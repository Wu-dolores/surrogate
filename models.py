"""
Neural network models for atmospheric radiation surrogate modeling.

This module contains the LocalGNO (Graph Neural Operator) architecture
for predicting heating rates and boundary fluxes.
"""

import torch
import torch.nn as nn
from typing import Tuple


class LocalGNOBlock(nn.Module):
    """
    Local Graph Neural Operator block for processing vertical profiles.

    This block performs message passing between neighboring vertical levels
    within a local window of size K.

    Args:
        hidden: Hidden dimension size
        K: Neighborhood size (processes levels within ±K distance)
    """

    def __init__(self, hidden: int = 128, K: int = 6):
        super().__init__()
        self.K = K

        # Message function: combines node features and coordinate distance
        self.msg = nn.Sequential(
            nn.Linear(hidden * 2 + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        # Update function: combines original features with aggregated messages
        self.upd = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        h: torch.Tensor,
        coord: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with local message passing.

        Args:
            h: Node features (B, N, hidden)
            coord: Vertical coordinates (B, N)

        Returns:
            Updated node features (B, N, hidden)
        """
        B, N, Hh = h.shape
        agg = torch.zeros_like(h)
        count = torch.zeros((1, N, 1), device=h.device, dtype=h.dtype)

        # Message passing within local neighborhood
        for off in range(-self.K, self.K + 1):
            if off == 0:
                continue

            if off > 0:
                hi, hj = h[:, :-off, :], h[:, off:, :]
                dcoord = (coord[:, off:] - coord[:, :-off]).unsqueeze(-1)
                m = self.msg(torch.cat([hi, hj, dcoord], dim=-1))
                agg[:, :-off, :] += m
                count[:, :-off, :] += 1.0
            else:
                k = -off
                hi, hj = h[:, k:, :], h[:, :-k, :]
                dcoord = (coord[:, :-k] - coord[:, k:]).unsqueeze(-1)
                m = self.msg(torch.cat([hi, hj, dcoord], dim=-1))
                agg[:, k:, :] += m
                count[:, k:, :] += 1.0

        # Average messages and update
        agg = agg / count.clamp_min(1.0)
        dh = self.upd(torch.cat([h, agg], dim=-1))
        return self.norm(h + dh)


class HR_TOA_BOA_Model(nn.Module):
    """
    Multi-task model for predicting heating rates and boundary fluxes.

    Predicts:
    - HR: Heating rate profile (vertical derivative of flux)
    - F_TOA: Net flux at top of atmosphere
    - F_BOA: Net flux at bottom of atmosphere (surface)

    Args:
        in_dim: Input feature dimension
        hidden: Hidden dimension size
        K: LocalGNO neighborhood size
        L: Number of LocalGNO blocks
        ts_idx: Index of surface temperature in input features
    """

    def __init__(
        self,
        in_dim: int = 7,
        hidden: int = 128,
        K: int = 6,
        L: int = 4,
        ts_idx: int = 3
    ):
        super().__init__()
        self.ts_idx = ts_idx

        # Input embedding
        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        # LocalGNO blocks for vertical processing
        self.blocks = nn.ModuleList([
            LocalGNOBlock(hidden=hidden, K=K) for _ in range(L)
        ])

        # Heating rate prediction head
        self.hr_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

        # TOA flux head (global + local context)
        self.toa_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

        # BOA flux head with surface temperature skip connection
        self.boa_head = nn.Sequential(
            nn.Linear(hidden * 2 + 1, hidden),  # +1 for Ts skip
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        coord: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features (B, N, in_dim)
            coord: Vertical coordinates (B, N)

        Returns:
            Tuple of (hr, f_toa, f_boa):
            - hr: Heating rate (B, N, 1)
            - f_toa: TOA flux (B, 1)
            - f_boa: BOA flux (B, 1)
        """
        # Embed and process through LocalGNO blocks
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h, coord)

        # Predict heating rate at each level
        hr = self.hr_head(h)  # (B, N, 1)

        # Global context from mean pooling
        g = h.mean(dim=1)  # (B, hidden)

        # Boundary-specific contexts
        h_toa = h[:, 0, :]   # Top of atmosphere
        h_boa = h[:, -1, :]  # Bottom of atmosphere

        # Surface temperature skip connection
        ts_skip = x[:, 0, self.ts_idx:self.ts_idx+1]  # (B, 1)

        # Predict boundary fluxes
        f_toa = self.toa_head(torch.cat([g, h_toa], dim=-1))
        f_boa = self.boa_head(torch.cat([g, h_boa, ts_skip], dim=-1))

        return hr, f_toa, f_boa
