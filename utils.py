"""
Utility functions for atmospheric radiation surrogate model.

This module contains common utility functions for data preprocessing,
normalization, and numerical integration.
"""

import numpy as np
import numpy.typing as npt
from typing import Tuple

# Constants
EPSILON_SMALL = 1e-12
EPSILON_LARGE = 1e-6


def zfit(x: npt.NDArray[np.float32]) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Compute mean and standard deviation for z-score normalization.

    Args:
        x: Input array of shape (N, D) where N is number of samples
           and D is number of features

    Returns:
        Tuple of (mean, std) arrays of shape (1, D)
    """
    mu = x.mean(axis=0, keepdims=True).astype(np.float32)
    std = (x.std(axis=0, keepdims=True) + EPSILON_LARGE).astype(np.float32)
    return mu, std


def zapply(
    x: npt.NDArray[np.float32],
    mu: npt.NDArray[np.float32],
    std: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """
    Apply z-score normalization.

    Args:
        x: Input array to normalize
        mu: Mean values
        std: Standard deviation values

    Returns:
        Normalized array with same shape as input
    """
    return ((x - mu) / (std + EPSILON_SMALL)).astype(np.float32)


def enforce_toa_to_boa_numpy(
    logp: npt.NDArray[np.float32],
    T: npt.NDArray[np.float32],
    q: npt.NDArray[np.float32],
    Fnet: npt.NDArray[np.float32],
    HR: npt.NDArray[np.float32] | None = None
) -> Tuple[npt.NDArray[np.float32], ...]:
    """
    Ensure vertical coordinate goes from TOA (top of atmosphere) to BOA (bottom).

    Checks if pressure increases from top to bottom (logp small -> large).
    If not, reverses all profiles along the vertical axis.

    Args:
        logp: Log-pressure coordinate array (S, N)
        T: Temperature profile (S, N)
        q: Specific humidity profile (S, N)
        Fnet: Net radiative flux profile (S, N)
        HR: Heating rate profile (S, N), optional

    Returns:
        Tuple of (logp, T, q, Fnet, HR) with consistent TOA->BOA ordering
    """
    # Check if majority of samples have decreasing pressure (wrong order)
    if np.mean(logp[:, 0] > logp[:, -1]) > 0.5:
        logp = logp[:, ::-1].copy()
        T = T[:, ::-1].copy()
        q = q[:, ::-1].copy()
        Fnet = Fnet[:, ::-1].copy()
        if HR is not None:
            HR = HR[:, ::-1].copy()
    return logp, T, q, Fnet, HR


def cumtrapz_batch_np(
    y: npt.NDArray[np.float32],
    x: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """
    Cumulative trapezoidal integration along axis=1 for batched data.

    Args:
        y: Integrand values (B, N)
        x: Integration coordinate (B, N)

    Returns:
        Cumulative integral (B, N) with first column set to 0
    """
    dx = x[:, 1:] - x[:, :-1]
    avg = 0.5 * (y[:, 1:] + y[:, :-1])
    inc = avg * dx
    I = np.zeros_like(y, dtype=np.float32)
    I[:, 1:] = np.cumsum(inc, axis=1)
    return I


def cwp_rw_from_q_logp_np(
    q: npt.NDArray[np.float32],
    logp: npt.NDArray[np.float32]
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Compute column water path (CWP) and residual water (RW) from humidity.

    CWP(i) = ∫_{top→i} q d(logp)  (water above level i)
    RW(i)  = ∫_{i→surf} q d(logp) (water below level i)

    Args:
        q: Specific humidity profile (B, N)
        logp: Log-pressure coordinate (B, N)

    Returns:
        Tuple of (cwp, rw) arrays of shape (B, N)
    """
    cwp = cumtrapz_batch_np(q, logp).astype(np.float32)
    total = cwp[:, -1:]  # Total column water
    rw = (total - cwp).astype(np.float32)
    return cwp, rw


def cwp_rw_norm_from_q_logp_np(
    q: npt.NDArray[np.float32],
    logp: npt.NDArray[np.float32]
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Compute normalized CWP and RW (divided by total column water).

    Args:
        q: Specific humidity profile (B, N)
        logp: Log-pressure coordinate (B, N)

    Returns:
        Tuple of (cwp_normalized, rw_normalized, total_water)
    """
    cwp = cumtrapz_batch_np(q, logp).astype(np.float32)
    total = cwp[:, -1:]
    rw = (total - cwp).astype(np.float32)
    denom = total + EPSILON_LARGE
    cwp_n = (cwp / denom).astype(np.float32)
    rw_n = (rw / denom).astype(np.float32)
    return cwp_n, rw_n, total


def alpha_full_column(
    logp: npt.NDArray[np.float32],
    alpha_gamma: float
) -> npt.NDArray[np.float32]:
    """
    Compute alpha weighting across full atmospheric column.

    Alpha ramps from 0 at TOA to 1 at BOA, raised to power gamma.

    Args:
        logp: Log-pressure coordinate (B, N)
        alpha_gamma: Exponent for alpha weighting

    Returns:
        Alpha weights (B, N) in range [0, 1]
    """
    a = (logp - logp[:, 0:1]) / (logp[:, -1:] - logp[:, 0:1] + EPSILON_LARGE)
    a = np.clip(a, 0.0, 1.0).astype(np.float32)
    return (a ** float(alpha_gamma)).astype(np.float32)


def alpha_bottom_window(
    logp: npt.NDArray[np.float32],
    alpha_gamma: float,
    bot_window_k: int
) -> npt.NDArray[np.float32]:
    """
    Compute alpha weighting only in bottom K layers.

    This concentrates boundary corrections near the surface.

    Args:
        logp: Log-pressure coordinate (B, N)
        alpha_gamma: Exponent for alpha weighting
        bot_window_k: Number of bottom layers to apply weighting

    Returns:
        Alpha weights (B, N) with zeros except in bottom K layers
    """
    B, N = logp.shape
    K = int(bot_window_k)

    if K <= 0:
        return alpha_full_column(logp, alpha_gamma)

    K = max(1, min(K, N))
    a = np.zeros((B, N), dtype=np.float32)

    if K == 1:
        a[:, -1] = 1.0
        return a

    ramp = np.linspace(0.0, 1.0, K, dtype=np.float32)[None, :] ** float(alpha_gamma)
    a[:, -K:] = ramp
    return a


def regrid_profile_batch(
    x: npt.NDArray[np.float32],
    logp: npt.NDArray[np.float32],
    new_logp: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """
    Interpolate profiles to new vertical grid.

    Args:
        x: Profile data (S, N) or (S, N, C)
        logp: Original coordinate (S, N)
        new_logp: Target coordinate (S, M)

    Returns:
        Interpolated profiles (S, M) or (S, M, C)
    """
    S = logp.shape[0]

    if x.ndim == 2:
        out = np.zeros((S, new_logp.shape[1]), dtype=np.float32)
        for i in range(S):
            out[i] = np.interp(new_logp[i], logp[i], x[i]).astype(np.float32)
        return out
    else:
        C = x.shape[2]
        out = np.zeros((S, new_logp.shape[1], C), dtype=np.float32)
        for i in range(S):
            for c in range(C):
                out[i, :, c] = np.interp(new_logp[i], logp[i], x[i, :, c]).astype(np.float32)
        return out


def make_logp_grid_like(
    logp: npt.NDArray[np.float32],
    M: int
) -> npt.NDArray[np.float32]:
    """
    Create uniform log-pressure grid matching sample boundaries.

    Args:
        logp: Original coordinate (S, N)
        M: Number of grid points

    Returns:
        Uniform grid (S, M) from each sample's top to bottom
    """
    top = logp[:, 0:1]
    bot = logp[:, -1:]
    a = np.linspace(0.0, 1.0, M, dtype=np.float32)[None, :]
    return (top + (bot - top) * a).astype(np.float32)

