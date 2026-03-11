"""
Data loading and preprocessing for atmospheric radiation surrogate model.
"""

import numpy as np
import numpy.typing as npt
from typing import Dict, Tuple
from pathlib import Path

from utils import (
    enforce_toa_to_boa_numpy,
    cwp_rw_norm_from_q_logp_np,
    zfit,
    zapply
)


class AtmosphericDataLoader:
    """
    Loader for atmospheric profile data in NPZ format.

    Expected data format:
    - logp_arr: Log-pressure coordinate (S, N)
    - T_arr: Temperature profile (S, N)
    - q_arr: Specific humidity profile (S, N)
    - Ts_K: Surface temperature (S,)
    - Fnet_arr: Net radiative flux profile (S, N)
    """

    def __init__(self, data_path: str):
        """
        Initialize data loader.

        Args:
            data_path: Path to NPZ data file
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self.data = np.load(str(self.data_path), allow_pickle=True)
        self._validate_data()

    def _validate_data(self) -> None:
        """Validate that required fields exist in data file."""
        required_fields = ['logp_arr', 'T_arr', 'q_arr', 'Ts_K', 'Fnet_arr']
        missing = [f for f in required_fields if f not in self.data]
        if missing:
            raise ValueError(f"Missing required fields in data: {missing}")

    def load_raw_data(self) -> Dict[str, npt.NDArray[np.float32]]:
        """
        Load and enforce TOA->BOA ordering.

        Returns:
            Dictionary with keys: logp, T, q, Ts, Fnet, HR
        """
        logp = self.data["logp_arr"].astype(np.float32)
        T = self.data["T_arr"].astype(np.float32)
        q = self.data["q_arr"].astype(np.float32)
        Ts = self.data["Ts_K"].astype(np.float32)
        Fnet = self.data["Fnet_arr"].astype(np.float32)

        # Compute heating rate using forward difference
        HR = np.zeros_like(Fnet)
        logp, T, q, Fnet, HR = enforce_toa_to_boa_numpy(logp, T, q, Fnet, HR)

        S, N = Fnet.shape
        dF = Fnet[:, 1:] - Fnet[:, :-1]
        dp = logp[:, 1:] - logp[:, :-1] + 1e-9
        slope = dF / dp
        HR[:, :-1] = slope
        HR[:, -1] = slope[:, -1]  # Extrapolate last level

        return {
            'logp': logp,
            'T': T,
            'q': q,
            'Ts': Ts,
            'Fnet': Fnet,
            'HR': HR
        }

    def build_features(
        self,
        raw_data: Dict[str, npt.NDArray[np.float32]]
    ) -> npt.NDArray[np.float32]:
        """
        Build feature array from raw data.

        Features: [T, logp, q, Ts_broadcast, cwp_norm, rw_norm, tpw]

        Args:
            raw_data: Dictionary from load_raw_data()

        Returns:
            Feature array (S, N, 7)
        """
        logp = raw_data['logp']
        T = raw_data['T']
        q = raw_data['q']
        Ts = raw_data['Ts']

        S, N = logp.shape

        # Broadcast surface temperature to all levels
        Ts_b = np.repeat(Ts[:, None], N, axis=1).astype(np.float32)

        # Compute water path features
        cwp_n, rw_n, tpw = cwp_rw_norm_from_q_logp_np(q, logp)
        tpw_b = np.repeat(tpw, N, axis=1).astype(np.float32)

        # Stack features
        X = np.stack([T, logp, q, Ts_b, cwp_n, rw_n, tpw_b], axis=-1)
        return X.astype(np.float32)


def split_train_val(
    data: Dict[str, npt.NDArray],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[Dict[str, npt.NDArray], Dict[str, npt.NDArray]]:
    """
    Split data into training and validation sets.

    Args:
        data: Dictionary of arrays with first dimension as samples
        train_ratio: Fraction of data for training
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, val_data) dictionaries
    """
    rng = np.random.RandomState(seed)

    # Get number of samples from first array
    S = next(iter(data.values())).shape[0]
    perm = rng.permutation(S)

    split_idx = int(S * train_ratio)
    train_idx = perm[:split_idx]
    val_idx = perm[split_idx:]

    train_data = {k: v[train_idx] for k, v in data.items()}
    val_data = {k: v[val_idx] for k, v in data.items()}

    return train_data, val_data


def compute_normalization_stats(
    X: npt.NDArray[np.float32],
    HR: npt.NDArray[np.float32],
    Ftoa: npt.NDArray[np.float32],
    Fboa: npt.NDArray[np.float32]
) -> Dict[str, npt.NDArray[np.float32]]:
    """
    Compute normalization statistics from training data.

    Args:
        X: Feature array (S, N, D)
        HR: Heating rate (S, N)
        Ftoa: TOA flux (S,)
        Fboa: BOA flux (S,)

    Returns:
        Dictionary with mean and std for each variable
    """
    X_mu, X_std = zfit(X.reshape(-1, X.shape[-1]))
    H_mu, H_std = zfit(HR.reshape(-1, 1))
    Ftoa_mu, Ftoa_std = zfit(Ftoa.reshape(-1, 1))
    Fboa_mu, Fboa_std = zfit(Fboa.reshape(-1, 1))

    return {
        'X_mu': X_mu,
        'X_std': X_std,
        'H_mu': H_mu,
        'H_std': H_std,
        'Ftoa_mu': Ftoa_mu,
        'Ftoa_std': Ftoa_std,
        'Fboa_mu': Fboa_mu,
        'Fboa_std': Fboa_std
    }


def normalize_data(
    data: Dict[str, npt.NDArray[np.float32]],
    stats: Dict[str, npt.NDArray[np.float32]]
) -> Dict[str, npt.NDArray[np.float32]]:
    """
    Apply normalization to data using precomputed statistics.

    Args:
        data: Dictionary with keys X, HR, Ftoa, Fboa
        stats: Normalization statistics from compute_normalization_stats()

    Returns:
        Dictionary with normalized arrays
    """
    X_n = zapply(data['X'], stats['X_mu'].reshape(1, 1, -1), stats['X_std'].reshape(1, 1, -1))
    HR_n = zapply(data['HR'][..., None], stats['H_mu'], stats['H_std'])
    Ftoa_n = zapply(data['Ftoa'][:, None], stats['Ftoa_mu'], stats['Ftoa_std'])
    Fboa_n = zapply(data['Fboa'][:, None], stats['Fboa_mu'], stats['Fboa_std'])

    return {
        'X': X_n.astype(np.float32),
        'HR': HR_n.astype(np.float32),
        'Ftoa': Ftoa_n.astype(np.float32),
        'Fboa': Fboa_n.astype(np.float32),
        'logp': data['logp']
    }
