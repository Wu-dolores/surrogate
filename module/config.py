"""
Configuration management for surrogate model training and evaluation.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    in_dim: int = 7
    hidden: int = 128
    K: int = 6  # LocalGNO neighborhood size
    L: int = 4  # Number of LocalGNO blocks
    ts_idx: int = 3  # Index of surface temperature in features


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 100
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-5

    # Loss weights: [HR, TOA, BOA, Physics]
    loss_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 0.0])

    # Data augmentation
    regrid_choices: List[int] = field(default_factory=lambda: [40, 80, 120, 160])
    regrid_mult: float = 1.0

    # Sample weighting
    Ts_tail: float = 320.0  # Temperature threshold for tail weighting
    tail_mult: float = 2.0  # Weight multiplier for tail samples


@dataclass
class DataConfig:
    """Data processing configuration."""
    train_ratio: float = 0.8
    random_seed: int = 42

    # Feature names in order
    features: List[str] = field(default_factory=lambda: [
        'T', 'logp', 'q', 'Ts_broadcast',
        'cwp_norm', 'rw_norm', 'tpw'
    ])


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    batch_size: int = 256
    alpha_gamma: float = 1.0
    bot_window_k: int = 0  # 0 = full column, >0 = bottom K layers
    generate_diagnostics: bool = False
    worst_k: int = 5  # Number of worst samples to analyze
