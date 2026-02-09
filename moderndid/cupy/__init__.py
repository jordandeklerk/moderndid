"""CuPy GPU acceleration module for moderndid."""

from .backend import HAS_CUPY, get_backend, set_backend, to_device, to_numpy
from .bootstrap import (
    _aggregate_by_cluster_cupy,
    _multiplier_bootstrap_cupy,
)
from .regression import cupy_logistic_irls, cupy_wls

__all__ = [
    "HAS_CUPY",
    "_aggregate_by_cluster_cupy",
    "_multiplier_bootstrap_cupy",
    "cupy_logistic_irls",
    "cupy_wls",
    "get_backend",
    "set_backend",
    "to_device",
    "to_numpy",
]
