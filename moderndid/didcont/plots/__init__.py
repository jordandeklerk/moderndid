"""Plotting functions for continuous treatment DiD."""

try:
    import matplotlib as _mpl  # noqa: F401
except ImportError as e:
    raise ImportError(
        "matplotlib is required for plotting functionality. Install it with: pip install moderndid[plots]"
    ) from e

from .core import plot_cont_did

__all__ = ["plot_cont_did"]
