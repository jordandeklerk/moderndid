"""Plotting functions for sensitivity analysis."""

try:
    import matplotlib as _mpl  # noqa: F401
except ImportError as e:
    raise ImportError(
        "matplotlib is required for plotting functionality. Install it with: pip install moderndid[plots]"
    ) from e

from moderndid.didhonest.plots.core import (
    plot_sensitivity_event_study,
    plot_sensitivity_rm,
    plot_sensitivity_sm,
)

__all__ = [
    "plot_sensitivity_event_study",
    "plot_sensitivity_rm",
    "plot_sensitivity_sm",
]
