"""Plots for DID models."""

try:
    import matplotlib as _mpl  # noqa: F401
except ImportError as e:
    raise ImportError(
        "matplotlib is required for plotting functionality. Install it with: pip install moderndid[plots]"
    ) from e

from moderndid.did.plots.core import (
    plot_att_gt,
    plot_did,
    plot_event_study,
)
from moderndid.did.plots.methods import add_plot_methods

# Add plotting methods to result objects
add_plot_methods()

__all__ = [
    "plot_att_gt",
    "plot_event_study",
    "plot_did",
]
