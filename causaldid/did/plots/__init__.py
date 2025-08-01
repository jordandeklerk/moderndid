"""Plots for DID models."""

from causaldid.did.plots.core import (
    plot_att_gt,
    plot_did,
    plot_event_study,
)
from causaldid.did.plots.methods import add_plot_methods

# Add plotting methods to result objects
add_plot_methods()

__all__ = [
    "plot_att_gt",
    "plot_event_study",
    "plot_did",
]
