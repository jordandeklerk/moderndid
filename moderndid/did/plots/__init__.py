"""Plots for DID models."""

from moderndid.did.plots.core import (
    plot_event_study,
    plot_gt,
)
from moderndid.did.plots.methods import add_plot_methods

add_plot_methods()

__all__ = [
    "plot_event_study",
    "plot_gt",
]
