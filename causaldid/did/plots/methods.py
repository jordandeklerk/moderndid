"""Add plotting methods to result objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from causaldid.did.aggte_obj import AGGTEResult
    from causaldid.did.multiperiod_obj import MPResult


def _mp_plot(self: MPResult, **kwargs: Any) -> Figure:
    """Plot method for MPResult objects.

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to plot_att_gt.

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    from causaldid.did.plots.core import plot_att_gt

    return plot_att_gt(self, **kwargs)


def _aggte_plot(self: AGGTEResult, plot_type: str | None = None, **kwargs: Any) -> Figure:
    """Plot method for AGGTEResult objects.

    Parameters
    ----------
    plot_type : str, optional
        Type of plot. If None, uses the aggregation type.
        Options: "dynamic" or "group".
    **kwargs
        Keyword arguments passed to the specific plotting function.

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    from causaldid.did.plots.core import plot_did

    return plot_did(self, plot_type=plot_type, **kwargs)


def add_plot_methods():
    """Add plot methods to result objects."""
    from causaldid.did.aggte_obj import AGGTEResult
    from causaldid.did.multiperiod_obj import MPResult

    MPResult.plot = _mp_plot
    AGGTEResult.plot = _aggte_plot
