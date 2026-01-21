"""Add plotting methods to result objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from plotnine import ggplot

    from moderndid.did.aggte_obj import AGGTEResult
    from moderndid.did.multiperiod_obj import MPResult


def _mp_plot(self: MPResult, **kwargs: Any) -> ggplot:
    """Plot method for MPResult objects.

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to plot_att_gt.

    Returns
    -------
    ggplot
        A plotnine ggplot object.
    """
    from moderndid.did.plots.core import plot_att_gt

    return plot_att_gt(self, **kwargs)


def _aggte_plot(self: AGGTEResult, **kwargs: Any) -> ggplot:
    """Plot method for AGGTEResult objects.

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to plot_event_study.

    Returns
    -------
    ggplot
        A plotnine ggplot object.
    """
    from moderndid.did.plots.core import plot_event_study

    return plot_event_study(self, **kwargs)


def add_plot_methods() -> None:
    """Add plot methods to result objects."""
    from moderndid.did.aggte_obj import AGGTEResult
    from moderndid.did.multiperiod_obj import MPResult

    MPResult.plot = _mp_plot
    AGGTEResult.plot = _aggte_plot
