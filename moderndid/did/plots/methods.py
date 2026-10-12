"""Add plotting methods to result objects."""

from moderndid.did.container import AGGTEResult, MPResult
from moderndid.did.plots.core import plot_event_study, plot_gt


def _mp_plot(self, **kwargs):
    """Plot method for MPResult objects.

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to plot_gt.

    Returns
    -------
    ggplot
        A plotnine ggplot object.
    """
    return plot_gt(self, **kwargs)


def _aggte_plot(self, **kwargs):
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
    return plot_event_study(self, **kwargs)


def add_plot_methods():
    """Add plot methods to result objects."""
    MPResult.plot = _mp_plot
    AGGTEResult.plot = _aggte_plot
