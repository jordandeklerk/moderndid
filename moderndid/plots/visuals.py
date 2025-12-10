"""Core visual functions for PlotCollection."""

import numpy as np


def scatter(
    data,
    target,
    x=None,
    y=None,
    color=None,
    marker=None,
    alpha=None,
    s=None,
    **kwargs,
):
    """Add scatter plot to target axes.

    Parameters
    ----------
    data : DataArray
        Data to plot. If x and y are None, uses data.values directly.
    target : matplotlib Axes
        Axes to plot on.
    x : array-like, optional
        X coordinates. If None, uses range(len(data)).
    y : array-like, optional
        Y coordinates. If None, uses data.values.
    color : color, optional
        Marker color.
    marker : str, optional
        Marker style.
    alpha : float, optional
        Transparency.
    s : float, optional
        Marker size.
    **kwargs
        Additional kwargs passed to ax.scatter().

    Returns
    -------
    PathCollection
        Scatter plot artist.
    """
    if y is None:
        y = data.values
    if x is None:
        x = np.arange(len(y))

    plot_kwargs = {}
    if color is not None:
        plot_kwargs["c"] = color
    if marker is not None:
        plot_kwargs["marker"] = marker
    if alpha is not None:
        plot_kwargs["alpha"] = alpha
    if s is not None:
        plot_kwargs["s"] = s

    plot_kwargs.update(kwargs)
    return target.scatter(x, y, **plot_kwargs)


def line(
    data,
    target,
    x=None,
    y=None,
    color=None,
    linestyle=None,
    linewidth=None,
    alpha=None,
    **kwargs,
):
    """Add line plot to target axes.

    Parameters
    ----------
    data : DataArray
        Data to plot. If x and y are None, uses data.values directly.
    target : matplotlib Axes
        Axes to plot on.
    x : array-like, optional
        X coordinates. If None, uses range(len(data)).
    y : array-like, optional
        Y coordinates. If None, uses data.values.
    color : color, optional
        Line color.
    linestyle : str, optional
        Line style ('-', '--', '-.', ':').
    linewidth : float, optional
        Line width.
    alpha : float, optional
        Transparency.
    **kwargs
        Additional kwargs passed to ax.plot().

    Returns
    -------
    Line2D
        Line artist.
    """
    if y is None:
        y = data.values
    if x is None:
        x = np.arange(len(y))

    plot_kwargs = {}
    if color is not None:
        plot_kwargs["color"] = color
    if linestyle is not None:
        plot_kwargs["linestyle"] = linestyle
    if linewidth is not None:
        plot_kwargs["linewidth"] = linewidth
    if alpha is not None:
        plot_kwargs["alpha"] = alpha

    plot_kwargs.update(kwargs)
    return target.plot(x, y, **plot_kwargs)[0]


def errorbar(
    data,
    target,
    x=None,
    y=None,
    yerr=None,
    xerr=None,
    color=None,
    marker=None,
    linestyle=None,
    capsize=None,
    alpha=None,
    **kwargs,
):
    """Add error bars to target axes.

    Parameters
    ----------
    data : DataArray
        Data to plot. If x and y are None, uses data.values directly.
    target : matplotlib Axes
        Axes to plot on.
    x : array-like, optional
        X coordinates. If None, uses range(len(data)).
    y : array-like, optional
        Y coordinates. If None, uses data.values.
    yerr : array-like or float, optional
        Y error values.
    xerr : array-like or float, optional
        X error values.
    color : color, optional
        Color for markers and error bars.
    marker : str, optional
        Marker style.
    linestyle : str, optional
        Line style connecting points.
    capsize : float, optional
        Length of error bar caps.
    alpha : float, optional
        Transparency.
    **kwargs
        Additional kwargs passed to ax.errorbar().

    Returns
    -------
    ErrorbarContainer
        Error bar artist container.
    """
    if y is None:
        y = data.values
    if x is None:
        x = np.arange(len(y))

    plot_kwargs = {}
    if yerr is not None:
        plot_kwargs["yerr"] = yerr
    if xerr is not None:
        plot_kwargs["xerr"] = xerr
    if color is not None:
        plot_kwargs["color"] = color
    if marker is not None:
        plot_kwargs["marker"] = marker
    if linestyle is not None:
        plot_kwargs["linestyle"] = linestyle
    if capsize is not None:
        plot_kwargs["capsize"] = capsize
    if alpha is not None:
        plot_kwargs["alpha"] = alpha

    plot_kwargs.update(kwargs)
    return target.errorbar(x, y, **plot_kwargs)


def fill_between(
    data,
    target,
    x=None,
    y1=None,
    y2=None,
    color=None,
    alpha=None,
    **kwargs,
):
    """Fill area between two curves.

    Parameters
    ----------
    data : DataArray
        Data to plot. If y1 and y2 are None, interprets data as y1.
    target : matplotlib Axes
        Axes to plot on.
    x : array-like, optional
        X coordinates. If None, uses range(len(data)).
    y1 : array-like, optional
        Lower bound. If None, uses data.values.
    y2 : array-like or float, optional
        Upper bound. If float, creates constant upper bound.
    color : color, optional
        Fill color.
    alpha : float, optional
        Transparency.
    **kwargs
        Additional kwargs passed to ax.fill_between().

    Returns
    -------
    PolyCollection
        Fill artist.
    """
    if y1 is None:
        y1 = data.values
    if x is None:
        x = np.arange(len(y1))

    if y2 is None:
        y2 = 0

    plot_kwargs = {}
    if color is not None:
        plot_kwargs["color"] = color
    if alpha is not None:
        plot_kwargs["alpha"] = alpha

    plot_kwargs.update(kwargs)
    return target.fill_between(x, y1, y2, **plot_kwargs)


def hline(
    data,
    target,
    y=None,
    color=None,
    linestyle=None,
    linewidth=None,
    alpha=None,
    **kwargs,
):
    """Add horizontal line to target axes.

    Parameters
    ----------
    data : DataArray
        Not used, for API consistency.
    target : matplotlib Axes
        Axes to plot on.
    y : float, optional
        Y-coordinate of the line. If None, uses data.values.item().
    color : color, optional
        Line color.
    linestyle : str, optional
        Line style.
    linewidth : float, optional
        Line width.
    alpha : float, optional
        Transparency.
    **kwargs
        Additional kwargs passed to ax.axhline().

    Returns
    -------
    Line2D
        Horizontal line artist.
    """
    if y is None:
        if hasattr(data, "item"):
            y = data.item()
        elif hasattr(data, "values"):
            y = data.values.item() if data.values.ndim == 0 else data.values[0]
        else:
            y = 0

    plot_kwargs = {}
    if color is not None:
        plot_kwargs["color"] = color
    if linestyle is not None:
        plot_kwargs["linestyle"] = linestyle
    if linewidth is not None:
        plot_kwargs["linewidth"] = linewidth
    if alpha is not None:
        plot_kwargs["alpha"] = alpha

    plot_kwargs.update(kwargs)
    return target.axhline(y=y, **plot_kwargs)


def vline(
    data,
    target,
    x=None,
    color=None,
    linestyle=None,
    linewidth=None,
    alpha=None,
    **kwargs,
):
    """Add vertical line to target axes.

    Parameters
    ----------
    data : DataArray
        Not used, for API consistency.
    target : matplotlib Axes
        Axes to plot on.
    x : float, optional
        X-coordinate of the line. If None, uses data.values.item().
    color : color, optional
        Line color.
    linestyle : str, optional
        Line style.
    linewidth : float, optional
        Line width.
    alpha : float, optional
        Transparency.
    **kwargs
        Additional kwargs passed to ax.axvline().

    Returns
    -------
    Line2D
        Vertical line artist.
    """
    if x is None:
        if hasattr(data, "item"):
            x = data.item()
        elif hasattr(data, "values"):
            x = data.values.item() if data.values.ndim == 0 else data.values[0]
        else:
            x = 0

    plot_kwargs = {}
    if color is not None:
        plot_kwargs["color"] = color
    if linestyle is not None:
        plot_kwargs["linestyle"] = linestyle
    if linewidth is not None:
        plot_kwargs["linewidth"] = linewidth
    if alpha is not None:
        plot_kwargs["alpha"] = alpha

    plot_kwargs.update(kwargs)
    return target.axvline(x=x, **plot_kwargs)
