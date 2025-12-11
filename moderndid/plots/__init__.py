"""Unified plotting infrastructure for moderndid."""

try:
    import matplotlib as _mpl  # noqa: F401
except ImportError as e:
    raise ImportError(
        "matplotlib is required for plotting functionality. Install it with: pip install moderndid[plots]"
    ) from e

from moderndid.plots.collection import PlotCollection
from moderndid.plots.themes import THEMES, PlotTheme, apply_theme
from moderndid.plots.visuals import (
    errorbar,
    fill_between,
    hline,
    line,
    scatter,
    vline,
)

__all__ = [
    "PlotCollection",
    "PlotTheme",
    "THEMES",
    "apply_theme",
    "scatter",
    "line",
    "errorbar",
    "fill_between",
    "hline",
    "vline",
]
