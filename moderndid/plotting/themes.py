"""Theming system for moderndid plots."""

from dataclasses import dataclass

import numpy as np


@dataclass
class PlotTheme:
    """Unified plot theme configuration.

    Attributes
    ----------
    primary_color : str
        Primary color for plots.
    secondary_color : str
        Secondary color for plots.
    pre_treatment_color : str
        Color for pre-treatment periods.
    post_treatment_color : str
        Color for post-treatment periods.
    reference_color : str
        Color for reference lines.
    marker : str
        Default marker style.
    marker_size : float
        Default marker size.
    line_width : float
        Default line width.
    linestyle : str
        Default line style.
    ci_alpha : float
        Alpha transparency for confidence intervals.
    ci_linewidth : float
        Line width for confidence interval bounds.
    figsize : tuple of float
        Default figure size in inches (width, height).
    dpi : int
        Dots per inch for figure.
    grid : bool
        Whether to show grid.
    grid_alpha : float
        Grid transparency.
    spine_width : float
        Width of axis spines.
    title_fontsize : int
        Font size for titles.
    label_fontsize : int
        Font size for axis labels.
    tick_fontsize : int
        Font size for tick labels.
    """

    primary_color: str = "C0"
    secondary_color: str = "C1"
    pre_treatment_color: str = "#3498db"
    post_treatment_color: str = "#e74c3c"
    reference_color: str = "gray"
    marker: str = "o"
    marker_size: float = 6.0
    line_width: float = 1.5
    linestyle: str = "-"
    ci_alpha: float = 0.3
    ci_linewidth: float = 0.7
    figsize: tuple = (10, 6)
    dpi: int = 100
    grid: bool = True
    grid_alpha: float = 0.3
    spine_width: float = 0.8
    title_fontsize: int = 14
    label_fontsize: int = 12
    tick_fontsize: int = 10

    def apply(self, fig=None, axes=None):
        """Apply theme to matplotlib figure and/or axes.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Figure to apply theme to.
        axes : matplotlib.axes.Axes or array-like, optional
            Axes to apply theme to.
        """
        if fig is not None:
            fig.set_dpi(self.dpi)

        if axes is not None:
            if not isinstance(axes, (list, np.ndarray)):
                axes = [axes]
            for ax in np.ravel(axes):
                if self.grid:
                    ax.grid(alpha=self.grid_alpha, linewidth=0.5)
                for spine in ax.spines.values():
                    spine.set_linewidth(self.spine_width)
                ax.tick_params(labelsize=self.tick_fontsize)


THEMES = {
    "default": PlotTheme(),
    "minimal": PlotTheme(
        grid=False,
        spine_width=0.5,
    ),
    "publication": PlotTheme(
        figsize=(6, 4),
        dpi=300,
        marker_size=4,
        line_width=1.0,
        grid=True,
        grid_alpha=0.2,
    ),
    "colorful": PlotTheme(
        pre_treatment_color="#9b59b6",
        post_treatment_color="#f39c12",
        primary_color="#e74c3c",
        secondary_color="#3498db",
    ),
}


def apply_theme(plot_collection, theme_name):
    """Apply a theme to a PlotCollection.

    Parameters
    ----------
    plot_collection : PlotCollection
        The PlotCollection to apply theme to.
    theme_name : str or PlotTheme
        Either a theme name from THEMES or a PlotTheme object.
    """
    if isinstance(theme_name, str):
        if theme_name not in THEMES:
            raise ValueError(f"Unknown theme '{theme_name}'. Available themes: {list(THEMES.keys())}")
        theme = THEMES[theme_name]
    else:
        theme = theme_name

    if "figure" in plot_collection.viz:
        fig = plot_collection.viz["figure"]
        if hasattr(fig, "values"):
            fig = fig.values.item() if fig.values.size == 1 else fig.values
        theme.apply(fig=fig)

    if "plot" in plot_collection.viz:
        plots = plot_collection.viz["plot"]
        if hasattr(plots, "values"):
            axes_array = plots.values
            theme.apply(axes=axes_array)
