"""Theme functions for moderndid plotnine plots."""

from __future__ import annotations

from plotnine import element_blank, element_line, element_rect, element_text, theme

COLORS = {
    "pre_treatment": "#3498db",
    "post_treatment": "#e74c3c",
    "line": "#3a3a3a",
    "ci_fill": "#bfbfbf",
    "reference": "gray",
    "original": "#2c3e50",
    "flci": "#3498db",
    "conditional": "#2ecc71",
    "c_f": "#9b59b6",
    "c_lf": "#e74c3c",
}


def theme_moderndid() -> theme:
    """Default moderndid theme for plotnine plots.

    Returns a clean, modern theme with sensible defaults for
    difference-in-differences visualizations.

    Returns
    -------
    theme
        A plotnine theme object.

    Examples
    --------
    >>> from moderndid.plots import plot_event_study, theme_moderndid
    >>> plot = plot_event_study(es_result) + theme_moderndid()
    """
    return theme(
        panel_background=element_rect(fill="white"),
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        axis_line=element_line(color="#333333", size=0.8),
        axis_ticks=element_line(color="#333333", size=0.5),
        axis_text=element_text(color="#333333", size=10),
        axis_title=element_text(color="#333333", size=12),
        plot_title=element_text(color="#333333", size=14, weight="bold"),
        legend_background=element_rect(fill="white", color=None),
        legend_key=element_rect(fill="white", color=None),
        strip_background=element_rect(fill="#F5F5F5"),
        strip_text=element_text(color="#333333", size=11, weight="bold"),
    )


def theme_publication() -> theme:
    """Publication-ready theme for academic papers.

    Returns a minimal theme optimized for print and publication,
    with clean lines and high contrast.

    Returns
    -------
    theme
        A plotnine theme object.

    Examples
    --------
    >>> from moderndid.plots import plot_event_study, theme_publication
    >>> plot = plot_event_study(es_result) + theme_publication()
    >>> plot.save("figure.pdf", width=6, height=4, dpi=300)
    """
    return theme(
        panel_background=element_rect(fill="white"),
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        panel_border=element_rect(color="#333333", size=0.5, fill=None),
        axis_line=element_blank(),
        axis_ticks=element_line(color="#333333", size=0.3),
        axis_text=element_text(color="#333333", size=9),
        axis_title=element_text(color="#333333", size=10),
        plot_title=element_text(color="#333333", size=11, weight="bold"),
        legend_background=element_rect(fill="white", color=None),
        legend_key=element_rect(fill="white", color=None),
        legend_position="bottom",
        strip_background=element_rect(fill="white", color="#333333", size=0.3),
        strip_text=element_text(color="#333333", size=9, weight="bold"),
        figure_size=(6, 4),
        dpi=300,
    )


def theme_minimal() -> theme:
    """Minimal theme with reduced visual elements.

    Returns a very clean theme with minimal chrome,
    suitable for dashboards or presentations.

    Returns
    -------
    theme
        A plotnine theme object.

    Examples
    --------
    >>> from moderndid.plots import plot_event_study, theme_minimal
    >>> plot = plot_event_study(es_result) + theme_minimal()
    """
    return theme(
        panel_background=element_rect(fill="white"),
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        axis_line=element_line(color="#666666", size=0.5),
        axis_ticks=element_line(color="#666666", size=0.3),
        axis_text=element_text(color="#666666", size=10),
        axis_title=element_text(color="#333333", size=11),
        plot_title=element_text(color="#333333", size=13, weight="bold"),
        legend_background=element_blank(),
        legend_key=element_blank(),
        strip_background=element_blank(),
        strip_text=element_text(color="#333333", size=10, weight="bold"),
    )
