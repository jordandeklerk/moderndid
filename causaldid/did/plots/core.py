"""Plotting functions for DID analysis."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from causaldid.did.aggte_obj import AGGTEResult
from causaldid.did.multiperiod_obj import MPResult


@dataclass
class PlotTheme:
    """Theme configuration for plots."""

    pre_color: str = "#FF6B6B"
    post_color: str = "#4ECDC4"
    accent_color: str = "#45B7D1"
    background_color: str = "white"
    grid_color: str = "#E0E0E0"
    text_color: str = "#2C3E50"

    title_size: int = 14
    label_size: int = 12
    tick_size: int = 10

    grid_style: str = "--"
    grid_alpha: float = 0.3
    marker_size: int = 8
    line_width: float = 2
    bar_width: float = 0.8

    figure_dpi: int = 100
    tight_layout: bool = True


THEMES = {
    "default": PlotTheme(),
    "minimal": PlotTheme(
        pre_color="#666666",
        post_color="#333333",
        accent_color="#000000",
        grid_alpha=0.1,
    ),
    "colorful": PlotTheme(
        pre_color="#E74C3C",
        post_color="#3498DB",
        accent_color="#9B59B6",
    ),
    "publication": PlotTheme(
        pre_color="#D32F2F",
        post_color="#1976D2",
        accent_color="#388E3C",
        marker_size=6,
        line_width=1.5,
        figure_dpi=300,
    ),
}


def _apply_theme(theme: PlotTheme):
    """Apply theme settings to matplotlib."""
    plt.rcParams.update(
        {
            "figure.dpi": theme.figure_dpi,
            "font.size": theme.tick_size,
            "axes.labelsize": theme.label_size,
            "axes.titlesize": theme.title_size,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": theme.text_color,
            "axes.labelcolor": theme.text_color,
            "text.color": theme.text_color,
            "xtick.color": theme.text_color,
            "ytick.color": theme.text_color,
        }
    )


def plot_att_gt(
    mp_result: MPResult,
    groups: Sequence[int] | None = None,
    ylim: tuple[float, float] | None = None,
    xlab: str | None = None,
    ylab: str | None = None,
    title: str = "Group",
    xgap: int = 1,
    legend: bool = True,
    ref_line: float = 0,
    figsize: tuple[float, float] | None = None,
    theme: str | PlotTheme = "default",
    show_ci: bool = True,
    **kwargs: Any,
) -> Figure:
    """Plot group-time average treatment effects.

    Parameters
    ----------
    mp_result : MPResult
        Multi-period DID result object containing group-time ATT estimates.
    groups : sequence of int, optional
        Specific groups to include in the plot. If None, all groups are plotted.
    ylim : tuple of float, optional
        Y-axis limits (min, max).
    xlab : str, optional
        X-axis label. Defaults to "Time".
    ylab : str, optional
        Y-axis label. Defaults to "ATT".
    title : str, default="Group"
        Title prefix for each panel.
    xgap : int, default=1
        Gap between x-axis labels (e.g., 2 shows every other label).
    ncol : int, default=2
        Number of columns in the subplot grid. (Deprecated - plots are now stacked vertically)
    legend : bool, default=True
        Whether to show the legend.
    ref_line : float, default=0
        Reference line value (typically 0).
    figsize : tuple of float, optional
        Figure size (width, height). If None, determined automatically.
    theme : str or PlotTheme, default="default"
        Theme name or custom PlotTheme object.
    show_ci : bool, default=True
        Whether to show confidence intervals.
    **kwargs
        Additional keyword arguments passed to matplotlib functions.

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    if isinstance(theme, str):
        theme_obj = THEMES.get(theme, THEMES["default"])
    else:
        theme_obj = theme

    _apply_theme(theme_obj)

    unique_groups = np.unique(mp_result.groups)

    if groups is not None:
        groups = np.array(groups)
        valid_groups = groups[np.isin(groups, unique_groups)]
        if len(valid_groups) != len(groups):
            invalid = groups[~np.isin(groups, unique_groups)]
            warnings.warn(f"Groups {invalid} not found in data. Using available groups.")
        unique_groups = valid_groups if len(valid_groups) > 0 else unique_groups

    n_groups = len(unique_groups)

    if figsize is None:
        figsize = (10, 3 * n_groups)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_groups, 1, hspace=0.6)

    for idx, group in enumerate(unique_groups):
        ax = fig.add_subplot(gs[idx, 0])

        mask = mp_result.groups == group
        times = mp_result.times[mask]
        att = mp_result.att_gt[mask]
        se = mp_result.se_gt[mask]

        sort_idx = np.argsort(times)
        times = times[sort_idx]
        att = att[sort_idx]
        se = se[sort_idx]

        is_post = times >= group

        for t, a, s, post in zip(times, att, se, is_post):
            color = theme_obj.post_color if post else theme_obj.pre_color

            if show_ci:
                ax.errorbar(
                    t,
                    a,
                    yerr=mp_result.critical_value * s,
                    fmt="o",
                    color=color,
                    markersize=theme_obj.marker_size,
                    capsize=5,
                    capthick=1,
                    alpha=0.8,
                    **kwargs,
                )
            else:
                ax.plot(t, a, "o", color=color, markersize=theme_obj.marker_size, **kwargs)

        if ref_line is not None:
            ax.axhline(y=ref_line, color="black", linestyle="--", linewidth=2, alpha=0.5, label="_nolegend_", zorder=1)

        unique_times_for_group = np.unique(times)
        if xgap > 1:
            time_labels = unique_times_for_group[::xgap]
        else:
            time_labels = unique_times_for_group
        ax.set_xticks(time_labels)
        ax.set_xticklabels([f"{int(t)}" if t.is_integer() else f"{t:.1f}" for t in time_labels])

        group_label = f"{int(group)}" if group.is_integer() else f"{group:.1f}"
        ax.set_title(f"{title} {group_label}", fontsize=theme_obj.title_size, fontweight="bold", pad=10, loc="left")

        ax.set_xlabel(xlab or "Time", fontsize=theme_obj.label_size)

        if idx == n_groups // 2:
            ax.set_ylabel(ylab or "ATT", fontsize=theme_obj.label_size)

        if ylim is not None:
            ax.set_ylim(ylim)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if legend and n_groups > 0:
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Pre-treatment",
                markerfacecolor=theme_obj.pre_color,
                markersize=theme_obj.marker_size,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Post-treatment",
                markerfacecolor=theme_obj.post_color,
                markersize=theme_obj.marker_size,
            ),
        ]

        fig.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=2,
            frameon=False,
            fontsize=theme_obj.label_size,
        )

    if theme_obj.tight_layout:
        fig.tight_layout()
        if legend and n_groups > 0:
            fig.subplots_adjust(bottom=0.1)

    plt.close(fig)
    return fig


def plot_event_study(
    aggte_result: AGGTEResult,
    ylim: tuple[float, float] | None = None,
    xlab: str | None = None,
    ylab: str | None = None,
    title: str | None = None,
    ref_line: float = 0,
    band_type: Literal["pointwise", "uniform"] = "pointwise",
    show_bands: bool = True,
    figsize: tuple[float, float] = (10, 6),
    theme: str | PlotTheme = "default",
    **kwargs: Any,
) -> Figure:
    """Create event study plot for dynamic treatment effects.

    Parameters
    ----------
    aggte_result : AGGTEResult
        Aggregated treatment effect result with dynamic aggregation.
    ylim : tuple of float, optional
        Y-axis limits (min, max).
    xlab : str, optional
        X-axis label. Defaults to "Time Relative to Treatment".
    ylab : str, optional
        Y-axis label. Defaults to "Average Treatment Effect".
    title : str, optional
        Plot title. Defaults based on aggregation type.
    ref_line : float, default=0
        Reference line value (typically 0).
    band_type : {"pointwise", "uniform"}, default="pointwise"
        Type of confidence bands to show.
    show_bands : bool, default=True
        Whether to show confidence bands.
    figsize : tuple of float, default=(10, 6)
        Figure size (width, height).
    theme : str or PlotTheme, default="default"
        Theme name or custom PlotTheme object.
    **kwargs
        Additional keyword arguments passed to plotting functions.

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    if aggte_result.aggregation_type != "dynamic":
        raise ValueError(f"Event study plot requires dynamic aggregation, got {aggte_result.aggregation_type}")

    if isinstance(theme, str):
        theme_obj = THEMES.get(theme, THEMES["default"])
    else:
        theme_obj = theme

    _apply_theme(theme_obj)

    fig, ax = plt.subplots(figsize=figsize)

    event_times = aggte_result.event_times
    att = aggte_result.att_by_event
    se = aggte_result.se_by_event

    sort_idx = np.argsort(event_times)
    event_times = event_times[sort_idx]
    att = att[sort_idx]
    se = se[sort_idx]

    pre_mask = event_times < 0
    post_mask = event_times >= 0

    if band_type == "uniform" and aggte_result.critical_values is not None:
        crit_vals = aggte_result.critical_values[sort_idx]
    else:
        crit_vals = np.full_like(se, 1.96)

    for t, a, s, cv in zip(event_times[pre_mask], att[pre_mask], se[pre_mask], crit_vals[pre_mask]):
        ax.errorbar(
            t,
            a,
            yerr=cv * s if show_bands else 0,
            fmt="o",
            color=theme_obj.pre_color,
            markersize=theme_obj.marker_size,
            capsize=5,
            capthick=1,
            alpha=0.8,
            label="_nolegend_",
            **kwargs,
        )

    for t, a, s, cv in zip(event_times[post_mask], att[post_mask], se[post_mask], crit_vals[post_mask]):
        ax.errorbar(
            t,
            a,
            yerr=cv * s if show_bands else 0,
            fmt="o",
            color=theme_obj.post_color,
            markersize=theme_obj.marker_size,
            capsize=5,
            capthick=1,
            alpha=0.8,
            label="_nolegend_",
            **kwargs,
        )

    if ref_line is not None:
        ax.axhline(y=ref_line, color="black", linestyle="--", linewidth=2, alpha=0.5, zorder=1)

    ax.set_xlabel(xlab or "Time Relative to Treatment", fontsize=theme_obj.label_size)
    ax.set_ylabel(ylab or "Average Treatment Effect", fontsize=theme_obj.label_size)

    if title is None:
        title = "Dynamic Treatment Effects"
    ax.set_title(title, fontsize=theme_obj.title_size, fontweight="bold", pad=15)

    if ylim is not None:
        ax.set_ylim(ylim)

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Pre-treatment",
            markerfacecolor=theme_obj.pre_color,
            markersize=theme_obj.marker_size,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Post-treatment",
            markerfacecolor=theme_obj.post_color,
            markersize=theme_obj.marker_size,
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
        frameon=False,
        fontsize=theme_obj.label_size,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_bands and band_type == "uniform":
        ax.text(
            0.02,
            0.98,
            "Note: Uniform confidence bands",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            style="italic",
            alpha=0.7,
        )

    if theme_obj.tight_layout:
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.15)

    plt.close(fig)
    return fig


def plot_did(
    result: MPResult | AGGTEResult,
    plot_type: str | None = None,
    theme: str | PlotTheme = "default",
    **kwargs: Any,
) -> Figure:
    """Plot DiD parameters.

    Parameters
    ----------
    result : MPResult or AGGTEResult
        DID result object to plot.
    plot_type : str, optional
        Force a specific plot type. Options depend on result type:
        - For MPResult: "att_gt" (default)
        - For AGGTEResult: "dynamic" or "group"
    theme : str or PlotTheme, default="default"
        Theme name or custom PlotTheme object.
    **kwargs
        Additional arguments passed to the specific plotting function.

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    if isinstance(result, MPResult):
        if plot_type is not None and plot_type != "att_gt":
            raise ValueError(f"Invalid plot_type '{plot_type}' for MPResult. Use 'att_gt' or None.")
        return plot_att_gt(result, theme=theme, **kwargs)

    if isinstance(result, AGGTEResult):
        if plot_type is None:
            if result.aggregation_type == "dynamic":
                plot_type = "dynamic"
            else:
                plot_type = "group"

        if plot_type == "dynamic":
            return plot_event_study(result, theme=theme, **kwargs)
        if plot_type == "group":
            return plot_att_gt(result, theme=theme, **kwargs)
        raise ValueError(f"Invalid plot_type '{plot_type}'. Options: 'dynamic' or 'group'")

    raise TypeError(f"Unknown result type: {type(result)}")
