"""Plotting functions for DID analysis."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

try:
    from matplotlib.lines import Line2D
except ImportError as e:
    raise ImportError(
        "matplotlib is required for plotting functionality. Install it with: pip install moderndid[plots]"
    ) from e

from moderndid.did.aggte_obj import AGGTEResult
from moderndid.did.multiperiod_obj import MPResult
from moderndid.plots import PlotCollection
from moderndid.plots.converters import aggte_to_dataset, mpresult_to_dataset
from moderndid.plots.themes import THEMES, PlotTheme, apply_theme


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
) -> PlotCollection:
    """Plot group-time average treatment effects.

    Creates a panel plot showing ATT estimates for each treatment cohort (group)
    over time. Points are color-coded by treatment status (pre vs post-treatment),
    with confidence intervals shown as error bars.

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
    legend : bool, default=True
        Whether to show the legend.
    ref_line : float, default=0
        Reference line value (typically 0).
    figsize : tuple of float, optional
        Figure size (width, height). If None, determined automatically.
    theme : str or PlotTheme, default="default"
        Theme name or custom PlotTheme object. Available themes: "default",
        "minimal", "publication", "colorful".
    show_ci : bool, default=True
        Whether to show confidence intervals.
    **kwargs
        Additional keyword arguments passed to matplotlib functions.

    Returns
    -------
    PlotCollection
        PlotCollection object for method chaining. Access the figure via
        ``pc.viz["figure"]`` and axes via ``pc.viz["plot"]``.

    See Also
    --------
    plot_event_study : Plot event study for dynamic aggregation.
    plot_did : Unified plotting interface for DID results.

    Examples
    --------
    Basic usage with the minimum wage dataset:

    .. plot::
        :context: close-figs

        >>> from moderndid import att_gt, load_mpdta, plot_att_gt
        >>> df = load_mpdta()
        >>> result = att_gt(
        ...     data=df,
        ...     yname="lemp",
        ...     tname="year",
        ...     gname="first.treat",
        ...     idname="countyreal",
        ... )
        >>> pc = plot_att_gt(result)
    """
    theme_obj = _get_theme(theme)
    dataset = mpresult_to_dataset(mp_result)
    unique_groups = np.unique(mp_result.groups)

    if groups is not None:
        groups_arr = np.array(groups)
        valid_groups = groups_arr[np.isin(groups_arr, unique_groups)]
        if len(valid_groups) != len(groups_arr):
            invalid = groups_arr[~np.isin(groups_arr, unique_groups)]
            warnings.warn(f"Groups {invalid} not found in data. Using available groups.")
        unique_groups = valid_groups if len(valid_groups) > 0 else unique_groups

    n_groups = len(unique_groups)

    if figsize is None:
        figsize = (10, 3 * n_groups)

    pc = PlotCollection.wrap(
        dataset,
        cols=["group"],
        col_wrap=1,
        figure_kwargs={"figsize": figsize},
    )
    apply_theme(pc, theme_obj)

    axes_data = pc.viz["plot"]
    if isinstance(axes_data.values, np.ndarray):
        axes_flat = axes_data.values.flatten()
    else:
        axes_flat = [axes_data.values]

    all_groups = dataset["att"].coords["group"]
    all_times = dataset["att"].coords["time"]
    att_array = dataset["att"].values
    se_array = dataset["se"].values
    treatment_status = dataset["treatment_status"].values

    for idx, group in enumerate(unique_groups):
        if idx >= len(axes_flat):
            break

        ax = axes_flat[idx]
        group_idx = np.where(all_groups == group)[0][0]

        times = all_times
        att = att_array[group_idx, :]
        se = se_array[group_idx, :]
        status = treatment_status[group_idx, :]

        sort_idx = np.argsort(times)
        times_sorted = times[sort_idx]
        att_sorted = att[sort_idx]
        se_sorted = se[sort_idx]
        status_sorted = status[sort_idx]

        for t, a, s, st in zip(times_sorted, att_sorted, se_sorted, status_sorted):
            if np.isnan(a):
                continue
            color = theme_obj.post_treatment_color if st == "post" else theme_obj.pre_treatment_color

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
            ax.axhline(y=ref_line, color="black", linestyle="--", linewidth=2, alpha=0.5, zorder=1)

        valid_times = times_sorted[~np.isnan(att_sorted)]
        if len(valid_times) > 0:
            if xgap > 1:
                time_labels = valid_times[::xgap]
            else:
                time_labels = valid_times
            ax.set_xticks(time_labels)
            ax.set_xticklabels([f"{int(t)}" if float(t).is_integer() else f"{t:.1f}" for t in time_labels])

        group_label = f"{int(group)}" if float(group).is_integer() else f"{group:.1f}"
        ax.set_title(f"{title} {group_label}", fontsize=theme_obj.title_fontsize, fontweight="bold", pad=10, loc="left")
        ax.set_xlabel(xlab or "Time", fontsize=theme_obj.label_fontsize)

        if idx == n_groups // 2:
            ax.set_ylabel(ylab or "ATT", fontsize=theme_obj.label_fontsize)

        if ylim is not None:
            ax.set_ylim(ylim)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if legend and n_groups > 0:
        fig = pc.viz["figure"]
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Pre-treatment",
                markerfacecolor=theme_obj.pre_treatment_color,
                markersize=theme_obj.marker_size,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Post-treatment",
                markerfacecolor=theme_obj.post_treatment_color,
                markersize=theme_obj.marker_size,
            ),
        ]
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=2,
            frameon=False,
            fontsize=theme_obj.label_fontsize,
        )

    fig = pc.viz["figure"]
    fig.tight_layout()
    if legend and n_groups > 0:
        fig.subplots_adjust(bottom=0.1)

    return pc


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
) -> PlotCollection:
    """Create event study plot for dynamic treatment effects.

    Visualizes treatment effects relative to the treatment timing, showing
    pre-treatment trends (for parallel trends assessment) and post-treatment
    effects. Points are color-coded by treatment status with confidence
    intervals.

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
        Theme name or custom PlotTheme object. Available themes: "default",
        "minimal", "publication", "colorful".
    **kwargs
        Additional keyword arguments passed to plotting functions.

    Returns
    -------
    PlotCollection
        PlotCollection object for method chaining. Access the figure via
        ``pc.viz["figure"]`` and axes via ``pc.viz["plot"]``.

    See Also
    --------
    plot_att_gt : Plot group-time ATT estimates.
    plot_did : Unified plotting interface for DID results.
    aggte : Aggregate group-time ATTs into event study.

    Examples
    --------
    Basic event study plot from aggregated results:

    .. plot::
        :context: close-figs

        >>> from moderndid import att_gt, aggte, load_mpdta, plot_event_study
        >>> df = load_mpdta()
        >>> mp_result = att_gt(
        ...     data=df,
        ...     yname="lemp",
        ...     tname="year",
        ...     gname="first.treat",
        ...     idname="countyreal",
        ... )
        >>> es_result = aggte(mp_result, type="dynamic")
        >>> pc = plot_event_study(es_result)
    """
    if aggte_result.aggregation_type != "dynamic":
        raise ValueError(f"Event study plot requires dynamic aggregation, got {aggte_result.aggregation_type}")

    theme_obj = _get_theme(theme)
    dataset = aggte_to_dataset(aggte_result)
    pc = PlotCollection.grid(dataset, figure_kwargs={"figsize": figsize})
    apply_theme(pc, theme_obj)

    event_times = dataset["att"].coords["event"]
    att = dataset["att"].values
    se = dataset["se"].values
    treatment_status = dataset["treatment_status"].values

    sort_idx = np.argsort(event_times)
    event_times = event_times[sort_idx]
    att = att[sort_idx]
    se = se[sort_idx]
    treatment_status = treatment_status[sort_idx]

    if band_type == "uniform" and aggte_result.critical_values is not None:
        crit_vals = aggte_result.critical_values[sort_idx]
    else:
        crit_vals = np.full_like(se, 1.96)

    ax = pc.viz["plot"].values.item() if pc.viz["plot"].size == 1 else pc.viz["plot"].values

    for t, a, s, cv, status in zip(event_times, att, se, crit_vals, treatment_status):
        color = theme_obj.post_treatment_color if status == "post" else theme_obj.pre_treatment_color
        yerr = cv * s if show_bands else 0
        ax.errorbar(
            t,
            a,
            yerr=yerr,
            fmt="o",
            color=color,
            markersize=theme_obj.marker_size,
            capsize=5,
            capthick=1,
            alpha=0.8,
            label="_nolegend_",
            **kwargs,
        )

    if ref_line is not None:
        ax.axhline(y=ref_line, color="black", linestyle="--", linewidth=2, alpha=0.5, zorder=1)

    ax.set_xlabel(xlab or "Time Relative to Treatment", fontsize=theme_obj.label_fontsize)
    ax.set_ylabel(ylab or "Average Treatment Effect", fontsize=theme_obj.label_fontsize)
    ax.set_title(title or "Dynamic Treatment Effects", fontsize=theme_obj.title_fontsize, fontweight="bold", pad=15)

    if ylim is not None:
        ax.set_ylim(ylim)

    _add_legend(ax, theme_obj)

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

    fig = pc.viz["figure"]
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)

    return pc


def plot_did(
    result: MPResult | AGGTEResult,
    plot_type: str | None = None,
    theme: str | PlotTheme = "default",
    **kwargs: Any,
) -> PlotCollection:
    """Plot DiD parameters.

    Unified plotting interface that automatically selects the appropriate
    visualization based on the result type. For `MPResult`, creates group-time
    ATT plots. For `AGGTEResult` with dynamic aggregation, creates event study
    plots.

    Parameters
    ----------
    result : MPResult or AGGTEResult
        DID result object to plot.
    plot_type : str, optional
        Force a specific plot type. Options depend on result type:

        - For MPResult: "att_gt" (default)
        - For AGGTEResult: "dynamic" or "group"
    theme : str or PlotTheme, default="default"
        Theme name or custom PlotTheme object. Available themes: "default",
        "minimal", "publication", "colorful".
    **kwargs
        Additional arguments passed to the specific plotting function.
        See :func:`plot_att_gt` and :func:`plot_event_study` for available
        options.

    Returns
    -------
    PlotCollection
        PlotCollection object for method chaining. Access the figure via
        ``pc.viz["figure"]`` and axes via ``pc.viz["plot"]``.

    See Also
    --------
    plot_att_gt : Plot group-time ATT estimates.
    plot_event_study : Plot event study for dynamic aggregation.

    Examples
    --------
    Automatically plot group-time ATTs from MPResult:

    .. plot::
        :context: close-figs

        >>> from moderndid import att_gt, load_mpdta, plot_did
        >>> df = load_mpdta()
        >>> mp_result = att_gt(
        ...     data=df,
        ...     yname="lemp",
        ...     tname="year",
        ...     gname="first.treat",
        ...     idname="countyreal",
        ... )
        >>> pc = plot_did(mp_result)

    Automatically plot event study from AGGTEResult:

    .. plot::
        :context: close-figs

        >>> from moderndid import att_gt, aggte, load_mpdta, plot_did
        >>> df = load_mpdta()
        >>> mp_result = att_gt(
        ...     data=df,
        ...     yname="lemp",
        ...     tname="year",
        ...     gname="first.treat",
        ...     idname="countyreal",
        ... )
        >>> es_result = aggte(mp_result, type="dynamic")
        >>> pc = plot_did(es_result)

    Pass customization options through:

    .. plot::
        :context: close-figs

        >>> from moderndid import att_gt, aggte, load_mpdta, plot_did
        >>> df = load_mpdta()
        >>> mp_result = att_gt(
        ...     data=df,
        ...     yname="lemp",
        ...     tname="year",
        ...     gname="first.treat",
        ...     idname="countyreal",
        ... )
        >>> es_result = aggte(mp_result, type="dynamic")
        >>> pc = plot_did(
        ...     es_result,
        ...     ylim=(-0.4, 0.2),
        ...     theme="publication",
        ...     figsize=(12, 6),
        ... )
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


def _get_theme(theme):
    """Resolve theme from string or PlotTheme object."""
    if isinstance(theme, str):
        return THEMES.get(theme, THEMES["default"])
    return theme


def _add_legend(ax, theme, fontsize=None):
    """Add pre/post treatment legend to axes."""
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Pre-treatment",
            markerfacecolor=theme.pre_treatment_color,
            markersize=theme.marker_size,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Post-treatment",
            markerfacecolor=theme.post_treatment_color,
            markersize=theme.marker_size,
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
        frameon=False,
        fontsize=fontsize or theme.label_fontsize,
    )
    return legend_elements
