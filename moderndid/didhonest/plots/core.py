"""Plotting functions for sensitivity analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from moderndid.plotting import PlotCollection
from moderndid.plotting.converters import sensitivity_to_dataset
from moderndid.plotting.themes import THEMES, PlotTheme, apply_theme

METHOD_COLORS = {
    "sensitivity": {
        "Original": "#3498db",  # Blue
        "FLCI": "#e74c3c",  # Red
        "Conditional": "#2ecc71",  # Green
        "C-F": "#9b59b6",  # Purple
        "C-LF": "#f39c12",  # Orange
    },
    "sensitivity_rm": {
        "Original": "#3498db",  # Blue
        "Conditional": "#2ecc71",  # Green
        "C-LF": "#f39c12",  # Orange
    },
}


def plot_sensitivity_sm(
    robust_results: pd.DataFrame,
    original_results,
    rescale_factor: float = 1,
    max_m: float = np.inf,
    add_x_axis: bool = True,
    figsize: tuple[float, float] = (12, 7),
    theme: str | PlotTheme = "default",
    **kwargs: Any,
) -> PlotCollection:
    """Create sensitivity plot showing how confidence intervals change with M.

    Visualizes how robust confidence intervals change as the smoothness
    parameter :math:`M` increases. The parameter :math:`M` bounds the maximum change in
    treatment effect trends between consecutive periods. Larger :math:`M` values
    allow for more violations of parallel trends.

    Parameters
    ----------
    robust_results : pd.DataFrame
        DataFrame from create_sensitivity_results_sm with columns:
        lb, ub, method, Delta, M.
    original_results : NamedTuple
        Result from construct_original_cs with lb, ub, method.
    rescale_factor : float, default=1
        Factor to rescale all values (M, lb, ub) for display.
    max_m : float, default=np.inf
        Maximum M value to display (after rescaling).
    add_x_axis : bool, default=True
        Whether to add horizontal line at y=0.
    figsize : tuple of float, default=(12, 7)
        Figure size (width, height).
    theme : str or PlotTheme, default="default"
        Theme name or custom PlotTheme object. Available themes: "default",
        "minimal", "publication", "colorful".
    **kwargs
        Additional keyword arguments passed to matplotlib functions.

    Returns
    -------
    PlotCollection
        PlotCollection object for method chaining. Access the figure via
        ``pc.viz["figure"]`` and axes via ``pc.viz["plot"]``.

    See Also
    --------
    plot_sensitivity_rm : Sensitivity plot for relative magnitude bounds.
    create_sensitivity_results_sm : Generate smoothness sensitivity results.
    honest_did : Main HonestDiD sensitivity analysis function.

    Examples
    --------
    Basic sensitivity plot with default settings:

    .. plot::
        :context: close-figs

        >>> from moderndid import att_gt, aggte, load_mpdta
        >>> from moderndid import plot_sensitivity_sm
        >>> from moderndid.didhonest import honest_did
        >>> df = load_mpdta()
        >>> gt_result = att_gt(
        ...     data=df,
        ...     yname="lemp",
        ...     tname="year",
        ...     gname="first.treat",
        ...     idname="countyreal",
        ...     bstrap=False,
        ... )
        >>> es_result = aggte(gt_result, type="dynamic")
        >>> hd_result = honest_did(
        ...     es_result,
        ...     event_time=0,
        ...     sensitivity_type="smoothness",
        ...     m_vec=[0.01, 0.02, 0.03],
        ... )
        >>> pc = plot_sensitivity_sm(
        ...     hd_result.robust_ci,
        ...     hd_result.original_ci,
        ... )
    """
    theme_obj = _get_theme(theme)
    m_col = "M" if "M" in robust_results.columns else "m"

    df = robust_results.copy()
    df[m_col] = df[m_col] * rescale_factor
    df["lb"] = df["lb"] * rescale_factor
    df["ub"] = df["ub"] * rescale_factor

    df = df[df[m_col] <= max_m]

    scaled_original = type(original_results)(
        lb=original_results.lb * rescale_factor,
        ub=original_results.ub * rescale_factor,
        method=original_results.method,
    )

    dataset = sensitivity_to_dataset(df, scaled_original, param_col=m_col)

    pc = PlotCollection.grid(dataset, figure_kwargs={"figsize": figsize})
    apply_theme(pc, theme_obj)

    ax = pc.viz["plot"].values.item() if pc.viz["plot"].size == 1 else pc.viz["plot"].values

    param_values = dataset["lb"].coords["param_value"]
    methods = dataset["lb"].coords["method"]
    midpoint = dataset["midpoint"].values
    halfwidth = dataset["halfwidth"].values

    value_range = param_values[-1] - param_values[0] if len(param_values) > 1 else 0
    offsets = _calculate_offsets(len(methods), value_range)

    palette = METHOD_COLORS["sensitivity"]
    for j, method in enumerate(methods):
        color = palette.get(method, "#34495e")
        x_positions = param_values + offsets[j]
        valid_mask = ~np.isnan(midpoint[:, j])

        ax.errorbar(
            x_positions[valid_mask],
            midpoint[valid_mask, j],
            yerr=halfwidth[valid_mask, j],
            fmt="o",
            color=color,
            label=method,
            capsize=7,
            capthick=2,
            linewidth=2,
            markersize=8,
            markeredgewidth=0,
            elinewidth=2,
            alpha=0.85,
            **kwargs,
        )

    if add_x_axis:
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.4, linewidth=1.5)

    ax.set_xticks(param_values)
    ax.set_xticklabels([f"{val:.3g}" for val in param_values], ha="center")

    ax.set_xlabel(r"$M$", fontsize=theme_obj.label_fontsize, fontweight="bold")
    ax.set_ylabel("Confidence Interval", fontsize=theme_obj.label_fontsize, fontweight="bold")
    ax.set_title("Sensitivity Analysis (Smoothness)", fontsize=theme_obj.title_fontsize, fontweight="bold", pad=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(len(methods), 5),
        frameon=True,
        fontsize=11,
    )

    fig = pc.viz["figure"]
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)

    return pc


def plot_sensitivity_rm(
    robust_results: pd.DataFrame,
    original_results,
    rescale_factor: float = 1,
    max_mbar: float = np.inf,
    add_x_axis: bool = True,
    figsize: tuple[float, float] = (12, 7),
    theme: str | PlotTheme = "default",
    **kwargs: Any,
) -> PlotCollection:
    r"""Create sensitivity plot for relative magnitude bounds.

    Visualizes how robust confidence intervals change as the relative
    magnitude parameter :math:`\bar{M}` increases. The parameter :math:`\bar{M}`
    bounds the ratio of post-treatment to pre-treatment trend differences.
    Values of :math:`\bar{M} = 1` mean post-treatment violations are no larger
    than pre-treatment violations.

    Parameters
    ----------
    robust_results : pd.DataFrame
        DataFrame from create_sensitivity_results_rm with
        columns: lb, ub, method, Delta, Mbar.
    original_results : NamedTuple
        Result from construct_original_cs with lb, ub, method.
    rescale_factor : float, default=1
        Factor to rescale all values for display.
    max_mbar : float, default=np.inf
        Maximum :math:`\bar{M}` value to display (after rescaling).
    add_x_axis : bool, default=True
        Whether to add horizontal line at y=0.
    figsize : tuple of float, default=(12, 7)
        Figure size (width, height).
    theme : str or PlotTheme, default="default"
        Theme name or custom PlotTheme object. Available themes: "default",
        "minimal", "publication", "colorful".
    **kwargs
        Additional keyword arguments passed to matplotlib functions.

    Returns
    -------
    PlotCollection
        PlotCollection object for method chaining. Access the figure via
        ``pc.viz["figure"]`` and axes via ``pc.viz["plot"]``.

    See Also
    --------
    plot_sensitivity_sm : Sensitivity plot for smoothness bounds.
    create_sensitivity_results_rm : Generate relative magnitude sensitivity results.
    honest_did : Main HonestDiD sensitivity analysis function.

    Examples
    --------
    Basic relative magnitude sensitivity plot:

    .. plot::
        :context: close-figs

        >>> from moderndid import att_gt, aggte, load_mpdta
        >>> from moderndid import plot_sensitivity_rm
        >>> from moderndid.didhonest import honest_did
        >>> df = load_mpdta()
        >>> gt_result = att_gt(
        ...     data=df,
        ...     yname="lemp",
        ...     tname="year",
        ...     gname="first.treat",
        ...     idname="countyreal",
        ...     bstrap=False,
        ... )
        >>> es_result = aggte(gt_result, type="dynamic")
        >>> hd_result = honest_did(
        ...     es_result,
        ...     event_time=0,
        ...     sensitivity_type="relative_magnitude",
        ...     m_bar_vec=[0.5, 1.0, 1.5, 2.0],
        ... )
        >>> pc = plot_sensitivity_rm(
        ...     hd_result.robust_ci,
        ...     hd_result.original_ci,
        ... )
    """
    theme_obj = _get_theme(theme)

    df = robust_results.copy()
    df["Mbar"] = df["Mbar"] * rescale_factor
    df["lb"] = df["lb"] * rescale_factor
    df["ub"] = df["ub"] * rescale_factor

    df = df[df["Mbar"] <= max_mbar]

    scaled_original = type(original_results)(
        lb=original_results.lb * rescale_factor,
        ub=original_results.ub * rescale_factor,
        method=original_results.method,
    )

    dataset = sensitivity_to_dataset(df, scaled_original, param_col="Mbar")

    pc = PlotCollection.grid(dataset, figure_kwargs={"figsize": figsize})
    apply_theme(pc, theme_obj)

    ax = pc.viz["plot"].values.item() if pc.viz["plot"].size == 1 else pc.viz["plot"].values

    param_values = dataset["lb"].coords["param_value"]
    methods = dataset["lb"].coords["method"]
    midpoint = dataset["midpoint"].values
    halfwidth = dataset["halfwidth"].values

    value_range = param_values[-1] - param_values[0] if len(param_values) > 1 else 0
    offsets = _calculate_offsets(len(methods), value_range)

    palette = METHOD_COLORS["sensitivity_rm"]
    for j, method in enumerate(methods):
        color = palette.get(method, "#34495e")
        x_positions = param_values + offsets[j]

        valid_mask = ~np.isnan(midpoint[:, j])

        ax.errorbar(
            x_positions[valid_mask],
            midpoint[valid_mask, j],
            yerr=halfwidth[valid_mask, j],
            fmt="o",
            color=color,
            label=method,
            capsize=7,
            capthick=2,
            linewidth=2,
            markersize=8,
            markeredgewidth=0,
            elinewidth=2,
            alpha=0.85,
            **kwargs,
        )

    if add_x_axis:
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.4, linewidth=1.5)

    ax.set_xticks(param_values)
    ax.set_xticklabels([f"{val:.3g}" for val in param_values], ha="center")

    ax.set_xlabel(r"$\bar{M}$", fontsize=theme_obj.label_fontsize, fontweight="bold")
    ax.set_ylabel("Confidence Interval", fontsize=theme_obj.label_fontsize, fontweight="bold")
    ax.set_title(
        "Sensitivity Analysis (Relative Magnitude)", fontsize=theme_obj.title_fontsize, fontweight="bold", pad=10
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(len(methods), 3),
        frameon=True,
        fontsize=11,
    )

    fig = pc.viz["figure"]
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)

    return pc


def plot_sensitivity_event_study(
    betahat: np.ndarray,
    std_errors: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
    num_pre_periods: int | None = None,
    num_post_periods: int | None = None,
    alpha: float = 0.05,
    time_vec: np.ndarray | None = None,
    reference_period: int | None = None,
    use_relative_event_time: bool = False,
    multiple_ci_data: list[dict] | None = None,
    figsize: tuple[float, float] = (14, 8),
    theme: str | PlotTheme = "default",
    **kwargs: Any,
) -> PlotCollection:
    """Create event study plot with confidence intervals.

    Parameters
    ----------
    betahat : ndarray
        Estimated event study coefficients.
    std_errors : ndarray, optional
        Standard errors for each coefficient. Either this or sigma must be provided.
    sigma : ndarray, optional
        Covariance matrix. Either this or std_errors must be provided.
    num_pre_periods : int, optional
        Number of pre-treatment periods. Required if time_vec not provided.
    num_post_periods : int, optional
        Number of post-treatment periods. Required if time_vec not provided.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    time_vec : ndarray, optional
        Time periods corresponding to coefficients.
    reference_period : int, optional
        Reference period to normalize to zero.
    use_relative_event_time : bool, default=False
        Whether to convert time periods to event time.
    multiple_ci_data : list of dict, optional
        Additional CI data to plot.
    figsize : tuple of float, default=(14, 8)
        Figure size (width, height).
    theme : str or PlotTheme, default="default"
        Theme name or custom PlotTheme object.
    **kwargs
        Additional keyword arguments passed to matplotlib functions.

    Returns
    -------
    PlotCollection
        PlotCollection object for method chaining.

    See Also
    --------
    plot_sensitivity_sm : Sensitivity plot for smoothness bounds.
    plot_sensitivity_rm : Sensitivity plot for relative magnitude bounds.
    honest_did : Main HonestDiD sensitivity analysis function.

    Examples
    --------
    Basic event study plot with confidence intervals:

    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> from moderndid import att_gt, aggte, load_mpdta
        >>> from moderndid import plot_sensitivity_event_study
        >>> df = load_mpdta()
        >>> gt_result = att_gt(
        ...     data=df,
        ...     yname="lemp",
        ...     tname="year",
        ...     gname="first.treat",
        ...     idname="countyreal",
        ...     bstrap=False,
        ... )
        >>> es_result = aggte(gt_result, type="dynamic")
        >>> betahat = es_result.att_by_event
        >>> std_errors = es_result.se_by_event
        >>> n_pre = np.sum(es_result.event_times < 0)
        >>> n_post = np.sum(es_result.event_times >= 0)
        >>> pc = plot_sensitivity_event_study(
        ...     betahat,
        ...     std_errors=std_errors,
        ...     num_pre_periods=n_pre,
        ...     num_post_periods=n_post,
        ... )

    Convert to relative event time centered at treatment:

    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> from moderndid import att_gt, aggte, load_mpdta
        >>> from moderndid import plot_sensitivity_event_study
        >>> df = load_mpdta()
        >>> gt_result = att_gt(
        ...     data=df,
        ...     yname="lemp",
        ...     tname="year",
        ...     gname="first.treat",
        ...     idname="countyreal",
        ...     bstrap=False,
        ... )
        >>> es_result = aggte(gt_result, type="dynamic")
        >>> betahat = es_result.att_by_event
        >>> std_errors = es_result.se_by_event
        >>> n_pre = np.sum(es_result.event_times < 0)
        >>> n_post = np.sum(es_result.event_times >= 0)
        >>> pc = plot_sensitivity_event_study(
        ...     betahat,
        ...     std_errors=std_errors,
        ...     num_pre_periods=n_pre,
        ...     num_post_periods=n_post,
        ...     use_relative_event_time=True,
        ... )
    """
    if std_errors is None and sigma is None:
        raise ValueError("Must specify either std_errors or sigma")

    if std_errors is None:
        std_errors = np.sqrt(np.diag(sigma))

    if num_pre_periods is None or num_post_periods is None:
        if time_vec is None:
            raise ValueError("Must provide either time_vec or both num_pre_periods and num_post_periods")
        total_periods = len(betahat)
        if reference_period is None:
            num_pre_periods = total_periods // 2
            num_post_periods = total_periods - num_pre_periods
        else:
            num_pre_periods = reference_period - 1
            num_post_periods = total_periods - num_pre_periods

    if time_vec is None:
        time_vec = np.arange(1, num_pre_periods + num_post_periods + 1)

    if reference_period is None:
        reference_period = num_pre_periods

    if use_relative_event_time:
        time_vec = time_vec - reference_period
        reference_period = 0

    plot_times = np.concatenate([time_vec[:num_pre_periods], [reference_period], time_vec[num_pre_periods:]])
    plot_betas = np.concatenate([betahat[:num_pre_periods], [0], betahat[num_pre_periods:]])
    plot_ses = np.concatenate([std_errors[:num_pre_periods], [np.nan], std_errors[num_pre_periods:]])

    theme_obj = _get_theme(theme)

    from moderndid.plotting.containers import Dataset

    coords = {"time": plot_times}
    data_vars = {
        "effect": {
            "values": plot_betas,
            "dims": ["time"],
            "coords": coords,
        },
        "se": {
            "values": plot_ses,
            "dims": ["time"],
            "coords": coords,
        },
    }
    dataset = Dataset(data_vars)

    pc = PlotCollection.grid(dataset, figure_kwargs={"figsize": figsize})
    apply_theme(pc, theme_obj)

    ax = pc.viz["plot"].values.item() if pc.viz["plot"].size == 1 else pc.viz["plot"].values

    z_crit = stats.norm.ppf(1 - alpha / 2)
    n_ci_sets = 1 + (len(multiple_ci_data) if multiple_ci_data else 0)

    if n_ci_sets > 1:
        time_range = plot_times.max() - plot_times.min()
        offset_amount = 0.08 * time_range / n_ci_sets
        offsets = np.linspace(-offset_amount * (n_ci_sets - 1) / 2, offset_amount * (n_ci_sets - 1) / 2, n_ci_sets)
    else:
        offsets = np.zeros(n_ci_sets)

    y_error = z_crit * plot_ses
    valid_mask = ~np.isnan(y_error)

    ax.errorbar(
        x=plot_times[valid_mask] + offsets[0],
        y=plot_betas[valid_mask],
        yerr=y_error[valid_mask],
        color=theme_obj.pre_treatment_color,
        fmt="none",
        capsize=5,
        capthick=1.5,
        elinewidth=2,
        alpha=0.8,
        **kwargs,
    )

    ax.scatter(
        plot_times + offsets[0],
        plot_betas,
        color=theme_obj.post_treatment_color,
        s=120,
        zorder=5,
        edgecolors="white",
        linewidth=2,
        label="Main",
    )

    if multiple_ci_data:
        additional_colors = ["#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]
        for i, ci_data in enumerate(multiple_ci_data):
            add_betahat = ci_data["betahat"]
            if "std_errors" in ci_data:
                add_ses = ci_data["std_errors"]
            else:
                add_ses = np.sqrt(np.diag(ci_data["sigma"]))

            add_plot_betas = np.concatenate([add_betahat[:num_pre_periods], [0], add_betahat[num_pre_periods:]])
            add_plot_ses = np.concatenate([add_ses[:num_pre_periods], [np.nan], add_ses[num_pre_periods:]])

            color = ci_data.get("color", additional_colors[i % len(additional_colors)])
            label = ci_data.get("label", f"CI {i + 1}")

            add_y_error = z_crit * add_plot_ses
            add_valid_mask = ~np.isnan(add_y_error)

            ax.errorbar(
                x=plot_times[add_valid_mask] + offsets[i + 1],
                y=add_plot_betas[add_valid_mask],
                yerr=add_y_error[add_valid_mask],
                color=color,
                fmt="none",
                capsize=5,
                capthick=1.5,
                elinewidth=2,
                alpha=0.8,
            )

            ax.scatter(
                plot_times + offsets[i + 1],
                add_plot_betas,
                color=color,
                s=120,
                zorder=5,
                edgecolors="white",
                linewidth=2,
                label=label,
            )

    ax.axhline(y=0, color="black", linestyle="-", alpha=0.5, linewidth=2)

    treatment_time = reference_period + 0.5 if not use_relative_event_time else 0.5
    ax.axvline(x=treatment_time, color="gray", linestyle="--", alpha=0.6, linewidth=2, label="Treatment")

    xlabel = "Event Time" if use_relative_event_time else "Time Period"
    ax.set_xlabel(xlabel, fontsize=theme_obj.label_fontsize, fontweight="bold")
    ax.set_ylabel("Treatment Effect", fontsize=theme_obj.label_fontsize, fontweight="bold")
    ax.set_title("Event Study", fontsize=theme_obj.title_fontsize, fontweight="bold", pad=15)

    time_range_vals = plot_times[~np.isnan(plot_betas)]
    ax.set_xticks(np.arange(np.min(time_range_vals), np.max(time_range_vals) + 1, 1))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if n_ci_sets > 1 or reference_period > 1:
        ax.legend(loc="upper left", frameon=True, fontsize=11)

    fig = pc.viz["figure"]
    fig.tight_layout()

    return pc


def _get_theme(theme):
    """Resolve theme from string or PlotTheme object."""
    if isinstance(theme, str):
        return THEMES.get(theme, THEMES["default"])
    return theme


def _calculate_offsets(n_series, value_range):
    """Calculate x-axis offsets for multiple series."""
    if n_series > 1:
        if value_range > 0:
            offset_amount = 0.08 * value_range / n_series
        else:
            offset_amount = 0.15
        return np.linspace(
            -offset_amount * (n_series - 1) / 2,
            offset_amount * (n_series - 1) / 2,
            n_series,
        )
    return np.zeros(n_series)
