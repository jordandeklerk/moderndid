"""Plotting functions for sensitivity analysis."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

mpl.rcParams["errorbar.capsize"] = 7

PLOT_CONFIG = {
    "figure_size": (12, 7),
    "figure_size_event_study": (14, 8),
    "font_scale": 1.3,
    "font_scale_event_study": 1.1,
    "label_fontsize": 15,
    "tick_labelsize": 12,
    "axis_color": "#2c3e50",
    "spine_linewidth": 1.5,
    "spine_linewidth_event_study": 2,
    "errorbar_style": {
        "fmt": "o",
        "capsize": 7,
        "capthick": 2,
        "linewidth": 2,
        "markersize": 8,
        "markeredgewidth": 0,
        "elinewidth": 2,
        "alpha": 0.85,
    },
    "event_study_ci_style": {
        "fmt": "none",
        "capsize": 5,
        "capthick": 1.5,
        "elinewidth": 2,
        "alpha": 0.8,
    },
    "legend_style": {
        "loc": "best",
        "frameon": True,
        "fancybox": False,
        "shadow": False,
        "fontsize": 11,
        "framealpha": 0.95,
        "edgecolor": "#e0e0e0",
        "facecolor": "white",
        "borderpad": 0.8,
        "columnspacing": 1.5,
        "handlelength": 1.5,
        "handletextpad": 0.8,
        "borderaxespad": 0.8,
    },
    "legend_style_bottom": {
        "loc": "upper center",
        "bbox_to_anchor": (0.5, -0.15),
        "frameon": True,
        "fancybox": False,
        "shadow": False,
        "fontsize": 11,
        "framealpha": 0.95,
        "edgecolor": "#e0e0e0",
        "facecolor": "white",
        "borderpad": 0.8,
        "columnspacing": 1.5,
        "handlelength": 1.5,
        "handletextpad": 0.8,
    },
}

COLOR_PALETTES = {
    "sensitivity": {
        "Original": "blue",
        "FLCI": "red",
        "Conditional": "red",
        "C-F": "red",
        "C-LF": "red",
    },
    "sensitivity_rm": {
        "Original": "blue",
        "Conditional": "red",
        "C-LF": "red",
    },
    "event_study": {
        "main": "red",
        "ci": "blue",
        "additional": ["#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"],
        "zero_line": "#2c3e50",
        "treatment_line": "#7f8c8d",
    },
}


def plot_sensitivity_sm(
    robust_results,
    original_results,
    rescale_factor=1,
    max_m=np.inf,
    add_x_axis=True,
):
    """Create sensitivity plot showing how confidence intervals change with :math:`M`.

    Creates a plot showing confidence intervals for different values of the
    smoothness parameter :math:`M`, comparing robust methods to the original
    (non-robust) confidence interval.

    Parameters
    ----------
    robust_results : pd.DataFrame
        DataFrame from create_sensitivity_results_sm with columns:
        lb, ub, method, Delta, M.
    original_results : pd.DataFrame
        DataFrame from construct_original_cs with columns:
        lb, ub, method.
    rescale_factor : float, default=1
        Factor to rescale all values (M, lb, ub) for display.
    max_m : float, default=np.inf
        Maximum M value to display (after rescaling).
    add_x_axis : bool, default=True
        Whether to add horizontal line at y=0.

    Returns
    -------
    matplotlib.figure.Figure
        The sensitivity plot figure.
    """
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=PLOT_CONFIG["font_scale"])

    m_col = "M" if "M" in robust_results.columns else "m"

    m_values = robust_results[m_col].unique()
    m_gap = np.min(np.diff(np.sort(m_values))) if len(m_values) > 1 else 1
    m_min = np.min(m_values)

    original_df = pd.DataFrame([original_results._asdict()])
    original_df[m_col] = m_min - m_gap

    df = pd.concat([original_df, robust_results], ignore_index=True)

    df[m_col] = df[m_col] * rescale_factor
    df["lb"] = df["lb"] * rescale_factor
    df["ub"] = df["ub"] * rescale_factor

    df = df[df[m_col] <= max_m]

    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figure_size"])

    methods = df["method"].unique()
    palette = COLOR_PALETTES["sensitivity"]

    m_values_unique = np.sort(df[m_col].unique())
    value_range = m_values_unique[-1] - m_values_unique[0] if len(m_values_unique) > 1 else 0
    offsets = _calculate_offsets(len(methods), value_range)

    for i, method in enumerate(methods):
        method_df = df[df["method"] == method]
        color = palette.get(method, "#34495e")

        x_positions = method_df[m_col].values + offsets[i]

        errorbar = ax.errorbar(
            x_positions,
            (method_df["lb"] + method_df["ub"]) / 2,
            yerr=(method_df["ub"] - method_df["lb"]) / 2,
            color=color,
            label=method,
            **PLOT_CONFIG["errorbar_style"],
        )

        if len(errorbar) > 1 and errorbar[1] is not None:
            for cap in errorbar[1]:
                cap.set_markersize(10)
                cap.set_markeredgewidth(2)

    if add_x_axis:
        ax.axhline(y=0, color=PLOT_CONFIG["axis_color"], linestyle="-", alpha=0.4, linewidth=1.5)

    ax.set_xticks(m_values_unique)
    ax.set_xticklabels([f"{val:.3g}" for val in m_values_unique], ha="center")

    if len(m_values_unique) > 1:
        minor_ticks = []
        for i in range(len(m_values_unique) - 1):
            minor_ticks.append((m_values_unique[i] + m_values_unique[i + 1]) / 2)
        ax.set_xticks(minor_ticks, minor=True)

    ax.tick_params(axis="x", which="major", length=8, width=2, color=PLOT_CONFIG["axis_color"], direction="out")
    ax.tick_params(axis="x", which="minor", length=4, width=1, color=PLOT_CONFIG["axis_color"], direction="out")

    ax.set_xlabel(r"$M$", fontsize=PLOT_CONFIG["label_fontsize"], fontweight="bold", color=PLOT_CONFIG["axis_color"])
    ax.set_ylabel(
        "Confidence Interval",
        fontsize=PLOT_CONFIG["label_fontsize"],
        fontweight="bold",
        color=PLOT_CONFIG["axis_color"],
    )

    legend_config = PLOT_CONFIG["legend_style_bottom"].copy()
    legend_config["ncol"] = min(len(methods), 5)

    legend = ax.legend(**legend_config)

    legend.get_frame().set_linewidth(0.8)
    for text in legend.get_texts():
        text.set_color("#2c3e50")

    _apply_axis_styling(ax)

    plt.tight_layout()
    return fig


def plot_sensitivity_rm(
    robust_results,
    original_results,
    rescale_factor=1,
    max_mbar=np.inf,
    add_x_axis=True,
):
    r"""Create sensitivity plot for relative magnitude bounds.

    Creates a plot showing confidence intervals for different values of the
    relative magnitude parameter :math:`\bar{M}`.

    Parameters
    ----------
    robust_results : pd.DataFrame
        DataFrame from create_sensitivity_results_rm with
        columns: lb, ub, method, Delta, Mbar.
    original_results : pd.DataFrame
        DataFrame from construct_original_cs with columns:
        lb, ub, method.
    rescale_factor : float, default=1
        Factor to rescale all values for display.
    max_mbar : float, default=np.inf
        Maximum :math:`\bar{M}` value to display (after rescaling).
    add_x_axis : bool, default=True
        Whether to add horizontal line at y=0.

    Returns
    -------
    matplotlib.figure.Figure
        The sensitivity plot figure.
    """
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=PLOT_CONFIG["font_scale"])

    mbar_values = robust_results["Mbar"].unique()
    mbar_gap = np.min(np.diff(np.sort(mbar_values))) if len(mbar_values) > 1 else 0.2
    mbar_min = np.min(mbar_values)

    original_df = pd.DataFrame([original_results._asdict()])
    original_df["Mbar"] = mbar_min - mbar_gap

    df = pd.concat([original_df, robust_results], ignore_index=True)

    df["Mbar"] = df["Mbar"] * rescale_factor
    df["lb"] = df["lb"] * rescale_factor
    df["ub"] = df["ub"] * rescale_factor

    df = df[df["Mbar"] <= max_mbar]

    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figure_size"])

    methods = df["method"].unique()
    palette = COLOR_PALETTES["sensitivity_rm"]

    mbar_values_unique = np.sort(df["Mbar"].unique())
    value_range = mbar_values_unique[-1] - mbar_values_unique[0] if len(mbar_values_unique) > 1 else 0
    offsets = _calculate_offsets(len(methods), value_range)

    for i, method in enumerate(methods):
        method_df = df[df["method"] == method]
        color = palette.get(method, "#34495e")

        x_positions = method_df["Mbar"].values + offsets[i]

        errorbar = ax.errorbar(
            x_positions,
            (method_df["lb"] + method_df["ub"]) / 2,
            yerr=(method_df["ub"] - method_df["lb"]) / 2,
            color=color,
            label=method,
            **PLOT_CONFIG["errorbar_style"],
        )

        if len(errorbar) > 1 and errorbar[1] is not None:
            for cap in errorbar[1]:
                cap.set_markersize(10)
                cap.set_markeredgewidth(2)

    if add_x_axis:
        ax.axhline(y=0, color=PLOT_CONFIG["axis_color"], linestyle="-", alpha=0.4, linewidth=1.5)

    ax.set_xticks(mbar_values_unique)
    ax.set_xticklabels([f"{val:.3g}" for val in mbar_values_unique], ha="center")

    if len(mbar_values_unique) > 1:
        minor_ticks = []
        for i in range(len(mbar_values_unique) - 1):
            minor_ticks.append((mbar_values_unique[i] + mbar_values_unique[i + 1]) / 2)
        ax.set_xticks(minor_ticks, minor=True)

    ax.tick_params(axis="x", which="major", length=8, width=2, color=PLOT_CONFIG["axis_color"], direction="out")
    ax.tick_params(axis="x", which="minor", length=4, width=1, color=PLOT_CONFIG["axis_color"], direction="out")

    ax.set_xlabel(
        r"$\bar{M}$", fontsize=PLOT_CONFIG["label_fontsize"], fontweight="bold", color=PLOT_CONFIG["axis_color"]
    )
    ax.set_ylabel(
        "Confidence Interval",
        fontsize=PLOT_CONFIG["label_fontsize"],
        fontweight="bold",
        color=PLOT_CONFIG["axis_color"],
    )

    legend_config = PLOT_CONFIG["legend_style_bottom"].copy()
    legend_config["ncol"] = min(len(methods), 3)

    legend = ax.legend(**legend_config)

    legend.get_frame().set_linewidth(0.8)
    for text in legend.get_texts():
        text.set_color("#2c3e50")

    _apply_axis_styling(ax)

    plt.tight_layout()
    return fig


def event_study_plot(
    betahat,
    std_errors=None,
    sigma=None,
    num_pre_periods=None,
    num_post_periods=None,
    alpha=0.05,
    time_vec=None,
    reference_period=None,
    use_relative_event_time=False,
    multiple_ci_data=None,
):
    """Create event study plot with confidence intervals.

    Creates a standard event study plot showing estimated coefficients and
    confidence intervals over time, with a reference period normalized to zero.

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
        Time periods corresponding to coefficients. If not provided, uses
        integers from 1 to num_pre_periods + num_post_periods.
    reference_period : int, optional
        Reference period to normalize to zero. If not provided, uses the
        period just before treatment.
    use_relative_event_time : bool, default=False
        Whether to convert time periods to event time (relative to reference).
    multiple_ci_data : list of dict, optional
        List of additional CI data to plot. Each dict should have keys:
        'betahat', 'std_errors' or 'sigma', 'label', 'color'.

    Returns
    -------
    matplotlib.figure.Figure
        The event study plot figure.
    """
    sns.set_style("white")
    sns.set_context("talk", font_scale=PLOT_CONFIG["font_scale_event_study"])

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

    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figure_size_event_study"])

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

    event_errorbar_style = PLOT_CONFIG["event_study_ci_style"].copy()
    event_errorbar_style["fmt"] = "none"

    errorbar_lines = ax.errorbar(
        x=plot_times[valid_mask] + offsets[0],
        y=plot_betas[valid_mask],
        yerr=y_error[valid_mask],
        color=COLOR_PALETTES["event_study"]["ci"],
        **event_errorbar_style,
    )

    for cap in errorbar_lines[1]:
        cap.set_markeredgewidth(1.5)

    ax.scatter(
        plot_times + offsets[0],
        plot_betas,
        color=COLOR_PALETTES["event_study"]["main"],
        s=120,
        zorder=5,
        edgecolors="white",
        linewidth=2,
        label="Main",
    )

    if multiple_ci_data:
        default_colors = COLOR_PALETTES["event_study"]["additional"]
        for i, ci_data in enumerate(multiple_ci_data):
            add_betahat = ci_data["betahat"]
            if "std_errors" in ci_data:
                add_ses = ci_data["std_errors"]
            else:
                add_ses = np.sqrt(np.diag(ci_data["sigma"]))

            add_plot_betas = np.concatenate([add_betahat[:num_pre_periods], [0], add_betahat[num_pre_periods:]])
            add_plot_ses = np.concatenate([add_ses[:num_pre_periods], [np.nan], add_ses[num_pre_periods:]])

            color = ci_data.get("color", default_colors[i % len(default_colors)])
            label = ci_data.get("label", f"CI {i + 1}")

            add_y_error = z_crit * add_plot_ses
            add_valid_mask = ~np.isnan(add_y_error)

            add_errorbar_lines = ax.errorbar(
                x=plot_times[add_valid_mask] + offsets[i + 1],
                y=add_plot_betas[add_valid_mask],
                yerr=add_y_error[add_valid_mask],
                color=color,
                **event_errorbar_style,
            )

            for cap in add_errorbar_lines[1]:
                cap.set_markeredgewidth(1.5)

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

    ax.axhline(y=0, color=COLOR_PALETTES["event_study"]["zero_line"], linestyle="-", alpha=0.5, linewidth=2)

    treatment_time = reference_period + 0.5 if not use_relative_event_time else 0.5
    ax.axvline(
        x=treatment_time,
        color=COLOR_PALETTES["event_study"]["treatment_line"],
        linestyle="--",
        alpha=0.6,
        linewidth=2,
        label="Treatment",
    )

    ax.set_xlabel(
        "Event Time" if use_relative_event_time else "Time Period",
        fontsize=PLOT_CONFIG["label_fontsize"],
        fontweight="bold",
        color=PLOT_CONFIG["axis_color"],
    )
    ax.set_ylabel(
        "Treatment Effect", fontsize=PLOT_CONFIG["label_fontsize"], fontweight="bold", color=PLOT_CONFIG["axis_color"]
    )

    time_range = plot_times[~np.isnan(plot_betas)]
    ax.set_xticks(np.arange(np.min(time_range), np.max(time_range) + 1, 1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(PLOT_CONFIG["spine_linewidth_event_study"])
    ax.spines["bottom"].set_linewidth(PLOT_CONFIG["spine_linewidth_event_study"])
    ax.spines["left"].set_color(PLOT_CONFIG["axis_color"])
    ax.spines["bottom"].set_color(PLOT_CONFIG["axis_color"])
    ax.grid(False)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=PLOT_CONFIG["tick_labelsize"],
        colors=PLOT_CONFIG["axis_color"],
    )

    if n_ci_sets > 1 or reference_period > 1:
        legend_config = PLOT_CONFIG["legend_style"].copy()
        legend_config["loc"] = "upper left"
        legend_config["ncol"] = 1

        legend = ax.legend(**legend_config)

        legend.get_frame().set_linewidth(0.8)
        for text in legend.get_texts():
            text.set_color("#2c3e50")

    plt.tight_layout()
    return fig


def _apply_axis_styling(ax):
    """Apply axis styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(PLOT_CONFIG["spine_linewidth"])
    ax.spines["bottom"].set_linewidth(PLOT_CONFIG["spine_linewidth"])
    ax.spines["left"].set_color(PLOT_CONFIG["axis_color"])
    ax.spines["bottom"].set_color(PLOT_CONFIG["axis_color"])
    ax.grid(False)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=PLOT_CONFIG["tick_labelsize"],
        colors=PLOT_CONFIG["axis_color"],
    )


def _calculate_offsets(n_series, value_range):
    """Calculate x-axis offsets."""
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
