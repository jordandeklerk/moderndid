"""Plotting functions for sensitivity analysis results."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def plot_sensitivity(
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
        DataFrame from create_sensitivity_results with columns:
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
    sns.set_context("paper", font_scale=1.3)

    m_values = robust_results["M"].unique()
    m_gap = np.min(np.diff(np.sort(m_values))) if len(m_values) > 1 else 1
    m_min = np.min(m_values)

    original_results = original_results.copy()
    original_results["M"] = m_min - m_gap

    df = pd.concat([original_results, robust_results], ignore_index=True)

    df["M"] = df["M"] * rescale_factor
    df["lb"] = df["lb"] * rescale_factor
    df["ub"] = df["ub"] * rescale_factor

    df = df[df["M"] <= max_m]

    fig, ax = plt.subplots(figsize=(12, 7))

    methods = df["method"].unique()
    palette = {"Original": "#e74c3c", "FLCI": "#3498db", "Conditional": "#2ecc71", "C-F": "#9b59b6", "C-LF": "#f39c12"}

    for method in methods:
        method_df = df[df["method"] == method]
        color = palette.get(method, "#34495e")

        ax.errorbar(
            method_df["M"],
            (method_df["lb"] + method_df["ub"]) / 2,
            yerr=(method_df["ub"] - method_df["lb"]) / 2,
            fmt="o",
            color=color,
            capsize=8,
            capthick=2.5,
            label=method,
            linewidth=2.5,
            markersize=8,
            markeredgewidth=0,
            elinewidth=2.5,
            alpha=0.85,
        )

    if add_x_axis:
        ax.axhline(y=0, color="#2c3e50", linestyle="-", alpha=0.4, linewidth=1.5)

    ax.set_xlabel("M", fontsize=14, fontweight="bold")
    ax.set_ylabel("Confidence Interval", fontsize=14, fontweight="bold")

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(len(methods), 5),
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=12,
    )
    legend.get_frame().set_alpha(0.9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    ax.grid(True, alpha=0.2, linestyle="--", linewidth=1)
    ax.set_axisbelow(True)

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
        DataFrame from create_sensitivity_results_relative_magnitudes with
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
    sns.set_context("paper", font_scale=1.3)

    mbar_values = robust_results["Mbar"].unique()
    mbar_gap = np.min(np.diff(np.sort(mbar_values))) if len(mbar_values) > 1 else 0.2
    mbar_min = np.min(mbar_values)

    original_results = original_results.copy()
    original_results["Mbar"] = mbar_min - mbar_gap

    df = pd.concat([original_results, robust_results], ignore_index=True)

    df["Mbar"] = df["Mbar"] * rescale_factor
    df["lb"] = df["lb"] * rescale_factor
    df["ub"] = df["ub"] * rescale_factor

    df = df[df["Mbar"] <= max_mbar]

    fig, ax = plt.subplots(figsize=(12, 7))

    methods = df["method"].unique()
    palette = {"Original": "#e74c3c", "Conditional": "#3498db", "C-LF": "#2ecc71"}

    for method in methods:
        method_df = df[df["method"] == method]
        color = palette.get(method, "#34495e")

        ax.errorbar(
            method_df["Mbar"],
            (method_df["lb"] + method_df["ub"]) / 2,
            yerr=(method_df["ub"] - method_df["lb"]) / 2,
            fmt="s",
            color=color,
            capsize=8,
            capthick=2.5,
            label=method,
            linewidth=2.5,
            markersize=8,
            markeredgewidth=0,
            elinewidth=2.5,
            alpha=0.85,
        )

    if add_x_axis:
        ax.axhline(y=0, color="#2c3e50", linestyle="-", alpha=0.4, linewidth=1.5)

    ax.set_xlabel(r"$\bar{M}$", fontsize=14, fontweight="bold")
    ax.set_ylabel("Confidence Interval", fontsize=14, fontweight="bold")

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(len(methods), 3),
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=12,
    )
    legend.get_frame().set_alpha(0.9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    ax.grid(True, alpha=0.2, linestyle="--", linewidth=1)
    ax.set_axisbelow(True)

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

    Returns
    -------
    matplotlib.figure.Figure
        The event study plot figure.
    """
    sns.set_style("white")
    sns.set_context("talk", font_scale=1.1)

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

    fig, ax = plt.subplots(figsize=(14, 8))

    z_crit = stats.norm.ppf(1 - alpha / 2)

    for t, beta, se in zip(plot_times, plot_betas, plot_ses):
        if not np.isnan(se):
            ci_lower = beta - z_crit * se
            ci_upper = beta + z_crit * se
            ax.plot([t, t], [ci_lower, ci_upper], color="#3498db", linewidth=3, alpha=0.6, solid_capstyle="round")

    ax.scatter(plot_times, plot_betas, color="#e74c3c", s=120, zorder=5, edgecolors="white", linewidth=2)

    ax.axhline(y=0, color="#2c3e50", linestyle="-", alpha=0.5, linewidth=2)

    treatment_time = reference_period + 0.5 if not use_relative_event_time else 0.5
    ax.axvline(x=treatment_time, color="#7f8c8d", linestyle="--", alpha=0.6, linewidth=2, label="Treatment")

    ax.set_xlabel(
        "Event Time" if use_relative_event_time else "Time Period", fontsize=15, fontweight="bold", color="#2c3e50"
    )
    ax.set_ylabel("Treatment Effect", fontsize=15, fontweight="bold", color="#2c3e50")

    time_range = plot_times[~np.isnan(plot_betas)]
    ax.set_xticks(np.arange(np.min(time_range), np.max(time_range) + 1, 1))
    ax.tick_params(axis="both", which="major", labelsize=12, colors="#2c3e50")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_color("#2c3e50")
    ax.spines["bottom"].set_color("#2c3e50")

    ax.grid(True, alpha=0.15, linestyle="--", linewidth=1, color="#95a5a6")
    ax.set_axisbelow(True)

    ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True, fontsize=12)

    plt.tight_layout()
    return fig
