# pylint: disable=unused-argument
"""Plotting functions for continuous treatment DiD results."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d

from moderndid.didcont.panel.container import DoseResult, PTEResult

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
    },
}

COLOR_PALETTES = {
    "dose": {
        "main": "#3498db",
        "ci": "#95a5a6",
        "zero_line": "#2c3e50",
        "overall": "#e74c3c",
    },
    "event_study": {
        "main": "#e74c3c",
        "ci": "#3498db",
        "zero_line": "#2c3e50",
        "treatment_line": "#7f8c8d",
    },
}


def plot_cont_did(
    dose_obj: DoseResult | PTEResult,
    type="att",
    alpha=0.05,
    show_overall=False,
    show_confidence_bands=True,
    rescale_factor=1.0,
    **kwargs,
):
    """Plot results from continuous treatment DiD estimation.

    Creates visualizations for continuous treatment DiD results, either
    showing treatment effects as a function of dose or as an event study.

    Parameters
    ----------
    dose_obj : DoseResult or PTEResult
        Result object from cont_did function.
    type : {"att", "acrt"}, default="att"
        Type of effect to plot:

        - "att": Average Treatment Effect on Treated as function of dose
        - "acrt": Average Causal Response on Treated (derivative of dose-response)
    alpha : float, default=0.05
        Significance level for confidence intervals.
    show_overall : bool, default=False
        Whether to show the overall treatment effect as a horizontal line.
    show_confidence_bands : bool, default=True
        Whether to display confidence bands around the estimates.
    rescale_factor : float, default=1.0
        Factor to rescale dose and effect values for display.
    **kwargs
        Additional keyword arguments passed to matplotlib functions.

    Returns
    -------
    Figure
        Matplotlib figure object containing the plot.
    """
    if hasattr(dose_obj, "event_study") and dose_obj.event_study is not None:
        return _plot_pte_event_study(dose_obj, type=type, alpha=alpha, **kwargs)

    if not hasattr(dose_obj, "dose"):
        raise ValueError("Input must be a DoseResult or PTEResult object from cont_did")

    return _plot_dose_response(
        dose_obj,
        type=type,
        alpha=alpha,
        show_overall=show_overall,
        show_confidence_bands=show_confidence_bands,
        rescale_factor=rescale_factor,
        **kwargs,
    )


def _plot_dose_response(
    dose_result: DoseResult,
    type="att",
    alpha=0.05,
    show_overall=False,
    show_confidence_bands=True,
    rescale_factor=1.0,
    smooth=True,
    **kwargs,
):
    """Plot dose-response function for continuous treatment."""
    sns.set_style("white")
    sns.set_context("notebook", font_scale=1.2)

    if type == "att":
        dose_values = dose_result.dose * rescale_factor
        effects = dose_result.att_d * rescale_factor
        se = dose_result.att_d_se * rescale_factor if dose_result.att_d_se is not None else None
        crit_val = dose_result.att_d_crit_val
        overall_effect = dose_result.overall_att * rescale_factor if show_overall else None
        ylabel = "att" if type == "att" else "ATT(d)"
    elif type == "acrt":
        dose_values = dose_result.dose * rescale_factor
        effects = dose_result.acrt_d * rescale_factor
        se = dose_result.acrt_d_se * rescale_factor if dose_result.acrt_d_se is not None else None
        crit_val = dose_result.acrt_d_crit_val
        overall_effect = dose_result.overall_acrt * rescale_factor if show_overall else None
        ylabel = "acrt" if type == "acrt" else "ACRT(d)"
    else:
        raise ValueError(f"Invalid type: {type}. Must be 'att' or 'acrt'")

    fig, ax = plt.subplots(figsize=(10, 6))

    if smooth and len(dose_values) > 3:
        dose_fine = np.linspace(dose_values.min(), dose_values.max(), 200)

        f_effects = interp1d(dose_values, effects, kind="cubic", fill_value="extrapolate")
        effects_smooth = f_effects(dose_fine)

        if show_confidence_bands and se is not None:
            f_se = interp1d(dose_values, se, kind="cubic", fill_value="extrapolate")
            se_smooth = f_se(dose_fine)

            if crit_val is not None and not np.isnan(crit_val):
                z_crit = crit_val
            else:
                z_crit = stats.norm.ppf(1 - alpha / 2)

            lower_bound = effects_smooth - z_crit * se_smooth
            upper_bound = effects_smooth + z_crit * se_smooth

            ax.fill_between(
                dose_fine,
                lower_bound,
                upper_bound,
                color="#CCCCCC",
                alpha=0.7,
                linewidth=0,
                edgecolor="none",
            )

        ax.plot(
            dose_fine,
            effects_smooth,
            color="#404040",
            linewidth=2.5,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
    else:
        if show_confidence_bands and se is not None:
            if crit_val is not None and not np.isnan(crit_val):
                z_crit = crit_val
            else:
                z_crit = stats.norm.ppf(1 - alpha / 2)

            lower_bound = effects - z_crit * se
            upper_bound = effects + z_crit * se

            ax.fill_between(
                dose_values,
                lower_bound,
                upper_bound,
                color="#CCCCCC",
                alpha=0.7,
                linewidth=0,
            )

        ax.plot(
            dose_values,
            effects,
            color="#404040",
            linewidth=2.5,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

    if overall_effect is not None and show_overall:
        ax.axhline(
            y=overall_effect,
            color="#E74C3C",
            linestyle="--",
            linewidth=1.5,
            alpha=0.6,
        )

    ax.axhline(y=0, color="#666666", linestyle="-", alpha=0.3, linewidth=0.8)

    ax.set_xlabel("dose", fontsize=12, color="#333333")
    ax.set_ylabel(ylabel, fontsize=12, color="#333333")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")

    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5, color="#CCCCCC")
    ax.set_axisbelow(True)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=10,
        colors="#666666",
        length=4,
        width=0.8,
    )

    ax.set_xlim(dose_values.min(), dose_values.max())

    plt.tight_layout()
    return fig


def _plot_pte_event_study(
    pte_result: PTEResult,
    type="att",
    alpha=0.05,
    **kwargs,
):
    """Plot event study for continuous treatment PTEResult."""
    sns.set_style("white")
    sns.set_context("talk", font_scale=PLOT_CONFIG["font_scale_event_study"])

    if not hasattr(pte_result, "event_study") or pte_result.event_study is None:
        raise ValueError("PTEResult does not contain event study results")

    event_study = pte_result.event_study

    event_times = event_study.event_times
    effects = event_study.att_by_event
    se = event_study.se_by_event

    is_acrt = (
        hasattr(pte_result, "ptep")
        and hasattr(pte_result.ptep, "target_parameter")
        and pte_result.ptep.target_parameter == "slope"
    )

    ylabel = "ACRT" if is_acrt or type == "acrt" else "ATT"

    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figure_size_event_study"])

    z_crit = stats.norm.ppf(1 - alpha / 2)
    if hasattr(event_study, "critical_value") and event_study.critical_value is not None:
        z_crit = event_study.critical_value

    ax.errorbar(
        event_times,
        effects,
        yerr=z_crit * se,
        fmt="none",
        ecolor=COLOR_PALETTES["event_study"]["ci"],
        elinewidth=2,
        capsize=5,
        capthick=1.5,
        alpha=0.8,
    )

    ax.scatter(
        event_times,
        effects,
        color=COLOR_PALETTES["event_study"]["main"],
        s=120,
        zorder=5,
        edgecolors="white",
        linewidth=2,
        label=ylabel,
    )

    ax.axhline(y=0, color=COLOR_PALETTES["event_study"]["zero_line"], linestyle="-", alpha=0.5, linewidth=2)

    if event_times.min() < 0 < event_times.max():
        ax.axvline(
            x=-0.5,
            color=COLOR_PALETTES["event_study"]["treatment_line"],
            linestyle="--",
            alpha=0.6,
            linewidth=2,
            label="Treatment",
        )

    ax.set_xlabel(
        "Event Time", fontsize=PLOT_CONFIG["label_fontsize"], fontweight="bold", color=PLOT_CONFIG["axis_color"]
    )
    ax.set_ylabel(ylabel, fontsize=PLOT_CONFIG["label_fontsize"], fontweight="bold", color=PLOT_CONFIG["axis_color"])
    ax.set_title(
        f"Event Study: {ylabel} Over Time",
        fontsize=PLOT_CONFIG["label_fontsize"] + 2,
        fontweight="bold",
        color=PLOT_CONFIG["axis_color"],
        pad=20,
    )

    ax.set_xticks(event_times)

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

    legend = ax.legend(**PLOT_CONFIG["legend_style"])
    legend.get_frame().set_linewidth(0.8)
    for text in legend.get_texts():
        text.set_color("#2c3e50")

    plt.tight_layout()
    return fig


def _apply_axis_styling(ax):
    """Apply consistent axis styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(PLOT_CONFIG["spine_linewidth"])
    ax.spines["bottom"].set_linewidth(PLOT_CONFIG["spine_linewidth"])
    ax.spines["left"].set_color(PLOT_CONFIG["axis_color"])
    ax.spines["bottom"].set_color(PLOT_CONFIG["axis_color"])
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=PLOT_CONFIG["tick_labelsize"],
        colors=PLOT_CONFIG["axis_color"],
    )
