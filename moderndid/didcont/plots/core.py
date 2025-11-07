# pylint: disable=unused-argument
"""Plotting functions for continuous treatment DiD results."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d

from moderndid.didcont.estimation.container import DoseResult, PTEResult

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
        "main": "red",
        "ci": "blue",
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
    ax=None,
    **kwargs,
):
    """Plot results from continuous treatment DiD estimation.

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
        return _plot_pte_event_study(dose_obj, type=type, alpha=alpha, ax=ax, **kwargs)

    if not hasattr(dose_obj, "dose"):
        raise ValueError("Input must be a DoseResult or PTEResult object from cont_did")

    return _plot_dose_response(
        dose_obj,
        type=type,
        alpha=alpha,
        show_overall=show_overall,
        show_confidence_bands=show_confidence_bands,
        rescale_factor=rescale_factor,
        ax=ax,
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
    smooth_line=None,
    smooth_band=True,
    ax=None,
    **kwargs,
):
    """Plot dose-response function for continuous treatment."""
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.15)

    if type == "att":
        dose_values = dose_result.dose * rescale_factor
        effects = dose_result.att_d * rescale_factor
        se = dose_result.att_d_se * rescale_factor if getattr(dose_result, "att_d_se", None) is not None else None
        if (se is None) or (not np.all(np.isfinite(se))) or np.allclose(se, 0):
            se = _se_from_inf(getattr(dose_result, "att_d_inf_func", None))

            if se is not None:
                se = se * rescale_factor
        crit_val = dose_result.att_d_crit_val
        overall_effect = dose_result.overall_att * rescale_factor if show_overall else None
        ylabel = "att d"
    elif type == "acrt":
        dose_values = dose_result.dose * rescale_factor
        effects = dose_result.acrt_d * rescale_factor
        se = dose_result.acrt_d_se * rescale_factor if getattr(dose_result, "acrt_d_se", None) is not None else None

        if (se is None) or (not np.all(np.isfinite(se))) or np.allclose(se, 0):
            se = _se_from_inf(getattr(dose_result, "acrt_d_inf_func", None))
            if se is not None:
                se = se * rescale_factor
        crit_val = dose_result.acrt_d_crit_val
        overall_effect = dose_result.overall_acrt * rescale_factor if show_overall else None
        ylabel = "acrt d"
    else:
        raise ValueError(f"Invalid type: {type}. Must be 'att' or 'acrt'")

    created_here = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        created_here = True
    else:
        fig = ax.figure

    if smooth_line is None:
        smooth_line = smooth

    if show_confidence_bands and se is not None:
        if crit_val is not None and not np.isnan(crit_val):
            z_crit = crit_val
        else:
            z_crit = stats.norm.ppf(1 - alpha / 2)

        if smooth_band and len(dose_values) > 3:
            x_band = np.linspace(dose_values.min(), dose_values.max(), 200)
            f_eff = interp1d(dose_values, effects, kind="cubic", fill_value="extrapolate")
            f_se = interp1d(dose_values, se, kind="cubic", fill_value="extrapolate")
            band_eff = f_eff(x_band)
            band_se = f_se(x_band)
            lb = band_eff - z_crit * band_se
            ub = band_eff + z_crit * band_se
        else:
            order = np.argsort(dose_values)
            x_band = dose_values[order]
            band_eff = effects[order]
            band_se = se[order]
            lb = band_eff - z_crit * band_se
            ub = band_eff + z_crit * band_se

        finite = _finite_interval(x_band, lb, ub)
        if finite is not None:
            xfb, lb, ub = finite
            ax.fill_between(
                xfb,
                lb,
                ub,
                color="#bfbfbf",
                alpha=0.5,
                linewidth=0,
                edgecolor="none",
                zorder=1,
            )

    if smooth_line and len(dose_values) > 3:
        x_line = np.linspace(dose_values.min(), dose_values.max(), 200)
        f_eff = interp1d(dose_values, effects, kind="cubic", fill_value="extrapolate")
        y_line = f_eff(x_line)
    else:
        order = np.argsort(dose_values)
        x_line = dose_values[order]
        y_line = effects[order]

    ax.plot(
        x_line,
        y_line,
        color="#3a3a3a",
        linewidth=2.8,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=2,
    )

    if overall_effect is not None and show_overall:
        ax.axhline(
            y=overall_effect,
            color="#E74C3C",
            linestyle="--",
            linewidth=1.5,
            alpha=0.6,
        )

    ax.axhline(y=0, color="#6b6b6b", linestyle="-", alpha=0.5, linewidth=1.1)

    ax.set_xlabel("dose", fontsize=12, color="#2f2f2f")
    ax.set_ylabel(ylabel, fontsize=12, color="#2f2f2f")

    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.9)
        ax.spines[side].set_color("#8a8a8a")

    ax.grid(True, axis="both", alpha=0.18, linestyle="-", linewidth=0.5, color="#d9d9d9")
    ax.set_axisbelow(True)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=10,
        colors="#4a4a4a",
        length=4,
        width=0.8,
    )

    ax.set_xlim(dose_values.min(), dose_values.max())

    if created_here:
        plt.tight_layout()
    return fig


def _plot_pte_event_study(
    pte_result: PTEResult,
    type="att",
    alpha=0.05,
    ax=None,
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

    created_here = False
    if ax is None:
        fig, ax = plt.subplots(figsize=PLOT_CONFIG["figure_size_event_study"])
        created_here = True
    else:
        fig = ax.figure

    z_crit = stats.norm.ppf(1 - alpha / 2)
    if hasattr(event_study, "critical_value") and event_study.critical_value is not None:
        z_crit = event_study.critical_value

    ci_style = PLOT_CONFIG.get("event_study_ci_style", {})
    ax.errorbar(
        event_times,
        effects,
        yerr=z_crit * se,
        color=COLOR_PALETTES["event_study"]["ci"],
        **ci_style,
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

    if created_here:
        plt.tight_layout()
    return fig


def _se_from_inf(inf):
    """Compute standard error from influence functions when provided."""
    if inf is None:
        return None
    if isinstance(inf, np.ndarray) and inf.ndim == 2 and inf.size > 0:
        n = inf.shape[0]
        return np.std(inf, axis=0) / np.sqrt(max(n, 1))
    return None


def _finite_interval(x_values, lower, upper):
    mask = np.isfinite(x_values) & np.isfinite(lower) & np.isfinite(upper)
    if not np.any(mask):
        return None
    return x_values[mask], lower[mask], upper[mask]
