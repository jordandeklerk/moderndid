"""Plotting functions for continuous treatment DiD results."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from matplotlib.lines import Line2D
from scipy import stats
from scipy.interpolate import interp1d

from moderndid.didcont.estimation.container import DoseResult, PTEResult
from moderndid.plotting import PlotCollection
from moderndid.plotting.converters import doseresult_to_dataset, pteresult_to_dataset
from moderndid.plotting.themes import THEMES, PlotTheme, apply_theme


def plot_cont_did(
    dose_obj: DoseResult | PTEResult,
    type: Literal["att", "acrt"] = "att",
    alpha: float = 0.05,
    show_overall: bool = False,
    show_ci: bool = True,
    rescale_factor: float = 1.0,
    figsize: tuple[float, float] | None = None,
    theme: str | PlotTheme = "default",
    **kwargs: Any,
) -> PlotCollection:
    """Plot results from continuous treatment DiD estimation.

    Creates visualizations for continuous treatment difference-in-differences
    analysis. For `DoseResult` objects, plots the dose-response function showing
    how treatment effects vary with dose intensity. For `PTEResult` objects with
    event studies, plots the dynamic treatment effects over time.

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
    show_ci : bool, default=True
        Whether to display confidence bands around the estimates.
    rescale_factor : float, default=1.0
        Factor to rescale dose and effect values for display.
    figsize : tuple of float, optional
        Figure size (width, height). Defaults to (9, 5.5) for dose-response
        and (10, 6) for event study.
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
    cont_did : Continuous treatment DiD estimation.
    plot_did : Unified plotting for standard DiD results.

    Examples
    --------
    Visualize dose-response relationships from continuous treatment DiD:

    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> from moderndid import plot_cont_did
        >>> from moderndid.didcont.estimation.container import DoseResult
        >>> np.random.seed(42)
        >>> dose_result = DoseResult(
        ...     dose=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        ...     overall_att=0.5,
        ...     overall_att_se=0.1,
        ...     overall_att_inf_func=np.random.randn(100),
        ...     overall_acrt=0.1,
        ...     overall_acrt_se=0.05,
        ...     overall_acrt_inf_func=np.random.randn(100),
        ...     att_d=np.array([0.1, 0.3, 0.5, 0.6, 0.65]),
        ...     att_d_se=np.array([0.08, 0.09, 0.1, 0.11, 0.12]),
        ...     acrt_d=np.array([0.2, 0.2, 0.1, 0.05, 0.02]),
        ...     acrt_d_se=np.array([0.05, 0.05, 0.04, 0.04, 0.03]),
        ... )
        >>> pc = plot_cont_did(dose_result, type="att")

    Plot the Average Causal Response on Treated (ACRT), which shows the
    marginal effect of increasing dose:

    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> from moderndid import plot_cont_did
        >>> from moderndid.didcont.estimation.container import DoseResult
        >>> np.random.seed(42)
        >>> dose_result = DoseResult(
        ...     dose=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        ...     overall_att=0.5,
        ...     overall_att_se=0.1,
        ...     overall_att_inf_func=np.random.randn(100),
        ...     overall_acrt=0.1,
        ...     overall_acrt_se=0.05,
        ...     overall_acrt_inf_func=np.random.randn(100),
        ...     att_d=np.array([0.1, 0.3, 0.5, 0.6, 0.65]),
        ...     att_d_se=np.array([0.08, 0.09, 0.1, 0.11, 0.12]),
        ...     acrt_d=np.array([0.2, 0.2, 0.1, 0.05, 0.02]),
        ...     acrt_d_se=np.array([0.05, 0.05, 0.04, 0.04, 0.03]),
        ... )
        >>> pc = plot_cont_did(dose_result, type="acrt")

    Add the overall average effect as a reference line and use "publication" theme:

    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> from moderndid import plot_cont_did
        >>> from moderndid.didcont.estimation.container import DoseResult
        >>> np.random.seed(42)
        >>> dose_result = DoseResult(
        ...     dose=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        ...     overall_att=0.5,
        ...     overall_att_se=0.1,
        ...     overall_att_inf_func=np.random.randn(100),
        ...     overall_acrt=0.1,
        ...     overall_acrt_se=0.05,
        ...     overall_acrt_inf_func=np.random.randn(100),
        ...     att_d=np.array([0.1, 0.3, 0.5, 0.6, 0.65]),
        ...     att_d_se=np.array([0.08, 0.09, 0.1, 0.11, 0.12]),
        ...     acrt_d=np.array([0.2, 0.2, 0.1, 0.05, 0.02]),
        ...     acrt_d_se=np.array([0.05, 0.05, 0.04, 0.04, 0.03]),
        ... )
        >>> pc = plot_cont_did(
        ...     dose_result,
        ...     type="att",
        ...     show_overall=True,
        ...     theme="publication",
        ...     figsize=(10, 6),
        ... )
    """
    if hasattr(dose_obj, "event_study") and dose_obj.event_study is not None:
        return _plot_pte_event_study(
            dose_obj,
            type=type,
            alpha=alpha,
            figsize=figsize,
            theme=theme,
            **kwargs,
        )

    if not hasattr(dose_obj, "dose"):
        raise ValueError("Input must be a DoseResult or PTEResult object from cont_did")

    return _plot_dose_response(
        dose_obj,
        type=type,
        alpha=alpha,
        show_overall=show_overall,
        show_ci=show_ci,
        rescale_factor=rescale_factor,
        figsize=figsize,
        theme=theme,
        **kwargs,
    )


def _plot_dose_response(
    dose_result: DoseResult,
    type: Literal["att", "acrt"] = "att",
    alpha: float = 0.05,
    show_overall: bool = False,
    show_ci: bool = True,
    rescale_factor: float = 1.0,
    smooth: bool = True,
    smooth_line: bool | None = None,
    smooth_band: bool = True,
    figsize: tuple[float, float] | None = None,
    theme: str | PlotTheme = "default",
    **kwargs: Any,
) -> PlotCollection:
    """Plot dose-response function for continuous treatment.

    Parameters
    ----------
    dose_result : DoseResult
        Dose-response result from continuous DiD estimation.
    type : {"att", "acrt"}, default="att"
        Type of effect to plot.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    show_overall : bool, default=False
        Whether to show the overall treatment effect as a horizontal line.
    show_ci : bool, default=True
        Whether to display confidence bands around the estimates.
    rescale_factor : float, default=1.0
        Factor to rescale dose and effect values for display.
    smooth : bool, default=True
        Whether to smooth the dose-response curve.
    smooth_line : bool, optional
        Whether to smooth the line. Defaults to smooth value.
    smooth_band : bool, default=True
        Whether to smooth the confidence band.
    figsize : tuple of float, optional
        Figure size (width, height). Defaults to (9, 5.5).
    theme : str or PlotTheme, default="default"
        Theme name or custom PlotTheme object.
    **kwargs
        Additional keyword arguments passed to matplotlib functions.

    Returns
    -------
    PlotCollection
        PlotCollection object for method chaining.
    """
    theme_obj = _get_theme(theme)

    if figsize is None:
        figsize = (9, 5.5)

    dataset = doseresult_to_dataset(dose_result)

    pc = PlotCollection.grid(dataset, figure_kwargs={"figsize": figsize})
    apply_theme(pc, theme_obj)

    ax = pc.viz["plot"].values.item() if pc.viz["plot"].size == 1 else pc.viz["plot"].values

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
        ylabel = "ATT(d)"
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
        ylabel = "ACRT(d)"
    else:
        raise ValueError(f"Invalid type: {type}. Must be 'att' or 'acrt'")

    if smooth_line is None:
        smooth_line = smooth

    if show_ci and se is not None:
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
                color=theme_obj.ci_color,
                alpha=theme_obj.ci_alpha,
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
        color=theme_obj.line_color,
        linewidth=theme_obj.line_width,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=2,
        **kwargs,
    )

    if overall_effect is not None and show_overall:
        ax.axhline(
            y=overall_effect,
            color=theme_obj.post_treatment_color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.6,
            label=f"Overall {type.upper()}",
        )

    ax.axhline(y=0, color="black", linestyle="-", alpha=0.5, linewidth=1.1)

    ax.set_xlabel("Dose", fontsize=theme_obj.label_fontsize)
    ax.set_ylabel(ylabel, fontsize=theme_obj.label_fontsize)
    ax.set_title(f"Dose-Response: {ylabel}", fontsize=theme_obj.title_fontsize, fontweight="bold", pad=10)

    ax.set_xlim(dose_values.min(), dose_values.max())

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(True, axis="both", alpha=0.2, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    fig = pc.viz["figure"]
    fig.tight_layout()

    return pc


def _plot_pte_event_study(
    pte_result: PTEResult,
    type: Literal["att", "acrt"] = "att",
    alpha: float = 0.05,
    show_ci: bool = True,
    ref_line: float = 0,
    figsize: tuple[float, float] | None = None,
    theme: str | PlotTheme = "default",
    **kwargs: Any,
) -> PlotCollection:
    """Plot event study for continuous treatment PTEResult.

    Parameters
    ----------
    pte_result : PTEResult
        Panel treatment effects result with event study.
    type : {"att", "acrt"}, default="att"
        Type of effect to plot.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    show_ci : bool, default=True
        Whether to display confidence bands.
    ref_line : float, default=0
        Reference line value (typically 0).
    figsize : tuple of float, optional
        Figure size (width, height). Defaults to (10, 6).
    theme : str or PlotTheme, default="default"
        Theme name or custom PlotTheme object.
    **kwargs
        Additional keyword arguments passed to matplotlib functions.

    Returns
    -------
    PlotCollection
        PlotCollection object for method chaining.
    """
    if not hasattr(pte_result, "event_study") or pte_result.event_study is None:
        raise ValueError("PTEResult does not contain event study results")

    theme_obj = _get_theme(theme)

    if figsize is None:
        figsize = (10, 6)

    dataset = pteresult_to_dataset(pte_result)

    pc = PlotCollection.grid(dataset, figure_kwargs={"figsize": figsize})
    apply_theme(pc, theme_obj)

    ax = pc.viz["plot"].values.item() if pc.viz["plot"].size == 1 else pc.viz["plot"].values

    event_study = pte_result.event_study
    event_times = dataset["att"].coords["event"]
    att = dataset["att"].values
    se = dataset["se"].values
    treatment_status = dataset["treatment_status"].values

    sort_idx = np.argsort(event_times)
    event_times = event_times[sort_idx]
    att = att[sort_idx]
    se = se[sort_idx]
    treatment_status = treatment_status[sort_idx]

    if hasattr(event_study, "critical_value") and event_study.critical_value is not None:
        z_crit = event_study.critical_value
    else:
        z_crit = stats.norm.ppf(1 - alpha / 2)

    is_acrt = (
        hasattr(pte_result, "ptep")
        and hasattr(pte_result.ptep, "target_parameter")
        and pte_result.ptep.target_parameter == "slope"
    )
    ylabel = "ACRT" if is_acrt or type == "acrt" else "ATT"

    for t, a, s, status in zip(event_times, att, se, treatment_status):
        color = theme_obj.post_treatment_color if status == "post" else theme_obj.pre_treatment_color
        yerr = z_crit * s if show_ci else 0

        ax.errorbar(
            t,
            a,
            yerr=yerr,
            fmt="o",
            color=color,
            markersize=theme_obj.marker_size,
            capsize=5,
            capthick=1.5,
            elinewidth=2,
            alpha=0.8,
            **kwargs,
        )

    if ref_line is not None:
        ax.axhline(y=ref_line, color="black", linestyle="-", alpha=0.5, linewidth=2, zorder=1)

    if event_times.min() < 0 < event_times.max():
        ax.axvline(x=-0.5, color="gray", linestyle="--", alpha=0.6, linewidth=2, label="Treatment")

    ax.set_xlabel("Event Time", fontsize=theme_obj.label_fontsize)
    ax.set_ylabel(ylabel, fontsize=theme_obj.label_fontsize)
    ax.set_title(f"Event Study: {ylabel} Over Time", fontsize=theme_obj.title_fontsize, fontweight="bold", pad=15)

    ax.set_xticks(event_times)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _add_legend(ax, theme_obj)

    fig = pc.viz["figure"]
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)

    return pc


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


def _se_from_inf(inf):
    """Compute standard error from influence functions when provided."""
    if inf is None:
        return None
    if isinstance(inf, np.ndarray) and inf.ndim == 2 and inf.size > 0:
        n = inf.shape[0]
        return np.std(inf, axis=0) / np.sqrt(max(n, 1))
    return None


def _finite_interval(x_values, lower, upper):
    """Filter to finite interval values."""
    mask = np.isfinite(x_values) & np.isfinite(lower) & np.isfinite(upper)
    if not np.any(mask):
        return None
    return x_values[mask], lower[mask], upper[mask]
