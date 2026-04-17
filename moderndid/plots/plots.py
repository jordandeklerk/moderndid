"""Plotnine-based plotting functions for moderndid."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import polars as pl
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_errorbar,
    geom_hline,
    geom_line,
    geom_point,
    geom_ribbon,
    geom_vline,
    ggplot,
    labs,
    position_dodge,
    scale_color_manual,
    scale_x_continuous,
    theme,
    theme_gray,
)

from moderndid.core.converters import (
    aggteresult_to_polars,
    dddaggresult_to_polars,
    dddmpresult_to_polars,
    didinterresult_to_polars,
    doseresult_to_polars,
    dynbalancingcoefs_to_polars,
    dynbalancinghetresult_to_polars,
    dynbalancinghistoryresult_to_polars,
    dynbalancingresult_to_polars,
    emfxresult_to_polars,
    honestdid_to_polars,
    mpresult_to_polars,
    pteresult_to_polars,
)
from moderndid.did.container import AGGTEResult, MPResult
from moderndid.diddynamic.container import DynBalancingHetResult, DynBalancingHistoryResult, DynBalancingResult
from moderndid.didinter.container import DIDInterResult
from moderndid.didtriple.container import DDDAggResult, DDDMultiPeriodRCResult, DDDMultiPeriodResult
from moderndid.etwfe.container import EmfxResult
from moderndid.plots.themes import COLORS

if TYPE_CHECKING:
    from moderndid.didcont.container import DoseResult, PTEResult
    from moderndid.didhonest.honest_did import HonestDiDResult


def plot_gt(
    result: MPResult | DDDMultiPeriodResult | DDDMultiPeriodRCResult,
    show_ci: bool = True,
    ref_line: float | None = 0,
    title: str = "Group",
    xlab: str | None = None,
    ylab: str | None = None,
    ncol: int = 1,
    **_kwargs: Any,
) -> ggplot:
    """Plot group-time average treatment effects.

    Parameters
    ----------
    result : MPResult, DDDMultiPeriodResult, or DDDMultiPeriodRCResult
        Multi-period result object containing group-time ATT estimates.
        This should be the output from ``att_gt()`` or ``ddd()``.
    show_ci : bool, default=True
        Whether to show confidence intervals as error bars.
    ref_line : float or None, default=0
        Y-value for reference line. Set to None to hide.
    title : str, default="Group"
        Title prefix for each facet panel.
    xlab : str, optional
        X-axis label. Defaults to "Time".
    ylab : str, optional
        Y-axis label. Defaults to "ATT".
    ncol : int, default=1
        Number of columns in the facet grid. Use 1 for vertical stacking.

    Returns
    -------
    ggplot
        A plotnine ggplot object that can be further customized.
    """
    if isinstance(result, MPResult):
        df = mpresult_to_polars(result)
        plot_title = "Group-Time Average Treatment Effects"
    elif isinstance(result, (DDDMultiPeriodResult, DDDMultiPeriodRCResult)):
        df = dddmpresult_to_polars(result)
        plot_title = "Group-Time DDD Treatment Effects"
    else:
        raise TypeError(
            f"plot_gt requires MPResult, DDDMultiPeriodResult, or DDDMultiPeriodRCResult, got {type(result).__name__}"
        )

    df = df.with_columns([df["group"].cast(int).cast(str).alias("group_label")])
    x_breaks = sorted(df["time"].unique().to_list())

    plot = (
        ggplot(df, aes(x="time", y="att", color="treatment_status"))
        + geom_point(size=3, alpha=0.8)
        + scale_color_manual(
            values={"Pre": COLORS["pre_treatment"], "Post": COLORS["post_treatment"]},
            limits=["Pre", "Post"],
            name="Treatment Status",
        )
        + scale_x_continuous(breaks=x_breaks)
        + facet_wrap("~group_label", ncol=ncol, labeller=lambda x: f"{title} {x}", scales="free_x")
        + labs(
            x=xlab or "Time",
            y=ylab or "ATT",
            title=plot_title,
        )
        + theme_gray()
        + theme(
            strip_text=element_text(size=11, weight="bold"),
            plot_title=element_text(margin={"b": 25}),
            legend_position="bottom",
        )
    )

    if show_ci:
        plot = plot + geom_errorbar(
            aes(ymin="ci_lower", ymax="ci_upper"),
            width=0.3,
            alpha=0.7,
        )

    if ref_line is not None:
        plot = plot + geom_hline(yintercept=ref_line, linetype="dashed", color="black", alpha=0.5)

    return plot


def plot_event_study(
    result: AGGTEResult | PTEResult | DDDAggResult | EmfxResult,
    show_ci: bool = True,
    ref_line: float | None = 0,
    ref_period: float | None = -1,
    xlab: str | None = None,
    ylab: str | None = None,
    title: str | None = None,
    **_kwargs: Any,
) -> ggplot:
    """Create event study plot for dynamic treatment effects.

    Parameters
    ----------
    result : AGGTEResult, PTEResult, or DDDAggResult
        Aggregated treatment effect result with dynamic/eventstudy aggregation,
        or PTEResult with event_study attribute.
    show_ci : bool, default=True
        Whether to show confidence intervals as error bars.
    ref_line : float or None, default=0
        Y-value for reference line. Set to None to hide.
    ref_period : float or None, default=-1
        X-value for vertical reference period line. Set to None to hide.
    xlab : str, optional
        X-axis label. Defaults to "Event Time".
    ylab : str, optional
        Y-axis label. Defaults to "ATT".
    title : str, optional
        Plot title. Defaults based on result type.

    Returns
    -------
    ggplot
        A plotnine ggplot object that can be further customized.
    """
    if isinstance(result, EmfxResult):
        if result.aggregation_type != "event":
            raise ValueError(f"Event study plot requires event aggregation, got {result.aggregation_type}")
        df = emfxresult_to_polars(result)
        default_title = "ETWFE Event Study"
    elif hasattr(result, "event_study") and not isinstance(result, (AGGTEResult, DDDAggResult)):
        df = pteresult_to_polars(result)
        default_title = "Event Study"
    elif isinstance(result, DDDAggResult):
        if result.aggregation_type != "eventstudy":
            raise ValueError(f"Event study plot requires eventstudy aggregation, got {result.aggregation_type}")
        df = dddaggresult_to_polars(result)
        default_title = "DDD Event Study"
    elif isinstance(result, AGGTEResult):
        if result.aggregation_type != "dynamic":
            raise ValueError(f"Event study plot requires dynamic aggregation, got {result.aggregation_type}")
        df = aggteresult_to_polars(result)
        default_title = "Event Study"
    else:
        raise TypeError(
            f"plot_event_study requires AGGTEResult, PTEResult, or DDDAggResult, got {type(result).__name__}"
        )

    plot = ggplot(df, aes(x="event_time", y="att"))

    if show_ci:
        plot = plot + geom_errorbar(
            aes(ymin="ci_lower", ymax="ci_upper", color="treatment_status"),
            width=0.2,
            size=0.8,
        )

    if ref_line is not None:
        plot = plot + geom_hline(yintercept=ref_line, linetype="dashed", color="#7f8c8d", alpha=0.7)

    if ref_period is not None:
        plot = plot + geom_vline(xintercept=ref_period, linetype="dashed", color="gray", size=0.4)
    else:
        plot = plot + geom_line(color=COLORS["line"], size=0.8, alpha=0.6, linetype="dotted")

    x_breaks = sorted(df["event_time"].unique().to_list())
    if ref_period is not None and ref_period not in x_breaks:
        x_breaks = sorted([*x_breaks, ref_period])

    present = df["treatment_status"].unique().to_list() if "treatment_status" in df.columns else []
    legend_limits = [s for s in ["Pre", "Post"] if s in present]

    plot = (
        plot
        + geom_point(aes(color="treatment_status"), size=3.5)
        + scale_color_manual(
            values={"Pre": COLORS["pre_treatment"], "Post": COLORS["post_treatment"]},
            limits=legend_limits,
            name="Treatment Status",
        )
        + scale_x_continuous(breaks=x_breaks)
        + labs(
            x=xlab or "Event Time",
            y=ylab or "ATT",
            title=title or default_title,
        )
        + theme_gray()
        + theme(legend_position="bottom")
    )

    return plot


def plot_agg(
    result: AGGTEResult | DDDAggResult,
    show_ci: bool = True,
    ref_line: float | None = 0,
    xlab: str | None = None,
    ylab: str | None = None,
    title: str | None = None,
    **_kwargs: Any,
) -> ggplot:
    """Create plot for aggregated treatment effects by group or calendar time.

    Parameters
    ----------
    result : AGGTEResult or DDDAggResult
        Aggregated treatment effect result with group or calendar aggregation.
    show_ci : bool, default=True
        Whether to show confidence intervals as error bars.
    ref_line : float or None, default=0
        Y-value for reference line. Set to None to hide.
    xlab : str, optional
        X-axis label. Defaults based on aggregation type.
    ylab : str, optional
        Y-axis label. Defaults to "ATT".
    title : str, optional
        Plot title. Defaults based on aggregation type.

    Returns
    -------
    ggplot
        A plotnine ggplot object that can be further customized.
    """
    if isinstance(result, DDDAggResult):
        if result.aggregation_type not in ("group", "calendar"):
            raise ValueError(
                f"plot_agg requires group or calendar aggregation, got {result.aggregation_type}. "
                f"Use plot_event_study for eventstudy aggregation."
            )
        df = dddaggresult_to_polars(result)
        is_ddd = True
    elif isinstance(result, AGGTEResult):
        if result.aggregation_type not in ("group", "calendar"):
            raise ValueError(
                f"plot_agg requires group or calendar aggregation, got {result.aggregation_type}. "
                f"Use plot_event_study for dynamic aggregation."
            )
        df = aggteresult_to_polars(result)
        is_ddd = False
    else:
        raise TypeError(f"plot_agg requires AGGTEResult or DDDAggResult, got {type(result).__name__}")

    if result.aggregation_type == "group":
        default_xlab = "Treatment Cohort"
        default_title = "DDD Effects by Treatment Cohort" if is_ddd else "Effects by Treatment Cohort"
    else:
        default_xlab = "Calendar Time"
        default_title = "DDD Effects by Calendar Time" if is_ddd else "Effects by Calendar Time"

    plot = ggplot(df, aes(x="event_time", y="att"))

    if show_ci:
        plot = plot + geom_errorbar(
            aes(ymin="ci_lower", ymax="ci_upper"),
            width=0.2,
            size=0.8,
            color=COLORS["post_treatment"],
        )

    if ref_line is not None:
        plot = plot + geom_hline(yintercept=ref_line, linetype="dashed", color="#7f8c8d", alpha=0.7)

    x_breaks = sorted(df["event_time"].unique().to_list())

    plot = (
        plot
        + geom_line(color=COLORS["line"], size=0.8, alpha=0.6, linetype="dotted")
        + geom_point(color=COLORS["post_treatment"], size=3.5)
        + scale_x_continuous(breaks=x_breaks)
        + labs(
            x=xlab or default_xlab,
            y=ylab or "ATT",
            title=title or default_title,
        )
        + theme_gray()
    )

    return plot


def plot_dose_response(
    result: DoseResult,
    effect_type: Literal["att", "acrt"] = "att",
    show_ci: bool = True,
    ref_line: float | None = 0,
    xlab: str | None = None,
    ylab: str | None = None,
    title: str | None = None,
    **_kwargs: Any,
) -> ggplot:
    """Plot dose-response function for continuous treatment.

    Parameters
    ----------
    result : DoseResult
        Continuous treatment dose-response result.
    effect_type : {'att', 'acrt'}, default='att'
        Type of effect to plot:
        - 'att': Average Treatment Effect on Treated
        - 'acrt': Average Causal Response on Treated (marginal effect)
    show_ci : bool, default=True
        Whether to show confidence bands.
    ref_line : float or None, default=0
        Y-value for reference line. Set to None to hide.
    xlab : str, optional
        X-axis label. Defaults to "Dose".
    ylab : str, optional
        Y-axis label. Defaults based on effect_type.
    title : str, optional
        Plot title. Defaults based on effect_type.

    Returns
    -------
    ggplot
        A plotnine ggplot object that can be further customized.
    """
    df = doseresult_to_polars(result, effect_type=effect_type)

    default_ylabel = "ATT(d)" if effect_type == "att" else "ACRT(d)"
    default_title = f"Dose-Response: {default_ylabel}"

    line_color = "#2c3e50"
    fill_color = "#5b7ea4"

    plot = ggplot(df, aes(x="dose", y="effect"))

    if show_ci:
        plot = (
            plot
            + geom_ribbon(aes(ymin="ci_lower", ymax="ci_upper"), fill=fill_color, alpha=0.2)
            + geom_line(aes(y="ci_upper"), linetype="dashed", color=line_color, size=0.5)
            + geom_line(aes(y="ci_lower"), linetype="dashed", color=line_color, size=0.5)
        )

    if ref_line is not None:
        plot = plot + geom_hline(yintercept=ref_line, linetype="dashed", color="#7f8c8d", alpha=0.7)

    plot = (
        plot
        + geom_line(color=line_color, size=1)
        + labs(
            x=xlab or "Dose",
            y=ylab or default_ylabel,
            title=title or default_title,
        )
        + theme_gray()
    )

    return plot


def plot_sensitivity(
    result: HonestDiDResult,
    ref_line: float | None = 0,
    xlab: str | None = None,
    ylab: str | None = None,
    title: str | None = None,
    **_kwargs: Any,
) -> ggplot:
    """Create sensitivity analysis plot for HonestDiD results.

    Parameters
    ----------
    result : HonestDiDResult
        Honest DiD sensitivity analysis result.
    ref_line : float or None, default=0
        Y-value for reference line. Set to None to hide.
    xlab : str, optional
        X-axis label. Defaults based on sensitivity type.
    ylab : str, optional
        Y-axis label. Defaults to "Confidence Interval".
    title : str, optional
        Plot title. Defaults based on sensitivity type.

    Returns
    -------
    ggplot
        A plotnine ggplot object that can be further customized.
    """
    df = honestdid_to_polars(result)

    is_smoothness = result.sensitivity_type == "smoothness"
    default_xlab = "M" if is_smoothness else r"$\bar{M}$"
    default_title = f"Sensitivity Analysis ({result.sensitivity_type.replace('_', ' ').title()})"

    method_colors = {
        "Original": COLORS["original"],
        "FLCI": COLORS["flci"],
        "Conditional": COLORS["conditional"],
        "C-F": COLORS["c_f"],
        "C-LF": COLORS["c_lf"],
    }

    methods = df["method"].unique().to_list()
    available_colors = {m: method_colors.get(m, "#34495e") for m in methods}

    n_methods = len(methods)
    dodge_width = 0.05 * (df["param_value"].max() - df["param_value"].min()) if n_methods > 1 else 0

    plot = (
        ggplot(df, aes(x="param_value", y="midpoint", color="method"))
        + geom_point(size=3, position=position_dodge(width=dodge_width))
        + geom_errorbar(
            aes(ymin="lb", ymax="ub"),
            width=0.02 * (df["param_value"].max() - df["param_value"].min()),
            size=0.8,
            position=position_dodge(width=dodge_width),
        )
        + scale_color_manual(values=available_colors, name="Method")
        + labs(
            x=xlab or default_xlab,
            y=ylab or "Confidence Interval",
            title=title or default_title,
        )
        + theme_gray()
        + theme(legend_position="bottom")
    )

    if ref_line is not None:
        plot = plot + geom_hline(yintercept=ref_line, linetype="solid", color="black", alpha=0.4)

    return plot


def plot_multiplegt(
    result: DIDInterResult,
    show_ci: bool = True,
    ref_line: float | None = 0,
    xlab: str | None = None,
    ylab: str | None = None,
    title: str | None = None,
    **_kwargs: Any,
) -> ggplot:
    """Create event study plot for intertemporal treatment effects.

    Parameters
    ----------
    result : DIDInterResult
        Intertemporal treatment effects result from did_multiplegt().
    show_ci : bool, default=True
        Whether to show confidence intervals as error bars.
    ref_line : float or None, default=0
        Y-value for reference line. Set to None to hide.
    xlab : str, optional
        X-axis label. Defaults to "Horizon".
    ylab : str, optional
        Y-axis label. Defaults to "Effect".
    title : str, optional
        Plot title. Defaults to "Intertemporal Treatment Effects".

    Returns
    -------
    ggplot
        A plotnine ggplot object that can be further customized.
    """
    df = didinterresult_to_polars(result)
    x_breaks = sorted(df["horizon"].unique().to_list())
    if 0 not in x_breaks:
        x_breaks = sorted([*x_breaks, 0])

    plot = ggplot(df, aes(x="horizon", y="att"))

    if show_ci:
        plot = plot + geom_errorbar(
            aes(ymin="ci_lower", ymax="ci_upper", color="treatment_status"),
            width=0.2,
            size=0.8,
        )

    if ref_line is not None:
        plot = plot + geom_hline(yintercept=ref_line, linetype="dashed", color="#7f8c8d", alpha=0.7)

    plot = plot + geom_vline(xintercept=0, linetype="dashed", color="gray", size=0.4)

    plot = (
        plot
        + geom_point(aes(color="treatment_status"), size=3.5)
        + scale_color_manual(
            values={"Pre": COLORS["pre_treatment"], "Post": COLORS["post_treatment"]},
            limits=["Pre", "Post"],
            name="Treatment Status",
        )
        + scale_x_continuous(breaks=x_breaks)
        + labs(
            x=xlab or "Horizon",
            y=ylab or "Effect",
            title=title or "Intertemporal Treatment Effects",
        )
        + theme_gray()
        + theme(legend_position="bottom")
    )

    return plot


def plot_dyn_balancing(
    result: DynBalancingResult,
    parameter: Literal["att", "mu1", "mu2", "all"] = "att",
    show_ci: bool = True,
    ci_type: Literal["robust", "gaussian"] = "robust",
    ref_line: float | None = 0,
    xlab: str | None = None,
    ylab: str | None = None,
    title: str | None = None,
    **_kwargs: Any,
) -> ggplot:
    """Plot point estimates from a dynamic covariate balancing result.

    By default shows the average treatment effect (``att``) as a point
    with an error bar. Set ``parameter="mu1"`` or ``"mu2"`` to plot one
    of the potential outcome estimates instead, or ``"all"`` to show all
    three side by side. The robust (chi-squared) confidence interval is
    drawn by default; set ``ci_type="gaussian"`` to use the Gaussian
    quantile instead.

    Parameters
    ----------
    result : DynBalancingResult
        Single dynamic covariate balancing result.
    parameter : {"att", "mu1", "mu2", "all"}, default="att"
        Which parameter to plot. ``"all"`` shows the ATE alongside both
        potential outcome estimates on a shared axis.
    show_ci : bool, default=True
        Whether to draw confidence intervals.
    ci_type : {"robust", "gaussian"}, default="robust"
        Which critical value to use when drawing confidence intervals.
    ref_line : float or None, default=0
        Y-value for a horizontal reference line. Set to None to hide.
    xlab : str, optional
        X-axis label. Defaults to "Parameter".
    ylab : str, optional
        Y-axis label. Defaults based on ``parameter``.
    title : str, optional
        Plot title. Defaults based on ``parameter``.

    Returns
    -------
    ggplot
        A plotnine ggplot object that can be further customized.
    """
    if not isinstance(result, DynBalancingResult):
        raise TypeError(f"plot_dyn_balancing requires DynBalancingResult, got {type(result).__name__}")

    if ci_type not in ("robust", "gaussian"):
        raise ValueError(f"ci_type must be 'robust' or 'gaussian', got {ci_type!r}")

    if parameter not in ("att", "mu1", "mu2", "all"):
        raise ValueError(f"parameter must be one of 'att', 'mu1', 'mu2', 'all', got {parameter!r}")

    df = dynbalancingresult_to_polars(result)

    label_map = {
        "att": ("ATE", "ATE", "Dynamic Covariate Balancing ATE"),
        "mu1": ("mu(ds1)", "mu(ds1)", "Potential Outcome under ds1"),
        "mu2": ("mu(ds2)", "mu(ds2)", "Potential Outcome under ds2"),
    }

    if parameter != "all":
        row_label, default_ylab, default_title = label_map[parameter]
        df = df.filter(pl.col("parameter") == row_label)
    else:
        default_ylab = "Estimate"
        default_title = "Dynamic Covariate Balancing Estimates"

    ymin_col = f"ci_lower_{ci_type}"
    ymax_col = f"ci_upper_{ci_type}"

    plot = ggplot(df, aes(x="parameter", y="estimate"))

    if ref_line is not None:
        plot = plot + geom_hline(yintercept=ref_line, linetype="dashed", color="#7f8c8d", alpha=0.7)

    if show_ci:
        plot = plot + geom_errorbar(
            aes(ymin=ymin_col, ymax=ymax_col),
            width=0.2,
            size=0.8,
            color=COLORS["post_treatment"],
        )

    plot = (
        plot
        + geom_point(color=COLORS["post_treatment"], size=3.5)
        + labs(
            x=xlab or "Parameter",
            y=ylab or default_ylab,
            title=title or default_title,
        )
        + theme_gray()
    )

    if parameter == "all":
        plot = (
            plot
            + facet_wrap("~parameter", scales="free", ncol=3)
            + theme(strip_text=element_text(size=11, weight="bold"))
        )

    return plot


def plot_dyn_balancing_history(
    result: DynBalancingHistoryResult,
    parameter: Literal["att", "mu1", "mu2"] = "att",
    show_ci: bool = True,
    ci_type: Literal["robust", "gaussian"] = "robust",
    ref_line: float | None = 0,
    xlab: str | None = None,
    ylab: str | None = None,
    title: str | None = None,
    **_kwargs: Any,
) -> ggplot:
    """Plot dynamic covariate balancing estimates across treatment history lengths.

    Shows how the chosen parameter (ATE by default) evolves as the length
    of the treatment history considered increases. Points mark each
    horizon with error bars for the selected confidence interval, and a
    dotted line connects successive estimates. By default the robust
    (chi-squared) critical values are used.

    Parameters
    ----------
    result : DynBalancingHistoryResult
        History-mode dynamic covariate balancing result.
    parameter : {"att", "mu1", "mu2"}, default="att"
        Which parameter to plot.
    show_ci : bool, default=True
        Whether to draw confidence intervals.
    ci_type : {"robust", "gaussian"}, default="robust"
        Which critical value to use when drawing confidence intervals.
    ref_line : float or None, default=0
        Y-value for a horizontal reference line. Set to None to hide.
    xlab : str, optional
        X-axis label. Defaults to "History Length".
    ylab : str, optional
        Y-axis label. Defaults based on ``parameter``.
    title : str, optional
        Plot title. Defaults based on ``parameter``.

    Returns
    -------
    ggplot
        A plotnine ggplot object that can be further customized.
    """
    if not isinstance(result, DynBalancingHistoryResult):
        raise TypeError(f"plot_dyn_balancing_history requires DynBalancingHistoryResult, got {type(result).__name__}")

    if ci_type not in ("robust", "gaussian"):
        raise ValueError(f"ci_type must be 'robust' or 'gaussian', got {ci_type!r}")

    df = dynbalancinghistoryresult_to_polars(result, parameter=parameter)

    label_map = {
        "att": ("ATE", "ATE by Treatment History Length"),
        "mu1": ("mu(ds1)", "Potential Outcome under ds1 by History Length"),
        "mu2": ("mu(ds2)", "Potential Outcome under ds2 by History Length"),
    }
    default_ylab, default_title = label_map[parameter]

    ymin_col = f"ci_lower_{ci_type}"
    ymax_col = f"ci_upper_{ci_type}"

    plot = ggplot(df, aes(x="period_length", y="estimate"))

    if show_ci:
        plot = plot + geom_errorbar(
            aes(ymin=ymin_col, ymax=ymax_col),
            width=0.2,
            size=0.8,
            color=COLORS["post_treatment"],
        )

    if ref_line is not None:
        plot = plot + geom_hline(yintercept=ref_line, linetype="dashed", color="#7f8c8d", alpha=0.7)

    x_breaks = sorted(df["period_length"].unique().to_list())

    plot = (
        plot
        + geom_line(color=COLORS["line"], size=0.8, alpha=0.6, linetype="dotted")
        + geom_point(color=COLORS["post_treatment"], size=3.5)
        + scale_x_continuous(breaks=x_breaks)
        + labs(
            x=xlab or "History Length",
            y=ylab or default_ylab,
            title=title or default_title,
        )
        + theme_gray()
    )

    return plot


def plot_dyn_balancing_het(
    result: DynBalancingHetResult,
    parameter: Literal["att", "mu1", "mu2"] = "att",
    show_ci: bool = True,
    ci_type: Literal["robust", "gaussian"] = "robust",
    ref_line: float | None = 0,
    xlab: str | None = None,
    ylab: str | None = None,
    title: str | None = None,
    **_kwargs: Any,
) -> ggplot:
    """Plot dynamic covariate balancing estimates across final time periods.

    Shows how the chosen parameter (ATE by default) varies when the
    treatment histories are evaluated at different final periods. Points
    mark each period with error bars for the selected confidence interval,
    and a dotted line connects successive estimates.

    Parameters
    ----------
    result : DynBalancingHetResult
        Het-mode dynamic covariate balancing result.
    parameter : {"att", "mu1", "mu2"}, default="att"
        Which parameter to plot.
    show_ci : bool, default=True
        Whether to draw confidence intervals.
    ci_type : {"robust", "gaussian"}, default="robust"
        Which critical value to use when drawing confidence intervals.
    ref_line : float or None, default=0
        Y-value for a horizontal reference line. Set to None to hide.
    xlab : str, optional
        X-axis label. Defaults to "Final Period".
    ylab : str, optional
        Y-axis label. Defaults based on ``parameter``.
    title : str, optional
        Plot title. Defaults based on ``parameter``.

    Returns
    -------
    ggplot
        A plotnine ggplot object that can be further customized.
    """
    if not isinstance(result, DynBalancingHetResult):
        raise TypeError(f"plot_dyn_balancing_het requires DynBalancingHetResult, got {type(result).__name__}")

    if ci_type not in ("robust", "gaussian"):
        raise ValueError(f"ci_type must be 'robust' or 'gaussian', got {ci_type!r}")

    df = dynbalancinghetresult_to_polars(result, parameter=parameter)

    label_map = {
        "att": ("ATE", "ATE by Final Period"),
        "mu1": ("mu(ds1)", "Potential Outcome under ds1 by Final Period"),
        "mu2": ("mu(ds2)", "Potential Outcome under ds2 by Final Period"),
    }
    default_ylab, default_title = label_map[parameter]

    ymin_col = f"ci_lower_{ci_type}"
    ymax_col = f"ci_upper_{ci_type}"

    plot = ggplot(df, aes(x="final_period", y="estimate"))

    if show_ci:
        plot = plot + geom_errorbar(
            aes(ymin=ymin_col, ymax=ymax_col),
            width=0.2,
            size=0.8,
            color=COLORS["post_treatment"],
        )

    if ref_line is not None:
        plot = plot + geom_hline(yintercept=ref_line, linetype="dashed", color="#7f8c8d", alpha=0.7)

    x_breaks = sorted(df["final_period"].unique().to_list())

    plot = (
        plot
        + geom_line(color=COLORS["line"], size=0.8, alpha=0.6, linetype="dotted")
        + geom_point(color=COLORS["post_treatment"], size=3.5)
        + scale_x_continuous(breaks=x_breaks)
        + labs(
            x=xlab or "Final Period",
            y=ylab or default_ylab,
            title=title or default_title,
        )
        + theme_gray()
    )

    return plot


def plot_dyn_balancing_coefs(
    result: DynBalancingResult,
    history: Literal["ds1", "ds2"] = "ds1",
    xlab: str | None = None,
    ylab: str | None = None,
    title: str | None = None,
    **_kwargs: Any,
) -> ggplot:
    """Plot LASSO coefficient estimates from a dynamic covariate balancing result.

    Shows a dot plot of the estimated coefficients for each covariate,
    faceted by time period. Non-zero coefficients (selected by the LASSO)
    are highlighted in red and zero coefficients are shown in gray.

    Parameters
    ----------
    result : DynBalancingResult
        Single dynamic covariate balancing result with non-empty
        ``coefficients`` attribute.
    history : {"ds1", "ds2"}, default="ds1"
        Which treatment history's coefficients to plot.
    xlab : str, optional
        X-axis label. Defaults to "Coefficient".
    ylab : str, optional
        Y-axis label. Defaults to "Covariate".
    title : str, optional
        Plot title. Defaults to "LASSO Coefficients by Period".

    Returns
    -------
    ggplot
        A plotnine ggplot object that can be further customized.
    """
    if not isinstance(result, DynBalancingResult):
        raise TypeError(f"plot_dyn_balancing_coefs requires DynBalancingResult, got {type(result).__name__}")

    df = dynbalancingcoefs_to_polars(result, history=history)

    if df.is_empty():
        raise ValueError("No coefficients available in this result. Run with balancing='dcb'.")

    plot = (
        ggplot(df, aes(x="coefficient", y="covariate", color="is_nonzero"))
        + geom_vline(xintercept=0, linetype="dashed", color="#7f8c8d", alpha=0.7)
        + geom_point(size=3)
        + scale_color_manual(
            values={True: COLORS["post_treatment"], False: COLORS["ci_fill"]},
            labels={True: "Non-zero", False: "Zero"},
            name="LASSO Selection",
        )
        + facet_wrap("~period", labeller="label_both")
        + labs(
            x=xlab or "Coefficient",
            y=ylab or "Covariate",
            title=title or "LASSO Coefficients by Period",
        )
        + theme_gray()
        + theme(legend_position="bottom")
    )

    return plot
