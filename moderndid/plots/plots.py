"""Plotnine-based plotting functions for moderndid."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_errorbar,
    geom_hline,
    geom_line,
    geom_point,
    geom_ribbon,
    ggplot,
    labs,
    position_dodge,
    scale_color_manual,
    theme,
)

from moderndid.plots.converters import (
    aggteresult_to_polars,
    doseresult_to_polars,
    honestdid_to_polars,
    mpresult_to_polars,
    pteresult_to_polars,
)
from moderndid.plots.themes import COLORS, theme_moderndid

if TYPE_CHECKING:
    from moderndid.did.aggte_obj import AGGTEResult
    from moderndid.did.multiperiod_obj import MPResult
    from moderndid.didcont.estimation.container import DoseResult, PTEResult
    from moderndid.didhonest.honest_did import HonestDiDResult


def plot_att_gt(
    result: MPResult,
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
    result : MPResult
        Multi-period DID result object containing group-time ATT estimates.
        This should be the output from ``att_gt()``, not ``drdid()``.
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
    from moderndid.did.multiperiod_obj import MPResult as MPResultClass

    if not isinstance(result, MPResultClass):
        raise TypeError(f"plot_att_gt requires MPResult from att_gt(), got {type(result).__name__}")

    df = mpresult_to_polars(result)

    df = df.with_columns([df["group"].cast(int).cast(str).alias("group_label")])

    plot = (
        ggplot(df, aes(x="time", y="att", color="treatment_status"))
        + geom_point(size=3, alpha=0.8)
        + scale_color_manual(
            values={"Pre": COLORS["pre_treatment"], "Post": COLORS["post_treatment"]},
            name="Treatment Status",
        )
        + facet_wrap("~group_label", ncol=ncol, labeller=lambda x: f"{title} {x}", scales="free_x")
        + labs(
            x=xlab or "Time",
            y=ylab or "ATT",
            title="Group-Time Average Treatment Effects",
        )
        + theme_moderndid()
        + theme(
            strip_text=element_text(size=11, weight="bold"),
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
    result: AGGTEResult | PTEResult,
    show_ci: bool = True,
    ref_line: float | None = 0,
    xlab: str | None = None,
    ylab: str | None = None,
    title: str | None = None,
    **_kwargs: Any,
) -> ggplot:
    """Create event study plot for dynamic treatment effects.

    Parameters
    ----------
    result : AGGTEResult or PTEResult
        Aggregated treatment effect result with dynamic aggregation,
        or PTEResult with event_study attribute.
    show_ci : bool, default=True
        Whether to show confidence intervals as error bars.
    ref_line : float or None, default=0
        Y-value for reference line. Set to None to hide.
    xlab : str, optional
        X-axis label. Defaults to "Event Time".
    ylab : str, optional
        Y-axis label. Defaults to "ATT".
    title : str, optional
        Plot title. Defaults to "Event Study".

    Returns
    -------
    ggplot
        A plotnine ggplot object that can be further customized.
    """
    from moderndid.didcont.estimation.container import PTEResult as PTEResultClass

    if isinstance(result, PTEResultClass):
        df = pteresult_to_polars(result)
    else:
        if result.aggregation_type != "dynamic":
            raise ValueError(f"Event study plot requires dynamic aggregation, got {result.aggregation_type}")
        df = aggteresult_to_polars(result)

    plot = ggplot(df, aes(x="event_time", y="att"))

    if show_ci:
        plot = plot + geom_errorbar(
            aes(ymin="ci_lower", ymax="ci_upper", color="treatment_status"),
            width=0.2,
            size=0.8,
        )

    if ref_line is not None:
        plot = plot + geom_hline(yintercept=ref_line, linetype="dashed", color="#7f8c8d", alpha=0.7)

    plot = (
        plot
        + geom_line(color=COLORS["line"], size=0.8, alpha=0.6)
        + geom_point(aes(color="treatment_status"), size=3.5)
        + scale_color_manual(
            values={"Pre": COLORS["pre_treatment"], "Post": COLORS["post_treatment"]},
            name="Treatment Status",
        )
        + labs(
            x=xlab or "Event Time",
            y=ylab or "ATT",
            title=title or "Event Study",
        )
        + theme_moderndid()
        + theme(legend_position="bottom")
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

    plot = ggplot(df, aes(x="dose", y="effect"))

    if show_ci:
        plot = plot + geom_ribbon(
            aes(ymin="ci_lower", ymax="ci_upper"),
            fill=COLORS["pre_treatment"],
            alpha=0.25,
        )

    if ref_line is not None:
        plot = plot + geom_hline(yintercept=ref_line, linetype="dashed", color="#7f8c8d", alpha=0.7)

    plot = (
        plot
        + geom_line(color=COLORS["pre_treatment"], size=1.2)
        + geom_point(color=COLORS["pre_treatment"], size=2.5)
        + labs(
            x=xlab or "Dose",
            y=ylab or default_ylabel,
            title=title or default_title,
        )
        + theme_moderndid()
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
        + theme_moderndid()
        + theme(legend_position="bottom")
    )

    if ref_line is not None:
        plot = plot + geom_hline(yintercept=ref_line, linetype="solid", color="black", alpha=0.4)

    return plot
