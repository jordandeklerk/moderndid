"""Converters for transforming DiD result objects to polars DataFrames."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from scipy.stats import chi2

if TYPE_CHECKING:
    from moderndid.did.container import AGGTEResult, MPResult
    from moderndid.didcont.container import DoseResult, PTEResult
    from moderndid.diddynamic.container import DynBalancingHetResult, DynBalancingHistoryResult, DynBalancingResult
    from moderndid.didhonest.honest_did import HonestDiDResult
    from moderndid.didinter.container import DIDInterResult, HeterogeneityResult
    from moderndid.didtriple.container import DDDAggResult, DDDMultiPeriodRCResult, DDDMultiPeriodResult
    from moderndid.etwfe.container import EmfxResult


def mpresult_to_polars(result: MPResult) -> pl.DataFrame:
    """Convert MPResult to polars DataFrame for plotting.

    Parameters
    ----------
    result : MPResult
        Multi-period DID result containing group-time ATT estimates.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - group: treatment cohort
        - time: time period
        - att: group-time ATT estimate
        - se: standard error
        - ci_lower: lower confidence interval
        - ci_upper: upper confidence interval
        - treatment_status: "Pre" or "Post" treatment
    """
    groups = result.groups
    times = result.times
    att = result.att_gt
    se = result.se_gt
    crit_val = result.critical_value

    ci_lower = att - crit_val * se
    ci_upper = att + crit_val * se

    treatment_status = np.array(["Pre" if t < g else "Post" for g, t in zip(groups, times, strict=False)])

    return pl.DataFrame(
        {
            "group": groups,
            "time": times,
            "att": att,
            "se": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "treatment_status": treatment_status,
        }
    )


def aggteresult_to_polars(result: AGGTEResult) -> pl.DataFrame:
    """Convert AGGTEResult to polars DataFrame for plotting.

    Parameters
    ----------
    result : AGGTEResult
        Aggregated treatment effect result (dynamic, group, or calendar).

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - event_time: event time (for dynamic), group (for group), or time (for calendar)
        - att: ATT estimate
        - se: standard error
        - ci_lower: lower confidence interval
        - ci_upper: upper confidence interval
        - treatment_status: "Pre" or "Post" (for dynamic aggregation)

    Raises
    ------
    ValueError
        If result is simple aggregation or missing required data.
    """
    if result.aggregation_type == "simple":
        raise ValueError("Simple aggregation does not produce event-level data for plotting.")

    if result.event_times is None or result.att_by_event is None or result.se_by_event is None:
        raise ValueError(
            f"AGGTEResult with aggregation_type='{result.aggregation_type}' "
            "must have event_times, att_by_event, and se_by_event"
        )

    event_times = result.event_times
    att = result.att_by_event
    se = result.se_by_event

    crit_vals = result.critical_values if result.critical_values is not None else np.full_like(se, 1.96)

    ci_lower = att - crit_vals * se
    ci_upper = att + crit_vals * se

    data = {
        "event_time": event_times,
        "att": att,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }

    if result.aggregation_type == "dynamic":
        data["treatment_status"] = np.array(["Pre" if e < 0 else "Post" for e in event_times])

    df = pl.DataFrame(data)
    return df.filter(~pl.col("se").is_nan())


def doseresult_to_polars(result: DoseResult, effect_type: str = "att") -> pl.DataFrame:
    """Convert DoseResult to polars DataFrame for plotting.

    Parameters
    ----------
    result : DoseResult
        Continuous treatment dose-response result.
    effect_type : {'att', 'acrt'}, default='att'
        Type of effect to extract:
        - 'att': Average Treatment Effect on Treated
        - 'acrt': Average Causal Response on Treated

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - dose: dose level
        - effect: effect estimate (ATT or ACRT)
        - se: standard error
        - ci_lower: lower confidence interval
        - ci_upper: upper confidence interval

    Raises
    ------
    ValueError
        If effect_type is invalid or required data is missing.
    """
    dose = result.dose

    if effect_type == "att":
        effect = result.att_d
        se = result.att_d_se
        crit_val = result.att_d_crit_val
    elif effect_type == "acrt":
        effect = result.acrt_d
        se = result.acrt_d_se
        crit_val = result.acrt_d_crit_val
    else:
        raise ValueError(f"effect_type must be 'att' or 'acrt', got '{effect_type}'")

    if effect is None or se is None:
        raise ValueError(f"DoseResult missing {effect_type.upper()} data")

    if crit_val is None or np.isnan(crit_val):
        crit_val = 1.96

    ci_lower = effect - crit_val * se
    ci_upper = effect + crit_val * se

    return pl.DataFrame(
        {
            "dose": dose,
            "effect": effect,
            "se": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
    )


def pteresult_to_polars(result: PTEResult) -> pl.DataFrame:
    """Convert PTEResult event study to polars DataFrame for plotting.

    Parameters
    ----------
    result : PTEResult
        Panel treatment effects result with event_study.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - event_time: event time relative to treatment
        - att: ATT estimate
        - se: standard error
        - ci_lower: lower confidence interval
        - ci_upper: upper confidence interval
        - treatment_status: "Pre" or "Post" treatment

    Raises
    ------
    ValueError
        If result does not contain event study.
    """
    if result.event_study is None:
        raise ValueError("PTEResult does not contain event study results")

    event_study = result.event_study
    event_times = event_study.event_times
    att = event_study.att_by_event
    se = event_study.se_by_event

    if hasattr(event_study, "critical_value") and event_study.critical_value is not None:
        crit_val = event_study.critical_value
    else:
        crit_val = 1.96

    ci_lower = att - crit_val * se
    ci_upper = att + crit_val * se
    treatment_status = np.array(["Pre" if e < 0 else "Post" for e in event_times])

    df = pl.DataFrame(
        {
            "event_time": event_times,
            "att": att,
            "se": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "treatment_status": treatment_status,
        }
    )
    return df.filter(~pl.col("se").is_nan())


def honestdid_to_polars(result: HonestDiDResult) -> pl.DataFrame:
    """Convert HonestDiDResult to polars DataFrame for plotting.

    Parameters
    ----------
    result : HonestDiDResult
        Honest DiD sensitivity analysis result.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - param_value: M or Mbar parameter value
        - method: CI method name
        - lb: lower bound of confidence interval
        - ub: upper bound of confidence interval
        - midpoint: (lb + ub) / 2
        Combined with original CI at param_value before the minimum robust value.

    Raises
    ------
    ValueError
        If result has empty robust_ci DataFrame.
    """
    robust_df = result.robust_ci
    original = result.original_ci

    if robust_df.is_empty():
        raise ValueError("HonestDiDResult has empty robust_ci DataFrame")

    if "M" in robust_df.columns:
        param_col = "M"
    elif "m" in robust_df.columns:
        param_col = "m"
    elif "Mbar" in robust_df.columns:
        param_col = "Mbar"
    else:
        raise ValueError("robust_ci must have 'M', 'm', or 'Mbar' column")

    m_values = robust_df[param_col].unique().sort().to_numpy()
    m_gap = np.min(np.diff(m_values)) if len(m_values) > 1 else m_values[0] if len(m_values) > 0 else 1.0
    original_m = m_values[0] - m_gap

    original_row = pl.DataFrame(
        {
            param_col: [original_m],
            "lb": [original.lb],
            "ub": [original.ub],
            "method": [getattr(original, "method", "Original")],
        }
    )

    combined = pl.concat([original_row, robust_df.select([param_col, "lb", "ub", "method"])])
    combined = combined.with_columns(
        [
            ((pl.col("lb") + pl.col("ub")) / 2).alias("midpoint"),
        ]
    )
    combined = combined.rename({param_col: "param_value"})

    return combined.sort(["method", "param_value"])


def dddmpresult_to_polars(result: DDDMultiPeriodResult | DDDMultiPeriodRCResult) -> pl.DataFrame:
    """Convert DDDMultiPeriodResult or DDDMultiPeriodRCResult to polars DataFrame for plotting.

    Parameters
    ----------
    result : DDDMultiPeriodResult or DDDMultiPeriodRCResult
        Multi-period DDD result containing group-time ATT estimates.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - group: treatment cohort
        - time: time period
        - att: group-time ATT estimate
        - se: standard error
        - ci_lower: lower confidence interval
        - ci_upper: upper confidence interval
        - treatment_status: "Pre" or "Post" treatment
    """
    groups = result.groups
    times = result.times
    att = result.att
    se = result.se

    ci_lower = result.lci
    ci_upper = result.uci

    treatment_status = np.array(["Pre" if t < g else "Post" for g, t in zip(groups, times, strict=False)])

    df = pl.DataFrame(
        {
            "group": groups,
            "time": times,
            "att": att,
            "se": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "treatment_status": treatment_status,
        }
    )
    return df.filter(~pl.col("se").is_nan())


def dddaggresult_to_polars(result: DDDAggResult) -> pl.DataFrame:
    """Convert DDDAggResult to polars DataFrame for plotting.

    Parameters
    ----------
    result : DDDAggResult
        Aggregated DDD treatment effect result (eventstudy, group, or calendar).

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - event_time: event time (for eventstudy), group (for group), or time (for calendar)
        - att: ATT estimate
        - se: standard error
        - ci_lower: lower confidence interval
        - ci_upper: upper confidence interval
        - treatment_status: "Pre" or "Post" (for eventstudy aggregation)

    Raises
    ------
    ValueError
        If result is simple aggregation or missing required data.
    """
    if result.aggregation_type == "simple":
        raise ValueError("Simple aggregation does not produce event-level data for plotting.")

    if result.egt is None or result.att_egt is None or result.se_egt is None:
        raise ValueError(
            f"DDDAggResult with aggregation_type='{result.aggregation_type}' must have egt, att_egt, and se_egt"
        )

    event_times = result.egt
    att = result.att_egt
    se = result.se_egt

    crit_val = result.crit_val if result.crit_val is not None else 1.96

    ci_lower = att - crit_val * se
    ci_upper = att + crit_val * se

    data = {
        "event_time": event_times,
        "att": att,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }

    if result.aggregation_type == "eventstudy":
        data["treatment_status"] = np.array(["Pre" if e < 0 else "Post" for e in event_times])

    df = pl.DataFrame(data)
    return df.filter(~pl.col("se").is_nan())


def didinterresult_to_polars(result: DIDInterResult) -> pl.DataFrame:
    """Convert DIDInterResult to polars DataFrame for plotting.

    Parameters
    ----------
    result : DIDInterResult
        Intertemporal treatment effects result from did_multiplegt().

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - horizon: event horizon (negative for placebos, positive for effects)
        - att: effect estimate
        - se: standard error
        - ci_lower: lower confidence interval
        - ci_upper: upper confidence interval
        - treatment_status: "Pre" or "Post" treatment
    """
    effects = result.effects
    placebos = result.placebos

    horizons = []
    att = []
    se = []
    ci_lower = []
    ci_upper = []
    treatment_status = []

    if placebos is not None:
        sorted_indices = np.argsort(placebos.horizons)
        horizons.extend(placebos.horizons[sorted_indices])
        att.extend(placebos.estimates[sorted_indices])
        se.extend(placebos.std_errors[sorted_indices])
        ci_lower.extend(placebos.ci_lower[sorted_indices])
        ci_upper.extend(placebos.ci_upper[sorted_indices])
        treatment_status.extend(["Pre"] * len(placebos.horizons))

    sorted_indices = np.argsort(effects.horizons)
    horizons.extend(effects.horizons[sorted_indices])
    att.extend(effects.estimates[sorted_indices])
    se.extend(effects.std_errors[sorted_indices])
    ci_lower.extend(effects.ci_lower[sorted_indices])
    ci_upper.extend(effects.ci_upper[sorted_indices])
    treatment_status.extend(["Post"] * len(effects.horizons))

    return pl.DataFrame(
        {
            "horizon": horizons,
            "att": att,
            "se": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "treatment_status": treatment_status,
        }
    )


def heterogeneityresult_to_polars(result: HeterogeneityResult) -> pl.DataFrame:
    """Convert HeterogeneityResult to polars DataFrame.

    Parameters
    ----------
    result : HeterogeneityResult
        Heterogeneous effects analysis result for a single horizon.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - Horizon: effect horizon analyzed
        - Covariate: covariate name
        - Estimate: coefficient estimate
        - Std. Error: standard error
        - t-stat: t-statistic
        - CI Lower: lower confidence interval bound
        - CI Upper: upper confidence interval bound
        - N: number of observations
        - F p-value: joint F-test p-value
    """
    return pl.DataFrame(
        {
            "Horizon": [result.horizon] * len(result.covariates),
            "Covariate": result.covariates,
            "Estimate": result.estimates,
            "Std. Error": result.std_errors,
            "t-stat": result.t_stats,
            "CI Lower": result.ci_lower,
            "CI Upper": result.ci_upper,
            "N": [result.n_obs] * len(result.covariates),
            "F p-value": [result.f_pvalue] * len(result.covariates),
        }
    )


def emfxresult_to_polars(result: EmfxResult) -> pl.DataFrame:
    """Convert EmfxResult to polars DataFrame for plotting.

    Parameters
    ----------
    result : EmfxResult
        Aggregated ETWFE marginal effects result (event, group, or calendar).

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - event_time: event time (for event), group (for group), or time (for calendar)
        - att: ATT estimate
        - se: standard error
        - ci_lower: lower confidence interval
        - ci_upper: upper confidence interval
        - treatment_status: "Pre" or "Post" (for event aggregation)

    Raises
    ------
    ValueError
        If result is simple aggregation or missing required data.
    """
    if result.aggregation_type == "simple":
        raise ValueError("Simple aggregation does not produce event-level data for plotting.")

    if result.event_times is None or result.att_by_event is None or result.se_by_event is None:
        raise ValueError(
            f"EmfxResult with aggregation_type='{result.aggregation_type}' "
            "must have event_times, att_by_event, and se_by_event"
        )

    event_times = result.event_times
    att = result.att_by_event
    se = result.se_by_event
    crit_val = result.critical_value

    ci_lower = att - crit_val * se
    ci_upper = att + crit_val * se

    data = {
        "event_time": event_times,
        "att": att,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }

    if result.aggregation_type == "event":
        data["treatment_status"] = np.array(["Pre" if e < 0 else "Post" for e in event_times])

    df = pl.DataFrame(data)
    return df.filter(~pl.col("se").is_nan())


def dynbalancingresult_to_polars(result: DynBalancingResult) -> pl.DataFrame:
    """Convert DynBalancingResult to polars DataFrame for plotting.

    Returns one row per parameter (ATE, ``mu(ds1)``, ``mu(ds2)``) with point
    estimates, standard errors, and both robust (chi-squared) and Gaussian
    confidence interval bounds. The robust quantile for the potential
    outcomes is recomputed using the appropriate degrees of freedom.

    Parameters
    ----------
    result : DynBalancingResult
        Single dynamic covariate balancing result.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - parameter: parameter label ("ATE", "mu(ds1)", or "mu(ds2)")
        - estimate: point estimate
        - se: standard error
        - ci_lower_robust: robust (chi-squared) lower CI
        - ci_upper_robust: robust (chi-squared) upper CI
        - ci_lower_gaussian: Gaussian lower CI
        - ci_upper_gaussian: Gaussian upper CI
    """
    params = result.estimation_params
    alpha = params.get("alpha", 0.05)
    n_periods = params.get("n_periods", 1)

    robust_q_ate = result.robust_quantile
    gaussian_q = result.gaussian_quantile

    robust_q_mu = math.sqrt(chi2.ppf(1.0 - alpha, n_periods)) if alpha is not None and n_periods >= 1 else robust_q_ate

    estimates = np.array([result.att, result.mu1, result.mu2])
    variances = np.array([result.var_att, result.var_mu1, result.var_mu2])
    ses = np.sqrt(np.maximum(variances, 0.0))
    robust_qs = np.array([robust_q_ate, robust_q_mu, robust_q_mu])

    return pl.DataFrame(
        {
            "parameter": ["ATE", "mu(ds1)", "mu(ds2)"],
            "estimate": estimates,
            "se": ses,
            "ci_lower_robust": estimates - robust_qs * ses,
            "ci_upper_robust": estimates + robust_qs * ses,
            "ci_lower_gaussian": estimates - gaussian_q * ses,
            "ci_upper_gaussian": estimates + gaussian_q * ses,
        }
    )


def dynbalancinghistoryresult_to_polars(
    result: DynBalancingHistoryResult,
    parameter: str = "att",
) -> pl.DataFrame:
    """Convert DynBalancingHistoryResult to polars DataFrame for plotting.

    Returns one row per history length with the chosen parameter's point
    estimate, standard error, and both robust and Gaussian confidence
    interval bounds.

    Parameters
    ----------
    result : DynBalancingHistoryResult
        History-mode dynamic covariate balancing result.
    parameter : {"att", "mu1", "mu2"}, default="att"
        Which parameter to extract.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - period_length: treatment history length
        - estimate: point estimate
        - se: standard error
        - ci_lower_robust: robust (chi-squared) lower CI
        - ci_upper_robust: robust (chi-squared) upper CI
        - ci_lower_gaussian: Gaussian lower CI
        - ci_upper_gaussian: Gaussian upper CI
    """
    if parameter not in ("att", "mu1", "mu2"):
        raise ValueError(f"parameter must be one of 'att', 'mu1', 'mu2', got {parameter!r}")

    var_col = f"var_{parameter}"
    summary = result.summary

    estimates = summary[parameter].to_numpy()
    variances = summary[var_col].to_numpy()
    ses = np.sqrt(np.maximum(variances, 0.0))
    robust_qs = summary["robust_quantile"].to_numpy()
    gaussian_qs = summary["gaussian_quantile"].to_numpy()

    return pl.DataFrame(
        {
            "period_length": summary["period_length"].to_numpy(),
            "estimate": estimates,
            "se": ses,
            "ci_lower_robust": estimates - robust_qs * ses,
            "ci_upper_robust": estimates + robust_qs * ses,
            "ci_lower_gaussian": estimates - gaussian_qs * ses,
            "ci_upper_gaussian": estimates + gaussian_qs * ses,
        }
    )


def dynbalancinghetresult_to_polars(
    result: DynBalancingHetResult,
    parameter: str = "att",
) -> pl.DataFrame:
    """Convert DynBalancingHetResult to polars DataFrame for plotting.

    Parameters
    ----------
    result : DynBalancingHetResult
        Het-mode dynamic covariate balancing result.
    parameter : {"att", "mu1", "mu2"}, default="att"
        Which parameter to extract.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``final_period``, ``estimate``, ``se``,
        ``ci_lower_robust``, ``ci_upper_robust``, ``ci_lower_gaussian``,
        ``ci_upper_gaussian``.
    """
    if parameter not in ("att", "mu1", "mu2"):
        raise ValueError(f"parameter must be one of 'att', 'mu1', 'mu2', got {parameter!r}")

    var_col = f"var_{parameter}"
    summary = result.summary

    estimates = summary[parameter].to_numpy()
    variances = summary[var_col].to_numpy()
    ses = np.sqrt(np.maximum(variances, 0.0))
    robust_qs = summary["robust_quantile"].to_numpy()
    gaussian_qs = summary["gaussian_quantile"].to_numpy()

    return pl.DataFrame(
        {
            "final_period": summary["final_period"].to_numpy(),
            "estimate": estimates,
            "se": ses,
            "ci_lower_robust": estimates - robust_qs * ses,
            "ci_upper_robust": estimates + robust_qs * ses,
            "ci_lower_gaussian": estimates - gaussian_qs * ses,
            "ci_upper_gaussian": estimates + gaussian_qs * ses,
        }
    )


def dynbalancingcoefs_to_polars(result: DynBalancingResult, history: str = "ds1") -> pl.DataFrame:
    """Convert LASSO coefficients from a DynBalancingResult to polars DataFrame.

    Returns one row per (period, covariate) combination with the
    estimated coefficient value. The intercept (index 0) is excluded.

    Parameters
    ----------
    result : DynBalancingResult
        Single dynamic covariate balancing result.
    history : {"ds1", "ds2"}, default="ds1"
        Which treatment history's coefficients to extract.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``period``, ``covariate``, ``coefficient``,
        ``is_nonzero``.
    """
    if history not in ("ds1", "ds2"):
        raise ValueError(f"history must be 'ds1' or 'ds2', got {history!r}")

    coefs = result.coefficients.get(history, [])
    if not coefs:
        return pl.DataFrame({"period": [], "covariate": [], "coefficient": [], "is_nonzero": []})

    rows: list[dict] = []
    for t, coef_arr in enumerate(coefs):
        coef_no_intercept = coef_arr[1:]
        for j, val in enumerate(coef_no_intercept):
            rows.append(
                {
                    "period": t + 1,
                    "covariate": f"X{j + 1}",
                    "coefficient": float(val),
                    "is_nonzero": bool(abs(val) > 1e-10),
                }
            )

    return pl.DataFrame(rows)


_DISPATCH: dict[str, Any] = {
    "MPResult": mpresult_to_polars,
    "AGGTEResult": aggteresult_to_polars,
    "DoseResult": doseresult_to_polars,
    "PTEResult": pteresult_to_polars,
    "HonestDiDResult": honestdid_to_polars,
    "DDDMultiPeriodResult": dddmpresult_to_polars,
    "DDDMultiPeriodRCResult": dddmpresult_to_polars,
    "DDDAggResult": dddaggresult_to_polars,
    "DIDInterResult": didinterresult_to_polars,
    "HeterogeneityResult": heterogeneityresult_to_polars,
    "EmfxResult": emfxresult_to_polars,
    "DynBalancingResult": dynbalancingresult_to_polars,
    "DynBalancingHistoryResult": dynbalancinghistoryresult_to_polars,
    "DynBalancingHetResult": dynbalancinghetresult_to_polars,
}


def to_df(result: Any, **kwargs: Any) -> pl.DataFrame:
    """Convert any ModernDiD result object to a polars DataFrame.

    Parameters
    ----------
    result : Any
        A ModernDiD result object. Supported types:

        - :class:`~moderndid.did.container.AGGTEResult`
        - :class:`~moderndid.did.container.MPResult`
        - :class:`~moderndid.didcont.container.DoseResult`
        - :class:`~moderndid.didcont.container.PTEResult`
        - :class:`~moderndid.didhonest.honest_did.HonestDiDResult`
        - :class:`~moderndid.didtriple.container.DDDAggResult`
        - :class:`~moderndid.didtriple.container.DDDMultiPeriodResult`
        - :class:`~moderndid.didtriple.container.DDDMultiPeriodRCResult`
        - :class:`~moderndid.didinter.container.DIDInterResult`
        - :class:`~moderndid.didinter.container.HeterogeneityResult`
    **kwargs
        Additional arguments passed to the underlying converter.
        For example, ``effect_type="acrt"`` for DoseResult.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns appropriate to the result type.

    Examples
    --------
    Convert group-time ATT results from :func:`~moderndid.att_gt` into a tidy DataFrame
    with one row per (group, time) cell:

    .. ipython::
        :okwarning:

        In [1]: from moderndid import att_gt, aggte, load_mpdta, to_df
           ...:
           ...: df = load_mpdta()
           ...: result = att_gt(
           ...:     data=df,
           ...:     yname="lemp",
           ...:     tname="year",
           ...:     gname="first.treat",
           ...:     idname="countyreal",
           ...:     est_method="dr",
           ...:     boot=False,
           ...: )
           ...: print(to_df(result).head())

    Aggregated event-study results work the same way:

    .. ipython::
        :okwarning:

        In [2]: agg = aggte(result, type="dynamic")
           ...: print(to_df(agg))
    """
    type_name = type(result).__name__
    converter = _DISPATCH.get(type_name)
    if converter is None:
        raise TypeError(f"No converter for {type_name!r}. Supported types: {', '.join(sorted(_DISPATCH))}")
    return converter(result, **kwargs)
