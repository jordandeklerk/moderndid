"""Converters for transforming DiD result objects to polars DataFrames."""

import math

import numpy as np
import polars as pl
from scipy.stats import chi2


def _whole_numbers_as_int(df, columns):
    """Cast each listed float column whose values are all whole numbers to integers."""
    casts = []
    for name in columns:
        values = df[name].drop_nulls()
        if values.dtype.is_float() and values.is_finite().all() and (values == values.round(0)).all():
            casts.append(pl.col(name).cast(pl.Int64))
    return df.with_columns(casts)


def mpresult_to_polars(result):
    """Convert MPResult or DIDMLResult to polars DataFrame for plotting.

    The ``group`` and ``time`` columns hold integers when every value in
    them is a whole number.

    Parameters
    ----------
    result : MPResult or DIDMLResult
        Multi-period DID result containing group-time ATT estimates.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - **group**: Treatment cohort
        - **time**: Time period
        - **att**: Group-time ATT estimate
        - **se**: Standard error
        - **ci_lower**: Lower end of the band
        - **ci_upper**: Upper end of the band
        - **treatment_status**: "Pre" or "Post" treatment
    """
    groups = result.groups
    times = result.times
    att = result.att_gt
    se = result.se_gt
    crit_val = result.critical_value

    ci_lower = att - crit_val * se
    ci_upper = att + crit_val * se

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
    return _whole_numbers_as_int(df, ["group", "time"])


def aggteresult_to_polars(result):
    """Convert AGGTEResult or DIDMLAggResult to polars DataFrame for plotting.

    The ``event_time`` column holds integers when every value in it is a
    whole number.

    Parameters
    ----------
    result : AGGTEResult or DIDMLAggResult
        Aggregated treatment effect result (dynamic, group, or calendar).

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - **event_time**: Event time for a dynamic aggregation, cohort for a group aggregation, or period for a
          calendar aggregation
        - **att**: ATT estimate
        - **se**: Standard error
        - **ci_lower**: Lower end of the band
        - **ci_upper**: Upper end of the band
        - **treatment_status**: "Pre" or "Post", for a dynamic aggregation only
    """
    if result.aggregation_type == "simple":
        raise ValueError(
            "Simple aggregation does not produce event-level data for plotting. "
            "Pass a dynamic, group, or calendar aggregation."
        )

    if result.event_times is None or result.att_by_event is None or result.se_by_event is None:
        raise ValueError(
            f"{type(result).__name__} with aggregation_type='{result.aggregation_type}' "
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

    df = pl.DataFrame(data).filter(~pl.col("se").is_nan())
    return _whole_numbers_as_int(df, ["event_time"])


def doseresult_to_polars(result, effect_type="att"):
    """Convert DoseResult to polars DataFrame for plotting.

    Parameters
    ----------
    result : DoseResult
        Continuous treatment dose-response result.
    effect_type : {'att', 'acrt'}, default='att'
        Curve to extract. ``'att'`` takes the level effects ATT(d) and
        ``'acrt'`` takes their slope ACRT(d).

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - **dose**: Dose level
        - **effect**: Estimate of ATT(d) or ACRT(d)
        - **se**: Standard error
        - **ci_lower**: Lower end of the band
        - **ci_upper**: Upper end of the band
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


def pteresult_to_polars(result):
    """Convert PTEResult event study to polars DataFrame for plotting.

    The row of a universal base period, whose standard error is undefined,
    is left out. The ``event_time`` column holds integers when every value
    in it is a whole number.

    Parameters
    ----------
    result : PTEResult
        Panel treatment effects result with event_study.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - **event_time**: Event time relative to treatment
        - **att**: Estimate at that event time
        - **se**: Standard error
        - **ci_lower**: Lower end of the band
        - **ci_upper**: Upper end of the band
        - **treatment_status**: "Pre" or "Post" treatment
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
    ).filter(~pl.col("se").is_nan())
    return _whole_numbers_as_int(df, ["event_time"])


def honestdid_to_polars(result):
    """Convert HonestDiDResult to polars DataFrame for plotting.

    Parameters
    ----------
    result : HonestDiDResult
        Honest DiD sensitivity analysis result.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - **param_value**: Value of M or Mbar
        - **lb**: Lower bound of the confidence interval
        - **ub**: Upper bound of the confidence interval
        - **method**: Name of the confidence interval method
        - **midpoint**: Midpoint of the bounds, (lb + ub) / 2

        The original confidence interval enters as one more row whose
        param_value comes before the smallest robust one.
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


def dddmpresult_to_polars(result):
    """Convert DDDMultiPeriodResult or DDDMultiPeriodRCResult to polars DataFrame for plotting.

    The ``group`` and ``time`` columns hold integers when every value in
    them is a whole number.

    Parameters
    ----------
    result : DDDMultiPeriodResult or DDDMultiPeriodRCResult
        Multi-period DDD result containing group-time ATT estimates.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - **group**: Treatment cohort
        - **time**: Time period
        - **att**: Group-time ATT estimate
        - **se**: Standard error
        - **ci_lower**: Lower end of the band
        - **ci_upper**: Upper end of the band
        - **treatment_status**: "Pre" or "Post" treatment
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
    ).filter(~pl.col("se").is_nan())
    return _whole_numbers_as_int(df, ["group", "time"])


def dddaggresult_to_polars(result):
    """Convert DDDAggResult to polars DataFrame for plotting.

    The ``event_time`` column holds integers when every value in it is a
    whole number.

    Parameters
    ----------
    result : DDDAggResult
        Aggregated DDD treatment effect result (eventstudy, group, or calendar).

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - **event_time**: Event time for an eventstudy aggregation, cohort for a group aggregation, or period for
          a calendar aggregation
        - **att**: ATT estimate
        - **se**: Standard error
        - **ci_lower**: Lower end of the band
        - **ci_upper**: Upper end of the band
        - **treatment_status**: "Pre" or "Post", for an eventstudy aggregation only
    """
    if result.aggregation_type == "simple":
        raise ValueError(
            "Simple aggregation does not produce event-level data for plotting. "
            "Pass an eventstudy, group, or calendar aggregation."
        )

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

    df = pl.DataFrame(data).filter(~pl.col("se").is_nan())
    return _whole_numbers_as_int(df, ["event_time"])


def didinterresult_to_polars(result):
    """Convert DIDInterResult to polars DataFrame for plotting.

    The ``horizon`` column holds integers when every value in it is a
    whole number.

    Parameters
    ----------
    result : DIDInterResult
        Intertemporal treatment effects result from did_multiplegt().

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - **horizon**: Event horizon, negative for placebos and positive for effects
        - **att**: Effect estimate
        - **se**: Standard error
        - **ci_lower**: Lower end of the confidence interval
        - **ci_upper**: Upper end of the confidence interval
        - **treatment_status**: "Pre" or "Post" treatment
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

    df = pl.DataFrame(
        {
            "horizon": horizons,
            "att": att,
            "se": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "treatment_status": treatment_status,
        }
    )
    return _whole_numbers_as_int(df, ["horizon"])


def heterogeneityresult_to_polars(result):
    """Convert HeterogeneityResult to polars DataFrame.

    The ``Horizon`` column holds an integer when the horizon is a whole
    number.

    Parameters
    ----------
    result : HeterogeneityResult
        Heterogeneous effects analysis result for a single horizon.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - **Horizon**: Effect horizon analyzed
        - **Covariate**: Covariate name
        - **Estimate**: Coefficient estimate
        - **Std. Error**: Standard error
        - **t-stat**: t-statistic
        - **CI Lower**: Lower end of the confidence interval
        - **CI Upper**: Upper end of the confidence interval
        - **N**: Number of observations
        - **F p-value**: p-value of the joint F-test
    """
    df = pl.DataFrame(
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
    return _whole_numbers_as_int(df, ["Horizon"])


def emfxresult_to_polars(result):
    """Convert EmfxResult to polars DataFrame for plotting.

    The ``event_time`` column holds integers when every value in it is a
    whole number.

    Parameters
    ----------
    result : EmfxResult
        Aggregated ETWFE marginal effects result (event, group, or calendar).

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - **event_time**: Event time for an event aggregation, cohort for a group aggregation, or period for a
          calendar aggregation
        - **att**: ATT estimate
        - **se**: Standard error
        - **ci_lower**: Lower end of the confidence interval
        - **ci_upper**: Upper end of the confidence interval
        - **treatment_status**: "Pre" or "Post", for an event aggregation only
    """
    if result.aggregation_type == "simple":
        raise ValueError(
            "Simple aggregation does not produce event-level data for plotting. "
            "Pass an event, group, or calendar aggregation."
        )

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

    df = pl.DataFrame(data).filter(~pl.col("se").is_nan())
    return _whole_numbers_as_int(df, ["event_time"])


def dynbalancingresult_to_polars(result):
    """Convert DynBalancingResult to polars DataFrame for plotting.

    Returns one row per parameter (ATE, ``mu(ds1)``, ``mu(ds2)``) with point
    estimates, standard errors, and both robust and Gaussian confidence
    interval bounds. The robust bounds follow the estimator's
    ``robust_quantile`` setting. With the setting on, the potential outcomes
    get a chi-squared quantile with their own degrees of freedom. With it
    off, the robust bounds equal the Gaussian ones.

    Parameters
    ----------
    result : DynBalancingResult
        Single dynamic covariate balancing result.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - **parameter**: Parameter label, "ATE", "mu(ds1)", or "mu(ds2)"
        - **estimate**: Point estimate
        - **se**: Standard error
        - **ci_lower_robust**: Lower end of the robust interval
        - **ci_upper_robust**: Upper end of the robust interval
        - **ci_lower_gaussian**: Lower end of the Gaussian interval
        - **ci_upper_gaussian**: Upper end of the Gaussian interval
    """
    robust_q_ate = result.robust_quantile
    gaussian_q = result.gaussian_quantile
    robust_q_mu = _dynbalancing_mu_quantile(result)

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


def dynbalancinghistoryresult_to_polars(result, parameter="att"):
    """Convert DynBalancingHistoryResult to polars DataFrame for plotting.

    Returns one row per history length with the chosen parameter's point
    estimate, standard error, and both robust and Gaussian confidence
    interval bounds. As in :func:`dynbalancingresult_to_polars`, the robust
    bounds of each row follow the settings of its estimate. The
    ``period_length`` column holds integers when every value in it is a
    whole number.

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

        - **period_length**: Treatment history length
        - **estimate**: Point estimate
        - **se**: Standard error
        - **ci_lower_robust**: Lower end of the robust interval
        - **ci_upper_robust**: Upper end of the robust interval
        - **ci_lower_gaussian**: Lower end of the Gaussian interval
        - **ci_upper_gaussian**: Upper end of the Gaussian interval
    """
    if parameter not in ("att", "mu1", "mu2"):
        raise ValueError(f"parameter must be one of 'att', 'mu1', 'mu2', got {parameter!r}")

    var_col = f"var_{parameter}"
    summary = result.summary

    estimates = summary[parameter].to_numpy()
    variances = summary[var_col].to_numpy()
    ses = np.sqrt(np.maximum(variances, 0.0))
    robust_qs = _dynbalancing_summary_quantiles(result, parameter)
    gaussian_qs = summary["gaussian_quantile"].to_numpy()

    df = pl.DataFrame(
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
    return _whole_numbers_as_int(df, ["period_length"])


def dynbalancinghetresult_to_polars(result, parameter="att"):
    """Convert DynBalancingHetResult to polars DataFrame for plotting.

    As in :func:`dynbalancingresult_to_polars`, the robust bounds of each row
    follow the settings of its estimate. The ``final_period`` column holds
    integers when every value in it is a whole number.

    Parameters
    ----------
    result : DynBalancingHetResult
        Het-mode dynamic covariate balancing result.
    parameter : {"att", "mu1", "mu2"}, default="att"
        Which parameter to extract.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - **final_period**: Final period of the treatment histories
        - **estimate**: Point estimate
        - **se**: Standard error
        - **ci_lower_robust**: Lower end of the robust interval
        - **ci_upper_robust**: Upper end of the robust interval
        - **ci_lower_gaussian**: Lower end of the Gaussian interval
        - **ci_upper_gaussian**: Upper end of the Gaussian interval
    """
    if parameter not in ("att", "mu1", "mu2"):
        raise ValueError(f"parameter must be one of 'att', 'mu1', 'mu2', got {parameter!r}")

    var_col = f"var_{parameter}"
    summary = result.summary

    estimates = summary[parameter].to_numpy()
    variances = summary[var_col].to_numpy()
    ses = np.sqrt(np.maximum(variances, 0.0))
    robust_qs = _dynbalancing_summary_quantiles(result, parameter)
    gaussian_qs = summary["gaussian_quantile"].to_numpy()

    df = pl.DataFrame(
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
    return _whole_numbers_as_int(df, ["final_period"])


def _dynbalancing_mu_quantile(result):
    """Return the robust critical value of a potential outcome in a dynamic covariate balancing result."""
    params = result.estimation_params
    # Since the robust quantile equals the Gaussian one only when the setting is off, a result
    # built without the setting still reveals it.
    if not params.get("robust_quantile", result.robust_quantile != result.gaussian_quantile):
        return result.gaussian_quantile
    return math.sqrt(chi2.ppf(1.0 - params.get("alpha", 0.05), params.get("n_periods", 1)))


def _dynbalancing_summary_quantiles(result, parameter):
    """Return the robust critical value of each row of a history or heterogeneity summary."""
    summary = result.summary
    # A summary built without its estimates carries only the critical value of the ATE.
    if parameter == "att" or len(result.results) != summary.height:
        return summary["robust_quantile"].to_numpy()
    return np.array([_dynbalancing_mu_quantile(r) for r in result.results])


def dynbalancingcoefs_to_polars(result, history="ds1"):
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
        DataFrame with columns:

        - **period**: Period of the coefficients, counted from 1
        - **covariate**: Covariate label, "X1", "X2", and so on
        - **coefficient**: Estimated coefficient
        - **is_nonzero**: Whether the LASSO kept the covariate
    """
    if history not in ("ds1", "ds2"):
        raise ValueError(f"history must be 'ds1' or 'ds2', got {history!r}")

    coefs = result.coefficients.get(history, [])
    if not coefs:
        return pl.DataFrame({"period": [], "covariate": [], "coefficient": [], "is_nonzero": []})

    rows = []
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


def blpresult_to_polars(result):
    """Convert BLPResult to polars DataFrame.

    Returns one row per event time and coefficient. Rows are ordered by event
    time and then by the key order of ``result.coefs``. The ``event_time``
    column holds integers when every value in it is a whole number.

    Parameters
    ----------
    result : BLPResult
        Best linear predictor result from :func:`~moderndid.blp_eventtimes`.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - **event_time**: Event time of the regression
        - **covariate**: Name of the coefficient, "Intercept" for the constant
        - **estimate**: Coefficient estimate
        - **se**: Standard error
        - **pvalue**: P-value of the coefficient
    """
    rows = [
        (event_time, name, result.coefs[name][k], result.ses[name][k], result.pvalues[name][k])
        for k, event_time in enumerate(result.event_times)
        for name in result.coefs
    ]

    df = pl.DataFrame(rows, schema=["event_time", "covariate", "estimate", "se", "pvalue"], orient="row")
    return _whole_numbers_as_int(df, ["event_time"])


def clanresult_to_polars(result):
    """Convert CLANResult to polars DataFrame.

    Returns one row per covariate in ``result.affected``, in that order.

    Parameters
    ----------
    result : CLANResult
        Classification analysis result for one event time, from
        :func:`~moderndid.clan_ttest` or :func:`~moderndid.clan_glhtest`.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - **covariate**: Covariate name
        - **high_mean**: Mean of the covariate in the top CATT group
        - **low_mean**: Mean of the covariate in the bottom CATT group
        - **diff**: High mean minus low mean
        - **pvalue**: P-value of the difference
    """
    return pl.DataFrame(
        {
            "covariate": result.affected,
            "high_mean": result.high_means,
            "low_mean": result.low_means,
            "diff": result.diffs,
            "pvalue": result.pvalues,
        }
    )


_DISPATCH = {
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
    "DIDMLResult": mpresult_to_polars,
    "DIDMLAggResult": aggteresult_to_polars,
    "BLPResult": blpresult_to_polars,
    "CLANResult": clanresult_to_polars,
}


def to_df(result, **kwargs):
    """Convert any ModernDiD result object to a polars DataFrame.

    Columns that hold cohorts, periods, event times, horizons, or history
    lengths come back as integers when every value in them is a whole
    number. Otherwise they stay float columns.

    See the :ref:`plotting guide <plotting-extracting-data>` for an event study
    converted to a polars DataFrame and the columns it holds.

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
        - :class:`~moderndid.EmfxResult`
        - :class:`~moderndid.diddynamic.DynBalancingResult`
        - :class:`~moderndid.diddynamic.DynBalancingHistoryResult`
        - :class:`~moderndid.diddynamic.DynBalancingHetResult`
        - :class:`~moderndid.DIDMLResult`
        - :class:`~moderndid.DIDMLAggResult`
        - :class:`~moderndid.BLPResult`
        - :class:`~moderndid.CLANResult`
    **kwargs
        Additional arguments passed to the underlying converter.
        For example, ``effect_type="acrt"`` for DoseResult.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns appropriate to the result type.

    """
    type_name = type(result).__name__
    converter = _DISPATCH.get(type_name)
    if converter is None:
        raise TypeError(f"No converter for {type_name!r}. Supported types: {', '.join(sorted(_DISPATCH))}")
    return converter(result, **kwargs)
