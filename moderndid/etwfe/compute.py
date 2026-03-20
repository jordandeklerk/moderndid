"""Data preparation, formula construction, and regression execution for ETWFE."""

from __future__ import annotations

import re
import warnings
from typing import Any

import formulaic
import numpy as np
import pandas as pd
import polars as pl
import pyfixest as pf
from scipy import stats

from moderndid.core.preprocess.config import EtwfeConfig
from moderndid.core.preprocess.utils import extract_vars_from_formula


def set_references(config: EtwfeConfig, data: pl.DataFrame) -> EtwfeConfig:
    """Auto-detect reference levels for time and cohort.

    Tries never-treated groups first (g > max(t)), then pre-period groups
    (g < min(t)), then the latest cohort when cgroup="notyet".

    Parameters
    ----------
    config : EtwfeConfig
        ETWFE configuration (mutated in place).
    data : pl.DataFrame
        Input data.

    Returns
    -------
    EtwfeConfig
        Updated configuration.
    """
    ug = sorted(data[config.gname].drop_nulls().unique().to_list())
    ut = sorted(data[config.tname].drop_nulls().unique().to_list())

    if config.tref is None:
        config.tref = int(min(ut))

    if config.gref is None:
        gref_cands = [g for g in ug if g > max(ut)]
        if len(gref_cands) == 0:
            gref_cands = [g for g in ug if g < min(ut)]
        if len(gref_cands) == 0 and config.cgroup == "notyet":
            gref_cands = [max(ug)]
        if len(gref_cands) == 0:
            raise ValueError(f"Could not identify '{config.cgroup}' control group.")
        config.gref = int(min(gref_cands))

    config._gref_min_flag = config.gref < min(ut)
    return config


def prepare_etwfe_data(data: pl.DataFrame, config: EtwfeConfig) -> pl.DataFrame:
    """Prepare data for ETWFE estimation.

    Creates the treatment indicator, demeans controls, and applies
    reference-level filtering.

    Parameters
    ----------
    data : pl.DataFrame
        Input panel data.
    config : EtwfeConfig
        ETWFE configuration.

    Returns
    -------
    pl.DataFrame
        Prepared data with ``_Dtreat``, ``_g``, ``_t``, and demeaned columns.
    """
    gname, tname = config.gname, config.tname
    gref = config.gref

    df = data.with_columns(
        [
            pl.col(gname).cast(pl.Float64).alias("_g"),
            pl.col(tname).cast(pl.Float64).alias("_t"),
        ]
    )

    if config.cgroup == "notyet":
        df = df.with_columns(
            pl.when((pl.col("_t") >= pl.col("_g")) & (pl.col("_g") != gref)).then(1.0).otherwise(0.0).alias("_Dtreat")
        )
        if not config._gref_min_flag:
            df = df.with_columns(
                pl.when(pl.col("_t") >= gref)
                .then(pl.lit(None, dtype=pl.Float64))
                .otherwise(pl.col("_Dtreat"))
                .alias("_Dtreat")
            )
        else:
            df = df.with_columns(
                pl.when(pl.col("_t") <= gref)
                .then(pl.lit(None, dtype=pl.Float64))
                .otherwise(pl.col("_Dtreat"))
                .alias("_Dtreat")
            )
    else:
        df = df.with_columns(
            pl.when(pl.col("_t") != (pl.col("_g") - 1.0)).then(1.0).otherwise(0.0).alias("_Dtreat")
        ).with_columns(pl.when(pl.col("_g") == gref).then(0.0).otherwise(pl.col("_Dtreat")).alias("_Dtreat"))

    ctrls = _get_control_vars(config)
    for ctrl in ctrls:
        df = df.with_columns(
            (pl.col(ctrl).cast(pl.Float64) - pl.col(ctrl).cast(pl.Float64).mean().over(gname)).alias(f"{ctrl}_dm")
        )

    xvar_dm_cols: list[str] = []
    xvar_time_dummies: list[str] = []
    if config.xvar:
        df, xvar_dm_cols = _build_xvar_dm_columns(df, config)
        all_times = sorted(df[tname].drop_nulls().unique().to_list())
        for dm_col in xvar_dm_cols:
            for t in all_times:
                if t == config.tref:
                    continue
                name = f"_t{int(t)}_{dm_col}"
                df = df.with_columns((pl.when(pl.col(tname) == t).then(pl.col(dm_col)).otherwise(0.0)).alias(name))
                xvar_time_dummies.append(name)

    config._ctrls = ctrls
    config._xvar_dm_cols = xvar_dm_cols
    config._xvar_time_dummies = xvar_time_dummies

    return df


def build_etwfe_formula(config: EtwfeConfig) -> str:
    """Build the regression formula for ETWFE.

    Parameters
    ----------
    config : EtwfeConfig
        ETWFE configuration (must have ``_ctrls``, ``_xvar_dm_cols``, etc. set).

    Returns
    -------
    str
        Complete formula string.
    """
    gcat = "__etwfe_gcat"
    tcat = "__etwfe_tcat"
    main_int = f"C({gcat}):C({tcat})"

    parts: list[str] = [f"_Dtreat:{main_int}"]

    if config._ctrls:
        for ctrl in config._ctrls:
            parts.append(f"_Dtreat:{main_int}:{ctrl}_dm")
        if config.fe != "vs":
            for ctrl in config._ctrls:
                parts.extend([ctrl, f"C({gcat}):{ctrl}", f"C({tcat}):{ctrl}"])

    if config.xvar:
        for dm_col in config._xvar_dm_cols:
            parts.append(f"_Dtreat:{main_int}:{dm_col}")
        if config._xvar_time_dummies:
            parts.extend(config._xvar_time_dummies)

    rhs = " + ".join(parts)

    if config.fe != "none":
        fe_var = config.idname if config.idname else config.gname
        formula = f"{config.yname} ~ {rhs} | {fe_var} + {config.tname}"
    else:
        parts.extend([f"C({gcat})", f"C({tcat})"])
        formula = f"{config.yname} ~ {' + '.join(parts)}"

    return formula


def run_etwfe_regression(
    formula: str,
    data: pl.DataFrame,
    config: EtwfeConfig,
    vcov: str | dict | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    """Run the ETWFE regression via pyfixest.

    Dispatches to ``feols``, ``fepois``, or ``feglm`` based on
    ``config.family``.

    Parameters
    ----------
    formula : str
        Regression formula string.
    data : pl.DataFrame
        Prepared data.
    config : EtwfeConfig
        ETWFE configuration.
    vcov : str or dict or None
        Variance-covariance specification.
    backend : str or None
        Demeaner backend.

    Returns
    -------
    dict
        Dictionary with keys ``model``, ``formula``, ``fit_data``.
    """
    df_clean = data.filter(pl.col("_Dtreat").is_not_null() & pl.col(config.yname).is_not_null())

    pdf = _to_pandas_with_categoricals(df_clean, config)

    vcov_spec = vcov if vcov else "hetero"
    family = config.family

    fit_kwargs: dict[str, Any] = {"fml": formula, "data": pdf, "vcov": vcov_spec}
    if config.weightsname:
        fit_kwargs["weights"] = config.weightsname
    if backend is not None:
        fit_kwargs["demeaner_backend"] = backend

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        if family is None or family == "gaussian":
            model = pf.feols(**fit_kwargs)
        elif family == "poisson":
            model = pf.fepois(**fit_kwargs)
        elif family in ("logit", "probit"):
            model = pf.feglm(**fit_kwargs, family=family)
        else:
            raise ValueError(f"Unsupported family: {family}")

    return {
        "model": model,
        "formula": formula,
        "fit_data": df_clean,
    }


def compute_emfx(
    model: Any,
    fit_data: pl.DataFrame,
    config: EtwfeConfig,
    agg_type: str = "simple",
    post_only: bool = True,
    window: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Compute marginal effects from the fitted ETWFE model.

    For linear models, uses direct coefficient extraction. For nonlinear
    models, constructs counterfactual predictions (Dtreat=1 vs Dtreat=0)
    and applies the inverse link function.

    Parameters
    ----------
    model : Feols, Fepois, or Feglm
        Fitted model object.
    fit_data : pl.DataFrame
        Data used for fitting.
    config : EtwfeConfig
        ETWFE configuration.
    agg_type : str
        Aggregation type: ``"simple"``, ``"group"``, ``"calendar"``, ``"event"``.
    post_only : bool
        Only include post-treatment observations.
    window : tuple or None
        Event window for filtering.

    Returns
    -------
    dict
        Keys: ``overall_att``, ``overall_se``, ``event_times``,
        ``att_by_event``, ``se_by_event``.
    """
    df = fit_data

    if config.cgroup == "never":
        df = df.filter(pl.col("_g") != config.gref)
        if agg_type != "event":
            df = df.filter((pl.col("_Dtreat") == 1.0) & (pl.col("_t") >= pl.col("_g")))
        elif not post_only:
            pass
        else:
            df = df.filter(pl.col("_Dtreat") == 1.0)
    else:
        df = df.filter(pl.col("_Dtreat") == 1.0)

    if agg_type == "event":
        df = df.with_columns((pl.col("_t") - pl.col("_g")).cast(pl.Int64).alias("event"))

    if window is not None and agg_type == "event":
        df = df.filter((pl.col("event") >= window[0]) & (pl.col("event") <= window[1]))

    coef_names = [str(c) for c in model._coefnames]
    beta = np.asarray(model._beta_hat, dtype=float)
    vcov_matrix = np.asarray(model._vcov, dtype=float)

    is_linear = config.family is None or config.family == "gaussian"

    if is_linear:
        slopes, jacobians = _compute_linear_slopes(df, coef_names, beta)
    else:
        slopes, jacobians = _compute_nonlinear_slopes(df, model, config, coef_names, beta)

    weights = np.ones(len(df), dtype=float)

    if agg_type == "simple":
        att, se = _weighted_agg(slopes, jacobians, weights, vcov_matrix)
        return {"overall_att": att, "overall_se": se, "event_times": None, "att_by_event": None, "se_by_event": None}

    if agg_type == "event":
        group_vals = df["event"].to_numpy()
    elif agg_type == "group":
        group_vals = df[config.gname].to_numpy()
    else:
        group_vals = df[config.tname].to_numpy()

    unique_vals = np.sort(np.unique(group_vals))
    att_list, se_list = [], []
    for val in unique_vals:
        mask = group_vals == val
        att_v, se_v = _weighted_agg(slopes[mask], jacobians[mask], weights[mask], vcov_matrix)
        att_list.append(att_v)
        se_list.append(se_v)

    att_arr = np.array(att_list)
    se_arr = np.array(se_list)
    egt_arr = np.array(unique_vals, dtype=float)

    overall_att, overall_se = _weighted_agg(slopes, jacobians, weights, vcov_matrix)

    return {
        "overall_att": overall_att,
        "overall_se": overall_se,
        "event_times": egt_arr,
        "att_by_event": att_arr,
        "se_by_event": se_arr,
    }


def _get_control_vars(config: EtwfeConfig) -> list[str]:
    """Extract control variable names from the formula."""
    if not config.xformla or config.xformla == "~1":
        return []
    return extract_vars_from_formula(config.xformla)


def _weighted_agg(
    slopes: np.ndarray,
    jacobians: np.ndarray,
    weights: np.ndarray,
    vcov_matrix: np.ndarray | None,
) -> tuple[float, float]:
    """Compute weighted average and delta-method SE for a subset."""
    w_sum = weights.sum()
    if w_sum == 0:
        return 0.0, np.nan
    est = float(np.average(slopes, weights=weights))
    if vcov_matrix is None:
        return est, np.nan
    WJ = jacobians * weights[:, None]
    gbar = WJ.sum(axis=0, keepdims=True) / w_sum
    se = float(np.sqrt(np.clip(gbar @ vcov_matrix @ gbar.T, 0, None)[0, 0]))
    return est, se


def _compute_linear_slopes(
    df: pl.DataFrame,
    coef_names: list[str],
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute slopes via direct coefficient lookup (linear models only)."""
    gt_map = _build_gt_coefficient_map(coef_names)

    g_arr = df["_g"].to_numpy()
    t_arr = df["_t"].to_numpy()

    n = len(df)
    slopes = np.zeros(n)
    jacobians = np.zeros((n, len(beta)))

    for i in range(n):
        key = (g_arr[i], t_arr[i])
        if key in gt_map:
            idx = gt_map[key]
            slopes[i] = beta[idx]
            jacobians[i, idx] = 1.0

    return slopes, jacobians


def _compute_nonlinear_slopes(
    df: pl.DataFrame,
    model: Any,
    config: EtwfeConfig,
    coef_names: list[str],
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute slopes via counterfactual predictions (nonlinear models)."""
    pdf = _to_pandas_with_categoricals(df, config)

    rhs_formula = config._formula.split("|")[0].strip()

    pdf1 = pdf.copy()
    pdf0 = pdf.copy()
    pdf1["_Dtreat"] = 1.0
    pdf0["_Dtreat"] = 0.0

    _, X1_full = formulaic.model_matrix(rhs_formula, pdf1)
    _, X0_full = formulaic.model_matrix(rhs_formula, pdf0)
    X1 = X1_full.reindex(columns=coef_names, fill_value=0.0).to_numpy()
    X0 = X0_full.reindex(columns=coef_names, fill_value=0.0).to_numpy()

    eta1 = X1 @ beta
    eta0 = X0 @ beta

    mu1, d1 = _invlink_and_deriv(eta1, config.family)
    mu0, d0 = _invlink_and_deriv(eta0, config.family)

    slopes = mu1 - mu0
    jacobians = (d1[:, None] * X1) - (d0[:, None] * X0)

    return slopes, jacobians


def _invlink_and_deriv(eta: np.ndarray, family: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Compute inverse link function and its derivative."""
    if family is None or family == "gaussian":
        return eta, np.ones_like(eta)

    if family == "poisson":
        mu = np.exp(eta)
        return mu, mu

    if family == "logit":
        mu = 1.0 / (1.0 + np.exp(-eta))
        return mu, mu * (1.0 - mu)

    if family == "probit":
        mu = stats.norm.cdf(eta)
        return mu, stats.norm.pdf(eta)

    raise ValueError(f"Unsupported family: {family}")


def _build_gt_coefficient_map(coef_names: list[str]) -> dict[tuple[float, float], int]:
    """Map (group, time) pairs to coefficient indices."""
    pattern = re.compile(
        r"_Dtreat:C\(__etwfe_gcat\)\[([^\]]+)\]"
        r":C\(__etwfe_tcat\)\[([^\]]+)\]$"
    )

    gt_map: dict[tuple[float, float], int] = {}
    for i, name in enumerate(coef_names):
        m = pattern.search(name)
        if m:
            try:
                g = float(m.group(1))
                t = float(m.group(2))
                gt_map[(g, t)] = i
            except ValueError:
                continue

    return gt_map


def _build_xvar_dm_columns(df: pl.DataFrame, config: EtwfeConfig) -> tuple[pl.DataFrame, list[str]]:
    """Build cohort-demeaned xvar columns for heterogeneous treatment effects."""
    gname = config.gname
    gref = config.gref

    w = (pl.col(gname).is_not_null() & (pl.col(gname) != gref)).cast(pl.Float64)

    x_col = config.xvar
    dm_cols: list[str] = []

    if df[x_col].dtype.is_numeric() or df[x_col].dtype == pl.Boolean:
        weighted_sum = (w * pl.col(x_col).cast(pl.Float64)).sum().over(gname)
        weight_total = w.sum().over(gname)
        cohort_mean = (weighted_sum / weight_total).fill_nan(0.0).fill_null(0.0)
        dm_name = f"{x_col}_dm"
        df = df.with_columns((pl.col(x_col).cast(pl.Float64) - cohort_mean).alias(dm_name))
        dm_cols.append(dm_name)
        return df, dm_cols

    cats = df[x_col].unique().sort().to_list()
    if len(cats) > 1:
        cats = cats[1:]

    for cat in cats:
        dummy_name = f"{x_col}_{cat}"
        df = df.with_columns(pl.when(pl.col(x_col) == cat).then(1.0).otherwise(0.0).alias(dummy_name))
        weighted_sum = (w * pl.col(dummy_name)).sum().over(gname)
        weight_total = w.sum().over(gname)
        cohort_mean = (weighted_sum / weight_total).fill_nan(0.0).fill_null(0.0)
        dm_name = f"{dummy_name}_dm"
        df = df.with_columns((pl.col(dummy_name) - cohort_mean).alias(dm_name))
        dm_cols.append(dm_name)
        df = df.drop(dummy_name)

    return df, dm_cols


def _to_pandas_with_categoricals(df: pl.DataFrame, config: EtwfeConfig) -> pd.DataFrame:
    """Convert to pandas with ordered categoricals for the regression backend."""
    gref = int(config.gref)
    tref = int(config.tref)

    g_vals = sorted(df[config.gname].drop_nulls().cast(pl.Int64).unique().to_list())
    t_vals = sorted(df[config.tname].drop_nulls().cast(pl.Int64).unique().to_list())

    if gref not in g_vals:
        g_vals = [gref, *g_vals]
    if tref not in t_vals:
        t_vals = [tref, *t_vals]

    g_cats = [gref] + [g for g in g_vals if g != gref]
    t_cats = [tref] + [t for t in t_vals if t != tref]

    pdf = df.to_pandas()
    pdf["_Dtreat"] = pdf["_Dtreat"].astype(float)

    pdf["__etwfe_gcat"] = pd.Categorical(
        pdf[config.gname].astype("Int64").astype(float).astype("Int64"),
        categories=g_cats,
        ordered=True,
    )
    pdf["__etwfe_tcat"] = pd.Categorical(
        pdf[config.tname].astype("Int64").astype(float).astype("Int64"),
        categories=t_cats,
        ordered=True,
    )
    return pdf
