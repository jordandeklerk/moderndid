"""Data preparation, formula construction, and regression execution for ETWFE."""

from __future__ import annotations

import keyword
import warnings

import formulaic
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

from moderndid.core.preprocess.utils import parse_formula
from moderndid.core.preprocess.validators import _duplicate_unit_period_error


def clean_etwfe_data(data, config, vcov=None):
    """Drop the rows and units that cannot enter the regression.

    Two rows for one unit in one period raise an error before any row is
    dropped, since the regression would count that period twice.

    Rows with a missing value in the time, cohort, or unit column, a control,
    the moderator, the weights, or a cluster variable leave the sample before
    any control is demeaned. A missing outcome does not, since the regression
    drops it later.

    Units already treated in the first period leave next. Since their cohort has
    no untreated period, no comparison identifies its effects. Each of the two
    steps warns with the number of rows or units it drops.

    Parameters
    ----------
    data : pl.DataFrame
        Input data.
    config : EtwfeConfig
        ETWFE configuration.
    vcov : str or dict or None, default=None
        Variance-covariance specification. A dict names the cluster variables.

    Returns
    -------
    pl.DataFrame
        The rows that can enter the regression.
    """
    ctrls = _get_control_vars(config)
    missing_cols = [c for c in ctrls if c not in data.columns]
    if missing_cols:
        raise ValueError(f"xformla columns {missing_cols} not found in data columns")

    if config.idname is not None:
        duplicate_error = _duplicate_unit_period_error(data, config.idname, config.tname)
        if duplicate_error is not None:
            raise ValueError(duplicate_error)

    clusters = _cluster_columns(vcov, data.columns)
    cols = [config.tname, config.gname, config.idname, *ctrls, config.xvar, config.weightsname, *clusters]
    cols = list(dict.fromkeys(c for c in cols if c is not None))
    incomplete = [c for c in cols if data.select(_is_missing(data, c).any()).item()]
    if incomplete:
        df = data.filter(~pl.any_horizontal([_is_missing(data, c) for c in incomplete]))
        warnings.warn(
            f"Dropped {data.height - df.height} rows with missing values in {', '.join(incomplete)}.",
            UserWarning,
            stacklevel=3,
        )
        data = df

    if data.height == 0:
        raise ValueError("No rows are left to estimate from.")
    first, last = data[config.tname].min(), data[config.tname].max()
    g = pl.col(config.gname).cast(pl.Float64)
    early = ~_never_treated(g, last) & (g <= first)
    early_rows = data.filter(early)
    if early_rows.height == 0:
        return data

    cohorts = ", ".join(f"{c:g}" for c in sorted(early_rows[config.gname].cast(pl.Float64).unique().to_list()))
    if config.idname is not None:
        units = early_rows[config.idname].unique()
        data = data.filter(~pl.col(config.idname).is_in(units.implode()))
        dropped = f"{units.len()} units"
    else:
        data = data.filter(~early)
        dropped = f"{early_rows.height} rows"
    warnings.warn(
        f"Dropped {dropped} of cohorts already treated in the first period ({cohorts}). Without an untreated "
        "period their effects are not identified.",
        UserWarning,
        stacklevel=3,
    )
    if data.height == 0:
        raise ValueError("No rows are left to estimate from.")
    return data


def set_references(config, data):
    """Pick and check the reference period and cohort.

    The reference period defaults to the first period. The reference cohort
    defaults to the never-treated units. Their cohort is 0, infinite, or later
    than the last period. All of these codes become 0. Any of them as the
    reference cohort refers to every never-treated unit. Without never-treated
    units, the not-yet-treated design uses the latest cohort.

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
    times = sorted(data[config.tname].drop_nulls().unique().to_list())
    first, last = times[0], times[-1]

    if config.tref is None:
        config.tref = int(first)
    elif float(config.tref) not in {float(t) for t in times}:
        raise ValueError(f"tref={config.tref} is not a period in '{config.tname}'.")

    cohorts = data.select(pl.col(config.gname).cast(pl.Float64)).to_series()
    never = data.select(_never_treated(pl.col(config.gname).cast(pl.Float64), last)).to_series()
    treated = sorted(cohorts.filter(~never & (cohorts > first)).unique().to_list())

    if config.gref is None:
        if never.any():
            config.gref = 0
        elif config.cgroup == "notyet" and treated:
            config.gref = int(max(treated))
        else:
            raise ValueError(
                f"Could not identify '{config.cgroup}' control group. Never-treated units have a cohort of 0 or "
                "infinity, or one later than the last period."
            )
    else:
        gref = float(config.gref)
        if gref == 0 or gref > last:
            if not never.any():
                raise ValueError(
                    f"gref={config.gref} refers to the never-treated units. No unit in '{config.gname}' is never "
                    "treated."
                )
            config.gref = 0
        elif gref <= first:
            if never.any():
                unset = " or leave gref unset to use the never-treated units"
            elif config.cgroup == "notyet" and treated:
                unset = f" or leave gref unset to use the latest cohort, {max(treated):g}"
            else:
                unset = ""
            raise ValueError(
                f"gref={config.gref} is not a cohort first treated after the first period, {first}. Pass a later "
                f"cohort{unset}."
            )
        elif gref not in set(cohorts.drop_nulls().to_list()):
            raise ValueError(f"gref={config.gref} is not a cohort in '{config.gname}'.")

    # 0 marks the never-treated units. With them as the reference, the not-yet-treated design keeps every period.
    config._gref_min_flag = config.gref == 0
    return config


def prepare_etwfe_data(data, config):
    """Prepare data for ETWFE estimation.

    Creates the treatment indicator, demeans controls, applies reference-level
    filtering, and adds an indicator column for each treated cohort-time cell.

    Since nothing identifies their effects, treated cohorts with no untreated
    row in the estimation sample leave with a warning. Under the never-treated
    design the only untreated period of cohort :math:`g` is :math:`g - 1`.

    Parameters
    ----------
    data : pl.DataFrame
        Input panel data.
    config : EtwfeConfig
        ETWFE configuration.

    Returns
    -------
    pl.DataFrame
        Prepared data with ``_Dtreat``, ``_g``, ``_t``, demeaned columns, and one
        indicator column per cell from :func:`treatment_cells`. ``_g`` holds the
        cohort with 0 for every never-treated unit.
    """
    tname = config.tname
    gref = config.gref
    cohort = pl.col(config.gname).cast(pl.Float64)

    df = data.with_columns(
        [
            pl.when(_never_treated(cohort, data[tname].max())).then(0.0).otherwise(cohort).alias("_g"),
            pl.col(tname).cast(pl.Float64).alias("_t"),
        ]
    )
    # Since the regression parses the names it is given, it reads user columns through copies with plain names.
    copies = [
        (config.yname, _outcome_column(config)),
        (config.idname, "__etwfe_id"),
        (config.weightsname, _weights_column(config)),
    ]
    df = df.with_columns([pl.col(name).alias(col) for name, col in copies if name is not None and col != name])

    # Never-treated units stay untreated whichever cohort is the reference.
    in_treated_cohort = (pl.col("_g") != 0.0) & (pl.col("_g") != gref)
    if config.cgroup == "notyet":
        df = df.with_columns(
            pl.when((pl.col("_t") >= pl.col("_g")) & in_treated_cohort).then(1.0).otherwise(0.0).alias("_Dtreat")
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
            pl.when((pl.col("_t") != (pl.col("_g") - 1.0)) & in_treated_cohort)
            .then(1.0)
            .otherwise(0.0)
            .alias("_Dtreat")
        )

    # Without an untreated row, a cohort's cells are measured only against whichever of them the regression drops.
    untreated = (
        _estimation_sample(df, config)
        .filter(in_treated_cohort)
        .group_by("_g")
        .agg((pl.col("_Dtreat") == 0.0).any().alias("__etwfe_untreated"))
    )
    unidentified = sorted(untreated.filter(~pl.col("__etwfe_untreated"))["_g"].to_list())
    if unidentified:
        rows = df.filter(pl.col("_g").is_in(unidentified))
        dropped = f"{rows[config.idname].n_unique()} units" if config.idname is not None else f"{rows.height} rows"
        df = df.filter(~pl.col("_g").is_in(unidentified))
        if config.cgroup == "never":
            lacking = ", ".join(f"{g:g} lacks {g - 1:g}" for g in unidentified)
            reason = (
                f"with no row in period g - 1 ({lacking}). The never-treated design measures each cohort's effects "
                "against that period."
            )
        else:
            cohorts = ", ".join(f"{g:g}" for g in unidentified)
            reason = (
                f"with no row before their first treated period ({cohorts}). Without an untreated period their "
                "effects are not identified."
            )
        warnings.warn(f"Dropped {dropped} of cohorts {reason}", UserWarning, stacklevel=3)

    ctrls = []
    for i, name in enumerate(_get_control_vars(config)):
        ctrl = _formula_name(name, f"__etwfe_x{i}")
        if ctrl != name:
            df = df.with_columns(pl.col(name).alias(ctrl))
        x = pl.col(ctrl).cast(pl.Float64)
        df = df.with_columns((x - x.mean().over("_g")).alias(f"{ctrl}_dm"))
        ctrls.append(ctrl)

    xvar_dm_cols = []
    xvar_time_dummies = []
    if config.xvar:
        df, dm_cols = _build_xvar_dm_columns(df, config)
        # A moderator column that varies within cells only along the controls repeats their cell interactions.
        # Its year terms would also add year-by-cell means that absorb the cells' identifying variation.
        xvar_dm_cols = [col for col in dm_cols if not _spanned_by_controls(df, config, col, ctrls)]
        all_times = sorted(df[tname].drop_nulls().unique().to_list())
        for dm_col in xvar_dm_cols:
            for t in all_times:
                if t == config.tref:
                    continue
                name = f"_t{int(t)}_{dm_col}"
                df = df.with_columns((pl.when(pl.col(tname) == t).then(pl.col(dm_col)).otherwise(0.0)).alias(name))
                xvar_time_dummies.append(name)

    cells = treatment_cells(df, config)
    df = df.with_columns(
        [((pl.col("_g") == g) & (pl.col("_t") == t)).cast(pl.UInt8).alias(_cell_column(g, t)) for g, t in cells]
    )

    config._ctrls = ctrls
    config._xvar_dm_cols = xvar_dm_cols
    config._xvar_time_dummies = xvar_time_dummies

    return df


def treatment_cells(data, config):
    """List the cohort-time cells that hold treated observations.

    Parameters
    ----------
    data : pl.DataFrame
        Data with the ``_g``, ``_t``, and ``_Dtreat`` columns from
        :func:`prepare_etwfe_data`.
    config : EtwfeConfig
        ETWFE configuration.

    Returns
    -------
    list of tuple
        The (group, time) pairs, ordered by time with the reference period
        first and then by group.
    """
    treated = _estimation_sample(data, config).filter(pl.col("_Dtreat") == 1.0).select("_g", "_t").unique()
    pairs = [(float(g), float(t)) for g, t in treated.iter_rows()]
    return sorted(pairs, key=lambda cell: (cell[1] != config.tref, cell[1], cell[0]))


def build_etwfe_formula(config, data):
    """Build the regression formula for ETWFE.

    Parameters
    ----------
    config : EtwfeConfig
        ETWFE configuration with ``_ctrls``, ``_xvar_dm_cols``, and
        ``_xvar_time_dummies`` set by :func:`prepare_etwfe_data`.
    data : pl.DataFrame
        Data prepared by :func:`prepare_etwfe_data`.

    Returns
    -------
    str
        Complete formula string.
    """
    cells = treatment_cells(data, config)
    if not cells:
        raise ValueError("No treated cohort-time cells found. Check gname, gref, and cgroup.")

    gcat = "__etwfe_gcat"
    tcat = "__etwfe_tcat"
    sample = _estimation_sample(data, config)

    # Terms enter the design by degree and then in written order. With the controls written first, a
    # treatment cell that they span is the column the regression drops.
    parts = []
    for ctrl in config._ctrls:
        parts.extend([ctrl, f"C({gcat}):{ctrl}", f"C({tcat}):{ctrl}"])
    parts.extend(config._xvar_time_dummies)
    parts.append(_cell_terms(cells))
    # A variable that is constant within a cell makes its interaction zero or a copy of the cell indicator.
    for var in [f"{ctrl}_dm" for ctrl in config._ctrls] + list(config._xvar_dm_cols):
        varying = _cells_where_varying(sample, cells, var)
        if varying:
            parts.append(f"{_cell_terms(varying)}:{var}")

    rhs = " + ".join(parts)
    yname = _outcome_column(config)

    if config.fe != "none":
        fe_var = "__etwfe_id" if config.idname else "_g"
        formula = f"{yname} ~ {rhs} | {fe_var} + _t"
    else:
        parts.extend([f"C({gcat})", f"C({tcat})"])
        formula = f"{yname} ~ {' + '.join(parts)}"

    return formula


def run_etwfe_regression(formula, data, config, vcov=None, backend=None):
    """Run the ETWFE regression.

    Fits a linear, Poisson, or binary response regression based on
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
        The fitted regression.

        - **model**: fitted regression model
        - **formula**: the formula that was fitted
        - **fit_data**: the rows that entered the regression
    """
    df_clean = _estimation_sample(data, config)

    pdf = _to_pandas_with_categoricals(df_clean, config)

    vcov_spec = vcov if vcov else "hetero"
    if isinstance(vcov_spec, dict):
        vcov_spec = dict(vcov_spec)
        for i, (key, name) in enumerate(list(vcov_spec.items())):
            # The regression parses cluster names too. One it cannot read goes through a copy with a plain name.
            if isinstance(name, str) and name in pdf.columns and _formula_name(name, "") != name:
                pdf[f"__etwfe_cluster{i}"] = pdf[name]
                vcov_spec[key] = f"__etwfe_cluster{i}"
    family = config.family

    fit_kwargs = {"fml": formula, "data": pdf, "vcov": vcov_spec}
    if config.weightsname:
        fit_kwargs["weights"] = _weights_column(config)
    if backend is not None:
        fit_kwargs["demeaner_backend"] = backend

    from pyfixest.estimation.estimation import feglm as _feglm
    from pyfixest.estimation.estimation import feols as _feols
    from pyfixest.estimation.estimation import fepois as _fepois

    if family is None or family == "gaussian":
        model = _feols(**fit_kwargs)
    elif family == "poisson":
        model = _fepois(**fit_kwargs)
    elif family in ("logit", "probit"):
        try:
            model = _feglm(**fit_kwargs, family=family)
        except np.linalg.LinAlgError as exc:
            found = _collinear_controls(df_clean, config)
            if found:
                names = ", ".join(name for name, _ in found)
                remedy = " ".join(reason for _, reason in found) + f" Drop {names} from xformla and refit."
            else:
                without_xvar = f" or without xvar='{config.xvar}'" if config.xvar else ""
                remedy = (
                    f"No control alone explains it. Refit with fewer controls{without_xvar} to find the collinear "
                    "terms."
                )
            raise ValueError(
                f"The {family} regression has collinear columns that the binary families cannot drop. {remedy}"
            ) from exc
    else:
        raise ValueError(f"Unsupported family: {family}")

    # The regression also drops units observed only once. The aggregates must average over the rows it used.
    dropped = np.asarray(model._na_index, dtype=int)
    if dropped.size:
        keep = np.ones(df_clean.height, dtype=bool)
        keep[dropped] = False
        df_clean = df_clean.filter(pl.Series(keep))

    return {
        "model": model,
        "formula": formula,
        "fit_data": df_clean,
    }


def compute_emfx(fit_data, config, coef_names, beta, vcov_matrix, agg_type="simple", post_only=True, window=None):
    """Compute marginal effects from the fitted ETWFE model.

    For linear models, uses direct coefficient extraction. For nonlinear
    models, constructs counterfactual predictions (Dtreat=1 vs Dtreat=0)
    and applies the inverse link function. Rows in a cell the regression
    dropped as collinear get a NaN effect. Every aggregate that includes them
    is NaN too.

    Parameters
    ----------
    fit_data : pl.DataFrame
        Data used for fitting.
    config : EtwfeConfig
        ETWFE configuration.
    coef_names : list of str
        Names of all regression coefficients.
    beta : ndarray
        Estimates of all regression coefficients, ordered as ``coef_names``.
    vcov_matrix : ndarray
        Variance-covariance matrix of all regression coefficients.
    agg_type : str
        Aggregation type: ``"simple"``, ``"group"``, ``"calendar"``, ``"event"``.
    post_only : bool
        For the event type, whether to keep only event times zero and later.
    window : tuple or None
        Event window for filtering.

    Returns
    -------
    dict
        The aggregated effects.

        - **overall_att**: summary effect for the aggregation type
        - **overall_se**: standard error of the summary effect
        - **event_times**: event times, groups, or calendar times, or None for the simple type
        - **att_by_event**: effect at each of those values
        - **se_by_event**: standard error of each of those effects
        - **dropped_cells**: aggregated (group, time) cells that the regression dropped
    """
    post = (pl.col("_t") >= pl.col("_g")) & (pl.col("_Dtreat") == 1.0)
    df = fit_data.filter((pl.col("_g") != config.gref) & (pl.col("_g") != 0.0))
    # Only the never-treated design estimates pre-treatment cells. The not-yet-treated design has no placebos.
    if agg_type == "event" and not post_only and config.cgroup == "never":
        df = df.filter(post | (pl.col("_t") < pl.col("_g")))
    else:
        df = df.filter(post)

    if agg_type == "event":
        df = df.with_columns((pl.col("_t") - pl.col("_g")).cast(pl.Int64).alias("event"))
        if window is not None:
            df = df.filter((pl.col("event") >= window[0]) & (pl.col("event") <= window[1]))

    cells = treatment_cells(fit_data, config)
    coef_pos = {str(name): i for i, name in enumerate(coef_names)}
    row_cell = _match_rows_to_cells(df, cells)
    in_cell = row_cell >= 0
    main_pos = np.array([coef_pos.get(f"_Dtreat:{_cell_column(g, t)}", -1) for g, t in cells], dtype=int)
    row_main = np.where(in_cell, main_pos[row_cell], -1)
    dropped = in_cell & (row_main < 0)

    beta = np.asarray(beta, dtype=float)
    vcov_matrix = np.asarray(vcov_matrix, dtype=float)

    if config.family is None or config.family == "gaussian":
        slopes, jacobians = _compute_linear_slopes(df, config, cells, coef_pos, beta, row_cell, row_main)
    else:
        slopes, jacobians = _compute_nonlinear_slopes(df, config, coef_names, beta)
    slopes[dropped] = np.nan

    dropped_cells = [cells[k] for k in np.unique(row_cell[dropped])]
    weights = np.ones(len(df), dtype=float)

    if agg_type == "simple":
        att, se = _weighted_agg(slopes, jacobians, weights, vcov_matrix)
        return {
            "overall_att": att,
            "overall_se": se,
            "event_times": None,
            "att_by_event": None,
            "se_by_event": None,
            "dropped_cells": dropped_cells,
        }

    level_col = {"event": "event", "group": "_g", "calendar": "_t"}[agg_type]
    level_vals = df[level_col].to_numpy()
    unique_vals = np.sort(np.unique(level_vals))

    att_list, se_list, grad_list = [], [], []
    for val in unique_vals:
        mask = level_vals == val
        if not in_cell[mask].any():
            # The never-treated design's reference period, normalized to zero.
            att_list.append(0.0)
            se_list.append(np.nan)
            grad_list.append(np.zeros(len(beta)))
            continue
        att_v, se_v = _weighted_agg(slopes[mask], jacobians[mask], weights[mask], vcov_matrix)
        att_list.append(att_v)
        se_list.append(se_v)
        grad_list.append(jacobians[mask].mean(axis=0))

    att_arr = np.array(att_list)
    se_arr = np.array(se_list)
    level_weights = _summary_weights(df, config, agg_type, unique_vals)
    overall_att, overall_se = _summary_effect(att_arr, np.array(grad_list), level_weights, vcov_matrix)

    return {
        "overall_att": overall_att,
        "overall_se": overall_se,
        "event_times": np.array(unique_vals, dtype=float),
        "att_by_event": att_arr,
        "se_by_event": se_arr,
        "dropped_cells": dropped_cells,
    }


def _summary_weights(df, config, agg_type, levels):
    """Weight the aggregation levels for the one-number summary of each type."""
    if agg_type == "event":
        return (levels >= 0).astype(float)
    if agg_type == "calendar":
        return np.ones(len(levels))
    if config.idname is not None and config.idname in df.columns:
        sizes = df.group_by("_g").agg(pl.col(config.idname).n_unique().alias("size"))
    else:
        sizes = df.group_by("_g").agg((pl.len() / pl.col("_t").n_unique()).alias("size"))
    size_map = dict(sizes.iter_rows())
    return np.array([float(size_map[g]) for g in levels])


def _summary_effect(att, grads, level_weights, vcov_matrix):
    """Average level effects with fixed weights and a delta-method standard error."""
    keep = level_weights > 0
    if not keep.any():
        return np.nan, np.nan
    w = level_weights[keep] / level_weights[keep].sum()
    est = float(np.sum(w * att[keep]))
    if np.isnan(est):
        return np.nan, np.nan
    gbar = w @ grads[keep]
    se = float(np.sqrt(max(gbar @ vcov_matrix @ gbar, 0.0)))
    return est, se


def _get_control_vars(config):
    """List the control columns that xformla names."""
    if not config.xformla or config.xformla == "~1":
        return []
    parsed = parse_formula(config.xformla)
    if parsed["outcome"]:
        raise ValueError(
            f"xformla='{config.xformla}' has a left-hand side. It lists only the controls, as in '~ x1 + x2'. "
            "The outcome goes in yname."
        )
    return parsed["predictors"]


def _never_treated(cohort, last_period):
    """Flag the cohort codes that mark never-treated units."""
    return (cohort == 0.0) | (cohort > last_period)


def _cluster_columns(vcov, columns):
    """List the data columns that a dict variance specification clusters by."""
    if not isinstance(vcov, dict):
        return []
    names = []
    for name in vcov.values():
        if isinstance(name, str):
            # Two-way clustering joins its variables with a plus sign.
            parts = [name] if name in columns else [part.strip() for part in name.split("+")]
            names.extend(part for part in parts if part in columns)
    return names


def _is_missing(data, col):
    """Flag the rows where a column is null or NaN."""
    missing = pl.col(col).is_null()
    if data.schema[col].is_float():
        missing = missing | pl.col(col).is_nan()
    return missing


def _formula_name(name, internal):
    """Return the name itself when the formula can read it and the internal name otherwise."""
    readable = name.isascii() and name.isidentifier() and not keyword.iskeyword(name)
    return name if readable else internal


def _outcome_column(config):
    """Name the column the formula reads the outcome from."""
    return _formula_name(config.yname, "__etwfe_y")


def _weights_column(config):
    """Name the column the regression reads the weights from."""
    return None if config.weightsname is None else _formula_name(config.weightsname, "__etwfe_w")


def _estimation_sample(data, config):
    """Keep the rows that can enter the regression."""
    return data.filter(pl.col("_Dtreat").is_not_null() & ~_is_missing(data, config.yname))


def _cell_column(g, t):
    """Name the indicator column of the (g, t) cell."""
    return f"__etwfe_cell_{int(g)}_{int(t)}".replace("-", "m")


def _format_cells(cells):
    """Write (group, time) cells for a message."""
    return ", ".join(f"({g:g}, {t:g})" for g, t in cells)


def _match_rows_to_cells(df, cells):
    """Return each row's position in ``cells`` or -1 for a row outside every treated cell."""
    lookup = pl.DataFrame(
        {
            "_g": [g for g, _ in cells],
            "_t": [t for _, t in cells],
            "__etwfe_pos": list(range(len(cells))),
        },
        schema={"_g": pl.Float64, "_t": pl.Float64, "__etwfe_pos": pl.Int64},
    )
    # A join may reorder the rows. The row index puts them back in the input order.
    matched = (
        df.select("_g", "_t")
        .with_row_index("__etwfe_row")
        .join(lookup, on=["_g", "_t"], how="left")
        .sort("__etwfe_row")
    )
    return matched["__etwfe_pos"].fill_null(-1).to_numpy()


def _weighted_agg(slopes, jacobians, weights, vcov_matrix):
    """Compute weighted average and delta-method SE for a subset."""
    w_sum = weights.sum()
    if w_sum == 0:
        return 0.0, np.nan
    est = float(np.average(slopes, weights=weights))
    if vcov_matrix is None or np.isnan(est):
        return est, np.nan
    WJ = jacobians * weights[:, None]
    gbar = WJ.sum(axis=0, keepdims=True) / w_sum
    se = float(np.sqrt(np.clip(gbar @ vcov_matrix @ gbar.T, 0, None)[0, 0]))
    return est, se


def _compute_linear_slopes(df, config, cells, coef_pos, beta, row_cell, row_main):
    """Compute slopes via direct coefficient lookup (linear models only)."""
    n = len(df)
    slopes = np.zeros(n)
    jacobians = np.zeros((n, len(beta)))

    rows = np.flatnonzero(row_main >= 0)
    slopes[rows] = beta[row_main[rows]]
    jacobians[rows, row_main[rows]] = 1.0

    for var in [f"{ctrl}_dm" for ctrl in config._ctrls] + list(config._xvar_dm_cols):
        var_pos = np.array([coef_pos.get(f"_Dtreat:{_cell_column(g, t)}:{var}", -1) for g, t in cells], dtype=int)
        row_var = np.where(row_cell >= 0, var_pos[row_cell], -1)
        rows = np.flatnonzero(row_var >= 0)
        values = df[var].cast(pl.Float64).to_numpy()[rows]
        slopes[rows] += beta[row_var[rows]] * values
        jacobians[rows, row_var[rows]] = values

    return slopes, jacobians


def _compute_nonlinear_slopes(df, config, coef_names, beta):
    """Compute slopes via counterfactual predictions (nonlinear models)."""
    if df.height == 0:
        return np.zeros(0), np.zeros((0, len(beta)))
    pdf = _to_pandas_with_categoricals(df, config)

    rhs_formula = config._formula.split("|")[0].strip()

    pdf1 = pdf.copy()
    pdf0 = pdf.copy()
    pdf1["_Dtreat"] = 1.0
    pdf0["_Dtreat"] = 0.0

    names = [str(c) for c in coef_names]
    _, X1_full = formulaic.model_matrix(rhs_formula, pdf1)
    _, X0_full = formulaic.model_matrix(rhs_formula, pdf0)
    X1 = X1_full.reindex(columns=names, fill_value=0.0).to_numpy()
    X0 = X0_full.reindex(columns=names, fill_value=0.0).to_numpy()

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


def _spanned_by_controls(df, config, col, ctrls):
    """Check whether a moderator column varies within cohort-time cells only along the controls."""
    cell = ["_g", "_t"]
    within = (
        _estimation_sample(df, config)
        .select(
            [
                (pl.col(c).cast(pl.Float64) - pl.col(c).cast(pl.Float64).mean().over(cell)).alias(f"__etwfe_w{i}")
                for i, c in enumerate([col, *ctrls])
            ]
        )
        .drop_nulls()
        .to_numpy()
    )
    x, controls = within[:, 0], within[:, 1:]
    resid = x - controls @ np.linalg.lstsq(controls, x, rcond=None)[0] if controls.shape[1] else x
    return bool(np.linalg.norm(resid) <= 1e-8 * np.linalg.norm(x))


def _cells_where_varying(sample, cells, col):
    """Keep the cells whose treated rows take more than one value of a column."""
    varying = (
        sample.filter(pl.col("_Dtreat") == 1.0)
        .group_by("_g", "_t")
        .agg(pl.col(col).n_unique().alias("__etwfe_n"))
        .filter(pl.col("__etwfe_n") > 1)
    )
    keep = {(float(g), float(t)) for g, t, _ in varying.iter_rows()}
    return [cell for cell in cells if cell in keep]


def _collinear_controls(sample, config):
    """Pair each control that repeats a cohort or period dummy or the controls before it with the reason."""
    collinear = []
    basis = np.ones((sample.height, 1))
    for name, ctrl in zip(_get_control_vars(config), config._ctrls, strict=True):
        constant = []
        for col, kind in (("_g", "cohort"), ("_t", "period")):
            levels = sample.group_by(col).agg(pl.col(ctrl).n_unique() == 1).filter(pl.col(ctrl))[col].sort()
            if levels.len():
                constant.append((kind, levels.to_list()))
        x = sample[ctrl].cast(pl.Float64).to_numpy()[:, None]
        if constant:
            where = " and ".join(
                f"{kind}{'s' if len(levels) > 1 else ''} {', '.join(f'{v:g}' for v in levels)}"
                for kind, levels in constant
            )
            if len(constant) == 1 and len(constant[0][1]) == 1:
                kind = constant[0][0]
                consequence = f"its interaction with that {kind} repeats the {kind} dummy"
            else:
                consequence = "its interactions with them repeat their dummies"
            collinear.append((name, f"Since {name} is constant within {where}, {consequence}."))
        elif np.linalg.matrix_rank(np.hstack([basis, x])) == basis.shape[1]:
            collinear.append((name, f"{name} is a linear combination of the controls before it."))
        else:
            basis = np.hstack([basis, x])
    return collinear


def _cell_terms(cells):
    """Write the treatment interactions of a list of cells as one formula term."""
    return "_Dtreat:(" + " + ".join(_cell_column(g, t) for g, t in cells) + ")"


def _build_xvar_dm_columns(df, config):
    """Demean the moderator within cohort-time cells for heterogeneous treatment effects.

    The new columns end in ``_xdm``. A moderator that is also a control then
    leaves the control's cohort-demeaned ``_dm`` column intact.
    """
    cell = ["_g", "_t"]
    x_col = config.xvar
    dm_cols = []

    # Demeaning within cells over every row makes the cell coefficients the effects at each cell's mean,
    # whatever the coding of the moderator.
    if df[x_col].dtype.is_numeric() or df[x_col].dtype == pl.Boolean:
        dm_name = _formula_name(f"{x_col}_xdm", "__etwfe_m0_xdm")
        x = pl.col(x_col).cast(pl.Float64)
        df = df.with_columns((x - x.mean().over(cell)).alias(dm_name))
        dm_cols.append(dm_name)
        return df, dm_cols

    cats = df[x_col].unique().sort().to_list()
    if len(cats) > 1:
        cats = cats[1:]

    # Since each category keeps its own column and year terms, the fit does not depend on which label sorts first.
    for i, cat in enumerate(cats):
        dummy = (pl.col(x_col) == cat).cast(pl.Float64)
        dm_name = _formula_name(f"{x_col}_{cat}_xdm", f"__etwfe_m{i}_xdm")
        df = df.with_columns((dummy - dummy.mean().over(cell)).alias(dm_name))
        dm_cols.append(dm_name)

    return df, dm_cols


def _to_pandas_with_categoricals(df, config):
    """Convert to pandas with ordered categoricals for the regression backend."""
    gref = int(config.gref)
    tref = int(config.tref)

    g_vals = sorted(df["_g"].drop_nulls().cast(pl.Int64).unique().to_list())
    t_vals = sorted(df["_t"].drop_nulls().cast(pl.Int64).unique().to_list())

    if gref not in g_vals:
        g_vals = [gref, *g_vals]
    if tref not in t_vals:
        t_vals = [tref, *t_vals]

    g_cats = [gref] + [g for g in g_vals if g != gref]
    t_cats = [tref] + [t for t in t_vals if t != tref]

    pdf = df.to_pandas()
    pdf["_Dtreat"] = pdf["_Dtreat"].astype(float)

    pdf["__etwfe_gcat"] = pd.Categorical(pdf["_g"].astype("Int64"), categories=g_cats, ordered=True)
    pdf["__etwfe_tcat"] = pd.Categorical(pdf["_t"].astype("Int64"), categories=t_cats, ordered=True)
    return pdf
