"""Distributed preprocessing for DIDCont partitions."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl

from moderndid.core.preprocess.constants import WEIGHTS_COLUMN
from moderndid.core.preprocess.utils import choose_knots_quantile
from moderndid.didcont.estimation.container import PTEParams

NEVER_TREATED = float("inf")


def partition_infer_group(df, idname, tname, dname):
    """Infer treatment group per unit within a single partition.

    For each unit, find the first period where treatment > 0.
    Returns a Polars DataFrame with columns ``[idname, '_G']`` where
    ``_G`` is the inferred first-treatment period (or ``inf`` if never
    treated within this partition).

    Parameters
    ----------
    df : pl.DataFrame
        A single partition.
    idname : str
        Unit identifier column.
    tname : str
        Time period column.
    dname : str
        Treatment / dose column.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``[idname, '_G']``.
    """
    if len(df) == 0:
        return pl.DataFrame({idname: pl.Series([], dtype=pl.Int64), "_G": pl.Series([], dtype=pl.Float64)})

    df_sorted = df.sort([idname, tname])

    treated = df_sorted.filter(pl.col(dname) > 0)
    first_treat = treated.group_by(idname).agg(pl.col(tname).min().alias("_G"))

    all_ids = df_sorted.select(idname).unique()
    result = all_ids.join(first_treat, on=idname, how="left")
    return result.with_columns(pl.col("_G").fill_null(NEVER_TREATED))


def partition_preprocess(df, col_config, max_time, gname_provided):
    """Run all partition-level preprocessing steps on a single partition.

    Performs column selection, treatment encoding, missing data handling,
    early-treatment filtering, dose validation, and weight normalization.

    Parameters
    ----------
    df : pl.DataFrame
        A single partition.
    col_config : dict
        Column name configuration with keys: ``yname``, ``tname``,
        ``idname``, ``gname``, ``dname``, ``weightsname``,
        ``anticipation``, ``required_pre_periods``.
    max_time : float
        Maximum time period in the dataset (used for treatment encoding).
    gname_provided : bool
        Whether the user provided gname (if False, ``_G`` was inferred
        and is already present).

    Returns
    -------
    pl.DataFrame
        Preprocessed partition.
    """
    if len(df) == 0:
        return df

    yname = col_config["yname"]
    tname = col_config["tname"]
    idname = col_config["idname"]
    gname = col_config["gname"]
    dname = col_config["dname"]
    weightsname = col_config.get("weightsname")

    cols_to_keep = [yname, tname, idname, gname, dname]
    if weightsname and weightsname in df.columns:
        cols_to_keep.append(weightsname)
    cols_to_keep = list(dict.fromkeys(c for c in cols_to_keep if c in df.columns))
    df = df.select(cols_to_keep)

    for col in [tname, gname, dname]:
        if col in df.columns and df[col].dtype not in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
            df = df.with_columns(pl.col(col).cast(pl.Float64))

    df = df.drop_nulls()
    if len(df) == 0:
        return df

    df = df.with_columns(pl.col(gname).cast(pl.Float64))
    df = df.with_columns(pl.when(pl.col(gname) == 0).then(pl.lit(NEVER_TREATED)).otherwise(pl.col(gname)).alias(gname))
    df = df.with_columns(
        pl.when(pl.col(gname) > max_time).then(pl.lit(NEVER_TREATED)).otherwise(pl.col(gname)).alias(gname)
    )

    df = df.with_columns(pl.when(pl.col(gname) == NEVER_TREATED).then(pl.lit(0)).otherwise(pl.col(dname)).alias(dname))

    if weightsname and weightsname in df.columns:
        df = df.with_columns(pl.col(weightsname).cast(pl.Float64).alias(WEIGHTS_COLUMN))
    else:
        df = df.with_columns(pl.lit(1.0).alias(WEIGHTS_COLUMN))

    return df


def filter_early_treated(df, gname, tname, anticipation, required_pre_periods):
    """Filter out early-treated groups from collected data.

    Removes groups treated before ``min_valid_group =
    required_pre_periods + anticipation + first_period``.

    Parameters
    ----------
    df : pl.DataFrame
        Collected preprocessed data.
    gname : str
        Group column name.
    tname : str
        Time column name.
    anticipation : int
        Anticipation periods.
    required_pre_periods : int
        Required pre-treatment periods.

    Returns
    -------
    pl.DataFrame
        Filtered data.
    """
    tlist = sorted(df[tname].unique().to_list())
    first_period = min(tlist)

    glist = sorted(g for g in df[gname].unique().to_list() if np.isfinite(g))
    if not glist:
        return df

    min_valid_group = required_pre_periods + anticipation + first_period
    groups_to_drop = [g for g in glist if g < min_valid_group]

    if groups_to_drop:
        warnings.warn(
            f"Dropped {len(groups_to_drop)} groups treated before period {min_valid_group} "
            f"(required_pre_periods={required_pre_periods}, anticipation={anticipation})"
        )
        df = df.filter(~pl.col(gname).is_in(groups_to_drop))

    return df


def recode_time_periods(df, tname):
    """Recode time periods to consecutive integers starting at 1.

    Parameters
    ----------
    df : pl.DataFrame
        Data with original time periods.
    tname : str
        Time column name.

    Returns
    -------
    tuple[pl.DataFrame, dict]
        Recoded data and the time_map ``{original: recoded}``.
    """
    original_periods = sorted(df[tname].unique().to_list())
    time_map = {t: i + 1 for i, t in enumerate(original_periods)}
    df = df.with_columns(pl.col(tname).replace(time_map).alias(tname))
    return df, time_map


def balance_panel(df, idname, tname):
    """Balance panel by keeping only units observed in all periods.

    Parameters
    ----------
    df : pl.DataFrame
        Panel data.
    idname : str
        Unit identifier column.
    tname : str
        Time period column.

    Returns
    -------
    pl.DataFrame
        Balanced panel data.
    """
    tlist = sorted(df[tname].unique().to_list())
    n_periods = len(tlist)

    unit_counts = df.group_by(idname).len()
    complete_units = unit_counts.filter(pl.col("len") == n_periods)[idname].to_list()

    n_old = df[idname].n_unique()
    df = df.filter(pl.col(idname).is_in(complete_units))
    n_new = df[idname].n_unique()

    if n_new < n_old:
        warnings.warn(f"Dropped {n_old - n_new} units while converting to balanced panel")

    if len(df) == 0:
        raise ValueError(
            "All observations dropped while converting to balanced panel. "
            "Consider setting panel=False and/or revisiting 'idname'"
        )

    return df


def normalize_weights(df, weightsname):
    """Normalize weights to have mean 1.

    Parameters
    ----------
    df : pl.DataFrame
        Data with weight column.
    weightsname : str or None
        Original weights column name.

    Returns
    -------
    pl.DataFrame
        Data with normalized ``WEIGHTS_COLUMN``.
    """
    if WEIGHTS_COLUMN not in df.columns:
        df = df.with_columns(pl.lit(1.0).alias(WEIGHTS_COLUMN))

    w_mean = df[WEIGHTS_COLUMN].mean()
    if w_mean and w_mean > 0:
        df = df.with_columns((pl.col(WEIGHTS_COLUMN) / w_mean).alias(WEIGHTS_COLUMN))

    return df


def compute_pte_params(
    collected_data,
    *,
    yname,
    tname,
    idname,
    gname,
    dname,
    time_map,
    weightsname,
    xformla,
    target_parameter,
    aggregation,
    treatment_type,
    dose_est_method,
    control_group,
    anticipation,
    base_period,
    boot_type,
    alp,
    cband,
    biters,
    degree,
    num_knots,
    dvals,
    required_pre_periods,
    gt_type,
):
    """Build PTEParams from collected preprocessed data and metadata.

    Replicates the logic from ``_build_pte_params()`` in
    ``process_panel.py`` but works directly from the collected
    preprocessed DataFrame rather than a ``ContDIDData`` object.

    Parameters
    ----------
    collected_data : pl.DataFrame
        Preprocessed, collected data with recoded time periods.
    yname, tname, idname, gname, dname : str
        Column names.
    time_map : dict
        Mapping from original to recoded time periods.
    weightsname : str or None
        Weights column name.
    xformla : str
        Covariate formula.
    target_parameter, aggregation, treatment_type, dose_est_method : str
        Estimation settings.
    control_group, base_period, boot_type : str
        Strategy settings (already string values, not enums).
    alp : float
        Significance level.
    cband : bool
        Uniform confidence bands.
    biters : int
        Bootstrap iterations.
    degree, num_knots : int
        Spline parameters.
    dvals : np.ndarray or None
        Dose values.
    required_pre_periods : int
        Required pre-treatment periods.
    gt_type : str
        Group-time effect type (``"att"`` or ``"dose"``).

    Returns
    -------
    PTEParams
        Parameters for panel treatment effects estimation.
    """
    data = collected_data.clone()

    data = data.with_columns(
        [
            pl.col(gname).alias("G"),
            pl.col(idname).alias("id"),
            pl.col(tname).alias("period"),
            pl.col(yname).alias("Y"),
        ]
    )
    data = data.with_columns(pl.col(dname).alias("D")) if dname else data.with_columns(pl.lit(0).alias("D"))

    if weightsname and WEIGHTS_COLUMN in data.columns:
        data = data.with_columns(pl.col(WEIGHTS_COLUMN).alias(".w"))
    else:
        data = data.with_columns(pl.lit(1.0).alias(".w"))

    time_periods = np.array(sorted(data["period"].unique().to_list()))
    groups = np.array(sorted(g for g in data["G"].unique().to_list() if np.isfinite(g)))

    if base_period == "universal":
        t_list = np.sort(time_periods)
        min_t_for_g = t_list[1] if len(t_list) > 1 else np.inf
    else:
        t_list = np.sort(time_periods)[required_pre_periods:]
        min_t_for_g = np.min(t_list) if len(t_list) > 0 else np.inf

    g_list = groups[np.isin(groups, t_list)]
    g_list = g_list[g_list >= (min_t_for_g + anticipation)]

    groups_to_drop = np.arange(1, required_pre_periods + anticipation + 1)
    data = data.filter(~pl.col("G").is_in(groups_to_drop))

    is_treated = data["G"].is_finite()
    is_post_treatment = data["period"] >= data["G"]
    mask = is_treated & is_post_treatment
    dose_values = data.filter(mask)["D"].to_numpy()
    positive_doses = dose_values[dose_values > 0]

    knots = choose_knots_quantile(positive_doses, num_knots)

    if dvals is None:
        dvals = np.linspace(positive_doses.min(), positive_doses.max(), 50) if len(positive_doses) > 0 else np.array([])

    return PTEParams(
        yname=yname,
        gname=gname,
        tname=tname,
        idname=idname,
        data=data,
        g_list=g_list,
        t_list=t_list,
        cband=cband,
        alp=alp,
        boot_type=boot_type,
        gt_type=gt_type,
        ret_quantile=0.5,
        biters=biters,
        anticipation=anticipation,
        base_period=base_period,
        weightsname=weightsname,
        control_group=control_group,
        dname=dname,
        degree=degree,
        num_knots=num_knots,
        knots=knots,
        dvals=dvals,
        target_parameter=target_parameter,
        aggregation=aggregation,
        treatment_type=treatment_type,
        xformula=xformla,
        dose_est_method=dose_est_method,
    )
