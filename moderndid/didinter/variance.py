"""Variance estimation for DIDInter."""

import numpy as np
import polars as pl
from scipy import stats

from .numba import compute_cluster_sums


def build_treatment_paths(df, horizon, config):
    """Build hierarchical cohort identifiers.

    Parameters
    ----------
    df : pl.DataFrame
        Data with F_g and d_sq columns.
    horizon : int
        Current horizon being computed.
    config : DIDInterConfig
        Configuration object with column names.

    Returns
    -------
    pl.DataFrame
        DataFrame with path_0, path_1, ..., path_h columns added.
    """
    gname = config.gname
    tname = config.tname
    dname = config.dname
    h = abs(horizon)

    df = df.with_columns(
        pl.when(pl.col(tname) == pl.col("F_g") + h - 1)
        .then(pl.col(dname))
        .otherwise(pl.lit(None))
        .alias("_treat_at_horizon")
    )

    df = df.with_columns(pl.col("_treat_at_horizon").mean().over(gname).alias(f"treat_h{h}"))

    if h == 1:
        df = df.with_columns(pl.col("d_sq").alias("treat_h0"))
        df = df.with_columns(
            pl.concat_str(
                [
                    pl.col("treat_h0").cast(pl.Utf8),
                    pl.lit("|"),
                    pl.col("F_g").cast(pl.Utf8),
                ]
            )
            .cast(pl.Categorical)
            .to_physical()
            .cast(pl.Int64)
            .add(1)
            .alias("path_0")
        )

    if h > 1 and f"treat_h{h - 1}" in df.columns:
        df = df.with_columns(
            pl.when(pl.col(f"treat_h{h}").is_null())
            .then(pl.col(f"treat_h{h - 1}"))
            .otherwise(pl.col(f"treat_h{h}"))
            .alias(f"treat_h{h}")
        )

    prev_path = f"path_{h - 1}" if h > 1 else "path_0"
    if prev_path in df.columns:
        df = df.with_columns(
            pl.concat_str(
                [
                    pl.col(prev_path).cast(pl.Utf8),
                    pl.lit("|"),
                    pl.col(f"treat_h{h}").cast(pl.Utf8),
                ]
            )
            .cast(pl.Categorical)
            .to_physical()
            .cast(pl.Int64)
            .add(1)
            .alias(f"path_{h}")
        )

    if h == 1 and "path_0" in df.columns:
        df = df.with_columns(pl.col(gname).n_unique().over("path_0").alias("n_groups_path_0"))
        df = df.with_columns((pl.col("n_groups_path_0") > 1).cast(pl.Int64).alias("valid_cohort_0"))

    if f"path_{h}" in df.columns:
        df = df.with_columns(pl.col(gname).n_unique().over(f"path_{h}").alias(f"n_groups_path_{h}"))
        df = df.with_columns((pl.col(f"n_groups_path_{h}") > 1).cast(pl.Int64).alias(f"valid_cohort_{h}"))

    df = df.drop("_treat_at_horizon")

    return df


def compute_cohort_dof(df, horizon, config, cluster_col=None):
    """Compute DOF-adjusted cohort means using hierarchical path groupings.

    Parameters
    ----------
    df : pl.DataFrame
        Data with path columns and switcher flags.
    horizon : int
        Current horizon.
    config : DIDInterConfig
        Configuration object.
    cluster_col : str or None
        Column name for cluster-robust DOF counting.

    Returns
    -------
    pl.DataFrame
        DataFrame with dof_switcher_{h} and cohort_mean_{h} columns.
    """
    h = abs(horizon)
    trends = config.trends_nonparam or []
    switcher_flag = f"is_switcher_{h}"
    weighted_diff = f"weighted_diff_{h}"

    is_switcher = pl.col(switcher_flag) == 1

    def _path_group(path_col):
        return [path_col] + list(trends)

    path_configs = [("path_0", "p0"), ("path_1", "p1"), (f"path_{h}", "p2")]

    for path_col, tag in path_configs:
        if path_col not in df.columns:
            continue

        weight_sum = f"weight_sum_{h}_{tag}"
        diff_sum = f"diff_sum_{h}_{tag}"

        val_weight = pl.when(is_switcher).then(pl.col("weight_gt")).otherwise(None)
        val_diff = pl.when(is_switcher).then(pl.col(weighted_diff)).otherwise(None)

        df = df.with_columns(
            pl.when(is_switcher).then(val_weight.sum().over(_path_group(path_col))).otherwise(None).alias(weight_sum)
        )
        df = df.with_columns(
            pl.when(is_switcher).then(val_diff.sum().over(_path_group(path_col))).otherwise(None).alias(diff_sum)
        )

    if cluster_col is None:
        val_dof = pl.when(is_switcher).then(pl.col(switcher_flag)).otherwise(None)
        for path_col, tag in path_configs:
            if path_col not in df.columns:
                continue
            dof_col = f"dof_{h}_{tag}"
            df = df.with_columns(
                pl.when(is_switcher).then(val_dof.sum().over(_path_group(path_col))).otherwise(None).alias(dof_col)
            )
    else:
        cluster_flag = f"_cluster_flag_{h}"
        df = df.with_columns(pl.when(is_switcher).then(pl.col(cluster_col)).otherwise(None).alias(cluster_flag))
        for path_col, tag in path_configs:
            if path_col not in df.columns:
                continue
            dof_col = f"dof_{h}_{tag}"
            df = df.with_columns(
                pl.when(pl.col(cluster_flag).is_not_null())
                .then(pl.col(cluster_flag).n_unique().over(_path_group(path_col)))
                .otherwise(None)
                .alias(dof_col)
            )
        df = df.drop(cluster_flag)

    dof_p0 = pl.col(f"dof_{h}_p0") if f"dof_{h}_p0" in df.columns else pl.lit(None)
    dof_p1 = pl.col(f"dof_{h}_p1") if f"dof_{h}_p1" in df.columns else pl.lit(None)
    dof_p2 = pl.col(f"dof_{h}_p2") if f"dof_{h}_p2" in df.columns else pl.lit(None)

    df = df.with_columns(
        pl.when(dof_p2 >= 2)
        .then(dof_p2)
        .when((dof_p2 < 2) & (dof_p1 >= 2))
        .then(dof_p1)
        .otherwise(dof_p0)
        .alias(f"dof_switcher_{h}")
    )

    ws_p0 = pl.col(f"weight_sum_{h}_p0") if f"weight_sum_{h}_p0" in df.columns else pl.lit(1.0)
    ws_p1 = pl.col(f"weight_sum_{h}_p1") if f"weight_sum_{h}_p1" in df.columns else pl.lit(1.0)
    ws_p2 = pl.col(f"weight_sum_{h}_p2") if f"weight_sum_{h}_p2" in df.columns else pl.lit(1.0)
    ds_p0 = pl.col(f"diff_sum_{h}_p0") if f"diff_sum_{h}_p0" in df.columns else pl.lit(0.0)
    ds_p1 = pl.col(f"diff_sum_{h}_p1") if f"diff_sum_{h}_p1" in df.columns else pl.lit(0.0)
    ds_p2 = pl.col(f"diff_sum_{h}_p2") if f"diff_sum_{h}_p2" in df.columns else pl.lit(0.0)

    df = df.with_columns(
        pl.when(dof_p2 >= 2)
        .then(ds_p2 / ws_p2)
        .when((dof_p2 < 2) & (dof_p1 >= 2))
        .then(ds_p1 / ws_p1)
        .otherwise(ds_p0 / ws_p0)
        .alias(f"cohort_mean_{h}")
    )

    return df


def compute_dof_scaling(df, horizon, config):
    """Compute DOF scaling factor for variance correction.

    Parameters
    ----------
    df : pl.DataFrame
        Data with dof_switcher and dof_control columns.
    horizon : int
        Current horizon.
    config : DIDInterConfig
        Configuration object.

    Returns
    -------
    pl.DataFrame
        DataFrame with dof_scale_{h} column.
    """
    h = abs(horizon)
    tname = config.tname

    dof_col = f"dof_scale_{h}"
    dof_s = f"dof_switcher_{h}"
    dof_ns = f"dof_control_{h}"

    df = df.with_columns(pl.lit(1.0).alias(dof_col))

    time = pl.col(tname)
    fg = pl.col("F_g")

    at_target_time = (fg - 1 + h) == time
    before_switch = time < fg

    if dof_s in df.columns:
        s = pl.col(dof_s)
        df = df.with_columns(
            pl.when(at_target_time & (s.fill_null(9999) > 1))
            .then((s / (s - 1)).sqrt())
            .otherwise(pl.col(dof_col))
            .alias(dof_col)
        )

    if dof_ns in df.columns:
        ns = pl.col(dof_ns)
        df = df.with_columns(
            pl.when(before_switch & (ns.fill_null(9999) > 1))
            .then((ns / (ns - 1)).sqrt())
            .otherwise(pl.col(dof_col))
            .alias(dof_col)
        )

    return df


def compute_clustered_variance(influence_func, cluster_ids, n_groups):
    """Compute clustered standard error from influence function.

    Parameters
    ----------
    influence_func : ndarray
        Influence function values for each unit.
    cluster_ids : ndarray
        Cluster identifiers for each unit.
    n_groups : int
        Number of groups.

    Returns
    -------
    float
        Clustered standard error.
    """
    cluster_sums, unique_clusters = compute_cluster_sums(influence_func, cluster_ids)
    n_clusters = len(unique_clusters)

    if n_clusters <= 1:
        return np.sqrt(np.var(influence_func, ddof=1) / n_groups)

    cluster_var = np.var(cluster_sums, ddof=1)
    std_error = np.sqrt(cluster_var / n_groups)

    return std_error


def compute_joint_test(estimates, vcov):
    """Compute joint test that all estimates are zero.

    Parameters
    ----------
    estimates : ndarray
        Point estimates.
    vcov : ndarray
        Variance-covariance matrix.

    Returns
    -------
    dict or None
        Dictionary with chi2_stat, df, and p_value, or None if test fails.
    """
    if vcov is None:
        return None

    valid_mask = ~np.isnan(estimates)
    if np.sum(valid_mask) < 1:
        return None

    valid_estimates = estimates[valid_mask]
    valid_vcov = vcov[np.ix_(valid_mask, valid_mask)]

    try:
        chi2_stat = float(valid_estimates @ np.linalg.solve(valid_vcov, valid_estimates))
        df = len(valid_estimates)
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)
        return {"chi2_stat": chi2_stat, "df": df, "p_value": p_value}
    except np.linalg.LinAlgError:
        return None
