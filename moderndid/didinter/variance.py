"""Variance estimation."""

import numpy as np
import polars as pl
from scipy import stats

from .numba import compute_cluster_sums


def build_treatment_paths(df, horizon, config):
    r"""Build hierarchical cohort identifiers based on treatment trajectories.

    Constructs treatment path variables that track each group's treatment history
    from period :math:`F_g` (first switch) through :math:`F_g - 1 + \ell`. Groups
    are assigned to cohorts based on their baseline treatment :math:`D_{g,1}`,
    switch timing :math:`F_g`, and subsequent treatment values. This enables
    comparison of switchers only to non-switchers with the same baseline treatment,
    which is required for the parallel trends assumption in [1]_.

    Parameters
    ----------
    df : pl.DataFrame
        Data with F_g (first switch period) and d_sq (baseline treatment) columns.
    horizon : int
        Current horizon :math:`\ell` being computed.
    config : DIDInterConfig
        Configuration object with column names.

    Returns
    -------
    pl.DataFrame
        DataFrame with path_0, path_1, ..., path_h columns identifying treatment
        trajectories, and validity flags for cohorts with sufficient observations.

    References
    ----------

    .. [1] de Chaisemartin, C., & D'Haultfoeuille, X. (2024). Difference-in-
           Differences Estimators of Intertemporal Treatment Effects.
           *Review of Economics and Statistics*, 106(6), 1723-1736.
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
        df = df.with_columns(pl.struct(["treat_h0", "F_g"]).hash(seed=42).alias("path_0"))

    if h > 1 and f"treat_h{h - 1}" in df.columns:
        df = df.with_columns(
            pl.when(pl.col(f"treat_h{h}").is_null())
            .then(pl.col(f"treat_h{h - 1}"))
            .otherwise(pl.col(f"treat_h{h}"))
            .alias(f"treat_h{h}")
        )

    prev_path = f"path_{h - 1}" if h > 1 else "path_0"
    if prev_path in df.columns:
        df = df.with_columns(pl.struct([prev_path, f"treat_h{h}"]).hash(seed=42).alias(f"path_{h}"))

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
    dist_col = f"dist_to_switch_{h}"

    is_switcher = pl.col(switcher_flag) == 1

    base_group_vars = ["d_sq", "F_g", "d_fg", dist_col]
    group_vars = base_group_vars + list(trends)
    group_vars = [c for c in group_vars if c in df.columns]

    weight_sum_col = f"weight_sum_{h}_switcher"
    diff_sum_col = f"diff_sum_{h}_switcher"

    val_weight = pl.when(is_switcher).then(pl.col("weight_gt")).otherwise(None)
    val_diff = pl.when(is_switcher).then(pl.col(weighted_diff)).otherwise(None)

    df = df.with_columns(
        pl.when(is_switcher).then(val_weight.sum().over(group_vars)).otherwise(None).alias(weight_sum_col),
        pl.when(is_switcher).then(val_diff.sum().over(group_vars)).otherwise(None).alias(diff_sum_col),
    )

    dof_col = f"dof_switcher_{h}"
    if cluster_col is None:
        val_dof = pl.when(is_switcher).then(pl.col(switcher_flag)).otherwise(None)
        df = df.with_columns(pl.when(is_switcher).then(val_dof.sum().over(group_vars)).otherwise(None).alias(dof_col))
    else:
        cluster_flag = f"_cluster_flag_{h}"
        df = df.with_columns(pl.when(is_switcher).then(pl.col(cluster_col)).otherwise(None).alias(cluster_flag))
        df = df.with_columns(
            pl.when(pl.col(cluster_flag).is_not_null())
            .then(pl.col(cluster_flag).n_unique().over(group_vars))
            .otherwise(None)
            .alias(dof_col)
        )
        df = df.drop(cluster_flag)

    ws = pl.col(weight_sum_col).fill_null(1.0)
    ds = pl.col(diff_sum_col).fill_null(0.0)

    df = df.with_columns((ds / ws).alias(f"cohort_mean_{h}"))

    return df


def compute_control_dof(df, horizon, config, cluster_col=None):
    """Compute DOF for control units (non-switchers).

    Parameters
    ----------
    df : pl.DataFrame
        Data with never_change column.
    horizon : int
        Current horizon.
    config : DIDInterConfig
        Configuration object.
    cluster_col : str or None
        Column name for cluster-robust DOF counting.

    Returns
    -------
    pl.DataFrame
        DataFrame with dof_control_{h} and control_mean_{h} columns.
    """
    h = abs(horizon)
    tname = config.tname
    trends = config.trends_nonparam or []

    never_col = f"never_change_{h}"
    weighted_diff = f"weighted_diff_{h}"

    if never_col not in df.columns:
        return df

    is_control = pl.col(never_col) == 1.0
    group_vars = [tname, "d_sq", *list(trends)]

    weight_sum_col = f"control_weight_sum_{h}"
    diff_sum_col = f"control_diff_sum_{h}"
    dof_col = f"dof_control_{h}"
    mean_col = f"control_mean_{h}"

    val_weight = pl.when(is_control).then(pl.col("weight_gt")).otherwise(None)
    val_diff = pl.when(is_control).then(pl.col(weighted_diff)).otherwise(None)

    df = df.with_columns(
        pl.when(is_control).then(val_weight.sum().over(group_vars)).otherwise(None).alias(weight_sum_col),
        pl.when(is_control).then(val_diff.sum().over(group_vars)).otherwise(None).alias(diff_sum_col),
    )

    if cluster_col is None:
        val_dof = pl.when(is_control).then(pl.lit(1)).otherwise(None)
        df = df.with_columns(pl.when(is_control).then(val_dof.sum().over(group_vars)).otherwise(None).alias(dof_col))
    else:
        cluster_flag = f"_control_cluster_{h}"
        df = df.with_columns(pl.when(is_control).then(pl.col(cluster_col)).otherwise(None).alias(cluster_flag))
        df = df.with_columns(
            pl.when(pl.col(cluster_flag).is_not_null())
            .then(pl.col(cluster_flag).n_unique().over(group_vars))
            .otherwise(None)
            .alias(dof_col)
        )
        df = df.drop(cluster_flag)

    df = df.with_columns(
        pl.when(pl.col(weight_sum_col) > 0)
        .then(pl.col(diff_sum_col) / pl.col(weight_sum_col))
        .otherwise(pl.lit(0.0))
        .alias(mean_col)
    )

    return df


def compute_union_dof(df, horizon, config, cluster_col=None):
    """Compute DOF for the union of switchers and controls.

    Parameters
    ----------
    df : pl.DataFrame
        Data with switcher and control flags.
    horizon : int
        Current horizon.
    config : DIDInterConfig
        Configuration object.
    cluster_col : str or None
        Column name for cluster-robust DOF counting.

    Returns
    -------
    pl.DataFrame
        DataFrame with dof_union_{h} and union_mean_{h} columns.
    """
    h = abs(horizon)
    tname = config.tname
    trends = config.trends_nonparam or []

    switcher_flag = f"is_switcher_{h}"
    never_col = f"never_change_{h}"
    weighted_diff = f"weighted_diff_{h}"

    if switcher_flag not in df.columns or never_col not in df.columns:
        return df

    is_union = (pl.col(switcher_flag) == 1) | (pl.col(never_col) == 1.0)
    group_vars = [tname, "d_sq", *list(trends)]

    union_flag = f"is_union_{h}"
    weight_sum_col = f"union_weight_sum_{h}"
    diff_sum_col = f"union_diff_sum_{h}"
    dof_col = f"dof_union_{h}"
    mean_col = f"union_mean_{h}"

    df = df.with_columns(is_union.cast(pl.Int64).alias(union_flag))

    val_weight = pl.when(is_union).then(pl.col("weight_gt")).otherwise(None)
    val_diff = pl.when(is_union).then(pl.col(weighted_diff)).otherwise(None)

    df = df.with_columns(
        pl.when(is_union).then(val_weight.sum().over(group_vars)).otherwise(None).alias(weight_sum_col),
        pl.when(is_union).then(val_diff.sum().over(group_vars)).otherwise(None).alias(diff_sum_col),
    )

    if cluster_col is None:
        val_dof = pl.when(is_union).then(pl.col(union_flag)).otherwise(None)
        df = df.with_columns(pl.when(is_union).then(val_dof.sum().over(group_vars)).otherwise(None).alias(dof_col))
    else:
        cluster_flag = f"_union_cluster_{h}"
        df = df.with_columns(pl.when(is_union).then(pl.col(cluster_col)).otherwise(None).alias(cluster_flag))
        df = df.with_columns(
            pl.when(pl.col(cluster_flag).is_not_null())
            .then(pl.col(cluster_flag).n_unique().over(group_vars))
            .otherwise(None)
            .alias(dof_col)
        )
        df = df.drop(cluster_flag)

    df = df.with_columns(
        pl.when(pl.col(weight_sum_col) > 0)
        .then(pl.col(diff_sum_col) / pl.col(weight_sum_col))
        .otherwise(pl.lit(0.0))
        .alias(mean_col)
    )

    return df


def compute_e_hat(df, horizon, config):
    """Compute cohort mean with 3-level DOF fallback.

    Parameters
    ----------
    df : pl.DataFrame
        Data with DOF and mean columns.
    horizon : int
        Current horizon.
    config : DIDInterConfig
        Configuration object.

    Returns
    -------
    pl.DataFrame
        DataFrame with E_hat_{h} column.
    """
    h = abs(horizon)
    tname = config.tname

    e_hat_col = f"E_hat_{h}"
    dof_s_col = f"dof_switcher_{h}"
    dof_ns_col = f"dof_control_{h}"
    dof_union_col = f"dof_union_{h}"
    mean_s_col = f"cohort_mean_{h}"
    mean_ns_col = f"control_mean_{h}"
    mean_union_col = f"union_mean_{h}"

    time = pl.col(tname)
    fg = pl.col("F_g")

    at_target = (fg - 1 + h) == time
    before_switch = time < fg
    relevant = at_target | before_switch

    dof_s = pl.col(dof_s_col) if dof_s_col in df.columns else pl.lit(None)
    dof_ns = pl.col(dof_ns_col) if dof_ns_col in df.columns else pl.lit(None)
    dof_union = pl.col(dof_union_col) if dof_union_col in df.columns else pl.lit(None)
    mean_s = pl.col(mean_s_col) if mean_s_col in df.columns else pl.lit(0.0)
    mean_ns = pl.col(mean_ns_col) if mean_ns_col in df.columns else pl.lit(0.0)
    mean_union = pl.col(mean_union_col) if mean_union_col in df.columns else pl.lit(0.0)

    s_safe = dof_s.fill_null(9999)
    ns_safe = dof_ns.fill_null(9999)
    union_safe = dof_union.fill_null(9999)

    use_switcher_mean = at_target & (s_safe >= 2)
    use_control_mean = before_switch & (ns_safe >= 2)
    use_union_mean = (union_safe >= 2) & ((at_target & (s_safe == 1)) | (before_switch & (ns_safe == 1)))

    df = df.with_columns(
        pl.when(~relevant)
        .then(pl.lit(None))
        .when(use_switcher_mean)
        .then(mean_s)
        .when(use_control_mean)
        .then(mean_ns)
        .when(use_union_mean)
        .then(mean_union)
        .otherwise(pl.lit(0.0))
        .alias(e_hat_col)
    )

    return df


def compute_dof_scaling(df, horizon, config):
    """Compute DOF scaling factor for variance correction with 3-level fallback.

    Parameters
    ----------
    df : pl.DataFrame
        Data with dof_switcher, dof_control, and dof_union columns.
    horizon : int
        Current horizon.
    config : DIDInterConfig
        Configuration object. Uses ``less_conservative_se`` to control whether
        DOF adjustments are applied.

    Returns
    -------
    pl.DataFrame
        DataFrame with dof_scale_{h} column.
    """
    h = abs(horizon)
    tname = config.tname

    dof_col = f"dof_scale_{h}"
    dof_s_col = f"dof_switcher_{h}"
    dof_ns_col = f"dof_control_{h}"
    dof_union_col = f"dof_union_{h}"

    df = df.with_columns(pl.lit(1.0).alias(dof_col))

    if getattr(config, "less_conservative_se", False):
        return df

    time = pl.col(tname)
    fg = pl.col("F_g")

    at_target_time = (fg - 1 + h) == time
    before_switch = time < fg

    dof_s = pl.col(dof_s_col) if dof_s_col in df.columns else pl.lit(9999)
    dof_ns = pl.col(dof_ns_col) if dof_ns_col in df.columns else pl.lit(9999)
    dof_union = pl.col(dof_union_col) if dof_union_col in df.columns else pl.lit(9999)

    s_safe = dof_s.fill_null(9999)
    ns_safe = dof_ns.fill_null(9999)
    union_safe = dof_union.fill_null(9999)

    use_s_dof = at_target_time & (s_safe > 1)
    use_ns_dof = before_switch & (ns_safe > 1)
    use_union_s = at_target_time & (s_safe == 1) & (union_safe >= 2)
    use_union_ns = before_switch & (ns_safe == 1) & (union_safe >= 2)

    df = df.with_columns(
        pl.when(use_s_dof)
        .then((dof_s / (dof_s - 1)).sqrt())
        .when(use_ns_dof)
        .then((dof_ns / (dof_ns - 1)).sqrt())
        .when(use_union_s | use_union_ns)
        .then((dof_union / (dof_union - 1)).sqrt())
        .otherwise(pl.col(dof_col))
        .alias(dof_col)
    )

    return df


def compute_clustered_variance(influence_func, cluster_ids, n_groups):
    r"""Compute clustered standard error from influence function.

    Computes standard errors for :math:`\text{DID}_\ell` estimators using the
    influence function approach with optional clustering. The variance is computed as

    .. math::

        \widehat{\text{Var}}(\text{DID}_\ell) = \frac{1}{G^2}
        \sum_{c=1}^{C} \left(\sum_{g \in c} \psi_g\right)^2

    where :math:`\psi_g` is the influence function for group :math:`g`, :math:`G`
    is the total number of groups, and :math:`C` is the number of clusters. When
    not clustering, each group is its own cluster.

    Parameters
    ----------
    influence_func : ndarray
        Influence function values :math:`\psi_g` for each group.
    cluster_ids : ndarray
        Cluster identifiers for each group.
    n_groups : int
        Total number of groups :math:`G`.

    Returns
    -------
    float
        Clustered standard error.

    References
    ----------

    .. [1] de Chaisemartin, C., & D'Haultfoeuille, X. (2024). Difference-in-
           Differences Estimators of Intertemporal Treatment Effects.
           *Review of Economics and Statistics*, 106(6), 1723-1736.
    """
    cluster_sums, unique_clusters = compute_cluster_sums(influence_func, cluster_ids)
    n_clusters = len(unique_clusters)

    if n_clusters <= 1:
        return np.sqrt(np.sum(influence_func**2)) / n_groups

    std_error = np.sqrt(np.sum(cluster_sums**2)) / n_groups

    return std_error


def compute_joint_test(estimates, vcov):
    r"""Compute joint Wald test that all estimates are zero.

    Computes a chi-squared test statistic for the null hypothesis
    :math:`H_0: \delta_1 = \delta_2 = \cdots = \delta_L = 0`. For placebo
    effects, this tests the parallel trends assumption by checking whether
    pre-treatment outcome trends differ between switchers and non-switchers.
    The test statistic is

    .. math::

        W = \hat{\boldsymbol{\delta}}' \widehat{\mathbf{V}}^{-1} \hat{\boldsymbol{\delta}}
        \sim \chi^2_L

    where :math:`\hat{\boldsymbol{\delta}}` is the vector of estimates and
    :math:`\widehat{\mathbf{V}}` is the variance-covariance matrix.

    Parameters
    ----------
    estimates : ndarray
        Point estimates :math:`\hat{\boldsymbol{\delta}}`.
    vcov : ndarray
        Variance-covariance matrix :math:`\widehat{\mathbf{V}}`.

    Returns
    -------
    dict or None
        Dictionary with chi2_stat, df, and p_value, or None if computation fails.

    References
    ----------

    .. [1] de Chaisemartin, C., & D'Haultfoeuille, X. (2024). Difference-in-
           Differences Estimators of Intertemporal Treatment Effects.
           *Review of Economics and Statistics*, 106(6), 1723-1736.
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
