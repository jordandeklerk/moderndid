"""Conditional expectation and degrees-of-freedom adjustments for influence function variance."""

import polars as pl


def build_treatment_paths_full(df, horizon, config):
    """Build treatment path identifiers for cohort grouping.

    Parameters
    ----------
    df : pl.DataFrame
        Data with F_g, d_sq, and treatment columns.
    horizon : int
        Maximum horizon for path construction.
    config : DIDInterConfig
        Configuration with column names.

    Returns
    -------
    pl.DataFrame
        DataFrame with path_0, path_1, ..., path_h columns added.
    """
    gname = config.gname
    tname = config.tname
    dname = config.dname

    d_sq_col = "d_sq" if "d_sq" in df.columns else dname

    df = df.with_columns(
        pl.concat_str(
            [
                pl.col(d_sq_col).cast(pl.Utf8),
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

    for i in range(1, horizon + 1):
        d_fg_col = f"d_fg_{i}"

        df = df.with_columns(
            pl.when(pl.col(tname) == pl.col("F_g") + i - 1).then(pl.col(dname)).otherwise(None).alias(f"_d_at_fg_{i}")
        )

        df = df.with_columns(pl.col(f"_d_at_fg_{i}").max().over(gname).alias(d_fg_col))

        prev_path = f"path_{i - 1}" if i > 1 else "path_0"
        df = df.with_columns(
            pl.concat_str(
                [
                    pl.col(prev_path).cast(pl.Utf8),
                    pl.lit("|"),
                    pl.col(d_fg_col).cast(pl.Utf8),
                ]
            )
            .cast(pl.Categorical)
            .to_physical()
            .cast(pl.Int64)
            .add(1)
            .alias(f"path_{i}")
        )

        df = df.with_columns(pl.col(gname).n_unique().over(f"path_{i}").alias(f"num_g_paths_{i}"))
        df = df.with_columns((pl.col(f"num_g_paths_{i}") > 1).cast(pl.Int64).alias(f"cohort_fullpath_{i}"))

        df = df.drop(f"_d_at_fg_{i}")

    return df


def compute_cohort_means(df, horizon, diff_col, cluster=None, trends_nonparam=None):
    """Compute cohort-specific means of outcome differences.

    Parameters
    ----------
    df : pl.DataFrame
        Data with path columns and outcome differences.
    horizon : int
        Current horizon.
    diff_col : str
        Name of the outcome difference column.
    cluster : str, optional
        Cluster variable for DOF counting.
    trends_nonparam : list, optional
        Non-parametric trend variables for grouping.

    Returns
    -------
    pl.DataFrame
        DataFrame with cohort mean and DOF columns added.
    """
    weight_col = "weight_gt" if "weight_gt" in df.columns else None
    n_gt = pl.col(weight_col) if weight_col else pl.lit(1.0)

    diff_yn_col = f"diff_y_{horizon}_N_gt"
    df = df.with_columns((pl.col(diff_col) * n_gt).alias(diff_yn_col))

    dof_col = f"dof_s_{horizon}"
    df = df.with_columns(pl.col("first_obs_by_gp").alias(dof_col))

    cond_dof = pl.col(diff_col).is_not_null() & pl.col(f"path_{horizon}").is_not_null()

    path_tags = [
        ("path_0", "s0"),
        ("path_1", "s1") if horizon >= 1 else (None, None),
        (f"path_{horizon}", "s2"),
    ]

    for path_col, s_tag in path_tags:
        if path_col is None:
            continue

        grp_cols = _get_group_cols(path_col, trends_nonparam)
        count_col = f"count_cohort_{horizon}_{s_tag}_t"
        total_col = f"total_cohort_{horizon}_{s_tag}_t"

        val_expr_count = pl.when(cond_dof).then(n_gt).otherwise(None)
        df = df.with_columns(
            pl.when(cond_dof).then(val_expr_count.sum().over(grp_cols)).otherwise(None).alias(count_col)
        )

        val_expr_total = pl.when(cond_dof).then(pl.col(diff_yn_col)).otherwise(None)
        df = df.with_columns(
            pl.when(cond_dof).then(val_expr_total.sum().over(grp_cols)).otherwise(None).alias(total_col)
        )

    if cluster is None:
        val_expr_dof = pl.when(cond_dof).then(pl.col(dof_col)).otherwise(None)
        for path_col, s_tag in path_tags:
            if path_col is None:
                continue
            grp_cols = _get_group_cols(path_col, trends_nonparam)
            dof_cohort_col = f"dof_cohort_{horizon}_{s_tag}_t"
            df = df.with_columns(
                pl.when(cond_dof).then(val_expr_dof.sum().over(grp_cols)).otherwise(None).alias(dof_cohort_col)
            )
    else:
        cluster_dof_col = f"cluster_dof_{horizon}_s"
        df = df.with_columns(pl.when(cond_dof).then(pl.col(cluster)).otherwise(None).alias(cluster_dof_col))
        for path_col, s_tag in path_tags:
            if path_col is None:
                continue
            grp_cols = _get_group_cols(path_col, trends_nonparam)
            dof_cohort_col = f"dof_cohort_{horizon}_{s_tag}_t"
            df = df.with_columns(
                pl.when(pl.col(cluster_dof_col).is_not_null())
                .then(pl.col(cluster_dof_col).n_unique().over(grp_cols))
                .otherwise(None)
                .alias(dof_cohort_col)
            )

    col_s0 = f"dof_cohort_{horizon}_s0_t"
    col_s1 = f"dof_cohort_{horizon}_s1_t" if horizon >= 1 else col_s0
    col_s2 = f"dof_cohort_{horizon}_s2_t"
    col_st = f"dof_cohort_{horizon}_s_t"

    df = df.with_columns(
        pl.when(pl.col(col_s2) >= 2)
        .then(pl.col(col_s2))
        .when((pl.col(col_s2) < 2) & (pl.col(col_s1) >= 2))
        .then(pl.col(col_s1))
        .otherwise(pl.col(col_s0))
        .alias(col_st)
    )

    col_cnt_s0 = f"count_cohort_{horizon}_s0_t"
    col_cnt_s1 = f"count_cohort_{horizon}_s1_t" if horizon >= 1 else col_cnt_s0
    col_cnt_s2 = f"count_cohort_{horizon}_s2_t"

    col_tot_s0 = f"total_cohort_{horizon}_s0_t"
    col_tot_s1 = f"total_cohort_{horizon}_s1_t" if horizon >= 1 else col_tot_s0
    col_tot_s2 = f"total_cohort_{horizon}_s2_t"

    mean_s_col = f"mean_cohort_{horizon}_s_t"

    df = df.with_columns(
        pl.when(pl.col(col_s2) >= 2)
        .then(pl.col(col_tot_s2) / pl.col(col_cnt_s2))
        .when((pl.col(col_s2) < 2) & (pl.col(col_s1) >= 2))
        .then(pl.col(col_tot_s1) / pl.col(col_cnt_s1))
        .otherwise(pl.col(col_tot_s0) / pl.col(col_cnt_s0))
        .alias(mean_s_col)
    )

    return df


def compute_e_hat(df, horizon, config):
    """Compute conditional expectation for variance estimation.

    Parameters
    ----------
    df : pl.DataFrame
        Data with cohort mean columns.
    horizon : int
        Current horizon.
    config : DIDInterConfig
        Configuration with column names.

    Returns
    -------
    pl.DataFrame
        DataFrame with E_hat_{horizon} column added.
    """
    tname = config.tname

    e_hat_col = f"E_hat_{horizon}"
    mean_s = f"mean_cohort_{horizon}_s_t"
    dof_s = f"dof_cohort_{horizon}_s_t"

    df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(e_hat_col))

    time = pl.col(tname)
    fg = pl.col("F_g")
    s = pl.col(dof_s)

    s9999 = s.fill_nan(9999).fill_null(9999)

    cond_a = (time < fg) | ((fg - 1 + horizon) == time)
    cond_b = ((fg - 1 + horizon) == time) & (s9999 >= 2)

    df = df.with_columns(pl.when(cond_a).then(0.0).otherwise(pl.col(e_hat_col)).alias(e_hat_col))
    df = df.with_columns(pl.when(cond_b).then(pl.col(mean_s)).otherwise(pl.col(e_hat_col)).alias(e_hat_col))

    df = df.with_columns(
        pl.when(pl.col(mean_s).is_null() | pl.col(mean_s).is_nan())
        .then(pl.lit(None))
        .otherwise(pl.col(e_hat_col))
        .alias(e_hat_col)
    )

    return df


def compute_dof(df, horizon, config):
    """Compute DOF scaling for variance estimation.

    Parameters
    ----------
    df : pl.DataFrame
        Data with DOF cohort columns.
    horizon : int
        Current horizon.
    config : DIDInterConfig
        Configuration with column names.

    Returns
    -------
    pl.DataFrame
        DataFrame with DOF_{horizon} column added.
    """
    tname = config.tname

    dof_col = f"DOF_{horizon}"
    dof_s = f"dof_cohort_{horizon}_s_t"

    df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(dof_col))

    time = pl.col(tname)
    fg = pl.col("F_g")
    s = pl.col(dof_s)

    cond_a = (time < fg) | ((fg - 1 + horizon) == time)
    cond_b = ((fg - 1 + horizon) == time) & (s.fill_null(9999) > 1)

    df = df.with_columns(pl.when(cond_a).then(1.0).otherwise(pl.col(dof_col)).alias(dof_col))
    df = df.with_columns(pl.when(cond_b).then((s / (s - 1)).sqrt()).otherwise(pl.col(dof_col)).alias(dof_col))
    df = df.with_columns(pl.when(s.is_null()).then(pl.lit(None)).otherwise(pl.col(dof_col)).alias(dof_col))

    return df


def compute_variance_influence(
    df,
    horizon,
    config,
    diff_col,
    dist_col,
    never_col,
    n_treated_col,
    n_control_col,
    n_groups,
    n_switchers,
):
    """Compute variance influence.

    Parameters
    ----------
    df : pl.DataFrame
        Data with E_hat and DOF columns.
    horizon : int
        Current horizon.
    config : DIDInterConfig
        Configuration with column names.
    diff_col : str
        Name of outcome difference column.
    dist_col : str
        Name of distance to switch column.
    never_col : str
        Name of never-switcher indicator column.
    n_treated_col : str
        Name of treated count column.
    n_control_col : str
        Name of control count column.
    n_groups : int
        Total number of groups.
    n_switchers : float
        Weighted number of switchers.

    Returns
    -------
    pl.DataFrame
        DataFrame with inf_var_{horizon} column added.
    """
    gname = config.gname

    e_hat_col = f"E_hat_{horizon}"
    dof_col = f"DOF_{horizon}"
    inf_var_col = f"inf_var_{horizon}"

    if e_hat_col not in df.columns or dof_col not in df.columns:
        return df

    safe_n_control = (
        pl.when(pl.col(n_control_col).is_null() | (pl.col(n_control_col) == 0))
        .then(1.0)
        .otherwise(pl.col(n_control_col))
    )

    df = df.with_columns(
        (
            (pl.lit(n_groups) / pl.lit(n_switchers))
            * pl.col("weight_gt")
            * (pl.col(dist_col) - (pl.col(n_treated_col) / safe_n_control) * pl.col(never_col).fill_null(0.0))
            * pl.col(dof_col).fill_null(1.0)
            * (pl.col(diff_col).fill_null(0.0) - pl.col(e_hat_col).fill_null(0.0))
        ).alias(inf_var_col)
    )

    df = df.with_columns((pl.col(inf_var_col).sum().over(gname) * pl.col("first_obs_by_gp")).alias(inf_var_col))

    return df


def _get_group_cols(path_col, trends_nonparam):
    """Return grouping columns for cohort aggregation."""
    base = [path_col]
    if trends_nonparam:
        base.extend(trends_nonparam)
    return base
