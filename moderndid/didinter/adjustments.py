"""Data adjustments for DIDInter estimation."""

import polars as pl


def compute_same_switchers_mask(df, config, n_horizons, t_max, horizon_type="effect"):
    """Compute mask for switchers valid at all horizons.

    Parameters
    ----------
    df : pl.DataFrame
        Data with F_g and outcome columns.
    config : DIDInterConfig
        Configuration object.
    n_horizons : int
        Number of horizons to check.
    t_max : int
        Maximum time period.
    horizon_type : str
        "effect" or "placebo".

    Returns
    -------
    pl.DataFrame
        DataFrame with same_switcher_valid column added.
    """
    gname = config.gname
    tname = config.tname
    yname = config.yname

    df = df.sort([gname, tname])
    df = df.with_columns(pl.lit(0).alias("valid_horizon_count"))

    for h in range(1, n_horizons + 1):
        if horizon_type == "effect":
            diff_col = f"_diff_check_{h}"
            df = df.with_columns(pl.col(yname).diff(h).over(gname).alias(diff_col))
            target_time = pl.col("F_g") + h - 1
        else:
            diff_col = f"_diff_check_pl_{h}"
            df = df.with_columns(
                (pl.col(yname).shift(2 * h).over(gname) - pl.col(yname).shift(h).over(gname)).alias(diff_col)
            )
            target_time = pl.col("F_g") - h - 1

        never_col = f"_never_check_{h}"
        df = df.with_columns(
            pl.when(pl.col(diff_col).is_not_null())
            .then((pl.col("F_g") > pl.col(tname)).cast(pl.Float64))
            .otherwise(pl.lit(None))
            .alias(never_col)
        )

        if config.only_never_switchers:
            df = df.with_columns(
                pl.when(
                    (pl.col("F_g") > pl.col(tname)) & (pl.col("F_g") < (t_max + 1)) & pl.col(diff_col).is_not_null()
                )
                .then(0.0)
                .otherwise(pl.col(never_col))
                .alias(never_col)
            )

        never_w_col = f"_never_w_check_{h}"
        df = df.with_columns((pl.col(never_col) * pl.col("weight_gt")).alias(never_w_col))

        n_control_col = f"_n_control_check_{h}"
        df = df.with_columns(pl.col(never_w_col).sum().over([tname, "d_sq"]).alias(n_control_col))

        n_control_at_target = f"_n_control_target_{h}"
        df = df.with_columns(
            pl.when(pl.col(tname) == target_time).then(pl.col(n_control_col)).otherwise(None).alias(n_control_at_target)
        )

        n_control_mean = f"_n_control_mean_{h}"
        df = df.with_columns(pl.col(n_control_at_target).mean().over(gname).alias(n_control_mean))

        diff_at_target = f"_diff_target_{h}"
        df = df.with_columns(
            pl.when(pl.col(tname) == target_time).then(pl.col(diff_col)).otherwise(None).alias(diff_at_target)
        )

        diff_mean = f"_diff_mean_{h}"
        df = df.with_columns(pl.col(diff_at_target).mean().over(gname).alias(diff_mean))

        df = df.with_columns(
            (
                pl.col("valid_horizon_count")
                + ((pl.col(n_control_mean) > 0) & pl.col(diff_mean).is_not_null()).cast(pl.Int64)
            ).alias("valid_horizon_count")
        )

    df = df.with_columns((pl.col("valid_horizon_count") == n_horizons).alias("same_switcher_valid"))

    cols_to_drop = [
        c
        for c in df.columns
        if c.startswith("_diff_check")
        or c.startswith("_never_check")
        or c.startswith("_never_w_check")
        or c.startswith("_n_control_check")
        or c.startswith("_n_control_target")
        or c.startswith("_n_control_mean")
        or c.startswith("_diff_target")
        or c.startswith("_diff_mean")
    ]
    df = df.drop(cols_to_drop)

    return df


def get_group_vars(config):
    """Get grouping variables for control matching.

    Parameters
    ----------
    config : DIDInterConfig
        Configuration object.

    Returns
    -------
    list
        List of grouping columns including trends if specified.
    """
    group_vars = [config.tname, "d_sq"]

    if config.trends_nonparam:
        group_vars.extend(config.trends_nonparam)

    return group_vars
