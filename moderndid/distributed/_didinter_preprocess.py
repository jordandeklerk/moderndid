"""Preprocessing functions for DIDInter partitions."""

from __future__ import annotations

import warnings

import polars as pl


def partition_preprocess_local(pdf, col_config, config_flags):
    """Column selection, missing-data handling, switcher identification/filtering, and partial data preparation.

    All operations group by gname which is partition-local after repartitioning by unit ID.
    Accepts a pandas or polars DataFrame. Always returns a polars DataFrame.
    """
    if len(pdf) == 0:
        return pdf

    gname = col_config["gname"]
    tname = col_config["tname"]
    yname = col_config["yname"]
    dname = col_config["dname"]

    df = pdf if isinstance(pdf, pl.DataFrame) else pl.from_pandas(pdf)

    cols_to_keep = [yname, tname, gname, dname]
    weightsname = config_flags.get("weightsname")
    if weightsname:
        cols_to_keep.append(weightsname)
    cluster = col_config.get("cluster")
    if cluster:
        cols_to_keep.append(cluster)
    xformla = config_flags.get("xformla")
    if xformla and xformla != "~1":
        cols_to_keep.extend(_extract_vars_from_formula(xformla))
    trends_nonparam = config_flags.get("trends_nonparam")
    if trends_nonparam:
        cols_to_keep.extend(trends_nonparam)
    het_covariates = col_config.get("het_covariates")
    if het_covariates:
        cols_to_keep.extend(het_covariates)
    cols_to_keep = list(dict.fromkeys(c for c in cols_to_keep if c is not None and c in df.columns))
    df = df.select(cols_to_keep)

    for col in [tname, gname, dname]:
        if col in df.columns and df[col].dtype not in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
            df = df.with_columns(pl.col(col).cast(pl.Float64))

    df = df.with_columns(
        [
            pl.col(dname).mean().over(gname).alias("_mean_D"),
            pl.col(yname).mean().over(gname).alias("_mean_Y"),
        ]
    )
    df = df.filter(pl.col("_mean_D").is_not_null() & pl.col("_mean_Y").is_not_null())
    df = df.drop(["_mean_D", "_mean_Y"])
    if len(df) == 0:
        return df

    df = df.sort([gname, tname])
    df = df.with_columns((pl.col(dname) - pl.col(dname).shift(1).over(gname)).alias("_d_diff"))

    base_treatment_pre = (
        df.filter(pl.col(tname) == pl.col(tname).min().over(gname))
        .select([gname, pl.col(dname).alias("_d_sq_pre")])
        .unique()
    )
    df = df.join(base_treatment_pre, on=gname, how="left")
    df = df.with_columns((pl.col(dname) - pl.col("_d_sq_pre")).alias("_diff_from_sq"))

    first_switch_pre = (
        df.filter((pl.col("_d_diff") != 0) & pl.col("_d_diff").is_not_null())
        .group_by(gname)
        .agg(pl.col(tname).min().alias("_F_g_pre"))
    )
    df = df.join(first_switch_pre, on=gname, how="left")

    t_max_per_unit = df.group_by(gname).agg(pl.col(tname).max().alias("_T_max_unit"))
    df = df.join(t_max_per_unit, on=gname, how="left")

    df = df.with_columns(
        pl.when(pl.col("_F_g_pre").is_not_null())
        .then(pl.col("_T_max_unit") - pl.col("_F_g_pre") + 1)
        .otherwise(pl.lit(0.0))
        .alias("L_g")
    )
    df = df.drop(["_F_g_pre", "_T_max_unit"])

    df = df.with_columns(
        [
            pl.when((pl.col("_diff_from_sq") > 0) & pl.col(dname).is_not_null())
            .then(1)
            .otherwise(0)
            .cum_sum()
            .clip(upper_bound=1)
            .over(gname)
            .alias("_ever_strict_increase"),
            pl.when((pl.col("_diff_from_sq") < 0) & pl.col(dname).is_not_null())
            .then(1)
            .otherwise(0)
            .cum_sum()
            .clip(upper_bound=1)
            .over(gname)
            .alias("_ever_strict_decrease"),
        ]
    )

    if not config_flags.get("keep_bidirectional_switchers", False):
        df = df.filter(~((pl.col("_ever_strict_increase") == 1) & (pl.col("_ever_strict_decrease") == 1)))

    df = df.drop(["_ever_strict_increase", "_ever_strict_decrease", "_d_sq_pre", "_diff_from_sq"])
    if len(df) == 0:
        return df

    first_switch = (
        df.filter((pl.col("_d_diff") != 0) & pl.col("_d_diff").is_not_null())
        .group_by(gname)
        .agg(pl.col(tname).min().alias("F_g"))
    )
    df = df.join(first_switch, on=gname, how="left")
    df = df.with_columns(pl.col("F_g").fill_null(float("inf")))

    base_treatment = (
        df.filter(pl.col(tname) == pl.col(tname).min().over(gname))
        .select([gname, pl.col(dname).alias("d_sq")])
        .unique()
    )
    df = df.join(base_treatment, on=gname, how="left")

    switch_treatment = df.filter(pl.col(tname) == pl.col("F_g")).select([gname, pl.col(dname).alias("d_fg")]).unique()
    df = df.join(switch_treatment, on=gname, how="left")

    switch_direction = (
        df.filter(pl.col("_d_diff").is_not_null() & (pl.col("_d_diff") != 0))
        .group_by(gname)
        .agg(pl.col("_d_diff").first().alias("_first_diff"))
    )
    df = df.join(switch_direction, on=gname, how="left")

    df = df.with_columns(
        pl.when(pl.col("F_g") == float("inf"))
        .then(0)
        .when(pl.col("_first_diff") > 0)
        .then(1)
        .when(pl.col("_first_diff") < 0)
        .then(-1)
        .otherwise(0)
        .alias("S_g")
    )

    if config_flags.get("drop_missing_preswitch", False):
        min_treat_time = (
            df.filter(pl.col(dname).is_not_null()).group_by(gname).agg(pl.col(tname).min().alias("_min_treat_time"))
        )
        df = df.join(min_treat_time, on=gname, how="left")
        df = df.filter(
            ~(
                (pl.col("_min_treat_time") < pl.col("F_g"))
                & (pl.col(tname) >= pl.col("_min_treat_time"))
                & (pl.col(tname) < pl.col("F_g"))
                & pl.col(dname).is_null()
            )
        )
        df = df.drop("_min_treat_time")

    df = df.drop(["_d_diff", "_first_diff"])

    switchers_mode = config_flags.get("switchers", "")
    if switchers_mode == "in":
        valid_units = df.filter((pl.col("S_g") == 1) | (pl.col("S_g") == 0))[gname].unique().to_list()
        df = df.filter(pl.col(gname).is_in(valid_units))
    elif switchers_mode == "out":
        valid_units = df.filter((pl.col("S_g") == -1) | (pl.col("S_g") == 0))[gname].unique().to_list()
        df = df.filter(pl.col(gname).is_in(valid_units))

    if len(df) == 0:
        return df

    df = df.sort([gname, tname])
    df = df.with_columns(pl.lit(1.0).alias("weight_gt"))
    if weightsname and weightsname in df.columns:
        df = df.with_columns(pl.col(weightsname).fill_null(0).alias("weight_gt"))

    df = df.with_columns(
        pl.when(pl.col(yname).is_null() | pl.col(dname).is_null())
        .then(0.0)
        .otherwise(pl.col("weight_gt"))
        .alias("weight_gt")
    )

    first_obs = df.group_by(gname).agg(pl.col(tname).min().alias("_first_t")).select([gname, "_first_t"])
    df = df.join(first_obs, on=gname, how="left")
    df = df.with_columns((pl.col(tname) == pl.col("_first_t")).cast(pl.Int64).alias("first_obs_by_gp"))
    df = df.drop("_first_t")

    t_max_by_group = df.group_by(gname).agg(pl.col(tname).max().alias("t_max_by_group"))
    df = df.join(t_max_by_group, on=gname, how="left")

    return df


def partition_extract_metadata(pdf, config_flags):
    """Extract partial metadata from a preprocessed partition for cross-partition aggregation."""
    if len(pdf) == 0:
        return {
            "weight_sum": 0.0,
            "weight_count": 0,
            "fg_variation": {},
            "controls_time": set(),
            "unique_times": set(),
            "unique_d_sq": set(),
            "t_min": float("inf"),
            "t_max": float("-inf"),
            "n_switchers": 0,
            "n_never_switchers": 0,
            "all_gnames_first_obs": set(),
            "max_effects_available": 0,
            "max_placebo_available": 0,
            "fg_trunc_max": {},
        }

    df = pl.from_pandas(pdf) if not isinstance(pdf, pl.DataFrame) else pdf
    gname_col = config_flags["gname"]
    tname_col = config_flags["tname"]
    trends_nonparam = config_flags.get("trends_nonparam")

    weight_sum = float(df["weight_gt"].sum())
    weight_count = len(df)

    group_cols = ["d_sq"]
    if trends_nonparam:
        group_cols.extend(trends_nonparam)

    df_with_fg = df.with_columns(
        pl.when(pl.col("F_g") == float("inf")).then(0.0).otherwise(pl.col("F_g")).alias("_F_g_for_std")
    )
    fg_variation = {}
    for key, grp in df_with_fg.group_by(group_cols):
        k = tuple(key) if isinstance(key, (list, tuple)) else (key,)
        vals = set(grp["_F_g_for_std"].drop_nulls().unique().to_list())
        fg_variation[k] = vals

    ctrl_cols = [tname_col, "d_sq"]
    if trends_nonparam:
        ctrl_cols.extend(trends_nonparam)
    never_df = df.filter(pl.col("F_g") == float("inf"))
    controls_time = set()
    if len(never_df) > 0:
        for key, _ in never_df.group_by(ctrl_cols):
            k = tuple(key) if isinstance(key, (list, tuple)) else (key,)
            controls_time.add(k)

    unique_times = set(df[tname_col].drop_nulls().unique().to_list())
    unique_d_sq = set(df["d_sq"].drop_nulls().unique().to_list())
    t_min = float(df[tname_col].min())
    t_max = float(df[tname_col].max())

    gname_fg = df.unique(subset=[gname_col])
    n_switchers = int((gname_fg["F_g"] != float("inf")).sum())
    n_never_switchers = int((gname_fg["F_g"] == float("inf")).sum())
    all_gnames_first_obs = set(df.filter(pl.col("first_obs_by_gp") == 1)[gname_col].to_list())

    switchers_df = gname_fg.filter(pl.col("F_g") != float("inf"))
    if len(switchers_df) > 0:
        max_effects = int((t_max - switchers_df["F_g"] + 1).max())
        max_placebo = int((switchers_df["F_g"] - t_min - 1).max())
    else:
        max_effects = 0
        max_placebo = 0

    df_trunc = df.with_columns(
        pl.when(pl.col("F_g") == float("inf"))
        .then(pl.col("t_max_by_group") + 1)
        .otherwise(pl.col("F_g"))
        .alias("_F_g_trunc")
    )
    fg_trunc_max = {}
    for key, grp in df_trunc.group_by(group_cols):
        k = tuple(key) if isinstance(key, (list, tuple)) else (key,)
        fg_trunc_max[k] = float(grp["_F_g_trunc"].max())

    return {
        "weight_sum": weight_sum,
        "weight_count": weight_count,
        "fg_variation": fg_variation,
        "controls_time": controls_time,
        "unique_times": unique_times,
        "unique_d_sq": unique_d_sq,
        "t_min": t_min,
        "t_max": t_max,
        "n_switchers": n_switchers,
        "n_never_switchers": n_never_switchers,
        "all_gnames_first_obs": all_gnames_first_obs,
        "max_effects_available": max_effects,
        "max_placebo_available": max_placebo,
        "fg_trunc_max": fg_trunc_max,
    }


def reduce_metadata(a, b):
    """Pairwise merge of metadata dicts."""
    fg_var = dict(a.get("fg_variation", {}))
    for k, v in b.get("fg_variation", {}).items():
        fg_var[k] = fg_var.get(k, set()) | v

    fg_tm = dict(a.get("fg_trunc_max", {}))
    for k, v in b.get("fg_trunc_max", {}).items():
        fg_tm[k] = max(fg_tm.get(k, v), v)

    return {
        "weight_sum": a["weight_sum"] + b["weight_sum"],
        "weight_count": a["weight_count"] + b["weight_count"],
        "fg_variation": fg_var,
        "controls_time": a["controls_time"] | b["controls_time"],
        "unique_times": a["unique_times"] | b["unique_times"],
        "unique_d_sq": a["unique_d_sq"] | b["unique_d_sq"],
        "t_min": min(a["t_min"], b["t_min"]),
        "t_max": max(a["t_max"], b["t_max"]),
        "n_switchers": a["n_switchers"] + b["n_switchers"],
        "n_never_switchers": a["n_never_switchers"] + b["n_never_switchers"],
        "all_gnames_first_obs": a["all_gnames_first_obs"] | b["all_gnames_first_obs"],
        "max_effects_available": max(a["max_effects_available"], b["max_effects_available"]),
        "max_placebo_available": max(a["max_placebo_available"], b["max_placebo_available"]),
        "fg_trunc_max": fg_tm,
    }


def partition_preprocess_global(pdf, metadata, col_config, config_flags):
    """Weight normalization, filtering, panel balancing, trends-lin, remaining data preparation, and sorting.

    Uses broadcast metadata from the driver to apply cross-partition transformations.
    Accepts pandas or polars, always returns polars.
    """
    if len(pdf) == 0:
        return pdf

    df = pl.from_pandas(pdf) if not isinstance(pdf, pl.DataFrame) else pdf
    gname = col_config["gname"]
    tname = col_config["tname"]
    yname = col_config["yname"]
    dname = col_config["dname"]
    trends_nonparam = config_flags.get("trends_nonparam")

    global_weight_mean = metadata["weight_sum"] / max(metadata["weight_count"], 1)
    if global_weight_mean > 0:
        df = df.with_columns((pl.col("weight_gt") / global_weight_mean).alias("weight_gt"))

    group_cols = ["d_sq"]
    if trends_nonparam:
        group_cols.extend(trends_nonparam)

    valid_groups = {k for k, vals in metadata["fg_variation"].items() if len(vals) > 1}
    if valid_groups:
        df = df.with_columns(
            pl.when(pl.col("F_g") == float("inf")).then(0.0).otherwise(pl.col("F_g")).alias("_F_g_for_std")
        )
        df = df.with_columns(pl.col("_F_g_for_std").std().over(group_cols).round(3).alias("_var_F_g"))
        df = df.filter(pl.col("_var_F_g") > 0)
        df = df.drop(["_var_F_g", "_F_g_for_std"])

    if len(df) == 0:
        return df

    df = df.with_columns(pl.when(pl.col("F_g") == float("inf")).then(1).otherwise(0).alias("_never_change_d"))
    ctrl_group = [tname, "d_sq"]
    if trends_nonparam:
        ctrl_group.extend(trends_nonparam)
    df = df.with_columns(pl.col("_never_change_d").max().over(ctrl_group).alias("_controls_time"))
    df = df.filter(pl.col("_controls_time") > 0)
    df = df.drop(["_never_change_d", "_controls_time"])

    if len(df) == 0:
        return df

    continuous = config_flags.get("continuous", 0)
    if continuous > 0:
        df = df.with_columns(pl.col("d_sq").alias("d_sq_orig"))
        for p in range(1, continuous + 1):
            df = df.with_columns((pl.col("d_sq_orig") ** p).alias(f"d_sq_{p}"))
        df = df.with_columns(pl.lit(0).alias("d_sq"))
        mapping_df = (
            df.select("d_sq")
            .filter(pl.col("d_sq").is_not_null())
            .unique()
            .sort("d_sq")
            .with_row_index("d_sq_int", offset=1)
            .select(["d_sq", "d_sq_int"])
        )
        df = df.join(mapping_df, on="d_sq", how="left")

    global_times = sorted(metadata["unique_times"])
    groups = df.select(gname).unique()
    times = pl.DataFrame({tname: global_times})
    full_index = groups.join(times, how="cross")
    df = full_index.join(df, on=[gname, tname], how="left")

    time_invariant_cols = ["F_g", "d_sq", "S_g", "L_g"]
    for col in time_invariant_cols:
        if col in df.columns:
            df = df.with_columns(pl.col(col).mean().over(gname).alias(col))

    df = df.with_columns(
        pl.when(pl.col("F_g") == float("inf"))
        .then(pl.col("t_max_by_group") + 1)
        .otherwise(pl.col("F_g"))
        .alias("_F_g_trunc")
    )
    df = df.with_columns((pl.col("_F_g_trunc").max().over(group_cols) - 1).alias("T_g"))
    df = df.drop("_F_g_trunc")

    if config_flags.get("trends_lin", False):
        t_min = metadata["t_min"]
        df = df.filter(pl.col("F_g") != t_min + 1)
        df = df.sort([gname, tname])
        df = df.with_columns((pl.col(yname) - pl.col(yname).shift(1).over(gname)).alias(yname))
        xformla = config_flags.get("xformla")
        if xformla and xformla != "~1":
            for ctrl in _extract_vars_from_formula(xformla):
                if ctrl in df.columns:
                    df = df.with_columns((pl.col(ctrl) - pl.col(ctrl).shift(1).over(gname)).alias(ctrl))
        df = df.filter(pl.col(tname) != t_min)

    first_obs = df.group_by(gname).agg(pl.col(tname).min().alias("_first_t"))
    df = df.join(first_obs, on=gname, how="left")
    df = df.with_columns((pl.col(tname) == pl.col("_first_t")).cast(pl.Int64).alias("first_obs_by_gp"))
    df = df.drop("_first_t")

    df = df.with_columns(pl.col("weight_gt").fill_null(0.0))
    df = df.with_columns(
        pl.when(pl.col(yname).is_null() | pl.col(dname).is_null())
        .then(0.0)
        .otherwise(pl.col("weight_gt"))
        .alias("weight_gt")
    )

    if "d_fg" in df.columns:
        df = df.with_columns(pl.col("d_fg").mean().over(gname).alias("d_fg"))
    if "t_max_by_group" in df.columns:
        df = df.with_columns(pl.col("t_max_by_group").max().over(gname).alias("t_max_by_group"))

    effects_n = config_flags.get("effects", 1)
    placebo_n = config_flags.get("placebo", 0)
    t_min = metadata["t_min"]

    if config_flags.get("same_switchers", False):
        df = df.with_columns((pl.col("L_g") >= effects_n).cast(pl.Float64).alias("same_switcher_valid"))
    if config_flags.get("same_switchers_pl", False) and placebo_n > 0:
        df = df.with_columns(((pl.col("F_g") - t_min) >= (placebo_n + 1)).cast(pl.Float64).alias("same_switcher_valid"))

    df = df.sort([gname, tname])

    return df


def cap_effects_placebo(config_flags, metadata):
    """Cap effects and placebo counts based on data availability from metadata."""
    max_eff = metadata["max_effects_available"]
    max_plac = metadata["max_placebo_available"]
    effects = config_flags["effects"]
    placebo = config_flags["placebo"]

    if effects > max_eff:
        warnings.warn(
            f"Requested effects={effects} but only {max_eff} "
            f"post-treatment periods available. Using effects={max_eff}.",
            UserWarning,
            stacklevel=4,
        )
        effects = max_eff
        config_flags["effects"] = effects

    if placebo > max_plac:
        warnings.warn(
            f"Requested placebo={placebo} but only {max_plac} "
            f"pre-treatment periods available. Using placebo={max_plac}.",
            UserWarning,
            stacklevel=4,
        )
        placebo = max_plac
        config_flags["placebo"] = placebo

    return effects, placebo


def validate_distributed(columns, col_config, config_flags):
    """Schema-level validation for distributed DataFrames."""
    errors = []
    col_set = set(columns)

    for col_type, col_name in {
        "yname": col_config["yname"],
        "tname": col_config["tname"],
        "gname": col_config["gname"],
        "dname": col_config["dname"],
    }.items():
        if col_name not in col_set:
            errors.append(f"{col_type} = '{col_name}' must be a column in the dataset")

    weightsname = config_flags.get("weightsname")
    if weightsname and weightsname not in col_set:
        errors.append(f"weightsname = '{weightsname}' must be a column in the dataset")

    cluster = col_config.get("cluster")
    if cluster and cluster not in col_set:
        errors.append(f"cluster = '{cluster}' must be a column in the dataset")

    xformla = config_flags.get("xformla")
    if xformla and xformla != "~1":
        for ctrl in _extract_vars_from_formula(xformla):
            if ctrl not in col_set:
                errors.append(f"xformla contains '{ctrl}' which is not in the dataset")

    if errors:
        raise ValueError("Validation failed:\n" + "\n".join(errors))


def _extract_vars_from_formula(formula):
    if not formula or formula == "~1":
        return []
    formula = formula.replace("~", "").strip()
    return [v.strip() for v in formula.split("+") if v.strip() and v.strip() != "1"]
