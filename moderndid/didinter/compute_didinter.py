"""Core computation for DIDInter estimation."""

import numpy as np
import polars as pl
from scipy import stats

from .results import ATEResult, DIDInterResult, EffectsResult, PlacebosResult
from .variance import compute_clustered_variance, compute_joint_test


def compute_effects(preprocessed):
    """Compute treatment effects.

    Parameters
    ----------
    preprocessed : DIDInterData
        Preprocessed data.

    Returns
    -------
    DIDInterResult
        Estimation results.
    """
    config = preprocessed.config
    data = preprocessed.data

    ci_level = config.ci_level
    alpha = 1 - ci_level / 100
    z_crit = stats.norm.ppf(1 - alpha / 2)

    n_groups = data[config.gname].n_unique()
    t_max = int(data[config.tname].max())

    df = _prepare_data(data, config)

    effects_results = _compute_did_effects(
        df=df,
        config=config,
        n_horizons=config.effects,
        n_groups=n_groups,
        t_max=t_max,
        horizon_type="effect",
    )

    placebos_results = None
    if config.placebo > 0:
        placebos_results = _compute_did_effects(
            df=df,
            config=config,
            n_horizons=config.placebo,
            n_groups=n_groups,
            t_max=t_max,
            horizon_type="placebo",
        )

    ate = _compute_ate(effects_results, z_crit) if effects_results else None

    effects_equal_test = None
    if config.effects_equal and config.effects > 1 and effects_results:
        effects_equal_test = _test_effects_equality(effects_results)

    placebo_joint_test = None
    if config.placebo > 1 and placebos_results is not None:
        placebo_joint_test = compute_joint_test(
            placebos_results["estimates"],
            placebos_results["vcov"],
        )

    effects = EffectsResult(
        horizons=effects_results["horizons"],
        estimates=effects_results["estimates"],
        std_errors=effects_results["std_errors"],
        ci_lower=effects_results["estimates"] - z_crit * effects_results["std_errors"],
        ci_upper=effects_results["estimates"] + z_crit * effects_results["std_errors"],
        n_switchers=effects_results["n_switchers"],
        n_observations=effects_results["n_observations"],
    )

    placebos = None
    if placebos_results is not None:
        placebos = PlacebosResult(
            horizons=placebos_results["horizons"],
            estimates=placebos_results["estimates"],
            std_errors=placebos_results["std_errors"],
            ci_lower=placebos_results["estimates"] - z_crit * placebos_results["std_errors"],
            ci_upper=placebos_results["estimates"] + z_crit * placebos_results["std_errors"],
            n_switchers=placebos_results["n_switchers"],
            n_observations=placebos_results["n_observations"],
        )

    return DIDInterResult(
        effects=effects,
        placebos=placebos,
        ate=ate,
        n_units=preprocessed.n_switchers + preprocessed.n_never_switchers,
        n_switchers=preprocessed.n_switchers,
        n_never_switchers=preprocessed.n_never_switchers,
        ci_level=ci_level,
        effects_equal_test=effects_equal_test,
        placebo_joint_test=placebo_joint_test,
        influence_effects=effects_results.get("influence_func"),
        influence_placebos=placebos_results.get("influence_func") if placebos_results else None,
        estimation_params={
            "effects": config.effects,
            "placebo": config.placebo,
            "normalized": config.normalized,
            "switchers": config.switchers,
            "controls": config.controls,
        },
    )


def _prepare_data(data, config):
    """Prepare data for computation."""
    df = data.clone()
    gname = config.gname
    tname = config.tname

    df = df.sort([gname, tname])

    df = df.with_columns(pl.lit(1.0).alias("weight_gt"))

    if config.weightsname:
        df = df.with_columns((pl.col("weight_gt") * pl.col(config.weightsname)).alias("weight_gt"))

    first_obs = df.group_by(gname).agg(pl.col(tname).min().alias("_first_t")).select([gname, "_first_t"])
    df = df.join(first_obs, on=gname, how="left")
    df = df.with_columns((pl.col(tname) == pl.col("_first_t")).cast(pl.Int64).alias("first_obs_by_gp"))
    df = df.drop("_first_t")

    t_max_by_group = df.group_by(gname).agg(pl.col(tname).max().alias("t_max_by_group"))
    df = df.join(t_max_by_group, on=gname, how="left")

    return df


def _compute_did_effects(df, config, n_horizons, n_groups, t_max, horizon_type):
    """Compute effects at multiple horizons."""
    gname = config.gname
    tname = config.tname
    yname = config.yname

    if horizon_type == "effect":
        horizons = np.arange(1, n_horizons + 1)
    else:
        horizons = -np.arange(1, n_horizons + 1)

    estimates = np.zeros(n_horizons)
    std_errors = np.zeros(n_horizons)
    n_switchers_arr = np.zeros(n_horizons)
    n_obs_arr = np.zeros(n_horizons)
    influence_funcs = []

    for idx, h in enumerate(horizons):
        abs_h = abs(h)

        df = df.sort([gname, tname])
        diff_col = f"diff_y_{abs_h}"

        if horizon_type == "effect":
            df = df.with_columns(pl.col(yname).diff(abs_h).over(gname).alias(diff_col))
            dist_col = f"dist_to_switch_{abs_h}"
            df = df.with_columns(
                pl.when(pl.col(tname) == pl.col("F_g") + abs_h - 1).then(1.0).otherwise(0.0).alias(dist_col)
            )
        else:
            df = df.with_columns(
                (pl.col(yname).shift(2 * abs_h).over(gname) - pl.col(yname).shift(abs_h).over(gname)).alias(diff_col)
            )
            dist_col = f"dist_to_switch_pl_{abs_h}"
            df = df.with_columns(
                pl.when(pl.col(tname) == pl.col("F_g") + abs_h - 1).then(1.0).otherwise(0.0).alias(dist_col)
            )

        never_col = f"never_change_{abs_h}"
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

        never_w_col = f"never_change_w_{abs_h}"
        df = df.with_columns((pl.col(never_col) * pl.col("weight_gt")).alias(never_w_col))

        group_vars = [tname, "d_sq"]
        n_control_col = f"n_control_{abs_h}"
        df = df.with_columns(pl.col(never_w_col).sum().over(group_vars).alias(n_control_col))

        if config.switchers == "in":
            switcher_mask = pl.col("S_g") == 1
        elif config.switchers == "out":
            switcher_mask = pl.col("S_g") == -1
        else:
            switcher_mask = pl.col("S_g") != 0

        dist_w_col = f"dist_to_switch_w_{abs_h}"
        df = df.with_columns(
            pl.when(switcher_mask).then(pl.col(dist_col) * pl.col("weight_gt")).otherwise(0.0).alias(dist_w_col)
        )

        n_treated_col = f"n_treated_{abs_h}"
        df = df.with_columns(pl.col(dist_w_col).sum().over(group_vars).alias(n_treated_col))

        n_switchers = df.filter(pl.col(dist_col) == 1.0)[gname].n_unique()

        if n_switchers == 0:
            estimates[idx] = np.nan
            std_errors[idx] = np.nan
            n_switchers_arr[idx] = 0
            n_obs_arr[idx] = 0
            continue

        inf_temp_col = f"inf_func_{abs_h}_temp"
        n_control_is_zero = pl.col(n_control_col).is_null() | (pl.col(n_control_col) == 0)
        safe_n_control = pl.when(n_control_is_zero).then(1.0).otherwise(pl.col(n_control_col))
        df = df.with_columns(
            (
                (pl.lit(n_groups) / pl.lit(n_switchers))
                * pl.col("weight_gt")
                * (pl.col(dist_col) - (pl.col(n_treated_col) / safe_n_control) * pl.col(never_col).fill_null(0.0))
                * pl.col(diff_col).fill_null(0.0)
            ).alias(inf_temp_col)
        )

        inf_col = f"inf_func_{abs_h}"
        df = df.with_columns((pl.col(inf_temp_col).sum().over(gname) * pl.col("first_obs_by_gp")).alias(inf_col))

        did_estimate = df.select(pl.col(inf_col).sum()).item() / n_groups

        if config.normalized:
            delta_d = _compute_delta_d(df, config, abs_h, horizon_type)
            if delta_d is not None and delta_d != 0:
                did_estimate = did_estimate / delta_d

        estimates[idx] = did_estimate

        inf_func = df.filter(pl.col("first_obs_by_gp") == 1).select(inf_col).to_numpy().flatten()
        influence_funcs.append(inf_func)

        if config.cluster:
            cluster_ids = df.filter(pl.col("first_obs_by_gp") == 1).select(config.cluster).to_numpy().flatten()
            std_error = compute_clustered_variance(inf_func, cluster_ids, n_groups)
        else:
            std_error = np.sqrt(np.var(inf_func, ddof=1) / n_groups)

        std_errors[idx] = std_error
        n_switchers_arr[idx] = n_switchers

        count_col = f"count_{abs_h}"
        df = df.with_columns(
            pl.when(
                (pl.col(inf_temp_col).is_not_null() & (pl.col(inf_temp_col) != 0))
                | ((pl.col(inf_temp_col) == 0) & (pl.col(diff_col) == 0))
            )
            .then(1)
            .otherwise(0)
            .alias(count_col)
        )
        n_obs_arr[idx] = df.select(pl.col(count_col).sum()).item()

    vcov = None
    if len(influence_funcs) == n_horizons and all(len(f) > 0 for f in influence_funcs):
        influence_matrix = np.column_stack(influence_funcs)
        vcov = np.cov(influence_matrix, rowvar=False) / n_groups

    return {
        "horizons": horizons.astype(float),
        "estimates": estimates,
        "std_errors": std_errors,
        "n_switchers": n_switchers_arr,
        "n_observations": n_obs_arr,
        "influence_func": np.column_stack(influence_funcs) if influence_funcs else None,
        "vcov": vcov,
    }


def _compute_delta_d(df, config, horizon, horizon_type):
    """Compute treatment intensity change."""
    gname = config.gname
    tname = config.tname
    dname = config.dname

    switchers = df.filter(pl.col("F_g") != float("inf"))
    if len(switchers) == 0:
        return None

    if horizon_type == "effect":
        target_time = pl.col("F_g") + horizon - 1
    else:
        target_time = pl.col("F_g") - horizon - 1

    treat_at_target = (
        switchers.filter(pl.col(tname) == target_time).select([gname, pl.col(dname).alias("treat_target")]).unique()
    )

    treat_at_base = (
        switchers.filter(pl.col(tname) == pl.col("F_g") - 1).select([gname, pl.col(dname).alias("treat_base")]).unique()
    )

    merged = treat_at_target.join(treat_at_base, on=gname, how="inner")
    if len(merged) == 0:
        return None

    delta_d = (merged["treat_target"] - merged["treat_base"]).mean()
    return delta_d


def _compute_ate(effects_results, z_crit):
    """Compute average total effect."""
    estimates = effects_results["estimates"]
    valid_mask = ~np.isnan(estimates)

    if not np.any(valid_mask):
        return None

    ate_estimate = np.nanmean(estimates)

    if effects_results.get("vcov") is not None:
        vcov = effects_results["vcov"]
        n_effects = np.sum(valid_mask)
        weights = np.zeros(len(estimates))
        weights[valid_mask] = 1.0 / n_effects
        ate_var = weights @ vcov @ weights
        ate_se = np.sqrt(ate_var) if ate_var > 0 else np.nan
    else:
        std_errors = effects_results["std_errors"]
        valid_se = std_errors[valid_mask]
        ate_se = np.sqrt(np.mean(valid_se**2)) if len(valid_se) > 0 else np.nan

    return ATEResult(
        estimate=ate_estimate,
        std_error=ate_se,
        ci_lower=ate_estimate - z_crit * ate_se,
        ci_upper=ate_estimate + z_crit * ate_se,
    )


def _test_effects_equality(effects_results):
    """Test whether all effects are equal."""
    estimates = effects_results["estimates"]
    vcov = effects_results.get("vcov")

    valid_mask = ~np.isnan(estimates)
    if vcov is None or np.sum(valid_mask) < 2:
        return None

    valid_estimates = estimates[valid_mask]
    valid_vcov = vcov[np.ix_(valid_mask, valid_mask)]

    n_valid = len(valid_estimates)
    contrast_matrix = np.zeros((n_valid - 1, n_valid))
    for i in range(n_valid - 1):
        contrast_matrix[i, i] = 1
        contrast_matrix[i, i + 1] = -1

    contrast_diff = contrast_matrix @ valid_estimates
    contrast_vcov = contrast_matrix @ valid_vcov @ contrast_matrix.T

    try:
        chi2_stat = float(contrast_diff @ np.linalg.solve(contrast_vcov, contrast_diff))
        df = n_valid - 1
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)
        return {"chi2_stat": chi2_stat, "df": df, "p_value": p_value}
    except np.linalg.LinAlgError:
        return None
