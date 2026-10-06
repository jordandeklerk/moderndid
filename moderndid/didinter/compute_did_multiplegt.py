"""Core computations for estimation in heterogeneous and dynamic ATT estimation."""

import warnings
from dataclasses import replace

import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy import stats

from moderndid.core.preprocess.transformers import DataTransformerPipeline, DIDInterConfigUpdater

from .bootstrap import cluster_bootstrap
from .container import ATEResult, DIDInterResult, EffectsResult, HeterogeneityResult, PlacebosResult
from .controls import apply_control_adjustment, compute_control_coefficients, compute_variance_adjustment
from .variance import (
    build_treatment_paths,
    compute_cluster_influence,
    compute_clustered_variance,
    compute_cohort_dof,
    compute_control_dof,
    compute_dof_scaling,
    compute_e_hat,
    compute_joint_test,
    compute_path_cohort_dof,
    compute_union_dof,
)


def compute_did_multiplegt(preprocessed, data):
    """Compute treatment effects.

    Parameters
    ----------
    preprocessed : DIDInterData
        Preprocessed data.
    data : DataFrame
        The panel before preprocessing that the bootstrap resamples.

    Returns
    -------
    DIDInterResult
        Estimation results.
    """
    config = preprocessed.config
    df = preprocessed.data

    ci_level = config.ci_level
    alpha = 1 - ci_level / 100
    z_crit = stats.norm.ppf(1 - alpha / 2)

    n_groups = df[config.gname].n_unique()
    t_max = int(df[config.tname].max())

    df, coefficients = compute_control_coefficients(df, config, n_groups)

    # Since a group's first row carries its influence function, clustering needs the cluster of each row the
    # balancer inserted. The heterogeneity regressions read the column as observed.
    het_df = df
    if config.cluster:
        cluster = pl.col(config.cluster)
        df = df.with_columns(cluster.fill_null(cluster.min().over(config.gname)))
        if df.select(cluster.n_unique().over(config.gname).max()).item() > 1:
            raise ValueError(
                f"Some groups belong to more than one cluster in '{config.cluster}'. Each group in "
                f"'{config.gname}' must be nested within a single cluster."
            )

    effects_results = _compute_did_effects(
        df=df,
        config=config,
        n_horizons=config.effects,
        n_groups=n_groups,
        t_max=t_max,
        horizon_type="effect",
        coefficients=coefficients,
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
            coefficients=coefficients,
        )

    if config.boot:
        warnings.warn(
            "did_multiplegt computes analytical standard errors by default. "
            "Bootstrapping is slower and recommended when using a continuous treatment.",
            UserWarning,
            stacklevel=4,
        )

        boot_result = cluster_bootstrap(
            data=data,
            config=config,
            compute_func=_compute_bootstrap_estimates,
            biters=config.biters,
            random_state=config.random_state,
        )
        effects_results["std_errors"] = boot_result.effects_se

        if placebos_results is not None and boot_result.placebos_se is not None:
            placebos_results["std_errors"] = boot_result.placebos_se

    ate = None
    if effects_results and not config.trends_lin:
        ate = _compute_ate(effects_results, z_crit, n_groups)

    if ate is not None and config.boot:
        ate = ate._replace(
            std_error=boot_result.ate_se,
            ci_lower=ate.estimate - z_crit * boot_result.ate_se,
            ci_upper=ate.estimate + z_crit * boot_result.ate_se,
        )

    vcov_warnings = []

    effects_equal_test = None
    if config.effects_equal and config.effects > 1 and effects_results:
        effects_equal_test = _test_effects_equality(effects_results, config=config)
        if effects_equal_test and effects_equal_test.get("warnings"):
            vcov_warnings.extend(effects_equal_test["warnings"])

    placebo_joint_test = None
    if config.placebo > 1 and placebos_results is not None:
        placebo_joint_test = compute_joint_test(
            placebos_results["estimates"],
            placebos_results["vcov"],
        )
        if placebo_joint_test and placebo_joint_test.get("warnings"):
            vcov_warnings.extend(placebo_joint_test["warnings"])

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

    heterogeneity = _compute_heterogeneity(het_df, config)

    # Since groups that switch in the direction left out are controls until they switch, they count as units
    # but not as switchers.
    groups = preprocessed.time_invariant_data
    switches = (groups["F_g"] != float("inf")) & groups["S_g"].is_in(DIDInterConfigUpdater.switcher_directions(config))

    return DIDInterResult(
        effects=effects,
        placebos=placebos,
        ate=ate,
        n_units=groups.height,
        n_switchers=int(switches.sum()),
        n_never_switchers=preprocessed.n_never_switchers,
        ci_level=ci_level,
        effects_equal_test=effects_equal_test,
        placebo_joint_test=placebo_joint_test,
        influence_effects=effects_results.get("influence_func"),
        influence_placebos=placebos_results.get("influence_func") if placebos_results else None,
        heterogeneity=heterogeneity,
        estimation_params={
            "yname": config.yname,
            "effects": config.effects,
            "placebo": config.placebo,
            "normalized": config.normalized,
            "switchers": config.switchers,
            "xformla": config.xformla,
            "cluster": config.cluster,
            "trends_lin": config.trends_lin,
            "trends_nonparam": config.trends_nonparam,
            "only_never_switchers": config.only_never_switchers,
            "same_switchers": config.same_switchers,
            "same_switchers_pl": config.same_switchers_pl,
            "continuous": config.continuous,
            "weightsname": config.weightsname,
            "boot": config.boot,
        },
        vcov_warnings=vcov_warnings,
    )


def _compute_bootstrap_estimates(df, config):
    """Preprocess one bootstrap draw and compute its point estimates."""
    nan_result = {"effects": np.full(config.effects, np.nan)}
    if config.placebo > 0:
        nan_result["placebos"] = np.full(config.placebo, np.nan)

    # Since the full sample passed the weight check, a draw can fail it only when none of its weights is positive.
    # Like a draw without switchers, such a draw has no estimates and drops out of the standard errors.
    if config.weightsname is not None and not (df[config.weightsname].cast(pl.Float64) > 0).any():
        return nan_result

    # Since point estimates never use the cluster, the draws skip the clustered variance computations.
    config = replace(config, cluster=None)
    for step in DataTransformerPipeline.get_didinter_pipeline().transformers:
        if df.is_empty():
            return nan_result
        df = step.transform(df, config)

    max_effects, max_placebo = DIDInterConfigUpdater.estimable_horizons(df, config)
    if max_effects == 0:
        return nan_result

    t_max = int(df[config.tname].max())
    # A draw estimates the horizons its own switchers reach, as the full sample does.
    config = replace(config, effects=min(config.effects, max_effects), placebo=min(config.placebo, max_placebo))
    n_groups = df[config.gname].n_unique()

    df, coefficients = compute_control_coefficients(df, config, n_groups)

    effects_results = _compute_did_effects(
        df=df,
        config=config,
        n_horizons=config.effects,
        n_groups=n_groups,
        t_max=t_max,
        horizon_type="effect",
        coefficients=coefficients,
    )

    result = {"effects": effects_results["estimates"]}

    if config.placebo > 0:
        placebos_results = _compute_did_effects(
            df=df,
            config=config,
            n_horizons=config.placebo,
            n_groups=n_groups,
            t_max=t_max,
            horizon_type="placebo",
            coefficients=coefficients,
        )
        result["placebos"] = placebos_results["estimates"]

    if not config.trends_lin:
        ci_level = config.ci_level
        alpha = 1 - ci_level / 100
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ate_result = _compute_ate(effects_results, z_crit, n_groups)
        if ate_result is not None:
            result["ate"] = ate_result.estimate

    return result


def _compute_did_effects(df, config, n_horizons, n_groups, t_max, horizon_type, coefficients):
    r"""Compute effects at multiple horizons.

    Switchers whose treatment rises and switchers whose treatment falls are estimated in separate
    passes against the same not-yet-switched controls. The passes combine in proportion to their
    weighted numbers of switchers. Since a fall enters with a minus sign, both passes measure the
    effect of moving away from the baseline treatment.

    With ``same_switchers`` every horizon uses only the switchers that reach all the requested
    effects. With ``same_switchers_pl`` the placebos also need all the requested placebos. With
    ``trends_lin`` the estimate at horizon :math:`\ell` sums the estimates of horizons 1 to
    :math:`\ell`. All of them come from the switchers that reach horizon :math:`\ell`.
    """
    gname = config.gname
    tname = config.tname

    horizons = np.arange(1, n_horizons + 1) if horizon_type == "effect" else -np.arange(1, n_horizons + 1)

    estimates = np.zeros(n_horizons)
    estimates_unnorm = np.zeros(n_horizons)
    std_errors = np.zeros(n_horizons)
    n_switchers_arr = np.zeros(n_horizons)
    n_switchers_weighted_arr = np.zeros(n_horizons)
    ate_delta_arr = np.zeros(n_horizons)
    n_obs_arr = np.zeros(n_horizons)
    influence_funcs = []
    influence_funcs_unnorm = []

    df = df.sort([gname, tname])
    unit_rows = df.filter(pl.col("first_obs_by_gp") == 1)
    unit_clusters = unit_rows[config.cluster] if config.cluster else None
    directions = [d for d in DIDInterConfigUpdater.switcher_directions(config) if (df["S_g"] == d).any()]

    use_placebo_mask = horizon_type == "placebo" and (config.same_switchers_pl or config.trends_lin)
    same_switchers = None
    if config.same_switchers or config.trends_lin:
        same_switchers = pl.col("_same_switcher")
        if use_placebo_mask:
            same_switchers = same_switchers & pl.col("_same_switcher_pl")
    if config.same_switchers and not config.trends_lin:
        df = _compute_same_switchers_mask(df, config, config.effects, config.placebo if use_placebo_mask else 0, t_max)

    for idx, h in enumerate(horizons):
        abs_h = abs(h)

        if config.trends_lin:
            df = _compute_same_switchers_mask(df, config, abs_h, abs_h if use_placebo_mask else 0, t_max)
            df, passes = _estimate_cumulated_horizon(
                df, config, abs_h, horizon_type, directions, n_groups, t_max, coefficients, same_switchers
            )
        else:
            df, passes = _estimate_horizon(
                df, config, abs_h, horizon_type, directions, n_groups, t_max, coefficients, same_switchers
            )

        if not passes:
            estimates[idx] = np.nan
            std_errors[idx] = np.nan
            n_switchers_arr[idx] = 0
            n_obs_arr[idx] = 0
            # A zero column keeps the influence functions aligned with the horizons for the average total effect.
            influence_funcs_unnorm.append(np.zeros(unit_rows.height))
            continue

        n_switchers_weighted = sum(result["n_weighted"] for result in passes)
        shares = [result["n_weighted"] / n_switchers_weighted for result in passes]
        did_estimate = sum(s * result["sign"] * result["estimate"] for s, result in zip(shares, passes, strict=True))
        inf_func = sum(s * result["sign"] * result["influence"] for s, result in zip(shares, passes, strict=True))
        deltas = [s * result["delta"] for s, result in zip(shares, passes, strict=True) if result["delta"] is not None]
        delta_d = sum(deltas) if deltas else None

        estimates_unnorm[idx] = did_estimate
        n_switchers_weighted_arr[idx] = n_switchers_weighted

        if horizon_type == "effect":
            ate_delta_arr[idx] = sum(s * result["ate_delta"] for s, result in zip(shares, passes, strict=True))

        if config.normalized and delta_d is not None and delta_d != 0:
            did_estimate = did_estimate / delta_d

        estimates[idx] = did_estimate

        if config.cluster:
            observed = unit_clusters.is_not_null()
            std_error = compute_clustered_variance(
                inf_func[observed.to_numpy()], unit_clusters.filter(observed).to_numpy(), n_groups
            )
        else:
            std_error = np.sqrt(np.sum(inf_func**2)) / n_groups

        influence_funcs_unnorm.append(inf_func.copy())

        if config.normalized and delta_d is not None and delta_d != 0:
            std_error = std_error / delta_d
            inf_func = inf_func / delta_d

        influence_funcs.append(inf_func)
        std_errors[idx] = std_error
        n_switchers_arr[idx] = sum(result["n_switchers"] for result in passes)
        n_obs_arr[idx] = df.select(pl.col(f"count_{abs_h}").sum()).item()

    # The covariances use the same uncentered cluster sums as the standard errors on the diagonal.
    vcov = None
    if len(influence_funcs) == n_horizons and all(len(f) > 0 for f in influence_funcs):
        cluster_sums = compute_cluster_influence(np.column_stack(influence_funcs), unit_clusters)
        vcov = cluster_sums.T @ cluster_sums / n_groups**2

    return {
        "horizons": horizons.astype(float),
        "estimates": estimates,
        "estimates_unnorm": estimates_unnorm,
        "std_errors": std_errors,
        "n_switchers": n_switchers_arr,
        "n_switchers_weighted": n_switchers_weighted_arr,
        "ate_delta": ate_delta_arr,
        "n_observations": n_obs_arr,
        "influence_func": np.column_stack(influence_funcs) if influence_funcs else None,
        "influence_func_unnorm": np.column_stack(influence_funcs_unnorm) if influence_funcs_unnorm else None,
        "unit_clusters": unit_clusters,
        "vcov": vcov,
        "df": df,
    }


def _estimate_horizon(df, config, horizon, horizon_type, directions, n_groups, t_max, coefficients, same_switchers):
    """Estimate one horizon direction by direction.

    Parameters
    ----------
    df : polars.DataFrame
        Preprocessed panel sorted by group and period.
    config : DIDInterConfig
        Configuration object.
    horizon : int
        Number of periods from the last period before the switch, forward for an effect and backward
        for a placebo.
    horizon_type : {"effect", "placebo"}
        Whether the horizon is an effect or a placebo.
    directions : list of int
        Values of ``S_g`` that get a pass.
    n_groups : int
        Number of groups in the sample.
    t_max : int
        Last period of the sample.
    coefficients : dict
        Coefficients of the control variables for each baseline treatment, empty without controls.
    same_switchers : polars.Expr or None
        Flags the switchers that the horizon may use. None lets every switcher in.

    Returns
    -------
    df : polars.DataFrame
        The panel with the columns of the horizon.
    passes : list of dict
        The results of :func:`_compute_direction_pass` for the directions with switchers at the horizon.
    """
    gname = config.gname
    tname = config.tname
    yname = config.yname
    diff_col = f"diff_y_{horizon}"

    if horizon_type == "effect":
        df = df.with_columns(pl.col(yname).diff(horizon).over(gname).alias(diff_col))
    else:
        df = df.with_columns(
            (pl.col(yname).shift(2 * horizon).over(gname) - pl.col(yname).shift(horizon).over(gname)).alias(diff_col)
        )

    df = build_treatment_paths(df, horizon, config)

    never_col = f"never_change_{horizon}"
    df = df.with_columns(
        pl.when(pl.col(diff_col).is_not_null())
        .then((pl.col("F_g") > pl.col(tname)).cast(pl.Float64))
        .otherwise(pl.lit(None))
        .alias(never_col)
    )

    if config.only_never_switchers:
        df = df.with_columns(
            pl.when((pl.col("F_g") > pl.col(tname)) & (pl.col("F_g") < (t_max + 1)) & pl.col(diff_col).is_not_null())
            .then(0.0)
            .otherwise(pl.col(never_col))
            .alias(never_col)
        )

    never_w_col = f"never_change_w_{horizon}"
    df = df.with_columns((pl.col(never_col) * pl.col("weight_gt")).alias(never_w_col))
    df = df.with_columns(pl.col(never_w_col).sum().over(_get_group_vars(config)).alias(f"n_control_{horizon}"))

    # Since the adjustment leaves missing outcome differences missing, the flags of each pass do not depend on it.
    if coefficients:
        df = apply_control_adjustment(df, config, horizon, coefficients, horizon_type)

    df = df.with_columns(pl.lit(0, dtype=pl.Int32).alias(f"count_{horizon}"))
    passes = []
    for direction in directions:
        df, result = _compute_direction_pass(
            df, config, horizon, horizon_type, direction, n_groups, coefficients, same_switchers
        )
        if result is not None:
            passes.append(result)

    return df, passes


def _estimate_cumulated_horizon(
    df, config, horizon, horizon_type, directions, n_groups, t_max, coefficients, same_switchers
):
    """Sum the estimates of the horizons up to one horizon for each direction.

    Since ``trends_lin`` first-differences the outcome, the estimate at a horizon sums the estimates
    of the horizons up to it. All of them use the switchers that ``same_switchers`` flags. A direction
    enters only when it has switchers at every horizon of the sum.

    Parameters
    ----------
    df : polars.DataFrame
        Preprocessed panel sorted by group and period.
    config : DIDInterConfig
        Configuration object.
    horizon : int
        Last horizon of the sum.
    horizon_type : {"effect", "placebo"}
        Whether the horizons are effects or placebos.
    directions : list of int
        Values of ``S_g`` that get a pass.
    n_groups : int
        Number of groups in the sample.
    t_max : int
        Last period of the sample.
    coefficients : dict
        Coefficients of the control variables for each baseline treatment, empty without controls.
    same_switchers : polars.Expr
        Flags the switchers that reach the last horizon.

    Returns
    -------
    df : polars.DataFrame
        The panel with the columns of the last horizon.
    passes : list of dict
        One dict per direction in the form of :func:`_compute_direction_pass`. The estimate and the
        influence function are sums over the horizons. The other fields come from the last horizon.
    """
    sums = {}
    for step in range(1, horizon + 1):
        df, passes = _estimate_horizon(
            df, config, step, horizon_type, directions, n_groups, t_max, coefficients, same_switchers
        )
        for result in passes:
            total = sums.setdefault(result["sign"], {"estimate": 0.0, "influence": 0.0, "steps": 0})
            total["estimate"] += result["estimate"]
            total["influence"] = total["influence"] + result["influence"]
            total["steps"] += 1
            if step == horizon:
                total.update({key: result[key] for key in ("sign", "n_switchers", "n_weighted", "delta", "ate_delta")})

    return df, [sums[d] for d in directions if d in sums and sums[d]["steps"] == horizon]


def _compute_direction_pass(df, config, horizon, horizon_type, direction, n_groups, coefficients, same_switchers=None):
    """Estimate one horizon from the switchers whose treatment moves in one direction.

    The pass also marks the cells it uses in the horizon's count column.

    Parameters
    ----------
    df : polars.DataFrame
        Preprocessed panel with the outcome differences and control flags of the horizon.
    config : DIDInterConfig
        Configuration object.
    horizon : int
        Number of periods from the last period before the switch, forward for an effect and backward
        for a placebo.
    horizon_type : {"effect", "placebo"}
        Whether the horizon is an effect or a placebo.
    direction : {1, -1}
        Value of ``S_g`` for the switchers of this pass.
    n_groups : int
        Number of groups in the sample.
    coefficients : dict
        Coefficients of the control variables for each baseline treatment, empty without controls.
    same_switchers : polars.Expr, optional
        Flags the switchers that the pass may use. None lets every switcher in.

    Returns
    -------
    df : polars.DataFrame
        The panel with the columns of this pass.
    result : dict or None
        None when no switcher in this direction reaches the horizon. Otherwise a dict with

        - **sign**: The direction of the pass
        - **n_switchers**: Number of switchers
        - **n_weighted**: Weighted number of switchers
        - **estimate**: Estimate from the switchers of this pass
        - **influence**: Influence function of the estimate, one entry per group
        - **delta**: Average treatment change that normalizes the estimate, or None
        - **ate_delta**: Average treatment change at the horizon for the average total effect, or None
          for a placebo
    """
    gname = config.gname
    tname = config.tname
    yname = config.yname
    dname = config.dname
    diff_col = f"diff_y_{horizon}"
    never_col = f"never_change_{horizon}"
    n_control_col = f"n_control_{horizon}"
    dist_col = f"dist_to_switch_{horizon}" if horizon_type == "effect" else f"dist_to_switch_pl_{horizon}"
    group_vars = _get_group_vars(config)

    switcher_mask = pl.col("S_g") == direction
    if same_switchers is not None:
        switcher_mask = switcher_mask & same_switchers

    base_cond = (
        (pl.col(tname) == (pl.col("F_g") - 1 + horizon))
        & (pl.col("L_g") >= horizon)
        & (pl.col(n_control_col) > 0)
        & pl.col(n_control_col).is_not_null()
    )

    # Since a placebo switcher must also count toward the effect at the same lag, its outcome at F_g - 1 + l
    # must be observed.
    if horizon_type == "placebo":
        base_cond = base_cond & pl.col(yname).is_not_null()

    if same_switchers is not None:
        base_cond = base_cond & same_switchers

    cond_expr = base_cond & (pl.col("S_g") == direction)
    df = df.with_columns(
        pl.when(pl.col(diff_col).is_null()).then(pl.lit(None)).otherwise(cond_expr.cast(pl.Float64)).alias(dist_col)
    )

    dist_w_col = f"dist_to_switch_w_{horizon}"
    df = df.with_columns(
        pl.when(switcher_mask).then(pl.col(dist_col) * pl.col("weight_gt")).otherwise(0.0).alias(dist_w_col)
    )

    n_treated_col = f"n_treated_{horizon}"
    df = df.with_columns(pl.col(dist_w_col).sum().over(group_vars).alias(n_treated_col))

    switcher_filter = (pl.col(dist_col) == 1.0) & pl.col(diff_col).is_not_null() & switcher_mask
    n_switchers_unweighted = df.filter(switcher_filter)[gname].n_unique()

    n_switchers_weighted = df.select(pl.col(dist_w_col).sum()).item()
    if n_switchers_weighted is None or n_switchers_weighted == 0:
        n_switchers_weighted = 0.0

    if n_switchers_unweighted == 0:
        return df, None

    inf_temp_col = f"inf_func_{horizon}_temp"
    n_control_is_zero = pl.col(n_control_col).is_null() | (pl.col(n_control_col) == 0)
    safe_n_control = pl.when(n_control_is_zero).then(1.0).otherwise(pl.col(n_control_col))
    safe_n_switchers = max(n_switchers_weighted, 1e-10)

    df = df.with_columns(
        (
            (pl.lit(n_groups) / pl.lit(safe_n_switchers))
            * pl.col("weight_gt")
            * (pl.col(dist_col) - (pl.col(n_treated_col) / safe_n_control) * pl.col(never_col).fill_null(0.0))
            * pl.col(diff_col).fill_null(0.0)
        ).alias(inf_temp_col)
    )

    inf_col = f"inf_func_{horizon}"
    df = df.with_columns((pl.col(inf_temp_col).sum().over(gname) * pl.col("first_obs_by_gp")).alias(inf_col))

    did_estimate = df.select(pl.col(inf_col).sum()).item() / n_groups
    delta_d = _compute_delta_d(df, config, horizon, horizon_type, dist_col)

    # The average total effect measures each switch by its treatment change at the horizon. S_g makes it positive.
    ate_delta = None
    if horizon_type == "effect":
        treat_col, base_col = (f"{dname}_orig", "d_sq_orig") if config.continuous > 0 else (dname, "d_sq")
        dose_change = pl.col("S_g") * (pl.col(treat_col) - pl.col(base_col))
        ate_delta = df.select((pl.col(dist_w_col) * dose_change).sum()).item() / safe_n_switchers

    if coefficients:
        df = compute_variance_adjustment(df, config, horizon, coefficients, safe_n_switchers, dist_col)

    switcher_flag = f"is_switcher_{horizon}"
    weighted_diff = f"weighted_diff_{horizon}"
    df = df.with_columns(
        pl.col(dist_col).cast(pl.Int64).alias(switcher_flag),
        (pl.col(diff_col).fill_null(0.0) * pl.col("weight_gt")).alias(weighted_diff),
    )

    # Placebos keep the default cohorts because path cohorts track treatment after the switch.
    if config.less_conservative_se and horizon_type == "effect":
        df = compute_path_cohort_dof(df, horizon, config)
    else:
        df = compute_cohort_dof(df, horizon, config, config.cluster)
    df = compute_control_dof(df, horizon, config, config.cluster)
    df = compute_union_dof(df, horizon, config, config.cluster)
    df = compute_dof_scaling(df, horizon, config)
    df = compute_e_hat(df, horizon, config)

    dof_scale_col = f"dof_scale_{horizon}"
    e_hat_col = f"E_hat_{horizon}"
    inf_var_col = f"inf_func_var_{horizon}"
    part2_col = f"part2_{horizon}"
    dof_scale_expr = pl.col(dof_scale_col).fill_null(1.0) if dof_scale_col in df.columns else pl.lit(1.0)
    dummy_u_gg_col = f"dummy_u_gg_{horizon}"
    time_constraint_col = f"time_constraint_{horizon}"

    df = df.with_columns(
        ((pl.col("T_g") - 1) >= horizon).cast(pl.Int64).alias(dummy_u_gg_col),
        ((pl.col(tname) >= (horizon + 1)) & (pl.col(tname) <= pl.col("T_g"))).cast(pl.Int64).alias(time_constraint_col),
    )

    if e_hat_col in df.columns:
        df = df.with_columns(
            (
                pl.col(dummy_u_gg_col)
                * (pl.lit(n_groups) / pl.lit(safe_n_switchers))
                * pl.col(time_constraint_col)
                * pl.col("weight_gt")
                * (pl.col(dist_col) - (pl.col(n_treated_col) / safe_n_control) * pl.col(never_col).fill_null(0.0))
                * dof_scale_expr
                * (pl.col(diff_col).fill_null(0.0) - pl.col(e_hat_col).fill_null(0.0))
            ).alias(inf_var_col)
        )
        df = df.with_columns((pl.col(inf_var_col).sum().over(gname) * pl.col("first_obs_by_gp")).alias(inf_var_col))

        if part2_col in df.columns:
            df = df.with_columns((pl.col(inf_var_col) - pl.col(part2_col).fill_null(0.0)).alias(inf_var_col))

        inf_func = df.filter(pl.col("first_obs_by_gp") == 1).select(inf_var_col).to_numpy().flatten()
    else:
        inf_func = df.filter(pl.col("first_obs_by_gp") == 1).select(inf_col).to_numpy().flatten()

        if part2_col in df.columns:
            part2_vals = df.filter(pl.col("first_obs_by_gp") == 1).select(part2_col).to_numpy().flatten()
            inf_func = inf_func - part2_vals

    counted = pl.when(
        (pl.col(inf_temp_col).is_not_null() & (pl.col(inf_temp_col) != 0))
        | ((pl.col(inf_temp_col) == 0) & (pl.col(diff_col) == 0))
    )
    count_col = f"count_{horizon}"
    # A cell that serves the switchers of both directions counts once.
    df = df.with_columns(pl.max_horizontal(pl.col(count_col), counted.then(1).otherwise(0)).alias(count_col))

    return df, {
        "sign": direction,
        "n_switchers": n_switchers_unweighted,
        "n_weighted": safe_n_switchers,
        "estimate": did_estimate,
        "influence": inf_func,
        "delta": delta_d,
        "ate_delta": ate_delta,
    }


def _run_het_regression(het_sample, covariates, horizon, config):
    """Run WLS regression for heterogeneity analysis at a given horizon.

    Uses HC2 standard errors by default. When ``predict_het_hc2bm`` is True on
    ``config``, uses HC2 clustered (Bell-McCaffrey) standard errors clustered
    by the ``cluster`` variable (or ``gname`` if no cluster is specified).
    """
    y = het_sample["_prod_het"].to_numpy()

    # Since the regression has one row per group, each group carries its user weight in its first period even when
    # the outcome is missing there. A group without a weight in that period drops out.
    weightsname = getattr(config, "weightsname", None)
    if weightsname is not None:
        weights = het_sample[weightsname].cast(pl.Float64).fill_null(0.0).to_numpy()
    else:
        weights = np.ones(len(y))

    X_cov_raw = het_sample.select(covariates).to_numpy()
    valid_mask = ~np.isnan(y) & np.all(np.isfinite(X_cov_raw), axis=1) & (weights > 0)
    if valid_mask.sum() < len(covariates) + 5:
        return None

    y = y[valid_mask]
    weights = weights[valid_mask]

    X_cov = X_cov_raw[valid_mask]

    interaction_cols = ["F_g", "d_sq", "S_g"]
    if config.trends_nonparam:
        interaction_cols.extend(c for c in config.trends_nonparam if c in het_sample.columns)

    fe_arrays = []
    for col in interaction_cols:
        if col in het_sample.columns:
            fe_arrays.append(het_sample[col].to_numpy()[valid_mask])

    X_parts = [np.ones((len(y), 1)), X_cov]
    if fe_arrays:
        stacked = np.column_stack(fe_arrays)
        _, inverse = np.unique(stacked, axis=0, return_inverse=True)
        n_groups = inverse.max() + 1
        if n_groups > 1:
            dummies = np.zeros((len(y), n_groups - 1))
            for i in range(1, n_groups):
                dummies[:, i - 1] = (inverse == i).astype(float)
            keep = dummies.std(axis=0) > 0
            dummies = dummies[:, keep]
            if dummies.shape[1] > 0:
                X_base = np.column_stack([np.ones((len(y), 1)), X_cov, dummies])
                _, R = np.linalg.qr(X_base, mode="reduced")
                n_base = 1 + X_cov.shape[1]
                tol = 1e-10 * np.abs(np.diag(R[:n_base, :n_base])).max()
                indep = np.abs(np.diag(R)[n_base:]) > tol
                dummies = dummies[:, indep]
            if dummies.shape[1] > 0:
                X_parts.append(dummies)

    X = np.column_stack(X_parts)

    use_hc2bm = getattr(config, "predict_het_hc2bm", False)

    if use_hc2bm:
        cluster_col = getattr(config, "cluster", None)
        if cluster_col and cluster_col in het_sample.columns:
            cluster_ids = het_sample[cluster_col].to_numpy()[valid_mask]
        else:
            cluster_col = None
            cluster_ids = None

        # Block BM only helps when clusters have >1 observation; otherwise
        # it degenerates to standard HC2.
        has_multi_obs_clusters = cluster_ids is not None and np.any(
            np.bincount(cluster_ids.astype(int) - cluster_ids.astype(int).min()) > 1
        )

        if not has_multi_obs_clusters:
            if cluster_col is None:
                warnings.warn(
                    "predict_het_hc2bm has no effect without an explicit "
                    "cluster variable. The heterogeneity sample has one row "
                    "per group, so Bell-McCaffrey clustering reduces to "
                    "standard HC2. Specify 'cluster' for multi-observation "
                    "clusters.",
                    UserWarning,
                    stacklevel=4,
                )
            else:
                warnings.warn(
                    f"predict_het_hc2bm has no effect because all clusters "
                    f"in '{cluster_col}' have a single observation in the "
                    "heterogeneity sample, so Bell-McCaffrey reduces to "
                    "standard HC2.",
                    UserWarning,
                    stacklevel=4,
                )
            use_hc2bm = False

    # HC2 divides each residual by the square root of one minus its leverage. A group that the regressors fit
    # exactly has leverage one and a zero residual. Its term in the variance is then 0/0.
    tolerance = np.sqrt(np.finfo(float).eps)
    if use_hc2bm:
        model_fit = sm.WLS(y, X, weights=weights).fit()

        try:
            XtWX_inv = np.linalg.inv((X * weights[:, None]).T @ X)
        except np.linalg.LinAlgError:
            XtWX_inv = np.linalg.pinv((X * weights[:, None]).T @ X)

        resid_wt = weights * model_fit.resid

        unique_clusters = np.unique(cluster_ids)
        k = X.shape[1]

        M = np.zeros((k, k))
        exact_fit = False
        for c in unique_clusters:
            ij = cluster_ids == c
            X_c = X[ij]
            sqrt_w = np.sqrt(weights[ij])
            res_c = resid_wt[ij]
            m_c = ij.sum()

            # I - X_c (X'WX)^-1 X_c' W_c is not symmetric when the weights vary within the cluster. Its inverse square
            # root conjugates that of the symmetric I - W_c^1/2 X_c (X'WX)^-1 X_c' W_c^1/2 by W_c^1/2.
            X_w = X_c * sqrt_w[:, None]
            eigvals, eigvecs = np.linalg.eigh(np.eye(m_c) - X_w @ XtWX_inv @ X_w.T)
            exact_fit = exact_fit or eigvals.min() < tolerance
            eigvals = np.maximum(eigvals, 1e-10)
            A_inv_sqrt = (eigvecs @ np.diag(eigvals ** (-0.5)) @ eigvecs.T) * (sqrt_w[None, :] / sqrt_w[:, None])

            res_adj = A_inv_sqrt @ res_c
            s_c = (res_adj[:, None] * X_c).sum(axis=0)
            M += np.outer(s_c, s_c)

        vcov_hc2bm = XtWX_inv @ M @ XtWX_inv

        model_fit._results.cov_params_default = vcov_hc2bm
        model = model_fit
    else:
        model = sm.WLS(y, X, weights=weights).fit(cov_type="HC2")
        weighted_X = X * np.sqrt(weights)[:, None]
        leverage = np.einsum("ij,ij->i", weighted_X @ np.linalg.pinv(weighted_X.T @ weighted_X), weighted_X)
        exact_fit = leverage.max() > 1 - tolerance

    n_cov = len(covariates)
    coef_indices = list(range(1, n_cov + 1))

    coefs = model.params[coef_indices]
    ses = np.sqrt(np.diag(model.cov_params()))[coef_indices] if use_hc2bm else model.bse[coef_indices]
    t_stats = coefs / ses

    t_crit = stats.t.ppf(0.975, model.df_resid)
    ci_lower = coefs - t_crit * ses
    ci_upper = coefs + t_crit * ses

    r_matrix = np.zeros((n_cov, len(model.params)))
    for i, idx in enumerate(coef_indices):
        r_matrix[i, idx] = 1

    if use_hc2bm:
        beta_r = r_matrix @ model.params
        v_r = r_matrix @ model.cov_params() @ r_matrix.T
        try:
            f_stat = float(beta_r @ np.linalg.solve(v_r, beta_r)) / n_cov
        except np.linalg.LinAlgError:
            f_stat = float(beta_r @ np.linalg.pinv(v_r) @ beta_r) / n_cov
        f_pvalue = stats.f.sf(f_stat, n_cov, model.df_resid)
    else:
        f_test = model.f_test(r_matrix)
        f_pvalue = float(f_test.pvalue)

    if exact_fit:
        ses = t_stats = ci_lower = ci_upper = np.full(n_cov, np.nan)
        f_pvalue = np.nan

    return HeterogeneityResult(
        horizon=horizon,
        covariates=covariates,
        estimates=np.array(coefs),
        std_errors=np.array(ses),
        t_stats=np.array(t_stats),
        ci_lower=np.array(ci_lower),
        ci_upper=np.array(ci_upper),
        n_obs=int(model.nobs),
        f_pvalue=f_pvalue,
    )


def _compute_delta_d(df, config, horizon, horizon_type, dist_col=None):
    """Compute cumulative treatment intensity change for normalization."""
    gname = config.gname
    tname = config.tname
    dname = config.dname

    treat_col, base_col = (f"{dname}_orig", "d_sq_orig") if config.continuous > 0 else (dname, "d_sq")

    if dist_col is None:
        dist_col = f"dist_to_switch_{horizon}" if horizon_type == "effect" else f"dist_to_switch_pl_{horizon}"

    switchers = df.filter(pl.col("F_g") != float("inf"))
    if len(switchers) == 0:
        return None

    # Delta_D uses post-switch periods for both effects and placebos
    time_start = pl.col("F_g")
    time_end = pl.col("F_g") - 1 + horizon

    mask = (pl.col(tname) >= time_start) & (pl.col(tname) <= time_end)

    switchers = switchers.with_columns(
        pl.when(mask).then(pl.col(treat_col) - pl.col(base_col)).otherwise(None).alias("_treat_diff_temp")
    )

    sum_by_unit = switchers.group_by(gname).agg(
        pl.col("_treat_diff_temp").sum().alias("sum_treat"),
        pl.col("S_g").first().alias("S_g"),
    )

    if dist_col in df.columns:
        valid_units = df.filter(pl.col(dist_col) == 1.0).select([gname, "weight_gt"]).unique()
        sum_by_unit = sum_by_unit.join(valid_units, on=gname, how="inner")
    else:
        sum_by_unit = sum_by_unit.with_columns(pl.lit(1.0).alias("weight_gt"))

    sum_by_unit = sum_by_unit.filter(pl.col("sum_treat").is_not_null())
    if len(sum_by_unit) == 0:
        return None

    total_weight = sum_by_unit["weight_gt"].sum()
    if total_weight == 0:
        return None

    sum_by_unit = sum_by_unit.with_columns(pl.when(pl.col("S_g") == 1).then(1).otherwise(0).alias("S_g_ind"))
    sum_by_unit = sum_by_unit.with_columns(
        (
            (pl.col("weight_gt") / total_weight)
            * (pl.col("S_g_ind") * pl.col("sum_treat") + (1 - pl.col("S_g_ind")) * (-pl.col("sum_treat")))
        ).alias("delta_contrib")
    )

    delta_d = sum_by_unit["delta_contrib"].sum()
    return delta_d


def _compute_ate(effects_results, z_crit, n_groups):
    r"""Compute average total effect.

    The effects are averaged with weights proportional to their weighted numbers of switchers. The
    average is divided by the same weighted average of the treatment changes. The change at horizon
    :math:`\ell` averages :math:`S_g (D_{g,F_g-1+\ell} - D_{g,1})` over the switchers of that horizon.
    Here :math:`S_g` is 1 for a rise and -1 for a fall.
    """
    estimates = effects_results.get("estimates_unnorm", effects_results["estimates"])
    n_sw = effects_results.get("n_switchers_weighted", effects_results["n_switchers"])
    ate_delta = effects_results["ate_delta"]

    valid_mask = ~np.isnan(estimates) & (n_sw > 0)

    if not np.any(valid_mask):
        return None

    total_n_sw = np.sum(n_sw[valid_mask])
    weights = n_sw[valid_mask] / total_n_sw if total_n_sw > 0 else np.ones(np.sum(valid_mask)) / np.sum(valid_mask)

    weighted_mean_effect = np.sum(weights * estimates[valid_mask])
    ate_denom = np.sum(weights * ate_delta[valid_mask])
    if ate_denom == 0:
        return None
    ate_estimate = weighted_mean_effect / ate_denom

    # Since a horizon that no switcher reaches keeps a zero influence column, the columns line up with the horizons.
    inf_func_unnorm = effects_results["influence_func_unnorm"]
    weighted_inf = np.zeros(inf_func_unnorm.shape[0])
    for column, weight in zip(inf_func_unnorm[:, valid_mask].T, weights, strict=True):
        weighted_inf += weight * column

    ate_inf = weighted_inf / ate_denom
    cluster_sums = compute_cluster_influence(ate_inf, effects_results.get("unit_clusters"))
    ate_se = np.sqrt(np.sum(cluster_sums**2)) / n_groups

    # A (group, period) cell that enters several horizons counts once in the sample size.
    count_cols = [f"count_{i + 1}" for i in np.flatnonzero(valid_mask)]
    total_n_obs = float(
        effects_results["df"].select(pl.any_horizontal([pl.col(c) == 1 for c in count_cols]).sum()).item()
    )
    total_n_switchers = np.sum(effects_results["n_switchers"][valid_mask])

    return ATEResult(
        estimate=ate_estimate,
        std_error=ate_se,
        ci_lower=ate_estimate - z_crit * ate_se,
        ci_upper=ate_estimate + z_crit * ate_se,
        n_observations=total_n_obs,
        n_switchers=total_n_switchers,
    )


def _test_effects_equality(effects_results, config=None):
    """Test whether effects are equal, optionally over a range of horizons.

    Parameters
    ----------
    effects_results : dict
        Dictionary with 'estimates' and 'vcov' keys.
    config : DIDInterConfig, optional
        Configuration object. When ``effects_equal_lb`` and ``effects_equal_ub``
        are set, only effects in that range are tested.

    Returns
    -------
    dict or None
        Dictionary with chi2_stat, df, p_value, and warnings list.
    """
    estimates = effects_results["estimates"]
    vcov = effects_results.get("vcov")

    if vcov is None or len(estimates) < 2:
        return None

    lb = getattr(config, "effects_equal_lb", None) if config else None
    ub = getattr(config, "effects_equal_ub", None) if config else None

    if lb is not None and ub is not None:
        idx_start = lb - 1
        idx_end = ub
        estimates = estimates[idx_start:idx_end]
        vcov = vcov[idx_start:idx_end, idx_start:idx_end]

    valid_mask = ~np.isnan(estimates)
    if np.sum(valid_mask) < 2:
        return None

    valid_estimates = estimates[valid_mask]
    valid_vcov = vcov[np.ix_(valid_mask, valid_mask)]

    k = len(valid_estimates)
    D = np.eye(k - 1, k) - np.ones((k - 1, k)) / k
    contrast_diff = D @ valid_estimates
    contrast_vcov = D @ valid_vcov @ D.T
    contrast_vcov = (contrast_vcov + contrast_vcov.T) / 2

    warnings_list = []

    eigenvalues = np.linalg.eigvalsh(contrast_vcov)
    positive_eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(positive_eigenvalues) < k - 1:
        warnings_list.append(
            "The variance-covariance matrix of the effects tested is not "
            "invertible. The equality test cannot be computed."
        )
        return {
            "chi2_stat": np.nan,
            "df": k - 1,
            "p_value": np.nan,
            "warnings": warnings_list,
        }

    condition_ratio = positive_eigenvalues.max() / positive_eigenvalues.min()
    if condition_ratio >= 1000:
        warnings_list.append(
            "The variance-covariance matrix of the effects tested is close "
            f"to singular (condition ratio: {condition_ratio:.1f}). The equality test "
            "may be unreliable."
        )

    try:
        chi2_stat = float(contrast_diff @ np.linalg.pinv(contrast_vcov) @ contrast_diff)
        df = k - 1
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)
        return {
            "chi2_stat": chi2_stat,
            "df": df,
            "p_value": p_value,
            "warnings": warnings_list,
        }
    except np.linalg.LinAlgError:
        return None


def _compute_heterogeneity(df, config):
    r"""Compute heterogeneous effects analysis via WLS regressions.

    Each requested effect and each placebo gets a regression. The placebo at horizon :math:`\ell`
    regresses the outcome change from :math:`F_g - 1` back to :math:`F_g - 1 - \ell` on the same
    groups as the effect at horizon :math:`\ell`. With ``trends_lin`` only the effects get a regression.
    """
    if config.predict_het is None or config.normalized:
        return None

    covariates, het_effects = config.predict_het

    if not isinstance(covariates, list) or not isinstance(het_effects, list) or len(covariates) == 0:
        return None

    gname = config.gname
    tname = config.tname
    # Since trends_lin differences the outcome, the regressions read the outcome in levels.
    outcome = "_outcome_levels" if config.trends_lin else config.yname

    valid_covariates = []
    for cov in covariates:
        if cov not in df.columns:
            continue
        n_unique = df.group_by(gname).agg(pl.col(cov).drop_nulls().n_unique().alias("n_uniq"))
        if (n_unique["n_uniq"] > 1).any():
            continue
        valid_covariates.append(cov)

    if len(valid_covariates) == 0:
        return None

    effects = [h for h in range(1, config.effects + 1) if -1 in het_effects or h in het_effects]
    placebos = [h for h in range(1, config.placebo + 1) if -1 in het_effects or h in het_effects]
    if config.trends_lin and placebos:
        warnings.warn(
            "predict_het runs no placebo regressions when trends_lin=True.",
            UserWarning,
            stacklevel=4,
        )
        placebos = []
    all_horizons = [*(-h for h in reversed(placebos)), *effects]

    if len(all_horizons) == 0:
        return None

    df = df.with_columns(
        pl.when(pl.col(tname) == pl.col("F_g") - 1).then(pl.col(outcome)).otherwise(None).alias("_Y_baseline")
    )
    df = df.with_columns(pl.col("_Y_baseline").mean().over(gname).alias("_Y_baseline"))
    df = df.with_columns(pl.col("_Y_baseline").is_not_null().alias("_feasible_het"))

    if config.trends_lin:
        df = df.with_columns(
            pl.when(pl.col(tname) == pl.col("F_g") - 2).then(pl.col(outcome)).otherwise(None).alias("_Y_baseline_m2")
        )
        df = df.with_columns(pl.col("_Y_baseline_m2").mean().over(gname).alias("_Y_baseline_m2"))
        df = df.with_columns((pl.col("_feasible_het") & pl.col("_Y_baseline_m2").is_not_null()).alias("_feasible_het"))

    df = df.sort([gname, tname])
    df = df.with_columns(pl.arange(0, pl.len()).over(gname).alias("_gr_id"))

    results = []
    for horizon in all_horizons:
        het_result = _compute_het_horizon(df, valid_covariates, horizon, config, outcome)
        if het_result is not None:
            results.append(het_result)

    # A regression whose HC2 variance is undefined reports NaN standard errors next to finite estimates.
    exact = [r.horizon for r in results if np.isnan(r.std_errors).all() and np.isfinite(r.estimates).all()]
    if exact:
        warnings.warn(
            f"The heterogeneity regressions at horizons {', '.join(map(str, exact))} fit a group exactly, for example "
            "the only group with its switch date, baseline treatment, and switch direction. Since HC2 standard errors "
            "are undefined for such a fit, their standard errors, confidence intervals, and F-test p-values are NaN.",
            UserWarning,
            stacklevel=4,
        )

    return results if len(results) > 0 else None


def _compute_het_horizon(df, covariates, horizon, config, outcome):
    """Compute the heterogeneity regression at one horizon.

    A negative horizon is a placebo.
    """
    gname = config.gname
    tname = config.tname

    df = df.with_columns(
        pl.when(pl.col(tname) == pl.col("F_g") - 1 + horizon)
        .then(pl.col(outcome))
        .otherwise(None)
        .alias(f"_Y_h{horizon}")
    )
    df = df.with_columns(pl.col(f"_Y_h{horizon}").mean().over(gname).alias(f"_Y_h{horizon}"))
    df = df.with_columns((pl.col(f"_Y_h{horizon}") - pl.col("_Y_baseline")).alias("_diff_het"))

    if config.trends_lin:
        df = df.with_columns(
            (pl.col("_diff_het") - horizon * (pl.col("_Y_baseline") - pl.col("_Y_baseline_m2"))).alias("_diff_het")
        )

    df = df.with_columns((pl.col("S_g") * pl.col("_diff_het")).alias("_prod_het"))
    df = df.with_columns(pl.when(pl.col("_gr_id") != 0).then(None).otherwise(pl.col("_prod_het")).alias("_prod_het"))

    het_sample = df.filter(
        (pl.col("F_g") - 1 + abs(horizon) <= pl.col("T_g"))
        & pl.col("_feasible_het")
        & pl.col("_prod_het").is_not_null()
    )

    if len(het_sample) < len(covariates) + 5:
        return None

    return _run_het_regression(het_sample, covariates, horizon, config)


def _compute_same_switchers_mask(df, config, n_effects, n_placebos, t_max):
    r"""Flag the switchers that every requested effect and placebo can use.

    A switcher reaches effect :math:`q` when its outcome difference :math:`Y_{g,F_g-1+q} - Y_{g,F_g-1}`
    is observed and a group with the same baseline treatment that has not switched by
    :math:`F_g - 1 + q` has an observed outcome difference over the same periods. It reaches placebo
    :math:`q` when :math:`Y_{g,F_g-1-q} - Y_{g,F_g-1}` is observed and some not-yet-switched group
    with the same baseline treatment has the same difference.

    Parameters
    ----------
    df : polars.DataFrame
        Preprocessed panel sorted by group and period.
    config : DIDInterConfig
        Configuration object.
    n_effects : int
        Number of effects every flagged switcher must reach.
    n_placebos : int
        Number of placebos every switcher flagged in ``_same_switcher_pl`` must reach, 0 for no placebo flag.
    t_max : int
        Last period of the sample.

    Returns
    -------
    polars.DataFrame
        The panel with the column ``_same_switcher`` for the effects and, when ``n_placebos`` is
        positive, the column ``_same_switcher_pl`` for the placebos.
    """
    reaches_effects = (pl.col("F_g") - 1 + n_effects) <= pl.col("T_g")
    for lag in range(1, n_effects + 1):
        df, reached = _reaches_horizon(df, config, lag, t_max)
        reaches_effects = reaches_effects & reached
    df = df.with_columns(reaches_effects.fill_null(False).alias("_same_switcher"))

    if n_placebos > 0:
        reaches_placebos = pl.lit(True)
        for lag in range(1, n_placebos + 1):
            df, reached = _reaches_horizon(df, config, -lag, t_max)
            reaches_placebos = reaches_placebos & reached
        df = df.with_columns(reaches_placebos.fill_null(False).alias("_same_switcher_pl"))

    return df.drop([name for name in df.columns if name.startswith("_reaches_")])


def _reaches_horizon(df, config, lag, t_max):
    """Flag the groups that reach the horizon ``lag`` periods from the last period before the switch."""
    gname = config.gname
    tname = config.tname
    yname = config.yname
    diff = f"_reaches_diff_{lag}"
    flag = f"_reaches_{lag}"

    # A negative lag shifts forward. A placebo then compares an earlier period with the period before the switch.
    df = df.with_columns((pl.col(yname) - pl.col(yname).shift(lag).over(gname)).alias(diff))
    target = pl.col(tname) == pl.col("F_g") - 1 + lag

    not_yet_switched = pl.col(diff).is_not_null() & (pl.col("F_g") > pl.col(tname))
    control = pl.when(not_yet_switched).then(1.0)
    if config.only_never_switchers:
        control = pl.when(not_yet_switched & (pl.col("F_g") < t_max + 1)).then(0.0).otherwise(control)
    df = df.with_columns((control * pl.col("weight_gt")).sum().over(_get_group_vars(config)).alias(flag))
    df = df.with_columns(
        (
            (pl.when(target).then(pl.col(flag)).mean().over(gname) > 0)
            & pl.when(target).then(pl.col(diff)).mean().over(gname).is_not_null()
        ).alias(flag)
    )

    return df, pl.col(flag).fill_null(False)


def _get_group_vars(config):
    """Get the columns whose values define a control pool."""
    group_vars = [config.tname, "d_sq_int"]

    if config.trends_nonparam:
        group_vars.extend(config.trends_nonparam)

    return group_vars
