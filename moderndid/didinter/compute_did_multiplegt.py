"""Core computations for estimation in heterogeneous and dynamic ATT estimation."""

import warnings

import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy import stats

from moderndid.core.preprocess.utils import get_covariate_names_from_formula

from .bootstrap import cluster_bootstrap
from .container import ATEResult, DIDInterResult, EffectsResult, HeterogeneityResult, PlacebosResult
from .controls import (
    apply_control_adjustment,
    compute_control_coefficients,
    compute_control_influence,
    compute_variance_adjustment,
)
from .variance import (
    build_treatment_paths,
    compute_clustered_variance,
    compute_cohort_dof,
    compute_control_dof,
    compute_dof_scaling,
    compute_e_hat,
    compute_joint_test,
    compute_union_dof,
)


def compute_did_multiplegt(preprocessed):
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
    df = preprocessed.data

    ci_level = config.ci_level
    alpha = 1 - ci_level / 100
    z_crit = stats.norm.ppf(1 - alpha / 2)

    n_groups = df[config.gname].n_unique()
    t_max = int(df[config.tname].max())

    if config.same_switchers:
        df = _compute_same_switchers_mask(df, config, config.effects, t_max, "effect")

    if config.same_switchers_pl and config.placebo > 0:
        df = _compute_same_switchers_mask(df, config, config.placebo, t_max, "placebo")

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

    if config.boot:
        warnings.warn(
            "did_multiplegt computes analytical standard errors by default. "
            "Bootstrapping is slower and recommended when using a continuous treatment.",
            UserWarning,
            stacklevel=4,
        )

        boot_result = cluster_bootstrap(
            data=df,
            config=config,
            compute_func=_compute_bootstrap_estimates,
            biters=config.biters,
            random_state=config.random_state,
        )
        effects_results["std_errors"] = boot_result.effects_se

        if placebos_results is not None and boot_result.placebos_se is not None:
            placebos_results["std_errors"] = boot_result.placebos_se

    ate = _compute_ate(effects_results, z_crit, n_groups) if effects_results else None

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

    heterogeneity = _compute_heterogeneity(df, config)

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
        },
        vcov_warnings=vcov_warnings,
    )


def _compute_bootstrap_estimates(df, config):
    """Compute estimates for a bootstrap sample of preprocessed data."""
    nan_effects = np.full(config.effects, np.nan)
    nan_placebos = np.full(config.placebo, np.nan) if config.placebo > 0 else None

    if df.height == 0:
        result = {"effects": nan_effects}
        if nan_placebos is not None:
            result["placebos"] = nan_placebos
        return result

    n_groups = df[config.gname].n_unique()
    if n_groups == 0:
        result = {"effects": nan_effects}
        if nan_placebos is not None:
            result["placebos"] = nan_placebos
        return result

    t_max_val = df[config.tname].max()
    if t_max_val is None:
        result = {"effects": nan_effects}
        if nan_placebos is not None:
            result["placebos"] = nan_placebos
        return result
    t_max = int(t_max_val)

    has_switchers = df.filter(pl.col("S_g") != 0).height > 0
    if not has_switchers:
        result = {"effects": nan_effects}
        if nan_placebos is not None:
            result["placebos"] = nan_placebos
        return result

    if config.same_switchers:
        df = _compute_same_switchers_mask(df, config, config.effects, t_max, "effect")

    if config.same_switchers_pl and config.placebo > 0:
        df = _compute_same_switchers_mask(df, config, config.placebo, t_max, "placebo")

    effects_results = _compute_did_effects(
        df=df,
        config=config,
        n_horizons=config.effects,
        n_groups=n_groups,
        t_max=t_max,
        horizon_type="effect",
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


def _compute_did_effects(df, config, n_horizons, n_groups, t_max, horizon_type):
    """Compute effects at multiple horizons."""
    gname = config.gname
    tname = config.tname
    yname = config.yname

    horizons = np.arange(1, n_horizons + 1) if horizon_type == "effect" else -np.arange(1, n_horizons + 1)

    estimates = np.zeros(n_horizons)
    estimates_unnorm = np.zeros(n_horizons)
    std_errors = np.zeros(n_horizons)
    n_switchers_arr = np.zeros(n_horizons)
    n_switchers_weighted_arr = np.zeros(n_horizons)
    delta_d_arr = np.zeros(n_horizons)
    n_obs_arr = np.zeros(n_horizons)
    influence_funcs = []
    influence_funcs_unnorm = []

    df = df.sort([gname, tname])

    for idx, h in enumerate(horizons):
        abs_h = abs(h)

        diff_col = f"diff_y_{abs_h}"

        if horizon_type == "effect":
            df = df.with_columns(pl.col(yname).diff(abs_h).over(gname).alias(diff_col))
            dist_col = f"dist_to_switch_{abs_h}"
        else:
            df = df.with_columns(
                (pl.col(yname).shift(2 * abs_h).over(gname) - pl.col(yname).shift(abs_h).over(gname)).alias(diff_col)
            )
            dist_col = f"dist_to_switch_pl_{abs_h}"

        df = build_treatment_paths(df, abs_h, config)

        coefficients = None
        covariate_names = get_covariate_names_from_formula(config.xformla)
        if covariate_names:
            for ctrl in covariate_names:
                lag_col = f"lag_{ctrl}_{abs_h}"
                df = df.with_columns(pl.col(ctrl).shift(abs_h).over(gname).alias(lag_col))

            coefficients = compute_control_coefficients(df, config, abs_h)
            df = apply_control_adjustment(df, config, abs_h, coefficients)

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

        group_vars = _get_group_vars(config)
        n_control_col = f"n_control_{abs_h}"
        df = df.with_columns(pl.col(never_w_col).sum().over(group_vars).alias(n_control_col))

        if config.switchers == "in":
            switcher_mask = pl.col("S_g") == 1
            increase_val = 1
        elif config.switchers == "out":
            switcher_mask = pl.col("S_g") == -1
            increase_val = -1
        else:
            switcher_mask = pl.col("S_g") != 0
            increase_val = None

        if config.same_switchers and "same_switcher_valid" in df.columns:
            switcher_mask = switcher_mask & pl.col("same_switcher_valid")

        base_cond = (
            (pl.col(tname) == (pl.col("F_g") - 1 + abs_h))
            & (pl.col("L_g") >= abs_h)
            & (pl.col(n_control_col) > 0)
            & pl.col(n_control_col).is_not_null()
        )

        if config.same_switchers and "same_switcher_valid" in df.columns:
            base_cond = base_cond & pl.col("same_switcher_valid")

        if increase_val is not None:
            cond_expr = base_cond & (pl.col("S_g") == increase_val)
        else:
            cond_expr = base_cond & (pl.col("S_g") != 0)

        df = df.with_columns(
            pl.when(pl.col(diff_col).is_null()).then(pl.lit(None)).otherwise(cond_expr.cast(pl.Float64)).alias(dist_col)
        )

        dist_w_col = f"dist_to_switch_w_{abs_h}"
        df = df.with_columns(
            pl.when(switcher_mask).then(pl.col(dist_col) * pl.col("weight_gt")).otherwise(0.0).alias(dist_w_col)
        )

        n_treated_col = f"n_treated_{abs_h}"
        df = df.with_columns(pl.col(dist_w_col).sum().over(group_vars).alias(n_treated_col))

        switcher_filter = (pl.col(dist_col) == 1.0) & pl.col(diff_col).is_not_null() & switcher_mask
        n_switchers_unweighted = df.filter(switcher_filter)[gname].n_unique()

        n_switchers_weighted = df.select(pl.col(dist_w_col).sum()).item()
        if n_switchers_weighted is None or n_switchers_weighted == 0:
            n_switchers_weighted = 0.0

        if n_switchers_unweighted == 0:
            estimates[idx] = np.nan
            std_errors[idx] = np.nan
            n_switchers_arr[idx] = 0
            n_obs_arr[idx] = 0
            continue

        inf_temp_col = f"inf_func_{abs_h}_temp"
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

        inf_col = f"inf_func_{abs_h}"
        df = df.with_columns((pl.col(inf_temp_col).sum().over(gname) * pl.col("first_obs_by_gp")).alias(inf_col))

        did_estimate = df.select(pl.col(inf_col).sum()).item() / n_groups

        estimates_unnorm[idx] = did_estimate
        n_switchers_weighted_arr[idx] = safe_n_switchers
        delta_d = _compute_delta_d(df, config, abs_h, horizon_type, dist_col)

        if horizon_type == "effect":
            delta_d_arr[idx] = delta_d if delta_d is not None else 0.0

        if config.normalized and delta_d is not None and delta_d != 0:
            did_estimate = did_estimate / delta_d

        estimates[idx] = did_estimate

        if covariate_names and coefficients:
            df = compute_control_influence(df, config, abs_h, coefficients, n_groups, safe_n_switchers)
            df = compute_variance_adjustment(df, config, abs_h, coefficients, n_groups)

        switcher_flag = f"is_switcher_{abs_h}"
        weighted_diff = f"weighted_diff_{abs_h}"
        df = df.with_columns(
            pl.col(dist_col).cast(pl.Int64).alias(switcher_flag),
            (pl.col(diff_col).fill_null(0.0) * pl.col("weight_gt")).alias(weighted_diff),
        )

        df = compute_cohort_dof(df, abs_h, config, config.cluster)
        df = compute_control_dof(df, abs_h, config, config.cluster)
        df = compute_union_dof(df, abs_h, config, config.cluster)
        df = compute_dof_scaling(df, abs_h, config)
        df = compute_e_hat(df, abs_h, config)

        dof_scale_col = f"dof_scale_{abs_h}"
        e_hat_col = f"E_hat_{abs_h}"
        inf_var_col = f"inf_func_var_{abs_h}"
        dof_scale_expr = pl.col(dof_scale_col).fill_null(1.0) if dof_scale_col in df.columns else pl.lit(1.0)
        dummy_u_gg_col = f"dummy_u_gg_{abs_h}"
        time_constraint_col = f"time_constraint_{abs_h}"

        df = df.with_columns(
            (pl.lit(abs_h) <= (pl.col("T_g") - 1)).cast(pl.Int64).alias(dummy_u_gg_col),
            ((pl.col(tname) >= pl.lit(abs_h + 1)) & (pl.col(tname) <= pl.col("T_g")))
            .cast(pl.Int64)
            .alias(time_constraint_col),
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

            part2_col = f"part2_{abs_h}"
            if part2_col in df.columns:
                df = df.with_columns((pl.col(inf_var_col) - pl.col(part2_col).fill_null(0.0)).alias(inf_var_col))

            inf_func = df.filter(pl.col("first_obs_by_gp") == 1).select(inf_var_col).to_numpy().flatten()
        else:
            inf_func = df.filter(pl.col("first_obs_by_gp") == 1).select(inf_col).to_numpy().flatten()

            part2_col = f"part2_{abs_h}"
            if part2_col in df.columns:
                part2_vals = df.filter(pl.col("first_obs_by_gp") == 1).select(part2_col).to_numpy().flatten()
                inf_func = inf_func - part2_vals

        if config.cluster:
            cluster_data = df.filter(pl.col("first_obs_by_gp") == 1).select(
                [config.cluster, inf_var_col if e_hat_col in df.columns else inf_col]
            )
            valid_cluster_mask = cluster_data[config.cluster].is_not_null()
            cluster_ids = cluster_data.filter(valid_cluster_mask)[config.cluster].to_numpy().flatten()
            inf_func_for_cluster = (
                cluster_data.filter(valid_cluster_mask)[inf_var_col if e_hat_col in df.columns else inf_col]
                .to_numpy()
                .flatten()
            )
            std_error = compute_clustered_variance(inf_func_for_cluster, cluster_ids, n_groups)
        else:
            std_error = np.sqrt(np.sum(inf_func**2)) / n_groups

        influence_funcs_unnorm.append(inf_func.copy())

        if config.normalized and delta_d is not None and delta_d != 0:
            std_error = std_error / delta_d
            inf_func = inf_func / delta_d

        influence_funcs.append(inf_func)
        std_errors[idx] = std_error
        n_switchers_arr[idx] = n_switchers_unweighted

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

    if config.trends_lin and len(influence_funcs) == n_horizons and n_horizons > 0:
        estimates, std_errors, influence_funcs = _apply_trends_lin(
            estimates, std_errors, influence_funcs, n_groups, config.cluster, df
        )

    vcov = None
    if len(influence_funcs) == n_horizons and all(len(f) > 0 for f in influence_funcs):
        influence_matrix = np.column_stack(influence_funcs)
        vcov = np.cov(influence_matrix, rowvar=False) / n_groups

    return {
        "horizons": horizons.astype(float),
        "estimates": estimates,
        "estimates_unnorm": estimates_unnorm,
        "std_errors": std_errors,
        "n_switchers": n_switchers_arr,
        "n_switchers_weighted": n_switchers_weighted_arr,
        "delta_d_arr": delta_d_arr,
        "n_observations": n_obs_arr,
        "influence_func": np.column_stack(influence_funcs) if influence_funcs else None,
        "influence_func_unnorm": np.column_stack(influence_funcs_unnorm) if influence_funcs_unnorm else None,
        "vcov": vcov,
        "df": df,
    }


def _run_het_regression(het_sample, covariates, horizon, config):
    """Run WLS regression for heterogeneity analysis at a given horizon.

    Uses HC2 standard errors by default. When ``predict_het_hc2bm`` is True on
    ``config``, uses HC2 clustered (Bell-McCaffrey) standard errors clustered
    by the ``cluster`` variable (or ``gname`` if no cluster is specified).
    """
    y = het_sample["_prod_het"].to_numpy()

    # Use uniform weights when the user has not specified a weight variable.
    # The preprocessed weight_gt column zeros out rows with missing Y/D at the
    # observation level, but the het regression uses a group-level dependent
    # variable (_prod_het) that can be valid even when the raw outcome is null
    # at _gr_id=0.  Inheriting those zeros would incorrectly drop valid
    # switchers from the het regression.
    has_user_weights = getattr(config, "weightsname", None) is not None
    if has_user_weights and "weight_gt" in het_sample.columns:
        weights = het_sample["weight_gt"].to_numpy()
    else:
        weights = np.ones(len(y))

    X_cov_raw = het_sample.select(covariates).to_numpy()
    valid_mask = ~np.isnan(y) & np.all(np.isfinite(X_cov_raw), axis=1)
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

    if use_hc2bm:
        model_fit = sm.WLS(y, X, weights=weights).fit()

        try:
            XtWX_inv = np.linalg.inv((X * weights[:, None]).T @ X)
        except np.linalg.LinAlgError:
            XtWX_inv = np.linalg.pinv((X * weights[:, None]).T @ X)

        resid_wt = weights * model_fit.resid

        unique_clusters = np.unique(cluster_ids)
        k = X.shape[1]

        # Bell-McCaffrey block HC2
        M = np.zeros((k, k))
        for c in unique_clusters:
            ij = cluster_ids == c
            X_c = X[ij]
            w_c = weights[ij]
            res_c = resid_wt[ij]
            m_c = ij.sum()

            H_cc = X_c @ XtWX_inv @ X_c.T @ np.diag(w_c)
            eigvals, eigvecs = np.linalg.eigh(np.eye(m_c) - H_cc)
            eigvals = np.maximum(eigvals, 1e-10)
            A_inv_sqrt = eigvecs @ np.diag(eigvals ** (-0.5)) @ eigvecs.T

            res_adj = A_inv_sqrt @ res_c
            s_c = (res_adj[:, None] * X_c).sum(axis=0)
            M += np.outer(s_c, s_c)

        vcov_hc2bm = XtWX_inv @ M @ XtWX_inv

        model_fit._results.cov_params_default = vcov_hc2bm
        model = model_fit
    else:
        model = sm.WLS(y, X, weights=weights).fit(cov_type="HC2")

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
        f_pvalue = 1 - stats.f.cdf(f_stat, n_cov, model.df_resid)
    else:
        f_test = model.f_test(r_matrix)
        f_pvalue = float(f_test.pvalue)

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

    treat_col = f"{dname}_orig" if config.continuous > 0 and f"{dname}_orig" in df.columns else dname

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
        pl.when(mask).then(pl.col(treat_col) - pl.col("d_sq")).otherwise(None).alias("_treat_diff_temp")
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
    """Compute average total effect."""
    estimates = effects_results.get("estimates_unnorm", effects_results["estimates"])
    n_sw = effects_results.get("n_switchers_weighted", effects_results["n_switchers"])
    delta_d_arr = effects_results.get("delta_d_arr", np.ones(len(estimates)))

    valid_mask = ~np.isnan(estimates) & (n_sw > 0)

    if not np.any(valid_mask):
        return None

    total_n_sw = np.sum(n_sw[valid_mask])
    weights = n_sw[valid_mask] / total_n_sw if total_n_sw > 0 else np.ones(np.sum(valid_mask)) / np.sum(valid_mask)

    weighted_mean_effect = np.sum(weights * estimates[valid_mask])

    per_period_delta = np.diff(delta_d_arr, prepend=0.0)
    valid_delta = per_period_delta[valid_mask]
    ate_denom = np.sum(weights * valid_delta)
    if ate_denom == 0:
        ate_denom = delta_d_arr[0] if len(delta_d_arr) > 0 and delta_d_arr[0] != 0 else 1.0
    ate_estimate = weighted_mean_effect / ate_denom

    inf_func_unnorm = effects_results.get("influence_func_unnorm")
    if inf_func_unnorm is not None and inf_func_unnorm.shape[1] == len(estimates):
        weighted_inf = np.zeros(inf_func_unnorm.shape[0])
        for i, (is_valid, wi) in enumerate(
            zip(valid_mask, n_sw / total_n_sw if total_n_sw > 0 else np.ones(len(n_sw)) / len(n_sw), strict=False)
        ):
            if is_valid:
                weighted_inf += wi * inf_func_unnorm[:, i]

        ate_inf = weighted_inf / ate_denom
        ate_se = np.sqrt(np.sum(ate_inf**2)) / n_groups
    elif effects_results.get("vcov") is not None:
        vcov = effects_results["vcov"]
        n_effects = np.sum(valid_mask)
        if n_effects == 1:
            ate_var = float(np.asarray(vcov).flat[0]) if vcov.size > 0 else np.nan
        else:
            ate_weights = np.zeros(len(estimates))
            ate_weights[valid_mask] = weights
            ate_var = ate_weights @ vcov @ ate_weights

        ate_var = ate_var / (ate_denom**2)

        ate_se = np.sqrt(ate_var) if ate_var > 0 else np.nan
    else:
        std_errors = effects_results["std_errors"]
        valid_se = std_errors[valid_mask]
        ate_se = np.sqrt(np.mean(valid_se**2)) if len(valid_se) > 0 else np.nan
        ate_se = ate_se / abs(ate_denom)

    n_observations = effects_results.get("n_observations", np.zeros(len(estimates)))
    total_n_obs = np.sum(n_observations[valid_mask])
    total_n_switchers = np.sum(n_sw[valid_mask])

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


def _apply_trends_lin(estimates, std_errors, influence_funcs, n_groups, cluster, df):
    """Apply linear trend adjustment by accumulating effects across horizons."""
    n_horizons = len(influence_funcs)
    if n_horizons == 0:
        return estimates, std_errors, influence_funcs

    valid_funcs = [f for f in influence_funcs if len(f) > 0 and not np.all(np.isnan(f))]
    if len(valid_funcs) == 0:
        return estimates, std_errors, influence_funcs

    cumulative_estimates = np.cumsum(estimates)

    for idx in range(1, n_horizons):
        estimates[idx] = cumulative_estimates[idx]

        cumulative_inf = np.zeros_like(valid_funcs[0])
        for j in range(idx + 1):
            if j < len(valid_funcs):
                cumulative_inf = cumulative_inf + valid_funcs[j]
        influence_funcs[idx] = cumulative_inf

        if cluster:
            cluster_ids = df.filter(df["first_obs_by_gp"] == 1).select(cluster).to_numpy().flatten()
            std_errors[idx] = compute_clustered_variance(cumulative_inf, cluster_ids, n_groups)
        else:
            std_errors[idx] = np.sqrt(np.sum(cumulative_inf**2)) / n_groups

    return estimates, std_errors, influence_funcs


def _compute_heterogeneity(df, config):
    """Compute heterogeneous effects analysis via WLS regressions."""
    if config.predict_het is None or config.normalized:
        return None

    covariates, het_effects = config.predict_het

    if not isinstance(covariates, list) or not isinstance(het_effects, list) or len(covariates) == 0:
        return None

    gname = config.gname
    tname = config.tname
    yname = config.yname

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

    all_horizons = (
        list(range(1, config.effects + 1))
        if -1 in het_effects
        else [h for h in het_effects if 1 <= h <= config.effects]
    )

    if len(all_horizons) == 0:
        return None

    df = df.with_columns(
        pl.when(pl.col(tname) == pl.col("F_g") - 1).then(pl.col(yname)).otherwise(None).alias("_Y_baseline")
    )
    df = df.with_columns(pl.col("_Y_baseline").mean().over(gname).alias("_Y_baseline"))
    df = df.with_columns(pl.col("_Y_baseline").is_not_null().alias("_feasible_het"))

    if config.trends_lin:
        df = df.with_columns(
            pl.when(pl.col(tname) == pl.col("F_g") - 2).then(pl.col(yname)).otherwise(None).alias("_Y_baseline_m2")
        )
        df = df.with_columns(pl.col("_Y_baseline_m2").mean().over(gname).alias("_Y_baseline_m2"))
        df = df.with_columns((pl.col("_feasible_het") & pl.col("_Y_baseline_m2").is_not_null()).alias("_feasible_het"))

    df = df.sort([gname, tname])
    df = df.with_columns(pl.arange(0, pl.len()).over(gname).alias("_gr_id"))

    results = []
    for horizon in all_horizons:
        het_result = _compute_het_horizon(df, valid_covariates, horizon, config)
        if het_result is not None:
            results.append(het_result)

    return results if len(results) > 0 else None


def _compute_het_horizon(df, covariates, horizon, config):
    """Compute heterogeneity regression for a single horizon."""
    gname = config.gname
    tname = config.tname
    yname = config.yname

    df = df.with_columns(
        pl.when(pl.col(tname) == pl.col("F_g") - 1 + horizon)
        .then(pl.col(yname))
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
        (pl.col("F_g") - 1 + horizon <= pl.col("t_max_by_group"))
        & pl.col("_feasible_het")
        & pl.col("_prod_het").is_not_null()
    )

    if len(het_sample) < len(covariates) + 5:
        return None

    return _run_het_regression(het_sample, covariates, horizon, config)


def _compute_same_switchers_mask(df, config, n_horizons, _t_max, horizon_type="effect"):
    """Compute mask for switchers valid at all horizons."""
    if "L_g" in df.columns:
        if horizon_type == "effect":
            df = df.with_columns((pl.col("L_g") >= n_horizons).alias("same_switcher_valid"))
        else:
            t_min = df[config.tname].min()
            df = df.with_columns(((pl.col("F_g") - t_min) >= (n_horizons + 1)).alias("same_switcher_valid"))
    else:
        df = df.with_columns(pl.lit(True).alias("same_switcher_valid"))

    return df


def _get_group_vars(config):
    """Get grouping variables for control matching."""
    group_vars = [config.tname, "d_sq"]

    if config.trends_nonparam:
        group_vars.extend(config.trends_nonparam)

    return group_vars
