"""Control variable adjustments."""

import numpy as np
import polars as pl

from moderndid.core.preprocess.utils import get_covariate_names_from_formula


def compute_control_coefficients(df, config, horizon):
    r"""Compute regression coefficients for covariate adjustment.

    Estimates coefficients :math:`\boldsymbol{\theta}` for adjusting the
    :math:`\text{DID}_{g,\ell}` estimator when covariates are included. The
    adjustment uses never-switchers with the same baseline treatment to estimate
    how covariates relate to outcome changes, then applies this relationship to
    remove covariate-driven differences between switchers and non-switchers.

    For each baseline treatment level :math:`d`, computes weighted least squares
    regression of outcome differences on covariate differences among never-switchers
    with :math:`D_{g,1} = d`.

    Parameters
    ----------
    df : pl.DataFrame
        Data with outcome and control columns.
    config : DIDInterConfig
        Configuration object.
    horizon : int
        Current horizon :math:`\ell`.

    Returns
    -------
    dict
        Mapping from baseline treatment level to dict with theta (coefficients),
        inv_denom (inverse of X'WX for variance), and useful (validity flag).
    """
    controls = get_covariate_names_from_formula(config.xformla)
    if not controls:
        return {}
    gname = config.gname

    diff_y_col = f"diff_y_{horizon}"
    results = {}

    baseline_levels = df.filter(pl.col("F_g") == float("inf"))["d_sq"].unique().to_list()
    n_groups = df[gname].n_unique()

    for d_level in baseline_levels:
        subset = df.filter(
            (pl.col("d_sq") == d_level) & (pl.col("F_g") == float("inf")) & pl.col(diff_y_col).is_not_null()
        )

        n_unique_groups = subset[gname].n_unique() if len(subset) > 0 else 0

        if len(subset) < len(controls) + 1 or n_unique_groups <= 1:
            results[d_level] = {
                "theta": np.zeros(len(controls)),
                "inv_denom": None,
                "useful": False,
            }
            continue

        y = subset.select(diff_y_col).to_numpy().flatten()
        weights = subset.select("weight_gt").to_numpy().flatten()

        X_cols = []
        for ctrl in controls:
            lag_col = f"lag_{ctrl}_{horizon}"
            if lag_col in subset.columns:
                diff_ctrl = subset.select(pl.col(ctrl) - pl.col(lag_col)).to_numpy().flatten()
            else:
                diff_ctrl = np.zeros(len(subset))
            X_cols.append(diff_ctrl)

        X = np.column_stack(X_cols)

        valid_mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
        if np.sum(valid_mask) < len(controls) + 1:
            results[d_level] = {
                "theta": np.zeros(len(controls)),
                "inv_denom": None,
                "useful": False,
            }
            continue

        y_valid = y[valid_mask]
        X_valid = X[valid_mask]
        w_valid = weights[valid_mask]

        try:
            W = np.diag(w_valid)
            XtWX = X_valid.T @ W @ X_valid
            XtWy = X_valid.T @ W @ y_valid

            if abs(np.linalg.det(XtWX)) <= 1e-16:
                theta = np.linalg.pinv(XtWX) @ XtWy
                inv_denom = None
                useful = False
            else:
                theta = np.linalg.solve(XtWX, XtWy)
                rsum = np.sum(w_valid)
                inv_denom = np.linalg.pinv(XtWX) * rsum * n_groups
                useful = True

            results[d_level] = {
                "theta": theta,
                "inv_denom": inv_denom,
                "useful": useful,
            }
        except np.linalg.LinAlgError:
            results[d_level] = {
                "theta": np.zeros(len(controls)),
                "inv_denom": None,
                "useful": False,
            }

    return results


def apply_control_adjustment(df, config, horizon, coefficients):
    r"""Apply covariate adjustment to outcome differences.

    Adjusts the outcome difference :math:`Y_{g,F_g-1+\ell} - Y_{g,F_g-1}` by
    removing the component explained by covariate changes. For each group,
    computes

    .. math::

        \widetilde{\Delta Y}_{g,\ell} = \Delta Y_{g,\ell} -
        \boldsymbol{\theta}_{D_{g,1}}' \Delta \mathbf{X}_{g,\ell}

    where :math:`\boldsymbol{\theta}_{D_{g,1}}` are the coefficients estimated
    from never-switchers with the same baseline treatment.

    Parameters
    ----------
    df : pl.DataFrame
        Data with outcome and control columns.
    config : DIDInterConfig
        Configuration object.
    horizon : int
        Current horizon :math:`\ell`.
    coefficients : dict
        Mapping from baseline treatment level to coefficient dict from
        :func:`compute_control_coefficients`.

    Returns
    -------
    pl.DataFrame
        DataFrame with adjusted outcome differences.
    """
    controls = get_covariate_names_from_formula(config.xformla)
    if not controls or not coefficients:
        return df

    gname = config.gname
    diff_y_col = f"diff_y_{horizon}"

    for ctrl in controls:
        lag_col = f"lag_{ctrl}_{horizon}"

        if lag_col not in df.columns:
            df = df.sort([gname, config.tname])
            df = df.with_columns(pl.col(ctrl).shift(horizon).over(gname).alias(lag_col))

        diff_ctrl_col = f"diff_{ctrl}_{horizon}"
        df = df.with_columns((pl.col(ctrl) - pl.col(lag_col)).alias(diff_ctrl_col))

    for d_level, coef_dict in coefficients.items():
        theta = coef_dict["theta"]
        adjustment = pl.lit(0.0)
        for ctrl_idx, ctrl in enumerate(controls):
            diff_ctrl_col = f"diff_{ctrl}_{horizon}"
            adjustment = adjustment + pl.lit(theta[ctrl_idx]) * pl.col(diff_ctrl_col).fill_null(0.0)

        df = df.with_columns(
            pl.when(pl.col("d_sq") == d_level)
            .then(pl.col(diff_y_col) - adjustment)
            .otherwise(pl.col(diff_y_col))
            .alias(diff_y_col)
        )

    return df


def compute_control_influence(df, config, horizon, coefficients, n_groups, n_switchers):
    r"""Compute influence function adjustment for covariate-adjusted estimator.

    Computes the additional influence function terms arising from estimating
    the covariate adjustment coefficients :math:`\boldsymbol{\theta}`. The
    influence function for the adjusted estimator accounts for uncertainty
    in both the treatment effect estimate and the covariate coefficients.

    Parameters
    ----------
    df : pl.DataFrame
        Data with control columns and DID computation columns.
    config : DIDInterConfig
        Configuration object.
    horizon : int
        Current horizon :math:`\ell`.
    coefficients : dict
        Mapping from baseline treatment level to coefficient dict.
    n_groups : int
        Total number of groups :math:`G`.
    n_switchers : int
        Number of switchers at this horizon.

    Returns
    -------
    pl.DataFrame
        DataFrame with control influence columns added.
    """
    controls = get_covariate_names_from_formula(config.xformla)
    if not controls or not coefficients or n_switchers == 0:
        return df

    gname = config.gname
    tname = config.tname
    dist_col = (
        f"dist_to_switch_{horizon}" if f"dist_to_switch_{horizon}" in df.columns else f"distance_to_switch_{horizon}"
    )
    never_col = f"never_change_{horizon}"
    n_treated_col = f"n_treated_{horizon}"
    n_control_col = f"n_control_{horizon}"

    if dist_col not in df.columns or never_col not in df.columns:
        return df

    safe_n_control = pl.when(pl.col(n_control_col) == 0).then(1.0).otherwise(pl.col(n_control_col))
    baseline_levels = [d for d, c in coefficients.items() if c.get("useful", False)]

    for ctrl_idx, ctrl in enumerate(controls):
        diff_ctrl_col = f"diff_{ctrl}_{horizon}"
        weighted_diff_ctrl = f"weighted_diff_{ctrl}_{horizon}"

        if diff_ctrl_col not in df.columns:
            continue

        df = df.with_columns((pl.col(diff_ctrl_col).fill_null(0.0) * pl.col("weight_gt")).alias(weighted_diff_ctrl))

        for d_level in baseline_levels:
            m_col = f"m_{ctrl_idx}_{d_level}_{horizon}"
            m_sum_col = f"m_sum_{ctrl_idx}_{d_level}_{horizon}"

            is_level = pl.col("d_sq") == d_level
            time_cond = (pl.col(tname) >= horizon + 1) & (pl.col(tname) <= pl.col("t_max_by_group"))

            df = df.with_columns(
                (
                    is_level.cast(pl.Float64)
                    * (pl.lit(n_groups) / pl.lit(n_switchers))
                    * (pl.col(dist_col) - (pl.col(n_treated_col) / safe_n_control) * pl.col(never_col).fill_null(0.0))
                    * time_cond.cast(pl.Float64)
                    * pl.col(weighted_diff_ctrl)
                ).alias(m_col)
            )

            df = df.with_columns(pl.col(m_col).sum().over(gname).alias(m_sum_col))
            df = df.with_columns((pl.col(m_sum_col) * pl.col("first_obs_by_gp")).alias(m_sum_col))

    for ctrl_idx, ctrl in enumerate(controls):
        weighted_diff_ctrl = f"weighted_diff_{ctrl}_{horizon}"

        if weighted_diff_ctrl not in df.columns:
            continue

        for d_level in baseline_levels:
            in_sum_col = f"in_sum_{ctrl_idx}_{d_level}_{horizon}"

            is_control = (pl.col("F_g") == float("inf")) & (pl.col("d_sq") == d_level)
            time_cond = (pl.col(tname) >= 2) & (pl.col(tname) < pl.col("F_g"))

            group_vars = [tname, "d_sq"]
            if config.trends_nonparam:
                group_vars.extend(config.trends_nonparam)

            df = df.with_columns(
                pl.when(is_control & time_cond)
                .then(pl.col(weighted_diff_ctrl).sum().over(group_vars))
                .otherwise(pl.lit(0.0))
                .alias(in_sum_col)
            )

    return df


def compute_variance_adjustment(df, config, horizon, coefficients, n_groups):
    r"""Compute variance adjustment for covariate-adjusted estimator.

    Computes the second component of the influence function variance that
    arises from estimating covariate adjustment coefficients. This accounts
    for the additional uncertainty introduced by the two-step estimation
    procedure (first estimating :math:`\boldsymbol{\theta}`, then applying
    the adjustment).

    Parameters
    ----------
    df : pl.DataFrame
        Data with control influence columns from :func:`compute_control_influence`.
    config : DIDInterConfig
        Configuration object.
    horizon : int
        Current horizon :math:`\ell`.
    coefficients : dict
        Mapping from baseline treatment level to coefficient dict.
    n_groups : int
        Total number of groups :math:`G`.

    Returns
    -------
    pl.DataFrame
        DataFrame with variance adjustment column for each group.
    """
    controls = get_covariate_names_from_formula(config.xformla)
    if not controls or not coefficients:
        return df

    gname = config.gname
    part2_col = f"part2_{horizon}"
    df = df.with_columns(pl.lit(0.0).alias(part2_col))

    baseline_levels = [d for d, c in coefficients.items() if c.get("useful", False)]

    for d_level in baseline_levels:
        coef_dict = coefficients[d_level]
        inv_denom = coef_dict.get("inv_denom")
        theta = coef_dict["theta"]

        if inv_denom is None:
            continue

        combined_col = f"combined_{d_level}_{horizon}"
        df = df.with_columns(pl.lit(0.0).alias(combined_col))

        for j in range(len(controls)):
            in_brackets_col = f"in_brackets_{d_level}_{j}_{horizon}"
            df = df.with_columns(pl.lit(0.0).alias(in_brackets_col))

            for k in range(len(controls)):
                in_sum_col = f"in_sum_{k}_{d_level}_{horizon}"
                if in_sum_col not in df.columns:
                    continue

                coef_jk = float(inv_denom[j, k])
                is_level = (pl.col("d_sq") == d_level) & (pl.col("F_g") != float("inf"))

                df = df.with_columns(
                    pl.when(is_level)
                    .then(pl.col(in_brackets_col) + pl.lit(coef_jk) * pl.col(in_sum_col))
                    .otherwise(pl.col(in_brackets_col))
                    .alias(in_brackets_col)
                )

            theta_j = float(theta[j])
            df = df.with_columns((pl.col(in_brackets_col) - pl.lit(theta_j)).alias(in_brackets_col))

            m_sum_col = f"m_sum_{j}_{d_level}_{horizon}"
            if m_sum_col in df.columns:
                M_total = df.filter(pl.col("first_obs_by_gp") == 1).select(pl.col(m_sum_col).sum()).item()
                M_scaled = M_total / n_groups if n_groups > 0 else 0.0

                df = df.with_columns(
                    (pl.col(combined_col) + pl.lit(M_scaled) * pl.col(in_brackets_col)).alias(combined_col)
                )

        df = df.with_columns((pl.col(part2_col) + pl.col(combined_col)).alias(part2_col))

    df = df.with_columns((pl.col(part2_col).sum().over(gname) * pl.col("first_obs_by_gp")).alias(part2_col))

    return df
