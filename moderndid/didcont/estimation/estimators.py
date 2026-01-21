# pylint: disable=unused-argument
"""ATT(g,t) estimator functions for panel treatment effects."""

import warnings

import numpy as np
import polars as pl
import statsmodels.api as sm
from formulaic import model_matrix

from ...core.dataframe import to_pandas
from ...drdid.estimators.drdid_panel import drdid_panel
from ...drdid.estimators.reg_did_panel import reg_did_panel
from .container import AttgtResult


def did_attgt(gt_data, xformula="~1", **kwargs):
    """Compute average treatment effect using doubly robust difference-in-differences.

    Parameters
    ----------
    gt_data : pl.DataFrame
        Data that is "local" to a particular group-time average treatment effect.
        Should contain columns: id, D, period, name, Y, and any covariates.
    xformula : str, default="~1"
        Formula string for covariates used in the propensity score and outcome
        regression models. Default is "~1" for no covariates (intercept only).
    **kwargs
        Additional arguments (not used, for compatibility).

    Returns
    -------
    AttgtResult
        NamedTuple containing:

        - **attgt**: The ATT(g,t) estimate
        - **inf_func**: The influence function
        - **extra_gt_returns**: Additional return values (None for this estimator)
    """
    pre_data = gt_data.filter(pl.col("name") == "pre")

    gt_data_wide = gt_data.pivot(index=["id", "D"], on="name", values="Y").sort("id")

    pre_data_aligned = pre_data.sort("id").filter(pl.col("id").is_in(gt_data_wide["id"].to_list()))
    pre_data_pd = to_pandas(pre_data_aligned)

    covariates = model_matrix(xformula, data=pre_data_pd).values

    if "post" not in gt_data_wide.columns or "pre" not in gt_data_wide.columns:
        raise ValueError("Data must contain both 'pre' and 'post' periods")

    d = gt_data_wide["D"].to_numpy()
    y_post = gt_data_wide["post"].to_numpy()
    y_pre = gt_data_wide["pre"].to_numpy()

    result = drdid_panel(y1=y_post, y0=y_pre, d=d, covariates=covariates, influence_func=True)

    return AttgtResult(attgt=result.att, inf_func=result.att_inf_func, extra_gt_returns=None)


def pte_attgt(
    gt_data,
    xformula="~1",
    d_outcome=False,
    d_covs_formula="~-1",
    lagged_outcome_cov=False,
    est_method="dr",
    **kwargs,
):
    """General group-time average treatment effect estimator with multiple estimation methods.

    Parameters
    ----------
    gt_data : pl.DataFrame
        Data that is "local" to a particular group-time average treatment effect.
        Should contain columns: id, D, G, period, name, Y, and any covariates.
    xformula : str, default="~1"
        Formula string for covariates used in the propensity score and outcome
        regression models. Default is "~1" for no covariates (intercept only).
    d_covs_formula : str, default="~-1"
        Formula string for covariates used in the propensity score and outcome
        regression models. Default is "~1" for no covariates (intercept only).
    lagged_outcome_cov : bool, default=False
        Whether to include the lagged (pre-treatment) outcome as a covariate.
    est_method : {"dr", "reg"}, default="dr"
        Estimation method. "dr" for doubly robust, "reg" for regression adjustment.
    **kwargs
        Additional arguments (not used).

    Returns
    -------
    AttgtResult
        NamedTuple containing:

        - **attgt**: The ATT(g,t) estimate
        - **inf_func**: The influence function
        - **extra_gt_returns**: Additional return values (None for this estimator)
    """
    if gt_data.is_empty():
        raise ValueError("Cannot compute ATT(g,t) with empty data")
    if gt_data["D"].n_unique() < 2:
        return AttgtResult(attgt=0.0, inf_func=np.zeros(gt_data["id"].n_unique()), extra_gt_returns=None)

    post_data = gt_data.filter(pl.col("name") == "post")
    treated_post = post_data.filter(pl.col("D") == 1)

    if "G" in treated_post.columns and len(treated_post) > 0:
        this_g = treated_post["G"][0]
    else:
        treated_any = gt_data.filter(pl.col("D") == 1)
        this_g = treated_any["G"][0] if ("G" in treated_any.columns and len(treated_any) > 0) else None
    this_tp = post_data["period"].unique()[0] if len(post_data) > 0 else None

    pre_data = gt_data.filter(pl.col("name") == "pre")

    gt_data_wide = gt_data.pivot(index=["id", "D"], on="name", values="Y").sort("id")

    d = gt_data_wide["D"].to_numpy()
    y_post = gt_data_wide["post"].to_numpy()
    y_pre = gt_data_wide["pre"].to_numpy()

    wide_ids = gt_data_wide["id"].to_list()
    pre_data_aligned = pre_data.filter(pl.col("id").is_in(wide_ids)).sort("id")
    post_data_aligned = post_data.filter(pl.col("id").is_in(wide_ids)).sort("id")

    pre_data_pd = to_pandas(pre_data_aligned)
    post_data_pd = to_pandas(post_data_aligned)

    covariates = model_matrix(xformula, data=pre_data_pd).values

    if d_covs_formula not in ("~-1", "~ -1"):
        d_covs_pre = model_matrix(d_covs_formula, data=pre_data_pd).values
        d_covs_post = model_matrix(d_covs_formula, data=post_data_pd).values
        d_covs = d_covs_post - d_covs_pre
        covariates = np.hstack([covariates, d_covs])

    if lagged_outcome_cov:
        covariates = np.hstack([covariates, y_pre[:, np.newaxis]])

    if ".w" in pre_data_aligned.columns:
        weights_aligned = pre_data_aligned[".w"].to_numpy()
    else:
        weights_aligned = np.ones(len(pre_data_aligned))

    weights_aligned = weights_aligned / np.sum(weights_aligned)

    if d_outcome:
        y = y_post - y_pre
    else:
        y = y_post

    n_control = np.sum(d == 0)
    if n_control > 0:
        control_covs = covariates[d == 0]
        if np.linalg.matrix_rank(control_covs) < control_covs.shape[1]:
            return AttgtResult(attgt=0.0, inf_func=np.zeros(len(gt_data_wide)), extra_gt_returns=None)

    if est_method == "dr" and n_control > 0 and np.sum(d) > 0:
        try:
            ps_model = sm.Logit(d, covariates)
            ps_results = ps_model.fit(disp=0)
            ps_scores = ps_results.predict(covariates)

            if np.max(ps_scores) > 0.99:
                est_method = "reg"
                if this_g is not None and this_tp is not None:
                    warnings.warn(
                        f"Switching to regression adjustment due to limited overlap "
                        f"for group {this_g} in period {this_tp}",
                        UserWarning,
                    )
        except (ValueError, np.linalg.LinAlgError, sm.tools.sm_exceptions.PerfectSeparationError):
            est_method = "reg"
            if this_g is not None and this_tp is not None:
                warnings.warn(
                    f"Switching to regression adjustment due to propensity score "
                    f"estimation issues for group {this_g} in period {this_tp}",
                    UserWarning,
                )

    y0 = np.zeros(len(gt_data_wide))

    if est_method == "dr":
        try:
            result = drdid_panel(
                y1=y, y0=y0, d=d, covariates=covariates, i_weights=weights_aligned, influence_func=True
            )
        except (ValueError, np.linalg.LinAlgError) as e:
            if "Failed to solve linear system" in str(e) or "singular" in str(e).lower():
                where = ""
                if this_g is not None and this_tp is not None:
                    where = f" for group {this_g} in period {this_tp}"
                warnings.warn(
                    f"DR estimator failed due to singularity issues{where}. Switching to regression adjustment.",
                    UserWarning,
                )
                result = reg_did_panel(
                    y1=y, y0=y0, d=d, covariates=covariates, i_weights=weights_aligned, influence_func=True
                )
            else:
                raise
    elif est_method == "reg":
        result = reg_did_panel(y1=y, y0=y0, d=d, covariates=covariates, i_weights=weights_aligned, influence_func=True)
    else:
        raise ValueError(f"Unsupported estimation method: {est_method}")

    if result.att_inf_func is not None:
        unique_ids = gt_data["id"].unique().to_numpy()
        n_unique = len(unique_ids)

        valid_mask = ~(np.isnan(y_post) | np.isnan(y_pre))
        valid_ids = gt_data_wide.filter(pl.lit(valid_mask))["id"].to_numpy()

        inf_func_final = np.zeros(n_unique)

        if np.sum(valid_mask) > 0:
            valid_inf_func = (
                result.att_inf_func[valid_mask]
                if len(result.att_inf_func) == len(valid_mask)
                else result.att_inf_func[: np.sum(valid_mask)]
            )
            for i, uid in enumerate(unique_ids):
                mask = valid_ids == uid
                if np.any(mask):
                    inf_func_final[i] = np.mean(valid_inf_func[mask])
    else:
        inf_func_final = None

    return AttgtResult(attgt=result.att, inf_func=inf_func_final, extra_gt_returns=None)
