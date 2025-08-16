# pylint: disable=unused-argument
"""ATT(g,t) estimator functions for panel treatment effects."""

import warnings

import numpy as np
import statsmodels.api as sm
from formulaic import model_matrix
from scipy.linalg import qr

from ...drdid.estimators.drdid_panel import drdid_panel
from ...drdid.estimators.reg_did_panel import reg_did_panel
from .process_panel import AttgtResult


def did_attgt(gt_data, xformula="~1", **kwargs):
    """Compute average treatment effect using doubly robust difference-in-differences.

    Parameters
    ----------
    gt_data : pd.DataFrame
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
    pre_data = gt_data[gt_data["name"] == "pre"].copy().set_index("id")

    gt_data_wide = gt_data.pivot_table(index=["id", "D"], columns="name", values="Y", aggfunc="first").reset_index()

    covariates = model_matrix(xformula, data=pre_data.loc[gt_data_wide["id"]]).values

    if "post" not in gt_data_wide.columns or "pre" not in gt_data_wide.columns:
        raise ValueError("Data must contain both 'pre' and 'post' periods")

    d = gt_data_wide["D"].values
    y_post = gt_data_wide["post"].values
    y_pre = gt_data_wide["pre"].values

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
    gt_data : pd.DataFrame
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
    post_data = gt_data[gt_data["name"] == "post"]
    treated_post = post_data[post_data["D"] == 1]
    if "G" in treated_post.columns and len(treated_post) > 0:
        this_g = treated_post["G"].iloc[0]
    else:
        this_g = None
    this_tp = post_data["period"].unique()[0] if len(post_data) > 0 else None

    pre_data = gt_data[gt_data["name"] == "pre"].copy()

    gt_data_wide = gt_data.pivot_table(index=["id", "D"], columns="name", values="Y", aggfunc="first").reset_index()

    d = gt_data_wide["D"].values
    y_post = gt_data_wide["post"].values
    y_pre = gt_data_wide["pre"].values

    pre_data_aligned = pre_data.set_index("id").loc[gt_data_wide["id"]].reset_index()
    post_data_aligned = post_data.set_index("id").loc[gt_data_wide["id"]].reset_index()

    covariates = model_matrix(xformula, data=pre_data_aligned).values

    if d_covs_formula not in ("~-1", "~ -1"):
        d_covs_pre = model_matrix(d_covs_formula, data=pre_data_aligned).values
        d_covs_post = model_matrix(d_covs_formula, data=post_data_aligned).values
        d_covs = d_covs_post - d_covs_pre
        covariates = np.hstack([covariates, d_covs])

    if lagged_outcome_cov:
        covariates = np.hstack([covariates, y_pre[:, np.newaxis]])

    if ".w" in pre_data_aligned.columns:
        weights_aligned = pre_data_aligned[".w"].values
    else:
        weights_aligned = np.ones(len(pre_data_aligned))

    weights_aligned = weights_aligned / np.sum(weights_aligned)

    if d_outcome:
        y = y_post - y_pre
    else:
        y = y_post

    control_covs = covariates[d == 0]
    n_control = np.sum(d == 0)

    if n_control > 0 and covariates.shape[1] > 1:
        try:
            _, r_matrix, p_matrix = qr(control_covs, pivoting=True)

            diag_r = np.abs(np.diag(r_matrix))
            if len(diag_r) > 0:
                tol = diag_r[0] * max(control_covs.shape) * np.finfo(r_matrix.dtype).eps
                rank = np.sum(diag_r > tol)

                if rank < covariates.shape[1]:
                    keep_cols_indices = p_matrix[:rank]
                    keep_cols_indices.sort()
                    covariates = covariates[:, keep_cols_indices]
        except (ValueError, np.linalg.LinAlgError):
            pass

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
        result = drdid_panel(y1=y, y0=y0, d=d, covariates=covariates, i_weights=weights_aligned, influence_func=True)
    elif est_method == "reg":
        result = reg_did_panel(y1=y, y0=y0, d=d, covariates=covariates, i_weights=weights_aligned, influence_func=True)
    else:
        raise ValueError(f"Unsupported estimation method: {est_method}")

    return AttgtResult(attgt=result.att, inf_func=result.att_inf_func, extra_gt_returns=None)
