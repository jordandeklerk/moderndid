"""Propensity score estimation using Inverse Probability Tilting (IPT)."""

import warnings

import numpy as np
import scipy.optimize
import scipy.special
import statsmodels.api as sm


def calculate_pscore_ipt(D, X, iw):
    """Calculate propensity scores using Inverse Probability Tilting.

    Parameters
    ----------
    D : ndarray
        Treatment indicator (1D array).
    X : ndarray
        Covariate matrix (2D array, n_obs x n_features), must include intercept.
    iw : ndarray
        Individual weights (1D array).

    Returns
    -------
    ndarray
        Propensity scores.
    """
    n_obs, k_features = X.shape
    init_gamma = _get_initial_gamma(D, X, iw, k_features)

    # Try trust-constr optimization first
    try:
        opt_cal_results = scipy.optimize.minimize(
            _loss_ps_cal,
            init_gamma.astype(np.float64),
            args=(D, X, iw),
            method="trust-constr",
            jac=lambda g, d_arr, x_arr, iw_arr: _loss_ps_cal(g, d_arr, x_arr, iw_arr)[1],
            hess=lambda g, d_arr, x_arr, iw_arr: _loss_ps_cal(g, d_arr, x_arr, iw_arr)[2],
            options={"maxiter": 1000},
        )
        if opt_cal_results.success:
            gamma_cal = opt_cal_results.x
        else:
            raise RuntimeError("trust-constr did not converge")
    except (ValueError, np.linalg.LinAlgError, RuntimeError, OverflowError) as e:
        warnings.warn(f"trust-constr optimization failed: {e}. Using IPT algorithm.", UserWarning)

        # Try IPT optimization
        try:
            opt_ipt_results = scipy.optimize.minimize(
                lambda g, d_arr, x_arr, iw_arr, n: _loss_ps_ipt(g, d_arr, x_arr, iw_arr, n)[0],
                init_gamma.astype(np.float64),
                args=(D, X, iw, n_obs),
                method="BFGS",
                jac=lambda g, d_arr, x_arr, iw_arr, n: _loss_ps_ipt(g, d_arr, x_arr, iw_arr, n)[1],
                options={"maxiter": 10000, "gtol": 1e-06},
            )
            if opt_ipt_results.success:
                gamma_cal = opt_ipt_results.x
            else:
                raise RuntimeError("IPT optimization did not converge") from None
        except (ValueError, np.linalg.LinAlgError, RuntimeError, OverflowError) as e_ipt:
            warnings.warn(f"IPT optimization failed: {e_ipt}. Using initial logit estimates.", UserWarning)
            gamma_cal = init_gamma

            # Validate logit fallback
            try:
                logit_model_refit = sm.Logit(D, X, weights=iw)
                logit_results_refit = logit_model_refit.fit(disp=0, start_params=init_gamma, maxiter=100)
                if not logit_results_refit.mle_retvals["converged"]:
                    warnings.warn("Initial Logit model (used as fallback) also did not converge.", UserWarning)
            except (ValueError, np.linalg.LinAlgError, RuntimeError, OverflowError):
                warnings.warn("Checking convergence of fallback Logit model failed.", UserWarning)

    # Compute propensity scores
    pscore_linear = X @ gamma_cal
    pscore = scipy.special.expit(pscore_linear)

    if np.any(np.isnan(pscore)):
        warnings.warn(
            "Propensity score model coefficients might have NA/Inf components. "
            "Multicollinearity or lack of variation in covariates is a likely reason. "
            "Resulting pscores contain NaNs.",
            UserWarning,
        )

    return pscore


def _loss_ps_cal(gamma, D, X, iw):
    """Loss function for calibrated propensity score estimation using trust.

    Parameters
    ----------
    gamma : ndarray
        Coefficient vector for propensity score model.
    D : ndarray
        Treatment indicator (1D array).
    X : ndarray
        Covariate matrix (2D array, n_obs x n_features), includes intercept.
    iw : ndarray
        Individual weights (1D array).

    Returns
    -------
    tuple
        (value, gradient, hessian)
    """
    n_obs, k_features = X.shape

    if np.any(np.isnan(gamma)):
        return np.inf, np.full(k_features, np.nan), np.full((k_features, k_features), np.nan)

    ps_ind = X @ gamma
    ps_ind_clipped = np.clip(ps_ind, -500, 500)
    exp_ps_ind = np.exp(ps_ind_clipped)

    value = -np.mean(np.where(D, ps_ind, -exp_ps_ind) * iw)

    grad_terms = np.where(D[:, np.newaxis], 1.0, -exp_ps_ind[:, np.newaxis]) * iw[:, np.newaxis] * X
    gradient = -np.mean(grad_terms, axis=0)

    hess_M_vector = np.where(D, 0.0, -exp_ps_ind) * iw
    hessian_term_matrix = X * hess_M_vector[:, np.newaxis]
    hessian = -(X.T @ hessian_term_matrix) / n_obs
    return value, gradient, hessian


def _loss_ps_ipt(gamma, D, X, iw, n_obs):
    """Loss function for inverse probability tilting propensity score estimation.

    Parameters
    ----------
    gamma : ndarray
        Coefficient vector for propensity score model.
    D : ndarray
        Treatment indicator (1D array).
    X : ndarray
        Covariate matrix (2D array, n_obs x n_features), includes intercept.
    iw : ndarray
        Individual weights (1D array).
    n_obs : int
        Number of observations.

    Returns
    -------
    tuple
        (value, gradient, hessian)
    """
    k_features = X.shape[1]
    if np.any(np.isnan(gamma)):
        return np.inf, np.full(k_features, np.nan), np.full((k_features, k_features), np.nan)

    if n_obs <= 1:
        # When n=1, we cannot compute log(n-1), so use a small epsilon
        epsilon = 1e-10
        log_n_minus_1 = np.log(epsilon)
        cn = -epsilon
        bn = -1.0
        an = -epsilon
        v_star = log_n_minus_1
    elif n_obs < 2.5:
        n_minus_1 = max(n_obs - 1, 0.1)
        log_n_minus_1 = np.log(n_minus_1)
        cn = -n_minus_1
        bn = -n_obs + n_minus_1 * log_n_minus_1
        an = -n_minus_1 * (1 - log_n_minus_1 + 0.5 * (log_n_minus_1**2))
        v_star = log_n_minus_1
    else:
        log_n_minus_1 = np.log(n_obs - 1)
        cn = -(n_obs - 1)
        bn = -n_obs + (n_obs - 1) * log_n_minus_1
        an = -(n_obs - 1) * (1 - log_n_minus_1 + 0.5 * (log_n_minus_1**2))
        v_star = log_n_minus_1

    v = X @ gamma
    v_clipped = np.clip(v, -500, 500)

    phi = np.where(v < v_star, -v - np.exp(v_clipped), an + bn * v + 0.5 * cn * (v**2))
    phi1 = np.where(v < v_star, -1.0 - np.exp(v_clipped), bn + cn * v)
    phi2 = np.where(v < v_star, -np.exp(v_clipped), cn)
    value = -np.sum((iw * (1 - D) * phi) + v)

    grad_vec_term = iw * ((1 - D) * phi1 + 1.0)
    gradient = -(X.T @ grad_vec_term)

    hess_M_ipt_vector = (1 - D) * iw * phi2
    hessian_term_matrix = X * hess_M_ipt_vector[:, np.newaxis]
    hessian = -(hessian_term_matrix.T @ X)
    return value, gradient, hessian


def _get_initial_gamma(D, X, iw, k_features):
    """Get initial gamma values for optimization."""
    try:
        logit_model = sm.Logit(D, X, weights=iw)
        logit_results = logit_model.fit(disp=0, maxiter=100)
        init_gamma = logit_results.params
        if not logit_results.mle_retvals["converged"]:
            warnings.warn(
                "Initial Logit model for IPT did not converge. Using pseudo-inverse for initial gamma.", UserWarning
            )
            try:
                init_gamma = np.linalg.pinv(X.T @ (iw[:, np.newaxis] * X)) @ (X.T @ (iw * D))
            except np.linalg.LinAlgError:
                warnings.warn("Pseudo-inverse for initial gamma failed. Using zeros.", UserWarning)
                init_gamma = np.zeros(k_features)
        return init_gamma
    except (np.linalg.LinAlgError, ValueError) as e:
        warnings.warn(f"Initial Logit model failed: {e}. Using zeros for initial gamma.", UserWarning)
        return np.zeros(k_features)
