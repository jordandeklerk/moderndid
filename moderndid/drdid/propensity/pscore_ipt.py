"""Propensity score estimation using Inverse Probability Tilting (IPT)."""

import warnings

import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.special
import statsmodels.api as sm


def calculate_pscore_ipt(D, X, iw, quantiles=None):
    r"""Calculate propensity scores using Inverse Probability Tilting for the ATT.

    Implements a specific variant of IPT tailored for estimating the
    Average Treatment Effect on the Treated (ATT). Instead of re-weighting both
    the treated and control groups to match the full sample, it estimates a
    propensity score model that implies a re-weighting of the control group to
    match the covariate distribution of the treated group. This is achieved by
    solving the following optimization problem for the propensity score
    parameters :math:`\gamma` given by

    .. math::
        \widehat{\gamma}^{ipt} = \arg\max_{\gamma \in \Gamma} \mathbb{E}_{n}
        \left[D X^{\prime} \gamma - (1-D) \exp(X^{\prime} \gamma)\right].

    The first-order condition of this problem implies the balancing property

    .. math::
        \sum_{i: D_i=1} w_i X_i = \sum_{i: D_i=0} w_i
        \frac{\widehat{p}(X_i)}{1-\widehat{p}(X_i)} X_i,

    where :math:`\widehat{p}(X) = \text{expit}(X'\widehat{\gamma}^{ipt})` is the estimated
    propensity score (i.e., the logistic function, :math:`\exp(v) / (1 + \exp(v))`)
    and :math:`w_i` are the observation weights. This property ensures that the
    weighted average of covariates in the control group matches the weighted
    average in the treated group, which is a key condition for identifying the ATT.

    Parameters
    ----------
    D : ndarray
        Treatment indicator (1D array).
    X : ndarray
        Covariate matrix (2D array, n_obs x n_features), must include intercept.
    iw : ndarray
        Individual weights (1D array).
    quantiles : dict[int, list[float]] | None
        Dict mapping column indices to quantiles (values between 0 and 1).
        For example, {1: [0.25, 0.5, 0.75]} adds 25th, 50th, 75th percentiles
        of the 2nd column as balance constraints. Default is None (no quantiles).

    Returns
    -------
    ndarray
        Propensity scores.

    Notes
    -----
    The general IPT framework described in [1]_ for the ATE involves solving
    two separate moment equations to find weights for the treated and control
    groups that balance covariates with the full sample. These are given by
    equations (8) and (11) in their paper

    .. math::
        \frac{1}{N} \sum_{i=1}^{N}\left\{\frac{D_{i}}
        {G\left(t\left(X_{i}\right)^{\prime} \delta_{I P T}^{1}\right)}-1\right\}
        t\left(X_{i}\right)=0

    and

    .. math::
        \frac{1}{N} \sum_{i=1}^{N}\left\{\frac{1-D_{i}}
        {1-G\left(t\left(X_{i}\right)^{\prime} \delta_{I P T}^{0}\right)}-1\right\}
        t\left(X_{i}\right)=0.

    This implementation, following [2]_, uses a single objective function tailored for
    ATT estimation. The function attempts to solve this using a trust-constr optimizer,
    falling back to a BFGS optimization of a modified loss function, and finally to a
    standard logit model if the IPT optimizations fail.

    References
    ----------

    .. [1] Graham, B., Pinto, C., and Egel, D. (2012), "Inverse Probability Tilting for Moment
        Condition Models with Missing Data," The Review of Economic Studies, 79(3), 1053-1079.
        https://doi.org/10.1093/restud/rdr047

    .. [2] Sant'Anna, P. H., and Zhao, J. (2020), "Inverse Probability Weighting with
        Missing Data," Journal of the American Statistical Association, 115(530), 1542-1552.
        https://doi.org/10.1080/01621459.2019.1635520
    """
    if np.all(iw == 0):
        warnings.warn(
            "All individual weights are zero. Propensity scores will be uninformative.",
            UserWarning,
        )
        return np.full_like(D, np.nan, dtype=float)

    unique_d = np.unique(D)
    if not np.array_equal(unique_d, [0, 1]):
        warnings.warn(
            f"Treatment indicator D contains values {unique_d}. Expected binary values [0, 1]. "
            "Results may be unreliable.",
            UserWarning,
        )

    X_processed, _ = _remove_collinear_columns(X)

    if quantiles is not None:
        X_processed = _add_quantile_constraints(X_processed, quantiles, iw)

    n_obs, k_features = X_processed.shape
    init_gamma = _get_initial_gamma(D, X_processed, iw, k_features)

    # Try trust-constr optimization first
    try:
        opt_cal_results = scipy.optimize.minimize(
            lambda g, d_arr, x_arr, iw_arr: _loss_ps_cal(g, d_arr, x_arr, iw_arr)[0],
            init_gamma.astype(np.float64),
            args=(D, X_processed, iw),
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
                args=(D, X_processed, iw, n_obs),
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
                logit_model_refit = sm.GLM(D, X_processed, family=sm.families.Binomial(), freq_weights=iw)
                logit_results_refit = logit_model_refit.fit(start_params=init_gamma, maxiter=100)
                if not logit_results_refit.converged:
                    warnings.warn("Initial Logit model (used as fallback) also did not converge.", UserWarning)
            except (ValueError, np.linalg.LinAlgError, RuntimeError, OverflowError):
                warnings.warn("Checking convergence of fallback Logit model failed.", UserWarning)

    # Compute propensity scores
    pscore_linear = X_processed @ gamma_cal
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
    value = -np.sum(iw * (1 - D) * phi + v)

    grad_vec_term = iw * (1 - D) * phi1 + 1.0
    gradient = -(X.T @ grad_vec_term)

    hess_M_ipt_vector = (1 - D) * iw * phi2
    hessian_term_matrix = X * hess_M_ipt_vector[:, np.newaxis]
    hessian = -(hessian_term_matrix.T @ X)
    return value, gradient, hessian


def _get_initial_gamma(D, X, iw, k_features):
    """Get initial gamma values for optimization."""
    try:
        logit_model = sm.GLM(D, X, family=sm.families.Binomial(), freq_weights=iw)
        logit_results = logit_model.fit(maxiter=100)
        init_gamma = logit_results.params
        if not logit_results.converged:
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


def _remove_collinear_columns(X):
    """Remove collinear columns from covariate matrix."""
    n_obs, n_cols = X.shape

    if n_cols == 1:
        return X, []

    _, R, P = scipy.linalg.qr(X, mode="economic", pivoting=True)

    diag_R = np.abs(np.diag(R))
    tol = diag_R[0] * max(n_obs, n_cols) * np.finfo(float).eps
    rank = np.sum(diag_R > tol)

    if rank == n_cols:
        return X, []

    independent_cols = P[:rank]
    removed_cols = P[rank:]

    if len(removed_cols) > 0:
        warnings.warn(
            f"Removed {len(removed_cols)} collinear column(s) from covariate matrix. "
            f"Column indices removed: {removed_cols.tolist()}",
            UserWarning,
        )

    return X[:, independent_cols], removed_cols.tolist()


def _add_quantile_constraints(X, quantiles, iw):
    """Add quantile balance constraints to covariate matrix."""
    quantile_features = []

    for col_idx, q_list in quantiles.items():
        if col_idx >= X.shape[1]:
            warnings.warn(f"Column index {col_idx} exceeds number of columns ({X.shape[1]}). Skipping.", UserWarning)
            continue

        if col_idx == 0 and np.all(X[:, 0] == X[0, 0]):
            warnings.warn("Skipping quantile constraints for intercept column.", UserWarning)
            continue

        col_values = X[:, col_idx]

        for q in q_list:
            if not 0 < q < 1:
                warnings.warn(f"Quantile {q} must be between 0 and 1. Skipping.", UserWarning)
                continue

            threshold = _weighted_quantile(col_values, q, iw)

            q_indicator = (col_values <= threshold).astype(float)
            quantile_features.append(q_indicator)

    if quantile_features:
        X_extended = np.column_stack([X] + quantile_features)
        return X_extended
    return X


def _weighted_quantile(values, q, weights):
    """Compute weighted quantile."""
    sorted_idx = np.argsort(values)
    sorted_values = values[sorted_idx]
    sorted_weights = weights[sorted_idx]

    # Cumulative weights
    cum_weights = np.cumsum(sorted_weights)
    total_weight = cum_weights[-1]

    # Find quantile position
    quantile_weight = q * total_weight
    idx = np.searchsorted(cum_weights, quantile_weight)

    if idx == 0:
        return sorted_values[0]
    if idx >= len(sorted_values):
        return sorted_values[-1]

    # Linear interpolation between adjacent values
    w_below = cum_weights[idx - 1]
    w_above = cum_weights[idx]

    if w_above == w_below:
        return sorted_values[idx]

    # Interpolation factor
    alpha = (quantile_weight - w_below) / (w_above - w_below)
    return sorted_values[idx - 1] + alpha * (sorted_values[idx] - sorted_values[idx - 1])
