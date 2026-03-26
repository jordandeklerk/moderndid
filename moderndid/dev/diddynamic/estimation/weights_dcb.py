"""DCB balancing weights via quadratic programming."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from moderndid.dev.diddynamic.estimation.coefficients import compute_coefficients


class DCBResult(NamedTuple):
    """Result of DCB weight estimation.

    Attributes
    ----------
    mu_hat : float
        Estimated potential outcome under target treatment history.
    gammas : np.ndarray
        Weight matrix of shape ``(n, T)`` with per-period balancing weights.
    predictions : np.ndarray
        Prediction matrix of shape ``(n, T)`` from the coefficient stage.
    not_nas : list[np.ndarray]
        Valid row indices per period.
    coef_t : list[np.ndarray]
        Coefficient vectors per period.
    bias : float
        Debiasing correction, ``nan`` if debiasing was not requested.
    """

    mu_hat: float
    gammas: np.ndarray
    predictions: np.ndarray
    not_nas: list[np.ndarray]
    coef_t: list[np.ndarray]
    bias: float


def compute_dcb_estimator(
    n_periods: int,
    outcome: np.ndarray,
    treatment_matrix: np.ndarray,
    covariates_t: dict[int, np.ndarray],
    ds: np.ndarray,
    *,
    method: str = "lasso_subsample",
    adaptive_balancing: bool = True,
    debias: bool = False,
    regularization: bool = True,
    nfolds: int = 10,
    lb: float = 1e-4,
    ub: float = 10.0,
    grid_length: int = 1000,
    n_beta_nonsparse: float = 1e-4,
    ratio_coefficients: float = 1 / 3,
    lags: int | None = None,
    dim_fe: int = 0,
    fast_adaptive: bool = False,
    tolerance: float = 1e-8,
) -> DCBResult:
    r"""Estimate potential outcomes using dynamic covariate balancing weights.

    Implements Algorithm 1 from [1]_. For each period :math:`t`, solves a
    quadratic program to find balancing weights :math:`\hat{\gamma}_t` that
    minimise the :math:`\ell_2` norm subject to dynamic covariate balance
    constraints

    .. math::

        \hat{\gamma}_t = \arg\min_{\gamma_t} \sum_{i=1}^{n} \gamma_{i,t}^2
        \quad \text{s.t.} \quad
        \left\| \hat{\gamma}_{t-1}^\top H_t - \gamma_t^\top H_t \right\|_\infty
        \leq K_{1,t} \, \delta_t(n, p_t),

    with :math:`\mathbf{1}^\top \gamma_t = 1`, :math:`\gamma_t \geq 0`, and
    :math:`\gamma_{i,t} = 0` for units with :math:`D_{i,1:t} \neq d_{1:t}`.
    The estimated potential outcome is then

    .. math::

        \hat{\mu}_T(d_{1:T}) = \hat{\gamma}_T^\top Y_T
        - \sum_{t=1}^{T} (\hat{\gamma}_t - \hat{\gamma}_{t-1})^\top
        H_t \hat{\beta}_{d_{1:T}}^{(t)}.

    Parameters
    ----------
    n_periods : int
        Number of time periods.
    outcome : ndarray, shape (n,)
        Outcome vector at the final period.
    treatment_matrix : ndarray, shape (n, T)
        Binary treatment assignments per unit and period.
    covariates_t : dict[int, ndarray]
        Per-period covariate matrices keyed by 0-based period index.
    ds : ndarray, shape (T,)
        Target treatment history.
    method : {'lasso_plain', 'lasso_subsample'}
        LASSO estimation strategy.
    adaptive_balancing : bool
        If True, use tighter balance constraints on covariates with
        large estimated coefficients.
    debias : bool
        If True, apply bootstrap debiasing with 20 replicates.
    regularization : bool
        If True use cross-validated LASSO, otherwise ridge.
    nfolds : int
        Cross-validation folds for LASSO.
    lb : float
        Lower bound for tuning constant grid search.
    ub : float
        Upper bound for tuning constant grid search.
    grid_length : int
        Number of grid points for tuning constant search.
    n_beta_nonsparse : float
        Threshold below which a rescaled coefficient is treated as zero.
    ratio_coefficients : float
        Fraction of largest coefficients to prioritise when sparsity is low.
    lags : int or None
        Treatment lags for the coefficient stage.
    dim_fe : int
        Number of fixed-effect columns at the end of each covariate matrix.
    fast_adaptive : bool
        If True, use flat grid with :math:`K_2 = 10 K_1` instead of the
        three-segment nested search.
    tolerance : float
        Lower bound on individual weights to enforce strict positivity.

    Returns
    -------
    DCBResult
        Estimated potential outcome, weight matrix, predictions,
        valid-row indices, coefficients, and bias correction.

    References
    ----------

    .. [1] Viviano, D. and Bradic, J. (2026). "Dynamic covariate balancing:
       estimating treatment effects over time with potential local projections."
       *Biometrika*, asag016. https://doi.org/10.1093/biomet/asag016
    """
    n = treatment_matrix.shape[0]
    coefs = compute_coefficients(
        n_periods, outcome, treatment_matrix, covariates_t, ds, method, regularization, nfolds, lags, dim_fe
    )
    pred_t = coefs.pred_t
    coef_t = coefs.coef_t
    covariates_nonna = coefs.covariates_nonna
    not_nas = coefs.not_nas

    grid_search = _grid_search_fast if fast_adaptive else _grid_search_standard

    gammas_first = grid_search(
        _solve_qp_first_period,
        lb=lb,
        ub=ub,
        grid_length=grid_length,
        adaptive_balancing=adaptive_balancing,
        x_all=covariates_nonna[0],
        d_col=treatment_matrix[not_nas[0], 0],
        d_target=ds[0],
        coef=coef_t[0],
        n_beta_nonsparse=n_beta_nonsparse,
        ratio_coefficients=ratio_coefficients,
        tolerance=tolerance,
    )
    if gammas_first is None:
        raise RuntimeError("Infeasible problem for period 1. Try increasing ub.")

    keep_gammas = np.zeros((n, n_periods))
    keep_gammas[not_nas[0], 0] = gammas_first

    previous_component = 1.0 / len(not_nas[0])
    component_mu = np.empty(n_periods)
    component_mu[0] = (gammas_first - previous_component) @ pred_t[0]

    for t in range(1, n_periods):
        gammas_t = grid_search(
            _solve_qp_sequential,
            lb=lb,
            ub=ub,
            grid_length=grid_length,
            adaptive_balancing=adaptive_balancing,
            gamma_prev=keep_gammas[not_nas[t], t - 1],
            x_all=covariates_nonna[t],
            d_mat=treatment_matrix[not_nas[t], : t + 1],
            d_target=ds[: t + 1],
            coef=coef_t[t],
            n_beta_nonsparse=n_beta_nonsparse,
            ratio_coefficients=ratio_coefficients,
            tolerance=tolerance,
        )
        if gammas_t is None:
            raise RuntimeError(f"Infeasible problem for period {t + 1}. Try increasing ub.")

        keep_gammas[not_nas[t], t] = gammas_t
        component_mu[t] = (keep_gammas[not_nas[t], t] - keep_gammas[not_nas[t], t - 1]) @ pred_t[t]

    final_predictions = np.zeros((n, n_periods))
    for t in range(n_periods):
        final_predictions[not_nas[t], t] = pred_t[t]

    last = n_periods - 1
    mu_hat = float(keep_gammas[not_nas[last], last] @ outcome[not_nas[last]] - component_mu.sum())

    final_bias = np.nan
    if debias:
        final_bias = _compute_bias(
            n,
            n_periods,
            outcome,
            treatment_matrix,
            covariates_t,
            ds,
            method,
            regularization,
            nfolds,
            lags,
            dim_fe,
            coef_t,
            covariates_nonna,
            not_nas,
            keep_gammas,
        )

    return DCBResult(
        mu_hat=mu_hat,
        gammas=keep_gammas,
        predictions=final_predictions,
        not_nas=list(not_nas),
        coef_t=list(coef_t),
        bias=final_bias,
    )


def _solve_qp_first_period(
    x_all, d_col, d_target, k1, k2, with_beta, coef, n_beta_nonsparse, ratio_coefficients, tolerance
):
    """Solve the balancing QP for the first period."""
    beta = coef.copy()
    beta[1:] *= np.nanstd(x_all, axis=0, ddof=1)

    mask = d_col == d_target
    x_sub = x_all[mask]
    x_bar = x_all.mean(axis=0)

    if x_sub.ndim == 1:
        x_sub = x_sub.reshape(1, -1)

    p = x_sub.shape[1]
    n = x_sub.shape[0]

    tol_balance = np.sqrt(np.log(p) / np.sqrt(n)) if p > 1 else 1.0

    tight = k1 * tol_balance
    loose = k2 * tol_balance
    bounds_vec = _build_balance_bounds(p, tight, loose, with_beta, beta, n_beta_nonsparse, ratio_coefficients)

    sol = _solve_balance_qp(x_sub, x_bar, n, bounds_vec, tolerance)
    if sol is None:
        return None
    gamma = np.zeros(x_all.shape[0])
    gamma[mask] = sol
    return gamma


def _solve_qp_sequential(
    gamma_prev, x_all, d_mat, d_target, k1, k2, with_beta, coef, n_beta_nonsparse, ratio_coefficients, tolerance
):
    r"""Solve the balancing QP for period :math:`t \geq 2`."""
    beta = coef.copy()
    beta[1:] *= np.nanstd(x_all, axis=0, ddof=1)

    subsample = d_mat == d_target if d_mat.ndim == 1 else np.all(d_mat == d_target, axis=1)

    x_sub = x_all[subsample]
    gamma_sum = gamma_prev.sum()
    x_bar = (gamma_prev @ x_all) / gamma_sum if gamma_sum != 0 else x_all.mean(axis=0)

    if x_sub.ndim == 1:
        x_sub = x_sub.reshape(1, -1)

    p = x_sub.shape[1]
    n = x_sub.shape[0]

    tol_balance = np.sqrt(np.log(p) / np.sqrt(n)) if p > 1 else 1.0

    tight = k1 * tol_balance
    loose = k2 * tol_balance
    bounds_vec = _build_balance_bounds(p, tight, loose, with_beta, beta, n_beta_nonsparse, ratio_coefficients)

    sol = _solve_balance_qp(x_sub, x_bar, n, bounds_vec, tolerance)
    if sol is None:
        return None
    gamma = np.zeros(x_all.shape[0])
    gamma[subsample] = sol
    return gamma


def _build_balance_bounds(p, tight, loose, with_beta, beta, n_beta_nonsparse, ratio_coefficients):
    """Return per-covariate balance tolerance vector."""
    bounds = np.full(p, tight)
    if not with_beta:
        return bounds

    non_zero = np.where(np.abs(beta[1:]) > n_beta_nonsparse)[0]

    if len(beta) >= 90 and np.sum(beta[1:] == 0) < (1 - ratio_coefficients) * len(beta):
        top_k = int(np.floor(ratio_coefficients * len(beta)))
        non_zero = np.argsort(np.abs(beta[1:]))[::-1][:top_k]

    bounds[:] = loose
    if len(non_zero) > 0:
        bounds[non_zero] = tight

    return bounds


def _solve_balance_qp(x_sub, x_bar, n_sub, bounds_vec, tolerance):
    r"""Solve :math:`\min \|\gamma\|^2` subject to balance, simplex, and box constraints."""
    upper_bound = np.log(n_sub) * n_sub ** (-2 / 3)

    x0 = np.full(n_sub, 1.0 / n_sub)
    x0 = np.clip(x0, tolerance, upper_bound)

    constraints = []

    constraints.append(LinearConstraint(np.ones((1, n_sub)), lb=1.0, ub=1.0))

    p = x_sub.shape[1]
    if p > 0:
        lb_bal = x_bar - bounds_vec
        ub_bal = x_bar + bounds_vec
        constraints.append(LinearConstraint(x_sub.T, lb=lb_bal, ub=ub_bal))

    bounds = [(tolerance, upper_bound)] * n_sub

    result = minimize(
        _qp_objective,
        x0,
        jac=_qp_gradient,
        method="trust-constr",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "gtol": 1e-12, "xtol": 1e-12},
    )

    if not result.success and result.status not in (1, 2):
        return None

    gamma = result.x
    if not np.isclose(gamma.sum(), 1.0, atol=1e-4):
        return None

    return gamma


def _qp_objective(x):
    r"""Objective :math:`0.5 \|x\|^2`."""
    return 0.5 * x @ x


def _qp_gradient(x):
    """Gradient of objective."""
    return x


def _grid_search_standard(
    solve_fn, *, lb, ub, grid_length, adaptive_balancing, n_beta_nonsparse, ratio_coefficients, **qp_kwargs
):
    """Three-segment nested grid search over tuning constants."""
    seg_len = max(int(np.floor(grid_length ** (1 / 3))), 2)
    seg_bounds = np.linspace(ub / 3, ub, 3)
    segments = [
        np.linspace(lb, seg_bounds[0], seg_len),
        np.linspace(seg_bounds[0], seg_bounds[1], seg_len),
        np.linspace(seg_bounds[1], seg_bounds[2], seg_len),
    ]

    with_beta = adaptive_balancing

    for seg in segments:
        for k1 in seg:
            k2_values = seg if adaptive_balancing else [k1]
            for k2 in k2_values:
                result = solve_fn(
                    k1=k1,
                    k2=k2,
                    with_beta=with_beta,
                    n_beta_nonsparse=n_beta_nonsparse,
                    ratio_coefficients=ratio_coefficients,
                    **qp_kwargs,
                )
                if result is not None:
                    return result
    return None


def _grid_search_fast(
    solve_fn, *, lb, ub, grid_length, adaptive_balancing, n_beta_nonsparse, ratio_coefficients, **qp_kwargs
):
    """Flat grid search with :math:`K_2 = 10 K_1`."""
    grid = np.linspace(lb, ub, grid_length)
    with_beta = adaptive_balancing

    for k1 in grid:
        k2 = 10.0 * k1
        result = solve_fn(
            k1=k1,
            k2=k2,
            with_beta=with_beta,
            n_beta_nonsparse=n_beta_nonsparse,
            ratio_coefficients=ratio_coefficients,
            **qp_kwargs,
        )
        if result is not None:
            return result
    return None


def _compute_bias(
    n,
    n_periods,
    outcome,
    treatment_matrix,
    covariates_t,
    ds,
    method,
    regularization,
    nfolds,
    lags,
    dim_fe,
    coef_t,
    covariates_nonna,
    not_nas,
    keep_gammas,
):
    """Bootstrap debiasing correction."""
    rng = np.random.default_rng()
    coef_accum = [c.copy() for c in coef_t]

    for _ in range(20):
        idx = rng.choice(n, size=n, replace=True)
        boot_covariates = {t: covariates_t[t][idx] for t in range(n_periods)}
        boot_coefs = compute_coefficients(
            n_periods,
            outcome[idx],
            treatment_matrix[idx],
            boot_covariates,
            ds,
            method,
            regularization,
            nfolds,
            lags,
            dim_fe,
        )
        for t in range(n_periods):
            coef_accum[t] = coef_accum[t] + boot_coefs.coef_t[t]

    bias_parts = np.empty(n_periods)
    coef_diff_0 = coef_t[0] - coef_accum[0] / 20.0
    coef_slope_0 = coef_diff_0[1:]
    diff_gamma_0 = keep_gammas[not_nas[0], 0] - 1.0 / n
    bias_parts[0] = coef_slope_0 @ (diff_gamma_0 @ covariates_nonna[0])

    for t in range(1, n_periods):
        coef_diff = coef_t[t] - coef_accum[t] / 20.0
        coef_slope = coef_diff[1:]
        diff_gamma = keep_gammas[not_nas[t], t] - keep_gammas[not_nas[t], t - 1]
        bias_parts[t] = coef_slope @ (diff_gamma @ covariates_nonna[t])

    return float(bias_parts.sum())
