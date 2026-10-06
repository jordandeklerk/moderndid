"""DCB balancing weights via quadratic programming."""

from __future__ import annotations

import numpy as np
from scipy.optimize import LinearConstraint, linprog, minimize

from moderndid.diddynamic.container import DCBResult
from moderndid.diddynamic.estimation.coefficients import compute_coefficients


def compute_dcb_estimator(
    n_periods,
    outcome,
    treatment_matrix,
    covariates_t,
    ds,
    *,
    method="lasso_subsample",
    adaptive_balancing=True,
    debias=False,
    regularization=True,
    nfolds=10,
    lb=1e-4,
    ub=10.0,
    grid_length=1000,
    n_beta_nonsparse=1e-4,
    ratio_coefficients=1 / 3,
    lags=None,
    dim_fe=0,
    fast_adaptive=False,
    tolerance=1e-8,
    random_state=None,
):
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
        If True, estimate the bias of the projection coefficients from 20
        bootstrap resamples. The coefficient bias times the covariate
        imbalance of each period gives the returned ``bias``.
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
        Number of most recent treatment indicators that ``lasso_plain``
        leaves unpenalized. Defaults to ``n_periods``.
    dim_fe : int
        Number of fixed-effect columns at the end of each covariate matrix.
    fast_adaptive : bool
        If True, use flat grid with :math:`K_2 = 10 K_1` instead of the
        three-segment nested search.
    tolerance : float
        Lower bound on individual weights to enforce strict positivity.
    random_state : int, Generator, optional
        Seeds the bootstrap resamples of ``debias=True``.

    Returns
    -------
    DCBResult
        Estimated potential outcome under the target treatment history.

        - **mu_hat**: Estimated potential outcome
        - **gammas**: Weight matrix of shape ``(n, T)`` with per-period balancing weights
        - **predictions**: Prediction matrix of shape ``(n, T)`` from the coefficient stage
        - **not_nas**: Valid row indices per period
        - **coef_t**: Coefficient vectors per period
        - **bias**: Debiasing correction (``nan`` if debiasing was not requested)

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
            np.random.default_rng(random_state),
        )

    return DCBResult(
        mu_hat=mu_hat,
        gammas=keep_gammas,
        predictions=final_predictions,
        not_nas=list(not_nas),
        coef_t=list(coef_t),
        bias=final_bias,
    )


def compute_imbalances(gammas, not_nas, covariates_t):
    r"""Compute the standardized covariate imbalance that the weights leave in each period.

    The first period compares the weights with equal weights on every unit
    that enters it. Each later period compares them with the weights of the
    period before. Up to the scaling, these are the differences that the
    balancing constraints of :func:`compute_dcb_estimator` keep small.

    Each covariate is divided by its standard deviation among the units that
    enter the period. A covariate that takes one value among them keeps its
    raw imbalance.

    Parameters
    ----------
    gammas : ndarray, shape (n, T)
        Weights per unit and period.
    not_nas : list[ndarray]
        Row indices that enter each period.
    covariates_t : dict[int, ndarray]
        Per-period covariate matrices keyed by 0-based period index.

    Returns
    -------
    ndarray, shape (T, p)
        Standardized imbalance of each covariate in each period.

    Notes
    -----
    With :math:`\hat{\gamma}_{i,0} = 1/n_1` for the :math:`n_1` units that
    enter the first period, the imbalance of covariate :math:`j` in period
    :math:`t` is

    .. math::

        I_{t,j} = \frac{1}{\hat{\sigma}_{t,j}} \sum_{i}
        \left(\hat{\gamma}_{i,t} - \hat{\gamma}_{i,t-1}\right) X_{i,t,j},

    where the sum runs over the units that enter period :math:`t` with all
    covariates observed and :math:`\hat{\sigma}_{t,j}` is the standard
    deviation of :math:`X_{t,j}` among them.
    """
    n_periods = gammas.shape[1]
    imbalances = np.empty((n_periods, covariates_t[0].shape[1]))
    for t in range(n_periods):
        rows = np.asarray(not_nas[t])
        x = covariates_t[t][rows]
        observed = ~np.isnan(x).any(axis=1)
        rows, x = rows[observed], x[observed]
        previous = np.full(len(rows), 1.0 / len(rows)) if t == 0 else gammas[rows, t - 1]
        scale = np.where(np.ptp(x, axis=0) > 0, x.std(axis=0, ddof=1), 1.0)
        imbalances[t] = ((gammas[rows, t] - previous) @ x) / scale
    return imbalances


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
    r"""Find the minimum-norm weights that satisfy the balancing constraints.

    A linear program first checks that the constraints admit a solution. Most
    tolerances in the grid search are infeasible. The linear program rejects
    such a tolerance in milliseconds. The quadratic solver would instead run
    to its iteration limit before giving up.

    An accepted solution must satisfy every constraint to within
    :math:`10^{-8}`. Balance violations are measured relative to the size of
    the balance bounds.
    """
    # With fewer than two units the weight cap log(n) n^(-2/3) cannot reach a sum of one.
    if n_sub < 2:
        return None

    upper_bound = np.log(n_sub) * n_sub ** (-2 / 3)
    lb_bal = x_bar - bounds_vec
    ub_bal = x_bar + bounds_vec

    if _is_infeasible(x_sub, lb_bal, ub_bal, tolerance, upper_bound):
        return None

    x0 = np.full(n_sub, 1.0 / n_sub)
    x0 = np.clip(x0, tolerance, upper_bound)

    constraints = []

    constraints.append(LinearConstraint(np.ones((1, n_sub)), lb=1.0, ub=1.0))

    p = x_sub.shape[1]
    if p > 0:
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

    # Status 4 marks a converged solve whose violation exceeds gtol. That limit is
    # far stricter than the direct check below.
    if result.status not in (1, 2, 4):
        return None

    gamma = result.x
    if _max_violation(gamma, x_sub, lb_bal, ub_bal, tolerance, upper_bound) > 1e-8:
        return None

    return gamma


def _is_infeasible(x_sub, lb_bal, ub_bal, lower, upper):
    """Return True when a linear program proves the balancing constraints infeasible."""
    a_ub = None
    b_ub = None
    if x_sub.shape[1] > 0:
        a_ub = np.vstack([x_sub.T, -x_sub.T])
        b_ub = np.concatenate([ub_bal, -lb_bal])

    n_sub = x_sub.shape[0]
    result = linprog(
        np.zeros(n_sub),
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=np.ones((1, n_sub)),
        b_eq=[1.0],
        bounds=(lower, upper),
        method="highs",
    )
    return result.status == 2


def _max_violation(gamma, x_sub, lb_bal, ub_bal, lower, upper):
    """Return the largest violation among the weight constraints."""
    violations = [
        abs(gamma.sum() - 1.0),
        np.max(lower - gamma, initial=0.0),
        np.max(gamma - upper, initial=0.0),
    ]
    if x_sub.shape[1] > 0:
        balance = x_sub.T @ gamma
        scale = 1.0 + np.maximum(np.abs(lb_bal), np.abs(ub_bal))
        violations.append(np.max(np.maximum(lb_bal - balance, balance - ub_bal) / scale, initial=0.0))
    return max(violations)


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
    rng,
):
    """Bootstrap debiasing correction."""
    # The full-sample fit stays out of the sum, since adding it would shift the bias estimate by a
    # twentieth of the coefficients.
    coef_accum = [np.zeros_like(c) for c in coef_t]

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
