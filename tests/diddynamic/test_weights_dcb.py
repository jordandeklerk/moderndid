"""Tests for DCB balancing weights via quadratic programming."""

import numpy as np
import pytest

from moderndid.dev.diddynamic.estimation.weights_dcb import (
    DCBResult,
    _build_balance_bounds,
    _solve_balance_qp,
    _solve_qp_first_period,
    _solve_qp_sequential,
    compute_dcb_estimator,
)


def test_qp_solution_sums_to_one(simple_qp_data):
    x_all, d_col, coef = simple_qp_data
    result = _solve_qp_first_period(
        x_all=x_all,
        d_col=d_col,
        d_target=1.0,
        k1=5.0,
        k2=5.0,
        with_beta=False,
        coef=coef,
        n_beta_nonsparse=1e-4,
        ratio_coefficients=1 / 3,
        tolerance=1e-8,
    )
    assert result is not None
    assert np.isclose(result.sum(), 1.0, atol=1e-4)


def test_qp_solution_nonneg(simple_qp_data):
    x_all, d_col, coef = simple_qp_data
    result = _solve_qp_first_period(
        x_all=x_all,
        d_col=d_col,
        d_target=1.0,
        k1=5.0,
        k2=5.0,
        with_beta=False,
        coef=coef,
        n_beta_nonsparse=1e-4,
        ratio_coefficients=1 / 3,
        tolerance=1e-8,
    )
    assert result is not None
    assert np.all(result >= -1e-8)


def test_qp_solution_correct_shape(simple_qp_data):
    x_all, d_col, coef = simple_qp_data
    result = _solve_qp_first_period(
        x_all=x_all,
        d_col=d_col,
        d_target=1.0,
        k1=5.0,
        k2=5.0,
        with_beta=False,
        coef=coef,
        n_beta_nonsparse=1e-4,
        ratio_coefficients=1 / 3,
        tolerance=1e-8,
    )
    assert result is not None
    assert result.shape == (x_all.shape[0],)
    assert np.all(result[d_col != 1.0] == 0.0)


@pytest.mark.parametrize("with_beta", [True, False])
def test_balance_bounds_shape(with_beta):
    p = 5
    coef = np.array([0.1, 0.5, 0.0, 0.3, 0.0, 0.1])
    bounds = _build_balance_bounds(
        p, tight=0.1, loose=1.0, with_beta=with_beta, beta=coef, n_beta_nonsparse=1e-4, ratio_coefficients=1 / 3
    )
    assert bounds.shape == (p,)


def test_balance_bounds_adaptive_tight_loose():
    p = 5
    coef = np.array([0.1, 0.5, 0.0, 0.3, 0.0, 0.1])
    bounds = _build_balance_bounds(
        p, tight=0.1, loose=1.0, with_beta=True, beta=coef, n_beta_nonsparse=1e-4, ratio_coefficients=1 / 3
    )
    zero_idx = np.where(np.abs(coef[1:]) <= 1e-4)[0]
    nonzero_idx = np.where(np.abs(coef[1:]) > 1e-4)[0]
    assert np.all(bounds[zero_idx] == 1.0)
    assert np.all(bounds[nonzero_idx] == 0.1)


def test_balance_bounds_non_adaptive_all_tight():
    p = 5
    coef = np.array([0.1, 0.5, 0.0, 0.3, 0.0, 0.1])
    bounds = _build_balance_bounds(
        p, tight=0.1, loose=1.0, with_beta=False, beta=coef, n_beta_nonsparse=1e-4, ratio_coefficients=1 / 3
    )
    assert np.all(bounds == 0.1)


def test_solve_balance_qp_feasible(rng):
    n_sub = 15
    p = 2
    x_sub = rng.standard_normal((n_sub, p))
    x_bar = x_sub.mean(axis=0)
    bounds_vec = np.full(p, 5.0)
    result = _solve_balance_qp(x_sub, x_bar, n_sub, bounds_vec, tolerance=1e-8)
    assert result is not None
    assert np.isclose(result.sum(), 1.0, atol=1e-4)
    assert np.all(result >= -1e-8)


def test_solve_balance_qp_infeasible(rng):
    n_sub = 5
    p = 2
    x_sub = rng.standard_normal((n_sub, p)) + 100
    x_bar = np.zeros(p)
    bounds_vec = np.full(p, 1e-10)
    result = _solve_balance_qp(x_sub, x_bar, n_sub, bounds_vec, tolerance=1e-8)
    assert result is None


def test_sequential_weights_respect_previous(rng):
    n = 30
    p = 2
    x_all = rng.standard_normal((n, p))
    d_mat = np.zeros((n, 2))
    d_mat[:15, 0] = 1.0
    d_mat[:15, 1] = 1.0
    d_target = np.array([1.0, 1.0])
    gamma_prev = np.zeros(n)
    gamma_prev[:15] = 1.0 / 15
    coef = np.zeros(p + 1)
    result = _solve_qp_sequential(
        gamma_prev=gamma_prev,
        x_all=x_all,
        d_mat=d_mat,
        d_target=d_target,
        k1=5.0,
        k2=5.0,
        with_beta=False,
        coef=coef,
        n_beta_nonsparse=1e-4,
        ratio_coefficients=1 / 3,
        tolerance=1e-8,
    )
    assert result is not None
    assert np.isclose(result.sum(), 1.0, atol=1e-4)


def test_sequential_weights_nonneg(rng):
    n = 30
    p = 2
    x_all = rng.standard_normal((n, p))
    d_mat = np.zeros((n, 2))
    d_mat[:15, :] = 1.0
    d_target = np.array([1.0, 1.0])
    gamma_prev = np.zeros(n)
    gamma_prev[:15] = 1.0 / 15
    coef = np.zeros(p + 1)
    result = _solve_qp_sequential(
        gamma_prev=gamma_prev,
        x_all=x_all,
        d_mat=d_mat,
        d_target=d_target,
        k1=5.0,
        k2=5.0,
        with_beta=False,
        coef=coef,
        n_beta_nonsparse=1e-4,
        ratio_coefficients=1 / 3,
        tolerance=1e-8,
    )
    assert result is not None
    assert np.all(result >= -1e-8)


@pytest.mark.parametrize("adaptive", [True, False])
def test_dcb_estimator_returns_result(estimation_panel, adaptive):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_dcb_estimator(
        3,
        outcome,
        treatment,
        covariates,
        ds,
        method="lasso_subsample",
        adaptive_balancing=adaptive,
        nfolds=3,
        ub=20.0,
        grid_length=50,
        tolerance=1e-8,
    )
    assert isinstance(result, DCBResult)


def test_dcb_estimator_mu_hat_finite(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_dcb_estimator(
        3,
        outcome,
        treatment,
        covariates,
        ds,
        method="lasso_subsample",
        adaptive_balancing=False,
        nfolds=3,
        ub=20.0,
        grid_length=50,
        tolerance=1e-8,
    )
    assert np.isfinite(result.mu_hat)


def test_dcb_estimator_gammas_shape(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_dcb_estimator(
        3,
        outcome,
        treatment,
        covariates,
        ds,
        method="lasso_subsample",
        adaptive_balancing=False,
        nfolds=3,
        ub=20.0,
        grid_length=50,
        tolerance=1e-8,
    )
    assert result.gammas.shape == (60, 3)


def test_dcb_estimator_predictions_shape(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_dcb_estimator(
        3,
        outcome,
        treatment,
        covariates,
        ds,
        method="lasso_subsample",
        adaptive_balancing=False,
        nfolds=3,
        ub=20.0,
        grid_length=50,
        tolerance=1e-8,
    )
    assert result.predictions.shape == (60, 3)


def test_dcb_estimator_gammas_sum_to_one(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_dcb_estimator(
        3,
        outcome,
        treatment,
        covariates,
        ds,
        method="lasso_subsample",
        adaptive_balancing=False,
        nfolds=3,
        ub=20.0,
        grid_length=50,
        tolerance=1e-8,
    )
    last = 3 - 1
    valid = result.not_nas[last]
    assert np.isclose(result.gammas[valid, last].sum(), 1.0, atol=1e-4)


def test_dcb_estimator_gammas_nonneg(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_dcb_estimator(
        3,
        outcome,
        treatment,
        covariates,
        ds,
        method="lasso_subsample",
        adaptive_balancing=False,
        nfolds=3,
        ub=20.0,
        grid_length=50,
        tolerance=1e-8,
    )
    assert np.all(result.gammas >= -1e-8)


def test_dcb_estimator_bias_nan_when_no_debias(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_dcb_estimator(
        3,
        outcome,
        treatment,
        covariates,
        ds,
        method="lasso_subsample",
        adaptive_balancing=False,
        nfolds=3,
        ub=20.0,
        grid_length=50,
        tolerance=1e-8,
    )
    assert np.isnan(result.bias)


def test_dcb_estimator_fast_adaptive(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_dcb_estimator(
        3,
        outcome,
        treatment,
        covariates,
        ds,
        method="lasso_subsample",
        adaptive_balancing=True,
        fast_adaptive=True,
        nfolds=3,
        ub=20.0,
        grid_length=50,
        tolerance=1e-8,
    )
    assert isinstance(result, DCBResult)
    assert np.isfinite(result.mu_hat)


def test_dcb_estimator_lasso_plain(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_dcb_estimator(
        3,
        outcome,
        treatment,
        covariates,
        ds,
        method="lasso_plain",
        adaptive_balancing=False,
        nfolds=3,
        ub=20.0,
        grid_length=50,
        tolerance=1e-8,
    )
    assert isinstance(result, DCBResult)
    assert np.isfinite(result.mu_hat)


def test_dcb_estimator_infeasible_raises(rng):
    n = 20
    treatment = np.zeros((n, 2))
    treatment[:10, :] = 1.0
    x0 = np.concatenate([np.full(10, 1000.0), np.full(10, -1000.0)])
    covariates = {t: np.column_stack([x0, x0]) for t in range(2)}
    outcome = rng.standard_normal(n)
    ds = np.array([1.0, 1.0])
    with pytest.raises(RuntimeError, match="Infeasible"):
        compute_dcb_estimator(
            2,
            outcome,
            treatment,
            covariates,
            ds,
            adaptive_balancing=False,
            nfolds=3,
            ub=1e-8,
            grid_length=3,
            tolerance=1e-8,
        )


@pytest.mark.parametrize("fast", [True, False])
def test_grid_search_finds_feasible(estimation_panel, fast):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_dcb_estimator(
        3,
        outcome,
        treatment,
        covariates,
        ds,
        method="lasso_subsample",
        adaptive_balancing=False,
        fast_adaptive=fast,
        nfolds=3,
        ub=20.0,
        grid_length=50,
        tolerance=1e-8,
    )
    for t in range(3):
        valid = result.not_nas[t]
        gamma_valid = result.gammas[valid, t]
        if gamma_valid.sum() > 0:
            assert np.isclose(gamma_valid.sum(), 1.0, atol=1e-3)


def test_gammas_zero_for_non_matching_units(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_dcb_estimator(
        3,
        outcome,
        treatment,
        covariates,
        ds,
        method="lasso_subsample",
        adaptive_balancing=False,
        nfolds=3,
        ub=20.0,
        grid_length=50,
        tolerance=1e-8,
    )
    for t in range(3):
        matching = np.all(treatment[:, : t + 1] == ds[: t + 1], axis=1)
        non_matching = ~matching
        assert np.all(result.gammas[non_matching, t] == 0.0)


def test_single_period_dcb(rng):
    n = 40
    treatment = np.zeros((n, 1))
    treatment[:20, 0] = 1.0
    covariates = {0: rng.standard_normal((n, 2))}
    outcome = rng.standard_normal(n) + 2.0 * treatment[:, 0]
    ds = np.array([1.0])
    result = compute_dcb_estimator(
        1,
        outcome,
        treatment,
        covariates,
        ds,
        method="lasso_subsample",
        adaptive_balancing=False,
        nfolds=3,
        ub=20.0,
        grid_length=50,
        tolerance=1e-8,
    )
    assert isinstance(result, DCBResult)
    assert np.isfinite(result.mu_hat)
    assert result.gammas.shape == (n, 1)


def test_small_n_weight_bounds(rng):
    n = 8
    treatment = np.zeros((n, 2))
    treatment[:4, :] = 1.0
    covariates = {t: rng.standard_normal((n, 2)) for t in range(2)}
    outcome = rng.standard_normal(n)
    ds = np.array([1.0, 1.0])
    result = compute_dcb_estimator(
        2,
        outcome,
        treatment,
        covariates,
        ds,
        method="lasso_subsample",
        adaptive_balancing=False,
        nfolds=2,
        ub=20.0,
        grid_length=50,
        tolerance=1e-8,
    )
    assert isinstance(result, DCBResult)
    upper_bound = np.log(4) * 4 ** (-2 / 3)
    for t in range(2):
        valid = result.not_nas[t]
        assert np.all(result.gammas[valid, t] <= upper_bound + 1e-6)
