"""Tests for DCB balancing weights via quadratic programming."""

import numpy as np
import pytest

from moderndid.diddynamic.container import DCBResult
from moderndid.diddynamic.estimation.coefficients import compute_coefficients
from moderndid.diddynamic.estimation.weights_dcb import (
    _build_balance_bounds,
    _compute_bias,
    _is_infeasible,
    _max_violation,
    _solve_balance_qp,
    _solve_qp_first_period,
    _solve_qp_sequential,
    compute_dcb_estimator,
    compute_imbalances,
)


def test_sums_to_one(simple_qp_data):
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


def test_nonnegative(simple_qp_data):
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


def test_correct_shape(simple_qp_data):
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


def test_zero_for_non_matching(simple_qp_data):
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
    assert np.all(result[d_col != 1.0] == 0.0)


@pytest.mark.parametrize("with_beta", [True, False])
def test_shape(with_beta):
    p = 5
    coef = np.array([0.1, 0.5, 0.0, 0.3, 0.0, 0.1])
    bounds = _build_balance_bounds(
        p,
        tight=0.1,
        loose=1.0,
        with_beta=with_beta,
        beta=coef,
        n_beta_nonsparse=1e-4,
        ratio_coefficients=1 / 3,
    )
    assert bounds.shape == (p,)


def test_adaptive_tight_for_nonzero_coefs():
    p = 5
    coef = np.array([0.1, 0.5, 0.0, 0.3, 0.0, 0.1])
    bounds = _build_balance_bounds(
        p,
        tight=0.1,
        loose=1.0,
        with_beta=True,
        beta=coef,
        n_beta_nonsparse=1e-4,
        ratio_coefficients=1 / 3,
    )
    zero_idx = np.where(np.abs(coef[1:]) <= 1e-4)[0]
    nonzero_idx = np.where(np.abs(coef[1:]) > 1e-4)[0]
    assert np.all(bounds[zero_idx] == 1.0)
    assert np.all(bounds[nonzero_idx] == 0.1)


def test_non_adaptive_all_tight():
    p = 5
    coef = np.array([0.1, 0.5, 0.0, 0.3, 0.0, 0.1])
    bounds = _build_balance_bounds(
        p,
        tight=0.1,
        loose=1.0,
        with_beta=False,
        beta=coef,
        n_beta_nonsparse=1e-4,
        ratio_coefficients=1 / 3,
    )
    assert np.all(bounds == 0.1)


def test_feasible_sums_to_one(rng):
    n_sub = 15
    p = 2
    x_sub = rng.standard_normal((n_sub, p))
    x_bar = x_sub.mean(axis=0)
    bounds_vec = np.full(p, 5.0)
    result = _solve_balance_qp(x_sub, x_bar, n_sub, bounds_vec, tolerance=1e-8)
    assert result is not None
    assert np.isclose(result.sum(), 1.0, atol=1e-4)
    assert np.all(result >= -1e-8)


@pytest.mark.filterwarnings("ignore:Singular Jacobian:UserWarning")
def test_infeasible_returns_none(rng):
    n_sub = 5
    p = 2
    x_sub = rng.standard_normal((n_sub, p)) + 100
    x_bar = np.zeros(p)
    bounds_vec = np.full(p, 1e-10)
    result = _solve_balance_qp(x_sub, x_bar, n_sub, bounds_vec, tolerance=1e-8)
    assert result is None


def test_sequential_sums_to_one(rng):
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
    assert np.isclose(result.sum(), 1.0, atol=1e-4)


def test_sequential_nonnegative(rng):
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
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_returns_dcb_result(estimation_panel, adaptive):
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


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_mu_hat_finite(estimation_panel):
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


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_gammas_shape(estimation_panel):
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


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_predictions_shape(estimation_panel):
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


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_gammas_sum_to_one(estimation_panel):
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
    last = 2
    valid = result.not_nas[last]
    assert np.isclose(result.gammas[valid, last].sum(), 1.0, atol=1e-4)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_gammas_nonneg(estimation_panel):
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


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_bias_nan_when_no_debias(estimation_panel):
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


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
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


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_fast_adaptive(estimation_panel):
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


def test_lasso_plain_method(estimation_panel):
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


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.filterwarnings("ignore:Singular Jacobian:UserWarning")
def test_raises_runtime_error(rng):
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
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_finds_feasible(estimation_panel, fast):
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


def test_single_period(rng):
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


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_weight_bounds_respected(rng):
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


def test_uniform_weights_for_identical_covariates(rng):
    n = 20
    treatment = np.zeros((n, 1))
    treatment[:10, 0] = 1.0
    covariates = {0: np.ones((n, 2))}
    outcome = rng.standard_normal(n)
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
    valid = result.not_nas[0]
    treated_gammas = result.gammas[valid, 0]
    treated_gammas = treated_gammas[treated_gammas > 0]
    assert np.allclose(treated_gammas, treated_gammas[0], atol=1e-3)


def test_mu_hat_approximates_conditional_mean(rng):
    n = 60
    treatment = np.zeros((n, 1))
    treatment[:30, 0] = 1.0
    covariates = {0: rng.standard_normal((n, 2)) * 0.01}
    outcome = np.full(n, 4.0) + rng.standard_normal(n) * 0.01
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
    assert result.mu_hat == pytest.approx(4.0, abs=0.5)


def test_linear_program_flags_infeasible_balance(rng):
    x_sub = rng.standard_normal((5, 2)) + 100
    bounds = np.full(2, 1e-10)
    upper = np.log(5) * 5 ** (-2 / 3)
    assert _is_infeasible(x_sub, -bounds, bounds, 1e-8, upper)


def test_linear_program_accepts_feasible_balance(rng):
    x_sub = rng.standard_normal((15, 2))
    x_bar = x_sub.mean(axis=0)
    upper = np.log(15) * 15 ** (-2 / 3)
    assert not _is_infeasible(x_sub, x_bar - 5.0, x_bar + 5.0, 1e-8, upper)


def test_accepted_weights_satisfy_constraints(rng):
    x_sub = rng.standard_normal((15, 2))
    x_bar = x_sub.mean(axis=0)
    bounds_vec = np.full(2, 0.5)
    gamma = _solve_balance_qp(x_sub, x_bar, 15, bounds_vec, tolerance=1e-8)
    upper = np.log(15) * 15 ** (-2 / 3)
    assert gamma is not None
    assert _max_violation(gamma, x_sub, x_bar - bounds_vec, x_bar + bounds_vec, 1e-8, upper) <= 1e-8


def test_max_violation_measures_balance_gap(rng):
    x_sub = rng.standard_normal((10, 2))
    gamma = np.full(10, 0.1)
    balance = x_sub.T @ gamma
    upper = np.log(10) * 10 ** (-2 / 3)
    violation = _max_violation(gamma, x_sub, balance + 1.0, balance + 2.0, 1e-8, upper)
    assert violation == pytest.approx(np.max(1.0 / (1.0 + np.abs(balance + 2.0))))


def test_fewer_than_two_units_is_infeasible(rng):
    x_sub = rng.standard_normal((1, 2))
    assert _solve_balance_qp(x_sub, x_sub[0], 1, np.full(2, 5.0), tolerance=1e-8) is None


def test_bias_vanishes_when_every_resample_refits_the_same_coefficients(exact_linear_panel, rng):
    outcome, treatment, covariates, ds = exact_linear_panel
    n = len(outcome)
    coefs = compute_coefficients(2, outcome, treatment, covariates, ds, "lasso_plain", False, 10)
    gammas = rng.dirichlet(np.ones(n), size=2).T
    bias = _compute_bias(
        n,
        2,
        outcome,
        treatment,
        covariates,
        ds,
        "lasso_plain",
        False,
        10,
        None,
        0,
        coefs.coef_t,
        coefs.covariates_nonna,
        coefs.not_nas,
        gammas,
        np.random.default_rng(0),
    )
    assert abs(bias) < 1e-6


def test_debias_depends_only_on_random_state(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    kwargs = dict(
        method="lasso_plain", regularization=False, debias=True, adaptive_balancing=False, ub=20.0, grid_length=50
    )
    first = compute_dcb_estimator(3, outcome, treatment, covariates, ds, random_state=3, **kwargs)
    second = compute_dcb_estimator(3, outcome, treatment, covariates, ds, random_state=3, **kwargs)
    assert np.isfinite(first.bias)
    assert first.bias == second.bias


def test_imbalances_compare_with_equal_then_previous_weights(rng):
    n = 30
    covariates = {t: rng.standard_normal((n, 2)) for t in range(2)}
    gammas = rng.dirichlet(np.ones(n), size=2).T
    not_nas = [np.arange(n), np.arange(n)]
    result = compute_imbalances(gammas, not_nas, covariates)
    first = (gammas[:, 0] - 1 / n) @ covariates[0] / covariates[0].std(axis=0, ddof=1)
    second = (gammas[:, 1] - gammas[:, 0]) @ covariates[1] / covariates[1].std(axis=0, ddof=1)
    np.testing.assert_allclose(result, np.vstack([first, second]), rtol=1e-12)


def test_imbalances_skip_missing_rows_and_keep_constant_columns_raw(rng):
    n = 30
    x = np.column_stack([rng.standard_normal(n), np.full(n, 2.0)])
    x[0, 0] = np.nan
    gammas = np.zeros((n, 1))
    gammas[1:, 0] = rng.dirichlet(np.ones(n - 1))
    result = compute_imbalances(gammas, [np.arange(n)], {0: x})
    expected = (gammas[1:, 0] - 1 / (n - 1)) @ x[1:, 0] / x[1:, 0].std(ddof=1)
    assert result[0, 0] == pytest.approx(expected, rel=1e-12)
    assert result[0, 1] == pytest.approx(0.0, abs=1e-12)
