"""Tests for variance estimation and inference."""

import numpy as np
import pytest

from moderndid.diddynamic.estimation.inference import (
    QuantileResult,
    compute_quantiles,
    compute_variance,
    compute_variance_clustered,
)


def test_nonnegative(multi_period_data):
    gammas, predictions, not_nas, y_t = multi_period_data
    var = compute_variance(gammas, predictions, not_nas, y_t)
    assert var >= 0.0


def test_finite(multi_period_data):
    gammas, predictions, not_nas, y_t = multi_period_data
    var = compute_variance(gammas, predictions, not_nas, y_t)
    assert np.isfinite(var)


def test_equals_final_term(single_period_data):
    gammas, predictions, not_nas, y_t = single_period_data
    var = compute_variance(gammas, predictions, not_nas, y_t)
    epsilon = y_t - predictions[:, 0]
    expected = float((gammas[:, 0] ** 2) @ (epsilon**2))
    assert np.isclose(var, expected)


def test_zero_residuals_give_zero_variance():
    n = 20
    n_periods = 2
    gammas = np.zeros((n, n_periods))
    gammas[:10, :] = 1.0 / 10
    predictions = np.ones((n, n_periods)) * 3.0
    not_nas = [np.arange(n)] * n_periods
    y_t = np.full(n, 3.0)
    var = compute_variance(gammas, predictions, not_nas, y_t)
    assert np.isclose(var, 0.0, atol=1e-15)


def test_zero_gammas_give_zero_variance():
    n = 20
    n_periods = 2
    gammas = np.zeros((n, n_periods))
    predictions = np.random.default_rng(42).standard_normal((n, n_periods))
    not_nas = [np.arange(n)] * n_periods
    y_t = np.random.default_rng(42).standard_normal(n)
    var = compute_variance(gammas, predictions, not_nas, y_t)
    assert np.isclose(var, 0.0, atol=1e-15)


def test_manual_single_unit_variance():
    gammas = np.array([[0.5], [0.5]])
    predictions = np.array([[3.0], [4.0]])
    not_nas = [np.array([0, 1])]
    y_t = np.array([3.5, 4.5])
    var = compute_variance(gammas, predictions, not_nas, y_t)
    epsilon = y_t - predictions[:, 0]
    expected = float((gammas[:, 0] ** 2) @ (epsilon**2))
    assert var == pytest.approx(expected)


@pytest.mark.parametrize("n_periods", [1, 3, 5])
def test_nonneg(rng, n_periods):
    n = 40
    gammas = np.zeros((n, n_periods))
    gammas[:20, :] = 1.0 / 20
    predictions = rng.standard_normal((n, n_periods))
    not_nas = [np.arange(n)] * n_periods
    y_t = rng.standard_normal(n)
    var = compute_variance(gammas, predictions, not_nas, y_t)
    assert var >= 0.0


def test_partial_not_nas(rng):
    n = 30
    n_periods = 3
    gammas = np.zeros((n, n_periods))
    gammas[:15, :] = 1.0 / 15
    predictions = rng.standard_normal((n, n_periods))
    not_nas = [np.arange(20), np.arange(15, 30), np.arange(10, 25)]
    y_t = rng.standard_normal(n)
    var = compute_variance(gammas, predictions, not_nas, y_t)
    assert var >= 0.0
    assert np.isfinite(var)


def test_variance_clustered_nonnegative(multi_period_data):
    gammas, predictions, not_nas, y_t = multi_period_data
    n = gammas.shape[0]
    clusters = np.repeat(np.arange(5), n // 5)
    var = compute_variance_clustered(gammas, predictions, not_nas, y_t, clusters)
    assert var >= 0.0


def test_unit_clusters_matches_plain(multi_period_data):
    gammas, predictions, not_nas, y_t = multi_period_data
    n = gammas.shape[0]
    unit_clusters = np.arange(n)
    var_clustered = compute_variance_clustered(gammas, predictions, not_nas, y_t, unit_clusters)
    var_plain = compute_variance(gammas, predictions, not_nas, y_t)
    assert np.isclose(var_clustered, var_plain)


def test_single_period_nonneg(single_period_data):
    gammas, predictions, not_nas, y_t = single_period_data
    n = gammas.shape[0]
    clusters = np.repeat(np.arange(4), n // 4)
    var = compute_variance_clustered(gammas, predictions, not_nas, y_t, clusters)
    assert var >= 0.0


def test_matches_manual_computation():
    n = 10
    gammas = np.zeros((n, 1))
    gammas[:, 0] = 0.1
    predictions = np.zeros((n, 1))
    not_nas = [np.arange(n)]
    y_t = np.arange(n, dtype=float)
    clusters = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    epsilon = y_t - predictions[:, 0]
    c0_sum = 0.1 * epsilon[:5].sum()
    c1_sum = 0.1 * epsilon[5:].sum()
    expected = c0_sum**2 + c1_sum**2

    var = compute_variance_clustered(gammas, predictions, not_nas, y_t, clusters)
    assert np.isclose(var, expected)


def test_gaussian_at_005():
    result = compute_quantiles(0.05, 3, robust_quantile=True)
    assert isinstance(result, QuantileResult)
    assert np.isclose(result.gaussian_quantile_ate, 1.959964, atol=1e-4)
    assert np.isclose(result.gaussian_quantile_mu, 1.959964, atol=1e-4)


def test_gaussian_at_001():
    result = compute_quantiles(0.01, 3, robust_quantile=True)
    assert np.isclose(result.gaussian_quantile_ate, 2.5758, atol=1e-3)


def test_robust_geq_gaussian():
    result = compute_quantiles(0.05, 3, robust_quantile=True)
    assert result.robust_quantile_ate >= result.gaussian_quantile_ate
    assert result.robust_quantile_mu >= result.gaussian_quantile_mu


def test_robust_ate_gt_robust_mu():
    result = compute_quantiles(0.05, 3, robust_quantile=True)
    assert result.robust_quantile_ate > result.robust_quantile_mu


@pytest.mark.parametrize("n_periods", [1, 2, 5, 10])
def test_positive(n_periods):
    result = compute_quantiles(0.05, n_periods, robust_quantile=True)
    assert result.robust_quantile_ate > 0
    assert result.robust_quantile_mu > 0


def test_equals_gaussian():
    result = compute_quantiles(0.05, 3, robust_quantile=False)
    assert np.isclose(result.robust_quantile_ate, result.gaussian_quantile_ate)
    assert np.isclose(result.robust_quantile_mu, result.gaussian_quantile_mu)


@pytest.mark.parametrize("alp", [0.01, 0.05, 0.10])
def test_tighter_alpha_gives_larger_quantile(alp):
    result_tight = compute_quantiles(0.01, 3, robust_quantile=True)
    result_loose = compute_quantiles(0.10, 3, robust_quantile=True)
    assert result_tight.robust_quantile_ate > result_loose.robust_quantile_ate


def test_robust_ate_increases_with_periods():
    q2 = compute_quantiles(0.05, 2, robust_quantile=True)
    q5 = compute_quantiles(0.05, 5, robust_quantile=True)
    assert q5.robust_quantile_ate > q2.robust_quantile_ate


def test_robust_mu_increases_with_periods():
    q2 = compute_quantiles(0.05, 2, robust_quantile=True)
    q5 = compute_quantiles(0.05, 5, robust_quantile=True)
    assert q5.robust_quantile_mu > q2.robust_quantile_mu
