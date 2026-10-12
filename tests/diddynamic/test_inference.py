"""Tests for variance estimation and inference."""

import numpy as np
import pytest
from scipy.stats import chi2

from moderndid.diddynamic.container import QuantileResult
from moderndid.diddynamic.estimation.inference import (
    compute_ate_variance,
    compute_quantiles,
    compute_variance,
    compute_variance_clustered,
)
from moderndid.diddynamic.estimation.weights_dcb import compute_dcb_estimator


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


def test_gaussian_at_001():
    result = compute_quantiles(0.01, 3, robust_quantile=True)
    assert np.isclose(result.gaussian_quantile_ate, 2.5758, atol=1e-3)


def test_robust_geq_gaussian():
    result = compute_quantiles(0.05, 3, robust_quantile=True)
    assert result.robust_quantile_ate >= result.gaussian_quantile_ate


@pytest.mark.parametrize("n_periods", [1, 2, 5, 10])
def test_positive(n_periods):
    result = compute_quantiles(0.05, n_periods, robust_quantile=True)
    assert result.robust_quantile_ate > 0


def test_equals_gaussian():
    result = compute_quantiles(0.05, 3, robust_quantile=False)
    assert np.isclose(result.robust_quantile_ate, result.gaussian_quantile_ate)


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
def test_tighter_alpha_gives_larger_quantile(alpha):
    result_tight = compute_quantiles(0.01, 3, robust_quantile=True)
    result_loose = compute_quantiles(0.10, 3, robust_quantile=True)
    assert result_tight.robust_quantile_ate > result_loose.robust_quantile_ate


def test_robust_ate_increases_with_periods():
    q2 = compute_quantiles(0.05, 2, robust_quantile=True)
    q5 = compute_quantiles(0.05, 5, robust_quantile=True)
    assert q5.robust_quantile_ate > q2.robust_quantile_ate


def test_quantiles_hold_only_the_ate_critical_values():
    result = compute_quantiles(0.05, 3, robust_quantile=True)
    assert result._fields == ("robust_quantile_ate", "gaussian_quantile_ate")
    assert result.robust_quantile_ate == pytest.approx(np.sqrt(chi2.ppf(0.95, 6)), rel=1e-12)


def test_variance_skips_a_row_missing_from_either_period():
    gammas = np.array([[0.5, 0.5], [0.5, 0.5]])
    predictions = np.array([[1.0, 2.0], [1.0, 4.0]])
    not_nas = [np.array([0, 1]), np.array([0])]
    y_t = np.array([3.0, 9.0])
    assert compute_variance(gammas, predictions, not_nas, y_t) == pytest.approx(0.5)


def test_ate_variance_subtracts_shared_period_covariance(two_history_results):
    first, second, y_t = two_history_results
    var_first = compute_variance(first.gammas, first.predictions, first.not_nas, y_t)
    var_second = compute_variance(second.gammas, second.predictions, second.not_nas, y_t)
    assert (var_first, var_second) == pytest.approx((2.0, 1.5))
    assert compute_ate_variance(first, second, y_t) == pytest.approx(2.5)


def test_ate_variance_adds_variances_when_weights_never_overlap(disjoint_history_results):
    first, second, y_t = disjoint_history_results
    assert compute_ate_variance(first, second, y_t) == pytest.approx(3.5)


def test_ate_variance_of_a_history_with_itself_is_zero(two_history_results):
    first, _, y_t = two_history_results
    assert compute_ate_variance(first, first, y_t) == 0.0


def test_clustered_ate_variance_sums_differences_within_clusters(two_history_results):
    first, second, y_t = two_history_results
    clusters = np.array([0, 0, 1, 1])
    var_first = compute_variance_clustered(first.gammas, first.predictions, first.not_nas, y_t, clusters)
    var_second = compute_variance_clustered(second.gammas, second.predictions, second.not_nas, y_t, clusters)
    assert (var_first, var_second) == pytest.approx((2.0, 1.0))
    assert compute_ate_variance(first, second, y_t, clusters) == pytest.approx(1.0)


def test_clustered_ate_variance_with_unit_clusters_matches_plain(two_history_results):
    first, second, y_t = two_history_results
    assert compute_ate_variance(first, second, y_t, np.arange(4)) == pytest.approx(2.5)


def test_clustered_ate_variance_keeps_dependence_between_histories_that_never_share_units(disjoint_history_results):
    first, second, y_t = disjoint_history_results
    clusters = np.array([0, 1, 0, 1])
    var_first = compute_variance_clustered(first.gammas, first.predictions, first.not_nas, y_t, clusters)
    var_second = compute_variance_clustered(second.gammas, second.predictions, second.not_nas, y_t, clusters)
    assert (var_first, var_second) == pytest.approx((2.0, 1.5))
    assert compute_ate_variance(first, second, y_t, clusters) == pytest.approx(2.5)


def test_lasso_plain_ate_variance_drops_shared_first_period_terms(estimation_panel):
    outcome, treatment, covariates, _ = estimation_panel
    kwargs = dict(
        method="lasso_plain", regularization=False, adaptive_balancing=False, nfolds=3, ub=20.0, grid_length=50
    )
    first = compute_dcb_estimator(3, outcome, treatment, covariates, np.array([0.0, 1.0, 1.0]), **kwargs)
    second = compute_dcb_estimator(3, outcome, treatment, covariates, np.array([0.0, 0.0, 0.0]), **kwargs)
    later_periods = np.array([0.0, 1.0, 1.0])
    trimmed = [r._replace(gammas=r.gammas * later_periods) for r in (first, second)]
    expected = sum(compute_variance(r.gammas, r.predictions, r.not_nas, outcome) for r in trimmed)
    np.testing.assert_array_equal(first.gammas[:, 0], second.gammas[:, 0])
    assert compute_ate_variance(first, second, outcome) == pytest.approx(expected, rel=1e-10)
