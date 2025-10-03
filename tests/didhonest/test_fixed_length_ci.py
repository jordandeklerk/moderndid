# pylint: disable=redefined-outer-name
"""Tests for FLCI (Fixed-Length Confidence Intervals) module."""

import numpy as np
import pytest

from moderndid.didhonest.fixed_length_ci import (
    FLCIResult,
    _optimize_flci_params,
    _optimize_h_bisection,
    affine_variance,
    compute_flci,
    folded_normal_quantile,
    get_min_bias_h,
    maximize_bias,
    minimize_variance,
)


@pytest.fixture
def test_data():
    np.random.seed(42)
    n_pre = 4
    n_post = 3
    beta_hat = np.random.randn(n_pre + n_post)
    A = np.random.randn(n_pre + n_post, n_pre + n_post)
    sigma = 0.01 * (A @ A.T) + 0.001 * np.eye(n_pre + n_post)
    post_period_weights = np.zeros(n_post)
    post_period_weights[0] = 1
    return beta_hat, sigma, post_period_weights, n_pre, n_post


@pytest.fixture
def test_data_custom(request):
    n_pre, n_post, seed = request.param
    np.random.seed(seed)
    beta_hat = np.random.randn(n_pre + n_post)
    A = np.random.randn(n_pre + n_post, n_pre + n_post)
    sigma = 0.01 * (A @ A.T) + 0.001 * np.eye(n_pre + n_post)
    post_period_weights = np.zeros(n_post)
    post_period_weights[0] = 1
    return beta_hat, sigma, post_period_weights, n_pre, n_post


def test_compute_flci_basic(test_data):
    beta_hat, sigma, post_period_weights, n_pre, n_post = test_data

    result = compute_flci(
        beta_hat=beta_hat,
        sigma=sigma,
        smoothness_bound=0.5,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        post_period_weights=post_period_weights,
        alpha=0.05,
    )

    assert isinstance(result, FLCIResult)
    assert len(result.flci) == 2
    assert result.flci[0] < result.flci[1]
    assert result.optimal_half_length > 0
    assert result.smoothness_bound == 0.5
    assert result.status == "optimal"
    assert len(result.optimal_vec) == n_pre + n_post
    assert len(result.optimal_pre_period_vec) == n_pre


def test_flci_with_default_l_vec(test_data):
    beta_hat, sigma, _, n_pre, n_post = test_data

    result = compute_flci(
        beta_hat=beta_hat,
        sigma=sigma,
        smoothness_bound=1.0,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        alpha=0.05,
    )

    assert isinstance(result, FLCIResult)
    expected_post_period_weights = np.array([1, 0, 0])
    np.testing.assert_array_equal(result.optimal_vec[n_pre:], expected_post_period_weights)


@pytest.mark.parametrize("alpha", [0.05])
def test_flci_different_alpha_levels(test_data, alpha):
    beta_hat, sigma, post_period_weights, n_pre, n_post = test_data

    result = compute_flci(
        beta_hat=beta_hat,
        sigma=sigma,
        smoothness_bound=1.0,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        post_period_weights=post_period_weights,
        alpha=alpha,
    )

    assert isinstance(result, FLCIResult)
    assert result.optimal_half_length > 0


def test_flci_alpha_ordering(test_data):
    beta_hat, sigma, post_period_weights, n_pre, n_post = test_data

    results = {}
    for alpha in [0.01, 0.05, 0.10]:
        result = compute_flci(
            beta_hat=beta_hat,
            sigma=sigma,
            smoothness_bound=1.0,
            n_pre_periods=n_pre,
            n_post_periods=n_post,
            post_period_weights=post_period_weights,
            alpha=alpha,
        )
        results[alpha] = result.optimal_half_length

    assert results[0.01] > results[0.05]
    assert results[0.05] > results[0.10]


@pytest.mark.parametrize("m", [0.5, 1.5])
def test_flci_different_m_values(test_data, m):
    beta_hat, sigma, post_period_weights, n_pre, n_post = test_data

    result = compute_flci(
        beta_hat=beta_hat,
        sigma=sigma,
        smoothness_bound=m,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        post_period_weights=post_period_weights,
        alpha=0.05,
    )

    assert isinstance(result, FLCIResult)
    assert result.optimal_half_length > 0


def test_flci_m_ordering(test_data):
    beta_hat, sigma, post_period_weights, n_pre, n_post = test_data

    results = {}
    for m in [0.5, 1.0, 2.0]:
        result = compute_flci(
            beta_hat=beta_hat,
            sigma=sigma,
            smoothness_bound=m,
            n_pre_periods=n_pre,
            n_post_periods=n_post,
            post_period_weights=post_period_weights,
            alpha=0.05,
        )
        results[m] = result.optimal_half_length

    assert results[2.0] > results[1.0]
    assert results[1.0] > results[0.5]


def test_maximize_bias_basic(test_data):
    _, sigma, post_period_weights, n_pre, n_post = test_data

    result = maximize_bias(
        h=1.0,
        sigma=sigma,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        post_period_weights=post_period_weights,
        smoothness_bound=1.0,
    )

    assert result["status"] == "optimal"
    assert result["value"] >= 0
    assert result["optimal_w"] is not None
    assert len(result["optimal_w"]) == n_pre
    assert result["optimal_l"] is not None
    assert len(result["optimal_l"]) == n_pre


@pytest.mark.parametrize("h", [0.5, 1.5])
def test_maximize_bias_different_h_values(test_data, h):
    _, sigma, post_period_weights, n_pre, n_post = test_data

    result = maximize_bias(
        h=h,
        sigma=sigma,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        post_period_weights=post_period_weights,
        smoothness_bound=1.0,
    )

    assert result["status"] == "optimal", f"Optimization failed for h={h} with status: {result['status']}"
    assert result["value"] >= 0


def test_maximize_bias_h_ordering(test_data):
    _, sigma, post_period_weights, n_pre, n_post = test_data

    biases = {}
    for h in [0.5, 1.0, 2.0]:
        result = maximize_bias(
            h=h,
            sigma=sigma,
            n_pre_periods=n_pre,
            n_post_periods=n_post,
            post_period_weights=post_period_weights,
            smoothness_bound=1.0,
        )
        biases[h] = result["value"]

    assert biases[2.0] >= biases[1.0]
    assert biases[1.0] >= biases[0.5]


def test_maximize_bias_weight_sum_constraint(test_data):
    _, sigma, post_period_weights, n_pre, n_post = test_data

    result = maximize_bias(
        h=1.5,
        sigma=sigma,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        post_period_weights=post_period_weights,
        smoothness_bound=1.0,
    )

    if result["status"] == "optimal":
        target_sum = np.dot(np.arange(1, n_post + 1), post_period_weights)
        actual_sum = np.sum(result["optimal_w"])
        np.testing.assert_allclose(actual_sum, target_sum, rtol=1e-5)


def test_affine_variance(test_data):
    _, sigma, post_period_weights, n_pre, _ = test_data

    l_pre = np.array([0.1, 0.2, 0.3, 0.4])

    variance = affine_variance(
        l_pre=l_pre,
        l_post=post_period_weights,
        sigma=sigma,
        n_pre_periods=n_pre,
    )

    assert variance > 0
    assert np.isfinite(variance)


def test_minimize_variance(test_data):
    _, sigma, post_period_weights, n_pre, n_post = test_data

    h_min = minimize_variance(
        sigma=sigma,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        post_period_weights=post_period_weights,
    )

    assert h_min > 0
    assert np.isfinite(h_min)


def test_get_min_bias_h(test_data):
    _, sigma, post_period_weights, n_pre, n_post = test_data

    h_min_bias = get_min_bias_h(
        sigma=sigma,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        post_period_weights=post_period_weights,
    )

    assert h_min_bias > 0
    assert np.isfinite(h_min_bias)


def test_h_ordering(test_data):
    _, sigma, post_period_weights, n_pre, n_post = test_data

    h_min_var = minimize_variance(
        sigma=sigma,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        post_period_weights=post_period_weights,
    )

    h_min_bias = get_min_bias_h(
        sigma=sigma,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        post_period_weights=post_period_weights,
    )

    assert h_min_var <= h_min_bias + 1e-6


@pytest.mark.parametrize(
    "p,mu,sd,min_val,max_val",
    [
        (0.95, 0, 1, 1.5, 2.5),
        (0.95, 2, 1, 2, None),
        (0.01, 0, 1, 0, 0.1),
        (0.99, 0, 1, 2.0, None),
    ],
)
def test_folded_normal_quantiles(p, mu, sd, min_val, max_val):
    q = folded_normal_quantile(p=p, mu=mu, sd=sd, seed=42)

    assert q > 0
    assert np.isfinite(q)
    if min_val is not None:
        assert q > min_val
    if max_val is not None:
        assert q < max_val


def test_folded_normal_negative_mean():
    q_pos = folded_normal_quantile(p=0.95, mu=2, sd=1, seed=42)
    q_neg = folded_normal_quantile(p=0.95, mu=-2, sd=1, seed=42)

    np.testing.assert_allclose(q_pos, q_neg, rtol=1e-3)


def test_folded_normal_invalid_sd():
    with pytest.raises(ValueError, match="Standard deviation must be positive"):
        folded_normal_quantile(p=0.95, mu=0, sd=-1)


def test_optimize_h_bisection(test_data):
    _, sigma, post_period_weights, n_pre, n_post = test_data

    h_min = 0.5
    h_max = 2.0

    h_optimal = _optimize_h_bisection(
        h_min=h_min,
        h_max=h_max,
        smoothness_bound=1.0,
        num_points=50,
        alpha=0.05,
        sigma=sigma,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        post_period_weights=post_period_weights,
        seed=42,
    )

    if not np.isnan(h_optimal):
        assert h_min <= h_optimal <= h_max


def test_optimize_flci_params(test_data):
    _, sigma, post_period_weights, n_pre, n_post = test_data

    result = _optimize_flci_params(
        sigma=sigma,
        smoothness_bound=1.0,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        post_period_weights=post_period_weights,
        num_points=50,
        alpha=0.05,
        seed=42,
    )

    assert isinstance(result, dict)
    assert "optimal_vec" in result
    assert "optimal_pre_period_vec" in result
    assert "optimal_half_length" in result
    assert "smoothness_bound" in result
    assert "status" in result

    assert len(result["optimal_vec"]) == n_pre + n_post
    assert len(result["optimal_pre_period_vec"]) == n_pre
    assert result["optimal_half_length"] > 0
    assert result["smoothness_bound"] == 1.0


@pytest.mark.parametrize("test_data_custom", [(2, 2, 42)], indirect=True)
def test_flci_small_pre_periods(test_data_custom):
    beta_hat, sigma, post_period_weights, n_pre, n_post = test_data_custom

    result = compute_flci(
        beta_hat=beta_hat,
        sigma=sigma,
        smoothness_bound=0.5,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        post_period_weights=post_period_weights,
        alpha=0.05,
    )

    assert isinstance(result, FLCIResult)
    assert len(result.optimal_pre_period_vec) == n_pre


@pytest.mark.parametrize("test_data_custom", [(4, 1, 42)], indirect=True)
def test_flci_single_post_period(test_data_custom):
    beta_hat, sigma, _, n_pre, n_post = test_data_custom

    result = compute_flci(
        beta_hat=beta_hat,
        sigma=sigma,
        smoothness_bound=1.0,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        alpha=0.05,
    )

    assert isinstance(result, FLCIResult)
    assert len(result.optimal_vec) == n_pre + n_post


def test_flci_large_m(test_data):
    beta_hat, sigma, post_period_weights, n_pre, n_post = test_data

    result = compute_flci(
        beta_hat=beta_hat,
        sigma=sigma,
        smoothness_bound=10.0,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        post_period_weights=post_period_weights,
        alpha=0.05,
    )

    assert isinstance(result, FLCIResult)
    assert result.optimal_half_length > 1


def test_flci_with_custom_l_vec(test_data):
    beta_hat, sigma, _, n_pre, n_post = test_data

    post_period_weights = np.ones(n_post) / n_post

    result = compute_flci(
        beta_hat=beta_hat,
        sigma=sigma,
        smoothness_bound=1.0,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        post_period_weights=post_period_weights,
        alpha=0.05,
    )

    assert isinstance(result, FLCIResult)
    np.testing.assert_array_almost_equal(result.optimal_vec[n_pre:], post_period_weights)


def test_flci_reproducibility(test_data):
    beta_hat, sigma, post_period_weights, n_pre, n_post = test_data

    result1 = compute_flci(
        beta_hat=beta_hat,
        sigma=sigma,
        smoothness_bound=1.0,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        post_period_weights=post_period_weights,
        alpha=0.05,
        seed=123,
    )

    result2 = compute_flci(
        beta_hat=beta_hat,
        sigma=sigma,
        smoothness_bound=1.0,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        post_period_weights=post_period_weights,
        alpha=0.05,
        seed=123,
    )

    np.testing.assert_array_almost_equal(result1.flci, result2.flci)
    assert result1.optimal_half_length == result2.optimal_half_length


def test_flci_grid_fallback():
    np.random.seed(99)
    n = 7
    beta_hat = np.random.randn(n) * 0.1

    A = np.random.randn(n, n) * 0.01
    sigma = A @ A.T + 1e-6 * np.eye(n)

    post_period_weights = np.array([1, 0, 0])

    result = compute_flci(
        beta_hat=beta_hat,
        sigma=sigma,
        smoothness_bound=0.1,
        n_pre_periods=4,
        n_post_periods=3,
        post_period_weights=post_period_weights,
        alpha=0.05,
        num_points=20,
    )

    assert isinstance(result, FLCIResult)
    assert result.optimal_half_length > 0
