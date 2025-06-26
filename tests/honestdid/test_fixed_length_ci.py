"""Tests for FLCI (Fixed-Length Confidence Intervals) module."""

import numpy as np
import pytest

from pydid.honestdid.fixed_length_ci import (
    FLCIResult,
    _optimize_flci_params,
    _optimize_h_bisection,
    affine_variance,
    compute_flci,
    folded_normal_quantile,
    get_min_bias_h,
    l_to_weights,
    maximize_bias,
    minimize_variance,
    weights_to_l,
)


def generate_test_data(n_pre=4, n_post=3, seed=42):
    np.random.seed(seed)

    betahat = np.random.randn(n_pre + n_post)

    A = np.random.randn(n_pre + n_post, n_pre + n_post)
    sigma = 0.01 * (A @ A.T) + 0.001 * np.eye(n_pre + n_post)

    l_vec = np.zeros(n_post)
    l_vec[0] = 1

    return betahat, sigma, l_vec


def test_compute_flci_basic():
    betahat, sigma, l_vec = generate_test_data()

    result = compute_flci(
        betahat=betahat,
        sigma=sigma,
        m=0.5,
        num_pre_periods=4,
        num_post_periods=3,
        l_vec=l_vec,
        alpha=0.05,
    )

    assert isinstance(result, FLCIResult)
    assert len(result.flci) == 2
    assert result.flci[0] < result.flci[1]
    assert result.optimal_half_length > 0
    assert result.m == 0.5
    assert result.status == "optimal"
    assert len(result.optimal_vec) == 7
    assert len(result.optimal_pre_period_vec) == 4


def test_flci_with_default_l_vec():
    betahat, sigma, _ = generate_test_data()

    result = compute_flci(
        betahat=betahat,
        sigma=sigma,
        m=1.0,
        num_pre_periods=4,
        num_post_periods=3,
        alpha=0.05,
    )

    assert isinstance(result, FLCIResult)
    expected_l_vec = np.array([1, 0, 0])
    np.testing.assert_array_equal(result.optimal_vec[4:], expected_l_vec)


def test_flci_different_alpha_levels():
    betahat, sigma, l_vec = generate_test_data()

    results = {}
    for alpha in [0.01, 0.05, 0.10]:
        result = compute_flci(
            betahat=betahat,
            sigma=sigma,
            m=1.0,
            num_pre_periods=4,
            num_post_periods=3,
            l_vec=l_vec,
            alpha=alpha,
        )

        assert isinstance(result, FLCIResult)
        assert result.optimal_half_length > 0
        results[alpha] = result.optimal_half_length

    assert results[0.01] > results[0.05]
    assert results[0.05] > results[0.10]


def test_flci_different_m_values():
    betahat, sigma, l_vec = generate_test_data()

    results = {}
    for m in [0.1, 0.5, 1.0, 2.0]:
        result = compute_flci(
            betahat=betahat,
            sigma=sigma,
            m=m,
            num_pre_periods=4,
            num_post_periods=3,
            l_vec=l_vec,
            alpha=0.05,
        )

        assert isinstance(result, FLCIResult)
        assert result.optimal_half_length > 0
        results[m] = result.optimal_half_length

    assert results[2.0] > results[1.0]
    assert results[1.0] > results[0.5]


def test_maximize_bias_basic():
    _, sigma, l_vec = generate_test_data()

    result = maximize_bias(
        h=1.0,
        sigma=sigma,
        num_pre_periods=4,
        num_post_periods=3,
        l_vec=l_vec,
        m=1.0,
    )

    assert result["status"] == "optimal"
    assert result["value"] >= 0
    assert result["optimal_w"] is not None
    assert len(result["optimal_w"]) == 4
    assert result["optimal_l"] is not None
    assert len(result["optimal_l"]) == 4


def test_maximize_bias_different_h_values():
    _, sigma, l_vec = generate_test_data()

    biases = {}
    for h in [0.5, 1.0, 2.0]:
        result = maximize_bias(
            h=h,
            sigma=sigma,
            num_pre_periods=4,
            num_post_periods=3,
            l_vec=l_vec,
            m=1.0,
        )

        assert result["status"] == "optimal", f"Optimization failed for h={h} with status: {result['status']}"
        biases[h] = result["value"]

    assert biases[2.0] >= biases[1.0]
    assert biases[1.0] >= biases[0.5]


def test_maximize_bias_weight_sum_constraint():
    _, sigma, l_vec = generate_test_data()

    result = maximize_bias(
        h=1.5,
        sigma=sigma,
        num_pre_periods=4,
        num_post_periods=3,
        l_vec=l_vec,
        m=1.0,
    )

    if result["status"] == "optimal":
        target_sum = np.dot(np.arange(1, 4), l_vec)
        actual_sum = np.sum(result["optimal_w"])
        np.testing.assert_allclose(actual_sum, target_sum, rtol=1e-5)


def test_affine_variance():
    _, sigma, l_vec = generate_test_data()

    l_pre = np.array([0.1, 0.2, 0.3, 0.4])

    variance = affine_variance(
        l_pre=l_pre,
        l_post=l_vec,
        sigma=sigma,
        num_pre_periods=4,
    )

    assert variance > 0
    assert np.isfinite(variance)


def test_minimize_variance():
    _, sigma, l_vec = generate_test_data()

    h_min = minimize_variance(
        sigma=sigma,
        num_pre_periods=4,
        num_post_periods=3,
        l_vec=l_vec,
    )

    assert h_min > 0
    assert np.isfinite(h_min)


def test_get_min_bias_h():
    _, sigma, l_vec = generate_test_data()

    h_min_bias = get_min_bias_h(
        sigma=sigma,
        num_pre_periods=4,
        num_post_periods=3,
        l_vec=l_vec,
    )

    assert h_min_bias > 0
    assert np.isfinite(h_min_bias)


def test_h_ordering():
    _, sigma, l_vec = generate_test_data()

    h_min_var = minimize_variance(
        sigma=sigma,
        num_pre_periods=4,
        num_post_periods=3,
        l_vec=l_vec,
    )

    h_min_bias = get_min_bias_h(
        sigma=sigma,
        num_pre_periods=4,
        num_post_periods=3,
        l_vec=l_vec,
    )

    assert h_min_var <= h_min_bias + 1e-6


def test_weights_to_l_conversion():
    weights = np.array([1, 2, 3, 4])
    l_vec = weights_to_l(weights)

    expected = np.array([1, 3, 6, 10])
    np.testing.assert_array_almost_equal(l_vec, expected)


def test_l_to_weights_conversion():
    l_vec = np.array([1, 3, 6, 10])
    weights = l_to_weights(l_vec)

    expected = np.array([1, 2, 3, 4])
    np.testing.assert_array_almost_equal(weights, expected)


def test_conversion_roundtrip():
    original_weights = np.array([0.5, -0.3, 0.8, 1.2])

    l_vec = weights_to_l(original_weights)
    recovered_weights = l_to_weights(l_vec)

    np.testing.assert_array_almost_equal(original_weights, recovered_weights)


def test_conversion_single_element():
    weights = np.array([5.0])
    l_vec = weights_to_l(weights)
    np.testing.assert_array_equal(l_vec, weights)

    recovered = l_to_weights(l_vec)
    np.testing.assert_array_equal(recovered, weights)


def test_folded_normal_zero_mean():
    q = folded_normal_quantile(p=0.95, mu=0, sd=1)

    assert q > 0
    assert np.isfinite(q)
    assert 1.5 < q < 2.5


def test_folded_normal_nonzero_mean():
    q = folded_normal_quantile(p=0.95, mu=2, sd=1, seed=42)

    assert q > 0
    assert np.isfinite(q)
    assert q > 2


def test_folded_normal_negative_mean():
    q_pos = folded_normal_quantile(p=0.95, mu=2, sd=1, seed=42)
    q_neg = folded_normal_quantile(p=0.95, mu=-2, sd=1, seed=42)

    np.testing.assert_allclose(q_pos, q_neg, rtol=1e-3)


def test_folded_normal_invalid_sd():
    with pytest.raises(ValueError, match="Standard deviation must be positive"):
        folded_normal_quantile(p=0.95, mu=0, sd=-1)


def test_folded_normal_extreme_quantiles():
    q_low = folded_normal_quantile(p=0.01, mu=0, sd=1)
    assert 0 < q_low < 0.1

    q_high = folded_normal_quantile(p=0.99, mu=0, sd=1)
    assert q_high > 2.0


def test_optimize_h_bisection():
    _, sigma, l_vec = generate_test_data()

    h_min = 0.5
    h_max = 2.0

    h_optimal = _optimize_h_bisection(
        h_min=h_min,
        h_max=h_max,
        m=1.0,
        num_points=50,
        alpha=0.05,
        sigma=sigma,
        num_pre_periods=4,
        num_post_periods=3,
        l_vec=l_vec,
        seed=42,
    )

    if not np.isnan(h_optimal):
        assert h_min <= h_optimal <= h_max


def test_optimize_flci_params():
    _, sigma, l_vec = generate_test_data()

    result = _optimize_flci_params(
        sigma=sigma,
        m=1.0,
        num_pre_periods=4,
        num_post_periods=3,
        l_vec=l_vec,
        num_points=50,
        alpha=0.05,
        seed=42,
    )

    assert isinstance(result, dict)
    assert "optimal_vec" in result
    assert "optimal_pre_period_vec" in result
    assert "optimal_half_length" in result
    assert "m" in result
    assert "status" in result

    assert len(result["optimal_vec"]) == 7
    assert len(result["optimal_pre_period_vec"]) == 4
    assert result["optimal_half_length"] > 0
    assert result["m"] == 1.0


def test_flci_small_pre_periods():
    n_pre = 2
    n_post = 2
    betahat, sigma, l_vec = generate_test_data(n_pre=n_pre, n_post=n_post)

    result = compute_flci(
        betahat=betahat,
        sigma=sigma,
        m=0.5,
        num_pre_periods=n_pre,
        num_post_periods=n_post,
        l_vec=l_vec,
        alpha=0.05,
    )

    assert isinstance(result, FLCIResult)
    assert len(result.optimal_pre_period_vec) == n_pre


def test_flci_single_post_period():
    n_pre = 4
    n_post = 1
    betahat, sigma, _ = generate_test_data(n_pre=n_pre, n_post=n_post)

    result = compute_flci(
        betahat=betahat,
        sigma=sigma,
        m=1.0,
        num_pre_periods=n_pre,
        num_post_periods=n_post,
        alpha=0.05,
    )

    assert isinstance(result, FLCIResult)
    assert len(result.optimal_vec) == n_pre + n_post


def test_flci_large_m():
    betahat, sigma, l_vec = generate_test_data()

    result = compute_flci(
        betahat=betahat,
        sigma=sigma,
        m=10.0,
        num_pre_periods=4,
        num_post_periods=3,
        l_vec=l_vec,
        alpha=0.05,
    )

    assert isinstance(result, FLCIResult)
    assert result.optimal_half_length > 1


def test_flci_with_custom_l_vec():
    betahat, sigma, _ = generate_test_data()

    l_vec = np.ones(3) / 3

    result = compute_flci(
        betahat=betahat,
        sigma=sigma,
        m=1.0,
        num_pre_periods=4,
        num_post_periods=3,
        l_vec=l_vec,
        alpha=0.05,
    )

    assert isinstance(result, FLCIResult)
    np.testing.assert_array_almost_equal(result.optimal_vec[4:], l_vec)


def test_flci_reproducibility():
    betahat, sigma, l_vec = generate_test_data()

    result1 = compute_flci(
        betahat=betahat,
        sigma=sigma,
        m=1.0,
        num_pre_periods=4,
        num_post_periods=3,
        l_vec=l_vec,
        alpha=0.05,
        seed=123,
    )

    result2 = compute_flci(
        betahat=betahat,
        sigma=sigma,
        m=1.0,
        num_pre_periods=4,
        num_post_periods=3,
        l_vec=l_vec,
        alpha=0.05,
        seed=123,
    )

    np.testing.assert_array_almost_equal(result1.flci, result2.flci)
    assert result1.optimal_half_length == result2.optimal_half_length


def test_flci_grid_fallback():
    np.random.seed(99)
    n = 7
    betahat = np.random.randn(n) * 0.1

    A = np.random.randn(n, n) * 0.01
    sigma = A @ A.T + 1e-6 * np.eye(n)

    l_vec = np.array([1, 0, 0])

    result = compute_flci(
        betahat=betahat,
        sigma=sigma,
        m=0.1,
        num_pre_periods=4,
        num_post_periods=3,
        l_vec=l_vec,
        alpha=0.05,
        num_points=20,
    )

    assert isinstance(result, FLCIResult)
    assert result.optimal_half_length > 0
