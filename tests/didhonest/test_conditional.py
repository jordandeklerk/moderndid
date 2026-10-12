"""Tests for the conditional test module."""

import numpy as np
import pytest
from scipy import stats

from moderndid.didhonest.conditional import (
    _create_pre_period_second_diff_constraints,
    _norminvp_generalized,
    estimate_lowerbound_m_conditional_test,
)
from moderndid.didhonest.conditional import (
    test_in_identified_set_max as in_identified_set_max_func,
)


def test_norminvp_generalized_no_truncation():
    p = 0.975
    result = _norminvp_generalized(p, lower=-np.inf, upper=np.inf)
    expected = 1.96
    assert np.isclose(result, expected, atol=0.01)


def test_norminvp_generalized_with_mean_sd():
    p = 0.5
    mu = 10
    sd = 2
    result = _norminvp_generalized(p, lower=-np.inf, upper=np.inf, mu=mu, sd=sd)
    assert np.isclose(result, mu)


def test_norminvp_generalized_truncated():
    p = 0.5
    lower = 0
    upper = 2
    result = _norminvp_generalized(p, lower=lower, upper=upper)
    assert lower <= result <= upper
    assert 0.5 < result < 1.5


def test_norminvp_generalized_edge_cases():
    result = _norminvp_generalized(0, lower=1, upper=10)
    assert result == 1

    result = _norminvp_generalized(1, lower=1, upper=10)
    assert result == 10

    with pytest.raises(ValueError, match="Standard deviation must be positive"):
        _norminvp_generalized(0.5, lower=0, upper=1, sd=0)


def test_norminvp_generalized_one_sided_truncation():
    p = 0.1
    lower = 0
    result = _norminvp_generalized(p, lower=lower, upper=np.inf)
    assert result >= lower

    p = 0.9
    upper = 0
    result = _norminvp_generalized(p, lower=-np.inf, upper=upper)
    assert result <= upper


def test_norminvp_generalized_upper_tail():
    for lower in [5.0, 8.0, 9.0, 12.0, 30.0]:
        result = _norminvp_generalized(0.95, lower=lower, upper=np.inf)
        expected = stats.truncnorm.ppf(0.95, lower, np.inf)
        assert lower < result < lower + 1
        assert np.isclose(result, expected, rtol=1e-10)

    assert np.isclose(_norminvp_generalized(0.95, lower=8.0, upper=np.inf), 8.36096063, atol=1e-8)
    assert np.isclose(_norminvp_generalized(0.95, lower=9.0, upper=np.inf), 9.32322299, atol=1e-8)


def test_norminvp_generalized_lower_tail_and_scale():
    result = _norminvp_generalized(0.95, lower=-np.inf, upper=-40.0)
    assert np.isclose(result, stats.truncnorm.ppf(0.95, -np.inf, -40.0), rtol=1e-10)

    result = _norminvp_generalized(0.995, lower=10.0, upper=10.5, mu=2.0, sd=0.5)
    expected = 2.0 + 0.5 * stats.truncnorm.ppf(0.995, 16.0, 17.0)
    assert np.isclose(result, expected, rtol=1e-10)


def test_norminvp_generalized_empty_interval():
    assert _norminvp_generalized(0.5, lower=2.0, upper=2.0) == 2.0
    assert _norminvp_generalized(0.5, lower=3.0, upper=1.0) == 3.0


def test_test_in_identified_set_max_adds_bound_back():
    A, d = _create_pre_period_second_diff_constraints(2)
    sigma = np.eye(2)
    m_value = np.sqrt(5)
    quantile = 1 + stats.truncnorm.ppf(0.95, -1, np.inf)

    y_above = np.array([np.sqrt(5) * (quantile + 0.05), 0.0])
    assert in_identified_set_max_func(m_value, y_above, sigma, A, 0.05, d)

    y_below = np.array([np.sqrt(5) * (quantile - 0.05), 0.0])
    assert not in_identified_set_max_func(m_value, y_below, sigma, A, 0.05, d)


def test_create_pre_period_second_diff_constraints():
    num_pre_periods = 3
    A, d = _create_pre_period_second_diff_constraints(num_pre_periods)

    assert A.shape == (4, 3)
    assert len(d) == 4
    assert np.all(d == 1)

    assert np.array_equal(A[0, :], [1, -2, 1])
    assert np.array_equal(A[1, :], [0, 1, -2])
    assert np.array_equal(A[2, :], [-1, 2, -1])
    assert np.array_equal(A[3, :], [0, -1, 2])

    num_pre_periods = 4
    A, d = _create_pre_period_second_diff_constraints(num_pre_periods)

    assert A.shape == (6, 4)
    assert len(d) == 6
    assert np.all(d == 1)

    assert np.array_equal(A[0, :], [1, -2, 1, 0])
    assert np.array_equal(A[1, :], [0, 1, -2, 1])
    assert np.array_equal(A[2, :], [0, 0, 1, -2])
    assert np.array_equal(A[3:, :], -A[:3, :])
    assert not np.any(np.all(A == 0, axis=1))


def test_create_pre_period_second_diff_constraints_two_periods():
    A, d = _create_pre_period_second_diff_constraints(2)

    assert np.array_equal(A, [[1, -2], [-1, 2]])
    assert np.array_equal(d, [1, 1])


def test_create_pre_period_second_diff_constraints_invalid():
    with pytest.raises(ValueError, match="Cannot estimate M"):
        _create_pre_period_second_diff_constraints(1)


def test_test_in_identified_set_max_simple():
    np.random.seed(42)

    y = np.array([0.1, 0.2, 0.4])
    sigma = np.eye(3) * 0.01

    A, d = _create_pre_period_second_diff_constraints(3)

    m_value = 1.0
    alpha = 0.05

    reject = in_identified_set_max_func(m_value, y, sigma, A, alpha, d)
    assert not reject

    m_value = 0.1
    reject = in_identified_set_max_func(m_value, y, sigma, A, alpha, d)
    assert reject

    y_violation = np.array([0.0, 0.0, 1.0])
    sigma_small = np.eye(3) * 0.0001
    m_value = 0.01
    reject = in_identified_set_max_func(m_value, y_violation, sigma_small, A, alpha, d)
    assert reject


def test_test_in_identified_set_max_zero_variance():
    y = np.array([0.1, 0.2, 0.3])
    sigma = np.eye(3) * 1e-20
    A, d = _create_pre_period_second_diff_constraints(3)

    m_value = 0.5
    alpha = 0.05

    reject = in_identified_set_max_func(m_value, y, sigma, A, alpha, d)
    assert isinstance(reject, bool)


def test_estimate_lowerbound_m_conditional_test_basic(fast_config):
    np.random.seed(123)

    pre_period_coef = np.array([0.0, 0.1, 0.2, 0.3])
    pre_period_covar = np.eye(4) * 0.01

    result = estimate_lowerbound_m_conditional_test(
        pre_period_coef=pre_period_coef,
        pre_period_covar=pre_period_covar,
        grid_ub=1.0,
        alpha=0.05,
        grid_points=fast_config["grid_points_small"],
    )

    assert np.isfinite(result)
    assert result >= 0
    assert result < 0.5


def test_estimate_lowerbound_m_conditional_test_all_rejected():
    pre_period_coef = np.array([0.0, 1.0, -1.0, 2.0])
    pre_period_covar = np.eye(4) * 0.0001

    result = estimate_lowerbound_m_conditional_test(
        pre_period_coef=pre_period_coef,
        pre_period_covar=pre_period_covar,
        grid_ub=0.01,
        alpha=0.05,
        grid_points=10,
    )

    assert isinstance(result, int | float)


def test_estimate_lowerbound_m_conditional_test_consistency(fast_config):
    np.random.seed(456)
    pre_period_coef = np.random.normal(0, 0.1, 4)
    pre_period_covar = np.eye(4) * 0.01

    result_05 = estimate_lowerbound_m_conditional_test(
        pre_period_coef=pre_period_coef,
        pre_period_covar=pre_period_covar,
        grid_ub=1.0,
        alpha=0.05,
        grid_points=fast_config["grid_points_small"],
    )

    result_10 = estimate_lowerbound_m_conditional_test(
        pre_period_coef=pre_period_coef,
        pre_period_covar=pre_period_covar,
        grid_ub=1.0,
        alpha=0.10,
        grid_points=fast_config["grid_points_small"],
    )

    assert result_10 <= result_05


def test_constraint_matrix_mathematical_correctness():
    num_pre_periods = 4
    beta = np.array([1.0, 2.0, 3.5, 5.5])

    A, _ = _create_pre_period_second_diff_constraints(num_pre_periods)

    second_diffs = np.diff(np.append(beta, 0.0), n=2)

    assert A.shape == (6, 4)
    assert np.allclose(A[:3, :] @ beta, second_diffs)
    assert np.allclose(A[:3, :] @ beta, [0.5, 0.5, -7.5])
    assert np.allclose(A[3:, :], -A[:3, :])


def test_integration_test_and_estimate(fast_config):
    np.random.seed(789)

    pre_period_coef = np.array([0.05, 0.10, 0.16, 0.23])
    pre_period_covar = np.eye(4) * 0.005

    lower_bound = estimate_lowerbound_m_conditional_test(
        pre_period_coef=pre_period_coef,
        pre_period_covar=pre_period_covar,
        grid_ub=2.0,
        alpha=0.05,
        grid_points=fast_config["grid_points_small"],
    )

    if np.isfinite(lower_bound) and lower_bound > 0:
        A, d = _create_pre_period_second_diff_constraints(4)

        m_below = lower_bound * 0.9
        reject_below = in_identified_set_max_func(m_below, pre_period_coef, pre_period_covar, A, 0.05, d)
        assert reject_below

        m_above = lower_bound * 1.1
        reject_above = in_identified_set_max_func(m_above, pre_period_coef, pre_period_covar, A, 0.05, d)
        assert not reject_above
