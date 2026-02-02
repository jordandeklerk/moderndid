"""Tests for variance estimation functions."""

import numpy as np
import pytest

from moderndid.didinter.variance import (
    compute_clustered_variance,
    compute_joint_test,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.mark.parametrize(
    "influence_func,cluster_ids,n_groups,expected_positive",
    [
        (np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([0, 0, 1, 1, 2]), 5, True),
        (np.array([1.0, 2.0, 3.0, 4.0]), np.array(["A", "A", "B", "B"]), 4, True),
        (np.zeros(5), np.array([0, 0, 1, 1, 2]), 5, False),
    ],
)
def test_compute_clustered_variance_basic(influence_func, cluster_ids, n_groups, expected_positive):
    se = compute_clustered_variance(influence_func, cluster_ids, n_groups)

    assert np.isfinite(se)
    if expected_positive:
        assert se > 0
    else:
        assert se == 0.0


@pytest.mark.parametrize(
    "cluster_pattern",
    [
        "single_cluster",
        "each_own_cluster",
    ],
)
def test_compute_clustered_variance_special_cases(cluster_pattern):
    influence_func = np.array([1.0, 2.0, 3.0, 4.0])

    if cluster_pattern == "single_cluster":
        cluster_ids = np.array([0, 0, 0, 0])
    else:
        cluster_ids = np.array([0, 1, 2, 3])

    n_groups = 4
    se = compute_clustered_variance(influence_func, cluster_ids, n_groups)

    expected = np.sqrt(np.sum(influence_func**2)) / n_groups
    np.testing.assert_almost_equal(se, expected)


def test_compute_clustered_variance_cluster_effect(rng):
    n = 100
    influence_func = rng.standard_normal(n)

    cluster_ids_many = np.arange(n)
    se_many = compute_clustered_variance(influence_func, cluster_ids_many, n)

    cluster_ids_few = np.repeat(np.arange(10), 10)
    se_few = compute_clustered_variance(influence_func, cluster_ids_few, n)

    assert se_few != se_many


@pytest.mark.parametrize(
    "estimates,vcov,expected_result",
    [
        (np.array([0.1, 0.2]), None, None),
        (np.array([np.nan, np.nan]), np.eye(2), None),
        (np.array([0.1, 0.2]), np.array([[1.0, 1.0], [1.0, 1.0]]), None),
    ],
)
def test_compute_joint_test_returns_none(estimates, vcov, expected_result):
    result = compute_joint_test(estimates, vcov)
    assert result is expected_result


def test_compute_joint_test_basic():
    estimates = np.array([0.1, 0.2, 0.15])
    vcov = np.array(
        [
            [0.01, 0.002, 0.001],
            [0.002, 0.015, 0.003],
            [0.001, 0.003, 0.012],
        ]
    )

    result = compute_joint_test(estimates, vcov)

    assert result is not None
    assert "chi2_stat" in result
    assert "df" in result
    assert "p_value" in result
    assert result["chi2_stat"] >= 0
    assert result["df"] == 3
    assert 0 <= result["p_value"] <= 1


def test_compute_joint_test_single_estimate():
    estimates = np.array([0.5])
    vcov = np.array([[0.04]])

    result = compute_joint_test(estimates, vcov)

    assert result is not None
    assert result["df"] == 1
    expected_chi2 = 0.5**2 / 0.04
    np.testing.assert_almost_equal(result["chi2_stat"], expected_chi2)


@pytest.mark.parametrize(
    "estimates,vcov_scale,chi2_threshold,pvalue_threshold,chi2_above",
    [
        (np.array([0.0, 0.0]), 1.0, 0.01, 0.99, False),
        (np.array([10.0, 10.0]), 0.01, 100, 0.01, True),
    ],
)
def test_compute_joint_test_extreme_values(estimates, vcov_scale, chi2_threshold, pvalue_threshold, chi2_above):
    vcov = np.eye(len(estimates)) * vcov_scale
    result = compute_joint_test(estimates, vcov)

    assert result is not None
    if chi2_above:
        assert result["chi2_stat"] > chi2_threshold
        assert result["p_value"] < pvalue_threshold
    else:
        assert result["chi2_stat"] < chi2_threshold
        assert result["p_value"] > pvalue_threshold


def test_compute_joint_test_handles_nan_estimates():
    estimates = np.array([0.1, np.nan, 0.2])
    vcov = np.array(
        [
            [0.01, 0.002, 0.001],
            [0.002, 0.015, 0.003],
            [0.001, 0.003, 0.012],
        ]
    )

    result = compute_joint_test(estimates, vcov)

    assert result is not None
    assert result["df"] == 2
