"""Tests for variance estimation functions."""

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid.didinter.variance import (
    compute_cluster_influence,
    compute_clustered_variance,
    compute_cohort_dof,
    compute_control_dof,
    compute_dof_scaling,
    compute_joint_test,
    compute_path_cohort_dof,
    compute_union_dof,
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
    ],
)
def test_compute_joint_test_returns_none(estimates, vcov, expected_result):
    result = compute_joint_test(estimates, vcov)
    assert result is expected_result


def test_compute_joint_test_singular_vcov():
    estimates = np.array([0.1, 0.2])
    vcov = np.array([[1.0, 1.0], [1.0, 1.0]])
    result = compute_joint_test(estimates, vcov)
    assert result is not None
    assert np.isnan(result["chi2_stat"])
    assert np.isnan(result["p_value"])
    assert len(result["warnings"]) == 1
    assert "not invertible" in result["warnings"][0]


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


def test_compute_cohort_dof_counts_only_weighted_switchers(variance_config):
    df = pl.DataFrame(
        {
            "is_switcher_1": [1, 1, 1, 0],
            "weight_gt": [1.0, 2.0, 0.0, 1.0],
            "weighted_diff_1": [1.0, 4.0, 0.0, 9.0],
            "d_sq": [0.0, 0.0, 0.0, 0.0],
            "F_g": [3.0, 3.0, 3.0, 4.0],
            "d_fg": [1.0, 1.0, 1.0, 1.0],
        }
    )

    result = compute_cohort_dof(df, 1, variance_config)

    assert result["dof_switcher_1"].to_list() == [2, 2, None, None]
    np.testing.assert_allclose(result["cohort_mean_1"].to_list()[:2], [5.0 / 3.0, 5.0 / 3.0])


def test_compute_cohort_dof_counts_distinct_clusters(variance_config):
    df = pl.DataFrame(
        {
            "is_switcher_1": [1, 1, 1, 0],
            "weight_gt": [1.0, 1.0, 1.0, 1.0],
            "weighted_diff_1": [0.1, 0.2, 0.3, 0.4],
            "d_sq": [0.0, 0.0, 0.0, 0.0],
            "F_g": [3.0, 3.0, 3.0, 3.0],
            "d_fg": [1.0, 1.0, 1.0, 1.0],
            "cl": [1, 1, 2, 3],
        }
    )

    result = compute_cohort_dof(df, 1, variance_config, "cl")

    assert result["dof_switcher_1"].to_list() == [2, 2, 2, None]


def test_compute_control_dof_skips_zero_weight_rows(variance_config):
    df = pl.DataFrame(
        {
            "time": [2, 2, 2, 2],
            "d_sq": [0.0, 0.0, 0.0, 0.0],
            "never_change_1": [1.0, 1.0, 1.0, None],
            "weight_gt": [1.0, 1.0, 0.0, 1.0],
            "weighted_diff_1": [0.5, 1.5, 0.0, 0.0],
        }
    )

    result = compute_control_dof(df, 1, variance_config)

    assert result["dof_control_1"].to_list() == [2, 2, None, None]
    assert result["control_mean_1"].to_list()[:2] == [1.0, 1.0]


def test_compute_control_dof_counts_distinct_clusters(variance_config):
    df = pl.DataFrame(
        {
            "time": [2, 2, 2, 2],
            "d_sq": [0.0, 0.0, 0.0, 0.0],
            "never_change_1": [1.0, 1.0, 0.0, 1.0],
            "weight_gt": [1.0, 1.0, 1.0, 1.0],
            "weighted_diff_1": [0.0, 0.0, 0.0, 0.0],
            "cl": [5, 6, 7, 6],
        }
    )

    result = compute_control_dof(df, 1, variance_config, "cl")

    assert result["dof_control_1"].to_list() == [2, 2, None, 2]


def test_compute_union_dof_counts_distinct_weighted_clusters(variance_config):
    df = pl.DataFrame(
        {
            "time": [3, 3, 3, 3, 3],
            "d_sq": [0.0, 0.0, 0.0, 0.0, 0.0],
            "is_switcher_1": [1, 0, 0, 0, None],
            "never_change_1": [0.0, 1.0, 1.0, 0.0, None],
            "weight_gt": [1.0, 1.0, 0.0, 1.0, 1.0],
            "weighted_diff_1": [0.0, 0.0, 0.0, 0.0, 0.0],
            "cl": [1, 2, 3, 4, 5],
        }
    )

    result = compute_union_dof(df, 1, variance_config, "cl")

    assert result["dof_union_1"].to_list() == [2, 2, None, None, None]


@pytest.mark.parametrize("less_conservative_se", [False, True])
def test_compute_dof_scaling_applies_with_less_conservative_se(variance_config, less_conservative_se):
    variance_config.less_conservative_se = less_conservative_se
    df = pl.DataFrame(
        {
            "time": [3, 2, 5],
            "F_g": [3.0, 3.0, 3.0],
            "dof_switcher_1": [4, None, None],
            "dof_control_1": [None, 5, None],
            "dof_union_1": [None, None, 1],
        }
    )

    result = compute_dof_scaling(df, 1, variance_config)

    np.testing.assert_allclose(result["dof_scale_1"].to_list(), [np.sqrt(4 / 3), np.sqrt(5 / 4), 1.0])


def test_compute_path_cohort_dof_falls_back_to_coarser_paths(variance_config):
    df = pl.DataFrame(
        {
            "is_switcher_2": [1, 1, 1, 1],
            "weight_gt": [1.0, 1.0, 1.0, 1.0],
            "weighted_diff_2": [1.0, 3.0, 5.0, 11.0],
            "path_0": [0, 0, 0, 0],
            "path_1": [1, 1, 1, 2],
            "path_2": [3, 3, 4, 5],
            "valid_cohort_1": [1, 1, 1, 0],
            "valid_cohort_2": [1, 1, 0, 0],
        }
    )

    result = compute_path_cohort_dof(df, 2, variance_config)

    assert result["dof_switcher_2"].to_list() == [2, 2, 3, 4]
    np.testing.assert_allclose(result["cohort_mean_2"].to_list(), [2.0, 2.0, 3.0, 5.0])
    assert not any(col.startswith("_path_") for col in result.columns)


def test_compute_path_cohort_dof_ignores_clusters_and_zero_weights(variance_config):
    df = pl.DataFrame(
        {
            "is_switcher_1": [1, 1, 1],
            "weight_gt": [1.0, 1.0, 0.0],
            "weighted_diff_1": [2.0, 4.0, 0.0],
            "path_0": [0, 0, 0],
            "path_1": [1, 1, 1],
            "valid_cohort_1": [1, 1, 1],
            "cl": [7, 7, 7],
        }
    )

    result = compute_path_cohort_dof(df, 1, variance_config)

    assert result["dof_switcher_1"].to_list() == [2, 2, None]
    np.testing.assert_allclose(result["cohort_mean_1"].to_list()[:2], [3.0, 3.0])


def test_compute_cluster_influence_sums_rows_within_clusters():
    influence = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    result = compute_cluster_influence(influence, pl.Series([7, 7, 9]))

    np.testing.assert_allclose(result, [[4.0, 6.0], [5.0, 6.0]])


def test_compute_cluster_influence_without_clusters_keeps_groups():
    result = compute_cluster_influence(np.array([1.0, 2.0, 3.0]))

    np.testing.assert_allclose(result, [[1.0], [2.0], [3.0]])


def test_compute_cluster_influence_drops_groups_without_cluster():
    result = compute_cluster_influence(np.array([1.0, 2.0, 3.0, 4.0]), pl.Series(["a", None, "b", "a"]))

    np.testing.assert_allclose(result, [[5.0], [3.0]])


def test_compute_cluster_influence_single_cluster_keeps_groups():
    result = compute_cluster_influence(np.array([1.0, 2.0]), pl.Series([3, 3]))

    np.testing.assert_allclose(result, [[1.0], [2.0]])
