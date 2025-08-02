# pylint: disable=redefined-outer-name
"""Tests for second differences under second differences restriction."""

import numpy as np
import pytest

from moderndid.didhonest.delta.sd.sd import (
    DeltaSDResult,
    _create_sd_constraint_matrix,
    _create_sd_constraint_vector,
    compute_conditional_cs_sd,
    compute_identified_set_sd,
)


@pytest.fixture
def simple_data():
    np.random.seed(42)
    num_pre_periods = 3
    num_post_periods = 2

    betahat = np.array([0.1, -0.05, 0.02, 0.15, 0.2])
    sigma = np.eye(5) * 0.01

    return {
        "betahat": betahat,
        "sigma": sigma,
        "num_pre_periods": num_pre_periods,
        "num_post_periods": num_post_periods,
        "l_vec": np.array([1, 0]),
    }


def test_create_sd_constraint_matrix_basic():
    A = _create_sd_constraint_matrix(3, 2)

    assert A.shape[0] == 8
    assert A.shape[1] == 5

    half = A.shape[0] // 2
    assert np.allclose(A[:half], -A[half:])


def test_create_sd_constraint_matrix_post_period_only():
    A_all = _create_sd_constraint_matrix(3, 2, post_period_moments_only=False)
    A_post = _create_sd_constraint_matrix(3, 2, post_period_moments_only=True)

    assert A_post.shape[0] < A_all.shape[0]

    post_period_cols = [3, 4]
    for i in range(A_post.shape[0]):
        assert np.any(A_post[i, post_period_cols] != 0)


def test_create_sd_constraint_vector():
    A = _create_sd_constraint_matrix(3, 2)
    m_bar = 0.5
    d = _create_sd_constraint_vector(A, m_bar)

    assert d.shape[0] == A.shape[0]
    assert np.all(d == m_bar)


def test_compute_identified_set_sd_basic(simple_data):
    result = compute_identified_set_sd(
        m_bar=0.1,
        true_beta=simple_data["betahat"],
        l_vec=simple_data["l_vec"],
        num_pre_periods=simple_data["num_pre_periods"],
        num_post_periods=simple_data["num_post_periods"],
    )

    assert isinstance(result, DeltaSDResult)
    assert hasattr(result, "id_lb")
    assert hasattr(result, "id_ub")
    assert result.id_lb <= result.id_ub


def test_compute_identified_set_sd_zero_m_bar(simple_data):
    result = compute_identified_set_sd(
        m_bar=0,
        true_beta=simple_data["betahat"],
        l_vec=simple_data["l_vec"],
        num_pre_periods=simple_data["num_pre_periods"],
        num_post_periods=simple_data["num_post_periods"],
    )

    assert np.isclose(result.id_lb, result.id_ub, rtol=1e-6)


def test_compute_identified_set_sd_large_m_bar(simple_data):
    result_small = compute_identified_set_sd(
        m_bar=0.1,
        true_beta=simple_data["betahat"],
        l_vec=simple_data["l_vec"],
        num_pre_periods=simple_data["num_pre_periods"],
        num_post_periods=simple_data["num_post_periods"],
    )

    result_large = compute_identified_set_sd(
        m_bar=1.0,
        true_beta=simple_data["betahat"],
        l_vec=simple_data["l_vec"],
        num_pre_periods=simple_data["num_pre_periods"],
        num_post_periods=simple_data["num_post_periods"],
    )

    assert result_large.id_ub - result_large.id_lb > result_small.id_ub - result_small.id_lb


def test_compute_conditional_cs_sd_basic(simple_data, fast_config):
    result = compute_conditional_cs_sd(
        betahat=simple_data["betahat"],
        sigma=simple_data["sigma"],
        num_pre_periods=simple_data["num_pre_periods"],
        num_post_periods=simple_data["num_post_periods"],
        l_vec=simple_data["l_vec"],
        m_bar=0.1,
        alpha=0.05,
        grid_points=fast_config["grid_points_medium"],
    )

    assert isinstance(result, dict)
    assert "grid" in result
    assert "accept" in result
    assert len(result["grid"]) == len(result["accept"])
    assert len(result["grid"]) == fast_config["grid_points_medium"]


@pytest.mark.parametrize("hybrid_flag", ["FLCI", "LF", "ARP"])
def test_compute_conditional_cs_sd_hybrid_flags(simple_data, hybrid_flag, fast_config):
    result = compute_conditional_cs_sd(
        betahat=simple_data["betahat"],
        sigma=simple_data["sigma"],
        num_pre_periods=simple_data["num_pre_periods"],
        num_post_periods=simple_data["num_post_periods"],
        l_vec=simple_data["l_vec"],
        m_bar=0.1,
        alpha=0.05,
        hybrid_flag=hybrid_flag,
        grid_points=fast_config["grid_points_small"],
    )

    assert isinstance(result, dict)
    assert "grid" in result
    assert "accept" in result


def test_compute_conditional_cs_sd_custom_grid_bounds(simple_data, fast_config):
    result = compute_conditional_cs_sd(
        betahat=simple_data["betahat"],
        sigma=simple_data["sigma"],
        num_pre_periods=simple_data["num_pre_periods"],
        num_post_periods=simple_data["num_post_periods"],
        l_vec=simple_data["l_vec"],
        m_bar=0.1,
        alpha=0.05,
        grid_lb=-1.0,
        grid_ub=1.0,
        grid_points=fast_config["grid_points_small"],
    )

    assert result["grid"][0] == pytest.approx(-1.0)
    assert result["grid"][-1] == pytest.approx(1.0)


def test_compute_conditional_cs_sd_post_period_moments_only(simple_data, fast_config):
    result_all = compute_conditional_cs_sd(
        betahat=simple_data["betahat"],
        sigma=simple_data["sigma"],
        num_pre_periods=simple_data["num_pre_periods"],
        num_post_periods=simple_data["num_post_periods"],
        l_vec=simple_data["l_vec"],
        m_bar=0.1,
        alpha=0.05,
        post_period_moments_only=False,
        grid_points=fast_config["grid_points_small"],
    )

    result_post = compute_conditional_cs_sd(
        betahat=simple_data["betahat"],
        sigma=simple_data["sigma"],
        num_pre_periods=simple_data["num_pre_periods"],
        num_post_periods=simple_data["num_post_periods"],
        l_vec=simple_data["l_vec"],
        m_bar=0.1,
        alpha=0.05,
        post_period_moments_only=True,
        grid_points=fast_config["grid_points_small"],
    )

    assert isinstance(result_all, dict)
    assert isinstance(result_post, dict)


def test_identified_set_monotonicity():
    np.random.seed(42)
    true_beta = np.array([0.1, 0.05, 0.02, 0.15, 0.2])
    l_vec = np.array([1, 0])

    m_values = [0, 0.1, 0.5, 1.0]
    widths = []

    for m in m_values:
        result = compute_identified_set_sd(
            m_bar=m,
            true_beta=true_beta,
            l_vec=l_vec,
            num_pre_periods=3,
            num_post_periods=2,
        )
        widths.append(result.id_ub - result.id_lb)

    for i in range(1, len(widths)):
        assert widths[i] >= widths[i - 1]


def test_edge_case_all_zero_beta():
    true_beta = np.zeros(5)
    l_vec = np.array([1, 0])

    result = compute_identified_set_sd(
        m_bar=0.1,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=3,
        num_post_periods=2,
    )

    assert isinstance(result, DeltaSDResult)
    assert result.id_lb <= 0 <= result.id_ub
