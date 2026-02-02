"""Tests for second differences with bias sign restrictions."""

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from moderndid.didhonest import (
    DeltaSDBResult,
    compute_conditional_cs_sdb,
    compute_identified_set_sdb,
)


@pytest.fixture
def simple_data():
    num_pre_periods = 3
    num_post_periods = 2
    betahat = np.array([0.1, -0.05, 0.02, 0.15, 0.25])
    sigma = np.eye(5) * 0.01
    l_vec = np.array([1, 0])
    return num_pre_periods, num_post_periods, betahat, sigma, l_vec


@pytest.fixture
def larger_data():
    num_pre_periods = 5
    num_post_periods = 4
    np.random.seed(42)
    betahat = np.array([0.05, -0.02, 0.03, 0.01, -0.01, 0.1, 0.15, 0.08, 0.12])
    sigma = np.eye(9) * 0.05
    l_vec = np.array([1, 0, 0, 0])
    return num_pre_periods, num_post_periods, betahat, sigma, l_vec


def test_compute_identified_set_sdb_basic(simple_data):
    num_pre_periods, num_post_periods, betahat, _, l_vec = simple_data

    result = compute_identified_set_sdb(
        m_bar=0.1,
        true_beta=betahat,
        l_vec=l_vec,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        bias_direction="positive",
    )

    assert isinstance(result, DeltaSDBResult)
    assert result.id_lb <= result.id_ub
    assert np.isfinite(result.id_lb)
    assert np.isfinite(result.id_ub)


def test_compute_identified_set_sdb_negative_bias():
    num_pre_periods = 3
    num_post_periods = 2
    betahat = np.array([0.1, -0.05, 0.02, -0.15, 0.25])
    l_vec = np.array([1, 0])

    result_pos = compute_identified_set_sdb(
        m_bar=0.1,
        true_beta=betahat,
        l_vec=l_vec,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        bias_direction="positive",
    )

    result_neg = compute_identified_set_sdb(
        m_bar=0.1,
        true_beta=betahat,
        l_vec=l_vec,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        bias_direction="negative",
    )

    assert result_pos.id_lb >= 0 or result_pos.id_ub >= result_neg.id_ub


def test_compute_identified_set_sdb_zero_smoothness(simple_data):
    num_pre_periods, num_post_periods, betahat, _, l_vec = simple_data

    result = compute_identified_set_sdb(
        m_bar=0,
        true_beta=betahat,
        l_vec=l_vec,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        bias_direction="positive",
    )

    observed_val = l_vec @ betahat[num_pre_periods:]
    assert result.id_lb >= 0
    assert result.id_ub >= observed_val


def test_compute_conditional_cs_sdb_single_post_period(fast_config):
    num_pre_periods = 3
    num_post_periods = 2
    betahat = np.array([0.1, -0.05, 0.02, 0.2, 0.0])
    sigma = np.eye(5) * 0.01
    l_vec = np.array([1, 0])

    result = compute_conditional_cs_sdb(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        l_vec=l_vec,
        m_bar=0.1,
        alpha=0.05,
        grid_points=fast_config["grid_points_medium"],
    )

    assert "grid" in result
    assert "accept" in result
    assert len(result["grid"]) == len(result["accept"])
    assert len(result["grid"]) == fast_config["grid_points_medium"]
    assert np.all((result["accept"] == 0) | (result["accept"] == 1))


def test_compute_conditional_cs_sdb_multiple_post_periods(simple_data, fast_config):
    num_pre_periods, num_post_periods, betahat, sigma, l_vec = simple_data

    result = compute_conditional_cs_sdb(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        l_vec=l_vec,
        m_bar=0.1,
        alpha=0.05,
        hybrid_flag="LF",
        grid_points=fast_config["grid_points_small"],
    )

    assert "grid" in result
    assert "accept" in result
    assert len(result["grid"]) == len(result["accept"])
    assert len(result["grid"]) == fast_config["grid_points_small"]


@pytest.mark.parametrize("hybrid_flag", ["FLCI", "LF", "ARP"])
def test_compute_conditional_cs_sdb_hybrid_flags(simple_data, hybrid_flag, fast_config):
    num_pre_periods, num_post_periods, betahat, sigma, l_vec = simple_data

    result = compute_conditional_cs_sdb(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        l_vec=l_vec,
        m_bar=0.05,
        alpha=0.05,
        hybrid_flag=hybrid_flag,
        grid_points=fast_config["grid_points_small"],
    )

    assert "grid" in result
    assert "accept" in result


@pytest.mark.parametrize("bias_direction", ["positive", "negative"])
def test_compute_conditional_cs_sdb_bias_directions(simple_data, bias_direction, fast_config):
    num_pre_periods, num_post_periods, betahat, sigma, l_vec = simple_data

    result = compute_conditional_cs_sdb(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        l_vec=l_vec,
        m_bar=0.1,
        alpha=0.05,
        bias_direction=bias_direction,
        grid_points=fast_config["grid_points_small"],
    )

    assert "grid" in result
    assert "accept" in result


def test_compute_conditional_cs_sdb_custom_grid_bounds(simple_data, fast_config):
    num_pre_periods, num_post_periods, betahat, sigma, l_vec = simple_data

    result = compute_conditional_cs_sdb(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        l_vec=l_vec,
        m_bar=0.1,
        alpha=0.05,
        grid_lb=-1,
        grid_ub=1,
        grid_points=fast_config["grid_points_small"],
    )

    assert np.min(result["grid"]) >= -1
    assert np.max(result["grid"]) <= 1


def test_compute_conditional_cs_sdb_post_period_moments_only(simple_data, fast_config):
    num_pre_periods, num_post_periods, betahat, sigma, l_vec = simple_data

    result_all = compute_conditional_cs_sdb(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        l_vec=l_vec,
        m_bar=0.05,
        alpha=0.05,
        post_period_moments_only=False,
        grid_points=fast_config["grid_points_small"],
    )

    result_post_only = compute_conditional_cs_sdb(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        l_vec=l_vec,
        m_bar=0.05,
        alpha=0.05,
        post_period_moments_only=True,
        grid_points=fast_config["grid_points_small"],
    )

    assert len(result_all["grid"]) == len(result_post_only["grid"])
    assert len(result_all["accept"]) == len(result_post_only["accept"])


def test_compute_identified_set_sdb_large_smoothness(simple_data):
    num_pre_periods, num_post_periods, betahat, _, l_vec = simple_data

    result = compute_identified_set_sdb(
        m_bar=10.0,
        true_beta=betahat,
        l_vec=l_vec,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        bias_direction="positive",
    )

    assert result.id_ub - result.id_lb > 0
