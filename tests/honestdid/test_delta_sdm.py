# pylint: disable=redefined-outer-name
"""Tests for second differences with monotonicity restrictions."""

import numpy as np
import pytest

from pydid.honestdid.delta.second_diff.sdm import (
    DeltaSDMResult,
    _create_sdm_constraint_matrix,
    _create_sdm_constraint_vector,
    compute_conditional_cs_sdm,
    compute_identified_set_sdm,
)


@pytest.fixture
def basic_setup():
    num_pre_periods = 4
    num_post_periods = 3
    betahat = np.array([0.1, -0.05, 0.02, -0.03, 0.15, 0.08, 0.12])

    sigma = np.eye(7) * 0.01
    np.fill_diagonal(sigma[1:], 0.005)
    np.fill_diagonal(sigma[:, 1:], 0.005)

    return {
        "num_pre_periods": num_pre_periods,
        "num_post_periods": num_post_periods,
        "betahat": betahat,
        "sigma": sigma,
    }


def test_compute_identified_set_sdm_basic(basic_setup):
    result = compute_identified_set_sdm(
        m_bar=0.1,
        true_beta=basic_setup["betahat"],
        l_vec=np.array([1, 0, 0]),
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        monotonicity_direction="increasing",
    )

    assert isinstance(result, DeltaSDMResult)
    assert result.id_lb <= result.id_ub
    assert not np.isnan(result.id_lb)
    assert not np.isnan(result.id_ub)


def test_compute_identified_set_sdm_decreasing(basic_setup):
    result = compute_identified_set_sdm(
        m_bar=0.1,
        true_beta=basic_setup["betahat"],
        l_vec=np.array([1, 0, 0]),
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        monotonicity_direction="decreasing",
    )

    assert isinstance(result, DeltaSDMResult)
    assert result.id_lb <= result.id_ub


def test_compute_identified_set_sdm_different_l_vec(basic_setup):
    l_vec = np.array([0.5, 0.3, 0.2])
    result = compute_identified_set_sdm(
        m_bar=0.05,
        true_beta=basic_setup["betahat"],
        l_vec=l_vec,
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
    )

    observed = l_vec @ basic_setup["betahat"][basic_setup["num_pre_periods"] :]
    assert result.id_lb <= observed <= result.id_ub


def test_compute_conditional_cs_sdm_single_post_period(fast_config):
    num_pre_periods = 3
    num_post_periods = 1
    betahat = np.array([0.1, -0.05, 0.02, 0.15])
    sigma = np.eye(4) * 0.01

    result = compute_conditional_cs_sdm(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        m_bar=0.1,
        alpha=0.05,
        grid_points=fast_config["grid_points_medium"],
    )

    assert isinstance(result, dict)
    assert "grid" in result
    assert "accept" in result
    assert len(result["grid"]) == len(result["accept"])
    assert len(result["grid"]) == fast_config["grid_points_medium"]
    assert all(x in [0, 1] for x in result["accept"])


def test_compute_conditional_cs_sdm_multiple_post_periods(basic_setup, fast_config):
    result = compute_conditional_cs_sdm(
        betahat=basic_setup["betahat"],
        sigma=basic_setup["sigma"],
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        m_bar=0.1,
        alpha=0.05,
        hybrid_flag="FLCI",
        grid_points=fast_config["grid_points_small"],
    )

    assert isinstance(result, dict)
    assert "grid" in result
    assert "accept" in result
    assert len(result["grid"]) == fast_config["grid_points_small"]


def test_compute_conditional_cs_sdm_arp_method(basic_setup, fast_config):
    result = compute_conditional_cs_sdm(
        betahat=basic_setup["betahat"],
        sigma=basic_setup["sigma"],
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        m_bar=0.1,
        alpha=0.05,
        hybrid_flag="ARP",
        grid_points=fast_config["grid_points_small"],
    )

    assert isinstance(result, dict)
    assert len(result["grid"]) == fast_config["grid_points_small"]


def test_compute_conditional_cs_sdm_lf_method_single_post(fast_config):
    num_pre_periods = 3
    num_post_periods = 1
    betahat = np.array([0.1, -0.05, 0.02, 0.15])
    sigma = np.eye(4) * 0.01

    result = compute_conditional_cs_sdm(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        m_bar=0.1,
        alpha=0.05,
        hybrid_flag="LF",
        grid_points=fast_config["grid_points_small"],
        seed=123,
    )

    assert isinstance(result, dict)
    assert len(result["grid"]) == fast_config["grid_points_small"]


def test_create_sdm_constraint_matrix():
    num_pre_periods = 3
    num_post_periods = 2

    A_sdm = _create_sdm_constraint_matrix(
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        monotonicity_direction="increasing",
    )

    assert isinstance(A_sdm, np.ndarray)
    assert A_sdm.shape[1] == num_pre_periods + num_post_periods

    A_sdm_dec = _create_sdm_constraint_matrix(
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        monotonicity_direction="decreasing",
    )

    assert A_sdm_dec.shape == A_sdm.shape


def test_create_sdm_constraint_vector():
    num_pre_periods = 3
    num_post_periods = 2
    m_bar = 0.1

    d_sdm = _create_sdm_constraint_vector(
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        m_bar=m_bar,
    )

    assert isinstance(d_sdm, np.ndarray)
    assert d_sdm.ndim == 1


def test_zero_m_bar(basic_setup):
    result = compute_identified_set_sdm(
        m_bar=0,
        true_beta=basic_setup["betahat"],
        l_vec=np.array([1, 0, 0]),
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
    )

    assert isinstance(result, DeltaSDMResult)
    assert result.id_lb <= result.id_ub


def test_custom_l_vec(basic_setup, fast_config):
    l_vec = np.array([0, 0, 1])

    result = compute_conditional_cs_sdm(
        betahat=basic_setup["betahat"],
        sigma=basic_setup["sigma"],
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        l_vec=l_vec,
        m_bar=0.1,
        grid_points=fast_config["grid_points_small"],
    )

    assert isinstance(result, dict)
    assert len(result["grid"]) == fast_config["grid_points_small"]


def test_post_period_moments_only_false(basic_setup, fast_config):
    result = compute_conditional_cs_sdm(
        betahat=basic_setup["betahat"],
        sigma=basic_setup["sigma"],
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        m_bar=0.1,
        post_period_moments_only=False,
        grid_points=fast_config["grid_points_small"],
    )

    assert isinstance(result, dict)
    assert len(result["grid"]) == fast_config["grid_points_small"]


def test_custom_grid_bounds(basic_setup, fast_config):
    result = compute_conditional_cs_sdm(
        betahat=basic_setup["betahat"],
        sigma=basic_setup["sigma"],
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        m_bar=0.1,
        grid_lb=-0.5,
        grid_ub=0.5,
        grid_points=fast_config["grid_points_small"],
    )

    assert result["grid"][0] >= -0.5
    assert result["grid"][-1] <= 0.5


def test_large_m_bar(basic_setup):
    result = compute_identified_set_sdm(
        m_bar=10.0,
        true_beta=basic_setup["betahat"],
        l_vec=np.array([1, 0, 0]),
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
    )

    assert isinstance(result, DeltaSDMResult)
    assert result.id_ub - result.id_lb >= 0


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
def test_different_alpha_levels(basic_setup, alpha, fast_config):
    result = compute_conditional_cs_sdm(
        betahat=basic_setup["betahat"],
        sigma=basic_setup["sigma"],
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        m_bar=0.1,
        alpha=alpha,
        grid_points=fast_config["grid_points_small"],
    )

    assert isinstance(result, dict)
    assert np.sum(result["accept"]) > 0


@pytest.mark.parametrize("direction", ["increasing", "decreasing"])
def test_monotonicity_directions(basic_setup, direction, fast_config):
    result = compute_conditional_cs_sdm(
        betahat=basic_setup["betahat"],
        sigma=basic_setup["sigma"],
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        m_bar=0.1,
        monotonicity_direction=direction,
        grid_points=fast_config["grid_points_small"],
    )

    assert isinstance(result, dict)
    assert len(result["grid"]) == fast_config["grid_points_small"]
