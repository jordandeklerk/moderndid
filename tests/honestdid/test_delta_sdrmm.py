# pylint: disable=redefined-outer-name
"""Tests for second differences with relative magnitudes and monotonicity."""

import numpy as np
import pytest

from pydid.honestdid.delta.second_diff_relative_magnitude.sdrmm import (
    DeltaSDRMMResult,
    _create_sdrmm_constraint_matrix,
    _create_sdrmm_constraint_vector,
    compute_conditional_cs_sdrmm,
    compute_identified_set_sdrmm,
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


def test_compute_identified_set_sdrmm_basic(basic_setup):
    result = compute_identified_set_sdrmm(
        m_bar=1.0,
        true_beta=basic_setup["betahat"],
        l_vec=np.array([1, 0, 0]),
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        monotonicity_direction="increasing",
    )

    assert isinstance(result, DeltaSDRMMResult)
    assert result.id_lb <= result.id_ub
    assert not np.isnan(result.id_lb)
    assert not np.isnan(result.id_ub)


def test_compute_identified_set_sdrmm_decreasing(basic_setup):
    result = compute_identified_set_sdrmm(
        m_bar=1.0,
        true_beta=basic_setup["betahat"],
        l_vec=np.array([1, 0, 0]),
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        monotonicity_direction="decreasing",
    )

    assert isinstance(result, DeltaSDRMMResult)
    assert result.id_lb <= result.id_ub


def test_compute_identified_set_sdrmm_zero_mbar(basic_setup):
    result = compute_identified_set_sdrmm(
        m_bar=0.0,
        true_beta=basic_setup["betahat"],
        l_vec=np.array([1, 0, 0]),
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
    )

    assert isinstance(result, DeltaSDRMMResult)
    assert result.id_lb <= result.id_ub


def test_compute_identified_set_sdrmm_different_l_vec(basic_setup):
    l_vec = np.array([0.5, 0.3, 0.2])
    result = compute_identified_set_sdrmm(
        m_bar=1.5,
        true_beta=basic_setup["betahat"],
        l_vec=l_vec,
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
    )

    observed = l_vec @ basic_setup["betahat"][basic_setup["num_pre_periods"] :]
    assert result.id_lb <= observed <= result.id_ub


def test_compute_conditional_cs_sdrmm_single_post_period(fast_config):
    num_pre_periods = 3
    num_post_periods = 1
    betahat = np.array([0.1, -0.05, 0.02, 0.15])
    sigma = np.eye(4) * 0.01

    result = compute_conditional_cs_sdrmm(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        m_bar=1.0,
        alpha=0.05,
        monotonicity_direction="increasing",
        grid_points=fast_config["grid_points_medium"],
    )

    assert isinstance(result, dict)
    assert "grid" in result
    assert "accept" in result
    assert len(result["grid"]) == len(result["accept"])
    assert len(result["grid"]) == fast_config["grid_points_medium"]
    assert all(x in [0, 1] for x in result["accept"])


def test_compute_conditional_cs_sdrmm_multiple_post_periods(basic_setup, fast_config):
    result = compute_conditional_cs_sdrmm(
        betahat=basic_setup["betahat"],
        sigma=basic_setup["sigma"],
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        m_bar=1.0,
        alpha=0.05,
        hybrid_flag="ARP",
        monotonicity_direction="increasing",
        grid_points=fast_config["grid_points_small"],
    )

    assert isinstance(result, dict)
    assert "grid" in result
    assert "accept" in result
    assert len(result["grid"]) == fast_config["grid_points_small"]


def test_compute_conditional_cs_sdrmm_flci_method(basic_setup, fast_config):
    result = compute_conditional_cs_sdrmm(
        betahat=basic_setup["betahat"],
        sigma=basic_setup["sigma"],
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        m_bar=1.0,
        alpha=0.05,
        hybrid_flag="FLCI",
        grid_points=fast_config["grid_points_small"],
    )

    assert isinstance(result, dict)
    assert len(result["grid"]) == fast_config["grid_points_small"]


def test_compute_conditional_cs_sdrmm_return_length(basic_setup, fast_config):
    result = compute_conditional_cs_sdrmm(
        betahat=basic_setup["betahat"],
        sigma=basic_setup["sigma"],
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        m_bar=1.0,
        return_length=True,
        grid_points=fast_config["grid_points_medium"],
    )

    assert isinstance(result, float)
    assert result >= 0


def test_compute_conditional_cs_sdrmm_lf_method_single_post(fast_config):
    num_pre_periods = 3
    num_post_periods = 1
    betahat = np.array([0.1, -0.05, 0.02, 0.15])
    sigma = np.eye(4) * 0.01

    result = compute_conditional_cs_sdrmm(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        m_bar=1.0,
        alpha=0.05,
        hybrid_flag="LF",
        grid_points=fast_config["grid_points_small"],
        seed=123,
    )

    assert isinstance(result, dict)
    assert len(result["grid"]) == fast_config["grid_points_small"]


def test_create_sdrmm_constraint_matrix():
    num_pre_periods = 3
    num_post_periods = 2

    A_sdrmm = _create_sdrmm_constraint_matrix(
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        m_bar=1.0,
        s=-1,
        max_positive=True,
        monotonicity_direction="increasing",
    )

    assert isinstance(A_sdrmm, np.ndarray)
    assert A_sdrmm.shape[1] == num_pre_periods + num_post_periods

    A_sdrmm_dec = _create_sdrmm_constraint_matrix(
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        m_bar=1.0,
        s=-1,
        max_positive=True,
        monotonicity_direction="decreasing",
    )

    assert A_sdrmm_dec.shape[1] == A_sdrmm.shape[1]


def test_create_sdrmm_constraint_vector():
    num_pre_periods = 3
    num_post_periods = 2

    A_sdrmm = _create_sdrmm_constraint_matrix(
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        m_bar=1.0,
        s=-1,
        max_positive=True,
    )

    d_sdrmm = _create_sdrmm_constraint_vector(A_sdrmm)

    assert isinstance(d_sdrmm, np.ndarray)
    assert d_sdrmm.ndim == 1
    assert len(d_sdrmm) == A_sdrmm.shape[0]
    assert np.all(d_sdrmm == 0)


def test_insufficient_pre_periods():
    with pytest.raises(ValueError, match="Not enough pre-periods"):
        compute_conditional_cs_sdrmm(
            betahat=np.array([0.1, 0.2]),
            sigma=np.eye(2),
            num_pre_periods=1,
            num_post_periods=1,
            m_bar=1.0,
        )


def test_invalid_hybrid_flag(basic_setup):
    with pytest.raises(ValueError, match="hybrid_flag must be"):
        compute_conditional_cs_sdrmm(
            betahat=basic_setup["betahat"],
            sigma=basic_setup["sigma"],
            num_pre_periods=basic_setup["num_pre_periods"],
            num_post_periods=basic_setup["num_post_periods"],
            m_bar=1.0,
            hybrid_flag="INVALID",
        )


def test_different_monotonicity_directions(basic_setup, fast_config):
    result_inc = compute_conditional_cs_sdrmm(
        betahat=basic_setup["betahat"],
        sigma=basic_setup["sigma"],
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        m_bar=1.0,
        monotonicity_direction="increasing",
        grid_points=fast_config["grid_points_small"],
    )

    result_dec = compute_conditional_cs_sdrmm(
        betahat=basic_setup["betahat"],
        sigma=basic_setup["sigma"],
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        m_bar=1.0,
        monotonicity_direction="decreasing",
        grid_points=fast_config["grid_points_small"],
    )

    assert isinstance(result_inc, dict)
    assert isinstance(result_dec, dict)


def test_custom_l_vec(basic_setup, fast_config):
    l_vec = np.array([0, 0, 1])

    result = compute_conditional_cs_sdrmm(
        betahat=basic_setup["betahat"],
        sigma=basic_setup["sigma"],
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        l_vec=l_vec,
        m_bar=1.0,
        grid_points=fast_config["grid_points_small"],
    )

    assert isinstance(result, dict)
    assert len(result["grid"]) == fast_config["grid_points_small"]


def test_post_period_moments_only_false(basic_setup, fast_config):
    result = compute_conditional_cs_sdrmm(
        betahat=basic_setup["betahat"],
        sigma=basic_setup["sigma"],
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        m_bar=1.0,
        post_period_moments_only=False,
        grid_points=fast_config["grid_points_small"],
    )

    assert isinstance(result, dict)
    assert len(result["grid"]) == fast_config["grid_points_small"]


def test_custom_grid_bounds(basic_setup, fast_config):
    result = compute_conditional_cs_sdrmm(
        betahat=basic_setup["betahat"],
        sigma=basic_setup["sigma"],
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        m_bar=1.0,
        grid_lb=-0.5,
        grid_ub=0.5,
        grid_points=fast_config["grid_points_small"],
    )

    assert result["grid"][0] >= -0.5
    assert result["grid"][-1] <= 0.5


def test_large_m_bar(basic_setup):
    result = compute_identified_set_sdrmm(
        m_bar=10.0,
        true_beta=basic_setup["betahat"],
        l_vec=np.array([1, 0, 0]),
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
    )

    assert isinstance(result, DeltaSDRMMResult)
    assert result.id_ub - result.id_lb >= 0


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
def test_different_alpha_levels(basic_setup, alpha, fast_config):
    result = compute_conditional_cs_sdrmm(
        betahat=basic_setup["betahat"],
        sigma=basic_setup["sigma"],
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        m_bar=1.0,
        alpha=alpha,
        grid_points=fast_config["grid_points_small"],
    )

    assert isinstance(result, dict)
    assert np.sum(result["accept"]) > 0


@pytest.mark.parametrize("m_bar", [0.5, 1.0, 2.0])
def test_different_m_bar_values(basic_setup, m_bar, fast_config):
    result = compute_conditional_cs_sdrmm(
        betahat=basic_setup["betahat"],
        sigma=basic_setup["sigma"],
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        m_bar=m_bar,
        grid_points=fast_config["grid_points_small"],
    )

    assert isinstance(result, dict)
    assert len(result["grid"]) == fast_config["grid_points_small"]


def test_multiple_s_values():
    num_pre_periods = 4
    num_post_periods = 2

    for s in [-(num_pre_periods - 2), -1, 0]:
        A_sdrmm = _create_sdrmm_constraint_matrix(
            num_pre_periods=num_pre_periods,
            num_post_periods=num_post_periods,
            m_bar=1.0,
            s=s,
            max_positive=True,
        )
        assert isinstance(A_sdrmm, np.ndarray)
        assert A_sdrmm.shape[1] == num_pre_periods + num_post_periods


def test_constraint_matrix_properties():
    num_pre_periods = 3
    num_post_periods = 2

    A_no_drop = _create_sdrmm_constraint_matrix(
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        m_bar=1.0,
        s=-1,
        max_positive=True,
        drop_zero=False,
    )

    assert A_no_drop.shape[1] == num_pre_periods + num_post_periods + 1

    A_drop = _create_sdrmm_constraint_matrix(
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        m_bar=1.0,
        s=-1,
        max_positive=True,
        drop_zero=True,
    )

    assert A_drop.shape[1] == num_pre_periods + num_post_periods
