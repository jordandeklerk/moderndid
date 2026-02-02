"""Tests for second differences with relative magnitudes and bias restrictions."""

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from moderndid.didhonest import (
    DeltaSDRMBResult,
    compute_conditional_cs_sdrmb,
    compute_identified_set_sdrmb,
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


def test_compute_identified_set_sdrmb_basic(basic_setup):
    result = compute_identified_set_sdrmb(
        m_bar=1.0,
        true_beta=basic_setup["betahat"],
        l_vec=np.array([1, 0, 0]),
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        bias_direction="positive",
    )

    assert isinstance(result, DeltaSDRMBResult)
    assert result.id_lb <= result.id_ub
    assert not np.isnan(result.id_lb)
    assert not np.isnan(result.id_ub)


def test_compute_identified_set_sdrmb_zero_mbar(basic_setup):
    result = compute_identified_set_sdrmb(
        m_bar=0.0,
        true_beta=basic_setup["betahat"],
        l_vec=np.array([1, 0, 0]),
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        bias_direction="positive",
    )

    assert isinstance(result, DeltaSDRMBResult)
    assert result.id_lb <= result.id_ub


def test_compute_identified_set_sdrmb_different_l_vec(basic_setup):
    l_vec = np.array([0.5, 0.3, 0.2])
    result = compute_identified_set_sdrmb(
        m_bar=1.5,
        true_beta=basic_setup["betahat"],
        l_vec=l_vec,
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        bias_direction="positive",
    )

    observed = l_vec @ basic_setup["betahat"][basic_setup["num_pre_periods"] :]
    assert result.id_lb <= observed <= result.id_ub


@pytest.mark.parametrize("bias_direction", ["positive", "negative"])
def test_compute_identified_set_sdrmb_bias_directions(basic_setup, bias_direction):
    result = compute_identified_set_sdrmb(
        m_bar=1.0,
        true_beta=basic_setup["betahat"],
        l_vec=np.array([1, 0, 0]),
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        bias_direction=bias_direction,
    )

    assert isinstance(result, DeltaSDRMBResult)
    assert result.id_lb <= result.id_ub


def test_compute_conditional_cs_sdrmb_single_post_period(fast_config):
    num_pre_periods = 3
    num_post_periods = 1
    betahat = np.array([0.1, -0.05, 0.02, 0.15])
    sigma = np.eye(4) * 0.01

    result = compute_conditional_cs_sdrmb(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        m_bar=1.0,
        alpha=0.05,
        grid_points=fast_config["grid_points_medium"],
    )

    assert isinstance(result, dict)
    assert "grid" in result
    assert "accept" in result
    assert len(result["grid"]) == len(result["accept"])
    assert len(result["grid"]) == fast_config["grid_points_medium"]
    assert all(x in [0, 1] for x in result["accept"])


def test_compute_conditional_cs_sdrmb_multiple_post_periods(basic_setup, fast_config):
    result = compute_conditional_cs_sdrmb(
        betahat=basic_setup["betahat"],
        sigma=basic_setup["sigma"],
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        m_bar=0.5,
        alpha=0.05,
        grid_points=fast_config["grid_points_small"],
    )

    assert isinstance(result, dict)
    assert len(result["grid"]) == fast_config["grid_points_small"]
    assert np.any(result["accept"] == 1)


@pytest.mark.parametrize("hybrid_flag", ["LF", "ARP"])
def test_compute_conditional_cs_sdrmb_hybrid_flags(basic_setup, fast_config, hybrid_flag):
    result = compute_conditional_cs_sdrmb(
        betahat=basic_setup["betahat"],
        sigma=basic_setup["sigma"],
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        m_bar=0.5,
        alpha=0.05,
        hybrid_flag=hybrid_flag,
        grid_points=fast_config["grid_points_small"],
    )

    assert isinstance(result, dict)
    assert len(result["grid"]) == fast_config["grid_points_small"]


@pytest.mark.parametrize("m_bar", [0.0, 0.5, 1.0, 2.0])
def test_compute_conditional_cs_sdrmb_different_mbars(basic_setup, fast_config, m_bar):
    result = compute_conditional_cs_sdrmb(
        betahat=basic_setup["betahat"],
        sigma=basic_setup["sigma"],
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        m_bar=m_bar,
        alpha=0.05,
        grid_points=fast_config["grid_points_small"],
    )

    assert isinstance(result, dict)
    assert len(result["grid"]) == fast_config["grid_points_small"]


def test_compute_conditional_cs_sdrmb_custom_l_vec(basic_setup, fast_config):
    l_vec = np.array([0.2, 0.3, 0.5])

    result = compute_conditional_cs_sdrmb(
        betahat=basic_setup["betahat"],
        sigma=basic_setup["sigma"],
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        l_vec=l_vec,
        m_bar=1.0,
        alpha=0.05,
        grid_points=fast_config["grid_points_small"],
    )

    assert isinstance(result, dict)
    assert len(result["grid"]) == fast_config["grid_points_small"]


def test_compute_conditional_cs_sdrmb_custom_grid_bounds(basic_setup, fast_config):
    result = compute_conditional_cs_sdrmb(
        betahat=basic_setup["betahat"],
        sigma=basic_setup["sigma"],
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        m_bar=1.0,
        alpha=0.05,
        grid_lb=-1.0,
        grid_ub=1.0,
        grid_points=fast_config["grid_points_small"],
    )

    assert result["grid"][0] == pytest.approx(-1.0)
    assert result["grid"][-1] == pytest.approx(1.0)


@pytest.mark.parametrize("post_period_moments_only", [True, False])
def test_compute_conditional_cs_sdrmb_post_period_moments(basic_setup, fast_config, post_period_moments_only):
    result = compute_conditional_cs_sdrmb(
        betahat=basic_setup["betahat"],
        sigma=basic_setup["sigma"],
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        m_bar=1.0,
        post_period_moments_only=post_period_moments_only,
        alpha=0.05,
        grid_points=fast_config["grid_points_small"],
    )

    assert isinstance(result, dict)
    assert len(result["grid"]) == fast_config["grid_points_small"]


def test_compute_conditional_cs_sdrmb_invalid_hybrid_flag(basic_setup):
    with pytest.raises(ValueError, match="hybrid_flag must be"):
        compute_conditional_cs_sdrmb(
            betahat=basic_setup["betahat"],
            sigma=basic_setup["sigma"],
            num_pre_periods=basic_setup["num_pre_periods"],
            num_post_periods=basic_setup["num_post_periods"],
            hybrid_flag="invalid",
        )


def test_compute_conditional_cs_sdrmb_insufficient_pre_periods():
    num_pre_periods = 1
    num_post_periods = 3
    betahat = np.array([0.1, 0.2, 0.15, 0.1])
    sigma = np.eye(4) * 0.01

    with pytest.raises(ValueError, match="Not enough pre-periods"):
        compute_conditional_cs_sdrmb(
            betahat=betahat,
            sigma=sigma,
            num_pre_periods=num_pre_periods,
            num_post_periods=num_post_periods,
        )


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
def test_compute_conditional_cs_sdrmb_different_alphas(basic_setup, fast_config, alpha):
    result = compute_conditional_cs_sdrmb(
        betahat=basic_setup["betahat"],
        sigma=basic_setup["sigma"],
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        m_bar=1.0,
        alpha=alpha,
        grid_points=fast_config["grid_points_small"],
    )

    assert isinstance(result, dict)
    assert len(result["grid"]) == fast_config["grid_points_small"]


def test_compute_conditional_cs_sdrmb_negative_bias(basic_setup, fast_config):
    result = compute_conditional_cs_sdrmb(
        betahat=basic_setup["betahat"],
        sigma=basic_setup["sigma"],
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        m_bar=1.0,
        bias_direction="negative",
        alpha=0.05,
        grid_points=fast_config["grid_points_small"],
    )

    assert isinstance(result, dict)
    assert len(result["grid"]) == fast_config["grid_points_small"]
