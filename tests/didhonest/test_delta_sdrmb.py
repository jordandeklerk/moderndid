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


def test_compute_conditional_cs_sdrmb_multiple_post_periods(basic_setup):
    result = compute_conditional_cs_sdrmb(
        betahat=basic_setup["betahat"],
        sigma=basic_setup["sigma"],
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        m_bar=0.5,
        alpha=0.05,
        grid_points=40,
    )

    assert isinstance(result, dict)
    assert len(result["grid"]) == 40
    assert np.any(result["accept"] == 1)
    assert result["accept"][0] == 0
    assert result["accept"][-1] == 0


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

    with pytest.raises(ValueError, match="need at least two pre-treatment periods"):
        compute_conditional_cs_sdrmb(
            betahat=betahat,
            sigma=sigma,
            num_pre_periods=num_pre_periods,
            num_post_periods=num_post_periods,
        )
    with pytest.raises(ValueError, match="need at least two pre-treatment periods"):
        compute_identified_set_sdrmb(
            m_bar=1.0,
            true_beta=betahat,
            l_vec=np.array([1.0, 0.0, 0.0]),
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


def test_conditional_cs_sdrmb_default_grid_covers_the_set_on_a_linear_pre_trend(linear_trend_event_study):
    kwargs = {
        "betahat": linear_trend_event_study["betahat"],
        "sigma": linear_trend_event_study["sigma"],
        "num_pre_periods": 3,
        "num_post_periods": 4,
        "l_vec": np.array([0.0, 0.0, 0.0, 1.0]),
        "m_bar": 0.5,
        "bias_direction": "positive",
    }

    default = compute_conditional_cs_sdrmb(**kwargs, grid_points=51)
    fine = compute_conditional_cs_sdrmb(**kwargs, grid_points=161, grid_lb=-0.4, grid_ub=0.4)

    accept = np.asarray(default["accept"]).astype(bool)
    fine_accepted = np.asarray(fine["grid"])[np.asarray(fine["accept"]).astype(bool)]
    tolerance = (default["grid"][1] - default["grid"][0]) + 0.005
    assert not accept[0]
    assert not accept[-1]
    assert default["grid"][accept].min() == pytest.approx(fine_accepted.min(), abs=tolerance)
    assert default["grid"][accept].max() == pytest.approx(fine_accepted.max(), abs=tolerance)


@pytest.mark.parametrize(
    "hybrid_flag,post_only_range,all_rows_range",
    [("ARP", (-0.09, 0.58), (-0.2025, 0.5875)), ("LF", (-0.0975, 0.585), (-0.21, 0.5925))],
)
def test_conditional_cs_sdrmb_one_post_period_uses_only_post_period_rows(
    two_pre_period_event_study, hybrid_flag, post_only_range, all_rows_range
):
    kwargs = {
        "betahat": two_pre_period_event_study["betahat"][:3],
        "sigma": two_pre_period_event_study["sigma"][:3, :3],
        "num_pre_periods": 2,
        "num_post_periods": 1,
        "l_vec": np.array([1.0]),
        "m_bar": 0.0,
        "bias_direction": "positive",
        "hybrid_flag": hybrid_flag,
        "grid_points": 1001,
        "grid_lb": -1.0,
        "grid_ub": 1.5,
    }

    post_only = compute_conditional_cs_sdrmb(**kwargs)
    all_rows = compute_conditional_cs_sdrmb(**kwargs, post_period_moments_only=False)

    post_only_grid = post_only["grid"][np.asarray(post_only["accept"]).astype(bool)]
    all_rows_grid = all_rows["grid"][np.asarray(all_rows["accept"]).astype(bool)]
    assert (post_only_grid.min(), post_only_grid.max()) == pytest.approx(post_only_range, abs=1e-9)
    assert (all_rows_grid.min(), all_rows_grid.max()) == pytest.approx(all_rows_range, abs=1e-9)
