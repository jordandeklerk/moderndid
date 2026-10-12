"""Tests for second differences with monotonicity restrictions."""

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from moderndid.didhonest.delta.sd.sdm import (
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


@pytest.mark.parametrize("m_bar,expected", [(0.01, (0.13, 0.15)), (0.1, (0.04, 0.15))])
def test_compute_identified_set_sdm_basic(monotone_event_study, m_bar, expected):
    result = compute_identified_set_sdm(
        m_bar=m_bar,
        true_beta=monotone_event_study["betahat"],
        l_vec=np.array([1, 0, 0]),
        num_pre_periods=4,
        num_post_periods=3,
        monotonicity_direction="increasing",
    )

    assert isinstance(result, DeltaSDMResult)
    assert (result.id_lb, result.id_ub) == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("m_bar,monotonicity_direction", [(0.005, "increasing"), (0.1, "decreasing")])
def test_compute_identified_set_sdm_empty_when_restriction_contradicts_pre_periods(
    monotone_event_study, m_bar, monotonicity_direction
):
    result = compute_identified_set_sdm(
        m_bar=m_bar,
        true_beta=monotone_event_study["betahat"],
        l_vec=np.array([1, 0, 0]),
        num_pre_periods=4,
        num_post_periods=3,
        monotonicity_direction=monotonicity_direction,
    )

    assert np.isnan(result.id_lb)
    assert np.isnan(result.id_ub)


@pytest.mark.parametrize("monotonicity_direction", ["increasing", "decreasing"])
def test_compute_identified_set_sdm_empty_when_pre_periods_are_not_monotone(basic_setup, monotonicity_direction):
    result = compute_identified_set_sdm(
        m_bar=1.0,
        true_beta=basic_setup["betahat"],
        l_vec=np.array([1, 0, 0]),
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        monotonicity_direction=monotonicity_direction,
    )

    assert np.isnan(result.id_lb)
    assert np.isnan(result.id_ub)


def test_compute_identified_set_sdm_different_l_vec(monotone_event_study):
    l_vec = np.array([0.5, 0.3, 0.2])
    result = compute_identified_set_sdm(
        m_bar=0.05,
        true_beta=monotone_event_study["betahat"],
        l_vec=l_vec,
        num_pre_periods=4,
        num_post_periods=3,
    )

    assert (result.id_lb, result.id_ub) == pytest.approx((-0.024, 0.123), abs=1e-8)


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


def test_zero_m_bar():
    result = compute_identified_set_sdm(
        m_bar=0,
        true_beta=np.array([-0.08, -0.06, -0.04, -0.02, 0.15, 0.08, 0.12]),
        l_vec=np.array([1, 0, 0]),
        num_pre_periods=4,
        num_post_periods=3,
    )

    assert isinstance(result, DeltaSDMResult)
    assert (result.id_lb, result.id_ub) == pytest.approx((0.13, 0.13), abs=1e-8)


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


def test_large_m_bar(monotone_event_study):
    result = compute_identified_set_sdm(
        m_bar=10.0,
        true_beta=monotone_event_study["betahat"],
        l_vec=np.array([1, 0, 0]),
        num_pre_periods=4,
        num_post_periods=3,
    )

    assert isinstance(result, DeltaSDMResult)
    assert (result.id_lb, result.id_ub) == pytest.approx((-9.86, 0.15), abs=1e-8)


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


def test_conditional_cs_sdm_single_post_period_conditional(one_post_event_study):
    kwargs = {
        "betahat": one_post_event_study["betahat"],
        "sigma": one_post_event_study["sigma"],
        "num_pre_periods": 4,
        "num_post_periods": 1,
        "l_vec": np.array([1.0]),
        "grid_points": 11,
        "grid_lb": 0.0,
        "grid_ub": 0.1,
    }

    result = compute_conditional_cs_sdm(**kwargs, m_bar=0.01, hybrid_flag="ARP", monotonicity_direction="increasing")

    assert result["accept"].astype(int).tolist() == [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]


@pytest.mark.parametrize(
    "alpha,direction,expected",
    [
        (0.01, "increasing", [0] + [1] * 29 + [0] * 11),
        (0.05, "decreasing", [0] * 10 + [1] * 27 + [0] * 4),
        (0.10, "increasing", [0] * 2 + [1] * 25 + [0] * 14),
    ],
)
def test_conditional_cs_sdm_flci_hybrid_rejects_the_ends_of_its_first_stage_interval(
    basic_setup, alpha, direction, expected
):
    result = compute_conditional_cs_sdm(
        betahat=basic_setup["betahat"],
        sigma=basic_setup["sigma"],
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        m_bar=0.1,
        alpha=alpha,
        monotonicity_direction=direction,
        hybrid_flag="FLCI",
        grid_points=41,
    )

    assert np.asarray(result["accept"]).astype(int).tolist() == expected


@pytest.mark.parametrize("num_post_periods", [1, 2])
@pytest.mark.parametrize("hybrid_flag", ["LF", "ARP"])
def test_conditional_cs_sdm_default_grid_covers_an_effect_far_from_zero(
    far_effect_event_study, num_post_periods, hybrid_flag
):
    size = 3 + num_post_periods
    kwargs = {
        "betahat": far_effect_event_study["betahat"][:size],
        "sigma": far_effect_event_study["sigma"][:size, :size],
        "num_pre_periods": 3,
        "num_post_periods": num_post_periods,
        "l_vec": np.eye(num_post_periods)[0],
        "m_bar": 0.005,
        "monotonicity_direction": "increasing",
        "hybrid_flag": hybrid_flag,
    }

    default = compute_conditional_cs_sdm(**kwargs, grid_points=101)
    fine = compute_conditional_cs_sdm(**kwargs, grid_points=401, grid_lb=0.4, grid_ub=0.6)

    accept = np.asarray(default["accept"]).astype(bool)
    fine_accepted = np.asarray(fine["grid"])[np.asarray(fine["accept"]).astype(bool)]
    tolerance = (default["grid"][1] - default["grid"][0]) + 0.0005
    assert not accept[0]
    assert not accept[-1]
    assert default["grid"][accept].min() == pytest.approx(fine_accepted.min(), abs=tolerance)
    assert default["grid"][accept].max() == pytest.approx(fine_accepted.max(), abs=tolerance)
