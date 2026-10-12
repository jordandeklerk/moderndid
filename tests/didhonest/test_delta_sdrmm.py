"""Tests for second differences with relative magnitudes and monotonicity."""

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from moderndid.didhonest.delta.sdrm.sdrmm import (
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


@pytest.mark.parametrize("m_bar,expected", [(0.0, (0.14, 0.14)), (1.0, (0.13, 0.15)), (2.0, (0.12, 0.15))])
def test_compute_identified_set_sdrmm_basic(monotone_event_study, m_bar, expected):
    result = compute_identified_set_sdrmm(
        m_bar=m_bar,
        true_beta=monotone_event_study["betahat"],
        l_vec=np.array([1, 0, 0]),
        num_pre_periods=4,
        num_post_periods=3,
        monotonicity_direction="increasing",
    )

    assert isinstance(result, DeltaSDRMMResult)
    assert (result.id_lb, result.id_ub) == pytest.approx(expected, abs=1e-8)


def test_compute_identified_set_sdrmm_decreasing(monotone_event_study):
    result = compute_identified_set_sdrmm(
        m_bar=1.0,
        true_beta=monotone_event_study["betahat"],
        l_vec=np.array([1, 0, 0]),
        num_pre_periods=4,
        num_post_periods=3,
        monotonicity_direction="decreasing",
    )

    assert np.isnan(result.id_lb)
    assert np.isnan(result.id_ub)


@pytest.mark.parametrize("monotonicity_direction", ["increasing", "decreasing"])
def test_compute_identified_set_sdrmm_empty_when_pre_periods_are_not_monotone(basic_setup, monotonicity_direction):
    result = compute_identified_set_sdrmm(
        m_bar=1.0,
        true_beta=basic_setup["betahat"],
        l_vec=np.array([1, 0, 0]),
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        monotonicity_direction=monotonicity_direction,
    )

    assert np.isnan(result.id_lb)
    assert np.isnan(result.id_ub)


def test_compute_identified_set_sdrmm_different_l_vec(monotone_event_study):
    l_vec = np.array([0.5, 0.3, 0.2])
    result = compute_identified_set_sdrmm(
        m_bar=1.5,
        true_beta=monotone_event_study["betahat"],
        l_vec=l_vec,
        num_pre_periods=4,
        num_post_periods=3,
    )

    assert (result.id_lb, result.id_ub) == pytest.approx((0.067, 0.123), abs=1e-8)


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
    with pytest.raises(ValueError, match="need at least two pre-treatment periods"):
        compute_identified_set_sdrmm(
            m_bar=1.0, true_beta=np.array([0.1, 0.2]), l_vec=np.array([1.0]), num_pre_periods=1, num_post_periods=1
        )
    with pytest.raises(ValueError, match="need at least two pre-treatment periods"):
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


def test_large_m_bar(monotone_event_study):
    result = compute_identified_set_sdrmm(
        m_bar=10.0,
        true_beta=monotone_event_study["betahat"],
        l_vec=np.array([1, 0, 0]),
        num_pre_periods=4,
        num_post_periods=3,
    )

    assert isinstance(result, DeltaSDRMMResult)
    assert (result.id_lb, result.id_ub) == pytest.approx((0.04, 0.15), abs=1e-8)


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
def test_different_alpha_levels(basic_setup, alpha):
    result = compute_conditional_cs_sdrmm(
        betahat=basic_setup["betahat"],
        sigma=basic_setup["sigma"],
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        m_bar=1.0,
        alpha=alpha,
        grid_points=40,
    )

    assert isinstance(result, dict)
    assert np.sum(result["accept"]) > 0
    assert result["accept"][0] == 0
    assert result["accept"][-1] == 0


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


def test_conditional_cs_sdrmm_default_grid_covers_the_set_on_a_linear_pre_trend(linear_trend_event_study):
    kwargs = {
        "betahat": linear_trend_event_study["betahat"],
        "sigma": linear_trend_event_study["sigma"],
        "num_pre_periods": 3,
        "num_post_periods": 4,
        "l_vec": np.array([0.0, 0.0, 0.0, 1.0]),
        "m_bar": 0.5,
        "monotonicity_direction": "increasing",
    }

    default = compute_conditional_cs_sdrmm(**kwargs, grid_points=51)
    fine = compute_conditional_cs_sdrmm(**kwargs, grid_points=161, grid_lb=-0.4, grid_ub=0.4)

    accept = np.asarray(default["accept"]).astype(bool)
    fine_accepted = np.asarray(fine["grid"])[np.asarray(fine["accept"]).astype(bool)]
    tolerance = (default["grid"][1] - default["grid"][0]) + 0.005
    assert not accept[0]
    assert not accept[-1]
    assert default["grid"][accept].min() == pytest.approx(fine_accepted.min(), abs=tolerance)
    assert default["grid"][accept].max() == pytest.approx(fine_accepted.max(), abs=tolerance)


@pytest.mark.parametrize(
    "hybrid_flag,post_only_range,all_rows_range",
    [("ARP", (-0.1975, 0.355), (-0.285, 0.3875)), ("LF", (-0.205, 0.36), (-0.2925, 0.3925))],
)
def test_conditional_cs_sdrmm_one_post_period_uses_only_post_period_rows(
    monotone_event_study, hybrid_flag, post_only_range, all_rows_range
):
    kwargs = {
        "betahat": monotone_event_study["betahat"][:5],
        "sigma": monotone_event_study["sigma"][:5, :5],
        "num_pre_periods": 4,
        "num_post_periods": 1,
        "l_vec": np.array([1.0]),
        "m_bar": 0.0,
        "monotonicity_direction": "increasing",
        "hybrid_flag": hybrid_flag,
        "grid_points": 1001,
        "grid_lb": -1.0,
        "grid_ub": 1.5,
    }

    post_only = compute_conditional_cs_sdrmm(**kwargs)
    all_rows = compute_conditional_cs_sdrmm(**kwargs, post_period_moments_only=False)

    post_only_grid = post_only["grid"][np.asarray(post_only["accept"]).astype(bool)]
    all_rows_grid = all_rows["grid"][np.asarray(all_rows["accept"]).astype(bool)]
    assert (post_only_grid.min(), post_only_grid.max()) == pytest.approx(post_only_range, abs=1e-9)
    assert (all_rows_grid.min(), all_rows_grid.max()) == pytest.approx(all_rows_range, abs=1e-9)
