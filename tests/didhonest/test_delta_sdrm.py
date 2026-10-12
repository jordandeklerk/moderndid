"""Tests for second differences with relative magnitudes."""

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from moderndid.didhonest.delta.sdrm.sdrm import (
    DeltaSDRMResult,
    _create_sdrm_constraint_matrix,
    _create_sdrm_constraint_vector,
    compute_conditional_cs_sdrm,
    compute_identified_set_sdrm,
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


def test_compute_identified_set_sdrm_basic(basic_setup):
    result = compute_identified_set_sdrm(
        m_bar=1.0,
        true_beta=basic_setup["betahat"],
        l_vec=np.array([1, 0, 0]),
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
    )

    assert isinstance(result, DeltaSDRMResult)
    assert result.id_lb <= result.id_ub
    assert not np.isnan(result.id_lb)
    assert not np.isnan(result.id_ub)


def test_compute_identified_set_sdrm_zero_mbar(basic_setup):
    result = compute_identified_set_sdrm(
        m_bar=0.0,
        true_beta=basic_setup["betahat"],
        l_vec=np.array([1, 0, 0]),
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
    )

    assert isinstance(result, DeltaSDRMResult)
    assert result.id_lb <= result.id_ub


def test_compute_identified_set_sdrm_different_l_vec(basic_setup):
    l_vec = np.array([0.5, 0.3, 0.2])
    result = compute_identified_set_sdrm(
        m_bar=1.5,
        true_beta=basic_setup["betahat"],
        l_vec=l_vec,
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
    )

    observed = l_vec @ basic_setup["betahat"][basic_setup["num_pre_periods"] :]
    assert result.id_lb <= observed <= result.id_ub


def test_compute_conditional_cs_sdrm_single_post_period(fast_config):
    num_pre_periods = 3
    num_post_periods = 1
    betahat = np.array([0.1, -0.05, 0.02, 0.15])
    sigma = np.eye(4) * 0.01

    result = compute_conditional_cs_sdrm(
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


def test_compute_conditional_cs_sdrm_multiple_post_periods(basic_setup, fast_config):
    result = compute_conditional_cs_sdrm(
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
    assert "grid" in result
    assert "accept" in result
    assert len(result["grid"]) == fast_config["grid_points_small"]


def test_compute_conditional_cs_sdrm_arp_method(basic_setup, fast_config):
    result = compute_conditional_cs_sdrm(
        betahat=basic_setup["betahat"],
        sigma=basic_setup["sigma"],
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
        m_bar=1.0,
        alpha=0.05,
        hybrid_flag="ARP",
        grid_points=fast_config["grid_points_small"],
    )

    assert isinstance(result, dict)
    assert len(result["grid"]) == fast_config["grid_points_small"]


def test_compute_conditional_cs_sdrm_lf_method_single_post(fast_config):
    num_pre_periods = 3
    num_post_periods = 1
    betahat = np.array([0.1, -0.05, 0.02, 0.15])
    sigma = np.eye(4) * 0.01

    result = compute_conditional_cs_sdrm(
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


def test_create_sdrm_constraint_matrix():
    num_pre_periods = 3
    num_post_periods = 2

    A_sdrm = _create_sdrm_constraint_matrix(
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        m_bar=1.0,
        s=-1,
        max_positive=True,
    )

    assert isinstance(A_sdrm, np.ndarray)
    assert A_sdrm.shape[1] == num_pre_periods + num_post_periods

    A_sdrm_neg = _create_sdrm_constraint_matrix(
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        m_bar=1.0,
        s=-1,
        max_positive=False,
    )

    assert A_sdrm_neg.shape == A_sdrm.shape


def test_create_sdrm_constraint_vector():
    num_pre_periods = 3
    num_post_periods = 2

    A_sdrm = _create_sdrm_constraint_matrix(
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        m_bar=1.0,
        s=-1,
        max_positive=True,
    )

    d_sdrm = _create_sdrm_constraint_vector(A_sdrm)

    assert isinstance(d_sdrm, np.ndarray)
    assert d_sdrm.ndim == 1
    assert len(d_sdrm) == A_sdrm.shape[0]
    assert np.all(d_sdrm == 0)


def test_insufficient_pre_periods():
    with pytest.raises(ValueError, match="need at least two pre-treatment periods"):
        compute_identified_set_sdrm(
            m_bar=1.0, true_beta=np.array([0.1, 0.2]), l_vec=np.array([1.0]), num_pre_periods=1, num_post_periods=1
        )
    with pytest.raises(ValueError, match="need at least two pre-treatment periods"):
        compute_conditional_cs_sdrm(
            betahat=np.array([0.1, 0.2]),
            sigma=np.eye(2),
            num_pre_periods=1,
            num_post_periods=1,
            m_bar=1.0,
        )


def test_invalid_hybrid_flag(basic_setup):
    with pytest.raises(ValueError, match="hybrid_flag must be"):
        compute_conditional_cs_sdrm(
            betahat=basic_setup["betahat"],
            sigma=basic_setup["sigma"],
            num_pre_periods=basic_setup["num_pre_periods"],
            num_post_periods=basic_setup["num_post_periods"],
            m_bar=1.0,
            hybrid_flag="INVALID",
        )


def test_zero_m_bar(basic_setup):
    result = compute_identified_set_sdrm(
        m_bar=0,
        true_beta=basic_setup["betahat"],
        l_vec=np.array([1, 0, 0]),
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
    )

    assert isinstance(result, DeltaSDRMResult)
    assert result.id_lb <= result.id_ub


def test_custom_l_vec(basic_setup, fast_config):
    l_vec = np.array([0, 0, 1])

    result = compute_conditional_cs_sdrm(
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
    result = compute_conditional_cs_sdrm(
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
    result = compute_conditional_cs_sdrm(
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
    result = compute_identified_set_sdrm(
        m_bar=10.0,
        true_beta=basic_setup["betahat"],
        l_vec=np.array([1, 0, 0]),
        num_pre_periods=basic_setup["num_pre_periods"],
        num_post_periods=basic_setup["num_post_periods"],
    )

    assert isinstance(result, DeltaSDRMResult)
    assert result.id_ub - result.id_lb >= 0


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
def test_different_alpha_levels(basic_setup, alpha):
    result = compute_conditional_cs_sdrm(
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
    result = compute_conditional_cs_sdrm(
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
        A_sdrm = _create_sdrm_constraint_matrix(
            num_pre_periods=num_pre_periods,
            num_post_periods=num_post_periods,
            m_bar=1.0,
            s=s,
            max_positive=True,
        )
        assert isinstance(A_sdrm, np.ndarray)
        assert A_sdrm.shape[1] == num_pre_periods + num_post_periods


def test_constraint_matrix_properties():
    num_pre_periods = 3
    num_post_periods = 2

    A_no_drop = _create_sdrm_constraint_matrix(
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        m_bar=1.0,
        s=-1,
        max_positive=True,
        drop_zero=False,
    )

    assert A_no_drop.shape[1] == num_pre_periods + num_post_periods + 1

    A_drop = _create_sdrm_constraint_matrix(
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        m_bar=1.0,
        s=-1,
        max_positive=True,
        drop_zero=True,
    )

    assert A_drop.shape[1] == num_pre_periods + num_post_periods


def test_conditional_cs_sdrm_default_grid_reaches_past_a_noisy_long_horizon_set(noisy_long_horizon_event_study):
    l_vec = np.zeros(8)
    l_vec[-1] = 1.0

    result = compute_conditional_cs_sdrm(
        betahat=noisy_long_horizon_event_study["betahat"],
        sigma=noisy_long_horizon_event_study["sigma"],
        num_pre_periods=3,
        num_post_periods=8,
        l_vec=l_vec,
        m_bar=2.0,
        hybrid_flag="LF",
        grid_points=30,
    )

    accept = np.asarray(result["accept"]).astype(bool)
    grid = np.asarray(result["grid"])
    assert not accept[0]
    assert not accept[-1]
    assert (grid[accept].min(), grid[accept].max()) == pytest.approx((-10.9360204, 11.1328204), abs=1e-6)


@pytest.mark.parametrize("target,m_bar", [(0, 0.0), (3, 0.0), (3, 2.0)])
def test_conditional_cs_sdrm_default_grid_covers_the_set_on_a_linear_pre_trend(linear_trend_event_study, target, m_bar):
    l_vec = np.zeros(4)
    l_vec[target] = 1.0
    kwargs = {
        "betahat": linear_trend_event_study["betahat"],
        "sigma": linear_trend_event_study["sigma"],
        "num_pre_periods": 3,
        "num_post_periods": 4,
        "l_vec": l_vec,
        "m_bar": m_bar,
    }

    default = compute_conditional_cs_sdrm(**kwargs, grid_points=51)
    fine = compute_conditional_cs_sdrm(**kwargs, grid_points=161, grid_lb=-0.4, grid_ub=0.4)

    accept = np.asarray(default["accept"]).astype(bool)
    fine_accepted = np.asarray(fine["grid"])[np.asarray(fine["accept"]).astype(bool)]
    tolerance = (default["grid"][1] - default["grid"][0]) + 0.005
    assert not accept[0]
    assert not accept[-1]
    assert default["grid"][accept].min() == pytest.approx(fine_accepted.min(), abs=tolerance)
    assert default["grid"][accept].max() == pytest.approx(fine_accepted.max(), abs=tolerance)


@pytest.mark.parametrize("m_bar", [0.0, 0.1, 0.5, 1.0, 2.0])
def test_identified_set_sdrm_two_pre_periods_matches_closed_form(two_pre_period_event_study, m_bar):
    betahat = two_pre_period_event_study["betahat"]
    bound = 0.09 * m_bar

    first = compute_identified_set_sdrm(m_bar, betahat, np.array([1.0, 0.0]), 2, 2)
    second = compute_identified_set_sdrm(m_bar, betahat, np.array([0.0, 1.0]), 2, 2)
    both = compute_identified_set_sdrm(m_bar, betahat, np.array([1.0, 1.0]), 2, 2)

    assert (first.id_lb, first.id_ub) == pytest.approx((0.28 - bound, 0.28 + bound), abs=1e-8)
    assert (second.id_lb, second.id_ub) == pytest.approx((0.36 - 3 * bound, 0.36 + 3 * bound), abs=1e-8)
    assert (both.id_lb, both.id_ub) == pytest.approx((0.64 - 4 * bound, 0.64 + 4 * bound), abs=1e-8)


def test_conditional_cs_sdrm_two_pre_periods(two_pre_period_event_study):
    result = compute_conditional_cs_sdrm(
        betahat=two_pre_period_event_study["betahat"],
        sigma=two_pre_period_event_study["sigma"],
        num_pre_periods=2,
        num_post_periods=2,
        l_vec=np.array([1.0, 0.0]),
        m_bar=0.5,
        hybrid_flag="ARP",
        grid_points=101,
        grid_lb=-1.0,
        grid_ub=1.5,
    )

    assert np.asarray(result["accept"]).astype(int).tolist() == [0] * 32 + [1] * 37 + [0] * 32


@pytest.mark.parametrize(
    "hybrid_flag,post_only_range,all_rows_range",
    [("ARP", (-0.09, 0.65), (-0.2025, 0.685)), ("LF", (-0.0975, 0.6575), (-0.21, 0.695))],
)
def test_conditional_cs_sdrm_one_post_period_uses_only_post_period_rows(
    two_pre_period_event_study, hybrid_flag, post_only_range, all_rows_range
):
    kwargs = {
        "betahat": two_pre_period_event_study["betahat"][:3],
        "sigma": two_pre_period_event_study["sigma"][:3, :3],
        "num_pre_periods": 2,
        "num_post_periods": 1,
        "l_vec": np.array([1.0]),
        "m_bar": 0.0,
        "hybrid_flag": hybrid_flag,
        "grid_points": 1001,
        "grid_lb": -1.0,
        "grid_ub": 1.5,
    }

    post_only = compute_conditional_cs_sdrm(**kwargs)
    all_rows = compute_conditional_cs_sdrm(**kwargs, post_period_moments_only=False)

    post_only_grid = post_only["grid"][np.asarray(post_only["accept"]).astype(bool)]
    all_rows_grid = all_rows["grid"][np.asarray(all_rows["accept"]).astype(bool)]
    assert (post_only_grid.min(), post_only_grid.max()) == pytest.approx(post_only_range, abs=1e-9)
    assert (all_rows_grid.min(), all_rows_grid.max()) == pytest.approx(all_rows_range, abs=1e-9)
