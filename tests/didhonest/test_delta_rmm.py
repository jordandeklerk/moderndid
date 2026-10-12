"""Tests for relative magnitudes with monotonicity restriction."""

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from moderndid.didhonest import (
    DeltaRMMResult,
    compute_conditional_cs_rmm,
    compute_identified_set_rmm,
)
from moderndid.didhonest.delta.rm.rmm import (
    _create_relative_magnitudes_monotonicity_constraint_matrix,
    _create_relative_magnitudes_monotonicity_constraint_vector,
)


@pytest.fixture
def constraint_matrix_params():
    return {
        "num_pre_periods": 3,
        "num_post_periods": 2,
        "m_bar": 1,
        "s": 0,
        "max_positive": True,
        "monotonicity_direction": "increasing",
    }


@pytest.fixture
def simple_event_study_data():
    num_pre_periods = 3
    num_post_periods = 2
    betahat = np.array([-0.1, 0.05, -0.02, 0.3, 0.5])
    sigma = np.eye(5) * 0.01
    true_beta = betahat.copy()
    l_vec = np.array([1, 0])
    return num_pre_periods, num_post_periods, betahat, sigma, true_beta, l_vec


@pytest.fixture
def monotonic_event_study_data():
    num_pre_periods = 3
    num_post_periods = 3
    betahat = np.array([-0.3, -0.2, -0.1, 0.3, 0.4, 0.5])
    sigma = np.eye(6) * 0.01
    true_beta = betahat.copy()
    l_vec = np.ones(num_post_periods) / num_post_periods
    return num_pre_periods, num_post_periods, betahat, sigma, true_beta, l_vec


def test_create_constraint_matrix_basic(constraint_matrix_params):
    A = _create_relative_magnitudes_monotonicity_constraint_matrix(**constraint_matrix_params)
    assert isinstance(A, np.ndarray)
    assert A.shape[1] == constraint_matrix_params["num_pre_periods"] + constraint_matrix_params["num_post_periods"]
    assert A.shape[0] > 0


@pytest.mark.parametrize("monotonicity_direction", ["increasing", "decreasing"])
def test_create_constraint_matrix_monotonicity_directions(constraint_matrix_params, monotonicity_direction):
    params = constraint_matrix_params.copy()
    params["monotonicity_direction"] = monotonicity_direction

    A = _create_relative_magnitudes_monotonicity_constraint_matrix(**params)
    assert A.shape[1] == params["num_pre_periods"] + params["num_post_periods"]
    assert A.shape[0] > 0


@pytest.mark.parametrize("num_pre_periods,num_post_periods", [(3, 2), (5, 4)])
@pytest.mark.parametrize("m_bar", [0.5, 2.0])
def test_create_constraint_matrix_different_configurations(num_pre_periods, num_post_periods, m_bar):
    for s in range(-(num_pre_periods - 1), 1):
        A = _create_relative_magnitudes_monotonicity_constraint_matrix(
            num_pre_periods=num_pre_periods,
            num_post_periods=num_post_periods,
            m_bar=m_bar,
            s=s,
            max_positive=True,
            monotonicity_direction="increasing",
        )
        assert A.shape[1] == num_pre_periods + num_post_periods
        assert A.shape[0] >= 0


def test_create_constraint_vector(constraint_matrix_params):
    A = _create_relative_magnitudes_monotonicity_constraint_matrix(**constraint_matrix_params)
    d = _create_relative_magnitudes_monotonicity_constraint_vector(A)
    assert isinstance(d, np.ndarray)
    assert np.all(d == 0)
    assert len(d) == A.shape[0]


@pytest.mark.parametrize("monotonicity_direction", ["increasing", "decreasing"])
def test_compute_identified_set_rmm_empty_when_pre_periods_are_not_monotone(
    simple_event_study_data, monotonicity_direction
):
    num_pre_periods, num_post_periods, _, _, true_beta, l_vec = simple_event_study_data

    result = compute_identified_set_rmm(
        m_bar=1,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        monotonicity_direction=monotonicity_direction,
    )

    assert isinstance(result, DeltaRMMResult)
    assert np.isnan(result.id_lb)
    assert np.isnan(result.id_ub)


def test_compute_identified_set_rmm_monotonic_data(monotonic_event_study_data):
    num_pre_periods, num_post_periods, _, _, true_beta, l_vec = monotonic_event_study_data

    result = compute_identified_set_rmm(
        m_bar=0.5,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        monotonicity_direction="increasing",
    )

    assert (result.id_lb, result.id_ub) == pytest.approx((0.3, 0.4), abs=1e-8)


def test_monotonicity_direction_effects(monotonic_event_study_data):
    num_pre_periods, num_post_periods, _, _, true_beta, l_vec = monotonic_event_study_data
    kwargs = {
        "m_bar": 1,
        "true_beta": true_beta,
        "l_vec": l_vec,
        "num_pre_periods": num_pre_periods,
        "num_post_periods": num_post_periods,
    }

    increasing = compute_identified_set_rmm(**kwargs, monotonicity_direction="increasing")
    decreasing = compute_identified_set_rmm(**kwargs, monotonicity_direction="decreasing")

    assert (increasing.id_lb, increasing.id_ub) == pytest.approx((0.2, 0.4), abs=1e-8)
    assert np.isnan(decreasing.id_lb)
    assert np.isnan(decreasing.id_ub)


def test_compute_identified_set_rmm_zero_m_bar():
    true_beta = np.array([-0.3, -0.2, -0.1, 0.3, 0.5])
    l_vec = np.array([1, 0])

    result = compute_identified_set_rmm(
        m_bar=0,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=3,
        num_post_periods=2,
        monotonicity_direction="increasing",
    )

    assert (result.id_lb, result.id_ub) == pytest.approx((0.3, 0.3), abs=1e-8)


def test_no_pre_periods_raises():
    betahat = np.array([0.1, 0.2, 0.3])

    with pytest.raises(ValueError, match="needs at least one pre-treatment period"):
        compute_identified_set_rmm(
            m_bar=1, true_beta=betahat, l_vec=np.ones(3) / 3, num_pre_periods=0, num_post_periods=3
        )
    with pytest.raises(ValueError, match="needs at least one pre-treatment period"):
        compute_conditional_cs_rmm(
            betahat=betahat, sigma=np.eye(3) * 0.01, num_pre_periods=0, num_post_periods=3, m_bar=1
        )


@pytest.mark.parametrize("m_bar", [0.0, 0.5, 1.0, 2.0])
def test_identified_set_rmm_one_pre_period_matches_closed_form(one_pre_period_event_study, m_bar):
    betahat = one_pre_period_event_study["betahat"]
    l_vec = np.array([1.0, 0.0])

    increasing = compute_identified_set_rmm(m_bar, betahat, l_vec, 1, 2, monotonicity_direction="increasing")
    decreasing = compute_identified_set_rmm(m_bar, betahat, l_vec, 1, 2, monotonicity_direction="decreasing")

    assert (increasing.id_lb, increasing.id_ub) == pytest.approx((0.10 - 0.03 * m_bar, 0.10), abs=1e-8)
    assert np.isnan(decreasing.id_lb)
    assert np.isnan(decreasing.id_ub)


def test_compute_conditional_cs_rmm_basic(simple_event_study_data, fast_config):
    num_pre_periods, num_post_periods, betahat, sigma, _, l_vec = simple_event_study_data

    result = compute_conditional_cs_rmm(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        l_vec=l_vec,
        m_bar=0.5,
        alpha=0.05,
        monotonicity_direction="increasing",
        grid_points=fast_config["grid_points_medium"],
    )

    assert isinstance(result, dict)
    assert "grid" in result
    assert "accept" in result
    assert len(result["grid"]) == len(result["accept"])
    assert len(result["grid"]) == fast_config["grid_points_medium"]

    assert np.any(result["accept"] > 0)


@pytest.mark.parametrize("hybrid_flag", ["LF", "ARP"])
def test_compute_conditional_cs_rmm_hybrid_flags(simple_event_study_data, hybrid_flag, fast_config):
    if fast_config["skip_expensive_params"]:
        pytest.skip("Skip RMM hybrid-flag stress case in fast mode")
    num_pre_periods, num_post_periods, betahat, sigma, _, l_vec = simple_event_study_data

    result = compute_conditional_cs_rmm(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        l_vec=l_vec,
        m_bar=0.5,
        alpha=0.05,
        hybrid_flag=hybrid_flag,
        monotonicity_direction="increasing",
        grid_points=fast_config["grid_points_small"],
    )

    assert "grid" in result
    assert "accept" in result
    assert np.any(result["accept"] > 0)


def test_compute_conditional_cs_rmm_custom_grid_bounds(simple_event_study_data, fast_config):
    num_pre_periods, num_post_periods, betahat, sigma, _, l_vec = simple_event_study_data

    result = compute_conditional_cs_rmm(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        l_vec=l_vec,
        m_bar=0.5,
        alpha=0.05,
        monotonicity_direction="increasing",
        grid_lb=-2,
        grid_ub=2,
        grid_points=fast_config["grid_points_small"],
    )

    assert result["grid"][0] == pytest.approx(-2)
    assert result["grid"][-1] == pytest.approx(2)


@pytest.mark.parametrize("grid_points", [16, 24])
def test_grid_resolution(simple_event_study_data, grid_points):
    num_pre_periods, num_post_periods, betahat, sigma, _, l_vec = simple_event_study_data

    result = compute_conditional_cs_rmm(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        l_vec=l_vec,
        m_bar=0.5,
        alpha=0.05,
        monotonicity_direction="increasing",
        grid_points=grid_points,
    )

    assert len(result["grid"]) == grid_points
    assert len(result["accept"]) == grid_points


def test_monotonicity_constraint_comparison():
    true_beta = np.array([0.2, 0.15, 0.1, 0.3, 0.4, 0.5])
    l_vec = np.array([1, 0, 0])

    result_increasing = compute_identified_set_rmm(
        m_bar=0.5,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=3,
        num_post_periods=3,
        monotonicity_direction="increasing",
    )

    result_decreasing = compute_identified_set_rmm(
        m_bar=0.5,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=3,
        num_post_periods=3,
        monotonicity_direction="decreasing",
    )

    assert np.isnan(result_increasing.id_lb)
    assert np.isnan(result_increasing.id_ub)
    assert (result_decreasing.id_lb, result_decreasing.id_ub) == pytest.approx((0.3, 0.35), abs=1e-8)


def test_different_s_values():
    num_pre_periods = 3
    num_post_periods = 2

    results = []
    for s in range(-(num_pre_periods - 1), 1):
        A = _create_relative_magnitudes_monotonicity_constraint_matrix(
            num_pre_periods=num_pre_periods,
            num_post_periods=num_post_periods,
            m_bar=1,
            s=s,
            max_positive=True,
            monotonicity_direction="increasing",
        )
        results.append(A.shape[0])

    assert all(r >= 0 for r in results)


def test_confidence_interval_coverage_ordering(fast_config):
    betahat = np.array([0, 0, 0, 0.5, 0.5])
    sigma = np.eye(5) * 0.01
    l_vec = np.array([1, 0])

    lengths = []
    alphas = [0.01, 0.05, 0.10]

    for alpha in alphas:
        length = compute_conditional_cs_rmm(
            betahat=betahat,
            sigma=sigma,
            num_pre_periods=3,
            num_post_periods=2,
            l_vec=l_vec,
            m_bar=0.5,
            alpha=alpha,
            monotonicity_direction="increasing",
            grid_points=fast_config["grid_points_medium"],
            return_length=True,
        )
        lengths.append(length)

    for i in range(1, len(lengths)):
        assert lengths[i] <= lengths[i - 1]


def test_monotonicity_with_violations():
    true_beta = np.array([0.1, 0.2, 0.15, 0.4, 0.3])
    l_vec = np.array([1, 0])

    result = compute_identified_set_rmm(
        m_bar=1,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=3,
        num_post_periods=2,
        monotonicity_direction="increasing",
    )

    assert np.isnan(result.id_lb)
    assert np.isnan(result.id_ub)


@pytest.mark.parametrize("m_bar,expected", [(0, (0.4, 0.4)), (1, (0.2, 0.4))])
def test_different_m_bar_values(monotonic_event_study_data, m_bar, expected):
    num_pre_periods, num_post_periods, _, _, true_beta, l_vec = monotonic_event_study_data

    result = compute_identified_set_rmm(
        m_bar=m_bar,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        monotonicity_direction="increasing",
    )

    assert (result.id_lb, result.id_ub) == pytest.approx(expected, abs=1e-8)


def test_post_period_moments_only_flag(fast_config):
    if fast_config["skip_expensive_params"]:
        pytest.skip("Skip post-period moments stress case in fast mode")
    betahat = np.array([0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
    sigma = np.eye(6) * 0.01
    l_vec = np.array([1, 0, 0])

    result_all = compute_conditional_cs_rmm(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=3,
        num_post_periods=3,
        l_vec=l_vec,
        m_bar=0.5,
        alpha=0.05,
        monotonicity_direction="increasing",
        post_period_moments_only=False,
        grid_points=fast_config["grid_points_small"],
    )

    result_post_only = compute_conditional_cs_rmm(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=3,
        num_post_periods=3,
        l_vec=l_vec,
        m_bar=0.5,
        alpha=0.05,
        monotonicity_direction="increasing",
        post_period_moments_only=True,
        grid_points=fast_config["grid_points_small"],
    )

    assert np.any(result_all["accept"] > 0)
    assert np.any(result_post_only["accept"] > 0)


@pytest.mark.parametrize(
    "case,m_bar,expected",
    [
        ("A", 1.0, [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]),
        ("B", 0.5, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
    ],
)
def test_conditional_cs_rmm_conditional_accepts(rm_cases, case, m_bar, expected):
    result = compute_conditional_cs_rmm(
        betahat=rm_cases[case],
        sigma=rm_cases["sigma"],
        num_pre_periods=3,
        num_post_periods=2,
        l_vec=np.array([1.0, 0.0]),
        m_bar=m_bar,
        hybrid_flag="ARP",
        monotonicity_direction="increasing",
        grid_points=11,
        grid_lb=-0.1,
        grid_ub=0.2,
    )

    assert result["accept"].tolist() == expected


def test_conditional_cs_rmm_single_post_period(one_post_event_study):
    kwargs = {
        "betahat": one_post_event_study["betahat"],
        "sigma": one_post_event_study["sigma"],
        "num_pre_periods": 4,
        "num_post_periods": 1,
        "l_vec": np.array([1.0]),
        "m_bar": 1.0,
        "monotonicity_direction": "increasing",
        "grid_points": 11,
        "grid_lb": 0.0,
        "grid_ub": 0.1,
    }

    arp = compute_conditional_cs_rmm(**kwargs, hybrid_flag="ARP")
    first = compute_conditional_cs_rmm(**kwargs, hybrid_flag="LF")
    second = compute_conditional_cs_rmm(**kwargs, hybrid_flag="LF")

    assert arp["accept"].astype(int).tolist() == [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    assert np.array_equal(first["accept"], second["accept"])


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("monotonicity_direction", ["increasing", "decreasing"])
def test_conditional_cs_rmm_at_zero_m_bar_with_duplicated_rows(one_pre_period_event_study, monotonicity_direction):
    result = compute_conditional_cs_rmm(
        betahat=one_pre_period_event_study["betahat"],
        sigma=one_pre_period_event_study["sigma"],
        num_pre_periods=1,
        num_post_periods=2,
        l_vec=np.array([0.5, 0.5]),
        m_bar=0.0,
        monotonicity_direction=monotonicity_direction,
        hybrid_flag="ARP",
        grid_points=76,
        grid_lb=-0.25,
        grid_ub=0.5,
    )

    assert np.asarray(result["accept"]).astype(int).tolist() == [0] * 32 + [1] * 9 + [0] * 35


def test_conditional_cs_rmm_default_grid_reaches_past_a_noisy_long_horizon_set(noisy_long_horizon_event_study):
    l_vec = np.zeros(8)
    l_vec[-1] = 1.0

    result = compute_conditional_cs_rmm(
        betahat=noisy_long_horizon_event_study["betahat"],
        sigma=noisy_long_horizon_event_study["sigma"],
        num_pre_periods=3,
        num_post_periods=8,
        l_vec=l_vec,
        m_bar=2.0,
        monotonicity_direction="decreasing",
        hybrid_flag="LF",
        grid_points=30,
    )

    accept = np.asarray(result["accept"]).astype(bool)
    grid = np.asarray(result["grid"])
    assert not accept[0]
    assert not accept[-1]
    assert (grid[accept].min(), grid[accept].max()) == pytest.approx((-0.2291392, 1.4055743), abs=1e-6)
