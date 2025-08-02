# pylint: disable=redefined-outer-name
"""Tests for relative magnitudes with monotonicity restriction."""

import numpy as np
import pytest

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
    betahat = np.array([0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
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


@pytest.mark.parametrize("num_pre_periods,num_post_periods", [(3, 2), (4, 3), (5, 4)])
@pytest.mark.parametrize("m_bar", [0.5, 1, 2])
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


def test_compute_identified_set_rmm_basic(simple_event_study_data):
    num_pre_periods, num_post_periods, _, _, true_beta, l_vec = simple_event_study_data

    result = compute_identified_set_rmm(
        m_bar=1,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        monotonicity_direction="increasing",
    )

    assert isinstance(result, DeltaRMMResult)
    assert result.id_lb <= result.id_ub

    observed_val = l_vec @ true_beta[num_pre_periods:]
    assert result.id_lb <= observed_val <= result.id_ub


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

    assert result.id_lb <= result.id_ub
    observed_val = l_vec @ true_beta[num_pre_periods:]
    assert result.id_lb <= observed_val <= result.id_ub


@pytest.mark.parametrize("monotonicity_direction", ["increasing", "decreasing"])
def test_monotonicity_direction_effects(simple_event_study_data, monotonicity_direction):
    num_pre_periods, num_post_periods, _, _, true_beta, l_vec = simple_event_study_data

    result = compute_identified_set_rmm(
        m_bar=1,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        monotonicity_direction=monotonicity_direction,
    )

    assert result.id_lb <= result.id_ub


def test_compute_identified_set_rmm_zero_m_bar():
    true_beta = np.array([-0.1, 0.05, -0.02, 0.3, 0.5])
    l_vec = np.array([1, 0])

    result = compute_identified_set_rmm(
        m_bar=0,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=3,
        num_post_periods=2,
        monotonicity_direction="increasing",
    )

    observed_effect = l_vec @ true_beta[3:]
    assert result.id_lb <= observed_effect
    assert result.id_ub >= observed_effect


@pytest.mark.parametrize(
    "num_pre,num_post,error_msg",
    [
        (1, 2, "Need at least 2 pre-periods"),
        (0, 3, "Need at least 2 pre-periods"),
    ],
)
def test_insufficient_pre_periods_errors(num_pre, num_post, error_msg):
    true_beta = np.random.randn(num_pre + num_post)
    l_vec = np.ones(num_post) / num_post if num_post > 0 else np.array([])

    with pytest.raises(ValueError, match=error_msg):
        compute_identified_set_rmm(
            m_bar=1,
            true_beta=true_beta,
            l_vec=l_vec,
            num_pre_periods=num_pre,
            num_post_periods=num_post,
            monotonicity_direction="increasing",
        )

    if num_pre + num_post >= 3:
        betahat = true_beta
        sigma = np.eye(num_pre + num_post) * 0.01

        with pytest.raises(ValueError, match=error_msg):
            compute_conditional_cs_rmm(
                betahat=betahat,
                sigma=sigma,
                num_pre_periods=num_pre,
                num_post_periods=num_post,
                m_bar=1,
            )


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


@pytest.mark.parametrize("grid_points", [20, 30, 50])
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
    true_beta = np.array([0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
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

    assert result_increasing.id_lb <= result_increasing.id_ub
    assert result_decreasing.id_lb <= result_decreasing.id_ub


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

    assert result.id_lb <= result.id_ub


@pytest.mark.parametrize("m_bar", [0, 0.5, 1, 2])
def test_different_m_bar_values(simple_event_study_data, m_bar):
    num_pre_periods, num_post_periods, _, _, true_beta, l_vec = simple_event_study_data

    result = compute_identified_set_rmm(
        m_bar=m_bar,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        monotonicity_direction="increasing",
    )

    assert result.id_lb <= result.id_ub


def test_post_period_moments_only_flag(fast_config):
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
