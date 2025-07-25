# pylint: disable=redefined-outer-name
"""Test relative magnitudes with bias restriction."""

import numpy as np
import pytest

from pydid.honestdid import (
    DeltaRMBResult,
    compute_conditional_cs_rmb,
    compute_identified_set_rmb,
)
from pydid.honestdid.delta.rm.rmb import (
    _compute_identified_set_rmb_fixed_s,
    _create_relative_magnitudes_bias_constraint_matrix,
    _create_relative_magnitudes_bias_constraint_vector,
)


@pytest.fixture
def basic_params():
    return {
        "num_pre_periods": 3,
        "num_post_periods": 2,
        "m_bar": 1,
        "s": 0,
        "max_positive": True,
        "bias_direction": "positive",
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
def larger_event_study_data():
    np.random.seed(42)
    num_pre_periods = 5
    num_post_periods = 4
    betahat = np.concatenate([np.random.normal(0, 0.1, num_pre_periods), np.random.normal(0.5, 0.2, num_post_periods)])
    n = num_pre_periods + num_post_periods
    sigma = np.eye(n) * 0.02
    for i in range(n):
        for j in range(n):
            if i != j:
                sigma[i, j] = 0.001 * np.exp(-abs(i - j))
    sigma = (sigma + sigma.T) / 2
    sigma += np.eye(n) * 1e-6
    true_beta = betahat.copy()
    l_vec = np.ones(num_post_periods) / num_post_periods
    return num_pre_periods, num_post_periods, betahat, sigma, true_beta, l_vec


def test_create_constraint_matrix_basic(basic_params):
    A_rmb = _create_relative_magnitudes_bias_constraint_matrix(**basic_params)
    assert isinstance(A_rmb, np.ndarray)
    expected_cols = basic_params["num_pre_periods"] + basic_params["num_post_periods"]
    assert A_rmb.shape[1] == expected_cols
    assert A_rmb.shape[0] > 0


@pytest.mark.parametrize("bias_direction", ["positive", "negative"])
def test_create_constraint_matrix_bias_direction(basic_params, bias_direction):
    params = basic_params.copy()
    params["bias_direction"] = bias_direction
    A_rmb = _create_relative_magnitudes_bias_constraint_matrix(**params)
    assert A_rmb.shape[0] > params["num_pre_periods"] + params["num_post_periods"]


def test_create_constraint_vector(basic_params):
    A_rmb = _create_relative_magnitudes_bias_constraint_matrix(**basic_params)
    d_rmb = _create_relative_magnitudes_bias_constraint_vector(A_rmb)
    assert isinstance(d_rmb, np.ndarray)
    assert np.all(d_rmb == 0)
    assert len(d_rmb) == A_rmb.shape[0]


@pytest.mark.parametrize("s", [-2, -1, 0])
@pytest.mark.parametrize("max_positive", [True, False])
def test_compute_identified_set_rmb_fixed_s(simple_event_study_data, s, max_positive):
    num_pre_periods, num_post_periods, _, _, true_beta, l_vec = simple_event_study_data

    result = _compute_identified_set_rmb_fixed_s(
        s=s,
        m_bar=1,
        max_positive=max_positive,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        bias_direction="positive",
    )

    assert isinstance(result, DeltaRMBResult)
    assert result.id_lb <= result.id_ub

    observed_val = l_vec @ true_beta[num_pre_periods:]
    assert result.id_lb <= observed_val <= result.id_ub


def test_compute_identified_set_rmb_basic(simple_event_study_data):
    num_pre_periods, num_post_periods, _, _, true_beta, l_vec = simple_event_study_data

    result = compute_identified_set_rmb(
        m_bar=1,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        bias_direction="positive",
    )

    assert isinstance(result, DeltaRMBResult)
    assert result.id_lb <= result.id_ub

    observed_val = l_vec @ true_beta[num_pre_periods:]
    assert result.id_lb <= observed_val <= result.id_ub


@pytest.mark.parametrize("bias_direction", ["positive", "negative"])
def test_compute_identified_set_rmb_bias_directions(simple_event_study_data, bias_direction):
    num_pre_periods, num_post_periods, _, _, true_beta, l_vec = simple_event_study_data

    result = compute_identified_set_rmb(
        m_bar=1,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        bias_direction=bias_direction,
    )

    assert isinstance(result, DeltaRMBResult)
    assert result.id_lb <= result.id_ub


def test_identified_set_zero_m_bar_positive_bias():
    true_beta = np.array([-0.1, 0.05, -0.02, 0.3, 0.5])
    l_vec = np.array([1, 0])

    result = compute_identified_set_rmb(
        m_bar=0,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=3,
        num_post_periods=2,
        bias_direction="positive",
    )

    observed_effect = l_vec @ true_beta[3:]
    assert result.id_lb >= 0
    assert result.id_lb <= observed_effect
    assert result.id_ub == pytest.approx(observed_effect, abs=1e-10)


def test_identified_set_sign_restriction_binding():
    true_beta = np.array([0, 0, 0, -0.3, 0.5])
    l_vec = np.array([1, 0])

    result = compute_identified_set_rmb(
        m_bar=2,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=3,
        num_post_periods=2,
        bias_direction="positive",
    )

    assert result.id_lb >= 0


def test_compute_conditional_cs_rmb_basic(simple_event_study_data, fast_config):
    num_pre_periods, num_post_periods, betahat, sigma, _, l_vec = simple_event_study_data

    result = compute_conditional_cs_rmb(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        l_vec=l_vec,
        m_bar=0.5,
        alpha=0.05,
        bias_direction="positive",
        grid_points=fast_config["grid_points_medium"],
    )

    assert isinstance(result, dict)
    assert "grid" in result
    assert "accept" in result
    assert len(result["grid"]) == len(result["accept"])
    assert len(result["grid"]) == fast_config["grid_points_medium"]
    assert np.any(result["accept"] > 0)


@pytest.mark.parametrize("hybrid_flag", ["LF", "ARP"])
def test_compute_conditional_cs_rmb_hybrid_flags(simple_event_study_data, hybrid_flag, fast_config):
    num_pre_periods, num_post_periods, betahat, sigma, _, l_vec = simple_event_study_data

    result = compute_conditional_cs_rmb(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        l_vec=l_vec,
        m_bar=0.5,
        alpha=0.05,
        hybrid_flag=hybrid_flag,
        bias_direction="positive",
        grid_points=fast_config["grid_points_small"],
    )

    assert "grid" in result
    assert "accept" in result
    assert np.any(result["accept"] > 0)


@pytest.mark.parametrize("bias_direction", ["positive", "negative"])
def test_compute_conditional_cs_rmb_bias_directions(simple_event_study_data, bias_direction, fast_config):
    num_pre_periods, num_post_periods, betahat, sigma, _, l_vec = simple_event_study_data

    betahat_modified = betahat.copy()
    if bias_direction == "negative":
        betahat_modified[num_pre_periods:] *= -1

    result = compute_conditional_cs_rmb(
        betahat=betahat_modified,
        sigma=sigma,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        l_vec=l_vec,
        m_bar=0.5,
        alpha=0.05,
        bias_direction=bias_direction,
        grid_points=fast_config["grid_points_small"],
    )

    assert "grid" in result
    assert "accept" in result


def test_rmb_vs_rm_restriction():
    true_beta = np.array([0, 0, 0, 0.5, 0.5])
    l_vec = np.array([1, 0])

    from pydid.honestdid import compute_identified_set_rm

    result_rm = compute_identified_set_rm(
        m_bar=1,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=3,
        num_post_periods=2,
    )

    result_rmb = compute_identified_set_rmb(
        m_bar=1,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=3,
        num_post_periods=2,
        bias_direction="positive",
    )

    assert result_rmb.id_lb >= result_rm.id_lb
    assert result_rmb.id_ub <= result_rm.id_ub
    assert result_rmb.id_ub - result_rmb.id_lb <= result_rm.id_ub - result_rm.id_lb


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
        compute_identified_set_rmb(
            m_bar=1,
            true_beta=true_beta,
            l_vec=l_vec,
            num_pre_periods=num_pre,
            num_post_periods=num_post,
        )

    if num_pre + num_post >= 3:
        betahat = true_beta
        sigma = np.eye(num_pre + num_post) * 0.01

        with pytest.raises(ValueError, match=error_msg):
            compute_conditional_cs_rmb(
                betahat=betahat,
                sigma=sigma,
                num_pre_periods=num_pre,
                num_post_periods=num_post,
                m_bar=1,
            )


def test_identified_set_monotonicity_in_m_bar():
    true_beta = np.array([0.1, 0.1, 0.1, 0.5, 0.5])
    l_vec = np.array([1, 0])

    m_bars = [0, 0.1, 0.5, 1, 2, 5]
    results = []

    for m_bar in m_bars:
        result = compute_identified_set_rmb(
            m_bar=m_bar,
            true_beta=true_beta,
            l_vec=l_vec,
            num_pre_periods=3,
            num_post_periods=2,
            bias_direction="positive",
        )
        results.append(result)

    for i in range(1, len(results)):
        assert results[i].id_ub - results[i].id_lb >= results[i - 1].id_ub - results[i - 1].id_lb
        assert results[i].id_lb <= results[i - 1].id_lb + 1e-10
        assert results[i].id_ub >= results[i - 1].id_ub - 1e-10


def test_confidence_interval_coverage_ordering(fast_config):
    betahat = np.array([0, 0, 0, 0.5, 0.5])
    sigma = np.eye(5) * 0.01
    l_vec = np.array([1, 0])

    lengths = []
    alphas = [0.01, 0.05, 0.10]

    for alpha in alphas:
        length = compute_conditional_cs_rmb(
            betahat=betahat,
            sigma=sigma,
            num_pre_periods=3,
            num_post_periods=2,
            l_vec=l_vec,
            m_bar=0.5,
            alpha=alpha,
            bias_direction="positive",
            grid_points=fast_config["grid_points_medium"],
            return_length=True,
        )
        lengths.append(length)

    for i in range(1, len(lengths)):
        assert lengths[i] <= lengths[i - 1]


def test_single_post_period_case(fast_config):
    betahat = np.array([0.1, -0.05, 0.02, 0.5])
    sigma = np.eye(4) * 0.01
    true_beta = betahat.copy()
    l_vec = np.array([1.0])

    result_id = compute_identified_set_rmb(
        m_bar=1,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=3,
        num_post_periods=1,
        bias_direction="positive",
    )

    assert result_id.id_lb <= result_id.id_ub
    assert result_id.id_lb >= 0

    result_cs = compute_conditional_cs_rmb(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=3,
        num_post_periods=1,
        l_vec=l_vec,
        m_bar=1,
        alpha=0.05,
        bias_direction="positive",
        grid_points=fast_config["grid_points_small"],
    )

    assert np.any(result_cs["accept"] > 0)


def test_larger_event_study_rmb(larger_event_study_data, fast_config):
    num_pre_periods, num_post_periods, betahat, sigma, true_beta, l_vec = larger_event_study_data

    result_id = compute_identified_set_rmb(
        m_bar=1,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        bias_direction="positive",
    )

    assert result_id.id_lb <= result_id.id_ub

    result_cs = compute_conditional_cs_rmb(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        l_vec=l_vec,
        m_bar=1,
        alpha=0.05,
        bias_direction="positive",
        grid_points=fast_config["grid_points_small"],
    )

    assert np.any(result_cs["accept"] > 0)


def test_negative_post_effects_with_positive_bias(fast_config):
    true_beta = np.array([0, 0, 0, -0.5, -0.3])
    betahat = true_beta + np.random.normal(0, 0.01, 5)
    sigma = np.eye(5) * 0.01
    l_vec = np.array([1, 0])

    result = compute_identified_set_rmb(
        m_bar=0.1,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=3,
        num_post_periods=2,
        bias_direction="positive",
    )

    assert result.id_lb >= 0

    cs_result = compute_conditional_cs_rmb(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=3,
        num_post_periods=2,
        l_vec=l_vec,
        m_bar=0.1,
        alpha=0.05,
        bias_direction="positive",
        grid_points=fast_config["grid_points_medium"],
        grid_lb=-1,
        grid_ub=1,
    )

    grid = cs_result["grid"]
    accept = cs_result["accept"]
    accepted_indices = np.where(accept)[0]

    if len(accepted_indices) > 0:
        assert grid[accepted_indices[0]] >= -0.1


def test_custom_grid_bounds(simple_event_study_data, fast_config):
    num_pre_periods, num_post_periods, betahat, sigma, _, l_vec = simple_event_study_data

    result = compute_conditional_cs_rmb(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        l_vec=l_vec,
        m_bar=0.5,
        alpha=0.05,
        bias_direction="positive",
        grid_lb=-2,
        grid_ub=2,
        grid_points=fast_config["grid_points_small"],
    )

    assert result["grid"][0] == pytest.approx(-2)
    assert result["grid"][-1] == pytest.approx(2)


def test_post_period_moments_only_flag(fast_config):
    betahat = np.array([0.1, 0.05, 0.02, 0.5, 0.6])
    sigma = np.eye(5) * 0.01
    l_vec = np.array([1, 0])

    result_all_moments = compute_conditional_cs_rmb(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=3,
        num_post_periods=2,
        l_vec=l_vec,
        m_bar=0.5,
        alpha=0.05,
        bias_direction="positive",
        post_period_moments_only=False,
        grid_points=fast_config["grid_points_small"],
    )

    result_post_only = compute_conditional_cs_rmb(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=3,
        num_post_periods=2,
        l_vec=l_vec,
        m_bar=0.5,
        alpha=0.05,
        bias_direction="positive",
        post_period_moments_only=True,
        grid_points=fast_config["grid_points_small"],
    )

    assert np.any(result_all_moments["accept"] > 0)
    assert np.any(result_post_only["accept"] > 0)
