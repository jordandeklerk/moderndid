# pylint: disable=redefined-outer-name
"""Tests for DeltaRM functions."""

import numpy as np
import pytest

from doublediff.didhonest import (
    DeltaRMResult,
    compute_conditional_cs_rm,
    compute_identified_set_rm,
)
from doublediff.didhonest.delta.rm.rm import (
    _create_relative_magnitudes_constraint_matrix,
    _create_relative_magnitudes_constraint_vector,
)


@pytest.fixture
def constraint_matrix_params():
    return {
        "num_pre_periods": 3,
        "num_post_periods": 2,
        "m_bar": 1,
        "s": 0,
        "max_positive": True,
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


def test_create_constraint_matrix_basic(constraint_matrix_params):
    A = _create_relative_magnitudes_constraint_matrix(**constraint_matrix_params)
    assert isinstance(A, np.ndarray)
    assert A.shape[1] == constraint_matrix_params["num_pre_periods"] + constraint_matrix_params["num_post_periods"]
    assert A.shape[0] > 0


@pytest.mark.parametrize("num_pre_periods,num_post_periods", [(3, 2), (4, 3)])
@pytest.mark.parametrize("m_bar", [0.5, 1])
def test_create_constraint_matrix_different_configurations(num_pre_periods, num_post_periods, m_bar):
    for s in range(-(num_pre_periods - 1), 1):
        A = _create_relative_magnitudes_constraint_matrix(
            num_pre_periods=num_pre_periods,
            num_post_periods=num_post_periods,
            m_bar=m_bar,
            s=s,
            max_positive=True,
        )
        assert A.shape[1] == num_pre_periods + num_post_periods
        assert A.shape[0] >= 0


@pytest.mark.parametrize("s", [-2, -1, 0])
def test_create_constraint_matrix_sign_change(constraint_matrix_params, s):
    params = constraint_matrix_params.copy()
    params["s"] = s

    params["max_positive"] = True
    A_pos = _create_relative_magnitudes_constraint_matrix(**params)

    params["max_positive"] = False
    A_neg = _create_relative_magnitudes_constraint_matrix(**params)

    if A_pos.shape[0] > 0 and A_neg.shape[0] > 0:
        assert not np.array_equal(A_pos, A_neg)


@pytest.mark.parametrize("invalid_s", [-5, 2, -10])
def test_create_constraint_matrix_invalid_s(constraint_matrix_params, invalid_s):
    params = constraint_matrix_params.copy()
    params["s"] = invalid_s

    with pytest.raises(ValueError, match="s must be between"):
        _create_relative_magnitudes_constraint_matrix(**params)


def test_create_constraint_vector(constraint_matrix_params):
    A = _create_relative_magnitudes_constraint_matrix(**constraint_matrix_params)
    d = _create_relative_magnitudes_constraint_vector(A)
    assert isinstance(d, np.ndarray)
    assert np.all(d == 0)
    assert len(d) == A.shape[0]


def test_compute_identified_set_rm_basic(simple_event_study_data):
    num_pre_periods, num_post_periods, _, _, true_beta, l_vec = simple_event_study_data

    result = compute_identified_set_rm(
        m_bar=1,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
    )

    assert isinstance(result, DeltaRMResult)
    assert result.id_lb <= result.id_ub

    observed_val = l_vec @ true_beta[num_pre_periods:]
    assert result.id_lb <= observed_val <= result.id_ub

    max_pre_effect = np.max(np.abs(true_beta[:num_pre_periods]))
    assert result.id_ub - result.id_lb <= 2 * max_pre_effect * 1


def test_compute_identified_set_rm_zero_m_bar():
    true_beta = np.array([-0.1, 0.05, -0.02, 0.3, 0.5])
    l_vec = np.array([1, 0])

    result = compute_identified_set_rm(
        m_bar=0,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=3,
        num_post_periods=2,
    )

    observed_effect = l_vec @ true_beta[3:]
    assert result.id_lb == pytest.approx(observed_effect, abs=1e-10)
    assert result.id_ub == pytest.approx(observed_effect, abs=1e-10)


def test_compute_identified_set_rm_known_bounds():
    true_beta = np.array([0.1, 0.1, 0.1, 0.5, 0.5])
    l_vec = np.array([0.5, 0.5])

    result = compute_identified_set_rm(
        m_bar=1,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=3,
        num_post_periods=2,
    )

    observed_effect = 0.5

    assert result.id_lb < observed_effect
    assert result.id_ub > observed_effect
    assert result.id_lb >= observed_effect - 0.5
    assert result.id_ub <= observed_effect + 0.5

    width = result.id_ub - result.id_lb
    assert 0 < width <= 1.0


def test_analytical_case_single_post_period():
    true_beta = np.array([0.2, -0.1, 0.1, 0.5])
    l_vec = np.array([1.0])

    result = compute_identified_set_rm(
        m_bar=1,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=3,
        num_post_periods=1,
    )

    observed_effect = 0.5
    pre_diffs = [abs(true_beta[1] - true_beta[0]), abs(true_beta[2] - true_beta[1]), abs(true_beta[2] - true_beta[0])]
    max_pre_diff = max(pre_diffs)

    assert result.id_lb <= observed_effect
    assert result.id_ub >= observed_effect
    assert result.id_ub - result.id_lb <= 2 * max_pre_diff


def test_compute_identified_set_rm_monotonicity():
    true_beta = np.array([0.1, 0.1, 0.1, 0.5, 0.5])
    l_vec = np.array([1, 0])

    m_bars = [0, 0.1, 0.5, 1, 2, 5]
    results = []

    for m_bar in m_bars:
        result = compute_identified_set_rm(
            m_bar=m_bar,
            true_beta=true_beta,
            l_vec=l_vec,
            num_pre_periods=3,
            num_post_periods=2,
        )
        results.append(result)

    for i in range(1, len(results)):
        assert results[i].id_ub - results[i].id_lb >= results[i - 1].id_ub - results[i - 1].id_lb
        assert results[i].id_lb <= results[i - 1].id_lb
        assert results[i].id_ub >= results[i - 1].id_ub


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
        compute_identified_set_rm(
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
            compute_conditional_cs_rm(
                betahat=betahat,
                sigma=sigma,
                num_pre_periods=num_pre,
                num_post_periods=num_post,
                m_bar=1,
            )


def test_compute_conditional_cs_rm_basic(simple_event_study_data, fast_config):
    num_pre_periods, num_post_periods, betahat, sigma, _, l_vec = simple_event_study_data

    result = compute_conditional_cs_rm(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        l_vec=l_vec,
        m_bar=0.5,
        alpha=0.05,
        grid_lb=-2,
        grid_ub=2,
        grid_points=fast_config["grid_points_small"],
    )

    assert isinstance(result, dict)
    assert "grid" in result
    assert "accept" in result
    assert len(result["grid"]) == len(result["accept"])
    assert len(result["grid"]) == fast_config["grid_points_small"]

    assert np.any(result["accept"] > 0)


@pytest.mark.parametrize("hybrid_flag", ["LF", "ARP"])
def test_compute_conditional_cs_rm_hybrid_flags(simple_event_study_data, hybrid_flag, fast_config):
    num_pre_periods, num_post_periods, betahat, sigma, _, l_vec = simple_event_study_data

    result = compute_conditional_cs_rm(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        l_vec=l_vec,
        m_bar=0.5,
        alpha=0.05,
        hybrid_flag=hybrid_flag,
        grid_lb=-2,
        grid_ub=2,
        grid_points=fast_config["grid_points_small"],
    )

    assert "grid" in result
    assert "accept" in result
    assert np.any(result["accept"] > 0)


def test_compute_conditional_cs_rm_custom_grid_bounds(simple_event_study_data, fast_config):
    num_pre_periods, num_post_periods, betahat, sigma, _, l_vec = simple_event_study_data

    result = compute_conditional_cs_rm(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        l_vec=l_vec,
        m_bar=0.5,
        alpha=0.05,
        grid_lb=-2,
        grid_ub=2,
        grid_points=fast_config["grid_points_small"],
    )

    assert result["grid"][0] == pytest.approx(-2)
    assert result["grid"][-1] == pytest.approx(2)


@pytest.mark.parametrize("grid_points", [15, 20, 30])
def test_grid_resolution(simple_event_study_data, grid_points):
    num_pre_periods, num_post_periods, betahat, sigma, _, l_vec = simple_event_study_data

    result = compute_conditional_cs_rm(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        l_vec=l_vec,
        m_bar=0.5,
        alpha=0.05,
        grid_lb=-2,
        grid_ub=2,
        grid_points=grid_points,
    )

    assert len(result["grid"]) == grid_points
    assert len(result["accept"]) == grid_points


def test_integration_identified_set_and_cs(fast_config):
    true_beta = np.array([0.05, 0.05, 0.05, 0.4, 0.6])
    betahat = true_beta + np.random.normal(0, 0.02, 5)
    sigma = np.eye(5) * 0.01
    l_vec = np.array([1, 0])

    id_set = compute_identified_set_rm(
        m_bar=1,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=3,
        num_post_periods=2,
    )

    cs = compute_conditional_cs_rm(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=3,
        num_post_periods=2,
        l_vec=l_vec,
        m_bar=1,
        alpha=0.05,
        hybrid_flag="ARP",
        grid_points=fast_config["grid_points_small"],
        grid_lb=id_set.id_lb - 0.2,
        grid_ub=id_set.id_ub + 0.2,
    )

    grid = cs["grid"]
    accept = cs["accept"]

    accepted_indices = np.where(accept)[0]
    if len(accepted_indices) > 0:
        cs_lb = grid[accepted_indices[0]]
        cs_ub = grid[accepted_indices[-1]]

        assert cs_lb <= id_set.id_ub
        assert cs_ub >= id_set.id_lb


@pytest.mark.parametrize("m_bar", [0, 0.5, 1, 2])
def test_different_m_bar_values(simple_event_study_data, m_bar):
    num_pre_periods, num_post_periods, _, _, true_beta, l_vec = simple_event_study_data

    result = compute_identified_set_rm(
        m_bar=m_bar,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
    )

    assert result.id_lb <= result.id_ub


def test_confidence_interval_coverage_ordering(fast_config):
    betahat = np.array([0, 0, 0, 0.5, 0.5])
    sigma = np.eye(5) * 0.01
    l_vec = np.array([1, 0])

    lengths = []
    alphas = [0.01, 0.05, 0.10]

    for alpha in alphas:
        length = compute_conditional_cs_rm(
            betahat=betahat,
            sigma=sigma,
            num_pre_periods=3,
            num_post_periods=2,
            l_vec=l_vec,
            m_bar=0.5,
            alpha=alpha,
            grid_lb=-1,
            grid_ub=2,
            grid_points=fast_config["grid_points_small"],
            return_length=True,
        )
        lengths.append(length)

    for i in range(1, len(lengths)):
        assert lengths[i] <= lengths[i - 1]


@pytest.mark.slow
def test_larger_event_study_basic(larger_event_study_data):
    num_pre_periods, num_post_periods, _, _, true_beta, l_vec = larger_event_study_data

    id_set = compute_identified_set_rm(
        m_bar=1,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
    )

    assert id_set.id_lb <= id_set.id_ub
    observed_val = l_vec @ true_beta[num_pre_periods:]
    assert id_set.id_lb <= observed_val <= id_set.id_ub


def test_parallel_trends_scenario():
    true_beta = np.array([0, 0, 0, 0.5, 0.7])
    l_vec = np.array([1, 0])

    result = compute_identified_set_rm(
        m_bar=0.1,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=3,
        num_post_periods=2,
    )

    observed_effect = 0.5
    assert result.id_lb <= observed_effect
    assert result.id_ub >= observed_effect
    assert result.id_ub - result.id_lb < 0.1


def test_violated_parallel_trends_scenario():
    true_beta = np.array([0.1, 0.2, 0.3, 0.5, 0.7])
    l_vec = np.array([1, 0])

    result_small_m = compute_identified_set_rm(
        m_bar=0.5,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=3,
        num_post_periods=2,
    )

    result_large_m = compute_identified_set_rm(
        m_bar=2,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=3,
        num_post_periods=2,
    )

    assert result_small_m.id_lb < result_small_m.id_ub
    assert result_large_m.id_lb < result_large_m.id_ub
    assert result_large_m.id_ub - result_large_m.id_lb >= result_small_m.id_ub - result_small_m.id_lb


def test_confidence_set_contains_true_value(fast_config):
    np.random.seed(123)
    true_effect = 0.5
    betahat = np.array([0.02, -0.01, 0.03, true_effect + 0.05, 0.6])
    sigma = np.diag([0.01, 0.01, 0.01, 0.02, 0.02])
    l_vec = np.array([1, 0])

    result = compute_conditional_cs_rm(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=3,
        num_post_periods=2,
        l_vec=l_vec,
        m_bar=0.5,
        alpha=0.05,
        grid_points=fast_config["grid_points_medium"],
        grid_lb=true_effect - 1,
        grid_ub=true_effect + 1,
    )

    grid = result["grid"]
    accept = result["accept"]

    true_idx = np.argmin(np.abs(grid - true_effect))
    assert accept[true_idx] == 1

    accepted_indices = np.where(accept)[0]
    if len(accepted_indices) > 0:
        ci_lb = grid[accepted_indices[0]]
        ci_ub = grid[accepted_indices[-1]]
        assert ci_lb <= true_effect <= ci_ub
