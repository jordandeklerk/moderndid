"""Test relative magnitudes with bias restriction."""

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from moderndid.didhonest import (
    DeltaRMBResult,
    compute_conditional_cs_rmb,
    compute_identified_set_rm,
    compute_identified_set_rmb,
)
from moderndid.didhonest.delta.rm.rmb import (
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
    rng = np.random.default_rng(42)
    num_pre_periods = 4
    num_post_periods = 3
    betahat = np.concatenate([rng.normal(0, 0.1, num_pre_periods), rng.normal(0.5, 0.2, num_post_periods)])
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


def test_compute_identified_set_rmb_fixed_s_at_the_largest_difference(simple_event_study_data):
    num_pre_periods, num_post_periods, _, _, true_beta, l_vec = simple_event_study_data

    result = _compute_identified_set_rmb_fixed_s(
        s=-2,
        m_bar=1,
        max_positive=True,
        true_beta=true_beta,
        l_vec=l_vec,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        bias_direction="positive",
    )

    assert isinstance(result, DeltaRMBResult)
    assert (result.id_lb, result.id_ub) == pytest.approx((0.15, 0.3), abs=1e-8)


@pytest.mark.parametrize("s,max_positive", [(-2, False), (-1, True), (-1, False), (0, True), (0, False)])
def test_compute_identified_set_rmb_fixed_s_empty_away_from_the_largest_difference(
    simple_event_study_data, s, max_positive
):
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

    assert np.isnan(result.id_lb)
    assert np.isnan(result.id_ub)


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

    observed_effect = l_vec @ true_beta[3:]
    assert result.id_lb == pytest.approx(observed_effect, abs=1e-6)
    assert result.id_ub == pytest.approx(observed_effect, abs=1e-6)


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
    if fast_config["skip_expensive_params"]:
        pytest.skip("Skip RMB hybrid-flag stress case in fast mode")
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


def test_no_pre_periods_raises():
    betahat = np.array([0.1, 0.2, 0.3])

    with pytest.raises(ValueError, match="needs at least one pre-treatment period"):
        compute_identified_set_rmb(
            m_bar=1, true_beta=betahat, l_vec=np.ones(3) / 3, num_pre_periods=0, num_post_periods=3
        )
    with pytest.raises(ValueError, match="needs at least one pre-treatment period"):
        compute_conditional_cs_rmb(
            betahat=betahat, sigma=np.eye(3) * 0.01, num_pre_periods=0, num_post_periods=3, m_bar=1
        )


@pytest.mark.parametrize("m_bar", [0.0, 0.5, 1.0, 2.0])
def test_identified_set_rmb_one_pre_period_matches_closed_form(one_pre_period_event_study, m_bar):
    betahat = one_pre_period_event_study["betahat"]
    first, second = np.array([1.0, 0.0]), np.array([0.0, 1.0])

    positive_first = compute_identified_set_rmb(m_bar, betahat, first, 1, 2, bias_direction="positive")
    positive_second = compute_identified_set_rmb(m_bar, betahat, second, 1, 2, bias_direction="positive")
    negative_first = compute_identified_set_rmb(m_bar, betahat, first, 1, 2, bias_direction="negative")

    assert (positive_first.id_lb, positive_first.id_ub) == pytest.approx((0.10 - 0.03 * m_bar, 0.10), abs=1e-8)
    assert (positive_second.id_lb, positive_second.id_ub) == pytest.approx((0.12 - 0.06 * m_bar, 0.12), abs=1e-8)
    assert (negative_first.id_lb, negative_first.id_ub) == pytest.approx((0.10, 0.10 + 0.03 * m_bar), abs=1e-8)


def test_identified_set_monotonicity_in_m_bar():
    true_beta = np.array([0.1, 0.1, 0.1, 0.5, 0.5])
    l_vec = np.array([1, 0])

    m_bars = [0, 0.5, 2]
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

    observed_effect = l_vec @ true_beta[3:]
    assert result_id.id_lb <= observed_effect + 1e-6
    assert result_id.id_ub >= observed_effect - 1e-6

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
    if fast_config["skip_expensive_params"]:
        pytest.skip("Skip larger event study in fast mode")
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
    if fast_config["skip_expensive_params"]:
        pytest.skip("Skip negative post-effects stress case in fast mode")
    rng = np.random.default_rng(123)
    true_beta = np.array([0, 0, 0, -0.5, -0.3])
    betahat = true_beta + rng.normal(0, 0.01, 5)
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

    observed_effect = l_vec @ true_beta[3:]
    assert result.id_lb <= result.id_ub
    assert result.id_lb <= observed_effect + 1e-6
    assert result.id_ub >= observed_effect - 1e-6

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

    assert np.any(cs_result["accept"] > 0)


def test_custom_grid_bounds(simple_event_study_data, fast_config):
    if fast_config["skip_expensive_params"]:
        pytest.skip("Skip custom grid bound stress case in fast mode")
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
    if fast_config["skip_expensive_params"]:
        pytest.skip("Skip post-period moments stress case in fast mode")
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


def test_identified_set_rmb_bounded_when_mbar_at_least_one(rm_cases):
    l_vec = np.array([1.0, 0.0])

    positive = compute_identified_set_rmb(1.0, rm_cases["A"], l_vec, 3, 2, bias_direction="positive")
    negative = compute_identified_set_rmb(1.0, rm_cases["A"], l_vec, 3, 2, bias_direction="negative")

    assert (positive.id_lb, positive.id_ub) == pytest.approx((0.03, 0.05))
    assert (negative.id_lb, negative.id_ub) == pytest.approx((0.05, 0.07))


@pytest.mark.parametrize(
    "case,m_bar,expected",
    [
        ("A", 1.0, [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]),
        ("B", 0.5, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
    ],
)
def test_conditional_cs_rmb_conditional_accepts(rm_cases, case, m_bar, expected):
    result = compute_conditional_cs_rmb(
        betahat=rm_cases[case],
        sigma=rm_cases["sigma"],
        num_pre_periods=3,
        num_post_periods=2,
        l_vec=np.array([1.0, 0.0]),
        m_bar=m_bar,
        hybrid_flag="ARP",
        grid_points=11,
        grid_lb=-0.1,
        grid_ub=0.2,
    )

    assert result["accept"].tolist() == expected


def test_conditional_cs_rmb_single_post_period(one_post_event_study):
    kwargs = {
        "betahat": one_post_event_study["betahat"],
        "sigma": one_post_event_study["sigma"],
        "num_pre_periods": 4,
        "num_post_periods": 1,
        "l_vec": np.array([1.0]),
        "m_bar": 1.0,
        "grid_points": 11,
        "grid_lb": 0.0,
        "grid_ub": 0.1,
    }

    arp = compute_conditional_cs_rmb(**kwargs, hybrid_flag="ARP")
    lf_first = compute_conditional_cs_rmb(**kwargs, hybrid_flag="LF")
    lf_second = compute_conditional_cs_rmb(**kwargs, hybrid_flag="LF")

    assert arp["accept"].astype(int).tolist() == [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    assert np.array_equal(lf_first["accept"], lf_second["accept"])


def test_conditional_cs_rmb_at_zero_m_bar_with_duplicated_rows(one_pre_period_event_study):
    result = compute_conditional_cs_rmb(
        betahat=one_pre_period_event_study["betahat"],
        sigma=one_pre_period_event_study["sigma"],
        num_pre_periods=1,
        num_post_periods=2,
        l_vec=np.array([0.5, 0.5]),
        m_bar=0.0,
        bias_direction="negative",
        hybrid_flag="ARP",
        grid_points=76,
        grid_lb=-0.25,
        grid_ub=0.5,
    )

    assert np.asarray(result["accept"]).astype(int).tolist() == [0] * 32 + [1] * 9 + [0] * 35


def test_conditional_cs_rmb_default_grid_reaches_past_a_noisy_long_horizon_set(noisy_long_horizon_event_study):
    l_vec = np.zeros(8)
    l_vec[-1] = 1.0

    result = compute_conditional_cs_rmb(
        betahat=noisy_long_horizon_event_study["betahat"],
        sigma=noisy_long_horizon_event_study["sigma"],
        num_pre_periods=3,
        num_post_periods=8,
        l_vec=l_vec,
        m_bar=2.0,
        bias_direction="positive",
        hybrid_flag="LF",
        grid_points=30,
    )

    accept = np.asarray(result["accept"]).astype(bool)
    grid = np.asarray(result["grid"])
    assert not accept[0]
    assert not accept[-1]
    assert (grid[accept].min(), grid[accept].max()) == pytest.approx((-1.4551743, -0.2291392), abs=1e-6)
