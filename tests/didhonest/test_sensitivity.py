"""Tests for sensitivity analysis functions."""

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid.didhonest.sensitivity import (
    OriginalCSResult,
    SensitivityResult,
    construct_original_cs,
    create_sensitivity_results_rm,
    create_sensitivity_results_sm,
)


@pytest.fixture
def basic_event_study_data():
    np.random.seed(42)
    num_pre_periods = 3
    num_post_periods = 2

    betahat = np.array([0.1, -0.05, 0.02, 0.5, 0.3])

    sigma = np.array(
        [
            [0.01, 0.002, 0.001, 0.0005, 0.0003],
            [0.002, 0.015, 0.003, 0.0006, 0.0004],
            [0.001, 0.003, 0.02, 0.0007, 0.0005],
            [0.0005, 0.0006, 0.0007, 0.025, 0.008],
            [0.0003, 0.0004, 0.0005, 0.008, 0.03],
        ]
    )

    return betahat, sigma, num_pre_periods, num_post_periods


def test_construct_original_cs(basic_event_study_data):
    betahat, sigma, num_pre_periods, num_post_periods = basic_event_study_data

    result = construct_original_cs(betahat, sigma, num_pre_periods, num_post_periods)

    assert isinstance(result, OriginalCSResult)
    assert result.method == "Original"
    assert result.delta is None
    assert result.lb < result.ub
    assert np.abs(result.lb - 0.1897) < 0.001
    assert np.abs(result.ub - 0.8103) < 0.001


@pytest.mark.parametrize(
    "method,monotonicity,bias,expected_delta,m_vec",
    [
        ("FLCI", None, None, "DeltaSD", np.array([0, 0.1, 0.2])),
        ("C-LF", "increasing", None, "DeltaSDI", np.array([0, 0.1])),
        ("C-F", None, "positive", "DeltaSDPB", np.array([0, 0.1])),
    ],
)
def test_create_sensitivity_results(basic_event_study_data, method, monotonicity, bias, expected_delta, m_vec):
    betahat, sigma, num_pre_periods, num_post_periods = basic_event_study_data

    results = create_sensitivity_results_sm(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        m_vec=m_vec,
        method=method,
        monotonicity_direction=monotonicity,
        bias_direction=bias,
    )

    assert isinstance(results, pl.DataFrame)
    assert len(results) == len(m_vec)
    assert results.columns == ["lb", "ub", "method", "delta", "m"]
    assert (results["method"] == method).all()
    assert (results["delta"] == expected_delta).all()
    assert (results["lb"] <= results["ub"]).all()

    row = results.row(0, named=True)
    expected_result = SensitivityResult(
        lb=row["lb"], ub=row["ub"], method=row["method"], delta=row["delta"], m=row["m"]
    )
    assert isinstance(expected_result, SensitivityResult)
    assert expected_result.lb == row["lb"]


@pytest.mark.parametrize(
    "bound,m_bar_vec,expected_delta,method",
    [
        ("deviation from parallel trends", np.array([0, 0.5, 1.0]), "DeltaRM", "C-LF"),
        ("deviation from linear trend", np.array([0, 1.0]), "DeltaSDRM", "Conditional"),
    ],
)
def test_create_sensitivity_results_relative_magnitudes(
    basic_event_study_data, bound, m_bar_vec, expected_delta, method
):
    betahat, sigma, num_pre_periods, num_post_periods = basic_event_study_data

    results = create_sensitivity_results_rm(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        bound=bound,
        m_bar_vec=m_bar_vec,
        method=method,
        grid_points=40,
    )

    assert isinstance(results, pl.DataFrame)
    assert len(results) == len(m_bar_vec)
    assert results.columns == ["lb", "ub", "method", "delta", "Mbar"]
    assert (results["delta"] == expected_delta).all()
    assert results["Mbar"].to_list() == list(m_bar_vec)
    assert (results["method"] == method).all()


@pytest.mark.parametrize(
    "kwargs,error_match",
    [
        (
            {
                "betahat": np.array([0.1, -0.05, 0.5, 0.3]),
                "sigma": np.eye(4) * 0.01,
                "num_pre_periods": 2,
                "num_post_periods": 2,
                "monotonicity_direction": "increasing",
                "bias_direction": "positive",
            },
            "Cannot specify both",
        ),
    ],
)
def test_create_sensitivity_results_errors(kwargs, error_match):
    with pytest.raises(ValueError, match=error_match):
        create_sensitivity_results_sm(**kwargs)


@pytest.mark.parametrize(
    "kwargs,error_match",
    [
        (
            {
                "betahat": np.array([0.1, -0.05, 0.02, 0.5, 0.3]),
                "sigma": np.eye(5) * 0.01,
                "num_pre_periods": 3,
                "num_post_periods": 2,
                "bound": "invalid bound",
            },
            "bound must be",
        ),
        (
            {
                "betahat": np.array([0.1, 0.5, 0.3]),
                "sigma": np.eye(3) * 0.01,
                "num_pre_periods": 1,
                "num_post_periods": 2,
                "bound": "deviation from linear trend",
            },
            "needs at least two pre-treatment periods",
        ),
    ],
)
def test_create_sensitivity_results_relative_magnitudes_errors(kwargs, error_match):
    with pytest.raises(ValueError, match=error_match):
        create_sensitivity_results_rm(**kwargs)


def test_default_m_vec_construction(basic_event_study_data):
    betahat, sigma, num_pre_periods, num_post_periods = basic_event_study_data

    results = create_sensitivity_results_sm(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        method="FLCI",
    )

    assert isinstance(results, pl.DataFrame)
    assert len(results) == 10
    assert results["m"].min() == 0
    assert results["m"].max() > 0


def test_flci_warning_with_restriction(basic_event_study_data):
    betahat, sigma, num_pre_periods, num_post_periods = basic_event_study_data

    with pytest.warns(UserWarning, match="shape/sign restriction"):
        create_sensitivity_results_sm(
            betahat=betahat,
            sigma=sigma,
            num_pre_periods=num_pre_periods,
            num_post_periods=num_post_periods,
            monotonicity_direction="increasing",
            method="FLCI",
            m_vec=np.array([0.1]),
        )


def test_single_post_period_least_favorable_results_reproducible(one_post_event_study):
    kwargs = {
        "betahat": one_post_event_study["betahat"],
        "sigma": one_post_event_study["sigma"],
        "num_pre_periods": 4,
        "num_post_periods": 1,
        "grid_points": 21,
        "grid_lb": 0.0,
        "grid_ub": 0.1,
    }

    sm_first = create_sensitivity_results_sm(**kwargs, method="C-LF", m_vec=[0.01])
    sm_second = create_sensitivity_results_sm(**kwargs, method="C-LF", m_vec=[0.01])
    rm_first = create_sensitivity_results_rm(**kwargs, method="C-LF", m_bar_vec=[1.0])
    rm_second = create_sensitivity_results_rm(**kwargs, method="C-LF", m_bar_vec=[1.0])

    assert sm_first.equals(sm_second)
    assert rm_first.equals(rm_second)
    assert sm_first["lb"][0] > 0.0


def test_default_m_vec_with_two_pre_periods():
    betahat = np.array([0.01, -0.02, 0.3])
    sigma = np.diag([0.0004, 0.0003, 0.0005])

    result = create_sensitivity_results_sm(betahat, sigma, num_pre_periods=2, num_post_periods=1)

    assert len(result) == 10
    assert result["m"][0] == 0.0
    assert result["m"][-1] > 0.0


def test_linear_trend_bound_with_two_pre_periods(two_pre_period_event_study):
    results = create_sensitivity_results_rm(
        betahat=two_pre_period_event_study["betahat"],
        sigma=two_pre_period_event_study["sigma"],
        num_pre_periods=2,
        num_post_periods=2,
        bound="deviation from linear trend",
        method="Conditional",
        m_bar_vec=np.array([0.0, 0.5, 1.0]),
        grid_points=201,
        grid_lb=-1.0,
        grid_ub=1.5,
    )

    assert results["delta"].to_list() == ["DeltaSDRM"] * 3
    assert results["lb"].to_list() == pytest.approx([-0.0875, -0.2, -0.35], abs=1e-9)
    assert results["ub"].to_list() == pytest.approx([0.65, 0.70, 0.75], abs=1e-9)


def test_relative_magnitudes_with_one_pre_period(one_pre_period_event_study):
    results = create_sensitivity_results_rm(
        betahat=one_pre_period_event_study["betahat"],
        sigma=one_pre_period_event_study["sigma"],
        num_pre_periods=1,
        num_post_periods=2,
        method="Conditional",
        m_bar_vec=np.array([0.0, 0.5, 1.0]),
        grid_points=201,
        grid_lb=-0.3,
        grid_ub=0.5,
    )

    assert results["delta"].to_list() == ["DeltaRM"] * 3
    assert results["lb"].to_list() == pytest.approx([0.044, 0.028, 0.008], abs=1e-9)
    assert results["ub"].to_list() == pytest.approx([0.156, 0.168, 0.184], abs=1e-9)


def test_linear_trend_bound_default_grid_covers_a_linear_pre_trend(linear_trend_event_study):
    kwargs = {
        "betahat": linear_trend_event_study["betahat"],
        "sigma": linear_trend_event_study["sigma"],
        "num_pre_periods": 3,
        "num_post_periods": 4,
        "bound": "deviation from linear trend",
        "m_bar_vec": np.array([0.0, 0.5]),
        "l_vec": np.array([0.0, 0.0, 0.0, 1.0]),
    }

    default = create_sensitivity_results_rm(**kwargs, grid_points=101)
    fine = create_sensitivity_results_rm(**kwargs, grid_points=161, grid_lb=-0.2, grid_ub=0.2)

    assert default["lb"].is_finite().all()
    assert default["ub"].is_finite().all()
    assert default["lb"].to_list() == pytest.approx(fine["lb"].to_list(), abs=0.01)
    assert default["ub"].to_list() == pytest.approx(fine["ub"].to_list(), abs=0.01)
