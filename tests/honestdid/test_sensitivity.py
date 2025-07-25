# pylint: disable=redefined-outer-name
"""Tests for sensitivity analysis functions."""

import numpy as np
import pandas as pd
import pytest

from didpy.honestdid.sensitivity import (
    OriginalCSResult,
    SensitivityResult,
    construct_original_cs,
    create_sensitivity_results,
    create_sensitivity_results_relative_magnitudes,
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

    results = create_sensitivity_results(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        m_vec=m_vec,
        method=method,
        monotonicity_direction=monotonicity,
        bias_direction=bias,
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) == len(m_vec)
    assert list(results.columns) == ["lb", "ub", "method", "delta", "m"]
    assert all(results["method"] == method)
    assert all(results["delta"] == expected_delta)
    assert all(results["lb"] <= results["ub"])

    row = results.iloc[0]
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

    results = create_sensitivity_results_relative_magnitudes(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        bound=bound,
        m_bar_vec=m_bar_vec,
        method=method,
        grid_points=100,
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) == len(m_bar_vec)
    assert list(results.columns) == ["lb", "ub", "method", "delta", "Mbar"]
    assert all(results["delta"] == expected_delta)
    assert list(results["Mbar"]) == list(m_bar_vec)
    assert all(results["method"] == method)


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
        create_sensitivity_results(**kwargs)


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
            "Not enough pre-periods",
        ),
    ],
)
def test_create_sensitivity_results_relative_magnitudes_errors(kwargs, error_match):
    with pytest.raises(ValueError, match=error_match):
        create_sensitivity_results_relative_magnitudes(**kwargs)


def test_default_m_vec_construction(basic_event_study_data):
    betahat, sigma, num_pre_periods, num_post_periods = basic_event_study_data

    results = create_sensitivity_results(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        method="FLCI",
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) == 10
    assert results["m"].min() == 0
    assert results["m"].max() > 0


def test_flci_warning_with_restriction(basic_event_study_data):
    betahat, sigma, num_pre_periods, num_post_periods = basic_event_study_data

    with pytest.warns(UserWarning, match="shape/sign restriction"):
        create_sensitivity_results(
            betahat=betahat,
            sigma=sigma,
            num_pre_periods=num_pre_periods,
            num_post_periods=num_post_periods,
            monotonicity_direction="increasing",
            method="FLCI",
            m_vec=np.array([0.1]),
        )
