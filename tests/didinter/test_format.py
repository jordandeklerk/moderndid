# pylint: disable=redefined-outer-name
"""Tests for result formatting functions."""

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid.didinter import (
    ATEResult,
    DIDInterResult,
    EffectsResult,
    PlacebosResult,
)
from moderndid.didinter.format import format_didinter_result


@pytest.fixture
def minimal_effects():
    return EffectsResult(
        horizons=np.array([1.0, 2.0]),
        estimates=np.array([0.5, 0.6]),
        std_errors=np.array([0.1, 0.12]),
        ci_lower=np.array([0.304, 0.365]),
        ci_upper=np.array([0.696, 0.835]),
        n_switchers=np.array([100.0, 90.0]),
        n_observations=np.array([500.0, 450.0]),
    )


@pytest.fixture
def minimal_result(minimal_effects):
    return DIDInterResult(
        effects=minimal_effects,
        n_units=200,
        n_switchers=100,
        n_never_switchers=100,
        ci_level=95.0,
        estimation_params={"effects": 2, "placebo": 0},
    )


@pytest.fixture
def full_result():
    effects = EffectsResult(
        horizons=np.array([1.0, 2.0, 3.0]),
        estimates=np.array([0.5, 0.6, 0.55]),
        std_errors=np.array([0.1, 0.12, 0.11]),
        ci_lower=np.array([0.304, 0.365, 0.334]),
        ci_upper=np.array([0.696, 0.835, 0.766]),
        n_switchers=np.array([100.0, 90.0, 85.0]),
        n_observations=np.array([500.0, 450.0, 420.0]),
    )

    placebos = PlacebosResult(
        horizons=np.array([-1.0, -2.0]),
        estimates=np.array([0.01, -0.02]),
        std_errors=np.array([0.05, 0.06]),
        ci_lower=np.array([-0.088, -0.138]),
        ci_upper=np.array([0.108, 0.098]),
        n_switchers=np.array([100.0, 95.0]),
        n_observations=np.array([500.0, 480.0]),
    )

    ate = ATEResult(
        estimate=0.55,
        std_error=0.08,
        ci_lower=0.393,
        ci_upper=0.707,
        n_observations=1370.0,
        n_switchers=275.0,
    )

    return DIDInterResult(
        effects=effects,
        placebos=placebos,
        ate=ate,
        n_units=200,
        n_switchers=100,
        n_never_switchers=100,
        ci_level=95.0,
        effects_equal_test={"chi2_stat": 1.5, "df": 2, "p_value": 0.47},
        placebo_joint_test={"chi2_stat": 0.8, "df": 2, "p_value": 0.67},
        estimation_params={
            "effects": 3,
            "placebo": 2,
            "normalized": False,
            "switchers": "",
            "controls": None,
            "cluster": None,
            "trends_lin": False,
            "trends_nonparam": None,
            "only_never_switchers": False,
            "same_switchers": False,
            "same_switchers_pl": False,
            "continuous": 0,
            "weightsname": None,
        },
    )


@pytest.mark.parametrize(
    "expected_text",
    [
        "Intertemporal Treatment Effects",
        "Treatment Effects by Horizon",
        "0.5",
        "0.6",
        "=" * 78,
        "Horizon",
        "Estimate",
        "Std. Error",
        "Conf. Interval",
        "Number of units: 200",
        "Switchers: 100",
        "Never-switchers: 100",
        "de Chaisemartin",
        "D'Haultfoeuille",
    ],
)
def test_format_minimal_result_contains(minimal_result, expected_text):
    formatted = format_didinter_result(minimal_result)

    assert isinstance(formatted, str)
    assert expected_text in formatted


@pytest.mark.parametrize(
    "expected_text",
    [
        "Placebo Effects",
        "-1",
        "-2",
        "Average Total Effect",
        "ATE",
        "Test of equal effects",
        "0.47",
        "Joint test",
        "0.67",
        "Estimation Details",
        "Effects estimated: 3",
        "Placebos estimated: 2",
        "Inference",
        "Confidence level: 95%",
    ],
)
def test_format_full_result_contains(full_result, expected_text):
    formatted = format_didinter_result(full_result)
    assert expected_text in formatted


@pytest.mark.parametrize(
    "param_key,param_value,expected_text",
    [
        ("cluster", "state", "Clustered standard errors: state"),
        ("normalized", True, "Normalized: Yes"),
        ("switchers", "in", "Switchers type: 'in'"),
        ("controls", ["x1", "x2"], "Controls: x1, x2"),
    ],
)
def test_format_with_estimation_params(minimal_effects, param_key, param_value, expected_text):
    params = {"effects": 1, "placebo": 0, param_key: param_value}
    result = DIDInterResult(effects=minimal_effects, estimation_params=params)

    formatted = format_didinter_result(result)
    assert expected_text in formatted


@pytest.mark.parametrize("method", ["__repr__", "__str__"])
def test_string_methods(minimal_result, method):
    result_str = getattr(minimal_result, method)()

    assert isinstance(result_str, str)
    assert "Intertemporal Treatment Effects" in result_str


def test_significance_markers():
    effects = EffectsResult(
        horizons=np.array([1.0, 2.0]),
        estimates=np.array([0.5, 0.01]),
        std_errors=np.array([0.1, 0.1]),
        ci_lower=np.array([0.304, -0.186]),
        ci_upper=np.array([0.696, 0.206]),
        n_switchers=np.array([100.0, 90.0]),
        n_observations=np.array([500.0, 450.0]),
    )

    result = DIDInterResult(
        effects=effects,
        estimation_params={"effects": 2, "placebo": 0},
    )

    formatted = format_didinter_result(result)
    assert "*" in formatted
    assert "confidence interval does not cover 0" in formatted
