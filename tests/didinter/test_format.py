"""Tests for result formatting functions."""

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid.didinter import (
    DIDInterResult,
    EffectsResult,
)
from moderndid.didinter.format import format_didinter_result


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
        ("cluster", "state", "Analytical standard errors clustered by state"),
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


@pytest.mark.parametrize(
    ("params", "expected", "absent"),
    [
        ({"boot": False}, [" Analytical standard errors"], ["Bootstrap", "clustered"]),
        ({"boot": True}, [" Bootstrap standard errors"], ["Analytical", "clustered"]),
        ({"boot": True, "cluster": "state"}, [" Bootstrap standard errors clustered by state"], ["Analytical"]),
        ({"boot": False, "cluster": "state"}, [" Analytical standard errors clustered by state"], ["Bootstrap"]),
    ],
)
def test_format_labels_bootstrap_standard_errors(minimal_effects, params, expected, absent):
    result = DIDInterResult(effects=minimal_effects, estimation_params={"effects": 2, "placebo": 0, **params})

    formatted = format_didinter_result(result)

    assert all(text in formatted for text in expected)
    assert not any(text in formatted for text in absent)


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
