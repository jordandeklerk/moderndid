# pylint: disable=redefined-outer-name
"""Tests for control variable adjustment functions."""

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid.core.preprocess.config import DIDInterConfig
from moderndid.didinter.controls import (
    apply_control_adjustment,
    compute_control_coefficients,
)


@pytest.fixture
def control_test_data():
    rng = np.random.default_rng(42)
    n_units = 30
    n_periods = 4

    units = np.repeat(np.arange(n_units), n_periods)
    periods = np.tile(np.arange(1, n_periods + 1), n_units)

    d_sq = np.zeros(len(units))
    d_sq[units < 10] = 0.0
    d_sq[(units >= 10) & (units < 20)] = 1.0
    d_sq[units >= 20] = 1.0

    f_g = np.full(len(units), float("inf"))
    for unit in range(10, 20):
        unit_mask = units == unit
        f_g[unit_mask] = 3

    x1 = rng.standard_normal(len(units))
    x2 = rng.standard_normal(len(units))
    y = rng.standard_normal(len(units)) + 0.5 * x1 + 0.3 * x2
    diff_y_1 = np.zeros(len(units))

    return pl.DataFrame(
        {
            "id": units,
            "time": periods,
            "y": y,
            "d_sq": d_sq,
            "F_g": f_g,
            "x1": x1,
            "x2": x2,
            "weight_gt": np.ones(len(units)),
            "diff_y_1": diff_y_1,
            "lag_x1_1": np.roll(x1, 1),
            "lag_x2_1": np.roll(x2, 1),
        }
    )


def test_compute_control_coefficients_basic(control_test_data):
    config = DIDInterConfig(
        yname="y",
        tname="time",
        gname="id",
        dname="d",
        controls=["x1", "x2"],
    )

    coefficients = compute_control_coefficients(control_test_data, config, horizon=1)

    assert isinstance(coefficients, dict)


@pytest.mark.parametrize("controls", [None, []])
def test_compute_control_coefficients_no_controls(controls):
    config = DIDInterConfig(
        yname="y",
        tname="time",
        gname="id",
        dname="d",
        controls=controls,
    )

    df = pl.DataFrame(
        {
            "id": [1, 2],
            "time": [1, 1],
            "y": [0.5, 0.6],
        }
    )

    coefficients = compute_control_coefficients(df, config, horizon=1)

    assert not coefficients


@pytest.mark.parametrize(
    "controls,expected_column",
    [
        (["x1", "x2"], "diff_y_1"),
        (None, "diff_y_1"),
    ],
)
def test_apply_control_adjustment_returns_diff_column(control_test_data, controls, expected_column):
    config = DIDInterConfig(
        yname="y",
        tname="time",
        gname="id",
        dname="d",
        controls=controls,
    )

    result = apply_control_adjustment(control_test_data, config, horizon=1, coefficients={})

    assert expected_column in result.columns


@pytest.mark.parametrize(
    "expected_column",
    ["diff_x1_1", "diff_x2_1"],
)
def test_apply_control_adjustment_creates_diff_columns(control_test_data, expected_column):
    config = DIDInterConfig(
        yname="y",
        tname="time",
        gname="id",
        dname="d",
        controls=["x1", "x2"],
    )

    coefficients = {
        0.0: {"theta": np.array([0.5, 0.3]), "inv_denom": None, "useful": True},
        1.0: {"theta": np.array([0.4, 0.2]), "inv_denom": None, "useful": True},
    }

    result = apply_control_adjustment(control_test_data, config, horizon=1, coefficients=coefficients)

    assert expected_column in result.columns


@pytest.mark.parametrize(
    "expected_key",
    ["theta", "inv_denom", "useful"],
)
def test_coefficient_structure_keys(control_test_data, expected_key):
    config = DIDInterConfig(
        yname="y",
        tname="time",
        gname="id",
        dname="d",
        controls=["x1", "x2"],
    )

    coefficients = compute_control_coefficients(control_test_data, config, horizon=1)

    for _, coef_dict in coefficients.items():
        assert expected_key in coef_dict


def test_coefficient_useful_is_bool(control_test_data):
    config = DIDInterConfig(
        yname="y",
        tname="time",
        gname="id",
        dname="d",
        controls=["x1", "x2"],
    )

    coefficients = compute_control_coefficients(control_test_data, config, horizon=1)

    for _, coef_dict in coefficients.items():
        assert isinstance(coef_dict["useful"], bool)


def test_compute_control_coefficients_insufficient_data():
    config = DIDInterConfig(
        yname="y",
        tname="time",
        gname="id",
        dname="d",
        controls=["x1", "x2"],
    )

    df = pl.DataFrame(
        {
            "id": [1],
            "time": [1],
            "y": [0.5],
            "d_sq": [0.0],
            "F_g": [float("inf")],
            "x1": [1.0],
            "x2": [2.0],
            "weight_gt": [1.0],
            "diff_y_1": [0.1],
            "lag_x1_1": [0.5],
            "lag_x2_1": [1.0],
        }
    )

    coefficients = compute_control_coefficients(df, config, horizon=1)

    for _, coef_dict in coefficients.items():
        assert coef_dict["useful"] is False
        np.testing.assert_array_equal(coef_dict["theta"], np.zeros(2))
