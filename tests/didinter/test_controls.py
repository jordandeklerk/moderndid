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


def test_compute_control_coefficients_basic(control_test_data):
    config = DIDInterConfig(
        yname="y",
        tname="time",
        gname="id",
        dname="d",
        xformla="~ x1 + x2",
    )

    _, coefficients = compute_control_coefficients(control_test_data, config, n_groups=30)

    assert isinstance(coefficients, dict)


@pytest.mark.parametrize("xformla", ["~1", ""])
def test_compute_control_coefficients_no_controls(xformla):
    config = DIDInterConfig(
        yname="y",
        tname="time",
        gname="id",
        dname="d",
        xformla=xformla if xformla else "~1",
    )

    df = pl.DataFrame(
        {
            "id": [1, 2],
            "time": [1, 1],
            "y": [0.5, 0.6],
        }
    )

    _, coefficients = compute_control_coefficients(df, config, n_groups=2)

    assert not coefficients


@pytest.mark.parametrize(
    "xformla,expected_column",
    [
        ("~ x1 + x2", ".diff_y_1"),
        ("~1", ".diff_y_1"),
    ],
)
def test_apply_control_adjustment_returns_diff_column(control_test_data, xformla, expected_column):
    config = DIDInterConfig(
        yname="y",
        tname="time",
        gname="id",
        dname="d",
        xformla=xformla,
    )

    result = apply_control_adjustment(control_test_data, config, horizon=1, coefficients={}, horizon_type="effect")

    assert expected_column in result.columns


@pytest.mark.parametrize(
    "expected_column",
    [".ctrl_diff_0_1", ".ctrl_diff_1_1"],
)
def test_apply_control_adjustment_creates_diff_columns(control_test_data, expected_column):
    config = DIDInterConfig(
        yname="y",
        tname="time",
        gname="id",
        dname="d",
        xformla="~ x1 + x2",
    )

    coefficients = {
        1: {"theta": np.array([0.5, 0.3]), "inv_denom": None, "useful": True},
        2: {"theta": np.array([0.4, 0.2]), "inv_denom": None, "useful": True},
    }

    result = apply_control_adjustment(
        control_test_data, config, horizon=1, coefficients=coefficients, horizon_type="effect"
    )

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
        xformla="~ x1 + x2",
    )

    _, coefficients = compute_control_coefficients(control_test_data, config, n_groups=30)

    for _, coef_dict in coefficients.items():
        assert expected_key in coef_dict


def test_coefficient_useful_is_bool(control_test_data):
    config = DIDInterConfig(
        yname="y",
        tname="time",
        gname="id",
        dname="d",
        xformla="~ x1 + x2",
    )

    _, coefficients = compute_control_coefficients(control_test_data, config, n_groups=30)

    for _, coef_dict in coefficients.items():
        assert isinstance(coef_dict["useful"], bool)


def test_compute_control_coefficients_insufficient_data():
    config = DIDInterConfig(
        yname="y",
        tname="time",
        gname="id",
        dname="d",
        xformla="~ x1 + x2",
    )

    df = pl.DataFrame(
        {
            "id": [1],
            "time": [1],
            "y": [0.5],
            "d_sq": [0.0],
            "d_sq_int": [1],
            "F_g": [float("inf")],
            "x1": [1.0],
            "x2": [2.0],
            "weight_gt": [1.0],
            ".diff_y_1": [0.1],
        }
    )

    _, coefficients = compute_control_coefficients(df, config, n_groups=1)

    for _, coef_dict in coefficients.items():
        assert coef_dict["useful"] is False
        np.testing.assert_array_equal(coef_dict["theta"], np.zeros(2))
