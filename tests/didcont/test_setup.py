# pylint: disable=redefined-outer-name
"""Tests for panel treatment effects setup."""

import numpy as np
import pandas as pd
import pytest

from moderndid.didcont.setup import (
    PTEParams,
    _convert_single_time,
    _original_to_new_time,
    setup_pte,
    setup_pte_basic,
)


def test_setup_pte_basic_balanced_panel(panel_data_balanced):
    params = setup_pte_basic(data=panel_data_balanced, yname="y", tname="time_id", dname="d", idname="unit_id")

    assert isinstance(params, PTEParams)
    assert len(params.y) == len(panel_data_balanced)
    assert len(params.d) == len(panel_data_balanced)
    assert params.n_units == 20
    assert params.n_periods == 6
    assert params.n_obs == 120
    assert params.balanced is True
    assert params.base_period == 1
    assert params.anticipation == 0
    assert params.control_group == "never_treated"


def test_setup_pte_basic_unbalanced_panel(panel_data_unbalanced):
    params = setup_pte_basic(data=panel_data_unbalanced, yname="y", tname="time_id", dname="d", idname="unit_id")

    assert isinstance(params, PTEParams)
    assert params.balanced is False
    assert params.n_units == 15
    assert params.n_obs == len(panel_data_unbalanced)


def test_setup_pte_basic_with_covariates(panel_data_with_covariates):
    params = setup_pte_basic(
        data=panel_data_with_covariates,
        yname="y",
        tname="time_id",
        dname="d",
        idname="unit_id",
        xformla="~ age + education + income",
    )

    assert params.x.shape[1] == 5
    assert params.n_obs == len(panel_data_with_covariates)


def test_setup_pte_basic_with_weights(panel_data_with_weights):
    params = setup_pte_basic(
        data=panel_data_with_weights, yname="y", tname="time_id", dname="d", idname="unit_id", weightsname="weights"
    )

    assert not np.allclose(params.weights, 1.0)
    assert len(params.weights) == len(panel_data_with_weights)


def test_setup_pte_basic_no_weights(panel_data_balanced):
    params = setup_pte_basic(data=panel_data_balanced, yname="y", tname="time_id", dname="d", idname="unit_id")

    assert np.allclose(params.weights, 1.0)


@pytest.mark.parametrize(
    "base_period,expected",
    [
        ("varying", 1),
        ("universal", 5),
        (2012, 3),
    ],
)
def test_setup_pte_basic_base_period_options(panel_data_balanced, base_period, expected):
    params = setup_pte_basic(
        data=panel_data_balanced, yname="y", tname="time_id", dname="d", idname="unit_id", base_period=base_period
    )

    assert params.base_period == expected


def test_setup_pte_basic_invalid_base_period(panel_data_balanced):
    with pytest.raises(ValueError, match="not found in data"):
        setup_pte_basic(
            data=panel_data_balanced, yname="y", tname="time_id", dname="d", idname="unit_id", base_period=1999
        )


def test_setup_pte_basic_invalid_base_period_type(panel_data_balanced):
    with pytest.raises(ValueError, match="must be 'varying', 'universal', or int"):
        setup_pte_basic(
            data=panel_data_balanced, yname="y", tname="time_id", dname="d", idname="unit_id", base_period="invalid"
        )


@pytest.mark.parametrize("anticipation", [0, 1, 2, 5])
def test_setup_pte_basic_anticipation_values(panel_data_balanced, anticipation):
    params = setup_pte_basic(
        data=panel_data_balanced, yname="y", tname="time_id", dname="d", idname="unit_id", anticipation=anticipation
    )

    assert params.anticipation == anticipation


@pytest.mark.parametrize("control_group", ["never_treated", "not_yet_treated"])
def test_setup_pte_basic_control_group_options(panel_data_balanced, control_group):
    params = setup_pte_basic(
        data=panel_data_balanced, yname="y", tname="time_id", dname="d", idname="unit_id", control_group=control_group
    )

    assert params.control_group == control_group


def test_setup_pte_basic_time_mapping(panel_data_balanced):
    params = setup_pte_basic(data=panel_data_balanced, yname="y", tname="time_id", dname="d", idname="unit_id")

    expected_orig_times = np.array([2010, 2011, 2012, 2013, 2014, 2015])
    np.testing.assert_array_equal(params.orig_time_periods, expected_orig_times)

    expected_map = {2010: 1, 2011: 2, 2012: 3, 2013: 4, 2014: 5, 2015: 6}
    assert params.time_map == expected_map

    assert np.min(params.time_ids) == 1
    assert np.max(params.time_ids) == 6


def test_setup_pte_basic_formula_error():
    data = pd.DataFrame({"y": [1, 2, 3], "time_id": [1, 2, 3], "d": [0.5, 1.0, 1.5], "unit_id": [1, 1, 1]})

    with pytest.raises(ValueError, match="Error processing xformla"):
        setup_pte_basic(
            data=data, yname="y", tname="time_id", dname="d", idname="unit_id", xformla="~ nonexistent_variable"
        )


def test_setup_pte_type_validation():
    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        setup_pte(data=[1, 2, 3], yname="y", tname="time_id", dname="d", idname="unit_id")


def test_setup_pte_empty_data():
    empty_data = pd.DataFrame()

    with pytest.raises(ValueError, match="cannot be empty"):
        setup_pte(data=empty_data, yname="y", tname="time_id", dname="d", idname="unit_id")


def test_setup_pte_missing_columns():
    data = pd.DataFrame({"y": [1, 2, 3], "time_id": [1, 2, 3]})

    with pytest.raises(KeyError, match="Required columns missing"):
        setup_pte(data=data, yname="y", tname="time_id", dname="d", idname="unit_id")


def test_setup_pte_missing_values_in_key_vars():
    data = pd.DataFrame({"y": [1, np.nan, 3], "time_id": [1, 2, 3], "d": [0.5, 1.0, 1.5], "unit_id": [1, 1, 1]})

    with pytest.raises(ValueError, match="Missing values found in y"):
        setup_pte(data=data, yname="y", tname="time_id", dname="d", idname="unit_id")


def test_setup_pte_missing_weights_column():
    data = pd.DataFrame({"y": [1, 2, 3], "time_id": [1, 2, 3], "d": [0.5, 1.0, 1.5], "unit_id": [1, 1, 1]})

    with pytest.raises(KeyError, match="Weights column 'weights' not found"):
        setup_pte(data=data, yname="y", tname="time_id", dname="d", idname="unit_id", weightsname="weights")


def test_setup_pte_missing_values_in_weights():
    data = pd.DataFrame(
        {
            "y": [1, 2, 3],
            "time_id": [1, 2, 3],
            "d": [0.5, 1.0, 1.5],
            "unit_id": [1, 1, 1],
            "weights": [1.0, np.nan, 1.0],
        }
    )

    with pytest.raises(ValueError, match="Missing values found in weights"):
        setup_pte(data=data, yname="y", tname="time_id", dname="d", idname="unit_id", weightsname="weights")


def test_setup_pte_negative_weights():
    data = pd.DataFrame(
        {"y": [1, 2, 3], "time_id": [1, 2, 3], "d": [0.5, 1.0, 1.5], "unit_id": [1, 1, 1], "weights": [1.0, -0.5, 1.0]}
    )

    with pytest.raises(ValueError, match="All weights must be non-negative"):
        setup_pte(data=data, yname="y", tname="time_id", dname="d", idname="unit_id", weightsname="weights")


def test_setup_pte_insufficient_observations():
    small_data = pd.DataFrame({"y": [1, 2], "time_id": [1, 2], "d": [0.5, 1.0], "unit_id": [1, 1]})

    with pytest.raises(ValueError, match="Insufficient observations.*Need at least 10"):
        setup_pte(data=small_data, yname="y", tname="time_id", dname="d", idname="unit_id")


def test_setup_pte_insufficient_units():
    data = pd.DataFrame(
        {
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "time_id": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
            "d": [0.5] * 12,
            "unit_id": [1] * 12,
        }
    )

    with pytest.raises(ValueError, match="Insufficient units.*Need at least 2"):
        setup_pte(data=data, yname="y", tname="time_id", dname="d", idname="unit_id")


def test_setup_pte_insufficient_periods():
    data = pd.DataFrame(
        {
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "time_id": [1] * 12,
            "d": [0.5] * 12,
            "unit_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
        }
    )

    with pytest.raises(ValueError, match="Insufficient time periods.*Need at least 2"):
        setup_pte(data=data, yname="y", tname="time_id", dname="d", idname="unit_id")


def test_setup_pte_invalid_anticipation():
    data = pd.DataFrame(
        {
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "time_id": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "d": [0.5] * 12,
            "unit_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
        }
    )

    with pytest.raises(ValueError, match="anticipation must be a non-negative integer"):
        setup_pte(data=data, yname="y", tname="time_id", dname="d", idname="unit_id", anticipation=-1)


def test_setup_pte_invalid_control_group():
    data = pd.DataFrame(
        {
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "time_id": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "d": [0.5] * 12,
            "unit_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
        }
    )

    with pytest.raises(ValueError, match="control_group must be"):
        setup_pte(data=data, yname="y", tname="time_id", dname="d", idname="unit_id", control_group="invalid_group")


def test_setup_pte_unbalanced_panel_not_allowed(panel_data_unbalanced):
    with pytest.raises(ValueError, match="Panel data is unbalanced"):
        setup_pte(
            data=panel_data_unbalanced,
            yname="y",
            tname="time_id",
            dname="d",
            idname="unit_id",
            allow_unbalanced_panel=False,
        )


def test_setup_pte_non_numeric_treatment():
    data = pd.DataFrame(
        {
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "time_id": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "d": ["low", "high"] * 6,
            "unit_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
        }
    )

    with pytest.raises(ValueError, match="Treatment variable.*must be numeric"):
        setup_pte(data=data, yname="y", tname="time_id", dname="d", idname="unit_id")


def test_setup_pte_non_numeric_outcome():
    data = pd.DataFrame(
        {
            "y": ["low", "high"] * 6,
            "time_id": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "d": [0.5] * 12,
            "unit_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
        }
    )

    with pytest.raises(ValueError, match="Outcome variable.*must be numeric"):
        setup_pte(data=data, yname="y", tname="time_id", dname="d", idname="unit_id")


def test_setup_pte_infinite_outcome():
    data = pd.DataFrame(
        {
            "y": [1, 2, np.inf, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "time_id": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "d": [0.5] * 12,
            "unit_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
        }
    )

    with pytest.raises(ValueError, match="Infinite values found in outcome"):
        setup_pte(data=data, yname="y", tname="time_id", dname="d", idname="unit_id")


def test_setup_pte_infinite_treatment():
    data = pd.DataFrame(
        {
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "time_id": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "d": [0.5, np.inf, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            "unit_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
        }
    )

    with pytest.raises(ValueError, match="Infinite values found in treatment"):
        setup_pte(data=data, yname="y", tname="time_id", dname="d", idname="unit_id")


def test_setup_pte_successful_run(panel_data_balanced):
    params = setup_pte(
        data=panel_data_balanced,
        yname="y",
        tname="time_id",
        dname="d",
        idname="unit_id",
        xformla="~ x1 + x2",
        weightsname="weights",
        base_period="universal",
        anticipation=1,
        control_group="not_yet_treated",
    )

    assert isinstance(params, PTEParams)
    assert params.x.shape[1] == 3
    assert params.base_period == 5
    assert params.anticipation == 1
    assert params.control_group == "not_yet_treated"
    assert params.balanced is True


def test_original_to_new_time_single_value():
    time_map = {2010: 1, 2011: 2, 2012: 3}

    result = _original_to_new_time(2011, time_map)
    assert result == 2

    result = _original_to_new_time(2020, time_map)
    assert np.isnan(result)


def test_original_to_new_time_array():
    time_map = {2010: 1, 2011: 2, 2012: 3}
    orig_times = [2010, 2012, 2020]

    result = _original_to_new_time(orig_times, time_map)
    expected = np.array([1, 3, np.nan])
    np.testing.assert_array_equal(result, expected)


def test_convert_single_time():
    time_map = {2010: 1, 2011: 2, 2012: 3}

    assert _convert_single_time(2011, time_map) == 2
    assert np.isnan(_convert_single_time(2020, time_map))


def test_setup_pte_complex_formula(panel_data_with_covariates):
    params = setup_pte(
        data=panel_data_with_covariates,
        yname="y",
        tname="time_id",
        dname="d",
        idname="unit_id",
        xformla="~ age + I(age**2) + education * income",
    )

    assert params.x.shape[1] > 4
    assert params.n_obs == len(panel_data_with_covariates)


@pytest.mark.parametrize(
    "data_fixture",
    ["panel_data_balanced", "panel_data_unbalanced", "panel_data_with_covariates", "panel_data_with_weights"],
)
def test_setup_pte_all_fixtures(data_fixture, request):
    data = request.getfixturevalue(data_fixture)

    params = setup_pte(data=data, yname="y", tname="time_id", dname="d", idname="unit_id")

    assert isinstance(params, PTEParams)
    assert params.n_obs == len(data)
    assert params.n_units == data["unit_id"].nunique()
    assert params.n_periods == data["time_id"].nunique()


def test_setup_pte_preserves_original_time_info(panel_data_with_weights):
    params = setup_pte(data=panel_data_with_weights, yname="y", tname="time_id", dname="d", idname="unit_id")

    orig_times = sorted(panel_data_with_weights["time_id"].unique())
    np.testing.assert_array_equal(params.orig_time_periods, orig_times)

    for orig_time in orig_times:
        assert orig_time in params.time_map
        new_time = params.time_map[orig_time]
        assert new_time >= 1
        assert new_time <= params.n_periods
