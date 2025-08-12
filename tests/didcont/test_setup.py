# pylint: disable=redefined-outer-name
"""Tests for setup functions."""

import numpy as np
import pandas as pd
import pytest

from moderndid.didcont.process.process_panel import (
    PTEParams,
    _choose_knots_quantile,
    _get_first_difference,
    _get_group,
    _get_group_inner,
    _make_balanced_panel,
    _map_to_idx,
    setup_pte,
    setup_pte_basic,
    setup_pte_cont,
)
from tests.didcont.dgp import simulate_contdid_data


@pytest.fixture
def contdid_data():
    return simulate_contdid_data(n=1000, seed=12345)


@pytest.fixture
def panel_data_with_group(panel_data_balanced):
    data = panel_data_balanced.copy()

    def assign_group(unit):
        if unit <= 10:
            return 2011
        if unit <= 15:
            return 2012
        return 0

    data["group"] = data["unit_id"].apply(assign_group)
    data.loc[data["time_id"] < data["group"], "d"] = 0
    data.loc[data["group"] == 0, "d"] = 0
    return data


def test_time_to_int():
    time_map = {2010: 1, 2011: 2, 2012: 3}
    assert _map_to_idx(2011, time_map) == 2
    assert _map_to_idx(2013, time_map) == 2013


def test_choose_knots_quantile():
    x = np.linspace(0, 10, 100)
    knots = _choose_knots_quantile(x, 3)
    assert len(knots) == 3
    assert knots[0] < knots[1] < knots[2]
    assert np.allclose(knots, [2.5, 5.0, 7.5], atol=0.2)


def test_choose_knots_quantile_zero_knots():
    x = np.linspace(0, 10, 100)
    knots = _choose_knots_quantile(x, 0)
    assert len(knots) == 0
    assert isinstance(knots, np.ndarray)


def test_setup_pte_basic(panel_data_with_group):
    params = setup_pte_basic(data=panel_data_with_group, yname="y", gname="group", tname="time_id", idname="unit_id")
    assert isinstance(params, PTEParams)
    assert "Y" in params.data.columns
    np.testing.assert_array_equal(params.g_list, [2011, 2012])
    np.testing.assert_array_equal(params.t_list, np.arange(2011, 2016))


def test_setup_pte_time_recoding(panel_data_with_group):
    params = setup_pte(data=panel_data_with_group, yname="y", gname="group", tname="time_id", idname="unit_id")
    assert params.data["period"].max() == 6
    assert set(params.data["G"].unique()) == {0, 2, 3}


def test_setup_pte_varying_base_period(panel_data_with_group):
    params = setup_pte(
        data=panel_data_with_group,
        yname="y",
        gname="group",
        tname="time_id",
        idname="unit_id",
        base_period="varying",
        required_pre_periods=1,
    )
    np.testing.assert_array_equal(params.t_list, [2, 3, 4, 5, 6])
    np.testing.assert_array_equal(params.g_list, [2, 3])


def test_setup_pte_universal_base_period(panel_data_with_group):
    params = setup_pte(
        data=panel_data_with_group,
        yname="y",
        gname="group",
        tname="time_id",
        idname="unit_id",
        base_period="universal",
        required_pre_periods=1,
    )
    np.testing.assert_array_equal(params.t_list, [1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(params.g_list, [2, 3])


def test_setup_pte_anticipation(panel_data_with_group):
    params = setup_pte(
        data=panel_data_with_group,
        yname="y",
        gname="group",
        tname="time_id",
        idname="unit_id",
        base_period="varying",
        anticipation=1,
    )
    np.testing.assert_array_equal(params.g_list, [3])
    params_anticipation_3 = setup_pte(
        data=panel_data_with_group,
        yname="y",
        gname="group",
        tname="time_id",
        idname="unit_id",
        base_period="varying",
        anticipation=3,
    )
    np.testing.assert_array_equal(params_anticipation_3.g_list, [])


def test_setup_pte_error_non_integer_time(panel_data_with_group):
    data = panel_data_with_group.copy()
    data["time_id"] = data["time_id"].astype(float) + 0.5
    with pytest.raises(ValueError, match="time periods must be positive integers"):
        setup_pte(data=data, yname="y", gname="group", tname="time_id", idname="unit_id")


@pytest.mark.parametrize("base_period", ["universal", "varying"])
def test_setup_pte_cont_base_periods(contdid_data, base_period):
    params = setup_pte_cont(
        data=contdid_data,
        yname="Y",
        gname="G",
        tname="time_period",
        idname="id",
        dname="D",
        base_period=base_period,
    )
    assert params.base_period == base_period


@pytest.mark.parametrize("anticipation", [0, 1])
def test_setup_pte_cont_anticipation(contdid_data, anticipation):
    params = setup_pte_cont(
        data=contdid_data,
        yname="Y",
        gname="G",
        tname="time_period",
        idname="id",
        dname="D",
        anticipation=anticipation,
    )
    assert params.anticipation == anticipation


def test_setup_pte_cont_knots(contdid_data):
    params = setup_pte_cont(
        data=contdid_data,
        yname="Y",
        gname="G",
        tname="time_period",
        idname="id",
        dname="D",
        num_knots=4,
    )
    assert params.num_knots == 4
    assert len(params.knots) == 4
    assert np.all(np.diff(params.knots) > 0)


def test_setup_pte_cont_dvals(contdid_data):
    dvals = [0.2, 0.4, 0.6, 0.8]
    params = setup_pte_cont(
        data=contdid_data,
        yname="Y",
        gname="G",
        tname="time_period",
        idname="id",
        dname="D",
        dvals=dvals,
    )
    np.testing.assert_array_equal(params.dvals, dvals)


def test_make_balanced_panel_basic(unbalanced_simple_panel):
    result = _make_balanced_panel(unbalanced_simple_panel, "id", "time")
    assert len(result["id"].unique()) == 3
    assert 3 not in result["id"].values
    assert all(result.groupby("id").size() == 3)


def test_make_balanced_panel_already_balanced(panel_data_balanced):
    result = _make_balanced_panel(panel_data_balanced, "unit_id", "time_id")
    assert len(result) == len(panel_data_balanced)
    assert len(result["unit_id"].unique()) == len(panel_data_balanced["unit_id"].unique())


def test_make_balanced_panel_invalid_input():
    with pytest.raises(TypeError, match="data must be a pandas DataFrame"):
        _make_balanced_panel([1, 2, 3], "id", "time")


def test_make_balanced_panel_empty():
    empty_df = pd.DataFrame({"id": [], "time": [], "y": []})
    result = _make_balanced_panel(empty_df, "id", "time")
    assert len(result) == 0


def test_get_first_difference_basic(panel_data_balanced):
    result = _get_first_difference(panel_data_balanced, "unit_id", "y", "time_id")
    grouped = panel_data_balanced.sort_values(["unit_id", "time_id"]).groupby("unit_id")

    for unit_id, group in grouped:
        unit_result = result[panel_data_balanced["unit_id"] == unit_id]
        assert pd.isna(unit_result.iloc[0])
        for i in range(1, len(group)):
            expected_diff = group["y"].iloc[i] - group["y"].iloc[i - 1]
            np.testing.assert_almost_equal(unit_result.iloc[i], expected_diff)


def test_get_first_difference_single_period():
    df = pd.DataFrame({"id": [1, 2, 3], "time": [1, 1, 1], "y": [10, 20, 30]})
    result = _get_first_difference(df, "id", "y", "time")
    assert result.isna().all()


def test_get_group_inner_never_treated():
    unit_df = pd.DataFrame({"time": [1, 2, 3], "treat": [0, 0, 0]})
    result = _get_group_inner(unit_df, "time", "treat")
    assert result == 0


def test_get_group_inner_always_treated():
    unit_df = pd.DataFrame({"time": [1, 2, 3], "treat": [1, 1, 1]})
    result = _get_group_inner(unit_df, "time", "treat")
    assert result == 1


def test_get_group_inner_staggered():
    unit_df = pd.DataFrame({"time": [1, 2, 3, 4], "treat": [0, 0, 1, 1]})
    result = _get_group_inner(unit_df, "time", "treat")
    assert result == 3


def test_get_group_basic(staggered_treatment_panel):
    result = _get_group(staggered_treatment_panel, "id", "time", "treat")
    expected = pd.Series([3, 3, 3, 3, 2, 2, 2, 2, 0, 0, 0, 0])
    pd.testing.assert_series_equal(result.reset_index(drop=True), expected)


def test_get_group_all_treated():
    df = pd.DataFrame({"id": [1, 1, 2, 2], "time": [1, 2, 1, 2], "treat": [1, 1, 1, 1]})
    result = _get_group(df, "id", "time", "treat")
    assert (result == 1).all()


def test_get_group_none_treated():
    df = pd.DataFrame({"id": [1, 1, 2, 2], "time": [1, 2, 1, 2], "treat": [0, 0, 0, 0]})
    result = _get_group(df, "id", "time", "treat")
    assert (result == 0).all()


@pytest.mark.parametrize(
    "treat_pattern,expected_group",
    [
        ([0, 1, 1], 2),
        ([0, 0, 1], 3),
        ([1, 1, 1], 1),
        ([0, 0, 0], 0),
    ],
)
def test_get_group_various_patterns(treat_pattern, expected_group):
    df = pd.DataFrame({"id": [1, 1, 1], "time": [1, 2, 3], "treat": treat_pattern})
    result = _get_group(df, "id", "time", "treat")
    assert (result == expected_group).all()


def test_integration_balanced_panel_with_groups(unbalanced_simple_panel):
    balanced = _make_balanced_panel(unbalanced_simple_panel, "id", "time")
    groups = _get_group(balanced, "id", "time", "treat")
    assert len(groups) == len(balanced)
    assert set(groups.unique()) == {0, 2, 3}
