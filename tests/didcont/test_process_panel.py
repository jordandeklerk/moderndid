# pylint: disable=redefined-outer-name, unused-argument
"""Tests for processing panel data."""

import numpy as np
import pandas as pd
import pytest

from moderndid.didcont.panel.container import (
    AttgtResult,
    DoseResult,
    PTEParams,
    PTEResult,
)
from moderndid.didcont.panel.process_panel import (
    _choose_knots_quantile,
    _get_first_difference,
    _get_group,
    _make_balanced_panel,
    _map_to_idx,
    compute_pte,
    pte,
    setup_pte,
    setup_pte_basic,
    setup_pte_cont,
)


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
    with pytest.raises(ValueError, match="Time periods must be positive integers."):
        setup_pte(data=data, yname="y", gname="group", tname="time_id", idname="unit_id")


@pytest.mark.parametrize("base_period", ["universal", "varying"])
def test_setup_pte_cont_base_periods(contdid_data, base_period):
    params = setup_pte_cont(
        data=contdid_data,
        yname="Y",
        gname="G",
        tname="period",
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
        tname="period",
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
        tname="period",
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
        tname="period",
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


def test_get_group_basic(staggered_treatment_panel):
    result = _get_group(staggered_treatment_panel, "id", "time", "treat")
    expected = pd.Series([3, 3, 3, 3, 2, 2, 2, 2, 0, 0, 0, 0])
    pd.testing.assert_series_equal(result.rename(None).reset_index(drop=True), expected)


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


def test_attgt_result_creation():
    result = AttgtResult(attgt=0.5, inf_func=np.array([0.1, 0.2, 0.3]), extra_gt_returns={"test": "data"})

    assert result.attgt == 0.5
    assert np.array_equal(result.inf_func, np.array([0.1, 0.2, 0.3]))
    assert result.extra_gt_returns == {"test": "data"}

    result_none = AttgtResult(attgt=1.0, inf_func=None, extra_gt_returns=None)

    assert result_none.attgt == 1.0
    assert result_none.inf_func is None
    assert result_none.extra_gt_returns is None


def test_pte_result_creation():
    params_dict = {
        "yname": "y",
        "gname": "g",
        "tname": "t",
        "idname": "id",
        "data": pd.DataFrame({"y": [1, 2], "g": [0, 1]}),
        "g_list": np.array([1, 2]),
        "t_list": np.array([1, 2]),
        "cband": True,
        "alp": 0.05,
        "boot_type": "multiplier",
        "anticipation": 0,
        "base_period": "varying",
        "weightsname": None,
        "control_group": "notyettreated",
        "gt_type": "att",
        "ret_quantile": 0.5,
        "biters": 20,
        "dname": None,
        "degree": None,
        "num_knots": None,
        "knots": None,
        "dvals": None,
        "target_parameter": None,
        "aggregation": None,
        "treatment_type": None,
        "xformula": "~1",
    }
    ptep = PTEParams(**params_dict)

    result = PTEResult(att_gt="mock_att_gt", overall_att="mock_overall", event_study="mock_event", ptep=ptep)

    assert result.att_gt == "mock_att_gt"
    assert result.overall_att == "mock_overall"
    assert result.event_study == "mock_event"
    assert result.ptep == ptep


def test_compute_pte_basic():
    data = pd.DataFrame(
        {
            "y": [1, 2, 3, 4, 5, 6],
            "id": [1, 1, 2, 2, 3, 3],
            "time": [1, 2, 1, 2, 1, 2],
            "group": [0, 0, 2, 2, 0, 0],
            "D": [0, 0, 0, 1, 0, 0],
        }
    )

    params_dict = {
        "yname": "y",
        "gname": "group",
        "tname": "time",
        "idname": "id",
        "data": data,
        "g_list": np.array([2]),
        "t_list": np.array([2]),
        "cband": True,
        "alp": 0.05,
        "boot_type": "multiplier",
        "anticipation": 0,
        "base_period": "varying",
        "weightsname": None,
        "control_group": "notyettreated",
        "gt_type": "att",
        "ret_quantile": 0.5,
        "biters": 20,
        "dname": None,
        "degree": None,
        "num_knots": None,
        "knots": None,
        "dvals": None,
        "target_parameter": None,
        "aggregation": None,
        "treatment_type": None,
        "xformula": "~1",
    }
    ptep = PTEParams(**params_dict)

    def mock_subset_fun(data, g, tp, **kwargs):
        return {"gt_data": data, "n1": len(data) // 2, "disidx": np.array([True, True, True])}

    def mock_attgt_fun(gt_data, **kwargs):
        return AttgtResult(attgt=0.5, inf_func=np.array([0.1, 0.2, 0.3]), extra_gt_returns={"test": "data"})

    result = compute_pte(ptep, mock_subset_fun, mock_attgt_fun)

    assert "attgt_list" in result
    assert "influence_func" in result
    assert "extra_gt_returns" in result

    assert len(result["attgt_list"]) == 1
    assert result["attgt_list"][0]["att"] == 0.5
    assert result["attgt_list"][0]["group"] == 2
    assert result["attgt_list"][0]["time_period"] == 2

    assert result["influence_func"].shape == (3, 1)


def test_compute_pte_universal_base_period():
    data = pd.DataFrame(
        {
            "y": np.random.randn(10),
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
            "time": [1, 2, 3, 1, 2, 3, 1, 2, 3, 4],
            "group": [0, 0, 0, 3, 3, 3, 0, 0, 0, 0],
        }
    )

    params_dict = {
        "yname": "y",
        "gname": "group",
        "tname": "time",
        "idname": "id",
        "data": data,
        "g_list": np.array([3]),
        "t_list": np.array([2, 3]),
        "cband": True,
        "alp": 0.05,
        "boot_type": "multiplier",
        "anticipation": 0,
        "base_period": "universal",
        "weightsname": None,
        "control_group": "notyettreated",
        "gt_type": "att",
        "ret_quantile": 0.5,
        "biters": 20,
        "dname": None,
        "degree": None,
        "num_knots": None,
        "knots": None,
        "dvals": None,
        "target_parameter": None,
        "aggregation": None,
        "treatment_type": None,
        "xformula": "~1",
    }
    ptep = PTEParams(**params_dict)

    def mock_subset_fun(data, g, tp, **kwargs):
        return {"gt_data": data, "n1": 2, "disidx": np.ones(3, dtype=bool)}

    def mock_attgt_fun(gt_data, **kwargs):
        return AttgtResult(attgt=0.3, inf_func=None, extra_gt_returns=None)

    result = compute_pte(ptep, mock_subset_fun, mock_attgt_fun)

    assert len(result["attgt_list"]) == 2

    boundary_result = [r for r in result["attgt_list"] if r["time_period"] == 2][0]
    assert boundary_result["att"] == 0


def test_bootstrap_logic():
    data = pd.DataFrame(
        {
            "y": np.random.randn(12),
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "time": [1, 2, 3] * 4,
            "group": [0, 0, 0, 2, 2, 2, 0, 0, 0, 2, 2, 2],
        }
    )

    def mock_setup_pte(yname, gname, tname, idname, data, **kwargs):
        params_dict = {
            "yname": yname,
            "gname": gname,
            "tname": tname,
            "idname": idname,
            "data": data,
            "g_list": np.array([2]),
            "t_list": np.array([2, 3]),
            "cband": kwargs.get("cband", True),
            "alp": kwargs.get("alp", 0.05),
            "boot_type": kwargs.get("boot_type", "multiplier"),
            "anticipation": 0,
            "base_period": "varying",
            "weightsname": kwargs.get("weightsname"),
            "control_group": "notyettreated",
            "gt_type": kwargs.get("gt_type", "att"),
            "ret_quantile": kwargs.get("ret_quantile", 0.5),
            "biters": kwargs.get("biters", 100),
            "dname": None,
            "degree": None,
            "num_knots": None,
            "knots": None,
            "dvals": None,
            "target_parameter": None,
            "aggregation": None,
            "treatment_type": None,
            "xformula": "~1",
        }
        return PTEParams(**params_dict)

    def mock_subset_fun(data, g, tp, **kwargs):
        return {"gt_data": data.head(4), "n1": 2, "disidx": np.array([True, True, False, False])}

    def mock_attgt_with_inf(gt_data, **kwargs):
        return AttgtResult(attgt=0.5, inf_func=np.array([0.1, 0.2]), extra_gt_returns=None)

    def mock_attgt_no_inf(gt_data, **kwargs):
        return AttgtResult(attgt=0.5, inf_func=None, extra_gt_returns=None)

    result_empirical = pte(
        yname="y",
        gname="group",
        tname="time",
        idname="id",
        data=data,
        setup_pte_fun=mock_setup_pte,
        subset_fun=mock_subset_fun,
        attgt_fun=mock_attgt_with_inf,
        boot_type="empirical",
        biters=10,
    )
    assert result_empirical.att_gt["influence_func"] is None

    result_multiplier = pte(
        yname="y",
        gname="group",
        tname="time",
        idname="id",
        data=data,
        setup_pte_fun=mock_setup_pte,
        subset_fun=mock_subset_fun,
        attgt_fun=mock_attgt_with_inf,
        boot_type="multiplier",
        biters=10,
    )
    assert hasattr(result_multiplier.att_gt, "influence_func")
    assert result_multiplier.att_gt.influence_func is not None

    result_fallback = pte(
        yname="y",
        gname="group",
        tname="time",
        idname="id",
        data=data,
        setup_pte_fun=mock_setup_pte,
        subset_fun=mock_subset_fun,
        attgt_fun=mock_attgt_no_inf,
        boot_type="multiplier",
        biters=10,
    )
    assert result_fallback.att_gt["influence_func"] is None


def test_pte_dose_type():
    data = pd.DataFrame(
        {
            "y": np.random.randn(10),
            "id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "time": [1, 2] * 5,
            "group": [0, 0, 2, 2, 0, 0, 2, 2, 0, 0],
            "D": [0, 0, 0, 0.5, 0, 0, 0, 0.8, 0, 0],
        }
    )

    def mock_setup_pte(yname, gname, tname, idname, data, **kwargs):
        params_dict = {
            "yname": yname,
            "gname": gname,
            "tname": tname,
            "idname": idname,
            "data": data,
            "g_list": np.array([2]),
            "t_list": np.array([2]),
            "cband": kwargs.get("cband", True),
            "alp": kwargs.get("alp", 0.05),
            "boot_type": kwargs.get("boot_type", "multiplier"),
            "anticipation": 0,
            "base_period": "varying",
            "weightsname": kwargs.get("weightsname"),
            "control_group": "notyettreated",
            "gt_type": kwargs.get("gt_type", "att"),
            "ret_quantile": kwargs.get("ret_quantile", 0.5),
            "biters": kwargs.get("biters", 100),
            "dname": "D",
            "degree": 1,
            "num_knots": 0,
            "knots": np.array([]),
            "dvals": np.array([0.5, 0.8]),
            "target_parameter": "ATT",
            "aggregation": "dose",
            "treatment_type": "continuous",
            "xformula": "~1",
        }
        return PTEParams(**params_dict)

    def mock_subset_fun(data, g, tp, **kwargs):
        return {"gt_data": data, "n1": 2, "disidx": np.ones(5, dtype=bool)}

    def mock_attgt_fun(gt_data, **kwargs):
        return AttgtResult(attgt=0.3, inf_func=None, extra_gt_returns={"att_dose": np.array([0.2, 0.4])})

    def mock_process_dose_gt(res, ptep, **kwargs):
        return DoseResult(dose=ptep.dvals, overall_att=0.3, overall_att_se=0.1)

    result = pte(
        yname="y",
        gname="group",
        tname="time",
        idname="id",
        data=data,
        setup_pte_fun=mock_setup_pte,
        subset_fun=mock_subset_fun,
        attgt_fun=mock_attgt_fun,
        gt_type="dose",
        process_dose_gt_fun=mock_process_dose_gt,
    )

    assert isinstance(result, DoseResult)
    assert result.overall_att == 0.3
    assert result.overall_att_se == 0.1
