"""Tests for processing panel data."""

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid.didcont.container import (
    AttgtResult,
    DoseResult,
    PTEParams,
    PTEResult,
)
from moderndid.didcont.estimation.process_panel import (
    _bootstrap_draw_params,
    compute_pte,
    pte,
    pte_default,
)


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
        "data": pl.DataFrame({"y": [1, 2], "g": [0, 1]}),
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
    data = pl.DataFrame(
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
    data = pl.DataFrame(
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


@pytest.mark.filterwarnings("ignore:No pre-treatment periods to test:UserWarning")
@pytest.mark.filterwarnings("ignore:Simultaneous band smaller than pointwise:UserWarning")
@pytest.mark.filterwarnings("ignore:No unit-level data available:UserWarning")
@pytest.mark.filterwarnings("ignore:Simultaneous critical value is NA/Inf:UserWarning")
@pytest.mark.filterwarnings("ignore:dropping some event times due to small groups:UserWarning")
@pytest.mark.filterwarnings("ignore:dropping some groups due to small groups:UserWarning")
def test_bootstrap_logic():
    data = pl.DataFrame(
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
        if isinstance(data, pl.DataFrame):
            return {"gt_data": data.head(4), "n1": 2, "disidx": np.array([True, True, False, False])}
        return {"gt_data": data[:4], "n1": 2, "disidx": np.array([True, True, False, False])}

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
    assert result_empirical.att_gt.influence_func is None

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
    assert result_fallback.att_gt.influence_func is None


def test_pte_dose_type():
    data = pl.DataFrame(
        {
            "y": np.random.randn(10),
            "id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "time": [1, 2] * 5,
            "group": [0, 0, 2, 2, 0, 0, 2, 2, 0, 0],
            "D": [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0],
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


def test_bootstrap_draw_params_keeps_estimate_settings_for_cohorts_in_draw(pte_params_basic):
    draw = pte_params_basic.data.filter(pl.col("G") != 2006)
    params = _bootstrap_draw_params(draw, ptep=pte_params_basic)

    np.testing.assert_array_equal(params.g_list, [2004, 2007])
    np.testing.assert_array_equal(params.t_list, pte_params_basic.t_list)
    assert params.data.height == draw.height
    assert params.base_period == pte_params_basic.base_period


@pytest.mark.filterwarnings("ignore:Simultaneous band smaller than pointwise:UserWarning")
@pytest.mark.parametrize(("gname", "tname"), [("G", "time"), ("cohort", "period"), ("G", "period")])
def test_pte_default_keeps_labels_when_columns_are_named_g_or_period(contdid_data, gname, tname):
    doubled = contdid_data.with_columns((pl.col("period") * 2).alias("period"), (pl.col("G") * 2).alias("G"))
    kwargs = {"yname": "Y", "idname": "id", "d_outcome": True, "biters": 10, "random_state": 0}
    reference = pte_default(
        data=doubled.rename({"G": "cohort", "period": "time"}), gname="cohort", tname="time", **kwargs
    )
    result = pte_default(data=doubled.rename({"G": gname, "period": tname}), gname=gname, tname=tname, **kwargs)

    np.testing.assert_array_equal(np.unique(reference.att_gt.groups), [4, 6, 8])
    np.testing.assert_array_equal(result.att_gt.groups, reference.att_gt.groups)
    np.testing.assert_array_equal(result.att_gt.times, reference.att_gt.times)
    np.testing.assert_array_equal(result.event_study.event_times, reference.event_study.event_times)
    np.testing.assert_allclose(result.event_study.att_by_event, reference.event_study.att_by_event, rtol=1e-12)
    np.testing.assert_allclose(
        result.event_study.influence_func["by_event"],
        reference.event_study.influence_func["by_event"],
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.filterwarnings("ignore:Simultaneous band smaller than pointwise:UserWarning")
@pytest.mark.parametrize("renamed", [{"G": "cohort", "id": "G"}, {"Y": "outcome", "id": "Y"}])
def test_pte_default_unit_column_named_like_a_working_column(contdid_data, renamed):
    names = {column: renamed.get(column, column) for column in ("Y", "id", "G")}
    kwargs = {"tname": "period", "d_outcome": True, "biters": 10, "random_state": 0}
    reference = pte_default(
        data=contdid_data.drop("D").rename({"Y": "outcome", "id": "unit", "G": "cohort"}),
        yname="outcome",
        idname="unit",
        gname="cohort",
        **kwargs,
    )
    result = pte_default(
        data=contdid_data.drop("D").rename(renamed), yname=names["Y"], idname=names["id"], gname=names["G"], **kwargs
    )

    np.testing.assert_allclose(result.att_gt.att, reference.att_gt.att, rtol=1e-12)
    np.testing.assert_allclose(result.event_study.att_by_event, reference.event_study.att_by_event, rtol=1e-12)
    np.testing.assert_allclose(result.event_study.se_by_event, reference.event_study.se_by_event, rtol=1e-12)


def test_pte_default_empirical_bootstrap_rejects_event_time_options(contdid_data):
    with pytest.raises(ValueError, match="The empirical bootstrap doesn't support min_e, balance_e\\."):
        pte_default(
            yname="Y",
            gname="G",
            tname="period",
            idname="id",
            data=contdid_data,
            d_outcome=True,
            boot_type="empirical",
            biters=5,
            min_e=-1,
            balance_e=1,
        )
