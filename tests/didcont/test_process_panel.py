"""Tests for processing panel data."""

import re

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid import cont_did, setup_pte, setup_pte_basic, setup_pte_cont
from moderndid.didcont import pte_attgt
from moderndid.didcont.cont_did import cont_did_acrt, cont_two_by_two_subset
from moderndid.didcont.container import (
    AttgtResult,
    DoseResult,
    PTEParams,
    PTEResult,
)
from moderndid.didcont.estimation.process_panel import (
    _bootstrap_draw_params,
    _check_cell_estimates,
    _two_by_two_subset,
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
        "alpha": 0.05,
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
        "alpha": 0.05,
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
        "alpha": 0.05,
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
            "alpha": kwargs.get("alpha", 0.05),
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
            "alpha": kwargs.get("alpha", 0.05),
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


@pytest.mark.parametrize(
    ("window", "message"),
    [
        ({"max_e": np.nan}, "max_e=nan is not valid. Must be a number or inf."),
        (
            {"min_e": 100},
            "No group-time effect has an event time within min_e=100 and max_e=inf. Widen the window or check the "
            "warnings for cells without an estimate.",
        ),
    ],
)
def test_pte_default_rejects_windows_without_event_times(contdid_data, window, message):
    spec = {"yname": "Y", "gname": "G", "tname": "period", "idname": "id", "d_outcome": True, "biters": 5}

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        pte_default(data=contdid_data, random_state=0, **spec, **window)


@pytest.mark.parametrize(
    ("setup_pte_fun", "options", "message"),
    [
        (
            setup_pte_basic,
            {"anticipation": 1},
            "anticipation=1 needs a setup function that takes it. setup_pte_basic() doesn't.",
        ),
        (
            setup_pte_basic,
            {"base_period": "universal", "control_group": "nevertreated"},
            "base_period='universal' and control_group='nevertreated' need a setup function that takes them. "
            "setup_pte_basic() doesn't.",
        ),
        (setup_pte_basic, {}, "setup_pte_basic() got an unexpected keyword argument 'weightsname'"),
        (setup_pte, {"dname": "D"}, "dname='D' needs a setup function that takes it. setup_pte() doesn't."),
    ],
)
def test_pte_rejects_options_that_its_setup_function_cannot_honor(contdid_data, setup_pte_fun, options, message):
    spec = {"yname": "Y", "gname": "G", "tname": "period", "idname": "id", "d_outcome": True, "biters": 5}

    with pytest.raises(TypeError, match=f"^{re.escape(message)}"):
        pte(
            data=contdid_data,
            setup_pte_fun=setup_pte_fun,
            subset_fun=_two_by_two_subset,
            attgt_fun=pte_attgt,
            random_state=0,
            **spec,
            **options,
        )


@pytest.mark.parametrize("control_group", ["notyettreated", "nevertreated"])
def test_pte_with_setup_pte_cont_reproduces_cont_did(contdid_data, control_group):
    spec = {
        "yname": "Y",
        "gname": "G",
        "tname": "period",
        "idname": "id",
        "dname": "D",
        "control_group": control_group,
        "degree": 3,
        "num_knots": 1,
        "cband": False,
        "biters": 20,
        "random_state": 7,
    }

    expected = cont_did(data=contdid_data, **spec)
    result = pte(
        data=contdid_data,
        setup_pte_fun=setup_pte_cont,
        subset_fun=cont_two_by_two_subset,
        attgt_fun=cont_did_acrt,
        gt_type="dose",
        aggregation="dose",
        **spec,
    )

    assert result.pte_params.control_group == control_group
    np.testing.assert_array_equal(np.unique(result.pte_params.data["G"]), [0, 2, 3, 4])
    np.testing.assert_array_equal(np.unique(expected.pte_params.data["G"]), [0, 2, 3, 4])
    assert np.isfinite(result.overall_att)
    assert result.overall_att == pytest.approx(expected.overall_att, rel=1e-12)
    assert result.overall_acrt == pytest.approx(expected.overall_acrt, rel=1e-12)
    assert result.overall_att_se == pytest.approx(expected.overall_att_se, rel=1e-12)
    np.testing.assert_allclose(result.att_d, expected.att_d, rtol=1e-12)
    np.testing.assert_allclose(result.acrt_d, expected.acrt_d, rtol=1e-12)
    np.testing.assert_allclose(result.att_d_se, expected.att_d_se, rtol=1e-12)
    np.testing.assert_allclose(result.att_d_inf_func, expected.att_d_inf_func, rtol=1e-12, atol=1e-14)
    assert result.att_d_crit_val == pytest.approx(expected.att_d_crit_val, rel=1e-12)


def test_check_cell_estimates_names_every_cohort_without_a_curve(pte_params_basic):
    ptep = pte_params_basic._replace(gt_type="dose", degree=3, num_knots=2)
    res = {
        "attgt_list": [
            {"att": 0.1, "group": 2004, "time_period": 2004},
            {"att": np.nan, "group": 2006, "time_period": 2006},
            {"att": np.nan, "group": 2007, "time_period": 2005},
        ]
    }
    message = (
        "The dose curve can't be estimated for cohorts 2006 and 2007 with degree=3 and num_knots=2. A cohort's "
        "spline needs at least 4 distinct positive doses plus one for each interior knot inside its dose range. The "
        "doses must also spread across those knots. Their cells are left out of the event study. Lower degree or "
        "num_knots to keep them."
    )

    with pytest.warns(UserWarning, match=re.escape(message)):
        _check_cell_estimates(res, ptep, "dose", "eventstudy")


def test_check_cell_estimates_warns_for_group_time_effects_without_estimates(pte_params_basic):
    res = {
        "attgt_list": [
            {"att": 0.1, "group": 2004, "time_period": 2004},
            {"att": np.nan, "group": 2006, "time_period": 2006},
        ]
    }
    message = "Some group-time effects of cohort 2006 have no estimate. Those cells are left out of the aggregation."

    with pytest.warns(UserWarning, match=re.escape(message)):
        _check_cell_estimates(res, pte_params_basic, "att", "eventstudy")


def test_check_cell_estimates_raises_without_an_estimate_after_treatment_starts(pte_params_basic):
    res = {
        "attgt_list": [
            {"att": 0.2, "group": 2006, "time_period": 2004},
            {"att": np.nan, "group": 2006, "time_period": 2006},
        ]
    }

    with pytest.raises(ValueError, match="No group-time effect after treatment starts is left to aggregate"):
        _check_cell_estimates(res, pte_params_basic, "att", "eventstudy")


def test_check_cell_estimates_advises_a_smaller_spline_only_for_cohorts_with_several_doses(pte_params_basic):
    ptep = pte_params_basic._replace(gt_type="dose", degree=3, num_knots=0)
    res = {
        "attgt_list": [
            {"att": 0.1, "group": 2004, "time_period": 2006},
            {"att": np.nan, "group": 2006, "time_period": 2006},
            {"att": np.nan, "group": 2007, "time_period": 2007},
        ],
        "extra_gt_returns": [
            {"extra_gt_returns": {"att_overall": 0.1}, "group": 2004, "time_period": 2006},
            {"extra_gt_returns": {"distinct_doses": 1, "coefficients": 4}, "group": 2006, "time_period": 2006},
            {"extra_gt_returns": {"distinct_doses": 3, "coefficients": 4}, "group": 2007, "time_period": 2007},
        ],
    }
    message = (
        "The dose curve can't be estimated for cohorts 2006 and 2007 with degree=3 and num_knots=0. A cohort needs at "
        "least 4 distinct positive doses. Every treated unit of cohort 2006 takes the same dose. Their cells are left "
        "out of the event study. Lower degree or num_knots to keep cohort 2007. Pass target_parameter='level' for an "
        "event study that keeps cohort 2006."
    )

    with pytest.warns(UserWarning, match=re.escape(message)):
        _check_cell_estimates(res, ptep, "dose", "eventstudy")


def test_check_cell_estimates_counts_nearly_coinciding_doses_as_one(pte_params_basic):
    ptep = pte_params_basic._replace(gt_type="dose", degree=2, num_knots=1)
    res = {
        "attgt_list": [{"att": np.nan, "group": 2006, "time_period": 2006}],
        "extra_gt_returns": [
            {"extra_gt_returns": {"distinct_doses": 5, "coefficients": 4}, "group": 2006, "time_period": 2006}
        ],
    }
    message = (
        "The dose curve can't be estimated for cohort 2006 with degree=2 and num_knots=1. A cohort's spline needs at "
        "least 3 distinct positive doses plus one for each interior knot inside its dose range. The doses must also "
        "spread across those knots. Doses that nearly coincide count as one. No cohort is left to aggregate. Lower "
        "degree or num_knots."
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        _check_cell_estimates(res, ptep, "dose", "dose")
