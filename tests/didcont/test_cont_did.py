"""Tests for continuous treatment difference-in-differences estimation."""

import re

import numpy as np
import pytest
from scipy import stats

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid import aggte, att_gt
from moderndid.didcont.cont_did import (
    cont_did,
    cont_did_acrt,
    cont_two_by_two_subset,
)
from moderndid.didcont.estimation import DoseResult, GroupTimeATTResult, PTEAggteResult, PTEResult
from moderndid.npiv import npiv


def test_cont_did_basic(contdid_data):
    result = cont_did(
        data=contdid_data,
        yname="Y",
        tname="period",
        idname="id",
        gname="G",
        dname="D",
        xformula="~1",
        target_parameter="level",
        aggregation="dose",
        treatment_type="continuous",
        dose_est_method="parametric",
        degree=2,
        num_knots=0,
        biters=10,
    )

    assert isinstance(result, DoseResult | PTEResult)
    assert np.isfinite(result.overall_att)
    assert np.isfinite(result.overall_att_se)
    assert result.overall_att_se > 0
    assert result.dose is not None
    assert len(result.dose) > 0
    assert result.att_d is not None
    assert len(result.att_d) == len(result.dose)
    assert np.all(np.isfinite(result.att_d))


def test_cont_did_value_validation(contdid_data):
    result = cont_did(
        data=contdid_data,
        yname="Y",
        tname="period",
        idname="id",
        gname="G",
        dname="D",
        target_parameter="level",
        aggregation="dose",
        degree=2,
        num_knots=0,
        biters=10,
    )

    assert isinstance(result, DoseResult)

    assert np.isfinite(result.overall_att)
    assert np.isfinite(result.overall_att_se)
    assert result.overall_att_se > 0

    assert len(result.dose) == len(result.att_d)
    assert len(result.dose) == len(result.att_d_se)

    assert np.all(np.isfinite(result.att_d))
    assert np.all(np.isfinite(result.att_d_se))
    assert np.all(result.att_d_se >= 0)

    assert np.all(np.diff(result.dose) >= 0)

    assert result.att_d_crit_val is not None
    assert np.isfinite(result.att_d_crit_val)
    assert result.att_d_crit_val > 0
    assert result.att_d_crit_val < 10


def test_cont_did_slope_parameter(contdid_data):
    result = cont_did(
        data=contdid_data,
        yname="Y",
        tname="period",
        idname="id",
        gname="G",
        dname="D",
        target_parameter="slope",
        aggregation="dose",
        degree=2,
        num_knots=0,
        biters=10,
    )

    assert isinstance(result, DoseResult | PTEResult)
    assert np.isfinite(result.overall_acrt)
    assert np.isfinite(result.overall_acrt_se)
    assert result.overall_acrt_se >= 0
    assert result.acrt_d is not None
    assert len(result.acrt_d) > 0
    assert np.all(np.isfinite(result.acrt_d))


def test_cont_did_event_study(contdid_data):
    result = cont_did(
        data=contdid_data,
        yname="Y",
        tname="period",
        idname="id",
        gname="G",
        dname="D",
        target_parameter="level",
        aggregation="eventstudy",
        biters=10,
    )

    assert isinstance(result, PTEResult)
    assert hasattr(result, "overall_att")
    assert result.overall_att is not None
    assert np.isfinite(result.overall_att.overall_att)
    assert np.isfinite(result.overall_att.overall_se)
    assert result.overall_att.overall_se > 0
    assert hasattr(result, "event_study") or hasattr(result, "att")


def test_cont_did_custom_dvals(contdid_data):
    custom_dvals = np.linspace(0.1, 0.9, 10)

    result = cont_did(
        data=contdid_data,
        yname="Y",
        tname="period",
        idname="id",
        gname="G",
        dname="D",
        dvals=custom_dvals,
        degree=2,
        num_knots=1,
        biters=10,
    )

    assert isinstance(result, DoseResult | PTEResult)
    if hasattr(result, "dose"):
        assert len(result.dose) == len(custom_dvals)
        assert np.allclose(result.dose, custom_dvals)


@pytest.mark.parametrize("control_group", ["notyettreated", "nevertreated"])
def test_cont_did_control_groups(contdid_data, control_group):
    result = cont_did(
        data=contdid_data,
        yname="Y",
        tname="period",
        idname="id",
        gname="G",
        dname="D",
        control_group=control_group,
        biters=10,
    )

    assert isinstance(result, DoseResult | PTEResult)
    assert np.isfinite(result.overall_att)
    assert np.isfinite(result.overall_att_se)
    assert result.overall_att_se > 0
    if isinstance(result, DoseResult):
        assert np.all(np.isfinite(result.att_d))
        assert np.all(np.isfinite(result.att_d_se))
        assert np.all(result.att_d_se >= 0)


def test_cont_did_base_period(contdid_data):
    result = cont_did(
        data=contdid_data,
        yname="Y",
        tname="period",
        idname="id",
        gname="G",
        dname="D",
        base_period="varying",
        biters=10,
    )

    assert isinstance(result, DoseResult | PTEResult)
    assert np.isfinite(result.overall_att)
    if isinstance(result, DoseResult):
        assert np.isfinite(result.overall_att_se)
        assert result.overall_att_se > 0
        assert np.all(np.isfinite(result.att_d))
        assert np.all(np.isfinite(result.att_d_se))
        assert np.all(result.att_d_se >= 0)


@pytest.mark.parametrize("boot_type", ["multiplier", "empirical"])
def test_cont_did_bootstrap_types(contdid_data, boot_type):
    result = cont_did(
        data=contdid_data,
        yname="Y",
        tname="period",
        idname="id",
        gname="G",
        dname="D",
        aggregation="eventstudy",
        boot_type=boot_type,
        biters=10,
    )

    assert isinstance(result, PTEResult)
    assert np.isfinite(result.overall_att.overall_att)
    assert np.isfinite(result.overall_att.overall_se)
    assert result.overall_att.overall_se > 0


@pytest.mark.parametrize("dose_est_method", ["parametric", "cck"])
def test_cont_did_empirical_bootstrap_rejects_dose_aggregation(contdid_data, dose_est_method):
    with pytest.raises(ValueError, match="boot_type='empirical' needs aggregation='eventstudy'"):
        cont_did(
            data=contdid_data,
            yname="Y",
            tname="period",
            idname="id",
            gname="G",
            dname="D",
            dose_est_method=dose_est_method,
            boot_type="empirical",
            biters=10,
        )


@pytest.mark.parametrize(("option", "value"), [("min_e", -1), ("max_e", 1), ("balance_e", 1)])
def test_cont_did_empirical_bootstrap_rejects_event_time_options(contdid_data, option, value):
    with pytest.raises(ValueError, match=f"The empirical bootstrap doesn't support {option}\\."):
        cont_did(
            data=contdid_data,
            yname="Y",
            tname="period",
            idname="id",
            gname="G",
            dname="D",
            aggregation="eventstudy",
            boot_type="empirical",
            biters=10,
            **{option: value},
        )


@pytest.mark.parametrize("cband", [False, True])
def test_cont_did_confidence_bands(contdid_data, cband):
    result = cont_did(
        data=contdid_data,
        yname="Y",
        tname="period",
        idname="id",
        gname="G",
        dname="D",
        cband=cband,
        biters=100,
    )

    assert isinstance(result, DoseResult | PTEResult)
    assert np.isfinite(result.overall_att)
    if isinstance(result, DoseResult):
        assert hasattr(result, "att_d_crit_val")
        assert result.att_d_crit_val is not None
        assert np.isfinite(result.att_d_crit_val)
        assert result.att_d_crit_val > 0
        assert result.att_d_crit_val < 10


@pytest.mark.parametrize("alp", [0.05, 0.10])
def test_cont_did_significance_level(contdid_data, alp):
    result = cont_did(
        data=contdid_data,
        yname="Y",
        tname="period",
        idname="id",
        gname="G",
        dname="D",
        alp=alp,
        biters=10,
    )

    assert isinstance(result, DoseResult | PTEResult)
    assert np.isfinite(result.overall_att)
    if isinstance(result, DoseResult):
        assert hasattr(result, "att_d_crit_val")
        assert result.att_d_crit_val is not None
        assert np.isfinite(result.att_d_crit_val)
        assert result.att_d_crit_val > 0
        assert result.att_d_crit_val < 10


@pytest.mark.parametrize("target_parameter", ["level", "slope"])
def test_cont_did_empirical_bootstrap_event_study(contdid_data, target_parameter):
    kwargs = {
        "yname": "Y",
        "tname": "period",
        "idname": "id",
        "gname": "G",
        "dname": "D",
        "target_parameter": target_parameter,
        "aggregation": "eventstudy",
        "biters": 10,
        "random_state": 1,
    }
    result = cont_did(data=contdid_data, boot_type="empirical", **kwargs)
    multiplier = cont_did(data=contdid_data, **kwargs)
    es = result.event_study

    assert isinstance(result, PTEResult)
    assert isinstance(es, PTEAggteResult)
    assert isinstance(result.att_gt, GroupTimeATTResult)
    np.testing.assert_array_equal(es.event_times, multiplier.event_study.event_times)
    np.testing.assert_allclose(es.att_by_event, multiplier.event_study.att_by_event, atol=1e-12)
    assert np.all(np.isfinite(es.se_by_event))
    assert np.all(es.se_by_event > 0)
    assert np.all(np.isfinite(result.att_gt.se))
    assert result.overall_att.overall_att == pytest.approx(np.mean(es.att_by_event[es.event_times >= 0]))
    assert np.isfinite(result.overall_att.overall_se)
    assert "Event time Effects" in str(result)


def test_cont_did_invalid_data():
    with pytest.raises(TypeError, match="__arrow_c_stream__"):
        cont_did(
            data=np.array([1, 2, 3]),
            yname="Y",
            tname="time",
            idname="id",
            dname="D",
        )


def test_cont_did_missing_columns():
    df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    message = (
        "yname='Y' is not a column in the data. Did you mean 'y'?\n"
        "tname='time' is not a column in the data.\n"
        "idname='id' is not a column in the data.\n"
        "dname='D' is not a column in the data."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        cont_did(
            data=df,
            yname="Y",
            tname="time",
            idname="id",
            dname="D",
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"gname": "GG"}, "gname='GG' is not a column in the data. Did you mean 'G'?"),
        ({"gname": None, "dname": "dose"}, "dname='dose' is not a column in the data."),
    ],
)
def test_cont_did_names_misspelled_columns(contdid_data, changes, message):
    spec = {"yname": "Y", "tname": "period", "idname": "id", "gname": "G", "dname": "D"} | changes

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        cont_did(contdid_data, **spec)


@pytest.mark.parametrize("contdid_one_missing_dose", [None, float("nan"), float("-inf")], indirect=True)
def test_cont_did_drops_a_missing_dose_like_any_missing_value(contdid_one_missing_dose):
    spec = {
        "yname": "Y",
        "tname": "period",
        "idname": "id",
        "gname": "G",
        "dname": "D",
        "biters": 99,
        "random_state": 3,
    }
    unit = contdid_one_missing_dose.filter(~pl.col("D").is_finite().fill_null(False))["id"].item()
    expected = cont_did(contdid_one_missing_dose.filter(pl.col("id") != unit), **spec)

    with pytest.warns(UserWarning) as record:
        result = cont_did(contdid_one_missing_dose, **spec)

    assert [str(warning.message) for warning in record] == [
        "Dropped 1 rows from original data due to missing values",
        "Dropped 1 units while converting to balanced panel",
    ]
    np.testing.assert_array_equal(result.att_d, expected.att_d)
    np.testing.assert_array_equal(result.att_d_se, expected.att_d_se)
    assert result.overall_att == expected.overall_att
    assert result.overall_acrt == expected.overall_acrt


def test_cont_did_covariates_not_supported(contdid_data):
    with pytest.raises(NotImplementedError, match="Covariates not currently supported"):
        cont_did(
            data=contdid_data,
            yname="Y",
            tname="period",
            idname="id",
            gname="G",
            dname="D",
            xformla="~x1+x2",
        )


def test_cont_did_discrete_treatment_not_supported(contdid_data):
    with pytest.raises(NotImplementedError, match="Discrete treatment not yet supported"):
        cont_did(
            data=contdid_data,
            yname="Y",
            tname="period",
            idname="id",
            gname="G",
            dname="D",
            treatment_type="discrete",
        )


def test_cont_did_unbalanced_panel_not_supported(contdid_data):
    with pytest.raises(NotImplementedError, match="Unbalanced panel not currently supported"):
        cont_did(
            data=contdid_data,
            yname="Y",
            tname="period",
            idname="id",
            gname="G",
            dname="D",
            allow_unbalanced_panel=True,
        )


@pytest.mark.filterwarnings("ignore:Simultaneous confidence band:UserWarning")
@pytest.mark.filterwarnings("ignore:Not returning pre-test Wald statistic:UserWarning")
def test_cont_did_clustering_warning(contdid_data):
    with pytest.warns(UserWarning, match="Two-way clustering not currently supported"):
        cont_did(
            data=contdid_data,
            yname="Y",
            tname="period",
            idname="id",
            gname="G",
            dname="D",
            clustervars="state",
            biters=10,
        )


@pytest.mark.filterwarnings("ignore:Dropped 1 groups treated before period:UserWarning")
@pytest.mark.filterwarnings("ignore:anticipation = 1:UserWarning")
@pytest.mark.filterwarnings("ignore:Dropped .* units that were already treated:UserWarning")
def test_cont_did_anticipation_controls_match_att_gt(contdid_data):
    kwargs = {"yname": "Y", "tname": "period", "idname": "id", "gname": "G"}
    result = cont_did(
        data=contdid_data, dname="D", aggregation="eventstudy", anticipation=1, biters=10, random_state=0, **kwargs
    )
    dose = cont_did(data=contdid_data, dname="D", anticipation=1, degree=2, biters=10, random_state=0, **kwargs)
    binary = att_gt(data=contdid_data, anticipation=1, control_group="notyettreated", boot=False, cband=False, **kwargs)

    observed = dict(zip(zip(result.att_gt.groups.tolist(), result.att_gt.times.tolist()), result.att_gt.att))
    expected = dict(zip(zip(binary.groups.tolist(), binary.times.tolist()), binary.att_gt))
    assert observed.keys() == expected.keys()
    np.testing.assert_allclose([observed[cell] for cell in expected], list(expected.values()), atol=1e-10)
    assert dose.overall_att == pytest.approx(aggte(binary, type="group").overall_att, abs=1e-10)


def test_cont_did_weights_not_supported(contdid_data):
    contdid_data = contdid_data.with_columns(pl.lit(1.0).alias("weights"))

    with pytest.raises(NotImplementedError, match="Sampling weights are not supported"):
        cont_did(
            data=contdid_data,
            yname="Y",
            tname="period",
            idname="id",
            gname="G",
            dname="D",
            weightsname="weights",
            biters=10,
        )


def test_cont_did_acrt_basic(simple_panel_data):
    result = cont_did_acrt(
        gt_data=simple_panel_data,
        dvals=np.linspace(0.1, 0.9, 10),
        degree=2,
        knots=[],
    )

    assert hasattr(result, "attgt")
    assert hasattr(result, "inf_func")
    assert hasattr(result, "extra_gt_returns")
    assert np.isfinite(result.attgt)
    assert result.inf_func is not None
    assert result.inf_func.shape[0] > 0
    assert np.all(np.isfinite(result.inf_func))


def test_cont_did_acrt_with_knots(simple_panel_data):
    result = cont_did_acrt(
        gt_data=simple_panel_data,
        dvals=np.linspace(0.1, 0.9, 10),
        degree=3,
        knots=[0.3, 0.7],
    )

    assert np.isfinite(result.attgt)
    assert result.inf_func is not None
    assert np.all(np.isfinite(result.inf_func))
    if result.extra_gt_returns:
        assert "att_d" in result.extra_gt_returns
        assert "acrt_d" in result.extra_gt_returns
        if result.extra_gt_returns["att_d"] is not None:
            assert np.all(np.isfinite(result.extra_gt_returns["att_d"]))
        if result.extra_gt_returns["acrt_d"] is not None:
            assert np.all(np.isfinite(result.extra_gt_returns["acrt_d"]))


def test_cont_did_acrt_auto_dvals(simple_panel_data):
    result = cont_did_acrt(
        gt_data=simple_panel_data,
        dvals=None,
        degree=2,
    )

    assert np.isfinite(result.attgt)
    assert result.inf_func is not None
    assert np.all(np.isfinite(result.inf_func))


def test_cont_did_acrt_no_treated(simple_panel_data):
    simple_panel_data = simple_panel_data.with_columns(pl.lit(0).alias("D"))

    result = cont_did_acrt(
        gt_data=simple_panel_data,
        degree=2,
    )

    assert result.attgt == 0.0
    assert np.all(result.inf_func == 0)


def test_cont_two_by_two_subset_notyettreated(contdid_data):
    result = cont_two_by_two_subset(
        data=contdid_data,
        g=2,
        tp=3,
        control_group="notyettreated",
        anticipation=0,
        base_period="varying",
        gname="G",
        tname="period",
        idname="id",
        dname="D",
    )

    assert "gt_data" in result
    assert "n1" in result
    assert "disidx" in result
    assert isinstance(result["gt_data"], pl.DataFrame)
    assert result["n1"] > 0
    assert isinstance(result["disidx"], np.ndarray)


def test_cont_two_by_two_subset_nevertreated(contdid_data):
    result = cont_two_by_two_subset(
        data=contdid_data,
        g=2,
        tp=3,
        control_group="nevertreated",
        anticipation=0,
        base_period="varying",
        gname="G",
        tname="time_period",
        idname="id",
        dname="D",
    )

    assert "gt_data" in result
    assert isinstance(result["gt_data"], pl.DataFrame)
    assert "name" in result["gt_data"].columns
    assert "D" in result["gt_data"].columns


def test_cont_two_by_two_subset_anticipation(contdid_data):
    coded = contdid_data.with_columns(pl.when(pl.col("G") == 0).then(np.inf).otherwise(pl.col("G")).alias("G"))
    result = cont_two_by_two_subset(
        data=coded,
        g=4,
        tp=2,
        control_group="notyettreated",
        anticipation=1,
        base_period="varying",
    )

    assert sorted(result["gt_data"]["G"].unique().to_list()) == [4.0, np.inf]
    assert sorted(result["gt_data"]["period"].unique().to_list()) == [1, 2]


def test_cont_two_by_two_subset_universal_base_pre_period_controls(contdid_data):
    coded = contdid_data.with_columns(pl.when(pl.col("G") == 0).then(np.inf).otherwise(pl.col("G")).alias("G"))
    result = cont_two_by_two_subset(
        data=coded,
        g=4,
        tp=1,
        control_group="notyettreated",
        anticipation=0,
        base_period="universal",
    )

    assert sorted(result["gt_data"]["G"].unique().to_list()) == [4.0, np.inf]
    assert sorted(result["gt_data"]["period"].unique().to_list()) == [1, 3]


def test_cont_two_by_two_subset_universal_base(contdid_data):
    result = cont_two_by_two_subset(
        data=contdid_data,
        g=2,
        tp=3,
        control_group="notyettreated",
        anticipation=0,
        base_period="universal",
        gname="G",
        tname="time_period",
        idname="id",
        dname="D",
    )

    assert "gt_data" in result
    gt_data = result["gt_data"]
    assert set(gt_data["name"].unique().to_list()) == {"pre", "post"}


@pytest.mark.filterwarnings("ignore:Using x_grid as x_eval:UserWarning")
@pytest.mark.filterwarnings("ignore:No pre-treatment periods to test:UserWarning")
@pytest.mark.filterwarnings("ignore:Simultaneous band smaller than pointwise:UserWarning")
def test_cck_estimator_basic(cck_test_data):
    result = cont_did(
        data=cck_test_data,
        yname="y",
        tname="time",
        idname="id",
        gname="g",
        dname="d",
        dose_est_method="cck",
        alp=0.05,
        cband=False,
        target_parameter="level",
    )

    assert isinstance(result, DoseResult)
    assert result.att_d is not None
    assert len(result.att_d) > 0
    assert np.all(np.isfinite(result.att_d))
    assert result.dose is not None
    assert len(result.dose) == len(result.att_d)
    assert np.all(np.diff(result.dose) >= 0)
    assert np.isfinite(result.overall_acrt)
    assert result.overall_acrt_se > 0


@pytest.mark.filterwarnings("ignore:Using x_grid as x_eval:UserWarning")
@pytest.mark.filterwarnings("ignore:No pre-treatment periods to test:UserWarning")
@pytest.mark.filterwarnings("ignore:Simultaneous band smaller than pointwise:UserWarning")
def test_cck_estimator_custom_dvals(cck_test_data):
    custom_dvals = np.linspace(0.1, 1.9, 20)

    result = cont_did(
        data=cck_test_data,
        yname="y",
        tname="time",
        idname="id",
        gname="g",
        dname="d",
        dose_est_method="cck",
        dvals=custom_dvals,
        alp=0.05,
        cband=True,
        target_parameter="slope",
    )

    assert isinstance(result, DoseResult)
    assert len(result.dose) == len(custom_dvals)
    assert np.allclose(result.dose, custom_dvals)
    assert np.isfinite(result.overall_acrt)
    assert result.acrt_d is not None
    assert len(result.acrt_d) == len(custom_dvals)
    assert np.all(np.isfinite(result.acrt_d))


@pytest.mark.filterwarnings("ignore:Be aware that there are some small groups:UserWarning")
def test_cck_estimator_invalid_groups():
    data = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "time": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "d": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "g": [0, 0, 0, 2, 2, 2, 3, 3, 3],
        }
    )

    with pytest.raises(ValueError, match=r"CCK estimator requires exactly 2 groups and 2 time periods"):
        cont_did(
            data=data,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            dname="d",
            dose_est_method="cck",
            alp=0.05,
            cband=False,
            target_parameter="level",
        )


@pytest.mark.filterwarnings("ignore:Be aware that there are some small groups:UserWarning")
def test_cck_estimator_invalid_times():
    data = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "time": [1, 2, 3, 1, 2, 3],
            "y": [1, 2, 3, 4, 5, 6],
            "d": [0, 0, 0, 1, 1, 1],
            "g": [0, 0, 0, 2, 2, 2],
        }
    )

    with pytest.raises(ValueError, match=r"CCK estimator requires exactly 2 groups and 2 time periods"):
        cont_did(
            data=data,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            dname="d",
            dose_est_method="cck",
            alp=0.05,
            cband=False,
            target_parameter="level",
        )


@pytest.mark.filterwarnings("ignore:Dropped .* post-treatment observations:UserWarning")
@pytest.mark.filterwarnings("ignore:Dropped 2 units while converting to balanced panel:UserWarning")
def test_cck_estimator_no_treated():
    data = pl.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3, 4, 4],
            "time": [1, 2, 1, 2, 1, 2, 1, 2],
            "y": [1, 2, 3, 4, 5, 6, 7, 8],
            "d": [0, 0, 0, 0, 0, 0, 0, 0],
            "g": [0, 0, 0, 0, 2, 2, 2, 2],
        }
    )

    with pytest.raises(ValueError, match="No valid groups"):
        cont_did(
            data=data,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            dname="d",
            dose_est_method="cck",
            alp=0.05,
            cband=False,
            target_parameter="level",
        )


@pytest.mark.filterwarnings("ignore:Using x_grid as x_eval:UserWarning")
@pytest.mark.filterwarnings("ignore:No pre-treatment periods to test:UserWarning")
@pytest.mark.filterwarnings("ignore:Simultaneous band smaller than pointwise:UserWarning")
def test_cont_did_cck_method(cck_test_data):
    result = cont_did(
        data=cck_test_data,
        yname="y",
        tname="time",
        idname="id",
        gname="g",
        dname="d",
        dose_est_method="cck",
        aggregation="dose",
    )

    assert isinstance(result, DoseResult)
    assert result.att_d is not None
    assert len(result.att_d) > 0
    assert np.all(np.isfinite(result.att_d))
    assert np.isfinite(result.overall_acrt)


def test_cont_did_cck_invalid_aggregation(cck_test_data):
    with pytest.raises(ValueError, match="Event study not supported with CCK estimator"):
        cont_did(
            data=cck_test_data,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            dname="d",
            dose_est_method="cck",
            aggregation="eventstudy",
        )


def test_cont_did_invalid_parameter_combination(contdid_data):
    with pytest.raises(ValueError, match="target_parameter='invalid' is not valid"):
        cont_did(
            data=contdid_data,
            yname="Y",
            tname="period",
            idname="id",
            gname="G",
            dname="D",
            target_parameter="invalid",
            aggregation="dose",
            treatment_type="continuous",
        )


@pytest.mark.parametrize("degree,num_knots", [(1, 0), (2, 0), (3, 0)])
def test_cont_did_various_degree_knot_combinations(contdid_data, degree, num_knots):
    result = cont_did(
        data=contdid_data,
        yname="Y",
        tname="period",
        idname="id",
        gname="G",
        dname="D",
        degree=degree,
        num_knots=num_knots,
        biters=10,
    )
    assert isinstance(result, DoseResult | PTEResult)
    assert np.isfinite(result.overall_att)
    if isinstance(result, DoseResult):
        assert np.isfinite(result.overall_att_se)
        assert result.overall_att_se > 0
        assert np.all(np.isfinite(result.att_d))
        assert np.all(np.isfinite(result.att_d_se))
        assert np.all(result.att_d_se >= 0)


@pytest.mark.parametrize("contdid_converted", ["pandas", "pyarrow", "duckdb"], indirect=True)
def test_cont_did_dataframe_interoperability(contdid_converted, cont_did_baseline_result):
    result = cont_did(
        data=contdid_converted,
        yname="Y",
        tname="period",
        idname="id",
        gname="G",
        dname="D",
        target_parameter="level",
        aggregation="dose",
        degree=2,
        num_knots=0,
        biters=10,
        random_state=42,
    )

    assert np.isclose(result.overall_att, cont_did_baseline_result.overall_att)
    assert np.isclose(result.overall_att_se, cont_did_baseline_result.overall_att_se)
    np.testing.assert_array_almost_equal(result.dose, cont_did_baseline_result.dose)
    np.testing.assert_array_almost_equal(result.att_d, cont_did_baseline_result.att_d)


def test_cont_did_overall_att_inf_func_is_binary_did(contdid_two_period_data):
    result = cont_did(
        data=contdid_two_period_data,
        yname="Y",
        tname="period",
        idname="id",
        gname="G",
        dname="D",
        degree=3,
        num_knots=0,
        biters=10,
        random_state=0,
    )

    wide = contdid_two_period_data.pivot(index=["id", "G", "D"], on="period", values="Y").sort("id")
    dy = (wide["2"] - wide["1"]).to_numpy()
    treated = wide["G"].to_numpy() > 0
    n = len(dy)
    expected = np.where(
        treated,
        n / treated.sum() * (dy - dy[treated].mean()),
        -n / (~treated).sum() * (dy - dy[~treated].mean()),
    )

    np.testing.assert_allclose(result.overall_att_inf_func, expected, atol=1e-10)
    np.testing.assert_allclose(result.overall_att, dy[treated].mean() - dy[~treated].mean(), atol=1e-12)


def test_cont_did_dose_inf_funcs_use_treated_count_and_comparison_mean(contdid_two_period_data):
    dvals = np.array([0.25, 0.5, 0.75])
    result = cont_did(
        data=contdid_two_period_data,
        yname="Y",
        tname="period",
        idname="id",
        gname="G",
        dname="D",
        degree=3,
        num_knots=0,
        dvals=dvals,
        biters=10,
        random_state=0,
    )

    wide = contdid_two_period_data.pivot(index=["id", "G", "D"], on="period", values="Y").sort("id")
    dy = (wide["2"] - wide["1"]).to_numpy()
    treated = wide["G"].to_numpy() > 0
    n, n_treated, n_comparison = len(dy), treated.sum(), (~treated).sum()
    d = wide["D"].to_numpy()[treated]
    x = np.column_stack([np.ones(n_treated), d, d**2, d**3])
    beta = np.linalg.lstsq(x, dy[treated], rcond=None)[0]
    score = (dy[treated] - x @ beta)[:, None] * x
    bread = np.linalg.inv(x.T @ x / n_treated)
    slope = beta[1] + 2 * beta[2] * d + 3 * beta[3] * d**2
    mean_slope_basis = np.column_stack([np.zeros(n_treated), np.ones(n_treated), 2 * d, 3 * d**2]).mean(axis=0)
    grid = np.column_stack([np.ones(3), dvals, dvals**2, dvals**3])

    acrt_expected = np.zeros(n)
    acrt_expected[treated] = n / n_treated * ((slope - slope.mean()) + score @ bread @ mean_slope_basis)
    att_d_expected = np.zeros((n, 3))
    att_d_expected[treated] = n / n_treated * score @ bread @ grid.T
    att_d_expected[~treated] = -n / n_comparison * (dy[~treated] - dy[~treated].mean())[:, None]

    np.testing.assert_allclose(result.overall_acrt_inf_func, acrt_expected, atol=1e-8)
    np.testing.assert_allclose(result.att_d_inf_func, att_d_expected, atol=1e-8)


def test_cont_did_acrt_inf_func_same_under_both_control_groups(contdid_data):
    kwargs = {
        "yname": "Y",
        "tname": "period",
        "idname": "id",
        "gname": "G",
        "dname": "D",
        "degree": 2,
        "num_knots": 0,
        "biters": 10,
        "random_state": 0,
    }
    not_yet = cont_did(data=contdid_data, control_group="notyettreated", **kwargs)
    never = cont_did(data=contdid_data, control_group="nevertreated", **kwargs)

    np.testing.assert_allclose(not_yet.overall_acrt_inf_func, never.overall_acrt_inf_func, atol=1e-12)
    np.testing.assert_allclose(not_yet.acrt_d_inf_func, never.acrt_d_inf_func, atol=1e-12)


def test_cont_did_dose_se_does_not_depend_on_grid(contdid_data):
    kwargs = {
        "yname": "Y",
        "tname": "period",
        "idname": "id",
        "gname": "G",
        "dname": "D",
        "degree": 3,
        "num_knots": 1,
        "biters": 10,
        "random_state": 0,
    }
    wide_grid = cont_did(data=contdid_data, dvals=np.array([0.1, 0.4, 0.9]), **kwargs)
    narrow_grid = cont_did(data=contdid_data, dvals=np.array([0.3, 0.4, 0.45]), **kwargs)
    n = wide_grid.att_d_inf_func.shape[0]
    wide_att_se = np.sqrt(np.sum(wide_grid.att_d_inf_func**2, axis=0)) / n
    narrow_att_se = np.sqrt(np.sum(narrow_grid.att_d_inf_func**2, axis=0)) / n
    wide_acrt_se = np.sqrt(np.sum(wide_grid.acrt_d_inf_func**2, axis=0)) / n
    narrow_acrt_se = np.sqrt(np.sum(narrow_grid.acrt_d_inf_func**2, axis=0)) / n

    np.testing.assert_allclose(wide_grid.att_d[1], narrow_grid.att_d[1], rtol=1e-12)
    np.testing.assert_allclose(wide_att_se[1], narrow_att_se[1], rtol=1e-10)
    np.testing.assert_allclose(wide_acrt_se[1], narrow_acrt_se[1], rtol=1e-10)


def test_cont_did_two_periods_keeps_uniform_band(contdid_two_period_data):
    result = cont_did(
        data=contdid_two_period_data,
        yname="Y",
        tname="period",
        idname="id",
        gname="G",
        dname="D",
        degree=3,
        num_knots=0,
        cband=True,
        biters=200,
        random_state=0,
    )

    assert result.pte_params.cband
    assert result.att_d_crit_val > 1.96
    assert result.acrt_d_crit_val > 1.96


@pytest.mark.parametrize("target_parameter", ["level", "slope"])
def test_cont_did_two_period_event_study_stays_pointwise(contdid_two_period_data, target_parameter):
    result = cont_did(
        data=contdid_two_period_data,
        yname="Y",
        tname="period",
        idname="id",
        gname="G",
        dname="D",
        aggregation="eventstudy",
        target_parameter=target_parameter,
        cband=True,
        biters=100,
        random_state=2,
    )

    assert not result.ptep.cband
    assert result.event_study.critical_value == pytest.approx(1.959964, abs=1e-6)


def test_cont_did_event_study_weight_term_follows_unit_ids(contdid_data):
    result = cont_did(
        data=contdid_data,
        yname="Y",
        tname="period",
        idname="id",
        gname="G",
        dname="D",
        aggregation="eventstudy",
        target_parameter="level",
        biters=10,
        random_state=0,
    )
    es, gt = result.event_study, result.att_gt

    group = contdid_data.filter(pl.col("period") == 1).sort("id")["G"].to_numpy()
    cells = np.flatnonzero(gt.times - gt.groups == 0)
    pg = np.array([np.mean(group == g) for g in gt.groups[cells]])
    indicators = np.column_stack([(group == g).astype(float) for g in gt.groups[cells]])
    weight_term = (indicators - pg) / pg.sum() - np.outer((indicators - pg).sum(axis=1), pg / pg.sum() ** 2)
    expected = gt.influence_func[:, cells] @ (pg / pg.sum()) + weight_term @ gt.att[cells]

    column = int(np.flatnonzero(es.event_times == 0)[0])
    np.testing.assert_allclose(es.influence_func["by_event"][:, column], expected, atol=1e-10)


@pytest.mark.parametrize("target_parameter", ["level", "slope"])
def test_cont_did_event_study_reproducible_with_random_state(contdid_data, target_parameter):
    kwargs = {
        "yname": "Y",
        "tname": "period",
        "idname": "id",
        "gname": "G",
        "dname": "D",
        "aggregation": "eventstudy",
        "target_parameter": target_parameter,
        "cband": True,
        "biters": 50,
        "random_state": 7,
    }
    first = cont_did(data=contdid_data, **kwargs).event_study
    second = cont_did(data=contdid_data, **kwargs).event_study

    np.testing.assert_array_equal(first.se_by_event, second.se_by_event)
    assert first.critical_value == second.critical_value
    assert first.overall_se == second.overall_se


@pytest.mark.filterwarnings("ignore:Using x_grid as x_eval:UserWarning")
@pytest.mark.filterwarnings("ignore:No pre-treatment periods to test:UserWarning")
def test_cck_att_d_band_includes_comparison_mean(cck_test_data):
    result = cont_did(
        data=cck_test_data,
        yname="y",
        tname="time",
        idname="id",
        gname="g",
        dname="d",
        dose_est_method="cck",
        cband=True,
        random_state=3,
    )

    wide = cck_test_data.pivot(index=["id", "g", "d"], on="time", values="y").sort("id")
    dy = (wide["2"] - wide["1"]).to_numpy()
    dose = wide["d"].to_numpy()
    control = dose == 0
    m0 = dy[control].mean()
    se_m0 = np.sqrt(np.sum((dy[control] - m0) ** 2)) / control.sum()
    curve = npiv(
        y=dy[~control] - m0,
        x=dose[~control].reshape(-1, 1),
        w=dose[~control].reshape(-1, 1),
        x_grid=np.asarray(result.dose).reshape(-1, 1),
        alpha=0.05,
        knots="quantiles",
        biters=999,
        j_x_degree=3,
        k_w_degree=3,
        seed=3,
    )

    assert result.pte_params.cband
    np.testing.assert_allclose(result.att_d, curve.h, rtol=1e-12)
    np.testing.assert_allclose(result.att_d_se, np.sqrt(curve.asy_se**2 + se_m0**2), rtol=1e-10)
    assert result.att_d_crit_val == pytest.approx(curve.cv)
    assert result.acrt_d_crit_val == pytest.approx(curve.cv_deriv)


@pytest.mark.filterwarnings("ignore:Using x_grid as x_eval:UserWarning")
@pytest.mark.filterwarnings("ignore:No pre-treatment periods to test:UserWarning")
@pytest.mark.filterwarnings("ignore:Simultaneous band smaller than pointwise:UserWarning")
def test_cck_band_is_never_narrower_than_pointwise_intervals(contdid_two_period_data):
    result = cont_did(
        data=contdid_two_period_data,
        yname="Y",
        tname="period",
        idname="id",
        gname="G",
        dname="D",
        dose_est_method="cck",
        cband=True,
        alp=1e-10,
        random_state=0,
    )
    pointwise = stats.norm.ppf(1 - 1e-10 / 2)

    assert result.att_d_crit_val >= pointwise
    assert result.acrt_d_crit_val >= pointwise


@pytest.mark.filterwarnings("ignore:Using x_grid as x_eval:UserWarning")
@pytest.mark.filterwarnings("ignore:No pre-treatment periods to test:UserWarning")
@pytest.mark.filterwarnings("ignore:Simultaneous band smaller than pointwise:UserWarning")
def test_cck_overall_att_se_reproducible_with_random_state(cck_test_data):
    kwargs = {
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "g",
        "dname": "d",
        "dose_est_method": "cck",
        "biters": 50,
        "random_state": 5,
    }
    first = cont_did(data=cck_test_data, **kwargs)
    second = cont_did(data=cck_test_data, **kwargs)

    assert first.overall_att_se == second.overall_att_se
    np.testing.assert_array_equal(first.att_d_se, second.att_d_se)


@pytest.mark.parametrize("control_group", ["notyettreated", "nevertreated"])
def test_cont_did_level_event_study_matches_binary_att_gt(contdid_data, control_group):
    kwargs = {"yname": "Y", "tname": "period", "idname": "id", "gname": "G"}
    result = cont_did(
        data=contdid_data,
        dname="D",
        aggregation="eventstudy",
        target_parameter="level",
        control_group=control_group,
        degree=3,
        num_knots=1,
        biters=10,
        random_state=0,
        **kwargs,
    )
    binary = aggte(att_gt(data=contdid_data, control_group=control_group, boot=False, cband=False, **kwargs), "dynamic")
    es = result.event_study
    inf_func = es.influence_func["by_event"]

    np.testing.assert_array_equal(es.event_times, binary.event_times)
    np.testing.assert_allclose(es.att_by_event, binary.att_by_event, atol=1e-12)
    np.testing.assert_allclose(np.sqrt(np.sum(inf_func**2, axis=0)) / inf_func.shape[0], binary.se_by_event, rtol=1e-8)


@pytest.mark.parametrize("scale, shift", [(1, 2000), (2, 0), (0.5, 2000.5), (0.1, 2001)])
def test_cont_did_results_do_not_depend_on_period_coding(contdid_data, scale, shift):
    recoded = contdid_data.with_columns(
        (pl.col("period") * scale + shift).alias("period"),
        pl.when(pl.col("G") > 0).then(pl.col("G") * scale + shift).otherwise(0).alias("G"),
    )
    kwargs = {
        "yname": "Y",
        "tname": "period",
        "idname": "id",
        "gname": "G",
        "dname": "D",
        "degree": 3,
        "num_knots": 1,
        "biters": 10,
        "random_state": 0,
    }
    base = cont_did(data=contdid_data, **kwargs)
    coded = cont_did(data=recoded, **kwargs)
    base_es = cont_did(data=contdid_data, aggregation="eventstudy", target_parameter="slope", **kwargs)
    coded_es = cont_did(data=recoded, aggregation="eventstudy", target_parameter="slope", **kwargs)

    assert coded.overall_att == pytest.approx(base.overall_att, abs=1e-12)
    assert coded.overall_acrt == pytest.approx(base.overall_acrt, abs=1e-12)
    np.testing.assert_allclose(coded.att_d, base.att_d, atol=1e-12)
    np.testing.assert_allclose(coded.att_d_inf_func, base.att_d_inf_func, atol=1e-10)
    np.testing.assert_allclose(coded.overall_acrt_inf_func, base.overall_acrt_inf_func, atol=1e-10)
    np.testing.assert_allclose(coded_es.event_study.event_times, base_es.event_study.event_times * scale, atol=1e-12)
    np.testing.assert_allclose(coded_es.event_study.att_by_event, base_es.event_study.att_by_event, atol=1e-12)
    assert coded_es.event_study.overall_att == pytest.approx(base_es.event_study.overall_att, abs=1e-12)
    np.testing.assert_allclose(coded_es.att_gt.groups, base_es.att_gt.groups * scale + shift, atol=1e-12)
    np.testing.assert_allclose(coded_es.att_gt.times, base_es.att_gt.times * scale + shift, atol=1e-12)


def test_cont_did_fractional_periods_match_att_gt(contdid_data):
    half_years = contdid_data.with_columns(
        (pl.col("period") * 0.5 + 2000.5).alias("period"),
        pl.when(pl.col("G") > 0).then(pl.col("G") * 0.5 + 2000.5).otherwise(0).alias("G"),
    )
    kwargs = {"yname": "Y", "tname": "period", "idname": "id", "gname": "G"}
    es_kwargs = {"dname": "D", "aggregation": "eventstudy", "random_state": 0, **kwargs}
    multiplier = cont_did(data=half_years, biters=10, **es_kwargs).event_study
    empirical = cont_did(data=half_years, boot_type="empirical", biters=5, **es_kwargs).event_study
    binary = aggte(att_gt(data=half_years, control_group="notyettreated", boot=False, cband=False, **kwargs), "dynamic")

    np.testing.assert_array_equal(multiplier.event_times, binary.event_times)
    np.testing.assert_array_equal(empirical.event_times, binary.event_times)
    np.testing.assert_allclose(multiplier.att_by_event, binary.att_by_event, atol=1e-12)
    np.testing.assert_allclose(empirical.att_by_event, binary.att_by_event, atol=1e-12)
    assert multiplier.overall_att == pytest.approx(binary.overall_att, abs=1e-12)
    assert empirical.overall_att == pytest.approx(binary.overall_att, abs=1e-12)


def test_cont_did_rejects_group_between_observed_periods(contdid_data):
    with pytest.raises(ValueError, match="Treatment starts between observed periods for group 3\\."):
        cont_did(
            data=contdid_data.filter(pl.col("period") != 3),
            yname="Y",
            tname="period",
            idname="id",
            gname="G",
            dname="D",
        )


def test_cont_did_universal_base_dose_path_matches_varying(contdid_data):
    kwargs = {
        "yname": "Y",
        "tname": "period",
        "idname": "id",
        "gname": "G",
        "dname": "D",
        "degree": 3,
        "num_knots": 1,
        "biters": 10,
        "random_state": 0,
    }
    varying = cont_did(data=contdid_data, **kwargs)
    universal = cont_did(data=contdid_data, base_period="universal", **kwargs)

    assert np.all(np.isfinite(universal.att_d))
    assert np.isfinite(universal.overall_att_se)
    assert np.isfinite(universal.overall_acrt_se)
    np.testing.assert_allclose(universal.att_d, varying.att_d, atol=1e-12)
    np.testing.assert_allclose(universal.overall_att_inf_func, varying.overall_att_inf_func, atol=1e-10)
    np.testing.assert_allclose(universal.overall_acrt_inf_func, varying.overall_acrt_inf_func, atol=1e-10)
    np.testing.assert_allclose(universal.att_d_inf_func, varying.att_d_inf_func, atol=1e-10)


def test_cont_did_universal_base_event_study_matches_att_gt(contdid_data):
    kwargs = {"yname": "Y", "tname": "period", "idname": "id", "gname": "G"}
    level = cont_did(
        data=contdid_data,
        dname="D",
        aggregation="eventstudy",
        base_period="universal",
        biters=10,
        random_state=0,
        **kwargs,
    )
    slope = cont_did(
        data=contdid_data,
        dname="D",
        aggregation="eventstudy",
        target_parameter="slope",
        base_period="universal",
        biters=10,
        random_state=0,
        **kwargs,
    )
    binary = att_gt(
        data=contdid_data, base_period="universal", control_group="notyettreated", boot=False, cband=False, **kwargs
    )
    dynamic = aggte(binary, type="dynamic")

    np.testing.assert_array_equal(level.event_study.event_times, dynamic.event_times)
    np.testing.assert_allclose(level.event_study.att_by_event, dynamic.att_by_event, atol=1e-12)
    assert level.att_gt.wald_stat == pytest.approx(binary.wald_stat, rel=1e-8)
    assert np.all(np.isfinite(slope.event_study.att_by_event))


def test_cont_did_without_never_treated_units_rejects_never_treated_controls(contdid_data):
    with pytest.raises(ValueError, match="needs never-treated units"):
        cont_did(
            data=contdid_data.filter(pl.col("G") > 0),
            yname="Y",
            tname="period",
            idname="id",
            gname="G",
            dname="D",
            control_group="nevertreated",
        )


@pytest.mark.filterwarnings("ignore:Not returning pre-test Wald statistic:UserWarning")
def test_cont_did_without_never_treated_units_drops_late_periods(contdid_data):
    treated_only = contdid_data.filter(pl.col("G") > 0)
    kwargs = {"yname": "Y", "tname": "period", "idname": "id", "gname": "G"}

    with pytest.warns(UserWarning, match="no unit is untreated from period 4 on"):
        result = cont_did(data=treated_only, dname="D", degree=2, biters=10, random_state=0, **kwargs)
    binary = att_gt(data=treated_only, control_group="notyettreated", boot=False, cband=False, **kwargs)

    assert result.overall_att == pytest.approx(aggte(binary, type="group").overall_att, abs=1e-10)


@pytest.mark.parametrize("aggregation", ["dose", "eventstudy"])
@pytest.mark.parametrize(("cohorts", "anticipation"), [([3], 0), ([3, 4], 1)])
def test_cont_did_without_never_treated_units_needs_a_treated_period(contdid_data, aggregation, cohorts, anticipation):
    with pytest.raises(ValueError, match="no cohort starts treatment before period 3"):
        cont_did(
            data=contdid_data.filter(pl.col("G").is_in(cohorts)),
            yname="Y",
            tname="period",
            idname="id",
            gname="G",
            dname="D",
            aggregation=aggregation,
            anticipation=anticipation,
            num_knots=0,
        )


@pytest.mark.parametrize(
    "target_parameter, method",
    [("level", "Doubly Robust (binarized treatment)"), ("slope", "Parametric (B-spline)")],
)
def test_cont_did_event_study_printout_names_method_and_overall(contdid_data, target_parameter, method):
    result = cont_did(
        data=contdid_data,
        yname="Y",
        tname="period",
        idname="id",
        gname="G",
        dname="D",
        aggregation="eventstudy",
        target_parameter=target_parameter,
        biters=10,
        random_state=0,
    )
    text = str(result)
    es = result.event_study

    assert f"Estimation Method: {method}" in text
    assert "average over event times e >= 0" in text
    assert es.overall_att == pytest.approx(np.mean(es.att_by_event[es.event_times >= 0]))


def test_cont_did_knots_use_unit_doses(contdid_data):
    result = cont_did(
        data=contdid_data,
        yname="Y",
        tname="period",
        idname="id",
        gname="G",
        dname="D",
        degree=3,
        num_knots=2,
        biters=10,
    )
    unit_doses = contdid_data.filter((pl.col("period") == 1) & (pl.col("G") > 0))["D"].to_numpy()

    np.testing.assert_allclose(result.pte_params.knots, np.quantile(unit_doses, [1 / 3, 2 / 3]), rtol=1e-12)


def test_cont_did_rejects_dose_that_changes_over_time(contdid_data):
    drifting = contdid_data.with_columns(
        pl.when(pl.col("period") == 4).then(pl.col("D") + 0.1).otherwise(pl.col("D")).alias("D")
    )

    with pytest.raises(ValueError, match="must stay the same over time"):
        cont_did(data=drifting, yname="Y", tname="period", idname="id", gname="G", dname="D")


def test_cont_did_accepts_dose_recorded_as_zero_before_treatment(contdid_data):
    zero_before = contdid_data.with_columns(
        pl.when(pl.col("period") < pl.col("G")).then(0.0).otherwise(pl.col("D")).alias("D")
    )
    kwargs = {"yname": "Y", "tname": "period", "idname": "id", "gname": "G", "dname": "D", "degree": 2, "biters": 10}
    es_kwargs = {"aggregation": "eventstudy", "target_parameter": "slope", **kwargs}
    recorded = cont_did(data=zero_before, **kwargs)
    constant = cont_did(data=contdid_data, **kwargs)
    recorded_es = cont_did(data=zero_before, **es_kwargs).event_study
    constant_es = cont_did(data=contdid_data, **es_kwargs).event_study

    assert recorded.overall_att == pytest.approx(constant.overall_att, abs=1e-12)
    assert recorded.overall_acrt == pytest.approx(constant.overall_acrt, abs=1e-12)
    np.testing.assert_allclose(recorded.att_d, constant.att_d, atol=1e-12)
    np.testing.assert_allclose(recorded_es.att_by_event, constant_es.att_by_event, atol=1e-12)


def test_cont_did_gname_none_reads_start_of_treatment_from_dose(contdid_data):
    zero_before = contdid_data.with_columns(
        pl.when(pl.col("period") < pl.col("G")).then(0.0).otherwise(pl.col("D")).alias("D")
    )
    kwargs = {"yname": "Y", "tname": "period", "idname": "id", "dname": "D", "degree": 2, "biters": 10}
    inferred = cont_did(data=zero_before.drop("G"), gname=None, **kwargs)
    given = cont_did(data=contdid_data, gname="G", **kwargs)

    assert inferred.overall_att == pytest.approx(given.overall_att, abs=1e-12)
    assert inferred.overall_acrt == pytest.approx(given.overall_acrt, abs=1e-12)
    np.testing.assert_allclose(inferred.att_d, given.att_d, atol=1e-12)


def test_cont_did_gname_none_needs_a_period_before_treatment(contdid_data):
    with pytest.raises(ValueError, match="With gname=None"):
        cont_did(data=contdid_data.drop("G"), yname="Y", tname="period", idname="id", dname="D", gname=None)


def test_cont_did_outcome_named_weights(contdid_data):
    kwargs = {"tname": "period", "idname": "id", "gname": "G", "dname": "D", "degree": 2, "biters": 10}
    renamed = cont_did(data=contdid_data.rename({"Y": "weights"}), yname="weights", random_state=0, **kwargs)
    expected = cont_did(data=contdid_data, yname="Y", random_state=0, **kwargs)

    assert renamed.overall_att == expected.overall_att
    assert renamed.overall_att_se == expected.overall_att_se
    np.testing.assert_array_equal(renamed.att_d, expected.att_d)


@pytest.mark.parametrize("contdid_one_infinite", ["Y", "D"], indirect=True)
def test_cont_did_drops_infinite_rows_like_missing_ones(contdid_one_infinite):
    kwargs = {"yname": "Y", "tname": "period", "idname": "id", "gname": "G", "dname": "D", "degree": 2, "biters": 10}
    expected = cont_did(
        data=contdid_one_infinite.filter(pl.col("Y").is_finite() & pl.col("D").is_finite()), random_state=0, **kwargs
    )

    with pytest.warns(UserWarning, match="^Dropped 1 rows from original data due to missing values$"):
        result = cont_did(data=contdid_one_infinite, random_state=0, **kwargs)

    assert result.overall_att == expected.overall_att
    assert result.overall_att_se == expected.overall_att_se
    np.testing.assert_array_equal(result.att_d, expected.att_d)


def test_cont_did_rejects_reserved_outcome_name(contdid_data):
    with pytest.raises(ValueError, match=re.escape("yname names the column '.w'")):
        cont_did(data=contdid_data.rename({"Y": ".w"}), yname=".w", tname="period", idname="id", gname="G", dname="D")


@pytest.mark.parametrize(
    "renamed",
    [
        {"Y": "outcome", "D": "Y"},
        {"G": "cohort", "D": "G"},
        {"period": "time", "D": "period"},
        {"id": "unit", "D": "id"},
        {"G": "cohort", "id": "G"},
        {"period": "time", "id": "period"},
        {"Y": "outcome", "id": "Y"},
        {"D": "dose", "id": "D"},
    ],
)
def test_cont_did_columns_named_like_its_working_columns_give_the_same_estimates(contdid_data, renamed):
    names = {column: renamed.get(column, column) for column in ("Y", "period", "id", "G", "D")}
    kwargs = {"target_parameter": "level", "aggregation": "dose", "degree": 2, "biters": 10, "random_state": 0}
    expected = cont_did(
        data=contdid_data.rename({"Y": "outcome", "period": "time", "id": "unit", "G": "cohort", "D": "dose"}),
        yname="outcome",
        tname="time",
        idname="unit",
        gname="cohort",
        dname="dose",
        **kwargs,
    )
    result = cont_did(
        data=contdid_data.rename(renamed),
        yname=names["Y"],
        tname=names["period"],
        idname=names["id"],
        gname=names["G"],
        dname=names["D"],
        **kwargs,
    )

    np.testing.assert_allclose(result.overall_att, expected.overall_att, rtol=1e-12)
    np.testing.assert_allclose(result.overall_att_se, expected.overall_att_se, rtol=1e-12)
    np.testing.assert_allclose(result.att_d, expected.att_d, rtol=1e-12)
    np.testing.assert_allclose(result.att_d_se, expected.att_d_se, rtol=1e-12)


def test_cont_did_empirical_bootstrap_with_a_unit_column_named_g(contdid_data):
    kwargs = {
        "yname": "Y",
        "tname": "period",
        "gname": "cohort",
        "dname": "D",
        "target_parameter": "level",
        "aggregation": "eventstudy",
        "boot_type": "empirical",
        "biters": 5,
        "random_state": 0,
    }
    expected = cont_did(data=contdid_data.rename({"G": "cohort", "id": "unit"}), idname="unit", **kwargs)
    result = cont_did(data=contdid_data.rename({"G": "cohort", "id": "G"}), idname="G", **kwargs)

    np.testing.assert_allclose(result.event_study.att_by_event, expected.event_study.att_by_event, rtol=1e-12)
    np.testing.assert_allclose(result.event_study.se_by_event, expected.event_study.se_by_event, rtol=1e-12)
    assert result.att_gt.n_units == expected.att_gt.n_units == 1000


@pytest.mark.parametrize("extra", ["G", ".G", "_group"])
def test_cont_did_without_gname_ignores_columns_named_like_its_group_columns(contdid_staggered_dose, extra):
    kwargs = {"yname": "Y", "tname": "period", "idname": "id", "dname": "D", "gname": None, "degree": 2, "biters": 10}
    expected = cont_did(data=contdid_staggered_dose, random_state=0, **kwargs)
    result = cont_did(data=contdid_staggered_dose.with_columns(pl.col("Y").alias(extra)), random_state=0, **kwargs)

    np.testing.assert_allclose(result.overall_att, expected.overall_att, rtol=1e-12)
    np.testing.assert_allclose(result.overall_att_se, expected.overall_att_se, rtol=1e-12)
    np.testing.assert_allclose(result.att_d, expected.att_d, rtol=1e-12)
    np.testing.assert_allclose(result.att_d_se, expected.att_d_se, rtol=1e-12)


@pytest.mark.parametrize("column", ["Y", "id", "D"])
def test_cont_did_without_gname_keeps_a_named_column_called_g(contdid_staggered_dose, column):
    names = {"Y": "Y", "id": "id", "D": "D", column: "G"}
    kwargs = {"tname": "period", "gname": None, "degree": 2, "biters": 10, "random_state": 0}
    expected = cont_did(data=contdid_staggered_dose, yname="Y", idname="id", dname="D", **kwargs)
    result = cont_did(
        data=contdid_staggered_dose.rename({column: "G"}),
        yname=names["Y"],
        idname=names["id"],
        dname=names["D"],
        **kwargs,
    )

    np.testing.assert_allclose(result.overall_att, expected.overall_att, rtol=1e-12)
    np.testing.assert_allclose(result.overall_att_se, expected.overall_att_se, rtol=1e-12)
    np.testing.assert_allclose(result.att_d, expected.att_d, rtol=1e-12)
    np.testing.assert_allclose(result.att_d_se, expected.att_d_se, rtol=1e-12)


def test_cont_did_without_gname_rejects_a_named_column_called_dot_g(contdid_staggered_dose):
    message = "yname names the column '.G'. Since moderndid uses that name for an internal column, rename the column."

    with pytest.raises(ValueError, match=re.escape(message)):
        cont_did(
            data=contdid_staggered_dose.rename({"Y": ".G"}),
            yname=".G",
            tname="period",
            idname="id",
            dname="D",
            gname=None,
        )


@pytest.mark.filterwarnings("error:.*unbalanced:UserWarning")
def test_cont_did_rejects_repeated_unit_periods(contdid_duplicated):
    message = (
        "The value of idname must be unique (by tname). Some units are observed more than once in a period. "
        "Rows repeat for the (id, period) pair (1, 2)."
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        cont_did(
            data=contdid_duplicated,
            yname="Y",
            tname="period",
            idname="id",
            gname="G",
            dname="D",
            biters=10,
            random_state=0,
        )
