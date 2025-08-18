"""Tests for continuous treatment difference-in-differences estimation."""

import numpy as np
import pandas as pd
import pytest

from moderndid.didcont.cont_did import (
    _cck_estimator,
    cont_did,
    cont_did_acrt,
    cont_two_by_two_subset,
)
from moderndid.didcont.panel import DoseResult, PTEResult


def test_cont_did_basic(contdid_data):
    result = cont_did(
        yname="Y",
        dname="D",
        tname="time_period",
        idname="id",
        data=contdid_data,
        gname="G",
        xformula="~1",
        target_parameter="level",
        aggregation="dose",
        treatment_type="continuous",
        dose_est_method="parametric",
        degree=2,
        num_knots=0,
        biters=100,
    )

    assert isinstance(result, DoseResult | PTEResult)
    assert result.overall_att is not None
    assert result.overall_att_se is not None
    assert result.dose is not None
    assert result.att_d is not None


def test_cont_did_slope_parameter(contdid_data):
    result = cont_did(
        yname="Y",
        dname="D",
        tname="time_period",
        idname="id",
        data=contdid_data,
        gname="G",
        target_parameter="slope",
        aggregation="dose",
        degree=3,
        num_knots=2,
        biters=100,
    )

    assert isinstance(result, DoseResult | PTEResult)
    assert result.overall_acrt is not None
    assert result.overall_acrt_se is not None
    assert result.acrt_d is not None


def test_cont_did_event_study(contdid_data):
    result = cont_did(
        yname="Y",
        dname="D",
        tname="time_period",
        idname="id",
        data=contdid_data,
        gname="G",
        target_parameter="level",
        aggregation="eventstudy",
        biters=100,
    )

    assert isinstance(result, PTEResult)
    assert result.overall_att is not None
    assert hasattr(result, "event_study") or hasattr(result, "att")


def test_cont_did_custom_dvals(contdid_data):
    custom_dvals = np.linspace(0.1, 0.9, 10)

    result = cont_did(
        yname="Y",
        dname="D",
        tname="time_period",
        idname="id",
        data=contdid_data,
        gname="G",
        dvals=custom_dvals,
        degree=2,
        num_knots=1,
        biters=100,
    )

    assert isinstance(result, DoseResult | PTEResult)
    if hasattr(result, "dose"):
        assert len(result.dose) == len(custom_dvals)
        assert np.allclose(result.dose, custom_dvals)


def test_cont_did_control_groups(contdid_data):
    result_nyt = cont_did(
        yname="Y",
        dname="D",
        tname="time_period",
        idname="id",
        data=contdid_data,
        gname="G",
        control_group="notyettreated",
        biters=100,
    )

    result_nt = cont_did(
        yname="Y",
        dname="D",
        tname="time_period",
        idname="id",
        data=contdid_data,
        gname="G",
        control_group="nevertreated",
        biters=100,
    )

    assert isinstance(result_nyt, DoseResult | PTEResult)
    assert isinstance(result_nt, DoseResult | PTEResult)


def test_cont_did_base_period(contdid_data):
    result_varying = cont_did(
        yname="Y",
        dname="D",
        tname="time_period",
        idname="id",
        data=contdid_data,
        gname="G",
        base_period="varying",
        biters=100,
    )

    result_universal = cont_did(
        yname="Y",
        dname="D",
        tname="time_period",
        idname="id",
        data=contdid_data,
        gname="G",
        base_period="universal",
        biters=100,
    )

    assert isinstance(result_varying, DoseResult | PTEResult)
    assert isinstance(result_universal, DoseResult | PTEResult)


def test_cont_did_bootstrap_types(contdid_data):
    result_mult = cont_did(
        yname="Y",
        dname="D",
        tname="time_period",
        idname="id",
        data=contdid_data,
        gname="G",
        boot_type="multiplier",
        biters=50,
    )

    result_emp = cont_did(
        yname="Y",
        dname="D",
        tname="time_period",
        idname="id",
        data=contdid_data,
        gname="G",
        boot_type="empirical",
        biters=50,
    )

    assert isinstance(result_mult, DoseResult | PTEResult)
    assert isinstance(result_emp, DoseResult | PTEResult)


def test_cont_did_confidence_bands(contdid_data):
    result_no_cband = cont_did(
        yname="Y",
        dname="D",
        tname="time_period",
        idname="id",
        data=contdid_data,
        gname="G",
        cband=False,
        biters=100,
    )

    result_cband = cont_did(
        yname="Y",
        dname="D",
        tname="time_period",
        idname="id",
        data=contdid_data,
        gname="G",
        cband=True,
        biters=100,
    )

    assert isinstance(result_no_cband, DoseResult | PTEResult)
    assert isinstance(result_cband, DoseResult | PTEResult)


def test_cont_did_significance_level(contdid_data):
    result_05 = cont_did(
        yname="Y",
        dname="D",
        tname="time_period",
        idname="id",
        data=contdid_data,
        gname="G",
        alp=0.05,
        biters=100,
    )

    result_10 = cont_did(
        yname="Y",
        dname="D",
        tname="time_period",
        idname="id",
        data=contdid_data,
        gname="G",
        alp=0.10,
        biters=100,
    )

    assert isinstance(result_05, DoseResult | PTEResult)
    assert isinstance(result_10, DoseResult | PTEResult)


def test_cont_did_auto_gname(contdid_data):
    data_no_g = contdid_data.drop(columns=["G"])

    result = cont_did(
        yname="Y",
        dname="D",
        tname="time_period",
        idname="id",
        data=data_no_g,
        gname=None,
        biters=100,
    )

    assert isinstance(result, DoseResult | PTEResult)
    assert result.overall_att is not None


def test_cont_did_invalid_data():
    with pytest.raises(TypeError, match="data must be a pandas DataFrame"):
        cont_did(
            yname="Y",
            dname="D",
            tname="time",
            idname="id",
            data=np.array([1, 2, 3]),
        )


def test_cont_did_missing_columns():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    with pytest.raises(ValueError, match="Missing columns"):
        cont_did(
            yname="Y",
            dname="D",
            tname="time",
            idname="id",
            data=df,
        )


def test_cont_did_covariates_not_supported(contdid_data):
    with pytest.raises(NotImplementedError, match="Covariates not currently supported"):
        cont_did(
            yname="Y",
            dname="D",
            tname="time_period",
            idname="id",
            data=contdid_data,
            gname="G",
            xformula="~x1+x2",
        )


def test_cont_did_discrete_treatment_not_supported(contdid_data):
    with pytest.raises(NotImplementedError, match="Discrete treatment not yet supported"):
        cont_did(
            yname="Y",
            dname="D",
            tname="time_period",
            idname="id",
            data=contdid_data,
            gname="G",
            treatment_type="discrete",
        )


def test_cont_did_unbalanced_panel_not_supported(contdid_data):
    with pytest.raises(NotImplementedError, match="Unbalanced panel not currently supported"):
        cont_did(
            yname="Y",
            dname="D",
            tname="time_period",
            idname="id",
            data=contdid_data,
            gname="G",
            allow_unbalanced_panel=True,
        )


def test_cont_did_est_method_error(contdid_data):
    with pytest.raises(ValueError, match="Covariates not supported"):
        cont_did(
            yname="Y",
            dname="D",
            tname="time_period",
            idname="id",
            data=contdid_data,
            gname="G",
            est_method="dr",
        )


def test_cont_did_clustering_warning(contdid_data):
    with pytest.warns(UserWarning, match="Two-way clustering not currently supported"):
        cont_did(
            yname="Y",
            dname="D",
            tname="time_period",
            idname="id",
            data=contdid_data,
            gname="G",
            clustervars="state",
            biters=100,
        )


def test_cont_did_anticipation_warning(contdid_data):
    with pytest.warns(UserWarning, match="Anticipation not fully tested"):
        cont_did(
            yname="Y",
            dname="D",
            tname="time_period",
            idname="id",
            data=contdid_data,
            gname="G",
            anticipation=1,
            biters=100,
        )


def test_cont_did_weights_warning(contdid_data):
    contdid_data["weights"] = np.random.uniform(0.5, 2.0, len(contdid_data))

    with pytest.warns(UserWarning, match="Sampling weights not fully tested"):
        cont_did(
            yname="Y",
            dname="D",
            tname="time_period",
            idname="id",
            data=contdid_data,
            gname="G",
            weightsname="weights",
            biters=100,
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
    assert result.attgt is not None
    assert result.inf_func is not None


def test_cont_did_acrt_with_knots(simple_panel_data):
    result = cont_did_acrt(
        gt_data=simple_panel_data,
        dvals=np.linspace(0.1, 0.9, 10),
        degree=3,
        knots=[0.3, 0.7],
    )

    assert result.attgt is not None
    assert result.inf_func is not None
    if result.extra_gt_returns:
        assert "att_d" in result.extra_gt_returns
        assert "acrt_d" in result.extra_gt_returns


def test_cont_did_acrt_auto_dvals(simple_panel_data):
    result = cont_did_acrt(
        gt_data=simple_panel_data,
        dvals=None,
        degree=2,
    )

    assert result.attgt is not None
    assert result.inf_func is not None


def test_cont_did_acrt_no_treated(simple_panel_data):
    simple_panel_data["D"] = 0

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
        tname="time_period",
        idname="id",
        dname="D",
    )

    assert "gt_data" in result
    assert "n1" in result
    assert "disidx" in result
    assert isinstance(result["gt_data"], pd.DataFrame)
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
    assert isinstance(result["gt_data"], pd.DataFrame)
    assert "name" in result["gt_data"].columns
    assert "D" in result["gt_data"].columns


def test_cont_two_by_two_subset_anticipation(contdid_data):
    result = cont_two_by_two_subset(
        data=contdid_data,
        g=3,
        tp=2,
        control_group="notyettreated",
        anticipation=1,
        base_period="varying",
        gname="G",
        tname="time_period",
        idname="id",
        dname="D",
    )

    assert "gt_data" in result
    assert result["n1"] > 0


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
    assert set(gt_data["name"].unique()) == {"pre", "post"}


def test_cck_estimator_basic(cck_test_data):
    result = _cck_estimator(
        data=cck_test_data,
        yname="y",
        dname="d",
        gname="g",
        tname="time",
        idname="id",
        dvals=None,
        alp=0.05,
        cband=False,
        target_parameter="level",
    )

    assert isinstance(result, DoseResult)
    assert result.overall_att is not None
    assert result.att_d is not None
    assert result.dose is not None


def test_cck_estimator_custom_dvals(cck_test_data):
    custom_dvals = np.linspace(0.1, 1.9, 20)

    result = _cck_estimator(
        data=cck_test_data,
        yname="y",
        dname="d",
        gname="g",
        tname="time",
        idname="id",
        dvals=custom_dvals,
        alp=0.05,
        cband=True,
        target_parameter="slope",
    )

    assert isinstance(result, DoseResult)
    assert len(result.dose) == len(custom_dvals)
    assert np.allclose(result.dose, custom_dvals)


def test_cck_estimator_invalid_groups():
    data = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "time": [1, 2, 1, 2, 1, 2],
            "y": [1, 2, 3, 4, 5, 6],
            "d": [0, 0, 1, 1, 2, 2],
            "g": [0, 0, 1, 1, 2, 2],
        }
    )

    with pytest.raises(ValueError, match="CCK estimator requires exactly 2 groups"):
        _cck_estimator(
            data=data,
            yname="y",
            dname="d",
            gname="g",
            tname="time",
            idname="id",
            dvals=None,
            alp=0.05,
            cband=False,
            target_parameter="level",
        )


def test_cck_estimator_invalid_times():
    data = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "time": [1, 2, 3, 1, 2, 3],
            "y": [1, 2, 3, 4, 5, 6],
            "d": [0, 0, 0, 1, 1, 1],
            "g": [0, 0, 0, 1, 1, 1],
        }
    )

    with pytest.raises(ValueError, match="CCK estimator requires exactly 2 time periods"):
        _cck_estimator(
            data=data,
            yname="y",
            dname="d",
            gname="g",
            tname="time",
            idname="id",
            dvals=None,
            alp=0.05,
            cband=False,
            target_parameter="level",
        )


def test_cck_estimator_no_treated():
    data = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3, 4, 4],
            "time": [1, 2, 1, 2, 1, 2, 1, 2],
            "y": [1, 2, 3, 4, 5, 6, 7, 8],
            "d": [0, 0, 0, 0, 0, 0, 0, 0],
            "g": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )

    with pytest.raises(ValueError, match="No treated units found"):
        _cck_estimator(
            data=data,
            yname="y",
            dname="d",
            gname="g",
            tname="time",
            idname="id",
            dvals=None,
            alp=0.05,
            cband=False,
            target_parameter="level",
        )


def test_cont_did_cck_method(cck_test_data):
    result = cont_did(
        yname="y",
        dname="d",
        tname="time",
        idname="id",
        data=cck_test_data,
        gname="g",
        dose_est_method="cck",
        aggregation="dose",
    )

    assert isinstance(result, DoseResult)
    assert result.overall_att is not None
    assert result.att_d is not None


def test_cont_did_cck_invalid_aggregation(cck_test_data):
    with pytest.raises(ValueError, match="Event study not supported with CCK estimator"):
        cont_did(
            yname="y",
            dname="d",
            tname="time",
            idname="id",
            data=cck_test_data,
            gname="g",
            dose_est_method="cck",
            aggregation="eventstudy",
        )


def test_cont_did_invalid_parameter_combination(contdid_data):
    with pytest.raises(ValueError, match="Invalid combination of parameters"):
        cont_did(
            yname="Y",
            dname="D",
            tname="time_period",
            idname="id",
            data=contdid_data,
            gname="G",
            target_parameter="invalid",
            aggregation="dose",
            treatment_type="continuous",
        )


def test_cont_did_various_degree_knot_combinations(contdid_data):
    configs = [
        (1, 0),
        (2, 0),
        (3, 0),
        (2, 1),
        (3, 2),
        (4, 3),
    ]

    for degree, num_knots in configs:
        result = cont_did(
            yname="Y",
            dname="D",
            tname="time_period",
            idname="id",
            data=contdid_data,
            gname="G",
            degree=degree,
            num_knots=num_knots,
            biters=50,
        )
        assert isinstance(result, DoseResult | PTEResult)
        assert result.overall_att is not None
