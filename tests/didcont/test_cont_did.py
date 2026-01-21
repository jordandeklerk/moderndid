"""Tests for continuous treatment difference-in-differences estimation."""

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid.didcont.cont_did import (
    cont_did,
    cont_did_acrt,
    cont_two_by_two_subset,
)
from moderndid.didcont.estimation import DoseResult, PTEResult


def test_cont_did_basic(contdid_data):
    result = cont_did(
        yname="Y",
        dname="D",
        tname="period",
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
        yname="Y",
        dname="D",
        tname="period",
        idname="id",
        data=contdid_data,
        gname="G",
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
        yname="Y",
        dname="D",
        tname="period",
        idname="id",
        data=contdid_data,
        gname="G",
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
        yname="Y",
        dname="D",
        tname="period",
        idname="id",
        data=contdid_data,
        gname="G",
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
        yname="Y",
        dname="D",
        tname="period",
        idname="id",
        data=contdid_data,
        gname="G",
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
        yname="Y",
        dname="D",
        tname="period",
        idname="id",
        data=contdid_data,
        gname="G",
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


@pytest.mark.parametrize("base_period", ["varying", "universal"])
def test_cont_did_base_period(contdid_data, base_period):
    result = cont_did(
        yname="Y",
        dname="D",
        tname="period",
        idname="id",
        data=contdid_data,
        gname="G",
        base_period=base_period,
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
        yname="Y",
        dname="D",
        tname="period",
        idname="id",
        data=contdid_data,
        gname="G",
        boot_type=boot_type,
        biters=10,
    )

    assert isinstance(result, DoseResult | PTEResult)
    assert np.isfinite(result.overall_att)

    if boot_type == "empirical":
        if isinstance(result, PTEResult):
            assert hasattr(result, "overall_att")
            if hasattr(result.overall_att, "se"):
                assert result.overall_att.se is not None
                assert np.isfinite(result.overall_att.se)
                assert result.overall_att.se > 0
        elif isinstance(result, DoseResult):
            assert result.overall_att_se is not None
            assert np.isfinite(result.overall_att_se)
            assert result.overall_att_se > 0


@pytest.mark.parametrize("cband", [False, True])
def test_cont_did_confidence_bands(contdid_data, cband):
    result = cont_did(
        yname="Y",
        dname="D",
        tname="period",
        idname="id",
        data=contdid_data,
        gname="G",
        cband=cband,
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


@pytest.mark.parametrize("alp", [0.05, 0.10])
def test_cont_did_significance_level(contdid_data, alp):
    result = cont_did(
        yname="Y",
        dname="D",
        tname="period",
        idname="id",
        data=contdid_data,
        gname="G",
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


def test_cont_did_auto_gname(contdid_data):
    data_no_g = contdid_data.drop("G")

    result = cont_did(
        yname="Y",
        dname="D",
        tname="period",
        idname="id",
        data=data_no_g,
        gname=None,
        biters=10,
    )

    assert isinstance(result, DoseResult | PTEResult)
    assert np.isfinite(result.overall_att)


def test_cont_did_empirical_bootstrap_fallback(contdid_data):
    result = cont_did(
        yname="Y",
        dname="D",
        tname="period",
        idname="id",
        data=contdid_data,
        gname="G",
        target_parameter="slope",
        aggregation="eventstudy",
        boot_type="empirical",
        biters=10,
    )

    assert isinstance(result, PTEResult)
    assert hasattr(result, "overall_att")
    if result.overall_att is not None:
        if hasattr(result.overall_att, "overall_se"):
            assert np.isfinite(result.overall_att.overall_se)
            assert result.overall_att.overall_se >= 0


def test_cont_did_invalid_data():
    with pytest.raises(TypeError, match="data must be a pandas or polars DataFrame"):
        cont_did(
            yname="Y",
            dname="D",
            tname="time",
            idname="id",
            data=np.array([1, 2, 3]),
        )


def test_cont_did_missing_columns():
    df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

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
            tname="period",
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
            tname="period",
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
            tname="period",
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
            tname="period",
            idname="id",
            data=contdid_data,
            gname="G",
            est_method="dr",
        )


@pytest.mark.filterwarnings("ignore:Simultaneous confidence band:UserWarning")
@pytest.mark.filterwarnings("ignore:Not returning pre-test Wald statistic:UserWarning")
def test_cont_did_clustering_warning(contdid_data):
    with pytest.warns(UserWarning, match="Two-way clustering not currently supported"):
        cont_did(
            yname="Y",
            dname="D",
            tname="period",
            idname="id",
            data=contdid_data,
            gname="G",
            clustervars="state",
            biters=10,
        )


@pytest.mark.filterwarnings("ignore:Not returning pre-test Wald statistic:UserWarning")
@pytest.mark.filterwarnings("ignore:Simultaneous confidence band is smaller than pointwise:UserWarning")
def test_cont_did_anticipation_warning(contdid_data):
    with pytest.warns(UserWarning, match="Anticipation not fully tested"):
        cont_did(
            yname="Y",
            dname="D",
            tname="period",
            idname="id",
            data=contdid_data,
            gname="G",
            anticipation=1,
            biters=10,
        )


@pytest.mark.filterwarnings("ignore:Simultaneous confidence band:UserWarning")
@pytest.mark.filterwarnings("ignore:Not returning pre-test Wald statistic:UserWarning")
@pytest.mark.filterwarnings("ignore:Simultaneous critical value is arguably 'too large':UserWarning")
def test_cont_did_weights_warning(contdid_data):
    contdid_data = contdid_data.with_columns(pl.lit(np.random.uniform(0.5, 2.0, len(contdid_data))).alias("weights"))

    with pytest.warns(UserWarning, match="Sampling weights not fully tested"):
        cont_did(
            yname="Y",
            dname="D",
            tname="period",
            idname="id",
            data=contdid_data,
            gname="G",
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
    assert set(gt_data["name"].unique().to_list()) == {"pre", "post"}


def test_cck_estimator_basic(cck_test_data):
    result = cont_did(
        data=cck_test_data,
        yname="y",
        dname="d",
        gname="g",
        tname="time",
        idname="id",
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


def test_cck_estimator_custom_dvals(cck_test_data):
    custom_dvals = np.linspace(0.1, 1.9, 20)

    result = cont_did(
        data=cck_test_data,
        yname="y",
        dname="d",
        gname="g",
        tname="time",
        idname="id",
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
            dname="d",
            gname="g",
            tname="time",
            idname="id",
            dose_est_method="cck",
            alp=0.05,
            cband=False,
            target_parameter="level",
        )


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
            dname="d",
            gname="g",
            tname="time",
            idname="id",
            dose_est_method="cck",
            alp=0.05,
            cband=False,
            target_parameter="level",
        )


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
            dname="d",
            gname="g",
            tname="time",
            idname="id",
            dose_est_method="cck",
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
    assert result.att_d is not None
    assert len(result.att_d) > 0
    assert np.all(np.isfinite(result.att_d))
    assert np.isfinite(result.overall_acrt)


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
            tname="period",
            idname="id",
            data=contdid_data,
            gname="G",
            target_parameter="invalid",
            aggregation="dose",
            treatment_type="continuous",
        )


@pytest.mark.parametrize("degree,num_knots", [(1, 0), (2, 0), (3, 0)])
def test_cont_did_various_degree_knot_combinations(contdid_data, degree, num_knots):
    result = cont_did(
        yname="Y",
        dname="D",
        tname="period",
        idname="id",
        data=contdid_data,
        gname="G",
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
