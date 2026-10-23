"""Tests for continuous treatment difference-in-differences estimation."""

import re
import warnings

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
        xformla="~1",
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


@pytest.mark.parametrize("boot_type", ["multiplier", "weighted", "empirical"])
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
        biters=2,
    )

    assert isinstance(result, PTEResult)
    assert np.isfinite(result.overall_att.overall_att)
    assert np.isfinite(result.overall_att.overall_se)
    assert result.overall_att.overall_se > 0


def test_cont_did_weighted_bootstrap_repeats_the_multiplier_dose_curves(contdid_data):
    kwargs = {
        "yname": "Y",
        "tname": "period",
        "idname": "id",
        "gname": "G",
        "dname": "D",
        "degree": 2,
        "cband": True,
        "biters": 49,
        "random_state": 11,
    }
    weighted = cont_did(data=contdid_data, boot_type="weighted", **kwargs)
    multiplier = cont_did(data=contdid_data, boot_type="multiplier", **kwargs)

    np.testing.assert_array_equal(weighted.att_d_se, multiplier.att_d_se)
    np.testing.assert_array_equal(weighted.acrt_d_se, multiplier.acrt_d_se)
    assert weighted.att_d_crit_val == multiplier.att_d_crit_val
    assert weighted.acrt_d_crit_val == multiplier.acrt_d_crit_val
    assert weighted.overall_att_se == multiplier.overall_att_se
    assert weighted.overall_acrt_se == multiplier.overall_acrt_se


@pytest.mark.parametrize("target_parameter", ["level", "slope"])
def test_cont_did_weighted_bootstrap_repeats_the_multiplier_event_study(contdid_data, target_parameter):
    kwargs = {
        "yname": "Y",
        "tname": "period",
        "idname": "id",
        "gname": "G",
        "dname": "D",
        "target_parameter": target_parameter,
        "aggregation": "eventstudy",
        "cband": True,
        "biters": 5,
        "random_state": 11,
    }
    weighted = cont_did(data=contdid_data, boot_type="weighted", **kwargs)
    multiplier = cont_did(data=contdid_data, boot_type="multiplier", **kwargs)
    empirical = cont_did(data=contdid_data, boot_type="empirical", **kwargs)

    np.testing.assert_array_equal(weighted.att_gt.se, multiplier.att_gt.se)
    np.testing.assert_array_equal(weighted.event_study.se_by_event, multiplier.event_study.se_by_event)
    assert weighted.event_study.critical_value == multiplier.event_study.critical_value
    assert weighted.overall_att.overall_se == multiplier.overall_att.overall_se
    assert not np.allclose(empirical.event_study.se_by_event, multiplier.event_study.se_by_event)


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


@pytest.mark.parametrize(
    ("window", "message"),
    [
        ({"min_e": -1}, "min_e=-1 needs aggregation='eventstudy'."),
        ({"max_e": 1}, "max_e=1 needs aggregation='eventstudy'."),
        ({"balance_e": 1}, "balance_e=1 needs aggregation='eventstudy'."),
        ({"min_e": -1, "max_e": 1}, "min_e=-1 and max_e=1 need aggregation='eventstudy'."),
        (
            {"min_e": -1, "max_e": 1, "balance_e": 1},
            "min_e=-1, max_e=1, and balance_e=1 need aggregation='eventstudy'.",
        ),
    ],
)
def test_cont_did_dose_aggregation_rejects_event_time_options(contdid_data, window, message):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        cont_did(data=contdid_data, yname="Y", tname="period", idname="id", gname="G", dname="D", **window)


@pytest.mark.parametrize(
    ("window", "options", "pronoun"),
    [({"min_e": -1}, "min_e=-1 needs", "it"), ({"max_e": 1, "balance_e": 1}, "max_e=1 and balance_e=1 need", "them")],
)
def test_cont_did_cck_rejects_event_time_options(contdid_two_period_data, window, options, pronoun):
    message = f"{options} an event study. Since dose_est_method='cck' has no event study, leave {pronoun} out."

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        cont_did(
            data=contdid_two_period_data,
            yname="Y",
            tname="period",
            idname="id",
            gname="G",
            dname="D",
            dose_est_method="cck",
            **window,
        )


@pytest.mark.parametrize(
    ("window", "message"),
    [
        ({"balance_e": -1}, "balance_e=-1 is not valid. Must be a non-negative event time."),
        ({"balance_e": np.nan}, "balance_e=nan is not valid. Must be a non-negative event time."),
        ({"min_e": 2, "max_e": 1}, "min_e=2 must be less than or equal to max_e=1."),
        ({"min_e": np.nan}, "min_e=nan is not valid. Must be a number or -inf."),
        ({"max_e": np.nan}, "max_e=nan is not valid. Must be a number or inf."),
        (
            {"min_e": 100},
            "No group-time effect has an event time within min_e=100 and max_e=inf. Widen the window or check the "
            "warnings for cells without an estimate.",
        ),
        (
            {"balance_e": 10},
            "No group-time effect has an event time within min_e=-inf and max_e=inf with balance_e=10. Widen the "
            "window or check the warnings for cells without an estimate.",
        ),
    ],
)
def test_cont_did_rejects_invalid_event_time_windows(contdid_data, window, message):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        cont_did(
            data=contdid_data,
            yname="Y",
            tname="period",
            idname="id",
            gname="G",
            dname="D",
            aggregation="eventstudy",
            biters=10,
            random_state=0,
            **window,
        )


def test_cont_did_event_study_warns_when_max_e_leaves_no_post_treatment_effect(contdid_data):
    spec = {"yname": "Y", "tname": "period", "idname": "id", "gname": "G", "dname": "D", "biters": 10}
    message = (
        "The overall effect of the event study is NaN because max_e=-1 leaves out every post-treatment group-time "
        "effect. Pass a larger max_e to estimate it."
    )

    full = cont_did(data=contdid_data, aggregation="eventstudy", random_state=0, **spec).event_study
    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$"):
        trimmed = cont_did(data=contdid_data, aggregation="eventstudy", max_e=-1, random_state=0, **spec)

    np.testing.assert_array_equal(trimmed.event_study.event_times, [-2, -1])
    np.testing.assert_allclose(trimmed.event_study.att_by_event, full.att_by_event[:2], rtol=1e-12)
    assert np.isnan(trimmed.overall_att.overall_att)
    assert np.isfinite(full.overall_att)


@pytest.mark.filterwarnings("ignore:The dose curve can't be estimated for cohort 2:UserWarning")
def test_cont_did_event_study_warns_when_balancing_leaves_no_post_treatment_effect(contdid_data):
    one_dose = contdid_data.with_columns(pl.when(pl.col("G") == 2).then(0.5).otherwise(pl.col("D")).alias("D"))
    message = (
        "The overall effect of the event study is NaN because no post-treatment group-time effect is left to "
        "average. An earlier warning names the cohorts whose cells have no estimate."
    )

    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$"):
        result = cont_did(
            data=one_dose,
            yname="Y",
            tname="period",
            idname="id",
            gname="G",
            dname="D",
            aggregation="eventstudy",
            target_parameter="slope",
            base_period="universal",
            balance_e=2,
            biters=10,
            random_state=0,
        )

    np.testing.assert_array_equal(result.event_study.event_times, [-1])
    np.testing.assert_array_equal(result.event_study.att_by_event, [0.0])
    assert np.isnan(result.overall_att.overall_att)


@pytest.mark.parametrize(
    ("window", "event_times"),
    [
        ({"min_e": -1, "max_e": 1}, [-1, 0, 1]),
        ({"max_e": 0}, [-2, -1, 0]),
        ({"balance_e": 1}, [-1, 0, 1]),
        ({"balance_e": 1, "min_e": 0}, [0, 1]),
    ],
)
def test_cont_did_event_study_window_matches_att_gt(contdid_data, window, event_times):
    kwargs = {"yname": "Y", "tname": "period", "idname": "id", "gname": "G"}

    result = cont_did(
        data=contdid_data, dname="D", aggregation="eventstudy", biters=10, random_state=0, **window, **kwargs
    )
    binary = aggte(
        att_gt(data=contdid_data, control_group="notyettreated", boot=False, cband=False, **kwargs), "dynamic", **window
    )

    np.testing.assert_array_equal(result.event_study.event_times, event_times)
    np.testing.assert_array_equal(binary.event_times, event_times)
    np.testing.assert_allclose(result.event_study.att_by_event, binary.att_by_event, atol=1e-12)
    assert result.overall_att.overall_att == pytest.approx(binary.overall_att, abs=1e-12)


def test_cont_did_slope_event_study_window_keeps_event_time_estimates(contdid_data):
    kwargs = {
        "yname": "Y",
        "tname": "period",
        "idname": "id",
        "gname": "G",
        "dname": "D",
        "aggregation": "eventstudy",
        "target_parameter": "slope",
        "biters": 10,
        "random_state": 0,
    }

    full = cont_did(data=contdid_data, **kwargs).event_study
    trimmed = cont_did(data=contdid_data, min_e=-1, max_e=1, **kwargs).event_study

    np.testing.assert_array_equal(full.event_times, [-2, -1, 0, 1, 2])
    np.testing.assert_array_equal(trimmed.event_times, [-1, 0, 1])
    np.testing.assert_allclose(trimmed.att_by_event, full.att_by_event[1:4], rtol=1e-12)
    assert trimmed.overall_att == pytest.approx(np.mean(full.att_by_event[2:4]), rel=1e-12)


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


@pytest.mark.parametrize("alpha", [0.05, 0.10])
def test_cont_did_significance_level(contdid_data, alpha):
    result = cont_did(
        data=contdid_data,
        yname="Y",
        tname="period",
        idname="id",
        gname="G",
        dname="D",
        alpha=alpha,
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
        "biters": 2,
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


@pytest.mark.parametrize(
    "data_fixture, options",
    [
        ("contdid_data", {"degree": 2, "num_knots": 0}),
        ("contdid_two_period_data", {"dose_est_method": "cck"}),
    ],
    ids=["parametric", "cck"],
)
def test_cont_did_reads_decimal_outcome_and_dose_as_doubles(request, data_fixture, options):
    decimal = request.getfixturevalue(data_fixture).with_columns(pl.col("Y", "D").cast(pl.Decimal(18, 6)))
    floats = decimal.with_columns(pl.col("Y", "D").cast(pl.Float64))
    spec = {
        "yname": "Y",
        "tname": "period",
        "idname": "id",
        "gname": "G",
        "dname": "D",
        "target_parameter": "level",
        "aggregation": "dose",
        "biters": 10,
        "random_state": 42,
        **options,
    }

    with warnings.catch_warnings(record=True) as expected_record:
        warnings.simplefilter("always")
        expected = cont_did(data=floats, **spec)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = cont_did(data=decimal, **spec)

    assert [str(warning.message) for warning in record] == [str(warning.message) for warning in expected_record]
    np.testing.assert_array_equal(result.dose, expected.dose)
    np.testing.assert_allclose(result.overall_att, expected.overall_att, rtol=1e-12)
    np.testing.assert_allclose(result.overall_att_se, expected.overall_att_se, rtol=1e-12)
    np.testing.assert_allclose(result.att_d, expected.att_d, rtol=1e-12)
    np.testing.assert_allclose(result.att_d_se, expected.att_d_se, rtol=1e-12)


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


@pytest.mark.parametrize(
    ("clustervars", "shown"),
    [
        ("state", "['state']"),
        (["state"], "['state']"),
        (["id", "state"], "['state']"),
        (["G", "state"], "['G', 'state']"),
    ],
)
def test_cont_did_ignores_cluster_variables_other_than_the_unit(contdid_data, clustervars, shown):
    kwargs = {
        "yname": "Y",
        "tname": "period",
        "idname": "id",
        "gname": "G",
        "dname": "D",
        "aggregation": "eventstudy",
        "biters": 19,
        "random_state": 5,
    }
    message = (
        f"cont_did clusters standard errors by unit only. It ignores clustervars={shown} since clustering on other "
        "variables isn't supported yet."
    )
    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$"):
        result = cont_did(data=contdid_data, clustervars=clustervars, **kwargs)
    unclustered = cont_did(data=contdid_data, **kwargs)

    np.testing.assert_array_equal(result.event_study.se_by_event, unclustered.event_study.se_by_event)
    assert result.overall_att.overall_se == unclustered.overall_att.overall_se


@pytest.mark.parametrize("clustervars", ["id", ["id"], []])
def test_cont_did_clustering_by_unit_gives_no_warning(contdid_data, clustervars, recwarn):
    kwargs = {
        "yname": "Y",
        "tname": "period",
        "idname": "id",
        "gname": "G",
        "dname": "D",
        "degree": 2,
        "biters": 19,
        "random_state": 5,
    }
    result = cont_did(data=contdid_data, clustervars=clustervars, **kwargs)
    unclustered = cont_did(data=contdid_data, **kwargs)

    assert not [w for w in recwarn if "cluster" in str(w.message)]
    np.testing.assert_array_equal(result.att_d_se, unclustered.att_d_se)
    assert result.overall_att_se == unclustered.overall_att_se


@pytest.mark.parametrize(
    ("option", "value"),
    [("boot", True), ("boot", False), ("xformula", "~1"), ("est_method", "reg"), ("max_event_time", 1)],
)
def test_cont_did_rejects_unknown_keyword_arguments(contdid_data, option, value):
    message = f"cont_did() got an unexpected keyword argument '{option}'"

    with pytest.raises(TypeError, match=f"^{re.escape(message)}"):
        cont_did(data=contdid_data, yname="Y", tname="period", idname="id", gname="G", dname="D", **{option: value})


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


def test_cont_did_acrt_matches_least_squares_on_a_quadratic(make_dose_cell):
    cell = make_dose_cell(np.random.default_rng(1).uniform(0.1, 1.0, 120))
    dvals = np.array([0.2, 0.5, 0.8])

    result = cont_did_acrt(cell, dvals=dvals, degree=2, knots=[])

    post = cell.filter(pl.col("name") == "post").sort("id")
    dy = post["Y"].to_numpy() - cell.filter(pl.col("name") == "pre").sort("id")["Y"].to_numpy()
    d = post["D"].to_numpy()
    treated = d > 0
    n_treated = treated.sum()
    x = np.column_stack([np.ones(n_treated), d[treated], d[treated] ** 2])
    beta = np.linalg.lstsq(x, dy[treated], rcond=None)[0]
    slope = beta[1] + 2 * beta[2] * d[treated]
    score = (dy[treated] - x @ beta)[:, None] * x
    bread = np.linalg.inv(x.T @ x / n_treated)
    mean_slope_basis = np.array([0.0, 1.0, 2 * d[treated].mean()])
    expected_inf_func = np.zeros(len(d))
    expected_inf_func[treated] = len(d) / n_treated * ((slope - slope.mean()) + score @ bread @ mean_slope_basis)
    expected_att_d = beta[0] + beta[1] * dvals + beta[2] * dvals**2 - dy[~treated].mean()

    assert result.attgt == pytest.approx(slope.mean(), rel=1e-10)
    np.testing.assert_allclose(result.inf_func, expected_inf_func, atol=1e-8)
    np.testing.assert_allclose(result.extra_gt_returns["att_d"], expected_att_d, atol=1e-10)
    np.testing.assert_allclose(result.extra_gt_returns["acrt_d"], beta[1] + 2 * beta[2] * dvals, atol=1e-10)


@pytest.mark.parametrize(("knots", "powers"), [([0.3, 0.7], [(0.3, 3), (0.7, 3)]), ([0.5, 0.5], [(0.5, 3), (0.5, 2)])])
def test_cont_did_acrt_with_knots_matches_truncated_power_basis(make_dose_cell, knots, powers):
    cell = make_dose_cell(np.random.default_rng(2).uniform(0.1, 1.0, 150))

    result = cont_did_acrt(cell, dvals=np.array([0.5]), degree=3, knots=knots)

    post = cell.filter(pl.col("name") == "post").sort("id")
    dy = post["Y"].to_numpy() - cell.filter(pl.col("name") == "pre").sort("id")["Y"].to_numpy()
    d = post["D"].to_numpy()
    dt = d[d > 0]
    columns = [np.ones(len(dt)), dt, dt**2, dt**3] + [np.clip(dt - knot, 0, None) ** p for knot, p in powers]
    derivatives = [np.zeros(len(dt)), np.ones(len(dt)), 2 * dt, 3 * dt**2]
    derivatives += [p * np.clip(dt - knot, 0, None) ** (p - 1) for knot, p in powers]
    beta = np.linalg.lstsq(np.column_stack(columns), dy[d > 0], rcond=None)[0]

    assert result.attgt == pytest.approx(np.mean(np.column_stack(derivatives) @ beta), rel=1e-8)
    np.testing.assert_array_equal(result.extra_gt_returns["knots"], knots)


def test_cont_did_acrt_drops_knots_outside_the_cell_doses(make_dose_cell):
    cell = make_dose_cell(np.append(np.random.default_rng(3).uniform(0.1, 0.4, 99), 0.4))
    dvals = np.array([0.2, 0.3, 0.9])

    with_knots = cont_did_acrt(cell, dvals=dvals, degree=3, knots=[0.05, 0.4, 0.6])
    without_knots = cont_did_acrt(cell, dvals=dvals, degree=3, knots=[])

    assert with_knots.extra_gt_returns["knots"].size == 0
    assert with_knots.attgt == pytest.approx(without_knots.attgt, rel=1e-12)
    np.testing.assert_allclose(with_knots.inf_func, without_knots.inf_func, atol=1e-12)
    np.testing.assert_allclose(
        with_knots.extra_gt_returns["att_d"], without_knots.extra_gt_returns["att_d"], atol=1e-12
    )


def test_cont_did_acrt_slope_is_nan_with_fewer_doses_than_coefficients(make_dose_cell):
    cell = make_dose_cell(np.tile([0.2, 0.5, 0.9], 40))

    result = cont_did_acrt(cell, dvals=np.array([0.5]), degree=3, knots=[])

    post = cell.filter(pl.col("name") == "post").sort("id")
    dy = post["Y"].to_numpy() - cell.filter(pl.col("name") == "pre").sort("id")["Y"].to_numpy()
    treated = post["D"].to_numpy() > 0
    assert np.isnan(result.attgt)
    assert np.all(np.isnan(result.inf_func))
    assert "att_d" not in result.extra_gt_returns
    assert result.extra_gt_returns["att_overall"] == pytest.approx(dy[treated].mean() - dy[~treated].mean(), rel=1e-12)


def test_cont_did_acrt_slope_is_nan_when_doses_leave_a_knot_interval_empty(make_dose_cell):
    cell = make_dose_cell(np.append(np.random.default_rng(4).uniform(0.1, 0.3, 60), 1.0))

    result = cont_did_acrt(cell, dvals=np.array([0.5]), degree=1, knots=[0.5, 0.7])

    assert np.isnan(result.attgt)
    assert np.all(np.isnan(result.inf_func))
    assert "att_inf_func" in result.extra_gt_returns


def test_cont_did_acrt_raises_when_ties_put_too_many_knots_at_one_dose(make_dose_cell):
    cell = make_dose_cell(np.random.default_rng(6).uniform(0.1, 1.0, 100))

    with pytest.raises(ValueError, match="put 2 interior knots at the dose 0.5. More than degree=1 knots"):
        cont_did_acrt(cell, dvals=np.array([0.5]), degree=1, knots=[0.5, 0.5])


def test_cont_did_acrt_influence_function_is_accurate_with_nearly_tied_doses(make_dose_cell):
    cell = make_dose_cell(np.concatenate([np.tile([0.2, 0.5, 0.9], 40), np.full(5, 0.9 + 1e-8)]))

    result = cont_did_acrt(cell, dvals=np.array([0.5]), degree=3, knots=[])

    post = cell.filter(pl.col("name") == "post").sort("id")
    dy = post["Y"].to_numpy() - cell.filter(pl.col("name") == "pre").sort("id")["Y"].to_numpy()
    d = post["D"].to_numpy()
    treated = d > 0
    z = (d[treated] - 0.55) / 0.35
    x = np.column_stack([np.ones(len(z)), z, z**2, z**3])
    q, r = np.linalg.qr(x)
    beta = np.linalg.solve(r, q.T @ dy[treated])
    slope = (beta[1] + 2 * beta[2] * z + 3 * beta[3] * z**2) / 0.35
    mean_slope_basis = np.array([0.0, 1.0, 2 * z.mean(), 3 * np.mean(z**2)]) / 0.35
    bread_times_basis = len(z) * np.linalg.solve(r, np.linalg.solve(r.T, mean_slope_basis))
    score = (dy[treated] - x @ beta)[:, None] * x
    expected = np.zeros(len(d))
    expected[treated] = len(d) / len(z) * ((slope - slope.mean()) + score @ bread_times_basis)

    assert result.attgt == pytest.approx(slope.mean(), rel=1e-5)
    np.testing.assert_allclose(result.inf_func, expected, rtol=1e-5, atol=1e-5 * np.max(np.abs(expected)))


def test_cont_did_acrt_without_dvals_skips_the_curves(make_dose_cell):
    cell = make_dose_cell(np.random.default_rng(5).uniform(0.1, 1.0, 100))

    without_grid = cont_did_acrt(cell, dvals=None, degree=2)
    with_grid = cont_did_acrt(cell, dvals=np.linspace(0.2, 0.8, 4), degree=2)

    assert without_grid.extra_gt_returns["att_d"].size == 0
    assert without_grid.extra_gt_returns["acrt_d"].size == 0
    assert without_grid.attgt == pytest.approx(with_grid.attgt, rel=1e-12)
    np.testing.assert_allclose(without_grid.inf_func, with_grid.inf_func, atol=1e-12)


def test_cont_did_acrt_without_treated_units_has_no_estimate(simple_panel_data):
    simple_panel_data = simple_panel_data.with_columns(pl.lit(0).alias("D"))

    result = cont_did_acrt(gt_data=simple_panel_data, dvals=np.array([0.5]), degree=2)

    assert np.isnan(result.attgt)
    assert np.all(np.isnan(result.inf_func))
    assert result.extra_gt_returns is None


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
        g=4,
        tp=2,
        control_group="notyettreated",
        anticipation=1,
        base_period="varying",
    )

    assert sorted(result["gt_data"]["G"].unique().to_list()) == [0, 4]
    assert sorted(result["gt_data"]["period"].unique().to_list()) == [1, 2]


def test_cont_two_by_two_subset_universal_base_pre_period_controls(contdid_data):
    result = cont_two_by_two_subset(
        data=contdid_data,
        g=4,
        tp=1,
        control_group="notyettreated",
        anticipation=0,
        base_period="universal",
    )

    assert sorted(result["gt_data"]["G"].unique().to_list()) == [0, 4]
    assert sorted(result["gt_data"]["period"].unique().to_list()) == [1, 3]


@pytest.mark.parametrize(("control_group", "groups"), [("notyettreated", [0, 2, 4]), ("nevertreated", [0, 2])])
def test_cont_two_by_two_subset_keeps_never_treated_units_coded_zero(contdid_data, control_group, groups):
    never_treated = contdid_data.filter(pl.col("G") == 0)["id"].n_unique()

    result = cont_two_by_two_subset(data=contdid_data, g=2, tp=3, control_group=control_group)

    cell = result["gt_data"]
    assert sorted(cell["G"].unique().to_list()) == groups
    assert cell.filter(pl.col("G") == 0)["id"].n_unique() == never_treated
    assert cell.filter(pl.col("G") != 2)["D"].unique().to_list() == [0.0]


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
        alpha=0.05,
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
        alpha=0.05,
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
            alpha=0.05,
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
            alpha=0.05,
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
            alpha=0.05,
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


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("control_group", None, "control_group=None is not valid. Must be 'notyettreated' or 'nevertreated'."),
        ("base_period", None, "base_period=None is not valid. Must be 'varying' or 'universal'."),
        ("base_period", 3, "base_period=3 is not valid. Must be 'varying' or 'universal'."),
        ("boot_type", None, "boot_type=None is not valid. Must be 'weighted', 'multiplier', or 'empirical'."),
        ("aggregation", None, "aggregation=None is not valid. Must be 'dose' or 'eventstudy'."),
        ("target_parameter", None, "target_parameter=None is not valid. Must be 'level' or 'slope'."),
        ("dose_est_method", None, "dose_est_method=None is not valid. Must be 'parametric' or 'cck'."),
        ("treatment_type", None, "treatment_type=None is not valid. Must be 'continuous' or 'discrete'."),
    ],
)
def test_cont_did_rejects_options_that_are_not_strings(contdid_data, option, value, message):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        cont_did(data=contdid_data, yname="Y", tname="period", idname="id", gname="G", dname="D", **{option: value})


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
@pytest.mark.filterwarnings("ignore:Simultaneous band smaller than pointwise:UserWarning")
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
        biters=1000,
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
        alpha=1e-10,
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
    base_window = cont_did(
        data=contdid_data, aggregation="eventstudy", target_parameter="slope", min_e=-1.5, max_e=1.5, **kwargs
    )
    coded_window = cont_did(
        data=recoded,
        aggregation="eventstudy",
        target_parameter="slope",
        min_e=-1.5 * scale,
        max_e=1.5 * scale,
        **kwargs,
    )

    np.testing.assert_array_equal(base_window.event_study.event_times, [-1, 0, 1])
    np.testing.assert_allclose(coded_window.event_study.event_times, [-scale, 0, scale], atol=1e-12)
    np.testing.assert_allclose(coded_window.event_study.att_by_event, base_window.event_study.att_by_event, atol=1e-12)
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
    empirical = cont_did(data=half_years, boot_type="empirical", biters=2, **es_kwargs).event_study
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
        "biters": 2,
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


@pytest.mark.parametrize(("aggregation", "target_parameter"), [("dose", "level"), ("eventstudy", "slope")])
def test_cont_did_raises_when_no_cohort_can_fit_the_spline(few_dose_panel, aggregation, target_parameter):
    message = "cohort 2 with degree=3 and num_knots=0. A cohort needs at least 4 distinct positive doses."

    with pytest.raises(ValueError, match=re.escape(message) + ".*Lower degree or num_knots"):
        cont_did(
            data=few_dose_panel,
            yname="Y",
            tname="period",
            idname="id",
            gname="G",
            dname="D",
            aggregation=aggregation,
            target_parameter=target_parameter,
            biters=10,
        )


@pytest.mark.parametrize(
    ("degree", "aggregation", "target_parameter"),
    [(3, "dose", "level"), (1, "dose", "level"), (1, "eventstudy", "slope")],
)
def test_cont_did_points_a_single_dose_to_the_level_event_study(few_dose_panel, degree, aggregation, target_parameter):
    one_dose = few_dose_panel.with_columns(pl.when(pl.col("D") > 0).then(0.5).otherwise(0.0).alias("D"))
    kwargs = {"yname": "Y", "tname": "period", "idname": "id", "gname": "G", "dname": "D", "biters": 10}
    message = (
        "The dose curve can't be estimated for cohort 2. Every treated unit of cohort 2 takes the same dose. No cohort "
        "is left to aggregate. Pass aggregation='eventstudy' with target_parameter='level' to compare treated and "
        "untreated units without the dose."
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        cont_did(data=one_dose, degree=degree, aggregation=aggregation, target_parameter=target_parameter, **kwargs)
    level = cont_did(data=one_dose, aggregation="eventstudy", target_parameter="level", random_state=0, **kwargs)

    wide = one_dose.pivot(index=["id", "G", "D"], on="period", values="Y").sort("id")
    dy = (wide["2"] - wide["1"]).to_numpy()
    treated = wide["D"].to_numpy() > 0

    assert level.event_study.att_by_event[0] == pytest.approx(dy[treated].mean() - dy[~treated].mean(), rel=1e-12)


def test_cont_did_counts_nearly_coinciding_doses_as_one(few_dose_panel):
    near_tie = few_dose_panel.with_columns(
        pl.when((pl.col("D") == 0.9) & (pl.col("id") % 2 == 0)).then(0.9 + 1e-15).otherwise(pl.col("D")).alias("D")
    )
    message = (
        "The dose curve can't be estimated for cohort 2 with degree=3 and num_knots=0. A cohort needs at least 4 "
        "distinct positive doses. Doses that nearly coincide count as one. No cohort is left to aggregate. Lower "
        "degree or num_knots."
    )

    assert near_tie.filter(pl.col("D") > 0)["D"].n_unique() == 4
    with pytest.raises(ValueError, match=re.escape(message)):
        cont_did(data=near_tie, yname="Y", tname="period", idname="id", gname="G", dname="D", biters=10)


def test_cont_did_fits_three_doses_with_a_quadratic(few_dose_panel):
    result = cont_did(
        data=few_dose_panel, yname="Y", tname="period", idname="id", gname="G", dname="D", degree=2, biters=10
    )

    wide = few_dose_panel.pivot(index=["id", "G", "D"], on="period", values="Y").sort("id")
    dy = (wide["2"] - wide["1"]).to_numpy()
    d = wide["D"].to_numpy()
    treated = d > 0
    coef = np.polyfit(d[treated], dy[treated], 2)

    assert result.overall_acrt == pytest.approx(np.mean(np.polyval(np.polyder(coef), d[treated])), rel=1e-10)
    assert result.overall_att == pytest.approx(dy[treated].mean() - dy[~treated].mean(), rel=1e-12)


def test_cont_did_leaves_out_cohort_that_cannot_fit_the_spline(single_dose_cohort_panel):
    kwargs = {
        "yname": "Y",
        "tname": "period",
        "idname": "id",
        "gname": "G",
        "dname": "D",
        "degree": 1,
        "control_group": "nevertreated",
        "biters": 10,
        "random_state": 0,
    }
    message = (
        "The dose curve can't be estimated for cohort 2003. Every treated unit of cohort 2003 takes the same dose. Its "
        "cells are left out of the dose curves and the overall ACRT. The overall ATT still includes it."
    )

    with pytest.warns(UserWarning, match=re.escape(message) + "$"):
        result = cont_did(data=single_dose_cohort_panel, **kwargs)
    without = cont_did(data=single_dose_cohort_panel.filter(pl.col("G") != 2003), **kwargs)

    wide = single_dose_cohort_panel.pivot(index=["id", "G", "D"], on="period", values="Y").sort("id")
    g = wide["G"].to_numpy()
    kept = g != 2003
    y = {t: wide[str(t)].to_numpy() for t in (2001, 2002, 2003)}
    binary_att = {
        (cohort, t): (y[t] - y[base])[g == cohort].mean() - (y[t] - y[base])[g == 0].mean()
        for cohort, t, base in ((2002, 2002, 2001), (2002, 2003, 2001), (2003, 2003, 2002))
    }
    share = np.array([np.sum(g == 2002), np.sum(g == 2003)]) / np.sum(g > 0)
    expected_att = share[0] * (binary_att[2002, 2002] + binary_att[2002, 2003]) / 2 + share[1] * binary_att[2003, 2003]
    n, n_kept = len(g), kept.sum()

    assert result.overall_att == pytest.approx(expected_att, rel=1e-12)
    assert result.overall_acrt == pytest.approx(without.overall_acrt, rel=1e-12)
    np.testing.assert_allclose(result.att_d, without.att_d, atol=1e-12)
    np.testing.assert_allclose(result.acrt_d, without.acrt_d, atol=1e-12)
    np.testing.assert_allclose(
        result.overall_acrt_inf_func[kept] / n, without.overall_acrt_inf_func / n_kept, atol=1e-14
    )
    np.testing.assert_allclose(result.overall_acrt_inf_func[~kept], 0.0, atol=1e-14)
    np.testing.assert_allclose(result.att_d_inf_func[kept] / n, without.att_d_inf_func / n_kept, atol=1e-14)


def test_cont_did_slope_event_study_leaves_out_cohort_that_cannot_fit_the_spline(single_dose_cohort_panel):
    kwargs = {
        "yname": "Y",
        "tname": "period",
        "idname": "id",
        "gname": "G",
        "dname": "D",
        "degree": 1,
        "control_group": "nevertreated",
        "aggregation": "eventstudy",
        "target_parameter": "slope",
        "biters": 10,
        "random_state": 0,
    }
    message = (
        "The dose curve can't be estimated for cohort 2003. Every treated unit of cohort 2003 takes the same dose. Its "
        "cells are left out of the event study. Pass target_parameter='level' for an event study that keeps cohort "
        "2003."
    )

    with pytest.warns(UserWarning, match=re.escape(message)):
        result = cont_did(data=single_dose_cohort_panel, **kwargs)
    without = cont_did(data=single_dose_cohort_panel.filter(pl.col("G") != 2003), **kwargs)

    kept = single_dose_cohort_panel.filter(pl.col("period") == 2001).sort("id")["G"].to_numpy() != 2003
    cohort_2003 = np.asarray(result.att_gt.groups) == 2003

    np.testing.assert_array_equal(result.event_study.event_times, [0, 1])
    np.testing.assert_allclose(result.event_study.att_by_event, without.event_study.att_by_event, atol=1e-12)
    np.testing.assert_allclose(
        result.event_study.influence_func["by_event"][kept] / kept.size,
        without.event_study.influence_func["by_event"] / kept.sum(),
        atol=1e-14,
    )
    assert np.all(np.isnan(np.asarray(result.att_gt.att)[cohort_2003]))
    assert np.all(np.isnan(np.asarray(result.att_gt.se)[cohort_2003]))


def test_cont_did_empirical_bootstrap_leaves_out_cohort_that_cannot_fit_the_spline(single_dose_cohort_panel):
    kwargs = {
        "yname": "Y",
        "tname": "period",
        "idname": "id",
        "gname": "G",
        "dname": "D",
        "degree": 1,
        "control_group": "nevertreated",
        "aggregation": "eventstudy",
        "target_parameter": "slope",
        "random_state": 0,
    }

    with pytest.warns(UserWarning, match="cohort 2003 .* Its cells are left out of the event study"):
        empirical = cont_did(data=single_dose_cohort_panel, boot_type="empirical", biters=2, **kwargs)
    with pytest.warns(UserWarning, match="cohort 2003 .* Its cells are left out of the event study"):
        multiplier = cont_did(data=single_dose_cohort_panel, biters=10, **kwargs)
    without = cont_did(data=single_dose_cohort_panel.filter(pl.col("G") != 2003), biters=10, **kwargs)

    cohort_2003 = np.asarray(empirical.att_gt.groups) == 2003

    np.testing.assert_array_equal(empirical.event_study.event_times, [0, 1])
    np.testing.assert_allclose(empirical.event_study.att_by_event, without.event_study.att_by_event, atol=1e-12)
    assert np.all(np.isfinite(empirical.event_study.se_by_event))
    np.testing.assert_array_equal(empirical.att_gt.groups, multiplier.att_gt.groups)
    np.testing.assert_array_equal(empirical.att_gt.times, multiplier.att_gt.times)
    np.testing.assert_allclose(empirical.att_gt.att, multiplier.att_gt.att, atol=1e-12)
    assert cohort_2003.sum() == 2
    assert np.all(np.isnan(empirical.att_gt.att[cohort_2003]))
    assert np.all(np.isnan(empirical.att_gt.se[cohort_2003]))
    assert np.all(np.isfinite(empirical.att_gt.se[~cohort_2003]))


def test_cont_did_fits_cohorts_whose_dose_range_excludes_a_knot(disjoint_dose_panel, recwarn):
    kwargs = {"yname": "Y", "tname": "period", "idname": "id", "gname": "G", "dname": "D", "biters": 10}

    one_knot = cont_did(data=disjoint_dose_panel, num_knots=1, random_state=0, **kwargs)
    no_knot = cont_did(data=disjoint_dose_panel, num_knots=0, random_state=0, **kwargs)

    assert 0.4 < one_knot.pte_params.knots[0] < 0.6
    assert not [w for w in recwarn if "can't be estimated" in str(w.message)]
    assert one_knot.overall_acrt == pytest.approx(no_knot.overall_acrt, rel=1e-12)
    np.testing.assert_allclose(one_knot.att_d, no_knot.att_d, atol=1e-12)
    np.testing.assert_allclose(one_knot.acrt_d_se, no_knot.acrt_d_se, rtol=1e-10)


@pytest.mark.parametrize(
    ("dvals", "message"),
    [
        ([], "dvals is empty"),
        (np.array([]), "dvals is empty"),
        ([0.3, np.nan], "dvals holds missing or infinite values"),
        ([0.3, np.inf], "dvals holds missing or infinite values"),
    ],
)
def test_cont_did_rejects_dvals_without_finite_doses(contdid_data, dvals, message):
    with pytest.raises(ValueError, match=message):
        cont_did(data=contdid_data, yname="Y", tname="period", idname="id", gname="G", dname="D", dvals=dvals)


def test_cont_did_accepts_a_single_dose_as_dvals(contdid_data):
    kwargs = {"yname": "Y", "tname": "period", "idname": "id", "gname": "G", "dname": "D", "degree": 2}

    scalar = cont_did(data=contdid_data, dvals=0.5, biters=10, random_state=0, **kwargs)
    listed = cont_did(data=contdid_data, dvals=[0.5], biters=10, random_state=0, **kwargs)

    np.testing.assert_array_equal(scalar.dose, [0.5])
    np.testing.assert_allclose(scalar.att_d, listed.att_d, atol=1e-14)


def test_cont_did_cck_raises_when_the_doses_cannot_pin_down_any_sieve(few_dose_panel):
    message = (
        "The smallest sieve of the CCK estimator is a cubic with 4 terms. The 3 distinct treated doses can't pin it "
        "down. Use dose_est_method='parametric' with a degree below 3."
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        cont_did(
            data=few_dose_panel,
            yname="Y",
            tname="period",
            idname="id",
            gname="G",
            dname="D",
            dose_est_method="cck",
            biters=10,
        )


def test_cont_did_cck_points_a_single_dose_to_the_level_event_study(few_dose_panel):
    one_dose = few_dose_panel.with_columns(pl.when(pl.col("D") > 0).then(0.5).otherwise(0.0).alias("D"))
    message = (
        "Since every treated unit takes the same dose, the CCK estimator has no curve to fit. Pass "
        "dose_est_method='parametric', aggregation='eventstudy', and target_parameter='level' to compare treated and "
        "untreated units without the dose."
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        cont_did(
            data=one_dose,
            yname="Y",
            tname="period",
            idname="id",
            gname="G",
            dname="D",
            dose_est_method="cck",
            biters=10,
        )


def test_cont_did_cck_raises_when_the_doses_cannot_pin_down_the_selected_sieve(few_dose_panel, cck_fixed_sieve):
    nine_doses = few_dose_panel.with_columns(
        pl.when(pl.col("D") > 0).then(pl.col("D") + 0.01 * (pl.col("id") % 3)).otherwise(0.0).alias("D")
    )
    cck_fixed_sieve(8)

    with pytest.raises(ValueError, match="selected a sieve with 11 terms that the 9 distinct treated doses can't pin"):
        cont_did(
            data=nine_doses,
            yname="Y",
            tname="period",
            idname="id",
            gname="G",
            dname="D",
            dose_est_method="cck",
            biters=10,
        )


def test_cont_did_cck_overall_acrt_averages_the_npiv_derivative_over_treated_doses(cck_test_data):
    treated_doses = np.sort(cck_test_data.filter((pl.col("time") == 2) & (pl.col("d") > 0))["d"].to_numpy())

    result = cont_did(
        data=cck_test_data,
        yname="y",
        tname="time",
        idname="id",
        gname="g",
        dname="d",
        dose_est_method="cck",
        dvals=treated_doses,
        biters=10,
        random_state=0,
    )

    assert result.overall_acrt == pytest.approx(np.mean(result.acrt_d), rel=1e-13)


def test_cont_did_cck_bands_follow_biters_at_a_fixed_random_state(cck_test_data):
    wide = cck_test_data.pivot(index=["id", "g", "d"], on="time", values="y").sort("id")
    dy = (wide["2"] - wide["1"]).to_numpy()
    dose = wide["d"].to_numpy()
    treated = dose > 0
    spec = {
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "g",
        "dname": "d",
        "dose_est_method": "cck",
        "cband": True,
        "random_state": 3,
    }

    few = cont_did(data=cck_test_data, biters=99, **spec)
    many = cont_did(data=cck_test_data, biters=999, **spec)
    curve = npiv(
        y=dy[treated] - dy[~treated].mean(),
        x=dose[treated].reshape(-1, 1),
        w=dose[treated].reshape(-1, 1),
        x_grid=np.asarray(few.dose).reshape(-1, 1),
        knots="quantiles",
        biters=99,
        j_x_degree=3,
        k_w_degree=3,
        seed=3,
    )

    np.testing.assert_allclose(few.att_d, curve.h, rtol=1e-12)
    np.testing.assert_allclose(few.att_d_crit_val, curve.cv, rtol=1e-12)
    np.testing.assert_allclose(few.acrt_d_crit_val, curve.cv_deriv, rtol=1e-12)
    assert abs(few.att_d_crit_val - many.att_d_crit_val) > 1e-2


def test_cont_did_rejects_the_removed_alp_argument(contdid_data):
    with pytest.raises(TypeError, match="unexpected keyword argument 'alp'"):
        cont_did(data=contdid_data, yname="Y", tname="period", idname="id", gname="G", dname="D", alp=0.1)
