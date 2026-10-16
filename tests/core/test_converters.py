"""Tests for core result converters and to_df dispatch."""

import re

import numpy as np
import polars as pl
import pytest
from scipy.stats import chi2

from moderndid.core.converters import (
    aggteresult_to_polars,
    dddaggresult_to_polars,
    dddmpresult_to_polars,
    didinterresult_to_polars,
    doseresult_to_polars,
    dynbalancinghistoryresult_to_polars,
    dynbalancingresult_to_polars,
    mpresult_to_polars,
    pteresult_to_polars,
    to_df,
)


class TestMpresultToPolars:
    def test_returns_dataframe(self, att_gt_analytical):
        df = mpresult_to_polars(att_gt_analytical)
        assert isinstance(df, pl.DataFrame)

    def test_has_expected_columns(self, att_gt_analytical):
        df = mpresult_to_polars(att_gt_analytical)
        for col in ("group", "time", "att", "se", "ci_lower", "ci_upper", "treatment_status"):
            assert col in df.columns

    def test_treatment_status_labels(self, att_gt_analytical):
        df = mpresult_to_polars(att_gt_analytical)
        assert set(df["treatment_status"].unique().to_list()) <= {"Pre", "Post"}


class TestAggteresultToPolars:
    def test_dynamic(self, aggte_dynamic):
        df = aggteresult_to_polars(aggte_dynamic)
        assert isinstance(df, pl.DataFrame)
        assert "event_time" in df.columns
        assert "treatment_status" in df.columns

    def test_group(self, aggte_group):
        df = aggteresult_to_polars(aggte_group)
        assert isinstance(df, pl.DataFrame)
        assert "event_time" in df.columns

    def test_calendar(self, aggte_calendar):
        df = aggteresult_to_polars(aggte_calendar)
        assert isinstance(df, pl.DataFrame)
        assert "event_time" in df.columns


class TestDoseresultToPolars:
    def test_att(self, cont_did_result):
        df = doseresult_to_polars(cont_did_result, effect_type="att")
        assert isinstance(df, pl.DataFrame)
        for col in ("dose", "effect", "se", "ci_lower", "ci_upper"):
            assert col in df.columns

    def test_acrt(self, cont_did_result):
        df = doseresult_to_polars(cont_did_result, effect_type="acrt")
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_invalid_effect_type(self, cont_did_result):
        with pytest.raises(ValueError, match="effect_type must be"):
            doseresult_to_polars(cont_did_result, effect_type="invalid")


class TestPteresultToPolars:
    def test_returns_dataframe(self, cont_did_event):
        df = pteresult_to_polars(cont_did_event)
        assert isinstance(df, pl.DataFrame)

    def test_has_expected_columns(self, cont_did_event):
        df = pteresult_to_polars(cont_did_event)
        for col in ("event_time", "att", "se", "ci_lower", "ci_upper", "treatment_status"):
            assert col in df.columns


class TestDddmpresultToPolars:
    def test_returns_dataframe(self, ddd_mp_result):
        df = dddmpresult_to_polars(ddd_mp_result)
        assert isinstance(df, pl.DataFrame)

    def test_has_expected_columns(self, ddd_mp_result):
        df = dddmpresult_to_polars(ddd_mp_result)
        for col in ("group", "time", "att", "se", "ci_lower", "ci_upper", "treatment_status"):
            assert col in df.columns


class TestDddaggresultToPolars:
    def test_eventstudy(self, ddd_agg_result):
        df = dddaggresult_to_polars(ddd_agg_result)
        assert isinstance(df, pl.DataFrame)
        assert "event_time" in df.columns


class TestDidinterresultToPolars:
    def test_returns_dataframe(self, didinter_result):
        df = didinterresult_to_polars(didinter_result)
        assert isinstance(df, pl.DataFrame)

    def test_has_expected_columns(self, didinter_result):
        df = didinterresult_to_polars(didinter_result)
        for col in ("horizon", "att", "se", "ci_lower", "ci_upper", "treatment_status"):
            assert col in df.columns


class TestToDf:
    def test_mpresult(self, att_gt_analytical):
        df = to_df(att_gt_analytical)
        expected = mpresult_to_polars(att_gt_analytical)
        assert df.equals(expected)

    def test_aggte_dynamic(self, aggte_dynamic):
        df = to_df(aggte_dynamic)
        expected = aggteresult_to_polars(aggte_dynamic)
        assert df.equals(expected)

    def test_aggte_group(self, aggte_group):
        df = to_df(aggte_group)
        expected = aggteresult_to_polars(aggte_group)
        assert df.equals(expected)

    def test_aggte_calendar(self, aggte_calendar):
        df = to_df(aggte_calendar)
        expected = aggteresult_to_polars(aggte_calendar)
        assert df.equals(expected)

    def test_doseresult(self, cont_did_result):
        df = to_df(cont_did_result)
        expected = doseresult_to_polars(cont_did_result)
        assert df.equals(expected)

    def test_doseresult_acrt(self, cont_did_result):
        df = to_df(cont_did_result, effect_type="acrt")
        expected = doseresult_to_polars(cont_did_result, effect_type="acrt")
        assert df.equals(expected)

    def test_pteresult(self, cont_did_event):
        df = to_df(cont_did_event)
        expected = pteresult_to_polars(cont_did_event)
        assert df.equals(expected)

    def test_dddmultiperiod(self, ddd_mp_result):
        df = to_df(ddd_mp_result)
        expected = dddmpresult_to_polars(ddd_mp_result)
        assert df.equals(expected)

    def test_dddagg(self, ddd_agg_result):
        df = to_df(ddd_agg_result)
        expected = dddaggresult_to_polars(ddd_agg_result)
        assert df.equals(expected)

    def test_didinter(self, didinter_result):
        df = to_df(didinter_result)
        expected = didinterresult_to_polars(didinter_result)
        assert df.equals(expected)

    def test_unsupported_type(self):
        with pytest.raises(TypeError, match="No converter for"):
            to_df("not a result")

    def test_unexpected_kwarg_for_converter(self, att_gt_analytical):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            to_df(att_gt_analytical, effect_type="att")

    def test_invalid_effect_type_via_to_df(self, cont_did_result):
        with pytest.raises(ValueError, match="effect_type must be"):
            to_df(cont_did_result, effect_type="invalid")

    def test_bogus_kwarg_for_dose(self, cont_did_result):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            to_df(cont_did_result, bogus="foo")


def test_dynbalancing_potential_outcome_bands_use_alpha(dyn_balancing_robust_result):
    df = dynbalancingresult_to_polars(dyn_balancing_robust_result)
    row = df.filter(pl.col("parameter") == "mu(ds1)").row(0, named=True)
    assert (row["ci_upper_robust"] - row["estimate"]) / row["se"] == pytest.approx(np.sqrt(chi2.ppf(0.9, 2)))


def test_dynbalancing_robust_bands_equal_gaussian_when_setting_off(dyn_balancing_robust_result):
    params = {**dyn_balancing_robust_result.estimation_params, "robust_quantile": False}
    result = dyn_balancing_robust_result._replace(
        robust_quantile=dyn_balancing_robust_result.gaussian_quantile, estimation_params=params
    )
    df = dynbalancingresult_to_polars(result)
    np.testing.assert_allclose(df["ci_lower_robust"].to_numpy(), df["ci_lower_gaussian"].to_numpy())
    np.testing.assert_allclose(df["ci_upper_robust"].to_numpy(), df["ci_upper_gaussian"].to_numpy())


@pytest.mark.parametrize("parameter", ["mu1", "mu2"])
def test_dynbalancing_history_potential_outcome_bands_use_their_own_degrees_of_freedom(
    dyn_balancing_robust_history_result, parameter
):
    df = dynbalancinghistoryresult_to_polars(dyn_balancing_robust_history_result, parameter=parameter)
    ratio = ((df["ci_upper_robust"] - df["estimate"]) / df["se"]).to_numpy()
    np.testing.assert_allclose(ratio, np.sqrt(chi2.ppf(0.9, [1, 2])))


def test_dynbalancing_history_ate_bands_use_ate_quantile(dyn_balancing_robust_history_result):
    df = dynbalancinghistoryresult_to_polars(dyn_balancing_robust_history_result)
    ratio = ((df["ci_upper_robust"] - df["estimate"]) / df["se"]).to_numpy()
    np.testing.assert_allclose(ratio, np.sqrt(chi2.ppf(0.9, [2, 4])))


def test_dynbalancing_history_bands_use_summary_quantile_without_results(dyn_balancing_robust_history_result):
    result = dyn_balancing_robust_history_result._replace(results=[])
    df = dynbalancinghistoryresult_to_polars(result, parameter="mu1")
    ratio = ((df["ci_upper_robust"] - df["estimate"]) / df["se"]).to_numpy()
    np.testing.assert_allclose(ratio, np.sqrt(chi2.ppf(0.9, [2, 4])))


def test_to_df_returns_integer_cohorts_and_periods_for_att_gt(att_gt_analytical):
    df = to_df(att_gt_analytical)
    assert df.schema["group"] == pl.Int64
    assert df.schema["time"] == pl.Int64
    assert sorted(df["group"].unique().to_list()) == [2004, 2006, 2007]
    assert sorted(df["time"].unique().to_list()) == [2004, 2005, 2006, 2007]


def test_to_df_returns_integer_event_times_for_dynamic_aggregation(aggte_dynamic):
    df = to_df(aggte_dynamic)
    assert df.schema["event_time"] == pl.Int64
    assert df["event_time"].to_list() == [-3, -2, -1, 0, 1, 2, 3]


@pytest.mark.parametrize(
    ("fixture_name", "expected"),
    [("aggte_group", [2004, 2006, 2007]), ("aggte_calendar", [2004, 2005, 2006, 2007])],
)
def test_to_df_returns_integer_labels_for_group_and_calendar_aggregations(fixture_name, expected, request):
    df = to_df(request.getfixturevalue(fixture_name))
    assert df.schema["event_time"] == pl.Int64
    assert df["event_time"].to_list() == expected


def test_to_df_returns_integer_horizons_for_did_multiplegt(didinter_result):
    df = to_df(didinter_result)
    assert df.schema["horizon"] == pl.Int64
    assert df["horizon"].to_list() == [-1, 1, 2]


def test_to_df_returns_integer_event_times_for_emfx(emfx_event_result):
    df = to_df(emfx_event_result)
    assert df.schema["event_time"] == pl.Int64
    assert df["event_time"].to_list() == [0, 1, 2]


def test_to_df_keeps_fractional_event_times_as_floats(aggte_fractional_event_times):
    df = to_df(aggte_fractional_event_times)
    assert df.schema["event_time"] == pl.Float64
    assert df["event_time"].to_list() == [-0.5, 0.5, 1.5]


def test_to_df_casts_event_times_after_dropping_reference_row(aggte_with_reference_row):
    df = to_df(aggte_with_reference_row)
    assert df.schema["event_time"] == pl.Int64
    assert df["event_time"].to_list() == [-2, 0]
    assert df["att"].to_list() == [0.05, 0.3]


@pytest.mark.parametrize(
    ("fixture_name", "aggregations"),
    [
        ("aggte_simple", "a dynamic, group, or calendar"),
        ("ddd_agg_simple", "an eventstudy, group, or calendar"),
        ("emfx_simple_result", "an event, group, or calendar"),
    ],
)
def test_simple_aggregation_error_names_the_aggregations_to_pass(fixture_name, aggregations, request):
    message = f"Simple aggregation does not produce event-level data for plotting. Pass {aggregations} aggregation."
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        to_df(request.getfixturevalue(fixture_name))
