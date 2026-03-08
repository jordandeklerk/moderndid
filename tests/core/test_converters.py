"""Tests for core result converters and to_df dispatch."""

import polars as pl
import pytest

from moderndid.core.converters import (
    aggteresult_to_polars,
    dddaggresult_to_polars,
    dddmpresult_to_polars,
    didinterresult_to_polars,
    doseresult_to_polars,
    honestdid_to_polars,
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


class TestHonestdidToPolars:
    def test_returns_dataframe(self, honest_did_result):
        df = honestdid_to_polars(honest_did_result)
        assert isinstance(df, pl.DataFrame)

    def test_has_expected_columns(self, honest_did_result):
        df = honestdid_to_polars(honest_did_result)
        for col in ("param_value", "method", "lb", "ub", "midpoint"):
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

    def test_honestdid(self, honest_did_result):
        df = to_df(honest_did_result)
        expected = honestdid_to_polars(honest_did_result)
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
