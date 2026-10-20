"""Tests for converting didml results with to_df."""

import re

import numpy as np
import polars as pl
import pytest

from moderndid import to_df


def test_to_df_on_a_didml_fit_returns_its_group_time_cells(mpdta_analytic_fit):
    fit = mpdta_analytic_fit
    margin = fit.critical_value * fit.se_gt

    df = to_df(fit)

    assert df.columns == ["group", "time", "att", "se", "ci_lower", "ci_upper", "treatment_status"]
    assert df.schema["group"] == pl.Int64
    assert df.schema["time"] == pl.Int64
    np.testing.assert_array_equal(df["group"].to_numpy(), fit.groups)
    np.testing.assert_array_equal(df["time"].to_numpy(), fit.times)
    np.testing.assert_allclose(df["att"].to_numpy(), fit.att_gt, rtol=1e-12)
    np.testing.assert_allclose(df["se"].to_numpy(), fit.se_gt, rtol=1e-12)
    np.testing.assert_allclose(df["ci_lower"].to_numpy(), fit.att_gt - margin, rtol=1e-12)
    np.testing.assert_allclose(df["ci_upper"].to_numpy(), fit.att_gt + margin, rtol=1e-12)
    np.testing.assert_array_equal(df["treatment_status"].to_numpy(), np.where(fit.times < fit.groups, "Pre", "Post"))


def test_to_df_on_a_didml_dynamic_aggregation_returns_its_event_times(mpdta_dynamic_agg):
    event_times = mpdta_dynamic_agg.event_times
    agg = mpdta_dynamic_agg._replace(
        se_by_event=np.where(event_times == -1, np.nan, mpdta_dynamic_agg.se_by_event),
        critical_values=np.linspace(2.0, 2.6, len(event_times)),
    )
    keep = np.isfinite(agg.se_by_event)
    margin = agg.critical_values[keep] * agg.se_by_event[keep]

    df = to_df(agg)

    assert df.columns == ["event_time", "att", "se", "ci_lower", "ci_upper", "treatment_status"]
    assert df.height == len(event_times) - 1
    assert df.schema["event_time"] == pl.Int64
    np.testing.assert_array_equal(df["event_time"].to_numpy(), event_times[keep])
    np.testing.assert_allclose(df["att"].to_numpy(), agg.att_by_event[keep], rtol=1e-12)
    np.testing.assert_allclose(df["se"].to_numpy(), agg.se_by_event[keep], rtol=1e-12)
    np.testing.assert_allclose(df["ci_lower"].to_numpy(), agg.att_by_event[keep] - margin, rtol=1e-12)
    np.testing.assert_allclose(df["ci_upper"].to_numpy(), agg.att_by_event[keep] + margin, rtol=1e-12)
    np.testing.assert_array_equal(df["treatment_status"].to_numpy(), np.where(event_times[keep] < 0, "Pre", "Post"))


def test_to_df_on_a_simple_didml_aggregation_raises(mpdta_simple_agg):
    with pytest.raises(ValueError, match="Simple aggregation"):
        to_df(mpdta_simple_agg)


def test_to_df_names_the_didml_class_when_an_aggregation_lacks_event_times(mpdta_dynamic_agg):
    agg = mpdta_dynamic_agg._replace(event_times=None)
    message = "DIDMLAggResult with aggregation_type='dynamic' must have event_times, att_by_event, and se_by_event"

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        to_df(agg)


def test_to_df_on_a_blp_result_has_one_row_per_event_time_and_covariate(het_blp):
    names = list(het_blp.coefs)
    estimate = np.column_stack([het_blp.coefs[name] for name in names]).ravel()
    se = np.column_stack([het_blp.ses[name] for name in names]).ravel()
    pvalue = np.column_stack([het_blp.pvalues[name] for name in names]).ravel()

    df = to_df(het_blp)

    assert df.columns == ["event_time", "covariate", "estimate", "se", "pvalue"]
    assert df.schema["event_time"] == pl.Int64
    assert df["event_time"].to_list() == [0, 0, 1, 1]
    assert df["covariate"].to_list() == ["Intercept", "x", "Intercept", "x"]
    np.testing.assert_allclose(df["estimate"].to_numpy(), estimate, rtol=1e-12)
    np.testing.assert_allclose(df["se"].to_numpy(), se, rtol=1e-12)
    np.testing.assert_allclose(df["pvalue"].to_numpy(), pvalue, rtol=1e-12)


def test_to_df_on_an_empty_blp_result_has_no_rows_and_typed_columns(het_blp):
    empty = het_blp._replace(coefs={}, ses={}, pvalues={})

    df = to_df(empty)

    assert df.height == 0
    assert df.columns == ["event_time", "covariate", "estimate", "se", "pvalue"]
    assert df.dtypes == [pl.Int64, pl.String, pl.Float64, pl.Float64, pl.Float64]


def test_to_df_on_a_clan_result_has_one_row_per_covariate(het_clan):
    df = to_df(het_clan)

    assert df.columns == ["covariate", "high_mean", "low_mean", "diff", "pvalue"]
    assert df["covariate"].to_list() == het_clan.affected
    np.testing.assert_allclose(df["high_mean"].to_numpy(), het_clan.high_means, rtol=1e-12)
    np.testing.assert_allclose(df["low_mean"].to_numpy(), het_clan.low_means, rtol=1e-12)
    np.testing.assert_allclose(df["diff"].to_numpy(), het_clan.diffs, rtol=1e-12)
    np.testing.assert_allclose(df["pvalue"].to_numpy(), het_clan.pvalues, rtol=1e-12)
