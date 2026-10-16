"""Tests for converting honest_did results to data frames."""

import polars as pl

from moderndid.core.converters import honestdid_to_polars, to_df


def test_honestdid_to_polars_returns_dataframe(honest_did_result):
    df = honestdid_to_polars(honest_did_result)
    assert isinstance(df, pl.DataFrame)


def test_honestdid_to_polars_has_expected_columns(honest_did_result):
    df = honestdid_to_polars(honest_did_result)
    for col in ("param_value", "method", "lb", "ub", "midpoint"):
        assert col in df.columns


def test_to_df_dispatches_honest_did_results(honest_did_result):
    df = to_df(honest_did_result)
    expected = honestdid_to_polars(honest_did_result)
    assert df.equals(expected)


def test_to_df_keeps_sensitivity_parameter_as_float(honest_did_result):
    df = to_df(honest_did_result)
    assert df.schema["param_value"] == pl.Float64
    assert sorted(df["param_value"].unique().to_list()) == [-1.0, 0.0, 1.0]
