"""Test the dataframe conversion functions."""

import re

import numpy as np
import pytest

from tests.helpers import importorskip

pd = importorskip("pandas")
pl = importorskip("polars")
pa = importorskip("pyarrow")

from polars.testing import assert_frame_equal

from moderndid import load_mpdta
from moderndid.core.dataframe import from_polars, to_polars


class TestPolarsInput:
    def test_passthrough(self):
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = to_polars(df)
        assert result is df

    def test_series_converts_to_dataframe(self):
        s = pl.Series("x", [1, 2, 3])
        result = to_polars(s)
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["x"]
        assert result["x"].to_list() == [1, 2, 3]

    def test_lazy_frame_raises(self):
        lf = pl.LazyFrame({"a": [1, 2, 3]})
        with pytest.raises(TypeError):
            to_polars(lf)


class TestPandasInput:
    def test_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = to_polars(df)
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["a", "b"]
        assert result["a"].to_list() == [1, 2, 3]

    def test_preserves_int_dtype(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = to_polars(df)
        assert result["x"].dtype == pl.Int64

    def test_preserves_float_dtype(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = to_polars(df)
        assert result["x"].dtype == pl.Float64

    def test_preserves_string_dtype(self):
        df = pd.DataFrame({"x": ["a", "b", "c"]})
        result = to_polars(df)
        assert result["x"].dtype == pl.String

    def test_preserves_bool_dtype(self):
        df = pd.DataFrame({"x": [True, False, True]})
        result = to_polars(df)
        assert result["x"].dtype == pl.Boolean

    def test_with_nulls(self):
        df = pd.DataFrame({"x": [1.0, None, 3.0]})
        result = to_polars(df)
        assert result["x"].null_count() == 1


class TestPyArrowInput:
    def test_table(self):
        table = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = to_polars(table)
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["a", "b"]
        assert result["a"].to_list() == [1, 2, 3]

    def test_record_batch(self):
        batch = pa.RecordBatch.from_pydict({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = to_polars(batch)
        assert isinstance(result, pl.DataFrame)
        assert result["a"].to_list() == [1, 2, 3]

    def test_chunked_array_table(self):
        chunked = pa.chunked_array([[1, 2], [3, 4]])
        table = pa.table({"x": chunked})
        result = to_polars(table)
        assert result["x"].to_list() == [1, 2, 3, 4]


class TestDuckDBInput:
    def test_arrow_table_from_query(self, duckdb_connection):
        arrow = duckdb_connection.execute("SELECT 1 as a, 2 as b").fetch_arrow_table()
        result = to_polars(arrow)
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["a", "b"]

    def test_arrow_table_with_multiple_rows(self, duckdb_connection):
        arrow = duckdb_connection.execute("SELECT * FROM range(10) AS t(x)").fetch_arrow_table()
        result = to_polars(arrow)
        assert len(result) == 10


class TestInvalidInput:
    def test_list_raises(self):
        with pytest.raises(TypeError, match="__arrow_c_stream__"):
            to_polars([1, 2, 3])

    def test_dict_raises(self):
        with pytest.raises(TypeError, match="__arrow_c_stream__"):
            to_polars({"a": [1, 2, 3]})

    def test_none_raises(self):
        with pytest.raises(TypeError, match="__arrow_c_stream__"):
            to_polars(None)

    def test_numpy_array_raises(self):
        with pytest.raises(TypeError, match="__arrow_c_stream__"):
            to_polars(np.array([1, 2, 3]))

    def test_string_raises(self):
        with pytest.raises(TypeError, match="__arrow_c_stream__"):
            to_polars("not a dataframe")

    @pytest.mark.xfail(
        hasattr(pd.Series, "__arrow_c_stream__"),
        raises=pl.exceptions.SchemaError,
        strict=True,
        reason="a pandas 3 Series takes the PyCapsule branch and raises polars' SchemaError in place of TypeError",
    )
    def test_pandas_series_raises(self):
        s = pd.Series([1, 2, 3])
        with pytest.raises(TypeError, match="__arrow_c_stream__"):
            to_polars(s)


@pytest.mark.skipif(hasattr(pd.DataFrame, "__arrow_c_stream__"), reason="needs pandas older than 2.2")
def test_to_polars_reads_old_pandas_like_polars_from_pandas(mixed_pandas_data):
    result = to_polars(mixed_pandas_data)

    assert isinstance(result, pl.DataFrame)
    assert_frame_equal(result, pl.from_pandas(mixed_pandas_data))
    assert result.null_count().row(0) == tuple(mixed_pandas_data.isna().sum())


@pytest.mark.skipif(hasattr(pd.DataFrame, "__arrow_c_stream__"), reason="needs pandas older than 2.2")
def test_to_polars_reads_the_county_data_from_old_pandas(mpdta, mpdta_pandas):
    result = to_polars(mpdta_pandas)

    assert_frame_equal(result, mpdta)


@pytest.mark.skipif(hasattr(pd.DataFrame, "__arrow_c_stream__"), reason="needs pandas older than 2.2")
def test_from_polars_returns_old_pandas_with_the_changed_rows(mpdta_pandas):
    expected = mpdta_pandas[mpdta_pandas["year"] == 2004].reset_index(drop=True)

    result = from_polars(to_polars(mpdta_pandas).filter(pl.col("year") == 2004), mpdta_pandas)

    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.skipif(hasattr(pd.DataFrame, "__arrow_c_stream__"), reason="needs pandas older than 2.2")
def test_load_mpdta_returns_the_county_data_on_old_pandas():
    result = load_mpdta()

    assert isinstance(result, pl.DataFrame)
    assert result.schema == {
        "year": pl.Int64,
        "countyreal": pl.Int64,
        "lpop": pl.Float64,
        "lemp": pl.Float64,
        "first.treat": pl.Int64,
        "treat": pl.Int64,
    }
    assert result["countyreal"].n_unique() == 500
    assert (result.group_by("countyreal").len()["len"] == 5).all()
    assert result["year"].unique().sort().to_list() == [2003, 2004, 2005, 2006, 2007]
    assert result["first.treat"].unique().sort().to_list() == [0, 2004, 2006, 2007]
    assert result["treat"].to_list() == (result["first.treat"] > 0).cast(pl.Int64).to_list()


def test_to_polars_error_names_the_accepted_inputs():
    with pytest.raises(TypeError, match=re.escape("'__arrow_c_stream__' or a pandas DataFrame, got: list")):
        to_polars([1, 2, 3])


def test_from_polars_rejects_a_like_it_cannot_rebuild():
    with pytest.raises(TypeError, match="Cannot convert back to list"):
        from_polars(pl.DataFrame({"a": [1, 2, 3]}), [1, 2, 3])
