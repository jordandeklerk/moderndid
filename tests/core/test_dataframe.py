# pylint: disable=no-self-use
"""Test the dataframe conversion functions."""

import numpy as np
import pytest

from tests.helpers import importorskip

pd = importorskip("pandas")
pl = importorskip("polars")
pa = importorskip("pyarrow")
duckdb = importorskip("duckdb")

from moderndid.core.dataframe import to_polars


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
        batch = pa.record_batch({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = to_polars(batch)
        assert isinstance(result, pl.DataFrame)
        assert result["a"].to_list() == [1, 2, 3]

    def test_chunked_array_table(self):
        chunked = pa.chunked_array([[1, 2], [3, 4]])
        table = pa.table({"x": chunked})
        result = to_polars(table)
        assert result["x"].to_list() == [1, 2, 3, 4]


class TestDuckDBInput:
    def test_arrow_table_from_query(self):
        conn = duckdb.connect()
        arrow = conn.execute("SELECT 1 as a, 2 as b").fetch_arrow_table()
        result = to_polars(arrow)
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["a", "b"]

    def test_arrow_table_with_multiple_rows(self):
        conn = duckdb.connect()
        arrow = conn.execute("SELECT * FROM range(10) AS t(x)").fetch_arrow_table()
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

    def test_pandas_series_raises(self):
        s = pd.Series([1, 2, 3])
        with pytest.raises(TypeError, match="__arrow_c_stream__"):
            to_polars(s)
