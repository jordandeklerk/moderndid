"""DataFrame compatibility layer for Arrow-compatible DataFrames."""

from typing import Any

import narwhals as nw
import polars as pl

DataFrame = Any  # Any object that to_polars accepts


def to_polars(df):
    """Convert any Arrow-compatible DataFrame to polars.

    Parameters
    ----------
    df : Any
        Input DataFrame. Supports any object implementing the Arrow PyCapsule
        Interface (``__arrow_c_stream__``), such as polars and pandas (2.2+)
        DataFrames, duckdb results, pyarrow Tables, ibis expressions, and
        cudf DataFrames. A pandas DataFrame older than 2.2 converts through
        narwhals.

    Returns
    -------
    pl.DataFrame
        Polars DataFrame.
    """
    if isinstance(df, pl.DataFrame):
        return df

    msg = f"Expected object implementing '__arrow_c_stream__' or a pandas DataFrame, got: {type(df).__name__}"
    # Since a pandas 3 Series also exports an Arrow stream, a Series other than a polars one stops here before the
    # PyCapsule branch reads it as a table.
    series = nw.from_native(df, series_only=True, pass_through=True)
    if isinstance(series, nw.Series) and not isinstance(df, pl.Series):
        raise TypeError(msg)

    # Arrow PyCapsule Interface
    if hasattr(df, "__arrow_c_stream__"):
        return nw.from_arrow(df, backend=pl).to_native()

    # Since pandas before 2.2 has no PyCapsule interface, narwhals reads its eager DataFrames instead.
    frame = nw.from_native(df, eager_only=True, pass_through=True)
    if isinstance(frame, nw.DataFrame):
        return frame.to_polars()

    raise TypeError(msg)


def from_polars(result, like):
    """Convert polars DataFrame back to the format of ``like``.

    Parameters
    ----------
    result : pl.DataFrame
        Polars DataFrame to convert.
    like : Any
        Original input whose type determines the output format.

    Returns
    -------
    Any
        DataFrame in the same format as ``like``.
    """
    if isinstance(like, pl.DataFrame):
        return result

    # Since pandas before 2.2 has no PyCapsule interface, narwhals decides whether like is an eager DataFrame.
    frame = nw.from_native(like, eager_only=True, pass_through=True)
    if isinstance(frame, nw.DataFrame):
        return nw.from_arrow(result.to_arrow(), backend=nw.get_native_namespace(frame)).to_native()

    msg = f"Cannot convert back to {type(like).__name__}"
    raise TypeError(msg)
