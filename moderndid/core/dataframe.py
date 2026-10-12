"""DataFrame compatibility layer for Arrow-compatible DataFrames."""

from typing import Any

import narwhals as nw
import polars as pl

DataFrame = Any  # Any object implementing __arrow_c_stream__


def to_polars(df):
    """Convert any Arrow-compatible DataFrame to polars.

    Parameters
    ----------
    df : Any
        Input DataFrame. Supports any object implementing the Arrow PyCapsule
        Interface (``__arrow_c_stream__``), such as polars and pandas (2.0+)
        DataFrames, duckdb results, pyarrow Tables, ibis expressions, and
        cudf DataFrames.

    Returns
    -------
    pl.DataFrame
        Polars DataFrame.
    """
    if isinstance(df, pl.DataFrame):
        return df

    # Arrow PyCapsule Interface
    if hasattr(df, "__arrow_c_stream__"):
        return nw.from_arrow(df, backend=pl).to_native()

    msg = f"Expected object implementing '__arrow_c_stream__', got: {type(df).__name__}"
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

    if hasattr(like, "__arrow_c_stream__"):
        native_ns = nw.get_native_namespace(nw.from_native(like, eager_only=True))
        return nw.from_arrow(result.to_arrow(), backend=native_ns).to_native()

    msg = f"Cannot convert back to {type(like).__name__}"
    raise TypeError(msg)
