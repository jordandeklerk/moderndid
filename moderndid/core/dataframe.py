"""DataFrame compatibility layer for pandas/polars interoperability."""

from typing import Any

import narwhals as nw
import polars as pl

DataFrame = Any  # Any object implementing __arrow_c_stream__


def to_polars(df: Any) -> pl.DataFrame:
    """Convert any Arrow-compatible DataFrame to polars.

    Parameters
    ----------
    df : Any
        Input DataFrame. Supports any object implementing the Arrow PyCapsule
        Interface (__arrow_c_stream__), including:
        - polars DataFrame
        - pandas DataFrame (2.0+)
        - duckdb results
        - pyarrow Table
        - ibis expressions
        - cudf DataFrame

    Returns
    -------
    pl.DataFrame
        Polars DataFrame.

    Raises
    ------
    TypeError
        If input doesn't implement __arrow_c_stream__.
    """
    if isinstance(df, pl.DataFrame):
        return df

    # Arrow PyCapsule Interface
    if hasattr(df, "__arrow_c_stream__"):
        return nw.from_arrow(df, backend=pl).to_native()

    msg = f"Expected object implementing '__arrow_c_stream__', got: {type(df).__name__}"
    raise TypeError(msg)
