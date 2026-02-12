"""DataFrame compatibility layer for Arrow-compatible DataFrames."""

from typing import Any

import narwhals as nw
import polars as pl

DataFrame = Any  # Any object implementing __arrow_c_stream__

_SUPPORTED_TYPES = (
    "polars.DataFrame, pandas.DataFrame, pyarrow.Table, "
    "duckdb.DuckDBPyRelation, cudf.DataFrame, or any object "
    "implementing __arrow_c_stream__"
)


def to_polars(df: Any) -> pl.DataFrame:
    """Convert any Arrow-compatible DataFrame to polars.

    Parameters
    ----------
    df : Any
        Input DataFrame. Supported types:

        - polars DataFrame
        - pandas DataFrame (any version)
        - pyarrow Table
        - duckdb relation
        - cudf DataFrame
        - Any object implementing ``__arrow_c_stream__``

    Returns
    -------
    pl.DataFrame
        Polars DataFrame.

    Raises
    ------
    TypeError
        If input is not a supported DataFrame type.
    """
    if isinstance(df, pl.DataFrame):
        return df

    # Dask DataFrames must use the distributed path, not be collected.
    from moderndid.dask import is_dask_dataframe

    if is_dask_dataframe(df):
        raise TypeError(
            "Cannot convert a Dask DataFrame to Polars via to_polars(). "
            "Pass the Dask DataFrame directly to the estimator (e.g. att_gt, ddd, cont_did) "
            "with backend='dask' to use the distributed execution path."
        )

    # PyArrow, DuckDB >= 0.8, cuDF, and pandas >= 3.0 all expose
    # __arrow_c_stream__.  We route through narwhals for a uniform
    # Arrow-to-polars conversion.
    if hasattr(df, "__arrow_c_stream__"):
        return nw.from_arrow(df, backend=pl).to_native()

    # pandas only gained __arrow_c_stream__ in 3.0.0 (Jan 2026).
    # Most production environments (Databricks, Colab, CI images) still
    # ship pandas 2.x, so we detect pandas DataFrames by module name and
    # convert via pl.from_pandas().  The _is_pandas check avoids importing
    # pandas as a required dependency.
    if _is_pandas(df):
        return pl.from_pandas(df)

    msg = f"Cannot convert {type(df).__module__}.{type(df).__name__} to polars. Supported types: {_SUPPORTED_TYPES}"
    raise TypeError(msg)


def from_polars(result: pl.DataFrame, like: Any) -> Any:
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

    Raises
    ------
    TypeError
        If ``like`` is not a recognized DataFrame type.
    """
    if isinstance(like, pl.DataFrame):
        return result

    if hasattr(like, "__arrow_c_stream__"):
        native_ns = nw.get_native_namespace(nw.from_native(like, eager_only=True))
        return nw.from_arrow(result.to_arrow(), backend=native_ns).to_native()

    if _is_pandas(like):
        return result.to_pandas()

    msg = (
        f"Cannot convert result back to {type(like).__module__}.{type(like).__name__}. "
        f"Supported types: {_SUPPORTED_TYPES}"
    )
    raise TypeError(msg)


def _is_pandas(df: Any) -> bool:
    """Check if ``df`` is a pandas DataFrame without importing pandas."""
    return type(df).__module__.startswith("pandas") and type(df).__name__ == "DataFrame"
