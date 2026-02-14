"""Worker-side utilities for Dask distributed execution."""

from __future__ import annotations

import numpy as np
import polars as pl


def combine_partitions(*partition_dfs, group_col, sentinel, required_groups=None, time_col=None, times=None):
    """Concatenate partition Pandas DataFrames, restore inf sentinel, convert to Polars.

    Parameters
    ----------
    *partition_dfs : pd.DataFrame
        Pandas DataFrames from Dask partition Futures.
    group_col : str
        Name of the group column.
    sentinel : float or None
        Value that replaced ``inf`` in the group column.  If not None,
        rows with this value are restored to ``np.inf``.
    required_groups : list or None
        Group values to keep (using the original values, i.e. ``inf`` not
        sentinel).  Filters to only the groups needed for this cell.
    time_col : str or None
        Time column name for early filtering.
    times : list or None
        Time values to keep.  When provided with *time_col*, each partition
        is filtered before concatenation to reduce peak memory.

    Returns
    -------
    pl.DataFrame
        Combined Polars DataFrame.
    """
    time_set = set(times) if (time_col is not None and times is not None) else None

    # Build a filter set that uses sentinel values (matching persisted data)
    # so we can filter BEFORE the copy-heavy sentinel restoration.
    if required_groups is not None and sentinel is not None:
        filter_set = {sentinel if (isinstance(g, float) and not np.isfinite(g)) else g for g in required_groups}
    elif required_groups is not None:
        filter_set = set(required_groups)
    else:
        filter_set = None

    pandas_parts = []
    for pdf in partition_dfs:
        # Time filter first — biggest reduction for panel data with many periods.
        if time_set is not None:
            pdf = pdf.loc[pdf[time_col].isin(time_set)]
            if len(pdf) == 0:
                continue

        # Group filter using sentinel-aware values — no copy needed.
        if filter_set is not None:
            pdf = pdf.loc[pdf[group_col].isin(filter_set)]
            if len(pdf) == 0:
                continue

        pandas_parts.append(pdf)

    if not pandas_parts:
        return pl.DataFrame()

    import pandas as pd

    combined = pd.concat(pandas_parts, copy=False)
    del pandas_parts

    # Convert to Polars once and restore sentinel there (zero-copy replace).
    result = pl.from_pandas(combined)
    del combined
    if sentinel is not None:
        result = result.with_columns(
            pl.when(pl.col(group_col) == sentinel).then(np.inf).otherwise(pl.col(group_col)).alias(group_col)
        )

    return result


def filter_by_times(df, time_col, times):
    """Filter a Polars DataFrame to rows matching the given time periods.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    time_col : str
        Time column name.
    times : list
        Time period values to keep.

    Returns
    -------
    pl.DataFrame
        Filtered DataFrame.
    """
    return df.filter(pl.col(time_col).is_in(times))
