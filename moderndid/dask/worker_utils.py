"""Worker-side utilities for Dask distributed execution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl


def combine_partitions(*partition_dfs, group_col, sentinel, required_groups=None):
    """Concatenate partition Pandas DataFrames, restore inf sentinel, convert to Polars.

    Parameters
    ----------
    *partition_dfs : pd.DataFrame
        Pandas DataFrames from Dask partition Futures. The index is the
        group column (set by ``set_index``).
    group_col : str
        Name of the group column (stored in the index after ``set_index``).
    sentinel : float or None
        Value that replaced ``inf`` in the group column.  If not None,
        rows with this value are restored to ``np.inf``.
    required_groups : list or None
        Group values to keep (using the original values, i.e. ``inf`` not
        sentinel).  Partitions from ``set_index`` may contain multiple
        groups; this filters to only the groups needed for this cell.

    Returns
    -------
    pl.DataFrame
        Combined Polars DataFrame with the group column restored as a
        regular column.
    """
    combined = pd.concat(partition_dfs, ignore_index=False)

    # Restore group column from index
    combined = combined.reset_index()

    if sentinel is not None:
        mask = combined[group_col] == sentinel
        if mask.any():
            combined.loc[mask, group_col] = np.inf

    df = pl.from_pandas(combined)

    # Filter to only the required groups for this cell
    if required_groups is not None:
        df = df.filter(pl.col(group_col).is_in(required_groups))

    return df


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
