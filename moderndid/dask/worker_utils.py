"""Worker-side utilities for Dask distributed execution."""

from __future__ import annotations

import numpy as np
import polars as pl


def combine_partitions(*partition_dfs, group_col, sentinel, required_groups=None, time_col=None, times=None):
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
    time_col : str or None
        Time column name for early filtering.
    times : list or None
        Time values to keep.  When provided with *time_col*, each partition
        is filtered before concatenation to reduce peak memory.

    Returns
    -------
    pl.DataFrame
        Combined Polars DataFrame with the group column restored as a
        regular column.
    """
    # Convert each partition to Polars individually to avoid holding a full
    # pandas concat alongside the full Polars DataFrame (saves ~N bytes peak).
    time_set = set(times) if (time_col is not None and times is not None) else None
    group_set = set(required_groups) if required_groups is not None else None

    polars_parts = []
    for pdf in partition_dfs:
        # Early time filter on the pandas partition (before any copies)
        if time_set is not None:
            pdf = pdf.loc[pdf[time_col].isin(time_set)]
            if len(pdf) == 0:
                continue

        pdf = pdf.reset_index()

        if sentinel is not None:
            mask = pdf[group_col] == sentinel
            if mask.any():
                pdf.loc[mask, group_col] = np.inf

        # Early group filter on the pandas partition
        if group_set is not None:
            pdf = pdf.loc[pdf[group_col].isin(group_set)]
            if len(pdf) == 0:
                continue

        polars_parts.append(pl.from_pandas(pdf))

    if not polars_parts:
        return pl.DataFrame()

    return pl.concat(polars_parts)


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
