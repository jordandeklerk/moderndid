"""Pre-partition utilities for group-time cell computation."""

from __future__ import annotations

import polars as pl


def build_gt_partitions(data, group_col, time_col):
    """Partition data by (group, time) with a single pass.

    Parameters
    ----------
    data : pl.DataFrame
        Input data.
    group_col : str
        Name of the group column.
    time_col : str
        Name of the time column.

    Returns
    -------
    dict[tuple, pl.DataFrame]
        Mapping from (group_val, time_val) to sub-DataFrame.
    """
    return data.partition_by(group_col, time_col, as_dict=True, maintain_order=True)


def concat_partitions(partitions, group_vals, time_vals):
    """Concatenate partitions for given (group, time) combinations.

    Parameters
    ----------
    partitions : dict[tuple, pl.DataFrame]
        Partition dictionary from :func:`build_gt_partitions`.
    group_vals : list
        Group values to include.
    time_vals : list
        Time values to include.

    Returns
    -------
    pl.DataFrame or None
        Concatenated data, or None if no matching partitions exist.
    """
    parts = [partitions[k] for g in group_vals for t in time_vals if (k := (g, t)) in partitions]
    if not parts:
        return None
    return pl.concat(parts)
