"""Core Dask DataFrame partitioning and execution logic."""

from __future__ import annotations

import numpy as np


def is_dask_dataframe(obj) -> bool:
    """Check if *obj* is a Dask DataFrame."""
    return type(obj).__module__.startswith("dask.dataframe") and type(obj).__name__ == "DataFrame"


def compute_dask_metadata(ddf, group_col, time_col, id_col=None):
    """Compute metadata from a Dask DataFrame via distributed aggregations.

    Parameters
    ----------
    ddf : dask.dataframe.DataFrame
        Distributed input data.
    group_col : str
        Treatment group column name.
    time_col : str
        Time period column name.
    id_col : str, optional
        Unit identifier column name.

    Returns
    -------
    dict
        Keys: ``tlist``, ``glist``, ``all_group_vals``, ``n_units``,
        ``unique_ids`` (None when *id_col* is None).
    """
    tlist = np.sort(ddf[time_col].unique().compute().to_numpy())

    glist_raw = ddf[group_col].unique().compute().to_numpy()
    all_group_vals = sorted(glist_raw)
    glist = np.sort([g for g in glist_raw if g > 0 and np.isfinite(g)])

    n_units = None
    unique_ids = None
    if id_col is not None:
        n_units = ddf[id_col].nunique().compute()
        unique_ids = ddf[id_col].unique().compute().to_numpy()

    return {
        "tlist": tlist,
        "glist": glist,
        "all_group_vals": all_group_vals,
        "n_units": n_units,
        "unique_ids": unique_ids,
    }


def persist_by_group(client, ddf, group_col):
    """Replace ``inf`` with a sentinel, ``set_index`` by *group_col*, and persist.

    Never-treated units have ``group = inf`` which is incompatible with
    ``set_index``.  This function replaces ``inf`` with a sentinel value
    one greater than the largest finite group value, persists the shuffled
    DataFrame, and returns the mapping from group values to partition indices.

    Parameters
    ----------
    client : distributed.Client
        Active Dask distributed client.
    ddf : dask.dataframe.DataFrame
        Input Dask DataFrame.
    group_col : str
        Column to partition by.

    Returns
    -------
    tuple of (persisted_ddf, group_to_partitions, sentinel)
        *persisted_ddf* : the persisted Dask DataFrame indexed by *group_col*.
        *group_to_partitions* : ``dict[group_value, list[int]]`` mapping each
        group value to its partition indices.
        *sentinel* : the value that replaced ``inf``, or ``None`` if no ``inf``
        values existed.
    """
    finite_groups = ddf[group_col].loc[np.isfinite(ddf[group_col])].unique().compute().to_numpy()

    sentinel = None
    if np.any(np.isinf(ddf[group_col].compute().to_numpy())):
        sentinel = float(np.max(finite_groups) + 1)
        ddf = ddf.map_partitions(_replace_inf_with_sentinel, group_col=group_col, sentinel=sentinel)

    ddf = ddf.set_index(group_col, sorted=False)
    ddf = client.persist(ddf)

    divisions = ddf.divisions
    group_to_partitions = _build_group_to_partitions(divisions, sentinel, finite_groups)

    return ddf, group_to_partitions, sentinel


def _replace_inf_with_sentinel(pdf, group_col, sentinel):
    """Worker-side: replace inf in *group_col* with *sentinel*."""
    pdf = pdf.copy()
    mask = np.isinf(pdf[group_col].values)
    if mask.any():
        pdf.loc[mask, group_col] = sentinel
    return pdf


def _build_group_to_partitions(divisions, sentinel, finite_groups):
    """Build a mapping from group value to list of partition indices.

    After ``set_index``, ``ddf.divisions`` gives the boundary values for each
    partition.  A group value ``g`` falls in partition ``i`` if
    ``divisions[i] <= g < divisions[i+1]`` (last partition uses ``<=``).
    """
    group_to_parts: dict[float, list[int]] = {}
    n_partitions = len(divisions) - 1

    all_vals = list(finite_groups)
    if sentinel is not None:
        all_vals.append(sentinel)

    for g in all_vals:
        parts = []
        for i in range(n_partitions):
            lo, hi = divisions[i], divisions[i + 1]
            if lo is None or hi is None:
                parts.append(i)
                continue
            if i == n_partitions - 1:
                if lo <= g <= hi:
                    parts.append(i)
            else:
                if lo <= g < hi:
                    parts.append(i)
        if not parts:
            # Fallback: scan all partitions
            parts = list(range(n_partitions))
        group_to_parts[g] = parts

    return group_to_parts


def submit_cell_tasks(client, persisted_ddf, group_to_partitions, cell_specs, worker_fn):
    """Submit tasks for all (g,t) cells with relevant partition Futures.

    Parameters
    ----------
    client : distributed.Client
        Active distributed client.
    persisted_ddf : dask.dataframe.DataFrame
        Persisted Dask DataFrame indexed by group column.
    group_to_partitions : dict
        Mapping from group values to partition indices.
    cell_specs : list of dict
        Each dict has keys ``required_groups`` (list of group values needed)
        and ``cell_kwargs`` (keyword arguments for the worker function).
    worker_fn : callable
        Worker function signature:
        ``worker_fn(*partition_dfs, **cell_kwargs)``.

    Returns
    -------
    list of Future
        One result Future per cell spec.
    """
    partition_futures = _get_partition_futures(persisted_ddf)

    result_futures = []
    for spec in cell_specs:
        required_groups = spec["required_groups"]
        cell_kwargs = spec["cell_kwargs"]

        needed_indices = set()
        for g in required_groups:
            needed_indices.update(group_to_partitions.get(g, []))

        part_futs = [partition_futures[i] for i in sorted(needed_indices) if i < len(partition_futures)]

        if not part_futs:
            result_futures.append(client.submit(lambda **kw: None, **cell_kwargs, pure=False))
        else:
            result_futures.append(client.submit(worker_fn, *part_futs, **cell_kwargs, pure=False))

    return result_futures


def _get_partition_futures(persisted_ddf):
    """Extract per-partition Futures from a persisted Dask DataFrame."""
    from distributed import futures_of

    return futures_of(persisted_ddf)


def gather_and_cleanup(client, result_futures, persisted_ddf):
    """Gather results and cancel partition futures.

    Parameters
    ----------
    client : distributed.Client
        Active distributed client.
    result_futures : list of Future
        Futures from ``submit_cell_tasks``.
    persisted_ddf : dask.dataframe.DataFrame
        The persisted Dask DataFrame whose partition futures should be
        cancelled after gathering.

    Returns
    -------
    list
        Gathered results, one per cell.
    """
    try:
        results = client.gather(result_futures)
    finally:
        partition_futures = _get_partition_futures(persisted_ddf)
        if partition_futures:
            client.cancel(partition_futures)
    return results
