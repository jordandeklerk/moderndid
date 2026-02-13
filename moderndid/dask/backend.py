"""Core Dask DataFrame partitioning and execution logic."""

from __future__ import annotations

import numpy as np


def is_dask_dataframe(obj) -> bool:
    """Check if *obj* is a Dask DataFrame."""
    return type(obj).__module__.startswith("dask.dataframe") and type(obj).__name__ == "DataFrame"


def compute_dask_metadata(ddf, group_col, time_col, id_col=None, need_unique_ids=True):
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
    need_unique_ids : bool, default True
        If False, skip materializing all unique IDs (only compute ``n_units``).
        Set to False when only analytical SEs are needed.

    Returns
    -------
    dict
        Keys: ``tlist``, ``glist``, ``all_group_vals``, ``n_units``,
        ``unique_ids`` (None when *id_col* is None or *need_unique_ids* is False).
    """
    tlist = np.sort(ddf[time_col].unique().compute().to_numpy())

    glist_raw = ddf[group_col].unique().compute().to_numpy()
    all_group_vals = sorted(glist_raw)
    glist = np.sort([g for g in glist_raw if g > 0 and np.isfinite(g)])

    n_units = None
    unique_ids = None
    if id_col is not None:
        n_units = ddf[id_col].nunique().compute()
        if need_unique_ids:
            unique_ids = ddf[id_col].unique().compute().to_numpy()

    return {
        "tlist": tlist,
        "glist": glist,
        "all_group_vals": all_group_vals,
        "n_units": n_units,
        "unique_ids": unique_ids,
    }


def persist_by_group(client, ddf, group_col, all_group_vals=None):
    """Persist the DataFrame and build a group-to-partition mapping.

    Replaces ``inf`` in *group_col* with a sentinel value, persists the
    DataFrame **without shuffling**, and scans each partition to determine
    which groups it contains.  This avoids the expensive all-to-all shuffle
    that ``set_index`` triggers on large datasets.

    Parameters
    ----------
    client : distributed.Client
        Active Dask distributed client.
    ddf : dask.dataframe.DataFrame
        Input Dask DataFrame.  Should already be persisted for best
        performance (avoids redundant Parquet reads).
    group_col : str
        Column to partition by.
    all_group_vals : list, optional
        Pre-computed sorted list of all group values (from
        ``compute_dask_metadata``).  When provided, avoids redundant
        ``.compute()`` calls to discover group values.

    Returns
    -------
    tuple of (persisted_ddf, group_to_partitions, sentinel)
        *persisted_ddf* : the persisted Dask DataFrame.
        *group_to_partitions* : ``dict[group_value, list[int]]`` mapping each
        group value to its partition indices.
        *sentinel* : the value that replaced ``inf``, or ``None`` if no ``inf``
        values existed.
    """
    if all_group_vals is not None:
        finite_groups = np.array([g for g in all_group_vals if np.isfinite(g)])
        has_inf = any(not np.isfinite(g) for g in all_group_vals)
    else:
        finite_groups = ddf[group_col].loc[np.isfinite(ddf[group_col])].unique().compute().to_numpy()
        has_inf = np.isinf(ddf[group_col].max().compute())

    sentinel = None
    if has_inf:
        sentinel = float(np.max(finite_groups) + 1)
        ddf = ddf.map_partitions(_replace_inf_with_sentinel, group_col=group_col, sentinel=sentinel)

    from distributed import futures_of

    # Avoid duplicating a collection that is already persisted upstream.
    if not futures_of(ddf):
        ddf = client.persist(ddf)

    group_to_partitions = _scan_group_partitions(client, ddf, group_col, finite_groups, sentinel)

    return ddf, group_to_partitions, sentinel


def _replace_inf_with_sentinel(pdf, group_col, sentinel):
    """Worker-side: replace inf in *group_col* with *sentinel*."""
    pdf = pdf.copy()
    mask = np.isinf(pdf[group_col].values)
    if mask.any():
        pdf.loc[mask, group_col] = sentinel
    return pdf


def _scan_group_partitions(client, persisted_ddf, group_col, finite_groups, sentinel):
    """Build group-to-partitions mapping by scanning persisted partitions.

    Submits a lightweight function to each partition that returns its unique
    group values, then builds the reverse mapping.
    """
    from distributed import futures_of

    partition_futures = futures_of(persisted_ddf)

    scan_futures = [client.submit(_partition_unique_groups, fut, group_col, pure=False) for fut in partition_futures]
    partition_group_sets = client.gather(scan_futures)

    all_vals = set(finite_groups.tolist())
    if sentinel is not None:
        all_vals.add(sentinel)

    group_to_parts: dict[float, list[int]] = {g: [] for g in all_vals}
    for i, groups_in_part in enumerate(partition_group_sets):
        for g in groups_in_part:
            if g in group_to_parts:
                group_to_parts[g].append(i)

    return group_to_parts


def _partition_unique_groups(pdf, group_col):
    """Worker-side: return unique group values in a single partition."""
    return {float(g) for g in pdf[group_col].unique()}


def execute_cell_tasks(client, persisted_ddf, group_to_partitions, cell_specs, worker_fn):
    """Submit cell tasks with concurrency control, gather results, and clean up.

    Pins exactly one cell to each worker at a time using a sliding-window
    approach.  When a worker finishes its cell, the next queued cell is
    submitted to that same worker.  This guarantees at most one cell per
    worker regardless of the worker's ``nthreads`` setting, preventing OOM
    when individual cells require large fractions of the dataset.

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
    list
        One result per cell spec (in order).
    """
    from collections import deque

    from distributed import as_completed

    partition_futures = _get_partition_futures(persisted_ddf)
    worker_addrs = list(client.scheduler_info()["workers"].keys())

    n_tasks = len(cell_specs)
    results = [None] * n_tasks

    def _make_future(idx, worker):
        spec = cell_specs[idx]
        needed = set()
        for g in spec["required_groups"]:
            needed.update(group_to_partitions.get(g, []))
        part_futs = [partition_futures[i] for i in sorted(needed) if i < len(partition_futures)]
        if not part_futs:
            return client.submit(
                lambda **kw: None,
                **spec["cell_kwargs"],
                workers=[worker],
                allow_other_workers=False,
                pure=False,
            )
        return client.submit(
            worker_fn,
            *part_futs,
            **spec["cell_kwargs"],
            workers=[worker],
            allow_other_workers=False,
            pure=False,
        )

    try:
        remaining = deque(range(n_tasks))
        # future -> (cell_idx, worker_addr)
        active = {}

        # Submit initial batch: 1 cell per worker
        for worker in worker_addrs:
            if not remaining:
                break
            idx = remaining.popleft()
            fut = _make_future(idx, worker)
            active[fut] = (idx, worker)

        # Sliding window: when a worker finishes, give it the next cell
        ac = as_completed(active)
        for completed in ac:
            idx, worker = active.pop(completed)
            results[idx] = completed.result()

            if remaining:
                next_idx = remaining.popleft()
                new_fut = _make_future(next_idx, worker)
                active[new_fut] = (next_idx, worker)
                ac.add(new_fut)
    finally:
        cleanup_persisted(client, persisted_ddf)

    return results


def _get_partition_futures(persisted_ddf):
    """Extract per-partition Futures from a persisted Dask DataFrame."""
    from distributed import futures_of

    return futures_of(persisted_ddf)


def cleanup_persisted(client, persisted_ddf):
    """Cancel partition futures for a persisted Dask DataFrame.

    Parameters
    ----------
    client : distributed.Client
        Active distributed client.
    persisted_ddf : dask.dataframe.DataFrame
        The persisted Dask DataFrame whose partition futures should be
        cancelled.
    """
    try:
        partition_futures = _get_partition_futures(persisted_ddf)
        if partition_futures:
            client.cancel(partition_futures)
    except Exception:  # noqa: BLE001
        pass
