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


def execute_cell_tasks(client, persisted_ddf, group_to_partitions, cell_specs, worker_fn):
    """Submit cell tasks with concurrency control, gather results, and clean up.

    Uses a sliding-window approach via ``as_completed`` to limit the number of
    simultaneously active tasks to ``2 * n_workers``.  This prevents worker OOM
    when individual cells require large fractions of the dataset (e.g. a large
    never-treated control group).

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
    from distributed import as_completed

    partition_futures = _get_partition_futures(persisted_ddf)
    n_workers = len(client.scheduler_info()["workers"])
    max_concurrent = max(n_workers * 2, 1)

    n_tasks = len(cell_specs)
    results = [None] * n_tasks

    def _make_future(idx):
        spec = cell_specs[idx]
        needed = set()
        for g in spec["required_groups"]:
            needed.update(group_to_partitions.get(g, []))
        part_futs = [partition_futures[i] for i in sorted(needed) if i < len(partition_futures)]
        if not part_futs:
            return client.submit(lambda **kw: None, **spec["cell_kwargs"], pure=False)
        return client.submit(worker_fn, *part_futs, **spec["cell_kwargs"], pure=False)

    try:
        task_iter = iter(range(n_tasks))
        pending = {}

        for _ in range(min(max_concurrent, n_tasks)):
            idx = next(task_iter)
            pending[_make_future(idx)] = idx

        ac = as_completed(pending)
        for completed in ac:
            idx = pending.pop(completed)
            results[idx] = completed.result()

            next_idx = next(task_iter, None)
            if next_idx is not None:
                new_fut = _make_future(next_idx)
                pending[new_fut] = next_idx
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
