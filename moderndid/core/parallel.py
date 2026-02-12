"""Parallel execution utilities for group-time estimation loops."""

from __future__ import annotations

import importlib.util
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import numpy as np
import polars as pl


@lru_cache
def dask_available() -> bool:
    """Check whether dask is importable."""
    return importlib.util.find_spec("dask") is not None


def parallel_map(func, args_list, n_jobs=1, backend="threads"):
    """Execute func(*args) for each args in args_list, optionally in parallel.

    Uses threads rather than processes because the per-cell computation is
    dominated by NumPy/scipy/statsmodels C extensions that release the GIL.
    Threads avoid the large serialization overhead of pickling data to
    subprocesses.

    Parameters
    ----------
    func : callable
        Function to call for each set of arguments.
    args_list : list of tuples
        Arguments for each call.
    n_jobs : int
        1 = sequential (default), -1 = all cores, >1 = that many workers.
    backend : {"threads", "dask"}, default="threads"
        Execution backend. ``"threads"`` uses a :class:`ThreadPoolExecutor`;
        ``"dask"`` uses :func:`dask.delayed` + :func:`dask.compute` to
        distribute work across a Dask cluster.

    Returns
    -------
    list
        Results in the same order as args_list.
    """
    if backend not in ("threads", "dask"):
        raise ValueError(f"backend='{backend}' is not valid. Must be 'threads' or 'dask'.")

    if n_jobs == 1:
        return [func(*args) for args in args_list]

    if backend == "dask":
        return _parallel_map_dask(func, args_list)

    max_workers = os.cpu_count() if n_jobs == -1 else n_jobs
    results = [None] * len(args_list)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(func, *args): i for i, args in enumerate(args_list)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
    return results


def _parallel_map_dask(func, args_list):
    """Execute func via dask for cluster-parallel workloads.

    When a distributed client is active, large objects (DataFrames, arrays)
    are scattered to workers with identity-based deduplication, and tasks are
    submitted via ``client.submit`` with all heavy data as top-level Future
    arguments so that Dask properly tracks dependencies.

    Falls back to ``dask.delayed`` when no distributed client is available.
    """
    if not dask_available():
        raise ImportError("Dask is required for backend='dask'. Install it with: uv pip install moderndid[parallel]")

    try:
        from distributed import get_client

        client = get_client()
    except (ImportError, ValueError):
        client = None

    if client is None:
        import dask

        delayed_results = [dask.delayed(func)(*args) for args in args_list]
        return list(dask.compute(*delayed_results))

    return _submit_with_scattered_args(client, func, args_list)


def _submit_with_scattered_args(client, func, args_list):
    """Scatter large objects and submit tasks via ``client.submit``.

    DataFrames and arrays are scattered to workers with identity-based
    deduplication, then swapped for their Future references inside each
    task's argument tuple.  ``client.submit`` resolves nested Futures
    automatically (via ``parse_input`` in Dask's task spec layer), so
    containers like ``cell_parts = (future1, future2)`` are transparently
    materialized on the worker.

    Following Dask's recommended ``as_completed`` pattern for
    memory-constrained workloads, up to ``max_inflight`` tasks are kept
    in flight at once.  As each task completes, its result is collected
    and the next task is submitted.  Scattered data futures persist
    across tasks so that partitions shared by multiple tasks are
    transferred to workers exactly once.  Reference counting tracks how
    many remaining tasks need each scattered object; once the last task
    using an object completes, its future is cancelled immediately so
    workers can reclaim memory.
    """
    has_large_objs = False
    for args in args_list:
        for arg in args:
            if isinstance(arg, (pl.DataFrame, np.ndarray)):
                has_large_objs = True
                break
            if isinstance(arg, (tuple, list)):
                for item in arg:
                    if isinstance(item, (pl.DataFrame, np.ndarray)):
                        has_large_objs = True
                        break
            if has_large_objs:
                break
        if has_large_objs:
            break

    if not has_large_objs:
        import dask

        delayed_results = [dask.delayed(func)(*args) for args in args_list]
        return list(dask.compute(*delayed_results))

    from distributed import as_completed as _as_completed

    n_workers = max(len(client.scheduler_info()["workers"]), 1)
    max_inflight = n_workers * 2

    # Pre-compute scatterable object IDs per task and reference counts so
    # scattered futures can be released as soon as all tasks referencing
    # them have completed, keeping worker memory bounded.
    task_obj_ids: list[set[int]] = []
    all_scatterable: dict[int, pl.DataFrame | np.ndarray] = {}
    ref_counts: dict[int, int] = {}
    for args in args_list:
        per_task: dict[int, pl.DataFrame | np.ndarray] = {}
        for arg in args:
            _collect_scatterable(arg, per_task)
        task_obj_ids.append(set(per_task.keys()))
        all_scatterable.update(per_task)
        for oid in per_task:
            ref_counts[oid] = ref_counts.get(oid, 0) + 1

    all_results = [None] * len(args_list)
    id_to_future: dict[int, object] = {}
    future_to_idx: dict[object, int] = {}

    def _scatter_new(task_indices):
        """Batch-scatter objects needed by the given tasks not yet on cluster."""
        new_oids: list[int] = []
        seen: set[int] = set()
        for i in task_indices:
            for oid in task_obj_ids[i]:
                if oid not in id_to_future and oid not in seen:
                    new_oids.append(oid)
                    seen.add(oid)
        if new_oids:
            objs = [all_scatterable[oid] for oid in new_oids]
            scattered = client.scatter(objs, hash=False)
            id_to_future.update(zip(new_oids, scattered, strict=True))

    def _submit_task(task_idx):
        """Submit a single task with its args resolved to scattered futures."""
        args = args_list[task_idx]
        resolved = tuple(_replace_with_futures(arg, id_to_future) for arg in args)
        fut = client.submit(func, *resolved, pure=False)
        future_to_idx[fut] = task_idx
        return fut

    try:
        # Scatter data and submit the initial window of tasks.
        initial = min(max_inflight, len(args_list))
        _scatter_new(range(initial))
        for i in range(initial):
            _submit_task(i)
        next_task = initial

        # Process results as they arrive, backfilling new tasks.
        ac = _as_completed(list(future_to_idx.keys()))
        for completed in ac:
            idx = future_to_idx.pop(completed)
            all_results[idx] = completed.result()

            # Release scattered data no longer needed by remaining tasks.
            for oid in task_obj_ids[idx]:
                ref_counts[oid] -= 1
                if ref_counts[oid] == 0 and oid in id_to_future:
                    client.cancel([id_to_future.pop(oid)])

            # Backfill: scatter any new data and submit the next task.
            if next_task < len(args_list):
                _scatter_new([next_task])
                fut = _submit_task(next_task)
                ac.add(fut)
                next_task += 1
    finally:
        if id_to_future:
            client.cancel(list(id_to_future.values()))
        if future_to_idx:
            client.cancel(list(future_to_idx.keys()))

    return all_results


def _scatter_args(args_list):
    """Scatter large objects to workers to keep the task graph lightweight.

    When a distributed client is active, DataFrames and arrays are sent
    directly to workers via ``client.scatter()``.  Each delayed task then
    receives a lightweight Future reference instead of the full object,
    preventing the scheduler from serializing large datasets into the task
    graph.  Objects that appear in multiple tasks (by identity) are scattered
    only once.

    Falls back to a no-op when no distributed client is available (e.g. when
    using the synchronous or threaded Dask schedulers).
    """
    if not args_list:
        return args_list

    try:
        from distributed import get_client

        client = get_client()
    except (ImportError, ValueError):
        return args_list

    to_scatter: dict[int, pl.DataFrame | np.ndarray] = {}
    for args in args_list:
        for arg in args:
            if isinstance(arg, (pl.DataFrame, np.ndarray)):
                oid = id(arg)
                if oid not in to_scatter:
                    to_scatter[oid] = arg

    if not to_scatter:
        return args_list

    oids = list(to_scatter.keys())
    objs = [to_scatter[oid] for oid in oids]
    futures = client.scatter(objs, hash=False)
    id_to_future = dict(zip(oids, futures, strict=True))

    return [tuple(id_to_future.get(id(arg), arg) for arg in args) for args in args_list]


def _collect_scatterable(obj, to_scatter):
    """Recursively collect DataFrames and arrays for scattering."""
    oid = id(obj)
    if oid in to_scatter:
        return
    if isinstance(obj, (pl.DataFrame, np.ndarray)):
        to_scatter[oid] = obj
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            _collect_scatterable(item, to_scatter)


def _replace_with_futures(obj, id_to_future):
    """Recursively replace DataFrames/arrays with their scattered Futures."""
    oid = id(obj)
    if oid in id_to_future:
        return id_to_future[oid]
    if isinstance(obj, tuple):
        return tuple(_replace_with_futures(item, id_to_future) for item in obj)
    if isinstance(obj, list):
        return [_replace_with_futures(item, id_to_future) for item in obj]
    return obj
