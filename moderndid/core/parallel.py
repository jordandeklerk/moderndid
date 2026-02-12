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
    """Execute func via dask.delayed for cluster-parallel workloads."""
    if not dask_available():
        raise ImportError("Dask is required for backend='dask'. Install it with: uv pip install moderndid[parallel]")
    import dask

    args_list = _scatter_args(args_list)
    delayed_results = [dask.delayed(func)(*args) for args in args_list]
    return list(dask.compute(*delayed_results))


def _scatter_args(args_list):
    """Scatter large objects to workers to keep the task graph lightweight.

    When a distributed client is active, DataFrames and arrays are sent
    directly to workers via ``client.scatter()``. Each delayed task then
    receives a lightweight Future reference instead of the full object,
    preventing the scheduler from serializing large datasets into the task
    graph. Objects that appear in multiple tasks (by identity) are scattered
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

    to_scatter = {}
    for args in args_list:
        for arg in args:
            oid = id(arg)
            if oid not in to_scatter and isinstance(arg, (pl.DataFrame, np.ndarray)):
                to_scatter[oid] = arg

    if not to_scatter:
        return args_list

    oids = list(to_scatter.keys())
    objs = [to_scatter[oid] for oid in oids]
    futures = client.scatter(objs, hash=False)
    id_to_future = dict(zip(oids, futures, strict=True))

    return [tuple(id_to_future.get(id(arg), arg) for arg in args) for args in args_list]
