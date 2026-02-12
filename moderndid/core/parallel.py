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
    """Scatter large objects and submit tasks via client.submit.

    ``client.submit`` only resolves Futures that are top-level positional
    arguments.  Futures nested inside tuples or lists are serialized into the
    task graph verbatim, which defeats the purpose of scattering.

    To handle arguments like ``cell_parts`` (a tuple of DataFrames), this
    function encodes each task's argument structure as a lightweight
    *template* and passes every DataFrame as a separate top-level Future.
    A thin wrapper on the worker reconstructs the original argument tuple
    from the template before calling *func*.
    """
    to_scatter: dict[int, pl.DataFrame | np.ndarray] = {}
    for args in args_list:
        for arg in args:
            _collect_scatterable(arg, to_scatter)

    if not to_scatter:
        import dask

        delayed_results = [dask.delayed(func)(*args) for args in args_list]
        return list(dask.compute(*delayed_results))

    oids = list(to_scatter.keys())
    objs = [to_scatter[oid] for oid in oids]
    futures = client.scatter(objs, hash=False)
    id_to_future = dict(zip(oids, futures, strict=True))

    task_futures = []
    for args in args_list:
        df_futures: list = []
        template: list = []
        for arg in args:
            _encode_arg(arg, id_to_future, df_futures, template)
        task_futures.append(client.submit(_apply_template, func, template, *df_futures, pure=False))

    return client.gather(task_futures)


def _apply_template(func, template, *resolved):
    """Reconstruct the original args from *template* + resolved DataFrames."""
    args = []
    for entry in template:
        kind = entry[0]
        if kind == "v":
            args.append(entry[1])
        elif kind == "f":
            args.append(resolved[entry[1]])
        elif kind == "t":
            args.append(tuple(resolved[i] if is_f else val for is_f, i, val in entry[1]))
        elif kind == "l":
            args.append([resolved[i] if is_f else val for is_f, i, val in entry[1]])
    return func(*args)


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


def _encode_arg(arg, id_to_future, df_futures, template):
    """Encode a single argument into *template* and append Futures to *df_futures*."""
    oid = id(arg)
    if oid in id_to_future:
        template.append(("f", len(df_futures)))
        df_futures.append(id_to_future[oid])
    elif isinstance(arg, (tuple, list)):
        items = []
        has_futures = False
        for item in arg:
            item_oid = id(item)
            if item_oid in id_to_future:
                items.append((True, len(df_futures), None))
                df_futures.append(id_to_future[item_oid])
                has_futures = True
            else:
                items.append((False, 0, item))
        if has_futures:
            template.append(("t" if isinstance(arg, tuple) else "l", items))
        else:
            template.append(("v", arg))
    else:
        template.append(("v", arg))
