"""Parallel execution utilities for group-time estimation loops."""

from __future__ import annotations

import contextvars
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


def parallel_map(func, args_list, n_jobs=1):
    """Execute func(*args) for each args in args_list, optionally in parallel.

    Uses threads rather than processes because the per-cell computation is
    dominated by NumPy/scipy/statsmodels C extensions that release the GIL.
    Threads avoid the large serialization overhead of pickling data to
    subprocesses.

    ``ContextVar`` values (e.g. the active backend set by
    :func:`~moderndid.cupy.backend.use_backend`) are propagated to each
    worker thread via :func:`contextvars.copy_context`.

    Parameters
    ----------
    func : callable
        Function to call for each set of arguments.
    args_list : list of tuples
        Arguments for each call.
    n_jobs : int
        1 = sequential (default), -1 = all cores, >1 = that many workers.

    Returns
    -------
    list
        Results in the same order as args_list.
    """
    if n_jobs == 1:
        return [func(*args) for args in args_list]

    max_workers = os.cpu_count() if n_jobs == -1 else n_jobs
    results = [None] * len(args_list)

    # Each task gets its own snapshot so Context.run() is never called
    # concurrently on the same object (which would raise RuntimeError).
    contexts = [contextvars.copy_context() for _ in args_list]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(ctx.run, func, *args): i
            for i, (ctx, args) in enumerate(zip(contexts, args_list, strict=True))
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
    return results
