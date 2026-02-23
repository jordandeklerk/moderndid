"""Distributed sufficient statistics via tree-reduce."""

from __future__ import annotations

from moderndid.distributed._gram import (  # noqa: F401
    _reduce_group,
    _sum_gram_pair,
    partition_gram,
    solve_gram,
)


def tree_reduce(client, futures, combine_fn, split_every=8):
    """Tree-reduce a list of futures with configurable fan-in.

    Groups ``split_every`` futures per reduction step and reduces each
    group in a single task on one worker, following the pattern used by
    Dask's internal reductions. With 64 futures and ``split_every=8``
    this produces 9 tasks (8 + 1) instead of the 63 tasks created by
    pairwise reduction.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    futures : list of Future
        Futures to reduce.
    combine_fn : callable
        Pairwise combiner ``(a, b) -> c``.
    split_every : int, default 8
        Number of futures to combine per reduction step.

    Returns
    -------
    result
        The fully reduced result, materialized on the driver.
    """
    while len(futures) > 1:
        new_futures = []
        for i in range(0, len(futures), split_every):
            group = futures[i : i + split_every]
            if len(group) == 1:
                new_futures.append(group[0])
            else:
                new_futures.append(client.submit(_reduce_group, combine_fn, *group))
        futures = new_futures
    return futures[0].result()


def distributed_gram(client, partitions):
    """Compute global sufficient statistics from distributed partitions.

    Submits :func:`partition_gram` to each worker and tree-reduces the
    results into global :math:`X^T W X` and :math:`X^T W y` matrices.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    partitions : list of (X, W, y) tuples
        Per-partition data arrays.

    Returns
    -------
    XtWX : ndarray of shape (k, k)
        Global Gram matrix.
    XtWy : ndarray of shape (k,)
        Global :math:`X^T W y` vector.
    n_total : int
        Total number of observations across all partitions.
    """
    futures = [client.submit(partition_gram, X, W, y) for X, W, y in partitions]
    return tree_reduce(client, futures, _sum_gram_pair)
