"""Distributed sufficient statistics via tree-reduce."""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("moderndid.dask.gram")


def partition_gram(X, W, y):
    """Compute local sufficient statistics :math:`X^T W X` and :math:`X^T W y` on one partition.

    Parameters
    ----------
    X : ndarray of shape (n_local, k)
        Design matrix for this partition.
    W : ndarray of shape (n_local,)
        Weight vector for this partition.
    y : ndarray of shape (n_local,)
        Response vector for this partition.

    Returns
    -------
    XtWX : ndarray of shape (k, k)
        Local :math:`X^T W X` Gram matrix.
    XtWy : ndarray of shape (k,)
        Local :math:`X^T W y` vector.
    n : int
        Number of observations in this partition.
    """
    XtW = X.T * W  # (k, n_local)
    return XtW @ X, XtW @ y, len(y)


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
    log.info("distributed_gram: %d partitions → tree-reduce", len(partitions))
    futures = [client.submit(partition_gram, X, W, y) for X, W, y in partitions]
    return tree_reduce(client, futures, _sum_gram_pair)


def solve_gram(XtWX, XtWy):
    r"""Solve the normal equations :math:`\\hat{\\beta} = (X^T W X)^{-1} X^T W y`.

    Parameters
    ----------
    XtWX : ndarray of shape (k, k)
        Gram matrix.
    XtWy : ndarray of shape (k,)
        Right-hand side vector.

    Returns
    -------
    beta : ndarray of shape (k,)
        Solution vector.
    """
    return np.linalg.solve(XtWX, XtWy)


def _sum_gram_pair(a, b):
    """Sum two (XtWX, XtWy, n) tuples element-wise."""
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def _reduce_group(combine_fn, *items):
    """Reduce a group of items by applying combine_fn pairwise."""
    result = items[0]
    for item in items[1:]:
        result = combine_fn(result, item)
    return result
