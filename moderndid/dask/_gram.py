"""Distributed sufficient statistics via tree-reduce."""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("moderndid.dask.gram")


def partition_gram(X, W, y):
    """Compute local sufficient statistics on one partition.

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
        Local X'WX gram matrix.
    XtWy : ndarray of shape (k,)
        Local X'Wy vector.
    n : int
        Number of observations in this partition.
    """
    XtW = X.T * W  # (k, n_local)
    return XtW @ X, XtW @ y, len(y)


def _sum_gram_pair(a, b):
    """Sum two (XtWX, XtWy, n) tuples element-wise."""
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def tree_reduce(client, futures, combine_fn):
    """Tree-reduce a list of futures using a pairwise combine function.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    futures : list of Future
        Futures to reduce.
    combine_fn : callable
        Function ``(a, b) -> c`` that combines two results.

    Returns
    -------
    result
        The fully reduced result.
    """
    while len(futures) > 1:
        new_futures = []
        for i in range(0, len(futures), 2):
            if i + 1 < len(futures):
                new_futures.append(client.submit(combine_fn, futures[i], futures[i + 1]))
            else:
                new_futures.append(futures[i])
        futures = new_futures
    return futures[0].result()


def distributed_gram(client, partitions):
    """Compute global sufficient statistics from distributed partitions.

    Submits ``partition_gram`` to each worker and tree-reduces the results
    into global ``X'WX`` (K x K) and ``X'Wy`` (K) matrices.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    partitions : list of (X, W, y) tuples
        Per-partition data arrays.

    Returns
    -------
    XtWX : ndarray of shape (k, k)
        Global gram matrix.
    XtWy : ndarray of shape (k,)
        Global X'Wy vector.
    n_total : int
        Total number of observations across all partitions.
    """
    log.info("distributed_gram: %d partitions → tree-reduce", len(partitions))
    futures = [client.submit(partition_gram, X, W, y) for X, W, y in partitions]
    return tree_reduce(client, futures, _sum_gram_pair)


def solve_gram(XtWX, XtWy):
    """Solve the normal equations from sufficient statistics.

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
