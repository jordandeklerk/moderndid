"""Distributed WLS and logistic IRLS via sufficient statistics."""

from __future__ import annotations

import numpy as np

from ._gram import _sum_gram_pair, partition_gram, solve_gram, tree_reduce


def distributed_wls(client, partitions):
    """Distributed weighted least squares.

    Each worker computes local ``X'WX`` and ``X'Wy``, then tree-reduce
    to global sufficient statistics and solve on the driver.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    partitions : list of (X, W, y) tuples
        Per-partition arrays: design matrix, weights, response.

    Returns
    -------
    beta : ndarray of shape (k,)
        Coefficient vector.
    """
    Xs, Ws, ys = zip(*partitions, strict=True)
    Xs_f = client.scatter(list(Xs))
    Ws_f = client.scatter(list(Ws))
    ys_f = client.scatter(list(ys))
    futures = [client.submit(partition_gram, xf, wf, yf) for xf, wf, yf in zip(Xs_f, Ws_f, ys_f, strict=True)]
    XtWX, XtWy, _ = tree_reduce(client, futures, _sum_gram_pair)
    return solve_gram(XtWX, XtWy)


def _irls_local_stats_with_y(X, weights, y, beta):
    """Compute local IRLS sufficient statistics for one partition.

    Parameters
    ----------
    X : ndarray of shape (n_local, k)
        Design matrix.
    weights : ndarray of shape (n_local,)
        Observation weights.
    y : ndarray of shape (n_local,)
        Binary response vector.
    beta : ndarray of shape (k,)
        Current coefficient estimate.

    Returns
    -------
    XtWX : ndarray of shape (k, k)
        Local gram matrix with IRLS weights.
    XtWz : ndarray of shape (k,)
        Local X'Wz vector.
    n : int
        Local sample size.
    """
    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-eta))
    mu = np.clip(mu, 1e-10, 1 - 1e-10)
    W_irls = weights * mu * (1 - mu)
    z = eta + (y - mu) / (mu * (1 - mu))
    XtW = X.T * W_irls
    return XtW @ X, XtW @ z, len(y)


def distributed_logistic_irls(client, partitions, max_iter=25, tol=1e-8):
    """Distributed logistic regression via iteratively reweighted least squares.

    Each IRLS iteration broadcasts ``beta`` to workers, who compute local
    sufficient statistics, then tree-reduces to solve on the driver.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    partitions : list of (X, weights, y) tuples
        Per-partition arrays: design matrix, observation weights, binary response.
    max_iter : int, default 25
        Maximum IRLS iterations.
    tol : float, default 1e-8
        Convergence tolerance on max absolute parameter change.

    Returns
    -------
    beta : ndarray of shape (k,)
        Coefficient vector.
    """
    # Get k from first partition
    k = partitions[0][0].shape[1]
    beta = np.zeros(k, dtype=np.float64)

    # Scatter partition data once; reuse across IRLS iterations
    Xs, Ws, ys = zip(*partitions, strict=True)
    Xs_f = client.scatter(list(Xs))
    Ws_f = client.scatter(list(Ws))
    ys_f = client.scatter(list(ys))

    for _ in range(max_iter):
        futures = [
            client.submit(_irls_local_stats_with_y, xf, wf, yf, beta)
            for xf, wf, yf in zip(Xs_f, Ws_f, ys_f, strict=True)
        ]
        XtWX, XtWz, _ = tree_reduce(client, futures, _sum_gram_pair)
        beta_new = solve_gram(XtWX, XtWz)

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    return beta


def distributed_logistic_irls_from_futures(client, part_futures, gram_fn, k, max_iter=25, tol=1e-8):
    """IRLS on partition futures.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    part_futures : list of Future
        Futures to partition dicts from ``_build_partition_arrays``.
    gram_fn : callable
        ``gram_fn(part_data, beta) -> (XtWX, XtWz, n) or None``.
    k : int
        Number of covariates (columns of X).
    max_iter : int, default 25
        Maximum IRLS iterations.
    tol : float, default 1e-8
        Convergence tolerance.

    Returns
    -------
    beta : ndarray of shape (k,)
    """
    beta = np.zeros(k, dtype=np.float64)

    for _ in range(max_iter):
        futures = [client.submit(gram_fn, pf, beta) for pf in part_futures]
        XtWX, XtWz, _ = tree_reduce(client, futures, _sum_gram_pair_or_none)
        if XtWX is None:
            break
        beta_new = solve_gram(XtWX, XtWz)

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    return beta


def distributed_wls_from_futures(client, part_futures, gram_fn):
    """WLS on partition futures.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    part_futures : list of Future
        Futures to partition dicts from ``_build_partition_arrays``.
    gram_fn : callable
        ``gram_fn(part_data) -> (XtWX, XtWy, n) or None``.

    Returns
    -------
    beta : ndarray of shape (k,)
    """
    futures = [client.submit(gram_fn, pf) for pf in part_futures]
    XtWX, XtWy, _ = tree_reduce(client, futures, _sum_gram_pair_or_none)
    if XtWX is None:
        raise ValueError("No data available for WLS regression.")
    return solve_gram(XtWX, XtWy)


def _sum_gram_pair_or_none(a, b):
    """Sum two (XtWX, XtWy, n) tuples, handling None from empty partitions."""
    if a is None:
        return b
    if b is None:
        return a
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]
