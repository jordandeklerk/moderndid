"""Distributed WLS and logistic IRLS via sufficient statistics."""

from __future__ import annotations

import numpy as np

from moderndid.distributed._regression import (
    _irls_local_stats_with_y,
    _sum_gram_pair_or_none,
)

from ._gram import _sum_gram_pair, partition_gram, solve_gram, tree_reduce


def distributed_wls(client, partitions):
    """Distributed weighted least squares.

    Each worker computes local :math:`X^T W X` and :math:`X^T W y`, then
    tree-reduce to global sufficient statistics and solve the normal equations
    on the driver.

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


def distributed_logistic_irls(client, partitions, max_iter=25, tol=1e-8):
    r"""Distributed logistic regression via iteratively reweighted least squares.

    Each IRLS iteration broadcasts :math:`\\beta` to workers, who compute
    local sufficient statistics :math:`X^T W X` and :math:`X^T W z` where
    :math:`W = \\text{diag}(\\mu(1 - \\mu))` and
    :math:`z = \\eta + (y - \\mu) / (\\mu(1 - \\mu))`, then tree-reduces
    to solve on the driver.

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
    k = partitions[0][0].shape[1]
    beta = np.zeros(k, dtype=np.float64)

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
    r"""Distributed logistic regression via IRLS on pre-scattered partition futures.

    Unlike :func:`distributed_logistic_irls`, this variant operates on futures
    to partition dicts that already reside on workers, avoiding re-scattering
    data each iteration. At each step the current :math:`\\beta` is broadcast
    to all workers, which compute local IRLS Gram matrices
    :math:`X^T W X` and working-response vectors :math:`X^T W z`, then
    tree-reduce to solve on the driver.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    part_futures : list of Future
        Futures to partition dicts produced by ``_build_partition_arrays``.
        Each dict contains ``{ids, y1, y0, subgroup, X, n}``.
    gram_fn : callable
        Function with signature ``gram_fn(part_data, beta)`` that returns
        ``(XtWX, XtWz, n)`` or ``None`` for empty partitions.
    k : int
        Number of columns in the design matrix :math:`X`.
    max_iter : int, default 25
        Maximum IRLS iterations.
    tol : float, default 1e-8
        Convergence tolerance on
        :math:`\\max |\\beta^{(t+1)} - \\beta^{(t)}|`.

    Returns
    -------
    beta : ndarray of shape (k,)
        Coefficient vector.
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
    r"""Distributed weighted least squares on pre-scattered partition futures.

    Unlike :func:`distributed_wls`, this variant operates on futures to
    partition dicts that already reside on workers. Each worker computes
    local sufficient statistics :math:`X^T W X` and :math:`X^T W y` via
    ``gram_fn``, then tree-reduce to the driver which solves the normal
    equations :math:`\\hat{\\beta} = (X^T W X)^{-1} X^T W y`.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    part_futures : list of Future
        Futures to partition dicts produced by ``_build_partition_arrays``.
        Each dict contains ``{ids, y1, y0, subgroup, X, n}``.
    gram_fn : callable
        Function with signature ``gram_fn(part_data)`` that returns
        ``(XtWX, XtWy, n)`` or ``None`` for empty partitions.

    Returns
    -------
    beta : ndarray of shape (k,)
        Coefficient vector.

    Raises
    ------
    ValueError
        If all partitions return ``None`` (no data available).
    """
    futures = [client.submit(gram_fn, pf) for pf in part_futures]
    XtWX, XtWy, _ = tree_reduce(client, futures, _sum_gram_pair_or_none)
    if XtWX is None:
        raise ValueError("No data available for WLS regression.")
    return solve_gram(XtWX, XtWy)
