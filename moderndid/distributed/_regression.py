"""Shared regression helpers for distributed backends."""

from __future__ import annotations

import numpy as np

from moderndid.cupy.backend import _array_module

from ._gram import solve_gram, weighted_gram
from ._utils import _reduce_gram_list


def _irls_local_stats_with_y(X, weights, y, beta):
    """Compute local IRLS sufficient statistics for one partition."""
    xp = _array_module(X)
    beta = xp.asarray(beta)
    eta = X @ beta
    mu = 1.0 / (1.0 + xp.exp(-eta))
    mu = xp.clip(mu, 1e-10, 1 - 1e-10)
    W_irls = weights * mu * (1 - mu)
    z = eta + (y - mu) / (mu * (1 - mu))
    XtWX, XtWz = weighted_gram(X, W_irls, z)
    return XtWX, XtWz, len(y)


def _sum_gram_pair_or_none(a, b):
    """Sum two (XtWX, XtWy, n) tuples, handling None from empty partitions."""
    if a is None:
        return b
    if b is None:
        return a
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def wls_from_partition_list(part_data_list, gram_fn):
    r"""Weighted least squares on a list of partition dicts (driver-side loop).

    Each partition dict is processed by ``gram_fn`` to produce local
    sufficient statistics, which are summed on the driver.

    Parameters
    ----------
    part_data_list : list of dict
        Partition dicts produced by ``_build_partition_arrays``.
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
    gram_list = [gram_fn(pd) for pd in part_data_list]
    result = _reduce_gram_list(gram_list)
    if result is None:
        raise ValueError("No data available for WLS regression.")
    XtWX, XtWy, _ = result
    return solve_gram(XtWX, XtWy)


def logistic_irls_from_partition_list(part_data_list, gram_fn, k, max_iter=25, tol=1e-8):
    r"""Logistic regression via IRLS on a list of partition dicts (driver-side loop).

    At each step the current :math:`\beta` is applied to all partitions via
    ``gram_fn``, results are collected and summed on the driver.

    Parameters
    ----------
    part_data_list : list of dict
        Partition dicts produced by ``_build_partition_arrays``.
    gram_fn : callable
        Function with signature ``gram_fn(part_data, beta)`` that returns
        ``(XtWX, XtWz, n)`` or ``None`` for empty partitions.
    k : int
        Number of columns in the design matrix :math:`X`.
    max_iter : int, default 25
        Maximum IRLS iterations.
    tol : float, default 1e-8
        Convergence tolerance.

    Returns
    -------
    beta : ndarray of shape (k,)
        Coefficient vector.
    """
    beta = np.zeros(k, dtype=np.float64)

    for _ in range(max_iter):
        gram_list = [gram_fn(pd, beta) for pd in part_data_list]
        result = _reduce_gram_list(gram_list)
        if result is None:
            break
        XtWX, XtWz, _ = result
        beta_new = solve_gram(XtWX, XtWz)

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    return beta
