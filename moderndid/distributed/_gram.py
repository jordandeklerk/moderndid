"""Shared Gram matrix functions for distributed backends."""

from __future__ import annotations

import numpy as np

from moderndid.cupy.backend import _array_module, to_numpy


def weighted_gram(X, w, z=None, block_size=128):
    r"""Memory-efficient :math:`X^T \mathrm{diag}(w) X` with optional :math:`X^T \mathrm{diag}(w) z`.

    For :math:`k \le` ``block_size``, uses direct computation.
    For :math:`k >` ``block_size``, tiles the k dimension in blocks
    to reduce peak intermediate memory from :math:`O(k \times n)` to
    :math:`O(\text{block\_size} \times n)`.

    Parameters
    ----------
    X : ndarray of shape (n, k)
        Design matrix.
    w : ndarray of shape (n,)
        Weight vector.
    z : ndarray of shape (n,) or None, default None
        Optional right-hand side vector.
    block_size : int, default 128
        Block size for tiling the k dimension.

    Returns
    -------
    gram : ndarray of shape (k, k)
        Weighted Gram matrix (NumPy).
    rhs : ndarray of shape (k,) or None
        Weighted right-hand side (NumPy), returned only when ``z`` is not None.
        When ``z`` is None, only ``gram`` is returned (not a tuple).
    """
    xp = _array_module(X)
    k = X.shape[1]

    if k <= block_size:
        XtW = X.T * w
        gram = XtW @ X
        if z is not None:
            return to_numpy(gram), to_numpy(XtW @ z)
        return to_numpy(gram)

    gram = xp.zeros((k, k), dtype=X.dtype)
    rhs = xp.zeros(k, dtype=X.dtype) if z is not None else None
    for i in range(0, k, block_size):
        bi = min(block_size, k - i)
        XtWi = X[:, i : i + bi].T * w  # (bi, n)
        if z is not None:
            rhs[i : i + bi] = XtWi @ z
        for j in range(i, k, block_size):
            bj = min(block_size, k - j)
            gram[i : i + bi, j : j + bj] = XtWi @ X[:, j : j + bj]
            if i != j:
                gram[j : j + bj, i : i + bi] = gram[i : i + bi, j : j + bj].T

    if rhs is not None:
        return to_numpy(gram), to_numpy(rhs)
    return to_numpy(gram)


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
        Local :math:`X^T W X` Gram matrix (NumPy).
    XtWy : ndarray of shape (k,)
        Local :math:`X^T W y` vector (NumPy).
    n : int
        Number of observations in this partition.
    """
    XtWX, XtWy = weighted_gram(X, W, y)
    return XtWX, XtWy, len(y)


def solve_gram(XtWX, XtWy):
    r"""Solve the normal equations :math:`\hat{\beta} = (X^T W X)^{-1} X^T W y`.

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
