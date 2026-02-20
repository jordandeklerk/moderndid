"""Shared Gram matrix functions for distributed backends."""

from __future__ import annotations

import numpy as np

from moderndid.cupy.backend import to_numpy


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
    XtW = X.T * W  # (k, n_local)
    return to_numpy(XtW @ X), to_numpy(XtW @ y), len(y)


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
