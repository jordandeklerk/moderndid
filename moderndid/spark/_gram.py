"""Distributed sufficient statistics via Spark collect and driver-side reduce."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from pyspark.sql.types import BinaryType, StructField, StructType

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


def distributed_gram(spark, cached_df, build_arrays_fn, build_args):
    r"""Compute global sufficient statistics from Spark DataFrame partitions.

    Uses ``mapInPandas`` to compute per-partition Gram matrices, collects
    the small :math:`k \\times k` results to the driver, and sums them.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Spark session.
    cached_df : pyspark.sql.DataFrame
        Cached Spark DataFrame.
    build_arrays_fn : callable
        Function that takes a pandas DataFrame and ``*build_args`` and
        returns ``(X, W, y)`` arrays.
    build_args : tuple
        Additional arguments for ``build_arrays_fn``.

    Returns
    -------
    XtWX : ndarray of shape (k, k)
        Global Gram matrix.
    XtWy : ndarray of shape (k,)
        Global :math:`X^T W y` vector.
    n_total : int
        Total number of observations across all partitions.
    """
    out_schema = StructType([StructField("gram_bytes", BinaryType(), False)])

    def _compute_gram_udf(iterator):
        for pdf in iterator:
            if len(pdf) == 0:
                continue
            X, W, y = build_arrays_fn(pdf, *build_args)
            gram_tuple = partition_gram(X, W, y)
            yield pd.DataFrame({"gram_bytes": [pickle.dumps(gram_tuple)]})

    result_df = cached_df.mapInPandas(_compute_gram_udf, schema=out_schema)
    rows = result_df.collect()

    gram_list = [pickle.loads(row["gram_bytes"]) for row in rows]

    return _reduce_gram_list(gram_list)


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


def _reduce_gram_list(gram_list):
    """Driver-side sum of collected (XtWX, XtWy, n) tuples.

    Parameters
    ----------
    gram_list : list of (ndarray, ndarray, int) or None
        Collected Gram tuples from partitions.

    Returns
    -------
    tuple of (ndarray, ndarray, int) or None
        Summed (XtWX, XtWy, n_total).
    """
    result = None
    for item in gram_list:
        if item is None:
            continue
        result = item if result is None else (result[0] + item[0], result[1] + item[1], result[2] + item[2])
    return result


def _reduce_group(combine_fn, *items):
    """Reduce a group of items by applying combine_fn pairwise."""
    result = items[0]
    for item in items[1:]:
        result = combine_fn(result, item)
    return result
