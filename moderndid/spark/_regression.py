"""Distributed WLS and logistic IRLS via sufficient statistics for Spark."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from pyspark.sql.types import BinaryType, StructField, StructType

from moderndid.distributed._regression import (
    _irls_local_stats_with_y,
    logistic_irls_from_partition_list,
    wls_from_partition_list,
)

from ._gram import _reduce_gram_list, partition_gram, solve_gram


def distributed_wls(spark, partitions):
    """Distributed weighted least squares.

    Each worker computes local :math:`X^T W X` and :math:`X^T W y`, results
    are collected to the driver and summed, then normal equations are solved.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Spark session.
    partitions : list of (X, W, y) tuples
        Per-partition arrays: design matrix, weights, response.

    Returns
    -------
    beta : ndarray of shape (k,)
        Coefficient vector.
    """
    gram_list = [partition_gram(X, W, y) for X, W, y in partitions]
    result = _reduce_gram_list(gram_list)
    if result is None:
        raise ValueError("No data available for WLS regression.")
    XtWX, XtWy, _ = result
    return solve_gram(XtWX, XtWy)


def distributed_logistic_irls(spark, partitions, max_iter=25, tol=1e-8):
    r"""Distributed logistic regression via iteratively reweighted least squares.

    Each IRLS iteration computes local sufficient statistics
    :math:`X^T W X` and :math:`X^T W z` on each partition, sums on
    the driver, and solves for the updated :math:`\beta`.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Spark session.
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

    for _ in range(max_iter):
        gram_list = [_irls_local_stats_with_y(X, W, y, beta) for X, W, y in partitions]
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


def distributed_logistic_irls_from_partitions(spark, part_data_list, gram_fn, k, max_iter=25, tol=1e-8):
    r"""Distributed logistic regression via IRLS on partition data dicts.

    Thin wrapper around :func:`moderndid.distributed._regression.logistic_irls_from_partition_list`
    that accepts and ignores the ``spark`` parameter for backwards compatibility.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Spark session (unused; kept for API compatibility).
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
    return logistic_irls_from_partition_list(part_data_list, gram_fn, k, max_iter=max_iter, tol=tol)


def distributed_wls_from_partitions(spark, part_data_list, gram_fn):
    r"""Distributed weighted least squares on partition data dicts.

    Thin wrapper around :func:`moderndid.distributed._regression.wls_from_partition_list`
    that accepts and ignores the ``spark`` parameter for backwards compatibility.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Spark session (unused; kept for API compatibility).
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
    return wls_from_partition_list(part_data_list, gram_fn)


def distributed_logistic_irls_spark_df(spark, cached_df, build_fn, build_args, k, max_iter=25, tol=1e-8):
    r"""Distributed logistic IRLS using Spark DataFrame ``mapInPandas``.

    Each iteration broadcasts :math:`\beta` to workers, which compute
    local IRLS Gram matrices via ``mapInPandas``, collect to driver,
    sum, and solve.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Spark session.
    cached_df : pyspark.sql.DataFrame
        Cached Spark DataFrame.
    build_fn : callable
        Function ``(pandas_df, *build_args, beta) -> (XtWX, XtWz, n)``.
    build_args : tuple
        Additional arguments for ``build_fn``.
    k : int
        Number of design matrix columns.
    max_iter : int, default 25
        Maximum IRLS iterations.
    tol : float, default 1e-8
        Convergence tolerance.

    Returns
    -------
    beta : ndarray of shape (k,)
        Coefficient vector.
    """
    out_schema = StructType([StructField("gram_bytes", BinaryType(), False)])
    beta = np.zeros(k, dtype=np.float64)
    sc = spark.sparkContext

    for _ in range(max_iter):
        beta_bc = sc.broadcast(beta)

        def _irls_udf(iterator, _beta_bc=beta_bc, _build_fn=build_fn, _build_args=build_args):
            beta_val = _beta_bc.value
            for pdf in iterator:
                if len(pdf) == 0:
                    continue
                gram_tuple = _build_fn(pdf, *_build_args, beta_val)
                if gram_tuple is not None:
                    yield pd.DataFrame({"gram_bytes": [pickle.dumps(gram_tuple)]})

        result_df = cached_df.mapInPandas(_irls_udf, schema=out_schema)
        rows = result_df.collect()
        beta_bc.destroy()

        gram_list = [pickle.loads(row["gram_bytes"]) for row in rows]
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
