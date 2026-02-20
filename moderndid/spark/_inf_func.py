"""Distributed influence function computation and variance estimation for Spark."""

from __future__ import annotations

import numpy as np

from moderndid.distributed._inf_func import compute_did_distributed  # noqa: F401

from ._gram import _sum_gram_pair


def compute_variance_distributed(spark, inf_func_partitions, n_total, k):
    r"""Compute variance from distributed influence function partitions.

    Uses Spark RDD ``treeReduce`` to compute
    :math:`V = (1/n) \, \Psi^\top \Psi` without materializing the full
    matrix on one node.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Spark session.
    inf_func_partitions : list of ndarray
        Per-partition influence function arrays.
    n_total : int
        Total number of observations.
    k : int
        Number of influence function columns.

    Returns
    -------
    se : ndarray of shape :math:`(k,)`
        Standard errors.
    """
    sc = spark.sparkContext
    rdd = sc.parallelize(inf_func_partitions, numSlices=len(inf_func_partitions))
    rdd = rdd.map(lambda chunk: (chunk.T @ chunk, np.zeros(chunk.shape[1]), chunk.shape[0]))

    PtP, _, _ = rdd.treeReduce(_sum_gram_pair, depth=3)
    V = PtP / n_total
    return np.sqrt(np.diag(V) / n_total)
