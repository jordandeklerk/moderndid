"""Distributed sufficient statistics via Spark collect and driver-side reduce."""

from __future__ import annotations

import pickle

import pandas as pd
from pyspark.sql.types import BinaryType, StructField, StructType

from moderndid.distributed._gram import (  # noqa: F401
    _reduce_group,
    _sum_gram_pair,
    partition_gram,
    solve_gram,
)
from moderndid.distributed._utils import _reduce_gram_list


def distributed_gram(spark, cached_df, build_arrays_fn, build_args):
    r"""Compute global sufficient statistics from Spark DataFrame partitions.

    Uses ``mapInPandas`` to compute per-partition Gram matrices, collects
    the small :math:`k \times k` results to the driver, and sums them.

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
