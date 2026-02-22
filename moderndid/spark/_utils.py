"""Shared Spark utilities, constants, and reusable helpers."""

from __future__ import annotations

import numpy as np

from moderndid.distributed._utils import (  # noqa: F401
    MEMMAP_THRESHOLD,
    auto_tune_partitions,
    chunked_vcov,
    sum_global_stats,
)


def is_spark_dataframe(data) -> bool:
    """Check if data is a PySpark DataFrame.

    Parameters
    ----------
    data : object
        Input data to check.

    Returns
    -------
    bool
        True if data is a ``pyspark.sql.DataFrame``.
    """
    try:
        from pyspark.sql import DataFrame as SparkDataFrame

        return isinstance(data, SparkDataFrame)
    except ImportError:
        _type_name = type(data).__module__ + "." + type(data).__qualname__
        if "pyspark" in _type_name.lower():
            raise ImportError(
                f"Input data appears to be a PySpark object ({_type_name}) but "
                "the spark extra is not installed. Install with: "
                "uv pip install 'moderndid[spark]'"
            ) from None
        return False


def validate_spark_input(sdf, required_cols):
    """Validate that a Spark DataFrame has the required columns.

    Parameters
    ----------
    sdf : pyspark.sql.DataFrame
        The Spark DataFrame to validate.
    required_cols : list of str
        Column names that must be present.

    Raises
    ------
    ValueError
        If any required columns are missing.
    """
    missing = [c for c in required_cols if c not in sdf.columns]
    if missing:
        raise ValueError(f"Columns not found in Spark DataFrame: {missing}")


def get_default_partitions(spark):
    """Compute default partition count from Spark default parallelism.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Spark session.

    Returns
    -------
    int
        Recommended number of partitions (default parallelism, minimum 1).
    """
    return max(spark.sparkContext.defaultParallelism, 1)


def get_or_create_spark(spark=None):
    """Get an existing SparkSession or create a local one.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession or None
        An existing Spark session. If None, attempts to get the active
        session or creates a new local session.

    Returns
    -------
    pyspark.sql.SparkSession
        A Spark session.
    """
    from pyspark.sql import SparkSession

    if spark is not None:
        return spark
    active = SparkSession.getActiveSession()
    if active is not None:
        return active
    return SparkSession.builder.master("local[*]").appName("moderndid").getOrCreate()


def prepare_cohort_wide_pivot(
    spark,
    sdf,
    g,
    cells,
    time_col,
    group_col,
    id_col,
    y_col,
    covariate_cols,
    n_partitions,
    extra_cols=None,
):
    """Build a single wide-pivoted DataFrame for all cells in a cohort.

    Instead of performing one shuffle join per :math:`(g, t)` cell, this
    function collects every time period required by the cohort's cells,
    merges them into a single wide DataFrame (one row per unit, one
    ``_y_{period}`` column per time period), and does a single
    ``repartition().cache()`` cycle.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Spark session.
    sdf : pyspark.sql.DataFrame
        Cached Spark DataFrame in long panel format.
    g : int or float
        Treatment cohort identifier.
    cells : list of tuple
        Cell specifications ``(counter, g, t, pret, post_treat, action)``
        for this cohort. Only ``action == "compute"`` cells are used.
    time_col, group_col, id_col, y_col : str
        Column names.
    covariate_cols : list of str or None
        Covariate column names.
    n_partitions : int
        Number of Spark partitions for the wide DataFrame.
    extra_cols : list of str or None, default None
        Additional columns to include in the base (e.g. partition column
        for DDD).

    Returns
    -------
    wide_sdf : pyspark.sql.DataFrame or None
        Cached wide DataFrame, or ``None`` if no data.
    n_wide : int
        Number of units in the wide DataFrame.
    """
    from pyspark.sql import functions as F

    all_times = set()
    for _counter, _g, t_cell, pret, _pt, action in cells:
        if action != "compute" or pret is None:
            continue
        all_times.add(t_cell)
        all_times.add(pret)

    all_times = sorted(all_times)
    if not all_times:
        return None, 0

    filtered = sdf.filter((F.col(group_col) == 0) | (F.col(group_col) == g))

    base_cols = [id_col, group_col]
    if extra_cols:
        base_cols = base_cols + [c for c in extra_cols if c not in base_cols]
    if covariate_cols:
        base_cols = base_cols + [c for c in covariate_cols if c not in base_cols]

    base = filtered.filter(F.col(time_col) == all_times[0]).select(*base_cols)

    period_data = filtered.filter(F.col(time_col).isin([int(t) for t in all_times])).select(id_col, time_col, y_col)
    pivoted_y = period_data.groupBy(id_col).pivot(time_col, [int(t) for t in all_times]).agg(F.first(y_col))
    for tp in all_times:
        pivoted_y = pivoted_y.withColumnRenamed(str(int(tp)), f"_y_{tp}")

    wide_sdf = base.join(pivoted_y, on=id_col, how="inner").repartition(n_partitions).cache()
    n_wide = wide_sdf.count()

    if n_wide == 0:
        wide_sdf.unpersist()
        return None, 0

    return wide_sdf, n_wide


def collect_partitions(cached_sdf, n_chunks=None):
    """Collect a cached Spark DataFrame as a list of pandas DataFrame chunks.

    Uses Arrow-based ``.toPandas()`` on the already-cached DataFrame, then
    splits the result into roughly equal chunks for downstream iteration.

    Parameters
    ----------
    cached_sdf : pyspark.sql.DataFrame
        A cached Spark DataFrame.
    n_chunks : int or None
        Number of chunks to split into.  When ``None`` the number of
        Spark RDD partitions is used so chunk boundaries mirror the
        original partitioning.

    Returns
    -------
    list of pandas.DataFrame
        Chunked pandas DataFrames.
    """
    full_pdf = cached_sdf.toPandas()
    if len(full_pdf) == 0:
        return []
    if n_chunks is None:
        n_chunks = max(1, cached_sdf.rdd.getNumPartitions())
    n_chunks = max(1, min(n_chunks, len(full_pdf)))
    boundaries = np.array_split(np.arange(len(full_pdf)), n_chunks)
    return [full_pdf.iloc[idx] for idx in boundaries if len(idx) > 0]
