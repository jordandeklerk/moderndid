"""Shared Spark utilities, constants, and reusable helpers."""

from __future__ import annotations

import numpy as np

MEMMAP_THRESHOLD = 1 * 1024**3
CHUNKED_SE_THRESHOLD = 10_000_000
SE_CHUNK_SIZE = 1_000_000


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


def auto_tune_partitions(n_default, n_units, k, target_bytes=500 * 1024**2):
    """Increase partition count when per-partition X matrices would exceed target_bytes.

    Parameters
    ----------
    n_default : int
        Default partition count (from cluster parallelism).
    n_units : int
        Total number of units.
    k : int
        Number of columns in the design matrix (intercept + covariates).
    target_bytes : int, default 500 MB
        Maximum per-partition X matrix size in bytes.

    Returns
    -------
    int
        Adjusted partition count.
    """
    rows_per_part = n_units / max(n_default, 1)
    part_bytes = rows_per_part * k * 8
    if part_bytes <= target_bytes:
        return n_default
    needed = int(np.ceil(n_units * k * 8 / target_bytes))
    return max(n_default, needed)


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


def sum_global_stats(a, b):
    """Pairwise sum for tree-reduce of global stats dicts.

    Parameters
    ----------
    a, b : dict or None
        Per-partition aggregate statistics.

    Returns
    -------
    dict or None
        Element-wise sum of the two dicts.
    """
    if a is None:
        return b
    if b is None:
        return a
    result = {}
    for key in a:
        if a[key] is None:
            result[key] = b[key]
        elif isinstance(a[key], (int, float)):
            result[key] = a[key] + b[key]
        else:
            result[key] = a[key] + b[key]
    return result


def chunked_vcov(inf_func, n_units):
    """Compute variance-covariance matrix, chunking for large n.

    Parameters
    ----------
    inf_func : ndarray of shape (n_units, n_cells)
        Influence function matrix.
    n_units : int
        Total number of units.

    Returns
    -------
    ndarray of shape (n_cells, n_cells)
        Variance-covariance matrix.
    """
    n_rows, n_cols = inf_func.shape
    if n_rows <= CHUNKED_SE_THRESHOLD:
        return inf_func.T @ inf_func / n_units

    V = np.zeros((n_cols, n_cols), dtype=np.float64)
    for start in range(0, n_rows, SE_CHUNK_SIZE):
        chunk = np.array(inf_func[start : start + SE_CHUNK_SIZE])
        V += chunk.T @ chunk
    V /= n_units
    return V


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

    for tp in all_times:
        period_y = filtered.filter(F.col(time_col) == tp).select(F.col(id_col), F.col(y_col).alias(f"_y_{tp}"))
        base = base.join(period_y, on=id_col, how="inner")

    wide_sdf = base.repartition(n_partitions).cache()
    n_wide = wide_sdf.count()

    if n_wide == 0:
        wide_sdf.unpersist()
        return None, 0

    return wide_sdf, n_wide


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
