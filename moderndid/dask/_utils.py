"""Shared Dask utilities, constants, and reusable helpers."""

from __future__ import annotations

import numpy as np

MEMMAP_THRESHOLD = 1 * 1024**3
CHUNKED_SE_THRESHOLD = 10_000_000
SE_CHUNK_SIZE = 1_000_000


def is_dask_collection(data) -> bool:
    """Check if data is a Dask DataFrame.

    Parameters
    ----------
    data : object
        Input data to check.

    Returns
    -------
    bool
        True if data is a ``dask.dataframe.DataFrame``.
    """
    try:
        import dask.dataframe as dd

        return isinstance(data, dd.DataFrame)
    except ImportError:
        return False


def validate_dask_input(ddf, required_cols):
    """Validate that a Dask DataFrame has the required columns.

    Parameters
    ----------
    ddf : dask.dataframe.DataFrame
        The Dask DataFrame to validate.
    required_cols : list of str
        Column names that must be present.

    Raises
    ------
    ValueError
        If any required columns are missing.
    """
    missing = [c for c in required_cols if c not in ddf.columns]
    if missing:
        raise ValueError(f"Columns not found in Dask DataFrame: {missing}")


def get_default_partitions(client):
    """Compute default partition count from total cluster threads.

    Uses the total number of threads across all workers so that every
    thread has at least one partition to work on, rather than defaulting
    to the number of workers (which leaves most threads idle).

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.

    Returns
    -------
    int
        Recommended number of partitions (total threads, minimum 1).
    """
    info = client.scheduler_info()
    workers = info.get("workers", {})
    total_threads = sum(w.get("nthreads", 1) for w in workers.values())
    return max(total_threads, 1)


def auto_tune_partitions(n_default, n_units, k, target_bytes=500 * 1024**2):
    """Increase partition count when per-partition X matrices would exceed target_bytes.

    Parameters
    ----------
    n_default : int
        Default partition count (from cluster thread count).
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


def get_or_create_client(client=None):
    """Get an existing Dask client or create a local one.

    Parameters
    ----------
    client : distributed.Client or None
        An existing Dask distributed client. If None, attempts to get
        the current client or creates a new ``LocalCluster`` client.

    Returns
    -------
    distributed.Client
        A Dask distributed client.
    """
    from distributed import Client

    if client is not None:
        return client
    try:
        return Client.current()
    except ValueError:
        return Client()


def detect_multiple_periods(ddf, tname, gname, client=None):
    """Detect whether data has more than 2 time periods or treatment groups.

    Parameters
    ----------
    ddf : dask.dataframe.DataFrame
        Dask DataFrame to inspect.
    tname : str
        Time period column name.
    gname : str
        Treatment group column name.
    client : distributed.Client or None
        Dask distributed client.

    Returns
    -------
    bool
        True if there are more than 2 time periods or treatment groups.
    """
    if client is not None:
        t_fut = client.compute(ddf[tname].nunique())
        g_fut = client.compute(ddf[gname].unique())
        n_time, gvals = client.gather([t_fut, g_fut])
        gvals = gvals.values
    else:
        n_time = ddf[tname].nunique().compute()
        gvals = ddf[gname].unique().compute().values

    finite_gvals = [g for g in gvals if np.isfinite(g)]
    n_groups = len(finite_gvals)

    return max(n_time, n_groups) > 2


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
    client,
    dask_data,
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
    ``repartition().persist()`` cycle.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    dask_data : dask.dataframe.DataFrame
        Persisted Dask DataFrame in long panel format.
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
        Number of Dask partitions for the wide DataFrame.
    extra_cols : list of str or None, default None
        Additional columns to include in the base (e.g. partition column
        for DDD).

    Returns
    -------
    wide_dask : dask.dataframe.DataFrame or None
        Persisted wide DataFrame, or ``None`` if no data.
    n_wide : int
        Number of units in the wide DataFrame.
    """
    all_times = set()
    for _counter, _g, t_cell, pret, _pt, action in cells:
        if action != "compute" or pret is None:
            continue
        all_times.add(t_cell)
        all_times.add(pret)

    all_times = sorted(all_times)
    if not all_times:
        return None, 0

    group_filter = (dask_data[group_col] == 0) | (dask_data[group_col] == g)
    filtered = dask_data.loc[group_filter]

    base_cols = [id_col, group_col]
    if extra_cols:
        base_cols = base_cols + [c for c in extra_cols if c not in base_cols]
    if covariate_cols:
        base_cols = base_cols + [c for c in covariate_cols if c not in base_cols]

    base = filtered.loc[filtered[time_col] == all_times[0]][base_cols]

    for tp in all_times:
        period_y = filtered.loc[filtered[time_col] == tp][[id_col, y_col]]
        period_y = period_y.rename(columns={y_col: f"_y_{tp}"})
        base = base.merge(period_y, on=id_col, how="inner")

    from distributed import wait

    wide_dask = base.repartition(npartitions=n_partitions).persist()
    wait(wide_dask)

    n_wide = len(wide_dask)
    if n_wide == 0:
        return None, 0

    return wide_dask, n_wide
