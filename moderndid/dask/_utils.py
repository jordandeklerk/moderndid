"""Dask collection detection and input validation."""

from __future__ import annotations


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
