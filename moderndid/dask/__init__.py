"""Dask DataFrame distributed backend for moderndid."""

from moderndid.dask.backend import (
    compute_dask_metadata,
    gather_and_cleanup,
    is_dask_dataframe,
    persist_by_group,
    submit_cell_tasks,
)

__all__ = [
    "compute_dask_metadata",
    "gather_and_cleanup",
    "is_dask_dataframe",
    "persist_by_group",
    "submit_cell_tasks",
]
