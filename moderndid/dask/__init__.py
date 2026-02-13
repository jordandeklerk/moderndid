"""Dask DataFrame distributed backend for moderndid."""

from moderndid.dask.backend import (
    cleanup_persisted,
    compute_dask_metadata,
    execute_cell_tasks,
    is_dask_dataframe,
    persist_by_group,
)

__all__ = [
    "cleanup_persisted",
    "compute_dask_metadata",
    "execute_cell_tasks",
    "is_dask_dataframe",
    "persist_by_group",
]
