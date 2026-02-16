"""Dask distributed backend for moderndid DDD estimators."""

from ._utils import get_default_partitions, get_or_create_client, is_dask_collection, validate_dask_input
from ._ddd import dask_ddd
from .monitor import monitor_cluster

__all__ = [
    "dask_ddd",
    "get_default_partitions",
    "get_or_create_client",
    "is_dask_collection",
    "monitor_cluster",
    "validate_dask_input",
]
