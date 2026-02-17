"""Dask distributed backend for moderndid estimators."""

from ._utils import get_default_partitions, get_or_create_client, is_dask_collection, validate_dask_input
from ._ddd import dask_ddd
from ._did import dask_att_gt
from .monitor import monitor_cluster

__all__ = [
    "dask_att_gt",
    "dask_ddd",
    "get_default_partitions",
    "get_or_create_client",
    "is_dask_collection",
    "monitor_cluster",
    "validate_dask_input",
]
