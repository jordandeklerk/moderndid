"""Dask distributed backend for moderndid estimators."""

from ._utils import get_default_partitions, get_or_create_client, is_dask_collection, validate_dask_input
from ._ddd import dask_ddd
from ._did import dask_att_gt
from ._didcont import dask_cont_did
from ._didinter import dask_did_multiplegt

__all__ = [
    "dask_att_gt",
    "dask_cont_did",
    "dask_ddd",
    "dask_did_multiplegt",
    "get_default_partitions",
    "get_or_create_client",
    "is_dask_collection",
    "validate_dask_input",
]
