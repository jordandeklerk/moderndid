"""GPU conversion utilities for distributed partition arrays."""

from __future__ import annotations

import numpy as np


def _to_gpu_partition(part_data):
    """Convert float64 arrays in a partition dict to CuPy device arrays.

    Called on a worker after ``_build_*_partition_arrays`` has produced
    a dict of NumPy arrays.  Integer arrays (``ids``, ``subgroup``) and
    scalars (``n``) are left on the host because only float arrays benefit
    from GPU acceleration.

    Parameters
    ----------
    part_data : dict or None
        Partition dict produced by a ``_build_*_partition_arrays`` function.

    Returns
    -------
    dict or None
        Same dict with float64 arrays replaced by CuPy arrays.
    """
    if part_data is None:
        return None
    import cupy as cp

    return {k: cp.asarray(v) if isinstance(v, np.ndarray) and v.dtype.kind == "f" else v for k, v in part_data.items()}
