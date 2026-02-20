"""GPU conversion utilities for Spark partition arrays."""

from __future__ import annotations

import numpy as np


def _to_gpu_partition(part_data):
    """Convert float64 arrays in a partition dict to CuPy device arrays.

    Called on a Spark worker after ``_build_*_partition_arrays`` has produced
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


def _maybe_to_gpu_dict(part_data, use_gpu):
    """Conditionally convert partition dict arrays to GPU.

    Parameters
    ----------
    part_data : dict or None
        Partition dict produced by a ``_build_*_partition_arrays`` function.
    use_gpu : bool
        Whether to convert float arrays to CuPy.

    Returns
    -------
    dict or None
        Original dict if ``use_gpu`` is False, otherwise GPU-resident dict.
    """
    if not use_gpu:
        return part_data
    return _to_gpu_partition(part_data)


def _get_gpu_device_id():
    """Get the GPU device ID assigned to the current Spark task.

    Uses ``TaskContext.get().resources().get("gpu")`` following the XGBoost
    PySpark pattern, falling back to device 0 if not available.

    Returns
    -------
    int
        GPU device ID.
    """
    try:
        from pyspark import TaskContext

        ctx = TaskContext.get()
        if ctx is not None:
            resources = ctx.resources()
            if "gpu" in resources:
                addrs = resources["gpu"].addresses
                if addrs:
                    return int(addrs[0])
    except (ImportError, RuntimeError, AttributeError):
        pass
    return 0
