"""GPU conversion utilities for Dask partition arrays."""

from __future__ import annotations

from moderndid.distributed._gpu import _to_gpu_partition


def _maybe_to_gpu(client, part_futures, use_gpu):
    """Wrap partition futures with GPU conversion if *use_gpu* is True.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    part_futures : list of Future
        Futures resolving to partition dicts on workers.
    use_gpu : bool
        Whether to convert float arrays to CuPy on the worker.

    Returns
    -------
    list of Future
        Original futures if ``use_gpu`` is False, otherwise new futures
        that resolve to GPU-resident partition dicts.
    """
    if not use_gpu:
        return part_futures
    return [client.submit(_to_gpu_partition, pf) for pf in part_futures]
