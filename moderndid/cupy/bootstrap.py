"""CuPy-accelerated bootstrap helpers."""

import numpy as np

from .backend import get_backend, to_numpy


def _multiplier_bootstrap_cupy(inf_func, biters, random_state=None):
    """Batched GPU multiplier bootstrap."""
    xp = get_backend()

    sqrt5 = float(np.sqrt(5))
    k1 = 0.5 * (1 - sqrt5)
    k2 = 0.5 * (1 + sqrt5)
    p_kappa = 0.5 * (1 + sqrt5) / sqrt5

    n = inf_func.shape[0]
    k = inf_func.shape[1] if inf_func.ndim == 2 else 1
    inf_gpu = xp.asarray(inf_func, dtype=xp.float64)

    max_batch_bytes = 1 << 30
    batch_size = max(1, max_batch_bytes // (16 * n))
    batch_size = min(batch_size, biters)

    if isinstance(random_state, np.random.Generator):
        random_state = int(random_state.integers(0, 2**31))
    rng = xp.random.default_rng(random_state)
    bres = np.empty((biters, k), dtype=np.float64)

    for start in range(0, biters, batch_size):
        end = min(start + batch_size, biters)
        b = end - start
        draws = rng.binomial(1, p_kappa, size=(b, n))
        v = xp.where(draws == 1, k1, k2).astype(xp.float64)
        bres[start:end] = to_numpy((v @ inf_gpu) / n)
        del draws, v

    return bres


def _aggregate_by_cluster_cupy(inf_func, cluster_int, n_clusters):
    """GPU cluster aggregation using scatter-add."""
    xp = get_backend()
    inf_gpu = xp.asarray(inf_func, dtype=xp.float64)
    cluster_gpu = xp.asarray(cluster_int)

    unique_gpu = xp.unique(cluster_gpu)
    idx = xp.searchsorted(unique_gpu, cluster_gpu)

    k = inf_gpu.shape[1]
    sums = xp.zeros((n_clusters, k), dtype=xp.float64)
    counts = xp.zeros(n_clusters, dtype=xp.float64)

    for j in range(k):
        xp.add.at(sums[:, j], idx, inf_gpu[:, j])
    xp.add.at(counts, idx, 1.0)

    counts = xp.maximum(counts, 1.0)
    result = sums / counts[:, None]
    return to_numpy(result)
