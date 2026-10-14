"""CuPy-accelerated bootstrap helpers."""

import numpy as np

from .backend import get_backend, to_numpy


def _multiplier_bootstrap_cupy(inf_func, biters, random_state=None):
    """Batched GPU multiplier bootstrap with Rademacher weights."""
    xp = get_backend()

    n = inf_func.shape[0]
    k = inf_func.shape[1] if inf_func.ndim == 2 else 1
    inf_gpu = xp.asarray(inf_func, dtype=xp.float64)

    max_batch_bytes = 1 << 30
    batch_size = max(1, max_batch_bytes // (16 * n))
    batch_size = min(batch_size, biters)

    if isinstance(random_state, np.random.Generator):
        random_state = int(random_state.bit_generator.seed_seq.spawn(1)[0].generate_state(1)[0])
    rng = xp.random.default_rng(random_state)
    bres = np.empty((biters, k), dtype=np.float64)

    for start in range(0, biters, batch_size):
        end = min(start + batch_size, biters)
        b = end - start
        draws = rng.binomial(1, 0.5, size=(b, n))
        v = xp.where(draws == 1, 1.0, -1.0).astype(xp.float64)
        bres[start:end] = to_numpy((v @ inf_gpu) / n)
        del draws, v

    return bres
