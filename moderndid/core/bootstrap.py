"""Bootstrap and variance estimation helpers."""

import hashlib

import numpy as np

from moderndid.cupy.backend import get_backend, to_numpy
from moderndid.cupy.bootstrap import _multiplier_bootstrap_cupy

__all__ = [
    "canonical_keys",
    "compute_cluster_sums",
    "compute_column_std",
    "gather_bootstrap_indices",
    "keyed_rademacher",
    "multiplier_bootstrap",
]


def canonical_keys(values):
    """Map identifiers to 64-bit keys that depend only on each identifier's value.

    An integer, or a float that holds an integer below 2**63 in absolute value,
    maps to the bits of that integer. Any other float maps to the bits of its
    value and a string to its 8-byte BLAKE2b digest read as a little-endian integer.

    Parameters
    ----------
    values : ndarray
        Identifiers of one numeric or string type.

    Returns
    -------
    ndarray
        Key of each identifier as unsigned 64-bit integers.
    """
    values = np.asarray(values)
    if values.dtype.kind in "biu":
        return values.astype(np.int64).view(np.uint64)
    if values.dtype.kind == "f":
        as_float = values.astype(np.float64)
        keys = as_float.view(np.uint64).copy()
        integral = np.isfinite(as_float) & (np.abs(as_float) < 2.0**63)
        integral[integral] = as_float[integral] == np.round(as_float[integral])
        keys[integral] = as_float[integral].astype(np.int64).view(np.uint64)
        return keys
    unique_values, inverse = np.unique(values.astype(str), return_inverse=True)
    digests = np.array(
        [int.from_bytes(hashlib.blake2b(v.encode(), digest_size=8).digest(), "little") for v in unique_values],
        dtype=np.uint64,
    )
    return digests[inverse.ravel()]


def keyed_rademacher(keys, random_state, biters):
    r"""Draw Rademacher multipliers that depend only on the seed and each key.

    A key's multiplier in draw :math:`b` stays the same however the keys are
    ordered or split across calls and however many draws a call makes.

    The multipliers come from the counter-based generator Philox4x32-10. The
    generator's key holds the low and high 32 bits of the seed. Its counter
    holds those of each key, the index of a block of 128 draws, and zero. Draw
    :math:`b` reads bit :math:`b \bmod 32` of word
    :math:`\lfloor b / 32 \rfloor \bmod 4` of block
    :math:`\lfloor b / 128 \rfloor`. A set bit gives 1 and a clear bit gives -1.

    Parameters
    ----------
    keys : ndarray
        Unsigned 64-bit keys of the units or clusters of shape (m,), such as
        those from :func:`canonical_keys`.
    random_state : int
        Seed of the multipliers, from 0 to 2**64 - 1.
    biters : int
        Number of bootstrap draws, at least 1.

    Returns
    -------
    ndarray
        Multipliers of shape (m, biters), each 1 or -1.
    """
    if isinstance(random_state, bool) or not isinstance(random_state, int | np.integer):
        raise ValueError(f"random_state={random_state!r} is not valid. Must be an integer from 0 to 2**64 - 1.")
    seed = int(random_state)
    if not 0 <= seed < 2**64:
        raise ValueError(f"random_state={random_state!r} is not valid. Must be an integer from 0 to 2**64 - 1.")
    if isinstance(biters, bool) or not isinstance(biters, int | np.integer) or biters < 1:
        raise ValueError(f"biters={biters!r} is not valid. Must be a positive integer.")

    return _keyed_sign_blocks(keys, seed, 0, -(-int(biters) // 128))[:, :biters]


def compute_cluster_sums(influence_func, cluster_ids):
    """Compute sum of influence function values within each cluster.

    Parameters
    ----------
    influence_func : ndarray
        Influence function values for each unit.
    cluster_ids : ndarray
        Cluster identifiers for each unit.

    Returns
    -------
    tuple of ndarray
        Tuple containing:

        - **cluster_sums**: Sum of the influence function values within each cluster
        - **unique_clusters**: Cluster identifiers in sorted order, one for each sum
    """
    influence_func = np.asarray(influence_func, dtype=np.float64)
    unique_clusters, cluster_codes = _cluster_codes(cluster_ids)

    cluster_sums = np.bincount(cluster_codes, weights=influence_func, minlength=len(unique_clusters))

    return cluster_sums, unique_clusters


def multiplier_bootstrap(inf_func, biters, random_state=None):
    """Run the multiplier bootstrap using Rademacher weights.

    Parameters
    ----------
    inf_func : ndarray
        Influence function matrix of shape (n, k).
    biters : int
        Number of bootstrap iterations.
    random_state : int, Generator, or None, default None
        Controls random number generation for reproducibility.

    Returns
    -------
    ndarray
        Bootstrap results matrix of shape (biters, k).
    """
    inf_func = np.asarray(inf_func, dtype=np.float64)
    if inf_func.ndim == 1:
        inf_func = inf_func.reshape(-1, 1)

    xp = get_backend()
    if xp is not np:
        return _multiplier_bootstrap_cupy(inf_func, biters, random_state)

    n = inf_func.shape[0]
    rng = np.random.default_rng(random_state)
    # A block's float64 uniforms coexist with the previous block's signs. With 2**24 signs per block, the peak stays
    # near 300 MB.
    block = max(1, min((1 << 24) // n, biters))

    bres = np.empty((biters, inf_func.shape[1]))
    for start in range(0, biters, block):
        stop = min(start + block, biters)
        # binomial(1, 0.5) returns one exactly when its uniform draw exceeds one half. Comparing the same uniforms
        # reproduces every seeded draw at a fraction of the cost.
        signs = np.where(rng.random((stop - start, n)) > 0.5, 1.0, -1.0)
        bres[start:stop] = signs @ inf_func / n
    return bres


def gather_bootstrap_indices(sampled_cluster_ids, cluster_starts, cluster_counts):
    """Gather row indices for a bootstrap sample.

    Parameters
    ----------
    sampled_cluster_ids : ndarray
        Array of sampled cluster indices (with replacement).
    cluster_starts : ndarray
        Starting row index for each cluster.
    cluster_counts : ndarray
        Number of rows in each cluster.

    Returns
    -------
    ndarray
        Row indices for the bootstrap sample.
    """
    sampled_cluster_ids = np.asarray(sampled_cluster_ids, dtype=np.int64)
    counts = np.asarray(cluster_counts, dtype=np.int64)[sampled_cluster_ids]
    starts = np.asarray(cluster_starts, dtype=np.int64)[sampled_cluster_ids]

    # The index at output position p is the cluster's start plus p minus its block's first position. Repeating each
    # block's shift and adding p builds every block at once.
    shifts = starts - (np.cumsum(counts) - counts)
    return np.repeat(shifts, counts) + np.arange(counts.sum(), dtype=np.int64)


def compute_column_std(matrix):
    """Compute column-wise standard deviation ignoring NaNs.

    Parameters
    ----------
    matrix : ndarray
        2D array of shape (n_samples, n_columns).

    Returns
    -------
    ndarray
        Standard deviation for each column.
    """
    xp = get_backend()
    if xp is not np:
        m = xp.asarray(matrix, dtype=xp.float64)
        return to_numpy(xp.nanstd(m, axis=0, ddof=1))

    matrix = np.asarray(matrix, dtype=np.float64)
    observed = ~np.isnan(matrix)
    count = observed.sum(axis=0)
    mean = np.where(observed, matrix, 0.0).sum(axis=0) / np.maximum(count, 1)
    deviations = np.where(observed, matrix - mean, 0.0)
    sum_sq = (deviations * deviations).sum(axis=0)

    # A column with fewer than two values has no degrees of freedom. Only the other columns divide.
    std = np.full(matrix.shape[1], np.nan)
    enough = count > 1
    std[enough] = np.sqrt(sum_sq[enough] / (count[enough] - 1))
    return std


def _cluster_codes(cluster_ids):
    """Map cluster identifiers to codes from zero to one less than the number of clusters.

    Parameters
    ----------
    cluster_ids : ndarray
        Cluster identifiers of any type that can be sorted.

    Returns
    -------
    tuple of ndarray
        Tuple containing:

        - **unique_clusters**: Cluster identifiers in sorted order
        - **cluster_codes**: Position of each identifier in ``unique_clusters``
    """
    # A negative or very large identifier can't index a table of sums. Integers get codes too.
    unique_clusters, cluster_codes = np.unique(np.asarray(cluster_ids).ravel(), return_inverse=True)
    return unique_clusters, np.ascontiguousarray(cluster_codes.ravel(), dtype=np.int64)


def _keyed_sign_blocks(keys, seed, first_block, n_blocks):
    """Return the multipliers of consecutive blocks of 128 keyed draws with one row for each key."""
    keys = np.asarray(keys).astype(np.uint64).ravel()
    shape = (len(keys), n_blocks)
    words = _philox4x32(
        np.broadcast_to((keys & np.uint64(0xFFFFFFFF))[:, None], shape),
        np.broadcast_to((keys >> np.uint64(32))[:, None], shape),
        np.broadcast_to(np.arange(first_block, first_block + n_blocks, dtype=np.uint64)[None, :], shape),
        np.zeros(shape, dtype=np.uint64),
        seed & 0xFFFFFFFF,
        seed >> 32,
    )
    # A little-endian word stores bit j in byte j // 8 at position j % 8. Unpacking its bytes in little bit order puts
    # bit j of word k of a block at position 32k + j.
    block_bytes = np.stack(words, axis=-1).astype("<u4").view(np.uint8)
    bits = np.unpackbits(block_bytes, axis=-1, bitorder="little").reshape(len(keys), n_blocks * 128)
    return np.where(bits == 1, 1.0, -1.0)


def _philox4x32(c0, c1, c2, c3, k0, k1):
    """Apply the ten rounds of the Philox4x32 bijection to counter words held in 64-bit integers."""
    low = np.uint64(0xFFFFFFFF)
    k0, k1 = np.uint64(k0), np.uint64(k1)
    for _ in range(10):
        # A product of two 32-bit words fits in 64 bits. Its high and low halves come out exact.
        product0 = np.uint64(0xD2511F53) * c0
        product1 = np.uint64(0xCD9E8D57) * c2
        c0, c1, c2, c3 = (
            (product1 >> np.uint64(32)) ^ c1 ^ k0,
            product1 & low,
            (product0 >> np.uint64(32)) ^ c3 ^ k1,
            product0 & low,
        )
        k0 = (k0 + np.uint64(0x9E3779B9)) & low
        k1 = (k1 + np.uint64(0xBB67AE85)) & low
    return c0, c1, c2, c3
