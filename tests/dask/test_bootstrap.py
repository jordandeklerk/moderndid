"""Tests for distributed bootstrap."""

import numpy as np

from moderndid.dask._bootstrap import _local_bootstrap, _sum_bootstrap_pair


def test_local_bootstrap_shape(rng):
    n, k, biters = 30, 4, 50
    inf_func = rng.standard_normal((n, k))
    bres, zeros, n_local = _local_bootstrap(inf_func, biters, seed=0)
    assert bres.shape == (biters, k)
    assert zeros.shape == (k,)
    assert n_local == n
    np.testing.assert_array_equal(zeros, np.zeros(k))


def test_local_bootstrap_reproducible(rng):
    inf_func = rng.standard_normal((20, 3))
    a = _local_bootstrap(inf_func, 10, seed=42)
    b = _local_bootstrap(inf_func, 10, seed=42)
    np.testing.assert_array_equal(a[0], b[0])


def test_local_bootstrap_different_seeds(rng):
    inf_func = rng.standard_normal((20, 3))
    a = _local_bootstrap(inf_func, 10, seed=1)
    b = _local_bootstrap(inf_func, 10, seed=2)
    assert not np.array_equal(a[0], b[0])


def test_sum_bootstrap_pair(rng):
    k, biters = 3, 10
    a = (rng.standard_normal((biters, k)), np.zeros(k), 5)
    b = (rng.standard_normal((biters, k)), np.zeros(k), 7)
    result = _sum_bootstrap_pair(a, b)
    np.testing.assert_allclose(result[0], a[0] + b[0])
    np.testing.assert_allclose(result[1], a[1] + b[1])
    assert result[2] == 12
