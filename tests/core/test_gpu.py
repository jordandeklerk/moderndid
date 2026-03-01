"""Tests for GPU backend dispatch and GPU-accelerated operations."""

import numpy as np
import pytest

import moderndid
import moderndid.cupy.backend as _backend_mod
from moderndid.core.numba_utils import aggregate_by_cluster, compute_column_std, multiplier_bootstrap
from moderndid.cupy.backend import _init_rmm_pool, get_backend, set_backend, to_device, to_numpy, use_backend
from moderndid.cupy.regression import cupy_logistic_irls, cupy_wls
from moderndid.did.mboot import mboot
from tests.helpers import importorskip

cp = importorskip("cupy")


def _has_cuda_gpu():
    try:
        set_backend("cupy")
        set_backend("numpy")
        return True
    except RuntimeError:
        return False


requires_gpu = pytest.mark.skipif(not _has_cuda_gpu(), reason="No CUDA GPU available")


def test_default_backend_is_numpy():
    set_backend("numpy")
    assert get_backend() is np


def test_set_backend_invalid():
    with pytest.raises(ValueError, match="Unknown backend"):
        set_backend("tensorflow")


def test_to_numpy_with_numpy_array():
    arr = np.array([1.0, 2.0, 3.0])
    result = to_numpy(arr)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, arr)


def test_to_device_with_numpy_backend():
    set_backend("numpy")
    arr = np.array([1.0, 2.0, 3.0])
    result = to_device(arr)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, arr)


@requires_gpu
def test_set_cupy_backend():
    set_backend("cupy")
    assert get_backend() is cp
    set_backend("numpy")


@requires_gpu
def test_to_device_cupy():
    set_backend("cupy")
    arr = np.array([1.0, 2.0, 3.0])
    result = to_device(arr)
    assert isinstance(result, cp.ndarray)
    np.testing.assert_array_equal(cp.asnumpy(result), arr)
    set_backend("numpy")


@requires_gpu
def test_to_numpy_cupy():
    gpu_arr = cp.array([1.0, 2.0, 3.0])
    result = to_numpy(gpu_arr)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


@requires_gpu
def test_roundtrip_numpy_cupy():
    set_backend("cupy")
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    on_gpu = to_device(arr)
    assert isinstance(on_gpu, cp.ndarray)
    back_cpu = to_numpy(on_gpu)
    assert isinstance(back_cpu, np.ndarray)
    np.testing.assert_array_equal(back_cpu, arr)
    set_backend("numpy")


def test_wls_simple_ols():
    set_backend("numpy")
    rng = np.random.default_rng(42)
    n, k = 200, 3
    X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
    beta_true = np.array([1.0, 2.0, -0.5])
    y = X @ beta_true + rng.standard_normal(n) * 0.1
    weights = np.ones(n)

    beta_hat, fitted = cupy_wls(y, X, weights)
    np.testing.assert_allclose(beta_hat, beta_true, atol=0.2)
    np.testing.assert_allclose(fitted, X @ beta_hat)


def test_wls_weighted():
    set_backend("numpy")
    rng = np.random.default_rng(123)
    n = 100
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    beta_true = np.array([3.0, -1.0])
    y = X @ beta_true + rng.standard_normal(n) * 0.5
    weights = rng.uniform(0.5, 2.0, n)

    beta_hat, fitted = cupy_wls(y, X, weights)
    np.testing.assert_allclose(beta_hat, beta_true, atol=0.3)
    np.testing.assert_allclose(fitted, X @ beta_hat)


@requires_gpu
def test_cupy_vs_cpu_wls():
    set_backend("numpy")
    rng = np.random.default_rng(99)
    n, k = 500, 4
    X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
    beta_true = np.array([1.0, -2.0, 0.5, 3.0])
    y = X @ beta_true + rng.standard_normal(n) * 0.3
    weights = rng.uniform(0.1, 3.0, n)

    beta_cpu, fitted_cpu = cupy_wls(y, X, weights)

    set_backend("cupy")
    beta_gpu, fitted_gpu = cupy_wls(cp.asarray(y), cp.asarray(X), cp.asarray(weights))
    beta_gpu = to_numpy(beta_gpu)
    fitted_gpu = to_numpy(fitted_gpu)

    np.testing.assert_allclose(beta_gpu, beta_cpu, rtol=1e-5)
    np.testing.assert_allclose(fitted_gpu, fitted_cpu, rtol=1e-5)
    set_backend("numpy")


def test_logistic_separable_data():
    set_backend("numpy")
    rng = np.random.default_rng(42)
    n = 300
    x1 = rng.standard_normal(n)
    prob = 1.0 / (1.0 + np.exp(-(1.0 + 2.0 * x1)))
    y = rng.binomial(1, prob).astype(np.float64)
    X = np.column_stack([np.ones(n), x1])
    weights = np.ones(n)

    beta, mu = cupy_logistic_irls(y, X, weights)
    np.testing.assert_allclose(beta, [1.0, 2.0], atol=0.5)
    assert mu.shape == (n,)
    assert np.all((mu > 0) & (mu < 1))


def test_logistic_weighted():
    set_backend("numpy")
    rng = np.random.default_rng(77)
    n = 500
    x1 = rng.standard_normal(n)
    prob = 1.0 / (1.0 + np.exp(-(0.5 + 1.5 * x1)))
    y = rng.binomial(1, prob).astype(np.float64)
    X = np.column_stack([np.ones(n), x1])
    weights = rng.uniform(0.5, 2.0, n)

    beta, mu = cupy_logistic_irls(y, X, weights)
    np.testing.assert_allclose(beta, [0.5, 1.5], atol=0.5)
    assert np.all((mu > 0) & (mu < 1))


@requires_gpu
def test_cupy_vs_cpu_logistic():
    set_backend("numpy")
    rng = np.random.default_rng(55)
    n = 400
    x1 = rng.standard_normal(n)
    prob = 1.0 / (1.0 + np.exp(-(1.0 + 1.0 * x1)))
    y = rng.binomial(1, prob).astype(np.float64)
    X = np.column_stack([np.ones(n), x1])
    weights = np.ones(n)

    beta_cpu, mu_cpu = cupy_logistic_irls(y, X, weights)

    set_backend("cupy")
    beta_gpu, mu_gpu = cupy_logistic_irls(cp.asarray(y), cp.asarray(X), cp.asarray(weights))
    beta_gpu = to_numpy(beta_gpu)
    mu_gpu = to_numpy(mu_gpu)

    np.testing.assert_allclose(beta_gpu, beta_cpu, rtol=1e-4)
    np.testing.assert_allclose(mu_gpu, mu_cpu, rtol=1e-4)
    set_backend("numpy")


def test_cpu_bootstrap_shape():
    set_backend("numpy")
    rng = np.random.default_rng(42)
    inf_func = rng.standard_normal((100, 3))
    result = multiplier_bootstrap(inf_func, biters=50, random_state=42)
    assert result.shape == (50, 3)


@requires_gpu
def test_cupy_bootstrap_shape():
    set_backend("cupy")
    rng = np.random.default_rng(42)
    inf_func = rng.standard_normal((100, 3))
    result = multiplier_bootstrap(inf_func, biters=50, random_state=42)
    assert isinstance(result, np.ndarray)
    assert result.shape == (50, 3)
    set_backend("numpy")


@requires_gpu
@pytest.mark.parametrize(
    "shape,seed,biters",
    [
        ((200, 4), 10, 99),
        ((150,), 20, 50),
    ],
    ids=["2d", "1d"],
)
def test_cupy_vs_cpu_bootstrap(shape, seed, biters):
    rng = np.random.default_rng(seed)
    inf_func = rng.standard_normal(shape)

    set_backend("numpy")
    cpu_result = multiplier_bootstrap(inf_func, biters=biters, random_state=42)

    set_backend("cupy")
    gpu_result = multiplier_bootstrap(inf_func, biters=biters, random_state=42)

    assert gpu_result.shape == cpu_result.shape
    assert np.all(np.isfinite(gpu_result))
    np.testing.assert_allclose(gpu_result.mean(axis=0), cpu_result.mean(axis=0), atol=0.15)
    np.testing.assert_allclose(gpu_result.std(axis=0), cpu_result.std(axis=0), atol=0.15)
    set_backend("numpy")


@requires_gpu
def test_cupy_vs_cpu_mboot():
    set_backend("numpy")
    rng = np.random.default_rng(42)
    n = 200
    k = 3
    inf_func = rng.standard_normal((n, k))

    cpu_result = mboot(inf_func, n_units=n, biters=999, random_state=42)

    set_backend("cupy")
    gpu_result = mboot(inf_func, n_units=n, biters=999, random_state=42)

    assert gpu_result["bres"].shape == cpu_result["bres"].shape
    assert np.all(np.isfinite(gpu_result["se"][~np.isnan(cpu_result["se"])]))
    np.testing.assert_allclose(gpu_result["se"], cpu_result["se"], rtol=0.3)
    np.testing.assert_allclose(gpu_result["bres"].mean(axis=0), cpu_result["bres"].mean(axis=0), atol=0.3)
    set_backend("numpy")


@requires_gpu
def test_cupy_vs_cpu_mboot_clustered():
    set_backend("numpy")
    rng = np.random.default_rng(42)
    n_units = 120
    n_clusters = 20
    cluster = np.repeat(np.arange(n_clusters), n_units // n_clusters)
    inf_func = rng.standard_normal(n_units)

    cpu_result = mboot(inf_func, n_units=n_units, biters=99, cluster=cluster, random_state=42)

    set_backend("cupy")
    gpu_result = mboot(inf_func, n_units=n_units, biters=99, cluster=cluster, random_state=42)

    assert gpu_result["bres"].shape == cpu_result["bres"].shape
    assert np.all(np.isfinite(gpu_result["se"][~np.isnan(cpu_result["se"])]))
    np.testing.assert_allclose(gpu_result["se"], cpu_result["se"], rtol=0.3)
    np.testing.assert_allclose(gpu_result["bres"].mean(axis=0), cpu_result["bres"].mean(axis=0), atol=0.15)
    set_backend("numpy")


def test_cpu_aggregate():
    set_backend("numpy")
    rng = np.random.default_rng(42)
    n, k = 100, 3
    inf_func = rng.standard_normal((n, k))
    cluster = np.repeat(np.arange(10), 10)

    result, n_clusters = aggregate_by_cluster(inf_func, cluster)
    assert result.shape == (10, k)
    assert n_clusters == 10


@requires_gpu
@pytest.mark.parametrize(
    "cluster_array,n,k",
    [
        (np.repeat(np.arange(20), 10), 200, 5),
        (np.concatenate([np.zeros(50), np.ones(30), np.full(20, 2)]), 100, 2),
    ],
    ids=["equal_clusters", "unequal_clusters"],
)
def test_cupy_vs_cpu_aggregate(cluster_array, n, k):
    rng = np.random.default_rng(42)
    inf_func = rng.standard_normal((n, k))

    set_backend("numpy")
    cpu_result, cpu_n = aggregate_by_cluster(inf_func, cluster_array)

    set_backend("cupy")
    gpu_result, gpu_n = aggregate_by_cluster(inf_func, cluster_array)

    assert cpu_n == gpu_n
    np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-5)
    set_backend("numpy")


def test_cpu_column_std():
    set_backend("numpy")
    rng = np.random.default_rng(42)
    matrix = rng.standard_normal((100, 4))
    result = compute_column_std(matrix)
    expected = np.nanstd(matrix, axis=0, ddof=1)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


@requires_gpu
@pytest.mark.parametrize("inject_nans", [False, True], ids=["no_nans", "with_nans"])
def test_cupy_vs_cpu_column_std(inject_nans):
    rng = np.random.default_rng(42)
    matrix = rng.standard_normal((200, 5)) if not inject_nans else rng.standard_normal((100, 3))

    if inject_nans:
        matrix[10:20, 0] = np.nan
        matrix[50:60, 2] = np.nan

    set_backend("numpy")
    cpu_result = compute_column_std(matrix)

    set_backend("cupy")
    gpu_result = compute_column_std(matrix)

    np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-5)
    set_backend("numpy")


@requires_gpu
def test_rmm_initialized_after_set_backend():
    set_backend("cupy")
    try:
        import rmm  # noqa: F401

        assert _backend_mod._rmm_initialized is True
    except ImportError:
        pass
    set_backend("numpy")


@requires_gpu
def test_rmm_initialized_after_use_backend():
    _backend_mod._rmm_initialized = False
    try:
        import rmm  # noqa: F401

        with use_backend("cupy"):
            assert _backend_mod._rmm_initialized is True
    except ImportError:
        pass
    finally:
        _init_rmm_pool()
        set_backend("numpy")


@requires_gpu
def test_rmm_skips_reinitialize_when_already_active():
    try:
        from unittest.mock import patch

        import rmm
        from rmm.allocators.cupy import rmm_cupy_allocator

        cp.cuda.set_allocator(rmm_cupy_allocator)
        _backend_mod._rmm_initialized = False

        with patch.object(rmm, "reinitialize") as mock_reinit:
            _init_rmm_pool()
            mock_reinit.assert_not_called()

        assert _backend_mod._rmm_initialized is True
    except ImportError:
        pytest.skip("rmm not installed")
    finally:
        set_backend("numpy")


@requires_gpu
def test_cont_did_cupy_runs():
    data = moderndid.simulate_cont_did_data(n=200, seed=42)
    result = moderndid.cont_did(
        data=data,
        yname="Y",
        tname="time_period",
        idname="id",
        gname="G",
        dname="D",
        target_parameter="level",
        aggregation="dose",
        degree=3,
        biters=20,
        random_state=42,
        backend="cupy",
    )
    assert result.att_d is not None
    assert np.all(np.isfinite(result.att_d))
    set_backend("numpy")
