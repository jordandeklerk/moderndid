"""Tests for GPU backend dispatch and GPU-accelerated operations."""

import numpy as np
import pytest

from moderndid.core.backend import HAS_CUPY, get_backend, set_backend, to_device, to_numpy
from moderndid.core.gpu import cupy_logistic_irls, cupy_wls
from moderndid.core.numba_utils import aggregate_by_cluster, compute_column_std, multiplier_bootstrap
from moderndid.did.mboot import mboot


def _cupy_available():
    """Check if CuPy is installed and a CUDA GPU is actually available."""
    if not HAS_CUPY:
        return False
    try:
        set_backend("cupy")
        set_backend("numpy")
        return True
    except (RuntimeError, ImportError):
        return False


_HAS_CUPY_GPU = _cupy_available()

requires_cupy = pytest.mark.skipif(not _HAS_CUPY_GPU, reason="CuPy not installed or no CUDA GPU")


class TestBackendDispatch:
    def test_default_backend_is_numpy(self):
        set_backend("numpy")
        assert get_backend() is np

    def test_set_backend_invalid(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            set_backend("tensorflow")

    def test_set_backend_cupy_without_install(self):
        if HAS_CUPY:
            pytest.skip("CuPy is installed")
        with pytest.raises(ImportError, match="CuPy is not installed"):
            set_backend("cupy")

    def test_set_backend_cupy_without_cupy_gpu(self):
        if not HAS_CUPY or _HAS_CUPY_GPU:
            pytest.skip("CuPy not installed or GPU is available")
        with pytest.raises(RuntimeError, match="no CUDA GPU is available"):
            set_backend("cupy")

    def test_to_numpy_with_numpy_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = to_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_to_device_with_numpy_backend(self):
        set_backend("numpy")
        arr = np.array([1.0, 2.0, 3.0])
        result = to_device(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    @requires_cupy
    def test_set_cupy_backend(self):
        set_backend("cupy")
        import cupy as cp

        assert get_backend() is cp
        set_backend("numpy")

    @requires_cupy
    def test_to_device_cupy(self):
        import cupy as cp

        set_backend("cupy")
        arr = np.array([1.0, 2.0, 3.0])
        result = to_device(arr)
        assert isinstance(result, cp.ndarray)
        np.testing.assert_array_equal(cp.asnumpy(result), arr)
        set_backend("numpy")

    @requires_cupy
    def test_to_numpy_cupy(self):
        import cupy as cp

        gpu_arr = cp.array([1.0, 2.0, 3.0])
        result = to_numpy(gpu_arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    @requires_cupy
    def test_roundtrip_numpy_cupy(self):
        import cupy as cp

        set_backend("cupy")
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        on_gpu = to_device(arr)
        assert isinstance(on_gpu, cp.ndarray)
        back_cpu = to_numpy(on_gpu)
        assert isinstance(back_cpu, np.ndarray)
        np.testing.assert_array_equal(back_cpu, arr)
        set_backend("numpy")


class TestGpuWlsNumpy:
    """Test cupy_wls on NumPy backend against known solutions."""

    def setup_method(self):
        set_backend("numpy")

    def test_simple_ols(self):
        rng = np.random.default_rng(42)
        n, k = 200, 3
        X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
        beta_true = np.array([1.0, 2.0, -0.5])
        y = X @ beta_true + rng.standard_normal(n) * 0.1
        weights = np.ones(n)

        beta_hat, fitted = cupy_wls(y, X, weights)
        np.testing.assert_allclose(beta_hat, beta_true, atol=0.2)
        np.testing.assert_allclose(fitted, X @ beta_hat)

    def test_weighted_ls(self):
        rng = np.random.default_rng(123)
        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        beta_true = np.array([3.0, -1.0])
        y = X @ beta_true + rng.standard_normal(n) * 0.5
        weights = rng.uniform(0.5, 2.0, n)

        beta_hat, fitted = cupy_wls(y, X, weights)
        np.testing.assert_allclose(beta_hat, beta_true, atol=0.3)
        np.testing.assert_allclose(fitted, X @ beta_hat)

    @requires_cupy
    def test_cupy_vs_cpu_wls(self):
        rng = np.random.default_rng(99)
        n, k = 500, 4
        X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
        beta_true = np.array([1.0, -2.0, 0.5, 3.0])
        y = X @ beta_true + rng.standard_normal(n) * 0.3
        weights = rng.uniform(0.1, 3.0, n)

        set_backend("numpy")
        beta_cpu, fitted_cpu = cupy_wls(y, X, weights)

        import cupy as cp

        set_backend("cupy")
        beta_gpu, fitted_gpu = cupy_wls(cp.asarray(y), cp.asarray(X), cp.asarray(weights))
        beta_gpu = to_numpy(beta_gpu)
        fitted_gpu = to_numpy(fitted_gpu)

        np.testing.assert_allclose(beta_gpu, beta_cpu, rtol=1e-5)
        np.testing.assert_allclose(fitted_gpu, fitted_cpu, rtol=1e-5)
        set_backend("numpy")


class TestGpuLogisticNumpy:
    """Test cupy_logistic_irls on NumPy backend against known solutions."""

    def setup_method(self):
        set_backend("numpy")

    def test_separable_data(self):
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

    def test_weighted_logistic(self):
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

    @requires_cupy
    def test_cupy_vs_cpu_logistic(self):
        rng = np.random.default_rng(55)
        n = 400
        x1 = rng.standard_normal(n)
        prob = 1.0 / (1.0 + np.exp(-(1.0 + 1.0 * x1)))
        y = rng.binomial(1, prob).astype(np.float64)
        X = np.column_stack([np.ones(n), x1])
        weights = np.ones(n)

        set_backend("numpy")
        beta_cpu, mu_cpu = cupy_logistic_irls(y, X, weights)

        import cupy as cp

        set_backend("cupy")
        beta_gpu, mu_gpu = cupy_logistic_irls(cp.asarray(y), cp.asarray(X), cp.asarray(weights))
        beta_gpu = to_numpy(beta_gpu)
        mu_gpu = to_numpy(mu_gpu)

        np.testing.assert_allclose(beta_gpu, beta_cpu, rtol=1e-4)
        np.testing.assert_allclose(mu_gpu, mu_cpu, rtol=1e-4)
        set_backend("numpy")


class TestMultiplierBootstrapGpu:
    """Test GPU-accelerated multiplier bootstrap against CPU."""

    def setup_method(self):
        set_backend("numpy")

    def test_cpu_bootstrap_shape(self):
        rng = np.random.default_rng(42)
        inf_func = rng.standard_normal((100, 3))
        result = multiplier_bootstrap(inf_func, biters=50, random_state=42)
        assert result.shape == (50, 3)

    @requires_cupy
    def test_cupy_bootstrap_shape(self):
        set_backend("cupy")
        rng = np.random.default_rng(42)
        inf_func = rng.standard_normal((100, 3))
        result = multiplier_bootstrap(inf_func, biters=50, random_state=42)
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 3)
        set_backend("numpy")

    @requires_cupy
    def test_cupy_vs_cpu_bootstrap(self):
        rng = np.random.default_rng(10)
        inf_func = rng.standard_normal((200, 4))

        set_backend("numpy")
        cpu_result = multiplier_bootstrap(inf_func, biters=99, random_state=42)

        set_backend("cupy")
        gpu_result = multiplier_bootstrap(inf_func, biters=99, random_state=42)

        assert gpu_result.shape == cpu_result.shape
        assert np.all(np.isfinite(gpu_result))
        np.testing.assert_allclose(gpu_result.mean(axis=0), cpu_result.mean(axis=0), atol=0.15)
        np.testing.assert_allclose(gpu_result.std(axis=0), cpu_result.std(axis=0), atol=0.15)
        set_backend("numpy")

    @requires_cupy
    def test_cupy_vs_cpu_bootstrap_1d(self):
        rng = np.random.default_rng(20)
        inf_func = rng.standard_normal(150)

        set_backend("numpy")
        cpu_result = multiplier_bootstrap(inf_func, biters=50, random_state=7)

        set_backend("cupy")
        gpu_result = multiplier_bootstrap(inf_func, biters=50, random_state=7)

        assert gpu_result.shape == cpu_result.shape
        assert np.all(np.isfinite(gpu_result))
        np.testing.assert_allclose(gpu_result.mean(axis=0), cpu_result.mean(axis=0), atol=0.15)
        np.testing.assert_allclose(gpu_result.std(axis=0), cpu_result.std(axis=0), atol=0.15)
        set_backend("numpy")


class TestMbootGpu:
    """Test the full mboot function with GPU backend."""

    def setup_method(self):
        set_backend("numpy")

    @requires_cupy
    def test_cupy_vs_cpu_mboot(self):
        rng = np.random.default_rng(42)
        n = 200
        k = 3
        inf_func = rng.standard_normal((n, k))

        set_backend("numpy")
        cpu_result = mboot(inf_func, n_units=n, biters=99, random_state=42)

        set_backend("cupy")
        gpu_result = mboot(inf_func, n_units=n, biters=99, random_state=42)

        assert gpu_result["bres"].shape == cpu_result["bres"].shape
        assert np.all(np.isfinite(gpu_result["se"][~np.isnan(cpu_result["se"])]))
        np.testing.assert_allclose(gpu_result["se"], cpu_result["se"], rtol=0.3)
        np.testing.assert_allclose(gpu_result["bres"].mean(axis=0), cpu_result["bres"].mean(axis=0), atol=0.15)
        set_backend("numpy")

    @requires_cupy
    def test_cupy_vs_cpu_mboot_clustered(self):
        rng = np.random.default_rng(42)
        n_units = 120
        n_clusters = 20
        cluster = np.repeat(np.arange(n_clusters), n_units // n_clusters)
        inf_func = rng.standard_normal(n_units)

        set_backend("numpy")
        cpu_result = mboot(inf_func, n_units=n_units, biters=99, cluster=cluster, random_state=42)

        set_backend("cupy")
        gpu_result = mboot(inf_func, n_units=n_units, biters=99, cluster=cluster, random_state=42)

        assert gpu_result["bres"].shape == cpu_result["bres"].shape
        assert np.all(np.isfinite(gpu_result["se"][~np.isnan(cpu_result["se"])]))
        np.testing.assert_allclose(gpu_result["se"], cpu_result["se"], rtol=0.3)
        np.testing.assert_allclose(gpu_result["bres"].mean(axis=0), cpu_result["bres"].mean(axis=0), atol=0.15)
        set_backend("numpy")


class TestAggregateByClusterGpu:
    """Test GPU-accelerated cluster aggregation."""

    def setup_method(self):
        set_backend("numpy")

    def test_cpu_aggregate(self):
        rng = np.random.default_rng(42)
        n, k = 100, 3
        inf_func = rng.standard_normal((n, k))
        cluster = np.repeat(np.arange(10), 10)

        result, n_clusters = aggregate_by_cluster(inf_func, cluster)
        assert result.shape == (10, k)
        assert n_clusters == 10

    @requires_cupy
    def test_cupy_vs_cpu_aggregate(self):
        rng = np.random.default_rng(42)
        n, k = 200, 5
        inf_func = rng.standard_normal((n, k))
        cluster = np.repeat(np.arange(20), 10)

        set_backend("numpy")
        cpu_result, cpu_n = aggregate_by_cluster(inf_func, cluster)

        set_backend("cupy")
        gpu_result, gpu_n = aggregate_by_cluster(inf_func, cluster)

        assert cpu_n == gpu_n
        np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-5)
        set_backend("numpy")

    @requires_cupy
    def test_cupy_vs_cpu_aggregate_unequal_clusters(self):
        rng = np.random.default_rng(77)
        cluster = np.concatenate([np.zeros(50), np.ones(30), np.full(20, 2)])
        inf_func = rng.standard_normal((100, 2))

        set_backend("numpy")
        cpu_result, _ = aggregate_by_cluster(inf_func, cluster)

        set_backend("cupy")
        gpu_result, _ = aggregate_by_cluster(inf_func, cluster)

        np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-5)
        set_backend("numpy")


class TestComputeColumnStdGpu:
    """Test GPU-accelerated column standard deviation."""

    def setup_method(self):
        set_backend("numpy")

    def test_cpu_column_std(self):
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 4))
        result = compute_column_std(matrix)
        expected = np.nanstd(matrix, axis=0, ddof=1)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    @requires_cupy
    def test_cupy_vs_cpu_column_std(self):
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((200, 5))

        set_backend("numpy")
        cpu_result = compute_column_std(matrix)

        set_backend("cupy")
        gpu_result = compute_column_std(matrix)

        np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-5)
        set_backend("numpy")

    @requires_cupy
    def test_cupy_column_std_with_nans(self):
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 3))
        matrix[10:20, 0] = np.nan
        matrix[50:60, 2] = np.nan

        set_backend("numpy")
        cpu_result = compute_column_std(matrix)

        set_backend("cupy")
        gpu_result = compute_column_std(matrix)

        np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-5)
        set_backend("numpy")
