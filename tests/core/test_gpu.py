"""Tests for the CuPy backend and the GPU-accelerated operations on a CUDA GPU."""

from unittest.mock import patch

import numpy as np
import pytest

import moderndid
from moderndid.core.bootstrap import compute_column_std, multiplier_bootstrap
from moderndid.cupy.backend import get_backend, set_backend, to_device, to_numpy, use_backend
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


@requires_gpu
def test_multiplier_bootstrap_on_gpu_draws_balanced_independent_signs():
    n_units = 40
    inf_func = n_units * np.eye(n_units)

    with use_backend("cupy"):
        draws = multiplier_bootstrap(inf_func, biters=500, random_state=42)

    assert isinstance(draws, np.ndarray)
    np.testing.assert_array_equal(np.abs(draws), np.ones((500, n_units)))
    np.testing.assert_allclose(draws.mean(axis=0), 0.0, atol=0.25)
    np.testing.assert_allclose(np.corrcoef(draws, rowvar=False), np.eye(n_units), atol=0.25)


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
    np.testing.assert_allclose(
        gpu_result["bres"].mean(axis=0), cpu_result["bres"].mean(axis=0), atol=0.15 * n_units / n_clusters
    )
    set_backend("numpy")


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
def test_cont_did_cupy_runs():
    data = moderndid.gen_cont_did_data(n=200, seed=42)
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


@requires_gpu
def test_att_gt_cupy_end_to_end():
    df = moderndid.load_mpdta()
    assert get_backend() is np

    result = moderndid.att_gt(
        data=df,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        est_method="dr",
        boot=False,
        backend="cupy",
    )

    assert get_backend() is np
    assert result is not None
    assert len(result.att_gt) > 0
    assert np.all(np.isfinite(result.att_gt))


@requires_gpu
@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
def test_att_gt_cupy_vs_numpy(est_method):
    df = moderndid.load_mpdta()

    result_cpu = moderndid.att_gt(
        data=df,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        est_method=est_method,
        boot=False,
        backend="numpy",
    )

    result_gpu = moderndid.att_gt(
        data=df,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        est_method=est_method,
        boot=False,
        backend="cupy",
    )

    np.testing.assert_allclose(result_gpu.att_gt, result_cpu.att_gt, rtol=1e-8)
    np.testing.assert_allclose(result_gpu.se_gt, result_cpu.se_gt, rtol=1e-6)


@requires_gpu
def test_ddd_cupy_end_to_end():
    dgp = moderndid.gen_ddd_2periods(n=500, dgp_type=1, random_state=42)

    result = moderndid.ddd(
        data=dgp["data"],
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
        xformla="~ cov1 + cov2 + cov3 + cov4",
        est_method="dr",
        backend="cupy",
    )

    assert get_backend() is np
    assert result is not None
    assert np.isfinite(result.att)


@requires_gpu
@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
def test_ddd_cupy_vs_numpy(est_method):
    dgp = moderndid.gen_ddd_2periods(n=500, dgp_type=1, random_state=42)

    result_cpu = moderndid.ddd(
        data=dgp["data"],
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
        xformla="~ cov1 + cov2 + cov3 + cov4",
        est_method=est_method,
        backend="numpy",
    )

    result_gpu = moderndid.ddd(
        data=dgp["data"],
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
        xformla="~ cov1 + cov2 + cov3 + cov4",
        est_method=est_method,
        backend="cupy",
    )

    np.testing.assert_allclose(result_gpu.att, result_cpu.att, rtol=1e-8)
    np.testing.assert_allclose(result_gpu.se, result_cpu.se, rtol=1e-6)


@requires_gpu
def test_wls_oom_raises_memory_error():
    set_backend("cupy")

    class FakeOOM(Exception):
        pass

    FakeOOM.__name__ = "OutOfMemoryError"

    with patch("moderndid.cupy.regression.get_backend") as mock_be:
        mock_xp = type("FakeXP", (), {"float64": np.float64})()
        mock_xp.asarray = lambda *a, **kw: (_ for _ in ()).throw(FakeOOM("fake OOM"))
        mock_be.return_value = mock_xp

        with pytest.raises(MemoryError, match="GPU out of memory during WLS"):
            cupy_wls(np.zeros(10), np.zeros((10, 2)), np.ones(10))

    set_backend("numpy")


@requires_gpu
def test_logistic_irls_oom_raises_memory_error():
    set_backend("cupy")

    class FakeOOM(Exception):
        pass

    FakeOOM.__name__ = "OutOfMemoryError"

    with patch("moderndid.cupy.regression.get_backend") as mock_be:
        mock_xp = type("FakeXP", (), {"float64": np.float64})()
        mock_xp.zeros = lambda *a, **kw: (_ for _ in ()).throw(FakeOOM("fake OOM"))
        mock_be.return_value = mock_xp

        with pytest.raises(MemoryError, match="GPU out of memory during logistic IRLS"):
            cupy_logistic_irls(np.zeros(10), np.zeros((10, 2)), np.ones(10))

    set_backend("numpy")
