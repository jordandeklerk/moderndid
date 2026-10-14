"""Tests for the array backend and its CuPy routes that run without a GPU."""

import contextvars
import re
import subprocess
import sys

import numpy as np
import pytest

from moderndid.core.bootstrap import compute_column_std, multiplier_bootstrap
from moderndid.cupy.backend import get_backend, set_backend, to_device, to_numpy, use_backend


def test_default_backend_is_numpy():
    assert contextvars.Context().run(get_backend) is np


def test_set_backend_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="Unknown backend 'tensorflow'"):
        set_backend("tensorflow")


def test_to_numpy_returns_a_numpy_array_unchanged():
    arr = np.array([1.0, 2.0, 3.0])

    result = to_numpy(arr)

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, arr)


def test_to_device_under_the_numpy_backend_keeps_a_numpy_array():
    arr = np.array([1.0, 2.0, 3.0])

    with use_backend("numpy"):
        result = to_device(arr)

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, arr)


@pytest.mark.parametrize(
    "probes",
    [
        [FileNotFoundError("nvidia-smi"), FileNotFoundError("nvcc")],
        [subprocess.CalledProcessError(9, "nvidia-smi"), subprocess.CalledProcessError(1, "nvcc")],
        [subprocess.TimeoutExpired("nvidia-smi", 5), subprocess.TimeoutExpired("nvcc", 5)],
        ["| NVIDIA-SMI 470.256.02    Driver Version: 470.256.02    CUDA Version: 11.4 |"],
    ],
    ids=["missing_tools", "failing_tools", "hanging_tools", "cuda_11"],
)
def test_set_backend_without_cupy_suggests_the_cuda_12_and_13_wheels(cuda_probes, probes):
    cuda_probes.side_effect = probes

    with pytest.raises(ImportError, match="CuPy is not installed") as raised:
        set_backend("cupy")

    assert 'uv add "cupy-cuda12x[ctk]"' in str(raised.value)
    assert 'uv add "cupy-cuda13x[ctk]"' in str(raised.value)
    assert "cupy-cuda11x" not in str(raised.value)


@pytest.mark.parametrize(
    "probes,wheel",
    [
        (["| NVIDIA-SMI 580.65.06    Driver Version: 580.65.06    CUDA Version: 13.0 |"], "cupy-cuda13x[ctk]"),
        (
            [subprocess.CalledProcessError(9, "nvidia-smi"), "Cuda compilation tools, release 12.4, V12.4.131"],
            "cupy-cuda12x[ctk]",
        ),
        (
            [subprocess.TimeoutExpired("nvidia-smi", 5), "Cuda compilation tools, release 12.4, V12.4.131"],
            "cupy-cuda12x[ctk]",
        ),
    ],
    ids=["nvidia_smi", "nvcc_after_failing_nvidia_smi", "nvcc_after_hanging_nvidia_smi"],
)
def test_set_backend_without_cupy_suggests_the_wheel_for_the_detected_cuda_version(cuda_probes, probes, wheel):
    cuda_probes.side_effect = probes

    with pytest.raises(ImportError, match=re.escape(f'uv add "{wheel}"')):
        set_backend("cupy")

    assert cuda_probes.call_count == len(probes)


def test_set_backend_reports_an_installed_cupy_that_fails_to_import(broken_cupy, load_backend):
    backend = load_backend()

    with pytest.raises(ImportError, match="CuPy is installed but failed to import") as raised:
        backend.set_backend("cupy")

    assert not backend.HAS_CUPY
    assert broken_cupy in str(raised.value)
    assert str(raised.value.__cause__) == broken_cupy


def test_set_backend_reports_a_missing_cupy_as_not_installed(monkeypatch, cuda_probes, load_backend):
    monkeypatch.setitem(sys.modules, "cupy", None)
    backend = load_backend()

    with pytest.raises(ImportError, match="CuPy is not installed") as raised:
        backend.set_backend("cupy")

    assert not backend.HAS_CUPY
    assert raised.value.__cause__ is None


def test_set_backend_without_a_cuda_device_raises(strict_cupy, monkeypatch):
    monkeypatch.setattr(strict_cupy, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="no CUDA GPU is available"):
        set_backend("cupy")


def test_multiplier_bootstrap_on_gpu_draws_balanced_independent_signs(strict_cupy):
    n_units = 40
    inf_func = n_units * np.eye(n_units)

    with use_backend("cupy"):
        draws = multiplier_bootstrap(inf_func, biters=500, random_state=42)

    assert isinstance(draws, np.ndarray)
    np.testing.assert_array_equal(np.abs(draws), np.ones((500, n_units)))
    np.testing.assert_allclose(draws.mean(axis=0), 0.0, atol=0.25)
    np.testing.assert_allclose(np.corrcoef(draws, rowvar=False), np.eye(n_units), atol=0.25)


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_compute_column_std_skips_missing_values_on_each_backend(strict_cupy, backend):
    rng = np.random.default_rng(42)
    matrix = rng.standard_normal((100, 3))
    matrix[10:20, 0] = np.nan

    with use_backend(backend):
        result = compute_column_std(matrix)

    np.testing.assert_allclose(result, np.nanstd(matrix, axis=0, ddof=1), rtol=1e-12)
