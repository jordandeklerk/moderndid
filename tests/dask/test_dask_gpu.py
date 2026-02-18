from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from moderndid.cupy.backend import HAS_CUPY, _array_module, _validate_backend_name, to_numpy
from moderndid.dask._gpu import _maybe_to_gpu, _to_gpu_partition
from moderndid.dask._gram import partition_gram
from moderndid.dask._regression import _irls_local_stats_with_y

_has_gpu = False
if HAS_CUPY:
    try:
        import cupy as cp

        cp.array([1.0])
        _has_gpu = True
    except (RuntimeError, AttributeError):
        pass

requires_gpu = pytest.mark.skipif(not _has_gpu, reason="CuPy with working GPU required")


@pytest.mark.parametrize(
    "arrays,expected",
    [
        ((), np),
        ((np.array([1.0]),), np),
        ((np.ones(5), np.zeros(3)), np),
    ],
)
def test_array_module_numpy(arrays, expected):
    assert _array_module(*arrays) is expected


@requires_gpu
def test_array_module_cupy_single():
    import cupy as cp

    assert _array_module(cp.ones(5)) is cp


@requires_gpu
def test_array_module_cupy_mixed():
    import cupy as cp

    assert _array_module(np.ones(5), cp.ones(3)) is cp


def test_to_gpu_partition_none():
    assert _to_gpu_partition(None) is None


@requires_gpu
def test_to_gpu_partition_converts_floats():
    import cupy as cp

    part = {
        "X": np.ones((10, 2)),
        "y1": np.zeros(10),
        "ids": np.arange(10),
        "n": 10,
    }
    result = _to_gpu_partition(part)
    assert isinstance(result["X"], cp.ndarray)
    assert isinstance(result["y1"], cp.ndarray)
    assert isinstance(result["ids"], np.ndarray)
    assert result["n"] == 10


@requires_gpu
def test_to_gpu_partition_preserves_values():
    import cupy as cp

    arr = np.array([1.0, 2.0, 3.0])
    result = _to_gpu_partition({"a": arr})
    np.testing.assert_array_equal(cp.asnumpy(result["a"]), arr)


@pytest.mark.parametrize("use_gpu", [True, False])
def test_maybe_to_gpu(use_gpu):
    client = MagicMock()
    sentinel = MagicMock()
    client.submit.return_value = sentinel
    futures = [MagicMock(), MagicMock(), MagicMock()]

    result = _maybe_to_gpu(client, futures, use_gpu=use_gpu)

    if use_gpu:
        assert len(result) == 3
        assert all(r is sentinel for r in result)
        assert client.submit.call_count == 3
        for call in client.submit.call_args_list:
            assert call[0][0] is _to_gpu_partition
    else:
        assert result is futures
        client.submit.assert_not_called()


def test_partition_gram_numpy_roundtrip():
    rng = np.random.default_rng(42)
    X = np.column_stack([np.ones(20), rng.standard_normal(20)])
    W = np.ones(20)
    y = rng.standard_normal(20)

    XtWX, XtWy, n = partition_gram(X, W, y)
    assert isinstance(XtWX, np.ndarray)
    assert isinstance(XtWy, np.ndarray)
    assert n == 20
    np.testing.assert_allclose(XtWX, X.T @ np.diag(W) @ X)
    np.testing.assert_allclose(XtWy, X.T @ np.diag(W) @ y)


def test_irls_local_stats_numpy_roundtrip():
    rng = np.random.default_rng(42)
    n, k = 50, 2
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    weights = np.ones(n)
    y = rng.integers(0, 2, size=n).astype(np.float64)
    beta = np.zeros(k)

    XtWX, XtWz, count = _irls_local_stats_with_y(X, weights, y, beta)
    assert isinstance(XtWX, np.ndarray)
    assert isinstance(XtWz, np.ndarray)
    assert XtWX.shape == (k, k)
    assert XtWz.shape == (k,)
    assert count == n


def test_to_numpy_passthrough():
    arr = np.array([1.0, 2.0, 3.0])
    assert to_numpy(arr) is arr


@requires_gpu
def test_to_numpy_cupy_converts():
    import cupy as cp

    gpu_arr = cp.array([1.0, 2.0, 3.0])
    result = to_numpy(gpu_arr)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


def test_validate_backend_invalid():
    with pytest.raises(ValueError, match="Unknown backend"):
        _validate_backend_name("tensorflow")


def test_validate_backend_numpy():
    _validate_backend_name("numpy")


@requires_gpu
def test_validate_backend_cupy_with_gpu():
    _validate_backend_name("cupy")


@pytest.mark.skipif(HAS_CUPY and _has_gpu, reason="Only runs when CuPy has no GPU")
def test_validate_backend_cupy_without_gpu():
    with pytest.raises((ImportError, RuntimeError)):
        _validate_backend_name("cupy")
