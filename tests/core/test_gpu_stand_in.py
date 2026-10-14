"""Tests for the strict CuPy stand-in that the GPU route tests run on."""

import importlib
import sys

import numpy as np
import polars as pl
import pytest
import scipy.interpolate
import scipy.linalg
import statsmodels.api as sm

from moderndid.cupy import backend as cupy_backend
from moderndid.cupy.backend import get_backend, to_device, to_numpy, use_backend
from moderndid.npiv import gsl_bspline
from tests.strict_cupy import BSpline, cupyx


@pytest.mark.parametrize(
    "consume,error,message",
    [
        (np.asarray, TypeError, "Implicit conversion to a NumPy array is not allowed"),
        (lambda a: scipy.linalg.solve(a, a), TypeError, "Implicit conversion to a NumPy array is not allowed"),
        (lambda a: sm.OLS(a[:, 0], a).fit(), ValueError, "unrecognized data structures"),
        (lambda a: pl.Series(a.ravel()), TypeError, "unsupported type"),
    ],
    ids=["numpy", "scipy", "statsmodels", "polars"],
)
def test_stand_in_refuses_implicit_conversion_to_numpy(strict_cupy, consume, error, message):
    device = strict_cupy.eye(3)

    with pytest.raises(error, match=message):
        consume(device)


@pytest.mark.parametrize(
    "combine",
    [
        lambda device, host: device + host,
        lambda device, host: host * device,
        lambda device, host: device - host[0:1].reshape(()),
        lambda device, host: device[None, :] @ host[:, None],
        lambda device, host: np.add(device, host),
        lambda device, host: np.where(device > 0, device, host),
        lambda device, host: np.concatenate([device, host]),
    ],
    ids=["add", "reflected_multiply", "zero_dimensional_host", "matmul", "ufunc", "where", "concatenate"],
)
def test_stand_in_refuses_host_array_operands(strict_cupy, combine):
    device = strict_cupy.arange(3.0)
    host = np.arange(3.0)

    with pytest.raises(TypeError, match="numpy.ndarray|Only cupy arrays"):
        combine(device, host)


def test_stand_in_computes_with_scalars_and_returns_zero_dimensional_reductions(strict_cupy):
    device = strict_cupy.arange(4.0)

    total = (2 * device + np.float64(1.0)).sum()

    assert isinstance(total, strict_cupy.ndarray)
    assert total.ndim == 0
    assert float(total) == 16.0


def test_stand_in_solves_a_singular_system_without_raising(strict_cupy):
    host = np.array([[1.0, 2.0], [2.0, 4.0]])

    with pytest.raises(np.linalg.LinAlgError, match="Singular matrix"):
        np.linalg.solve(host, np.ones(2))
    solution = strict_cupy.linalg.solve(strict_cupy.asarray(host), strict_cupy.ones(2))
    dispatched = np.linalg.solve(strict_cupy.asarray(host), strict_cupy.ones(2))
    inverse = strict_cupy.linalg.inv(strict_cupy.asarray(host))

    assert not np.all(np.isfinite(solution.get()))
    assert not np.all(np.isfinite(dispatched.get()))
    assert not np.all(np.isfinite(inverse.get()))


def test_stand_in_raises_on_a_singular_system_inside_linalg_raise_mode(strict_cupy):
    singular = strict_cupy.asarray(np.array([[1.0, 2.0], [2.0, 4.0]]))

    with cupyx.errstate(linalg="raise"), pytest.raises(np.linalg.LinAlgError, match="getrf"):
        strict_cupy.linalg.solve(singular, strict_cupy.ones(2))


def test_stand_in_linalg_refuses_host_arrays(strict_cupy):
    host = np.eye(2)

    with pytest.raises(np.linalg.LinAlgError, match="cupy.linalg only supports cupy.ndarray"):
        strict_cupy.linalg.solve(host, np.ones(2))


@pytest.mark.parametrize("method", ["choice", "permutation", "shuffle", "normal"])
def test_stand_in_generator_lacks_the_methods_cupy_lacks(strict_cupy, method):
    rng = strict_cupy.random.default_rng(0)

    assert not hasattr(rng, method)


def test_stand_in_binomial_takes_only_a_python_float_probability(strict_cupy):
    rng = strict_cupy.random.default_rng(0)

    with pytest.raises(TypeError, match="p is required"):
        rng.binomial(1, np.float64(0.5), size=3)


def test_stand_in_draws_differ_from_numpy_for_the_same_seed(strict_cupy):
    device_draws = strict_cupy.random.default_rng(42).standard_normal(5)

    numpy_draws = np.random.default_rng(42).standard_normal(5)

    assert not np.allclose(device_draws.get(), numpy_draws)


@pytest.mark.parametrize("name", ["nanquantile", "nanpercentile", "insert", "errstate"])
def test_stand_in_lacks_the_functions_cupy_lacks(strict_cupy, name):
    assert not hasattr(strict_cupy, name)


def test_stand_in_refuses_numpy_functions_cupy_lacks(strict_cupy):
    device = strict_cupy.arange(3.0)

    with pytest.raises(TypeError, match="no implementation found"):
        np.nanquantile(device, 0.5)


def test_stand_in_basic_assignment_refuses_host_arrays_and_advanced_assignment_takes_them(strict_cupy):
    device = strict_cupy.zeros(3)

    with pytest.raises(ValueError, match="non-scalar numpy.ndarray cannot be used for fill"):
        device[:2] = np.ones(2)
    device[np.array([0, 2])] = np.array([4.0, 5.0])

    np.testing.assert_array_equal(device.get(), [4.0, 0.0, 5.0])


def test_stand_in_spline_refuses_host_knots_and_matches_scipy(strict_cupy):
    knots = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
    x = np.array([0.1, 0.4, 0.8])

    with pytest.raises(TypeError, match="incorrect type"):
        BSpline(knots, np.ones(5), 3)
    design = BSpline.design_matrix(x, strict_cupy.asarray(knots), 3).toarray()

    np.testing.assert_allclose(design.get(), scipy.interpolate.BSpline.design_matrix(x, knots, 3).toarray())


def test_strict_cupy_fixture_patches_moderndid_import_sites(strict_cupy):
    host = np.arange(3.0)

    with use_backend("cupy"):
        device = to_device(host)
        backend = get_backend()

    assert backend is strict_cupy
    assert isinstance(device, strict_cupy.ndarray)
    np.testing.assert_array_equal(to_numpy(device), host)
    assert cupy_backend.HAS_CUPY
    assert gsl_bspline.CupyBSpline is BSpline
    assert importlib.import_module("cupyx.scipy.sparse") is sys.modules["cupyx.scipy.sparse"]
    assert sys.modules["cupy"] is strict_cupy
