"""Tests for the cont_did and NPIV array routines on the NumPy and CuPy backends."""

import numpy as np
import pytest

from moderndid import BSpline, cont_did, gen_cont_did_data, npiv_est
from moderndid.cupy.backend import to_numpy, use_backend
from moderndid.npiv import gsl_bs
from moderndid.npiv.estimators import _ginv


def test_ginv_of_identity_is_identity(xp):
    matrix = xp.asarray(np.eye(4))

    result = _ginv(matrix)

    assert isinstance(result, xp.ndarray)
    np.testing.assert_allclose(to_numpy(result), np.eye(4), rtol=1e-7, atol=1e-10)


def test_ginv_of_zero_matrix_is_zero(xp):
    matrix = xp.asarray(np.zeros((3, 3)))

    result = _ginv(matrix)

    assert isinstance(result, xp.ndarray)
    np.testing.assert_array_equal(to_numpy(result), np.zeros((3, 3)))


def test_ginv_of_well_conditioned_matrix_matches_pinv(xp):
    design = np.random.default_rng(42).standard_normal((5, 5))
    matrix = design.T @ design + 0.1 * np.eye(5)

    result = _ginv(xp.asarray(matrix))

    assert isinstance(result, xp.ndarray)
    np.testing.assert_allclose(to_numpy(result), np.linalg.pinv(matrix), rtol=1e-10)


@pytest.mark.gpu
def test_npiv_est_on_gpu_matches_cpu(cp):
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, (200, 1))
    y = np.sin(2 * np.pi * x[:, 0]) + 0.3 * rng.standard_normal(200)
    spec = {
        "y": y,
        "x": x,
        "w": x.copy(),
        "basis": "tensor",
        "j_x_degree": 3,
        "j_x_segments": 4,
        "k_w_degree": 3,
        "k_w_segments": 4,
        "knots": "uniform",
    }

    cpu = npiv_est(**spec)
    with use_backend("cupy"):
        gpu = npiv_est(**spec)

    assert isinstance(gpu.h, cp.ndarray)
    for field in ("h", "deriv", "beta", "asy_se"):
        assert np.all(np.isfinite(getattr(cpu, field)))
        np.testing.assert_allclose(to_numpy(getattr(gpu, field)), getattr(cpu, field), rtol=1e-7, atol=1e-6)


@pytest.mark.gpu
def test_cont_did_parametric_dose_on_gpu_matches_cpu():
    spec = {
        "data": gen_cont_did_data(n=300, seed=42),
        "yname": "Y",
        "tname": "time_period",
        "idname": "id",
        "gname": "G",
        "dname": "D",
        "target_parameter": "level",
        "aggregation": "dose",
        "degree": 3,
        "biters": 50,
        "random_state": 42,
    }

    cpu = cont_did(**spec, backend="numpy")
    gpu = cont_did(**spec, backend="cupy")

    for field in ("att_d", "acrt_d", "overall_att", "overall_acrt"):
        assert np.all(np.isfinite(getattr(cpu, field)))
        np.testing.assert_allclose(getattr(gpu, field), getattr(cpu, field), rtol=1e-7, atol=1e-4)


@pytest.mark.gpu
def test_cont_did_cck_dose_on_gpu_matches_cpu():
    spec = {
        "data": gen_cont_did_data(n=300, num_time_periods=2, seed=42),
        "yname": "Y",
        "tname": "time_period",
        "idname": "id",
        "gname": "G",
        "dname": "D",
        "dose_est_method": "cck",
        "target_parameter": "level",
        "aggregation": "dose",
        "biters": 50,
        "random_state": 42,
    }

    cpu = cont_did(**spec, backend="numpy")
    gpu = cont_did(**spec, backend="cupy")

    for field in ("att_d", "acrt_d"):
        assert np.all(np.isfinite(getattr(cpu, field)))
        np.testing.assert_allclose(getattr(gpu, field), getattr(cpu, field), rtol=1e-7, atol=1e-4)


@pytest.mark.gpu
def test_cont_did_event_study_on_gpu_matches_cpu():
    spec = {
        "data": gen_cont_did_data(n=300, seed=42),
        "yname": "Y",
        "tname": "time_period",
        "idname": "id",
        "gname": "G",
        "dname": "D",
        "target_parameter": "slope",
        "aggregation": "eventstudy",
        "degree": 3,
        "biters": 50,
        "random_state": 42,
    }

    cpu = cont_did(**spec, backend="numpy")
    gpu = cont_did(**spec, backend="cupy")

    assert np.isfinite(cpu.overall_att.overall_att)
    np.testing.assert_allclose(gpu.overall_att.overall_att, cpu.overall_att.overall_att, rtol=1e-7, atol=1e-4)


@pytest.mark.gpu
def test_bspline_with_df_on_gpu_matches_cpu(cp):
    x = np.sort(np.random.default_rng(42).uniform(0, 1, 100))

    cpu_spline = BSpline(x=x, degree=3, df=7)
    cpu_basis = cpu_spline.basis(complete_basis=True)
    cpu_deriv = cpu_spline.derivative(derivs=1, complete_basis=True)
    with use_backend("cupy"):
        gpu_spline = BSpline(x=x, degree=3, df=7)
        gpu_basis = gpu_spline.basis(complete_basis=True)
        gpu_deriv = gpu_spline.derivative(derivs=1, complete_basis=True)

    assert isinstance(gpu_basis, cp.ndarray)
    assert isinstance(gpu_deriv, cp.ndarray)
    np.testing.assert_allclose(to_numpy(gpu_basis), cpu_basis, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(to_numpy(gpu_deriv), cpu_deriv, rtol=1e-7, atol=1e-10)


@pytest.mark.gpu
def test_bspline_with_internal_knots_on_gpu_matches_cpu(cp):
    x = np.sort(np.random.default_rng(99).uniform(0, 10, 200))
    internal_knots = np.array([2.5, 5.0, 7.5])

    cpu_spline = BSpline(x=x, degree=3, internal_knots=internal_knots)
    cpu_basis = cpu_spline.basis(complete_basis=False)
    cpu_deriv = cpu_spline.derivative(derivs=1, complete_basis=False)
    with use_backend("cupy"):
        gpu_spline = BSpline(x=x, degree=3, internal_knots=internal_knots)
        gpu_basis = gpu_spline.basis(complete_basis=False)
        gpu_deriv = gpu_spline.derivative(derivs=1, complete_basis=False)

    assert isinstance(gpu_basis, cp.ndarray)
    assert isinstance(gpu_deriv, cp.ndarray)
    np.testing.assert_allclose(to_numpy(gpu_basis), cpu_basis, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(to_numpy(gpu_deriv), cpu_deriv, rtol=1e-7, atol=1e-10)


@pytest.mark.gpu
def test_gsl_bs_on_gpu_matches_cpu(cp):
    x = np.sort(np.random.default_rng(42).uniform(0, 1, 150))

    cpu_basis = gsl_bs(x, degree=3, nbreak=6).basis
    cpu_deriv = gsl_bs(x, degree=3, nbreak=6, deriv=1).basis
    with use_backend("cupy"):
        gpu_basis = gsl_bs(x, degree=3, nbreak=6).basis
        gpu_deriv = gsl_bs(x, degree=3, nbreak=6, deriv=1).basis

    assert isinstance(gpu_basis, cp.ndarray)
    assert isinstance(gpu_deriv, cp.ndarray)
    np.testing.assert_allclose(to_numpy(gpu_basis), cpu_basis, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(to_numpy(gpu_deriv), cpu_deriv, rtol=1e-7, atol=1e-10)
