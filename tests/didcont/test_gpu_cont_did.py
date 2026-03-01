"""GPU backend tests for cont_did and NPIV modules."""

import numpy as np
import pytest

import moderndid
from moderndid.cupy.backend import set_backend, to_numpy
from moderndid.didcont.npiv.estimators import _ginv, npiv_est
from moderndid.didcont.npiv.gsl_bspline import gsl_bs
from moderndid.didcont.spline import BSpline
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
def test_ginv_identity():
    set_backend("cupy")
    A = cp.eye(4, dtype=np.float64)
    result = _ginv(A)
    np.testing.assert_allclose(to_numpy(result), np.eye(4), atol=1e-10)
    set_backend("numpy")


@requires_gpu
def test_ginv_cupy_vs_numpy():
    rng = np.random.default_rng(42)
    A_np = rng.standard_normal((5, 5))
    A_np = A_np.T @ A_np + 0.1 * np.eye(5)

    set_backend("numpy")
    result_cpu = _ginv(A_np)

    set_backend("cupy")
    result_gpu = _ginv(cp.asarray(A_np))

    np.testing.assert_allclose(to_numpy(result_gpu), result_cpu, atol=1e-8)
    set_backend("numpy")


@requires_gpu
def test_ginv_singular():
    set_backend("cupy")
    A = cp.zeros((3, 3), dtype=np.float64)
    result = _ginv(A)
    np.testing.assert_allclose(to_numpy(result), np.zeros((3, 3)), atol=1e-10)
    set_backend("numpy")


@requires_gpu
def test_npiv_est_cupy_vs_numpy():
    rng = np.random.default_rng(42)
    n = 200
    x = rng.uniform(0, 1, (n, 1))
    w = x.copy()
    y = np.sin(2 * np.pi * x[:, 0]) + rng.standard_normal(n) * 0.3

    set_backend("numpy")
    result_cpu = npiv_est(
        y=y,
        x=x,
        w=w,
        basis="tensor",
        j_x_degree=3,
        j_x_segments=4,
        k_w_degree=3,
        k_w_segments=4,
        knots="uniform",
    )

    set_backend("cupy")
    result_gpu = npiv_est(
        y=y,
        x=x,
        w=w,
        basis="tensor",
        j_x_degree=3,
        j_x_segments=4,
        k_w_degree=3,
        k_w_segments=4,
        knots="uniform",
    )

    np.testing.assert_allclose(to_numpy(result_gpu.h), to_numpy(result_cpu.h), atol=1e-6)
    np.testing.assert_allclose(to_numpy(result_gpu.deriv), to_numpy(result_cpu.deriv), atol=1e-6)
    np.testing.assert_allclose(to_numpy(result_gpu.beta), to_numpy(result_cpu.beta), atol=1e-6)
    np.testing.assert_allclose(to_numpy(result_gpu.asy_se), to_numpy(result_cpu.asy_se), atol=1e-6)
    set_backend("numpy")


@requires_gpu
def test_parametric_dose_cupy_vs_numpy():
    data = moderndid.simulate_cont_did_data(n=300, seed=42)

    result_cpu = moderndid.cont_did(
        data=data,
        yname="Y",
        tname="time_period",
        idname="id",
        gname="G",
        dname="D",
        target_parameter="level",
        aggregation="dose",
        degree=3,
        biters=50,
        random_state=42,
        backend="numpy",
    )

    result_gpu = moderndid.cont_did(
        data=data,
        yname="Y",
        tname="time_period",
        idname="id",
        gname="G",
        dname="D",
        target_parameter="level",
        aggregation="dose",
        degree=3,
        biters=50,
        random_state=42,
        backend="cupy",
    )

    np.testing.assert_allclose(result_gpu.att_d, result_cpu.att_d, atol=1e-4)
    np.testing.assert_allclose(result_gpu.acrt_d, result_cpu.acrt_d, atol=1e-4)
    np.testing.assert_allclose(result_gpu.overall_att, result_cpu.overall_att, atol=1e-4)
    np.testing.assert_allclose(result_gpu.overall_acrt, result_cpu.overall_acrt, atol=1e-4)


@requires_gpu
def test_cck_dose_cupy_vs_numpy():
    data = moderndid.simulate_cont_did_data(n=300, num_time_periods=2, seed=42)

    result_cpu = moderndid.cont_did(
        data=data,
        yname="Y",
        tname="time_period",
        idname="id",
        gname="G",
        dname="D",
        dose_est_method="cck",
        target_parameter="level",
        aggregation="dose",
        biters=50,
        random_state=42,
        backend="numpy",
    )

    result_gpu = moderndid.cont_did(
        data=data,
        yname="Y",
        tname="time_period",
        idname="id",
        gname="G",
        dname="D",
        dose_est_method="cck",
        target_parameter="level",
        aggregation="dose",
        biters=50,
        random_state=42,
        backend="cupy",
    )

    np.testing.assert_allclose(result_gpu.att_d, result_cpu.att_d, atol=1e-4)
    np.testing.assert_allclose(result_gpu.acrt_d, result_cpu.acrt_d, atol=1e-4)


@requires_gpu
def test_eventstudy_cupy_vs_numpy():
    data = moderndid.simulate_cont_did_data(n=300, seed=42)

    result_cpu = moderndid.cont_did(
        data=data,
        yname="Y",
        tname="time_period",
        idname="id",
        gname="G",
        dname="D",
        target_parameter="slope",
        aggregation="eventstudy",
        degree=3,
        biters=50,
        random_state=42,
        backend="numpy",
    )

    result_gpu = moderndid.cont_did(
        data=data,
        yname="Y",
        tname="time_period",
        idname="id",
        gname="G",
        dname="D",
        target_parameter="slope",
        aggregation="eventstudy",
        degree=3,
        biters=50,
        random_state=42,
        backend="cupy",
    )

    np.testing.assert_allclose(result_gpu.overall_att.overall_att, result_cpu.overall_att.overall_att, atol=1e-4)


@requires_gpu
def test_bspline_basis_cupy_vs_scipy():
    rng = np.random.default_rng(42)
    x = np.sort(rng.uniform(0, 1, 100))

    set_backend("numpy")
    bs_cpu = BSpline(x=x, degree=3, df=7)
    basis_cpu = bs_cpu.basis(complete_basis=True)
    deriv_cpu = bs_cpu.derivative(derivs=1, complete_basis=True)

    set_backend("cupy")
    bs_gpu = BSpline(x=x, degree=3, df=7)
    basis_gpu = to_numpy(bs_gpu.basis(complete_basis=True))
    deriv_gpu = to_numpy(bs_gpu.derivative(derivs=1, complete_basis=True))

    np.testing.assert_allclose(basis_gpu, basis_cpu, atol=1e-10)
    np.testing.assert_allclose(deriv_gpu, deriv_cpu, atol=1e-10)
    set_backend("numpy")


@requires_gpu
def test_bspline_basis_with_knots_cupy_vs_scipy():
    rng = np.random.default_rng(99)
    x = np.sort(rng.uniform(0, 10, 200))
    internal_knots = np.array([2.5, 5.0, 7.5])

    set_backend("numpy")
    bs_cpu = BSpline(x=x, degree=3, internal_knots=internal_knots)
    basis_cpu = bs_cpu.basis(complete_basis=False)
    deriv_cpu = bs_cpu.derivative(derivs=1, complete_basis=False)

    set_backend("cupy")
    bs_gpu = BSpline(x=x, degree=3, internal_knots=internal_knots)
    basis_gpu = to_numpy(bs_gpu.basis(complete_basis=False))
    deriv_gpu = to_numpy(bs_gpu.derivative(derivs=1, complete_basis=False))

    np.testing.assert_allclose(basis_gpu, basis_cpu, atol=1e-10)
    np.testing.assert_allclose(deriv_gpu, deriv_cpu, atol=1e-10)
    set_backend("numpy")


@requires_gpu
def test_gsl_bs_cupy_vs_scipy():
    rng = np.random.default_rng(42)
    x = np.sort(rng.uniform(0, 1, 150))

    set_backend("numpy")
    result_cpu = gsl_bs(x, degree=3, nbreak=6)
    basis_cpu = result_cpu.basis
    deriv_cpu = gsl_bs(x, degree=3, nbreak=6, deriv=1).basis

    set_backend("cupy")
    result_gpu = gsl_bs(x, degree=3, nbreak=6)
    basis_gpu = to_numpy(result_gpu.basis)
    deriv_gpu = to_numpy(gsl_bs(x, degree=3, nbreak=6, deriv=1).basis)

    np.testing.assert_allclose(basis_gpu, basis_cpu, atol=1e-10)
    np.testing.assert_allclose(deriv_gpu, deriv_cpu, atol=1e-10)
    set_backend("numpy")
