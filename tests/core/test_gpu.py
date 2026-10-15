"""Tests for the estimators on a CUDA GPU against the same calls on the CPU."""

import re

import numpy as np
import pytest

from moderndid import (
    att_gt,
    cont_did,
    ddd,
    drdid,
    ipwdid,
    mboot,
    npiv,
    ordid,
    wboot_dr_tr_panel,
    wboot_drdid_imp_panel,
    wboot_drdid_ipt_rc1,
    wboot_drdid_ipt_rc2,
    wboot_drdid_rc1,
    wboot_drdid_rc2,
    wboot_ipw_panel,
    wboot_ipw_rc,
    wboot_reg_panel,
    wboot_reg_rc,
    wboot_std_ipw_panel,
    wboot_std_ipw_rc,
    wboot_twfe_panel,
    wboot_twfe_rc,
)
from moderndid.cupy.backend import get_backend, use_backend

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
@pytest.mark.parametrize(
    "data_fixture,layout",
    [
        ("mpdta", {}),
        ("mpdta", {"panel": False}),
        ("mpdta_unbalanced", {"allow_unbalanced_panel": True}),
    ],
    ids=["balanced_panel", "repeated_cross_sections", "unbalanced_panel"],
)
def test_att_gt_on_gpu_matches_cpu(request, data_fixture, layout, est_method):
    spec = {
        "data": request.getfixturevalue(data_fixture),
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~ lpop",
        "est_method": est_method,
        "boot": False,
        "cband": False,
        **layout,
    }

    cpu = att_gt(**spec)
    gpu = att_gt(**spec, backend="cupy")

    assert get_backend() is np
    assert np.all(np.isfinite(cpu.se_gt))
    np.testing.assert_allclose(gpu.att_gt, cpu.att_gt, rtol=1e-8)
    np.testing.assert_allclose(gpu.se_gt, cpu.se_gt, rtol=1e-6)


def test_att_gt_repeated_cross_sections_multiplier_bootstrap_on_gpu_matches_cpu_loosely(mpdta):
    spec = {
        "data": mpdta,
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~ lpop",
        "est_method": "dr",
        "panel": False,
        "boot": True,
        "biters": 999,
        "random_state": 7,
    }

    cpu = att_gt(**spec)
    gpu = att_gt(**spec, backend="cupy")

    assert np.all(np.isfinite(cpu.se_gt))
    np.testing.assert_allclose(gpu.att_gt, cpu.att_gt, rtol=1e-8)
    np.testing.assert_allclose(gpu.se_gt, cpu.se_gt, rtol=0.3)


@pytest.mark.parametrize("est_method", ["imp", "trad"])
@pytest.mark.parametrize("panel", [True, False], ids=["panel", "repeated_cross_sections"])
def test_drdid_on_gpu_matches_cpu(drdid_panel_data, panel, est_method):
    spec = {
        "data": drdid_panel_data,
        "yname": "y",
        "tname": "year",
        "idname": "id",
        "treatname": "treat",
        "xformla": "~ x",
        "panel": panel,
        "est_method": est_method,
    }

    cpu = drdid(**spec)
    with use_backend("cupy"):
        gpu = drdid(**spec)

    assert np.isfinite(cpu.se)
    np.testing.assert_allclose(gpu.att, cpu.att, rtol=1e-8)
    np.testing.assert_allclose(gpu.se, cpu.se, rtol=1e-6)


@pytest.mark.parametrize("transfer", ["asarray", "asnumpy"], ids=["device_inputs", "host_inputs"])
@pytest.mark.parametrize(
    "bootstrap,layout",
    [
        pytest.param(wboot_drdid_imp_panel, "panel", id="drdid_imp_panel"),
        pytest.param(wboot_dr_tr_panel, "panel", id="dr_tr_panel"),
        pytest.param(wboot_ipw_panel, "panel", id="ipw_panel"),
        pytest.param(wboot_std_ipw_panel, "panel", id="std_ipw_panel"),
        pytest.param(wboot_reg_panel, "panel", id="reg_panel"),
        pytest.param(wboot_twfe_panel, "stacked_panel", id="twfe_panel"),
        pytest.param(wboot_drdid_rc1, "repeated_cross_sections", id="drdid_rc1"),
        pytest.param(wboot_drdid_rc2, "repeated_cross_sections", id="drdid_rc2"),
        pytest.param(wboot_drdid_ipt_rc1, "repeated_cross_sections", id="drdid_ipt_rc1"),
        pytest.param(wboot_drdid_ipt_rc2, "repeated_cross_sections", id="drdid_ipt_rc2"),
        pytest.param(wboot_ipw_rc, "repeated_cross_sections", id="ipw_rc"),
        pytest.param(wboot_std_ipw_rc, "repeated_cross_sections", id="std_ipw_rc"),
        pytest.param(wboot_reg_rc, "repeated_cross_sections", id="reg_rc"),
        pytest.param(wboot_twfe_rc, "repeated_cross_sections", id="twfe_rc"),
    ],
)
def test_weighted_bootstrap_on_gpu_matches_cpu_exactly(cp, weighted_bootstrap_inputs, bootstrap, layout, transfer):
    inputs = weighted_bootstrap_inputs[layout]
    gpu_inputs = {name: getattr(cp, transfer)(value) for name, value in inputs.items()}

    cpu = bootstrap(**inputs, biters=20, random_state=3)
    with use_backend("cupy"):
        gpu = bootstrap(**gpu_inputs, biters=20, random_state=3)

    assert np.all(np.isfinite(cpu))
    assert np.ptp(cpu) > 0
    np.testing.assert_array_equal(cp.asnumpy(gpu), cpu)


@pytest.mark.parametrize("estimator", [drdid, ipwdid, ordid], ids=["drdid", "ipwdid", "ordid"])
@pytest.mark.parametrize("panel", [True, False], ids=["panel", "repeated_cross_sections"])
def test_two_period_weighted_bootstrap_on_gpu_returns_finite_standard_error(drdid_panel_data, estimator, panel):
    spec = {
        "data": drdid_panel_data,
        "yname": "y",
        "tname": "year",
        "idname": "id",
        "treatname": "treat",
        "xformla": "~ x",
        "panel": panel,
        "boot": True,
        "boot_type": "weighted",
        "biters": 20,
    }

    with use_backend("cupy"):
        result = estimator(**spec)

    assert np.isfinite(result.se)
    assert result.se > 0


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
def test_ddd_on_gpu_matches_cpu(ddd_two_period_data, est_method):
    spec = {
        "data": ddd_two_period_data,
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "state",
        "pname": "partition",
        "xformla": "~ cov1 + cov2 + cov3 + cov4",
        "est_method": est_method,
    }

    cpu = ddd(**spec)
    gpu = ddd(**spec, backend="cupy")

    assert get_backend() is np
    assert np.isfinite(cpu.se)
    np.testing.assert_allclose(gpu.att, cpu.att, rtol=1e-8)
    np.testing.assert_allclose(gpu.se, cpu.se, rtol=1e-6)


@pytest.mark.parametrize("panel", [True, False], ids=["panel", "repeated_cross_sections"])
def test_ddd_weighted_bootstrap_on_gpu_matches_cpu(cp, ddd_two_period_data, panel):
    spec = {
        "data": ddd_two_period_data,
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "state",
        "pname": "partition",
        "xformla": "~ cov1 + cov2 + cov3 + cov4",
        "panel": panel,
        "boot": True,
        "boot_type": "weighted",
        "biters": 49,
        "random_state": 5,
    }

    cpu = ddd(**spec)
    gpu = ddd(**spec, backend="cupy")

    assert np.ptp(cpu.boots) > 0
    np.testing.assert_allclose(gpu.att, cpu.att, rtol=1e-8)
    np.testing.assert_allclose(cp.asnumpy(gpu.boots), cpu.boots, rtol=1e-10)
    np.testing.assert_allclose(gpu.se, cpu.se, rtol=1e-6)


@pytest.mark.parametrize("panel", [True, False], ids=["panel", "repeated_cross_sections"])
def test_ddd_multiplier_bootstrap_on_gpu_matches_cpu_loosely(ddd_two_period_data, panel):
    spec = {
        "data": ddd_two_period_data,
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "state",
        "pname": "partition",
        "xformla": "~ cov1 + cov2 + cov3 + cov4",
        "panel": panel,
        "boot": True,
        "boot_type": "multiplier",
        "biters": 999,
        "random_state": 5,
    }

    cpu = ddd(**spec)
    gpu = ddd(**spec, backend="cupy")

    assert np.isfinite(cpu.se)
    np.testing.assert_allclose(gpu.att, cpu.att, rtol=1e-8)
    np.testing.assert_allclose(gpu.se, cpu.se, rtol=0.3)


def test_mboot_on_gpu_matches_cpu_loosely():
    inf_func = np.random.default_rng(42).standard_normal((200, 3))

    cpu = mboot(inf_func, n_units=200, biters=999, random_state=42)
    with use_backend("cupy"):
        gpu = mboot(inf_func, n_units=200, biters=999, random_state=42)

    assert gpu["bres"].shape == cpu["bres"].shape
    assert np.all(np.isfinite(cpu["se"]))
    np.testing.assert_allclose(gpu["se"], cpu["se"], rtol=0.3)
    np.testing.assert_allclose(gpu["bres"].mean(axis=0), cpu["bres"].mean(axis=0), atol=0.3)


def test_clustered_mboot_on_gpu_matches_cpu_loosely():
    inf_func = np.random.default_rng(42).standard_normal(120)
    cluster = np.repeat(np.arange(20), 6)

    cpu = mboot(inf_func, n_units=120, biters=999, cluster=cluster, random_state=42)
    with use_backend("cupy"):
        gpu = mboot(inf_func, n_units=120, biters=999, cluster=cluster, random_state=42)

    assert gpu["bres"].shape == cpu["bres"].shape
    assert np.all(np.isfinite(cpu["se"]))
    np.testing.assert_allclose(gpu["se"], cpu["se"], rtol=0.3)
    np.testing.assert_allclose(gpu["bres"].mean(axis=0), cpu["bres"].mean(axis=0), atol=0.9)


@pytest.mark.parametrize("instrument", ["w", "x"], ids=["instrument_differs_from_x", "instrument_is_x"])
def test_npiv_with_selected_dimension_on_gpu_matches_cpu(cp, iv_data, instrument):
    spec = {"y": iv_data["y"], "x": iv_data["x"], "w": iv_data[instrument], "j_x_segments": None, "seed": 3}

    cpu = npiv(**spec)
    with use_backend("cupy"):
        gpu = npiv(**spec)

    assert np.isfinite(cpu.cv)
    assert gpu.j_x_segments == cpu.j_x_segments
    assert gpu.k_w_segments == cpu.k_w_segments
    np.testing.assert_allclose(cp.asnumpy(gpu.h), cpu.h, rtol=1e-8)
    np.testing.assert_allclose(cp.asnumpy(gpu.deriv), cpu.deriv, rtol=1e-8)
    np.testing.assert_allclose(cp.asnumpy(gpu.asy_se), cpu.asy_se, rtol=1e-6)
    np.testing.assert_allclose(cp.asnumpy(gpu.deriv_asy_se), cpu.deriv_asy_se, rtol=1e-6)
    np.testing.assert_allclose(cp.asnumpy(gpu.cv), cpu.cv, rtol=1e-6)
    np.testing.assert_allclose(cp.asnumpy(gpu.cv_deriv), cpu.cv_deriv, rtol=1e-6)


def test_cont_did_wald_pretest_on_gpu_matches_cpu(cp, cont_did_four_periods):
    spec = {
        "data": cont_did_four_periods,
        "yname": "Y",
        "tname": "time_period",
        "idname": "id",
        "gname": "G",
        "dname": "D",
        "dose_est_method": "parametric",
        "aggregation": "eventstudy",
        "biters": 20,
        "random_state": 1,
    }

    cpu = cont_did(**spec)
    gpu = cont_did(**spec, backend="cupy")

    assert cpu.att_gt.wald_stat is not None
    np.testing.assert_allclose(cp.asnumpy(gpu.att_gt.att), cpu.att_gt.att, rtol=1e-8)
    assert gpu.att_gt.wald_stat == pytest.approx(cpu.att_gt.wald_stat, rel=1e-6)


@pytest.mark.parametrize("panel", [True, False], ids=["panel", "repeated_cross_sections"])
def test_drdid_collinear_covariate_on_gpu_raises_like_cpu(collinear_covariate_panel, panel):
    spec = {
        "data": collinear_covariate_panel,
        "yname": "y",
        "tname": "year",
        "idname": "id",
        "treatname": "treat",
        "xformla": "~ x + x2",
        "est_method": "trad",
        "panel": panel,
    }

    with pytest.raises(ValueError, match="regression design matrix of the comparison group") as cpu_error:
        drdid(**spec)

    with use_backend("cupy"), pytest.raises(ValueError, match=re.escape(str(cpu_error.value))):
        drdid(**spec)


@pytest.mark.parametrize("panel", [True, False], ids=["panel", "repeated_cross_sections"])
def test_ipwdid_separation_warnings_on_gpu_match_cpu(recorded_warnings, separated_treatment_panel, panel):
    spec = {
        "data": separated_treatment_panel,
        "yname": "y",
        "tname": "year",
        "idname": "id",
        "treatname": "treat",
        "xformla": "~ x + x2",
        "panel": panel,
    }

    with recorded_warnings() as cpu_warnings:
        ipwdid(**spec)
    with use_backend("cupy"), recorded_warnings() as gpu_warnings:
        ipwdid(**spec)

    assert cpu_warnings
    assert gpu_warnings == cpu_warnings


def test_npiv_bands_without_variance_on_gpu_match_cpu(cp, iv_data, recorded_warnings):
    spec = {"y": np.zeros_like(iv_data["y"]), "x": iv_data["x"], "w": iv_data["w"], "j_x_segments": 4, "seed": 3}

    with recorded_warnings() as cpu_warnings:
        cpu = npiv(**spec)
    with use_backend("cupy"), recorded_warnings() as gpu_warnings:
        gpu = npiv(**spec)

    assert ("UserWarning", "Using unstudentized confidence bands for function estimates") in cpu_warnings
    assert ("UserWarning", "Using unstudentized confidence bands for derivative estimates") in cpu_warnings
    assert gpu_warnings == cpu_warnings
    np.testing.assert_array_equal(cp.asnumpy(gpu.h_lower), cpu.h_lower)
    np.testing.assert_array_equal(cp.asnumpy(gpu.h_upper), cpu.h_upper)
    np.testing.assert_array_equal(cp.asnumpy(gpu.h_lower_deriv), cpu.h_lower_deriv)
    np.testing.assert_array_equal(cp.asnumpy(gpu.h_upper_deriv), cpu.h_upper_deriv)
