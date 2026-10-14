"""Tests for the two-period estimators on the strict CuPy stand-in."""

import functools
import importlib
import warnings

import numpy as np
import pytest
import statsmodels.api as sm

from moderndid import (
    drdid_panel,
    drdid_rc,
    drdid_trad_rc,
    ipw_did_panel,
    ipw_did_rc,
    reg_did_rc,
    std_ipw_did_panel,
    std_ipw_did_rc,
)
from moderndid.cupy.backend import use_backend
from moderndid.cupy.regression import cupy_logistic_irls


@pytest.mark.parametrize("estimator", [drdid_rc, drdid_trad_rc, std_ipw_did_rc, reg_did_rc])
def test_repeated_cross_section_estimators_of_device_arrays_match_cpu(strict_cupy, two_period_specs, estimator):
    spec = two_period_specs["repeated_cross_sections"]
    device_spec = {name: strict_cupy.asarray(value) for name, value in spec.items()}

    expected = estimator(**spec, influence_func=True)
    with use_backend("cupy"):
        result = estimator(**device_spec, influence_func=True)

    assert isinstance(result.att_inf_func, np.ndarray)
    np.testing.assert_allclose(result.att, expected.att, rtol=1e-8)
    np.testing.assert_allclose(result.se, expected.se, rtol=1e-8)
    np.testing.assert_allclose(result.att_inf_func, expected.att_inf_func, rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize(
    "estimator,layout",
    [
        (drdid_panel, "panel"),
        (ipw_did_panel, "panel"),
        (std_ipw_did_panel, "panel"),
        (drdid_rc, "repeated_cross_sections"),
        (drdid_trad_rc, "repeated_cross_sections"),
        (ipw_did_rc, "repeated_cross_sections"),
        (std_ipw_did_rc, "repeated_cross_sections"),
    ],
)
def test_logit_stopped_after_one_iteration_on_gpu_warns_like_cpu(
    strict_cupy, monkeypatch, two_period_specs, estimator, layout
):
    spec = two_period_specs[layout]
    monkeypatch.setattr(sm.GLM, "fit", functools.partialmethod(sm.GLM.fit, maxiter=1))
    monkeypatch.setattr(
        importlib.import_module(estimator.__module__),
        "cupy_logistic_irls",
        functools.partial(cupy_logistic_irls, max_iter=1),
    )

    with warnings.catch_warnings(record=True) as cpu_warnings:
        warnings.simplefilter("always")
        estimator(**spec)
    with use_backend("cupy"), warnings.catch_warnings(record=True) as gpu_warnings:
        warnings.simplefilter("always")
        estimator(**spec)

    cpu_messages = {str(w.message) for w in cpu_warnings if "did not converge" in str(w.message)}
    gpu_messages = {str(w.message) for w in gpu_warnings if "did not converge" in str(w.message)}
    assert cpu_messages
    assert gpu_messages == cpu_messages
