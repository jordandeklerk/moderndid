"""Tests for IPW, AIPW, and IPW-MSM weight estimation."""

import numpy as np
import pytest

from moderndid.dev.diddynamic.estimation.weights_ipw import (
    IPWResult,
    _estimate_joint_propensity,
    _match_mask,
    compute_ipw_estimator,
)


@pytest.mark.parametrize("method", ["ipw", "aipw", "ipw_msm"])
def test_returns_ipw_result(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_ipw_estimator(3, outcome, treatment, covariates, ds, method=method)
    assert isinstance(result, IPWResult)


@pytest.mark.parametrize("method", ["ipw", "aipw", "ipw_msm"])
def test_result_has_two_fields(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_ipw_estimator(3, outcome, treatment, covariates, ds, method=method)
    assert len(result) == 2


@pytest.mark.parametrize("method", ["ipw", "aipw", "ipw_msm"])
def test_mu_hat_is_finite(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_ipw_estimator(3, outcome, treatment, covariates, ds, method=method)
    assert np.isfinite(result.mu_hat)


@pytest.mark.parametrize("method", ["ipw", "aipw", "ipw_msm"])
def test_variance_non_negative(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_ipw_estimator(3, outcome, treatment, covariates, ds, method=method)
    assert result.variance >= 0.0


def test_invalid_method_raises(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    with pytest.raises(ValueError, match="Unknown method"):
        compute_ipw_estimator(3, outcome, treatment, covariates, ds, method="invalid")


def test_joint_propensity_shape(estimation_panel):
    _, treatment, covariates, ds = estimation_panel
    ps = _estimate_joint_propensity(3, treatment, covariates, ds, (0.01, 0.99))
    assert ps.shape == (60,)


def test_joint_propensity_clipped(estimation_panel):
    _, treatment, covariates, ds = estimation_panel
    ps = _estimate_joint_propensity(3, treatment, covariates, ds, (0.01, 0.99))
    assert np.all(ps >= 0.01)
    assert np.all(ps <= 0.99)


def test_joint_propensity_positive(estimation_panel):
    _, treatment, covariates, ds = estimation_panel
    ps = _estimate_joint_propensity(3, treatment, covariates, ds, (0.01, 0.99))
    assert np.all(ps > 0)


def test_match_mask_single_period(rng):
    treatment = rng.integers(0, 2, size=(20, 1)).astype(float)
    ds = np.array([1.0])
    mask = _match_mask(treatment, ds)
    assert mask.shape == (20,)
    assert mask.dtype == bool


def test_match_mask_multi_period(estimation_panel):
    _, treatment, _, ds = estimation_panel
    mask = _match_mask(treatment, ds)
    assert mask.shape == (60,)
    np.testing.assert_array_equal(mask, np.all(treatment == ds, axis=1))


@pytest.mark.parametrize("method", ["ipw", "aipw", "ipw_msm"])
def test_mu_hat_is_float(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_ipw_estimator(3, outcome, treatment, covariates, ds, method=method)
    assert isinstance(result.mu_hat, float)


@pytest.mark.parametrize("method", ["ipw", "aipw", "ipw_msm"])
def test_variance_is_float(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_ipw_estimator(3, outcome, treatment, covariates, ds, method=method)
    assert isinstance(result.variance, float)


def test_ipw_and_msm_both_finite(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    ipw = compute_ipw_estimator(3, outcome, treatment, covariates, ds, method="ipw")
    msm = compute_ipw_estimator(3, outcome, treatment, covariates, ds, method="ipw_msm")
    assert np.isfinite(ipw.mu_hat)
    assert np.isfinite(msm.mu_hat)


@pytest.mark.parametrize("method", ["ipw", "aipw", "ipw_msm"])
def test_single_period(rng, method):
    n = 60
    treatment = rng.integers(0, 2, size=(n, 1)).astype(float)
    covariates = {0: rng.standard_normal((n, 3))}
    outcome = rng.standard_normal(n) + treatment[:, 0]
    ds = np.array([1.0])
    result = compute_ipw_estimator(1, outcome, treatment, covariates, ds, method=method)
    assert np.isfinite(result.mu_hat)
    assert result.variance >= 0.0


@pytest.mark.parametrize("method", ["ipw", "ipw_msm"])
def test_custom_clip_bounds(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_ipw_estimator(3, outcome, treatment, covariates, ds, method=method, clip_bounds=(0.05, 0.95))
    assert np.isfinite(result.mu_hat)
    assert result.variance >= 0.0


def test_all_units_match_ds(rng):
    n = 30
    treatment = np.ones((n, 2))
    covariates = {t: rng.standard_normal((n, 2)) for t in range(2)}
    outcome = rng.standard_normal(n) + 2.0
    ds = np.array([1.0, 1.0])
    result = compute_ipw_estimator(2, outcome, treatment, covariates, ds, method="ipw")
    assert np.isfinite(result.mu_hat)
    assert result.variance >= 0.0


def test_no_units_match_ds(rng):
    n = 30
    treatment = np.zeros((n, 2))
    covariates = {t: rng.standard_normal((n, 2)) for t in range(2)}
    outcome = rng.standard_normal(n)
    ds = np.array([1.0, 1.0])
    result = compute_ipw_estimator(2, outcome, treatment, covariates, ds, method="ipw")
    assert np.isfinite(result.mu_hat) or np.isnan(result.mu_hat)


def test_nan_in_covariates_handled(rng):
    n = 40
    treatment = np.zeros((n, 2))
    treatment[:20, :] = 1.0
    covariates = {t: rng.standard_normal((n, 3)) for t in range(2)}
    covariates[0][0, 0] = np.nan
    outcome = rng.standard_normal(n)
    ds = np.array([1.0, 1.0])
    result = compute_ipw_estimator(2, outcome, treatment, covariates, ds, method="ipw")
    assert np.isfinite(result.mu_hat)


def test_aipw_differs_from_ipw(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    ipw_result = compute_ipw_estimator(3, outcome, treatment, covariates, ds, method="ipw")
    aipw_result = compute_ipw_estimator(3, outcome, treatment, covariates, ds, method="aipw")
    assert ipw_result.mu_hat != pytest.approx(aipw_result.mu_hat, abs=1e-10)
