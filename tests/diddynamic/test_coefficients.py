"""Tests for LASSO coefficient estimation in dynamic covariate balancing."""

import numpy as np
import pytest

from moderndid.dev.diddynamic.estimation.coefficients import CoefficientResult, compute_coefficients


@pytest.mark.parametrize("method", ["lasso_plain", "lasso_subsample"])
def test_returns_coefficient_result(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method=method, nfolds=3)
    assert isinstance(result, CoefficientResult)


@pytest.mark.parametrize("method", ["lasso_plain", "lasso_subsample"])
def test_coef_t_length(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method=method, nfolds=3)
    assert len(result.coef_t) == 3


@pytest.mark.parametrize("method", ["lasso_plain", "lasso_subsample"])
def test_coef_t_shape(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method=method, nfolds=3)
    for t in range(3):
        assert result.coef_t[t].shape[0] == 1 + covariates[t].shape[1]


@pytest.mark.parametrize("method", ["lasso_plain", "lasso_subsample"])
def test_pred_t_length(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method=method, nfolds=3)
    assert len(result.pred_t) == 3


@pytest.mark.parametrize("method", ["lasso_plain", "lasso_subsample"])
def test_pred_t_values_finite(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method=method, nfolds=3)
    for t in range(3):
        assert np.all(np.isfinite(result.pred_t[t]))


@pytest.mark.parametrize("method", ["lasso_plain", "lasso_subsample"])
def test_covariates_nonna_length(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method=method, nfolds=3)
    assert len(result.covariates_nonna) == 3


@pytest.mark.parametrize("method", ["lasso_plain", "lasso_subsample"])
def test_not_nas_length(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method=method, nfolds=3)
    assert len(result.not_nas) == 3


@pytest.mark.parametrize("method", ["lasso_plain", "lasso_subsample"])
def test_all_valid_when_no_nan(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method=method, nfolds=3)
    assert len(result.not_nas[3 - 1]) == 60


@pytest.mark.parametrize("method", ["lasso_plain", "lasso_subsample"])
def test_no_regularization(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method=method, regularization=False, nfolds=3)
    assert isinstance(result, CoefficientResult)
    for t in range(3):
        assert np.all(np.isfinite(result.coef_t[t]))


def test_invalid_method_raises(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    with pytest.raises(ValueError, match="Unknown method"):
        compute_coefficients(3, outcome, treatment, covariates, ds, method="invalid")


def test_model_effect_populated_for_lasso_plain(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method="lasso_plain", nfolds=3)
    assert len(result.model_effect) == 3
    for eff in result.model_effect:
        assert np.isfinite(eff)


def test_model_effect_empty_for_subsample(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method="lasso_subsample", nfolds=3)
    assert result.model_effect == []


def test_handles_nan_in_outcome(rng):
    treatment = np.zeros((60, 3))
    covariates = {t: rng.standard_normal((60, 2)) for t in range(3)}
    outcome = rng.standard_normal(60)
    outcome[0] = np.nan
    ds = np.zeros(3)
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method="lasso_plain", nfolds=3)
    assert len(result.not_nas[3 - 1]) == 60 - 1


@pytest.mark.parametrize("method", ["lasso_plain", "lasso_subsample"])
def test_lags_parameter(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method=method, lags=2, nfolds=3)
    assert isinstance(result, CoefficientResult)


@pytest.mark.parametrize("method", ["lasso_plain", "lasso_subsample"])
def test_dim_fe_zeroing(rng, method):
    n_cov = 3
    n_fe = 2
    treatment = np.zeros((60, 3))
    covariates = {t: rng.standard_normal((60, n_cov + n_fe)) for t in range(3)}
    outcome = rng.standard_normal(60)
    ds = np.zeros(3)
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method=method, dim_fe=n_fe, nfolds=3)
    assert isinstance(result, CoefficientResult)
    assert len(result.coef_t) == 3


def test_single_period(rng):
    n = 30
    treatment = rng.integers(0, 2, size=(n, 1)).astype(float)
    covariates = {0: rng.standard_normal((n, 2))}
    outcome = rng.standard_normal(n) + treatment[:, 0]
    ds = np.array([1.0])
    result = compute_coefficients(1, outcome, treatment, covariates, ds, method="lasso_plain", nfolds=3)
    assert len(result.coef_t) == 1
    assert len(result.pred_t) == 1


def test_pred_t_matches_not_nas_count(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method="lasso_plain", nfolds=3)
    for t in range(3):
        assert result.pred_t[t].shape[0] == len(result.not_nas[t])


def test_covariates_nonna_no_nans(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method="lasso_plain", nfolds=3)
    for t in range(3):
        assert not np.any(np.isnan(result.covariates_nonna[t]))


def test_handles_nan_in_covariates(rng):
    n = 40
    treatment = np.zeros((n, 2))
    treatment[:20, :] = 1.0
    covariates = {t: rng.standard_normal((n, 2)) for t in range(2)}
    covariates[0][0, 0] = np.nan
    outcome = rng.standard_normal(n)
    ds = np.array([1.0, 1.0])
    result = compute_coefficients(2, outcome, treatment, covariates, ds, method="lasso_plain", nfolds=3)
    assert len(result.not_nas[0]) < n


def test_subsample_excludes_non_matching_units(rng):
    n = 40
    treatment = np.zeros((n, 2))
    treatment[:20, :] = 1.0
    covariates = {t: rng.standard_normal((n, 3)) for t in range(2)}
    outcome = rng.standard_normal(n) + 2.0 * treatment[:, -1]
    ds_treated = np.array([1.0, 1.0])
    ds_control = np.array([0.0, 0.0])
    result_treated = compute_coefficients(
        2, outcome, treatment, covariates, ds_treated, method="lasso_subsample", nfolds=3
    )
    result_control = compute_coefficients(
        2, outcome, treatment, covariates, ds_control, method="lasso_subsample", nfolds=3
    )
    assert not np.allclose(result_treated.coef_t[1], result_control.coef_t[1], atol=0.1)


def test_very_few_matching_units(rng):
    n = 40
    treatment = np.zeros((n, 2))
    treatment[:3, :] = 1.0
    covariates = {t: rng.standard_normal((n, 2)) for t in range(2)}
    outcome = rng.standard_normal(n)
    ds = np.array([1.0, 1.0])
    result = compute_coefficients(2, outcome, treatment, covariates, ds, method="lasso_subsample", nfolds=2)
    assert isinstance(result, CoefficientResult)
    assert len(result.coef_t) == 2
