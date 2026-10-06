"""Tests for LASSO coefficient estimation in dynamic covariate balancing."""

import numpy as np
import pytest

from moderndid.diddynamic.container import CoefficientResult
from moderndid.diddynamic.estimation.coefficients import (
    _fit_penalized,
    _free_coefficients,
    _interpolation_weights,
    _one_se_index,
    compute_coefficients,
)


@pytest.mark.parametrize("method", ["lasso_plain", "lasso_subsample"])
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_returns_coefficient_result(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method=method, nfolds=3)
    assert isinstance(result, CoefficientResult)


@pytest.mark.parametrize("method", ["lasso_plain", "lasso_subsample"])
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_length_equals_n_periods(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method=method, nfolds=3)
    assert len(result.coef_t) == 3


@pytest.mark.parametrize("method", ["lasso_plain", "lasso_subsample"])
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_shape_includes_intercept(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method=method, nfolds=3)
    for t in range(3):
        assert result.coef_t[t].shape[0] == 1 + covariates[t].shape[1]


@pytest.mark.parametrize("method", ["lasso_plain", "lasso_subsample"])
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_values_finite(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method=method, nfolds=3)
    for t in range(3):
        assert np.all(np.isfinite(result.coef_t[t]))


@pytest.mark.parametrize("method", ["lasso_plain", "lasso_subsample"])
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_pred_t_length_equals_n_periods(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method=method, nfolds=3)
    assert len(result.pred_t) == 3


@pytest.mark.parametrize("method", ["lasso_plain", "lasso_subsample"])
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_pred_t_values_finite(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method=method, nfolds=3)
    for t in range(3):
        assert np.all(np.isfinite(result.pred_t[t]))


def test_matches_not_nas_count(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method="lasso_plain", nfolds=3)
    for t in range(3):
        assert result.pred_t[t].shape[0] == len(result.not_nas[t])


@pytest.mark.parametrize("method", ["lasso_plain", "lasso_subsample"])
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_length(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method=method, nfolds=3)
    assert len(result.covariates_nonna) == 3


def test_no_nans(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method="lasso_plain", nfolds=3)
    for t in range(3):
        assert not np.any(np.isnan(result.covariates_nonna[t]))


@pytest.mark.parametrize("method", ["lasso_plain", "lasso_subsample"])
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_not_nas_length(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method=method, nfolds=3)
    assert len(result.not_nas) == 3


@pytest.mark.parametrize("method", ["lasso_plain", "lasso_subsample"])
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_all_valid_when_no_nan(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method=method, nfolds=3)
    assert len(result.not_nas[2]) == 60


def test_populated_for_lasso_plain(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method="lasso_plain", nfolds=3)
    assert len(result.model_effect) == 3
    for eff in result.model_effect:
        assert np.isfinite(eff)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_empty_for_subsample(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method="lasso_subsample", nfolds=3)
    assert result.model_effect == []


@pytest.mark.parametrize("method", ["lasso_plain", "lasso_subsample"])
def test_returns_finite_coefficients(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_coefficients(
        3,
        outcome,
        treatment,
        covariates,
        ds,
        method=method,
        regularization=False,
        nfolds=3,
    )
    assert isinstance(result, CoefficientResult)
    for t in range(3):
        assert np.all(np.isfinite(result.coef_t[t]))


def test_raises_value_error(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    with pytest.raises(ValueError, match="Unknown method"):
        compute_coefficients(3, outcome, treatment, covariates, ds, method="invalid")


def test_nan_in_outcome_reduces_not_nas(rng):
    treatment = np.zeros((60, 3))
    covariates = {t: rng.standard_normal((60, 2)) for t in range(3)}
    outcome = rng.standard_normal(60)
    outcome[0] = np.nan
    ds = np.zeros(3)
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method="lasso_plain", nfolds=3)
    assert len(result.not_nas[2]) == 59


def test_nan_in_covariates_reduces_not_nas(rng):
    n = 40
    treatment = np.zeros((n, 2))
    treatment[:20, :] = 1.0
    covariates = {t: rng.standard_normal((n, 2)) for t in range(2)}
    covariates[0][0, 0] = np.nan
    outcome = rng.standard_normal(n)
    ds = np.array([1.0, 1.0])
    result = compute_coefficients(2, outcome, treatment, covariates, ds, method="lasso_plain", nfolds=3)
    assert len(result.not_nas[0]) < n


@pytest.mark.parametrize("method", ["lasso_plain", "lasso_subsample"])
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_lags_accepted(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method=method, lags=2, nfolds=3)
    assert isinstance(result, CoefficientResult)


@pytest.mark.parametrize("method", ["lasso_plain", "lasso_subsample"])
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_zeroing_with_dim_fe(rng, method):
    n_cov = 3
    n_fe = 2
    treatment = np.zeros((60, 3))
    covariates = {t: rng.standard_normal((60, n_cov + n_fe)) for t in range(3)}
    outcome = rng.standard_normal(60)
    ds = np.zeros(3)
    result = compute_coefficients(3, outcome, treatment, covariates, ds, method=method, dim_fe=n_fe, nfolds=3)
    assert isinstance(result, CoefficientResult)
    assert len(result.coef_t) == 3


def test_single_period_basic(rng):
    n = 30
    treatment = rng.integers(0, 2, size=(n, 1)).astype(float)
    covariates = {0: rng.standard_normal((n, 2))}
    outcome = rng.standard_normal(n) + treatment[:, 0]
    ds = np.array([1.0])
    result = compute_coefficients(1, outcome, treatment, covariates, ds, method="lasso_plain", nfolds=3)
    assert len(result.coef_t) == 1
    assert len(result.pred_t) == 1


def test_excludes_non_matching_units(rng):
    n = 40
    treatment = np.zeros((n, 2))
    treatment[:20, :] = 1.0
    covariates = {t: rng.standard_normal((n, 3)) for t in range(2)}
    outcome = rng.standard_normal(n) + 2.0 * treatment[:, -1]
    ds_treated = np.array([1.0, 1.0])
    ds_control = np.array([0.0, 0.0])
    result_treated = compute_coefficients(
        2,
        outcome,
        treatment,
        covariates,
        ds_treated,
        method="lasso_subsample",
        nfolds=3,
    )
    result_control = compute_coefficients(
        2,
        outcome,
        treatment,
        covariates,
        ds_control,
        method="lasso_subsample",
        nfolds=3,
    )
    assert not np.allclose(result_treated.coef_t[1], result_control.coef_t[1], atol=0.1)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
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


@pytest.mark.parametrize("n_matching", [0, 1])
def test_lasso_subsample_needs_two_matching_units(rng, n_matching):
    n = 40
    treatment = np.zeros((n, 2))
    treatment[:n_matching, :] = 1.0
    covariates = {t: rng.standard_normal((n, 2)) for t in range(2)}
    outcome = rng.standard_normal(n)
    with pytest.raises(ValueError, match=r"Fewer than two units .* history \[1, 1\] through period 2"):
        compute_coefficients(2, outcome, treatment, covariates, np.ones(2), method="lasso_subsample")


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_constant_outcome_intercept_near_value(rng):
    n = 40
    treatment = np.zeros((n, 2))
    treatment[:20, :] = 1.0
    covariates = {t: rng.standard_normal((n, 2)) for t in range(2)}
    outcome = np.full(n, 5.0)
    ds = np.array([0.0, 0.0])
    result = compute_coefficients(2, outcome, treatment, covariates, ds, method="lasso_subsample", nfolds=3)
    assert result.coef_t[1][0] == pytest.approx(5.0, abs=1.0)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_zero_outcome_predictions_near_zero(rng):
    n = 60
    treatment = np.zeros((n, 2))
    treatment[:30, :] = 1.0
    covariates = {t: rng.standard_normal((n, 3)) for t in range(2)}
    outcome = np.zeros(n)
    ds = np.array([0.0, 0.0])
    result = compute_coefficients(2, outcome, treatment, covariates, ds, method="lasso_subsample", nfolds=3)
    assert np.allclose(result.pred_t[1], 0.0, atol=0.5)


def test_lasso_plain_predictions_come_from_penalized_fit(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    lasso = compute_coefficients(3, outcome, treatment, covariates, ds, method="lasso_plain", nfolds=3)
    ridge = compute_coefficients(
        3, outcome, treatment, covariates, ds, method="lasso_plain", regularization=False, nfolds=3
    )
    assert not np.allclose(lasso.pred_t[2], ridge.pred_t[2], atol=1e-3)


def test_lasso_plain_stores_intercept(rng):
    n = 80
    treatment = np.zeros((n, 2))
    treatment[:40, :] = 1.0
    covariates = {t: rng.standard_normal((n, 2)) for t in range(2)}
    outcome = 5.0 + rng.standard_normal(n)
    result = compute_coefficients(2, outcome, treatment, covariates, np.zeros(2), method="lasso_plain", nfolds=5)
    assert result.coef_t[1][0] == pytest.approx(5.0, abs=0.5)


def test_lasso_plain_invariant_to_covariate_scale(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    scaled = {t: cov * np.array([10.0, 1.0, 0.1]) for t, cov in covariates.items()}
    base = compute_coefficients(3, outcome, treatment, covariates, ds, method="lasso_plain", nfolds=3)
    rescaled = compute_coefficients(3, outcome, treatment, scaled, ds, method="lasso_plain", nfolds=3)
    for t in range(3):
        np.testing.assert_allclose(rescaled.pred_t[t], base.pred_t[t], atol=1e-6)


def test_lags_penalize_older_treatments(lagged_effect_panel):
    outcome, treatment, covariates, ds = lagged_effect_panel
    free = compute_coefficients(2, outcome, treatment, covariates, ds, method="lasso_plain", nfolds=5)
    one_lag = compute_coefficients(2, outcome, treatment, covariates, ds, method="lasso_plain", nfolds=5, lags=1)
    assert not np.allclose(free.pred_t[1], one_lag.pred_t[1], atol=1e-6)


def test_unpenalized_column_keeps_least_squares_value(lagged_effect_panel):
    outcome, treatment, covariates, _ = lagged_effect_panel
    x = np.column_stack([covariates[1], treatment])
    penalized = np.array([True, True, True, False, False])
    _, coefs = _fit_penalized(x, outcome, penalized, 5)
    free = np.column_stack([np.ones(len(outcome)), treatment])
    expected = np.linalg.lstsq(free, outcome - covariates[1] @ coefs[:3], rcond=None)[0]
    np.testing.assert_allclose(coefs[3:], expected[1:], atol=1e-10)


def test_one_se_index_picks_largest_penalty_within_one_se():
    folds = np.repeat(np.arange(3), 10)
    fold_errors = np.array([[1.0, 0.5, 0.3], [1.0, 0.6, 0.6], [1.0, 0.4, 0.45]])
    errors = fold_errors[folds]
    assert _one_se_index(errors, folds, 30) == 1


def test_interpolation_weights_hold_ends_and_interpolate():
    grid = np.array([4.0, 2.0, 1.0])
    weights = _interpolation_weights(grid, np.array([5.0, 4.0, 3.0, 1.5, 0.5]))
    np.testing.assert_allclose(np.array([10.0, 20.0, 30.0]) @ weights, [10.0, 10.0, 15.0, 25.0, 30.0])


def test_free_coefficients_zero_for_repeated_column(rng):
    d = rng.integers(0, 2, size=50).astype(float)
    free = np.column_stack([np.ones(50), d, d])
    coefs = _free_coefficients(free, 1.0 + 2.0 * d)
    np.testing.assert_allclose(coefs, [1.0, 2.0, 0.0], atol=1e-10)
