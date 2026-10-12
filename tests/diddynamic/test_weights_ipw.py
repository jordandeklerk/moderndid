"""Tests for IPW and AIPW weight estimation."""

import numpy as np
import pytest
import statsmodels.api as sm

from moderndid.diddynamic.container import IPWResult
from moderndid.diddynamic.estimation.coefficients import compute_coefficients
from moderndid.diddynamic.estimation.weights_ipw import (
    _independent_columns,
    _is_separated,
    _period_propensities,
    compute_ipw_estimator,
)


@pytest.mark.parametrize("method", ["ipw", "aipw"])
def test_returns_ipw_result(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_ipw_estimator(3, outcome, treatment, covariates, ds, method=method)
    assert isinstance(result, IPWResult)
    assert result.gammas.shape == (60, 3)
    assert result.predictions.shape == (60, 3)
    assert len(result.not_nas) == 3


@pytest.mark.parametrize("method", ["ipw", "aipw"])
def test_is_finite(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_ipw_estimator(3, outcome, treatment, covariates, ds, method=method)
    assert np.isfinite(result.mu_hat)


@pytest.mark.parametrize("method", ["ipw", "aipw"])
def test_is_float(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_ipw_estimator(3, outcome, treatment, covariates, ds, method=method)
    assert isinstance(result.mu_hat, float)


@pytest.mark.parametrize("method", ["ipw", "aipw"])
def test_non_negative(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_ipw_estimator(3, outcome, treatment, covariates, ds, method=method)
    assert result.variance >= 0.0


@pytest.mark.parametrize("method", ["ipw", "aipw"])
def test_variance_is_float(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_ipw_estimator(3, outcome, treatment, covariates, ds, method=method)
    assert isinstance(result.variance, float)


def test_raises_value_error(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    with pytest.raises(ValueError, match="Unknown method"):
        compute_ipw_estimator(3, outcome, treatment, covariates, ds, method="invalid")


def test_ipw_msm_is_unknown(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    with pytest.raises(ValueError, match="Unknown method 'ipw_msm'"):
        compute_ipw_estimator(3, outcome, treatment, covariates, ds, method="ipw_msm")


@pytest.mark.parametrize("method", ["ipw", "aipw"])
def test_weights_sum_to_one_on_target_history(estimation_panel, method):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_ipw_estimator(3, outcome, treatment, covariates, ds, method=method)
    for t in range(3):
        off_path = ~np.all(treatment[:, : t + 1] == ds[: t + 1], axis=1)
        assert result.gammas[:, t].sum() == pytest.approx(1.0)
        assert np.all(result.gammas[off_path, t] == 0.0)


def test_aipw_differs_from_ipw(lagged_effect_panel):
    outcome, treatment, covariates, ds = lagged_effect_panel
    ipw_result = compute_ipw_estimator(2, outcome, treatment, covariates, ds, method="ipw")
    aipw_result = compute_ipw_estimator(2, outcome, treatment, covariates, ds, method="aipw")
    assert ipw_result.mu_hat != pytest.approx(aipw_result.mu_hat, abs=1e-10)


def test_ipw_variance_is_weighted_residual_variance(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_ipw_estimator(3, outcome, treatment, covariates, ds, method="ipw")
    weights = result.gammas[:, -1]
    assert result.mu_hat == pytest.approx(weights @ outcome, rel=1e-12)
    assert result.variance == pytest.approx(np.sum(weights**2 * (outcome - result.mu_hat) ** 2), rel=1e-12)


def test_aipw_single_period_is_augmented_mean(rng):
    n = 200
    treatment = rng.integers(0, 2, size=(n, 1)).astype(float)
    covariates = {0: rng.standard_normal((n, 2))}
    outcome = covariates[0] @ np.array([1.0, -0.5]) + treatment[:, 0] + rng.standard_normal(n)
    ds = np.array([1.0])
    result = compute_ipw_estimator(1, outcome, treatment, covariates, ds, method="aipw", regularization=False)
    fitted = compute_coefficients(1, outcome, treatment, covariates, ds, "lasso_subsample", False, 10).pred_t[0]
    weights = result.gammas[:, 0]
    assert result.mu_hat == pytest.approx(fitted.mean() + weights @ (outcome - fitted), rel=1e-12)
    assert result.variance == pytest.approx(np.sum(weights**2 * (outcome - fitted) ** 2), rel=1e-12)


def test_aipw_corrects_projections_in_every_period(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_ipw_estimator(3, outcome, treatment, covariates, ds, method="aipw")
    gammas, preds, rows = result.gammas, result.predictions, result.not_nas
    adjustment = (gammas[rows[0], 0] - 1.0 / len(rows[0])) @ preds[rows[0], 0]
    for t in range(1, 3):
        adjustment += (gammas[rows[t], t] - gammas[rows[t], t - 1]) @ preds[rows[t], t]
    assert result.mu_hat == pytest.approx(gammas[rows[2], 2] @ outcome[rows[2]] - adjustment, rel=1e-12)


def test_aipw_recovers_potential_outcomes_when_covariates_respond_to_treatment(responsive_panel):
    outcome, treatment, covariates, mu_treated, mu_control = responsive_panel
    treated = compute_ipw_estimator(2, outcome, treatment, covariates, np.ones(2), method="aipw", regularization=False)
    control = compute_ipw_estimator(2, outcome, treatment, covariates, np.zeros(2), method="aipw", regularization=False)
    assert treated.mu_hat == pytest.approx(mu_treated, abs=0.15)
    assert control.mu_hat == pytest.approx(mu_control, abs=0.15)
    assert treated.mu_hat - control.mu_hat == pytest.approx(mu_treated - mu_control, abs=0.15)


def test_ipw_recovers_potential_outcomes_when_treatment_persists(responsive_panel):
    outcome, treatment, covariates, mu_treated, mu_control = responsive_panel
    treated = compute_ipw_estimator(2, outcome, treatment, covariates, np.ones(2), method="ipw")
    control = compute_ipw_estimator(2, outcome, treatment, covariates, np.zeros(2), method="ipw")
    assert treated.mu_hat == pytest.approx(mu_treated, abs=0.15)
    assert control.mu_hat == pytest.approx(mu_control, abs=0.15)
    assert treated.mu_hat - control.mu_hat == pytest.approx(mu_treated - mu_control, abs=0.15)


def test_period_propensities_condition_on_target_history(estimation_panel):
    _, treatment, covariates, ds = estimation_panel
    ps = _period_propensities(3, treatment, covariates, ds, (0.01, 0.99))
    followed = np.column_stack([np.all(treatment[:, :t] == ds[:t], axis=1) for t in range(3)])
    assert ps.shape == (60, 3)
    assert np.array_equal(np.isnan(ps), ~followed)
    assert np.all((ps[followed] >= 0.01) & (ps[followed] <= 0.99))


def test_period_propensity_fits_units_on_target_history(rng):
    n = 300
    covariates = {t: rng.standard_normal((n, 2)) for t in range(2)}
    first = rng.integers(0, 2, size=n).astype(float)
    index = covariates[1][:, 0] + 2.0 * (2.0 * first - 1.0)
    second = (rng.random(n) < 1 / (1 + np.exp(-index))).astype(float)
    ps = _period_propensities(2, np.column_stack([first, second]), covariates, np.ones(2), (0.01, 0.99))
    on_path = first == 1.0
    design = sm.add_constant(covariates[1][on_path])
    expected = sm.Logit(second[on_path], design).fit(disp=0).predict(design)
    np.testing.assert_allclose(ps[on_path, 1], np.clip(expected, 0.01, 0.99), rtol=1e-6)
    assert np.all(np.isnan(ps[~on_path, 1]))


def test_missing_covariate_removes_unit_from_later_weights(rng):
    n = 40
    treatment = rng.integers(0, 2, size=(n, 2)).astype(float)
    treatment[3] = 1.0
    covariates = {t: rng.standard_normal((n, 2)) for t in range(2)}
    covariates[1][3, 0] = np.nan
    outcome = rng.standard_normal(n)
    result = compute_ipw_estimator(2, outcome, treatment, covariates, np.ones(2), method="ipw")
    assert result.gammas[3, 0] > 0.0
    assert result.gammas[3, 1] == 0.0
    assert result.gammas[:, 1].sum() == pytest.approx(1.0)


def test_ipw_skips_units_with_missing_outcome(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    outcome = outcome.copy()
    outcome[0] = np.nan
    result = compute_ipw_estimator(3, outcome, treatment, covariates, ds, method="ipw")
    assert np.isfinite(result.mu_hat)
    assert np.isfinite(result.variance)
    assert np.all(result.gammas[0] == 0.0)


def test_independent_columns_drop_constant_and_spanned_columns(rng):
    a = rng.standard_normal(50)
    b = rng.standard_normal(50)
    x = np.column_stack([a, np.zeros(50), 2.0 * a, b, a + b, np.full(50, 3.0)])
    assert _independent_columns(x).tolist() == [True, False, False, True, False, False]


def test_independent_columns_drop_last_dummy_level(rng):
    levels = rng.integers(0, 3, size=60)
    dummies = np.column_stack([(levels == k).astype(float) for k in range(3)])
    assert _independent_columns(dummies).tolist() == [True, True, False]


def test_independent_columns_drop_near_duplicate_columns(rng):
    a = rng.standard_normal(200)
    b = rng.standard_normal(200)
    near = a + 1e-9 * rng.standard_normal(200)
    distinct = a + 1e-5 * rng.standard_normal(200)
    x = np.column_stack([a, near, b, a.astype(np.float32).astype(float), distinct])
    assert _independent_columns(x).tolist() == [True, False, True, False, True]


def test_is_separated_detects_complete_separation(rng):
    x = rng.standard_normal((80, 2))
    treatment = (x[:, 0] + 0.3 * x[:, 1] > 0).astype(float)
    assert _is_separated(np.column_stack([np.ones(80), x]), treatment)


def test_is_separated_detects_quasi_separation(rng):
    group = (np.arange(80) < 8).astype(float)
    treatment = rng.integers(0, 2, size=80).astype(float)
    treatment[:8] = 1.0
    design = np.column_stack([np.ones(80), rng.standard_normal(80), group])
    assert _is_separated(design, treatment)


def test_is_separated_false_when_groups_overlap(rng):
    x = rng.standard_normal((80, 2))
    treatment = rng.integers(0, 2, size=80).astype(float)
    assert not _is_separated(np.column_stack([np.ones(80), x]), treatment)


def test_constant_and_duplicate_columns_leave_ipw_unchanged(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    padded = {t: np.column_stack([x, np.zeros(len(x)), x[:, 0]]) for t, x in covariates.items()}
    base = compute_ipw_estimator(3, outcome, treatment, covariates, ds, method="ipw")
    result = compute_ipw_estimator(3, outcome, treatment, padded, ds, method="ipw")
    assert result.mu_hat == pytest.approx(base.mu_hat, rel=1e-12)
    assert result.variance == pytest.approx(base.variance, rel=1e-12)


def test_near_duplicate_covariate_leaves_ipw_unchanged(rng):
    n = 2000
    x = rng.standard_normal((n, 2))
    treatment = (rng.random((n, 1)) < 1 / (1 + np.exp(-x[:, :1]))).astype(float)
    outcome = x @ np.array([1.0, -0.5]) + treatment[:, 0] + rng.standard_normal(n)
    near = np.column_stack([x, x[:, 0] + 1e-8 * rng.standard_normal(n)])
    base = compute_ipw_estimator(1, outcome, treatment, {0: x}, np.ones(1), method="ipw")
    result = compute_ipw_estimator(1, outcome, treatment, {0: near}, np.ones(1), method="ipw")
    assert result.mu_hat == base.mu_hat
    assert result.variance == base.variance


@pytest.mark.parametrize("method", ["ipw", "aipw"])
def test_separating_covariate_raises(rng, method):
    n = 80
    treatment = rng.integers(0, 2, size=(n, 2)).astype(float)
    treatment[:10, 0] = 1.0
    group = (np.arange(n) < 10).astype(float)
    covariates = {t: np.column_stack([rng.standard_normal(n), group]) for t in range(2)}
    outcome = rng.standard_normal(n)
    with pytest.raises(ValueError, match="perfectly predict the treatment in period 1"):
        compute_ipw_estimator(2, outcome, treatment, covariates, np.ones(2), method=method)


def test_separation_among_units_on_target_history_raises(rng):
    n = 200
    treatment = rng.integers(0, 2, size=(n, 2)).astype(float)
    second_covariate = rng.standard_normal(n)
    on_path = treatment[:, 0] == 1.0
    treatment[on_path, 1] = (second_covariate[on_path] > 0).astype(float)
    covariates = {0: rng.standard_normal((n, 1)), 1: second_covariate[:, None]}
    outcome = rng.standard_normal(n)
    with pytest.raises(ValueError, match="perfectly predict the treatment in period 2"):
        compute_ipw_estimator(2, outcome, treatment, covariates, np.ones(2), method="ipw")
    result = compute_ipw_estimator(2, outcome, treatment, covariates, np.zeros(2), method="ipw")
    assert np.isfinite(result.mu_hat)


@pytest.mark.parametrize("method", ["ipw", "aipw"])
def test_single_period_finite(rng, method):
    n = 60
    treatment = rng.integers(0, 2, size=(n, 1)).astype(float)
    covariates = {0: rng.standard_normal((n, 3))}
    outcome = rng.standard_normal(n) + treatment[:, 0]
    ds = np.array([1.0])
    result = compute_ipw_estimator(1, outcome, treatment, covariates, ds, method=method)
    assert np.isfinite(result.mu_hat)
    assert result.variance >= 0.0


def test_custom_clip_bounds(estimation_panel):
    outcome, treatment, covariates, ds = estimation_panel
    result = compute_ipw_estimator(
        3,
        outcome,
        treatment,
        covariates,
        ds,
        method="ipw",
        clip_bounds=(0.05, 0.95),
    )
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


@pytest.mark.parametrize("method", ["ipw", "aipw"])
@pytest.mark.parametrize(
    ("first_period", "message"), [(0.0, r"\[1\] through period 1"), (1.0, r"\[1, 1\] through period 2")]
)
def test_no_units_match_ds_raises(rng, method, first_period, message):
    n = 30
    treatment = np.zeros((n, 2))
    treatment[:, 0] = first_period
    covariates = {t: rng.standard_normal((n, 2)) for t in range(2)}
    outcome = rng.standard_normal(n)
    with pytest.raises(ValueError, match=f"follows the treatment history {message}"):
        compute_ipw_estimator(2, outcome, treatment, covariates, np.ones(2), method=method)


def test_aipw_with_one_unit_on_target_history_raises(rng):
    n = 60
    covariates = {t: rng.standard_normal((n, 2)) for t in range(2)}
    treatment = np.zeros((n, 2))
    treatment[: n // 2, 0] = 1.0
    treatment[np.argmin(np.linalg.norm(covariates[1][: n // 2], axis=1)), 1] = 1.0
    outcome = rng.standard_normal(n)
    with pytest.raises(ValueError, match=r"Fewer than two units .* history \[1, 1\] through period 2"):
        compute_ipw_estimator(2, outcome, treatment, covariates, np.ones(2), method="aipw")


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


def test_all_same_treatment_gives_sample_mean(rng):
    n = 50
    treatment = np.ones((n, 1))
    covariates = {0: rng.standard_normal((n, 2)) * 0.001}
    outcome = np.full(n, 3.0)
    ds = np.array([1.0])
    result = compute_ipw_estimator(1, outcome, treatment, covariates, ds, method="ipw")
    assert result.mu_hat == pytest.approx(3.0, abs=0.1)


def test_constant_outcome_mu_hat_near_constant(rng):
    n = 60
    treatment = np.zeros((n, 1))
    treatment[:30, 0] = 1.0
    covariates = {0: rng.standard_normal((n, 2))}
    outcome = np.full(n, 7.0)
    ds = np.array([1.0])
    result = compute_ipw_estimator(1, outcome, treatment, covariates, ds, method="ipw")
    assert result.mu_hat == pytest.approx(7.0, abs=0.5)


def test_zero_variance_when_all_same_outcome_and_treatment(rng):
    n = 40
    treatment = np.ones((n, 1))
    covariates = {0: np.ones((n, 1))}
    outcome = np.full(n, 5.0)
    ds = np.array([1.0])
    result = compute_ipw_estimator(1, outcome, treatment, covariates, ds, method="ipw")
    assert result.variance == pytest.approx(0.0, abs=1e-6)
