"""Tests for the pscore module."""

import numpy as np
import pytest
import statsmodels.api as sm

from drsynthdid import pscore


@pytest.fixture(scope="module")
def simple_logit_data():
    rng = np.random.default_rng(12345)
    n_obs = 100
    n_covars = 2
    covariates = rng.standard_normal((n_obs, n_covars))
    covariates_with_intercept = sm.add_constant(covariates, prepend=True)

    true_beta = np.array([-0.5, 0.8, -0.6])
    linear_combination = covariates_with_intercept @ true_beta
    prob = 1 / (1 + np.exp(-linear_combination))
    treatment_indicator = rng.binomial(1, prob, n_obs)
    observation_weights = rng.uniform(0.5, 1.5, n_obs)

    return treatment_indicator, covariates_with_intercept, observation_weights, n_obs


def test_estimate_propensity_score_mle_basic(simple_logit_data):
    treatment, exog, weights, _ = simple_logit_data

    ps_fit, cov_params = pscore._estimate_propensity_score_mle(treatment, exog, weights)

    assert ps_fit is not None
    assert cov_params is not None
    assert ps_fit.shape == (treatment.shape[0],)
    assert cov_params.shape == (exog.shape[1], exog.shape[1])
    assert np.all((ps_fit >= 1e-16) & (ps_fit <= 1.0 - 1e-16))


def test_estimate_propensity_score_mle_freq_weights(simple_logit_data):
    treatment, exog, _, _ = simple_logit_data
    freq_weights = np.random.randint(1, 5, size=treatment.shape[0])

    ps_fit, cov_params = pscore._estimate_propensity_score_mle(treatment, exog, freq_weights, is_frequency_weights=True)

    assert ps_fit is not None
    assert cov_params is not None
    assert ps_fit.shape == (treatment.shape[0],)
    assert np.all((ps_fit >= 1e-16) & (ps_fit <= 1.0 - 1e-16))


def test_calculate_ipt_loss_graham(simple_logit_data):
    treatment, exog, weights, n_obs = simple_logit_data
    gamma_params = np.random.rand(exog.shape[1])
    loss = pscore._calculate_ipt_loss_graham(gamma_params, treatment, exog, weights, n_obs)
    assert isinstance(loss, float)


def test_calculate_ipt_gradient_graham(simple_logit_data):
    treatment, exog, weights, n_obs = simple_logit_data
    gamma_params = np.random.rand(exog.shape[1])
    gradient = pscore._calculate_ipt_gradient_graham(gamma_params, treatment, exog, weights, n_obs)
    assert gradient.shape == (exog.shape[1],)


def test_calculate_ipt_hessian_graham(simple_logit_data):
    treatment, exog, weights, n_obs = simple_logit_data
    gamma_params = np.random.rand(exog.shape[1])
    hessian = pscore._calculate_ipt_hessian_graham(gamma_params, treatment, exog, weights, n_obs)
    assert hessian.shape == (exog.shape[1], exog.shape[1])


def test_objective_calibrated_propensity_score_tan(simple_logit_data):
    treatment, exog, weights, _ = simple_logit_data
    gamma_params = np.random.rand(exog.shape[1])
    value, gradient, hessian = pscore._objective_calibrated_propensity_score_tan(gamma_params, treatment, exog, weights)
    assert isinstance(value, float)
    assert gradient.shape == (exog.shape[1],)
    assert hessian.shape == (exog.shape[1], exog.shape[1])


def test_calculate_propensity_score_perfect_separation_mle_fallback(caplog):
    caplog.set_level("WARNING")
    n_obs = 50
    covariates = np.linspace(-1, 1, n_obs)
    covariates_with_intercept = sm.add_constant(covariates, prepend=True)
    treatment_indicator = (covariates > 0).astype(int)
    observation_weights = np.ones(n_obs)

    original_obj_cal = pscore._objective_calibrated_propensity_score_tan
    original_loss_ipt = pscore._calculate_ipt_loss_graham

    def failing_obj_cal(*args, **kwargs):
        raise RuntimeError("Forced failure for Tan method")

    def failing_loss_ipt(*args, **kwargs):
        raise RuntimeError("Forced failure for IPT method")

    pscore._objective_calibrated_propensity_score_tan = failing_obj_cal
    pscore._calculate_ipt_loss_graham = failing_loss_ipt

    with pytest.warns(UserWarning, match="Falling back to MLE"):
        estimated_ps, flag = pscore.calculate_propensity_score(
            treatment_indicator, covariates_with_intercept, observation_weights, n_obs
        )

    pscore._objective_calibrated_propensity_score_tan = original_obj_cal
    pscore._calculate_ipt_loss_graham = original_loss_ipt

    assert flag == 2
    assert not np.any(np.isnan(estimated_ps))


def test_all_methods_fail_returns_nan_and_flag_3(caplog):
    caplog.set_level("WARNING")
    rng = np.random.default_rng(9876)
    n_obs = 20
    covariates = sm.add_constant(rng.standard_normal((n_obs, 1)), prepend=True)
    treatment_indicator = np.zeros(n_obs)
    observation_weights = np.ones(n_obs)

    original_obj_cal = pscore._objective_calibrated_propensity_score_tan
    original_loss_ipt = pscore._calculate_ipt_loss_graham
    original_mle = pscore._estimate_propensity_score_mle

    def failing_obj_cal(*args, **kwargs):
        raise RuntimeError("Forced failure for Tan method")

    def failing_loss_ipt(*args, **kwargs):
        raise RuntimeError("Forced failure for IPT method")

    def failing_mle(*args, **kwargs):
        return np.full(args[0].shape, np.nan), None

    pscore._objective_calibrated_propensity_score_tan = failing_obj_cal
    pscore._calculate_ipt_loss_graham = failing_loss_ipt
    pscore._estimate_propensity_score_mle = failing_mle

    with pytest.warns(UserWarning, match="All propensity score estimation methods failed"):
        estimated_ps, flag = pscore.calculate_propensity_score(
            treatment_indicator, covariates, observation_weights, n_obs
        )

    pscore._objective_calibrated_propensity_score_tan = original_obj_cal
    pscore._calculate_ipt_loss_graham = original_loss_ipt
    pscore._estimate_propensity_score_mle = original_mle

    assert flag == 3
    assert np.all(np.isnan(estimated_ps))
