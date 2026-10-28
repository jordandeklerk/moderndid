"""Tests for the IPT propensity score calculation and estimation."""

import re
import warnings

import numpy as np
import pytest
import scipy.optimize

from moderndid.drdid.propensity.pscore_ipt import (
    _add_quantile_constraints,
    _loss_ps_cal,
    _loss_ps_ipt,
    _remove_collinear_columns,
    _weighted_quantile,
    calculate_pscore_ipt,
)


def test_loss_ps_cal_matches_a_hand_computed_weighted_tilting_loss():
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, -1.0]])
    D = np.array([1, 0, 1, 0])
    iw = np.array([1.0, 2.0, 1.0, 3.0])
    gamma = np.array([0.0, np.log(2.0)])

    value, gradient, hessian = _loss_ps_cal(gamma, D, X, iw)

    assert isinstance(value, float)
    assert isinstance(gradient, np.ndarray)
    assert isinstance(hessian, np.ndarray)
    np.testing.assert_allclose(value, (5.5 - 2 * np.log(2.0)) / 4, rtol=1e-12)
    np.testing.assert_allclose(gradient, [0.875, 0.125], rtol=1e-12)
    np.testing.assert_allclose(hessian, [[1.375, 0.625], [0.625, 1.375]], rtol=1e-12)


def test_loss_ps_cal_gradient_and_hessian_are_derivatives_of_its_value():
    rng = np.random.default_rng(42)
    X = np.column_stack([np.ones(100), rng.standard_normal((100, 2))])
    D = rng.binomial(1, 0.3, 100)
    iw = rng.exponential(1, 100)
    gamma = np.array([0.1, 0.2, -0.3])
    steps = 1e-5 * np.eye(3)

    value, gradient, hessian = _loss_ps_cal(gamma, D, X, iw)
    above = [_loss_ps_cal(gamma + step, D, X, iw) for step in steps]
    below = [_loss_ps_cal(gamma - step, D, X, iw) for step in steps]

    assert isinstance(value, float)
    assert np.isfinite(value)
    assert isinstance(gradient, np.ndarray)
    assert isinstance(hessian, np.ndarray)
    np.testing.assert_allclose(gradient, [(up[0] - down[0]) / 2e-5 for up, down in zip(above, below)], rtol=1e-7)
    np.testing.assert_allclose(hessian, [(up[1] - down[1]) / 2e-5 for up, down in zip(above, below)], rtol=1e-7)


def test_loss_ps_cal_py_nan_gamma():
    rng = np.random.default_rng(42)
    n_obs = 100
    k_features = 3

    X = np.column_stack([np.ones(n_obs), rng.standard_normal((n_obs, k_features - 1))])
    D = rng.binomial(1, 0.3, n_obs)
    iw = np.ones(n_obs)

    gamma = np.array([0.1, np.nan, -0.3])

    value, gradient, hessian = _loss_ps_cal(gamma, D, X, iw)

    assert value == np.inf
    assert np.all(np.isnan(gradient))
    assert np.all(np.isnan(hessian))


def test_loss_ps_cal_py_shape_validation():
    rng = np.random.default_rng(42)
    n_obs = 100
    k_features = 3

    X = np.column_stack([np.ones(n_obs), rng.standard_normal((n_obs, k_features - 1))])
    D = rng.binomial(1, 0.3, n_obs)
    iw = np.ones(n_obs)

    gamma = np.array([0.1, 0.2])

    with pytest.raises(ValueError, match=re.escape("(size 2 is different from 3)")):
        _loss_ps_cal(gamma, D, X, iw)


def test_loss_ps_ipt_matches_the_tilting_loss_with_its_quadratic_extension():
    X = np.array([[1.0, 0.0], [1.0, 0.5], [1.0, 1.25], [1.0, -1.0]])
    D = np.array([1, 0, 0, 1])
    gamma = np.array([0.0, 1.0])
    knot = np.log(3.0)
    gap = 1.25 - knot
    objective = 0.0 - np.exp(0.5) + (1.25 - knot - 3 - 4 * gap - 1.5 * gap**2) - 1.0
    slopes = np.array([1.0, -np.exp(0.5), -3 - 3 * gap, 1.0])
    curvatures = np.array([0.0, -np.exp(0.5), -3.0, 0.0])

    value, gradient, hessian = _loss_ps_ipt(gamma, D, X, np.ones(4), 4)

    assert isinstance(value, float)
    assert isinstance(gradient, np.ndarray)
    assert isinstance(hessian, np.ndarray)
    np.testing.assert_allclose(value, -objective, rtol=1e-12)
    np.testing.assert_allclose(gradient, -X.T @ slopes, rtol=1e-12)
    np.testing.assert_allclose(hessian, -(X * curvatures[:, None]).T @ X, rtol=1e-12)


def test_loss_ps_ipt_gradient_and_hessian_are_derivatives_of_its_value():
    X = np.column_stack([np.ones(12), np.linspace(-2.0, 3.5, 12)])
    D = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    iw = np.random.default_rng(5).uniform(0.5, 2.0, 12)
    gamma = np.array([0.2, 0.9])
    steps = 1e-5 * np.eye(2)

    value, gradient, hessian = _loss_ps_ipt(gamma, D, X, iw, 12)
    above = [_loss_ps_ipt(gamma + step, D, X, iw, 12) for step in steps]
    below = [_loss_ps_ipt(gamma - step, D, X, iw, 12) for step in steps]

    assert isinstance(value, float)
    assert np.isfinite(value)
    assert isinstance(gradient, np.ndarray)
    assert isinstance(hessian, np.ndarray)
    np.testing.assert_allclose(gradient, [(up[0] - down[0]) / 2e-5 for up, down in zip(above, below)], rtol=1e-7)
    np.testing.assert_allclose(hessian, [(up[1] - down[1]) / 2e-5 for up, down in zip(above, below)], rtol=1e-7)


def test_loss_ps_ipt_py_nan_gamma():
    rng = np.random.default_rng(42)
    n_obs = 100
    k_features = 3

    X = np.column_stack([np.ones(n_obs), rng.standard_normal((n_obs, k_features - 1))])
    D = rng.binomial(1, 0.3, n_obs)
    iw = np.ones(n_obs)

    gamma = np.array([0.1, np.nan, -0.3])

    value, gradient, hessian = _loss_ps_ipt(gamma, D, X, iw, n_obs)

    assert value == np.inf
    assert np.all(np.isnan(gradient))
    assert np.all(np.isnan(hessian))


def test_calculate_pscore_ipt_balances_the_covariates_of_the_two_groups():
    rng = np.random.default_rng(42)
    X = np.column_stack([np.ones(200), rng.standard_normal((200, 2))])
    D = rng.binomial(1, 0.3, 200)
    iw = np.ones(200)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pscore = calculate_pscore_ipt(D, X, iw)
    odds = pscore / (1 - pscore)

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (200,)
    assert np.all((pscore >= 0) & (pscore <= 1))
    np.testing.assert_allclose(X.T @ ((1 - D) * odds), X.T @ D, rtol=1e-6)
    assert not [w for w in caught if issubclass(w.category, UserWarning)]


def test_calculate_pscore_ipt_balances_the_weighted_covariates():
    rng = np.random.default_rng(42)
    X = np.column_stack([np.ones(200), rng.standard_normal((200, 2))])
    D = rng.binomial(1, 0.3, 200)
    iw = rng.exponential(1, 200)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pscore = calculate_pscore_ipt(D, X, iw)
    odds = pscore / (1 - pscore)

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (200,)
    assert np.all((pscore >= 0) & (pscore <= 1))
    np.testing.assert_allclose(X.T @ (iw * (1 - D) * odds), X.T @ (iw * D), rtol=1e-6)
    assert not [w for w in caught if issubclass(w.category, UserWarning)]


@pytest.mark.parametrize("n_controls", [30, 300])
def test_calculate_pscore_ipt_balances_twenty_covariates(n_controls):
    rng = np.random.default_rng(8)
    X = np.column_stack([np.ones(200 + n_controls), 1 + rng.standard_normal((200 + n_controls, 20))])
    D = np.r_[np.ones(200), np.zeros(n_controls)].astype(int)
    iw = rng.uniform(0.5, 2.0, 200 + n_controls)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pscore = calculate_pscore_ipt(D, X, iw)
    odds = pscore / (1 - pscore)

    np.testing.assert_allclose(X.T @ (iw * (1 - D) * odds), X.T @ (iw * D), rtol=1e-6)
    assert not [w for w in caught if issubclass(w.category, UserWarning)]


def test_calculate_pscore_ipt_falls_back_to_the_logit_fit_when_twenty_covariates_cannot_be_balanced():
    rng = np.random.default_rng(8)
    X = np.column_stack([np.ones(225), 1 + rng.standard_normal((225, 20))])
    D = np.r_[np.ones(200), np.zeros(25)].astype(int)
    iw = rng.uniform(0.5, 2.0, 225)
    message = "IPT optimization failed: IPT optimization did not converge. Using initial logit estimates."

    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$"):
        pscore = calculate_pscore_ipt(D, X, iw)

    np.testing.assert_allclose(X.T @ (iw * pscore), X.T @ (iw * D), rtol=1e-8)


def test_calculate_pscore_ipt_shape_validation():
    rng = np.random.default_rng(42)
    n_obs = 100
    k_features = 3

    X = np.column_stack([np.ones(n_obs), rng.standard_normal((n_obs, k_features - 1))])
    D = rng.binomial(1, 0.3, n_obs)
    iw = np.ones(n_obs)

    with pytest.warns(UserWarning):
        pscore_d = calculate_pscore_ipt(D[:-1], X, iw)

    with pytest.warns(UserWarning):
        pscore_iw = calculate_pscore_ipt(D, X, iw[:-1])

    assert isinstance(pscore_d, np.ndarray)
    assert isinstance(pscore_iw, np.ndarray)


def test_calculate_pscore_ipt_drops_a_repeated_covariate_and_still_balances_it():
    rng = np.random.default_rng(42)
    X_base = np.column_stack([np.ones(100), rng.standard_normal(100)])
    X = np.column_stack([X_base, X_base[:, 1] * 2])
    D = rng.binomial(1, 0.3, 100)
    iw = np.ones(100)

    with pytest.warns(UserWarning, match=re.escape("Removed 1 collinear column(s) from covariate matrix.")):
        pscore = calculate_pscore_ipt(D, X, iw)
    odds = pscore / (1 - pscore)

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (100,)
    np.testing.assert_allclose(X.T @ ((1 - D) * odds), X.T @ D, rtol=1e-6)


def test_calculate_pscore_ipt_all_treated():
    rng = np.random.default_rng(42)
    n_obs = 100
    k_features = 3

    X = np.column_stack([np.ones(n_obs), rng.standard_normal((n_obs, k_features - 1))])
    D = np.ones(n_obs)
    iw = np.ones(n_obs)
    message = "Treatment indicator D contains values [1.]. Expected binary values [0, 1]. Results may be unreliable."

    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$"):
        pscore = calculate_pscore_ipt(D, X, iw)

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (n_obs,)
    assert np.all(np.isfinite(pscore))
    assert np.all(pscore >= 0.5)


def test_calculate_pscore_ipt_all_control():
    rng = np.random.default_rng(42)
    n_obs = 100
    k_features = 3

    X = np.column_stack([np.ones(n_obs), rng.standard_normal((n_obs, k_features - 1))])
    D = np.zeros(n_obs)
    iw = np.ones(n_obs)
    message = "Treatment indicator D contains values [0.]. Expected binary values [0, 1]. Results may be unreliable."

    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$"):
        pscore = calculate_pscore_ipt(D, X, iw)

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (n_obs,)
    assert np.all(np.isfinite(pscore))
    assert np.all(pscore < 0.5)


def test_calculate_pscore_ipt_perfect_separation():
    rng = np.random.default_rng(42)
    n_obs = 100

    x1 = rng.standard_normal(n_obs)
    D = (x1 > 0).astype(int)

    X = np.column_stack([np.ones(n_obs), x1, rng.standard_normal((n_obs, 2))])
    iw = np.ones(n_obs)

    with pytest.warns(UserWarning, match="trust-constr optimization failed"):
        pscore = calculate_pscore_ipt(D, X, iw)

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (n_obs,)
    assert not np.any(np.isnan(pscore))
    assert np.all((pscore >= 0) & (pscore <= 1))
    assert pscore[D == 1].min() > pscore[D == 0].max()


def test_calculate_pscore_ipt_falls_back_to_the_ipt_loss_when_trust_constr_fails(monkeypatch):
    rng = np.random.default_rng(42)
    X = np.column_stack([np.ones(200), rng.standard_normal((200, 2))])
    D = rng.binomial(1, 0.3, 200)
    iw = np.ones(200)
    minimize = scipy.optimize.minimize

    def fail_trust_constr(fun, x0, method=None, **kwargs):
        if method == "trust-constr":
            return scipy.optimize.OptimizeResult(x=x0, success=False)
        return minimize(fun, x0, method=method, **kwargs)

    monkeypatch.setattr(scipy.optimize, "minimize", fail_trust_constr)

    with pytest.warns(UserWarning, match="trust-constr optimization failed") as caught:
        pscore = calculate_pscore_ipt(D, X, iw)
    odds = pscore / (1 - pscore)

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (200,)
    np.testing.assert_allclose(X.T @ ((1 - D) * odds), X.T @ D, rtol=1e-6)
    assert not [w for w in caught if "IPT optimization failed" in str(w.message)]


def test_calculate_pscore_ipt_falls_back_to_the_logit_fit_when_both_optimizers_fail(monkeypatch):
    rng = np.random.default_rng(42)
    X = np.column_stack([np.ones(200), rng.standard_normal((200, 2))])
    D = rng.binomial(1, 0.3, 200)
    iw = rng.uniform(0.5, 2.0, 200)

    def fail(fun, x0, **kwargs):
        return scipy.optimize.OptimizeResult(x=x0, success=False)

    monkeypatch.setattr(scipy.optimize, "minimize", fail)

    with pytest.warns(UserWarning, match="Using initial logit estimates") as caught:
        pscore = calculate_pscore_ipt(D, X, iw)

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (200,)
    np.testing.assert_allclose(X.T @ (iw * pscore), X.T @ (iw * D), rtol=1e-8)
    assert not [w for w in caught if "Logit model" in str(w.message)]


def test_calculate_pscore_ipt_non_binary_treatment():
    rng = np.random.default_rng(42)
    n_obs = 100
    k_features = 3

    X = np.column_stack([np.ones(n_obs), rng.standard_normal((n_obs, k_features - 1))])
    D = rng.choice([0, 1, 2], size=n_obs)
    iw = np.ones(n_obs)
    message = "Treatment indicator D contains values [0 1 2]. Expected binary values [0, 1]. Results may be unreliable."

    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$"):
        pscore = calculate_pscore_ipt(D, X, iw)

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (n_obs,)


def test_calculate_pscore_ipt_zero_weights():
    rng = np.random.default_rng(42)
    n_obs = 100
    k_features = 3

    X = np.column_stack([np.ones(n_obs), rng.standard_normal((n_obs, k_features - 1))])
    D = rng.binomial(1, 0.3, n_obs)
    iw = np.zeros(n_obs)
    message = "All individual weights are zero. Propensity scores will be uninformative."

    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$"):
        pscore = calculate_pscore_ipt(D, X, iw)

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (n_obs,)
    assert np.all(np.isnan(pscore))


def test_remove_collinear_columns_keeps_linearly_independent_columns():
    rng = np.random.default_rng(42)
    X = np.column_stack([np.ones(100), rng.standard_normal(100), rng.standard_normal(100)])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        X_reduced, removed = _remove_collinear_columns(X)

    np.testing.assert_array_equal(X_reduced, X)
    assert removed == []
    assert not caught


def test_remove_collinear_columns_keeps_a_covariate_measured_on_a_large_scale():
    rng = np.random.default_rng(42)
    X = np.column_stack([np.ones(100), rng.standard_normal(100), 1e16 * rng.standard_normal(100)])

    X_reduced, removed = _remove_collinear_columns(X)

    np.testing.assert_array_equal(X_reduced, X)
    assert removed == []


def test_remove_collinear_columns_drops_one_of_two_proportional_columns():
    rng = np.random.default_rng(42)
    X_base = np.column_stack([np.ones(100), rng.standard_normal(100)])
    X = np.column_stack([X_base, X_base[:, 1] * 2])

    with pytest.warns(UserWarning, match=re.escape("Removed 1 collinear column(s)")):
        X_reduced, removed = _remove_collinear_columns(X)

    assert len(removed) == 1
    assert removed[0] in (1, 2)
    assert X_reduced.shape == (100, 2)
    assert np.linalg.matrix_rank(X_reduced) == 2
    assert all(any(np.array_equal(column, original) for original in X.T) for column in X_reduced.T)


def test_remove_collinear_columns_drops_every_column_spanned_by_the_others():
    rng = np.random.default_rng(42)
    col1 = rng.standard_normal(100)
    X = np.column_stack([np.ones(100), col1, col1, col1 * 2 + 1])

    with pytest.warns(UserWarning, match=re.escape("Removed 2 collinear column(s)")):
        X_reduced, removed = _remove_collinear_columns(X)

    assert len(removed) == 2
    assert X_reduced.shape == (100, 2)
    assert np.linalg.matrix_rank(X_reduced) == 2
    assert all(any(np.array_equal(column, original) for original in X.T) for column in X_reduced.T)


def test_remove_collinear_columns_single_column():
    X = np.ones((100, 1))

    X_reduced, removed = _remove_collinear_columns(X)

    np.testing.assert_array_equal(X_reduced, X)
    assert removed == []


def test_weighted_quantile_with_equal_weights_interpolates_the_inverted_cdf():
    values = np.array([4.0, 1.0, 5.0, 2.0, 3.0])
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    result = [_weighted_quantile(values, q, np.ones(5)) for q in quantiles]

    np.testing.assert_allclose(result, np.quantile(values, quantiles, method="interpolated_inverted_cdf"), rtol=1e-12)


def test_weighted_quantile_splits_the_weight_at_the_requested_share():
    rng = np.random.default_rng(3)
    values = rng.standard_normal(400)
    weights = np.where(values > 0, 10.0, 1.0) * rng.uniform(0.5, 1.5, 400)
    quantiles = [0.25, 0.5, 0.75]

    thresholds = [_weighted_quantile(values, q, weights) for q in quantiles]
    shares = [np.sum(weights * (values <= threshold)) / np.sum(weights) for threshold in thresholds]
    lumpy_median = _weighted_quantile(np.arange(1.0, 6.0), 0.5, np.array([0.1, 0.1, 0.6, 0.1, 0.1]))

    np.testing.assert_allclose(shares, quantiles, rtol=0, atol=np.max(weights) / np.sum(weights))
    assert 2.5 <= lumpy_median <= 3.5


def test_weighted_quantile_edge():
    values = np.array([1.0, 2.0, 3.0])
    weights = np.ones(3)

    q0 = _weighted_quantile(values, 0.0, weights)
    q100 = _weighted_quantile(values, 1.0, weights)

    assert q0 == 1.0
    assert q100 == 3.0


def test_add_quantile_constraints_appends_an_indicator_below_each_quantile():
    rng = np.random.default_rng(42)
    X = np.column_stack([np.ones(101), rng.standard_normal(101), rng.standard_normal(101)])
    iw = np.ones(101)
    thresholds = np.quantile(X[:, 1], [0.005, 0.25, 0.5, 0.75], method="interpolated_inverted_cdf")

    X_extended = _add_quantile_constraints(X, {1: [0.005, 0.25, 0.5, 0.75]}, iw)

    np.testing.assert_array_equal(X_extended, np.column_stack([X, X[:, [1]] <= thresholds]))


def test_add_quantile_constraints_invalid_column():
    rng = np.random.default_rng(42)
    n_obs = 100
    X = np.column_stack([np.ones(n_obs), rng.standard_normal(n_obs)])
    iw = np.ones(n_obs)

    quantiles = {2: [0.5], 5: [0.5]}

    with pytest.warns(UserWarning, match="exceeds number of columns") as caught:
        X_extended = _add_quantile_constraints(X, quantiles, iw)

    assert [str(w.message) for w in caught] == [
        f"Column index {index} exceeds number of columns (2). Skipping." for index in (2, 5)
    ]
    np.testing.assert_array_equal(X_extended, X)


def test_add_quantile_constraints_invalid_quantile():
    rng = np.random.default_rng(42)
    n_obs = 101
    X = np.column_stack([np.ones(n_obs), rng.standard_normal(n_obs)])
    iw = np.ones(n_obs)
    median = np.quantile(X[:, 1], 0.5, method="interpolated_inverted_cdf")

    quantiles = {1: [0.0, 0.5, 1.0, 1.5]}

    with pytest.warns(UserWarning, match="must be between 0 and 1") as caught:
        X_extended = _add_quantile_constraints(X, quantiles, iw)

    assert [str(w.message) for w in caught] == [
        f"Quantile {q} must be between 0 and 1. Skipping." for q in (0.0, 1.0, 1.5)
    ]
    np.testing.assert_array_equal(X_extended, np.column_stack([X, X[:, 1] <= median]))


def test_add_quantile_constraints_intercept_skip():
    rng = np.random.default_rng(42)
    n_obs = 101
    X = np.column_stack([np.ones(n_obs), rng.standard_normal(n_obs)])
    iw = np.ones(n_obs)
    median = np.quantile(X[:, 1], 0.5, method="interpolated_inverted_cdf")

    quantiles = {0: [0.5], 1: [0.5]}

    with pytest.warns(UserWarning, match="Skipping quantile constraints for intercept"):
        X_extended = _add_quantile_constraints(X, quantiles, iw)

    np.testing.assert_array_equal(X_extended, np.column_stack([X, X[:, 1] <= median]))


def test_add_quantile_constraints_keeps_a_first_column_that_is_not_an_intercept():
    X = np.random.default_rng(42).standard_normal((101, 2))
    median = np.quantile(X[:, 0], 0.5, method="interpolated_inverted_cdf")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        X_extended = _add_quantile_constraints(X, {0: [0.5]}, np.ones(101))

    np.testing.assert_array_equal(X_extended, np.column_stack([X, X[:, 0] <= median]))
    assert not caught


def test_calculate_pscore_ipt_balances_the_requested_quantile_indicators():
    rng = np.random.default_rng(42)
    X = np.column_stack([np.ones(200), rng.standard_normal((200, 2))])
    D = rng.binomial(1, 0.3, 200)
    iw = np.where(X[:, 1] > 0, 4.0, 1.0) * rng.uniform(0.5, 1.5, 200)
    below = [(X[:, [j]] >= X[:, j]) @ iw < q * np.sum(iw) for j, q in ((1, 0.25), (1, 0.75), (2, 0.5))]
    moments = np.column_stack([X, *below])

    pscore = calculate_pscore_ipt(D, X, iw, quantiles={1: [0.25, 0.75], 2: [0.5]})
    odds = pscore / (1 - pscore)

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (200,)
    assert np.all((pscore >= 0) & (pscore <= 1))
    np.testing.assert_allclose(moments.T @ (iw * (1 - D) * odds), moments.T @ (iw * D), rtol=1e-6)


def test_calculate_pscore_ipt_drops_two_dependent_covariates_and_still_balances_them():
    rng = np.random.default_rng(42)
    X_base = np.column_stack([np.ones(100), rng.standard_normal((100, 2))])
    X = np.column_stack([X_base, X_base[:, 1] * 2, X_base[:, 2] - X_base[:, 1]])
    D = rng.binomial(1, 0.3, 100)
    iw = np.ones(100)

    with pytest.warns(UserWarning, match=re.escape("Removed 2 collinear column(s)")):
        pscore = calculate_pscore_ipt(D, X, iw)
    odds = pscore / (1 - pscore)

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (100,)
    assert np.all((pscore >= 0) & (pscore <= 1))
    np.testing.assert_allclose(X.T @ ((1 - D) * odds), X.T @ D, rtol=1e-6)


def test_calculate_pscore_ipt_balances_a_quantile_indicator_after_dropping_a_repeated_covariate():
    rng = np.random.default_rng(42)
    X_base = np.column_stack([np.ones(101), rng.standard_normal(101)])
    X = np.column_stack([X_base, X_base[:, 1] * 2])
    D = rng.binomial(1, 0.3, 101)
    iw = np.ones(101)
    moments = np.column_stack([X, X[:, 1] <= np.quantile(X[:, 1], 0.5, method="interpolated_inverted_cdf")])

    with pytest.warns(UserWarning, match=re.escape("Removed 1 collinear column(s)")):
        pscore = calculate_pscore_ipt(D, X, iw, quantiles={1: [0.5]})
    odds = pscore / (1 - pscore)

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (101,)
    assert np.all((pscore >= 0) & (pscore <= 1))
    np.testing.assert_allclose(moments.T @ ((1 - D) * odds), moments.T @ D, rtol=1e-6)
