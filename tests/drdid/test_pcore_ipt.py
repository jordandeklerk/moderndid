"""Tests for the IPT propensity score calculation and estimation."""

import numpy as np
import pytest
import scipy.optimize

from doublediff.diddr.propensity.pscore_ipt import (
    _add_quantile_constraints,
    _loss_ps_cal,
    _loss_ps_ipt,
    _remove_collinear_columns,
    _weighted_quantile,
    calculate_pscore_ipt,
)


def test_loss_ps_cal_py_basic():
    np.random.seed(42)
    n_obs = 100
    k_features = 3

    X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, k_features - 1)])
    D = np.random.binomial(1, 0.3, n_obs)
    iw = np.ones(n_obs)

    gamma = np.array([0.1, 0.2, -0.3])

    value, gradient, hessian = _loss_ps_cal(gamma, D, X, iw)

    assert isinstance(value, float)
    assert not np.isnan(value)
    assert isinstance(gradient, np.ndarray)
    assert gradient.shape == (k_features,)
    assert isinstance(hessian, np.ndarray)
    assert hessian.shape == (k_features, k_features)


def test_loss_ps_cal_py_with_weights():
    np.random.seed(42)
    n_obs = 100
    k_features = 3

    X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, k_features - 1)])
    D = np.random.binomial(1, 0.3, n_obs)
    iw = np.random.exponential(1, n_obs)

    gamma = np.array([0.1, 0.2, -0.3])

    value, gradient, hessian = _loss_ps_cal(gamma, D, X, iw)

    assert isinstance(value, float)
    assert not np.isnan(value)
    assert isinstance(gradient, np.ndarray)
    assert gradient.shape == (k_features,)
    assert isinstance(hessian, np.ndarray)
    assert hessian.shape == (k_features, k_features)


def test_loss_ps_cal_py_nan_gamma():
    n_obs = 100
    k_features = 3

    X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, k_features - 1)])
    D = np.random.binomial(1, 0.3, n_obs)
    iw = np.ones(n_obs)

    gamma = np.array([0.1, np.nan, -0.3])

    value, gradient, hessian = _loss_ps_cal(gamma, D, X, iw)

    assert value == np.inf
    assert np.all(np.isnan(gradient))
    assert np.all(np.isnan(hessian))


def test_loss_ps_cal_py_shape_validation():
    n_obs = 100
    k_features = 3

    X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, k_features - 1)])
    D = np.random.binomial(1, 0.3, n_obs)
    iw = np.ones(n_obs)

    gamma = np.array([0.1, 0.2])

    with pytest.raises(ValueError):
        _loss_ps_cal(gamma, D, X, iw)


def test_loss_ps_ipt_py_basic():
    np.random.seed(42)
    n_obs = 100
    k_features = 3

    X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, k_features - 1)])
    D = np.random.binomial(1, 0.3, n_obs)
    iw = np.ones(n_obs)

    gamma = np.array([0.1, 0.2, -0.3])

    value, gradient, hessian = _loss_ps_ipt(gamma, D, X, iw, n_obs)

    assert isinstance(value, float)
    assert not np.isnan(value)
    assert isinstance(gradient, np.ndarray)
    assert gradient.shape == (k_features,)
    assert isinstance(hessian, np.ndarray)
    assert hessian.shape == (k_features, k_features)


def test_loss_ps_ipt_py_with_weights():
    np.random.seed(42)
    n_obs = 100
    k_features = 3

    X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, k_features - 1)])
    D = np.random.binomial(1, 0.3, n_obs)
    iw = np.random.exponential(1, n_obs)

    gamma = np.array([0.1, 0.2, -0.3])

    value, gradient, hessian = _loss_ps_ipt(gamma, D, X, iw, n_obs)

    assert isinstance(value, float)
    assert not np.isnan(value)
    assert isinstance(gradient, np.ndarray)
    assert gradient.shape == (k_features,)
    assert isinstance(hessian, np.ndarray)
    assert hessian.shape == (k_features, k_features)


def test_loss_ps_ipt_py_nan_gamma():
    n_obs = 100
    k_features = 3

    X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, k_features - 1)])
    D = np.random.binomial(1, 0.3, n_obs)
    iw = np.ones(n_obs)

    gamma = np.array([0.1, np.nan, -0.3])

    value, gradient, hessian = _loss_ps_ipt(gamma, D, X, iw, n_obs)

    assert value == np.inf
    assert np.all(np.isnan(gradient))
    assert np.all(np.isnan(hessian))


def test_calculate_pscore_ipt_basic():
    np.random.seed(42)
    n_obs = 200
    k_features = 3

    X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, k_features - 1)])
    D = np.random.binomial(1, 0.3, n_obs)
    iw = np.ones(n_obs)

    pscore = calculate_pscore_ipt(D, X, iw)

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (n_obs,)
    assert np.all((pscore >= 0) & (pscore <= 1))


def test_calculate_pscore_ipt_with_weights():
    np.random.seed(42)
    n_obs = 200
    k_features = 3

    X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, k_features - 1)])
    D = np.random.binomial(1, 0.3, n_obs)
    iw = np.random.exponential(1, n_obs)

    pscore = calculate_pscore_ipt(D, X, iw)

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (n_obs,)
    assert np.all((pscore >= 0) & (pscore <= 1))


def test_calculate_pscore_ipt_shape_validation():
    n_obs = 100
    k_features = 3

    X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, k_features - 1)])
    D = np.random.binomial(1, 0.3, n_obs)
    iw = np.ones(n_obs)

    with pytest.warns(UserWarning):
        pscore_d = calculate_pscore_ipt(D[:-1], X, iw)

    with pytest.warns(UserWarning):
        pscore_iw = calculate_pscore_ipt(D, X, iw[:-1])

    assert isinstance(pscore_d, np.ndarray)
    assert isinstance(pscore_iw, np.ndarray)


def test_calculate_pscore_ipt_collinear_covariates():
    np.random.seed(42)
    n_obs = 100

    X_base = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, 1)])
    X = np.column_stack([X_base, X_base[:, 1] * 2])
    D = np.random.binomial(1, 0.3, n_obs)
    iw = np.ones(n_obs)

    with pytest.warns(UserWarning):
        pscore = calculate_pscore_ipt(D, X, iw)

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (n_obs,)


def test_calculate_pscore_ipt_all_treated():
    np.random.seed(42)
    n_obs = 100
    k_features = 3

    X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, k_features - 1)])
    D = np.ones(n_obs)
    iw = np.ones(n_obs)

    with pytest.warns(UserWarning):
        pscore = calculate_pscore_ipt(D, X, iw)

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (n_obs,)
    assert np.all(pscore >= 0.5)


def test_calculate_pscore_ipt_all_control():
    np.random.seed(42)
    n_obs = 100
    k_features = 3

    X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, k_features - 1)])
    D = np.zeros(n_obs)
    iw = np.ones(n_obs)

    with pytest.warns(UserWarning):
        pscore = calculate_pscore_ipt(D, X, iw)

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (n_obs,)
    assert np.all(pscore < 0.5)


def test_calculate_pscore_ipt_perfect_separation():
    np.random.seed(42)
    n_obs = 100

    x1 = np.random.randn(n_obs)
    D = (x1 > 0).astype(int)

    X = np.column_stack([np.ones(n_obs), x1, np.random.randn(n_obs, 2)])
    iw = np.ones(n_obs)

    with pytest.warns(UserWarning):
        pscore = calculate_pscore_ipt(D, X, iw)

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (n_obs,)

    assert not np.any(np.isnan(pscore))
    assert np.all((pscore >= 0) & (pscore <= 1))


def test_calculate_pscore_ipt_trust_constr_fail_ipt_success():
    np.random.seed(42)
    n_obs = 200
    k_features = 3

    X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, k_features - 1)])
    D = np.random.binomial(1, 0.3, n_obs)
    iw = np.ones(n_obs)

    original_minimize = scipy.optimize.minimize

    def mock_minimize(fun, x0, args=(), method="", jac=None, **_):
        if method == "trust-constr":
            result = type("obj", (object,), {"success": False, "x": x0, "message": "Simulated failure"})
            return result
        return original_minimize(fun, x0, args=args, method=method, jac=jac)

    scipy.optimize.minimize = mock_minimize

    with pytest.warns(UserWarning, match="trust-constr optimization failed"):
        pscore = calculate_pscore_ipt(D, X, iw)

    scipy.optimize.minimize = original_minimize

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (n_obs,)


def test_calculate_pscore_ipt_both_methods_fail():
    np.random.seed(42)
    n_obs = 200
    k_features = 3

    X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, k_features - 1)])
    D = np.random.binomial(1, 0.3, n_obs)
    iw = np.ones(n_obs)

    original_minimize = scipy.optimize.minimize

    def mock_minimize(_, x0, **__):
        result = type("obj", (object,), {"success": False, "x": x0, "message": "Simulated failure"})
        return result

    scipy.optimize.minimize = mock_minimize

    with pytest.warns(UserWarning):
        pscore = calculate_pscore_ipt(D, X, iw)

    scipy.optimize.minimize = original_minimize

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (n_obs,)


def test_calculate_pscore_ipt_non_binary_treatment():
    np.random.seed(42)
    n_obs = 100
    k_features = 3

    X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, k_features - 1)])
    D = np.random.choice([0, 1, 2], size=n_obs)
    iw = np.ones(n_obs)

    with pytest.warns(UserWarning):
        pscore = calculate_pscore_ipt(D, X, iw)

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (n_obs,)


def test_calculate_pscore_ipt_zero_weights():
    np.random.seed(42)
    n_obs = 100
    k_features = 3

    X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, k_features - 1)])
    D = np.random.binomial(1, 0.3, n_obs)
    iw = np.zeros(n_obs)

    with pytest.warns(UserWarning):
        pscore = calculate_pscore_ipt(D, X, iw)

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (n_obs,)


def test_remove_collinear_columns():
    n_obs = 100
    X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs), np.random.randn(n_obs)])

    X_reduced, removed = _remove_collinear_columns(X)

    assert X_reduced.shape == (n_obs, 3)
    assert len(removed) == 0


def test_remove_collinear_columns_with_collinearity():
    n_obs = 100
    X_base = np.column_stack([np.ones(n_obs), np.random.randn(n_obs)])
    X = np.column_stack([X_base, X_base[:, 1] * 2])

    with pytest.warns(UserWarning, match="Removed.*collinear"):
        X_reduced, removed = _remove_collinear_columns(X)

    assert X_reduced.shape[1] < X.shape[1]
    assert len(removed) > 0


def test_remove_collinear_columns_perfect_collinearity():
    n_obs = 100
    col1 = np.random.randn(n_obs)
    X = np.column_stack([np.ones(n_obs), col1, col1, col1 * 2 + 1])

    with pytest.warns(UserWarning, match="Removed.*collinear"):
        X_reduced, removed = _remove_collinear_columns(X)

    assert X_reduced.shape[1] <= 3
    assert len(removed) >= 1


def test_remove_collinear_columns_single_column():
    n_obs = 100
    X = np.ones((n_obs, 1))

    X_reduced, removed = _remove_collinear_columns(X)

    assert X_reduced.shape == (n_obs, 1)
    assert len(removed) == 0


def test_weighted_quantile():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.ones(5)

    q25 = _weighted_quantile(values, 0.25, weights)
    q50 = _weighted_quantile(values, 0.50, weights)
    q75 = _weighted_quantile(values, 0.75, weights)

    assert 1.0 <= q25 <= 2.5
    assert 2.0 <= q50 <= 4.0
    assert 3.5 <= q75 <= 5.0


def test_weighted_quantile_with_weights():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.array([0.1, 0.1, 0.6, 0.1, 0.1])

    q50 = _weighted_quantile(values, 0.50, weights)

    assert 2.5 <= q50 <= 3.5


def test_weighted_quantile_edge():
    values = np.array([1.0, 2.0, 3.0])
    weights = np.ones(3)

    q0 = _weighted_quantile(values, 0.0, weights)
    q100 = _weighted_quantile(values, 1.0, weights)

    assert q0 == 1.0
    assert q100 == 3.0


def test_add_quantile_constraints():
    np.random.seed(42)
    n_obs = 100
    X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs), np.random.randn(n_obs)])
    iw = np.ones(n_obs)

    quantiles = {1: [0.25, 0.5, 0.75]}

    X_extended = _add_quantile_constraints(X, quantiles, iw)

    assert X_extended.shape == (n_obs, 6)


def test_add_quantile_constraints_invalid_column():
    n_obs = 100
    X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs)])
    iw = np.ones(n_obs)

    quantiles = {5: [0.5]}

    with pytest.warns(UserWarning, match="exceeds number of columns"):
        X_extended = _add_quantile_constraints(X, quantiles, iw)

    assert X_extended.shape == X.shape


def test_add_quantile_constraints_invalid_quantile():
    n_obs = 100
    X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs)])
    iw = np.ones(n_obs)

    quantiles = {1: [0.0, 0.5, 1.5]}

    with pytest.warns(UserWarning, match="must be between 0 and 1"):
        X_extended = _add_quantile_constraints(X, quantiles, iw)

    assert X_extended.shape == (n_obs, 3)


def test_add_quantile_constraints_intercept_skip():
    n_obs = 100
    X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs)])
    iw = np.ones(n_obs)

    quantiles = {0: [0.5], 1: [0.5]}

    with pytest.warns(UserWarning, match="Skipping quantile constraints for intercept"):
        X_extended = _add_quantile_constraints(X, quantiles, iw)

    assert X_extended.shape == (n_obs, 3)


def test_calculate_pscore_ipt_with_quantiles():
    np.random.seed(42)
    n_obs = 200
    k_features = 3

    X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, k_features - 1)])
    D = np.random.binomial(1, 0.3, n_obs)
    iw = np.ones(n_obs)

    quantiles = {1: [0.25, 0.75], 2: [0.5]}

    pscore = calculate_pscore_ipt(D, X, iw, quantiles=quantiles)

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (n_obs,)
    assert np.all((pscore >= 0) & (pscore <= 1))


def test_calculate_pscore_ipt_automatic_collinearity_removal():
    np.random.seed(42)
    n_obs = 100

    X_base = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, 2)])
    X = np.column_stack([X_base, X_base[:, 1] * 2, X_base[:, 2] - X_base[:, 1]])
    D = np.random.binomial(1, 0.3, n_obs)
    iw = np.ones(n_obs)

    with pytest.warns(UserWarning, match="Removed.*collinear"):
        pscore = calculate_pscore_ipt(D, X, iw)

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (n_obs,)
    assert np.all((pscore >= 0) & (pscore <= 1))


def test_calculate_pscore_ipt_collinearity_and_quantiles():
    np.random.seed(42)
    n_obs = 100

    X_base = np.column_stack([np.ones(n_obs), np.random.randn(n_obs)])
    X = np.column_stack([X_base, X_base[:, 1] * 2])
    D = np.random.binomial(1, 0.3, n_obs)
    iw = np.ones(n_obs)

    quantiles = {1: [0.5]}

    with pytest.warns(UserWarning, match="Removed.*collinear"):
        pscore = calculate_pscore_ipt(D, X, iw, quantiles=quantiles)

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (n_obs,)
    assert np.all((pscore >= 0) & (pscore <= 1))
