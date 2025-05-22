import numpy as np
import pytest
import scipy.optimize

from pydid.drdid.pscore_ipt import _loss_ps_cal_py, _loss_ps_ipt_py, calculate_pscore_ipt


def test_loss_ps_cal_py_basic():
    np.random.seed(42)
    n_obs = 100
    k_features = 3

    X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, k_features - 1)])
    D = np.random.binomial(1, 0.3, n_obs)
    iw = np.ones(n_obs)

    gamma = np.array([0.1, 0.2, -0.3])

    value, gradient, hessian = _loss_ps_cal_py(gamma, D, X, iw)

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

    value, gradient, hessian = _loss_ps_cal_py(gamma, D, X, iw)

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

    value, gradient, hessian = _loss_ps_cal_py(gamma, D, X, iw)

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
        _loss_ps_cal_py(gamma, D, X, iw)


def test_loss_ps_ipt_py_basic():
    np.random.seed(42)
    n_obs = 100
    k_features = 3

    X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, k_features - 1)])
    D = np.random.binomial(1, 0.3, n_obs)
    iw = np.ones(n_obs)

    gamma = np.array([0.1, 0.2, -0.3])

    value, gradient, hessian = _loss_ps_ipt_py(gamma, D, X, iw, n_obs)

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

    value, gradient, hessian = _loss_ps_ipt_py(gamma, D, X, iw, n_obs)

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

    value, gradient, hessian = _loss_ps_ipt_py(gamma, D, X, iw, n_obs)

    assert value == np.inf
    assert np.all(np.isnan(gradient))
    assert np.all(np.isnan(hessian))


def test_loss_ps_ipt_py_small_sample():
    n_obs = 1
    k_features = 3

    X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, k_features - 1)])
    D = np.array([1])
    iw = np.ones(n_obs)

    gamma = np.array([0.1, 0.2, -0.3])

    value, gradient, hessian = _loss_ps_ipt_py(gamma, D, X, iw, n_obs)

    assert isinstance(value, float)
    assert not np.isnan(value)
    assert isinstance(gradient, np.ndarray)
    assert gradient.shape == (k_features,)
    assert isinstance(hessian, np.ndarray)
    assert hessian.shape == (k_features, k_features)


def test_calculate_pscore_ipt_basic():
    np.random.seed(42)
    n_obs = 200
    k_features = 3

    X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, k_features - 1)])
    D = np.random.binomial(1, 0.3, n_obs)
    iw = np.ones(n_obs)

    pscore, flag = calculate_pscore_ipt(D, X, iw)

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (n_obs,)
    assert np.all((pscore >= 0) & (pscore <= 1))
    assert isinstance(flag, int)
    assert flag in [0, 1, 2]


def test_calculate_pscore_ipt_with_weights():
    np.random.seed(42)
    n_obs = 200
    k_features = 3

    X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, k_features - 1)])
    D = np.random.binomial(1, 0.3, n_obs)
    iw = np.random.exponential(1, n_obs)

    pscore, flag = calculate_pscore_ipt(D, X, iw)

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (n_obs,)
    assert np.all((pscore >= 0) & (pscore <= 1))
    assert isinstance(flag, int)
    assert flag in [0, 1, 2]


def test_calculate_pscore_ipt_shape_validation():
    n_obs = 100
    k_features = 3

    X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, k_features - 1)])
    D = np.random.binomial(1, 0.3, n_obs)
    iw = np.ones(n_obs)

    with pytest.warns(UserWarning):
        pscore_d, _ = calculate_pscore_ipt(D[:-1], X, iw)

    with pytest.warns(UserWarning):
        pscore_iw, _ = calculate_pscore_ipt(D, X, iw[:-1])

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
        pscore, flag = calculate_pscore_ipt(D, X, iw)
        assert flag in [0, 1, 2]

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
        pscore, flag = calculate_pscore_ipt(D, X, iw)
        assert flag in [0, 1, 2]

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
        pscore, flag = calculate_pscore_ipt(D, X, iw)
        assert flag in [0, 1, 2]

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (n_obs,)
    assert np.all(pscore < 0.5)


def test_calculate_pscore_ipt_small_sample():
    np.random.seed(42)
    n_obs = 10
    k_features = 3

    X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, k_features - 1)])
    D = np.random.binomial(1, 0.3, n_obs)
    iw = np.ones(n_obs)

    with pytest.warns(UserWarning):
        pscore, flag = calculate_pscore_ipt(D, X, iw)
        assert flag in [0, 1, 2]

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (n_obs,)


def test_calculate_pscore_ipt_perfect_separation():
    np.random.seed(42)
    n_obs = 100

    x1 = np.random.randn(n_obs)
    D = (x1 > 0).astype(int)

    X = np.column_stack([np.ones(n_obs), x1, np.random.randn(n_obs, 2)])
    iw = np.ones(n_obs)

    with pytest.warns(UserWarning):
        pscore, flag = calculate_pscore_ipt(D, X, iw)
        assert flag in [0, 1, 2]

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (n_obs,)

    assert not np.any(np.isnan(pscore))
    assert np.all((pscore >= 0) & (pscore <= 1))


def test_calculate_pscore_ipt_reproducibility():
    np.random.seed(42)
    n_obs = 200
    k_features = 3

    X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, k_features - 1)])
    D = np.random.binomial(1, 0.3, n_obs)
    iw = np.ones(n_obs)

    pscore1, flag1 = calculate_pscore_ipt(D, X, iw)
    pscore2, flag2 = calculate_pscore_ipt(D, X, iw)

    np.testing.assert_array_equal(pscore1, pscore2)
    assert flag1 == flag2


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

    with pytest.warns(UserWarning, match="trust-constr algorithm for loss_ps_cal did not converge"):
        pscore, flag = calculate_pscore_ipt(D, X, iw)

    scipy.optimize.minimize = original_minimize

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (n_obs,)
    assert flag == 1


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
        pscore, flag = calculate_pscore_ipt(D, X, iw)
        assert flag in [0, 1, 2]

    scipy.optimize.minimize = original_minimize

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (n_obs,)
    assert flag == 2


def test_calculate_pscore_ipt_edge_case_one_observation():
    np.random.seed(42)
    n_obs = 1
    k_features = 3

    X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, k_features - 1)])
    D = np.array([1])
    iw = np.ones(n_obs)

    with pytest.warns(UserWarning):
        pscore, flag = calculate_pscore_ipt(D, X, iw)
        assert flag in [0, 1, 2]

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
        pscore, flag = calculate_pscore_ipt(D, X, iw)
        assert flag in [0, 1, 2]

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
        pscore, flag = calculate_pscore_ipt(D, X, iw)

    assert isinstance(pscore, np.ndarray)
    assert pscore.shape == (n_obs,)
    assert flag in [0, 1, 2]
