"""Tests for IPW estimators."""

import numpy as np
import pytest

from pydid.drdid.bootstrap_ipw_rc import wboot_ipw_rc
from pydid.drdid.ipw_estimators import ipw_did_rc


def test_ipw_did_rc_basic():
    rng = np.random.RandomState(42)
    n = 100
    y = rng.normal(1, 1, n)
    post = rng.binomial(1, 0.5, n)
    d = rng.binomial(1, 0.3, n)
    ps = 0.3 * np.ones(n)
    i_weights = np.ones(n)

    result = ipw_did_rc(y, post, d, ps, i_weights)
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_ipw_did_rc_with_trimming():
    rng = np.random.RandomState(42)
    n = 100
    y = rng.normal(1, 1, n)
    post = rng.binomial(1, 0.5, n)
    d = rng.binomial(1, 0.3, n)
    ps = rng.uniform(0.1, 0.9, n)
    i_weights = np.ones(n)

    trim_ps = np.ones(n, dtype=bool)
    trim_ps[d == 0] = ps[d == 0] < 0.8

    result = ipw_did_rc(y, post, d, ps, i_weights, trim_ps)
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_ipw_did_rc_all_treated():
    n = 50
    y = np.ones(n)
    post = np.random.binomial(1, 0.5, n)
    d = np.ones(n)
    ps = 0.5 * np.ones(n)
    i_weights = np.ones(n)

    result = ipw_did_rc(y, post, d, ps, i_weights)
    assert np.isfinite(result)


def test_ipw_did_rc_no_treated():
    n = 50
    y = np.ones(n)
    post = np.random.binomial(1, 0.5, n)
    d = np.zeros(n)  # No treated units
    ps = 0.5 * np.ones(n)
    i_weights = np.ones(n)

    with pytest.warns(UserWarning, match="No treated units found"):
        result = ipw_did_rc(y, post, d, ps, i_weights)
    assert np.isnan(result)


def test_ipw_did_rc_extreme_propensity():
    n = 50
    y = np.ones(n)
    post = np.random.binomial(1, 0.5, n)
    d = np.zeros(n)
    d[:10] = 1
    ps = np.ones(n)
    ps[d == 1] = 0.5
    i_weights = np.ones(n)

    with pytest.warns(UserWarning, match="Propensity score is 1 for some control units"):
        result = ipw_did_rc(y, post, d, ps, i_weights)
    assert np.isnan(result)


def test_ipw_did_rc_lambda_edge_cases():
    n = 50
    y = np.ones(n)
    d = np.random.binomial(1, 0.3, n)
    ps = 0.3 * np.ones(n)
    i_weights = np.ones(n)

    post = np.zeros(n)
    with pytest.warns(UserWarning, match="Lambda is 0"):
        result = ipw_did_rc(y, post, d, ps, i_weights)
    assert np.isnan(result)

    post = np.ones(n)
    with pytest.warns(UserWarning, match="Lambda is 1"):
        result = ipw_did_rc(y, post, d, ps, i_weights)
    assert np.isnan(result)


def test_ipw_did_rc_invalid_inputs():
    n = 50
    y = np.ones(n)
    post = np.random.binomial(1, 0.5, n)
    d = np.random.binomial(1, 0.3, n)
    ps = 0.3 * np.ones(n)
    i_weights = np.ones(n)

    with pytest.raises(TypeError, match="All inputs must be NumPy arrays"):
        ipw_did_rc(list(y), post, d, ps, i_weights)

    with pytest.raises(ValueError, match="All input arrays must be 1-dimensional"):
        ipw_did_rc(y.reshape(-1, 1), post, d, ps, i_weights)

    with pytest.raises(ValueError, match="All input arrays must have the same shape"):
        ipw_did_rc(y[:-1], post, d, ps, i_weights)


def test_wboot_ipw_rc_basic():
    rng = np.random.RandomState(42)
    n = 100
    y = rng.normal(1, 1, n)
    post = rng.binomial(1, 0.5, n)
    d = rng.binomial(1, 0.3, n)
    x = rng.normal(0, 1, (n, 3))
    x[:, 0] = 1
    i_weights = np.ones(n)

    bootstrap_estimates = wboot_ipw_rc(y, post, d, x, i_weights, n_bootstrap=50, random_state=42)

    assert bootstrap_estimates.shape == (50,)
    assert np.sum(np.isfinite(bootstrap_estimates)) > 45


def test_wboot_ipw_rc_convergence():
    rng = np.random.RandomState(123)
    n = 200
    true_effect = 0.5

    x = rng.normal(0, 1, (n, 2))
    x[:, 0] = 1
    ps_true = 1 / (1 + np.exp(-0.5 * x[:, 1]))
    d = rng.binomial(1, ps_true, n)
    post = rng.binomial(1, 0.5, n)

    y = 1 + 0.5 * x[:, 1] + true_effect * d * post + rng.normal(0, 0.5, n)
    i_weights = np.ones(n)

    bootstrap_estimates = wboot_ipw_rc(y, post, d, x, i_weights, n_bootstrap=100, random_state=123)

    mean_estimate = np.nanmean(bootstrap_estimates)
    assert np.abs(mean_estimate - true_effect) < 0.7
