"""Tests for the improved and locally efficient DR-DiD estimator for repeated cross-section data."""

import numpy as np
import pytest

from pydid import drdid_imp_local_rc


@pytest.mark.parametrize("covariates", [None, "with_covariates"])
def test_drdid_imp_local_rc_basic(covariates):
    n_units = 200
    rng = np.random.RandomState(42)

    post = rng.binomial(1, 0.5, n_units)
    d = rng.binomial(1, 0.5, n_units)

    if covariates == "with_covariates":
        x = rng.randn(n_units, 3)
        x = np.column_stack([np.ones(n_units), x])
    else:
        x = None

    y = rng.randn(n_units) + 2 * d * post

    result = drdid_imp_local_rc(y=y, post=post, d=d, covariates=x)

    assert isinstance(result.att, float)
    assert isinstance(result.se, float)
    assert result.se > 0
    assert result.lci < result.uci
    assert result.boots is None
    assert result.att_inf_func is None


def test_drdid_imp_local_rc_with_weights():
    n_units = 200
    rng = np.random.RandomState(42)

    post = rng.binomial(1, 0.5, n_units)
    d = rng.binomial(1, 0.5, n_units)
    x = np.column_stack([np.ones(n_units), rng.randn(n_units, 2)])
    y = rng.randn(n_units) + 2 * d * post
    i_weights = rng.uniform(0.5, 1.5, n_units)

    result = drdid_imp_local_rc(y=y, post=post, d=d, covariates=x, i_weights=i_weights)

    assert isinstance(result.att, float)
    assert result.se > 0


def test_drdid_imp_local_rc_influence_function():
    n_units = 200
    rng = np.random.RandomState(42)

    post = rng.binomial(1, 0.5, n_units)
    d = rng.binomial(1, 0.5, n_units)
    x = np.column_stack([np.ones(n_units), rng.randn(n_units, 2)])
    y = rng.randn(n_units) + 2 * d * post

    result = drdid_imp_local_rc(y=y, post=post, d=d, covariates=x, influence_func=True)

    assert result.att_inf_func is not None
    assert len(result.att_inf_func) == n_units
    assert np.abs(np.mean(result.att_inf_func)) < 0.1


@pytest.mark.parametrize("boot_type", ["weighted", "multiplier"])
def test_drdid_imp_local_rc_bootstrap(boot_type):
    n_units = 100
    rng = np.random.RandomState(42)

    post = rng.binomial(1, 0.5, n_units)
    d = rng.binomial(1, 0.5, n_units)
    x = np.column_stack([np.ones(n_units), rng.randn(n_units, 2)])
    y = rng.randn(n_units) + 2 * d * post

    result = drdid_imp_local_rc(y=y, post=post, d=d, covariates=x, boot=True, boot_type=boot_type, nboot=100)

    assert result.boots is not None
    assert len(result.boots) == 100
    assert result.se > 0
    assert result.lci < result.uci


def test_drdid_imp_local_rc_no_treated_in_post():
    n_units = 100
    rng = np.random.RandomState(42)

    post = rng.binomial(1, 0.5, n_units)
    d = np.zeros(n_units)
    x = np.column_stack([np.ones(n_units), rng.randn(n_units, 2)])
    y = rng.randn(n_units)

    with pytest.raises(ValueError, match="No units found"):
        drdid_imp_local_rc(y=y, post=post, d=d, covariates=x)


def test_drdid_imp_local_rc_all_treated():
    n_units = 100
    rng = np.random.RandomState(42)

    post = rng.binomial(1, 0.5, n_units)
    d = np.ones(n_units)
    x = np.column_stack([np.ones(n_units), rng.randn(n_units, 2)])
    y = rng.randn(n_units) + 2 * post

    with pytest.raises(ValueError, match="No units found"):
        drdid_imp_local_rc(y=y, post=post, d=d, covariates=x)


def test_drdid_imp_local_rc_trimming():
    n_units = 200
    rng = np.random.RandomState(42)

    post = rng.binomial(1, 0.5, n_units)
    d = rng.binomial(1, 0.05, n_units)
    x = np.column_stack([np.ones(n_units), rng.randn(n_units, 2)])
    y = rng.randn(n_units) + 2 * d * post

    result = drdid_imp_local_rc(y=y, post=post, d=d, covariates=x, trim_level=0.9)

    assert isinstance(result.att, float)
    assert result.se > 0


def test_drdid_imp_local_rc_negative_weights():
    n_units = 100
    rng = np.random.RandomState(42)

    post = rng.binomial(1, 0.5, n_units)
    d = rng.binomial(1, 0.5, n_units)
    x = np.column_stack([np.ones(n_units), rng.randn(n_units, 2)])
    y = rng.randn(n_units) + 2 * d * post
    i_weights = rng.randn(n_units)

    with pytest.raises(ValueError, match="non-negative"):
        drdid_imp_local_rc(y=y, post=post, d=d, covariates=x, i_weights=i_weights)


def test_drdid_imp_local_rc_dgp():
    n_units = 2000
    rng = np.random.RandomState(42)

    x1 = rng.normal(0, 1, n_units)
    x2 = rng.normal(0, 1, n_units)
    x3 = rng.normal(0, 1, n_units)
    x4 = rng.normal(0, 1, n_units)

    post = rng.binomial(1, 0.5, n_units)

    d_propensity = 1 / (1 + np.exp(-(x1 + x3)))
    d = (rng.uniform(size=n_units) < d_propensity).astype(int)

    att_true = 1

    y = 1 + 2 * x1 + 3 * x2 + 4 * x3 + 5 * x4 + post + d * post * att_true + rng.normal(0, 1, n_units)

    covariates = np.column_stack((np.ones(n_units), x1, x2, x3, x4))

    result = drdid_imp_local_rc(y=y, post=post, d=d, covariates=covariates)

    assert 0.8 < result.att < 1.2
    assert result.se > 0
    assert result.args["estMethod"] == "imp2"


def test_drdid_imp_local_rc_no_post_period():
    n_units = 100
    rng = np.random.RandomState(42)

    post = np.zeros(n_units)
    d = rng.binomial(1, 0.5, n_units)
    x = np.column_stack([np.ones(n_units), rng.randn(n_units, 2)])
    y = rng.randn(n_units)

    with pytest.raises(ValueError, match="No units found"):
        drdid_imp_local_rc(y=y, post=post, d=d, covariates=x)


def test_drdid_imp_local_rc_no_pre_period():
    n_units = 100
    rng = np.random.RandomState(42)

    post = np.ones(n_units)
    d = rng.binomial(1, 0.5, n_units)
    x = np.column_stack([np.ones(n_units), rng.randn(n_units, 2)])
    y = rng.randn(n_units) + 2 * d

    with pytest.raises(ValueError, match="No units found"):
        drdid_imp_local_rc(y=y, post=post, d=d, covariates=x)


def test_drdid_imp_local_rc_heterogeneous_effects():
    n_units = 2000
    rng = np.random.RandomState(42)

    x1 = rng.normal(0, 1, n_units)
    x2 = rng.normal(0, 1, n_units)

    post = rng.binomial(1, 0.5, n_units)
    d = rng.binomial(1, 0.5, n_units)

    treatment_effect = 1 + 0.5 * x1
    y = x1 + x2 + post + d * post * treatment_effect + rng.normal(0, 0.5, n_units)

    covariates = np.column_stack((np.ones(n_units), x1, x2))

    result = drdid_imp_local_rc(y=y, post=post, d=d, covariates=covariates)

    expected_att = np.mean(treatment_effect[d == 1])
    assert abs(result.att - expected_att) < 0.2
    assert result.se > 0


def test_drdid_imp_local_rc_small_sample():
    n_units = 50
    rng = np.random.RandomState(42)

    post = rng.binomial(1, 0.5, n_units)
    d = rng.binomial(1, 0.5, n_units)
    y = rng.randn(n_units) + d * post
    x = np.column_stack([np.ones(n_units), rng.randn(n_units, 2)])

    result = drdid_imp_local_rc(y=y, post=post, d=d, covariates=x)

    assert isinstance(result.att, float)
    assert result.se > 0
