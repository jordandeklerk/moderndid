"""Tests for outcome regression DiD estimator with repeated cross-sections."""

import numpy as np
import pytest

from causaldid.drdid import reg_did_rc


def test_reg_did_rc_basic():
    np.random.seed(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
    d = np.random.binomial(1, 0.3, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5, -0.3] + 2 * d * post + np.random.randn(n)

    result = reg_did_rc(y=y, post=post, d=d, covariates=x)

    assert isinstance(result.att, float)
    assert isinstance(result.se, float)
    assert result.se > 0
    assert result.lci < result.att < result.uci
    assert result.boots is None
    assert result.att_inf_func is None


def test_reg_did_rc_with_weights():
    np.random.seed(42)
    n = 200
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + np.random.randn(n)
    weights = np.random.exponential(1, n)

    result = reg_did_rc(y=y, post=post, d=d, covariates=x, i_weights=weights)

    assert isinstance(result.att, float)
    assert result.se > 0


def test_reg_did_rc_with_influence_func():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + np.random.randn(n)

    result = reg_did_rc(y=y, post=post, d=d, covariates=x, influence_func=True)

    assert result.att_inf_func is not None
    assert len(result.att_inf_func) == n
    assert np.isclose(np.mean(result.att_inf_func), 0, atol=1e-10)


def test_reg_did_rc_no_covariates():
    np.random.seed(42)
    n = 200
    d = np.random.binomial(1, 0.3, n)
    post = np.random.binomial(1, 0.5, n)
    y = 1 + 2 * d * post + np.random.randn(n)

    result = reg_did_rc(y=y, post=post, d=d, covariates=None)

    assert isinstance(result.att, float)
    assert result.se > 0


def test_reg_did_rc_bootstrap_weighted():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + np.random.randn(n)

    result = reg_did_rc(y=y, post=post, d=d, covariates=x, boot=True, boot_type="weighted", nboot=50)

    assert result.boots is not None
    assert len(result.boots) == 50
    assert not np.all(np.isnan(result.boots))


def test_reg_did_rc_bootstrap_multiplier():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + np.random.randn(n)

    result = reg_did_rc(y=y, post=post, d=d, covariates=x, boot=True, boot_type="multiplier", nboot=50)

    assert result.boots is not None
    assert len(result.boots) == 50


def test_reg_did_rc_invalid_inputs():
    n = 50
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)

    with pytest.raises(ValueError, match="i_weights must be non-negative"):
        reg_did_rc(y=y, post=post, d=d, covariates=x, i_weights=-np.ones(n))


def test_reg_did_rc_edge_cases():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)

    d_all_treated = np.ones(n, dtype=int)
    post = np.random.binomial(1, 0.5, n)
    with pytest.raises(ValueError, match="No control units in pre-treatment period"):
        reg_did_rc(y=y, post=post, d=d_all_treated, covariates=x)

    d_all_control = np.zeros(n, dtype=int)
    result = reg_did_rc(y=y, post=post, d=d_all_control, covariates=x)
    assert np.isnan(result.att)


def test_reg_did_rc_no_pre_control():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.zeros(n, dtype=int)
    d[50:] = 1
    post = np.ones(n, dtype=int)
    y = x @ [1, 0.5] + 2 * d * post + np.random.randn(n)

    with pytest.raises(ValueError, match="No control units in pre-treatment period"):
        reg_did_rc(y=y, post=post, d=d, covariates=x)


def test_reg_did_rc_no_post_control():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.zeros(n, dtype=int)
    d[50:] = 1
    post = np.zeros(n, dtype=int)
    y = x @ [1, 0.5] + np.random.randn(n)

    with pytest.raises(ValueError, match="No control units in post-treatment period"):
        reg_did_rc(y=y, post=post, d=d, covariates=x)


def test_reg_did_rc_insufficient_control_pre():
    np.random.seed(42)
    n = 10
    x = np.column_stack([np.ones(n), np.random.randn(n, 3)])
    d = np.ones(n, dtype=int)
    d[0:2] = 0
    post = np.zeros(n, dtype=int)
    post[5:] = 1
    y = np.random.randn(n)

    with pytest.raises(ValueError, match="Insufficient control units in pre-treatment period for regression"):
        reg_did_rc(y=y, post=post, d=d, covariates=x)


def test_reg_did_rc_insufficient_control_post():
    np.random.seed(42)
    n = 10
    x = np.column_stack([np.ones(n), np.random.randn(n, 3)])
    d = np.ones(n, dtype=int)
    d[0:6] = 0
    post = np.zeros(n, dtype=int)
    post[4:] = 1
    y = np.random.randn(n)

    with pytest.raises(ValueError, match="Insufficient control units in post-treatment period for regression"):
        reg_did_rc(y=y, post=post, d=d, covariates=x)


def test_reg_did_rc_args_output():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + np.random.randn(n)

    result = reg_did_rc(y=y, post=post, d=d, covariates=x, boot=True, nboot=50)

    assert result.args["panel"] is False
    assert result.args["boot"] is True
    assert result.args["boot_type"] == "weighted"
    assert result.args["nboot"] == 50
    assert result.args["type"] == "or"


def test_reg_did_rc_reproducibility():
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + np.random.randn(n)

    np.random.seed(42)
    result1 = reg_did_rc(y=y, post=post, d=d, covariates=x)

    np.random.seed(42)
    result2 = reg_did_rc(y=y, post=post, d=d, covariates=x)

    assert result1.att == result2.att
    assert result1.se == result2.se


def test_reg_did_rc_multicollinear_covariates():
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    x2 = 2 * x1
    x = np.column_stack([np.ones(n), x1, x2])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = 1 + 0.5 * x1 + 2 * d * post + np.random.randn(n)

    with pytest.raises(ValueError, match="singular"):
        reg_did_rc(y=y, post=post, d=d, covariates=x)


def test_reg_did_rc_near_singular_gram():
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    x2 = x1 + 1e-10 * np.random.randn(n)
    x = np.column_stack([np.ones(n), x1, x2])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = 1 + 0.5 * x1 + 2 * d * post + np.random.randn(n)

    with pytest.raises(ValueError, match="singular"):
        reg_did_rc(y=y, post=post, d=d, covariates=x)


def test_reg_did_rc_1d_covariates():
    np.random.seed(42)
    n = 100
    x = np.random.randn(n)
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = 1 + 0.5 * x + 2 * d * post + np.random.randn(n)

    result = reg_did_rc(y=y, post=post, d=d, covariates=x)

    assert isinstance(result.att, float)
    assert result.se > 0
