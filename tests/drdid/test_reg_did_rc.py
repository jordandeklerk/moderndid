"""Tests for outcome regression DiD estimator with repeated cross-sections."""

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from moderndid.drdid import reg_did_rc


def test_reg_did_rc_basic():
    rng = np.random.default_rng(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    d = rng.binomial(1, 0.3, n)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5, -0.3] + 2 * d * post + rng.standard_normal(n)

    result = reg_did_rc(y=y, post=post, d=d, covariates=x)

    assert isinstance(result.att, float)
    assert isinstance(result.se, float)
    assert result.se > 0
    assert result.lci < result.att < result.uci
    assert result.boots is None
    assert result.att_inf_func is None


def test_reg_did_rc_with_weights():
    rng = np.random.default_rng(42)
    n = 200
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + rng.standard_normal(n)
    weights = rng.exponential(1, n)

    result = reg_did_rc(y=y, post=post, d=d, covariates=x, i_weights=weights)

    assert isinstance(result.att, float)
    assert result.se > 0


def test_reg_did_rc_with_influence_func():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + rng.standard_normal(n)

    result = reg_did_rc(y=y, post=post, d=d, covariates=x, influence_func=True)

    assert result.att_inf_func is not None
    assert len(result.att_inf_func) == n
    assert np.isclose(np.mean(result.att_inf_func), 0, atol=1e-10)


def test_reg_did_rc_no_covariates():
    rng = np.random.default_rng(42)
    n = 200
    d = rng.binomial(1, 0.3, n)
    post = rng.binomial(1, 0.5, n)
    y = 1 + 2 * d * post + rng.standard_normal(n)

    result = reg_did_rc(y=y, post=post, d=d, covariates=None)

    assert isinstance(result.att, float)
    assert result.se > 0


def test_reg_did_rc_bootstrap_weighted():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + rng.standard_normal(n)

    result = reg_did_rc(y=y, post=post, d=d, covariates=x, boot=True, boot_type="weighted", nboot=10)

    assert result.boots is not None
    assert len(result.boots) == 10
    assert not np.all(np.isnan(result.boots))


def test_reg_did_rc_bootstrap_multiplier():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + rng.standard_normal(n)

    result = reg_did_rc(y=y, post=post, d=d, covariates=x, boot=True, boot_type="multiplier", nboot=10)

    assert result.boots is not None
    assert len(result.boots) == 10


def test_reg_did_rc_invalid_inputs():
    rng = np.random.default_rng(42)
    n = 50
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = rng.standard_normal(n)
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)

    with pytest.raises(ValueError, match="i_weights must be non-negative"):
        reg_did_rc(y=y, post=post, d=d, covariates=x, i_weights=-np.ones(n))


def test_reg_did_rc_edge_cases():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = rng.standard_normal(n)

    d_all_treated = np.ones(n, dtype=int)
    post = rng.binomial(1, 0.5, n)
    with pytest.raises(ValueError, match="No control units in pre-treatment period"):
        reg_did_rc(y=y, post=post, d=d_all_treated, covariates=x)

    d_all_control = np.zeros(n, dtype=int)
    result = reg_did_rc(y=y, post=post, d=d_all_control, covariates=x)
    assert np.isnan(result.att)


def test_reg_did_rc_no_pre_control():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = np.zeros(n, dtype=int)
    d[50:] = 1
    post = np.ones(n, dtype=int)
    y = x @ [1, 0.5] + 2 * d * post + rng.standard_normal(n)

    with pytest.raises(ValueError, match="No control units in pre-treatment period"):
        reg_did_rc(y=y, post=post, d=d, covariates=x)


def test_reg_did_rc_no_post_control():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = np.zeros(n, dtype=int)
    d[50:] = 1
    post = np.zeros(n, dtype=int)
    y = x @ [1, 0.5] + rng.standard_normal(n)

    with pytest.raises(ValueError, match="No control units in post-treatment period"):
        reg_did_rc(y=y, post=post, d=d, covariates=x)


def test_reg_did_rc_insufficient_control_pre():
    rng = np.random.default_rng(42)
    n = 10
    x = np.column_stack([np.ones(n), rng.standard_normal((n, 3))])
    d = np.ones(n, dtype=int)
    d[0:2] = 0
    post = np.zeros(n, dtype=int)
    post[5:] = 1
    y = rng.standard_normal(n)

    with pytest.raises(ValueError, match="Insufficient control units in pre-treatment period for regression"):
        reg_did_rc(y=y, post=post, d=d, covariates=x)


def test_reg_did_rc_insufficient_control_post():
    rng = np.random.default_rng(42)
    n = 10
    x = np.column_stack([np.ones(n), rng.standard_normal((n, 3))])
    d = np.ones(n, dtype=int)
    d[0:6] = 0
    post = np.zeros(n, dtype=int)
    post[4:] = 1
    y = rng.standard_normal(n)

    with pytest.raises(ValueError, match="Insufficient control units in post-treatment period for regression"):
        reg_did_rc(y=y, post=post, d=d, covariates=x)


def test_reg_did_rc_args_output():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + rng.standard_normal(n)

    result = reg_did_rc(y=y, post=post, d=d, covariates=x, boot=True, nboot=10)

    assert result.args["panel"] is False
    assert result.args["boot"] is True
    assert result.args["boot_type"] == "weighted"
    assert result.args["nboot"] == 10
    assert result.args["type"] == "or"


def test_reg_did_rc_reproducibility():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + rng.standard_normal(n)

    result1 = reg_did_rc(y=y, post=post, d=d, covariates=x)
    result2 = reg_did_rc(y=y, post=post, d=d, covariates=x)

    assert result1.att == result2.att
    assert result1.se == result2.se


def test_reg_did_rc_multicollinear_covariates():
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.standard_normal(n)
    x2 = 2 * x1
    x = np.column_stack([np.ones(n), x1, x2])
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = 1 + 0.5 * x1 + 2 * d * post + rng.standard_normal(n)

    with pytest.raises(ValueError, match="singular"):
        reg_did_rc(y=y, post=post, d=d, covariates=x)


def test_reg_did_rc_near_singular_gram():
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.standard_normal(n)
    x2 = x1 + 1e-10 * rng.standard_normal(n)
    x = np.column_stack([np.ones(n), x1, x2])
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = 1 + 0.5 * x1 + 2 * d * post + rng.standard_normal(n)

    with pytest.raises(ValueError, match="singular"):
        reg_did_rc(y=y, post=post, d=d, covariates=x)


def test_reg_did_rc_1d_covariates():
    rng = np.random.default_rng(42)
    n = 100
    x = rng.standard_normal(n)
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = 1 + 0.5 * x + 2 * d * post + rng.standard_normal(n)

    result = reg_did_rc(y=y, post=post, d=d, covariates=x)

    assert isinstance(result.att, float)
    assert result.se > 0
