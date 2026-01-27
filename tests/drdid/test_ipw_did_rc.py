"""Tests for IPW DiD estimator with repeated cross-sections."""

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from moderndid.drdid import ipw_did_rc


def test_ipw_did_rc_basic():
    rng = np.random.default_rng(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    d = rng.binomial(1, 0.3, n)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5, -0.3] + 2 * d * post + rng.standard_normal(n)

    result = ipw_did_rc(y=y, post=post, d=d, covariates=x)

    assert isinstance(result.att, float)
    assert isinstance(result.se, float)
    assert result.se > 0
    assert result.lci < result.att < result.uci
    assert result.boots is None
    assert result.att_inf_func is None


def test_ipw_did_rc_with_weights():
    rng = np.random.default_rng(42)
    n = 200
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + rng.standard_normal(n)
    weights = rng.exponential(1, n)

    result = ipw_did_rc(y=y, post=post, d=d, covariates=x, i_weights=weights)

    assert isinstance(result.att, float)
    assert result.se > 0


def test_ipw_did_rc_with_influence_func():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + rng.standard_normal(n)

    result = ipw_did_rc(y=y, post=post, d=d, covariates=x, influence_func=True)

    assert result.att_inf_func is not None
    assert len(result.att_inf_func) == n
    assert np.isclose(np.mean(result.att_inf_func), 0, atol=1e-10)


def test_ipw_did_rc_no_covariates():
    rng = np.random.default_rng(42)
    n = 200
    d = rng.binomial(1, 0.3, n)
    post = rng.binomial(1, 0.5, n)
    y = 1 + 2 * d * post + rng.standard_normal(n)

    result = ipw_did_rc(y=y, post=post, d=d, covariates=None)

    assert isinstance(result.att, float)
    assert result.se > 0


def test_ipw_did_rc_bootstrap_weighted():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + rng.standard_normal(n)

    result = ipw_did_rc(y=y, post=post, d=d, covariates=x, boot=True, boot_type="weighted", nboot=10)

    assert result.boots is not None
    assert len(result.boots) == 10
    assert not np.all(np.isnan(result.boots))


def test_ipw_did_rc_bootstrap_multiplier():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + rng.standard_normal(n)

    result = ipw_did_rc(y=y, post=post, d=d, covariates=x, boot=True, boot_type="multiplier", nboot=10)

    assert result.boots is not None
    assert len(result.boots) == 10


def test_ipw_did_rc_invalid_inputs():
    rng = np.random.default_rng(42)
    n = 50
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = rng.standard_normal(n)
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)

    with pytest.raises(ValueError, match="i_weights must be non-negative"):
        ipw_did_rc(y=y, post=post, d=d, covariates=x, i_weights=-np.ones(n))


def test_ipw_did_rc_edge_cases():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = rng.standard_normal(n)
    post = rng.binomial(1, 0.5, n)

    d_all_treated = np.ones(n, dtype=int)
    with pytest.raises(ValueError, match="No control units found"):
        ipw_did_rc(y=y, post=post, d=d_all_treated, covariates=x)

    d_all_control = np.zeros(n, dtype=int)
    with pytest.raises(ValueError, match="No treated units found"):
        ipw_did_rc(y=y, post=post, d=d_all_control, covariates=x)


def test_ipw_did_rc_all_post():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    post = np.ones(n, dtype=int)
    y = x @ [1, 0.5] + 2 * d * post + rng.standard_normal(n)

    with pytest.raises(ValueError, match="No pre-treatment observations found"):
        ipw_did_rc(y=y, post=post, d=d, covariates=x)


def test_ipw_did_rc_all_pre():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    post = np.zeros(n, dtype=int)
    y = x @ [1, 0.5] + rng.standard_normal(n)

    with pytest.raises(ValueError, match="No post-treatment observations found"):
        ipw_did_rc(y=y, post=post, d=d, covariates=x)


def test_ipw_did_rc_extreme_pscore():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), np.linspace(-10, 10, n)])
    d = (x[:, 1] > 0).astype(int)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + rng.standard_normal(n)

    with pytest.warns(UserWarning):
        result = ipw_did_rc(y=y, post=post, d=d, covariates=x)

    assert isinstance(result.att, float)


def test_ipw_did_rc_lambda_zero():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    post = np.zeros(n, dtype=int)
    y = x @ [1, 0.5] + rng.standard_normal(n)

    with pytest.raises(ValueError, match="No post-treatment observations found"):
        ipw_did_rc(y=y, post=post, d=d, covariates=x)


def test_ipw_did_rc_trim_level():
    rng = np.random.default_rng(42)
    n = 200
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + rng.standard_normal(n)

    result1 = ipw_did_rc(y=y, post=post, d=d, covariates=x, trim_level=0.99)
    result2 = ipw_did_rc(y=y, post=post, d=d, covariates=x, trim_level=0.95)

    assert isinstance(result1.att, float)
    assert isinstance(result2.att, float)


def test_ipw_did_rc_args_output():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + rng.standard_normal(n)

    result = ipw_did_rc(y=y, post=post, d=d, covariates=x, boot=True, nboot=10, trim_level=0.99)

    assert result.args["panel"] is False
    assert result.args["normalized"] is False
    assert result.args["boot"] is True
    assert result.args["boot_type"] == "weighted"
    assert result.args["nboot"] == 10
    assert result.args["type"] == "ipw"
    assert result.args["trim_level"] == 0.99


def test_ipw_did_rc_reproducibility():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + rng.standard_normal(n)

    result1 = ipw_did_rc(y=y, post=post, d=d, covariates=x)
    result2 = ipw_did_rc(y=y, post=post, d=d, covariates=x)

    assert result1.att == result2.att
    assert result1.se == result2.se


def test_ipw_did_rc_multicollinear_covariates():
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.standard_normal(n)
    x2 = 2 * x1
    x = np.column_stack([np.ones(n), x1, x2])
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = 1 + 0.5 * x1 + 2 * d * post + rng.standard_normal(n)

    result = ipw_did_rc(y=y, post=post, d=d, covariates=x)
    assert isinstance(result.att, float)
    assert not np.isnan(result.att)
    assert result.se > 0
