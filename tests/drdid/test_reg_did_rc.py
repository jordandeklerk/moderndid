"""Tests for outcome regression DiD estimator with repeated cross-sections."""

import re
import warnings

import numpy as np
import pytest

from moderndid.drdid import reg_did_rc

pytestmark = pytest.mark.slow


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

    result = reg_did_rc(y=y, post=post, d=d, covariates=x, boot=True, boot_type="weighted", biters=10)

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

    result = reg_did_rc(y=y, post=post, d=d, covariates=x, boot=True, boot_type="multiplier", biters=10)

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
    message = (
        "No units found in the comparison group in the pre-period. Pass data with comparison units in both periods."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        reg_did_rc(y=y, post=post, d=d_all_treated, covariates=x)

    d_all_control = np.zeros(n, dtype=int)
    message = (
        "No units found in the treated group in the pre-period. Pass a treatment indicator that equals 1 for "
        "treated units in both periods."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        reg_did_rc(y=y, post=post, d=d_all_control, covariates=x)


def test_reg_did_rc_no_pre_control():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = np.zeros(n, dtype=int)
    d[50:] = 1
    post = np.ones(n, dtype=int)
    y = x @ [1, 0.5] + 2 * d * post + rng.standard_normal(n)
    message = (
        "No units found in the comparison group in the pre-period. Pass data with comparison units in both periods."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        reg_did_rc(y=y, post=post, d=d, covariates=x)


def test_reg_did_rc_no_post_control():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = np.zeros(n, dtype=int)
    d[50:] = 1
    post = np.zeros(n, dtype=int)
    y = x @ [1, 0.5] + rng.standard_normal(n)
    message = (
        "No units found in the comparison group in the post-period. Pass data with comparison units in both periods."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
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
    message = (
        "The comparison group in the pre-period has fewer observations (2) than regression coefficients (4). "
        "Pass fewer covariates."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
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
    message = (
        "The comparison group in the post-period has fewer observations (2) than regression coefficients (4). "
        "Pass fewer covariates."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        reg_did_rc(y=y, post=post, d=d, covariates=x)


def test_reg_did_rc_args_output():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + rng.standard_normal(n)

    result = reg_did_rc(y=y, post=post, d=d, covariates=x, boot=True, biters=10)

    assert result.args["panel"] is False
    assert result.args["boot"] is True
    assert result.args["boot_type"] == "weighted"
    assert result.args["biters"] == 10
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


def test_reg_did_rc_fits_exact_comparison_regressions_without_warnings(rc_exact_comparison_fit):
    spec = rc_exact_comparison_fit
    y, post, d = spec["y"], spec["post"], spec["d"]
    x, weights = spec["covariates"], spec["i_weights"]
    coefficients = [
        np.linalg.lstsq(np.sqrt(weights[rows])[:, None] * x[rows], np.sqrt(weights[rows]) * y[rows], rcond=None)[0]
        for rows in ((d == 0) & (post == 0), (d == 0) & (post == 1))
    ]
    change = x @ (coefficients[1] - coefficients[0])
    w_post, w_pre, w_treated = weights * d * post, weights * d * (1 - post), weights * d
    att = np.average(y, weights=w_post) - np.average(y, weights=w_pre) - np.average(change, weights=w_treated)
    inf_func = (
        w_post * (y - np.average(y, weights=w_post)) / np.mean(w_post)
        - w_pre * (y - np.average(y, weights=w_pre)) / np.mean(w_pre)
        - w_treated * (change - np.average(change, weights=w_treated)) / np.mean(w_treated)
    )

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = reg_did_rc(**spec, influence_func=True)

    assert [str(warning.message) for warning in record] == []
    np.testing.assert_allclose(result.att, att, rtol=1e-12)
    np.testing.assert_allclose(result.att_inf_func, inf_func, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.se, np.sqrt(np.mean(inf_func**2) / y.size), rtol=1e-12)


@pytest.mark.parametrize("weight", [1.0, 0.0], ids=["positive_weight", "zero_weight"])
@pytest.mark.parametrize("value", [np.nan, np.inf], ids=["nan", "inf"])
@pytest.mark.parametrize(("period", "post_value"), [("pre", 0), ("post", 1)], ids=["pre", "post"])
def test_reg_did_rc_rejects_a_comparison_outcome_that_is_not_finite_without_warnings(
    two_period_specs, forbid_errstate, period, post_value, value, weight
):
    spec = two_period_specs["repeated_cross_sections"]
    row = np.flatnonzero((spec["d"] == 0) & (spec["post"] == post_value))[0]
    y, i_weights = spec["y"].copy(), spec["i_weights"].copy()
    y[row] = value
    i_weights[row] = weight
    message = (
        f"Failed to fit {period}-period regression: y holds missing or infinite values for some observations of the "
        f"comparison group in the {period}-period. Pass finite outcomes for those observations."
    )
    forbid_errstate()

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            reg_did_rc(**{**spec, "y": y, "i_weights": i_weights})

    assert [str(warning.message) for warning in record] == []
