"""Tests for IPW DiD estimator with repeated cross-sections."""

import functools
import re
import warnings
from importlib import import_module

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats

from moderndid.drdid import ipw_did_rc

pytestmark = pytest.mark.slow


def test_ipw_did_rc_basic(rc_cell_arrays, cell_rc_att):
    arrays = rc_cell_arrays

    result = ipw_did_rc(**arrays)
    inf_func = ipw_did_rc(**arrays, influence_func=True).att_inf_func
    att = cell_rc_att(**arrays)
    se = np.sqrt(np.mean(inf_func**2) / inf_func.size)

    assert isinstance(result.att, float)
    assert isinstance(result.se, float)
    np.testing.assert_allclose(result.att, att, rtol=1e-10)
    np.testing.assert_allclose([result.se, result.lci, result.uci], [se, att - 1.96 * se, att + 1.96 * se], rtol=1e-10)
    assert result.boots is None
    assert result.att_inf_func is None


def test_ipw_did_rc_with_weights(two_covariate_design):
    design = two_covariate_design
    covariates = np.column_stack([np.ones(design["d"].size), design["z"]])
    spec = {"y": design["y"], "post": design["post"], "d": design["d"], "covariates": covariates}
    counts = np.random.default_rng(6).integers(1, 4, size=design["d"].size)

    weighted = ipw_did_rc(**spec, i_weights=counts)
    replicated = ipw_did_rc(**{name: np.repeat(value, counts, axis=0) for name, value in spec.items()})
    unweighted = ipw_did_rc(**spec)

    assert abs(weighted.att - unweighted.att) > 1e-2
    np.testing.assert_allclose(weighted.att, replicated.att, rtol=1e-12)


def test_ipw_did_rc_with_influence_func(rc_misspecified_score_arrays, weight_derivative):
    arrays = rc_misspecified_score_arrays
    cell = 2 * arrays["d"] + arrays["post"]
    rows = np.concatenate([np.flatnonzero(cell == value)[:3] for value in range(4)])

    result = ipw_did_rc(**arrays, influence_func=True)
    expected = weight_derivative(lambda weights: ipw_did_rc(**arrays, i_weights=weights).att, cell.size, rows)

    assert len(result.att_inf_func) == cell.size
    assert abs(np.mean(result.att_inf_func)) < 1e-10
    np.testing.assert_allclose(result.att_inf_func[rows], expected, rtol=1e-6, atol=1e-6)


def test_ipw_did_rc_no_covariates(rc_cell_arrays, cell_rc_att):
    arrays = {**rc_cell_arrays, "covariates": np.ones((rc_cell_arrays["d"].size, 1))}

    result = ipw_did_rc(**{**arrays, "covariates": None}, influence_func=True)
    se = np.sqrt(np.mean(result.att_inf_func**2) / result.att_inf_func.size)

    np.testing.assert_allclose([result.att, result.se], [cell_rc_att(**arrays), se], rtol=1e-12)


def test_ipw_did_rc_bootstrap_weighted(rc_cell_arrays, cell_rc_att, preset_bootstrap_weights):
    arrays = rc_cell_arrays
    draws = np.random.default_rng(3).exponential(size=(25, arrays["d"].size))
    preset_bootstrap_weights(draws)

    result = ipw_did_rc(**arrays, boot=True, boot_type="weighted", biters=25, trim_level=0.95)
    att = cell_rc_att(**arrays, trim_level=0.95)
    boots = np.array([cell_rc_att(**{**arrays, "i_weights": arrays["i_weights"] * v}, trim_level=0.95) for v in draws])
    se = stats.iqr(boots - att) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
    cv = np.quantile(np.abs(boots - att) / se, 0.95)

    np.testing.assert_allclose(result.boots, boots, rtol=1e-7)
    np.testing.assert_allclose([result.se, result.lci, result.uci], [se, att - cv * se, att + cv * se], rtol=1e-8)


def test_ipw_did_rc_bootstrap_multiplier(monkeypatch, draws_from_weights, rc_cell_arrays, cell_rc_att):
    arrays = rc_cell_arrays
    weights = np.random.default_rng(4).choice([-1.0, 1.0], size=(25, arrays["d"].size))
    inf_func = ipw_did_rc(**arrays, influence_func=True).att_inf_func
    module = import_module("moderndid.drdid.estimators.ipw_did_rc")
    monkeypatch.setattr(module, "mboot_did", draws_from_weights(weights))

    result = ipw_did_rc(**arrays, boot=True, boot_type="multiplier", biters=25)
    att = cell_rc_att(**arrays)
    boots = weights @ inf_func / inf_func.size
    se = stats.iqr(boots) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
    cv = np.quantile(np.abs(boots) / se, 0.95)

    np.testing.assert_allclose(result.boots, boots, rtol=1e-12)
    np.testing.assert_allclose([result.se, result.lci, result.uci], [se, att - cv * se, att + cv * se], rtol=1e-10)


def test_ipw_did_rc_invalid_inputs():
    rng = np.random.default_rng(42)
    n = 50
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = rng.standard_normal(n)
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)

    with pytest.raises(ValueError, match="i_weights must be non-negative"):
        ipw_did_rc(y=y, post=post, d=d, covariates=x, i_weights=-np.ones(n))


def test_ipw_did_rc_accepts_zero_weights_and_drops_their_rows(rc_cell_arrays, cell_rc_att):
    weights = rc_cell_arrays["i_weights"].copy()
    weights[:10] = 0.0
    arrays = {**rc_cell_arrays, "i_weights": weights}

    result = ipw_did_rc(**arrays)

    assert abs(result.att - cell_rc_att(**rc_cell_arrays)) > 1e-3
    np.testing.assert_allclose(result.att, cell_rc_att(**arrays), rtol=1e-10)


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
    lam = post.mean()
    weights = np.where(d == 1, 1.0, 1e-6 / (1 - 1e-6))

    with pytest.warns(UserWarning, match="^Perfect separation or prediction detected"):
        result = ipw_did_rc(y=y, post=post, d=d, covariates=x)
    expected = np.sum(weights * (2 * d - 1) * (post - lam) / (lam * (1 - lam)) * y) / d.sum()

    np.testing.assert_allclose(result.att, expected, rtol=1e-10)


def test_ipw_did_rc_trim_level(rc_cell_arrays, cell_rc_att):
    kept = ipw_did_rc(**rc_cell_arrays, trim_level=0.99)
    trimmed = ipw_did_rc(**rc_cell_arrays, trim_level=0.95)

    assert abs(kept.att - trimmed.att) > 0.1
    np.testing.assert_allclose(kept.att, cell_rc_att(**rc_cell_arrays, trim_level=0.99), rtol=1e-10)
    np.testing.assert_allclose(trimmed.att, cell_rc_att(**rc_cell_arrays, trim_level=0.95), rtol=1e-10)


@pytest.mark.parametrize("boot_type", ["weighted", "multiplier"])
def test_ipw_did_rc_args_output(rc_cell_arrays, boot_type):
    result = ipw_did_rc(**rc_cell_arrays, boot=True, boot_type=boot_type, biters=10, trim_level=0.99)

    assert len(result.boots) == 10
    assert result.args["panel"] is False
    assert result.args["normalized"] is False
    assert result.args["boot"] is True
    assert result.args["boot_type"] == boot_type
    assert result.args["biters"] == 10
    assert result.args["type"] == "ipw"
    assert result.args["trim_level"] == 0.99


def test_ipw_did_rc_does_not_depend_on_the_order_of_rows(two_period_specs):
    spec = two_period_specs["repeated_cross_sections"]
    order = np.random.default_rng(7).permutation(spec["d"].size)

    expected = ipw_did_rc(**spec)
    result = ipw_did_rc(**{name: value[order] for name, value in spec.items()})

    assert np.isfinite([expected.att, expected.se]).all()
    np.testing.assert_allclose([result.att, result.se], [expected.att, expected.se], rtol=1e-12)


def test_ipw_did_rc_multicollinear_covariates():
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.standard_normal(n)
    x2 = 2 * x1
    x = np.column_stack([np.ones(n), x1, x2])
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = 1 + 0.5 * x1 + 2 * d * post + rng.standard_normal(n)

    with pytest.warns(UserWarning, match=re.escape("Failed to invert Hessian matrix. Using pseudo-inverse.")):
        result = ipw_did_rc(y=y, post=post, d=d, covariates=x)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        expected = ipw_did_rc(y=y, post=post, d=d, covariates=x[:, :2])

    assert not any("Hessian" in str(w.message) for w in caught)
    assert np.isfinite([expected.att, expected.se]).all()
    np.testing.assert_allclose([result.att, result.se], [expected.att, expected.se], rtol=1e-10)


def test_ipw_did_rc_warns_when_the_propensity_score_fit_stops_before_converging(monkeypatch, two_period_specs):
    spec = two_period_specs["repeated_cross_sections"]
    message = "GLM algorithm did not converge."

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ipw_did_rc(**spec)
    monkeypatch.setattr(sm.GLM, "fit", functools.partialmethod(sm.GLM.fit, maxiter=1))
    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$"):
        ipw_did_rc(**spec)

    assert not any("did not converge" in str(w.message) for w in caught)
