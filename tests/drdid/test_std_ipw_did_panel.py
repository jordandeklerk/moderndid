"""Tests for standardized IPW DiD estimator with panel data."""

import functools
import re
import warnings
from importlib import import_module

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats

from moderndid.drdid import ipw_did_panel, std_ipw_did_panel

pytestmark = pytest.mark.slow


def test_std_ipw_did_panel_basic(panel_cell_arrays, cell_panel_att, cell_panel_inf_func):
    arrays = panel_cell_arrays

    result = std_ipw_did_panel(**arrays)
    att = cell_panel_att(**arrays, normalized=True)
    se = np.sqrt(np.sum(cell_panel_inf_func(**arrays) ** 2)) / arrays["d"].size

    assert isinstance(result.att, float)
    assert isinstance(result.se, float)
    np.testing.assert_allclose(result.att, att, rtol=1e-10)
    np.testing.assert_allclose([result.se, result.lci, result.uci], [se, att - 1.96 * se, att + 1.96 * se], rtol=1e-10)
    assert result.boots is None
    assert result.att_inf_func is None


def test_std_ipw_did_panel_with_weights(two_covariate_design):
    design = two_covariate_design
    covariates = np.column_stack([np.ones(design["d"].size), design["z"]])
    spec = {"y1": design["y1"], "y0": design["y0"], "d": design["d"], "covariates": covariates}
    counts = np.random.default_rng(6).integers(1, 4, size=design["d"].size)

    weighted = std_ipw_did_panel(**spec, i_weights=counts)
    replicated = std_ipw_did_panel(**{name: np.repeat(value, counts, axis=0) for name, value in spec.items()})
    unweighted = std_ipw_did_panel(**spec)

    assert abs(weighted.att - unweighted.att) > 1e-2
    np.testing.assert_allclose(weighted.att, replicated.att, rtol=1e-12)


def test_std_ipw_did_panel_with_influence_func(panel_cell_arrays, cell_panel_inf_func):
    result = std_ipw_did_panel(**panel_cell_arrays, influence_func=True)

    np.testing.assert_allclose(result.att_inf_func, cell_panel_inf_func(**panel_cell_arrays), rtol=1e-8, atol=1e-8)


def test_std_ipw_did_panel_no_covariates(panel_cell_arrays, cell_panel_att, cell_panel_inf_func):
    arrays = {**panel_cell_arrays, "covariates": np.ones((panel_cell_arrays["d"].size, 1))}

    result = std_ipw_did_panel(**{**arrays, "covariates": None})
    att = cell_panel_att(**arrays, normalized=True)
    se = np.sqrt(np.sum(cell_panel_inf_func(**arrays) ** 2)) / arrays["d"].size

    np.testing.assert_allclose([result.att, result.se], [att, se], rtol=1e-12)


def test_std_ipw_did_panel_bootstrap_weighted(panel_cell_arrays, cell_panel_att, preset_bootstrap_weights):
    arrays = panel_cell_arrays
    draws = np.random.default_rng(3).exponential(size=(25, arrays["d"].size))
    preset_bootstrap_weights(draws)

    result = std_ipw_did_panel(**arrays, boot=True, boot_type="weighted", biters=25, trim_level=0.95)
    att = cell_panel_att(**arrays, trim_level=0.95, normalized=True)
    boots = np.array(
        [
            cell_panel_att(**{**arrays, "i_weights": arrays["i_weights"] * v}, trim_level=0.95, normalized=True)
            for v in draws
        ]
    )
    se = stats.iqr(boots - att) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
    cv = np.quantile(np.abs(boots - att) / se, 0.95)

    np.testing.assert_allclose(result.boots, boots, rtol=1e-8)
    np.testing.assert_allclose([result.se, result.lci, result.uci], [se, att - cv * se, att + cv * se], rtol=1e-8)


def test_std_ipw_did_panel_bootstrap_multiplier(
    monkeypatch, draws_from_weights, panel_cell_arrays, cell_panel_att, cell_panel_inf_func
):
    arrays = panel_cell_arrays
    weights = np.random.default_rng(4).choice([-1.0, 1.0], size=(25, arrays["d"].size))
    module = import_module("moderndid.drdid.estimators.std_ipw_did_panel")
    monkeypatch.setattr(module, "mboot_did", draws_from_weights(weights))

    result = std_ipw_did_panel(**arrays, boot=True, boot_type="multiplier", biters=25)
    att = cell_panel_att(**arrays, normalized=True)
    boots = weights @ cell_panel_inf_func(**arrays) / arrays["d"].size
    se = stats.iqr(boots) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
    cv = np.quantile(np.abs(boots) / se, 0.95)

    np.testing.assert_allclose(result.boots, boots, rtol=1e-8)
    np.testing.assert_allclose([result.se, result.lci, result.uci], [se, att - cv * se, att + cv * se], rtol=1e-8)


def test_std_ipw_did_panel_invalid_inputs():
    rng = np.random.default_rng(42)
    n = 50
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y0 = rng.standard_normal(n)
    y1 = rng.standard_normal(n)
    d = rng.binomial(1, 0.5, n)

    with pytest.raises(ValueError, match="i_weights must be non-negative"):
        std_ipw_did_panel(y1=y1, y0=y0, d=d, covariates=x, i_weights=-np.ones(n))


def test_std_ipw_did_panel_accepts_zero_weights_and_drops_their_units(panel_cell_arrays, cell_panel_att):
    weights = panel_cell_arrays["i_weights"].copy()
    weights[:10] = 0.0
    arrays = {**panel_cell_arrays, "i_weights": weights}

    result = std_ipw_did_panel(**arrays)

    assert abs(result.att - cell_panel_att(**panel_cell_arrays, normalized=True)) > 1e-3
    np.testing.assert_allclose(result.att, cell_panel_att(**arrays, normalized=True), rtol=1e-10)


def test_std_ipw_did_panel_edge_cases():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y0 = rng.standard_normal(n)
    y1 = rng.standard_normal(n)

    d_all_treated = np.ones(n, dtype=int)
    with pytest.raises(ValueError, match="No control units found"):
        std_ipw_did_panel(y1=y1, y0=y0, d=d_all_treated, covariates=x)

    d_all_control = np.zeros(n, dtype=int)
    with pytest.raises(ValueError, match="No treated units found"):
        std_ipw_did_panel(y1=y1, y0=y0, d=d_all_control, covariates=x)


def test_std_ipw_did_panel_extreme_pscore():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), np.linspace(-10, 10, n)])
    d = (x[:, 1] > 0).astype(int)
    y0 = x @ [1, 0.5] + rng.standard_normal(n)
    y1 = y0 + 2 * d + rng.standard_normal(n) * 0.5
    dy = y1 - y0

    with pytest.warns(UserWarning, match="^Perfect separation or prediction detected"):
        result = std_ipw_did_panel(y1=y1, y0=y0, d=d, covariates=x)

    np.testing.assert_allclose(result.att, dy[d == 1].mean() - dy[d == 0].mean(), rtol=1e-10)


def test_std_ipw_did_panel_trim_level(panel_cell_arrays, cell_panel_att):
    kept = std_ipw_did_panel(**panel_cell_arrays, trim_level=0.99)
    trimmed = std_ipw_did_panel(**panel_cell_arrays, trim_level=0.95)

    assert abs(kept.att - trimmed.att) > 0.1
    np.testing.assert_allclose(
        kept.att, cell_panel_att(**panel_cell_arrays, trim_level=0.99, normalized=True), rtol=1e-10
    )
    np.testing.assert_allclose(
        trimmed.att, cell_panel_att(**panel_cell_arrays, trim_level=0.95, normalized=True), rtol=1e-10
    )


@pytest.mark.parametrize("boot_type", ["weighted", "multiplier"])
def test_std_ipw_did_panel_args_output(panel_cell_arrays, boot_type):
    result = std_ipw_did_panel(**panel_cell_arrays, boot=True, boot_type=boot_type, biters=10, trim_level=0.99)

    assert len(result.boots) == 10
    assert result.args["panel"] is True
    assert result.args["normalized"] is True
    assert result.args["boot"] is True
    assert result.args["boot_type"] == boot_type
    assert result.args["biters"] == 10
    assert result.args["type"] == "ipw"
    assert result.args["trim_level"] == 0.99


def test_std_ipw_did_panel_does_not_depend_on_the_order_of_units(two_period_specs):
    spec = two_period_specs["panel"]
    order = np.random.default_rng(7).permutation(spec["d"].size)

    expected = std_ipw_did_panel(**spec)
    result = std_ipw_did_panel(**{name: value[order] for name, value in spec.items()})

    assert np.isfinite([expected.att, expected.se]).all()
    np.testing.assert_allclose([result.att, result.se], [expected.att, expected.se], rtol=1e-12)


def test_std_ipw_did_panel_multicollinear_covariates():
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.standard_normal(n)
    x2 = 2 * x1
    x = np.column_stack([np.ones(n), x1, x2])
    d = rng.binomial(1, 0.5, n)
    y0 = 1 + 0.5 * x1 + rng.standard_normal(n)
    y1 = y0 + 2 * d + rng.standard_normal(n) * 0.5

    result = std_ipw_did_panel(y1=y1, y0=y0, d=d, covariates=x)
    expected = std_ipw_did_panel(y1=y1, y0=y0, d=d, covariates=x[:, :2])

    assert np.isfinite([expected.att, expected.se]).all()
    np.testing.assert_allclose([result.att, result.se], [expected.att, expected.se], rtol=1e-10)


def test_std_ipw_did_panel_arrays_vs_lists():
    rng = np.random.default_rng(42)
    n = 50
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    y0 = x @ [1, 0.5] + rng.standard_normal(n)
    y1 = y0 + 2 * d + rng.standard_normal(n) * 0.5

    result_array = std_ipw_did_panel(y1=y1, y0=y0, d=d, covariates=x)
    result_list = std_ipw_did_panel(y1=y1.tolist(), y0=y0.tolist(), d=d.tolist(), covariates=x.tolist())

    assert np.isfinite([result_array.att, result_array.se]).all()
    assert result_list.att == result_array.att
    assert result_list.se == result_array.se


def test_std_ipw_did_panel_matches_a_hand_computed_att_on_ten_units():
    b = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    d = np.array([1, 1, 0, 0, 0, 1, 1, 1, 0, 0])
    y1 = np.array([3.0, 5.0, 1.0, 2.0, 3.0, 6.0, 8.0, 10.0, 4.0, 6.0])

    result = std_ipw_did_panel(y1=y1, y0=np.zeros(10), d=d, covariates=np.column_stack([np.ones(10), b]))

    np.testing.assert_allclose([result.att, result.se], [2.6, np.sqrt(62.8 + 32 / 9) / 10], rtol=1e-12)


def test_std_ipw_did_panel_vs_non_standardized(two_period_specs):
    spec = two_period_specs["panel"]
    shifted = {**spec, "y1": spec["y1"] + 3}

    result_std = std_ipw_did_panel(**spec)
    shifted_std = std_ipw_did_panel(**shifted)
    result_non_std = ipw_did_panel(**spec)
    shifted_non_std = ipw_did_panel(**shifted)

    assert abs(shifted_non_std.att - result_non_std.att) > 1e-2
    assert np.isfinite([result_std.att, result_std.se]).all()
    np.testing.assert_allclose([shifted_std.att, shifted_std.se], [result_std.att, result_std.se], rtol=1e-12)
    assert result_std.args["normalized"] is True
    assert result_non_std.args["normalized"] is False


def test_std_ipw_did_panel_warns_when_trimming_drops_every_comparison_unit(panel_cell_arrays):
    message = "No effectively control units after trimming."

    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$"):
        result = std_ipw_did_panel(**panel_cell_arrays, trim_level=0.3)

    assert np.isnan([result.att, result.se, result.lci, result.uci]).all()
    assert result.att_inf_func is None


def test_std_ipw_did_panel_warns_when_the_propensity_score_fit_stops_before_converging(monkeypatch, two_period_specs):
    spec = two_period_specs["panel"]
    message = "Propensity score estimation did not converge."

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        std_ipw_did_panel(**spec)
    monkeypatch.setattr(sm.GLM, "fit", functools.partialmethod(sm.GLM.fit, maxiter=1))
    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$"):
        std_ipw_did_panel(**spec)

    assert not any("did not converge" in str(w.message) for w in caught)
