"""Tests for IPW estimators."""

import re
import warnings

import numpy as np
import pytest

from moderndid import ipw_did_rc, ipw_rc, wboot_ipw_rc

pytestmark = pytest.mark.slow


def test_ipw_rc_matches_a_hand_computed_att():
    y = np.array([2.0, 4.0, 7.0, 9.0, 1.0, 3.0, 4.0, 2.0])
    post = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    d = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    ps = np.array([0.5, 0.25, 0.5, 0.75, 0.5, 0.2, 0.5, 0.8])
    i_weights = np.array([1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0])

    result = ipw_rc(y, post, d, ps, i_weights)

    assert isinstance(result, float)
    np.testing.assert_allclose(result, -0.5, rtol=1e-12)


def test_ipw_rc_drops_trimmed_rows_from_the_comparison_terms_only():
    y = np.array([2.0, 4.0, 7.0, 9.0, 1.0, 3.0, 4.0, 2.0])
    post = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    d = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    ps = np.array([0.5, 0.25, 0.5, 0.75, 0.5, 0.2, 0.5, 0.8])
    i_weights = np.array([1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0])
    trim_ps = np.array([True, True, True, True, True, True, True, False])

    result = ipw_rc(y, post, d, ps, i_weights, trim_ps)

    assert isinstance(result, float)
    np.testing.assert_allclose(result, 3.5, rtol=1e-12)


def test_ipw_rc_with_only_treated_rows_returns_their_change_in_means():
    y = np.array([1.0, 3.0, 4.0, 8.0])
    post = np.array([0, 0, 1, 1])
    i_weights = np.array([1.0, 2.0, 1.0, 3.0])

    result = ipw_rc(y, post, np.ones(4), np.full(4, 0.5), i_weights)

    np.testing.assert_allclose(result, 14 / 3, rtol=1e-12)


def test_ipw_rc_returns_nan_with_a_warning_for_a_non_finite_estimate():
    y = np.array([2.0, 4.0, 7.0, np.inf, 1.0, 3.0, 4.0, 2.0])
    post = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    d = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    ps = np.array([0.5, 0.25, 0.5, 0.75, 0.5, 0.2, 0.5, 0.8])

    with pytest.warns(UserWarning, match=f"^{re.escape('IPW estimator is not finite: inf.')}$"):
        result = ipw_rc(y, post, d, ps, np.ones(8))

    assert np.isnan(result)


def test_ipw_did_rc_no_treated():
    rng = np.random.default_rng(42)
    n = 50
    y = np.ones(n)
    post = rng.binomial(1, 0.5, n)
    d = np.zeros(n)
    ps = 0.5 * np.ones(n)
    i_weights = np.ones(n)

    with pytest.warns(UserWarning, match="No treated units found"):
        result = ipw_rc(y, post, d, ps, i_weights)
    assert np.isnan(result)


def test_ipw_did_rc_extreme_propensity():
    rng = np.random.default_rng(42)
    n = 50
    y = np.ones(n)
    post = rng.binomial(1, 0.5, n)
    d = np.zeros(n)
    d[:10] = 1
    ps = np.ones(n)
    ps[d == 1] = 0.5
    i_weights = np.ones(n)

    with pytest.warns(UserWarning, match="Propensity score is 1 for some control units"):
        result = ipw_rc(y, post, d, ps, i_weights)
    assert np.isnan(result)


@pytest.mark.filterwarnings("error")
def test_ipw_did_rc_treated_unit_with_propensity_score_one(forbid_errstate):
    y = np.arange(1.0, 9.0)
    post = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    d = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    ps = np.array([0.3, 0.4, 0.5, 0.6, 0.5, 0.5, 0.5, 0.5])
    expected = ipw_rc(y, post, d, ps, np.ones(8))
    ps[-1] = 1.0
    forbid_errstate()

    assert ipw_rc(y, post, d, ps, np.ones(8)) == expected


def test_ipw_did_rc_lambda_edge_cases():
    rng = np.random.default_rng(42)
    n = 50
    y = np.ones(n)
    d = rng.binomial(1, 0.3, n)
    ps = 0.3 * np.ones(n)
    i_weights = np.ones(n)

    post = np.zeros(n)
    with pytest.warns(UserWarning, match="Lambda is 0"):
        result = ipw_rc(y, post, d, ps, i_weights)
    assert np.isnan(result)

    post = np.ones(n)
    with pytest.warns(UserWarning, match="Lambda is 1"):
        result = ipw_rc(y, post, d, ps, i_weights)
    assert np.isnan(result)


@pytest.mark.parametrize("scale", [0.5, 1.5])
def test_ipw_rc_does_not_depend_on_the_scale_of_the_weights(scale):
    rng = np.random.default_rng(5)
    n = 400
    post = rng.binomial(1, 0.3, n)
    d = rng.binomial(1, 0.4, n)
    ps = rng.uniform(0.2, 0.6, n)
    y = 10 + 20 * d + post + rng.normal(0, 1, n)
    i_weights = rng.exponential(1, n)

    np.testing.assert_allclose(ipw_rc(y, post, d, ps, scale * i_weights), ipw_rc(y, post, d, ps, i_weights), rtol=1e-12)


def test_ipw_did_rc_invalid_inputs():
    rng = np.random.default_rng(42)
    n = 50
    y = np.ones(n)
    post = rng.binomial(1, 0.5, n)
    d = rng.binomial(1, 0.3, n)
    ps = 0.3 * np.ones(n)
    i_weights = np.ones(n)

    with pytest.raises(TypeError, match="All inputs must be NumPy arrays"):
        ipw_rc(list(y), post, d, ps, i_weights)

    with pytest.raises(ValueError, match="All input arrays must be 1-dimensional"):
        ipw_rc(y.reshape(-1, 1), post, d, ps, i_weights)

    with pytest.raises(ValueError, match="All input arrays must have the same shape"):
        ipw_rc(y[:-1], post, d, ps, i_weights)


def test_wboot_ipw_rc_basic(rc_cell_arrays, cell_rc_att, preset_bootstrap_weights):
    arrays = rc_cell_arrays
    draws = np.random.default_rng(5).exponential(size=(10, arrays["d"].size))
    preset_bootstrap_weights(draws)

    boots = wboot_ipw_rc(
        arrays["y"], arrays["post"], arrays["d"], arrays["covariates"], arrays["i_weights"], biters=10, random_state=42
    )
    expected = [cell_rc_att(**{**arrays, "i_weights": arrays["i_weights"] * v}) for v in draws]

    np.testing.assert_allclose(boots, expected, rtol=1e-8)


def test_wboot_ipw_rc_rejects_invalid_inputs(rc_cell_arrays):
    arrays = rc_cell_arrays
    rest = (arrays["post"], arrays["d"], arrays["covariates"], arrays["i_weights"])

    with pytest.raises(TypeError, match=f"^{re.escape('y must be a NumPy array.')}$"):
        wboot_ipw_rc(list(arrays["y"]), *rest, biters=5)
    with pytest.raises(ValueError, match=f"^{re.escape('All arrays must have the same number of observations.')}$"):
        wboot_ipw_rc(arrays["y"][:-1], *rest, biters=5)
    with pytest.raises(ValueError, match=f"^{re.escape('biters must be a positive integer.')}$"):
        wboot_ipw_rc(arrays["y"], *rest, biters=0)
    with pytest.raises(ValueError, match=f"^{re.escape('trim_level must be between 0 and 1.')}$"):
        wboot_ipw_rc(arrays["y"], *rest, biters=5, trim_level=1.5)


def test_wboot_ipw_rc_draws_repeat_with_their_random_state(rc_cell_arrays):
    arrays = rc_cell_arrays
    data = (arrays["y"], arrays["post"], arrays["d"], arrays["covariates"], arrays["i_weights"])

    first = wboot_ipw_rc(*data, biters=5, random_state=3)
    repeated = wboot_ipw_rc(*data, biters=5, random_state=3)
    other = wboot_ipw_rc(*data, biters=5, random_state=4)

    assert np.isfinite(first).all()
    np.testing.assert_array_equal(repeated, first)
    assert np.min(np.abs(other - first)) > 1e-6


def test_wboot_ipw_rc_warns_when_its_draws_fail(rc_cell_arrays):
    arrays = rc_cell_arrays
    rest = (arrays["post"], arrays["d"], arrays["covariates"], arrays["i_weights"])
    y = np.where(np.arange(arrays["y"].size) == 0, np.inf, arrays["y"])

    with warnings.catch_warnings(record=True) as failing:
        warnings.simplefilter("always")
        failed = wboot_ipw_rc(y, *rest, biters=5, random_state=3)
    with warnings.catch_warnings(record=True) as passing:
        warnings.simplefilter("always")
        wboot_ipw_rc(arrays["y"], *rest, biters=5, random_state=3)
    messages = [str(w.message) for w in failing if "iterations failed" in str(w.message)]

    assert np.isnan(failed).all()
    assert messages == ["5 out of 5 bootstrap iterations failed. Results may be unreliable."]
    assert not any("iterations failed" in str(w.message) for w in passing)


def test_wboot_ipw_rc_reproduces_ipw_did_rc_at_unit_weights(unit_bootstrap_weights, trimmed_two_period_data):
    data = trimmed_two_period_data
    expected = ipw_did_rc(data["y"], data["post"], data["d"], data["x"], trim_level=0.9).att

    draws = wboot_ipw_rc(
        data["y"], data["post"], data["d"], data["x"], np.ones(len(data["d"])), biters=2, trim_level=0.9
    )

    np.testing.assert_allclose(draws, expected, rtol=1e-8)
