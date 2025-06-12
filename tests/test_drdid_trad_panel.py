"""Tests for the locally efficient doubly robust DiD estimator for panel data."""

import numpy as np
import pytest

from pydid.drdid import drdid_panel


@pytest.mark.parametrize("covariates", [None, "with_covariates"])
def test_drdid_panel_basic(covariates):
    n_units = 200
    rng = np.random.RandomState(42)

    d = rng.binomial(1, 0.5, n_units)

    if covariates == "with_covariates":
        x = rng.randn(n_units, 3)
        x = np.column_stack([np.ones(n_units), x])
    else:
        x = None

    y0 = rng.randn(n_units)
    y1 = y0 + 2 * d + rng.randn(n_units)

    result = drdid_panel(y1=y1, y0=y0, d=d, covariates=x)

    assert isinstance(result.att, float)
    assert isinstance(result.se, float)
    assert result.se > 0
    assert result.lci < result.uci
    assert result.boots is None
    assert result.att_inf_func is None


def test_drdid_panel_with_weights():
    n_units = 200
    rng = np.random.RandomState(42)

    d = rng.binomial(1, 0.5, n_units)
    x = np.column_stack([np.ones(n_units), rng.randn(n_units, 2)])
    y0 = rng.randn(n_units)
    y1 = y0 + 2 * d + rng.randn(n_units)
    i_weights = rng.uniform(0.5, 1.5, n_units)

    result = drdid_panel(y1=y1, y0=y0, d=d, covariates=x, i_weights=i_weights)

    assert isinstance(result.att, float)
    assert result.se > 0


def test_drdid_panel_influence_function():
    n_units = 200
    rng = np.random.RandomState(42)

    d = rng.binomial(1, 0.5, n_units)
    x = np.column_stack([np.ones(n_units), rng.randn(n_units, 2)])
    y0 = rng.randn(n_units)
    y1 = y0 + 2 * d + rng.randn(n_units)

    result = drdid_panel(y1=y1, y0=y0, d=d, covariates=x, influence_func=True)

    assert result.att_inf_func is not None
    assert len(result.att_inf_func) == n_units
    # The mean of the influence function should be close to zero
    assert np.abs(np.mean(result.att_inf_func)) < 0.1


@pytest.mark.parametrize("boot_type", ["weighted", "multiplier"])
def test_drdid_panel_bootstrap(boot_type):
    n_units = 100
    rng = np.random.RandomState(42)

    d = rng.binomial(1, 0.5, n_units)
    x = np.column_stack([np.ones(n_units), rng.randn(n_units, 2)])
    y0 = rng.randn(n_units)
    y1 = y0 + 2 * d + rng.randn(n_units)

    result = drdid_panel(y1=y1, y0=y0, d=d, covariates=x, boot=True, boot_type=boot_type, nboot=100)

    assert result.boots is not None
    assert len(result.boots) == 100
    assert result.se > 0
    assert result.lci < result.uci


def test_drdid_panel_no_treated():
    n_units = 100
    rng = np.random.RandomState(42)

    d = np.zeros(n_units)
    x = np.column_stack([np.ones(n_units), rng.randn(n_units, 2)])
    y0 = rng.randn(n_units)
    y1 = y0 + rng.randn(n_units)

    with pytest.raises(ValueError, match="No effectively treated units"):
        drdid_panel(y1=y1, y0=y0, d=d, covariates=x)


def test_drdid_panel_all_treated():
    n_units = 100
    rng = np.random.RandomState(42)

    d = np.ones(n_units)
    x = np.column_stack([np.ones(n_units), rng.randn(n_units, 2)])
    y0 = rng.randn(n_units)
    y1 = y0 + 2 + rng.randn(n_units)

    with pytest.raises(ValueError, match="No control units"):
        drdid_panel(y1=y1, y0=y0, d=d, covariates=x)


def test_drdid_panel_trimming():
    n_units = 200
    rng = np.random.RandomState(42)

    d = rng.binomial(1, 0.05, n_units)
    x = np.column_stack([np.ones(n_units), rng.randn(n_units, 2)])
    y0 = rng.randn(n_units)
    y1 = y0 + 2 * d + rng.randn(n_units)

    result = drdid_panel(y1=y1, y0=y0, d=d, covariates=x, trim_level=0.9)

    assert isinstance(result.att, float)
    assert result.se > 0


def test_drdid_panel_singular_covariate_matrix():
    n_units = 100
    rng = np.random.RandomState(42)

    d = rng.binomial(1, 0.5, n_units)
    x1 = rng.randn(n_units)
    x = np.column_stack([np.ones(n_units), x1, x1])
    y0 = rng.randn(n_units)
    y1 = y0 + 2 * d + rng.randn(n_units)

    with pytest.raises(ValueError, match="singular"):
        drdid_panel(y1=y1, y0=y0, d=d, covariates=x)


def test_drdid_panel_negative_weights():
    n_units = 100
    rng = np.random.RandomState(42)

    d = rng.binomial(1, 0.5, n_units)
    x = np.column_stack([np.ones(n_units), rng.randn(n_units, 2)])
    y0 = rng.randn(n_units)
    y1 = y0 + 2 * d + rng.randn(n_units)
    i_weights = rng.randn(n_units)

    with pytest.raises(ValueError, match="non-negative"):
        drdid_panel(y1=y1, y0=y0, d=d, covariates=x, i_weights=i_weights)


def test_drdid_panel_dgp():
    """Test on a data generating process with known ATT."""
    n_units = 2000
    rng = np.random.RandomState(42)

    # Generate covariates
    x1 = rng.normal(0, 1, n_units)
    x2 = rng.normal(0, 1, n_units)
    x3 = rng.normal(0, 1, n_units)
    x4 = rng.normal(0, 1, n_units)

    # Baseline outcome
    y00 = x1 + x2 + rng.normal(0, 1, n_units)

    # Outcome in second period
    y10 = y00 + x1 + x3 + rng.normal(0, 1, n_units)

    # True treatment effect
    att_true = 1

    # Treatment indicator based on propensity score
    d_propensity = 1 / (1 + np.exp(-(x1 + x3)))
    d = (rng.uniform(size=n_units) < d_propensity).astype(int)

    # Observed outcomes
    y1 = y10 + att_true * d
    y0 = y00

    # Covariate matrix
    covariates = np.column_stack((np.ones(n_units), x1, x2, x3, x4))

    result = drdid_panel(y1=y1, y0=y0, d=d, covariates=covariates)

    # We expect the ATT to be close to the true value
    assert 0.8 < result.att < 1.2
    assert result.se > 0
    assert result.args["estMethod"] == "trad"
