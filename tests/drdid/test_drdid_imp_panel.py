"""Tests for the improved doubly robust DiD estimator for panel data."""

import numpy as np

from doublediff import drdid_imp_panel


def dgp_panel_for_test(n=2000):
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    x3 = np.random.normal(0, 1, n)
    x4 = np.random.normal(0, 1, n)
    y00 = x1 + x2 + np.random.normal(0, 1, n)
    y10 = y00 + x1 + x3 + np.random.normal(0, 1, n)
    att = 1
    d_propensity = 1 / (1 + np.exp(-(x1 + x3)))
    d = (np.random.uniform(size=n) < d_propensity).astype(int)
    y1 = y10 + att * d
    y0 = y00
    covariates = np.column_stack((np.ones(n), x1, x2, x3, x4))
    return y1, y0, d, covariates


def get_test_data():
    y1, y0, d, covariates = dgp_panel_for_test()
    i_weights = np.ones(len(y1))
    return {
        "y1": y1,
        "y0": y0,
        "d": d,
        "covariates": covariates,
        "i_weights": i_weights,
        "n": len(y1),
    }


def test_analytical_inference():
    setup = get_test_data()
    result = drdid_imp_panel(
        y1=setup["y1"],
        y0=setup["y0"],
        d=setup["d"],
        covariates=setup["covariates"],
        i_weights=setup["i_weights"],
        boot=False,
    )
    assert result.att is not None
    assert result.se is not None
    assert np.isclose(result.att, 1.0, atol=0.2)
    assert result.boots is None


def test_weighted_bootstrap():
    setup = get_test_data()
    result = drdid_imp_panel(
        y1=setup["y1"],
        y0=setup["y0"],
        d=setup["d"],
        covariates=setup["covariates"],
        i_weights=setup["i_weights"],
        boot=True,
        boot_type="weighted",
        nboot=100,
    )
    assert result.att is not None
    assert result.se is not None
    assert np.isclose(result.att, 1.0, atol=0.2)
    assert result.boots is not None
    assert len(result.boots) == 100


def test_multiplier_bootstrap():
    setup = get_test_data()
    result = drdid_imp_panel(
        y1=setup["y1"],
        y0=setup["y0"],
        d=setup["d"],
        covariates=setup["covariates"],
        i_weights=setup["i_weights"],
        boot=True,
        boot_type="multiplier",
        nboot=100,
    )
    assert result.att is not None
    assert result.se is not None
    assert np.isclose(result.att, 1.0, atol=0.2)
    assert result.boots is not None
    assert len(result.boots) == 100


def test_influence_function():
    setup = get_test_data()
    result = drdid_imp_panel(
        y1=setup["y1"],
        y0=setup["y0"],
        d=setup["d"],
        covariates=setup["covariates"],
        i_weights=setup["i_weights"],
        influence_func=True,
    )
    assert result.att_inf_func is not None
    assert len(result.att_inf_func) == setup["n"]


def test_no_covariates():
    setup = get_test_data()
    result = drdid_imp_panel(
        y1=setup["y1"],
        y0=setup["y0"],
        d=setup["d"],
        covariates=None,
        i_weights=setup["i_weights"],
        boot=False,
    )
    assert result.att is not None
    assert result.se is not None
    assert not np.isnan(result.att)
    assert not np.isnan(result.se)
