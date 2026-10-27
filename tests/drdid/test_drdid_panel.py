"""Tests for the locally efficient doubly robust DiD estimator for panel data."""

import re
import warnings

import numpy as np
import pytest

from moderndid import drdid_panel


@pytest.mark.parametrize("covariates", [None, "with_covariates"])
def test_drdid_panel_basic(covariates):
    n_units = 200
    rng = np.random.default_rng(42)

    d = rng.binomial(1, 0.5, n_units)

    if covariates == "with_covariates":
        x = rng.standard_normal((n_units, 3))
        x = np.column_stack([np.ones(n_units), x])
    else:
        x = None

    y0 = rng.standard_normal(n_units)
    y1 = y0 + 2 * d + rng.standard_normal(n_units)

    result = drdid_panel(y1=y1, y0=y0, d=d, covariates=x)

    assert isinstance(result.att, float)
    assert isinstance(result.se, float)
    assert result.se > 0
    assert result.lci < result.uci
    assert result.boots is None
    assert result.att_inf_func is None


def test_drdid_panel_with_weights():
    n_units = 200
    rng = np.random.default_rng(42)

    d = rng.binomial(1, 0.5, n_units)
    x = np.column_stack([np.ones(n_units), rng.standard_normal((n_units, 2))])
    y0 = rng.standard_normal(n_units)
    y1 = y0 + 2 * d + rng.standard_normal(n_units)
    i_weights = rng.uniform(0.5, 1.5, n_units)

    result = drdid_panel(y1=y1, y0=y0, d=d, covariates=x, i_weights=i_weights)

    assert isinstance(result.att, float)
    assert result.se > 0


def test_drdid_panel_influence_function():
    n_units = 200
    rng = np.random.default_rng(42)

    d = rng.binomial(1, 0.5, n_units)
    x = np.column_stack([np.ones(n_units), rng.standard_normal((n_units, 2))])
    y0 = rng.standard_normal(n_units)
    y1 = y0 + 2 * d + rng.standard_normal(n_units)

    result = drdid_panel(y1=y1, y0=y0, d=d, covariates=x, influence_func=True)

    assert result.att_inf_func is not None
    assert len(result.att_inf_func) == n_units
    assert np.abs(np.mean(result.att_inf_func)) < 0.1


@pytest.mark.parametrize("boot_type", ["weighted", "multiplier"])
def test_drdid_panel_bootstrap(boot_type):
    n_units = 100
    rng = np.random.default_rng(42)

    d = rng.binomial(1, 0.5, n_units)
    x = np.column_stack([np.ones(n_units), rng.standard_normal((n_units, 2))])
    y0 = rng.standard_normal(n_units)
    y1 = y0 + 2 * d + rng.standard_normal(n_units)

    result = drdid_panel(y1=y1, y0=y0, d=d, covariates=x, boot=True, boot_type=boot_type, biters=20)

    assert result.boots is not None
    assert len(result.boots) == result.args["biters"]
    assert result.se > 0
    assert result.lci < result.uci


def test_drdid_panel_no_treated():
    n_units = 100
    rng = np.random.default_rng(42)

    d = np.zeros(n_units)
    x = np.column_stack([np.ones(n_units), rng.standard_normal((n_units, 2))])
    y0 = rng.standard_normal(n_units)
    y1 = y0 + rng.standard_normal(n_units)
    message = "No units found in the treated group. Pass data in which some units have a treatment indicator of 1."

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        drdid_panel(y1=y1, y0=y0, d=d, covariates=x)


def test_drdid_panel_all_treated():
    n_units = 100
    rng = np.random.default_rng(42)

    d = np.ones(n_units)
    x = np.column_stack([np.ones(n_units), rng.standard_normal((n_units, 2))])
    y0 = rng.standard_normal(n_units)
    y1 = y0 + 2 + rng.standard_normal(n_units)
    message = "No units found in the comparison group. Pass data in which some units have a treatment indicator of 0."

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        drdid_panel(y1=y1, y0=y0, d=d, covariates=x)


def test_drdid_panel_trimming():
    n_units = 200
    rng = np.random.default_rng(42)

    d = rng.binomial(1, 0.05, n_units)
    x = np.column_stack([np.ones(n_units), rng.standard_normal((n_units, 2))])
    y0 = rng.standard_normal(n_units)
    y1 = y0 + 2 * d + rng.standard_normal(n_units)

    result = drdid_panel(y1=y1, y0=y0, d=d, covariates=x, trim_level=0.9)

    assert isinstance(result.att, float)
    assert result.se > 0


def test_drdid_panel_negative_weights():
    n_units = 100
    rng = np.random.default_rng(42)

    d = rng.binomial(1, 0.5, n_units)
    x = np.column_stack([np.ones(n_units), rng.standard_normal((n_units, 2))])
    y0 = rng.standard_normal(n_units)
    y1 = y0 + 2 * d + rng.standard_normal(n_units)
    i_weights = rng.standard_normal(n_units)

    with pytest.raises(ValueError, match="non-negative"):
        drdid_panel(y1=y1, y0=y0, d=d, covariates=x, i_weights=i_weights)


def test_drdid_panel_dgp():
    n_units = 2000
    rng = np.random.default_rng(42)

    x1 = rng.normal(0, 1, n_units)
    x2 = rng.normal(0, 1, n_units)
    x3 = rng.normal(0, 1, n_units)
    x4 = rng.normal(0, 1, n_units)

    y00 = x1 + x2 + rng.normal(0, 1, n_units)

    y10 = y00 + x1 + x3 + rng.normal(0, 1, n_units)

    att_true = 1

    d_propensity = 1 / (1 + np.exp(-(x1 + x3)))
    d = (rng.uniform(size=n_units) < d_propensity).astype(int)

    y1 = y10 + att_true * d
    y0 = y00

    covariates = np.column_stack((np.ones(n_units), x1, x2, x3, x4))

    result = drdid_panel(y1=y1, y0=y0, d=d, covariates=covariates)

    assert 0.8 < result.att < 1.2
    assert result.se > 0
    assert result.args["estMethod"] == "trad"


def test_drdid_panel_rejects_the_removed_nboot_argument():
    rng = np.random.default_rng(42)
    d = rng.binomial(1, 0.5, 50)
    y0 = rng.standard_normal(50)

    with pytest.raises(TypeError, match="unexpected keyword argument 'nboot'"):
        drdid_panel(y1=y0 + d, y0=y0, d=d, covariates=None, boot=True, nboot=20)


@pytest.mark.parametrize(
    ("redundant", "message"),
    [
        ("duplicate", "The propensity score model dropped 1 covariate that is collinear with the others."),
        ("intercept", "The propensity score model dropped 1 covariate that is collinear with the others."),
        ("zero", "The propensity score model dropped 1 covariate that is collinear with the others."),
        ("affine", "The propensity score model dropped 1 covariate that is collinear with the others."),
        ("duplicate_and_zero", "The propensity score model dropped 2 covariates that are collinear with the others."),
    ],
    ids=["duplicate", "intercept", "zero", "affine", "duplicate_and_zero"],
)
def test_drdid_panel_warns_about_a_redundant_propensity_covariate_before_its_outcome_regression_fails(
    two_period_specs, redundant_columns, redundant, message
):
    spec = two_period_specs["panel"]
    covariates = spec["covariates"]
    padded = np.column_stack([covariates[:, :2], redundant_columns[redundant], covariates[:, 2:]])
    singular = (
        "The regression design matrix of the comparison group is singular. Consider removing covariates that are "
        "constant or collinear among the comparison units."
    )

    with (
        pytest.warns(UserWarning, match=f"^{re.escape(message)}$") as record,
        pytest.raises(ValueError, match=f"^{re.escape(singular)}$"),
    ):
        drdid_panel(**{**spec, "covariates": padded})

    assert len(record) == 1


@pytest.mark.parametrize("level", [0.0, 1e6])
def test_drdid_panel_keeps_every_covariate_of_a_full_rank_design_without_a_warning(two_period_specs, level):
    spec = two_period_specs["panel"]
    shifted = spec["covariates"] + np.array([0.0, 0.0, level])

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = drdid_panel(**{**spec, "covariates": shifted})
    expected = drdid_panel(**spec)
    without_last = drdid_panel(**{**spec, "covariates": spec["covariates"][:, :2]})

    assert [str(w.message) for w in record] == []
    np.testing.assert_allclose(result.att, expected.att, rtol=1e-10)
    assert abs(result.att - without_last.att) > 1e-3
