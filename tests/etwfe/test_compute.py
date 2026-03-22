"""Tests for ETWFE internal computation functions."""

import numpy as np
import polars as pl
import pytest

from tests.helpers import importorskip

importorskip("pyfixest")

from moderndid.etwfe.compute import (
    _invlink_and_deriv,
    _weighted_agg,
    build_etwfe_formula,
    prepare_etwfe_data,
    set_references,
)


def test_set_references_auto_tref(mpdta_data, base_config):
    config = set_references(base_config, mpdta_data)
    assert config.tref == 2003


@pytest.mark.parametrize("cgroup,expected_gref", [("notyet", 0), ("never", 0)])
def test_set_references_auto_gref(mpdta_data, base_config, cgroup, expected_gref):
    base_config.cgroup = cgroup
    config = set_references(base_config, mpdta_data)
    assert config.gref == expected_gref


def test_set_references_respects_explicit_refs(mpdta_data, base_config):
    base_config.tref = 2004
    base_config.gref = 2006
    config = set_references(base_config, mpdta_data)
    assert config.tref == 2004
    assert config.gref == 2006


@pytest.mark.parametrize("gref,expected_flag", [(None, True), (2007, False)])
def test_set_references_gref_min_flag(mpdta_data, base_config, gref, expected_flag):
    if gref is None:
        base_config.cgroup = "never"
    else:
        base_config.gref = gref
    config = set_references(base_config, mpdta_data)
    assert config._gref_min_flag is expected_flag


def test_set_references_no_control_group_raises(base_config):
    df = pl.DataFrame({"g": [1, 2], "t": [1, 2], "y": [1.0, 2.0]})
    base_config.gname = "g"
    base_config.tname = "t"
    base_config.cgroup = "never"
    with pytest.raises(ValueError, match="Could not identify"):
        set_references(base_config, df)


def test_set_references_notyet_fallback_to_max_group(base_config):
    df = pl.DataFrame({"g": [2, 3, 4], "t": [1, 2, 3], "y": [1.0, 2.0, 3.0]})
    base_config.gname = "g"
    base_config.tname = "t"
    base_config.cgroup = "notyet"
    config = set_references(base_config, df)
    assert config.gref == 4


def test_prepare_notyet_treatment_indicator(mpdta_data, base_config):
    config = set_references(base_config, mpdta_data)
    df = prepare_etwfe_data(mpdta_data, config)

    treated = df.filter((pl.col("_g") == 2004) & (pl.col("_t") == 2004))["_Dtreat"]
    assert (treated == 1.0).all()

    untreated = df.filter((pl.col("_g") == 2006) & (pl.col("_t") == 2004))["_Dtreat"]
    assert (untreated == 0.0).all()


def test_prepare_notyet_drops_ref_cohort_periods(mpdta_data, base_config):
    base_config.gref = 2007
    config = set_references(base_config, mpdta_data)
    df = prepare_etwfe_data(mpdta_data, config)

    ref_at_treat = df.filter(pl.col("_t") >= 2007)["_Dtreat"]
    assert ref_at_treat.is_null().all()


def test_prepare_notyet_control_group_untreated(mpdta_data, base_config):
    config = set_references(base_config, mpdta_data)
    df = prepare_etwfe_data(mpdta_data, config)

    control = df.filter((pl.col("_g") == 2007) & (pl.col("_t") < 2007))["_Dtreat"]
    assert (control == 0.0).all()


def test_prepare_never_treatment_indicator(mpdta_data, base_config):
    base_config.cgroup = "never"
    config = set_references(base_config, mpdta_data)
    df = prepare_etwfe_data(mpdta_data, config)

    never_treated = df.filter(pl.col("_g") == 0)["_Dtreat"]
    assert (never_treated == 0.0).all()

    pre_period = df.filter((pl.col("_g") == 2004) & (pl.col("_t") == 2003))["_Dtreat"]
    assert (pre_period == 0.0).all()

    post_period = df.filter((pl.col("_g") == 2004) & (pl.col("_t") == 2004))["_Dtreat"]
    assert (post_period == 1.0).all()


def test_prepare_creates_internal_columns(mpdta_data, base_config):
    config = set_references(base_config, mpdta_data)
    df = prepare_etwfe_data(mpdta_data, config)
    for col in ("_g", "_t", "_Dtreat"):
        assert col in df.columns


def test_prepare_internal_columns_are_float64(mpdta_data, base_config):
    config = set_references(base_config, mpdta_data)
    df = prepare_etwfe_data(mpdta_data, config)
    assert df["_g"].dtype == pl.Float64
    assert df["_t"].dtype == pl.Float64


def test_prepare_demeans_controls(mpdta_data, base_config):
    base_config.xformla = "~ lpop"
    config = set_references(base_config, mpdta_data)
    df = prepare_etwfe_data(mpdta_data, config)

    assert "lpop_dm" in df.columns
    cohort_means = df.group_by("first.treat").agg(pl.col("lpop_dm").mean().alias("mean_dm"))
    assert np.allclose(cohort_means["mean_dm"].to_numpy(), 0.0, atol=1e-10)


def test_prepare_no_controls_returns_empty_ctrls(mpdta_data, base_config):
    config = set_references(base_config, mpdta_data)
    prepare_etwfe_data(mpdta_data, config)
    assert config._ctrls == []


def test_formula_contains_treatment_interaction(mpdta_data, base_config):
    config = set_references(base_config, mpdta_data)
    prepare_etwfe_data(mpdta_data, config)
    formula = build_etwfe_formula(config)
    assert "_Dtreat:C(__etwfe_gcat):C(__etwfe_tcat)" in formula


@pytest.mark.parametrize(
    "fe,idname,has_pipe,has_explicit_cats",
    [
        ("vs", "countyreal", True, False),
        ("feo", "countyreal", True, False),
        ("none", None, False, True),
    ],
)
def test_formula_fe_modes(mpdta_data, base_config, fe, idname, has_pipe, has_explicit_cats):
    base_config.fe = fe
    base_config.idname = idname
    config = set_references(base_config, mpdta_data)
    prepare_etwfe_data(mpdta_data, config)
    formula = build_etwfe_formula(config)
    assert ("|" in formula) == has_pipe
    assert ("C(__etwfe_gcat) +" in formula or "C(__etwfe_gcat)" in formula.split("+")[-1]) == has_explicit_cats


def test_formula_vs_absorbs_idname(mpdta_data, base_config):
    config = set_references(base_config, mpdta_data)
    prepare_etwfe_data(mpdta_data, config)
    formula = build_etwfe_formula(config)
    assert "countyreal" in formula
    assert "year" in formula


def test_formula_with_controls(mpdta_data, base_config):
    base_config.xformla = "~ lpop"
    config = set_references(base_config, mpdta_data)
    prepare_etwfe_data(mpdta_data, config)
    formula = build_etwfe_formula(config)
    assert "lpop_dm" in formula


def test_formula_feo_with_controls_explicit(mpdta_data, base_config):
    base_config.xformla = "~ lpop"
    base_config.fe = "feo"
    config = set_references(base_config, mpdta_data)
    prepare_etwfe_data(mpdta_data, config)
    formula = build_etwfe_formula(config)
    assert "C(__etwfe_gcat):lpop" in formula
    assert "C(__etwfe_tcat):lpop" in formula


def test_formula_vs_includes_control_interactions(mpdta_data, base_config):
    base_config.xformla = "~ lpop"
    base_config.fe = "vs"
    config = set_references(base_config, mpdta_data)
    prepare_etwfe_data(mpdta_data, config)
    formula = build_etwfe_formula(config)
    assert "C(__etwfe_gcat):lpop" in formula
    assert "C(__etwfe_tcat):lpop" in formula


def test_prepare_xvar_creates_dm_columns(mpdta_data, base_config):
    base_config.xvar = "lpop"
    config = set_references(base_config, mpdta_data)
    df = prepare_etwfe_data(mpdta_data, config)
    assert "lpop_dm" in df.columns
    assert len(config._xvar_dm_cols) > 0


def test_prepare_xvar_creates_time_dummies(mpdta_data, base_config):
    base_config.xvar = "lpop"
    config = set_references(base_config, mpdta_data)
    df = prepare_etwfe_data(mpdta_data, config)
    assert len(config._xvar_time_dummies) > 0
    for col in config._xvar_time_dummies:
        assert col in df.columns


def test_formula_with_xvar(mpdta_data, base_config):
    base_config.xvar = "lpop"
    config = set_references(base_config, mpdta_data)
    prepare_etwfe_data(mpdta_data, config)
    formula = build_etwfe_formula(config)
    assert "lpop_dm" in formula
    for td in config._xvar_time_dummies:
        assert td in formula


@pytest.mark.parametrize(
    "family,eta,expected_mu",
    [
        ("gaussian", np.array([0.0, 1.0]), np.array([0.0, 1.0])),
        ("poisson", np.array([0.0, 1.0]), np.exp([0.0, 1.0])),
        ("logit", np.array([0.0]), np.array([0.5])),
        ("probit", np.array([0.0]), np.array([0.5])),
    ],
)
def test_invlink_and_deriv_mu(family, eta, expected_mu):
    mu, _ = _invlink_and_deriv(eta, family)
    np.testing.assert_allclose(mu, expected_mu, atol=1e-6)


@pytest.mark.parametrize("family", ["gaussian", "poisson", "logit", "probit"])
def test_invlink_and_deriv_positive_derivative(family):
    eta = np.array([-1.0, 0.0, 1.0])
    _, deriv = _invlink_and_deriv(eta, family)
    assert np.all(deriv > 0)


def test_invlink_and_deriv_poisson_mu_equals_deriv():
    mu, deriv = _invlink_and_deriv(np.array([0.0, 1.0, -1.0]), "poisson")
    np.testing.assert_array_equal(mu, deriv)


def test_invlink_and_deriv_unsupported_family():
    with pytest.raises(ValueError, match="Unsupported family"):
        _invlink_and_deriv(np.array([0.0]), "invalid")


def test_weighted_agg_zero_weights():
    slopes = np.array([1.0, 2.0])
    jac = np.eye(2)
    att, se = _weighted_agg(slopes, jac, np.array([0.0, 0.0]), np.eye(2))
    assert att == 0.0
    assert np.isnan(se)


def test_weighted_agg_none_vcov():
    slopes = np.array([1.0, 2.0])
    jac = np.eye(2)
    att, se = _weighted_agg(slopes, jac, np.ones(2), None)
    np.testing.assert_allclose(att, 1.5)
    assert np.isnan(se)
