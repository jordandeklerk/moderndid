"""Tests for inverse propensity weighted DiD."""

import re
import warnings
from importlib import import_module

import numpy as np
import pytest
import statsmodels.api as sm

from moderndid import drdid, ipw_did_panel, ipwdid, ordid, std_ipw_did_rc
from tests.helpers import importorskip

pl = importorskip("polars")


@pytest.mark.parametrize("est_method", ["ipw", "std_ipw"])
def test_ipwdid_panel_basic(nsw_data, nsw_units, est_method):
    treated = nsw_units.filter(pl.col("experimental") == 1)["dy"].to_numpy()
    comparison = nsw_units.filter(pl.col("experimental") == 0)["dy"].to_numpy()
    att = treated.mean() - comparison.mean()
    se = np.sqrt(treated.var() / treated.size + comparison.var() / comparison.size)

    result = ipwdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        panel=True,
        est_method=est_method,
    )

    assert isinstance(result.att, float)
    assert isinstance(result.se, float)
    np.testing.assert_allclose(
        [result.att, result.se, result.lci, result.uci], [att, se, att - 1.96 * se, att + 1.96 * se], rtol=1e-10
    )
    assert result.boots is None
    assert result.att_inf_func is None
    assert result.call_params["panel"] is True
    assert result.call_params["est_method"] == est_method
    assert result.args["type"] == "ipw"
    assert result.args["estMethod"] == est_method


def test_ipwdid_panel_with_covariates(nsw_data, nsw_units):
    d = nsw_units["experimental"].to_numpy()
    x = np.column_stack(
        [np.ones(nsw_units.height), nsw_units.select("age", "educ", "black", "married", "nodegree", "hisp").to_numpy()]
    )
    score = sm.GLM(d, x, family=sm.families.Binomial()).fit(tol=1e-12).predict(x)
    att = np.mean((d - score) / (1 - score) * nsw_units["dy"].to_numpy()) / np.mean(d)

    result = ipwdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ age + educ + black + married + nodegree + hisp",
        panel=True,
        est_method="ipw",
    )

    assert isinstance(result.att, float)
    np.testing.assert_allclose(result.att, att, rtol=1e-8)
    assert np.isfinite(result.se)
    assert result.se > 0
    assert result.call_params["xformla"] == "~ age + educ + black + married + nodegree + hisp"


@pytest.mark.parametrize("est_method", ["ipw", "std_ipw"])
def test_ipwdid_panel_with_weights(nsw_weighted, nsw_units, est_method):
    d = nsw_units["experimental"].to_numpy()
    dy = nsw_units["dy"].to_numpy()
    w = nsw_units["w"].to_numpy()
    x = np.column_stack([np.ones(nsw_units.height), nsw_units.select("age", "educ").to_numpy()])
    score = sm.GLM(d, x, family=sm.families.Binomial(), var_weights=w).fit(tol=1e-12).predict(x)
    comparison_weights = w * score / (1 - score) * (1 - d)
    expected = {
        "ipw": np.sum((w * d - comparison_weights) * dy) / np.sum(w * d),
        "std_ipw": np.average(dy, weights=w * d) - np.average(dy, weights=comparison_weights),
    }

    result = ipwdid(
        data=nsw_weighted,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ age + educ",
        panel=True,
        weightsname="w",
        est_method=est_method,
    )

    assert isinstance(result.att, float)
    np.testing.assert_allclose(result.att, expected[est_method], rtol=1e-8)
    assert np.isfinite(result.se)
    assert result.se > 0
    assert result.call_params["weightsname"] == "w"


def test_ipwdid_panel_with_influence_func(nsw_data, nsw_units):
    d = nsw_units["experimental"].to_numpy()
    dy = nsw_units["dy"].to_numpy()
    share = d.mean()
    expected = d * (dy - dy[d == 1].mean()) / share - (1 - d) * (dy - dy[d == 0].mean()) / (1 - share)

    result = ipwdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        panel=True,
        inf_func=True,
        est_method="ipw",
    )

    assert result.att_inf_func is not None
    assert np.all(np.isfinite(result.att_inf_func))
    np.testing.assert_allclose(result.att_inf_func, expected, rtol=1e-10, atol=1e-6)


@pytest.mark.parametrize("est_method", ["ipw", "std_ipw"])
def test_ipwdid_panel_bootstrap(nsw_data, unit_bootstrap_weights, est_method):
    result = ipwdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ age + educ",
        panel=True,
        boot=True,
        boot_type="weighted",
        biters=50,
        est_method=est_method,
    )

    assert np.isfinite(result.att)
    assert result.boots is not None
    np.testing.assert_allclose(result.boots, np.full(50, result.att), rtol=1e-10)
    assert result.args["biters"] == 50
    assert result.args["boot"] is True
    assert result.args["boot_type"] == "weighted"


@pytest.mark.parametrize(
    ("est_method", "module"),
    [
        ("ipw", "moderndid.drdid.estimators.ipw_did_panel"),
        ("std_ipw", "moderndid.drdid.estimators.std_ipw_did_panel"),
    ],
)
def test_ipwdid_panel_multiplier_bootstrap_draws_from_the_influence_function(
    monkeypatch, draws_from_weights, nsw_data, est_method, module
):
    weights = np.where(np.random.default_rng(3).random((50, nsw_data["id"].n_unique())) < 0.5, -1.0, 1.0)
    monkeypatch.setattr(import_module(module), "mboot_did", draws_from_weights(weights))

    result = ipwdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ age + educ",
        panel=True,
        boot=True,
        boot_type="multiplier",
        biters=50,
        inf_func=True,
        est_method=est_method,
    )

    assert result.att_inf_func is not None
    assert result.boots is not None
    assert np.all(np.isfinite(result.att_inf_func))
    np.testing.assert_allclose(result.boots, weights @ result.att_inf_func / weights.shape[1], rtol=1e-12, atol=1e-9)
    assert result.args["boot"] is True
    assert result.args["boot_type"] == "multiplier"


@pytest.mark.parametrize("est_method", ["ipw", "std_ipw"])
def test_ipwdid_repeated_cross_section(nsw_weighted, est_method):
    y = nsw_weighted["re"].to_numpy()
    post = (nsw_weighted["year"] == 1978).cast(pl.Float64).to_numpy()
    d = nsw_weighted["experimental"].to_numpy()
    w = nsw_weighted["w"].to_numpy()
    x = np.column_stack([np.ones(nsw_weighted.height), nsw_weighted.select("age", "educ", "black").to_numpy()])
    score = sm.GLM(d, x, family=sm.families.Binomial(), var_weights=w).fit(tol=1e-12).predict(x)
    odds = score / (1 - score)
    share_post = np.average(post, weights=w)
    expected = {
        "ipw": np.sum(w * (d - score) / (1 - score) * (post - share_post) / (share_post * (1 - share_post)) * y)
        / np.sum(w * d),
        "std_ipw": np.average(y, weights=w * d * post)
        - np.average(y, weights=w * d * (1 - post))
        - np.average(y, weights=w * odds * (1 - d) * post)
        + np.average(y, weights=w * odds * (1 - d) * (1 - post)),
    }

    result = ipwdid(
        data=nsw_weighted,
        yname="re",
        tname="year",
        treatname="experimental",
        panel=False,
        xformla="~ age + educ + black",
        weightsname="w",
        est_method=est_method,
    )

    assert isinstance(result.att, float)
    np.testing.assert_allclose(result.att, expected[est_method], rtol=1e-8)
    assert np.isfinite(result.se)
    assert result.se > 0
    assert result.call_params["panel"] is False
    assert result.args["panel"] is False
    assert result.args["estMethod"] == est_method


@pytest.mark.parametrize("est_method", ["ipw", "std_ipw"])
def test_ipwdid_rc_with_bootstrap(nsw_data, unit_bootstrap_weights, unit_generator_bootstrap_weights, est_method):
    result = ipwdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        panel=False,
        xformla="~ age + educ",
        boot=True,
        biters=50,
        est_method=est_method,
    )

    assert np.isfinite(result.att)
    assert result.boots is not None
    np.testing.assert_allclose(result.boots, np.full(50, result.att), rtol=1e-10)
    assert result.args["biters"] == 50


@pytest.mark.parametrize(
    ("est_method", "module"),
    [
        ("ipw", "moderndid.drdid.estimators.ipw_did_rc"),
        ("std_ipw", "moderndid.drdid.estimators.std_ipw_did_rc"),
    ],
)
def test_ipwdid_rc_multiplier_bootstrap_draws_from_the_influence_function(
    monkeypatch, draws_from_weights, nsw_data, est_method, module
):
    weights = np.where(np.random.default_rng(3).random((50, nsw_data.height)) < 0.5, -1.0, 1.0)
    monkeypatch.setattr(import_module(module), "mboot_did", draws_from_weights(weights))

    result = ipwdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        panel=False,
        xformla="~ age + educ",
        boot=True,
        boot_type="multiplier",
        biters=50,
        inf_func=True,
        est_method=est_method,
    )

    assert result.att_inf_func is not None
    assert result.boots is not None
    assert np.all(np.isfinite(result.att_inf_func))
    np.testing.assert_allclose(result.boots, weights @ result.att_inf_func / weights.shape[1], rtol=1e-12, atol=1e-9)
    assert result.args["boot_type"] == "multiplier"


@pytest.mark.parametrize("est_method", ["ipw", "std_ipw"])
def test_ipwdid_repeated_cross_sections_trim_comparison_rows_at_the_trim_level(
    trimmed_two_period_data, trimmed_two_period_cross_sections, est_method
):
    data = trimmed_two_period_data
    d = data["d"]
    post = data["post"]
    y = data["y"]
    score = sm.GLM(d, data["x"], family=sm.families.Binomial()).fit(tol=1e-12).predict(data["x"])
    comparison_weights = score / (1 - score) * (1 - d) * (score < 0.9)
    share_post = post.mean()
    expected = {
        "ipw": np.mean((d - comparison_weights) * (post - share_post) / (share_post * (1 - share_post)) * y)
        / np.mean(d),
        "std_ipw": np.average(y, weights=d * post)
        - np.average(y, weights=d * (1 - post))
        - np.average(y, weights=comparison_weights * post)
        + np.average(y, weights=comparison_weights * (1 - post)),
    }

    result = ipwdid(
        data=trimmed_two_period_cross_sections,
        yname="y",
        tname="year",
        treatname="d",
        xformla="~ x1",
        panel=False,
        trim_level=0.9,
        est_method=est_method,
    )

    np.testing.assert_allclose(result.att, expected[est_method], rtol=1e-8)
    assert np.isfinite(result.se)
    assert result.args["trim_level"] == 0.9


@pytest.mark.parametrize("trim_level", [0.9, 0.95])
def test_ipwdid_trim_level(trimmed_two_period_data, trimmed_two_period_panel, trim_level):
    data = trimmed_two_period_data
    d = data["d"]
    dy = data["y1"] - data["y0"]
    score = sm.GLM(d, data["x"], family=sm.families.Binomial()).fit(tol=1e-12).predict(data["x"])
    kept = (d == 0) & (score < trim_level)
    att = (np.sum(d * dy) - np.sum(score[kept] / (1 - score[kept]) * dy[kept])) / np.sum(d)

    result = ipwdid(
        data=trimmed_two_period_panel,
        yname="y",
        tname="year",
        treatname="d",
        idname="id",
        xformla="~ x1",
        panel=True,
        trim_level=trim_level,
        est_method="ipw",
    )

    assert isinstance(result.att, float)
    np.testing.assert_allclose(result.att, att, rtol=1e-8)
    assert result.args["trim_level"] == trim_level


def test_ipwdid_missing_id_col_panel():
    rng = np.random.default_rng(42)
    df = pl.DataFrame(
        {
            "y": rng.standard_normal(100),
            "time": np.repeat([0, 1], 50),
            "treat": np.tile([0, 1], 50),
        }
    )

    with pytest.raises(ValueError, match="idname must be provided when panel=True"):
        ipwdid(
            data=df,
            yname="y",
            tname="time",
            treatname="treat",
            panel=True,
        )


@pytest.mark.parametrize(
    ("formula", "term"),
    [
        ("~ age + educ + age:educ", pl.col("age") * pl.col("educ")),
        ("~ age + I(age**2) + educ", pl.col("age") ** 2),
    ],
)
def test_ipwdid_formula_variations(nsw_data, nsw_units, formula, term):
    d = nsw_units["experimental"].to_numpy()
    x = np.column_stack([np.ones(nsw_units.height), nsw_units.select("age", "educ", term.alias("term")).to_numpy()])
    score = sm.GLM(d, x, family=sm.families.Binomial()).fit(tol=1e-12).predict(x)
    att = np.mean((d - score) / (1 - score) * nsw_units["dy"].to_numpy()) / np.mean(d)

    result = ipwdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla=formula,
        panel=True,
        est_method="ipw",
    )

    assert isinstance(result.att, float)
    np.testing.assert_allclose(result.att, att, rtol=1e-8)
    assert np.isfinite(result.se)
    assert result.se > 0


def test_ipwdid_call_params_stored(nsw_data):
    result = ipwdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ age + educ",
        panel=True,
        boot=True,
        boot_type="multiplier",
        biters=50,
        inf_func=True,
        est_method="std_ipw",
        trim_level=0.99,
    )

    assert result.call_params["yname"] == "re"
    assert result.call_params["tname"] == "year"
    assert result.call_params["treatname"] == "experimental"
    assert result.call_params["idname"] == "id"
    assert result.call_params["xformla"] == "~ age + educ"
    assert result.call_params["panel"] is True
    assert result.call_params["boot"] is True
    assert result.call_params["boot_type"] == "multiplier"
    assert result.call_params["biters"] == 50
    assert result.call_params["inf_func"] is True
    assert result.call_params["est_method"] == "std_ipw"
    assert result.call_params["trim_level"] == 0.99
    assert result.call_params["data_shape"] == nsw_data.shape


def test_ipwdid_args_output(nsw_data):
    result = ipwdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        panel=True,
        boot=True,
        boot_type="weighted",
        biters=50,
        est_method="ipw",
    )

    assert result.args["panel"] is True
    assert result.args["normalized"] is False
    assert result.args["boot"] is True
    assert result.args["boot_type"] == "weighted"
    assert result.args["biters"] == 50
    assert result.args["type"] == "ipw"
    assert result.args["estMethod"] == "ipw"


@pytest.mark.parametrize("panel", [True, False])
@pytest.mark.parametrize(
    ("est_method", "normalized", "weight_type"),
    [
        ("ipw", False, "Weight type: Unnormalized (Horvitz-Thompson-type)"),
        ("std_ipw", True, "Weight type: Normalized (Hajek-type estimator)"),
    ],
)
def test_ipwdid_records_the_weights_it_used(nsw_data, panel, est_method, normalized, weight_type):
    result = ipwdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ age + educ",
        panel=panel,
        est_method=est_method,
    )

    assert result.args["normalized"] is normalized
    assert weight_type in [line.strip() for line in str(result).splitlines()]


def test_ipwdid_estimate_does_not_depend_on_the_row_order(nsw_data, nsw_units):
    treated_ids = nsw_units.filter(pl.col("experimental") == 1)["id"].head(50).to_list()
    ids = treated_ids + nsw_units.filter(pl.col("experimental") == 0)["id"].head(50).to_list()
    small_data = nsw_data.filter(pl.col("id").is_in(ids))
    spec = {
        "yname": "re",
        "tname": "year",
        "treatname": "experimental",
        "idname": "id",
        "xformla": "~ age + educ",
        "panel": True,
        "est_method": "ipw",
    }
    expected = ipwdid(data=small_data, **spec)

    result = ipwdid(data=small_data.sample(fraction=1.0, shuffle=True, seed=0), **spec)

    assert np.isfinite(expected.att)
    assert np.isfinite(expected.se)
    assert result.att == expected.att
    assert result.se == expected.se


def test_ipwdid_subset_columns(nsw_data):
    spec = {
        "yname": "re",
        "tname": "year",
        "treatname": "experimental",
        "idname": "id",
        "xformla": "~ age + educ",
        "panel": True,
        "est_method": "std_ipw",
    }
    expected = ipwdid(data=nsw_data, **spec)

    result = ipwdid(data=nsw_data.select("id", "year", "re", "experimental", "age", "educ"), **spec)

    assert isinstance(result.att, float)
    assert np.isfinite(expected.att)
    assert result.se > 0
    assert result.att == expected.att
    assert result.se == expected.se


@pytest.mark.parametrize("covariates_formula", [None, "~ 1"])
def test_ipwdid_no_covariates(nsw_weighted, nsw_units, covariates_formula):
    d = nsw_units["experimental"].to_numpy()
    dy = nsw_units["dy"].to_numpy()
    w = nsw_units["w"].to_numpy() / nsw_units["w"].mean()
    treated_mean = np.average(dy, weights=w * d)
    comparison_mean = np.average(dy, weights=w * (1 - d))
    share = np.mean(w * d)
    influence = w * d * (dy - treated_mean) / share - w * (1 - d) * (dy - comparison_mean) / (1 - share)

    result = ipwdid(
        data=nsw_weighted,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla=covariates_formula,
        panel=True,
        weightsname="w",
        est_method="ipw",
    )

    assert isinstance(result.att, float)
    np.testing.assert_allclose(
        [result.att, result.se],
        [treated_mean - comparison_mean, np.sqrt(np.sum(influence**2)) / dy.size],
        rtol=1e-10,
    )


def test_ipwdid_categorical_covariates(nsw_data, nsw_units):
    age_group = (
        pl.when(pl.col("age") <= 25)
        .then(pl.lit("young"))
        .when(pl.col("age") <= 35)
        .then(pl.lit("middle"))
        .otherwise(pl.lit("old"))
        .alias("age_group")
    )
    groups = nsw_units.select(age_group)["age_group"].to_numpy()
    d = nsw_units["experimental"].to_numpy()
    x = np.column_stack(
        [np.ones(nsw_units.height), groups == "old", groups == "young", nsw_units.select("educ", "black").to_numpy()]
    ).astype(float)
    score = sm.GLM(d, x, family=sm.families.Binomial()).fit(tol=1e-12).predict(x)
    att = np.mean((d - score) / (1 - score) * nsw_units["dy"].to_numpy()) / np.mean(d)

    result = ipwdid(
        data=nsw_data.with_columns(age_group),
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ C(age_group) + educ + black",
        panel=True,
        est_method="ipw",
    )

    assert isinstance(result.att, float)
    np.testing.assert_allclose(result.att, att, rtol=1e-8)
    assert np.isfinite(result.se)
    assert result.se > 0


def test_ipwdid_missing_values_raise_in_panel():
    rng = np.random.default_rng(42)
    y_vals = rng.standard_normal(100)
    y_vals[5] = np.nan
    df = pl.DataFrame(
        {
            "id": np.repeat(range(50), 2),
            "time": np.tile([0, 1], 50),
            "y": y_vals,
            "treat": np.repeat([0, 1], 50),
            "x1": rng.standard_normal(100),
        }
    )

    with pytest.raises(ValueError, match="Missing values found in panel data"):
        ipwdid(
            data=df,
            yname="y",
            tname="time",
            treatname="treat",
            idname="id",
            xformla="~ x1",
            panel=True,
        )


def test_ipwdid_unbalanced_panel_warning(nsw_data, nsw_units):
    unbalanced = nsw_data.filter(~((pl.col("id") == 15995) & (pl.col("year") == 1978)))
    balanced = nsw_units.filter(pl.col("id") != 15995)
    treated = balanced.filter(pl.col("experimental") == 1)["dy"].to_numpy()
    comparison = balanced.filter(pl.col("experimental") == 0)["dy"].to_numpy()
    att = treated.mean() - comparison.mean()
    se = np.sqrt(treated.var() / treated.size + comparison.var() / comparison.size)

    with pytest.warns(UserWarning, match="Panel data is unbalanced"):
        result = ipwdid(
            data=unbalanced,
            yname="re",
            tname="year",
            treatname="experimental",
            idname="id",
            panel=True,
        )

    assert isinstance(result.att, float)
    np.testing.assert_allclose([result.att, result.se], [att, se], rtol=1e-10)
    assert result.args["n_obs"] == 32832


def test_ipwdid_balanced_panel_raises_no_warning(nsw_data):
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        ipwdid(data=nsw_data, yname="re", tname="year", treatname="experimental", idname="id", panel=True)

    assert [str(w.message) for w in recorded if issubclass(w.category, UserWarning)] == []


def test_ipwdid_more_than_two_periods_error():
    rng = np.random.default_rng(42)
    df = pl.DataFrame(
        {
            "id": np.repeat(range(50), 3),
            "time": np.tile([0, 1, 2], 50),
            "y": rng.standard_normal(150),
            "treat": np.repeat([0, 1], 75),
        }
    )

    with pytest.raises(
        ValueError, match=re.escape("This package currently supports only two time periods (pre and post).")
    ):
        ipwdid(
            data=df,
            yname="y",
            tname="time",
            treatname="treat",
            idname="id",
            panel=True,
        )


@pytest.mark.parametrize("method", ["ipw", "std_ipw"])
def test_ipwdid_estimators_consistency(nsw_data, nsw_units, method):
    treated_ids = nsw_units.filter(pl.col("experimental") == 1)["id"].head(200).to_list()
    ids = treated_ids + nsw_units.filter(pl.col("experimental") == 0)["id"].head(200).to_list()
    spec = {
        "data": nsw_data.filter(pl.col("id").is_in(ids)),
        "yname": "re",
        "tname": "year",
        "treatname": "experimental",
        "idname": "id",
        "xformla": "~ age + educ + black",
        "est_method": method,
        "trim_level": 0.99,
    }

    panel = ipwdid(panel=True, **spec)
    cross_sections = ipwdid(panel=False, **spec)

    assert np.isfinite(panel.att)
    assert panel.se > 0
    assert cross_sections.se > 0
    np.testing.assert_allclose(cross_sections.att, panel.att, rtol=1e-8)


def test_ipwdid_comparison_std_vs_regular(nsw_data, nsw_units):
    treated_ids = nsw_units.filter(pl.col("experimental") == 1)["id"].head(300).to_list()
    ids = treated_ids + nsw_units.filter(pl.col("experimental") == 0)["id"].head(300).to_list()
    units = nsw_units.filter(pl.col("id").is_in(ids))
    d = units["experimental"].to_numpy()
    dy = units["dy"].to_numpy()
    x = np.column_stack([np.ones(units.height), units.select("age", "educ").to_numpy()])
    score = sm.GLM(d, x, family=sm.families.Binomial()).fit(tol=1e-12).predict(x)
    comparison_weights = score / (1 - score) * (1 - d) * (score < 0.99)
    spec = {
        "data": nsw_data.filter(pl.col("id").is_in(ids)),
        "yname": "re",
        "tname": "year",
        "treatname": "experimental",
        "idname": "id",
        "xformla": "~ age + educ",
        "panel": True,
        "trim_level": 0.99,
    }

    ipw_result = ipwdid(est_method="ipw", **spec)
    std_ipw_result = ipwdid(est_method="std_ipw", **spec)

    np.testing.assert_allclose(
        [ipw_result.att, std_ipw_result.att],
        [
            np.sum((d - comparison_weights) * dy) / np.sum(d),
            np.average(dy, weights=d) - np.average(dy, weights=comparison_weights),
        ],
        rtol=1e-8,
    )
    assert ipw_result.se > 0
    assert std_ipw_result.se > 0


def test_ipwdid_extreme_trimming(trimmed_two_period_data, trimmed_two_period_panel):
    data = trimmed_two_period_data
    d = data["d"]
    dy = data["y1"] - data["y0"]
    score = sm.GLM(d, data["x"], family=sm.families.Binomial()).fit(tol=1e-12).predict(data["x"])
    comparison_weights = score / (1 - score) * (1 - d) * (score < 0.9)

    result = ipwdid(
        data=trimmed_two_period_panel,
        yname="y",
        tname="year",
        treatname="d",
        idname="id",
        xformla="~ x1",
        panel=True,
        est_method="std_ipw",
        trim_level=0.9,
    )

    assert isinstance(result.att, float)
    np.testing.assert_allclose(
        result.att, np.average(dy, weights=d) - np.average(dy, weights=comparison_weights), rtol=1e-8
    )
    assert np.isfinite(result.se)
    assert result.se > 0


def test_ipwdid_comparison_with_other_estimators(nsw_data, nsw_units):
    treated_ids = nsw_units.filter(pl.col("experimental") == 1)["id"].head(100).to_list()
    ids = treated_ids + nsw_units.filter(pl.col("experimental") == 0)["id"].head(100).to_list()
    units = nsw_units.filter(pl.col("id").is_in(ids))
    treated = units.filter(pl.col("experimental") == 1)["dy"].to_numpy()
    comparison = units.filter(pl.col("experimental") == 0)["dy"].to_numpy()
    att = treated.mean() - comparison.mean()
    se = np.sqrt(treated.var() / treated.size + comparison.var() / comparison.size)
    spec = {
        "data": nsw_data.filter(pl.col("id").is_in(ids)),
        "yname": "re",
        "tname": "year",
        "treatname": "experimental",
        "idname": "id",
        "panel": True,
    }

    ipw_result = ipwdid(est_method="ipw", **spec)
    dr_result = drdid(est_method="trad", **spec)
    or_result = ordid(**spec)

    assert all(isinstance(r.att, float) for r in [ipw_result, dr_result, or_result])
    np.testing.assert_allclose([[r.att, r.se] for r in [ipw_result, dr_result, or_result]], [[att, se]] * 3, rtol=1e-10)


def test_ipwdid_panel_plain_covariates_without_formulaic(nsw_data, without_formulaic):
    post = nsw_data.filter(pl.col("year") == nsw_data["year"].max()).sort("id")
    pre = nsw_data.filter(pl.col("year") == nsw_data["year"].min()).sort("id")
    covariates = np.column_stack([np.ones(post.height), post.select("age", "educ", "black").to_numpy()])
    expected = ipw_did_panel(
        y1=post["re"].to_numpy(),
        y0=pre["re"].to_numpy(),
        d=post["experimental"].to_numpy(),
        covariates=covariates,
        i_weights=np.ones(post.height),
    )

    result = ipwdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ age + educ + black",
        est_method="ipw",
    )

    np.testing.assert_allclose([result.att, result.se], [expected.att, expected.se], rtol=1e-12)


def test_ipwdid_rc_plain_covariates_without_formulaic(nsw_data, without_formulaic):
    covariates = np.column_stack([np.ones(nsw_data.height), nsw_data.select("age", "educ", "black").to_numpy()])
    expected = std_ipw_did_rc(
        y=nsw_data["re"].to_numpy(),
        post=(nsw_data["year"] == nsw_data["year"].max()).cast(pl.Int64).to_numpy(),
        d=nsw_data["experimental"].to_numpy(),
        covariates=covariates,
        i_weights=np.ones(nsw_data.height),
    )

    result = ipwdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        panel=False,
        xformla="~ age + educ + black",
        est_method="std_ipw",
    )

    np.testing.assert_allclose([result.att, result.se], [expected.att, expected.se], rtol=1e-12)


def test_ipwdid_rejects_repeated_unit_periods(nsw_duplicated):
    message = (
        "The value of idname must be unique (by tname). Some units are observed more than once in a period. "
        "Rows repeat for the (id, year) pair (15995, 1975)."
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        ipwdid(data=nsw_duplicated, yname="re", tname="year", treatname="experimental", idname="id", panel=True)


@pytest.mark.parametrize("nsw_one_infinite", [("re", float("inf")), ("age", float("-inf"))], indirect=True)
def test_ipwdid_drops_infinite_rows_of_repeated_cross_sections(nsw_one_infinite):
    spec = {"yname": "re", "tname": "year", "treatname": "experimental", "idname": "id", "xformla": "~ age + educ"}
    expected = ipwdid(
        data=nsw_one_infinite.filter(pl.all_horizontal(pl.col("re", "age").is_finite())), panel=False, **spec
    )

    with pytest.warns(UserWarning, match="^Dropped 1 rows from original data due to missing values$"):
        result = ipwdid(data=nsw_one_infinite, panel=False, **spec)

    assert result.att == expected.att
    assert result.se == expected.se
    assert result.args["n_obs"] == 32833


@pytest.mark.parametrize("panel", [True, False])
def test_ipwdid_rejects_weights_without_positive_mean(nsw_data, panel):
    message = "The weights variable 'w' must be non-negative with a positive mean."

    with pytest.raises(ValueError, match=re.escape(message)):
        ipwdid(
            data=nsw_data.with_columns(pl.lit(0.0).alias("w")),
            yname="re",
            tname="year",
            treatname="experimental",
            idname="id",
            weightsname="w",
            panel=panel,
        )


def test_ipwdid_rejects_the_removed_n_boot_argument(nsw_data):
    with pytest.raises(TypeError, match="unexpected keyword argument 'n_boot'"):
        ipwdid(data=nsw_data, yname="re", tname="year", treatname="experimental", idname="id", boot=True, n_boot=50)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"treatname": None}, "treatname is required. Please specify the treatment column."),
        ({"est_method": "imp"}, "est_method='imp' is not valid. Must be 'ipw' or 'std_ipw'."),
        ({"boot_type": "wild"}, "boot_type='wild' is not valid. Must be 'weighted' or 'multiplier'."),
        ({"biters": 0}, "biters=0 is not valid. Must be a positive integer."),
        ({"biters": 2.5}, "biters=2.5 is not valid. Must be a positive integer."),
        ({"trim_level": 0}, "trim_level=0 is not valid. Must be between 0 and 1 (exclusive)."),
        ({"trim_level": 1}, "trim_level=1 is not valid. Must be between 0 and 1 (exclusive)."),
    ],
)
def test_ipwdid_rejects_an_invalid_argument_by_name(nsw_data, changes, message):
    spec = {"yname": "re", "tname": "year", "treatname": "experimental", "idname": "id"} | changes

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        ipwdid(data=nsw_data, **spec)


def test_ipwdid_repeated_cross_sections_ignore_idname(nsw_data):
    spec = {
        "data": nsw_data,
        "yname": "re",
        "tname": "year",
        "treatname": "experimental",
        "xformla": "~ age + educ",
        "panel": False,
    }
    expected = ipwdid(**spec)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        result = ipwdid(idname="id", **spec)

    assert [str(w.message) for w in recorded if issubclass(w.category, UserWarning)] == []
    assert np.isfinite(expected.att)
    assert result.att == expected.att
    assert result.se == expected.se
