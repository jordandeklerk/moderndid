"""Tests for outcome regression DiD."""

import re
import warnings
from importlib import import_module

import numpy as np
import pytest
import statsmodels.api as sm

from moderndid import ordid, reg_did_panel, reg_did_rc
from tests.helpers import importorskip

pl = importorskip("polars")


def test_ordid_panel_basic(nsw_data, nsw_units):
    treated = nsw_units.filter(pl.col("experimental") == 1)["dy"].to_numpy()
    comparison = nsw_units.filter(pl.col("experimental") == 0)["dy"].to_numpy()
    att = treated.mean() - comparison.mean()
    se = np.sqrt(treated.var() / treated.size + comparison.var() / comparison.size)

    result = ordid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        panel=True,
    )

    assert isinstance(result.att, float)
    assert isinstance(result.se, float)
    np.testing.assert_allclose(
        [result.att, result.se, result.lci, result.uci], [att, se, att - 1.96 * se, att + 1.96 * se], rtol=1e-10
    )
    assert result.boots is None
    assert result.att_inf_func is None
    assert result.call_params["panel"] is True
    assert result.args["type"] == "or"


def test_ordid_panel_with_covariates(nsw_data, nsw_units):
    covariates = ["age", "educ", "black", "married", "nodegree", "hisp"]
    comparison = nsw_units.filter(pl.col("experimental") == 0)
    treated = nsw_units.filter(pl.col("experimental") == 1)
    x_comparison = np.column_stack([np.ones(comparison.height), comparison.select(covariates).to_numpy()])
    x_treated = np.column_stack([np.ones(treated.height), treated.select(covariates).to_numpy()])
    fit = sm.OLS(comparison["dy"].to_numpy(), x_comparison).fit(cov_type="HC0")
    residuals = treated["dy"].to_numpy() - x_treated @ fit.params
    x_bar = x_treated.mean(axis=0)
    se = np.sqrt(residuals.var() / residuals.size + x_bar @ fit.cov_params() @ x_bar)

    result = ordid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ age + educ + black + married + nodegree + hisp",
        panel=True,
    )

    assert isinstance(result.att, float)
    np.testing.assert_allclose([result.att, result.se], [residuals.mean(), se], rtol=1e-10)
    assert result.call_params["xformla"] == "~ age + educ + black + married + nodegree + hisp"


def test_ordid_panel_with_weights(nsw_weighted, nsw_units):
    comparison = nsw_units.filter(pl.col("experimental") == 0)
    treated = nsw_units.filter(pl.col("experimental") == 1)
    x_comparison = np.column_stack([np.ones(comparison.height), comparison.select("age", "educ").to_numpy()])
    x_treated = np.column_stack([np.ones(treated.height), treated.select("age", "educ").to_numpy()])
    fit = sm.WLS(comparison["dy"].to_numpy(), x_comparison, weights=comparison["w"].to_numpy()).fit(cov_type="HC0")
    w = treated["w"].to_numpy()
    residuals = treated["dy"].to_numpy() - x_treated @ fit.params
    att = np.average(residuals, weights=w)
    x_bar = np.average(x_treated, axis=0, weights=w)
    se = np.sqrt(np.sum((w * (residuals - att)) ** 2) / np.sum(w) ** 2 + x_bar @ fit.cov_params() @ x_bar)

    result = ordid(
        data=nsw_weighted,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ age + educ",
        panel=True,
        weightsname="w",
    )

    assert isinstance(result.att, float)
    np.testing.assert_allclose([result.att, result.se], [att, se], rtol=1e-10)
    assert result.call_params["weightsname"] == "w"


def test_ordid_panel_with_influence_func(nsw_data, nsw_units):
    d = nsw_units["experimental"].to_numpy()
    dy = nsw_units["dy"].to_numpy()
    share = d.mean()
    expected = d * (dy - dy[d == 1].mean()) / share - (1 - d) * (dy - dy[d == 0].mean()) / (1 - share)

    result = ordid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        panel=True,
        inf_func=True,
    )

    assert result.att_inf_func is not None
    np.testing.assert_allclose(np.mean(result.att_inf_func), 0, atol=1e-10)
    np.testing.assert_allclose(result.att_inf_func, expected, rtol=1e-10, atol=1e-6)


def test_ordid_panel_bootstrap(nsw_data, unit_bootstrap_weights):
    result = ordid(
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
    )

    assert np.isfinite(result.att)
    assert result.boots is not None
    np.testing.assert_allclose(result.boots, np.full(50, result.att), rtol=1e-10)
    assert result.args["biters"] == 50
    assert result.args["boot"] is True
    assert result.args["boot_type"] == "weighted"


def test_ordid_panel_multiplier_bootstrap_draws_from_the_influence_function(monkeypatch, draws_from_weights, nsw_data):
    weights = np.where(np.random.default_rng(3).random((50, nsw_data["id"].n_unique())) < 0.5, -1.0, 1.0)
    monkeypatch.setattr(
        import_module("moderndid.drdid.estimators.reg_did_panel"), "mboot_did", draws_from_weights(weights)
    )

    result = ordid(
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
    )

    assert result.att_inf_func is not None
    assert result.boots is not None
    assert np.all(np.isfinite(result.att_inf_func))
    np.testing.assert_allclose(result.boots, weights @ result.att_inf_func / weights.shape[1], rtol=1e-12, atol=1e-9)
    assert result.args["boot"] is True
    assert result.args["boot_type"] == "multiplier"


def test_ordid_repeated_cross_section(nsw_weighted):
    y = nsw_weighted["re"].to_numpy()
    post = (nsw_weighted["year"] == 1978).to_numpy()
    d = nsw_weighted["experimental"].to_numpy() == 1
    w = nsw_weighted["w"].to_numpy()
    x = np.column_stack([np.ones(nsw_weighted.height), nsw_weighted.select("age", "educ", "black").to_numpy()])
    change = (
        sm.WLS(y[~d & post], x[~d & post], weights=w[~d & post]).fit().params
        - sm.WLS(y[~d & ~post], x[~d & ~post], weights=w[~d & ~post]).fit().params
    )
    att = (
        np.average(y[d & post], weights=w[d & post])
        - np.average(y[d & ~post], weights=w[d & ~post])
        - np.average(x[d] @ change, weights=w[d])
    )

    result = ordid(
        data=nsw_weighted,
        yname="re",
        tname="year",
        treatname="experimental",
        panel=False,
        xformla="~ age + educ + black",
        weightsname="w",
    )

    assert isinstance(result.att, float)
    np.testing.assert_allclose(result.att, att, rtol=1e-10)
    assert np.isfinite(result.se)
    assert result.se > 0
    assert result.call_params["panel"] is False
    assert result.args["panel"] is False


def test_ordid_rc_with_bootstrap(nsw_data, unit_bootstrap_weights):
    result = ordid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        panel=False,
        xformla="~ age + educ",
        boot=True,
        biters=50,
    )

    assert np.isfinite(result.att)
    assert result.boots is not None
    np.testing.assert_allclose(result.boots, np.full(50, result.att), rtol=1e-10)
    assert result.args["biters"] == 50


def test_ordid_rc_multiplier_bootstrap_draws_from_the_influence_function(monkeypatch, draws_from_weights, nsw_data):
    weights = np.where(np.random.default_rng(3).random((50, nsw_data.height)) < 0.5, -1.0, 1.0)
    monkeypatch.setattr(
        import_module("moderndid.drdid.estimators.reg_did_rc"), "mboot_did", draws_from_weights(weights)
    )

    result = ordid(
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
    )

    assert result.att_inf_func is not None
    assert result.boots is not None
    assert np.all(np.isfinite(result.att_inf_func))
    np.testing.assert_allclose(result.boots, weights @ result.att_inf_func / weights.shape[1], rtol=1e-12, atol=1e-9)
    assert result.args["boot_type"] == "multiplier"


def test_ordid_missing_id_col_panel():
    rng = np.random.default_rng(42)
    df = pl.DataFrame(
        {
            "y": rng.standard_normal(100),
            "time": np.repeat([0, 1], 50),
            "treat": np.tile([0, 1], 50),
        }
    )

    with pytest.raises(ValueError, match="idname must be provided when panel=True"):
        ordid(
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
def test_ordid_formula_variations(nsw_data, nsw_units, formula, term):
    d = nsw_units["experimental"].to_numpy() == 1
    dy = nsw_units["dy"].to_numpy()
    x = np.column_stack([np.ones(nsw_units.height), nsw_units.select("age", "educ", term.alias("term")).to_numpy()])
    coefficients = np.linalg.lstsq(x[~d], dy[~d], rcond=None)[0]

    result = ordid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla=formula,
        panel=True,
    )

    assert isinstance(result.att, float)
    np.testing.assert_allclose(result.att, np.mean(dy[d] - x[d] @ coefficients), rtol=1e-10)
    assert np.isfinite(result.se)
    assert result.se > 0


def test_ordid_call_params_stored(nsw_data):
    result = ordid(
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
    assert result.call_params["data_shape"] == nsw_data.shape


def test_ordid_args_output(nsw_data):
    result = ordid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        panel=True,
        boot=True,
        boot_type="weighted",
        biters=50,
    )

    assert result.args["panel"] is True
    assert result.args["normalized"] is True
    assert result.args["boot"] is True
    assert result.args["boot_type"] == "weighted"
    assert result.args["biters"] == 50
    assert result.args["type"] == "or"


def test_ordid_estimate_does_not_depend_on_the_row_order(nsw_data, nsw_units):
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
    }
    expected = ordid(data=small_data, **spec)

    result = ordid(data=small_data.sample(fraction=1.0, shuffle=True, seed=0), **spec)

    assert np.isfinite(expected.att)
    assert np.isfinite(expected.se)
    assert result.att == expected.att
    assert result.se == expected.se


def test_ordid_subset_columns(nsw_data):
    spec = {
        "yname": "re",
        "tname": "year",
        "treatname": "experimental",
        "idname": "id",
        "xformla": "~ age + educ",
        "panel": True,
    }
    expected = ordid(data=nsw_data, **spec)

    result = ordid(data=nsw_data.select("id", "year", "re", "experimental", "age", "educ"), **spec)

    assert isinstance(result.att, float)
    assert np.isfinite(expected.att)
    assert result.se > 0
    assert result.att == expected.att
    assert result.se == expected.se


@pytest.mark.parametrize("covariates_formula", [None, "~ 1"])
def test_ordid_no_covariates(nsw_weighted, nsw_units, covariates_formula):
    d = nsw_units["experimental"].to_numpy()
    dy = nsw_units["dy"].to_numpy()
    w = nsw_units["w"].to_numpy() / nsw_units["w"].mean()
    treated_mean = np.average(dy, weights=w * d)
    comparison_mean = np.average(dy, weights=w * (1 - d))
    share = np.mean(w * d)
    influence = w * d * (dy - treated_mean) / share - w * (1 - d) * (dy - comparison_mean) / (1 - share)

    result = ordid(
        data=nsw_weighted,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla=covariates_formula,
        panel=True,
        weightsname="w",
    )

    assert isinstance(result.att, float)
    np.testing.assert_allclose(
        [result.att, result.se],
        [treated_mean - comparison_mean, np.sqrt(np.sum(influence**2)) / dy.size],
        rtol=1e-10,
    )


def test_ordid_categorical_covariates(nsw_data, nsw_units):
    age_group = (
        pl.when(pl.col("age") <= 25)
        .then(pl.lit("young"))
        .when(pl.col("age") <= 35)
        .then(pl.lit("middle"))
        .otherwise(pl.lit("old"))
        .alias("age_group")
    )
    groups = nsw_units.select(age_group)["age_group"].to_numpy()
    d = nsw_units["experimental"].to_numpy() == 1
    dy = nsw_units["dy"].to_numpy()
    x = np.column_stack(
        [np.ones(nsw_units.height), groups == "old", groups == "young", nsw_units.select("educ", "black").to_numpy()]
    ).astype(float)
    coefficients = np.linalg.lstsq(x[~d], dy[~d], rcond=None)[0]

    result = ordid(
        data=nsw_data.with_columns(age_group),
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ C(age_group) + educ + black",
        panel=True,
    )

    assert isinstance(result.att, float)
    np.testing.assert_allclose(result.att, np.mean(dy[d] - x[d] @ coefficients), rtol=1e-10)
    assert np.isfinite(result.se)
    assert result.se > 0


def test_ordid_missing_values_raise_in_panel():
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
        ordid(
            data=df,
            yname="y",
            tname="time",
            treatname="treat",
            idname="id",
            xformla="~ x1",
            panel=True,
        )


def test_ordid_unbalanced_panel_warning(nsw_data, nsw_units):
    unbalanced = nsw_data.filter(~((pl.col("id") == 15995) & (pl.col("year") == 1978)))
    balanced = nsw_units.filter(pl.col("id") != 15995)
    treated = balanced.filter(pl.col("experimental") == 1)["dy"].to_numpy()
    comparison = balanced.filter(pl.col("experimental") == 0)["dy"].to_numpy()
    att = treated.mean() - comparison.mean()
    se = np.sqrt(treated.var() / treated.size + comparison.var() / comparison.size)

    with pytest.warns(UserWarning, match="Panel data is unbalanced"):
        result = ordid(
            data=unbalanced,
            yname="re",
            tname="year",
            treatname="experimental",
            idname="id",
            panel=True,
        )

    np.testing.assert_allclose([result.att, result.se], [att, se], rtol=1e-10)
    assert result.args["n_obs"] == 32832


def test_ordid_balanced_panel_raises_no_warning(nsw_data):
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        ordid(data=nsw_data, yname="re", tname="year", treatname="experimental", idname="id", panel=True)

    assert [str(w.message) for w in recorded if issubclass(w.category, UserWarning)] == []


def test_ordid_more_than_two_periods_error():
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
        ordid(
            data=df,
            yname="y",
            tname="time",
            treatname="treat",
            idname="id",
            panel=True,
        )


def test_ordid_panel_plain_covariates_without_formulaic(nsw_data, without_formulaic):
    post = nsw_data.filter(pl.col("year") == nsw_data["year"].max()).sort("id")
    pre = nsw_data.filter(pl.col("year") == nsw_data["year"].min()).sort("id")
    covariates = np.column_stack([np.ones(post.height), post.select("age", "educ", "black").to_numpy()])
    expected = reg_did_panel(
        y1=post["re"].to_numpy(),
        y0=pre["re"].to_numpy(),
        d=post["experimental"].to_numpy(),
        covariates=covariates,
        i_weights=np.ones(post.height),
    )

    result = ordid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ age + educ + black",
    )

    np.testing.assert_allclose([result.att, result.se], [expected.att, expected.se], rtol=1e-12)


def test_ordid_rc_plain_covariates_without_formulaic(nsw_data, without_formulaic):
    covariates = np.column_stack([np.ones(nsw_data.height), nsw_data.select("age", "educ", "black").to_numpy()])
    expected = reg_did_rc(
        y=nsw_data["re"].to_numpy(),
        post=(nsw_data["year"] == nsw_data["year"].max()).cast(pl.Int64).to_numpy(),
        d=nsw_data["experimental"].to_numpy(),
        covariates=covariates,
        i_weights=np.ones(nsw_data.height),
    )

    result = ordid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        panel=False,
        xformla="~ age + educ + black",
    )

    np.testing.assert_allclose([result.att, result.se], [expected.att, expected.se], rtol=1e-12)


def test_ordid_rejects_repeated_unit_periods(nsw_duplicated):
    message = (
        "The value of idname must be unique (by tname). Some units are observed more than once in a period. "
        "Rows repeat for the (id, year) pair (15995, 1975)."
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        ordid(data=nsw_duplicated, yname="re", tname="year", treatname="experimental", idname="id", panel=True)


@pytest.mark.parametrize("nsw_one_infinite", [("re", float("inf")), ("age", float("-inf"))], indirect=True)
def test_ordid_drops_infinite_rows_of_repeated_cross_sections(nsw_one_infinite):
    spec = {"yname": "re", "tname": "year", "treatname": "experimental", "idname": "id", "xformla": "~ age + educ"}
    expected = ordid(
        data=nsw_one_infinite.filter(pl.all_horizontal(pl.col("re", "age").is_finite())), panel=False, **spec
    )

    with pytest.warns(UserWarning, match="^Dropped 1 rows from original data due to missing values$"):
        result = ordid(data=nsw_one_infinite, panel=False, **spec)

    assert result.att == expected.att
    assert result.se == expected.se
    assert result.args["n_obs"] == 32833


@pytest.mark.parametrize("panel", [True, False])
def test_ordid_rejects_weights_without_positive_mean(nsw_data, panel):
    message = "The weights variable 'w' must be non-negative with a positive mean."

    with pytest.raises(ValueError, match=re.escape(message)):
        ordid(
            data=nsw_data.with_columns(pl.lit(0.0).alias("w")),
            yname="re",
            tname="year",
            treatname="experimental",
            idname="id",
            weightsname="w",
            panel=panel,
        )


def test_ordid_rejects_the_removed_n_boot_argument(nsw_data):
    with pytest.raises(TypeError, match="unexpected keyword argument 'n_boot'"):
        ordid(data=nsw_data, yname="re", tname="year", treatname="experimental", idname="id", boot=True, n_boot=50)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"treatname": None}, "treatname is required. Please specify the treatment column."),
        ({"boot_type": "wild"}, "boot_type='wild' is not valid. Must be 'weighted' or 'multiplier'."),
        ({"biters": 0}, "biters=0 is not valid. Must be a positive integer."),
        ({"biters": 2.5}, "biters=2.5 is not valid. Must be a positive integer."),
    ],
)
def test_ordid_rejects_an_invalid_argument_by_name(nsw_data, changes, message):
    spec = {"yname": "re", "tname": "year", "treatname": "experimental", "idname": "id"} | changes

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        ordid(data=nsw_data, **spec)


def test_ordid_repeated_cross_sections_ignore_idname(nsw_data):
    spec = {
        "data": nsw_data,
        "yname": "re",
        "tname": "year",
        "treatname": "experimental",
        "xformla": "~ age + educ",
        "panel": False,
    }
    expected = ordid(**spec)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        result = ordid(idname="id", **spec)

    assert [str(w.message) for w in recorded if issubclass(w.category, UserWarning)] == []
    assert np.isfinite(expected.att)
    assert result.att == expected.att
    assert result.se == expected.se
