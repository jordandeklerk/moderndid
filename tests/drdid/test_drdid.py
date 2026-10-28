"""Tests for doubly robust DiD."""

import re
import warnings

import numpy as np
import pytest
from scipy import stats

from moderndid import (
    drdid,
    drdid_imp_local_rc,
    drdid_imp_panel,
    drdid_imp_rc,
    drdid_panel,
    drdid_rc,
    drdid_trad_rc,
    ipw_did_panel,
    ipw_did_rc,
    ipwdid,
    ordid,
    reg_did_panel,
    reg_did_rc,
    std_ipw_did_panel,
    std_ipw_did_rc,
    twfe_did_panel,
    twfe_did_rc,
)
from tests.helpers import importorskip

pl = importorskip("polars")


@pytest.mark.parametrize("xformla", [None, "~ 1"])
@pytest.mark.parametrize("est_method", ["imp", "trad"])
def test_drdid_panel_without_covariates_gives_the_difference_in_mean_changes(
    nsw_data, nsw_changes, est_method, xformla
):
    treated = nsw_changes.filter(pl.col("experimental") == 1)["dre"].to_numpy()
    comparison = nsw_changes.filter(pl.col("experimental") == 0)["dre"].to_numpy()
    att = treated.mean() - comparison.mean()
    se = np.sqrt(treated.var() / treated.size + comparison.var() / comparison.size)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = drdid(
            data=nsw_data,
            yname="re",
            tname="year",
            treatname="experimental",
            idname="id",
            xformla=xformla,
            panel=True,
            est_method=est_method,
        )

    assert isinstance(result.att, float)
    assert isinstance(result.se, float)
    np.testing.assert_allclose(
        [result.att, result.se, result.lci, result.uci], [att, se, att - 1.96 * se, att + 1.96 * se], rtol=1e-12
    )
    assert result.boots is None
    assert result.att_inf_func is None
    assert result.call_params["panel"] is True
    assert result.call_params["est_method"] == est_method
    assert result.args["type"] == "dr"
    assert result.args["estMethod"] == est_method
    assert [str(warning.message) for warning in caught] == []


@pytest.mark.parametrize("est_method", ["imp", "trad"])
def test_drdid_panel_with_binary_covariates_gives_the_exact_matching_estimate(
    nsw_data, nsw_changes, panel_matching_did, est_method
):
    att, inf_func = panel_matching_did(
        nsw_changes["dre"].to_numpy(),
        nsw_changes["experimental"].to_numpy(),
        (nsw_changes["black"] + 2 * nsw_changes["hisp"]).to_numpy(),
        np.ones(nsw_changes.height),
    )

    result = drdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ black + hisp",
        panel=True,
        inf_func=True,
        est_method=est_method,
    )

    np.testing.assert_allclose([result.att, result.se], [att, np.sqrt(np.sum(inf_func**2)) / inf_func.size], rtol=1e-10)
    assert np.shape(result.att_inf_func) == inf_func.shape
    np.testing.assert_allclose(result.att_inf_func, inf_func, rtol=1e-9, atol=1e-6)
    assert result.call_params["xformla"] == "~ black + hisp"


@pytest.mark.parametrize(
    ("est_method", "data_fixture"), [("imp", "nsw_data"), ("trad", "nsw_data"), ("trad", "nsw_thinned")]
)
def test_drdid_repeated_cross_sections_with_binary_covariates_give_the_exact_matching_estimate(
    request, rc_matching_did, est_method, data_fixture
):
    data = request.getfixturevalue(data_fixture)
    att, inf_func = rc_matching_did(
        data["re"].to_numpy(),
        (data["year"] == 1978).cast(pl.Int64).to_numpy(),
        data["experimental"].to_numpy(),
        (data["black"] + 2 * data["hisp"]).to_numpy(),
        np.ones(data.height),
    )

    result = drdid(
        data=data,
        yname="re",
        tname="year",
        treatname="experimental",
        xformla="~ black + hisp",
        panel=False,
        inf_func=True,
        est_method=est_method,
    )

    np.testing.assert_allclose([result.att, result.se], [att, np.sqrt(np.sum(inf_func**2)) / inf_func.size], rtol=1e-10)
    assert np.shape(result.att_inf_func) == inf_func.shape
    np.testing.assert_allclose(result.att_inf_func, inf_func, rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize("est_method", ["imp", "trad"])
def test_drdid_panel_weights_enter_the_estimate_and_its_standard_error(
    nsw_data, nsw_changes, panel_matching_did, est_method
):
    weight = (1 + (pl.col("id") % 5) / 2).alias("weight")
    units = nsw_changes.with_columns(weight)
    att, inf_func = panel_matching_did(
        units["dre"].to_numpy(),
        units["experimental"].to_numpy(),
        (units["black"] + 2 * units["hisp"]).to_numpy(),
        units["weight"].to_numpy(),
    )

    result = drdid(
        data=nsw_data.with_columns(weight),
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ black + hisp",
        panel=True,
        weightsname="weight",
        est_method=est_method,
    )

    np.testing.assert_allclose([result.att, result.se], [att, np.sqrt(np.sum(inf_func**2)) / inf_func.size], rtol=1e-10)
    assert result.call_params["weightsname"] == "weight"


@pytest.mark.parametrize("est_method", ["imp", "trad"])
def test_drdid_repeated_cross_section_weights_enter_the_estimate_and_its_standard_error(
    nsw_data, rc_matching_did, est_method
):
    data = nsw_data.with_columns((1 + (pl.col("id") % 5) / 2).alias("weight"))
    att, inf_func = rc_matching_did(
        data["re"].to_numpy(),
        (data["year"] == 1978).cast(pl.Int64).to_numpy(),
        data["experimental"].to_numpy(),
        (data["black"] + 2 * data["hisp"]).to_numpy(),
        data["weight"].to_numpy(),
    )

    result = drdid(
        data=data,
        yname="re",
        tname="year",
        treatname="experimental",
        xformla="~ black + hisp",
        panel=False,
        weightsname="weight",
        est_method=est_method,
    )

    np.testing.assert_allclose([result.att, result.se], [att, np.sqrt(np.sum(inf_func**2)) / inf_func.size], rtol=1e-10)


@pytest.mark.parametrize("est_method", ["imp", "trad"])
@pytest.mark.parametrize("panel", [True, False])
def test_drdid_weighted_bootstrap_reestimates_the_att_under_each_draw_of_weights(
    trimmed_two_period_data,
    trimmed_two_period_frames,
    panel_matching_did,
    rc_matching_did,
    seeded_exponential_draws,
    panel,
    est_method,
):
    data = trimmed_two_period_data
    n = data["d"].size
    weights = np.vstack([np.ones(n), np.random.default_rng(8).exponential(1.0, (5, n))])
    if panel:
        estimates = [panel_matching_did(data["y1"] - data["y0"], data["d"], np.zeros(n), w)[0] for w in weights]
    else:
        estimates = [rc_matching_did(data["y"], data["post"], data["d"], np.zeros(n), w)[0] for w in weights]
    quartiles = np.percentile(np.array(estimates[1:]) - estimates[0], [25, 75])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = drdid(
            data=trimmed_two_period_frames["panel" if panel else "repeated_cross_sections"],
            yname="y",
            tname="year",
            treatname="d",
            idname="id" if panel else None,
            panel=panel,
            boot=True,
            boot_type="weighted",
            biters=5,
            est_method=est_method,
        )

    assert np.shape(result.boots) == (5,)
    np.testing.assert_allclose(result.boots, estimates[1:], rtol=1e-10)
    np.testing.assert_allclose(
        [result.att, result.se],
        [estimates[0], (quartiles[1] - quartiles[0]) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))],
        rtol=1e-10,
    )
    assert result.args["boot"] is True
    assert result.args["boot_type"] == "weighted"
    assert [str(warning.message) for warning in caught] == []


@pytest.mark.parametrize("est_method", ["imp", "trad"])
@pytest.mark.parametrize("panel", [True, False])
def test_drdid_multiplier_bootstrap_draws_center_on_zero_and_return_the_influence_function(
    trimmed_two_period_data,
    trimmed_two_period_frames,
    panel_matching_did,
    rc_matching_did,
    seeded_default_rng,
    panel,
    est_method,
):
    data = trimmed_two_period_data
    n = data["d"].size
    biters = 499
    if panel:
        _, inf_func = panel_matching_did(data["y1"] - data["y0"], data["d"], np.zeros(n), np.ones(n))
    else:
        _, inf_func = rc_matching_did(data["y"], data["post"], data["d"], np.zeros(n), np.ones(n))
    se = np.sqrt(np.sum(inf_func**2)) / n

    result = drdid(
        data=trimmed_two_period_frames["panel" if panel else "repeated_cross_sections"],
        yname="y",
        tname="year",
        treatname="d",
        idname="id" if panel else None,
        panel=panel,
        boot=True,
        boot_type="multiplier",
        biters=biters,
        inf_func=True,
        est_method=est_method,
    )

    assert np.shape(result.boots) == (biters,)
    assert abs(np.mean(result.boots)) < 5 * se / np.sqrt(biters)
    np.testing.assert_allclose(
        result.se,
        np.subtract(*np.percentile(result.boots, [75, 25])) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25)),
        rtol=1e-12,
    )
    assert np.shape(result.att_inf_func) == inf_func.shape
    np.testing.assert_allclose(result.att_inf_func, inf_func, rtol=1e-9, atol=1e-12)
    assert result.args["boot"] is True
    assert result.args["boot_type"] == "multiplier"


@pytest.mark.parametrize("est_method", ["imp", "trad"])
def test_drdid_repeated_cross_sections_without_covariates_give_the_difference_of_cell_means(nsw_data, est_method):
    y = nsw_data["re"].to_numpy()
    post = nsw_data["year"].to_numpy() == 1978
    treated = nsw_data["experimental"].to_numpy() == 1
    cells = [y[treated & post], y[treated & ~post], y[~treated & post], y[~treated & ~post]]
    att = cells[0].mean() - cells[1].mean() - cells[2].mean() + cells[3].mean()
    se = np.sqrt(sum(cell.var() / cell.size for cell in cells))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = drdid(
            data=nsw_data,
            yname="re",
            tname="year",
            treatname="experimental",
            panel=False,
            est_method=est_method,
        )

    assert isinstance(result.att, float)
    np.testing.assert_allclose(
        [result.att, result.se, result.lci, result.uci], [att, se, att - 1.96 * se, att + 1.96 * se], rtol=1e-12
    )
    assert result.att_inf_func is None
    assert result.args["panel"] is False
    assert result.args["estMethod"] == est_method
    assert [str(warning.message) for warning in caught] == []


@pytest.mark.parametrize(
    ("panel", "est_method", "estimator"),
    [
        (True, "imp", drdid_imp_panel),
        (True, "trad", drdid_panel),
        (False, "imp", drdid_imp_local_rc),
        (False, "trad", drdid_rc),
    ],
)
def test_drdid_passes_trim_level_to_the_estimator(
    trimmed_two_period_data, trimmed_two_period_frames, panel, est_method, estimator
):
    data = trimmed_two_period_data
    outcomes = {"y1": data["y1"], "y0": data["y0"]} if panel else {"y": data["y"], "post": data["post"]}
    expected = estimator(d=data["d"], covariates=data["x"], trim_level=0.9, **outcomes)
    untrimmed = estimator(d=data["d"], covariates=data["x"], **outcomes)

    result = drdid(
        data=trimmed_two_period_frames["panel" if panel else "repeated_cross_sections"],
        yname="y",
        tname="year",
        treatname="d",
        idname="id" if panel else None,
        xformla="~ x1",
        panel=panel,
        trim_level=0.9,
        est_method=est_method,
    )

    assert abs(expected.att - untrimmed.att) > 1e-2
    assert np.all(np.isfinite([expected.att, expected.se]))
    assert expected.se > 0
    np.testing.assert_allclose([result.att, result.se], [expected.att, expected.se], rtol=1e-12)
    assert result.args["trim_level"] == 0.9


@pytest.mark.parametrize("panel", [True, False])
@pytest.mark.parametrize("est_method", ["imp_local", "trad_local", "local"])
def test_drdid_rejects_an_unknown_est_method(nsw_data, est_method, panel):
    message = f"est_method='{est_method}' is not valid. Must be 'imp' or 'trad'."

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        drdid(
            data=nsw_data,
            yname="re",
            tname="year",
            treatname="experimental",
            idname="id",
            panel=panel,
            est_method=est_method,
        )


def test_drdid_missing_id_col_panel():
    rng = np.random.default_rng(42)
    df = pl.DataFrame(
        {
            "y": rng.standard_normal(100),
            "time": np.repeat([0, 1], 50),
            "treat": np.tile([0, 1], 50),
        }
    )

    with pytest.raises(ValueError, match="idname must be provided when panel=True"):
        drdid(
            data=df,
            yname="y",
            tname="time",
            treatname="treat",
            panel=True,
        )


def test_drdid_call_params_stored(nsw_data, seeded_default_rng):
    result = drdid(
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
        est_method="imp",
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
    assert result.call_params["est_method"] == "imp"
    assert result.call_params["trim_level"] == 0.99
    assert result.call_params["weightsname"] is None
    assert result.call_params["data_shape"] == nsw_data.shape


def test_drdid_args_output(trimmed_two_period_frames, seeded_exponential_draws):
    result = drdid(
        data=trimmed_two_period_frames["panel"],
        yname="y",
        tname="year",
        treatname="d",
        idname="id",
        panel=True,
        boot=True,
        boot_type="weighted",
        biters=10,
        est_method="trad",
    )

    assert result.args["panel"] is True
    assert result.args["normalized"] is True
    assert result.args["boot"] is True
    assert result.args["boot_type"] == "weighted"
    assert result.args["biters"] == 10
    assert result.args["type"] == "dr"
    assert result.args["estMethod"] == "trad"


@pytest.mark.parametrize("est_method", ["imp", "trad"])
def test_drdid_panel_pairs_the_two_periods_of_each_unit_by_id(nsw_data, nsw_changes, panel_matching_did, est_method):
    shuffled = nsw_data[np.random.default_rng(3).permutation(nsw_data.height)]
    att, inf_func = panel_matching_did(
        nsw_changes["dre"].to_numpy(),
        nsw_changes["experimental"].to_numpy(),
        (nsw_changes["black"] + 2 * nsw_changes["hisp"]).to_numpy(),
        np.ones(nsw_changes.height),
    )

    result = drdid(
        data=shuffled,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ black + hisp",
        panel=True,
        est_method=est_method,
    )

    np.testing.assert_allclose([result.att, result.se], [att, np.sqrt(np.sum(inf_func**2)) / inf_func.size], rtol=1e-10)


@pytest.mark.parametrize("panel", [True, False])
def test_drdid_ignores_data_columns_outside_the_model(nsw_data, panel):
    spec = {
        "yname": "re",
        "tname": "year",
        "treatname": "experimental",
        "idname": "id",
        "xformla": "~ age + educ",
        "panel": panel,
    }

    result = drdid(data=nsw_data, **spec)
    expected = drdid(data=nsw_data.select("id", "year", "re", "experimental", "age", "educ"), **spec)

    assert np.all(np.isfinite([result.att, result.se]))
    assert result.se > 0
    assert result.att == expected.att
    assert result.se == expected.se


def test_drdid_categorical_covariate_gives_the_exact_matching_estimate(nsw_data, nsw_changes, panel_matching_did):
    age_group = (
        pl.when(pl.col("age") <= 25)
        .then(pl.lit("young"))
        .when(pl.col("age") <= 35)
        .then(pl.lit("middle"))
        .otherwise(pl.lit("old"))
        .alias("age_group")
    )
    units = nsw_changes.with_columns(age_group)
    att, inf_func = panel_matching_did(
        units["dre"].to_numpy(),
        units["experimental"].to_numpy(),
        units["age_group"].to_numpy(),
        np.ones(units.height),
    )

    result = drdid(
        data=nsw_data.with_columns(age_group),
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ C(age_group)",
        panel=True,
        est_method="imp",
    )

    np.testing.assert_allclose([result.att, result.se], [att, np.sqrt(np.sum(inf_func**2)) / inf_func.size], rtol=1e-10)


def test_drdid_missing_values_error():
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

    with pytest.raises(ValueError, match=r"^Missing values found in panel data\. Dropped 1 rows\."):
        drdid(
            data=df,
            yname="y",
            tname="time",
            treatname="treat",
            idname="id",
            xformla="~ x1",
            panel=True,
        )


def test_drdid_unbalanced_panel_warns_and_drops_the_units_seen_once(nsw_data, nsw_changes):
    unbalanced = nsw_data.filter(~((pl.col("id") == 15995) & (pl.col("year") == 1978)))
    balanced = nsw_changes.filter(pl.col("id") != 15995)
    treated = balanced.filter(pl.col("experimental") == 1)["dre"].to_numpy()
    comparison = balanced.filter(pl.col("experimental") == 0)["dre"].to_numpy()
    message = "Panel data is unbalanced. Dropping units with incomplete observations."

    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$"):
        result = drdid(data=unbalanced, yname="re", tname="year", treatname="experimental", idname="id", panel=True)

    np.testing.assert_allclose(
        [result.att, result.se],
        [
            treated.mean() - comparison.mean(),
            np.sqrt(treated.var() / treated.size + comparison.var() / comparison.size),
        ],
        rtol=1e-12,
    )


def test_drdid_more_than_two_periods_error():
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
        drdid(
            data=df,
            yname="y",
            tname="time",
            treatname="treat",
            idname="id",
            panel=True,
        )


@pytest.mark.parametrize(
    ("panel", "method", "estimator", "other"),
    [
        (True, "imp", drdid_imp_panel, drdid_panel),
        (True, "trad", drdid_panel, drdid_imp_panel),
        (False, "imp", drdid_imp_local_rc, drdid_rc),
        (False, "trad", drdid_rc, drdid_imp_local_rc),
    ],
)
def test_drdid_runs_the_estimator_that_panel_and_est_method_select(nsw_data, panel, method, estimator, other):
    first_units = nsw_data.filter(pl.col("year") == 1978).sort("id").group_by("experimental").head(50)
    small_data = nsw_data.filter(pl.col("id").is_in(first_units["id"].to_list()))
    post = small_data.filter(pl.col("year") == 1978).sort("id")
    pre = small_data.filter(pl.col("year") == 1975).sort("id")
    rows = post if panel else small_data
    outcomes = (
        {"y1": post["re"].to_numpy(), "y0": pre["re"].to_numpy()}
        if panel
        else {"y": rows["re"].to_numpy(), "post": (rows["year"] == 1978).cast(pl.Int64).to_numpy()}
    )
    covariates = np.column_stack([np.ones(rows.height), rows.select("age", "educ").to_numpy()])
    expected = estimator(d=rows["experimental"].to_numpy(), covariates=covariates, **outcomes)
    alternative = other(d=rows["experimental"].to_numpy(), covariates=covariates, **outcomes)

    result = drdid(
        data=small_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id" if panel else None,
        xformla="~ age + educ",
        panel=panel,
        est_method=method,
    )

    assert isinstance(result.att, float)
    assert np.all(np.isfinite([result.att, result.se]))
    assert result.se > 0
    assert abs(expected.att - alternative.att) > 100
    np.testing.assert_allclose([result.att, result.se], [expected.att, expected.se], rtol=1e-12)


def test_drdid_matches_ordid_without_covariates(nsw_data, nsw_changes):
    treated = nsw_changes.filter(pl.col("experimental") == 1)["dre"].to_numpy()
    comparison = nsw_changes.filter(pl.col("experimental") == 0)["dre"].to_numpy()
    att = treated.mean() - comparison.mean()
    se = np.sqrt(treated.var() / treated.size + comparison.var() / comparison.size)

    or_result = ordid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        panel=True,
    )
    dr_result = drdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        panel=True,
        est_method="trad",
    )

    np.testing.assert_allclose([or_result.att, or_result.se], [att, se], rtol=1e-12)
    np.testing.assert_allclose([dr_result.att, dr_result.se], [att, se], rtol=1e-12)


@pytest.mark.parametrize(
    "transformed, explicit",
    [
        ("~ age + I(age**2) + educ", "~ age + age_sq + educ"),
        ("~ age + educ + age:educ", "~ age + educ + age_educ"),
        ("~ center(age) + educ", "~ age_c + educ"),
    ],
)
@pytest.mark.parametrize("panel", [True, False])
def test_drdid_transformed_covariates_match_explicit_columns(nsw_data, transformed, explicit, panel):
    data = nsw_data.with_columns(
        (pl.col("age") ** 2).alias("age_sq"),
        (pl.col("age") * pl.col("educ")).alias("age_educ"),
        (pl.col("age") - pl.col("age").mean()).alias("age_c"),
    )
    spec = {"yname": "re", "tname": "year", "treatname": "experimental", "panel": panel}
    if panel:
        spec["idname"] = "id"

    result = drdid(data=data, xformla=transformed, **spec)
    expected = drdid(data=data, xformla=explicit, **spec)

    assert np.all(np.isfinite([result.att, result.se]))
    assert result.se > 0
    np.testing.assert_allclose(result.att, expected.att, rtol=1e-10)
    np.testing.assert_allclose(result.se, expected.se, rtol=1e-10)


@pytest.mark.parametrize("panel", [True, False])
def test_drdid_dotted_and_backticked_covariate_names(nsw_data, panel):
    data = nsw_data.with_columns(pl.col("age").alias("age.yrs"), pl.col("educ").alias("years of school"))
    spec = {"yname": "re", "tname": "year", "treatname": "experimental", "panel": panel}
    if panel:
        spec["idname"] = "id"

    result = drdid(data=data, xformla="~ age.yrs + `years of school`", **spec)
    expected = drdid(data=data, xformla="~ age + educ", **spec)

    assert result.att == expected.att
    assert result.se == expected.se


@pytest.mark.parametrize("panel", [True, False])
def test_drdid_reads_decimal_outcome_treatment_covariates_and_weights_as_doubles(nsw_data, panel):
    data = nsw_data.with_columns((1 + (pl.col("id") % 4) / 4).alias("w")).with_columns(
        pl.col("re", "w").cast(pl.Decimal(18, 6)), pl.col("experimental", "age", "educ").cast(pl.Decimal(18, 0))
    )
    floats = data.with_columns(pl.col("re", "w", "experimental", "age", "educ").cast(pl.Float64))
    spec = {
        "yname": "re",
        "tname": "year",
        "idname": "id" if panel else None,
        "treatname": "experimental",
        "xformla": "~ age + I(age**2) + educ",
        "weightsname": "w",
        "panel": panel,
    }

    result = drdid(data=data, **spec)
    expected = drdid(data=floats, **spec)

    assert np.all(np.isfinite([result.att, result.se]))
    np.testing.assert_allclose(result.att, expected.att, rtol=1e-12)
    np.testing.assert_allclose(result.se, expected.se, rtol=1e-12)


def test_drdid_formula_without_tilde_raises(nsw_data):
    with pytest.raises(ValueError, match="must be in the form"):
        drdid(
            data=nsw_data,
            yname="re",
            tname="year",
            treatname="experimental",
            idname="id",
            panel=True,
            xformla="age + educ",
        )


def test_drdid_panel_plain_covariates_without_formulaic(nsw_data, without_formulaic):
    post = nsw_data.filter(pl.col("year") == nsw_data["year"].max()).sort("id")
    pre = nsw_data.filter(pl.col("year") == nsw_data["year"].min()).sort("id")
    covariates = np.column_stack([np.ones(post.height), post.select("age", "educ", "black").to_numpy()])
    expected = drdid_imp_panel(
        y1=post["re"].to_numpy(),
        y0=pre["re"].to_numpy(),
        d=post["experimental"].to_numpy(),
        covariates=covariates,
        i_weights=np.ones(post.height),
    )

    result = drdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ age + educ + black",
    )

    assert np.all(np.isfinite([result.att, result.se]))
    np.testing.assert_allclose([result.att, result.se], [expected.att, expected.se], rtol=1e-12)


def test_drdid_rc_plain_covariates_without_formulaic(nsw_data, without_formulaic):
    covariates = np.column_stack([np.ones(nsw_data.height), nsw_data.select("age", "educ", "black").to_numpy()])
    expected = drdid_imp_local_rc(
        y=nsw_data["re"].to_numpy(),
        post=(nsw_data["year"] == nsw_data["year"].max()).cast(pl.Int64).to_numpy(),
        d=nsw_data["experimental"].to_numpy(),
        covariates=covariates,
        i_weights=np.ones(nsw_data.height),
    )

    result = drdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        panel=False,
        xformla="~ age + educ + black",
    )

    assert np.all(np.isfinite([result.att, result.se]))
    np.testing.assert_allclose([result.att, result.se], [expected.att, expected.se], rtol=1e-12)


def test_drdid_transformed_covariate_without_formulaic_raises(nsw_data, without_formulaic):
    with pytest.raises(ImportError, match=re.escape("xformla term 'I(age**2)' requires formulaic")):
        drdid(
            data=nsw_data,
            yname="re",
            tname="year",
            treatname="experimental",
            idname="id",
            xformla="~ age + I(age**2)",
        )


def test_drdid_string_covariate_without_formulaic_raises(nsw_data, without_formulaic):
    data = nsw_data.with_columns(
        pl.when(pl.col("age") <= 25).then(pl.lit("young")).otherwise(pl.lit("old")).alias("age_group")
    )

    with pytest.raises(ImportError, match="xformla column 'age_group' holds strings or categories"):
        drdid(
            data=data,
            yname="re",
            tname="year",
            treatname="experimental",
            idname="id",
            xformla="~ age_group + educ",
        )


@pytest.mark.parametrize("panel", [True, False])
def test_drdid_string_covariate_matches_categorical_term(nsw_data, panel):
    data = nsw_data.with_columns(
        pl.when(pl.col("age") <= 25).then(pl.lit("young")).otherwise(pl.lit("old")).alias("age_group")
    )
    spec = {"yname": "re", "tname": "year", "treatname": "experimental", "panel": panel}
    if panel:
        spec["idname"] = "id"

    result = drdid(data=data, xformla="~ age_group + educ", **spec)
    expected = drdid(data=data, xformla="~ C(age_group) + educ", **spec)

    assert result.att == expected.att
    assert result.se == expected.se


@pytest.mark.parametrize("xformla", ["re ~ age", "age ~ 1", "re ~ I(age**2)"])
def test_drdid_rejects_left_hand_side(nsw_data, xformla):
    with pytest.raises(ValueError, match="has a left-hand side"):
        drdid(
            data=nsw_data,
            yname="re",
            tname="year",
            treatname="experimental",
            idname="id",
            xformla=xformla,
        )


@pytest.mark.parametrize("xformla", ["~ age - 1", "~ 0 + age", "~ I(age**2) - 1"])
def test_drdid_rejects_dropping_the_intercept(nsw_data, xformla):
    with pytest.raises(ValueError, match="drops the intercept"):
        drdid(
            data=nsw_data,
            yname="re",
            tname="year",
            treatname="experimental",
            idname="id",
            xformla=xformla,
        )


def test_drdid_missing_covariate_raises(nsw_data):
    with pytest.raises(ValueError, match="^'nope' in xformla is not a column in the data\\.$"):
        drdid(
            data=nsw_data,
            yname="re",
            tname="year",
            treatname="experimental",
            idname="id",
            xformla="~ age + nope",
        )


@pytest.mark.parametrize("panel", [True, False])
def test_drdid_outcome_named_weights(nsw_data, panel):
    spec = dict(tname="year", treatname="experimental", idname="id", xformla="~ age + educ", panel=panel)
    renamed = drdid(data=nsw_data.rename({"re": "weights"}), yname="weights", **spec)
    expected = drdid(data=nsw_data, yname="re", **spec)

    assert renamed.att == expected.att
    assert renamed.se == expected.se


@pytest.mark.parametrize(
    ("column", "argument", "xformla"),
    [("re", "yname", "~ age + educ"), ("educ", "xformla", "~ age + {}"), ("educ", "xformla", "~ age + I({}**2)")],
)
@pytest.mark.parametrize("reserved", [".w", "Intercept"])
def test_drdid_rejects_reserved_column_names(nsw_data, column, argument, xformla, reserved):
    with pytest.raises(ValueError, match=re.escape(f"{argument} names the column '{reserved}'")):
        drdid(
            data=nsw_data.rename({column: reserved}),
            yname=reserved if column == "re" else "re",
            tname="year",
            treatname="experimental",
            idname="id",
            xformla=xformla.format(reserved),
        )


def test_drdid_rejects_repeated_unit_periods(nsw_duplicated):
    message = (
        "The value of idname must be unique (by tname). Some units are observed more than once in a period. "
        "Rows repeat for the (id, year) pair (15995, 1975)."
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        drdid(data=nsw_duplicated, yname="re", tname="year", treatname="experimental", idname="id", panel=True)


@pytest.mark.parametrize(
    ("nsw_one_infinite", "weightsname"),
    [(("re", float("inf")), None), (("age", float("-inf")), None), (("w", float("inf")), "w")],
    indirect=["nsw_one_infinite"],
)
def test_drdid_drops_infinite_rows_of_repeated_cross_sections(nsw_one_infinite, weightsname):
    spec = {
        "yname": "re",
        "tname": "year",
        "treatname": "experimental",
        "idname": "id",
        "xformla": "~ age + educ",
        "weightsname": weightsname,
        "panel": False,
    }
    expected = drdid(data=nsw_one_infinite.filter(pl.all_horizontal(pl.col("re", "age", "w").is_finite())), **spec)

    with pytest.warns(UserWarning, match="^Dropped 1 rows from original data due to missing values$"):
        result = drdid(data=nsw_one_infinite, **spec)

    assert result.att == expected.att
    assert result.se == expected.se


@pytest.mark.parametrize("nsw_one_infinite", [("year", float("nan")), ("year", float("inf"))], indirect=True)
@pytest.mark.parametrize("estimator", [drdid, ipwdid, ordid])
def test_repeated_cross_sections_count_the_periods_of_the_rows_that_remain(nsw_one_infinite, estimator):
    spec = {"yname": "re", "tname": "year", "treatname": "experimental", "panel": False}
    expected = estimator(data=nsw_one_infinite.filter(pl.col("year").is_finite()), **spec)

    with pytest.warns(UserWarning, match="^Dropped 1 rows from original data due to missing values$"):
        result = estimator(data=nsw_one_infinite, **spec)

    assert result.att == expected.att
    assert result.se == expected.se


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"tname": "yaer"}, "tname='yaer' is not a column in the data. Did you mean 'year'?"),
        (
            {"treatname": "experimentl"},
            "treatname='experimentl' is not a column in the data. Did you mean 'experimental'?",
        ),
        ({"idname": "idd"}, "idname='idd' is not a column in the data. Did you mean 'id'?"),
        ({"tname": "yaer", "panel": False}, "tname='yaer' is not a column in the data. Did you mean 'year'?"),
    ],
)
@pytest.mark.parametrize("estimator", [drdid, ipwdid, ordid])
def test_two_period_estimators_name_misspelled_columns(nsw_data, estimator, changes, message):
    spec = {"yname": "re", "tname": "year", "treatname": "experimental", "idname": "id"} | changes

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        estimator(data=nsw_data, **spec)


@pytest.mark.parametrize("nsw_one_infinite", [("re", float("inf")), ("w", float("-inf"))], indirect=True)
def test_drdid_panel_rejects_infinite_values_like_missing_ones(nsw_one_infinite):
    with pytest.raises(ValueError, match="^Missing values found in panel data\\. Dropped 1 rows\\."):
        drdid(
            data=nsw_one_infinite,
            yname="re",
            tname="year",
            treatname="experimental",
            idname="id",
            weightsname="w",
            panel=True,
        )


@pytest.mark.parametrize("panel", [True, False])
def test_drdid_rejects_weights_without_positive_mean(nsw_data, panel):
    message = "The weights variable 'w' must be non-negative with a positive mean."

    with pytest.raises(ValueError, match=re.escape(message)):
        drdid(
            data=nsw_data.with_columns(pl.lit(0.0).alias("w")),
            yname="re",
            tname="year",
            treatname="experimental",
            idname="id",
            weightsname="w",
            panel=panel,
        )


@pytest.mark.parametrize(
    ("panel", "est_method", "title", "regressions"),
    [
        (
            True,
            "imp",
            "Doubly Robust DiD Estimator (Improved Method)",
            ["Outcome regression: Weighted least squares"],
        ),
        (
            True,
            "trad",
            "Doubly Robust DiD Estimator (Traditional Method)",
            ["Outcome regression: Ordinary least squares"],
        ),
        (
            False,
            "imp",
            "Doubly Robust DiD Estimator (Improved Method)",
            [
                "Outcome regression (comparison group): Weighted least squares",
                "Outcome regression (treated group): Ordinary least squares",
            ],
        ),
        (
            False,
            "trad",
            "Doubly Robust DiD Estimator (Traditional Method)",
            ["Outcome regression: Ordinary least squares"],
        ),
    ],
)
def test_drdid_printout_describes_the_estimator_it_ran(nsw_data, panel, est_method, title, regressions):
    result = drdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ age + educ",
        panel=panel,
        est_method=est_method,
    )

    lines = [line.strip() for line in str(result).splitlines()]
    assert title in lines
    assert [line for line in lines if line.startswith("Outcome regression")] == regressions


@pytest.mark.parametrize(
    ("estimator", "xformla", "panel", "header"),
    [
        (
            drdid,
            "~ age + educ + black + married + nodegree + hisp + re74",
            True,
            "Computed from 32834 observations and 7 covariates.",
        ),
        (ipwdid, None, True, "Computed from 32834 observations and 0 covariates."),
        (ordid, "~ age + C(black)", False, "Computed from 32834 observations and 2 covariates."),
        (drdid, "~ age", False, "Computed from 32834 observations and 1 covariate."),
    ],
)
def test_two_period_header_counts_the_covariates_in_the_model(nsw_data, estimator, xformla, panel, header):
    result = estimator(
        data=nsw_data, yname="re", tname="year", treatname="experimental", idname="id", xformla=xformla, panel=panel
    )

    assert header in [line.strip() for line in str(result).splitlines()]
    assert result.__maketables_stat__("N") == 32834


@pytest.mark.parametrize("nsw_one_infinite", [("age", float("-inf"))], indirect=True)
def test_drdid_header_and_table_count_the_rows_left_after_dropping(nsw_one_infinite):
    with pytest.warns(UserWarning, match="^Dropped 1 rows from original data due to missing values$"):
        result = drdid(
            data=nsw_one_infinite, yname="re", tname="year", treatname="experimental", xformla="~ age", panel=False
        )

    assert "Computed from 32833 observations and 1 covariate." in [line.strip() for line in str(result).splitlines()]
    assert result.__maketables_stat__("N") == 32833


def test_drdid_panel_header_and_table_count_the_balanced_units(nsw_data):
    unbalanced = nsw_data.filter(~((pl.col("id") == 15995) & (pl.col("year") == 1978)))

    with pytest.warns(UserWarning, match="Panel data is unbalanced"):
        result = drdid(
            data=unbalanced, yname="re", tname="year", treatname="experimental", idname="id", xformla="~ age"
        )

    assert "Computed from 32832 observations and 1 covariate." in [line.strip() for line in str(result).splitlines()]
    assert result.__maketables_stat__("N") == 32832


@pytest.mark.parametrize("est_method", ["imp", "trad"])
def test_drdid_rc_methods_run_the_locally_efficient_estimators(
    nsw_thinned, rc_matching_did, rc_period_matching_did, est_method
):
    arrays = (
        nsw_thinned["re"].to_numpy(),
        (nsw_thinned["year"] == 1978).cast(pl.Int64).to_numpy(),
        nsw_thinned["experimental"].to_numpy(),
        (nsw_thinned["black"] + 2 * nsw_thinned["hisp"]).to_numpy(),
        np.ones(nsw_thinned.height),
    )
    att, _ = rc_matching_did(*arrays)
    period_att, _ = rc_period_matching_did(*arrays)

    result = drdid(
        data=nsw_thinned,
        yname="re",
        tname="year",
        treatname="experimental",
        xformla="~ black + hisp",
        panel=False,
        est_method=est_method,
    )

    assert abs(att - period_att) > 50
    np.testing.assert_allclose(result.att, att, rtol=1e-10)


@pytest.mark.parametrize("estimator", [drdid_imp_rc, drdid_trad_rc])
def test_rc_estimators_that_are_not_locally_efficient_stay_reachable(
    nsw_thinned, rc_matching_did, rc_period_matching_did, estimator
):
    y = nsw_thinned["re"].to_numpy()
    post = (nsw_thinned["year"] == 1978).cast(pl.Int64).to_numpy()
    d = nsw_thinned["experimental"].to_numpy()
    cells = (nsw_thinned["black"] + 2 * nsw_thinned["hisp"]).to_numpy()
    att, _ = rc_period_matching_did(y, post, d, cells, np.ones(y.size))
    local_att, _ = rc_matching_did(y, post, d, cells, np.ones(y.size))
    covariates = np.column_stack([np.ones(y.size), nsw_thinned.select("black", "hisp").to_numpy()])

    result = estimator(y=y, post=post, d=d, covariates=covariates)

    assert abs(att - local_att) > 50
    np.testing.assert_allclose(result.att, att, rtol=1e-10)


@pytest.mark.parametrize(
    ("estimator", "data_fixture"),
    [(drdid_imp_rc, "nsw_data"), (drdid_trad_rc, "nsw_data"), (drdid_trad_rc, "nsw_thinned")],
)
def test_rc_estimators_that_are_not_locally_efficient_return_the_matching_influence_function(
    request, rc_period_matching_did, estimator, data_fixture
):
    data = request.getfixturevalue(data_fixture)
    y = data["re"].to_numpy()
    post = (data["year"] == 1978).cast(pl.Int64).to_numpy()
    d = data["experimental"].to_numpy()
    covariates = np.column_stack([np.ones(y.size), data.select("black", "hisp").to_numpy()])
    att, inf_func = rc_period_matching_did(y, post, d, covariates[:, 1] + 2 * covariates[:, 2], np.ones(y.size))

    result = estimator(y=y, post=post, d=d, covariates=covariates, influence_func=True)

    np.testing.assert_allclose([result.att, result.se], [att, np.sqrt(np.sum(inf_func**2)) / y.size], rtol=1e-10)
    assert np.shape(result.att_inf_func) == inf_func.shape
    np.testing.assert_allclose(result.att_inf_func, inf_func, rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize(
    ("estimator", "label"),
    [(drdid_imp_rc, "imp1"), (drdid_imp_local_rc, "imp2"), (drdid_trad_rc, "trad1"), (drdid_rc, "trad2")],
)
def test_rc_estimators_label_their_results_with_the_paper_subscript(rc_correct_models_arrays, estimator, label):
    result = estimator(**rc_correct_models_arrays, trim_level=0.99)

    assert result.args["estMethod"] == label
    assert result.args["trim_level"] == 0.99
    assert [key for key in result.args if "." in key] == []


@pytest.mark.parametrize("panel", [True, False])
@pytest.mark.parametrize("est_method", ["imp", "trad"])
def test_drdid_args_spell_each_setting_once(nsw_data, panel, est_method):
    result = drdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        panel=panel,
        est_method=est_method,
        trim_level=0.99,
    )

    assert result.args["trim_level"] == 0.99
    assert [key for key in result.args if "." in key] == []


@pytest.mark.parametrize("est_method", ["imp", "trad"])
def test_drdid_rc_methods_reject_a_treated_cell_with_a_constant_covariate(rc_married_in_treated_pre, est_method):
    message = (
        "The regression design matrix of the treated group in the pre-period is singular. Consider removing "
        "covariates that are constant or collinear within that group and period. "
        "The estimators drdid_imp_rc and drdid_trad_rc fit no regression for the treated group and accept this design."
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            drdid(
                data=rc_married_in_treated_pre,
                yname="y",
                tname="year",
                treatname="d",
                xformla="~ x0 + married",
                panel=False,
                est_method=est_method,
            )

    assert [str(warning.message) for warning in caught] == []


@pytest.mark.parametrize("est_method", ["imp", "trad"])
def test_drdid_rc_methods_reject_a_treated_cell_with_fewer_observations_than_coefficients(
    rc_small_treated_pre, est_method
):
    message = (
        "The treated group in the pre-period has fewer observations (4) than regression coefficients (6). "
        "Pass fewer covariates. The estimators drdid_imp_rc and drdid_trad_rc fit no regression for the treated "
        "group and accept this design."
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            drdid(
                data=rc_small_treated_pre,
                yname="y",
                tname="year",
                treatname="d",
                xformla="~ x0 + x1 + x2 + x3 + x4",
                panel=False,
                est_method=est_method,
            )

    assert [str(warning.message) for warning in caught] == []


@pytest.mark.parametrize("est_method", ["imp", "trad"])
def test_drdid_panel_methods_reject_a_comparison_group_with_fewer_units_than_coefficients(
    panel_small_comparison, est_method
):
    message = "The comparison group has fewer units (4) than regression coefficients (6). Pass fewer covariates."

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            drdid(
                data=panel_small_comparison,
                yname="y",
                tname="year",
                idname="id",
                treatname="d",
                xformla="~ x0 + x1 + x2 + x3 + x4",
                est_method=est_method,
            )

    assert [str(warning.message) for warning in caught] == []


@pytest.mark.parametrize("estimator", [drdid_imp_local_rc, drdid_rc])
def test_locally_efficient_rc_estimators_reject_a_treated_cell_with_fewer_observations_than_coefficients(
    rc_small_treated_pre_arrays, estimator
):
    message = (
        "The treated group in the pre-period has fewer observations (4) than regression coefficients (6). "
        "Pass fewer covariates. The estimators drdid_imp_rc and drdid_trad_rc fit no regression for the treated "
        "group and accept this design."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        estimator(**rc_small_treated_pre_arrays)


@pytest.mark.parametrize("estimator", [drdid_imp_rc, drdid_trad_rc])
@pytest.mark.parametrize("design", ["rc_binary_one_treated_pre_arrays", "rc_binary_constant_treated_pre_arrays"])
def test_rc_estimators_without_treated_regressions_fit_an_unidentified_treated_cell(request, design, estimator):
    arrays = request.getfixturevalue(design)
    y, post, d, b = arrays["y"], arrays["post"], arrays["d"], arrays["covariates"][:, 1]
    comparison_means = np.zeros(y.size)
    for period in (0, 1):
        for value in (0, 1):
            rows = (post == period) & (b == value)
            comparison_means[rows] = y[rows & (d == 0)].mean()
    residuals = y - comparison_means
    att = residuals[(d == 1) & (post == 1)].mean() - residuals[(d == 1) & (post == 0)].mean()

    result = estimator(**arrays, influence_func=True)

    assert np.isfinite(result.se)
    np.testing.assert_allclose(result.att, att, rtol=1e-12)
    np.testing.assert_allclose(result.se, np.std(result.att_inf_func) / np.sqrt(y.size), rtol=1e-12)


@pytest.mark.parametrize("estimator", [drdid_imp_local_rc, drdid_rc, drdid_imp_rc, drdid_trad_rc, reg_did_rc])
def test_rc_estimators_reject_a_comparison_cell_with_fewer_observations_than_coefficients(
    rc_small_comparison_post_arrays, estimator
):
    message = (
        "The comparison group in the post-period has fewer observations (4) than regression coefficients (6). "
        "Pass fewer covariates."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        estimator(**rc_small_comparison_post_arrays)


@pytest.mark.parametrize("estimator", [drdid_imp_panel, drdid_panel, reg_did_panel])
def test_panel_estimators_reject_a_comparison_group_with_fewer_units_than_coefficients(
    panel_small_comparison_arrays, estimator
):
    message = "The comparison group has fewer units (4) than regression coefficients (6). Pass fewer covariates."

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        estimator(**panel_small_comparison_arrays)


@pytest.mark.parametrize(
    "estimator",
    [drdid_imp_local_rc, drdid_rc, drdid_imp_rc, drdid_trad_rc, reg_did_rc, ipw_did_rc, std_ipw_did_rc, twfe_did_rc],
)
def test_rc_estimators_reject_a_period_without_treated_rows(rc_correct_models_arrays, estimator):
    arrays = rc_correct_models_arrays
    d = np.where((arrays["d"] == 1) & (arrays["post"] == 0), 0, arrays["d"])
    message = (
        "No units found in the treated group in the pre-period. Pass a treatment indicator that equals 1 for "
        "treated units in both periods."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        estimator(**{**arrays, "d": d})


@pytest.mark.parametrize(
    "estimator",
    [drdid_imp_local_rc, drdid_rc, drdid_imp_rc, drdid_trad_rc, reg_did_rc, ipw_did_rc, std_ipw_did_rc, twfe_did_rc],
)
def test_rc_estimators_reject_a_period_whose_treated_rows_have_no_weight(rc_correct_models_arrays, estimator):
    arrays = rc_correct_models_arrays
    weights = np.where((arrays["d"] == 1) & (arrays["post"] == 0), 0.0, 1.0)
    message = (
        "The treated group in the pre-period has no observations with positive weight. Pass positive weights for "
        "some of its observations."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        estimator(**arrays, i_weights=weights)


@pytest.mark.parametrize("estimator", [drdid_imp_panel, drdid_panel, reg_did_panel, twfe_did_panel])
@pytest.mark.parametrize(
    ("treated", "message"),
    [
        (0, "No units found in the treated group. Pass data in which some units have a treatment indicator of 1."),
        (1, "No units found in the comparison group. Pass data in which some units have a treatment indicator of 0."),
    ],
)
def test_panel_estimators_reject_a_group_without_units(panel_small_comparison_arrays, estimator, treated, message):
    arrays = panel_small_comparison_arrays
    d = np.full(arrays["d"].size, treated)

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        estimator(**{**arrays, "d": d})


@pytest.mark.parametrize(
    "estimator", [drdid_imp_panel, drdid_panel, reg_did_panel, ipw_did_panel, std_ipw_did_panel, twfe_did_panel]
)
def test_panel_estimators_reject_a_treated_group_without_weight(two_covariate_design, estimator):
    design = two_covariate_design
    covariates = np.column_stack([np.ones(design["d"].size), design["z"]])
    weights = np.where(design["d"] == 1, 0.0, 1.0)
    message = "The treated group has no units with positive weight. Pass positive weights for some of its units."

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        estimator(y1=design["y1"], y0=design["y0"], d=design["d"], covariates=covariates, i_weights=weights)


@pytest.mark.parametrize("scale", [1e9, 1e16])
@pytest.mark.parametrize(
    ("estimator", "panel", "rtol"),
    [
        (drdid_imp_local_rc, False, 1e-8),
        (drdid_rc, False, 1e-12),
        (drdid_imp_rc, False, 1e-8),
        (drdid_trad_rc, False, 1e-12),
        (reg_did_rc, False, 1e-12),
        (ipw_did_rc, False, 1e-12),
        (std_ipw_did_rc, False, 1e-12),
        (twfe_did_rc, False, 1e-12),
        (drdid_imp_panel, True, 1e-8),
        (drdid_panel, True, 1e-12),
        (reg_did_panel, True, 1e-12),
        (ipw_did_panel, True, 1e-12),
        (std_ipw_did_panel, True, 1e-12),
        (twfe_did_panel, True, 1e-12),
    ],
)
def test_estimators_give_the_same_estimate_whatever_the_scale_of_a_covariate(
    two_covariate_design, estimator, panel, rtol, scale
):
    design = two_covariate_design
    ones = np.ones(design["d"].size)
    outcomes = {"y1": design["y1"], "y0": design["y0"]} if panel else {"y": design["y"], "post": design["post"]}

    expected = estimator(d=design["d"], covariates=np.column_stack([ones, design["z"]]), **outcomes)
    result = estimator(
        d=design["d"], covariates=np.column_stack([ones, design["z"][:, 0], scale * design["z"][:, 1]]), **outcomes
    )

    assert np.all(np.isfinite([result.att, result.se]))
    np.testing.assert_allclose([result.att, result.se], [expected.att, expected.se], rtol=rtol)


def test_drdid_rejects_the_removed_n_boot_argument(nsw_data):
    with pytest.raises(TypeError, match="unexpected keyword argument 'n_boot'"):
        drdid(data=nsw_data, yname="re", tname="year", treatname="experimental", idname="id", boot=True, n_boot=50)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"treatname": None}, "treatname is required. Please specify the treatment column."),
        ({"trim_level": 1.0}, "trim_level=1.0 is not valid. Must be between 0 and 1 (exclusive)."),
        ({"trim_level": 0}, "trim_level=0 is not valid. Must be between 0 and 1 (exclusive)."),
        ({"boot_type": "wild"}, "boot_type='wild' is not valid. Must be 'weighted' or 'multiplier'."),
        ({"biters": 0}, "biters=0 is not valid. Must be a positive integer."),
        ({"biters": 10.0}, "biters=10.0 is not valid. Must be a positive integer."),
    ],
)
def test_drdid_rejects_invalid_settings(nsw_data, changes, message):
    spec = {"yname": "re", "tname": "year", "treatname": "experimental", "idname": "id"} | changes

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        drdid(data=nsw_data, **spec)


def test_drdid_repeated_cross_sections_ignore_idname(
    trimmed_two_period_data, trimmed_two_period_frames, rc_matching_did
):
    data = trimmed_two_period_data
    n = data["d"].size
    att, inf_func = rc_matching_did(data["y"], data["post"], data["d"], np.zeros(n), np.ones(n))

    result = drdid(
        data=trimmed_two_period_frames["repeated_cross_sections"],
        yname="y",
        tname="year",
        treatname="d",
        idname="id",
        panel=False,
    )

    np.testing.assert_allclose([result.att, result.se], [att, np.sqrt(np.sum(inf_func**2)) / n], rtol=1e-12)


@pytest.mark.parametrize(
    ("column", "message"),
    [
        (
            "experimental",
            "Treatment indicator ('experimental') must be unique for each ID ('id'). IDs with varying treatment: [1].",
        ),
        ("w", "Weights must be time-invariant in panel data."),
    ],
)
def test_drdid_panel_rejects_a_treatment_or_weight_that_changes_over_time(nsw_data, column, message):
    row = (pl.col("id") == 1) & (pl.col("year") == 1978)
    data = nsw_data.with_columns(pl.lit(1.0).alias("w"))

    with pytest.raises(ValueError, match=f"{re.escape(message)}$"):
        drdid(
            data=data.with_columns(pl.when(row).then(1 - pl.col(column)).otherwise(pl.col(column)).alias(column)),
            yname="re",
            tname="year",
            treatname="experimental",
            idname="id",
            weightsname="w",
        )
