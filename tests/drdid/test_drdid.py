"""Tests for doubly robust DiD."""

import re
import warnings

import numpy as np
import pytest

from moderndid.drdid import (
    drdid_imp_local_rc,
    drdid_imp_panel,
    drdid_imp_rc,
    drdid_panel,
    drdid_rc,
    drdid_trad_rc,
    ipw_did_panel,
    ipw_did_rc,
    reg_did_panel,
    reg_did_rc,
    std_ipw_did_panel,
    std_ipw_did_rc,
    twfe_did_panel,
    twfe_did_rc,
)
from moderndid.drdid.drdid import drdid
from moderndid.drdid.ipwdid import ipwdid
from moderndid.drdid.ordid import ordid
from tests.helpers import importorskip

pl = importorskip("polars")


@pytest.mark.parametrize("est_method", ["imp", "trad"])
def test_drdid_panel_basic(nsw_data, est_method):
    result = drdid(
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
    assert result.se > 0
    assert result.lci < result.att < result.uci
    assert result.boots is None
    assert result.att_inf_func is None
    assert result.call_params["panel"] is True
    assert result.call_params["est_method"] == est_method
    assert result.args["type"] == "dr"
    assert result.args["estMethod"] == est_method


def test_drdid_panel_with_covariates(nsw_data):
    result = drdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ age + educ + black + married + nodegree + hisp",
        panel=True,
        est_method="imp",
    )

    assert isinstance(result.att, float)
    assert result.se > 0
    assert result.call_params["xformla"] == "~ age + educ + black + married + nodegree + hisp"


@pytest.mark.filterwarnings("ignore:Extreme weight ratios detected:UserWarning")
def test_drdid_panel_with_weights(nsw_data):
    rng = np.random.default_rng(42)
    unique_ids = nsw_data["id"].unique().to_list()
    unit_weights = rng.exponential(1, len(unique_ids))
    weight_dict = dict(zip(unique_ids, unit_weights))
    nsw_data = nsw_data.with_columns(pl.col("id").replace_strict(weight_dict, default=1.0).alias("weight"))

    result = drdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ age + educ",
        panel=True,
        weightsname="weight",
        est_method="imp",
    )

    assert isinstance(result.att, float)
    assert result.se > 0
    assert result.call_params["weightsname"] == "weight"


def test_drdid_panel_with_influence_func(nsw_data):
    result = drdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        panel=True,
        inf_func=True,
        est_method="imp",
    )

    assert result.att_inf_func is not None
    n_units = len(nsw_data["id"].unique())
    assert len(result.att_inf_func) == n_units
    assert np.isclose(np.mean(result.att_inf_func), 0, atol=1e-10)


@pytest.mark.filterwarnings("ignore:Extreme weight ratios detected:UserWarning")
@pytest.mark.parametrize(
    "boot_type,est_method",
    [
        ("weighted", "imp"),
        ("multiplier", "trad"),
    ],
)
def test_drdid_panel_bootstrap(nsw_data, boot_type, est_method):
    result = drdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ age + educ",
        panel=True,
        boot=True,
        boot_type=boot_type,
        biters=10,
        inf_func=(boot_type == "multiplier"),
        est_method=est_method,
    )

    assert result.boots is not None
    assert len(result.boots) == 10
    assert not np.all(np.isnan(result.boots))
    assert result.args["boot"] is True
    assert result.args["boot_type"] == boot_type
    if boot_type == "multiplier":
        assert result.att_inf_func is not None


@pytest.mark.parametrize("est_method", ["imp", "trad"])
def test_drdid_repeated_cross_section(nsw_data, est_method):
    result = drdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        panel=False,
        xformla="~ age + educ + black",
        est_method=est_method,
    )

    assert isinstance(result.att, float)
    assert result.se > 0
    assert result.args["panel"] is False
    assert result.args["estMethod"] == est_method


@pytest.mark.filterwarnings("ignore:Extreme weight ratios detected:UserWarning")
def test_drdid_rc_with_bootstrap(nsw_data):
    result = drdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        panel=False,
        xformla="~ age + educ",
        boot=True,
        biters=10,
        est_method="imp",
    )

    assert result.boots is not None
    assert len(result.boots) == 10


@pytest.mark.parametrize("trim_level", [0.99, 0.995])
def test_drdid_trim_level(nsw_data, trim_level):
    result = drdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        panel=True,
        trim_level=trim_level,
        est_method="imp",
    )

    assert isinstance(result.att, float)
    assert result.args["trim_level"] == trim_level


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


@pytest.mark.parametrize(
    "formula",
    [
        "~ age + educ + age:educ",
        "~ age + I(age**2) + educ",
    ],
)
def test_drdid_formula_variations(nsw_data, formula):
    result = drdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla=formula,
        panel=True,
        est_method="imp",
    )

    assert isinstance(result.att, float)
    assert result.se > 0


@pytest.mark.filterwarnings("ignore:Extreme weight ratios detected:UserWarning")
def test_drdid_call_params_stored(nsw_data):
    result = drdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ age + educ",
        panel=True,
        boot=True,
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
    assert result.call_params["biters"] == 50
    assert result.call_params["inf_func"] is True
    assert result.call_params["est_method"] == "imp"
    assert result.call_params["trim_level"] == 0.99
    assert "data_shape" in result.call_params


@pytest.mark.filterwarnings("ignore:Extreme weight ratios detected:UserWarning")
def test_drdid_args_output(nsw_data):
    result = drdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
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


def test_drdid_reproducibility(nsw_data):
    treated_ids = nsw_data.filter(pl.col("experimental") == 1)["id"].unique()[:50].to_list()
    control_ids = nsw_data.filter(pl.col("experimental") == 0)["id"].unique()[:50].to_list()
    selected_ids = treated_ids + control_ids
    small_data = nsw_data.filter(pl.col("id").is_in(selected_ids))

    result1 = drdid(
        data=small_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ age + educ",
        panel=True,
        est_method="imp",
    )

    result2 = drdid(
        data=small_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ age + educ",
        panel=True,
        est_method="imp",
    )

    assert result1.att == result2.att
    assert result1.se == result2.se


def test_drdid_subset_columns(nsw_data):
    subset_cols = ["id", "year", "re", "experimental", "age", "educ"]
    subset_data = nsw_data.select(subset_cols)

    result = drdid(
        data=subset_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ age + educ",
        panel=True,
        est_method="imp",
    )

    assert isinstance(result.att, float)
    assert result.se > 0


@pytest.mark.parametrize("covariates_formula", [None, "~ 1"])
def test_drdid_no_covariates(nsw_data, covariates_formula):
    result = drdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla=covariates_formula,
        panel=True,
        est_method="imp",
    )

    assert isinstance(result.att, float)
    assert result.se > 0


def test_drdid_no_covariates_equivalence(nsw_data):
    result1 = drdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla=None,
        panel=True,
        est_method="imp",
    )

    result2 = drdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ 1",
        panel=True,
        est_method="imp",
    )

    assert np.isclose(result1.att, result2.att, rtol=1e-10)
    assert np.isclose(result1.se, result2.se, rtol=1e-10)


def test_drdid_categorical_covariates(nsw_data):
    nsw_data = nsw_data.with_columns(
        pl.when(pl.col("age") <= 25)
        .then(pl.lit("young"))
        .when(pl.col("age") <= 35)
        .then(pl.lit("middle"))
        .otherwise(pl.lit("old"))
        .alias("age_group")
    )

    result = drdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ C(age_group) + educ + black",
        panel=True,
        est_method="imp",
    )

    assert isinstance(result.att, float)
    assert result.se > 0


@pytest.mark.filterwarnings("ignore:Missing values found:UserWarning")
@pytest.mark.filterwarnings("ignore:Dropped.*rows due to missing values:UserWarning")
@pytest.mark.filterwarnings("ignore:Panel data is unbalanced:UserWarning")
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

    with pytest.raises(ValueError):
        drdid(
            data=df,
            yname="y",
            tname="time",
            treatname="treat",
            idname="id",
            xformla="~ x1",
            panel=True,
        )


@pytest.mark.filterwarnings("ignore:Small group size detected:UserWarning")
@pytest.mark.filterwarnings("ignore:Only.*control units available:UserWarning")
def test_drdid_unbalanced_panel_warning():
    rng = np.random.default_rng(42)
    df = pl.DataFrame(
        {
            "id": [1, 1, 2, 2, 3],
            "time": [0, 1, 0, 1, 0],
            "y": rng.standard_normal(5),
            "treat": [0, 0, 1, 1, 0],
        }
    )

    with pytest.warns(UserWarning, match="Panel data is unbalanced"):
        result = drdid(
            data=df,
            yname="y",
            tname="time",
            treatname="treat",
            idname="id",
            panel=True,
        )
        assert isinstance(result.att, float)


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

    with pytest.raises(ValueError):
        drdid(
            data=df,
            yname="y",
            tname="time",
            treatname="treat",
            idname="id",
            panel=True,
        )


@pytest.mark.parametrize(
    "panel,method",
    [
        (True, "imp"),
        (True, "trad"),
        (False, "imp"),
        (False, "trad"),
    ],
)
def test_drdid_all_estimators_consistency(nsw_data, panel, method):
    treated_ids = nsw_data.filter(pl.col("experimental") == 1)["id"].unique()[:50].to_list()
    control_ids = nsw_data.filter(pl.col("experimental") == 0)["id"].unique()[:50].to_list()
    selected_ids = treated_ids + control_ids
    small_data = nsw_data.filter(pl.col("id").is_in(selected_ids))

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
    assert result.se > 0
    assert np.isfinite(result.att)
    assert -10000 < result.att < 10000


def test_drdid_comparison_with_ordid(nsw_data):
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

    assert abs(or_result.att - dr_result.att) < abs(or_result.att) * 2


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


@pytest.mark.parametrize(
    ("est_method", "estimator", "att", "se"),
    [
        ("imp", drdid_imp_local_rc, 0.08027838510, 0.2008845114),
        ("trad", drdid_rc, 0.1013594144, 0.2042126933),
    ],
)
def test_drdid_rc_methods_run_the_locally_efficient_estimators(
    rc_correct_models, rc_correct_models_arrays, est_method, estimator, att, se
):
    result = drdid(
        data=rc_correct_models,
        yname="y",
        tname="year",
        treatname="d",
        xformla="~ z1 + z2 + z3 + z4",
        panel=False,
        est_method=est_method,
    )
    expected = estimator(**rc_correct_models_arrays)

    np.testing.assert_allclose([result.att, result.se], [att, se], rtol=1e-6)
    np.testing.assert_allclose([result.att, result.se], [expected.att, expected.se], rtol=1e-10)


@pytest.mark.parametrize(
    ("estimator", "att", "se"),
    [(drdid_imp_rc, -1.941300004, 3.104453917), (drdid_trad_rc, -1.918270052, 3.105141979)],
)
def test_rc_estimators_that_are_not_locally_efficient_stay_reachable(rc_correct_models_arrays, estimator, att, se):
    result = estimator(**rc_correct_models_arrays)

    np.testing.assert_allclose([result.att, result.se], [att, se], rtol=1e-6)


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

    np.testing.assert_allclose([result.att, result.se], [expected.att, expected.se], rtol=rtol)


def test_drdid_rejects_the_removed_n_boot_argument(nsw_data):
    with pytest.raises(TypeError, match="unexpected keyword argument 'n_boot'"):
        drdid(data=nsw_data, yname="re", tname="year", treatname="experimental", idname="id", boot=True, n_boot=50)
