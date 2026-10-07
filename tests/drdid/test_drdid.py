"""Tests for doubly robust DiD."""

import re

import numpy as np
import pytest

from moderndid.core.data import load_nsw
from moderndid.drdid import drdid_imp_panel, drdid_imp_rc
from moderndid.drdid.drdid import drdid
from moderndid.drdid.ipwdid import ipwdid
from moderndid.drdid.ordid import ordid
from tests.helpers import importorskip

pl = importorskip("polars")


@pytest.fixture
def nsw_data():
    return load_nsw()


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
        n_boot=10,
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


@pytest.mark.parametrize("est_method", ["imp", "trad", "imp_local", "trad_local"])
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
        n_boot=10,
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


@pytest.mark.parametrize("est_method", ["imp_local", "trad_local"])
def test_drdid_invalid_est_method_for_panel(est_method):
    rng = np.random.default_rng(42)
    df = pl.DataFrame(
        {
            "id": np.repeat(range(50), 2),
            "time": np.tile([0, 1], 50),
            "y": rng.standard_normal(100),
            "treat": np.repeat([0, 1], 50),
        }
    )

    with pytest.raises(ValueError, match=f"est_method='{est_method}' is only available for repeated cross-sections"):
        drdid(
            data=df,
            yname="y",
            tname="time",
            treatname="treat",
            idname="id",
            panel=True,
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
        n_boot=50,
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
    assert result.call_params["n_boot"] == 50
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
        n_boot=10,
        est_method="trad",
    )

    assert result.args["panel"] is True
    assert result.args["normalized"] is True
    assert result.args["boot"] is True
    assert result.args["boot_type"] == "weighted"
    assert result.args["nboot"] == 10
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
        (False, "imp_local"),
        (False, "trad_local"),
    ],
)
def test_drdid_all_estimators_consistency(nsw_data, panel, method):
    if panel and method in ["imp_local", "trad_local"]:
        pytest.skip("Methods imp_local and trad_local are only for repeated cross-sections")

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
    from moderndid.drdid.ordid import ordid

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
    expected = drdid_imp_rc(
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
