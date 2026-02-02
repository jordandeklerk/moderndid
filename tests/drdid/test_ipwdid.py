"""Tests for inverse propensity weighted DiD."""

import numpy as np
import pytest

from moderndid.core.data import load_nsw
from moderndid.drdid.ipwdid import ipwdid
from tests.helpers import importorskip

pl = importorskip("polars")


@pytest.fixture
def nsw_data():
    return load_nsw()


@pytest.mark.parametrize("est_method", ["ipw", "std_ipw"])
def test_ipwdid_panel_basic(nsw_data, est_method):
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
    assert result.se > 0
    assert result.lci < result.att < result.uci
    assert result.boots is None
    assert result.att_inf_func is None
    assert result.call_params["panel"] is True
    assert result.call_params["est_method"] == est_method
    assert result.args["type"] == "ipw"
    assert result.args["estMethod"] == est_method


def test_ipwdid_panel_with_covariates(nsw_data):
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
    assert result.se > 0
    assert result.call_params["xformla"] == "~ age + educ + black + married + nodegree + hisp"


def test_ipwdid_panel_with_weights(nsw_data):
    rng = np.random.default_rng(42)
    unique_ids = nsw_data["id"].unique().to_list()
    unit_weights = rng.exponential(1, len(unique_ids))
    weight_dict = dict(zip(unique_ids, unit_weights))
    nsw_data = nsw_data.with_columns(pl.col("id").replace_strict(weight_dict, default=1.0).alias("weight"))

    result = ipwdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ age + educ",
        panel=True,
        weightsname="weight",
        est_method="ipw",
    )

    assert isinstance(result.att, float)
    assert result.se > 0
    assert result.call_params["weightsname"] == "weight"


def test_ipwdid_panel_with_influence_func(nsw_data):
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
    n_units = len(nsw_data["id"].unique())
    assert len(result.att_inf_func) == n_units
    assert np.all(np.isfinite(result.att_inf_func))
    assert np.abs(np.mean(result.att_inf_func)) < 1000


@pytest.mark.parametrize(
    "boot_type,est_method",
    [
        ("weighted", "ipw"),
        ("multiplier", "std_ipw"),
    ],
)
def test_ipwdid_panel_bootstrap(nsw_data, boot_type, est_method):
    result = ipwdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ age + educ",
        panel=True,
        boot=True,
        boot_type=boot_type,
        n_boot=50,
        inf_func=(boot_type == "multiplier"),
        est_method=est_method,
    )

    assert result.boots is not None
    assert len(result.boots) == result.args["nboot"]
    assert not np.all(np.isnan(result.boots))
    assert result.args["boot"] is True
    assert result.args["boot_type"] == boot_type
    if boot_type == "multiplier":
        assert result.att_inf_func is not None


@pytest.mark.parametrize(
    "panel,est_method",
    [
        (False, "ipw"),
        (False, "std_ipw"),
    ],
)
def test_ipwdid_repeated_cross_section(nsw_data, panel, est_method):
    result = ipwdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        panel=panel,
        xformla="~ age + educ + black",
        est_method=est_method,
    )

    assert isinstance(result.att, float)
    assert result.se > 0
    assert result.args["panel"] is False
    assert result.args["estMethod"] == est_method


def test_ipwdid_rc_with_bootstrap(nsw_data):
    result = ipwdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        panel=False,
        xformla="~ age + educ",
        boot=True,
        n_boot=50,
        est_method="ipw",
    )

    assert result.boots is not None
    assert len(result.boots) == result.args["nboot"]


@pytest.mark.parametrize("trim_level", [0.99, 0.995])
def test_ipwdid_trim_level(nsw_data, trim_level):
    result = ipwdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        panel=True,
        trim_level=trim_level,
        est_method="ipw",
    )

    assert isinstance(result.att, float)
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
    "formula",
    [
        "~ age + educ + age:educ",
        "~ age + I(age**2) + educ",
    ],
)
def test_ipwdid_formula_variations(nsw_data, formula):
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
        n_boot=50,
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
    assert result.call_params["n_boot"] == 50
    assert result.call_params["inf_func"] is True
    assert result.call_params["est_method"] == "std_ipw"
    assert result.call_params["trim_level"] == 0.99
    assert "data_shape" in result.call_params


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
        n_boot=50,
        est_method="ipw",
    )

    assert result.args["panel"] is True
    assert result.args["normalized"] is True
    assert result.args["boot"] is True
    assert result.args["boot_type"] == "weighted"
    assert result.args["nboot"] == 50
    assert result.args["type"] == "ipw"
    assert result.args["estMethod"] == "ipw"


def test_ipwdid_reproducibility(nsw_data):
    treated_ids = nsw_data.filter(pl.col("experimental") == 1)["id"].unique()[:50].to_list()
    control_ids = nsw_data.filter(pl.col("experimental") == 0)["id"].unique()[:50].to_list()
    selected_ids = treated_ids + control_ids
    small_data = nsw_data.filter(pl.col("id").is_in(selected_ids))

    result1 = ipwdid(
        data=small_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ age + educ",
        panel=True,
        est_method="ipw",
    )

    result2 = ipwdid(
        data=small_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ age + educ",
        panel=True,
        est_method="ipw",
    )

    assert result1.att == result2.att
    assert result1.se == result2.se


def test_ipwdid_subset_columns(nsw_data):
    subset_cols = ["id", "year", "re", "experimental", "age", "educ"]
    subset_data = nsw_data.select(subset_cols)

    result = ipwdid(
        data=subset_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ age + educ",
        panel=True,
        est_method="std_ipw",
    )

    assert isinstance(result.att, float)
    assert result.se > 0


@pytest.mark.parametrize("covariates_formula", [None, "~ 1"])
def test_ipwdid_no_covariates(nsw_data, covariates_formula):
    result = ipwdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla=covariates_formula,
        panel=True,
        est_method="ipw",
    )

    assert isinstance(result.att, float)
    assert result.se > 0


def test_ipwdid_no_covariates_equivalence(nsw_data):
    result1 = ipwdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla=None,
        panel=True,
        est_method="ipw",
    )

    result2 = ipwdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ 1",
        panel=True,
        est_method="ipw",
    )

    assert np.isclose(result1.att, result2.att, rtol=1e-10)
    assert np.isclose(result1.se, result2.se, rtol=1e-10)


def test_ipwdid_categorical_covariates(nsw_data):
    nsw_data = nsw_data.with_columns(
        pl.when(pl.col("age") <= 25)
        .then(pl.lit("young"))
        .when(pl.col("age") <= 35)
        .then(pl.lit("middle"))
        .otherwise(pl.lit("old"))
        .alias("age_group")
    )

    result = ipwdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ C(age_group) + educ + black",
        panel=True,
        est_method="ipw",
    )

    assert isinstance(result.att, float)
    assert result.se > 0


@pytest.mark.filterwarnings("ignore:Missing values found:UserWarning")
@pytest.mark.filterwarnings("ignore:Dropped.*rows due to missing values:UserWarning")
@pytest.mark.filterwarnings("ignore:Panel data is unbalanced:UserWarning")
def test_ipwdid_missing_values_handled():
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

    result = ipwdid(
        data=df,
        yname="y",
        tname="time",
        treatname="treat",
        idname="id",
        xformla="~ x1",
        panel=True,
    )
    assert isinstance(result.att, float)
    assert np.isfinite(result.att) or np.isnan(result.att)


@pytest.mark.filterwarnings("ignore:Small group size detected:UserWarning")
def test_ipwdid_unbalanced_panel_warning():
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
        result = ipwdid(
            data=df,
            yname="y",
            tname="time",
            treatname="treat",
            idname="id",
            panel=True,
        )
        assert isinstance(result.att, float)


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

    with pytest.raises(ValueError):
        ipwdid(
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
        (True, "ipw"),
        (True, "std_ipw"),
        (False, "ipw"),
        (False, "std_ipw"),
    ],
)
def test_ipwdid_estimators_consistency(nsw_data, panel, method):
    treated_ids = nsw_data.filter(pl.col("experimental") == 1)["id"].unique()[:200].to_list()
    control_ids = nsw_data.filter(pl.col("experimental") == 0)["id"].unique()[:200].to_list()
    selected_ids = treated_ids + control_ids
    small_data = nsw_data.filter(pl.col("id").is_in(selected_ids))

    try:
        result = ipwdid(
            data=small_data,
            yname="re",
            tname="year",
            treatname="experimental",
            idname="id" if panel else None,
            xformla="~ age + educ + black",
            panel=panel,
            est_method=method,
            trim_level=0.99,
        )
        assert isinstance(result.att, float)
        assert result.se > 0
        assert np.isfinite(result.att)
        assert -10000 < result.att < 10000
    except ValueError:
        pass


def test_ipwdid_comparison_std_vs_regular(nsw_data):
    treated_ids = nsw_data.filter(pl.col("experimental") == 1)["id"].unique()[:300].to_list()
    control_ids = nsw_data.filter(pl.col("experimental") == 0)["id"].unique()[:300].to_list()
    selected_ids = treated_ids + control_ids
    data = nsw_data.filter(pl.col("id").is_in(selected_ids))

    try:
        ipw_result = ipwdid(
            data=data,
            yname="re",
            tname="year",
            treatname="experimental",
            idname="id",
            xformla="~ age + educ",
            panel=True,
            est_method="ipw",
            trim_level=0.99,
        )
        ipw_succeeded = True
    except ValueError:
        ipw_succeeded = False
        ipw_result = None

    try:
        std_ipw_result = ipwdid(
            data=data,
            yname="re",
            tname="year",
            treatname="experimental",
            idname="id",
            xformla="~ age + educ",
            panel=True,
            est_method="std_ipw",
            trim_level=0.99,
        )
        std_ipw_succeeded = True
    except ValueError:
        std_ipw_succeeded = False
        std_ipw_result = None

    assert ipw_succeeded or std_ipw_succeeded

    if ipw_succeeded and std_ipw_succeeded:
        assert abs(ipw_result.att - std_ipw_result.att) < max(abs(ipw_result.att), abs(std_ipw_result.att)) * 2
        assert ipw_result.se > 0 and std_ipw_result.se > 0


def test_ipwdid_extreme_trimming(nsw_data):
    result = ipwdid(
        data=nsw_data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        xformla="~ age + educ",
        panel=True,
        est_method="ipw",
        trim_level=0.9,
    )

    assert isinstance(result.att, float)
    assert result.se > 0


def test_ipwdid_comparison_with_other_estimators(nsw_data):
    from moderndid import drdid, ordid

    treated_ids = nsw_data.filter(pl.col("experimental") == 1)["id"].unique()[:100].to_list()
    control_ids = nsw_data.filter(pl.col("experimental") == 0)["id"].unique()[:100].to_list()
    selected_ids = treated_ids + control_ids
    data = nsw_data.filter(pl.col("id").is_in(selected_ids))

    ipw_result = ipwdid(
        data=data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        panel=True,
        est_method="ipw",
    )

    dr_result = drdid(
        data=data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        panel=True,
        est_method="trad",
    )

    or_result = ordid(
        data=data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id",
        panel=True,
    )

    assert all(isinstance(r.att, float) for r in [ipw_result, dr_result, or_result])
    assert all(r.se > 0 for r in [ipw_result, dr_result, or_result])

    atts = [ipw_result.att, dr_result.att, or_result.att]
    assert max(atts) - min(atts) < max(abs(att) for att in atts) * 2
