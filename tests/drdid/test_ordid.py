# pylint: disable=redefined-outer-name
"""Tests for outcome regression DiD."""

import numpy as np
import pytest

from moderndid.data import load_nsw
from moderndid.drdid.ordid import ordid

from ..helpers import importorskip

pd = importorskip("pandas")


@pytest.fixture
def nsw_data():
    return load_nsw()


def test_ordid_panel_basic(nsw_data):
    result = ordid(
        data=nsw_data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
        panel=True,
    )

    assert isinstance(result.att, float)
    assert isinstance(result.se, float)
    assert result.se > 0
    assert result.lci < result.att < result.uci
    assert result.boots is None
    assert result.att_inf_func is None
    assert result.call_params["panel"] is True
    assert result.args["type"] == "or"


def test_ordid_panel_with_covariates(nsw_data):
    result = ordid(
        data=nsw_data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
        covariates_formula="~ age + educ + black + married + nodegree + hisp",
        panel=True,
    )

    assert isinstance(result.att, float)
    assert result.se > 0
    assert result.call_params["covariates_formula"] == "~ age + educ + black + married + nodegree + hisp"


def test_ordid_panel_with_weights(nsw_data):
    np.random.seed(42)
    unique_ids = nsw_data["id"].unique()
    unit_weights = np.random.exponential(1, len(unique_ids))
    weight_dict = dict(zip(unique_ids, unit_weights))
    nsw_data["weight"] = nsw_data["id"].map(weight_dict)

    result = ordid(
        data=nsw_data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
        covariates_formula="~ age + educ",
        panel=True,
        weights_col="weight",
    )

    assert isinstance(result.att, float)
    assert result.se > 0
    assert result.call_params["weights_col"] == "weight"


def test_ordid_panel_with_influence_func(nsw_data):
    result = ordid(
        data=nsw_data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
        panel=True,
        inf_func=True,
    )

    assert result.att_inf_func is not None
    n_units = len(nsw_data["id"].unique())
    assert len(result.att_inf_func) == n_units
    assert np.isclose(np.mean(result.att_inf_func), 0, atol=1e-10)


@pytest.mark.parametrize("boot_type", ["weighted", "multiplier"])
def test_ordid_panel_bootstrap(nsw_data, boot_type):
    result = ordid(
        data=nsw_data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
        covariates_formula="~ age + educ",
        panel=True,
        boot=True,
        boot_type=boot_type,
        n_boot=50,
        inf_func=(boot_type == "multiplier"),
    )

    assert result.boots is not None
    assert len(result.boots) == result.args["nboot"]
    assert not np.all(np.isnan(result.boots))
    assert result.args["boot"] is True
    assert result.args["boot_type"] == boot_type
    if boot_type == "multiplier":
        assert result.att_inf_func is not None


def test_ordid_repeated_cross_section(nsw_data):
    result = ordid(
        data=nsw_data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        panel=False,
        covariates_formula="~ age + educ + black",
    )

    assert isinstance(result.att, float)
    assert result.se > 0
    assert result.args["panel"] is False


def test_ordid_rc_with_bootstrap(nsw_data):
    result = ordid(
        data=nsw_data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        panel=False,
        covariates_formula="~ age + educ",
        boot=True,
        n_boot=50,
    )

    assert result.boots is not None
    assert len(result.boots) == result.args["nboot"]


def test_ordid_missing_id_col_panel():
    df = pd.DataFrame(
        {
            "y": np.random.randn(100),
            "time": np.repeat([0, 1], 50),
            "treat": np.tile([0, 1], 50),
        }
    )

    with pytest.raises(ValueError, match="id_col must be provided when panel=True"):
        ordid(
            data=df,
            y_col="y",
            time_col="time",
            treat_col="treat",
            panel=True,
        )


@pytest.mark.parametrize(
    "formula",
    [
        "~ age + educ + age:educ",
        "~ age + I(age**2) + educ",
    ],
)
def test_ordid_formula_variations(nsw_data, formula):
    result = ordid(
        data=nsw_data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
        covariates_formula=formula,
        panel=True,
    )

    assert isinstance(result.att, float)
    assert result.se > 0


def test_ordid_call_params_stored(nsw_data):
    result = ordid(
        data=nsw_data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
        covariates_formula="~ age + educ",
        panel=True,
        boot=True,
        n_boot=50,
        inf_func=True,
    )

    assert result.call_params["y_col"] == "re"
    assert result.call_params["time_col"] == "year"
    assert result.call_params["treat_col"] == "experimental"
    assert result.call_params["id_col"] == "id"
    assert result.call_params["covariates_formula"] == "~ age + educ"
    assert result.call_params["panel"] is True
    assert result.call_params["boot"] is True
    assert result.call_params["n_boot"] == 50
    assert result.call_params["inf_func"] is True
    assert "data_shape" in result.call_params


def test_ordid_args_output(nsw_data):
    result = ordid(
        data=nsw_data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
        panel=True,
        boot=True,
        boot_type="weighted",
        n_boot=50,
    )

    assert result.args["panel"] is True
    assert result.args["normalized"] is True
    assert result.args["boot"] is True
    assert result.args["boot_type"] == "weighted"
    assert result.args["nboot"] == 50
    assert result.args["type"] == "or"


def test_ordid_reproducibility(nsw_data):
    treated_ids = nsw_data[nsw_data["experimental"] == 1]["id"].unique()[:50]
    control_ids = nsw_data[nsw_data["experimental"] == 0]["id"].unique()[:50]
    selected_ids = np.concatenate([treated_ids, control_ids])
    small_data = nsw_data[nsw_data["id"].isin(selected_ids)].copy()

    np.random.seed(42)
    result1 = ordid(
        data=small_data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
        covariates_formula="~ age + educ",
        panel=True,
    )

    np.random.seed(42)
    result2 = ordid(
        data=small_data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
        covariates_formula="~ age + educ",
        panel=True,
    )

    assert result1.att == result2.att
    assert result1.se == result2.se


def test_ordid_subset_columns(nsw_data):
    subset_cols = ["id", "year", "re", "experimental", "age", "educ"]
    subset_data = nsw_data[subset_cols].copy()

    result = ordid(
        data=subset_data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
        covariates_formula="~ age + educ",
        panel=True,
    )

    assert isinstance(result.att, float)
    assert result.se > 0


@pytest.mark.parametrize("covariates_formula", [None, "~ 1"])
def test_ordid_no_covariates(nsw_data, covariates_formula):
    result = ordid(
        data=nsw_data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
        covariates_formula=covariates_formula,
        panel=True,
    )

    assert isinstance(result.att, float)
    assert result.se > 0


def test_ordid_no_covariates_equivalence(nsw_data):
    result1 = ordid(
        data=nsw_data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
        covariates_formula=None,
        panel=True,
    )

    result2 = ordid(
        data=nsw_data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
        covariates_formula="~ 1",
        panel=True,
    )

    assert np.isclose(result1.att, result2.att, rtol=1e-10)
    assert np.isclose(result1.se, result2.se, rtol=1e-10)


def test_ordid_categorical_covariates(nsw_data):
    nsw_data["age_group"] = pd.cut(nsw_data["age"], bins=[0, 25, 35, 100], labels=["young", "middle", "old"])

    result = ordid(
        data=nsw_data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
        covariates_formula="~ C(age_group) + educ + black",
        panel=True,
    )

    assert isinstance(result.att, float)
    assert result.se > 0


def test_ordid_missing_values_error():
    df = pd.DataFrame(
        {
            "id": np.repeat(range(50), 2),
            "time": np.tile([0, 1], 50),
            "y": np.random.randn(100),
            "treat": np.repeat([0, 1], 50),
            "x1": np.random.randn(100),
        }
    )
    df.loc[5, "y"] = np.nan

    with pytest.raises(ValueError):
        ordid(
            data=df,
            y_col="y",
            time_col="time",
            treat_col="treat",
            id_col="id",
            covariates_formula="~ x1",
            panel=True,
        )


def test_ordid_unbalanced_panel_error():
    df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3],
            "time": [0, 1, 0, 1, 0],
            "y": np.random.randn(5),
            "treat": [0, 0, 1, 1, 0],
        }
    )

    with pytest.raises(ValueError):
        ordid(
            data=df,
            y_col="y",
            time_col="time",
            treat_col="treat",
            id_col="id",
            panel=True,
        )


def test_ordid_more_than_two_periods_error():
    df = pd.DataFrame(
        {
            "id": np.repeat(range(50), 3),
            "time": np.tile([0, 1, 2], 50),
            "y": np.random.randn(150),
            "treat": np.repeat([0, 1], 75),
        }
    )

    with pytest.raises(ValueError):
        ordid(
            data=df,
            y_col="y",
            time_col="time",
            treat_col="treat",
            id_col="id",
            panel=True,
        )
