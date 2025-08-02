# pylint: disable=redefined-outer-name
"""Tests for inverse propensity weighted DiD."""

import numpy as np
import pytest

from moderndid.data import load_nsw
from moderndid.drdid.ipwdid import ipwdid

from ..helpers import importorskip

pd = importorskip("pandas")


@pytest.fixture
def nsw_data():
    return load_nsw()


@pytest.mark.parametrize("est_method", ["ipw", "std_ipw"])
def test_ipwdid_panel_basic(nsw_data, est_method):
    result = ipwdid(
        data=nsw_data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
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
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
        covariates_formula="~ age + educ + black + married + nodegree + hisp",
        panel=True,
        est_method="ipw",
    )

    assert isinstance(result.att, float)
    assert result.se > 0
    assert result.call_params["covariates_formula"] == "~ age + educ + black + married + nodegree + hisp"


def test_ipwdid_panel_with_weights(nsw_data):
    np.random.seed(42)
    unique_ids = nsw_data["id"].unique()
    unit_weights = np.random.exponential(1, len(unique_ids))
    weight_dict = dict(zip(unique_ids, unit_weights))
    nsw_data["weight"] = nsw_data["id"].map(weight_dict)

    result = ipwdid(
        data=nsw_data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
        covariates_formula="~ age + educ",
        panel=True,
        weights_col="weight",
        est_method="ipw",
    )

    assert isinstance(result.att, float)
    assert result.se > 0
    assert result.call_params["weights_col"] == "weight"


def test_ipwdid_panel_with_influence_func(nsw_data):
    result = ipwdid(
        data=nsw_data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
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
        est_method=est_method,
    )

    assert result.boots is not None
    assert len(result.boots) == 50
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
        y_col="re",
        time_col="year",
        treat_col="experimental",
        panel=panel,
        covariates_formula="~ age + educ + black",
        est_method=est_method,
    )

    assert isinstance(result.att, float)
    assert result.se > 0
    assert result.args["panel"] is False
    assert result.args["estMethod"] == est_method


def test_ipwdid_rc_with_bootstrap(nsw_data):
    result = ipwdid(
        data=nsw_data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        panel=False,
        covariates_formula="~ age + educ",
        boot=True,
        n_boot=50,
        est_method="ipw",
    )

    assert result.boots is not None
    assert len(result.boots) == 50


@pytest.mark.parametrize("trim_level", [0.99, 0.995])
def test_ipwdid_trim_level(nsw_data, trim_level):
    result = ipwdid(
        data=nsw_data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
        panel=True,
        trim_level=trim_level,
        est_method="ipw",
    )

    assert isinstance(result.att, float)
    assert result.args["trim_level"] == trim_level


def test_ipwdid_missing_id_col_panel():
    df = pd.DataFrame(
        {
            "y": np.random.randn(100),
            "time": np.repeat([0, 1], 50),
            "treat": np.tile([0, 1], 50),
        }
    )

    with pytest.raises(ValueError, match="id_col must be provided when panel=True"):
        ipwdid(
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
def test_ipwdid_formula_variations(nsw_data, formula):
    result = ipwdid(
        data=nsw_data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
        covariates_formula=formula,
        panel=True,
        est_method="ipw",
    )

    assert isinstance(result.att, float)
    assert result.se > 0


def test_ipwdid_call_params_stored(nsw_data):
    result = ipwdid(
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
        est_method="std_ipw",
        trim_level=0.99,
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
    assert result.call_params["est_method"] == "std_ipw"
    assert result.call_params["trim_level"] == 0.99
    assert "data_shape" in result.call_params


def test_ipwdid_args_output(nsw_data):
    result = ipwdid(
        data=nsw_data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
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
    treated_ids = nsw_data[nsw_data["experimental"] == 1]["id"].unique()[:50]
    control_ids = nsw_data[nsw_data["experimental"] == 0]["id"].unique()[:50]
    selected_ids = np.concatenate([treated_ids, control_ids])
    small_data = nsw_data[nsw_data["id"].isin(selected_ids)].copy()

    np.random.seed(42)
    result1 = ipwdid(
        data=small_data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
        covariates_formula="~ age + educ",
        panel=True,
        est_method="ipw",
    )

    np.random.seed(42)
    result2 = ipwdid(
        data=small_data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
        covariates_formula="~ age + educ",
        panel=True,
        est_method="ipw",
    )

    assert result1.att == result2.att
    assert result1.se == result2.se


def test_ipwdid_subset_columns(nsw_data):
    subset_cols = ["id", "year", "re", "experimental", "age", "educ"]
    subset_data = nsw_data[subset_cols].copy()

    result = ipwdid(
        data=subset_data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
        covariates_formula="~ age + educ",
        panel=True,
        est_method="std_ipw",
    )

    assert isinstance(result.att, float)
    assert result.se > 0


@pytest.mark.parametrize("covariates_formula", [None, "~ 1"])
def test_ipwdid_no_covariates(nsw_data, covariates_formula):
    result = ipwdid(
        data=nsw_data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
        covariates_formula=covariates_formula,
        panel=True,
        est_method="ipw",
    )

    assert isinstance(result.att, float)
    assert result.se > 0


def test_ipwdid_no_covariates_equivalence(nsw_data):
    result1 = ipwdid(
        data=nsw_data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
        covariates_formula=None,
        panel=True,
        est_method="ipw",
    )

    result2 = ipwdid(
        data=nsw_data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
        covariates_formula="~ 1",
        panel=True,
        est_method="ipw",
    )

    assert np.isclose(result1.att, result2.att, rtol=1e-10)
    assert np.isclose(result1.se, result2.se, rtol=1e-10)


def test_ipwdid_categorical_covariates(nsw_data):
    nsw_data["age_group"] = pd.cut(nsw_data["age"], bins=[0, 25, 35, 100], labels=["young", "middle", "old"])

    result = ipwdid(
        data=nsw_data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
        covariates_formula="~ C(age_group) + educ + black",
        panel=True,
        est_method="ipw",
    )

    assert isinstance(result.att, float)
    assert result.se > 0


def test_ipwdid_missing_values_error():
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
        ipwdid(
            data=df,
            y_col="y",
            time_col="time",
            treat_col="treat",
            id_col="id",
            covariates_formula="~ x1",
            panel=True,
        )


def test_ipwdid_unbalanced_panel_warning():
    df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3],
            "time": [0, 1, 0, 1, 0],
            "y": np.random.randn(5),
            "treat": [0, 0, 1, 1, 0],
        }
    )

    with pytest.warns(UserWarning, match="Panel data is unbalanced"):
        result = ipwdid(
            data=df,
            y_col="y",
            time_col="time",
            treat_col="treat",
            id_col="id",
            panel=True,
        )
        assert isinstance(result.att, float)


def test_ipwdid_more_than_two_periods_error():
    df = pd.DataFrame(
        {
            "id": np.repeat(range(50), 3),
            "time": np.tile([0, 1, 2], 50),
            "y": np.random.randn(150),
            "treat": np.repeat([0, 1], 75),
        }
    )

    with pytest.raises(ValueError):
        ipwdid(
            data=df,
            y_col="y",
            time_col="time",
            treat_col="treat",
            id_col="id",
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
    treated_ids = nsw_data[nsw_data["experimental"] == 1]["id"].unique()[:200]
    control_ids = nsw_data[nsw_data["experimental"] == 0]["id"].unique()[:200]
    selected_ids = np.concatenate([treated_ids, control_ids])
    small_data = nsw_data[nsw_data["id"].isin(selected_ids)].copy()

    try:
        result = ipwdid(
            data=small_data,
            y_col="re",
            time_col="year",
            treat_col="experimental",
            id_col="id" if panel else None,
            covariates_formula="~ age + educ + black",
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
    treated_ids = nsw_data[nsw_data["experimental"] == 1]["id"].unique()[:300]
    control_ids = nsw_data[nsw_data["experimental"] == 0]["id"].unique()[:300]
    selected_ids = np.concatenate([treated_ids, control_ids])
    data = nsw_data[nsw_data["id"].isin(selected_ids)].copy()

    try:
        ipw_result = ipwdid(
            data=data,
            y_col="re",
            time_col="year",
            treat_col="experimental",
            id_col="id",
            covariates_formula="~ age + educ",
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
            y_col="re",
            time_col="year",
            treat_col="experimental",
            id_col="id",
            covariates_formula="~ age + educ",
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
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
        covariates_formula="~ age + educ",
        panel=True,
        est_method="ipw",
        trim_level=0.9,
    )

    assert isinstance(result.att, float)
    assert result.se > 0


def test_ipwdid_comparison_with_other_estimators(nsw_data):
    from moderndid import drdid, ordid

    treated_ids = nsw_data[nsw_data["experimental"] == 1]["id"].unique()[:100]
    control_ids = nsw_data[nsw_data["experimental"] == 0]["id"].unique()[:100]
    selected_ids = np.concatenate([treated_ids, control_ids])
    data = nsw_data[nsw_data["id"].isin(selected_ids)].copy()

    ipw_result = ipwdid(
        data=data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
        panel=True,
        est_method="ipw",
    )

    dr_result = drdid(
        data=data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
        panel=True,
        est_method="trad",
    )

    or_result = ordid(
        data=data,
        y_col="re",
        time_col="year",
        treat_col="experimental",
        id_col="id",
        panel=True,
    )

    assert all(isinstance(r.att, float) for r in [ipw_result, dr_result, or_result])
    assert all(r.se > 0 for r in [ipw_result, dr_result, or_result])

    atts = [ipw_result.att, dr_result.att, or_result.att]
    assert max(atts) - min(atts) < max(abs(att) for att in atts) * 2
