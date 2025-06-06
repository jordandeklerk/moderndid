"""Tests for the preprocessing functions."""

import numpy as np
import pytest

from .helpers import importorskip

pd = importorskip("pandas")

from pydid.utils import preprocess_drdid, preprocess_synth
from tests.dgp import SantAnnaZhaoDRDiD


def test_pre_process_santanna_zhao_rcs():
    dgp = SantAnnaZhaoDRDiD(n_units=100, random_seed=42)
    data_dict = dgp.generate_data(att=0.5)
    df = data_dict["df"]

    y_col = "y"
    time_col = "post"
    id_col = "id"
    treat_col = "d"

    result = preprocess_drdid(
        data=df,
        y_col=y_col,
        time_col=time_col,
        id_col=id_col,
        treat_col=treat_col,
        covariates_formula="~ x1 + x2 + x3 + x4",
        panel=False,
    )

    assert isinstance(result, dict)
    assert result["panel"] is False
    assert "y" in result
    assert "D" in result
    assert "post" in result
    assert "covariates" in result
    assert "weights" in result
    assert "n_obs" in result
    assert len(result["y"]) == result["n_obs"]
    assert len(result["y"]) == len(df)
    assert result["covariates"].shape == (len(df), 5)


@pytest.mark.parametrize("panel_setting", [True, False])
def test_preprocess_drdid_empty_df(panel_setting):
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Missing required columns"):
        preprocess_drdid(
            data=empty_df,
            y_col="outcome",
            time_col="time",
            id_col="id",
            treat_col="treatment",
            panel=panel_setting,
        )


@pytest.mark.parametrize(
    "invalid_treatment_data",
    [
        pd.DataFrame({"time": [0, 1], "outcome": [1, 2], "id": [1, 1], "treatment": [0, 2]}),
        pd.DataFrame({"time": [0, 1], "outcome": [1, 2], "id": [1, 1], "treatment": [-1, 1]}),
    ],
)
@pytest.mark.parametrize("panel_setting", [True, False])
def test_preprocess_drdid_invalid_treatment_values(invalid_treatment_data, panel_setting):
    with pytest.raises(ValueError, match="Treatment indicator column must contain only 0 .* and 1"):
        preprocess_drdid(
            data=invalid_treatment_data,
            y_col="outcome",
            time_col="time",
            id_col="id",
            treat_col="treatment",
            panel=panel_setting,
        )


@pytest.mark.parametrize(
    "single_group_data",
    [
        pd.DataFrame(
            {
                "time": [0, 1, 0, 1],
                "outcome": [1, 2, 3, 4],
                "id": [1, 1, 2, 2],
                "treatment": [0, 0, 0, 0],
            }
        ),
        pd.DataFrame(
            {
                "time": [0, 1, 0, 1],
                "outcome": [1, 2, 3, 4],
                "id": [1, 1, 2, 2],
                "treatment": [1, 1, 1, 1],
            }
        ),
    ],
)
@pytest.mark.parametrize("panel_setting", [True, False])
def test_preprocess_drdid_single_treatment_group(single_group_data, panel_setting):
    match_err = "Treatment indicator column must contain only 0 .* and 1"
    with pytest.raises(ValueError, match=match_err):
        preprocess_drdid(
            data=single_group_data,
            y_col="outcome",
            time_col="time",
            id_col="id",
            treat_col="treatment",
            panel=panel_setting,
        )


@pytest.mark.parametrize("panel_setting", [True, False])
def test_preprocess_drdid_non_numeric_weights(panel_setting):
    data = pd.DataFrame(
        {
            "time": [0, 1, 0, 1, 0, 1, 0, 1],
            "outcome": [1, 2, 3, 4, 5, 6, 7, 8],
            "id": [1, 1, 2, 2, 3, 3, 4, 4],
            "treatment": [0, 0, 0, 0, 1, 1, 1, 1],
            "w": ["a", "b", "c", "d", "e", "f", "g", "h"],
        }
    )
    with pytest.raises(TypeError, match="Column 'w' must be numeric. Could not convert."):
        preprocess_drdid(
            data=data,
            y_col="outcome",
            time_col="time",
            id_col="id",
            treat_col="treatment",
            weights_col="w",
            panel=panel_setting,
        )


@pytest.mark.parametrize("panel_setting", [True, False])
def test_preprocess_drdid_negative_weights(panel_setting):
    data = pd.DataFrame(
        {
            "time": [0, 1, 0, 1, 0, 1, 0, 1],
            "outcome": [1, 2, 3, 4, 5, 6, 7, 8],
            "id": [1, 1, 2, 2, 3, 3, 4, 4],
            "treatment_val": [0, 0, 0, 0, 1, 1, 1, 1],
            "w": [1, 1, -1, -1, 1, 1, -2, -2],
        }
    )
    if panel_setting:
        unit_treatment_status = data.groupby("id")["treatment_val"].first()
        data["treat_col_for_test"] = data["id"].map(unit_treatment_status)
        treat_col_to_use = "treat_col_for_test"
    else:
        data["treat_col_for_test"] = data["treatment_val"]
        treat_col_to_use = "treat_col_for_test"

    with pytest.warns(UserWarning, match="Some weights are negative."):
        preprocess_drdid(
            data=data,
            y_col="outcome",
            time_col="time",
            id_col="id",
            treat_col=treat_col_to_use,
            weights_col="w",
            panel=panel_setting,
        )


@pytest.mark.parametrize("panel_setting", [True, False])
def test_preprocess_drdid_zero_weights(panel_setting):
    data = pd.DataFrame(
        {
            "time": [0, 1, 0, 1, 0, 1, 0, 1],
            "outcome": [1, 2, 3, 4, 5, 6, 7, 8],
            "id": [1, 1, 2, 2, 3, 3, 4, 4],
            "treatment": [0, 0, 0, 0, 1, 1, 1, 1],
            "w": [0, 0, 0, 0, 0, 0, 0, 0],
        }
    )
    with pytest.warns(UserWarning, match="Mean of weights is zero or negative. Cannot normalize."):
        result = preprocess_drdid(
            data=data,
            y_col="outcome",
            time_col="time",
            id_col="id",
            treat_col="treatment",
            weights_col="w",
            panel=panel_setting,
            normalized=True,
        )
    assert "weights" in result
    if panel_setting:
        assert np.all(result["weights"] == 0)
    else:
        assert np.all(result["weights"] == 0)


def test_preprocess_drdid_panel_missing_id_col():
    data = pd.DataFrame({"time": [0, 1, 0, 1], "outcome": [1, 2, 3, 4], "treatment": [0, 0, 1, 1]})
    with pytest.raises(ValueError, match="Missing required columns: id"):
        preprocess_drdid(
            data=data,
            y_col="outcome",
            time_col="time",
            id_col="id",
            treat_col="treatment",
            panel=True,
        )


def test_preprocess_drdid_panel_duplicate_id_time():
    data = pd.DataFrame(
        {
            "time": [0, 1, 0, 1, 0, 1, 0, 1],
            "outcome": [1, 2, 3, 4, 5, 6, 7, 8],
            "id": [1, 1, 1, 1, 2, 2, 2, 2],
            "treatment": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    unit_treatment_status = data.groupby("id")["treatment"].first()
    data["is_treated_unit"] = data["id"].map(unit_treatment_status)

    with pytest.raises(ValueError, match="ID 'id' is not unique within time period 'time'"):
        preprocess_drdid(
            data=data,
            y_col="outcome",
            time_col="time",
            id_col="id",
            treat_col="is_treated_unit",
            panel=True,
        )


def test_preprocess_synth_empty_df():
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Missing required columns"):
        preprocess_synth(
            data=empty_df,
            y_col="outcome",
            time_col="time",
            id_col="id",
            treat_col="treatment",
            treatment_period=1,
        )


def test_preprocess_synth_treatment_period_not_in_time():
    data = pd.DataFrame(
        {
            "time": [0, 1, 0, 1],
            "outcome": [1, 2, 3, 4],
            "id": [1, 1, 2, 2],
            "treatment": [0, 0, 1, 1],
        }
    )
    with pytest.raises(ValueError, match="treatment_period .* not found in time column"):
        preprocess_synth(
            data=data,
            y_col="outcome",
            time_col="time",
            id_col="id",
            treat_col="treatment",
            treatment_period=2,
        )


def test_preprocess_synth_treatment_period_is_first():
    data = pd.DataFrame(
        {
            "time": [0, 1, 2, 0, 1, 2],
            "outcome": [1, 2, 3, 4, 5, 6],
            "id": [1, 1, 1, 2, 2, 2],
            "treatment": [0, 0, 0, 1, 1, 1],
        }
    )
    with pytest.raises(ValueError, match="treatment_period cannot be the first period"):
        preprocess_synth(
            data=data,
            y_col="outcome",
            time_col="time",
            id_col="id",
            treat_col="treatment",
            treatment_period=0,
        )


def test_preprocess_synth_no_pre_treatment_periods():
    data = pd.DataFrame(
        {
            "time": [1, 2, 3, 1, 2, 3],
            "outcome": [1, 2, 3, 4, 5, 6],
            "id": [1, 1, 1, 2, 2, 2],
            "treatment": [0, 0, 0, 1, 1, 1],
        }
    )
    with pytest.raises(ValueError, match="treatment_period cannot be the first period"):
        preprocess_synth(
            data=data,
            y_col="outcome",
            time_col="time",
            id_col="id",
            treat_col="treatment",
            treatment_period=1,
        )


def test_preprocess_synth_non_numeric_weights():
    data = pd.DataFrame(
        {
            "time": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "outcome": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "treatment": [0, 0, 0, 1, 1, 1, 0, 0, 0],
            "w": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
        }
    )
    with pytest.raises(TypeError, match="Weights column 'w' must be numeric"):
        preprocess_synth(
            data=data,
            y_col="outcome",
            time_col="time",
            id_col="id",
            treat_col="treatment",
            treatment_period=1,
            weights_col="w",
        )


def test_preprocess_synth_zero_weights():
    data = pd.DataFrame(
        {
            "time": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "outcome": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "treatment": [0, 0, 0, 1, 1, 1, 0, 0, 0],
            "w": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        }
    )
    with pytest.warns(UserWarning, match="Mean of weights is zero or negative. Cannot normalize."):
        result = preprocess_synth(
            data=data,
            y_col="outcome",
            time_col="time",
            id_col="id",
            treat_col="treatment",
            treatment_period=1,
            weights_col="w",
            normalized=True,
        )
    assert "weights_treat" in result
    assert "weights_control" in result
    assert np.all(result["weights_treat"] == 0)
    assert np.all(result["weights_control"] == 0)


def test_preprocess_synth_collinearity_warning():
    data = pd.DataFrame(
        {
            "time": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "id": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            "outcome": [5, 6, 7, 20, 10, 11, 12, 21, 13, 14, 15, 22, 16, 17, 18, 23],
            "X1": [0, 0, 0, 0, 10, 11, 12, 21, 0, 0, 0, 0, 0, 0, 0, 0],
            "treatment": [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        }
    )
    data["is_treated_unit"] = data["id"].map(data.groupby("id")["treatment"].first())

    with pytest.warns(UserWarning, match="Potential collinearity detected among predictors"):
        preprocess_synth(
            data=data,
            y_col="outcome",
            time_col="time",
            id_col="id",
            treat_col="is_treated_unit",
            treatment_period=2,
            l_outcome_periods=[1],
            covariates_formula="~ X1",
        )


def test_preprocess_synth_negative_weights():
    data = pd.DataFrame(
        {
            "time": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "outcome": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "treatment": [0, 0, 0, 1, 1, 1, 0, 0, 0],
            "w": [1, 1, 1, -1, -1, -1, 1, 1, 1],
        }
    )
    with pytest.warns(UserWarning, match="Some weights are negative."):
        preprocess_synth(
            data=data,
            y_col="outcome",
            time_col="time",
            id_col="id",
            treat_col="treatment",
            treatment_period=1,
            weights_col="w",
        )


@pytest.mark.parametrize("scenario", ["no_treated_after_balance", "no_control_after_balance"])
def test_preprocess_synth_no_units_after_balancing(scenario):
    base_data = {
        "time": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
        "outcome": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        "treatment": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    }
    df = pd.DataFrame(base_data)

    treatment_period = 2

    if scenario == "no_treated_after_balance":
        df_modified = df[~((df["id"] == 2) & (df["time"] == 0))]
        expected_error_match = "No treated units remain after balancing panel"
    else:
        df_modified = df[~((df["id"] == 1) & (df["time"] == 1))]
        df_modified = df_modified[~((df_modified["id"] == 3) & (df_modified["time"] == 0))]
        expected_error_match = "No control units remain after balancing panel"

    with pytest.raises(ValueError, match=expected_error_match):
        preprocess_synth(
            data=df_modified,
            y_col="outcome",
            time_col="time",
            id_col="id",
            treat_col="treatment",
            treatment_period=treatment_period,
            post_periods_of_interest=[2, 3],
        )


def test_preprocess_synth_invalid_l_outcome_periods():
    data = pd.DataFrame(
        {
            "time": [0, 1, 2, 3, 0, 1, 2, 3],
            "outcome": [1, 2, 3, 4, 5, 6, 7, 8],
            "id": [1, 1, 1, 1, 2, 2, 2, 2],
            "treatment": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    with pytest.raises(ValueError, match="All l_outcome_periods must be in the pre-treatment period"):
        preprocess_synth(
            data=data,
            y_col="outcome",
            time_col="time",
            id_col="id",
            treat_col="treatment",
            treatment_period=2,
            l_outcome_periods=[0, 1, 2],
        )


@pytest.mark.parametrize("invalid_post_periods", [([0, 1, 2]), ([2, 3, 4])])
def test_preprocess_synth_invalid_post_periods_of_interest(invalid_post_periods):
    data = pd.DataFrame(
        {
            "time": [0, 1, 2, 3, 0, 1, 2, 3],
            "outcome": [1, 2, 3, 4, 5, 6, 7, 8],
            "id": [1, 1, 1, 1, 2, 2, 2, 2],
            "treatment": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    match_str = (
        "All post_periods_of_interest must be >= treatment_period"
        if any(p < 2 for p in invalid_post_periods)
        else "Not all post_periods_of_interest found in time column"
    )
    with pytest.raises(ValueError, match=match_str):
        preprocess_synth(
            data=data,
            y_col="outcome",
            time_col="time",
            id_col="id",
            treat_col="treatment",
            treatment_period=2,
            post_periods_of_interest=invalid_post_periods,
        )


@pytest.mark.parametrize(
    "treatment_setup",
    [
        {
            "data": pd.DataFrame(
                {
                    "time": [0, 1, 0, 1],
                    "outcome": [1, 2, 3, 4],
                    "id": [1, 1, 2, 2],
                    "treatment": [0, 0, 0, 0],
                }
            ),
            "match": "No treated units found",
        },
        {
            "data": pd.DataFrame(
                {
                    "time": [0, 1, 0, 1],
                    "outcome": [1, 2, 3, 4],
                    "id": [1, 1, 2, 2],
                    "treatment": [1, 1, 1, 1],
                }
            ),
            "match": "No control units found",
        },
    ],
)
def test_preprocess_synth_no_treated_or_control_units(treatment_setup):
    data = treatment_setup["data"]
    match_error = treatment_setup["match"]

    with pytest.raises(ValueError, match=match_error):
        preprocess_synth(
            data=data,
            y_col="outcome",
            time_col="time",
            id_col="id",
            treat_col="treatment",
            treatment_period=1,
        )
