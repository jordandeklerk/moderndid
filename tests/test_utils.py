"""Tests for the preprocessing functions."""

import numpy as np
import pytest

from .helpers import importorskip

pd = importorskip("pandas")

from didpy.drdid.utils import preprocess_drdid
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
