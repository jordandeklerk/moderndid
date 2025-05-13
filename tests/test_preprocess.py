"""Tests for the preprocessing functions."""

import numpy as np
import pytest

from drsynthdid.preprocess import pre_process_drdid
from tests.dgp import DiD, SantAnnaZhaoDRDiD


def test_pre_process_panel():
    dgp = DiD(n_units=50, n_time=4, treatment_time=2, n_features=2)
    data_dict = dgp.generate_data()
    df = data_dict["df"]

    y_col = "outcome"
    time_col = "time_id"
    id_col = "unit_id"
    unit_treatment_status = df.groupby(id_col)["treatment"].any()
    df["is_treated_unit"] = df[id_col].map(unit_treatment_status)
    treat_col = "is_treated_unit"

    df_filtered = df[df[time_col].isin([dgp.treatment_time - 1, dgp.treatment_time])].copy()

    result_filtered = pre_process_drdid(
        data=df_filtered,
        y_col=y_col,
        time_col=time_col,
        id_col=id_col,
        treat_col=treat_col,
        covariates_formula="~ X1 + X2",
        panel=True,
    )

    assert isinstance(result_filtered, dict)
    assert result_filtered["panel"] is True
    assert "y1" in result_filtered
    assert "y0" in result_filtered
    assert "D" in result_filtered
    assert "covariates" in result_filtered
    assert "weights" in result_filtered
    assert "n_units" in result_filtered
    expected_units = df_filtered[id_col].nunique()
    assert result_filtered["n_units"] == expected_units
    assert len(result_filtered["y1"]) == expected_units


def test_repeated_cross_section():
    dgp = DiD(n_units=60, n_time=2, treatment_time=1, n_features=1)
    data_dict = dgp.generate_data()
    df = data_dict["df"]

    y_col = "outcome"
    time_col = "time_id"
    id_col = "unit_id"
    treat_col = "treatment"

    result = pre_process_drdid(
        data=df,
        y_col=y_col,
        time_col=time_col,
        id_col=id_col,
        treat_col=treat_col,
        covariates_formula="~ X1",
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


def test_missing_value_handling():
    dgp = DiD(n_units=50, n_time=2, treatment_time=1, n_features=1)
    data_dict = dgp.generate_data()
    df = data_dict["df"]

    y_col = "outcome"
    time_col = "time_id"
    id_col = "unit_id"
    unit_treatment_status = df.groupby(id_col)["treatment"].any()
    df["is_treated_unit"] = df[id_col].map(unit_treatment_status)
    treat_col = "is_treated_unit"

    df.loc[0, "outcome"] = np.nan
    df.loc[1, "X1"] = np.nan

    with pytest.warns(UserWarning, match="Missing values found"):
        result = pre_process_drdid(
            data=df,
            y_col=y_col,
            time_col=time_col,
            id_col=id_col,
            treat_col=treat_col,
            covariates_formula="~ X1",
            panel=True,
        )

    ids_with_na = df[df[[y_col, "X1", treat_col]].isna().any(axis=1)][id_col].unique()
    expected_units_after_drop = df[id_col].nunique() - len(ids_with_na)

    assert result["n_units"] == expected_units_after_drop
    assert len(result["y1"]) == expected_units_after_drop


def test_collinearity_warning():
    dgp = DiD(n_units=50, n_time=2, treatment_time=1, n_features=1)
    data_dict = dgp.generate_data()
    df = data_dict["df"]

    df["X2"] = df["X1"] * 2

    y_col = "outcome"
    time_col = "time_id"
    id_col = "unit_id"
    unit_treatment_status = df.groupby(id_col)["treatment"].any()
    df["is_treated_unit"] = df[id_col].map(unit_treatment_status)
    treat_col = "is_treated_unit"

    with pytest.warns(UserWarning, match="Potential collinearity detected"):
        result = pre_process_drdid(
            data=df,
            y_col=y_col,
            time_col=time_col,
            id_col=id_col,
            treat_col=treat_col,
            covariates_formula="~ X1 + X2",
            panel=True,
        )
    assert "covariates" in result


def test_time_invariance_error_panel():
    dgp = DiD(n_units=50, n_time=2, treatment_time=1, n_features=1)
    data_dict = dgp.generate_data()
    df = data_dict["df"]

    y_col = "outcome"
    time_col = "time_id"
    id_col = "unit_id"
    treat_col = "treatment"

    with pytest.raises(ValueError, match="Treatment indicator.*must be unique for each ID"):
        pre_process_drdid(
            data=df.copy(),
            y_col=y_col,
            time_col=time_col,
            id_col=id_col,
            treat_col=treat_col,
            covariates_formula="~ X1",
            panel=True,
        )

    df_mod = df.copy()
    df_mod.loc[df_mod[time_col] == 1, "X1"] = df_mod["X1"] + 1
    unit_treatment_status = df_mod.groupby(id_col)["treatment"].any()
    df_mod["is_treated_unit"] = df_mod[id_col].map(unit_treatment_status)
    treat_col_invariant = "is_treated_unit"

    with pytest.raises(ValueError, match="Covariates must be time-invariant"):
        pre_process_drdid(
            data=df_mod,
            y_col=y_col,
            time_col=time_col,
            id_col=id_col,
            treat_col=treat_col_invariant,
            covariates_formula="~ X1",
            panel=True,
        )


def test_pre_process_santanna_zhao_rcs():
    dgp = SantAnnaZhaoDRDiD(n_units=100, random_seed=42)
    data_dict = dgp.generate_data(att=0.5)
    df = data_dict["df"]

    y_col = "y"
    time_col = "post"
    id_col = "id"
    treat_col = "d"

    result = pre_process_drdid(
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
