"""Tests for the data generating process for DiD and Synthetic Controls."""

import numpy as np
import pandas as pd
import pytest

from .dgp import BaseDGP, DiDDGP, SyntheticControlsDGP


def test_abstract_methods():
    with pytest.raises(TypeError):
        BaseDGP()  # pylint: disable=abstract-class-instantiated


def test_set_seed():
    dgp = DiDDGP()

    dgp.set_seed(42)
    data1 = dgp.generate_data()

    dgp.set_seed(42)
    data2 = dgp.generate_data()

    np.testing.assert_array_equal(data1["features"], data2["features"])
    np.testing.assert_array_equal(data1["unit_effects"], data2["unit_effects"])

    dgp.set_seed(99)
    data3 = dgp.generate_data()

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(data1["features"], data3["features"])


def test_did_dgp_init():
    dgp = DiDDGP(n_units=50, n_time=10, treatment_time=5, n_features=3, random_seed=42)

    assert dgp.n_units == 50
    assert dgp.n_time == 10
    assert dgp.treatment_time == 5
    assert dgp.n_features == 3
    assert dgp.random_seed == 42


def test_did_dgp_generate_data():
    dgp = DiDDGP(n_units=100, n_time=5, treatment_time=3, random_seed=42)
    data = dgp.generate_data(
        treatment_fraction=0.6,
        unit_effect_scale=0.8,
        time_effect_scale=0.4,
        noise_scale=0.1,
        effect_base=2.0,
        effect_coef=1.0,
        effect_feature_idx=0,
        confounding_strength=0.7,
    )

    assert "df" in data
    assert "treatment_effects" in data
    assert "unit_effects" in data
    assert "time_effects" in data
    assert "features" in data

    df = data["df"]

    assert len(df) == 100 * 5
    assert set(df.columns) == {"unit_id", "time_id", "treatment", "X1", "X2", "outcome"}

    treated_units = df[df["treatment"]]["unit_id"].unique()
    assert len(treated_units) >= 50
    assert len(treated_units) <= 70

    treatments = df[df["treatment"]]
    assert all(t >= 3 for t in treatments["time_id"])

    assert len(data["unit_effects"]) == 100
    assert len(data["time_effects"]) == 5
    assert data["features"].shape == (100, 2)

    assert len(data["treatment_effects"]) == len(treated_units)


def test_did_dgp_confounding():
    dgp = DiDDGP(n_units=1000, n_time=4, treatment_time=2, random_seed=42)

    data_conf = dgp.generate_data(confounding_strength=1.0)
    df_conf = data_conf["df"]

    data_no_conf = dgp.generate_data(confounding_strength=0.0)
    df_no_conf = data_no_conf["df"]

    units_conf = df_conf[df_conf["time_id"] == 0]
    x_conf = units_conf[["X1", "X2"]].values

    treat_units_conf = set(df_conf[df_conf["treatment"]]["unit_id"].unique())
    t_conf = np.array([1.0 if i in treat_units_conf else 0.0 for i in units_conf["unit_id"]])

    units_no_conf = df_no_conf[df_no_conf["time_id"] == 0]
    x_no_conf = units_no_conf[["X1", "X2"]].values

    treat_units_no_conf = set(df_no_conf[df_no_conf["treatment"]]["unit_id"].unique())
    t_no_conf = np.array([1 if i in treat_units_no_conf else 0 for i in units_no_conf["unit_id"]])

    corr_conf = np.corrcoef(x_conf[:, 0], t_conf)[0, 1]
    corr_no_conf = np.corrcoef(x_no_conf[:, 0], t_no_conf)[0, 1]

    assert abs(corr_conf) > abs(corr_no_conf) + 0.1


def test_did_dgp_heterogeneity():
    dgp = DiDDGP(n_units=100, n_time=4, treatment_time=2, random_seed=42)

    data_homogeneous = dgp.generate_data(effect_base=1.0, effect_coef=0.0)

    data_heterogeneous = dgp.generate_data(effect_base=1.0, effect_coef=2.0)

    var_homogeneous = np.var(data_homogeneous["treatment_effects"])
    var_heterogeneous = np.var(data_heterogeneous["treatment_effects"])

    assert var_homogeneous < var_heterogeneous


def test_did_dgp_staggered():
    dgp = DiDDGP(n_units=100, n_time=6, random_seed=42)

    treatment_times = [2, 3, 4]
    dynamic_effects = [0.5, 1.0, 1.5, 2.0]

    data = dgp.generate_staggered_adoption(
        treatment_times=treatment_times,
        dynamic_effects=dynamic_effects,
        treatment_fraction=0.7,
    )

    df = data["df"]

    first_treated_values = df[~df["first_treated"].isna()]["first_treated"].unique()
    assert all(t in treatment_times for t in first_treated_values)

    treated_units = df[~df["first_treated"].isna()]["unit_id"].unique()
    assert len(treated_units) >= 60
    assert len(treated_units) <= 80

    assert data["dynamic_effects"] == dynamic_effects

    treated_data = df[df["treatment"]].copy()
    treated_data["time_since_treatment"] = treated_data["time_id"] - treated_data["first_treated"]

    outcomes_by_duration = []
    for unit in treated_units[:10]:
        unit_data = treated_data[treated_data["unit_id"] == unit]
        if len(unit_data) >= 3:
            for t_idx in range(min(3, len(dynamic_effects))):
                if t_idx in unit_data["time_since_treatment"].values:
                    time_filter = unit_data["time_since_treatment"] == t_idx
                    outcome_series = unit_data[time_filter]["outcome"]
                    outcomes_by_duration.append(outcome_series.values[0])

    if len(outcomes_by_duration) > 0:
        outcomes_array = np.array(outcomes_by_duration)
        assert np.mean(outcomes_array) > 0


def test_sc_dgp_init():
    dgp = SyntheticControlsDGP(
        n_treated_units=2,
        n_control_units=10,
        n_time_pre=8,
        n_time_post=4,
        n_features=2,
        random_seed=123,
    )
    assert dgp.n_treated_units == 2
    assert dgp.n_control_units == 10
    assert dgp.n_time_pre == 8
    assert dgp.n_time_post == 4
    assert dgp.n_features == 2
    assert dgp.random_seed == 123

    with pytest.raises(ValueError):
        SyntheticControlsDGP(n_treated_units=0)
    with pytest.raises(ValueError):
        SyntheticControlsDGP(n_control_units=0)
    with pytest.raises(ValueError):
        SyntheticControlsDGP(n_time_pre=0)
    with pytest.raises(ValueError):
        SyntheticControlsDGP(n_time_post=0)


def test_sc_dgp_generate_data():
    n_treated = 2
    n_control = 15
    n_pre = 7
    n_post = 3
    n_feat = 2

    dgp = SyntheticControlsDGP(
        n_treated_units=n_treated,
        n_control_units=n_control,
        n_time_pre=n_pre,
        n_time_post=n_post,
        n_features=n_feat,
        random_seed=42,
    )
    data = dgp.generate_data(
        feature_effect_scale=0.8,
        unit_effect_scale=0.6,
        time_effect_scale=0.2,
        noise_scale=0.15,
        treatment_effect_value=1.5,
        confounding_strength=0.4,
    )

    assert "df" in data
    assert "true_treatment_effect" in data
    assert "unit_effects" in data
    assert "time_effects" in data
    assert "features" in data
    assert "feature_weights" in data
    assert "n_treated_units" in data
    assert "n_control_units" in data
    assert "n_time_pre" in data
    assert "n_time_post" in data

    df = data["df"]
    n_total_units = n_treated + n_control
    n_total_time = n_pre + n_post

    assert len(df) == n_total_units * n_total_time
    expected_cols = {"unit_id", "time_id", "outcome", "is_treated_unit", "is_post_period"}
    for i in range(1, n_feat + 1):
        expected_cols.add(f"X{i}")
    assert set(df.columns) == expected_cols

    assert data["true_treatment_effect"] == 1.5
    assert len(data["unit_effects"]) == n_total_units
    assert len(data["time_effects"]) == n_total_time
    assert data["features"].shape == (n_total_units, n_feat)
    assert len(data["feature_weights"]) == n_feat
    assert data["n_treated_units"] == n_treated
    assert data["n_control_units"] == n_control
    assert data["n_time_pre"] == n_pre
    assert data["n_time_post"] == n_post

    assert df["is_treated_unit"].sum() == n_treated * n_total_time
    assert df["is_post_period"].sum() == n_total_units * n_post

    treated_post_outcome_mean = df[(df["is_treated_unit"]) & (df["is_post_period"])]["outcome"].mean()
    control_post_outcome_mean = df[(~df["is_treated_unit"]) & (df["is_post_period"])]["outcome"].mean()

    if n_treated > 0 and n_control > 0 and n_post > 0:
        assert treated_post_outcome_mean > control_post_outcome_mean - 1.0


def test_sc_dgp_generate_data_no_features():
    dgp = SyntheticControlsDGP(n_features=0, random_seed=43)
    data = dgp.generate_data()
    df = data["df"]
    assert "X1" not in df.columns
    assert data["features"].shape == (dgp.n_treated_units + dgp.n_control_units, 0)
    assert len(data["feature_weights"]) == 0


def test_sc_dgp_confounding():
    n_total_units = 500
    dgp_conf = SyntheticControlsDGP(
        n_treated_units=10, n_control_units=n_total_units - 10, n_features=1, random_seed=42
    )
    data_conf = dgp_conf.generate_data(confounding_strength=0.9, unit_effect_scale=0.1)

    dgp_no_conf = SyntheticControlsDGP(
        n_treated_units=10, n_control_units=n_total_units - 10, n_features=1, random_seed=42
    )
    data_no_conf = dgp_no_conf.generate_data(confounding_strength=0.0, unit_effect_scale=0.1)

    features_conf = data_conf["features"][:, 0]
    unit_effects_conf = data_conf["unit_effects"]

    features_no_conf = data_no_conf["features"][:, 0]
    unit_effects_no_conf = data_no_conf["unit_effects"]

    corr_conf = np.corrcoef(features_conf, unit_effects_conf)[0, 1]
    corr_no_conf = np.corrcoef(features_no_conf, unit_effects_no_conf)[0, 1]

    assert abs(corr_conf) > abs(corr_no_conf) + 0.1


def test_set_seed_sc():
    dgp = SyntheticControlsDGP(n_features=1, random_seed=420)

    dgp.set_seed(420)
    data1 = dgp.generate_data()

    dgp.set_seed(420)
    data2 = dgp.generate_data()

    np.testing.assert_array_equal(data1["features"], data2["features"])
    np.testing.assert_array_equal(data1["unit_effects"], data2["unit_effects"])
    np.testing.assert_array_equal(data1["time_effects"], data2["time_effects"])
    np.testing.assert_array_equal(data1["feature_weights"], data2["feature_weights"])
    pd.testing.assert_frame_equal(data1["df"], data2["df"])

    dgp.set_seed(990)
    data3 = dgp.generate_data()

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(data1["features"], data3["features"])
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(data1["df"], data3["df"])
