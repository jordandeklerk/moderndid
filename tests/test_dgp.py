"""Tests for the data generating process."""

import numpy as np
import pytest

from .dgp import BaseDGP, DiDDGP


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
