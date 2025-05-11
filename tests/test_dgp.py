"""Tests for the data generating process for DiD and Synthetic Controls."""

import numpy as np
import pandas as pd
import pytest

from .dgp import BaseDGP, DiD, DRDiDSC, SantAnnaZhaoDRDiD, SyntheticControl


def test_abstract_methods():
    with pytest.raises(TypeError):
        BaseDGP()  # pylint: disable=abstract-class-instantiated


def test_set_seed():
    dgp = DiD()

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
    dgp = DiD(n_units=50, n_time=10, treatment_time=5, n_features=3, random_seed=42)

    assert dgp.n_units == 50
    assert dgp.n_time == 10
    assert dgp.treatment_time == 5
    assert dgp.n_features == 3
    assert dgp.random_seed == 42


def test_did_dgp_generate_data():
    dgp = DiD(n_units=100, n_time=5, treatment_time=3, random_seed=42)
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
    dgp = DiD(n_units=1000, n_time=4, treatment_time=2, random_seed=42)

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
    dgp = DiD(n_units=100, n_time=4, treatment_time=2, random_seed=42)

    data_homogeneous = dgp.generate_data(effect_base=1.0, effect_coef=0.0)

    data_heterogeneous = dgp.generate_data(effect_base=1.0, effect_coef=2.0)

    var_homogeneous = np.var(data_homogeneous["treatment_effects"])
    var_heterogeneous = np.var(data_heterogeneous["treatment_effects"])

    assert var_homogeneous < var_heterogeneous


def test_did_dgp_staggered():
    dgp = DiD(n_units=100, n_time=6, random_seed=42)

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


def test_sc_dgp_init_success():
    dgp = SyntheticControl(
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


@pytest.mark.parametrize(
    "invalid_params",
    [
        {"n_treated_units": 0},
        {"n_control_units": 0},
        {"n_time_pre": 0},
        {"n_time_post": 0},
    ],
)
def test_sc_dgp_init_value_errors(invalid_params):
    with pytest.raises(ValueError):
        SyntheticControl(**invalid_params)


def test_sc_dgp_generate_data():
    n_treated = 2
    n_control = 15
    n_pre = 7
    n_post = 3
    n_feat = 2

    dgp = SyntheticControl(
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
    dgp = SyntheticControl(n_features=0, random_seed=43)
    data = dgp.generate_data()
    df = data["df"]
    assert "X1" not in df.columns
    assert data["features"].shape == (dgp.n_treated_units + dgp.n_control_units, 0)
    assert len(data["feature_weights"]) == 0


def test_sc_dgp_confounding():
    n_total_units = 500
    dgp_conf = SyntheticControl(n_treated_units=10, n_control_units=n_total_units - 10, n_features=1, random_seed=42)
    data_conf = dgp_conf.generate_data(confounding_strength=0.9, unit_effect_scale=0.1)

    dgp_no_conf = SyntheticControl(n_treated_units=10, n_control_units=n_total_units - 10, n_features=1, random_seed=42)
    data_no_conf = dgp_no_conf.generate_data(confounding_strength=0.0, unit_effect_scale=0.1)

    features_conf = data_conf["features"][:, 0]
    unit_effects_conf = data_conf["unit_effects"]

    features_no_conf = data_no_conf["features"][:, 0]
    unit_effects_no_conf = data_no_conf["unit_effects"]

    corr_conf = np.corrcoef(features_conf, unit_effects_conf)[0, 1]
    corr_no_conf = np.corrcoef(features_no_conf, unit_effects_no_conf)[0, 1]

    assert abs(corr_conf) > abs(corr_no_conf) + 0.1


def test_set_seed_sc():
    dgp = SyntheticControl(n_features=1, random_seed=420)

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


def test_drdidsc_dgp_init():
    dgp = DRDiDSC(
        n_units=50,
        n_time_pre=8,
        n_time_post=4,
        n_features=2,
        n_latent_factors=1,
        random_seed=123,
    )
    assert dgp.n_units == 50
    assert dgp.n_time_pre == 8
    assert dgp.n_time_post == 4
    assert dgp.n_total_time == 12
    assert dgp.treatment_time == 8
    assert dgp.n_features == 2
    assert dgp.n_latent_factors == 1
    assert dgp.random_seed == 123


def test_drdidsc_dgp_generate_data_defaults():
    n_units, n_time_pre, n_time_post, n_features, n_latent_factors = 20, 5, 3, 2, 1
    dgp = DRDiDSC(
        n_units=n_units,
        n_time_pre=n_time_pre,
        n_time_post=n_time_post,
        n_features=n_features,
        n_latent_factors=n_latent_factors,
        random_seed=42,
    )
    data = dgp.generate_data()

    expected_keys = {
        "df",
        "X",
        "Y0_it",
        "actual_tau_it",
        "true_prop_scores",
        "D_treatment_assignment",
        "L_factor_loadings",
        "F_time_factors",
        "alpha_i_unit_effects",
        "delta_t_time_effects",
        "beta_prop_propensity_coefs",
        "beta_outcome_outcome_coefs",
        "att_hetero_coefs",
        "avg_true_att",
        "n_treated_units_actual",
        "n_control_units_actual",
        "treatment_time",
    }
    assert set(data.keys()) == expected_keys

    df = data["df"]
    n_total_time = n_time_pre + n_time_post
    assert isinstance(df, pd.DataFrame)
    assert len(df) == n_units * n_total_time
    expected_df_cols = {"unit_id", "time_id", "outcome", "is_treated_unit", "treatment_in_period"}
    for i in range(1, n_features + 1):
        expected_df_cols.add(f"X{i}")
    assert set(df.columns) == expected_df_cols

    assert data["X"].shape == (n_units, n_features)
    assert data["Y0_it"].shape == (n_units, n_total_time)
    assert data["actual_tau_it"].shape == (n_units, n_total_time)
    assert data["true_prop_scores"].shape == (n_units,)
    assert data["D_treatment_assignment"].shape == (n_units,)
    assert data["L_factor_loadings"].shape == (n_units, n_latent_factors)
    assert data["F_time_factors"].shape == (n_total_time, n_latent_factors)
    assert data["alpha_i_unit_effects"].shape == (n_units,)
    assert data["delta_t_time_effects"].shape == (n_total_time,)

    if n_features > 0:
        assert data["beta_prop_propensity_coefs"].shape == (n_features,)
        assert data["beta_outcome_outcome_coefs"].shape == (n_features,)
    else:
        assert len(data["beta_prop_propensity_coefs"]) == 0
        assert len(data["beta_outcome_outcome_coefs"]) == 0

    assert isinstance(data["avg_true_att"], float | np.floating)
    assert data["treatment_time"] == n_time_pre

    assert data["n_treated_units_actual"] == np.sum(data["D_treatment_assignment"])
    assert data["n_control_units_actual"] == n_units - np.sum(data["D_treatment_assignment"])
    assert df["is_treated_unit"].sum() == np.sum(data["D_treatment_assignment"]) * n_total_time

    for _, row in df.iterrows():
        is_treated_unit = data["D_treatment_assignment"][int(row["unit_id"])]
        is_post_period = row["time_id"] >= n_time_pre
        assert row["treatment_in_period"] == (is_treated_unit and is_post_period)


def test_drdidsc_dgp_no_features_no_factors():
    n_units, n_time_pre, n_time_post = 10, 3, 2
    dgp = DRDiDSC(
        n_units=n_units,
        n_time_pre=n_time_pre,
        n_time_post=n_time_post,
        n_features=0,
        n_latent_factors=0,
        random_seed=43,
    )
    data = dgp.generate_data()

    assert data["X"].shape == (n_units, 0)
    assert "X1" not in data["df"].columns
    assert data["L_factor_loadings"].shape == (n_units, 0)
    assert data["F_time_factors"].shape == (n_time_pre + n_time_post, 0)
    assert len(data["beta_prop_propensity_coefs"]) == 0
    assert len(data["beta_outcome_outcome_coefs"]) == 0
    assert len(data["att_hetero_coefs"]) == 0


def test_drdidsc_dgp_custom_params():
    n_units, n_time_pre, n_time_post, n_features = 15, 4, 3, 1
    dgp = DRDiDSC(
        n_units=n_units,
        n_time_pre=n_time_pre,
        n_time_post=n_time_post,
        n_features=n_features,
        random_seed=44,
    )

    custom_prop_coefs = np.array([2.0])
    custom_outcome_coefs = np.array([-1.0])
    custom_att_hetero_coefs = np.array([0.5])
    custom_dynamic_coeffs = [1.0, 1.5]

    data = dgp.generate_data(
        feature_mean=5.0,
        feature_cov=np.array([[4.0]]),
        prop_score_coefs=custom_prop_coefs,
        prop_score_intercept=-0.5,
        treatment_fraction_target=0.33,
        outcome_coefs=custom_outcome_coefs,
        att_base=0.5,
        att_hetero_coefs=custom_att_hetero_coefs,
        att_dynamic_coeffs=custom_dynamic_coeffs,
    )

    assert data["X"].shape == (n_units, n_features)
    if n_units > 0 and n_features > 0:
        assert np.isclose(np.mean(data["X"][:, 0]), 5.0, atol=1.5)

    np.testing.assert_array_equal(data["beta_prop_propensity_coefs"], custom_prop_coefs)
    np.testing.assert_array_equal(data["beta_outcome_outcome_coefs"], custom_outcome_coefs)
    np.testing.assert_array_equal(data["att_hetero_coefs"], custom_att_hetero_coefs)

    expected_treated = int(n_units * 0.33)
    if n_units * 0.33 == n_units and n_units > 1:
        expected_treated = n_units - 1
    elif n_units * 0.33 == n_units and n_units <= 1:
        expected_treated = 0

    assert data["n_treated_units_actual"] == expected_treated or data["n_treated_units_actual"] == expected_treated + 1

    treated_mask = data["D_treatment_assignment"]
    if np.any(treated_mask):
        first_treated_unit_idx = np.where(treated_mask)[0][0]
        tau_unit = data["actual_tau_it"][first_treated_unit_idx, :]

        assert np.all(tau_unit[:n_time_pre] == 0)

        base_effect_for_unit = 0.5
        if n_features > 0:
            base_effect_for_unit += data["X"][first_treated_unit_idx, 0] * custom_att_hetero_coefs[0]

        if n_time_post > 0:
            assert np.isclose(tau_unit[n_time_pre], base_effect_for_unit * custom_dynamic_coeffs[0])
        if n_time_post > 1:
            assert np.isclose(tau_unit[n_time_pre + 1], base_effect_for_unit * custom_dynamic_coeffs[1])
        if n_time_post > 2:
            assert np.isclose(tau_unit[n_time_pre + 2], base_effect_for_unit * custom_dynamic_coeffs[-1])


def test_drdidsc_dgp_set_seed():
    dgp_params = dict(n_units=10, n_time_pre=3, n_time_post=2, n_features=1, random_seed=777)
    dgp1 = DRDiDSC(**dgp_params)
    data1 = dgp1.generate_data()

    dgp2 = DRDiDSC(**dgp_params)
    data2 = dgp2.generate_data()

    pd.testing.assert_frame_equal(data1["df"], data2["df"])
    np.testing.assert_array_equal(data1["X"], data2["X"])
    np.testing.assert_array_equal(data1["Y0_it"], data2["Y0_it"])
    np.testing.assert_array_equal(data1["actual_tau_it"], data2["actual_tau_it"])
    np.testing.assert_array_equal(data1["true_prop_scores"], data2["true_prop_scores"])
    assert data1["avg_true_att"] == data2["avg_true_att"]

    dgp1.set_seed(888)
    data3 = dgp1.generate_data()

    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(data1["df"], data3["df"])
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(data1["X"], data3["X"])


@pytest.mark.parametrize(
    "n_units, treatment_fraction_target, expected_n_treated_approx",
    [
        (10, 0.0, 0),
        (10, 1.0, 9),
        (10, 0.5, 5),
        (1, 0.0, 0),
        (1, 1.0, 0),
        (2, 1.0, 1),
        (0, 0.5, 0),
    ],
)
def test_drdidsc_dgp_treatment_assignment_edge_cases(n_units, treatment_fraction_target, expected_n_treated_approx):
    dgp = DRDiDSC(n_units=n_units, n_time_pre=2, n_time_post=1, n_features=0, random_seed=45)
    data = dgp.generate_data(treatment_fraction_target=treatment_fraction_target)

    assert data["n_treated_units_actual"] == expected_n_treated_approx
    if n_units > 0:
        assert data["n_control_units_actual"] == n_units - expected_n_treated_approx
    else:
        assert data["n_control_units_actual"] == 0

    assert np.sum(data["D_treatment_assignment"]) == expected_n_treated_approx

    df = data["df"]
    if n_units > 0:
        assert df["is_treated_unit"].sum() == expected_n_treated_approx * (dgp.n_time_pre + dgp.n_time_post)

        assert df["treatment_in_period"].sum() == expected_n_treated_approx * dgp.n_time_post
    else:
        assert len(df) == 0


@pytest.mark.parametrize(
    "dgp_init_kwargs, generate_data_args, generate_data_kwargs, match_pattern",
    [
        (
            {"n_features": 2},
            (),
            {"feature_mean": np.array([1.0, 2.0, 3.0])},
            "feature_mean must be scalar or",
        ),
        (
            {"n_features": 2},
            (),
            {"feature_cov": np.eye(3)},
            "feature_cov must be",
        ),
        (
            {"n_features": 2},
            (),
            {"prop_score_coefs": np.array([1.0])},
            "prop_score_coefs must be",
        ),
        (
            {"n_features": 2},
            (),
            {"outcome_coefs": np.array([0.5, 0.1, 0.2])},
            "outcome_coefs must be",
        ),
        (
            {"n_features": 2},
            (),
            {"att_hetero_coefs": np.array([0.5])},
            "att_hetero_coefs must be",
        ),
        (
            {},
            (1, 2, 3),
            {},
            "Unexpected positional arguments",
        ),
        (
            {},
            (),
            {"unexpected_param": 100},
            "Unexpected keyword arguments",
        ),
    ],
)
def test_drdidsc_dgp_generate_data_value_errors(
    dgp_init_kwargs, generate_data_args, generate_data_kwargs, match_pattern
):
    dgp = DRDiDSC(**dgp_init_kwargs)
    with pytest.raises(ValueError, match=match_pattern):
        dgp.generate_data(*generate_data_args, **generate_data_kwargs)


def test_santannazhaodrdid_init():
    dgp = SantAnnaZhaoDRDiD(
        n_units=500,
        treatment_fraction=0.7,
        common_support_strength=0.9,
        random_seed=4321,
    )

    assert dgp.n_units == 500
    assert dgp.treatment_fraction == 0.7
    assert dgp.common_support_strength == 0.9
    assert dgp.random_seed == 4321


def test_santannazhaodrdid_generate_data():
    dgp = SantAnnaZhaoDRDiD(n_units=200, random_seed=1234)
    data = dgp.generate_data(att=0.5)

    assert "df" in data
    assert "true_att" in data
    assert "raw_covariates" in data
    assert "transformed_covariates" in data
    assert "propensity_scores" in data
    assert "potential_outcomes_pre" in data
    assert "potential_outcomes_post" in data

    df = data["df"]
    assert len(df) == 200
    assert set(df.columns) == {"id", "post", "y", "d", "x1", "x2", "x3", "x4"}

    assert data["true_att"] == 0.5
    assert data["raw_covariates"].shape == (200, 4)
    assert data["transformed_covariates"].shape == (200, 4)
    assert data["propensity_scores"].shape == (200,)
    assert all(0 <= p <= 1 for p in data["propensity_scores"])

    assert all(d in [0, 1] for d in df["d"])
    assert all(p in [0, 1] for p in df["post"])

    assert "y0" in data["potential_outcomes_pre"]
    assert "y0" in data["potential_outcomes_post"]
    assert "y1" in data["potential_outcomes_post"]
    assert len(data["potential_outcomes_pre"]["y0"]) == 200
    assert len(data["potential_outcomes_post"]["y0"]) == 200
    assert len(data["potential_outcomes_post"]["y1"]) == 200


def test_santannazhaodrdid_set_seed():
    dgp1 = SantAnnaZhaoDRDiD(n_units=100, random_seed=5678)
    data1 = dgp1.generate_data(att=0.0)

    dgp2 = SantAnnaZhaoDRDiD(n_units=100, random_seed=5678)
    data2 = dgp2.generate_data(att=0.0)

    pd.testing.assert_frame_equal(data1["df"], data2["df"])

    np.testing.assert_array_equal(data1["raw_covariates"], data2["raw_covariates"])
    np.testing.assert_array_equal(data1["transformed_covariates"], data2["transformed_covariates"])
    np.testing.assert_array_equal(data1["propensity_scores"], data2["propensity_scores"])

    dgp3 = SantAnnaZhaoDRDiD(n_units=100, random_seed=9999)
    data3 = dgp3.generate_data(att=0.0)

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(data1["raw_covariates"], data3["raw_covariates"])


def test_santannazhaodrdid_att_parameter():
    dgp = SantAnnaZhaoDRDiD(n_units=500, random_seed=1234)

    data_att0 = dgp.generate_data(att=0.0)
    dgp.set_seed(1234)
    data_att1 = dgp.generate_data(att=1.0)

    diff_y1 = data_att1["potential_outcomes_post"]["y1"] - data_att0["potential_outcomes_post"]["y1"]

    treated_units = data_att0["df"]["d"] == 1
    assert np.allclose(diff_y1[treated_units], 1.0)

    np.testing.assert_array_equal(
        data_att0["potential_outcomes_post"]["y0"], data_att1["potential_outcomes_post"]["y0"]
    )


@pytest.mark.parametrize(
    "args, kwargs, match_pattern",
    [
        ((1, 2), {}, "Unexpected positional arguments"),
        ((), {"unknown_param": 5}, "Unexpected keyword arguments"),
    ],
)
def test_santannazhaodrdid_value_errors(args, kwargs, match_pattern):
    dgp = SantAnnaZhaoDRDiD()
    with pytest.raises(ValueError, match=match_pattern):
        dgp.generate_data(*args, **kwargs)
