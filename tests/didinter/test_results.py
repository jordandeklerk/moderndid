"""Tests for didinter result containers."""

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

import moderndid
from moderndid.core.converters import to_df
from moderndid.didinter import (
    ATEResult,
    DIDInterResult,
    EffectsResult,
    HeterogeneityResult,
    PlacebosResult,
)


def test_effects_result_creation(effects_result):
    assert len(effects_result.horizons) == 3
    assert len(effects_result.estimates) == 3
    np.testing.assert_array_equal(effects_result.horizons, [1, 2, 3])


def test_placebos_result_creation(placebos_result):
    assert len(placebos_result.horizons) == 2
    np.testing.assert_array_equal(placebos_result.horizons, [-1, -2])


@pytest.mark.parametrize(
    "field,expected_value",
    [
        ("estimate", 0.55),
        ("std_error", 0.08),
        ("ci_lower", 0.39),
        ("ci_upper", 0.71),
        ("n_observations", 1500.0),
        ("n_switchers", 270.0),
    ],
)
def test_ate_result_fields(field, expected_value):
    result = ATEResult(
        estimate=0.55,
        std_error=0.08,
        ci_lower=0.39,
        ci_upper=0.71,
        n_observations=1500.0,
        n_switchers=270.0,
    )

    assert getattr(result, field) == expected_value


@pytest.mark.parametrize(
    "field,expected_default",
    [
        ("n_observations", 0.0),
        ("n_switchers", 0.0),
    ],
)
def test_ate_result_defaults(field, expected_default):
    result = ATEResult(
        estimate=0.55,
        std_error=0.08,
        ci_lower=0.39,
        ci_upper=0.71,
    )

    assert getattr(result, field) == expected_default


def test_heterogeneity_result_creation():
    result = HeterogeneityResult(
        horizon=1,
        covariates=["x1", "x2"],
        estimates=np.array([0.1, 0.2]),
        std_errors=np.array([0.05, 0.06]),
        t_stats=np.array([2.0, 3.33]),
        ci_lower=np.array([0.0, 0.08]),
        ci_upper=np.array([0.2, 0.32]),
        n_obs=200,
        f_pvalue=0.01,
    )

    assert result.horizon == 1
    assert result.covariates == ["x1", "x2"]
    assert len(result.estimates) == 2
    assert result.n_obs == 200
    assert result.f_pvalue == 0.01


@pytest.mark.parametrize(
    "expected_column",
    ["Horizon", "Covariate", "Estimate", "t-stat", "F p-value"],
)
def test_heterogeneity_result_to_dataframe_columns(expected_column):
    result = HeterogeneityResult(
        horizon=2,
        covariates=["x1", "x2", "x3"],
        estimates=np.array([0.1, 0.2, 0.15]),
        std_errors=np.array([0.05, 0.06, 0.04]),
        t_stats=np.array([2.0, 3.33, 3.75]),
        ci_lower=np.array([0.0, 0.08, 0.07]),
        ci_upper=np.array([0.2, 0.32, 0.23]),
        n_obs=200,
        f_pvalue=0.01,
    )

    df = to_df(result)

    assert isinstance(df, pl.DataFrame)
    assert expected_column in df.columns


def test_heterogeneity_result_to_dataframe_values():
    result = HeterogeneityResult(
        horizon=2,
        covariates=["x1", "x2", "x3"],
        estimates=np.array([0.1, 0.2, 0.15]),
        std_errors=np.array([0.05, 0.06, 0.04]),
        t_stats=np.array([2.0, 3.33, 3.75]),
        ci_lower=np.array([0.0, 0.08, 0.07]),
        ci_upper=np.array([0.2, 0.32, 0.23]),
        n_obs=200,
        f_pvalue=0.01,
    )

    df = to_df(result)

    assert len(df) == 3
    assert df["Horizon"].to_list() == [2, 2, 2]


def test_didinter_result_creation(single_horizon_effects):
    result = DIDInterResult(
        effects=single_horizon_effects,
        n_units=200,
        n_switchers=100,
        n_never_switchers=100,
        alpha=0.05,
    )

    assert result.effects is single_horizon_effects
    assert result.placebos is None
    assert result.ate is None
    assert result.n_units == 200
    assert result.n_switchers == 100
    assert result.n_never_switchers == 100
    assert result.alpha == 0.05


def test_didinter_result_with_placebos(single_horizon_effects, placebos_result):
    result = DIDInterResult(
        effects=single_horizon_effects,
        placebos=placebos_result,
        n_units=200,
        n_switchers=100,
        n_never_switchers=100,
    )

    assert result.placebos is not None
    assert len(result.placebos.horizons) == 2


def test_didinter_result_with_tests(single_horizon_effects):
    effects_equal_test = {"chi2_stat": 1.5, "df": 1, "p_value": 0.22}
    placebo_joint_test = {"chi2_stat": 2.3, "df": 2, "p_value": 0.32}

    result = DIDInterResult(
        effects=single_horizon_effects,
        effects_equal_test=effects_equal_test,
        placebo_joint_test=placebo_joint_test,
    )

    assert result.effects_equal_test == effects_equal_test
    assert result.placebo_joint_test == placebo_joint_test


@pytest.mark.parametrize(
    "field,expected_default",
    [
        ("placebos", None),
        ("ate", None),
        ("n_units", 0),
        ("n_switchers", 0),
        ("n_never_switchers", 0),
        ("alpha", 0.05),
        ("effects_equal_test", None),
        ("placebo_joint_test", None),
        ("influence_effects", None),
        ("influence_placebos", None),
        ("heterogeneity", None),
    ],
)
def test_didinter_result_defaults(single_horizon_effects, field, expected_default):
    result = DIDInterResult(effects=single_horizon_effects)

    assert getattr(result, field) == expected_default


def test_didinter_result_estimation_params_default(single_horizon_effects):
    result = DIDInterResult(effects=single_horizon_effects)

    assert result.estimation_params == {}


@pytest.mark.parametrize(
    ("params", "se_type", "vcov_info"),
    [
        ({}, "Analytical", {"vcov_type": "analytical", "clustervar": None}),
        ({"cluster": "state"}, "Analytical clustered by state", {"vcov_type": "analytical", "clustervar": "state"}),
        ({"boot": True}, "Bootstrap", {"vcov_type": "bootstrap", "clustervar": None}),
        (
            {"boot": True, "cluster": "state"},
            "Bootstrap clustered by state",
            {"vcov_type": "bootstrap", "clustervar": "state"},
        ),
    ],
)
def test_maketables_labels_the_standard_errors(single_horizon_effects, params, se_type, vcov_info):
    result = DIDInterResult(effects=single_horizon_effects, estimation_params=params)

    assert result.__maketables_stat__("se_type") == se_type
    assert result.__maketables_vcov_info__ == vcov_info


def test_didinter_result_estimation_params(single_horizon_effects):
    params = {
        "effects": 3,
        "placebo": 2,
        "normalized": True,
        "switchers": "in",
    }

    result = DIDInterResult(effects=single_horizon_effects, estimation_params=params)

    assert result.estimation_params == params
    assert result.estimation_params["effects"] == 3
    assert result.estimation_params["normalized"] is True


def test_top_level_names_resolve_to_the_didinter_containers():
    assert moderndid.DIDInterATEResult is ATEResult
    assert moderndid.DIDInterEffectsResult is EffectsResult
    assert moderndid.DIDInterPlacebosResult is PlacebosResult
    assert moderndid.DIDInterResult is DIDInterResult
