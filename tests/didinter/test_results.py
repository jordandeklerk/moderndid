"""Tests for didinter result containers."""

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid.didinter import (
    ATEResult,
    DIDInterResult,
    EffectsResult,
    HeterogeneityResult,
    PlacebosResult,
)


@pytest.fixture
def effects_result():
    return EffectsResult(
        horizons=np.array([1, 2, 3]),
        estimates=np.array([0.5, 0.6, 0.7]),
        std_errors=np.array([0.1, 0.12, 0.15]),
        ci_lower=np.array([0.3, 0.36, 0.4]),
        ci_upper=np.array([0.7, 0.84, 1.0]),
        n_switchers=np.array([100, 90, 80]),
        n_observations=np.array([500, 450, 400]),
    )


@pytest.fixture
def placebos_result():
    return PlacebosResult(
        horizons=np.array([-1, -2]),
        estimates=np.array([0.01, -0.02]),
        std_errors=np.array([0.05, 0.06]),
        ci_lower=np.array([-0.09, -0.14]),
        ci_upper=np.array([0.11, 0.10]),
        n_switchers=np.array([100, 95]),
        n_observations=np.array([500, 480]),
    )


@pytest.fixture
def minimal_effects():
    return EffectsResult(
        horizons=np.array([1]),
        estimates=np.array([0.5]),
        std_errors=np.array([0.1]),
        ci_lower=np.array([0.3]),
        ci_upper=np.array([0.7]),
        n_switchers=np.array([100]),
        n_observations=np.array([500]),
    )


def test_effects_result_creation(effects_result):
    assert len(effects_result.horizons) == 3
    assert len(effects_result.estimates) == 3
    np.testing.assert_array_equal(effects_result.horizons, [1, 2, 3])


@pytest.mark.parametrize(
    "expected_column",
    ["Horizon", "Estimate", "Std. Error", "CI Lower", "CI Upper", "N Switchers", "N Obs"],
)
def test_effects_result_to_dataframe_columns(effects_result, expected_column):
    df = effects_result.to_dataframe()

    assert isinstance(df, pl.DataFrame)
    assert expected_column in df.columns


def test_effects_result_to_dataframe_values(effects_result):
    df = effects_result.to_dataframe()

    assert len(df) == 3
    np.testing.assert_array_almost_equal(df["Estimate"].to_numpy(), [0.5, 0.6, 0.7])


def test_placebos_result_creation(placebos_result):
    assert len(placebos_result.horizons) == 2
    np.testing.assert_array_equal(placebos_result.horizons, [-1, -2])


def test_placebos_result_to_dataframe(placebos_result):
    df = placebos_result.to_dataframe()

    assert isinstance(df, pl.DataFrame)
    assert len(df) == 2
    np.testing.assert_array_almost_equal(df["Horizon"].to_numpy(), [-1, -2])


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

    df = result.to_dataframe()

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

    df = result.to_dataframe()

    assert len(df) == 3
    assert df["Horizon"].to_list() == [2, 2, 2]


def test_didinter_result_creation(minimal_effects):
    result = DIDInterResult(
        effects=minimal_effects,
        n_units=200,
        n_switchers=100,
        n_never_switchers=100,
        ci_level=95.0,
    )

    assert result.effects is minimal_effects
    assert result.placebos is None
    assert result.ate is None
    assert result.n_units == 200
    assert result.n_switchers == 100
    assert result.n_never_switchers == 100
    assert result.ci_level == 95.0


def test_didinter_result_with_placebos(minimal_effects, placebos_result):
    result = DIDInterResult(
        effects=minimal_effects,
        placebos=placebos_result,
        n_units=200,
        n_switchers=100,
        n_never_switchers=100,
    )

    assert result.placebos is not None
    assert len(result.placebos.horizons) == 2


def test_didinter_result_with_tests(minimal_effects):
    effects_equal_test = {"chi2_stat": 1.5, "df": 1, "p_value": 0.22}
    placebo_joint_test = {"chi2_stat": 2.3, "df": 2, "p_value": 0.32}

    result = DIDInterResult(
        effects=minimal_effects,
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
        ("ci_level", 95.0),
        ("effects_equal_test", None),
        ("placebo_joint_test", None),
        ("influence_effects", None),
        ("influence_placebos", None),
        ("heterogeneity", None),
    ],
)
def test_didinter_result_defaults(minimal_effects, field, expected_default):
    result = DIDInterResult(effects=minimal_effects)

    assert getattr(result, field) == expected_default


def test_didinter_result_estimation_params_default(minimal_effects):
    result = DIDInterResult(effects=minimal_effects)

    assert result.estimation_params == {}


def test_didinter_result_estimation_params(minimal_effects):
    params = {
        "effects": 3,
        "placebo": 2,
        "normalized": True,
        "switchers": "in",
    }

    result = DIDInterResult(effects=minimal_effects, estimation_params=params)

    assert result.estimation_params == params
    assert result.estimation_params["effects"] == 3
    assert result.estimation_params["normalized"] is True
