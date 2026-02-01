# pylint: disable=redefined-outer-name
"""Tests for the did_multiplegt main entry point."""

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid import did_multiplegt
from moderndid.didinter import ATEResult, DIDInterResult, EffectsResult, PlacebosResult


def test_basic_estimation(simple_panel_data):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=1,
    )

    assert isinstance(result, DIDInterResult)
    assert isinstance(result.effects, EffectsResult)
    assert len(result.effects.horizons) == 1
    assert len(result.effects.estimates) == 1
    assert result.effects.std_errors[0] > 0
    assert result.n_switchers > 0
    assert result.n_units > 0


def test_multiple_effects_horizons(simple_panel_data):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=3,
    )

    assert len(result.effects.horizons) == 3
    np.testing.assert_array_equal(result.effects.horizons, [1, 2, 3])
    assert all(se > 0 or np.isnan(se) for se in result.effects.std_errors)


def test_with_placebos(simple_panel_data):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        placebo=2,
    )

    assert result.placebos is not None
    assert isinstance(result.placebos, PlacebosResult)
    assert len(result.placebos.horizons) == 2
    np.testing.assert_array_equal(result.placebos.horizons, [-1, -2])


def test_ate_computation(simple_panel_data):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
    )

    assert result.ate is not None
    assert isinstance(result.ate, ATEResult)
    assert isinstance(result.ate.estimate, float)
    assert result.ate.std_error > 0
    assert result.ate.ci_lower < result.ate.ci_upper


def test_normalized_effects(simple_panel_data):
    result_unnorm = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        normalized=False,
    )

    result_norm = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        normalized=True,
    )

    assert not np.allclose(result_norm.effects.estimates, result_unnorm.effects.estimates, rtol=0.01)


@pytest.mark.parametrize("switchers", ["", "in"])
def test_switcher_types(simple_panel_data, switchers):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        switchers=switchers,
    )

    assert result.n_switchers > 0


def test_same_switchers_option(simple_panel_data):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=3,
        same_switchers=True,
    )

    n_switchers = result.effects.n_switchers
    unique_counts = np.unique(n_switchers[~np.isnan(n_switchers)])
    if len(unique_counts) > 0:
        assert len(unique_counts) == 1


def test_only_never_switchers_control(simple_panel_data):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        only_never_switchers=True,
    )

    assert isinstance(result, DIDInterResult)
    assert result.n_never_switchers > 0


def test_with_weights(weighted_panel_data):
    result = did_multiplegt(
        weighted_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        weightsname="w",
    )

    assert isinstance(result, DIDInterResult)
    assert result.estimation_params.get("weightsname") == "w"


def test_with_clustering(clustered_panel_data):
    result = did_multiplegt(
        clustered_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        cluster="cluster",
    )

    assert isinstance(result, DIDInterResult)
    assert result.estimation_params.get("cluster") == "cluster"
    assert result.effects.std_errors[0] > 0


def test_with_controls(panel_with_controls):
    result = did_multiplegt(
        panel_with_controls,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        controls=["x1", "x2"],
    )

    assert isinstance(result, DIDInterResult)
    assert result.estimation_params.get("controls") == ["x1", "x2"]


@pytest.mark.parametrize(
    "test_key",
    ["chi2_stat", "p_value"],
)
def test_effects_equal_test_keys(simple_panel_data, test_key):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=3,
        effects_equal=True,
    )

    assert result.effects_equal_test is not None
    assert test_key in result.effects_equal_test


def test_effects_equal_test_p_value_range(simple_panel_data):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=3,
        effects_equal=True,
    )

    assert 0 <= result.effects_equal_test["p_value"] <= 1


@pytest.mark.parametrize(
    "test_key",
    ["chi2_stat", "p_value"],
)
def test_placebo_joint_test_keys(simple_panel_data, test_key):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        placebo=2,
    )

    if result.placebo_joint_test is not None:
        assert test_key in result.placebo_joint_test


def test_placebo_joint_test_p_value_range(simple_panel_data):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        placebo=2,
    )

    if result.placebo_joint_test is not None:
        assert 0 <= result.placebo_joint_test["p_value"] <= 1


@pytest.mark.parametrize("ci_level", [90.0, 95.0, 99.0])
def test_confidence_interval_levels(simple_panel_data, ci_level):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        ci_level=ci_level,
    )

    assert result.ci_level == ci_level
    assert np.all(result.effects.ci_lower <= result.effects.estimates)
    assert np.all(result.effects.ci_upper >= result.effects.estimates)


def test_influence_functions_returned(simple_panel_data):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        placebo=1,
    )

    assert result.influence_effects is not None
    assert result.influence_effects.shape[1] == 2
    if result.influence_placebos is not None:
        assert result.influence_placebos.shape[1] == 1


@pytest.mark.parametrize(
    "param_key,expected_value",
    [
        ("effects", 3),
        ("placebo", 2),
        ("normalized", True),
        ("same_switchers", True),
    ],
)
def test_estimation_params_stored(simple_panel_data, param_key, expected_value):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=3,
        placebo=2,
        normalized=True,
        same_switchers=True,
    )

    assert result.estimation_params[param_key] == expected_value


@pytest.mark.parametrize("keep_bidirectional", [True, False])
def test_bidirectional_switchers_handling(bidirectional_panel_data, keep_bidirectional):
    result = did_multiplegt(
        bidirectional_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        keep_bidirectional_switchers=keep_bidirectional,
    )

    assert isinstance(result, DIDInterResult)


def test_unbalanced_panel(unbalanced_panel_data):
    result = did_multiplegt(
        unbalanced_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
    )

    assert isinstance(result, DIDInterResult)


def test_result_counts(simple_panel_data):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
    )

    assert result.n_units == result.n_switchers + result.n_never_switchers
    assert result.n_switchers >= 0
    assert result.n_never_switchers >= 0


def test_n_switchers_per_horizon(simple_panel_data):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=3,
    )

    assert len(result.effects.n_switchers) == 3
    assert len(result.effects.n_observations) == 3
    assert np.all(result.effects.n_switchers >= 0)
    assert np.all(result.effects.n_observations >= 0)


def test_real_data_basic(favara_imbs_data):
    result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=2,
    )

    assert isinstance(result, DIDInterResult)
    assert result.n_switchers > 0
    assert result.n_never_switchers > 0
    assert len(result.effects.estimates) == 2


def test_real_data_with_cluster(favara_imbs_data):
    result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=2,
        cluster="state_n",
    )

    assert isinstance(result, DIDInterResult)
    assert all(se > 0 for se in result.effects.std_errors)


def test_real_data_normalized(favara_imbs_data):
    result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=3,
        normalized=True,
    )

    assert isinstance(result, DIDInterResult)
    assert len(result.effects.estimates) == 3
