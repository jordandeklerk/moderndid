# pylint: disable=redefined-outer-name
"""Tests for aggregate treatment effects."""

import numpy as np
import pytest

from pydid import aggte, att_gt, load_mpdta


@pytest.fixture
def mp_result():
    df = load_mpdta()
    df["first_treat"] = df["first.treat"].replace(0, np.inf)

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first_treat",
        idname="countyreal",
        xformla="~ 1",
        est_method="reg",
    )

    return result


@pytest.mark.parametrize("agg_type", ["simple", "dynamic", "group", "calendar"])
def test_aggte_basic_structure(mp_result, agg_type):
    result = aggte(mp_result, type=agg_type)

    assert result.aggregation_type == agg_type
    assert isinstance(result.overall_att, float | np.floating)
    assert isinstance(result.overall_se, float | np.floating)

    if agg_type == "simple":
        assert result.event_times is None
        assert result.att_by_event is None
        assert result.se_by_event is None
    else:
        assert result.event_times is not None
        assert result.att_by_event is not None
        assert result.se_by_event is not None
        assert len(result.event_times) == len(result.att_by_event)
        assert len(result.event_times) == len(result.se_by_event)


def test_aggte_invalid_type(mp_result):
    with pytest.raises(ValueError, match="must be one of"):
        aggte(mp_result, type="invalid")


def test_aggte_dynamic_with_balance(mp_result):
    result = aggte(mp_result, type="dynamic", balance_e=1)

    assert result.aggregation_type == "dynamic"
    assert result.balanced_event_threshold == 1
    assert isinstance(result.overall_att, float)
    assert isinstance(result.overall_se, float)


def test_aggte_dynamic_with_min_max_e(mp_result):
    result = aggte(mp_result, type="dynamic", min_e=-1, max_e=2)

    assert result.aggregation_type == "dynamic"
    assert result.min_event_time == -1
    assert result.max_event_time == 2
    assert np.all(result.event_times >= -1)
    assert np.all(result.event_times <= 2)


@pytest.mark.parametrize("agg_type", ["simple", "group", "calendar", "dynamic"])
def test_aggte_with_na_rm(mp_result, agg_type):
    result = aggte(mp_result, type=agg_type, na_rm=True)

    assert result.aggregation_type == agg_type
    assert isinstance(result.overall_att, float)
    assert isinstance(result.overall_se, float)


@pytest.mark.parametrize(
    "params,expected",
    [
        ({"bstrap": True, "biters": 99}, {"bootstrap": True, "biters": 99}),
        ({"cband": True, "bstrap": True, "biters": 99}, {"uniform_bands": True, "bootstrap": True}),
        ({"alp": 0.1}, {"alpha": 0.1}),
    ],
)
def test_aggte_parameter_passing(mp_result, params, expected):
    result = aggte(mp_result, type="simple", **params)

    for key, value in expected.items():
        assert result.estimation_params.get(key) == value


def test_aggte_clustering(mp_result):
    result = aggte(
        mp_result,
        type="simple",
        clustervars=["countyreal"],
        bstrap=True,
        biters=99,
    )

    assert result.aggregation_type == "simple"
    assert result.estimation_params.get("clustervars") == ["countyreal"]


@pytest.mark.parametrize("agg_type", ["simple", "dynamic", "group", "calendar"])
def test_aggte_result_attributes(mp_result, agg_type):
    result = aggte(mp_result, type=agg_type)

    required_attrs = [
        "overall_att",
        "overall_se",
        "aggregation_type",
        "event_times",
        "att_by_event",
        "se_by_event",
        "critical_values",
        "influence_func",
        "estimation_params",
        "call_info",
    ]

    for attr in required_attrs:
        assert hasattr(result, attr)

    assert "function" in result.call_info
    assert "aggte" in result.call_info["function"]


def test_aggte_influence_function(mp_result):
    result = aggte(mp_result, type="simple")

    assert result.influence_func is not None
    assert isinstance(result.influence_func, np.ndarray)
    assert len(result.influence_func) == mp_result.n_units


@pytest.mark.parametrize(
    "agg_type,expected_text",
    [
        ("simple", "Aggregate Treatment Effects"),
        ("dynamic", "Event Study"),
        ("group", "Group/Cohort"),
        ("calendar", "Calendar Time"),
    ],
)
def test_aggte_print_output(mp_result, agg_type, expected_text):
    result = aggte(mp_result, type=agg_type)
    output = str(result)
    assert "Aggregate Treatment Effects" in output
    assert expected_text in output
