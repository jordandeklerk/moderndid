"""Tests for aggregate treatment effect (AGGTE) result objects."""

import re

import numpy as np
import pytest

from moderndid import AGGTEResult, format_aggte_result
from moderndid.did.container import aggte


def test_aggte_simple():
    result = aggte(
        overall_att=1.5,
        overall_se=0.3,
        aggregation_type="simple",
        estimation_params={"alpha": 0.05},
    )

    assert isinstance(result, AGGTEResult)
    assert result.overall_att == 1.5
    assert result.overall_se == 0.3
    assert result.aggregation_type == "simple"
    assert result.estimation_params["alpha"] == 0.05
    assert result.event_times is None
    assert result.att_by_event is None


def test_aggte_dynamic():
    event_times = np.array([-2, -1, 0, 1, 2, 3])
    att_by_event = np.array([0.1, 0.2, 1.5, 1.8, 2.0, 1.9])
    se_by_event = np.array([0.15, 0.16, 0.25, 0.3, 0.35, 0.4])
    critical_values = np.array([2.2, 2.3, 2.4, 2.5, 2.6, 2.7])

    result = aggte(
        overall_att=1.6,
        overall_se=0.25,
        aggregation_type="dynamic",
        event_times=event_times,
        att_by_event=att_by_event,
        se_by_event=se_by_event,
        critical_values=critical_values,
        min_event_time=-2,
        max_event_time=3,
        balanced_event_threshold=2,
    )

    assert result.aggregation_type == "dynamic"
    np.testing.assert_array_equal(result.event_times, event_times)
    np.testing.assert_array_equal(result.att_by_event, att_by_event)
    np.testing.assert_array_equal(result.se_by_event, se_by_event)
    np.testing.assert_array_equal(result.critical_values, critical_values)
    assert (result.min_event_time, result.max_event_time, result.balanced_event_threshold) == (-2, 3, 2)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {"aggregation_type": "invalid"},
            "Invalid aggregation_type: invalid. Must be one of 'simple', 'dynamic', 'group', 'calendar'.",
        ),
        ({"att_by_event": np.array([1.0, 1.5])}, "att_by_event must have same length as event_times."),
        ({"se_by_event": np.array([0.2, 0.3])}, "se_by_event must have same length as event_times."),
        ({"critical_values": np.array([2.0, 2.0])}, "critical_values must have same length as event_times."),
    ],
)
def test_aggte_validation(arguments, message):
    spec = {"overall_att": 1.25, "overall_se": 0.3, "aggregation_type": "dynamic", "event_times": np.array([0, 1, 2])}

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        aggte(**{**spec, **arguments})


def test_aggte_with_estimation_params():
    estimation_params = {
        "alpha": 0.05,
        "bootstrap": True,
        "uniform_bands": False,
        "control_group": "nevertreated",
        "anticipation_periods": 0,
        "estimation_method": "dr",
    }

    call_info = {
        "function": 'aggte(MP_obj, type="simple")',
    }

    result = aggte(
        overall_att=1.8,
        overall_se=0.4,
        aggregation_type="simple",
        estimation_params=estimation_params,
        call_info=call_info,
    )

    assert result.estimation_params["bootstrap"] is True
    assert result.estimation_params["control_group"] == "nevertreated"
    assert result.call_info["function"] == 'aggte(MP_obj, type="simple")'


def test_aggte_formatting_simple():
    result = aggte(
        overall_att=1.234,
        overall_se=0.456,
        aggregation_type="simple",
        estimation_params={
            "alpha": 0.05,
            "control_group": "notyettreated",
            "anticipation_periods": 2,
            "estimation_method": "ipw",
        },
    )

    lines = format_aggte_result(result).splitlines()
    rows = [line.replace(" ", "").split("│")[1:-1] for line in lines if line.startswith("│")]

    assert " Aggregate Treatment Effects" in lines
    assert " Overall ATT:" in lines
    assert rows == [["ATT", "Std.Error", "[95%Conf.Interval]"], ["1.2340", "0.4560", "[0.3403,2.1277]*"]]
    assert " Control Group: Not Yet Treated" in lines
    assert " Anticipation Periods: 2" in lines
    assert " Estimation Method: Inverse Probability Weighting" in lines


def test_aggte_formatting_dynamic():
    result = aggte(
        overall_att=1.5,
        overall_se=0.2,
        aggregation_type="dynamic",
        event_times=np.array([-1, 0, 1, 2]),
        att_by_event=np.array([0.2, 1.5, 1.8, 1.6]),
        se_by_event=np.array([0.3, 0.25, 0.28, 0.35]),
        critical_values=np.full(4, 2.5),
        estimation_params={
            "alpha": 0.10,
            "bootstrap": True,
            "uniform_bands": True,
            "estimation_method": "dr",
        },
    )

    lines = format_aggte_result(result).splitlines()
    rows = [line.replace(" ", "").split("│")[1:-1] for line in lines if line.startswith("│")]

    assert " Aggregate Treatment Effects (Event Study)" in lines
    assert " Overall summary of ATT's based on event-study/dynamic aggregation:" in lines
    assert " Dynamic Effects:" in lines
    assert rows == [
        ["ATT", "Std.Error", "[90%Conf.Interval]"],
        ["1.5000", "0.2000", "[1.1710,1.8290]*"],
        ["Eventtime", "Estimate", "Std.Error", "[90%Simult.Conf.Band]"],
        ["-1", "0.2000", "0.3000", "[-0.5500,0.9500]"],
        ["0", "1.5000", "0.2500", "[0.8750,2.1250]*"],
        ["1", "1.8000", "0.2800", "[1.1000,2.5000]*"],
        ["2", "1.6000", "0.3500", "[0.7250,2.4750]*"],
    ]


def test_aggte_formatting_group():
    groups = np.array([2000, 2001, 2002])
    att_by_group = np.array([1.2, 1.5, 1.8])
    se_by_group = np.array([0.3, 0.35, 0.4])

    result = aggte(
        overall_att=1.5,
        overall_se=0.25,
        aggregation_type="group",
        event_times=groups,
        att_by_event=att_by_group,
        se_by_event=se_by_group,
        estimation_params={
            "control_group": "nevertreated",
        },
    )

    lines = format_aggte_result(result).splitlines()
    rows = [line.replace(" ", "").split("│")[1:-1] for line in lines if line.startswith("│")]

    assert " Aggregate Treatment Effects (Group/Cohort)" in lines
    assert " Overall summary of ATT's based on group/cohort aggregation:" in lines
    assert " Group Effects:" in lines
    assert rows == [
        ["ATT", "Std.Error", "[95%Conf.Interval]"],
        ["1.5000", "0.2500", "[1.0100,1.9900]*"],
        ["Group", "Estimate", "Std.Error", "[95%PointwiseConf.Band]"],
        ["2000", "1.2000", "0.3000", "[0.6120,1.7880]*"],
        ["2001", "1.5000", "0.3500", "[0.8140,2.1860]*"],
        ["2002", "1.8000", "0.4000", "[1.0160,2.5840]*"],
    ]
    assert " Control Group: Never Treated" in lines


@pytest.mark.parametrize(
    ("estimation_params", "line"),
    [
        ({"bootstrap": False}, " Analytical standard errors"),
        ({"bootstrap": True}, " Bootstrap standard errors"),
        (
            {"bootstrap": True, "clustervars": ["state"], "cluster": np.repeat(np.arange(50), 2)},
            " Bootstrap standard errors clustered by state",
        ),
        (
            {"bootstrap": False, "clustervars": ["state"], "cluster": np.repeat(np.arange(50), 2)},
            " Analytical standard errors clustered by state",
        ),
        ({"bootstrap": True, "clustervars": ["copy"], "cluster": np.arange(100)}, " Bootstrap standard errors"),
    ],
)
def test_aggte_formatting_names_the_clusters_of_the_standard_errors(estimation_params, line):
    result = aggte(
        overall_att=1.5,
        overall_se=0.2,
        aggregation_type="simple",
        estimation_params={"alpha": 0.05, **estimation_params},
    )

    lines = format_aggte_result(result).splitlines()

    assert line in lines
    assert lines[lines.index(line) - 1] == " Significance level: 0.05"


def test_aggte_display_methods():
    result = aggte(
        overall_att=2.0,
        overall_se=0.5,
        aggregation_type="calendar",
        event_times=np.array([2020, 2021, 2022]),
        att_by_event=np.array([1.8, 2.0, 2.2]),
        se_by_event=np.array([0.4, 0.5, 0.6]),
    )

    report = format_aggte_result(result)
    lines = report.splitlines()
    rows = [line.replace(" ", "").split("│")[1:-1] for line in lines if line.startswith("│")]

    assert str(result) == report
    assert repr(result) == report
    assert " Aggregate Treatment Effects (Calendar Time)" in lines
    assert " Overall summary of ATT's based on calendar time aggregation:" in lines
    assert " Time Effects:" in lines
    assert rows == [
        ["ATT", "Std.Error", "[95%Conf.Interval]"],
        ["2.0000", "0.5000", "[1.0200,2.9800]*"],
        ["Time", "Estimate", "Std.Error", "[95%PointwiseConf.Band]"],
        ["2020", "1.8000", "0.4000", "[1.0160,2.5840]*"],
        ["2021", "2.0000", "0.5000", "[1.0200,2.9800]*"],
        ["2022", "2.2000", "0.6000", "[1.0240,3.3760]*"],
    ]
