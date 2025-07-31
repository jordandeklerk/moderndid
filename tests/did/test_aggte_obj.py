"""Tests for aggregate treatment effect (AGGTE) result objects."""

import numpy as np
import pytest

from doublediff import AGGTEResult, format_aggte_result
from doublediff.did.aggte_obj import aggte


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
    critical_values = np.array([2.2, 2.2, 2.2, 2.2, 2.2, 2.2])

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
    assert len(result.event_times) == 6
    assert len(result.att_by_event) == 6
    assert len(result.se_by_event) == 6
    assert len(result.critical_values) == 6
    assert result.min_event_time == -2
    assert result.max_event_time == 3


def test_aggte_validation():
    with pytest.raises(ValueError, match="Invalid aggregation_type"):
        aggte(
            overall_att=1.0,
            overall_se=0.2,
            aggregation_type="invalid",
        )

    event_times = np.array([0, 1, 2])
    att_by_event = np.array([1.0, 1.5])

    with pytest.raises(ValueError, match="att_by_event must have same length"):
        aggte(
            overall_att=1.25,
            overall_se=0.3,
            aggregation_type="dynamic",
            event_times=event_times,
            att_by_event=att_by_event,
        )


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

    formatted = format_aggte_result(result)

    assert "Aggregate Treatment Effects" in formatted
    assert "Overall ATT:" in formatted
    assert "1.2340" in formatted
    assert "0.4560" in formatted
    assert "Control Group: Not Yet Treated" in formatted
    assert "Anticipation Periods: 2" in formatted
    assert "Estimation Method: Inverse Probability Weighting" in formatted


def test_aggte_formatting_dynamic():
    event_times = np.array([-1, 0, 1, 2])
    att_by_event = np.array([0.2, 1.5, 1.8, 1.6])
    se_by_event = np.array([0.3, 0.25, 0.28, 0.35])

    result = aggte(
        overall_att=1.5,
        overall_se=0.2,
        aggregation_type="dynamic",
        event_times=event_times,
        att_by_event=att_by_event,
        se_by_event=se_by_event,
        estimation_params={
            "alpha": 0.10,
            "bootstrap": True,
            "uniform_bands": True,
            "estimation_method": "dr",
        },
    )

    formatted = format_aggte_result(result)

    assert "Aggregate Treatment Effects (Event Study)" in formatted
    assert "Overall summary of ATT's based on event-study/dynamic aggregation:" in formatted
    assert "Dynamic Effects:" in formatted
    assert "Event time" in formatted
    assert "[90% Simult. Conf. Band]" in formatted
    assert all(str(e) in formatted for e in event_times)


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

    formatted = format_aggte_result(result)

    assert "Aggregate Treatment Effects (Group/Cohort)" in formatted
    assert "Group Effects:" in formatted
    assert "Group" in formatted


def test_aggte_display_methods():
    result = aggte(
        overall_att=2.0,
        overall_se=0.5,
        aggregation_type="calendar",
        event_times=np.array([2020, 2021, 2022]),
        att_by_event=np.array([1.8, 2.0, 2.2]),
        se_by_event=np.array([0.4, 0.5, 0.6]),
    )

    str_output = str(result)
    repr_output = repr(result)

    assert isinstance(str_output, str)
    assert isinstance(repr_output, str)
    assert "Aggregate Treatment Effects (Calendar Time)" in str_output
    assert "Time Effects:" in str_output
