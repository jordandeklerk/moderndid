import numpy as np
import pytest

from moderndid.didtriple.agg_ddd_obj import DDDAggResult
from moderndid.didtriple.format import format_ddd_agg_result


@pytest.mark.parametrize(
    "expected_text",
    [
        "Aggregate DDD Treatment Effects",
        "Overall ATT:",
        "ATT",
        "Std. Error",
        "Conf. Interval",
    ],
)
def test_format_ddd_agg_simple_contains_text(ddd_agg_result_simple, expected_text):
    output = format_ddd_agg_result(ddd_agg_result_simple)
    assert expected_text in output


@pytest.mark.parametrize(
    "expected_text",
    [
        "Event Study",
        "event-study aggregation",
        "Dynamic Effects",
        "Event time",
    ],
)
def test_format_ddd_agg_eventstudy_contains_text(ddd_agg_result_eventstudy, expected_text):
    output = format_ddd_agg_result(ddd_agg_result_eventstudy)
    assert expected_text in output


def test_format_ddd_agg_group_aggregation():
    result = DDDAggResult(
        overall_att=2.0,
        overall_se=0.3,
        aggregation_type="group",
        egt=np.array([3, 4]),
        att_egt=np.array([1.8, 2.2]),
        se_egt=np.array([0.3, 0.35]),
        crit_val=1.96,
        args={"alpha": 0.05, "boot": False, "cband": False},
    )
    output = format_ddd_agg_result(result)
    assert "Group/Cohort" in output
    assert "group/cohort aggregation" in output
    assert "Group Effects" in output


def test_format_ddd_agg_calendar_aggregation():
    result = DDDAggResult(
        overall_att=2.0,
        overall_se=0.3,
        aggregation_type="calendar",
        egt=np.array([2020, 2021, 2022]),
        att_egt=np.array([1.5, 2.0, 2.5]),
        se_egt=np.array([0.25, 0.3, 0.35]),
        crit_val=1.96,
        args={"alpha": 0.05, "boot": False, "cband": False},
    )
    output = format_ddd_agg_result(result)
    assert "Calendar Time" in output
    assert "calendar time aggregation" in output
    assert "Time Effects" in output


def test_format_ddd_agg_simultaneous_conf_band():
    result = DDDAggResult(
        overall_att=2.0,
        overall_se=0.3,
        aggregation_type="eventstudy",
        egt=np.array([-1, 0, 1]),
        att_egt=np.array([0.1, 1.8, 2.0]),
        se_egt=np.array([0.2, 0.3, 0.35]),
        crit_val=2.5,
        args={"alpha": 0.05, "boot": True, "cband": True},
    )
    output = format_ddd_agg_result(result)
    assert "Simult. Conf. Band" in output


def test_format_ddd_agg_pointwise_conf_band(ddd_agg_result_eventstudy):
    output = format_ddd_agg_result(ddd_agg_result_eventstudy)
    assert "Pointwise Conf. Band" in output


def test_format_ddd_agg_significance_marker(ddd_agg_result_eventstudy):
    output = format_ddd_agg_result(ddd_agg_result_eventstudy)
    assert "*" in output


@pytest.mark.parametrize("method", [repr, str])
def test_format_ddd_agg_repr_str(ddd_agg_result_simple, method):
    output = method(ddd_agg_result_simple)
    assert "Aggregate DDD Treatment Effects" in output


def test_ddd_agg_result_namedtuple_fields():
    result = DDDAggResult(
        overall_att=2.0,
        overall_se=0.3,
        aggregation_type="simple",
    )
    assert result.overall_att == 2.0
    assert result.overall_se == 0.3
    assert result.aggregation_type == "simple"
    assert result.egt is None
    assert result.att_egt is None
    assert result.se_egt is None
    assert result.crit_val == 1.96
    assert result.inf_func is None
    assert result.inf_func_overall is None
    assert result.args == {}
