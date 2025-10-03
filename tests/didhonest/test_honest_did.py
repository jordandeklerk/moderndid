# pylint: disable=redefined-outer-name
"""Tests for sensitivity analysis using the approach of Rambachan and Roth (2021)."""

import numpy as np
import pandas as pd
import pytest

from moderndid import AGGTEResult, HonestDiDResult, honest_did


@pytest.fixture
def sample_aggte_result():
    np.random.seed(42)

    event_times = np.array([-4, -3, -2, -1, 1, 2, 3])
    att_by_event = np.array([0.05, 0.1, -0.05, 0.02, 0.5, 0.3, 0.4])
    se_by_event = np.array([0.09, 0.1, 0.122, 0.141, 0.158, 0.173, 0.187])

    n_units = 120
    n_events = len(event_times)
    influence_func = np.random.normal(0, 0.1, (n_units, n_events))

    for i in range(1, n_events):
        influence_func[:, i] = 0.5 * influence_func[:, i - 1] + 0.5 * influence_func[:, i]

    return AGGTEResult(
        overall_att=0.25,
        overall_se=0.08,
        aggregation_type="dynamic",
        event_times=event_times,
        att_by_event=att_by_event,
        se_by_event=se_by_event,
        influence_func=influence_func,
        estimation_params={"alpha": 0.05, "control_group": "nevertreated"},
    )


@pytest.fixture
def sample_aggte_result_with_ref():
    np.random.seed(42)

    event_times = np.array([-4, -3, -2, -1, 0, 1, 2, 3])
    att_by_event = np.array([0.05, 0.1, -0.05, 0.02, 0.0, 0.5, 0.3, 0.4])
    se_by_event = np.array([0.09, 0.1, 0.122, 0.141, 0.0, 0.158, 0.173, 0.187])

    n_units = 120
    n_events = len(event_times)
    influence_func = np.random.normal(0, 0.1, (n_units, n_events))

    influence_func[:, 4] = 0

    return AGGTEResult(
        overall_att=0.25,
        overall_se=0.08,
        aggregation_type="dynamic",
        event_times=event_times,
        att_by_event=att_by_event,
        se_by_event=se_by_event,
        influence_func=influence_func,
        estimation_params={"alpha": 0.05},
    )


def test_honest_did_basic(sample_aggte_result):
    result = honest_did(sample_aggte_result, event_time=1)

    assert isinstance(result, HonestDiDResult)
    assert isinstance(result.robust_ci, pd.DataFrame)
    assert result.sensitivity_type == "smoothness"

    assert "lb" in result.robust_ci.columns
    assert "ub" in result.robust_ci.columns
    assert "method" in result.robust_ci.columns
    assert "delta" in result.robust_ci.columns
    assert "m" in result.robust_ci.columns

    assert result.original_ci.lb < result.original_ci.ub
    assert result.original_ci.method == "Original"


def test_honest_did_with_reference_period(sample_aggte_result_with_ref):
    result = honest_did(sample_aggte_result_with_ref, event_time=1)

    assert isinstance(result, HonestDiDResult)
    assert isinstance(result.robust_ci, pd.DataFrame)
    assert len(result.robust_ci) > 0


def test_honest_did_relative_magnitude(sample_aggte_result):
    result = honest_did(
        sample_aggte_result,
        event_time=2,
        sensitivity_type="relative_magnitude",
        m_bar_vec=np.array([0.0, 1.0]),
    )

    assert result.sensitivity_type == "relative_magnitude"
    assert len(result.robust_ci) == 2
    assert "Mbar" in result.robust_ci.columns
    assert list(result.robust_ci["Mbar"]) == [0.0, 1.0]


@pytest.mark.parametrize(
    "event_time,method",
    [
        (1, "FLCI"),
        (1, "Conditional"),
    ],
)
def test_honest_did_methods(sample_aggte_result, event_time, method):
    result = honest_did(
        sample_aggte_result,
        event_time=event_time,
        method=method,
        m_vec=np.array([0, 0.1]),
    )

    assert all(result.robust_ci["method"] == method)


@pytest.mark.xfail(reason="Numerical instability with synthetic test data")
def test_honest_did_with_shape_restrictions(sample_aggte_result):
    result_mono = honest_did(
        sample_aggte_result,
        event_time=1,
        monotonicity_direction="increasing",
        m_vec=np.array([0, 0.1]),
    )
    assert "SDI" in result_mono.robust_ci["delta"].iloc[0]

    result_bias = honest_did(
        sample_aggte_result,
        event_time=1,
        bias_direction="positive",
        m_vec=np.array([0, 0.1]),
    )
    assert "SDPB" in result_bias.robust_ci["delta"].iloc[0]


def test_honest_did_custom_alpha(sample_aggte_result):
    result = honest_did(
        sample_aggte_result,
        event_time=1,
        alpha=0.10,
    )

    result_05 = honest_did(
        sample_aggte_result,
        event_time=1,
        alpha=0.05,
    )

    ci_width_10 = result.original_ci.ub - result.original_ci.lb
    ci_width_05 = result_05.original_ci.ub - result_05.original_ci.lb
    assert ci_width_10 < ci_width_05


def test_honest_did_grid_points(sample_aggte_result):
    result = honest_did(
        sample_aggte_result,
        event_time=1,
        sensitivity_type="smoothness",
        method="Conditional",
        grid_points=12,
        m_vec=np.array([0.1]),
    )

    assert isinstance(result.robust_ci, pd.DataFrame)


def test_honest_did_errors():
    static_result = AGGTEResult(
        overall_att=0.25,
        overall_se=0.08,
        aggregation_type="simple",
    )

    with pytest.raises(ValueError, match="event study"):
        honest_did(static_result)

    no_inf_result = AGGTEResult(
        overall_att=0.25,
        overall_se=0.08,
        aggregation_type="dynamic",
        event_times=np.array([-1, 1, 2]),
        att_by_event=np.array([0.1, 0.5, 0.3]),
        influence_func=None,
    )

    with pytest.raises(ValueError, match="influence functions"):
        honest_did(no_inf_result)

    non_consec_result = AGGTEResult(
        overall_att=0.25,
        overall_se=0.08,
        aggregation_type="dynamic",
        event_times=np.array([-4, -2, -1, 1, 3]),
        att_by_event=np.array([0.05, -0.05, 0.02, 0.5, 0.4]),
        influence_func=np.random.normal(0, 0.1, (500, 5)),
    )

    with pytest.raises(ValueError, match="consecutive"):
        honest_did(non_consec_result)


def test_honest_did_invalid_event_time(sample_aggte_result):
    with pytest.raises(ValueError, match="Event time .* not found"):
        honest_did(sample_aggte_result, event_time=5)


def test_honest_did_invalid_sensitivity_type(sample_aggte_result):
    with pytest.raises(ValueError, match="sensitivity_type must be"):
        honest_did(sample_aggte_result, event_time=1, sensitivity_type="invalid")


def test_honest_did_invalid_object_type():
    with pytest.raises(TypeError, match="not implemented for object of type"):
        honest_did("not an event study object")


def test_honest_did_no_pre_periods():
    result = AGGTEResult(
        overall_att=0.25,
        overall_se=0.08,
        aggregation_type="dynamic",
        event_times=np.array([1, 2, 3]),
        att_by_event=np.array([0.5, 0.3, 0.4]),
        influence_func=np.random.normal(0, 0.1, (500, 3)),
    )

    with pytest.raises(ValueError, match="pre-treatment periods"):
        honest_did(result)


def test_honest_did_no_post_periods():
    result = AGGTEResult(
        overall_att=0.25,
        overall_se=0.08,
        aggregation_type="dynamic",
        event_times=np.array([-3, -2, -1]),
        att_by_event=np.array([0.1, -0.05, 0.02]),
        influence_func=np.random.normal(0, 0.1, (500, 3)),
    )

    with pytest.raises(ValueError, match="post-treatment periods"):
        honest_did(result)
