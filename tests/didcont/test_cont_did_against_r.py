"""Tests against R examples."""

import numpy as np
import pandas as pd
import pytest

from moderndid.didcont.cont_did import cont_did


def load_r_data():
    df = pd.read_csv("tests/didcont/cont_test_data.csv")
    df.loc[df["G"] == 0, "D"] = 0
    return df


def test_slope_dose_with_r_data():
    df = load_r_data()

    result = cont_did(
        yname="Y",
        tname="time_period",
        idname="id",
        dname="D",
        data=df,
        gname="G",
        target_parameter="slope",
        aggregation="dose",
        treatment_type="continuous",
        control_group="notyettreated",
        biters=100,
        cband=True,
        num_knots=1,
        degree=3,
    )

    att_diff = abs(result.overall_att - (-0.0265))
    acrt_diff = abs(result.overall_acrt - 0.1341)

    assert np.isclose(att_diff, 0, atol=0.001)
    assert np.isclose(acrt_diff, 0, atol=0.01)


def test_level_eventstudy_with_r_data():
    df = load_r_data()

    result = cont_did(
        yname="Y",
        tname="time_period",
        idname="id",
        dname="D",
        data=df,
        gname="G",
        target_parameter="level",
        aggregation="eventstudy",
        treatment_type="continuous",
        control_group="notyettreated",
        biters=100,
        cband=True,
        num_knots=1,
        degree=3,
    )

    r_att = -0.0346
    r_dynamic = {
        -2: -0.0222,
        -1: 0.0116,
        0: -0.0039,
        1: -0.0160,
        2: -0.0839,
    }

    py_att = result.overall_att.overall_att
    att_diff = abs(py_att - r_att)
    assert att_diff < 0.1, f"Overall ATT diff {att_diff:.4f} exceeds tolerance"

    for e, att in zip(result.event_study.event_times, result.event_study.att_by_event):
        r_val = r_dynamic.get(e)
        if r_val is not None:
            dynamic_diff = abs(att - r_val)
            assert dynamic_diff < 0.001


@pytest.mark.skip(
    reason="Need to fix the panel processing to pass spline basis and knots to the cont_did_acrt function"
)
def test_slope_eventstudy_with_r_data():
    df = load_r_data()

    result = cont_did(
        yname="Y",
        tname="time_period",
        idname="id",
        dname="D",
        data=df,
        gname="G",
        target_parameter="slope",
        aggregation="eventstudy",
        treatment_type="continuous",
        control_group="notyettreated",
        biters=100,
        cband=True,
        num_knots=1,
        degree=3,
    )

    r_acrt = 0.1341
    r_dynamic = {
        -2: -0.0701,
        -1: -0.2212,
        0: 0.1592,
        1: 0.0551,
        2: -0.5405,
    }

    py_acrt = result.overall_att.overall_att
    acrt_diff = abs(py_acrt - r_acrt)
    assert acrt_diff < 0.2, f"Overall ACRT diff {acrt_diff:.4f} exceeds tolerance"

    for e, att in zip(result.event_study.event_times, result.event_study.att_by_event):
        r_val = r_dynamic.get(e)
        if r_val is not None:
            dynamic_diff = abs(att - r_val)
            assert dynamic_diff < 0.2, f"Dynamic effect for event time {e} diff {dynamic_diff:.4f} exceeds tolerance"
