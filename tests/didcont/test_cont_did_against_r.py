"""Tests against R examples."""

import numpy as np

from moderndid.didcont.cont_did import cont_did
from tests.helpers import importorskip

pl = importorskip("polars")


def load_r_data():
    df = pl.read_csv("tests/didcont/data/cont_test_data.csv.gz")
    df = df.with_columns(pl.when(pl.col("G") == 0).then(0).otherwise(pl.col("D")).alias("D"))
    return df


def load_cck_data():
    df = pl.read_csv("tests/didcont/data/cont_test_data_cck.csv.gz")
    df = df.with_columns(pl.when(pl.col("G") == 0).then(0).otherwise(pl.col("D")).alias("D"))
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

    assert np.allclose(result.overall_att, -0.0265, atol=0.001)
    assert np.allclose(result.overall_acrt, 0.1341, atol=0.01)


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
    assert np.allclose(py_att, r_att, atol=0.1)

    for e, att in zip(result.event_study.event_times, result.event_study.att_by_event):
        r_val = r_dynamic.get(e)
        if r_val is not None:
            assert np.allclose(att, r_val, atol=0.1)


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

    # NOTE: R reports 0.1341 but that's using "group" aggregation. Is that right?
    # For event study, the correct value is the average of post-treatment effects
    r_acrt = -0.1087  # Average of [0.1592, 0.0551, -0.5405]
    r_dynamic = {
        -2: -0.0701,
        -1: -0.2212,
        0: 0.1592,
        1: 0.0551,
        2: -0.5405,
    }

    py_acrt = result.overall_att.overall_att
    assert np.allclose(py_acrt, r_acrt, atol=0.001)

    for e, att in zip(result.event_study.event_times, result.event_study.att_by_event):
        r_val = r_dynamic.get(e)
        if r_val is not None:
            assert np.allclose(att, r_val, atol=0.01)


def test_cck_estimator_with_r_data():
    df = load_cck_data()

    result = cont_did(
        yname="Y",
        tname="time_period",
        idname="id",
        dname="D",
        data=df,
        gname="G",
        target_parameter="level",
        aggregation="dose",
        treatment_type="continuous",
        dose_est_method="cck",
        control_group="notyettreated",
        biters=100,
        cband=True,
    )

    r_att = 0.3399
    r_att_se = 0.037
    r_acrt = 0.6595
    r_acrt_se = 0.1853

    assert np.allclose(result.overall_att, r_att, atol=0.01)
    assert np.allclose(result.overall_att_se, r_att_se, atol=0.1)

    assert np.allclose(result.overall_acrt, r_acrt, atol=0.1)
    assert np.allclose(result.overall_acrt_se, r_acrt_se, atol=0.1)
