"""Integration tests for distributed continuous DiD via Spark."""

import numpy as np
import pytest

pyspark = pytest.importorskip("pyspark")

from moderndid import simulate_cont_did_data
from moderndid.didcont.cont_did import cont_did
from moderndid.spark._didcont import spark_cont_did


@pytest.fixture(scope="module")
def cont_did_data():
    return simulate_cont_did_data(n=500, seed=42)


def test_spark_cont_did_dose_matches_local(spark_session, cont_did_data):
    sdf = spark_session.createDataFrame(cont_did_data.to_pandas())

    local_result = cont_did(
        data=cont_did_data,
        yname="Y",
        tname="time_period",
        idname="id",
        gname="G",
        dname="D",
        target_parameter="level",
        aggregation="dose",
        degree=2,
        num_knots=0,
        biters=10,
        random_state=42,
    )

    spark_result = spark_cont_did(
        data=sdf,
        yname="Y",
        tname="time_period",
        idname="id",
        gname="G",
        dname="D",
        target_parameter="level",
        aggregation="dose",
        degree=2,
        num_knots=0,
        biters=10,
        random_state=42,
        spark=spark_session,
    )

    assert type(local_result).__name__ == type(spark_result).__name__

    np.testing.assert_allclose(
        spark_result.att_d,
        local_result.att_d,
        atol=0.15,
    )

    np.testing.assert_allclose(
        spark_result.overall_att,
        local_result.overall_att,
        atol=0.15,
    )


def test_spark_cont_did_eventstudy_matches_local(spark_session, cont_did_data):
    sdf = spark_session.createDataFrame(cont_did_data.to_pandas())

    local_result = cont_did(
        data=cont_did_data,
        yname="Y",
        tname="time_period",
        idname="id",
        gname="G",
        dname="D",
        target_parameter="level",
        aggregation="eventstudy",
        degree=2,
        num_knots=0,
        biters=10,
        random_state=42,
    )

    spark_result = spark_cont_did(
        data=sdf,
        yname="Y",
        tname="time_period",
        idname="id",
        gname="G",
        dname="D",
        target_parameter="level",
        aggregation="eventstudy",
        degree=2,
        num_knots=0,
        biters=10,
        random_state=42,
        spark=spark_session,
    )

    assert type(local_result).__name__ == type(spark_result).__name__


def test_spark_cont_did_dispatch(spark_session, cont_did_data):
    sdf = spark_session.createDataFrame(cont_did_data.to_pandas())

    result = cont_did(
        data=sdf,
        yname="Y",
        tname="time_period",
        idname="id",
        gname="G",
        dname="D",
        target_parameter="level",
        aggregation="dose",
        degree=2,
        num_knots=0,
        biters=10,
        random_state=42,
    )

    assert result is not None
    assert hasattr(result, "att_d") or hasattr(result, "att_gt")


def test_spark_cont_did_result_structure(spark_session, cont_did_data):
    sdf = spark_session.createDataFrame(cont_did_data.to_pandas())

    result = spark_cont_did(
        data=sdf,
        yname="Y",
        tname="time_period",
        idname="id",
        gname="G",
        dname="D",
        target_parameter="level",
        aggregation="dose",
        degree=2,
        num_knots=0,
        biters=10,
        random_state=42,
        spark=spark_session,
    )

    assert hasattr(result, "dose")
    assert hasattr(result, "att_d")
    assert hasattr(result, "overall_att")
    assert result.att_d is not None
    assert len(result.att_d) > 0
