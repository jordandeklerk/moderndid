"""Integration tests for distributed intertemporal DiD via Spark."""

import numpy as np
import polars as pl
import pytest

pyspark = pytest.importorskip("pyspark")

from moderndid.didinter.did_multiplegt import did_multiplegt
from moderndid.spark._didinter import spark_did_multiplegt


@pytest.fixture(scope="module")
def simple_panel_data():
    rng = np.random.default_rng(42)
    n_units = 50
    n_periods = 6

    units = np.repeat(np.arange(n_units), n_periods)
    periods = np.tile(np.arange(1, n_periods + 1), n_units)

    treatment = np.zeros(len(units))
    for unit in range(n_units):
        unit_mask = units == unit
        if unit < 20:
            treatment[unit_mask & (periods >= 3)] = 1
        elif unit < 30:
            treatment[unit_mask & (periods >= 4)] = 1

    y = rng.standard_normal(len(units))
    y += 2.0 * treatment

    return pl.DataFrame(
        {
            "id": units,
            "time": periods,
            "y": y,
            "d": treatment,
        }
    )


def test_spark_didinter_matches_local(spark_session, simple_panel_data):
    sdf = spark_session.createDataFrame(simple_panel_data.to_pandas())

    local_result = did_multiplegt(
        data=simple_panel_data,
        yname="y",
        tname="time",
        idname="id",
        dname="d",
        effects=2,
        random_state=42,
    )

    spark_result = spark_did_multiplegt(
        data=sdf,
        yname="y",
        tname="time",
        idname="id",
        dname="d",
        effects=2,
        random_state=42,
        spark=spark_session,
    )

    assert type(local_result).__name__ == type(spark_result).__name__

    np.testing.assert_allclose(
        spark_result.effects.estimates,
        local_result.effects.estimates,
        atol=0.15,
    )

    finite = np.isfinite(local_result.effects.std_errors) & np.isfinite(spark_result.effects.std_errors)
    if np.any(finite):
        np.testing.assert_allclose(
            spark_result.effects.std_errors[finite],
            local_result.effects.std_errors[finite],
            atol=0.15,
        )


def test_spark_didinter_dispatch(spark_session, simple_panel_data):
    sdf = spark_session.createDataFrame(simple_panel_data.to_pandas())

    result = did_multiplegt(
        data=sdf,
        yname="y",
        tname="time",
        idname="id",
        dname="d",
        effects=1,
    )

    assert result is not None
    assert hasattr(result, "effects")


def test_spark_didinter_result_structure(spark_session, simple_panel_data):
    sdf = spark_session.createDataFrame(simple_panel_data.to_pandas())

    result = spark_did_multiplegt(
        data=sdf,
        yname="y",
        tname="time",
        idname="id",
        dname="d",
        effects=2,
        random_state=42,
        spark=spark_session,
    )

    assert hasattr(result, "effects")
    assert hasattr(result, "n_units")
    assert hasattr(result, "n_switchers")
    assert hasattr(result, "ci_level")
    assert result.effects.estimates is not None
    assert len(result.effects.estimates) > 0
    assert len(result.effects.estimates) == len(result.effects.std_errors)
