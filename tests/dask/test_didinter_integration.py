"""Integration tests for distributed intertemporal DiD via Dask."""

import numpy as np
import polars as pl
import pytest

distributed = pytest.importorskip("distributed")

import dask.dataframe as dd

from moderndid.dask._didinter import dask_did_multiplegt
from moderndid.didinter.did_multiplegt import did_multiplegt


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


def test_dask_didinter_matches_local(dask_client, simple_panel_data):
    ddf = dd.from_pandas(simple_panel_data.to_pandas(), npartitions=4)

    local_result = did_multiplegt(
        data=simple_panel_data,
        yname="y",
        tname="time",
        idname="id",
        dname="d",
        effects=2,
        random_state=42,
    )

    dask_result = dask_did_multiplegt(
        data=ddf,
        yname="y",
        tname="time",
        idname="id",
        dname="d",
        effects=2,
        random_state=42,
        client=dask_client,
    )

    assert type(local_result).__name__ == type(dask_result).__name__

    np.testing.assert_allclose(
        dask_result.effects.estimates,
        local_result.effects.estimates,
        atol=0.15,
    )

    finite = np.isfinite(local_result.effects.std_errors) & np.isfinite(dask_result.effects.std_errors)
    if np.any(finite):
        np.testing.assert_allclose(
            dask_result.effects.std_errors[finite],
            local_result.effects.std_errors[finite],
            atol=0.15,
        )


def test_dask_didinter_dispatch(dask_client, simple_panel_data):
    ddf = dd.from_pandas(simple_panel_data.to_pandas(), npartitions=2)

    with dask_client.as_current():
        result = did_multiplegt(
            data=ddf,
            yname="y",
            tname="time",
            idname="id",
            dname="d",
            effects=1,
        )

    assert result is not None
    assert hasattr(result, "effects")


def test_dask_didinter_result_structure(dask_client, simple_panel_data):
    ddf = dd.from_pandas(simple_panel_data.to_pandas(), npartitions=2)

    result = dask_did_multiplegt(
        data=ddf,
        yname="y",
        tname="time",
        idname="id",
        dname="d",
        effects=2,
        random_state=42,
        client=dask_client,
    )

    assert hasattr(result, "effects")
    assert hasattr(result, "n_units")
    assert hasattr(result, "n_switchers")
    assert hasattr(result, "ci_level")
    assert result.effects.estimates is not None
    assert len(result.effects.estimates) > 0
    assert len(result.effects.estimates) == len(result.effects.std_errors)
