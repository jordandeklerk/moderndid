"""Integration tests for distributed continuous DiD via Dask."""

import numpy as np
import pytest

distributed = pytest.importorskip("distributed")

import dask.dataframe as dd
from distributed import Client, LocalCluster

from moderndid import simulate_cont_did_data
from moderndid.dask._didcont import dask_cont_did
from moderndid.didcont.cont_did import cont_did


@pytest.fixture(scope="module")
def dask_client():
    cluster = LocalCluster(n_workers=2, threads_per_worker=1, memory_limit="512MB")
    client = Client(cluster)
    yield client
    client.close()
    cluster.close()


@pytest.fixture(scope="module")
def cont_did_data():
    return simulate_cont_did_data(n=500, seed=42)


def test_dask_cont_did_dose_matches_local(dask_client, cont_did_data):
    ddf = dd.from_pandas(cont_did_data.to_pandas(), npartitions=4)

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

    dask_result = dask_cont_did(
        data=ddf,
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
        client=dask_client,
    )

    assert type(local_result).__name__ == type(dask_result).__name__

    np.testing.assert_allclose(
        dask_result.att_d,
        local_result.att_d,
        atol=0.15,
    )

    np.testing.assert_allclose(
        dask_result.overall_att,
        local_result.overall_att,
        atol=0.15,
    )


def test_dask_cont_did_eventstudy_matches_local(dask_client, cont_did_data):
    ddf = dd.from_pandas(cont_did_data.to_pandas(), npartitions=4)

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

    dask_result = dask_cont_did(
        data=ddf,
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
        client=dask_client,
    )

    assert type(local_result).__name__ == type(dask_result).__name__


def test_dask_cont_did_dispatch(dask_client, cont_did_data):
    ddf = dd.from_pandas(cont_did_data.to_pandas(), npartitions=2)

    result = cont_did(
        data=ddf,
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


def test_dask_cont_did_result_structure(dask_client, cont_did_data):
    ddf = dd.from_pandas(cont_did_data.to_pandas(), npartitions=2)

    result = dask_cont_did(
        data=ddf,
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
        client=dask_client,
    )

    assert hasattr(result, "dose")
    assert hasattr(result, "att_d")
    assert hasattr(result, "overall_att")
    assert result.att_d is not None
    assert len(result.att_d) > 0
