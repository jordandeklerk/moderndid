"""End-to-end integration tests for the Dask DDD backend."""

import numpy as np
import pytest

distributed = pytest.importorskip("distributed")

from distributed import Client, LocalCluster

from moderndid.dask._ddd import dask_ddd
from moderndid.didtriple.ddd import ddd
from moderndid.didtriple.dgp import gen_dgp_mult_periods


@pytest.fixture(scope="module")
def dask_client():
    cluster = LocalCluster(n_workers=2, threads_per_worker=1, memory_limit="512MB")
    client = Client(cluster)
    yield client
    client.close()
    cluster.close()


@pytest.fixture(scope="module")
def mp_dgp_data():
    return gen_dgp_mult_periods(n=300, dgp_type=1, random_state=42)


def test_point_estimates_close_to_local(dask_client, mp_dgp_data):
    import dask.dataframe as dd

    data = mp_dgp_data["data"]
    data_pd = data.to_pandas()
    ddf = dd.from_pandas(data_pd, npartitions=4)

    local_result = ddd(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method="reg",
        random_state=42,
    )

    dask_result = dask_ddd(
        data=ddf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method="reg",
        random_state=42,
        client=dask_client,
    )

    local_order = np.lexsort((local_result.times, local_result.groups))
    dask_order = np.lexsort((dask_result.times, dask_result.groups))

    local_att = local_result.att[local_order]
    dask_att = dask_result.att[dask_order]

    n_compare = min(len(local_att), len(dask_att))
    np.testing.assert_allclose(dask_att[:n_compare], local_att[:n_compare], atol=0.5)


def test_result_structure(dask_client, mp_dgp_data):
    import dask.dataframe as dd

    data = mp_dgp_data["data"]
    data_pd = data.to_pandas()
    ddf = dd.from_pandas(data_pd, npartitions=4)

    result = dask_ddd(
        data=ddf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method="reg",
        random_state=42,
        client=dask_client,
    )

    assert hasattr(result, "att")
    assert hasattr(result, "se")
    assert hasattr(result, "groups")
    assert hasattr(result, "times")
    assert hasattr(result, "inf_func_mat")

    assert len(result.att) == len(result.se)
    assert len(result.att) == len(result.groups)
    assert len(result.att) == len(result.times)
    assert result.n > 0
    assert np.any(np.isfinite(result.se))


def test_dask_dispatch_via_ddd(dask_client, mp_dgp_data):
    import dask.dataframe as dd

    data = mp_dgp_data["data"]
    data_pd = data.to_pandas()
    ddf = dd.from_pandas(data_pd, npartitions=4)

    with dask_client.as_current():
        result = ddd(
            data=ddf,
            yname="y",
            tname="time",
            idname="id",
            gname="group",
            pname="partition",
            est_method="reg",
        )

    assert hasattr(result, "glist")
    assert hasattr(result, "tlist")
    assert len(result.glist) > 0
