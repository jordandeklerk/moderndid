import dask.dataframe as dd
import numpy as np
import pytest

distributed = pytest.importorskip("distributed")

from distributed import Client, LocalCluster

from moderndid.core.data import gen_did_scalable
from moderndid.dask._ddd import dask_ddd
from moderndid.dask._did import dask_att_gt
from moderndid.did.att_gt import att_gt
from moderndid.didtriple.ddd import ddd
from moderndid.didtriple.dgp import gen_dgp_mult_periods, gen_dgp_scalable


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


@pytest.fixture(scope="module")
def did_2period_panel_data():
    return gen_did_scalable(n=200, dgp_type=1, n_periods=2, n_cohorts=1, panel=True, random_state=42)


@pytest.fixture(scope="module")
def ddd_2period_panel_data():
    return gen_dgp_scalable(n=200, dgp_type=1, n_periods=2, n_cohorts=1, panel=True, random_state=42)


def test_ddd_mp_point_estimates_close_to_local(dask_client, mp_dgp_data):
    data = mp_dgp_data["data"]
    ddf = dd.from_pandas(data.to_pandas(), npartitions=4)

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
    local_se = local_result.se[local_order]
    dask_se = dask_result.se[dask_order]

    n_compare = min(len(local_att), len(dask_att))
    np.testing.assert_allclose(dask_att[:n_compare], local_att[:n_compare], atol=1e-10)

    finite = np.isfinite(local_se[:n_compare]) & np.isfinite(dask_se[:n_compare])
    np.testing.assert_allclose(dask_se[:n_compare][finite], local_se[:n_compare][finite], atol=1e-10)


def test_ddd_mp_result_structure(dask_client, mp_dgp_data):
    data = mp_dgp_data["data"]
    ddf = dd.from_pandas(data.to_pandas(), npartitions=4)

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

    assert len(result.att) == len(result.se)
    assert len(result.att) == len(result.groups)
    assert len(result.att) == len(result.times)
    assert result.n > 0
    assert np.any(np.isfinite(result.se))


def test_ddd_mp_dask_dispatch(dask_client, mp_dgp_data):
    data = mp_dgp_data["data"]
    ddf = dd.from_pandas(data.to_pandas(), npartitions=4)

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

    assert len(result.glist) > 0
    assert len(result.tlist) > 0


def test_did_2period_panel_estimates_close_to_local(dask_client, did_2period_panel_data):
    data = did_2period_panel_data["data"]
    ddf = dd.from_pandas(data.to_pandas(), npartitions=4)

    local_result = att_gt(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        est_method="reg",
        panel=True,
        random_state=42,
    )

    dask_result = dask_att_gt(
        data=ddf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        est_method="reg",
        panel=True,
        random_state=42,
        client=dask_client,
    )

    local_order = np.lexsort((local_result.times, local_result.groups))
    dask_order = np.lexsort((dask_result.times, dask_result.groups))

    local_att = local_result.att_gt[local_order]
    dask_att = dask_result.att_gt[dask_order]
    local_se = local_result.se_gt[local_order]
    dask_se = dask_result.se_gt[dask_order]

    n_compare = min(len(local_att), len(dask_att))
    np.testing.assert_allclose(dask_att[:n_compare], local_att[:n_compare], atol=1e-10)

    finite = np.isfinite(local_se[:n_compare]) & np.isfinite(dask_se[:n_compare])
    np.testing.assert_allclose(dask_se[:n_compare][finite], local_se[:n_compare][finite], atol=1e-10)


def test_did_2period_panel_result_structure(dask_client, did_2period_panel_data):
    data = did_2period_panel_data["data"]
    ddf = dd.from_pandas(data.to_pandas(), npartitions=4)

    result = dask_att_gt(
        data=ddf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        est_method="reg",
        panel=True,
        random_state=42,
        client=dask_client,
    )

    assert len(result.att_gt) == len(result.se_gt)
    assert len(result.att_gt) == len(result.groups)
    assert len(result.att_gt) == len(result.times)
    assert result.n_units > 0
    assert len(result.att_gt) >= 1


def test_ddd_2period_panel_estimates_close_to_local(dask_client, ddd_2period_panel_data):
    data = ddd_2period_panel_data["data"]
    ddf = dd.from_pandas(data.to_pandas(), npartitions=4)

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

    compute_mask = np.isfinite(dask_result.se)
    dask_att = dask_result.att[compute_mask]
    dask_se = dask_result.se[compute_mask]
    local_att = np.atleast_1d(local_result.att)
    local_se = np.atleast_1d(local_result.se)

    np.testing.assert_allclose(dask_att, local_att, atol=1e-10)
    np.testing.assert_allclose(dask_se, local_se, rtol=0.01)


def test_ddd_2period_panel_result_structure(dask_client, ddd_2period_panel_data):
    data = ddd_2period_panel_data["data"]
    ddf = dd.from_pandas(data.to_pandas(), npartitions=4)

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

    assert len(result.att) == len(result.se)
    assert len(result.att) == len(result.groups)
    assert len(result.att) == len(result.times)
    assert result.n > 0
    assert len(result.att) >= 1
