import numpy as np
import pytest

pyspark = pytest.importorskip("pyspark")

from moderndid.core.data import gen_did_scalable
from moderndid.did.att_gt import att_gt
from moderndid.didtriple.ddd import ddd
from moderndid.didtriple.dgp import gen_dgp_mult_periods, gen_dgp_scalable
from moderndid.spark._ddd import spark_ddd
from moderndid.spark._did import spark_att_gt


@pytest.fixture(scope="module")
def did_rc_data():
    return gen_did_scalable(
        n=200,
        dgp_type=1,
        n_periods=4,
        n_cohorts=2,
        n_covariates=4,
        panel=False,
        random_state=42,
    )


@pytest.fixture(scope="module")
def ddd_rc_data():
    return gen_dgp_mult_periods(n=300, dgp_type=1, panel=False, random_state=42)


@pytest.fixture(scope="module")
def did_2period_rc_data():
    return gen_did_scalable(n=200, dgp_type=1, n_periods=2, n_cohorts=1, panel=False, random_state=42)


@pytest.fixture(scope="module")
def ddd_2period_rc_data():
    return gen_dgp_scalable(n=200, dgp_type=1, n_periods=2, n_cohorts=1, panel=False, random_state=42)


@pytest.mark.parametrize("est_method", ["reg", "dr"])
def test_did_rc_estimates_close_to_local(spark_session, did_rc_data, est_method):
    data = did_rc_data["data"]
    sdf = spark_session.createDataFrame(data.to_pandas())

    local_result = att_gt(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        xformla="~ cov1 + cov2 + cov3 + cov4",
        est_method=est_method,
        panel=False,
        random_state=42,
    )

    spark_result = spark_att_gt(
        data=sdf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        xformla="~ cov1 + cov2 + cov3 + cov4",
        est_method=est_method,
        panel=False,
        random_state=42,
        spark=spark_session,
    )

    local_order = np.lexsort((local_result.times, local_result.groups))
    spark_order = np.lexsort((spark_result.times, spark_result.groups))

    local_att = local_result.att_gt[local_order]
    spark_att = spark_result.att_gt[spark_order]
    local_se = local_result.se_gt[local_order]
    spark_se = spark_result.se_gt[spark_order]

    n_compare = min(len(local_att), len(spark_att))
    np.testing.assert_allclose(spark_att[:n_compare], local_att[:n_compare], atol=1e-10)

    finite = np.isfinite(local_se[:n_compare]) & np.isfinite(spark_se[:n_compare])
    np.testing.assert_allclose(spark_se[:n_compare][finite], local_se[:n_compare][finite], atol=1e-10)


def test_did_rc_result_structure(spark_session, did_rc_data):
    data = did_rc_data["data"]
    sdf = spark_session.createDataFrame(data.to_pandas())

    result = spark_att_gt(
        data=sdf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        xformla="~ cov1 + cov2 + cov3 + cov4",
        est_method="reg",
        panel=False,
        random_state=42,
        spark=spark_session,
    )

    assert len(result.att_gt) == len(result.se_gt)
    assert len(result.att_gt) == len(result.groups)
    assert len(result.att_gt) == len(result.times)
    assert result.n_units > 0
    assert np.any(np.isfinite(result.se_gt))
    assert result.influence_func.shape[1] == len(result.att_gt)


def test_did_rc_dispatch_via_att_gt(spark_session, did_rc_data):
    data = did_rc_data["data"]
    sdf = spark_session.createDataFrame(data.to_pandas())

    result = att_gt(
        data=sdf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        xformla="~ cov1 + cov2 + cov3 + cov4",
        est_method="reg",
        panel=False,
    )

    assert len(result.groups) > 0
    assert len(result.times) > 0


@pytest.mark.parametrize("est_method", ["reg", "dr"])
def test_ddd_rc_estimates_close_to_local(spark_session, ddd_rc_data, est_method):
    data = ddd_rc_data["data"]
    sdf = spark_session.createDataFrame(data.to_pandas())

    local_result = ddd(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method=est_method,
        panel=False,
        random_state=42,
    )

    spark_result = spark_ddd(
        data=sdf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method=est_method,
        panel=False,
        random_state=42,
        spark=spark_session,
    )

    local_order = np.lexsort((local_result.times, local_result.groups))
    spark_order = np.lexsort((spark_result.times, spark_result.groups))

    local_att = local_result.att[local_order]
    spark_att = spark_result.att[spark_order]
    local_se = local_result.se[local_order]
    spark_se = spark_result.se[spark_order]

    n_compare = min(len(local_att), len(spark_att))
    np.testing.assert_allclose(spark_att[:n_compare], local_att[:n_compare], atol=1e-10)

    finite = np.isfinite(local_se[:n_compare]) & np.isfinite(spark_se[:n_compare])
    np.testing.assert_allclose(spark_se[:n_compare][finite], local_se[:n_compare][finite], atol=1e-10)


def test_ddd_rc_result_structure(spark_session, ddd_rc_data):
    data = ddd_rc_data["data"]
    sdf = spark_session.createDataFrame(data.to_pandas())

    result = spark_ddd(
        data=sdf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method="reg",
        panel=False,
        random_state=42,
        spark=spark_session,
    )

    assert len(result.att) == len(result.se)
    assert len(result.att) == len(result.groups)
    assert len(result.att) == len(result.times)
    assert result.n > 0
    assert np.any(np.isfinite(result.se))
    assert result.inf_func_mat.shape[1] == len(result.att)


def test_ddd_rc_dispatch_via_ddd(spark_session, ddd_rc_data):
    data = ddd_rc_data["data"]
    sdf = spark_session.createDataFrame(data.to_pandas())

    result = ddd(
        data=sdf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method="reg",
        panel=False,
    )

    assert len(result.glist) > 0
    assert len(result.tlist) > 0


def test_did_2period_rc_estimates_close_to_local(spark_session, did_2period_rc_data):
    data = did_2period_rc_data["data"]
    sdf = spark_session.createDataFrame(data.to_pandas())

    local_result = att_gt(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        est_method="reg",
        panel=False,
        random_state=42,
    )

    spark_result = spark_att_gt(
        data=sdf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        est_method="reg",
        panel=False,
        random_state=42,
        spark=spark_session,
    )

    local_order = np.lexsort((local_result.times, local_result.groups))
    spark_order = np.lexsort((spark_result.times, spark_result.groups))

    local_att = local_result.att_gt[local_order]
    spark_att = spark_result.att_gt[spark_order]
    local_se = local_result.se_gt[local_order]
    spark_se = spark_result.se_gt[spark_order]

    n_compare = min(len(local_att), len(spark_att))
    np.testing.assert_allclose(spark_att[:n_compare], local_att[:n_compare], atol=1e-10)

    finite = np.isfinite(local_se[:n_compare]) & np.isfinite(spark_se[:n_compare])
    np.testing.assert_allclose(spark_se[:n_compare][finite], local_se[:n_compare][finite], atol=1e-10)


def test_did_2period_rc_result_structure(spark_session, did_2period_rc_data):
    data = did_2period_rc_data["data"]
    sdf = spark_session.createDataFrame(data.to_pandas())

    result = spark_att_gt(
        data=sdf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        est_method="reg",
        panel=False,
        random_state=42,
        spark=spark_session,
    )

    assert len(result.att_gt) == len(result.se_gt)
    assert len(result.att_gt) == len(result.groups)
    assert len(result.att_gt) == len(result.times)
    assert result.n_units > 0
    assert len(result.att_gt) >= 1


def test_ddd_2period_rc_estimates_close_to_local(spark_session, ddd_2period_rc_data):
    data = ddd_2period_rc_data["data"]
    sdf = spark_session.createDataFrame(data.to_pandas())

    local_result = ddd(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method="reg",
        panel=False,
        random_state=42,
    )

    spark_result = spark_ddd(
        data=sdf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method="reg",
        panel=False,
        random_state=42,
        spark=spark_session,
    )

    compute_mask = np.isfinite(spark_result.se)
    spark_att = spark_result.att[compute_mask]
    spark_se = spark_result.se[compute_mask]
    local_att = np.atleast_1d(local_result.att)
    local_se = np.atleast_1d(local_result.se)

    np.testing.assert_allclose(spark_att, local_att, atol=1e-10)
    np.testing.assert_allclose(spark_se, local_se, rtol=0.01)


def test_ddd_2period_rc_result_structure(spark_session, ddd_2period_rc_data):
    data = ddd_2period_rc_data["data"]
    sdf = spark_session.createDataFrame(data.to_pandas())

    result = spark_ddd(
        data=sdf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method="reg",
        panel=False,
        random_state=42,
        spark=spark_session,
    )

    assert len(result.att) == len(result.se)
    assert len(result.att) == len(result.groups)
    assert len(result.att) == len(result.times)
    assert result.n > 0
    assert len(result.att) >= 1
