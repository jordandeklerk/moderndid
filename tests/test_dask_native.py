"""Tests for Dask DataFrame distributed backend."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import requires_distributed

distributed = pytest.importorskip("distributed")
dd = pytest.importorskip("dask.dataframe")
Client = distributed.Client


def _polars_to_dask(pl_df, npartitions=4):
    pdf = pl_df.to_pandas()
    return dd.from_pandas(pdf, npartitions=npartitions)


@requires_distributed
def test_is_dask_dataframe_true():
    import polars as pl

    from moderndid.dask import is_dask_dataframe

    df = pl.DataFrame({"a": [1, 2, 3]})
    ddf = _polars_to_dask(df)
    assert is_dask_dataframe(ddf)


@requires_distributed
def test_is_dask_dataframe_false_for_polars():
    import polars as pl

    from moderndid.dask import is_dask_dataframe

    df = pl.DataFrame({"a": [1, 2, 3]})
    assert not is_dask_dataframe(df)


@requires_distributed
def test_is_dask_dataframe_false_for_pandas():
    import pandas as pd

    from moderndid.dask import is_dask_dataframe

    df = pd.DataFrame({"a": [1, 2, 3]})
    assert not is_dask_dataframe(df)


@requires_distributed
def test_compute_dask_metadata():
    import polars as pl

    from moderndid.dask import compute_dask_metadata

    df = pl.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "time": [1, 2, 1, 2, 1, 2],
            "group": [0.0, 0.0, 2.0, 2.0, float("inf"), float("inf")],
        }
    )
    ddf = _polars_to_dask(df)

    with Client(n_workers=1, threads_per_worker=1, dashboard_address=None):
        meta = compute_dask_metadata(ddf, "group", "time", "id")

    np.testing.assert_array_equal(meta["tlist"], [1, 2])
    np.testing.assert_array_equal(meta["glist"], [2.0])
    assert meta["n_units"] == 3
    assert len(meta["unique_ids"]) == 3


@requires_distributed
def test_persist_by_group_handles_inf():
    import polars as pl

    from moderndid.dask import persist_by_group

    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "group": [0.0, 2.0, float("inf"), float("inf")],
        }
    )
    ddf = _polars_to_dask(df, npartitions=2)

    with Client(n_workers=1, threads_per_worker=1, dashboard_address=None) as client:
        persisted, group_to_parts, sentinel = persist_by_group(client, ddf, "group")

        assert sentinel is not None
        assert sentinel > 2.0
        assert isinstance(group_to_parts, dict)

        result = persisted.compute().reset_index()
        assert len(result) == 4

        from distributed import futures_of

        futs = futures_of(persisted)
        if futs:
            client.cancel(futs)


@requires_distributed
def test_persist_by_group_no_inf():
    import polars as pl

    from moderndid.dask import persist_by_group

    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "group": [0.0, 2.0, 3.0],
        }
    )
    ddf = _polars_to_dask(df, npartitions=2)

    with Client(n_workers=1, threads_per_worker=1, dashboard_address=None) as client:
        persisted, group_to_parts, sentinel = persist_by_group(client, ddf, "group")

        assert sentinel is None
        assert isinstance(group_to_parts, dict)

        from distributed import futures_of

        futs = futures_of(persisted)
        if futs:
            client.cancel(futs)


@requires_distributed
def test_ddd_panel_dask_matches_regular():
    from moderndid.didtriple.ddd import ddd
    from moderndid.didtriple.dgp import gen_dgp_mult_periods

    dgp = gen_dgp_mult_periods(n=200, random_state=42)
    pl_data = dgp["data"]
    ddf = _polars_to_dask(pl_data, npartitions=4)

    common = dict(
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method="reg",
    )

    result_regular = ddd(data=pl_data, **common)

    with Client(n_workers=2, threads_per_worker=1, dashboard_address=None):
        result_dask = ddd(data=ddf, **common, backend="dask")

    np.testing.assert_allclose(result_dask.att, result_regular.att, rtol=1e-10)
    np.testing.assert_allclose(result_dask.se, result_regular.se, rtol=1e-10)
    np.testing.assert_array_equal(result_dask.groups, result_regular.groups)
    np.testing.assert_array_equal(result_dask.times, result_regular.times)


@requires_distributed
def test_ddd_panel_dask_notyettreated():
    from moderndid.didtriple.ddd import ddd
    from moderndid.didtriple.dgp import gen_dgp_mult_periods

    dgp = gen_dgp_mult_periods(n=200, random_state=123)
    pl_data = dgp["data"]
    ddf = _polars_to_dask(pl_data, npartitions=4)

    common = dict(
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method="reg",
        control_group="notyettreated",
    )

    result_regular = ddd(data=pl_data, **common)

    with Client(n_workers=2, threads_per_worker=1, dashboard_address=None):
        result_dask = ddd(data=ddf, **common, backend="dask")

    np.testing.assert_allclose(result_dask.att, result_regular.att, rtol=1e-10)
    np.testing.assert_allclose(result_dask.se, result_regular.se, rtol=1e-10)


@requires_distributed
def test_ddd_panel_dask_varying_base_period():
    from moderndid.didtriple.ddd import ddd
    from moderndid.didtriple.dgp import gen_dgp_mult_periods

    dgp = gen_dgp_mult_periods(n=200, random_state=7)
    pl_data = dgp["data"]
    ddf = _polars_to_dask(pl_data, npartitions=4)

    common = dict(
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method="reg",
        base_period="varying",
    )

    result_regular = ddd(data=pl_data, **common)

    with Client(n_workers=2, threads_per_worker=1, dashboard_address=None):
        result_dask = ddd(data=ddf, **common, backend="dask")

    np.testing.assert_allclose(result_dask.att, result_regular.att, rtol=1e-10)
    np.testing.assert_allclose(result_dask.se, result_regular.se, rtol=1e-10)


@requires_distributed
def test_ddd_rc_dask_matches_regular():
    from moderndid.didtriple.ddd import ddd
    from moderndid.didtriple.dgp import gen_dgp_mult_periods

    dgp = gen_dgp_mult_periods(n=200, panel=False, random_state=42)
    pl_data = dgp["data"]
    ddf = _polars_to_dask(pl_data, npartitions=4)

    common = dict(
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        panel=False,
        est_method="reg",
    )

    result_regular = ddd(data=pl_data, **common)

    with Client(n_workers=2, threads_per_worker=1, dashboard_address=None):
        result_dask = ddd(data=ddf, **common, backend="dask")

    np.testing.assert_allclose(result_dask.att, result_regular.att, rtol=1e-10)
    np.testing.assert_allclose(result_dask.se, result_regular.se, rtol=1e-10)
    np.testing.assert_array_equal(result_dask.groups, result_regular.groups)
    np.testing.assert_array_equal(result_dask.times, result_regular.times)


@requires_distributed
def test_did_dask_matches_regular():
    from moderndid.core.data import load_mpdta
    from moderndid.did.att_gt import att_gt

    pl_data = load_mpdta()
    ddf = _polars_to_dask(pl_data, npartitions=4)

    common = dict(
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        panel=True,
        control_group="nevertreated",
        base_period="varying",
    )

    result_regular = att_gt(data=pl_data, **common)

    with Client(n_workers=2, threads_per_worker=1, dashboard_address=None):
        result_dask = att_gt(data=ddf, **common, backend="dask")

    np.testing.assert_allclose(result_dask.att_gt, result_regular.att_gt, rtol=1e-10)
    np.testing.assert_allclose(result_dask.se_gt, result_regular.se_gt, rtol=1e-10)
    np.testing.assert_array_equal(result_dask.groups, result_regular.groups)
    np.testing.assert_array_equal(result_dask.times, result_regular.times)


@requires_distributed
def test_did_dask_notyettreated():
    from moderndid.core.data import load_mpdta
    from moderndid.did.att_gt import att_gt

    pl_data = load_mpdta()
    ddf = _polars_to_dask(pl_data, npartitions=4)

    common = dict(
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        panel=True,
        control_group="notyettreated",
        base_period="varying",
    )

    result_regular = att_gt(data=pl_data, **common)

    with Client(n_workers=2, threads_per_worker=1, dashboard_address=None):
        result_dask = att_gt(data=ddf, **common, backend="dask")

    np.testing.assert_allclose(result_dask.att_gt, result_regular.att_gt, rtol=1e-10)
    np.testing.assert_allclose(result_dask.se_gt, result_regular.se_gt, rtol=1e-10)


@requires_distributed
def test_did_dask_universal_base_period():
    from moderndid.core.data import load_mpdta
    from moderndid.did.att_gt import att_gt

    pl_data = load_mpdta()
    ddf = _polars_to_dask(pl_data, npartitions=4)

    common = dict(
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        panel=True,
        control_group="nevertreated",
        base_period="universal",
    )

    result_regular = att_gt(data=pl_data, **common)

    with Client(n_workers=2, threads_per_worker=1, dashboard_address=None):
        result_dask = att_gt(data=ddf, **common, backend="dask")

    np.testing.assert_allclose(result_dask.att_gt, result_regular.att_gt, rtol=1e-10)
    np.testing.assert_allclose(result_dask.se_gt, result_regular.se_gt, rtol=1e-10)


@requires_distributed
def test_pte_dose_dask_matches_regular():
    from moderndid.core.data import simulate_cont_did_data
    from moderndid.didcont.cont_did import cont_did

    pl_data = simulate_cont_did_data(n=200, seed=42).rename({"time_period": "period"})
    ddf = _polars_to_dask(pl_data, npartitions=4)

    common = dict(
        yname="Y",
        tname="period",
        idname="id",
        gname="G",
        dname="D",
        degree=2,
        num_knots=0,
        biters=10,
        random_state=42,
        aggregation="dose",
    )

    result_regular = cont_did(data=pl_data, **common)

    with Client(n_workers=2, threads_per_worker=1, dashboard_address=None):
        result_dask = cont_did(data=ddf, **common, backend="dask")

    np.testing.assert_allclose(result_dask.overall_att, result_regular.overall_att, rtol=1e-10)
    np.testing.assert_allclose(result_dask.att_d, result_regular.att_d, rtol=1e-10)
    np.testing.assert_allclose(result_dask.acrt_d, result_regular.acrt_d, rtol=1e-10)


@requires_distributed
def test_pte_eventstudy_dask_matches_regular():
    from moderndid.core.data import simulate_cont_did_data
    from moderndid.didcont.cont_did import cont_did

    pl_data = simulate_cont_did_data(n=200, seed=42).rename({"time_period": "period"})
    ddf = _polars_to_dask(pl_data, npartitions=4)

    common = dict(
        yname="Y",
        tname="period",
        idname="id",
        gname="G",
        dname="D",
        degree=2,
        num_knots=0,
        biters=10,
        random_state=42,
        aggregation="eventstudy",
    )

    result_regular = cont_did(data=pl_data, **common)

    with Client(n_workers=2, threads_per_worker=1, dashboard_address=None):
        result_dask = cont_did(data=ddf, **common, backend="dask")

    np.testing.assert_allclose(result_dask.att_gt.att, result_regular.att_gt.att, rtol=1e-10)
    np.testing.assert_allclose(result_dask.att_gt.se, result_regular.att_gt.se, rtol=1e-10)
    np.testing.assert_array_equal(result_dask.att_gt.groups, result_regular.att_gt.groups)
    np.testing.assert_array_equal(result_dask.att_gt.times, result_regular.att_gt.times)


@requires_distributed
def test_dask_auto_switches_backend():
    from moderndid.didtriple.ddd import ddd
    from moderndid.didtriple.dgp import gen_dgp_mult_periods

    dgp = gen_dgp_mult_periods(n=200, random_state=42)
    ddf = _polars_to_dask(dgp["data"], npartitions=4)

    with (
        Client(n_workers=1, threads_per_worker=1, dashboard_address=None),
        pytest.warns(UserWarning, match="Switching to backend='dask'"),
    ):
        result = ddd(
            data=ddf,
            yname="y",
            tname="time",
            idname="id",
            gname="group",
            pname="partition",
            est_method="reg",
            backend="threads",
        )

    assert len(result.att) > 0


@requires_distributed
def test_dask_multiple_npartitions():
    from moderndid.didtriple.ddd import ddd
    from moderndid.didtriple.dgp import gen_dgp_mult_periods

    dgp = gen_dgp_mult_periods(n=200, random_state=42)
    pl_data = dgp["data"]

    common = dict(
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method="reg",
        backend="dask",
    )

    with Client(n_workers=2, threads_per_worker=1, dashboard_address=None):
        result_2 = ddd(data=_polars_to_dask(pl_data, npartitions=2), **common)
        result_8 = ddd(data=_polars_to_dask(pl_data, npartitions=8), **common)

    np.testing.assert_allclose(result_2.att, result_8.att, rtol=1e-10)
    np.testing.assert_allclose(result_2.se, result_8.se, rtol=1e-10)
