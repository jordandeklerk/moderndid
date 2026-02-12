"""Tests for the parallel execution backends."""

from __future__ import annotations

import pickle
from unittest.mock import patch

import numpy as np
import pytest

dask = pytest.importorskip("dask")
distributed = pytest.importorskip("distributed")
Client = distributed.Client
Future = distributed.Future

from moderndid import att_gt, load_mpdta, simulate_cont_did_data
from moderndid.core.parallel import (
    _parallel_map_dask,
    _scatter_args,
    dask_available,
    parallel_map,
)
from moderndid.core.preprocess import preprocess_did
from moderndid.did.compute_att_gt import compute_att_gt
from moderndid.didcont.cont_did import cont_did
from moderndid.didtriple.ddd import ddd
from moderndid.didtriple.dgp import gen_dgp_2periods, gen_dgp_mult_periods
from tests.conftest import (
    CountingScheduler,
    _add,
    _ols_cell,
    _raise_on_two,
    _square,
    requires_dask,
    requires_distributed,
)


def test_dask_available_returns_bool():
    assert isinstance(dask_available(), bool)


def test_sequential_fallback():
    args = [(i,) for i in range(5)]
    assert parallel_map(_square, args, n_jobs=1) == [0, 1, 4, 9, 16]


@pytest.mark.parametrize("backend", ["threads", "dask"])
def test_sequential_ignores_backend(backend):
    args = [(i,) for i in range(5)]
    assert parallel_map(_square, args, n_jobs=1, backend=backend) == [0, 1, 4, 9, 16]


def test_threads_backend():
    args = [(i,) for i in range(10)]
    assert parallel_map(_square, args, n_jobs=2, backend="threads") == [i * i for i in range(10)]


def test_threads_backend_all_cores():
    args = [(i,) for i in range(10)]
    assert parallel_map(_square, args, n_jobs=-1, backend="threads") == [i * i for i in range(10)]


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_empty_args_list(n_jobs):
    assert parallel_map(_square, [], n_jobs=n_jobs) == []


def test_invalid_backend_raises():
    with pytest.raises(ValueError, match="backend='invalid' is not valid"):
        parallel_map(_square, [(1,)], n_jobs=2, backend="invalid")


def test_dask_import_error_when_unavailable():
    with (
        patch("moderndid.core.parallel.dask_available", return_value=False),
        pytest.raises(ImportError, match="Dask is required"),
    ):
        _parallel_map_dask(_square, [(1,)])


def test_exception_propagates_threads():
    with pytest.raises(ValueError, match="boom"):
        parallel_map(_raise_on_two, [(1,), (2,), (3,)], n_jobs=2, backend="threads")


@requires_dask
def test_dask_delayed_builds_lazy_graph():
    call_count = 0

    def _counting_square(x):
        nonlocal call_count
        call_count += 1
        return x * x

    delayed_results = [dask.delayed(_counting_square)(i) for i in range(5)]
    assert call_count == 0

    results = list(dask.compute(*delayed_results))
    assert call_count == 5
    assert results == [0, 1, 4, 9, 16]


@requires_dask
def test_dask_dispatches_through_scheduler():
    scheduler = CountingScheduler()
    with dask.config.set(scheduler=scheduler):
        result = parallel_map(_square, [(i,) for i in range(5)], n_jobs=2, backend="dask")
    assert scheduler.total_computes > 0
    assert result == [0, 1, 4, 9, 16]


@requires_dask
@pytest.mark.parametrize("scheduler", ["synchronous", "threads"])
def test_dask_backend_across_schedulers(scheduler):
    args = [(i,) for i in range(10)]
    with dask.config.set(scheduler=scheduler):
        result = parallel_map(_square, args, n_jobs=2, backend="dask")
    assert result == [i * i for i in range(10)]


@requires_dask
@pytest.mark.parametrize(
    "func,args,expected",
    [
        (_square, [(i,) for i in range(10)], [i * i for i in range(10)]),
        (_add, [(i, i + 1) for i in range(5)], [1, 3, 5, 7, 9]),
    ],
    ids=["single_arg", "multi_arg"],
)
def test_dask_backend_correctness(func, args, expected):
    assert parallel_map(func, args, n_jobs=2, backend="dask") == expected


@requires_dask
def test_dask_matches_threads():
    args = [(i,) for i in range(20)]
    result_threads = parallel_map(_square, args, n_jobs=2, backend="threads")
    result_dask = parallel_map(_square, args, n_jobs=2, backend="dask")
    assert result_threads == result_dask


@requires_dask
def test_dask_preserves_order():
    args = [(i,) for i in range(20)]
    result = parallel_map(_square, args, n_jobs=4, backend="dask")
    np.testing.assert_array_equal(result, [i * i for i in range(20)])


@requires_dask
def test_dask_numpy_workload():
    rng = np.random.default_rng(42)
    n, k = 200, 3
    args = []
    for _ in range(8):
        x = rng.standard_normal((n, k))
        x[:, 0] = 1.0
        y = x @ np.array([1.0, 2.0, -0.5]) + rng.standard_normal(n) * 0.1
        args.append((x, y))

    result_seq = parallel_map(_ols_cell, args, n_jobs=1)
    result_dask = parallel_map(_ols_cell, args, n_jobs=2, backend="dask")

    for r_seq, r_dask in zip(result_seq, result_dask):
        np.testing.assert_allclose(r_seq, r_dask, rtol=1e-12)


@requires_dask
def test_dask_empty_args_list():
    assert parallel_map(_square, [], n_jobs=2, backend="dask") == []


@requires_dask
def test_exception_propagates_dask():
    with pytest.raises(ValueError, match="boom"):
        parallel_map(_raise_on_two, [(1,), (2,), (3,)], n_jobs=2, backend="dask")


def test_att_gt_invalid_backend_raises():
    with pytest.raises(ValueError, match="backend='invalid' is not valid"):
        att_gt(
            data=load_mpdta(),
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
            backend="invalid",
        )


@pytest.mark.parametrize("n_jobs", [0, -2])
def test_att_gt_invalid_n_jobs_raises(n_jobs):
    with pytest.raises(ValueError, match="n_jobs=.* is not valid"):
        att_gt(
            data=load_mpdta(),
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
            n_jobs=n_jobs,
        )


def test_cont_did_invalid_backend_raises():
    data = simulate_cont_did_data(n=100, seed=42).rename({"time_period": "period"})
    with pytest.raises(ValueError, match="backend='invalid' is not valid"):
        cont_did(
            data=data,
            yname="Y",
            tname="period",
            idname="id",
            gname="G",
            dname="D",
            backend="invalid",
        )


@pytest.mark.parametrize("n_jobs", [0, -2])
def test_cont_did_invalid_n_jobs_raises(n_jobs):
    data = simulate_cont_did_data(n=100, seed=42).rename({"time_period": "period"})
    with pytest.raises(ValueError, match="n_jobs=.* is not valid"):
        cont_did(
            data=data,
            yname="Y",
            tname="period",
            idname="id",
            gname="G",
            dname="D",
            n_jobs=n_jobs,
        )


def test_ddd_invalid_backend_raises():
    dgp = gen_dgp_mult_periods(n=100, random_state=42)
    with pytest.raises(ValueError, match="backend='invalid' is not valid"):
        ddd(
            data=dgp["data"],
            yname="y",
            tname="time",
            idname="id",
            gname="group",
            pname="partition",
            backend="invalid",
        )


@pytest.mark.parametrize("n_jobs", [0, -2])
def test_ddd_invalid_n_jobs_raises(n_jobs):
    dgp = gen_dgp_mult_periods(n=100, random_state=42)
    with pytest.raises(ValueError, match="n_jobs=.* is not valid"):
        ddd(
            data=dgp["data"],
            yname="y",
            tname="time",
            idname="id",
            gname="group",
            pname="partition",
            n_jobs=n_jobs,
        )


def test_ddd_warns_parallel_args_ignored_for_2_period_panel():
    dgp = gen_dgp_2periods(n=200, dgp_type=1, random_state=42)
    with pytest.warns(UserWarning, match="ignored for 2-period data"):
        ddd(
            data=dgp["data"],
            yname="y",
            tname="time",
            idname="id",
            gname="state",
            pname="partition",
            n_jobs=2,
            backend="dask",
        )


def test_ddd_warns_parallel_args_ignored_for_2_period_rcs():
    dgp = gen_dgp_2periods(n=200, dgp_type=1, panel=False, random_state=42)
    with pytest.warns(UserWarning, match="ignored for 2-period data"):
        ddd(
            data=dgp["data"],
            yname="y",
            tname="time",
            gname="state",
            pname="partition",
            panel=False,
            n_jobs=2,
            backend="dask",
        )


@pytest.mark.parametrize(
    "backend",
    [
        "threads",
        pytest.param("dask", marks=requires_dask),
    ],
)
def test_att_gt_parallel_matches_sequential(backend):
    data = preprocess_did(
        load_mpdta(),
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        panel=True,
        control_group="nevertreated",
        base_period="varying",
    )

    result_seq = compute_att_gt(data, n_jobs=1)
    result_par = compute_att_gt(data, n_jobs=2, backend=backend)

    assert len(result_seq.attgt_list) == len(result_par.attgt_list)
    for r1, r2 in zip(result_seq.attgt_list, result_par.attgt_list):
        np.testing.assert_allclose(r1.att, r2.att, rtol=1e-10)
        assert r1.group == r2.group
        assert r1.year == r2.year
        assert r1.post == r2.post

    np.testing.assert_allclose(
        result_seq.influence_functions.toarray(),
        result_par.influence_functions.toarray(),
        rtol=1e-10,
    )


@pytest.mark.parametrize(
    "backend",
    [
        "threads",
        pytest.param("dask", marks=requires_dask),
    ],
)
def test_cont_did_parallel_matches_sequential(backend):
    data = simulate_cont_did_data(n=5000, seed=12345).rename({"time_period": "period"})

    common = dict(
        data=data,
        yname="Y",
        tname="period",
        idname="id",
        gname="G",
        dname="D",
        degree=2,
        num_knots=0,
        biters=10,
        random_state=42,
    )

    result_seq = cont_did(**common, n_jobs=1)
    result_par = cont_did(**common, n_jobs=2, backend=backend)

    np.testing.assert_allclose(result_seq.overall_att, result_par.overall_att, rtol=1e-10)
    np.testing.assert_allclose(result_seq.att_d, result_par.att_d, rtol=1e-10)
    np.testing.assert_allclose(result_seq.acrt_d, result_par.acrt_d, rtol=1e-10)


@pytest.mark.parametrize(
    "backend",
    [
        "threads",
        pytest.param("dask", marks=requires_dask),
    ],
)
def test_ddd_parallel_matches_sequential(backend):
    dgp = gen_dgp_mult_periods(n=5000, random_state=42)

    common = dict(
        data=dgp["data"],
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method="reg",
    )

    result_seq = ddd(**common, n_jobs=1)
    result_par = ddd(**common, n_jobs=2, backend=backend)

    np.testing.assert_allclose(result_seq.att, result_par.att, rtol=1e-10)
    np.testing.assert_array_equal(result_seq.groups, result_par.groups)
    np.testing.assert_array_equal(result_seq.times, result_par.times)
    np.testing.assert_allclose(result_seq.inf_func_mat, result_par.inf_func_mat, rtol=1e-10)


@requires_dask
def test_att_gt_dask_dispatches_through_scheduler():
    data = preprocess_did(
        load_mpdta(),
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        panel=True,
        control_group="nevertreated",
        base_period="varying",
    )

    scheduler = CountingScheduler()
    with dask.config.set(scheduler=scheduler):
        result = compute_att_gt(data, n_jobs=2, backend="dask")

    assert scheduler.total_computes > 0
    assert len(result.attgt_list) > 0


@pytest.mark.parametrize(
    "args_list",
    [
        [],
        [("a", "b", "c")],
    ],
    ids=["empty", "single_task"],
)
def test_scatter_noop_short_args(args_list):
    result = _scatter_args(args_list)
    assert result is args_list


def test_scatter_noop_no_distributed():
    shared = [1, 2, 3]
    args_list = [(shared, 0), (shared, 1), (shared, 2)]
    result = _scatter_args(args_list)
    assert result is args_list


def test_scatter_noop_no_shared_objects():
    args_list = [([1], "a"), ([2], "b"), ([3], "c")]
    result = _scatter_args(args_list)
    assert result is args_list


@requires_dask
@pytest.mark.parametrize(
    "scheduler_override",
    [
        "synchronous",
        "threads",
        pytest.param(CountingScheduler(), id="custom_callable"),
    ],
)
def test_scatter_noop_when_scheduler_overridden(scheduler_override):
    shared = list(range(1000))
    args_list = [(shared, i) for i in range(5)]

    with dask.config.set(scheduler=scheduler_override):
        result = _scatter_args(args_list)
    assert result is args_list


@requires_dask
def test_dask_with_custom_scheduler_and_shared_data():
    shared_data = np.arange(100)
    args_list = [(shared_data, i) for i in range(5)]

    def _sum_with_offset(data, offset):
        return data.sum() + offset

    scheduler = CountingScheduler()
    with dask.config.set(scheduler=scheduler):
        result = parallel_map(_sum_with_offset, args_list, n_jobs=2, backend="dask")

    expected_sum = shared_data.sum()
    assert result == [expected_sum + i for i in range(5)]
    assert scheduler.total_computes > 0


@requires_dask
@pytest.mark.parametrize("scheduler", ["synchronous", "threads"])
def test_dask_with_builtin_scheduler_and_shared_data(scheduler):
    shared_data = list(range(50))
    args_list = [(shared_data, i) for i in range(4)]

    def _len_plus(data, offset):
        return len(data) + offset

    with dask.config.set(scheduler=scheduler):
        result = parallel_map(_len_plus, args_list, n_jobs=2, backend="dask")

    assert result == [50, 51, 52, 53]


@requires_distributed
def test_scatter_replaces_shared_args_with_futures():
    shared_data = np.arange(100)
    args_list = [(shared_data, 0), (shared_data, 1), (shared_data, 2)]

    with Client(n_workers=1, threads_per_worker=1, dashboard_address=None):
        result = _scatter_args(args_list)

    assert all(isinstance(args[0], Future) for args in result)
    assert result[0][0] is result[1][0] is result[2][0]
    assert [args[1] for args in result] == [0, 1, 2]


@requires_distributed
def test_scatter_leaves_unique_args_as_values():
    shared_data = np.arange(100)
    args_list = [(shared_data, 0), (shared_data, 1)]

    with Client(n_workers=1, threads_per_worker=1, dashboard_address=None):
        result = _scatter_args(args_list)

    assert isinstance(result[0][0], Future)
    assert not isinstance(result[0][1], Future)
    assert not isinstance(result[1][1], Future)


@requires_distributed
def test_scatter_multiple_shared_objects():
    data_a = np.arange(50)
    data_b = np.arange(30)
    args_list = [(data_a, data_b, 0), (data_a, data_b, 1), (data_a, data_b, 2)]

    with Client(n_workers=1, threads_per_worker=1, dashboard_address=None):
        result = _scatter_args(args_list)

    assert all(isinstance(args[0], Future) for args in result)
    assert all(isinstance(args[1], Future) for args in result)
    assert all(isinstance(args[2], int) for args in result)


@requires_distributed
def test_scatter_noop_with_no_shared_objects_and_client():
    args_list = [([1], "a"), ([2], "b"), ([3], "c")]

    with Client(n_workers=1, threads_per_worker=1, dashboard_address=None):
        result = _scatter_args(args_list)

    assert result is args_list


@requires_distributed
def test_scatter_unique_per_task_arrays():
    arr_a = np.arange(100)
    arr_b = np.arange(200)
    arr_c = np.arange(300)
    args_list = [(arr_a, 0), (arr_b, 1), (arr_c, 2)]

    with Client(n_workers=1, threads_per_worker=1, dashboard_address=None):
        result = _scatter_args(args_list)

    assert all(isinstance(args[0], Future) for args in result)
    assert all(not isinstance(args[1], Future) for args in result)


@requires_distributed
def test_parallel_map_dask_with_distributed_client():
    shared_data = np.arange(100)
    args_list = [(shared_data, i) for i in range(5)]

    def _sum_with_offset(data, offset):
        return data.sum() + offset

    with Client(n_workers=2, threads_per_worker=1, dashboard_address=None):
        result = parallel_map(_sum_with_offset, args_list, n_jobs=2, backend="dask")

    expected_sum = shared_data.sum()
    assert result == [expected_sum + i for i in range(5)]


@requires_distributed
def test_parallel_map_dask_distributed_matches_sequential():
    rng = np.random.default_rng(42)
    n, k = 200, 3
    args = []
    for _ in range(8):
        x = rng.standard_normal((n, k))
        x[:, 0] = 1.0
        y = x @ np.array([1.0, 2.0, -0.5]) + rng.standard_normal(n) * 0.1
        args.append((x, y))

    result_seq = parallel_map(_ols_cell, args, n_jobs=1)

    with Client(n_workers=2, threads_per_worker=1, dashboard_address=None):
        result_dist = parallel_map(_ols_cell, args, n_jobs=2, backend="dask")

    for r_seq, r_dist in zip(result_seq, result_dist):
        np.testing.assert_allclose(r_seq, r_dist, rtol=1e-12)


@requires_distributed
def test_scattered_futures_persist_across_batches():
    import polars as pl

    from moderndid.core.parallel import _submit_with_scattered_args

    shared_df = pl.DataFrame({"x": list(range(100))})

    def _row_count_plus(df, offset):
        return df.shape[0] + offset

    args_list = [(shared_df, i) for i in range(6)]

    with Client(n_workers=1, threads_per_worker=1, dashboard_address=None) as client:
        original_scatter = client.scatter

        scatter_call_count = 0

        def _counting_scatter(*args, **kwargs):
            nonlocal scatter_call_count
            scatter_call_count += 1
            return original_scatter(*args, **kwargs)

        client.scatter = _counting_scatter

        result = _submit_with_scattered_args(client, _row_count_plus, args_list)

    assert result == [100 + i for i in range(6)]
    assert scatter_call_count == 1


@requires_distributed
def test_scattered_futures_cleaned_on_exception():
    """Scattered data futures are cancelled even when a task raises."""
    import polars as pl

    from moderndid.core.parallel import _submit_with_scattered_args

    shared_df = pl.DataFrame({"x": list(range(100))})

    def _fail_on_three(df, offset):
        if offset == 3:
            raise ValueError("boom")
        return df.shape[0] + offset

    args_list = [(shared_df, i) for i in range(6)]

    with Client(n_workers=1, threads_per_worker=1, dashboard_address=None) as client:
        original_cancel = client.cancel
        cancelled_futures = []

        def _tracking_cancel(futures, **kwargs):
            if isinstance(futures, list):
                cancelled_futures.extend(futures)
            else:
                cancelled_futures.append(futures)
            return original_cancel(futures, **kwargs)

        client.cancel = _tracking_cancel

        with pytest.raises(ValueError, match="boom"):
            _submit_with_scattered_args(client, _fail_on_three, args_list)

    has_scattered_cancel = any(
        not hasattr(f, "key") or "row_count" not in str(getattr(f, "key", "")) for f in cancelled_futures
    )
    assert has_scattered_cancel


@requires_distributed
def test_scattered_futures_released_after_last_use():
    """Scattered data for early-only tasks is released before later tasks run."""
    import polars as pl

    from moderndid.core.parallel import _submit_with_scattered_args

    df_early = pl.DataFrame({"x": list(range(100))})
    df_late = pl.DataFrame({"x": list(range(200))})

    def _row_count(df):
        return df.shape[0]

    # With 1 worker → max_inflight=2.
    # Initial window (tasks 0-1): scatter df_early.
    # Backfill task 2: scatter df_late (df_early released when ref hits 0).
    # Backfill task 3: df_late already scattered.
    args_list = [(df_early,), (df_early,), (df_late,), (df_late,)]

    with Client(n_workers=1, threads_per_worker=1, dashboard_address=None) as client:
        original_scatter = client.scatter
        scatter_object_counts: list[int] = []

        def _tracking_scatter(objs, **kwargs):
            scatter_object_counts.append(len(objs))
            return original_scatter(objs, **kwargs)

        client.scatter = _tracking_scatter

        result = _submit_with_scattered_args(client, _row_count, args_list)

    assert result == [100, 100, 200, 200]
    # Each unique DataFrame scattered exactly once (total 2 objects),
    # in separate calls (incremental, not all upfront).
    assert sum(scatter_object_counts) == 2
    assert len(scatter_object_counts) == 2


@requires_distributed
def test_att_gt_with_distributed_client():
    data = preprocess_did(
        load_mpdta(),
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        panel=True,
        control_group="nevertreated",
        base_period="varying",
    )

    result_seq = compute_att_gt(data, n_jobs=1)

    with Client(n_workers=2, threads_per_worker=1, dashboard_address=None):
        result_dist = compute_att_gt(data, n_jobs=2, backend="dask")

    assert len(result_seq.attgt_list) == len(result_dist.attgt_list)
    for r1, r2 in zip(result_seq.attgt_list, result_dist.attgt_list):
        np.testing.assert_allclose(r1.att, r2.att, rtol=1e-10)


@requires_distributed
def test_exception_propagates_distributed():
    with Client(n_workers=2, threads_per_worker=1, dashboard_address=None), pytest.raises(ValueError, match="boom"):
        parallel_map(_raise_on_two, [(1,), (2,), (3,)], n_jobs=2, backend="dask")


@requires_distributed
def test_result_objects_survive_pickle_roundtrip():
    data = preprocess_did(
        load_mpdta(),
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        panel=True,
        control_group="nevertreated",
        base_period="varying",
    )

    with Client(n_workers=2, threads_per_worker=1, dashboard_address=None):
        result = compute_att_gt(data, n_jobs=2, backend="dask")

    roundtripped = pickle.loads(pickle.dumps(result))
    assert len(roundtripped.attgt_list) == len(result.attgt_list)
    for r1, r2 in zip(result.attgt_list, roundtripped.attgt_list):
        np.testing.assert_allclose(r1.att, r2.att, rtol=1e-14)
