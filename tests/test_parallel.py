"""Tests for the parallel execution backends."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from moderndid import att_gt, load_mpdta, simulate_cont_did_data
from moderndid.core.parallel import _parallel_map_dask, dask_available, parallel_map
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
    import dask

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
    import dask

    scheduler = CountingScheduler()
    with dask.config.set(scheduler=scheduler):
        result = parallel_map(_square, [(i,) for i in range(5)], n_jobs=2, backend="dask")
    assert scheduler.total_computes > 0
    assert result == [0, 1, 4, 9, 16]


@requires_dask
@pytest.mark.parametrize("scheduler", ["synchronous", "threads"])
def test_dask_backend_across_schedulers(scheduler):
    import dask

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
    import dask

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
