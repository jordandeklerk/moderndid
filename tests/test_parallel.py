"""Tests for the parallel execution backends."""

from __future__ import annotations

import numpy as np
import pytest

from moderndid import att_gt, load_mpdta, simulate_cont_did_data
from moderndid.core.parallel import (
    dask_available,
    parallel_map,
)
from moderndid.core.preprocess import preprocess_did
from moderndid.did.compute_att_gt import compute_att_gt
from moderndid.didcont.cont_did import cont_did
from moderndid.didtriple.ddd import ddd
from moderndid.didtriple.dgp import gen_dgp_2periods, gen_dgp_mult_periods
from tests.conftest import (
    _raise_on_two,
    _square,
)


def test_dask_available_returns_bool():
    assert isinstance(dask_available(), bool)


def test_sequential_fallback():
    args = [(i,) for i in range(5)]
    assert parallel_map(_square, args, n_jobs=1) == [0, 1, 4, 9, 16]


def test_sequential_ignores_backend():
    args = [(i,) for i in range(5)]
    assert parallel_map(_square, args, n_jobs=1) == [0, 1, 4, 9, 16]


def test_threads_backend():
    args = [(i,) for i in range(10)]
    assert parallel_map(_square, args, n_jobs=2) == [i * i for i in range(10)]


def test_threads_backend_all_cores():
    args = [(i,) for i in range(10)]
    assert parallel_map(_square, args, n_jobs=-1) == [i * i for i in range(10)]


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_empty_args_list(n_jobs):
    assert parallel_map(_square, [], n_jobs=n_jobs) == []


def test_exception_propagates_threads():
    with pytest.raises(ValueError, match="boom"):
        parallel_map(_raise_on_two, [(1,), (2,), (3,)], n_jobs=2)


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
            backend="threads",
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
            backend="threads",
        )


def test_att_gt_parallel_matches_sequential():
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
    result_par = compute_att_gt(data, n_jobs=2)

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


def test_cont_did_parallel_matches_sequential():
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
    result_par = cont_did(**common, n_jobs=2)

    np.testing.assert_allclose(result_seq.overall_att, result_par.overall_att, rtol=1e-10)
    np.testing.assert_allclose(result_seq.att_d, result_par.att_d, rtol=1e-10)
    np.testing.assert_allclose(result_seq.acrt_d, result_par.acrt_d, rtol=1e-10)


def test_ddd_parallel_matches_sequential():
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
    result_par = ddd(**common, n_jobs=2)

    np.testing.assert_allclose(result_seq.att, result_par.att, rtol=1e-10)
    np.testing.assert_array_equal(result_seq.groups, result_par.groups)
    np.testing.assert_array_equal(result_seq.times, result_par.times)
    np.testing.assert_allclose(result_seq.inf_func_mat, result_par.inf_func_mat, rtol=1e-10)


# --- Tests: backend="dask" with regular DataFrame raises TypeError ---


def test_att_gt_dask_backend_regular_df_raises():
    with pytest.raises(TypeError, match="backend='dask' requires a Dask DataFrame"):
        att_gt(
            data=load_mpdta(),
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
            backend="dask",
        )


def test_ddd_dask_backend_regular_df_raises():
    dgp = gen_dgp_mult_periods(n=100, random_state=42)
    with pytest.raises(TypeError, match="backend='dask' requires a Dask DataFrame"):
        ddd(
            data=dgp["data"],
            yname="y",
            tname="time",
            idname="id",
            gname="group",
            pname="partition",
            backend="dask",
        )


def test_cont_did_dask_backend_regular_df_raises():
    data = simulate_cont_did_data(n=100, seed=42).rename({"time_period": "period"})
    with pytest.raises(TypeError, match="backend='dask' requires a Dask DataFrame"):
        cont_did(
            data=data,
            yname="Y",
            tname="period",
            idname="id",
            gname="G",
            dname="D",
            backend="dask",
        )
