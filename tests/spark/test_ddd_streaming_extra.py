"""Tests for remaining DDD streaming helpers (wide arrays, RC, global agg)."""

import numpy as np
import pandas as pd
import pytest

from moderndid.spark._ddd_streaming import (
    _build_ddd_rc_partition_arrays,
    _build_partition_arrays_wide,
    _build_rc_global_agg,
    _filter_rc_partition_for_ctrl,
    _partition_compute_ddd_if,
    _partition_ddd_rc_global_stats,
    _partition_ddd_rc_or_gram,
    _partition_ddd_rc_pscore_gram,
)


def _make_ddd_rc_pdf(n=40, g=3, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    groups = rng.choice([0, g], size=n)
    parts = rng.choice([0, 1], size=n)
    return pd.DataFrame(
        {
            "y": rng.standard_normal(n),
            "_post": rng.choice([0, 1], size=n),
            "group": groups,
            "partition": parts,
        }
    )


def test_build_partition_arrays_wide_basic():
    rng = np.random.default_rng(20)
    n = 20
    pdf = pd.DataFrame(
        {
            "id": np.arange(n),
            "group": rng.choice([0, 3], size=n),
            "partition": rng.choice([0, 1], size=n),
            "_y_2": rng.standard_normal(n),
            "_y_1": rng.standard_normal(n),
        }
    )
    result = _build_partition_arrays_wide(pdf, "id", "group", "partition", 3, None, "_y_2", "_y_1")
    assert result is not None
    assert set(result.keys()) == {"ids", "y1", "y0", "subgroup", "X", "n", "groups_raw", "parts_raw", "weights"}
    assert result["n"] == n
    assert result["X"].shape == (n, 1)


def test_build_partition_arrays_wide_covariates():
    rng = np.random.default_rng(21)
    n = 15
    pdf = pd.DataFrame(
        {
            "id": np.arange(n),
            "group": rng.choice([0, 3], size=n),
            "partition": rng.choice([0, 1], size=n),
            "_y_2": rng.standard_normal(n),
            "_y_1": rng.standard_normal(n),
            "cov1": rng.standard_normal(n),
        }
    )
    result = _build_partition_arrays_wide(pdf, "id", "group", "partition", 3, ["cov1"], "_y_2", "_y_1")
    assert result["X"].shape == (n, 2)


def test_build_partition_arrays_wide_empty():
    pdf = pd.DataFrame({"id": [], "group": [], "partition": [], "_y_2": [], "_y_1": []})
    result = _build_partition_arrays_wide(pdf, "id", "group", "partition", 3, None, "_y_2", "_y_1")
    assert result is None


def test_build_partition_arrays_wide_subgroup_encoding():
    pdf = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "group": [3, 3, 0, 0],
            "partition": [1, 0, 1, 0],
            "_y_2": [1.0, 2.0, 3.0, 4.0],
            "_y_1": [0.5, 1.5, 2.5, 3.5],
        }
    )
    result = _build_partition_arrays_wide(pdf, "id", "group", "partition", 3, None, "_y_2", "_y_1")
    np.testing.assert_array_equal(result["subgroup"], np.array([4, 3, 2, 1]))


def test_build_ddd_rc_partition_arrays_basic():
    pdf = _make_ddd_rc_pdf(n=30, g=3)
    result = _build_ddd_rc_partition_arrays(pdf, 0, "y", "group", "partition", 3, None)
    assert result is not None
    assert set(result.keys()) == {"ids", "y", "post", "subgroup", "X", "n", "weights", "groups_raw", "parts_raw"}
    assert result["n"] == 30
    np.testing.assert_array_equal(result["ids"], np.arange(30, dtype=np.int64))


def test_build_ddd_rc_partition_arrays_offset():
    pdf = _make_ddd_rc_pdf(n=10, g=3)
    result = _build_ddd_rc_partition_arrays(pdf, 50, "y", "group", "partition", 3, None)
    np.testing.assert_array_equal(result["ids"], np.arange(50, 60, dtype=np.int64))


def test_build_ddd_rc_partition_arrays_with_covariates():
    rng = np.random.default_rng(30)
    pdf = _make_ddd_rc_pdf(n=20, g=2, rng=rng)
    pdf["cov1"] = rng.standard_normal(20)
    result = _build_ddd_rc_partition_arrays(pdf, 0, "y", "group", "partition", 2, ["cov1"])
    assert result["X"].shape == (20, 2)


def test_build_ddd_rc_partition_arrays_empty():
    pdf = pd.DataFrame({"y": [], "_post": [], "group": [], "partition": []})
    result = _build_ddd_rc_partition_arrays(pdf, 0, "y", "group", "partition", 3, None)
    assert result is None


def test_filter_rc_partition_for_ctrl_none():
    assert _filter_rc_partition_for_ctrl(None, 3, 0) is None


def test_filter_rc_partition_for_ctrl_valid(ddd_rc_partition):
    result = _filter_rc_partition_for_ctrl(ddd_rc_partition, 3, 0)
    assert result is not None
    assert result["n"] > 0
    unique_groups = set(result["groups_raw"])
    assert unique_groups.issubset({3, 0})
    assert "post" in result
    assert "y" in result


def test_filter_rc_partition_for_ctrl_no_match(ddd_rc_partition):
    result = _filter_rc_partition_for_ctrl(ddd_rc_partition, 3, 99)
    if result is not None:
        assert all(gv == 3 for gv in result["groups_raw"])


def test_partition_ddd_rc_pscore_gram_none():
    assert _partition_ddd_rc_pscore_gram(None, 3, np.zeros(1)) is None


def test_partition_ddd_rc_pscore_gram_valid(ddd_rc_partition):
    k = ddd_rc_partition["X"].shape[1]
    result = _partition_ddd_rc_pscore_gram(ddd_rc_partition, 3, np.zeros(k))
    assert result is not None
    XtWX, XtWz, n = result
    assert XtWX.shape == (k, k)
    assert XtWz.shape == (k,)
    assert n > 0


def test_partition_ddd_rc_pscore_gram_no_match(ddd_rc_partition):
    part = {**ddd_rc_partition, "subgroup": np.ones(ddd_rc_partition["n"], dtype=int)}
    result = _partition_ddd_rc_pscore_gram(part, 3, np.zeros(part["X"].shape[1]))
    assert result is None


@pytest.mark.parametrize("d_val,post_val", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_partition_ddd_rc_or_gram_none(d_val, post_val):
    assert _partition_ddd_rc_or_gram(None, 3, d_val, post_val) is None


def test_partition_ddd_rc_or_gram_valid(ddd_rc_partition):
    result = _partition_ddd_rc_or_gram(ddd_rc_partition, 1, 0, 0)
    if result is not None:
        XtWX, XtWy, n = result
        k = ddd_rc_partition["X"].shape[1]
        assert XtWX.shape == (k, k)
        assert XtWy.shape == (k,)
        assert n > 0


@pytest.mark.parametrize("est_method", ["dr", "reg"])
def test_partition_ddd_rc_global_stats_keys(ddd_rc_partition, est_method):
    k = ddd_rc_partition["X"].shape[1]
    or_betas = {key: np.zeros(k) for key in ["cont_pre", "cont_post", "treat_pre", "treat_post"]}
    result = _partition_ddd_rc_global_stats(ddd_rc_partition, 3, np.zeros(k), or_betas, est_method, 0.995)
    assert result is not None
    assert result["n_sub"] > 0
    assert "sum_w_treat_pre" in result
    assert "sum_w_treat_post" in result
    assert "info_gram" in result


def test_partition_ddd_rc_global_stats_none():
    or_betas = {key: np.zeros(1) for key in ["cont_pre", "cont_post", "treat_pre", "treat_post"]}
    result = _partition_ddd_rc_global_stats(None, 3, np.zeros(1), or_betas, "reg", 0.995)
    assert result is None


def test_build_rc_global_agg_basic():
    n_sub = 100
    agg = {
        "sum_w_treat_pre": 10.0,
        "sum_w_treat_post": 12.0,
        "sum_w_cont_pre": 8.0,
        "sum_w_cont_post": 9.0,
        "sum_w_d": 22.0,
        "sum_w_dt1": 12.0,
        "sum_w_dt0": 10.0,
        "sum_eta_treat_pre": 5.0,
        "sum_eta_treat_post": 6.0,
        "sum_eta_cont_pre": 3.0,
        "sum_eta_cont_post": 4.0,
        "sum_eta_d_post": 2.0,
        "sum_eta_dt1_post": 1.5,
        "sum_eta_d_pre": 1.0,
        "sum_eta_dt0_pre": 0.8,
        "n_sub": n_sub,
    }
    result = _build_rc_global_agg(agg, n_sub)
    assert result["n_sub"] == n_sub
    assert result["mean_w_treat_pre"] == pytest.approx(10.0 / n_sub)
    assert result["mean_w_treat_post"] == pytest.approx(12.0 / n_sub)
    assert "dr_att" in result
    assert np.isfinite(result["dr_att"])


def test_build_rc_global_agg_zero_weights():
    n_sub = 50
    agg = {f"sum_w_{key}": 0.0 for key in ["treat_pre", "treat_post", "cont_pre", "cont_post", "d", "dt1", "dt0"]}
    agg.update(
        {
            f"sum_eta_{key}": 0.0
            for key in [
                "treat_pre",
                "treat_post",
                "cont_pre",
                "cont_post",
                "d_post",
                "dt1_post",
                "d_pre",
                "dt0_pre",
            ]
        }
    )
    agg["n_sub"] = n_sub
    result = _build_rc_global_agg(agg, n_sub)
    assert result["dr_att"] == 0.0
    assert result["att_treat_pre"] == 0.0


def test_partition_compute_ddd_if_none():
    ids, if_arr = _partition_compute_ddd_if(
        None,
        {},
        {},
        {},
        "reg",
        0.995,
        1.0,
        1.0,
        1.0,
        {},
        {},
        {},
    )
    assert len(ids) == 0
    assert len(if_arr) == 0


@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_partition_compute_ddd_if_valid(partition_dict, est_method):
    k = partition_dict["X"].shape[1]

    def _make_agg(sg):
        return {
            "mean_w_treat": 1.0,
            "mean_w_control": 1.0,
            "att_treat": 0.5,
            "att_control": 0.3,
        }

    global_agg = {cs: _make_agg(cs) for cs in [3, 2, 1]}
    ps_betas = {cs: np.zeros(k) for cs in [3, 2, 1]}
    or_betas = {cs: np.zeros(k) for cs in [3, 2, 1]}
    hm2 = {cs: np.zeros(k) for cs in [3, 2, 1]}
    xim1 = {cs: np.zeros(k) for cs in [3, 2, 1]}
    xim3 = {cs: np.zeros(k) for cs in [3, 2, 1]}

    ids, if_arr = _partition_compute_ddd_if(
        partition_dict,
        ps_betas,
        or_betas,
        global_agg,
        est_method,
        0.995,
        1.0,
        1.0,
        1.0,
        hm2,
        xim1,
        xim3,
    )
    assert len(ids) == partition_dict["n"]
    assert len(if_arr) == partition_dict["n"]
    assert np.all(np.isfinite(if_arr))
