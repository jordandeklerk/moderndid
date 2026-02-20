"""Tests for DiD distributed streaming cell helpers."""

import numpy as np
import pandas as pd
import pytest

from moderndid.spark._did_streaming import (
    _build_did_partition_arrays,
    _build_did_partition_arrays_wide,
    _build_did_rc_partition_arrays,
    _partition_compute_did_if,
    _partition_did_global_stats,
    _partition_did_or_gram,
    _partition_did_pscore_gram,
    _partition_did_rc_global_stats,
    _partition_did_rc_or_gram,
    _partition_did_rc_pscore_gram,
)


def _make_did_merged_pdf(n=40, g=3, covariate_cols=None, weightsname=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    groups = rng.choice([0, g], size=n)
    pdf = pd.DataFrame(
        {
            "id": np.arange(n),
            "y": rng.standard_normal(n),
            "_y_pre": rng.standard_normal(n),
            "group": groups,
        }
    )
    if covariate_cols:
        for col in covariate_cols:
            pdf[col] = rng.standard_normal(n)
    if weightsname:
        pdf[weightsname] = rng.uniform(0.5, 1.5, size=n)
    return pdf


def _make_did_rc_pdf(n=40, g=3, covariate_cols=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    groups = rng.choice([0, g], size=n)
    pdf = pd.DataFrame(
        {
            "y": rng.standard_normal(n),
            "_post": rng.choice([0, 1], size=n),
            "group": groups,
        }
    )
    if covariate_cols:
        for col in covariate_cols:
            pdf[col] = rng.standard_normal(n)
    return pdf


def test_build_did_partition_arrays_basic():
    pdf = _make_did_merged_pdf(n=30, g=3)
    result = _build_did_partition_arrays(pdf, "id", "y", "group", 3, None)
    assert result is not None
    assert set(result.keys()) == {"ids", "y1", "y0", "D", "X", "n", "groups_raw", "weights"}
    assert result["n"] == 30
    assert result["X"].shape == (30, 1)
    np.testing.assert_array_equal(result["D"], (pdf["group"].values == 3).astype(np.float64))


def test_build_did_partition_arrays_with_covariates():
    pdf = _make_did_merged_pdf(n=20, g=2, covariate_cols=["cov1", "cov2"])
    result = _build_did_partition_arrays(pdf, "id", "y", "group", 2, ["cov1", "cov2"])
    assert result["X"].shape == (20, 3)


def test_build_did_partition_arrays_with_weights():
    pdf = _make_did_merged_pdf(n=20, g=2, weightsname="w")
    result = _build_did_partition_arrays(pdf, "id", "y", "group", 2, None, weightsname="w")
    assert not np.all(result["weights"] == 1.0)


def test_build_did_partition_arrays_empty():
    pdf = pd.DataFrame({"id": [], "y": [], "_y_pre": [], "group": []})
    result = _build_did_partition_arrays(pdf, "id", "y", "group", 3, None)
    assert result is None


def test_build_did_partition_arrays_wide_basic():
    rng = np.random.default_rng(10)
    n = 20
    pdf = pd.DataFrame(
        {
            "id": np.arange(n),
            "group": rng.choice([0, 3], size=n),
            "_y_2": rng.standard_normal(n),
            "_y_1": rng.standard_normal(n),
        }
    )
    result = _build_did_partition_arrays_wide(pdf, "id", "group", 3, None, "_y_2", "_y_1")
    assert result is not None
    assert result["n"] == n
    assert result["X"].shape == (n, 1)
    np.testing.assert_array_equal(result["y1"], pdf["_y_2"].values)
    np.testing.assert_array_equal(result["y0"], pdf["_y_1"].values)


def test_build_did_partition_arrays_wide_with_covariates():
    rng = np.random.default_rng(11)
    n = 15
    pdf = pd.DataFrame(
        {
            "id": np.arange(n),
            "group": rng.choice([0, 3], size=n),
            "_y_2": rng.standard_normal(n),
            "_y_1": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
        }
    )
    result = _build_did_partition_arrays_wide(pdf, "id", "group", 3, ["x1"], "_y_2", "_y_1")
    assert result["X"].shape == (n, 2)


def test_build_did_partition_arrays_wide_empty():
    pdf = pd.DataFrame({"id": [], "group": [], "_y_2": [], "_y_1": []})
    result = _build_did_partition_arrays_wide(pdf, "id", "group", 3, None, "_y_2", "_y_1")
    assert result is None


def test_build_did_rc_partition_arrays_basic():
    pdf = _make_did_rc_pdf(n=30, g=3)
    result = _build_did_rc_partition_arrays(pdf, 0, "y", "group", 3, None)
    assert result is not None
    assert set(result.keys()) == {"ids", "y", "post", "D", "X", "n", "weights"}
    assert result["n"] == 30
    np.testing.assert_array_equal(result["ids"], np.arange(30, dtype=np.int64))


def test_build_did_rc_partition_arrays_offset():
    pdf = _make_did_rc_pdf(n=10, g=3)
    result = _build_did_rc_partition_arrays(pdf, 100, "y", "group", 3, None)
    np.testing.assert_array_equal(result["ids"], np.arange(100, 110, dtype=np.int64))


def test_build_did_rc_partition_arrays_with_covariates():
    pdf = _make_did_rc_pdf(n=20, g=2, covariate_cols=["c1"])
    result = _build_did_rc_partition_arrays(pdf, 0, "y", "group", 2, ["c1"])
    assert result["X"].shape == (20, 2)


def test_build_did_rc_partition_arrays_empty():
    pdf = pd.DataFrame({"y": [], "_post": [], "group": []})
    result = _build_did_rc_partition_arrays(pdf, 0, "y", "group", 3, None)
    assert result is None


def test_partition_did_pscore_gram_none():
    assert _partition_did_pscore_gram(None, np.zeros(1)) is None


def test_partition_did_pscore_gram_valid(did_partition):
    k = did_partition["X"].shape[1]
    result = _partition_did_pscore_gram(did_partition, np.zeros(k))
    assert result is not None
    XtWX, XtWz, n = result
    assert XtWX.shape == (k, k)
    assert XtWz.shape == (k,)
    assert n == did_partition["n"]


def test_partition_did_or_gram_none():
    assert _partition_did_or_gram(None) is None


def test_partition_did_or_gram_no_controls(did_partition):
    part = {**did_partition, "D": np.ones(did_partition["n"])}
    result = _partition_did_or_gram(part)
    assert result is None


def test_partition_did_or_gram_valid(did_partition):
    k = did_partition["X"].shape[1]
    result = _partition_did_or_gram(did_partition)
    assert result is not None
    XtWX, XtWy, n = result
    assert XtWX.shape == (k, k)
    assert XtWy.shape == (k,)
    assert n > 0


def test_partition_did_rc_pscore_gram_none():
    assert _partition_did_rc_pscore_gram(None, np.zeros(1)) is None


def test_partition_did_rc_pscore_gram_valid(did_rc_partition):
    k = did_rc_partition["X"].shape[1]
    result = _partition_did_rc_pscore_gram(did_rc_partition, np.zeros(k))
    assert result is not None
    XtWX, XtWz, n = result
    assert XtWX.shape == (k, k)
    assert XtWz.shape == (k,)
    assert n == did_rc_partition["n"]


@pytest.mark.parametrize("d_val,post_val", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_partition_did_rc_or_gram_none(d_val, post_val):
    assert _partition_did_rc_or_gram(None, d_val, post_val) is None


def test_partition_did_rc_or_gram_valid(did_rc_partition):
    result = _partition_did_rc_or_gram(did_rc_partition, 0, 0)
    if result is not None:
        XtWX, XtWy, n = result
        k = did_rc_partition["X"].shape[1]
        assert XtWX.shape == (k, k)
        assert XtWy.shape == (k,)
        assert n > 0


def test_partition_did_rc_or_gram_no_match(did_rc_partition):
    part = {**did_rc_partition, "D": np.ones(did_rc_partition["n"]), "post": np.ones(did_rc_partition["n"])}
    result = _partition_did_rc_or_gram(part, 0, 0)
    assert result is None


@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_partition_did_global_stats_keys(did_partition, est_method):
    k = did_partition["X"].shape[1]
    result = _partition_did_global_stats(did_partition, np.zeros(k), np.zeros(k), est_method, 0.995)
    assert result is not None
    expected_keys = {
        "sum_w_treat",
        "sum_w_control",
        "sum_riesz_treat",
        "sum_riesz_control",
        "n_sub",
        "sum_wt_X",
        "sum_wc_X",
        "sum_wc_dy_or_X",
        "or_xpx",
        "info_gram",
    }
    assert expected_keys.issubset(set(result.keys()))
    assert result["n_sub"] == did_partition["n"]


def test_partition_did_global_stats_none():
    result = _partition_did_global_stats(None, np.zeros(1), np.zeros(1), "reg", 0.995)
    assert result is None


@pytest.mark.parametrize("est_method", ["dr", "reg"])
def test_partition_did_rc_global_stats_keys(did_rc_partition, est_method):
    k = did_rc_partition["X"].shape[1]
    or_betas = {key: np.zeros(k) for key in ["cont_pre", "cont_post", "treat_pre", "treat_post"]}
    result = _partition_did_rc_global_stats(did_rc_partition, np.zeros(k), or_betas, est_method, 0.995)
    assert result is not None
    assert result["n_sub"] == did_rc_partition["n"]


def test_partition_did_rc_global_stats_none():
    or_betas = {key: np.zeros(1) for key in ["cont_pre", "cont_post", "treat_pre", "treat_post"]}
    result = _partition_did_rc_global_stats(None, np.zeros(1), or_betas, "reg", 0.995)
    assert result is None


def test_partition_compute_did_if_none():
    ids, if_arr = _partition_compute_did_if(
        None,
        np.zeros(1),
        np.zeros(1),
        {},
        "reg",
        0.995,
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
    )
    assert len(ids) == 0
    assert len(if_arr) == 0


@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_partition_compute_did_if_valid(did_partition, est_method):
    k = did_partition["X"].shape[1]
    ps_beta = np.zeros(k)
    or_beta = np.zeros(k)
    global_agg = {
        "mean_w_treat": 1.0,
        "mean_w_control": 1.0,
        "att_treat": 0.5,
        "att_control": 0.3,
    }
    ids, if_arr = _partition_compute_did_if(
        did_partition,
        ps_beta,
        or_beta,
        global_agg,
        est_method,
        0.995,
        np.zeros(k),
        np.zeros(k),
        np.zeros(k),
    )
    assert len(ids) == did_partition["n"]
    assert len(if_arr) == did_partition["n"]
    assert np.all(np.isfinite(if_arr))
