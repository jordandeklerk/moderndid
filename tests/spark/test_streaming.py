"""Tests for distributed streaming cell computation."""

import numpy as np
import pandas as pd
import pytest

from moderndid.spark._ddd_streaming import (
    _build_partition_arrays,
    _filter_partition_for_ctrl,
    _partition_global_stats,
    _partition_or_gram,
    _partition_pscore_gram,
)
from moderndid.spark._utils import sum_global_stats


def _make_merged_pdf(n=40, g=3, covariate_cols=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    groups = rng.choice([0, g], size=n)
    parts = rng.choice([0, 1], size=n)
    pdf = pd.DataFrame(
        {
            "id": np.arange(n),
            "y": rng.standard_normal(n),
            "_y_pre": rng.standard_normal(n),
            "group": groups,
            "partition": parts,
        }
    )
    if covariate_cols:
        for col in covariate_cols:
            pdf[col] = rng.standard_normal(n)
    return pdf


def test_build_partition_arrays_basic():
    pdf = _make_merged_pdf(n=30, g=3)
    result = _build_partition_arrays(pdf, "id", "y", "group", "partition", 3, None)
    assert result is not None
    assert set(result.keys()) == {"ids", "y1", "y0", "subgroup", "X", "n", "groups_raw", "parts_raw", "weights"}
    assert result["n"] == 30
    assert result["X"].shape == (30, 1)


def test_build_partition_arrays_with_covariates():
    pdf = _make_merged_pdf(n=20, g=2, covariate_cols=["cov1", "cov2"])
    result = _build_partition_arrays(pdf, "id", "y", "group", "partition", 2, ["cov1", "cov2"])
    assert result["X"].shape == (20, 3)


def test_build_partition_arrays_empty():
    pdf = pd.DataFrame({"id": [], "y": [], "_y_pre": [], "group": [], "partition": []})
    result = _build_partition_arrays(pdf, "id", "y", "group", "partition", 3, None)
    assert result is None


def test_build_partition_arrays_subgroup_encoding():
    pdf = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "y": [1.0, 2.0, 3.0, 4.0],
            "_y_pre": [0.5, 1.5, 2.5, 3.5],
            "group": [3, 3, 0, 0],
            "partition": [1, 0, 1, 0],
        }
    )
    result = _build_partition_arrays(pdf, "id", "y", "group", "partition", 3, None)
    expected_sg = np.array([4, 3, 2, 1])
    np.testing.assert_array_equal(result["subgroup"], expected_sg)


def test_filter_partition_for_ctrl_basic(partition_dict):
    filtered = _filter_partition_for_ctrl(partition_dict, g=3, ctrl=0)
    assert filtered is not None
    assert filtered["n"] > 0
    unique_groups = set(filtered["groups_raw"])
    assert unique_groups.issubset({3, 0})


def test_filter_partition_for_ctrl_none():
    result = _filter_partition_for_ctrl(None, g=3, ctrl=0)
    assert result is None


def test_filter_partition_for_ctrl_no_match(partition_dict):
    filtered = _filter_partition_for_ctrl(partition_dict, g=3, ctrl=99)
    if filtered is not None:
        assert all(gv == 3 for gv in filtered["groups_raw"])


def test_partition_pscore_gram_none():
    result = _partition_pscore_gram(None, comp_sg=3, beta=np.zeros(1))
    assert result is None


def test_partition_pscore_gram_no_subgroups(partition_dict):
    part = partition_dict.copy()
    part["subgroup"] = np.ones(part["n"], dtype=int)
    result = _partition_pscore_gram(part, comp_sg=3, beta=np.zeros(part["X"].shape[1]))
    assert result is None


def test_partition_pscore_gram_valid(partition_dict):
    k = partition_dict["X"].shape[1]
    beta = np.zeros(k)
    result = _partition_pscore_gram(partition_dict, comp_sg=3, beta=beta)
    assert result is not None
    XtWX, XtWz, n = result
    assert XtWX.shape == (k, k)
    assert XtWz.shape == (k,)
    assert n > 0


def test_partition_or_gram_none():
    result = _partition_or_gram(None, comp_sg=1)
    assert result is None


def test_partition_or_gram_no_control(partition_dict):
    part = partition_dict.copy()
    part["subgroup"] = np.full(part["n"], 4)
    result = _partition_or_gram(part, comp_sg=1)
    assert result is None


def test_partition_or_gram_valid(partition_dict):
    k = partition_dict["X"].shape[1]
    result = _partition_or_gram(partition_dict, comp_sg=1)
    assert result is not None
    XtWX, XtWy, n = result
    assert XtWX.shape == (k, k)
    assert XtWy.shape == (k,)
    assert n > 0


def _make_stats_dict(rng, k=2):
    return {
        "sum_w_treat": rng.random(),
        "sum_w_control": rng.random(),
        "sum_riesz_treat": rng.random(),
        "sum_riesz_control": rng.random(),
        "n_sub": rng.integers(5, 50),
        "sum_wt_X": rng.standard_normal(k),
        "sum_wc_X": rng.standard_normal(k),
        "sum_wc_dy_or_X": rng.standard_normal(k),
        "sum_wc_att_part": rng.random(),
        "sum_or_x_X": rng.random(),
        "or_xpx": rng.standard_normal((k, k)),
        "sum_or_ex": rng.standard_normal(k),
        "info_gram": rng.standard_normal((k, k)),
        "sum_score_ps": None,
    }


@pytest.mark.parametrize(
    "use_a,use_b",
    [(False, True), (True, False), (False, False)],
    ids=["none_a", "none_b", "both_none"],
)
def testsum_global_stats_none_handling(use_a, use_b, rng):
    a = _make_stats_dict(rng) if use_a else None
    b = _make_stats_dict(rng) if use_b else None
    result = sum_global_stats(a, b)
    if not use_a and not use_b:
        assert result is None
    elif not use_a:
        assert result is b
    else:
        assert result is a


def testsum_global_stats_sum(rng):
    a = _make_stats_dict(rng)
    b = _make_stats_dict(rng)
    result = sum_global_stats(a, b)
    np.testing.assert_allclose(result["sum_w_treat"], a["sum_w_treat"] + b["sum_w_treat"])
    np.testing.assert_allclose(result["or_xpx"], a["or_xpx"] + b["or_xpx"])
    assert result["n_sub"] == a["n_sub"] + b["n_sub"]


@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_partition_global_stats_keys(partition_dict, est_method):
    k = partition_dict["X"].shape[1]
    ps_beta = np.zeros(k)
    or_beta = np.zeros(k)
    result = _partition_global_stats(
        partition_dict, comp_sg=3, ps_beta=ps_beta, or_beta=or_beta, est_method=est_method, trim_level=0.995
    )
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
        "sum_wc_att_part",
        "sum_or_x_X",
        "or_xpx",
        "sum_or_ex",
        "info_gram",
    }
    assert expected_keys.issubset(set(result.keys()))


def test_partition_global_stats_none():
    result = _partition_global_stats(
        None, comp_sg=3, ps_beta=np.zeros(1), or_beta=np.zeros(1), est_method="reg", trim_level=0.995
    )
    assert result is None
