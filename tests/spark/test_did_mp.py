"""Tests for distributed DiD multi-period estimation."""

import numpy as np
import pytest

from moderndid.spark._did_mp import _compute_wald_pretest


def test_compute_wald_pretest_valid(rng):
    groups = np.array([5, 5, 5, 3, 3])
    times = np.array([1, 2, 3, 4, 5])
    att = rng.standard_normal(5) * 0.1
    se = np.abs(rng.standard_normal(5)) + 0.01
    vcov = np.eye(5) * 0.01
    stat, pval = _compute_wald_pretest(att, groups, times, vcov, se, n_units=100)
    assert isinstance(stat, float)
    assert isinstance(pval, float)
    assert stat >= 0
    assert 0 <= pval <= 1


def test_compute_wald_pretest_no_pretreatment():
    groups = np.array([1, 2, 3])
    times = np.array([5, 5, 5])
    att = np.array([0.1, 0.2, 0.3])
    se = np.array([0.01, 0.02, 0.03])
    vcov = np.eye(3)
    with pytest.warns(UserWarning, match="No pre-treatment"):
        stat, pval = _compute_wald_pretest(att, groups, times, vcov, se, n_units=100)
    assert stat is None
    assert pval is None


def test_compute_wald_pretest_all_nan_se():
    groups = np.array([5, 5])
    times = np.array([1, 2])
    att = np.array([0.1, 0.2])
    se = np.array([np.nan, np.nan])
    vcov = np.eye(2)
    with pytest.warns(UserWarning, match="No pre-treatment"):
        stat, pval = _compute_wald_pretest(att, groups, times, vcov, se, n_units=100)
    assert stat is None
    assert pval is None


def test_compute_wald_pretest_nan_vcov():
    groups = np.array([5, 5])
    times = np.array([1, 2])
    att = np.array([0.1, 0.2])
    se = np.array([0.01, 0.02])
    vcov = np.array([[0.01, np.nan], [np.nan, 0.01]])
    with pytest.warns(UserWarning, match="NA pre-treatment"):
        stat, pval = _compute_wald_pretest(att, groups, times, vcov, se, n_units=100)
    assert stat is None
    assert pval is None


def test_compute_wald_pretest_singular_vcov():
    groups = np.array([5, 5])
    times = np.array([1, 2])
    att = np.array([0.1, 0.2])
    se = np.array([0.01, 0.02])
    vcov = np.zeros((2, 2))
    with pytest.warns(UserWarning, match="singular covariance"):
        stat, pval = _compute_wald_pretest(att, groups, times, vcov, se, n_units=100)
    assert stat is None
    assert pval is None


def test_compute_wald_pretest_linalg_error(monkeypatch):
    groups = np.array([5, 5])
    times = np.array([1, 2])
    att = np.array([0.1, 0.2])
    se = np.array([0.01, 0.02])
    vcov = np.eye(2) * 0.01

    def _raise(*args, **kwargs):
        raise np.linalg.LinAlgError("mocked")

    monkeypatch.setattr(np.linalg, "solve", _raise)
    with pytest.warns(UserWarning, match="numerical issues"):
        stat, pval = _compute_wald_pretest(att, groups, times, vcov, se, n_units=100)
    assert stat is None
    assert pval is None
