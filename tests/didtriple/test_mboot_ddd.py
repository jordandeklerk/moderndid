"""Tests for the DDD multiplier bootstrap."""

import numpy as np
import pytest

from moderndid import mboot_ddd, wboot_ddd


def test_mboot_ddd_basic():
    rng = np.random.default_rng(42)
    inf_func = rng.standard_normal(100)

    result = mboot_ddd(inf_func, biters=10, random_state=42)

    assert result.bres.shape == (10, 1)
    assert len(result.se) == 1
    assert np.isfinite(result.se[0])
    assert np.isfinite(result.crit_val)


def test_mboot_ddd_reproducibility():
    rng = np.random.default_rng(42)
    inf_func = rng.standard_normal(100)

    result1 = mboot_ddd(inf_func, biters=10, random_state=123)
    result2 = mboot_ddd(inf_func, biters=10, random_state=123)

    assert np.allclose(result1.bres, result2.bres)
    assert np.allclose(result1.se, result2.se)
    assert result1.crit_val == result2.crit_val


def test_mboot_ddd_different_seeds():
    rng = np.random.default_rng(42)
    inf_func = rng.standard_normal(100)

    result1 = mboot_ddd(inf_func, biters=10, random_state=123)
    result2 = mboot_ddd(inf_func, biters=10, random_state=456)

    assert not np.allclose(result1.bres, result2.bres)


def test_wboot_ddd_basic(ddd_data_no_covariates):
    ddd_data, covariates = ddd_data_no_covariates

    boots = wboot_ddd(
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
        covariates=covariates,
        i_weights=np.ones(len(ddd_data.y1)),
        est_method="reg",
        biters=3,
        random_state=42,
    )

    assert boots.shape == (3,)
    valid_boots = boots[~np.isnan(boots)]
    assert len(valid_boots) > 0


def test_wboot_ddd_reproducibility(ddd_data_no_covariates):
    ddd_data, covariates = ddd_data_no_covariates

    boots1 = wboot_ddd(
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
        covariates=covariates,
        i_weights=np.ones(len(ddd_data.y1)),
        est_method="reg",
        biters=3,
        random_state=123,
    )

    boots2 = wboot_ddd(
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
        covariates=covariates,
        i_weights=np.ones(len(ddd_data.y1)),
        est_method="reg",
        biters=3,
        random_state=123,
    )

    assert np.allclose(boots1, boots2, equal_nan=True)


def test_mboot_ddd_2d_influence_function():
    rng = np.random.default_rng(42)
    inf_func = rng.standard_normal((100, 5))

    result = mboot_ddd(inf_func, biters=20, random_state=42)

    assert result.bres.shape == (20, 5)
    assert len(result.se) == 5
    assert all(np.isfinite(se) for se in result.se)
    assert np.isfinite(result.crit_val)


def test_mboot_ddd_clustered():
    rng = np.random.default_rng(42)
    inf_func = rng.standard_normal(100)
    cluster = np.repeat(np.arange(20), 5)

    result = mboot_ddd(inf_func, biters=20, cluster=cluster, random_state=42)

    assert result.bres.shape == (20, 1)
    assert np.isfinite(result.se[0])
    assert np.isfinite(result.crit_val)


def test_mboot_ddd_clustered_2d():
    rng = np.random.default_rng(42)
    inf_func = rng.standard_normal((100, 3))
    cluster = np.repeat(np.arange(20), 5)

    result = mboot_ddd(inf_func, biters=20, cluster=cluster, random_state=42)

    assert result.bres.shape == (20, 3)
    assert len(result.se) == 3
    assert all(np.isfinite(se) for se in result.se)


def test_mboot_ddd_clustered_vs_unclustered():
    rng = np.random.default_rng(42)
    inf_func = rng.standard_normal(100)
    cluster = np.repeat(np.arange(20), 5)

    result_unclustered = mboot_ddd(inf_func, biters=50, random_state=42)
    result_clustered = mboot_ddd(inf_func, biters=50, cluster=cluster, random_state=42)

    assert result_unclustered.se[0] != result_clustered.se[0]


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10, 0.20])
def test_mboot_ddd_alpha_levels(alpha):
    rng = np.random.default_rng(42)
    inf_func = rng.standard_normal(100)

    result = mboot_ddd(inf_func, biters=50, alpha=alpha, random_state=42)

    assert np.isfinite(result.crit_val)
    assert result.crit_val > 0


def test_mboot_ddd_alpha_ordering():
    rng = np.random.default_rng(42)
    inf_func = rng.standard_normal(100)

    result_01 = mboot_ddd(inf_func, biters=100, alpha=0.01, random_state=42)
    result_05 = mboot_ddd(inf_func, biters=100, alpha=0.05, random_state=42)
    result_10 = mboot_ddd(inf_func, biters=100, alpha=0.10, random_state=42)

    assert result_01.crit_val > result_05.crit_val > result_10.crit_val


def test_mboot_ddd_larger_biters():
    rng = np.random.default_rng(42)
    inf_func = rng.standard_normal(200)

    result = mboot_ddd(inf_func, biters=500, random_state=42)

    assert result.bres.shape == (500, 1)
    assert np.isfinite(result.se[0])


@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_wboot_ddd_all_methods(ddd_data_no_covariates, est_method):
    ddd_data, covariates = ddd_data_no_covariates

    boots = wboot_ddd(
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
        covariates=covariates,
        i_weights=np.ones(len(ddd_data.y1)),
        est_method=est_method,
        biters=5,
        random_state=42,
    )

    assert boots.shape == (5,)
    valid_boots = boots[~np.isnan(boots)]
    assert len(valid_boots) > 0


def test_wboot_ddd_with_weights(ddd_data_no_covariates):
    ddd_data, covariates = ddd_data_no_covariates
    rng = np.random.default_rng(42)
    weights = rng.uniform(0.5, 2.0, len(ddd_data.y1))

    boots = wboot_ddd(
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
        covariates=covariates,
        i_weights=weights,
        est_method="reg",
        biters=5,
        random_state=42,
    )

    assert boots.shape == (5,)
    valid_boots = boots[~np.isnan(boots)]
    assert len(valid_boots) > 0


def test_wboot_ddd_with_covariates(ddd_data_with_covariates):
    ddd_data, covariates = ddd_data_with_covariates

    boots = wboot_ddd(
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
        covariates=covariates,
        i_weights=np.ones(len(ddd_data.y1)),
        est_method="dr",
        biters=5,
        random_state=42,
    )

    assert boots.shape == (5,)
    valid_boots = boots[~np.isnan(boots)]
    assert len(valid_boots) > 0


def test_wboot_ddd_different_seeds(ddd_data_no_covariates):
    ddd_data, covariates = ddd_data_no_covariates

    boots1 = wboot_ddd(
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
        covariates=covariates,
        i_weights=np.ones(len(ddd_data.y1)),
        est_method="reg",
        biters=5,
        random_state=123,
    )

    boots2 = wboot_ddd(
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
        covariates=covariates,
        i_weights=np.ones(len(ddd_data.y1)),
        est_method="reg",
        biters=5,
        random_state=456,
    )

    assert not np.allclose(boots1, boots2, equal_nan=True)
