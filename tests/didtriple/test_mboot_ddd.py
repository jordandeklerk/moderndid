"""Tests for the DDD multiplier bootstrap."""

from importlib import import_module

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


def test_mboot_ddd_clustered_matches_presummed_clusters():
    rng = np.random.default_rng(0)
    cluster = np.repeat(np.arange(40), np.arange(1, 41))
    inf_func = rng.standard_normal(len(cluster))
    sums = np.bincount(cluster, weights=inf_func)

    clustered = mboot_ddd(inf_func, biters=50, cluster=cluster, random_state=3)
    presummed = mboot_ddd(sums, biters=50, cluster=np.arange(40), random_state=3)

    np.testing.assert_array_equal(clustered.bres, presummed.bres)
    np.testing.assert_allclose(clustered.se, presummed.se * 40 / len(inf_func), rtol=1e-12)


def test_mboot_ddd_cluster_length_mismatch():
    with pytest.raises(ValueError, match="cluster has 5 entries but inf_func has 10 rows"):
        mboot_ddd(np.ones(10), biters=5, cluster=np.arange(5))


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


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_mboot_ddd_zero_scale_column_is_ignored_in_every_draw(monkeypatch, zero_scale_draws):
    monkeypatch.setattr(
        import_module("moderndid.didtriple.bootstrap.mboot_ddd"),
        "multiplier_bootstrap",
        lambda *args, **kwargs: zero_scale_draws,
    )

    result = mboot_ddd(np.zeros((1, 2)), biters=21)

    np.testing.assert_allclose(result.crit_val, 10 / (10 / 1.3489795), rtol=1e-12)
    np.testing.assert_allclose(result.se[0], 10 / 1.3489795, rtol=1e-12)
    assert np.isnan(result.se[1])


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_mboot_ddd_negligible_scale_column_is_ignored_in_every_draw(monkeypatch, zero_scale_draws):
    first = zero_scale_draws[:, 0]
    draws = np.column_stack([first, np.where(first == -10, 1.0, 1e-9 * first)])
    monkeypatch.setattr(
        import_module("moderndid.didtriple.bootstrap.mboot_ddd"), "multiplier_bootstrap", lambda *args, **kwargs: draws
    )

    result = mboot_ddd(np.zeros((1, 2)), biters=21)

    np.testing.assert_allclose(result.crit_val, 10 / (10 / 1.3489795), rtol=1e-12)
    assert np.isnan(result.se[1])


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_mboot_ddd_critical_value_is_nan_when_every_column_is_ignored(monkeypatch):
    draws = np.ones((21, 2))
    monkeypatch.setattr(
        import_module("moderndid.didtriple.bootstrap.mboot_ddd"), "multiplier_bootstrap", lambda *args, **kwargs: draws
    )

    result = mboot_ddd(np.zeros((1, 2)), biters=21)

    assert np.isnan(result.crit_val)
    assert np.all(np.isnan(result.se))


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_mboot_ddd_zero_scale_column_critical_value_from_bootstrap_draws(
    monkeypatch, draws_from_weights, offsetting_spike_weights, inf_func_with_zero_scale_column
):
    monkeypatch.setattr(
        import_module("moderndid.didtriple.bootstrap.mboot_ddd"),
        "multiplier_bootstrap",
        draws_from_weights(offsetting_spike_weights(200, 0, 1)),
    )

    result = mboot_ddd(inf_func_with_zero_scale_column, biters=999)

    bres = result.bres
    quartiles = np.percentile(bres, [75, 25], axis=0, method="inverted_cdf")
    scale = (quartiles[0] - quartiles[1]) / 1.3489795
    largest = np.max(np.abs(bres[:, [0, 2]]) / scale[[0, 2]], axis=1)

    assert scale[1] == 0
    np.testing.assert_allclose(result.crit_val, np.percentile(largest, 95, method="inverted_cdf"), rtol=1e-12)
    assert np.isnan(result.se[1])


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_mboot_ddd_critical_value_without_ignored_columns_is_quantile_of_largest_standardized_draw():
    inf_func = np.random.default_rng(3).standard_normal((150, 4))

    result = mboot_ddd(inf_func, biters=499, random_state=5)

    bres = result.bres
    quartiles = np.percentile(bres, [75, 25], axis=0, method="inverted_cdf")
    scale = (quartiles[0] - quartiles[1]) / 1.3489795
    expected = np.percentile(np.max(np.abs(bres / scale), axis=1), 95, method="inverted_cdf")

    assert result.crit_val == expected


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
