"""Tests for the DDD multiplier bootstrap."""

import numpy as np

from moderndid import mboot_ddd, wboot_ddd


def test_mboot_ddd_basic():
    rng = np.random.default_rng(42)
    inf_func = rng.standard_normal(100)

    boots = mboot_ddd(inf_func, nboot=10, random_state=42)

    assert boots.shape == (10,)
    assert np.all(np.isfinite(boots))


def test_mboot_ddd_reproducibility():
    rng = np.random.default_rng(42)
    inf_func = rng.standard_normal(100)

    boots1 = mboot_ddd(inf_func, nboot=10, random_state=123)
    boots2 = mboot_ddd(inf_func, nboot=10, random_state=123)

    assert np.allclose(boots1, boots2)


def test_mboot_ddd_different_seeds():
    rng = np.random.default_rng(42)
    inf_func = rng.standard_normal(100)

    boots1 = mboot_ddd(inf_func, nboot=10, random_state=123)
    boots2 = mboot_ddd(inf_func, nboot=10, random_state=456)

    assert not np.allclose(boots1, boots2)


def test_wboot_ddd_basic(ddd_data_no_covariates):
    ddd_data, covariates = ddd_data_no_covariates

    boots = wboot_ddd(
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
        covariates=covariates,
        i_weights=np.ones(len(ddd_data.y1)),
        est_method="reg",
        nboot=3,
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
        nboot=3,
        random_state=123,
    )

    boots2 = wboot_ddd(
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
        covariates=covariates,
        i_weights=np.ones(len(ddd_data.y1)),
        est_method="reg",
        nboot=3,
        random_state=123,
    )

    assert np.allclose(boots1, boots2, equal_nan=True)
