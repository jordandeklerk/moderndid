"""Tests for the multiplier bootstrap function."""

import numpy as np
import pytest

from pydid.did import mboot


def test_basic_functionality():
    n = 200
    inf_func = np.random.normal(0, 1, n)
    result = mboot(inf_func, n_units=n, biters=99)

    assert isinstance(result, dict)
    assert "bres" in result
    assert "V" in result
    assert "se" in result
    assert "crit_val" in result

    assert result["bres"].shape[0] == 99
    assert len(result["se"]) == 1
    assert isinstance(result["crit_val"], float | np.floating)


def test_multivariate_influence_function():
    n = 200
    k = 3
    inf_func = np.random.normal(0, 1, (n, k))
    result = mboot(inf_func, n_units=n, biters=99)

    assert result["bres"].shape == (99, k)
    assert result["V"].shape == (k, k)
    assert result["se"].shape == (k,)


def test_clustering():
    n_units = 100
    n_clusters = 20

    cluster = np.repeat(np.arange(n_clusters), n_units // n_clusters)
    inf_func = np.random.normal(0, 1, n_units)

    result = mboot(inf_func, n_units=n_units, biters=99, cluster=cluster)

    assert result["bres"].shape == (99, 1)
    assert result["se"].shape == (1,)


def test_degenerate_columns():
    n = 200
    inf_func = np.column_stack([np.random.normal(0, 1, n), np.zeros(n), np.random.normal(0, 1, n)])

    result = mboot(inf_func, n_units=n, biters=99)

    assert not np.isnan(result["se"][0])
    assert np.isnan(result["se"][1])
    assert not np.isnan(result["se"][2])


def test_reproducibility():
    n = 100
    inf_func = np.random.normal(0, 1, n)

    result1 = mboot(inf_func, n_units=n, biters=99, random_state=42)
    result2 = mboot(inf_func, n_units=n, biters=99, random_state=42)

    np.testing.assert_array_equal(result1["bres"], result2["bres"])
    assert result1["se"] == result2["se"]
    assert result1["crit_val"] == result2["crit_val"]


def test_significance_level():
    n = 200
    inf_func = np.random.normal(0, 1, n)

    result_05 = mboot(inf_func, n_units=n, biters=199, alp=0.05)
    result_10 = mboot(inf_func, n_units=n, biters=199, alp=0.10, random_state=42)

    assert result_10["crit_val"] < result_05["crit_val"]


def test_invalid_inputs():
    with pytest.raises(ValueError, match="cluster must have length"):
        mboot(np.random.normal(0, 1, 100), n_units=100, cluster=np.arange(50))


def test_edge_cases():
    n = 10
    inf_func = np.random.normal(0, 1, n)
    result = mboot(inf_func, n_units=n, biters=50)

    assert result["bres"].shape == (50, 1)
    assert result["se"].shape == (1,)

    cluster = np.arange(n)
    result = mboot(inf_func, n_units=n, biters=50, cluster=cluster)

    assert result["bres"].shape == (50, 1)
