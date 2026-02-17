"""Shared fixtures for Dask backend tests."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_design_matrix(rng):
    X = np.column_stack([np.ones(20), rng.standard_normal(20)])
    return X


@pytest.fixture
def small_weights():
    return np.ones(20)


@pytest.fixture
def small_binary_response(rng):
    return rng.integers(0, 2, size=20).astype(np.float64)


@pytest.fixture
def small_continuous_response(rng):
    return rng.standard_normal(20)


@pytest.fixture
def gram_tuple_pair(rng):
    k = 2
    a = (rng.standard_normal((k, k)), rng.standard_normal(k), 10)
    b = (rng.standard_normal((k, k)), rng.standard_normal(k), 15)
    return a, b


@pytest.fixture
def ddd_subgroup_arrays(rng):
    n = 200
    subgroup = rng.choice([1, 2, 3, 4], size=n)
    y1 = rng.standard_normal(n)
    y0 = rng.standard_normal(n)
    covariates = np.column_stack([np.ones(n), rng.standard_normal(n)])
    weights = np.ones(n)
    return {
        "subgroup": subgroup,
        "y1": y1,
        "y0": y0,
        "covariates": covariates,
        "weights": weights,
        "n": n,
    }


@pytest.fixture
def partition_dict(rng):
    n = 50
    ids = np.arange(n)
    y1 = rng.standard_normal(n)
    y0 = rng.standard_normal(n)
    subgroup = np.array([4] * 15 + [3] * 10 + [2] * 10 + [1] * 15)
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    groups_raw = np.array([3] * 15 + [3] * 10 + [0] * 10 + [0] * 15)
    parts_raw = np.array([1] * 15 + [0] * 10 + [1] * 10 + [0] * 15)
    return {
        "ids": ids,
        "y1": y1,
        "y0": y0,
        "subgroup": subgroup,
        "X": X,
        "n": n,
        "groups_raw": groups_raw,
        "parts_raw": parts_raw,
    }
