"""Shared fixtures for distributed partition tests."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def basic_partition(rng):
    """Balanced panel with 3 units x 5 periods: two switchers and one never-switcher.

    Unit 1: switches at t=3 (S_g=1, F_g=3, L_g=3)
    Unit 2: switches at t=4 (S_g=1, F_g=4, L_g=2)
    Unit 3: never switches  (S_g=0, F_g=inf, L_g=inf)
    """
    n_units = 3
    n_periods = 5
    n = n_units * n_periods

    gname = np.repeat([1.0, 2.0, 3.0], n_periods)
    tname = np.tile(np.arange(1.0, n_periods + 1), n_units)

    d = np.zeros(n)
    d[(gname == 1) & (tname >= 3)] = 1.0
    d[(gname == 2) & (tname >= 4)] = 1.0

    y = np.array(
        [
            1.0,
            2.0,
            5.0,
            6.0,
            7.0,  # unit 1: jump at switch
            1.5,
            2.5,
            3.5,
            7.0,
            8.0,  # unit 2: jump at switch
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,  # unit 3: flat (never-switcher)
        ]
    )

    F_g = np.where(gname == 1, 3.0, np.where(gname == 2, 4.0, np.inf))
    L_g = np.where(gname == 1, 3.0, np.where(gname == 2, 2.0, np.inf))
    S_g = np.where(gname == 3, 0.0, 1.0)
    d_sq = np.zeros(n)
    T_g = np.full(n, 5.0)
    weight_gt = np.ones(n)

    first_obs = np.zeros(n)
    for _i, g in enumerate([1.0, 2.0, 3.0]):
        idx = np.where(gname == g)[0][0]
        first_obs[idx] = 1.0

    return {
        "gname": gname,
        "tname": tname,
        "y": y,
        "d": d,
        "F_g": F_g,
        "L_g": L_g,
        "S_g": S_g,
        "d_sq": d_sq,
        "weight_gt": weight_gt,
        "first_obs_by_gp": first_obs,
        "T_g": T_g,
        "n_rows": n,
    }


@pytest.fixture
def partition_with_covariates(basic_partition, rng):
    """Basic partition augmented with two covariates."""
    n = basic_partition["n_rows"]
    basic_partition["x1"] = rng.standard_normal(n)
    basic_partition["x2"] = np.linspace(0, 1, n)
    return basic_partition


@pytest.fixture
def partition_with_clusters(basic_partition):
    """Basic partition augmented with cluster identifiers."""
    gname = basic_partition["gname"]
    basic_partition["cluster"] = np.where(gname <= 2, "A", "B")
    return basic_partition


@pytest.fixture
def bidirectional_partition(rng):
    """Panel with both in- and out-switchers plus a never-switcher.

    Unit 1: switches in at t=3 (S_g=1)
    Unit 2: switches out at t=4 (S_g=-1, d goes 1->0)
    Unit 3: never switches (S_g=0)
    """
    n_periods = 5
    n = 3 * n_periods

    gname = np.repeat([1.0, 2.0, 3.0], n_periods)
    tname = np.tile(np.arange(1.0, n_periods + 1), 3)

    d = np.zeros(n)
    d[(gname == 1) & (tname >= 3)] = 1.0
    d[(gname == 2) & (tname < 4)] = 1.0

    y = np.array(
        [
            1.0,
            2.0,
            5.0,
            6.0,
            7.0,
            5.0,
            5.5,
            6.0,
            3.0,
            3.5,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
        ]
    )

    F_g = np.where(gname == 1, 3.0, np.where(gname == 2, 4.0, np.inf))
    L_g = np.where(gname == 1, 3.0, np.where(gname == 2, 2.0, np.inf))
    S_g = np.where(gname == 1, 1.0, np.where(gname == 2, -1.0, 0.0))
    d_sq = np.where(gname == 2, 1.0, 0.0)
    T_g = np.full(n, 5.0)
    weight_gt = np.ones(n)

    first_obs = np.zeros(n)
    for g in [1.0, 2.0, 3.0]:
        idx = np.where(gname == g)[0][0]
        first_obs[idx] = 1.0

    return {
        "gname": gname,
        "tname": tname,
        "y": y,
        "d": d,
        "F_g": F_g,
        "L_g": L_g,
        "S_g": S_g,
        "d_sq": d_sq,
        "weight_gt": weight_gt,
        "first_obs_by_gp": first_obs,
        "T_g": T_g,
        "n_rows": n,
    }


@pytest.fixture
def large_partition(rng):
    """Larger partition for aggregate-level tests: 20 units x 8 periods.

    Units 0-9:  switch in at t=4 (S_g=1, d_sq=0)
    Units 10-14: switch in at t=5 (S_g=1, d_sq=0)
    Units 15-19: never switch  (S_g=0)
    """
    n_units = 20
    n_periods = 8
    n = n_units * n_periods

    gname = np.repeat(np.arange(1.0, n_units + 1), n_periods)
    tname = np.tile(np.arange(1.0, n_periods + 1), n_units)

    d = np.zeros(n)
    for u in range(1, 11):
        d[(gname == u) & (tname >= 4)] = 1.0
    for u in range(11, 16):
        d[(gname == u) & (tname >= 5)] = 1.0

    effect = 2.0
    y = rng.standard_normal(n) + effect * d

    F_g = np.full(n, np.inf)
    L_g = np.full(n, np.inf)
    S_g = np.zeros(n)

    for u in range(1, 11):
        mask = gname == u
        F_g[mask] = 4.0
        L_g[mask] = 5.0
        S_g[mask] = 1.0
    for u in range(11, 16):
        mask = gname == u
        F_g[mask] = 5.0
        L_g[mask] = 4.0
        S_g[mask] = 1.0

    d_sq = np.zeros(n)
    T_g = np.full(n, float(n_periods))
    weight_gt = np.ones(n)

    first_obs = np.zeros(n)
    for u in range(1, n_units + 1):
        idx = np.where(gname == u)[0][0]
        first_obs[idx] = 1.0

    return {
        "gname": gname,
        "tname": tname,
        "y": y,
        "d": d,
        "F_g": F_g,
        "L_g": L_g,
        "S_g": S_g,
        "d_sq": d_sq,
        "weight_gt": weight_gt,
        "first_obs_by_gp": first_obs,
        "T_g": T_g,
        "n_rows": n,
    }
