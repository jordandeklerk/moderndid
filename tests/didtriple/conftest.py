# pylint: disable=redefined-outer-name
"""Shared fixtures for didtriple tests."""

import numpy as np
import pandas as pd
import pytest

from moderndid import ddd_mp
from moderndid.core.preprocessing import preprocess_ddd_2periods
from moderndid.didtriple.dgp import gen_dgp_2periods, gen_dgp_mult_periods


@pytest.fixture
def ddd_data_with_covariates():
    """Generate DDD data with covariates."""
    result = gen_dgp_2periods(n=1000, dgp_type=1, random_state=42)
    data = result["data"]

    ddd_data = preprocess_ddd_2periods(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
        xformla="~ cov1 + cov2 + cov3 + cov4",
    )

    covariates = np.column_stack([np.ones(ddd_data.n_units), ddd_data.covariates])

    return ddd_data, covariates


@pytest.fixture
def ddd_data_no_covariates():
    """Generate DDD data without covariates."""
    result = gen_dgp_2periods(n=1000, dgp_type=1, random_state=42)
    data = result["data"]

    ddd_data = preprocess_ddd_2periods(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
    )

    covariates = np.ones((ddd_data.n_units, 1))

    return ddd_data, covariates


@pytest.fixture
def mp_ddd_data():
    """Generate multi-period panel data for DDD testing."""
    rng = np.random.default_rng(42)
    n_units = 500
    time_periods = [1, 2, 3, 4, 5]

    unit_ids = np.arange(n_units)
    groups = rng.choice([0, 3, 4], size=n_units, p=[0.5, 0.25, 0.25])
    partition = rng.choice([0, 1], size=n_units, p=[0.5, 0.5])

    records = []
    for unit in unit_ids:
        g = groups[unit]
        p = partition[unit]
        unit_effect = rng.normal(0, 1)

        for t in time_periods:
            time_effect = 0.5 * t
            treat_effect = 0.0
            if 0 < g <= t and p == 1:
                treat_effect = 2.0

            y = unit_effect + time_effect + treat_effect + rng.normal(0, 0.5)
            records.append({"id": unit, "time": t, "y": y, "group": g, "partition": p})

    return pd.DataFrame(records)


@pytest.fixture
def mp_ddd_result(mp_ddd_data):
    """Get multi-period DDD result for aggregation tests."""
    return ddd_mp(
        data=mp_ddd_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        est_method="reg",
    )


@pytest.fixture
def two_period_df():
    """Raw 2-period DataFrame for ddd() wrapper tests."""
    dgp = gen_dgp_2periods(n=1000, dgp_type=1, random_state=42)
    return dgp["data"]


@pytest.fixture
def multi_period_df():
    """Raw multi-period DataFrame for ddd() wrapper tests."""
    dgp = gen_dgp_mult_periods(n=500, dgp_type=1, random_state=42)
    return dgp["data"]


@pytest.fixture
def two_period_dgp_result():
    """Full 2-period DGP result including true ATT and oracle ATT."""
    result = gen_dgp_2periods(n=1000, dgp_type=1, random_state=42)
    return result["data"], result["true_att"], result["oracle_att"]


def convert_r_array(arr):
    """Convert R array to numpy array."""
    result = []
    for val in arr:
        if val == "NA" or val is None:
            result.append(np.nan)
        else:
            result.append(float(val))
    return np.array(result)


@pytest.fixture
def bootstrap_data():
    """Bootstrap data for numba tests."""
    rng = np.random.default_rng(42)
    n = 100
    k = 5
    inf_func = rng.standard_normal((n, k))
    return inf_func


@pytest.fixture
def cluster_data():
    """Cluster data for numba tests."""
    rng = np.random.default_rng(42)
    n = 100
    k = 5
    inf_func = rng.standard_normal((n, k))
    cluster = np.repeat(np.arange(10), 10)
    return inf_func, cluster


@pytest.fixture
def agg_inf_func_data():
    """Aggregation data for numba tests."""
    rng = np.random.default_rng(42)
    n = 100
    num_gt_cells = 20
    inf_func_mat = rng.standard_normal((n, num_gt_cells))
    whichones = np.array([0, 3, 7, 12, 15])
    weights = rng.random(len(whichones))
    weights = weights / weights.sum()
    return inf_func_mat, whichones, weights
