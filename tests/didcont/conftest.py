# pylint: disable=redefined-outer-name
"""Shared fixtures for tests."""

import numpy as np
import pandas as pd
import pytest

from moderndid.data import load_engel


@pytest.fixture
def simple_data():
    np.random.seed(42)
    n = 200
    w = np.random.uniform(0, 1, (n, 1))
    x = w + 0.2 * np.random.normal(0, 1, (n, 1))
    y = np.sin(2 * np.pi * x).ravel() + 0.1 * np.random.normal(0, 1, n)
    return y, x, w


@pytest.fixture
def multivariate_data():
    np.random.seed(42)
    n = 300
    w = np.random.uniform(0, 1, (n, 2))
    x = w @ np.array([[1.0, 0.5], [0.3, 0.8]]).T + 0.1 * np.random.normal(0, 1, (n, 2))
    y = np.sin(x[:, 0]) * np.cos(x[:, 1]) + 0.1 * np.random.normal(0, 1, n)
    return y, x, w


@pytest.fixture
def regression_data():
    np.random.seed(42)
    n = 150
    x = np.random.uniform(0, 1, (n, 1))
    y = 2 * x.ravel() + x.ravel() ** 2 + 0.1 * np.random.normal(0, 1, n)
    return y, x, x


@pytest.fixture
def continuous_data():
    np.random.seed(42)
    return np.random.normal(0, 1, (200, 3))


@pytest.fixture
def discrete_data():
    np.random.seed(42)
    n = 200
    return np.column_stack(
        [
            np.random.choice([0, 1, 2], n),
            np.random.choice([0, 1], n),
        ]
    )


@pytest.fixture
def degree_matrix():
    return np.array([[3, 3], [2, 2], [3, 4]])


@pytest.fixture
def indicator_vector():
    return np.array([1, 1])


@pytest.fixture
def simple_setup():
    np.random.seed(42)
    x = np.random.uniform(0, 1, (100, 2))
    K = np.array([[3, 4], [2, 3]])
    return x, K


@pytest.fixture
def basis_list():
    np.random.seed(42)
    return [
        np.random.normal(0, 1, (50, 3)),
        np.random.normal(0, 1, (50, 2)),
        np.random.normal(0, 1, (50, 4)),
    ]


@pytest.fixture
def simple_bases():
    return [
        np.array([[1, 2], [3, 4], [5, 6]]),
        np.array([[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]]),
    ]


@pytest.fixture
def three_bases():
    return [
        np.array([[1, 2], [3, 4]]),
        np.array([[1], [2]]),
        np.array([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]]),
    ]


@pytest.fixture
def bspline_simple_data():
    return np.linspace(0, 1, 100)


@pytest.fixture
def random_uniform_data():
    np.random.seed(42)
    return np.random.uniform(0, 10, 200)


@pytest.fixture
def sparse_data():
    return np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])


@pytest.fixture
def selection_result():
    return {
        "j_x_seg": 3,
        "k_w_seg": 4,
        "j_tilde": 5,
        "theta_star": 1.2,
        "j_x_segments_set": np.array([2, 3, 4, 5]),
        "k_w_segments_set": np.array([3, 4, 5, 6]),
    }


@pytest.fixture
def matrices_small():
    np.random.seed(42)
    n = 50
    x = np.random.randn(n, 10)
    y = np.random.randn(n)
    y_pred = y + 0.1 * np.random.randn(n)
    symmetric_matrix = np.eye(5) + 0.1 * np.random.randn(5, 5)
    symmetric_matrix = (symmetric_matrix + symmetric_matrix.T) / 2
    return x, y, y_pred, symmetric_matrix


@pytest.fixture
def matrices_large():
    np.random.seed(42)
    n = 5000
    p = 100
    x = np.random.randn(n, p)
    y = np.random.randn(n)
    y_pred = y + 0.1 * np.random.randn(n)
    symmetric_matrix = np.eye(50) + 0.1 * np.random.randn(50, 50)
    symmetric_matrix = (symmetric_matrix + symmetric_matrix.T) / 2
    return x, y, y_pred, symmetric_matrix


@pytest.fixture
def basis_matrices():
    np.random.seed(42)
    n_obs = 100
    bases = [
        np.random.randn(n_obs, 3),
        np.random.randn(n_obs, 4),
        np.random.randn(n_obs, 2),
    ]
    return bases


@pytest.fixture
def engel_data():
    engel_df = load_engel()
    engel_df = engel_df.sort_values("logexp")

    return {
        "food": engel_df["food"].values,
        "logexp": engel_df["logexp"].values.reshape(-1, 1),
        "logwages": engel_df["logwages"].values.reshape(-1, 1),
    }


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def simple_matrix(rng):
    return rng.randn(100, 5)


@pytest.fixture
def rank_deficient_matrix(rng):
    x = np.ones((100, 3))
    x[:, 1] = 2 * x[:, 0]
    x[:, 2] = rng.randn(100)
    return x


@pytest.fixture
def symmetric_psd_matrix(rng):
    A = rng.randn(5, 5)
    return A @ A.T


@pytest.fixture
def panel_data_balanced():
    np.random.seed(42)
    n_units = 20
    n_periods = 6
    unit_ids = np.repeat(np.arange(1, n_units + 1), n_periods)
    time_ids = np.tile(np.arange(2010, 2010 + n_periods), n_units)

    d_base = np.repeat(np.random.normal(0, 1, n_units), n_periods)
    d_variation = 0.5 * np.random.normal(0, 1, len(unit_ids))
    d = np.maximum(0, d_base + d_variation)

    unit_effects = np.repeat(np.random.normal(0, 1, n_units), n_periods)
    time_effects = np.tile(np.random.normal(0, 0.5, n_periods), n_units)
    treatment_effect = 2.0 * d + 0.5 * d**2
    noise = np.random.normal(0, 1, len(unit_ids))
    y = unit_effects + time_effects + treatment_effect + noise

    x1 = np.random.normal(1, 0.5, len(unit_ids))
    x2 = np.random.binomial(1, 0.3, len(unit_ids))

    return pd.DataFrame(
        {
            "unit_id": unit_ids,
            "time_id": time_ids,
            "y": y,
            "d": d,
            "x1": x1,
            "x2": x2,
            "weights": np.ones(len(unit_ids)),
        }
    )


@pytest.fixture
def panel_data_unbalanced():
    np.random.seed(42)
    n_units = 15
    max_periods = 8

    data_list = []
    for unit in range(1, n_units + 1):
        n_periods_unit = np.random.randint(3, max_periods + 1)
        start_year = np.random.randint(2008, 2012)
        periods = np.arange(start_year, start_year + n_periods_unit)

        unit_ids = np.full(n_periods_unit, unit)
        time_ids = periods

        d_base = np.random.normal(1, 0.8)
        d = np.maximum(0, d_base + 0.3 * np.random.normal(0, 1, n_periods_unit))

        unit_effect = np.random.normal(0, 1.2)
        time_effects = 0.1 * (periods - 2010)
        treatment_effect = 1.8 * d
        noise = np.random.normal(0, 0.8, n_periods_unit)
        y = unit_effect + time_effects + treatment_effect + noise

        x1 = np.random.normal(0, 1, n_periods_unit)
        x2 = np.random.exponential(1, n_periods_unit)
        weights = np.random.gamma(2, 0.5, n_periods_unit)

        unit_data = pd.DataFrame(
            {"unit_id": unit_ids, "time_id": time_ids, "y": y, "d": d, "x1": x1, "x2": x2, "weights": weights}
        )
        data_list.append(unit_data)

    return pd.concat(data_list, ignore_index=True)


@pytest.fixture
def staggered_treatment_panel():
    data = {
        "id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        "time": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
        "y": [10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 35, 38],
        "treat": [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def unbalanced_simple_panel():
    data = {
        "id": [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4],
        "time": [1, 2, 3, 1, 2, 3, 2, 3, 1, 2, 3],
        "y": [10, 12, 15, 20, 22, 25, 32, 35, 40, 42, 45],
        "treat": [0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def panel_data_with_covariates():
    np.random.seed(42)
    n_units = 25
    n_periods = 5
    n_obs = n_units * n_periods

    unit_ids = np.repeat(np.arange(1, n_units + 1), n_periods)
    time_ids = np.tile(np.arange(2015, 2015 + n_periods), n_units)

    age = np.repeat(np.random.randint(25, 65, n_units), n_periods)
    education = np.repeat(np.random.choice(["HS", "College", "Grad"], n_units), n_periods)
    income_base = np.repeat(np.random.lognormal(10, 0.8, n_units), n_periods)
    income = income_base + 0.05 * (time_ids - 2015) * income_base

    treatment_propensity = (
        0.02 * (age - 40) + 0.3 * (income / 50000) + 0.1 * (education == "College") + 0.2 * (education == "Grad")
    )
    d = np.maximum(0, treatment_propensity + 0.5 * np.random.normal(0, 1, n_obs))

    y = (
        2
        + 0.03 * age
        + 0.5 * (education == "College")
        + 0.8 * (education == "Grad")
        + 0.00002 * income
        + 1.5 * d
        - 0.1 * d**2
        + 0.2 * (time_ids - 2017)
        + np.random.normal(0, 1, n_obs)
    )

    return pd.DataFrame(
        {
            "unit_id": unit_ids,
            "time_id": time_ids,
            "y": y,
            "d": d,
            "age": age,
            "education": education,
            "income": income,
            "weights": np.random.uniform(0.5, 1.5, n_obs),
        }
    )


@pytest.fixture
def panel_data_with_weights():
    np.random.seed(42)
    n_units = 30
    n_periods = 4
    n_obs = n_units * n_periods

    unit_ids = np.repeat(np.arange(1, n_units + 1), n_periods)
    time_ids = np.tile([2018, 2019, 2020, 2021], n_units)

    stratum = np.repeat(np.random.choice(["A", "B", "C"], n_units, p=[0.5, 0.3, 0.2]), n_periods)
    base_weights = np.where(stratum == "A", 1.0, np.where(stratum == "B", 2.0, 3.0))

    weights = base_weights * np.random.uniform(0.8, 1.2, n_obs)

    treatment_multiplier = np.where(stratum == "A", 1.0, np.where(stratum == "B", 1.5, 0.8))
    d = treatment_multiplier * np.maximum(0, np.random.normal(1, 0.8, n_obs))

    stratum_effect = np.where(stratum == "A", 0, np.where(stratum == "B", 1.2, -0.8))
    y = stratum_effect + 2.0 * d + 0.1 * (time_ids - 2019) + np.random.normal(0, 1, n_obs)

    return pd.DataFrame(
        {"unit_id": unit_ids, "time_id": time_ids, "y": y, "d": d, "stratum": stratum, "weights": weights}
    )
