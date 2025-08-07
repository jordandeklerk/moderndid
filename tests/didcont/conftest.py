# pylint: disable=redefined-outer-name
"""Shared fixtures for didcont tests."""

import numpy as np
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
