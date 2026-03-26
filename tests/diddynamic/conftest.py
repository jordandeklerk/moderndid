"""Shared fixtures for diddynamic tests."""

import numpy as np
import polars as pl
import pytest

from moderndid.core.preprocess import DynBalancingConfig, PreprocessDataBuilder


def build_dyn_balancing(data, **config_kwargs):
    """Run the full builder pipeline for DynBalancing."""
    config = DynBalancingConfig(**config_kwargs)
    return PreprocessDataBuilder().with_data(data).with_config(config).validate().transform().build()


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def simple_panel():
    """Balanced panel with 10 units and 4 periods."""
    rng = np.random.default_rng(42)
    n_units = 10
    n_periods = 4
    ids = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(1, n_periods + 1), n_units)
    treatment = np.zeros(n_units * n_periods)
    for i in range(n_units):
        if i < 5:
            treatment[i * n_periods + 2 : (i + 1) * n_periods] = 1.0
    y = rng.standard_normal(n_units * n_periods)
    x1 = rng.standard_normal(n_units * n_periods)
    x2 = rng.standard_normal(n_units * n_periods)
    cluster = np.repeat(np.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 3]), n_periods)

    return pl.DataFrame(
        {
            "id": ids,
            "time": times,
            "y": y,
            "D": treatment,
            "X1": x1,
            "X2": x2,
            "cluster_var": cluster,
        }
    )


@pytest.fixture
def unbalanced_panel():
    """Panel where unit 1 is missing period 3."""
    return pl.DataFrame(
        {
            "id": [0, 0, 0, 1, 1, 2, 2, 2],
            "time": [1, 2, 3, 1, 2, 1, 2, 3],
            "y": [1.0] * 8,
            "D": [0.0] * 8,
        }
    )


@pytest.fixture
def estimation_panel(rng):
    """Synthetic panel for estimation module tests (60 units, 3 periods, 3 covariates)."""
    n_units = 60
    n_periods = 3
    n_covariates = 3
    treatment = np.zeros((n_units, n_periods))
    for i in range(n_units // 2):
        treatment[i, 1:] = 1.0
    covariates = {t: rng.standard_normal((n_units, n_covariates)) for t in range(n_periods)}
    outcome = rng.standard_normal(n_units) + 0.5 * covariates[n_periods - 1][:, 0] + 2.0 * treatment[:, -1]
    ds = np.array([0.0, 1.0, 1.0])
    return outcome, treatment, covariates, ds


@pytest.fixture
def simple_qp_data(rng):
    """Small balanced data for direct QP testing."""
    n = 20
    p = 2
    x_all = rng.standard_normal((n, p))
    d_col = np.array([1.0] * 10 + [0.0] * 10)
    coef = np.zeros(p + 1)
    return x_all, d_col, coef
