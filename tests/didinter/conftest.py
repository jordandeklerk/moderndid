"""Shared fixtures for didinter tests."""

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid import load_favara_imbs


@pytest.fixture(scope="module")
def favara_imbs_data():
    """Load the Favara and Imbs dataset."""
    return load_favara_imbs()


@pytest.fixture
def rng():
    """Seeded random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def simple_panel_data(rng):
    """Simple panel data with known switchers at periods 3 and 4."""
    n_units = 50
    n_periods = 6

    units = np.repeat(np.arange(n_units), n_periods)
    periods = np.tile(np.arange(1, n_periods + 1), n_units)

    treatment = np.zeros(len(units))
    for unit in range(n_units):
        unit_mask = units == unit
        if unit < 20:
            switch_time = 3
            treatment[unit_mask & (periods >= switch_time)] = 1
        elif unit < 30:
            switch_time = 4
            treatment[unit_mask & (periods >= switch_time)] = 1

    y = rng.standard_normal(len(units))
    treatment_effect = 2.0
    y += treatment_effect * treatment

    return pl.DataFrame(
        {
            "id": units,
            "time": periods,
            "y": y,
            "d": treatment,
        }
    )


@pytest.fixture
def bidirectional_panel_data(rng):
    """Panel data with units that switch treatment in both directions."""
    n_units = 30
    n_periods = 8

    units = np.repeat(np.arange(n_units), n_periods)
    periods = np.tile(np.arange(1, n_periods + 1), n_units)

    treatment = np.zeros(len(units))
    for unit in range(n_units):
        unit_mask = units == unit
        if unit < 5:
            treatment[unit_mask & (periods >= 3) & (periods <= 5)] = 1
        elif unit < 15:
            treatment[unit_mask & (periods >= 4)] = 1

    y = rng.standard_normal(len(units)) + 1.5 * treatment

    return pl.DataFrame(
        {
            "id": units,
            "time": periods,
            "y": y,
            "d": treatment,
        }
    )


@pytest.fixture
def weighted_panel_data(simple_panel_data, rng):
    """Panel data with sampling weights."""
    weights = rng.uniform(0.5, 2.0, len(simple_panel_data))
    return simple_panel_data.with_columns(pl.Series("w", weights))


@pytest.fixture
def clustered_panel_data(simple_panel_data):
    """Panel data with cluster variable."""
    df = simple_panel_data.clone()
    cluster = (df["id"] // 10).cast(pl.Int64)
    return df.with_columns(cluster.alias("cluster"))


@pytest.fixture
def panel_with_controls(simple_panel_data, rng):
    """Panel data with control variables."""
    n = len(simple_panel_data)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    return simple_panel_data.with_columns(
        [
            pl.Series("x1", x1),
            pl.Series("x2", x2),
        ]
    )


@pytest.fixture
def unbalanced_panel_data(rng):
    """Unbalanced panel data with randomly missing observations."""
    n_units = 40
    n_periods = 6

    units = np.repeat(np.arange(n_units), n_periods)
    periods = np.tile(np.arange(1, n_periods + 1), n_units)

    treatment = np.zeros(len(units))
    for unit in range(n_units):
        unit_mask = units == unit
        if unit < 15:
            treatment[unit_mask & (periods >= 3)] = 1

    y = rng.standard_normal(len(units)) + 2.0 * treatment

    df = pl.DataFrame(
        {
            "id": units,
            "time": periods,
            "y": y,
            "d": treatment,
        }
    )

    keep_mask = rng.uniform(size=len(df)) > 0.15
    return df.filter(pl.Series(keep_mask))


@pytest.fixture
def basic_config():
    """Basic DIDInterConfig for testing."""
    from moderndid.core.preprocess.config import DIDInterConfig

    return DIDInterConfig(
        yname="y",
        tname="time",
        gname="id",
        dname="d",
    )


@pytest.fixture
def panel_data():
    """Small panel data with pre-computed switcher columns."""
    return pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "time": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "y": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 2.0, 2.0, 2.0],
            "d": [0, 0, 1, 0, 1, 1, 0, 0, 0],
            "F_g": [3.0, 3.0, 3.0, 2.0, 2.0, 2.0, float("inf"), float("inf"), float("inf")],
            "d_sq": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "S_g": [1, 1, 1, 1, 1, 1, 0, 0, 0],
            "L_g": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, float("inf"), float("inf"), float("inf")],
        }
    )


@pytest.fixture
def switcher_data():
    """Panel data for testing delta_d computation."""
    return pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "time": [1, 2, 3, 1, 2, 3],
            "d": [0, 0, 1, 0, 1, 1],
            "d_sq": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "F_g": [3.0, 3.0, 3.0, 2.0, 2.0, 2.0],
            "S_g": [1, 1, 1, 1, 1, 1],
            "weight_gt": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "dist_to_switch_1": [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        }
    )


@pytest.fixture
def effects_results_basic():
    """Basic effects results dict for testing ATE computation."""
    return {
        "estimates": np.array([0.5, 0.6, 0.7]),
        "estimates_unnorm": np.array([0.5, 0.6, 0.7]),
        "std_errors": np.array([0.1, 0.12, 0.15]),
        "n_switchers": np.array([100.0, 90.0, 80.0]),
        "n_switchers_weighted": np.array([100.0, 90.0, 80.0]),
        "delta_d_arr": np.array([1.0, 1.0, 1.0]),
        "n_observations": np.array([500.0, 450.0, 400.0]),
        "vcov": np.diag([0.01, 0.0144, 0.0225]),
    }
