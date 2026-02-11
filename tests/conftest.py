"""Shared fixtures for the top-level test directory."""

import numpy as np
import polars as pl
import pytest

from moderndid.core.parallel import dask_available

requires_dask = pytest.mark.skipif(not dask_available(), reason="dask not installed")


class CountingScheduler:
    """Dask scheduler that counts the number of compute calls.

    Adapted from xarray's test infrastructure. Plugs into
    ``dask.config.set(scheduler=...)`` and proves that work actually
    flows through the dask graph rather than being silently bypassed.
    """

    def __init__(self):
        self.total_computes = 0

    def __call__(self, dsk, keys, **kwargs):
        import dask

        self.total_computes += 1
        return dask.local.get_sync(dsk, keys, **kwargs)


def _square(x):
    return x * x


def _add(a, b):
    return a + b


def _ols_cell(x, y):
    xtx_inv = np.linalg.inv(x.T @ x)
    return xtx_inv @ (x.T @ y)


def _raise_on_two(x):
    if x == 2:
        raise ValueError("boom")
    return x


@pytest.fixture
def balanced_panel():
    return pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "time": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "y": [10, 12, 15, 20, 22, 25, 30, 32, 35],
            "x": [1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2],
        }
    )


@pytest.fixture
def unbalanced_panel():
    return pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4],
            "time": [1, 2, 3, 1, 2, 3, 2, 3, 1, 2, 3],
            "y": [10, 12, 15, 20, 22, 25, 32, 35, 40, 42, 45],
            "treat": [0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
        }
    )


@pytest.fixture
def panel_with_duplicates():
    return pl.DataFrame(
        {
            "id": [1, 1, 1, 1, 2, 2, 2],
            "time": [1, 1, 2, 3, 1, 2, 2],
            "y": [10.0, 11.0, 12.0, 15.0, 20.0, 22.0, 24.0],
            "cat": ["a", "b", "a", "a", "c", "c", "d"],
        }
    )


@pytest.fixture
def staggered_panel():
    return pl.DataFrame(
        {
            "id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "time": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            "y": [10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 35, 38],
            "treat": [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
        }
    )


def _distributed_available():
    try:
        import distributed  # noqa: F401

        return True
    except ImportError:
        return False


requires_distributed = pytest.mark.skipif(not _distributed_available(), reason="distributed not installed")
