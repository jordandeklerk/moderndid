"""Shared fixtures for core tests."""

import pytest

import polars as pl


@pytest.fixture
def sample_data():
    return pl.DataFrame(
        {
            "group": [1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0],
            "time": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "y": [10, 11, 20, 21, 30, 31, 12, 13, 22, 23, 32, 33],
            "id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
        }
    )
