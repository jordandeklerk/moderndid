"""Shared fixtures for the top-level test directory."""

import polars as pl
import pytest


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
