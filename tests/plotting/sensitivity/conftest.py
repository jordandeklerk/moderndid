"""Shared fixtures for the sensitivity plot tests."""

import polars as pl
import pytest

from tests.helpers import importorskip

importorskip("moderndid.didhonest")

from moderndid.didhonest.honest_did import HonestDiDResult
from moderndid.didhonest.sensitivity import OriginalCSResult


@pytest.fixture
def honest_result():
    """Return a smoothness sensitivity result with three fixed-length intervals."""
    df = pl.DataFrame({"M": [0.5, 1.0, 1.5], "lb": [0.1, 0.0, -0.1], "ub": [0.9, 0.8, 0.7], "method": ["FLCI"] * 3})

    original_ci = OriginalCSResult(
        lb=0.3,
        ub=0.7,
        method="Original",
    )

    return HonestDiDResult(
        robust_ci=df,
        original_ci=original_ci,
        sensitivity_type="smoothness",
    )
