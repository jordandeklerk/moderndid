"""Result storage for att_gt benchmark outputs."""

from __future__ import annotations

from dataclasses import dataclass

from benchmark.common.storage import BaseBenchmarkResult


@dataclass
class ATTgtBenchmarkResult(BaseBenchmarkResult):
    """Container for att_gt benchmark result."""

    n_periods: int = 0
    n_groups: int = 0
    n_covariates: int = 0
    control_group: str = "nevertreated"
