"""Result storage for cont_did benchmark outputs."""

from __future__ import annotations

from dataclasses import dataclass

from benchmark.common.storage import BaseBenchmarkResult


@dataclass
class ContDIDBenchmarkResult(BaseBenchmarkResult):
    """Container for cont_did benchmark result."""

    n_periods: int = 4
    target_parameter: str = "level"
    aggregation: str = "dose"
    dose_est_method: str = "parametric"
    degree: int = 3
    num_knots: int = 0
