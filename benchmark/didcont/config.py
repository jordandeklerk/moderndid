"""Benchmark configurations for cont_did estimator."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ContDIDBenchmarkConfig:
    """Configuration for cont_did benchmark runs."""

    n_units: int = 500
    n_periods: int = 4
    target_parameter: str = "level"
    aggregation: str = "dose"
    dose_est_method: str = "parametric"
    degree: int = 3
    num_knots: int = 0
    boot: bool = False
    biters: int = 100
    n_warmup: int = 1
    n_runs: int = 5
    random_seed: int = 42

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


CONTDID_BENCHMARK_SUITES: dict[str, list[ContDIDBenchmarkConfig]] = {
    "scaling_units": [
        ContDIDBenchmarkConfig(n_units=100, n_periods=4),
        ContDIDBenchmarkConfig(n_units=500, n_periods=4),
        ContDIDBenchmarkConfig(n_units=1000, n_periods=4),
        ContDIDBenchmarkConfig(n_units=2000, n_periods=4),
        ContDIDBenchmarkConfig(n_units=5000, n_periods=4),
        ContDIDBenchmarkConfig(n_units=10000, n_periods=4),
    ],
    "scaling_periods": [
        ContDIDBenchmarkConfig(n_units=500, n_periods=3),
        ContDIDBenchmarkConfig(n_units=500, n_periods=4),
        ContDIDBenchmarkConfig(n_units=500, n_periods=5),
        ContDIDBenchmarkConfig(n_units=500, n_periods=6),
        ContDIDBenchmarkConfig(n_units=500, n_periods=8),
    ],
    "target_parameters": [
        ContDIDBenchmarkConfig(n_units=500, n_periods=4, target_parameter="level"),
        ContDIDBenchmarkConfig(n_units=500, n_periods=4, target_parameter="slope"),
    ],
    "aggregation_types": [
        ContDIDBenchmarkConfig(n_units=500, n_periods=4, aggregation="dose"),
        ContDIDBenchmarkConfig(n_units=500, n_periods=4, aggregation="eventstudy"),
    ],
    "dose_methods": [
        ContDIDBenchmarkConfig(n_units=500, n_periods=4, dose_est_method="parametric", degree=2),
        ContDIDBenchmarkConfig(n_units=500, n_periods=4, dose_est_method="parametric", degree=3),
        ContDIDBenchmarkConfig(n_units=500, n_periods=4, dose_est_method="parametric", degree=3, num_knots=2),
    ],
    "cck_method": [
        ContDIDBenchmarkConfig(n_units=100, n_periods=2, dose_est_method="cck"),
        ContDIDBenchmarkConfig(n_units=500, n_periods=2, dose_est_method="cck"),
        ContDIDBenchmarkConfig(n_units=1000, n_periods=2, dose_est_method="cck"),
        ContDIDBenchmarkConfig(n_units=2000, n_periods=2, dose_est_method="cck"),
    ],
    "bootstrap": [
        ContDIDBenchmarkConfig(n_units=500, n_periods=4, boot=False),
        ContDIDBenchmarkConfig(n_units=500, n_periods=4, boot=True, biters=100),
        ContDIDBenchmarkConfig(n_units=500, n_periods=4, boot=True, biters=500),
    ],
    "quick": [
        ContDIDBenchmarkConfig(n_units=100, n_periods=4),
        ContDIDBenchmarkConfig(n_units=500, n_periods=4),
        ContDIDBenchmarkConfig(n_units=1000, n_periods=4),
    ],
    "comprehensive": [
        # Parametric scaling
        ContDIDBenchmarkConfig(n_units=500, n_periods=4, dose_est_method="parametric"),
        ContDIDBenchmarkConfig(n_units=1000, n_periods=4, dose_est_method="parametric"),
        ContDIDBenchmarkConfig(n_units=2000, n_periods=4, dose_est_method="parametric"),
        # Target parameters
        ContDIDBenchmarkConfig(n_units=500, n_periods=4, target_parameter="level"),
        ContDIDBenchmarkConfig(n_units=500, n_periods=4, target_parameter="slope"),
        # Aggregations
        ContDIDBenchmarkConfig(n_units=500, n_periods=4, aggregation="dose"),
        ContDIDBenchmarkConfig(n_units=500, n_periods=4, aggregation="eventstudy"),
        # CCK method (requires 2 periods)
        ContDIDBenchmarkConfig(n_units=500, n_periods=2, dose_est_method="cck"),
        ContDIDBenchmarkConfig(n_units=1000, n_periods=2, dose_est_method="cck"),
    ],
    "large_scale": [
        ContDIDBenchmarkConfig(n_units=100000, n_periods=4, n_runs=3),
        ContDIDBenchmarkConfig(n_units=250000, n_periods=4, n_runs=3),
        ContDIDBenchmarkConfig(n_units=500000, n_periods=4, n_runs=3),
        ContDIDBenchmarkConfig(n_units=1000000, n_periods=4, n_runs=3),
        ContDIDBenchmarkConfig(n_units=2500000, n_periods=4, n_runs=2),
        ContDIDBenchmarkConfig(n_units=5000000, n_periods=4, n_runs=2),
    ],
}
