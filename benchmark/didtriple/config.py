"""Benchmark configurations for ddd estimator."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class DDDBenchmarkConfig:
    """Configuration for DDD benchmark runs."""

    n_units: int = 1000
    dgp_type: int = 1
    panel: bool = True
    multi_period: bool = False
    est_method: str = "dr"
    control_group: str = "nevertreated"
    base_period: str = "varying"
    boot: bool = False
    biters: int = 100
    xformla: str = "~ cov1 + cov2 + cov3 + cov4"
    n_warmup: int = 1
    n_runs: int = 5
    random_seed: int = 42

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


DDD_BENCHMARK_SUITES: dict[str, list[DDDBenchmarkConfig]] = {
    "scaling_units_2period": [
        DDDBenchmarkConfig(n_units=100, multi_period=False),
        DDDBenchmarkConfig(n_units=500, multi_period=False),
        DDDBenchmarkConfig(n_units=1000, multi_period=False),
        DDDBenchmarkConfig(n_units=5000, multi_period=False),
        DDDBenchmarkConfig(n_units=10000, multi_period=False),
        DDDBenchmarkConfig(n_units=50000, multi_period=False),
        DDDBenchmarkConfig(n_units=100000, multi_period=False),
    ],
    "scaling_units_multiperiod": [
        DDDBenchmarkConfig(n_units=100, multi_period=True),
        DDDBenchmarkConfig(n_units=500, multi_period=True),
        DDDBenchmarkConfig(n_units=1000, multi_period=True),
        DDDBenchmarkConfig(n_units=5000, multi_period=True),
        DDDBenchmarkConfig(n_units=10000, multi_period=True),
        DDDBenchmarkConfig(n_units=50000, multi_period=True),
    ],
    "est_methods": [
        DDDBenchmarkConfig(n_units=1000, multi_period=False, est_method="dr"),
        DDDBenchmarkConfig(n_units=1000, multi_period=False, est_method="ipw"),
        DDDBenchmarkConfig(n_units=1000, multi_period=False, est_method="reg"),
        DDDBenchmarkConfig(n_units=1000, multi_period=True, est_method="dr"),
        DDDBenchmarkConfig(n_units=1000, multi_period=True, est_method="ipw"),
        DDDBenchmarkConfig(n_units=1000, multi_period=True, est_method="reg"),
    ],
    "bootstrap": [
        DDDBenchmarkConfig(n_units=500, multi_period=False, boot=False),
        DDDBenchmarkConfig(n_units=500, multi_period=False, boot=True, biters=100),
        DDDBenchmarkConfig(n_units=500, multi_period=False, boot=True, biters=500),
        DDDBenchmarkConfig(n_units=500, multi_period=False, boot=True, biters=1000),
    ],
    "panel_vs_rcs": [
        DDDBenchmarkConfig(n_units=1000, multi_period=False, panel=True),
        DDDBenchmarkConfig(n_units=1000, multi_period=False, panel=False),
        DDDBenchmarkConfig(n_units=1000, multi_period=True, panel=True),
        DDDBenchmarkConfig(n_units=1000, multi_period=True, panel=False),
    ],
    "quick": [
        DDDBenchmarkConfig(n_units=100, multi_period=False),
        DDDBenchmarkConfig(n_units=500, multi_period=False),
        DDDBenchmarkConfig(n_units=1000, multi_period=False),
    ],
}
