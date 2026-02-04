"""Benchmark configurations for did_multiplegt estimator."""

from dataclasses import dataclass


@dataclass
class DIDInterBenchmarkConfig:
    """Configuration for a single did_multiplegt benchmark run."""

    n_units: int = 500
    n_periods: int = 10
    effects: int = 3
    placebo: int = 2
    normalized: bool = False
    boot: bool = False
    biters: int = 100
    n_warmup: int = 1
    n_runs: int = 5
    random_seed: int = 42

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.effects < 1:
            raise ValueError("effects must be >= 1")
        if self.placebo < 0:
            raise ValueError("placebo must be >= 0")
        if self.n_units < 10:
            raise ValueError("n_units must be >= 10")
        if self.n_periods < 4:
            raise ValueError("n_periods must be >= 4")


DIDINTER_BENCHMARK_SUITES: dict[str, list[DIDInterBenchmarkConfig]] = {
    # Quick validation suite
    "quick": [
        DIDInterBenchmarkConfig(n_units=100, n_periods=6, effects=2, placebo=1),
        DIDInterBenchmarkConfig(n_units=500, n_periods=8, effects=3, placebo=2),
        DIDInterBenchmarkConfig(n_units=1000, n_periods=10, effects=3, placebo=2),
    ],
    # Unit scaling
    "scaling_units": [
        DIDInterBenchmarkConfig(n_units=100, n_periods=8),
        DIDInterBenchmarkConfig(n_units=500, n_periods=8),
        DIDInterBenchmarkConfig(n_units=1000, n_periods=8),
        DIDInterBenchmarkConfig(n_units=2000, n_periods=8),
        DIDInterBenchmarkConfig(n_units=5000, n_periods=8),
        DIDInterBenchmarkConfig(n_units=10000, n_periods=8),
    ],
    # Period scaling
    "scaling_periods": [
        DIDInterBenchmarkConfig(n_units=1000, n_periods=5, effects=2, placebo=1),
        DIDInterBenchmarkConfig(n_units=1000, n_periods=8, effects=3, placebo=2),
        DIDInterBenchmarkConfig(n_units=1000, n_periods=12, effects=5, placebo=3),
        DIDInterBenchmarkConfig(n_units=1000, n_periods=15, effects=6, placebo=4),
        DIDInterBenchmarkConfig(n_units=1000, n_periods=20, effects=8, placebo=5),
    ],
    # Effects/placebo scaling
    "scaling_effects": [
        DIDInterBenchmarkConfig(n_units=1000, n_periods=15, effects=1, placebo=0),
        DIDInterBenchmarkConfig(n_units=1000, n_periods=15, effects=3, placebo=2),
        DIDInterBenchmarkConfig(n_units=1000, n_periods=15, effects=5, placebo=3),
        DIDInterBenchmarkConfig(n_units=1000, n_periods=15, effects=8, placebo=5),
    ],
    # Normalized vs non-normalized
    "normalized": [
        DIDInterBenchmarkConfig(n_units=1000, n_periods=10, normalized=False),
        DIDInterBenchmarkConfig(n_units=1000, n_periods=10, normalized=True),
    ],
    # Bootstrap scaling
    "bootstrap": [
        DIDInterBenchmarkConfig(n_units=500, n_periods=8, boot=False),
        DIDInterBenchmarkConfig(n_units=500, n_periods=8, boot=True, biters=100),
        DIDInterBenchmarkConfig(n_units=500, n_periods=8, boot=True, biters=500),
        DIDInterBenchmarkConfig(n_units=500, n_periods=8, boot=True, biters=1000),
    ],
    # Large scale
    "large_scale": [
        DIDInterBenchmarkConfig(n_units=10000, n_periods=10, effects=3, placebo=2),
        DIDInterBenchmarkConfig(n_units=50000, n_periods=10, effects=3, placebo=2),
        DIDInterBenchmarkConfig(n_units=100000, n_periods=10, effects=3, placebo=2),
    ],
}
