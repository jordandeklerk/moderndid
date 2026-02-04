"""Benchmark suite for did_multiplegt intertemporal treatment effects estimator."""

from benchmark.didinter.config import DIDINTER_BENCHMARK_SUITES, DIDInterBenchmarkConfig
from benchmark.didinter.runners import (
    R_DIDINTER_AVAILABLE,
    DIDInterPythonRunner,
    DIDInterRRunner,
)
from benchmark.didinter.storage import DIDInterBenchmarkResult

__all__ = [
    "DIDINTER_BENCHMARK_SUITES",
    "R_DIDINTER_AVAILABLE",
    "DIDInterBenchmarkConfig",
    "DIDInterBenchmarkResult",
    "DIDInterPythonRunner",
    "DIDInterRRunner",
]
