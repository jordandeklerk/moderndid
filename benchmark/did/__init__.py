"""Benchmark suite for att_gt estimator vs R did package."""

from benchmark.did.config import ATTGT_BENCHMARK_SUITES, ATTgtBenchmarkConfig
from benchmark.did.dgp import StaggeredDIDDGP
from benchmark.did.runners import R_DID_AVAILABLE, ATTgtPythonRunner, ATTgtRRunner
from benchmark.did.storage import ATTgtBenchmarkResult

__all__ = [
    "ATTGT_BENCHMARK_SUITES",
    "R_DID_AVAILABLE",
    "ATTgtBenchmarkConfig",
    "ATTgtBenchmarkResult",
    "ATTgtPythonRunner",
    "ATTgtRRunner",
    "StaggeredDIDDGP",
]
