"""CLI entry point for running benchmarks."""

from __future__ import annotations

import argparse
import logging
import sys

from benchmark.did.config import ATTGT_BENCHMARK_SUITES
from benchmark.did.run_benchmark import main as attgt_main
from benchmark.didtriple.config import DDD_BENCHMARK_SUITES
from benchmark.didtriple.run_benchmark import main as ddd_main

logger = logging.getLogger(__name__)


def main():
    """Run benchmark CLI."""
    parser = argparse.ArgumentParser(
        description="Benchmark Python moderndid package vs R equivalents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Estimator to benchmark")

    # att_gt subcommand
    attgt_parser = subparsers.add_parser("attgt", help="Benchmark att_gt vs R did package")
    attgt_parser.add_argument(
        "--suite",
        type=str,
        choices=list(ATTGT_BENCHMARK_SUITES.keys()),
        help="Predefined benchmark suite to run",
    )
    attgt_parser.add_argument("--n-units", type=int, default=1000, help="Number of units")
    attgt_parser.add_argument("--n-periods", type=int, default=5, help="Number of time periods")
    attgt_parser.add_argument("--n-groups", type=int, default=3, help="Number of treatment groups")
    attgt_parser.add_argument("--n-covariates", type=int, default=0, help="Number of covariates")
    attgt_parser.add_argument(
        "--est-method", type=str, default="dr", choices=["dr", "ipw", "reg"], help="Estimation method"
    )
    attgt_parser.add_argument("--boot", action="store_true", help="Use bootstrap")
    attgt_parser.add_argument("--biters", type=int, default=100, help="Bootstrap iterations")
    attgt_parser.add_argument("--warmup", type=int, default=1, help="Number of warmup runs")
    attgt_parser.add_argument("--runs", type=int, default=5, help="Number of timed runs")
    attgt_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    attgt_parser.add_argument("--python-only", action="store_true", help="Skip R benchmarks")
    attgt_parser.add_argument("--output-dir", type=str, default="benchmark/output", help="Output directory")
    attgt_parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    # ddd subcommand
    ddd_parser = subparsers.add_parser("ddd", help="Benchmark ddd vs R triplediff package")
    ddd_parser.add_argument(
        "--suite",
        type=str,
        choices=list(DDD_BENCHMARK_SUITES.keys()),
        help="Predefined benchmark suite to run",
    )
    ddd_parser.add_argument("--n-units", type=int, default=1000, help="Number of units")
    ddd_parser.add_argument("--dgp-type", type=int, default=1, choices=[1, 2, 3, 4], help="DGP type")
    ddd_parser.add_argument("--panel", action="store_true", default=True, help="Use panel data")
    ddd_parser.add_argument("--rcs", action="store_true", help="Use repeated cross-section data")
    ddd_parser.add_argument("--multi-period", action="store_true", help="Use multi-period DGP")
    ddd_parser.add_argument(
        "--est-method", type=str, default="dr", choices=["dr", "ipw", "reg"], help="Estimation method"
    )
    ddd_parser.add_argument("--boot", action="store_true", help="Use bootstrap")
    ddd_parser.add_argument("--biters", type=int, default=100, help="Bootstrap iterations")
    ddd_parser.add_argument("--warmup", type=int, default=1, help="Number of warmup runs")
    ddd_parser.add_argument("--runs", type=int, default=5, help="Number of timed runs")
    ddd_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    ddd_parser.add_argument("--python-only", action="store_true", help="Skip R benchmarks")
    ddd_parser.add_argument("--output-dir", type=str, default="benchmark/output", help="Output directory")
    ddd_parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "attgt":
        sys.argv = ["benchmark.did.run_benchmark", *sys.argv[2:]]
        attgt_main()
    elif args.command == "ddd":
        sys.argv = ["benchmark.didtriple.run_benchmark", *sys.argv[2:]]
        ddd_main()


if __name__ == "__main__":
    main()
