"""CLI entry point for running benchmarks."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime

from benchmark.config import (
    ATTGT_BENCHMARK_SUITES,
    DDD_BENCHMARK_SUITES,
    ATTgtBenchmarkConfig,
    DDDBenchmarkConfig,
)
from benchmark.dgp.staggered_did import StaggeredDIDDGP
from benchmark.results.storage import ATTgtBenchmarkResult, DDDBenchmarkResult, ResultStorage
from benchmark.runners.python_runner import PythonBenchmarkRunner
from benchmark.runners.r_runner import R_DID_AVAILABLE, R_TRIPLEDIFF_AVAILABLE, RBenchmarkRunner

logger = logging.getLogger(__name__)


def run_single_attgt_benchmark(
    config: ATTgtBenchmarkConfig,
    python_runner: PythonBenchmarkRunner,
    r_runner: RBenchmarkRunner | None,
) -> ATTgtBenchmarkResult:
    """Run a single benchmark configuration."""
    logger.info(
        "Generating data: %d units, %d periods, %d groups",
        config.n_units,
        config.n_periods,
        config.n_groups,
    )

    dgp = StaggeredDIDDGP(
        n_units=config.n_units,
        n_periods=config.n_periods,
        n_groups=config.n_groups,
        n_covariates=config.n_covariates,
        random_seed=config.random_seed,
    )
    data_result = dgp.generate_data()
    data = data_result["df"]

    logger.info("Running Python benchmark (%d runs)...", config.n_runs)

    python_result = python_runner.time_att_gt(
        data=data,
        est_method=config.est_method,
        control_group=config.control_group,
        boot=config.boot,
        biters=config.biters,
        xformla=config.xformla,
        n_warmup=config.n_warmup,
        n_runs=config.n_runs,
    )

    if python_result.success:
        logger.info("Python: %.4fs (std: %.4fs)", python_result.mean_time, python_result.std_time)
    else:
        logger.error("Python: FAILED - %s", python_result.error)

    r_result = None
    if r_runner is not None:
        logger.info("Running R benchmark (%d runs)...", config.n_runs)

        r_result = r_runner.time_att_gt(
            data=data,
            est_method=config.est_method,
            control_group=config.control_group,
            boot=config.boot,
            biters=config.biters,
            xformla=config.xformla,
            n_warmup=config.n_warmup,
            n_runs=config.n_runs,
        )

        if r_result.success:
            logger.info("R: %.4fs (std: %.4fs)", r_result.mean_time, r_result.std_time)
        else:
            logger.error("R: FAILED - %s", r_result.error)

    speedup = float("nan")
    if python_result.success and r_result is not None and r_result.success and python_result.mean_time > 0:
        speedup = r_result.mean_time / python_result.mean_time

    if speedup == speedup:
        faster = "Python faster" if speedup > 1 else "R faster"
        logger.info("Speedup: %.2fx (%s)", speedup, faster)

    return ATTgtBenchmarkResult(
        n_units=config.n_units,
        n_periods=config.n_periods,
        n_groups=config.n_groups,
        n_covariates=config.n_covariates,
        est_method=config.est_method,
        control_group=config.control_group,
        boot=config.boot,
        biters=config.biters,
        xformla=config.xformla,
        python_mean_time=python_result.mean_time,
        python_std_time=python_result.std_time,
        python_min_time=python_result.min_time,
        python_max_time=python_result.max_time,
        python_success=python_result.success,
        python_error=python_result.error,
        r_mean_time=r_result.mean_time if r_result else float("nan"),
        r_std_time=r_result.std_time if r_result else float("nan"),
        r_min_time=r_result.min_time if r_result else float("nan"),
        r_max_time=r_result.max_time if r_result else float("nan"),
        r_success=r_result.success if r_result else False,
        r_error=r_result.error if r_result else "Skipped",
        speedup=speedup,
        n_observations=data_result["n_observations"],
        n_estimates=python_result.n_estimates,
        timestamp=datetime.now().isoformat(),
    )


def run_attgt_benchmark_suite(
    configs: list[ATTgtBenchmarkConfig],
    python_only: bool = False,
) -> list[ATTgtBenchmarkResult]:
    """Run a suite of att_gt benchmark configurations."""
    python_runner = PythonBenchmarkRunner()
    r_runner = None if python_only else (RBenchmarkRunner() if R_DID_AVAILABLE else None)

    if r_runner is None and not python_only:
        logger.warning("R or did package not available, running Python-only benchmarks")

    results = []
    for i, config in enumerate(configs):
        logger.info("Benchmark %d/%d:", i + 1, len(configs))
        result = run_single_attgt_benchmark(config, python_runner, r_runner)
        results.append(result)

    return results


def run_single_ddd_benchmark(
    config: DDDBenchmarkConfig,
    python_runner: PythonBenchmarkRunner,
    r_runner: RBenchmarkRunner | None,
) -> DDDBenchmarkResult:
    """Run a single DDD benchmark configuration."""
    from moderndid import gen_dgp_2periods, gen_dgp_mult_periods

    if config.multi_period:
        logger.info(
            "Generating multi-period DDD data: %d units, dgp_type=%d, panel=%s",
            config.n_units,
            config.dgp_type,
            config.panel,
        )
        dgp_result = gen_dgp_mult_periods(
            n=config.n_units,
            dgp_type=config.dgp_type,
            panel=config.panel,
            random_state=config.random_seed,
        )
    else:
        logger.info(
            "Generating 2-period DDD data: %d units, dgp_type=%d, panel=%s",
            config.n_units,
            config.dgp_type,
            config.panel,
        )
        dgp_result = gen_dgp_2periods(
            n=config.n_units,
            dgp_type=config.dgp_type,
            panel=config.panel,
            random_state=config.random_seed,
        )

    data = dgp_result["data"]
    n_observations = len(data)

    logger.info("Running Python DDD benchmark (%d runs)...", config.n_runs)

    python_result = python_runner.time_ddd(
        data=data,
        multi_period=config.multi_period,
        panel=config.panel,
        est_method=config.est_method,
        control_group=config.control_group,
        base_period=config.base_period,
        boot=config.boot,
        biters=config.biters,
        xformla=config.xformla,
        n_warmup=config.n_warmup,
        n_runs=config.n_runs,
    )

    if python_result.success:
        logger.info("Python: %.4fs (std: %.4fs)", python_result.mean_time, python_result.std_time)
    else:
        logger.error("Python: FAILED - %s", python_result.error)

    r_result = None
    if r_runner is not None and r_runner.is_ddd_available:
        logger.info("Running R DDD benchmark (%d runs)...", config.n_runs)

        r_result = r_runner.time_ddd(
            data=data,
            multi_period=config.multi_period,
            panel=config.panel,
            est_method=config.est_method,
            control_group=config.control_group,
            base_period=config.base_period,
            boot=config.boot,
            biters=config.biters,
            xformla=config.xformla,
            n_warmup=config.n_warmup,
            n_runs=config.n_runs,
        )

        if r_result.success:
            logger.info("R: %.4fs (std: %.4fs)", r_result.mean_time, r_result.std_time)
        else:
            logger.error("R: FAILED - %s", r_result.error)

    speedup = float("nan")
    if python_result.success and r_result is not None and r_result.success and python_result.mean_time > 0:
        speedup = r_result.mean_time / python_result.mean_time

    if speedup == speedup:  # Check for NaN
        faster = "Python faster" if speedup > 1 else "R faster"
        logger.info("Speedup: %.2fx (%s)", speedup, faster)

    return DDDBenchmarkResult(
        n_units=config.n_units,
        dgp_type=config.dgp_type,
        panel=config.panel,
        multi_period=config.multi_period,
        est_method=config.est_method,
        control_group=config.control_group,
        base_period=config.base_period,
        boot=config.boot,
        biters=config.biters,
        xformla=config.xformla,
        python_mean_time=python_result.mean_time,
        python_std_time=python_result.std_time,
        python_min_time=python_result.min_time,
        python_max_time=python_result.max_time,
        python_success=python_result.success,
        python_error=python_result.error,
        r_mean_time=r_result.mean_time if r_result else float("nan"),
        r_std_time=r_result.std_time if r_result else float("nan"),
        r_min_time=r_result.min_time if r_result else float("nan"),
        r_max_time=r_result.max_time if r_result else float("nan"),
        r_success=r_result.success if r_result else False,
        r_error=r_result.error if r_result else "Skipped",
        speedup=speedup,
        n_observations=n_observations,
        n_estimates=python_result.n_estimates,
        timestamp=datetime.now().isoformat(),
    )


def run_ddd_benchmark_suite(
    configs: list[DDDBenchmarkConfig],
    python_only: bool = False,
) -> list[DDDBenchmarkResult]:
    """Run a suite of DDD benchmark configurations."""
    python_runner = PythonBenchmarkRunner()
    r_runner = None if python_only else (RBenchmarkRunner() if R_TRIPLEDIFF_AVAILABLE else None)

    if r_runner is None and not python_only:
        logger.warning("R or triplediff package not available, running Python-only benchmarks")

    results = []
    for i, config in enumerate(configs):
        logger.info("DDD Benchmark %d/%d:", i + 1, len(configs))
        result = run_single_ddd_benchmark(config, python_runner, r_runner)
        results.append(result)

    return results


def main():
    """Run benchmark CLI."""
    parser = argparse.ArgumentParser(
        description="Benchmark Python estimators vs R packages",
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

    # Default to attgt if no subcommand given
    if args.command is None:
        parser.print_help()
        return

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[logging.StreamHandler()],
    )

    storage = ResultStorage(output_dir=args.output_dir)

    if args.command == "attgt":
        if args.suite:
            configs = ATTGT_BENCHMARK_SUITES[args.suite]
            suite_name = f"attgt_{args.suite}"
        else:
            configs = [
                ATTgtBenchmarkConfig(
                    n_units=args.n_units,
                    n_periods=args.n_periods,
                    n_groups=args.n_groups,
                    n_covariates=args.n_covariates,
                    est_method=args.est_method,
                    boot=args.boot,
                    biters=args.biters,
                    n_warmup=args.warmup,
                    n_runs=args.runs,
                    random_seed=args.seed,
                )
            ]
            suite_name = "attgt_custom"

        logger.info("Running att_gt benchmark suite: %s", suite_name)
        logger.info("Number of configurations: %d", len(configs))
        logger.info("Python-only: %s", args.python_only)

        results = run_attgt_benchmark_suite(configs, python_only=args.python_only)

        for r in results:
            status = "OK" if r.python_success else "FAILED"
            msg = f"  {r.n_units} units, {r.n_periods} periods: Python {r.python_mean_time:.4f}s [{status}]"
            if r.r_success:
                msg += f", R {r.r_mean_time:.4f}s, Speedup {r.speedup:.2f}x"
            logger.info(msg)

    elif args.command == "ddd":
        panel = not args.rcs if args.rcs else args.panel

        if args.suite:
            configs = DDD_BENCHMARK_SUITES[args.suite]
            suite_name = f"ddd_{args.suite}"
        else:
            configs = [
                DDDBenchmarkConfig(
                    n_units=args.n_units,
                    dgp_type=args.dgp_type,
                    panel=panel,
                    multi_period=args.multi_period,
                    est_method=args.est_method,
                    boot=args.boot,
                    biters=args.biters,
                    n_warmup=args.warmup,
                    n_runs=args.runs,
                    random_seed=args.seed,
                )
            ]
            suite_name = "ddd_custom"

        logger.info("Running DDD benchmark suite: %s", suite_name)
        logger.info("Number of configurations: %d", len(configs))
        logger.info("Python-only: %s", args.python_only)

        results = run_ddd_benchmark_suite(configs, python_only=args.python_only)

        for r in results:
            status = "OK" if r.python_success else "FAILED"
            mode = "multi-period" if r.multi_period else "2-period"
            data_type = "panel" if r.panel else "RCS"
            msg = f"  {r.n_units} units ({mode}, {data_type}): Python {r.python_mean_time:.4f}s [{status}]"
            if r.r_success:
                msg += f", R {r.r_mean_time:.4f}s, Speedup {r.speedup:.2f}x"
            logger.info(msg)

    csv_filename = storage.generate_filename(suite_name, "csv")
    json_filename = storage.generate_filename(suite_name, "json")

    csv_path = storage.save_csv(results, csv_filename)
    json_path = storage.save_json(results, json_filename)

    logger.info("Results saved to:")
    logger.info("  CSV: %s", csv_path)
    logger.info("  JSON: %s", json_path)


if __name__ == "__main__":
    main()
