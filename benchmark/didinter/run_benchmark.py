"""CLI entry point for did_multiplegt benchmarks."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime

from benchmark.common.storage import ResultStorage
from benchmark.didinter.config import DIDINTER_BENCHMARK_SUITES, DIDInterBenchmarkConfig
from benchmark.didinter.runners import (
    R_DIDINTER_AVAILABLE,
    DIDInterPythonRunner,
    DIDInterRRunner,
    generate_didinter_data,
)
from benchmark.didinter.storage import DIDInterBenchmarkResult

logger = logging.getLogger(__name__)


def run_single_benchmark(
    config: DIDInterBenchmarkConfig,
    python_runner: DIDInterPythonRunner,
    r_runner: DIDInterRRunner | None,
    python_only: bool = False,
) -> DIDInterBenchmarkResult:
    """Run a single benchmark configuration."""
    logger.info("Generating data: %d units, %d periods", config.n_units, config.n_periods)

    data = generate_didinter_data(
        n_units=config.n_units,
        n_periods=config.n_periods,
        random_seed=config.random_seed,
    )
    n_observations = len(data)

    logger.info("Running Python benchmark (%d runs)...", config.n_runs)

    py_result = python_runner.time_did_multiplegt(
        data=data,
        effects=config.effects,
        placebo=config.placebo,
        normalized=config.normalized,
        boot=config.boot,
        biters=config.biters,
        n_warmup=config.n_warmup,
        n_runs=config.n_runs,
        random_state=config.random_seed,
    )

    if py_result.success:
        logger.info("Python: %.4fs (std: %.4fs)", py_result.mean_time, py_result.std_time)

    r_result = None
    if not python_only and r_runner is not None and r_runner.is_available:
        logger.info("Running R benchmark (%d runs)...", config.n_runs)

        r_result = r_runner.time_did_multiplegt(
            data=data,
            effects=config.effects,
            placebo=config.placebo,
            normalized=config.normalized,
            boot=config.boot,
            biters=config.biters,
            n_warmup=config.n_warmup,
            n_runs=config.n_runs,
            random_state=config.random_seed,
        )

        if r_result.success:
            logger.info("R: %.4fs (std: %.4fs)", r_result.mean_time, r_result.std_time)

    speedup = float("nan")
    if py_result.success and r_result is not None and r_result.success:
        speedup = r_result.mean_time / py_result.mean_time
        logger.info("Speedup: %.2fx (Python faster)", speedup)

    return DIDInterBenchmarkResult(
        n_units=config.n_units,
        est_method="did_multiplegt",
        boot=config.boot,
        biters=config.biters,
        xformla="~1",
        python_mean_time=py_result.mean_time,
        python_std_time=py_result.std_time,
        python_min_time=py_result.min_time,
        python_max_time=py_result.max_time,
        python_success=py_result.success,
        python_error=py_result.error,
        r_mean_time=r_result.mean_time if r_result else float("nan"),
        r_std_time=r_result.std_time if r_result else float("nan"),
        r_min_time=r_result.min_time if r_result else float("nan"),
        r_max_time=r_result.max_time if r_result else float("nan"),
        r_success=r_result.success if r_result else False,
        r_error=r_result.error if r_result else "Skipped",
        speedup=speedup,
        n_observations=n_observations,
        n_estimates=py_result.n_estimates,
        timestamp=datetime.now().isoformat(),
        n_periods=config.n_periods,
        effects=config.effects,
        placebo=config.placebo,
        normalized=config.normalized,
    )


def main():
    """Run the benchmark CLI."""
    parser = argparse.ArgumentParser(description="Run did_multiplegt benchmarks")

    parser.add_argument(
        "--suite",
        choices=list(DIDINTER_BENCHMARK_SUITES.keys()),
        help="Predefined benchmark suite",
    )
    parser.add_argument("--n-units", type=int, default=500, help="Number of units")
    parser.add_argument("--n-periods", type=int, default=10, help="Number of time periods")
    parser.add_argument("--effects", type=int, default=3, help="Number of effects to estimate")
    parser.add_argument("--placebo", type=int, default=2, help="Number of placebos to estimate")
    parser.add_argument("--normalized", action="store_true", help="Use normalized effects")
    parser.add_argument("--boot", action="store_true", help="Enable bootstrap")
    parser.add_argument("--biters", type=int, default=100, help="Bootstrap iterations")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs")
    parser.add_argument("--runs", type=int, default=5, help="Timed runs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--python-only", action="store_true", help="Skip R benchmarks")
    parser.add_argument("--output-dir", default="benchmark/output", help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[logging.StreamHandler()],
    )

    if args.suite:
        configs = DIDINTER_BENCHMARK_SUITES[args.suite]
        suite_name = f"didinter_{args.suite}"
    else:
        configs = [
            DIDInterBenchmarkConfig(
                n_units=args.n_units,
                n_periods=args.n_periods,
                effects=args.effects,
                placebo=args.placebo,
                normalized=args.normalized,
                boot=args.boot,
                biters=args.biters,
                n_warmup=args.warmup,
                n_runs=args.runs,
                random_seed=args.seed,
            )
        ]
        suite_name = "didinter_custom"

    logger.info("Running did_multiplegt benchmark suite: %s", suite_name)
    logger.info("Number of configurations: %d", len(configs))
    logger.info("Python-only: %s", args.python_only)

    if not args.python_only and not R_DIDINTER_AVAILABLE:
        logger.warning("R or DIDmultiplegtDYN package not available, running Python-only")

    python_runner = DIDInterPythonRunner()
    r_runner = DIDInterRRunner() if not args.python_only else None

    results = []
    for i, config in enumerate(configs):
        logger.info("Benchmark %d/%d:", i + 1, len(configs))
        result = run_single_benchmark(
            config=config,
            python_runner=python_runner,
            r_runner=r_runner,
            python_only=args.python_only,
        )
        results.append(result)

        status = "OK" if result.python_success else f"FAIL: {result.python_error}"
        msg = f"  {config.n_units} units, {config.n_periods} periods: Python {result.python_mean_time:.4f}s [{status}]"
        if result.r_success:
            msg += f", R {result.r_mean_time:.4f}s, Speedup {result.speedup:.2f}x"
        logger.info(msg)

    storage = ResultStorage(output_dir=args.output_dir)
    csv_filename = storage.generate_filename(suite_name, "csv")
    json_filename = storage.generate_filename(suite_name, "json")

    csv_path = storage.save_csv(results, csv_filename)
    json_path = storage.save_json(results, json_filename)

    logger.info("Results saved to:")
    logger.info("  CSV: %s", csv_path)
    logger.info("  JSON: %s", json_path)


if __name__ == "__main__":
    main()
