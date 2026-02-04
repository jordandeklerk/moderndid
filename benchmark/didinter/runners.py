"""Benchmark runners for did_multiplegt intertemporal treatment effects estimator."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from benchmark.common.base import BaseBenchmarkRunner, TimingResult
from moderndid import did_multiplegt


def check_r_didmultiplegt_available() -> bool:
    """Check if R and the DIDmultiplegtDYN package are available."""
    try:
        result = subprocess.run(
            ["R", "--vanilla", "--quiet"],
            input='library(DIDmultiplegtDYN); library(jsonlite); cat("OK")',
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return "OK" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


R_DIDINTER_AVAILABLE = check_r_didmultiplegt_available()


def generate_didinter_data(
    n_units: int = 500,
    n_periods: int = 10,
    treatment_effect: float = 2.0,
    switcher_fraction: float = 0.4,
    random_seed: int = 42,
) -> pl.DataFrame:
    """Generate panel data for did_multiplegt benchmarks."""
    rng = np.random.default_rng(random_seed)

    n_switchers = int(n_units * switcher_fraction)
    n_never_switchers = n_units - n_switchers

    records = []

    for i in range(n_never_switchers):
        unit_id = i + 1
        unit_fe = rng.normal(0, 1)

        for t in range(1, n_periods + 1):
            time_fe = 0.5 * t
            y = unit_fe + time_fe + rng.normal(0, 0.5)
            records.append(
                {
                    "id": unit_id,
                    "time": t,
                    "D": 0,
                    "Y": y,
                }
            )

    switch_times = rng.integers(3, n_periods - 1, size=n_switchers)

    for i in range(n_switchers):
        unit_id = n_never_switchers + i + 1
        unit_fe = rng.normal(0, 1)
        switch_time = switch_times[i]
        treat_level = rng.integers(1, 4)

        for t in range(1, n_periods + 1):
            time_fe = 0.5 * t

            if t < switch_time:
                d = 0
                y = unit_fe + time_fe + rng.normal(0, 0.5)
            else:
                d = treat_level
                exposure = t - switch_time + 1
                effect = treatment_effect * d * min(exposure, 3) / 3
                y = unit_fe + time_fe + effect + rng.normal(0, 0.5)

            records.append(
                {
                    "id": unit_id,
                    "time": t,
                    "D": d,
                    "Y": y,
                }
            )

    return pl.DataFrame(records)


class DIDInterPythonRunner(BaseBenchmarkRunner):
    """Benchmark runner for Python did_multiplegt implementation."""

    def time_did_multiplegt(
        self,
        data: pl.DataFrame,
        effects: int = 3,
        placebo: int = 2,
        normalized: bool = False,
        boot: bool = False,
        biters: int = 100,
        n_warmup: int = 1,
        n_runs: int = 5,
        random_state: int | None = None,
    ) -> TimingResult:
        """Time Python did_multiplegt estimation."""

        def run_estimation():
            return did_multiplegt(
                data=data,
                yname="Y",
                tname="time",
                idname="id",
                dname="D",
                effects=effects,
                placebo=placebo,
                normalized=normalized,
                boot=boot,
                biters=biters,
                random_state=random_state,
            )

        def get_n_estimates(result):
            n = 0
            if hasattr(result, "effects") and result.effects is not None:
                n += len(result.effects.estimates)
            if hasattr(result, "placebos") and result.placebos is not None:
                n += len(result.placebos.estimates)
            return max(n, 1)

        try:
            for _ in range(n_warmup):
                self.gc_collect()
                run_estimation()

            times = []
            n_estimates = 0

            for _ in range(n_runs):
                self.gc_collect()
                elapsed, result = self.time_execution(run_estimation)
                times.append(elapsed)
                n_estimates = get_n_estimates(result)

            return TimingResult(
                mean_time=float(np.mean(times)),
                std_time=float(np.std(times)),
                min_time=float(np.min(times)),
                max_time=float(np.max(times)),
                times=times,
                n_estimates=n_estimates,
                success=True,
            )

        except (ValueError, RuntimeError, KeyError, IndexError) as e:
            return TimingResult(
                mean_time=float("nan"),
                std_time=float("nan"),
                min_time=float("nan"),
                max_time=float("nan"),
                times=[],
                n_estimates=0,
                success=False,
                error=str(e),
            )


class DIDInterRRunner(BaseBenchmarkRunner):
    """Benchmark runner for R DIDmultiplegtDYN package implementation."""

    def __init__(self):
        self._r_available = R_DIDINTER_AVAILABLE

    @property
    def is_available(self) -> bool:
        """Check if R DIDmultiplegtDYN runner is available."""
        return self._r_available

    def time_did_multiplegt(
        self,
        data: pl.DataFrame,
        effects: int = 3,
        placebo: int = 2,
        normalized: bool = False,
        boot: bool = False,
        biters: int = 100,
        n_warmup: int = 1,
        n_runs: int = 5,
        random_state: int | None = None,
    ) -> TimingResult:
        """Time R DIDmultiplegtDYN package estimation."""
        if not self._r_available:
            return TimingResult(
                mean_time=float("nan"),
                std_time=float("nan"),
                min_time=float("nan"),
                max_time=float("nan"),
                times=[],
                n_estimates=0,
                success=False,
                error="R or DIDmultiplegtDYN package not available",
            )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            data.write_csv(f.name)
            data_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            result_path = f.name

        normalized_str = "TRUE" if normalized else "FALSE"
        bootstrap_str = str(biters) if boot else "NULL"
        seed_str = f"set.seed({random_state})" if random_state is not None else ""

        r_script = f"""
# Memory optimization settings
options(rgl.useNULL = TRUE)
options(warn = -1)

# Clear environment and force garbage collection
rm(list = ls(all.names = TRUE))
gc(verbose = FALSE, full = TRUE, reset = TRUE)

# Load packages with minimal overhead
suppressPackageStartupMessages({{
    library(DIDmultiplegtDYN)
    library(jsonlite)
    library(data.table)
}})

{seed_str}

# Use data.table for efficient CSV reading (less memory than read.csv)
data <- as.data.frame(fread("{data_path}", data.table = FALSE))
gc(verbose = FALSE, full = TRUE)

times <- numeric({n_runs})
n_estimates <- 0

# Skip warmup to reduce memory pressure - go straight to timed runs
for (i in 1:{n_runs}) {{
    # Aggressive garbage collection before each run
    gc(verbose = FALSE, full = TRUE, reset = TRUE)

    start_time <- Sys.time()
    result <- suppressWarnings(did_multiplegt_dyn(
        df = data,
        outcome = "Y",
        group = "id",
        time = "time",
        treatment = "D",
        effects = {effects},
        placebo = {placebo},
        normalized = {normalized_str},
        bootstrap = {bootstrap_str},
        graph_off = TRUE
    ))
    end_time <- Sys.time()
    times[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))

    n_eff <- if (!is.null(result$effects)) nrow(result$effects) else 0
    n_plac <- if (!is.null(result$placebos)) nrow(result$placebos) else 0
    n_estimates <- n_eff + n_plac

    # Immediately free result memory
    rm(result)
    gc(verbose = FALSE, full = TRUE, reset = TRUE)
}}

out <- list(
    mean_time = mean(times),
    std_time = sd(times),
    min_time = min(times),
    max_time = max(times),
    times = as.list(times),
    n_estimates = n_estimates,
    success = TRUE
)

writeLines(toJSON(out, auto_unbox = TRUE), "{result_path}")
"""

        try:
            import os

            env = os.environ.copy()
            env["R_MAX_VSIZE"] = "32Gb"

            proc = subprocess.run(
                ["R", "--vanilla", "--quiet"],
                input=r_script,
                capture_output=True,
                text=True,
                timeout=3600,
                check=False,
                env=env,
            )

            if proc.returncode != 0:
                return TimingResult(
                    mean_time=float("nan"),
                    std_time=float("nan"),
                    min_time=float("nan"),
                    max_time=float("nan"),
                    times=[],
                    n_estimates=0,
                    success=False,
                    error=f"R script failed: {proc.stderr}",
                )

            with open(result_path, encoding="utf-8") as f:
                r_result = json.load(f)

            std_time = r_result["std_time"]
            if std_time is None or std_time == "NA" or (isinstance(std_time, float) and std_time != std_time):
                std_time = 0.0

            return TimingResult(
                mean_time=float(r_result["mean_time"]),
                std_time=float(std_time),
                min_time=float(r_result["min_time"]),
                max_time=float(r_result["max_time"]),
                times=[float(t) for t in r_result["times"]],
                n_estimates=int(r_result["n_estimates"]),
                success=bool(r_result["success"]),
            )

        except subprocess.TimeoutExpired:
            return TimingResult(
                mean_time=float("nan"),
                std_time=float("nan"),
                min_time=float("nan"),
                max_time=float("nan"),
                times=[],
                n_estimates=0,
                success=False,
                error="R script timed out",
            )

        except (OSError, json.JSONDecodeError, KeyError) as e:
            return TimingResult(
                mean_time=float("nan"),
                std_time=float("nan"),
                min_time=float("nan"),
                max_time=float("nan"),
                times=[],
                n_estimates=0,
                success=False,
                error=str(e),
            )

        finally:
            Path(data_path).unlink(missing_ok=True)
            Path(result_path).unlink(missing_ok=True)
