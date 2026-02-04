"""Benchmark runners for cont_did continuous treatment DiD estimator."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from benchmark.common.base import BaseBenchmarkRunner, TimingResult
from moderndid import cont_did


def check_r_contdid_available() -> bool:
    """Check if R and the contdid package are available."""
    try:
        result = subprocess.run(
            ["R", "--vanilla", "--quiet"],
            input='library(contdid); library(jsonlite); cat("OK")',
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return "OK" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


R_CONTDID_AVAILABLE = check_r_contdid_available()


class ContDIDPythonRunner(BaseBenchmarkRunner):
    """Benchmark runner for Python cont_did implementation."""

    def time_cont_did(
        self,
        data: pl.DataFrame,
        target_parameter: str = "level",
        aggregation: str = "dose",
        dose_est_method: str = "parametric",
        degree: int = 3,
        num_knots: int = 0,
        boot: bool = False,
        biters: int = 100,
        n_warmup: int = 1,
        n_runs: int = 5,
        random_state: int | None = None,
    ) -> TimingResult:
        """Time Python cont_did estimation."""

        def run_estimation():
            return cont_did(
                data=data,
                yname="Y",
                tname="time_period",
                idname="id",
                gname="G",
                dname="D",
                target_parameter=target_parameter,
                aggregation=aggregation,
                dose_est_method=dose_est_method,
                degree=degree,
                num_knots=num_knots,
                boot=boot,
                biters=biters,
                random_state=random_state,
            )

        def get_n_estimates(result):
            if hasattr(result, "dose") and result.dose is not None:
                return len(result.dose)
            if hasattr(result, "att_d") and result.att_d is not None:
                return len(result.att_d)
            return 1

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


class ContDIDRRunner(BaseBenchmarkRunner):
    """Benchmark runner for R contdid package implementation."""

    def __init__(self):
        self._r_available = R_CONTDID_AVAILABLE

    @property
    def is_available(self) -> bool:
        """Check if R contdid runner is available."""
        return self._r_available

    def time_cont_did(
        self,
        data: pl.DataFrame,
        target_parameter: str = "level",
        aggregation: str = "dose",
        dose_est_method: str = "parametric",
        degree: int = 3,
        num_knots: int = 0,
        boot: bool = False,
        biters: int = 100,
        n_warmup: int = 1,
        n_runs: int = 5,
        random_state: int | None = None,
    ) -> TimingResult:
        """Time R contdid package estimation."""
        if not self._r_available:
            return TimingResult(
                mean_time=float("nan"),
                std_time=float("nan"),
                min_time=float("nan"),
                max_time=float("nan"),
                times=[],
                n_estimates=0,
                success=False,
                error="R or contdid package not available",
            )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            data.write_csv(f.name)
            data_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            result_path = f.name

        bstrap_str = "TRUE" if boot else "FALSE"
        seed_str = f"set.seed({random_state})" if random_state is not None else ""

        r_script = f"""
library(contdid)
library(jsonlite)

{seed_str}

data <- read.csv("{data_path}")

run_estimation <- function() {{
    suppressWarnings(cont_did(
        yname = "Y",
        tname = "time_period",
        idname = "id",
        gname = "G",
        dname = "D",
        data = data,
        target_parameter = "{target_parameter}",
        aggregation = "{aggregation}",
        treatment_type = "continuous",
        control_group = "notyettreated",
        base_period = "varying",
        dose_est_method = "{dose_est_method}",
        degree = {degree},
        num_knots = {num_knots},
        bstrap = {bstrap_str},
        biters = {biters}
    ))
}}

for (i in 1:{n_warmup}) {{
    gc()
    run_estimation()
}}

times <- numeric({n_runs})
n_estimates <- 0

for (i in 1:{n_runs}) {{
    gc()
    start_time <- Sys.time()
    result <- run_estimation()
    end_time <- Sys.time()
    times[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
    if (!is.null(result$dose)) {{
        n_estimates <- length(result$dose)
    }} else if (!is.null(result$att_d)) {{
        n_estimates <- length(result$att_d)
    }} else {{
        n_estimates <- 1
    }}
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
            proc = subprocess.run(
                ["R", "--vanilla", "--quiet"],
                input=r_script,
                capture_output=True,
                text=True,
                timeout=3600,
                check=False,
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
            if std_time is None or (isinstance(std_time, float) and std_time != std_time):
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
