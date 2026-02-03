"""Benchmark runners for ddd (Python) vs triplediff (R) package."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from benchmark.common.base import BaseBenchmarkRunner, TimingResult
from moderndid import ddd


def check_r_triplediff_available() -> bool:
    """Check if R and the triplediff package are available."""
    try:
        result = subprocess.run(
            ["R", "--vanilla", "--quiet"],
            input='library(triplediff); library(jsonlite); cat("OK")',
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return "OK" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


R_TRIPLEDIFF_AVAILABLE = check_r_triplediff_available()


class DDDPythonRunner(BaseBenchmarkRunner):
    """Benchmark runner for Python ddd implementation."""

    def time_ddd(
        self,
        data: pl.DataFrame,
        multi_period: bool = False,
        panel: bool = True,
        est_method: str = "dr",
        control_group: str = "nevertreated",
        base_period: str = "varying",
        boot: bool = False,
        biters: int = 100,
        xformla: str = "~ cov1 + cov2 + cov3 + cov4",
        n_warmup: int = 1,
        n_runs: int = 5,
    ) -> TimingResult:
        """Time Python DDD estimation."""
        gname = "group" if multi_period else "state"

        def run_estimation():
            return ddd(
                data=data,
                yname="y",
                tname="time",
                idname="id",
                gname=gname,
                pname="partition",
                xformla=xformla,
                est_method=est_method,
                control_group=control_group,
                base_period=base_period,
                panel=panel,
                boot=boot,
                biters=biters if boot else 0,
            )

        def get_n_estimates(result):
            if hasattr(result, "att") and hasattr(result.att, "__len__"):
                return len(result.att)
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

        except (ValueError, RuntimeError, KeyError) as e:
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


class DDDRRunner(BaseBenchmarkRunner):
    """Benchmark runner for R triplediff package ddd implementation."""

    def __init__(self):
        self._r_available = R_TRIPLEDIFF_AVAILABLE

    @property
    def is_available(self) -> bool:
        """Check if R triplediff runner is available."""
        return self._r_available

    def time_ddd(
        self,
        data: pl.DataFrame,
        multi_period: bool = False,
        panel: bool = True,
        est_method: str = "dr",
        control_group: str = "nevertreated",
        base_period: str = "varying",
        boot: bool = False,
        biters: int = 100,
        xformla: str = "~ cov1 + cov2 + cov3 + cov4",
        n_warmup: int = 1,
        n_runs: int = 5,
    ) -> TimingResult:
        """Time R triplediff package ddd estimation."""
        if not self._r_available:
            return TimingResult(
                mean_time=float("nan"),
                std_time=float("nan"),
                min_time=float("nan"),
                max_time=float("nan"),
                times=[],
                n_estimates=0,
                success=False,
                error="R or triplediff package not available",
            )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            data.write_csv(f.name)
            data_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            result_path = f.name

        gname = "group" if multi_period else "state"
        panel_str = "TRUE" if panel else "FALSE"
        boot_str = "TRUE" if boot else "FALSE"

        # Only include biters if boot is TRUE
        biters_arg = f",\n        biters = {biters}" if boot else ""

        r_script = f"""
library(triplediff)
library(jsonlite)

data <- read.csv("{data_path}")

run_estimation <- function() {{
    ddd(
        yname = "y",
        tname = "time",
        idname = "id",
        gname = "{gname}",
        pname = "partition",
        xformla = {xformla},
        data = data,
        est_method = "{est_method}",
        control_group = "{control_group}",
        base_period = "{base_period}",
        panel = {panel_str},
        boot = {boot_str}{biters_arg}
    )
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
    if (is.null(result$ATT)) {{
        n_estimates <- 1
    }} else if (length(result$ATT) > 1) {{
        n_estimates <- length(result$ATT)
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
                timeout=600,
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
