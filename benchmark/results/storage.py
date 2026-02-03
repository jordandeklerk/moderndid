"""Result storage utilities for benchmark outputs."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class BaseBenchmarkResult:
    """Base container for benchmark results."""

    n_units: int
    est_method: str
    boot: bool
    biters: int
    xformla: str

    python_mean_time: float
    python_std_time: float
    python_min_time: float
    python_max_time: float
    python_success: bool
    python_error: str | None

    r_mean_time: float
    r_std_time: float
    r_min_time: float
    r_max_time: float
    r_success: bool
    r_error: str | None

    speedup: float
    n_observations: int
    n_estimates: int
    timestamp: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ATTgtBenchmarkResult(BaseBenchmarkResult):
    """Container for att_gt benchmark result."""

    n_periods: int = 0
    n_groups: int = 0
    n_covariates: int = 0
    control_group: str = "nevertreated"


@dataclass
class DDDBenchmarkResult(BaseBenchmarkResult):
    """Container for DDD benchmark result."""

    dgp_type: int = 1
    panel: bool = True
    multi_period: bool = False
    control_group: str = "nevertreated"
    base_period: str = "varying"


class ResultStorage:
    """Handles saving and loading benchmark results."""

    def __init__(self, output_dir: str | Path = "benchmark/output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_csv(self, results: list[BaseBenchmarkResult], filename: str) -> Path:
        """Save results to CSV file."""
        filepath = self.output_dir / filename
        if not results:
            return filepath

        fieldnames = list(results[0].to_dict().keys())

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result.to_dict())

        return filepath

    def save_json(self, results: list[BaseBenchmarkResult], filename: str) -> Path:
        """Save results to JSON file."""
        filepath = self.output_dir / filename

        data = {
            "timestamp": datetime.now().isoformat(),
            "n_results": len(results),
            "results": [r.to_dict() for r in results],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return filepath

    def load_csv(self, filename: str) -> list[dict]:
        """Load results from CSV file."""
        filepath = self.output_dir / filename
        results = []

        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)

        return results

    def load_json(self, filename: str) -> dict:
        """Load results from JSON file."""
        filepath = self.output_dir / filename

        with open(filepath, encoding="utf-8") as f:
            return json.load(f)

    def generate_filename(self, suite_name: str | None = None, extension: str = "csv") -> str:
        """Generate a timestamped filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if suite_name:
            return f"benchmark_{suite_name}_{timestamp}.{extension}"
        return f"benchmark_{timestamp}.{extension}"
