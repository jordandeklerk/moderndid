"""Save comparison results together with the environment needed to interpret them."""

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from benchmarks.references import PACKAGES, package_version, r_environment, r_version

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "R_DATATABLE_NUM_THREADS",
)
TOLERANCES = {"absolute": 1e-6, "estimate_relative": 1e-5, "se_relative": 1e-3}


def git_output(root, *arguments):
    """Read Git metadata without refreshing or otherwise writing the index."""
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout if result.returncode == 0 else None


def source_fingerprint(root):
    """Hash the paths and contents of tracked source and of source added in the working tree."""
    tracked = git_output(root, "ls-files", "-z", "--cached", "--", "moderndid", "benchmarks")
    untracked = git_output(root, "ls-files", "-z", "--others", "--exclude-standard", "--", "moderndid", "benchmarks")
    if tracked is None:
        return {"sha256": None, "tracked_sha256": None, "files": [], "includes_untracked": True}
    tracked_paths = {name for name in tracked.split("\0") if name.endswith(".py")}
    paths = tracked_paths | {name for name in (untracked or "").split("\0") if name.endswith(".py")}

    def digest(names):
        fingerprint = hashlib.sha256()
        for name in sorted(names):
            path = root / name
            fingerprint.update(name.encode() + b"\0")
            fingerprint.update(path.read_bytes() if path.is_file() else b"<deleted>")
            fingerprint.update(b"\0")
        return fingerprint.hexdigest()

    return {
        "sha256": digest(paths),
        "tracked_sha256": digest(tracked_paths),
        "files": sorted(paths),
        "includes_untracked": True,
    }


def cpu_name():
    """Read the CPU model on macOS and Linux and fall back to platform metadata."""
    if platform.system() == "Darwin":
        # Since managed environments often leave /usr/sbin off PATH, the system location comes first.
        sysctl = shutil.which("sysctl", path="/usr/sbin") or shutil.which("sysctl")
        if sysctl is not None:
            try:
                result = subprocess.run(
                    [sysctl, "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
            except OSError:
                pass
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or platform.machine()


def capture_environment(estimators, root=None):
    """Capture the software and hardware behind one report with its source and thread settings."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    versions = {}
    for package in (
        "moderndid",
        "numpy",
        "polars",
        "scipy",
        "statsmodels",
        "numba",
        "llvmlite",
        "pyarrow",
        "narwhals",
        "formulaic",
        "pandas",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    commit = git_output(root, "rev-parse", "HEAD")
    status = git_output(root, "status", "--porcelain")
    r_threads = r_environment()
    r_executable = shutil.which("Rscript")
    r_libraries = []
    if r_executable is not None:
        try:
            libraries = subprocess.run(
                [r_executable, "--vanilla", "-e", 'cat(.libPaths(), sep = "\\n")'],
                env=r_threads,
                capture_output=True,
                text=True,
                check=False,
            )
            if libraries.returncode == 0:
                r_libraries = libraries.stdout.strip().splitlines()
        except OSError:
            pass
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu_name(),
        "logical_cpus": os.cpu_count(),
        "package_versions": versions,
        "r_version": r_version(),
        "r_executable": r_executable,
        "r_library_paths": r_libraries,
        "r_packages": {
            package: package_version(package) for package in sorted({PACKAGES[item] for item in estimators})
        },
        "thread_environment": {
            "python": {name: os.environ.get(name) for name in THREAD_VARIABLES},
            "R": {name: r_threads.get(name) for name in THREAD_VARIABLES},
        },
        "git": {
            "commit": commit.strip() if commit is not None else None,
            "dirty": bool(status) if status is not None else None,
        },
        "source": source_fingerprint(root),
    }


def make_report(rows, arguments, environment):
    """Combine measured rows and provenance into the versioned report schema."""
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "environment": environment,
        "configuration": {
            "estimators": list(arguments.estimators),
            "workloads": list(arguments.workloads),
            "aggregations": list(arguments.aggregations),
            "repeats": arguments.repeats,
            "warmup_calls": 1,
            "seed": 42,
            "tolerances": TOLERANCES,
            "command": [sys.executable, "-m", "benchmarks.compare", *sys.argv[1:]],
        },
        "summary": dict(Counter(row["status"] for row in rows)),
        "results": rows,
    }


def json_safe(value):
    """Represent unavailable numeric results as JSON null at any nesting depth."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_report(report, output=None, csv_path=None):
    """Write strict JSON and a flat CSV without starting any benchmark calls."""
    report = json_safe(report)
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if csv_path is not None:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = report["results"]
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: json.dumps(value, allow_nan=False) if isinstance(value, (dict, list)) else value
                        for key, value in row.items()
                    }
                )
