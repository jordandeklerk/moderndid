"""Compare moderndid with the R implementations of the same estimators."""

import argparse
import math
import platform
import statistics
import time

import numpy as np
import polars as pl
from prettytable import PrettyTable

import moderndid
from benchmarks.cases import WORKLOADS
from benchmarks.common import aggregation_type, check_estimate, make_aggregation, make_estimator, resolve_workload
from benchmarks.references import PACKAGES, package_version, r_version, run_reference

ESTIMATORS = ("att_gt", "aggte", "ddd", "agg_ddd", "cont_did", "did_multiplegt", "drdid", "ipwdid", "ordid")
AGGREGATIONS = ("simple", "group", "dynamic")
AGGREGATES = {"aggte": "att_gt", "agg_ddd": "ddd"}


def parse_args():
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimators", nargs="+", choices=ESTIMATORS, default=ESTIMATORS)
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=("baseline",),
        help="Workload profiles from benchmarks/cases.py, or ipw, reg, and normalized.",
    )
    parser.add_argument("--aggregations", nargs="+", choices=AGGREGATIONS, default=AGGREGATIONS)
    parser.add_argument("--repeats", type=int, default=5, help="Timed calls on each side after one warm-up.")
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        parser.error("--repeats must be at least 1")
    # Since bootstrap draws differ between the two languages, their standard errors would never agree.
    if "bootstrap" in arguments.workloads:
        parser.error("the comparison uses analytical standard errors, so the bootstrap workload is not available")
    unknown = set(arguments.workloads) - set(WORKLOADS) - {"ipw", "reg", "normalized"}
    if unknown:
        parser.error(f"unknown workloads: {', '.join(sorted(unknown))}")
    return arguments


def cases(arguments):
    """List the estimator, workload, and aggregation of every comparison to run."""
    for estimator in arguments.estimators:
        for profile in arguments.workloads:
            if estimator in AGGREGATES:
                for aggregation in arguments.aggregations:
                    yield estimator, profile, aggregation
            else:
                yield estimator, profile, None


def prepare_python(estimator, profile, aggregation):
    """Return the moderndid call for one case and the data R needs to repeat it."""
    if estimator in AGGREGATES:
        fitted = AGGREGATES[estimator]
        data = make_estimator(fitted, profile).keywords["data"]
        return make_aggregation(fitted, profile, aggregation), data
    call = make_estimator(estimator, profile)
    return call, call.keywords["data"]


def reference_options(estimator, profile, aggregation):
    """Collect the settings the R call needs to match the moderndid call."""
    fitted = AGGREGATES.get(estimator, estimator)
    workload = resolve_workload(fitted, profile)
    return {
        "covariates": [f"x{index + 1}" for index in range(workload.n_covariates)],
        "est_method": profile if profile in {"ipw", "reg"} else "dr",
        "normalized": profile == "normalized",
        "biters": workload.biters,
        "aggregation": aggregation_type(fitted, aggregation),
    }


def aggregation_rows(result, labels, label_estimates, label_errors):
    """Return the overall effect of an aggregation followed by its effects at each label."""
    keys, estimates, errors = ["overall"], [result.overall_att], [result.overall_se]
    if labels is not None:
        keys += [f"{label:.0f}" for label in labels]
        estimates += list(label_estimates)
        errors += list(label_errors)
    return keys, estimates, errors


def python_rows(estimator, result):
    """Return the keys, estimates, and standard errors of a moderndid result."""
    if estimator in {"att_gt", "ddd"}:
        estimates = result.att_gt if estimator == "att_gt" else result.att
        errors = result.se_gt if estimator == "att_gt" else result.se
        keys = [f"{group:.0f} {period:.0f}" for group, period in zip(result.groups, result.times, strict=True)]
        return keys, list(estimates), list(errors)
    if estimator == "aggte":
        return aggregation_rows(result, result.event_times, result.att_by_event, result.se_by_event)
    if estimator == "agg_ddd":
        return aggregation_rows(result, result.egt, result.att_egt, result.se_egt)
    if estimator == "cont_did":
        return (
            ["ATT", "ACRT"],
            [result.overall_att, result.overall_acrt],
            [result.overall_att_se, result.overall_acrt_se],
        )
    if estimator in {"drdid", "ipwdid", "ordid"}:
        return ["ATT"], [result.att], [result.se]
    effects, placebos = result.effects, result.placebos
    keys = [f"effect {index + 1}" for index in range(len(effects.estimates))]
    keys += [f"placebo {index + 1}" for index in range(len(placebos.estimates))]
    estimates = list(effects.estimates) + list(placebos.estimates)
    errors = list(effects.std_errors) + list(placebos.std_errors)
    return keys, estimates, errors


def as_float(value):
    """Convert a JSON value to a float, reading R's missing values as NaN."""
    if value is None or value == "NA":
        return math.nan
    return float(value)


def largest_differences(python, reference, compare_errors, skip_errors=()):
    """Return the largest estimate and standard-error gaps and how many cells both sides report."""
    keys, estimates, errors = python
    lookup = {
        str(key): (as_float(estimate), as_float(error))
        for key, estimate, error in zip(reference["keys"], reference["estimates"], reference["std_errors"], strict=True)
    }
    gaps, error_gaps = [], []
    for key, estimate, error in zip(keys, estimates, errors, strict=True):
        if key not in lookup:
            continue
        reference_estimate, reference_error = lookup[key]
        if math.isfinite(estimate) and math.isfinite(reference_estimate):
            gaps.append((abs(estimate - reference_estimate), abs(reference_estimate)))
        if compare_errors and key not in skip_errors and math.isfinite(error) and math.isfinite(reference_error):
            error_gaps.append((abs(error - reference_error), abs(reference_error)))
    return gaps, error_gaps


def within(gaps, relative):
    """Check every gap against an absolute floor plus a relative tolerance."""
    return all(gap <= 1e-6 + relative * size for gap, size in gaps)


def format_seconds(seconds):
    """Write a duration with a unit that keeps three significant figures."""
    if seconds < 1:
        return f"{seconds * 1000:.3g} ms"
    return f"{seconds:.3g} s"


def run_case(estimator, profile, aggregation, repeats):
    """Time moderndid and R on one case and measure how far their results are apart."""
    label = f"{estimator} ({aggregation})" if aggregation else estimator
    row = {"case": label, "workload": profile}
    package = PACKAGES[estimator]
    try:
        call, data = prepare_python(estimator, profile, aggregation)
    except ValueError as error:
        return {**row, "status": f"skipped, {error}"}
    result = call()
    if estimator not in AGGREGATES:
        check_estimate(estimator, result)
    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        seconds.append(time.perf_counter() - start)
    row["moderndid"] = statistics.median(seconds)
    if package_version(package) is None:
        return {**row, "status": f"R package {package} not installed"}
    try:
        reference = run_reference(estimator, data, reference_options(estimator, profile, aggregation), repeats)
    except RuntimeError as error:
        return {**row, "status": f"R failed, {error}"}
    row["R"] = statistics.median(reference["seconds"])
    compare_errors = estimator != "cont_did"
    # Since triplediff matches some cohorts to the wrong units in the overall standard error of its group
    # aggregation, the comparison leaves that number out.
    skip_errors = {"overall"} if estimator == "agg_ddd" and aggregation == "group" else set()
    gaps, error_gaps = largest_differences(python_rows(estimator, result), reference, compare_errors, skip_errors)
    row["cells"] = len(gaps)
    row["estimate"] = max((gap for gap, _ in gaps), default=math.nan)
    row["error"] = max((gap for gap, _ in error_gaps), default=math.nan) if compare_errors else None
    if not gaps:
        row["status"] = "no shared cells"
    elif within(gaps, 1e-5) and within(error_gaps, 1e-3):
        row["status"] = "match"
    else:
        row["status"] = "differs"
    return row


def print_environment(estimators):
    """Print the software versions behind the comparison."""
    print(f"Python {platform.python_version()} on {platform.platform()}")
    print(f"moderndid {moderndid.__version__}, numpy {np.__version__}, polars {pl.__version__}")
    print(r_version() or "R is not installed")
    for package in sorted({PACKAGES[estimator] for estimator in estimators}):
        print(f"  {package} {package_version(package) or 'not installed'}")
    print()


def print_report(rows, repeats):
    """Print one line per case with timings and the largest differences from R."""
    table = PrettyTable(
        ["case", "workload", "moderndid", "R", "R / moderndid", "max |Δ estimate|", "max |Δ SE|", "cells", "status"]
    )
    for row in rows:
        python_seconds, r_seconds = row.get("moderndid"), row.get("R")
        error = row.get("error")
        table.add_row(
            [
                row["case"],
                row["workload"],
                format_seconds(python_seconds) if python_seconds is not None else "",
                format_seconds(r_seconds) if r_seconds is not None else "",
                f"{r_seconds / python_seconds:.1f}x" if python_seconds and r_seconds else "",
                f"{row['estimate']:.2e}" if "estimate" in row else "",
                "bootstrap" if "cells" in row and error is None else (f"{error:.2e}" if error is not None else ""),
                row.get("cells", ""),
                row["status"],
            ]
        )
    table.align = "r"
    table.align["case"] = table.align["workload"] = table.align["status"] = "l"
    print(table)
    print()
    print(
        f"Timings are medians of {repeats} calls after one warm-up. Every math library runs on one thread. "
        "R times only the estimator call, after loading its package and the data."
    )
    print(
        "Estimates match when each gap is within 1e-6 + 1e-5 x |R| and standard errors when it is within "
        "1e-6 + 1e-3 x |R|. Since the two cont_did bootstraps draw different samples, their standard errors "
        "are not compared. Because triplediff matches some cohorts to the wrong units in the overall "
        "standard error of its group aggregation, the comparison leaves that number out for agg_ddd."
    )
    print(
        "R's contdid stops on the small workload because rounding leaves its overall weights about 1e-16 "
        "short of the exact sum of one it requires."
    )


def main():
    """Run every requested comparison and print the report."""
    arguments = parse_args()
    print_environment(arguments.estimators)
    rows = [run_case(*case, arguments.repeats) for case in cases(arguments)]
    print_report(rows, arguments.repeats)
    if any(row["status"] == "differs" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
