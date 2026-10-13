"""Compare moderndid with the R implementations of the same estimators."""

import argparse
import math
import platform
import statistics
import time
from dataclasses import asdict

import numpy as np
import polars as pl
from prettytable import PrettyTable

import moderndid
from benchmarks.cases import WORKLOADS
from benchmarks.common import (
    aggregation_type,
    check_aggregation,
    check_estimate,
    make_aggregation,
    make_estimator,
    resolve_workload,
)
from benchmarks.references import PACKAGES, package_version, r_version, run_reference
from benchmarks.reporting import TOLERANCES, capture_environment, make_report, write_report

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
    parser.add_argument("--output", help="Save measurements, coverage, and environment metadata as JSON.")
    parser.add_argument("--csv", help="Save one row per comparison as CSV.")
    parser.add_argument(
        "--strict", action="store_true", help="Fail for unavailable, failed, or incomplete comparisons."
    )
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        parser.error("--repeats must be at least 1")
    # Independent bootstrap draws do not give a deterministic numerical comparison.
    if "bootstrap" in arguments.workloads:
        parser.error("the bootstrap workload is not available because the comparison uses analytical standard errors")
    unknown = set(arguments.workloads) - set(WORKLOADS) - {"ipw", "reg", "normalized"}
    if unknown:
        parser.error(f"unknown workloads: {', '.join(sorted(unknown))}")
    return arguments


def cases(arguments):
    """Yield every requested case as an estimator and workload with its aggregation."""
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
        return make_aggregation(fitted, profile, aggregation, warmup=False), data
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
    """Return the keyed estimates and standard errors of a moderndid result."""
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
        # Since the two bootstraps use independent draws, the influence functions give deterministic standard errors.
        influence_functions = (
            np.asarray(result.overall_att_inf_func),
            np.asarray(result.overall_acrt_inf_func),
            np.asarray(result.att_d_inf_func),
            np.asarray(result.acrt_d_inf_func),
        )
        errors = [np.sqrt(np.sum(values**2, axis=0)) / values.shape[0] for values in influence_functions]
        return (
            ["ATT", "ACRT"]
            + [f"ATT dose {index + 1}" for index in range(len(result.dose))]
            + [f"ACRT dose {index + 1}" for index in range(len(result.dose))],
            [result.overall_att, result.overall_acrt, *result.att_d, *result.acrt_d],
            [float(errors[0]), float(errors[1]), *errors[2], *errors[3]],
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
    """Convert a JSON value to a float and read R's missing values as NaN."""
    if value is None or value == "NA":
        return math.nan
    return float(value)


def within(gaps, relative):
    """Check every gap against an absolute floor plus a relative tolerance."""
    return all(gap <= TOLERANCES["absolute"] + relative * size for gap, size in gaps)


def normalized_reference_key(estimator, aggregation, key):
    """Identify triple-difference cells whose reference-period effect is fixed at zero."""
    if estimator == "ddd":
        group, period = (float(value) for value in key.split())
        return period == group - 1
    return estimator == "agg_ddd" and aggregation == "dynamic" and key == "-1"


def averaged_cohorts(fitted, aggregation):
    """Count the cohorts that each output of a triple-difference aggregation averages."""
    groups = np.asarray(fitted.groups)
    events = np.asarray(fitted.times) - groups
    # Since every cohort's base-period effect is fixed at zero, that cell adds nothing to a cohort-share term.
    cohorts = {event: set(groups[(events == event) & (events != -1)]) for event in np.unique(events)}
    if aggregation == "dynamic":
        counts = {f"{event:.0f}": len(members) for event, members in cohorts.items()}
        counts["overall"] = max((len(members) for event, members in cohorts.items() if event >= 0), default=0)
        return counts
    treated = set().union(*(members for event, members in cohorts.items() if event >= 0))
    return {"overall": len(treated)} | {f"{group:.0f}": 1 for group in treated}


def known_differences(estimator, aggregation, keys, cohorts=None):
    """Name the outputs that the reference computes differently and say why.

    The comparison records the largest gap of these outputs without holding them to the tolerances.
    A missing or nonfinite value among them still makes the coverage partial. For agg_ddd, ``cohorts``
    maps each output to the number of cohorts it averages, as ``averaged_cohorts`` counts them.
    """
    if estimator == "cont_did":
        return (
            {key for key in keys if key not in {"ATT", "ACRT"}},
            set(keys),
            "The reference evaluates the dose curves on a basis whose boundary knots are the grid's endpoints. "
            "Its influence functions also hold the cohort weights fixed, rescale the cohort shares to sum to one, "
            "and scale each ACRT cell by its size instead of its treated count.",
        )
    if estimator == "agg_ddd":
        # Since every cohort of make_panel is half eligible, shares of eligible units and of all units give the same
        # estimates. A workload whose cohorts differ in eligible share would need its multi-cohort estimates here.
        reason = (
            "The reference counts every unit of an enabling cohort in the cohort shares instead of only the "
            "eligible units that receive treatment. Since every cohort of the generated panels is half eligible, "
            "the estimates agree. The standard errors of outputs that average two or more cohorts differ through "
            "the influence function's term for the estimated shares."
        )
        if aggregation == "group":
            reason += (
                " The reference also builds that term from group-time cohort labels instead of the distinct cohorts."
            )
        return set(), {key for key in keys if (cohorts or {}).get(key, 0) > 1}, reason
    return set(), set(), None


def comparison_details(python, reference, estimator, aggregation, cohorts=None):
    """Compare all output keys and report finite coverage separately from numerical gaps."""
    keys, estimates, errors = python
    python_lookup = {
        str(key): (as_float(estimate), as_float(error))
        for key, estimate, error in zip(keys, estimates, errors, strict=True)
    }
    r_lookup = {
        str(key): (as_float(estimate), as_float(error))
        for key, estimate, error in zip(reference["keys"], reference["estimates"], reference["std_errors"], strict=True)
    }
    all_keys = python_lookup.keys() | r_lookup.keys()
    known_estimates, known_errors, reason = known_differences(estimator, aggregation, all_keys, cohorts)
    gaps, error_gaps, known_gaps, known_error_gaps = [], [], [], []
    nonfinite_estimates, nonfinite_errors = [], []
    unavailable_errors, normalized_errors = [], []
    for key in sorted(python_lookup.keys() & r_lookup.keys()):
        estimate, error = python_lookup[key]
        r_estimate, r_error = r_lookup[key]
        if not (math.isfinite(estimate) and math.isfinite(r_estimate)):
            nonfinite_estimates.append(key)
        elif key in known_estimates:
            known_gaps.append(abs(estimate - r_estimate))
        else:
            gaps.append((abs(estimate - r_estimate), abs(r_estimate)))
        if math.isfinite(error) and math.isfinite(r_error):
            if key in known_errors:
                known_error_gaps.append(abs(error - r_error))
            else:
                error_gaps.append((abs(error - r_error), abs(r_error)))
        elif (
            normalized_reference_key(estimator, aggregation, key)
            and estimate == r_estimate == 0
            and (math.isnan(error) or error == 0)
            and (math.isnan(r_error) or r_error == 0)
        ):
            if math.isnan(error) and math.isnan(r_error):
                unavailable_errors.append(key)
            else:
                normalized_errors.append(key)
        else:
            nonfinite_errors.append(key)
    missing_python = sorted(r_lookup.keys() - python_lookup.keys())
    missing_r = sorted(python_lookup.keys() - r_lookup.keys())
    # Although known differences leave the denominators, a missing or nonfinite one still makes the coverage partial.
    expected_cells = len(all_keys - known_estimates)
    expected_se = len(all_keys - known_errors) - len(set(unavailable_errors + normalized_errors) - known_errors)
    complete = not (missing_python or missing_r or nonfinite_estimates or nonfinite_errors)
    if not gaps:
        status = "no shared cells"
    elif not within(gaps, TOLERANCES["estimate_relative"]) or not within(error_gaps, TOLERANCES["se_relative"]):
        status = "differs"
    elif not complete:
        status = "partial"
    else:
        status = "match"
    return {
        "cells": len(gaps),
        "expected_cells": expected_cells,
        "python_cells": len(python_lookup),
        "r_cells": len(r_lookup),
        "se_cells": len(error_gaps),
        "expected_se_cells": expected_se,
        "excluded_estimate_keys": sorted(known_estimates),
        "excluded_se_keys": sorted(known_errors),
        "excluded_estimate_gap": max(known_gaps, default=None),
        "excluded_error_gap": max(known_error_gaps, default=None),
        "exclusion_reason": reason if known_estimates or known_errors else None,
        "unavailable_se_keys": unavailable_errors,
        "normalized_se_keys": normalized_errors,
        "missing_python_keys": missing_python,
        "missing_r_keys": missing_r,
        "nonfinite_estimate_keys": nonfinite_estimates,
        "nonfinite_se_keys": nonfinite_errors,
        "estimate": max((gap for gap, _ in gaps), default=math.nan),
        "error": max((gap for gap, _ in error_gaps), default=math.nan) if expected_se else None,
        "status": status,
    }


def timing_iqr(seconds):
    """Return the interquartile range of the timings in seconds."""
    return float(np.percentile(seconds, 75) - np.percentile(seconds, 25))


def format_seconds(seconds):
    """Write a duration with a unit that keeps three significant figures."""
    if seconds < 1:
        return f"{seconds * 1000:.3g} ms"
    return f"{seconds:.3g} s"


def run_case(estimator, profile, aggregation, repeats):
    """Time moderndid and R on one case and measure how far their results are apart."""
    label = f"{estimator} ({aggregation})" if aggregation else estimator
    package = PACKAGES[estimator]
    row = {
        "case": label,
        "estimator": estimator,
        "workload": profile,
        "aggregation": aggregation,
        "reference_package": package,
    }
    try:
        workload = resolve_workload(AGGREGATES.get(estimator, estimator), profile)
    except ValueError as error:
        return {**row, "status": f"skipped, {error}"}
    row["workload_spec"] = asdict(workload)
    try:
        call, data = prepare_python(estimator, profile, aggregation)
        row["data_rows"] = len(data)
        result = call()
        if estimator in AGGREGATES:
            check_aggregation(result)
        else:
            check_estimate(estimator, result)
        extracted = python_rows(estimator, result)
        # Since make_aggregation binds the fitted ddd result as the first argument, the call carries its cohorts.
        cohorts = averaged_cohorts(call.args[0], aggregation) if estimator == "agg_ddd" else None
        row["python_cells"] = len(extracted[0])
        seconds = []
        for _ in range(repeats):
            start = time.perf_counter()
            call()
            seconds.append(time.perf_counter() - start)
        row.update(moderndid=statistics.median(seconds), python_seconds=seconds, python_iqr=timing_iqr(seconds))
    except Exception as error:  # noqa: BLE001 - retain other cases and persist the failed case in the report
        return {**row, "status": f"Python failed, {type(error).__name__}: {error}"}
    if package_version(package) is None:
        return {**row, "status": f"R package {package} not installed"}
    try:
        reference = run_reference(estimator, data, reference_options(estimator, profile, aggregation), repeats)
        r_seconds = reference["seconds"]
        if not isinstance(r_seconds, list):
            r_seconds = [r_seconds]
        row.update(R=statistics.median(r_seconds), r_seconds=r_seconds, r_iqr=timing_iqr(r_seconds))
        row.update(comparison_details(extracted, reference, estimator, aggregation, cohorts))
    except Exception as error:  # noqa: BLE001 - an unavailable reference must remain visible in saved results
        return {**row, "status": f"R failed, {error}"}
    return row


def print_environment(estimators):
    """Print the software versions behind the comparison."""
    print(f"Python {platform.python_version()} on {platform.platform()}")
    print(f"moderndid {moderndid.__version__}, numpy {np.__version__}, polars {pl.__version__}")
    print(r_version() or "R is not installed")
    for package in sorted({PACKAGES[estimator] for estimator in estimators}):
        print(f"  {package} {package_version(package) or 'not installed'}")
    print()


def known_difference_note(row):
    """Describe the known differences of one case in a sentence followed by their reason."""
    parts = []
    for keys, gap, noun in (
        (row.get("excluded_estimate_keys", []), row.get("excluded_estimate_gap"), "estimate"),
        (row.get("excluded_se_keys", []), row.get("excluded_error_gap"), "standard error"),
    ):
        if keys:
            largest = f" (largest gap {gap:.2e})" if gap is not None else ""
            parts.append(f"{len(keys)} {noun}{'' if len(keys) == 1 else 's'}{largest}")
    return (
        f"{row['case']} on the {row['workload']} workload has known differences in {' and '.join(parts)}. "
        f"{row['exclusion_reason']}"
    )


def print_report(rows, repeats):
    """Print one line per case with timings and the largest differences from R."""
    table = PrettyTable(
        [
            "case",
            "workload",
            "moderndid",
            "R",
            "R / moderndid",
            "max |Δ estimate|",
            "max |Δ SE|",
            "cells",
            "SE cells",
            "status",
        ]
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
                f"{error:.2e}" if error is not None else ("known" if row.get("excluded_se_keys") else ""),
                f"{row['cells']}/{row['expected_cells']}" if "cells" in row else "",
                f"{row['se_cells']}/{row['expected_se_cells']}" if "se_cells" in row else "",
                row["status"],
            ]
        )
    table.align = "r"
    table.align["case"] = table.align["workload"] = table.align["status"] = "l"
    print(table)
    print()
    print(
        f"Timings are medians of {repeats} calls after one warm-up. "
        "R times only the estimator call, after loading its package and the data."
    )
    print(
        "Estimates match when each gap is within 1e-6 + 1e-5 x |R| and standard errors when it is within "
        "1e-6 + 1e-3 x |R|. Known differences keep their gaps out of that check and never fail a run."
    )
    print(
        "Cell counts show finite comparisons over expected comparisons. Undefined or normalized "
        "reference-period standard errors and known differences leave the denominators and are recorded in the "
        "saved report. Missing keys or other nonfinite values produce partial coverage."
    )
    print(
        "R's contdid stops on the small workload because rounding leaves its overall weights about 1e-16 "
        "short of the exact sum of one it requires."
    )
    for row in rows:
        if row.get("excluded_estimate_keys") or row.get("excluded_se_keys"):
            print(known_difference_note(row))


def main():
    """Run every requested comparison and print the report."""
    arguments = parse_args()
    environment = capture_environment(arguments.estimators)
    print_environment(arguments.estimators)
    rows = [run_case(*case, arguments.repeats) for case in cases(arguments)]
    print_report(rows, arguments.repeats)
    write_report(make_report(rows, arguments, environment), arguments.output, arguments.csv)
    if any(
        row["status"] == "differs"
        or (arguments.strict and row["status"] != "match" and not row["status"].startswith("skipped,"))
        for row in rows
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
