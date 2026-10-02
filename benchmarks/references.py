"""Run the R implementations that the comparison measures moderndid against."""

import functools
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

# Each comparison names the R package behind its reference. A missing package then
# marks that reference unavailable instead of failing the whole run.
PACKAGES = {
    "att_gt": "did",
    "aggte": "did",
    "ddd": "triplediff",
    "agg_ddd": "triplediff",
    "cont_did": "contdid",
    "did_multiplegt": "DIDmultiplegtDYN",
    "drdid": "DRDID",
    "ipwdid": "DRDID",
    "ordid": "DRDID",
}

SCRIPT = """
suppressPackageStartupMessages({{
  library({package})
  library(jsonlite)
}})
data <- read.csv("{data_path}")
{setup}
fit <- function() {call}
result <- fit()
seconds <- vapply(seq_len({repeats}), function(i) system.time(fit())[["elapsed"]], numeric(1))
write_json(c(list(seconds = seconds), {extract}), "{result_path}", digits = NA, na = "null")
"""


def r_formula(covariates):
    """Write the covariates as an R formula."""
    return "~1" if not covariates else "~" + " + ".join(covariates)


def r_logical(value):
    """Write a Python boolean as an R logical."""
    return "TRUE" if value else "FALSE"


def att_gt_call(options):
    """Return the R call that matches the benchmark's ``att_gt`` workload."""
    return (
        'did::att_gt(yname = "y", tname = "time", idname = "id", gname = "group", '
        f"xformla = {r_formula(options['covariates'])}, data = data, "
        f'est_method = "{options["est_method"]}", control_group = "nevertreated", '
        'base_period = "varying", bstrap = FALSE)'
    )


def ddd_call(options):
    """Return the R call that matches the benchmark's ``ddd`` workload."""
    return (
        'triplediff::ddd(yname = "y", tname = "time", idname = "id", gname = "group", '
        f'pname = "partition", xformla = {r_formula(options["covariates"])}, data = data, '
        f'control_group = "nevertreated", base_period = "universal", est_method = "{options["est_method"]}", '
        "boot = FALSE)"
    )


def two_period_call(estimator, options):
    """Return the R call that matches the benchmark's two-period workload."""
    # Each call pairs R with the est_method default of moderndid's function of the same name.
    # Since R's ipwdid normalizes its weights unless told otherwise, the call turns that off.
    method = {"drdid": ', estMethod = "imp"', "ipwdid": ", normalized = FALSE", "ordid": ""}[estimator]
    return (
        f'DRDID::{estimator}(yname = "y", tname = "time", idname = "id", dname = "treated", '
        f"xformla = {r_formula(options['covariates'])}, data = data, panel = TRUE{method})"
    )


def reference_code(estimator, options):
    """Return the setup, timed call, and result extraction for one R reference."""
    if estimator == "att_gt":
        return (
            "",
            att_gt_call(options),
            "list(keys = paste(result$group, result$t), estimates = result$att, std_errors = result$se)",
        )
    if estimator == "aggte":
        return (
            f"group_time <- {att_gt_call(options)}",
            f'did::aggte(group_time, type = "{options["aggregation"]}", bstrap = FALSE, cband = FALSE)',
            'list(keys = c("overall", as.character(result$egt)), '
            "estimates = c(result$overall.att, result$att.egt), "
            "std_errors = c(result$overall.se, result$se.egt))",
        )
    if estimator == "ddd":
        return (
            "",
            ddd_call(options),
            "list(keys = paste(result$groups, result$periods), estimates = result$ATT, std_errors = result$se)",
        )
    if estimator == "agg_ddd":
        return (
            f"group_time <- {ddd_call(options)}",
            f'triplediff::agg_ddd(group_time, type = "{options["aggregation"]}", boot = FALSE, cband = FALSE)',
            'list(keys = c("overall", as.character(result$aggte_ddd$egt)), '
            "estimates = c(result$aggte_ddd$overall.att, result$aggte_ddd$att.egt), "
            "std_errors = c(result$aggte_ddd$overall.se, result$aggte_ddd$se.egt))",
        )
    if estimator == "cont_did":
        # Since the Python estimator bootstraps its dose-response errors, R draws as many samples to time the same work.
        return (
            "",
            'contdid::cont_did(yname = "y", tname = "time", idname = "id", dname = "dose", gname = "group", '
            'data = data, target_parameter = "level", aggregation = "dose", treatment_type = "continuous", '
            'dose_est_method = "parametric", control_group = "notyettreated", degree = 3, num_knots = 0, '
            f"bstrap = TRUE, biters = {options['biters']})",
            'list(keys = c("ATT", "ACRT"), estimates = c(result$overall_att, result$overall_acrt), '
            "std_errors = c(result$overall_att_se, result$overall_acrt_se))",
        )
    if estimator == "did_multiplegt":
        controls = "NULL"
        if options["covariates"]:
            controls = "c(" + ", ".join(f'"{name}"' for name in options["covariates"]) + ")"
        # Since DIDmultiplegtDYN 2.4.0 calls polars through pl without importing it, the setup attaches polars.
        return (
            "suppressPackageStartupMessages(library(polars))",
            'suppressWarnings(DIDmultiplegtDYN::did_multiplegt_dyn(df = data, outcome = "y", group = "id", '
            f'time = "time", treatment = "treatment", effects = 2, placebo = 1, '
            f"normalized = {r_logical(options['normalized'])}, controls = {controls}, graph_off = TRUE))",
            "list(keys = c(paste('effect', seq_len(nrow(result$results$Effects))), "
            "paste('placebo', seq_len(nrow(result$results$Placebos)))), "
            'estimates = c(result$results$Effects[, "Estimate"], result$results$Placebos[, "Estimate"]), '
            'std_errors = c(result$results$Effects[, "SE"], result$results$Placebos[, "SE"]))',
        )
    if estimator in {"drdid", "ipwdid", "ordid"}:
        return (
            "",
            two_period_call(estimator, options),
            'list(keys = "ATT", estimates = result$ATT, std_errors = result$se)',
        )
    raise ValueError(f"No R reference for {estimator!r}")


def r_environment():
    """Return the environment for R with every math library limited to one thread."""
    environment = dict(os.environ)
    # Since the benchmark environment runs moderndid on one thread, R gets the same limit.
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "R_DATATABLE_NUM_THREADS",
    ):
        environment[name] = "1"
    return environment


@functools.cache
def r_version():
    """Return the R version string, or None when R is not installed."""
    if shutil.which("Rscript") is None:
        return None
    result = subprocess.run(
        ["Rscript", "--vanilla", "-e", "cat(R.version.string)"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


@functools.cache
def package_version(package):
    """Return the version of an R package that loads, or None when it is missing or broken."""
    if r_version() is None:
        return None
    # A package can be installed and still fail to load, as when one of its own dependencies is missing.
    check = (
        f'if (!requireNamespace("{package}", quietly = TRUE)) quit(status = 1); '
        f'cat(as.character(packageVersion("{package}")))'
    )
    result = subprocess.run(
        ["Rscript", "--vanilla", "-e", check],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def run_reference(estimator, data, options, repeats):
    """Run one R reference and return its timings, keys, estimates, and standard errors."""
    setup, call, extract = reference_code(estimator, options)
    with tempfile.TemporaryDirectory() as directory:
        data_path = Path(directory) / "data.csv"
        result_path = Path(directory) / "result.json"
        script_path = Path(directory) / "reference.R"
        data.write_csv(data_path)
        script_path.write_text(
            SCRIPT.format(
                package=PACKAGES[estimator],
                data_path=data_path.as_posix(),
                setup=setup,
                call=call,
                repeats=repeats,
                extract=extract,
                result_path=result_path.as_posix(),
            )
        )
        process = subprocess.run(
            ["Rscript", "--vanilla", str(script_path)],
            capture_output=True,
            text=True,
            env=r_environment(),
            check=False,
        )
        if process.returncode != 0 or not result_path.exists():
            raise RuntimeError(r_error(process.stderr))
        return json.loads(result_path.read_text())


def r_error(stderr):
    """Pull the message of an R error out of its standard error stream."""
    lines = stderr.strip().splitlines()
    start = next((index for index, line in enumerate(lines) if line.startswith("Error")), None)
    if start is None:
        return lines[-1] if lines else "R failed without a message"
    message = " ".join(line.strip() for line in lines[start:] if not line.startswith(("Calls:", "Execution halted")))
    return message.split(" : ", 1)[-1] if " : " in message else message
