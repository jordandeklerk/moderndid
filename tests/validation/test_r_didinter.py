"""Validation tests comparing Python did_multiplegt implementation with R DIDmultiplegtDYN package."""

import json
import subprocess
import tempfile

import pytest

pytestmark = pytest.mark.slow

from tests.helpers import importorskip

pl = importorskip("polars")
np = importorskip("numpy")

from moderndid import did_multiplegt, load_favara_imbs


def _run_r_script(r_script, result_path, timeout=120):
    proc = subprocess.run(
        ["R", "--vanilla", "--quiet"],
        input=r_script,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"R script failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}")

    with open(result_path, encoding="utf-8") as f:
        return json.load(f)


def check_r_available():
    try:
        result = subprocess.run(
            ["R", "--vanilla", "--quiet"],
            input=(
                "options(rgl.useNULL=TRUE); "
                "suppressPackageStartupMessages(library(DIDmultiplegtDYN)); "
                "library(polars); "
                'library(jsonlite); cat("OK")'
            ),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return "OK" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


R_AVAILABLE = check_r_available()


def r_did_multiplegt(
    data_path,
    effects=1,
    placebo=0,
    normalized=False,
    cluster=None,
    effects_equal=False,
    trends_lin=False,
    switchers="",
    only_never_switchers=False,
    same_switchers=False,
    same_switchers_pl=False,
    less_conservative_se=False,
    more_granular_demeaning=False,
    continuous=None,
    controls=None,
    predict_het=None,
    predict_het_hc2bm=False,
    weight=None,
    trends_nonparam=None,
    outcome="Dl_vloans_b",
    group="county",
    time="year",
    treatment="inter_bra",
):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    normalized_str = "TRUE" if normalized else "FALSE"
    trends_lin_str = "TRUE" if trends_lin else "FALSE"
    only_never_str = "TRUE" if only_never_switchers else "FALSE"
    same_switchers_str = "TRUE" if same_switchers else "FALSE"
    same_switchers_pl_str = "TRUE" if same_switchers_pl else "FALSE"
    continuous_str = "NULL" if continuous is None else str(continuous)
    less_conservative_str = "TRUE" if less_conservative_se else "FALSE"
    more_granular_str = "TRUE" if more_granular_demeaning else "FALSE"
    predict_het_hc2bm_str = "TRUE" if predict_het_hc2bm else "FALSE"

    if isinstance(effects_equal, str):
        effects_equal_str = f'"{effects_equal}"'
    elif effects_equal is True:
        effects_equal_str = "TRUE"
    else:
        effects_equal_str = "FALSE"

    cluster_str = f'"{cluster}"' if cluster is not None else "NULL"
    weight_str = f'"{weight}"' if weight is not None else "NULL"

    if switchers == "in":
        switchers_str = '"in"'
    elif switchers == "out":
        switchers_str = '"out"'
    else:
        switchers_str = '""'

    if controls is not None:
        controls_str = "c(" + ", ".join(f'"{c}"' for c in controls) + ")"
    else:
        controls_str = "NULL"

    if trends_nonparam is not None:
        trends_nonparam_str = "c(" + ", ".join(f'"{v}"' for v in trends_nonparam) + ")"
    else:
        trends_nonparam_str = "NULL"

    if predict_het is not None:
        covs, horizons = predict_het
        covs_str = "c(" + ", ".join(f'"{c}"' for c in covs) + ")"
        if horizons == [-1]:
            horizons_str = "-1"
        else:
            horizons_str = "c(" + ", ".join(str(h) for h in horizons) + ")"
        predict_het_str = f"list({covs_str}, {horizons_str})"
    else:
        predict_het_str = "NULL"

    r_script = f"""
options(rgl.useNULL = TRUE)
suppressPackageStartupMessages(library(DIDmultiplegtDYN))
library(polars)
library(jsonlite)

data <- read.csv("{data_path}")

result <- tryCatch(
  suppressWarnings(did_multiplegt_dyn(
    df = data,
    outcome = "{outcome}",
    group = "{group}",
    time = "{time}",
    treatment = "{treatment}",
    effects = {effects},
    placebo = {placebo},
    normalized = {normalized_str},
    cluster = {cluster_str},
    weight = {weight_str},
    effects_equal = {effects_equal_str},
    trends_lin = {trends_lin_str},
    switchers = {switchers_str},
    only_never_switchers = {only_never_str},
    same_switchers = {same_switchers_str},
    same_switchers_pl = {same_switchers_pl_str},
    continuous = {continuous_str},
    less_conservative_se = {less_conservative_str},
    more_granular_demeaning = {more_granular_str},
    controls = {controls_str},
    trends_nonparam = {trends_nonparam_str},
    predict_het = {predict_het_str},
    predict_het_hc2bm = {predict_het_hc2bm_str},
    graph_off = TRUE
  )),
  error = function(e) NULL
)

if (is.null(result)) {{
    write_json(list(error = "R estimation failed"), "{result_path}")
    quit(status = 0)
}}

r <- result$results
out <- list()

if (!is.null(r$Effects)) {{
    out$effect_estimates <- as.numeric(r$Effects[, "Estimate"])
    out$effect_se <- as.numeric(r$Effects[, "SE"])
    out$effect_ci_lower <- as.numeric(r$Effects[, "LB CI"])
    out$effect_ci_upper <- as.numeric(r$Effects[, "UB CI"])
    out$effect_n_switchers <- as.integer(r$Effects[, "Switchers"])
    out$effect_n_switchers_w <- as.numeric(r$Effects[, "Switchers.w"])
    out$effect_n <- as.numeric(r$Effects[, "N"])
}}

if (!is.null(r$Placebos)) {{
    out$placebo_estimates <- as.numeric(r$Placebos[, "Estimate"])
    out$placebo_se <- as.numeric(r$Placebos[, "SE"])
    out$placebo_n_switchers <- as.integer(r$Placebos[, "Switchers"])
    out$placebo_n_switchers_w <- as.numeric(r$Placebos[, "Switchers.w"])
    out$placebo_n <- as.numeric(r$Placebos[, "N"])
}}

if (!is.null(r$ATE)) {{
    out$ate_estimate <- r$ATE[1, "Estimate"]
    out$ate_se <- r$ATE[1, "SE"]
    out$ate_n <- r$ATE[1, "N"]
    out$ate_switchers <- r$ATE[1, "Switchers"]
}}
out$ate_missing <- is.null(r$ATE) || is.na(r$ATE[1, "Estimate"])

if (!is.null(r$p_equality_effects)) {{
    out$effects_equal_pvalue <- r$p_equality_effects
}}

if (!is.null(r$p_jointplacebo)) {{
    out$placebo_joint_pvalue <- r$p_jointplacebo
}}

if (!is.null(r$predict_het)) {{
    het <- r$predict_het
    out$het_effects <- het$effect
    out$het_covariates <- het$covariate
    out$het_estimates <- het$Estimate
    out$het_se <- het$SE
    out$het_t <- het$t
    out$het_pf <- het$pF
}}

if (!is.null(r$vcov_warnings)) {{
    out$vcov_warnings <- r$vcov_warnings
}}

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path, timeout=300)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_did_multiplegt_bootstrap(
    data_path,
    sample_dir,
    reps,
    seed,
    cluster=None,
    effects=3,
    placebo=2,
    switchers="",
    outcome="Dl_vloans_b",
    group="county",
    time="year",
    treatment="inter_bra",
):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    cluster_str = f'"{cluster}"' if cluster is not None else "NULL"

    r_script = f"""
options(rgl.useNULL = TRUE, DID_BOOTSTRAP_SAMPLE_DIR = "{sample_dir}")
suppressPackageStartupMessages(library(DIDmultiplegtDYN))
library(polars)
library(jsonlite)

data <- read.csv("{data_path}")

r <- suppressMessages(did_multiplegt_dyn(
    df = data,
    outcome = "{outcome}",
    group = "{group}",
    time = "{time}",
    treatment = "{treatment}",
    effects = {effects},
    placebo = {placebo},
    switchers = "{switchers}",
    cluster = {cluster_str},
    bootstrap = c({reps}, {seed}),
    graph_off = TRUE
))$results

out <- list(
    effect_se = as.numeric(r$Effects[, "SE"]),
    placebo_se = as.numeric(r$Placebos[, "SE"]),
    ate_se = r$ATE[1, "SE"]
)
write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path, timeout=600)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


@pytest.fixture(scope="module")
def favara_imbs_data():
    return load_favara_imbs()


@pytest.fixture(scope="module")
def favara_imbs_csv_path(favara_imbs_data):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        favara_imbs_data.write_csv(f.name)
        return f.name


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
@pytest.mark.parametrize("effects", [1, 3, 5])
def test_effect_estimates(favara_imbs_data, favara_imbs_csv_path, effects):
    r_result = r_did_multiplegt(favara_imbs_csv_path, effects=effects)

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=effects,
    )

    np.testing.assert_allclose(
        py_result.effects.estimates,
        np.array(r_result["effect_estimates"]),
        rtol=1e-4,
        atol=1e-5,
        err_msg=f"effects={effects}: Effect estimates mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
@pytest.mark.parametrize("effects", [1, 3, 5])
def test_effect_standard_errors(favara_imbs_data, favara_imbs_csv_path, effects):
    r_result = r_did_multiplegt(favara_imbs_csv_path, effects=effects)

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=effects,
    )

    np.testing.assert_allclose(
        py_result.effects.std_errors,
        np.array(r_result["effect_se"]),
        rtol=5e-4,
        atol=1e-5,
        err_msg=f"effects={effects}: Standard errors mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_effect_confidence_intervals(favara_imbs_data, favara_imbs_csv_path):
    r_result = r_did_multiplegt(favara_imbs_csv_path, effects=3)

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=3,
    )

    np.testing.assert_allclose(
        py_result.effects.ci_lower,
        np.array(r_result["effect_ci_lower"]),
        rtol=3e-3,
        atol=1e-4,
        err_msg="CI lower bounds mismatch",
    )
    np.testing.assert_allclose(
        py_result.effects.ci_upper,
        np.array(r_result["effect_ci_upper"]),
        rtol=5e-4,
        atol=1e-4,
        err_msg="CI upper bounds mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_n_switchers(favara_imbs_data, favara_imbs_csv_path):
    r_result = r_did_multiplegt(favara_imbs_csv_path, effects=4)

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=4,
    )

    np.testing.assert_array_equal(
        py_result.effects.n_switchers,
        np.array(r_result["effect_n_switchers"]),
        err_msg="Number of switchers mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
@pytest.mark.parametrize("placebo", [1, 2])
def test_placebo_estimates(favara_imbs_data, favara_imbs_csv_path, placebo):
    r_result = r_did_multiplegt(favara_imbs_csv_path, effects=3, placebo=placebo)

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=3,
        placebo=placebo,
    )

    np.testing.assert_allclose(
        py_result.placebos.estimates,
        np.array(r_result["placebo_estimates"]),
        rtol=1e-4,
        atol=1e-5,
        err_msg=f"placebo={placebo}: Placebo estimates mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_placebo_standard_errors(favara_imbs_data, favara_imbs_csv_path):
    r_result = r_did_multiplegt(favara_imbs_csv_path, effects=3, placebo=2)

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=3,
        placebo=2,
    )

    np.testing.assert_allclose(
        py_result.placebos.std_errors,
        np.array(r_result["placebo_se"]),
        rtol=1e-10,
        err_msg="Placebo standard errors mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_ate_estimate(favara_imbs_data, favara_imbs_csv_path):
    r_result = r_did_multiplegt(favara_imbs_csv_path, effects=3)

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    if "ate_estimate" not in r_result:
        pytest.fail("R did not return ATE estimate")

    py_result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=3,
    )

    np.testing.assert_allclose(
        py_result.ate.estimate,
        r_result["ate_estimate"],
        rtol=1e-6,
        atol=1e-10,
        err_msg="ATE estimate mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_ate_standard_error(favara_imbs_data, favara_imbs_csv_path):
    r_result = r_did_multiplegt(favara_imbs_csv_path, effects=3)

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    if "ate_se" not in r_result:
        pytest.fail("R did not return ATE standard error")

    py_result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=3,
    )

    np.testing.assert_allclose(
        py_result.ate.std_error,
        r_result["ate_se"],
        rtol=1e-10,
        err_msg="ATE standard error mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_normalized_effects(favara_imbs_data, favara_imbs_csv_path):
    r_result = r_did_multiplegt(favara_imbs_csv_path, effects=3, normalized=True)

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=3,
        normalized=True,
    )

    np.testing.assert_allclose(
        py_result.effects.estimates,
        np.array(r_result["effect_estimates"]),
        rtol=5e-4,
        atol=1e-5,
        err_msg="Normalized effect estimates mismatch",
    )
    np.testing.assert_allclose(
        py_result.effects.std_errors,
        np.array(r_result["effect_se"]),
        rtol=2e-3,
        atol=1e-4,
        err_msg="Normalized standard errors mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_clustered_se(favara_imbs_data, favara_imbs_csv_path):
    r_result = r_did_multiplegt(favara_imbs_csv_path, effects=3, cluster="state_n")

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=3,
        cluster="state_n",
    )

    np.testing.assert_allclose(
        py_result.effects.estimates,
        np.array(r_result["effect_estimates"]),
        rtol=1e-4,
        atol=1e-5,
        err_msg="Clustered: Effect estimates mismatch",
    )
    np.testing.assert_allclose(
        py_result.effects.std_errors,
        np.array(r_result["effect_se"]),
        rtol=1e-10,
        err_msg="Clustered: Standard errors mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_switchers_in(favara_imbs_data, favara_imbs_csv_path):
    r_result = r_did_multiplegt(favara_imbs_csv_path, effects=3, switchers="in")

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=3,
        switchers="in",
    )

    np.testing.assert_allclose(
        py_result.effects.estimates,
        np.array(r_result["effect_estimates"]),
        rtol=1e-4,
        atol=1e-5,
        err_msg="switchers=in: Effect estimates mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_only_never_switchers(favara_imbs_data, favara_imbs_csv_path):
    r_result = r_did_multiplegt(favara_imbs_csv_path, effects=2, only_never_switchers=True)

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=2,
        only_never_switchers=True,
    )

    np.testing.assert_allclose(
        py_result.effects.estimates,
        np.array(r_result["effect_estimates"]),
        rtol=1e-4,
        atol=1e-5,
        err_msg="only_never_switchers: Effect estimates mismatch",
    )
    np.testing.assert_allclose(
        py_result.effects.std_errors,
        np.array(r_result["effect_se"]),
        rtol=5e-4,
        atol=1e-5,
        err_msg="only_never_switchers: Standard errors mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_same_switchers(favara_imbs_data, favara_imbs_csv_path):
    r_result = r_did_multiplegt(favara_imbs_csv_path, effects=3, same_switchers=True)

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=3,
        same_switchers=True,
    )

    np.testing.assert_allclose(
        py_result.effects.estimates,
        np.array(r_result["effect_estimates"]),
        rtol=1e-10,
        err_msg="same_switchers: Effect estimates mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_less_conservative_se(favara_imbs_data, favara_imbs_csv_path):
    r_result = r_did_multiplegt(favara_imbs_csv_path, effects=2, less_conservative_se=True)

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=2,
        less_conservative_se=True,
    )

    np.testing.assert_allclose(
        py_result.effects.estimates,
        np.array(r_result["effect_estimates"]),
        rtol=1e-4,
        atol=1e-5,
        err_msg="less_conservative_se: Effect estimates mismatch",
    )
    np.testing.assert_allclose(
        py_result.effects.std_errors,
        np.array(r_result["effect_se"]),
        rtol=1e-10,
        err_msg="less_conservative_se: Standard errors mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_trends_lin(favara_imbs_data, favara_imbs_csv_path):
    r_result = r_did_multiplegt(favara_imbs_csv_path, effects=2, trends_lin=True)

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=2,
        trends_lin=True,
    )

    np.testing.assert_allclose(
        py_result.effects.estimates,
        np.array(r_result["effect_estimates"]),
        rtol=1e-10,
        err_msg="trends_lin: Effect estimates mismatch",
    )
    np.testing.assert_allclose(
        py_result.effects.std_errors,
        np.array(r_result["effect_se"]),
        rtol=1e-10,
        err_msg="trends_lin: Standard errors mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_effects_equal_test(favara_imbs_data, favara_imbs_csv_path):
    r_result = r_did_multiplegt(favara_imbs_csv_path, effects=3, effects_equal=True)

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    if "effects_equal_pvalue" not in r_result:
        pytest.fail("R did not return effects equality test")

    py_result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=3,
        effects_equal=True,
    )

    np.testing.assert_allclose(
        py_result.effects_equal_test["p_value"],
        r_result["effects_equal_pvalue"],
        rtol=1e-10,
        err_msg="Effects equality test p-value mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_placebo_joint_test(favara_imbs_data, favara_imbs_csv_path):
    r_result = r_did_multiplegt(favara_imbs_csv_path, effects=3, placebo=2)

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    if "placebo_joint_pvalue" not in r_result:
        pytest.fail("R did not return placebo joint test")

    py_result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=3,
        placebo=2,
    )

    np.testing.assert_allclose(
        py_result.placebo_joint_test["p_value"],
        r_result["placebo_joint_pvalue"],
        rtol=1e-10,
        err_msg="Placebo joint test p-value mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_full_pipeline_effects_and_placebos(favara_imbs_data, favara_imbs_csv_path):
    r_result = r_did_multiplegt(favara_imbs_csv_path, effects=3, placebo=2)

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=3,
        placebo=2,
    )

    np.testing.assert_allclose(
        py_result.effects.estimates,
        np.array(r_result["effect_estimates"]),
        rtol=1e-4,
        atol=1e-5,
        err_msg="Pipeline: Effect estimates mismatch",
    )
    np.testing.assert_allclose(
        py_result.placebos.estimates,
        np.array(r_result["placebo_estimates"]),
        rtol=1e-4,
        atol=1e-5,
        err_msg="Pipeline: Placebo estimates mismatch",
    )
    np.testing.assert_allclose(
        py_result.effects.std_errors,
        np.array(r_result["effect_se"]),
        rtol=5e-4,
        atol=1e-5,
        err_msg="Pipeline: Effect standard errors mismatch",
    )
    np.testing.assert_allclose(
        py_result.placebos.std_errors,
        np.array(r_result["placebo_se"]),
        rtol=1e-10,
        err_msg="Pipeline: Placebo standard errors mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_normalized_with_cluster(favara_imbs_data, favara_imbs_csv_path):
    r_result = r_did_multiplegt(favara_imbs_csv_path, effects=3, normalized=True, cluster="state_n")

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=3,
        normalized=True,
        cluster="state_n",
    )

    np.testing.assert_allclose(
        py_result.effects.estimates,
        np.array(r_result["effect_estimates"]),
        rtol=5e-4,
        atol=1e-5,
        err_msg="Normalized + clustered: Effect estimates mismatch",
    )
    np.testing.assert_allclose(
        py_result.effects.std_errors,
        np.array(r_result["effect_se"]),
        rtol=1e-10,
        err_msg="Normalized + clustered: Standard errors mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
@pytest.mark.parametrize("effects_equal", ["2, 4", "1, 5"])
def test_effects_equal_range(favara_imbs_data, favara_imbs_csv_path, effects_equal):
    r_result = r_did_multiplegt(
        favara_imbs_csv_path,
        effects=5,
        effects_equal=effects_equal,
    )

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    if "effects_equal_pvalue" not in r_result:
        pytest.fail("R did not return effects equality test")

    lb, ub = (int(x.strip()) for x in effects_equal.split(","))
    py_result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=5,
        effects_equal=(lb, ub),
    )

    r_pvalue = r_result["effects_equal_pvalue"]
    py_pvalue = py_result.effects_equal_test["p_value"]

    if np.isnan(r_pvalue):
        assert np.isnan(py_pvalue), "R p-value is NaN but Python is not"
    else:
        np.testing.assert_allclose(
            py_pvalue,
            r_pvalue,
            rtol=1e-10,
            err_msg=f"effects_equal='{effects_equal}': p-value mismatch",
        )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_more_granular_demeaning(favara_imbs_data, favara_imbs_csv_path):
    r_result = r_did_multiplegt(
        favara_imbs_csv_path,
        effects=3,
        more_granular_demeaning=True,
    )

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=3,
        more_granular_demeaning=True,
    )

    np.testing.assert_allclose(
        py_result.effects.estimates,
        np.array(r_result["effect_estimates"]),
        rtol=1e-4,
        atol=1e-5,
        err_msg="more_granular_demeaning: Effect estimates mismatch",
    )
    np.testing.assert_allclose(
        py_result.effects.std_errors,
        np.array(r_result["effect_se"]),
        rtol=1e-10,
        err_msg="more_granular_demeaning: Standard errors mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_more_granular_matches_less_conservative(favara_imbs_csv_path):
    r_granular = r_did_multiplegt(
        favara_imbs_csv_path,
        effects=3,
        more_granular_demeaning=True,
    )
    r_less_cons = r_did_multiplegt(
        favara_imbs_csv_path,
        effects=3,
        less_conservative_se=True,
    )

    if r_granular is None or "error" in r_granular:
        pytest.fail("R estimation failed for more_granular_demeaning")
    if r_less_cons is None or "error" in r_less_cons:
        pytest.fail("R estimation failed for less_conservative_se")

    np.testing.assert_allclose(
        np.array(r_granular["effect_estimates"]),
        np.array(r_less_cons["effect_estimates"]),
        rtol=1e-10,
        err_msg="R: more_granular_demeaning should match less_conservative_se estimates",
    )
    np.testing.assert_allclose(
        np.array(r_granular["effect_se"]),
        np.array(r_less_cons["effect_se"]),
        rtol=1e-10,
        err_msg="R: more_granular_demeaning should match less_conservative_se SEs",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
@pytest.mark.parametrize("normalized", [False, True])
def test_controls_effects(favara_imbs_data, favara_imbs_csv_path, normalized):
    r_result = r_did_multiplegt(favara_imbs_csv_path, effects=3, normalized=normalized, controls=["Dl_hpi"])

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=3,
        normalized=normalized,
        xformla="~ Dl_hpi",
    )

    np.testing.assert_allclose(
        py_result.effects.estimates,
        np.array(r_result["effect_estimates"]),
        rtol=1e-10,
        err_msg=f"normalized={normalized}: Controls effect estimates mismatch",
    )
    np.testing.assert_allclose(
        py_result.effects.std_errors,
        np.array(r_result["effect_se"]),
        rtol=1e-10,
        err_msg=f"normalized={normalized}: Controls effect standard errors mismatch",
    )
    np.testing.assert_array_equal(
        py_result.effects.n_switchers,
        np.array(r_result["effect_n_switchers"]),
        err_msg=f"normalized={normalized}: Controls number of switchers mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
@pytest.mark.parametrize("normalized", [False, True])
def test_controls_placebos(favara_imbs_data, favara_imbs_csv_path, normalized):
    r_result = r_did_multiplegt(favara_imbs_csv_path, effects=3, placebo=2, normalized=normalized, controls=["Dl_hpi"])

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=3,
        placebo=2,
        normalized=normalized,
        xformla="~ Dl_hpi",
    )

    np.testing.assert_allclose(
        py_result.placebos.estimates,
        np.array(r_result["placebo_estimates"]),
        rtol=1e-10,
        err_msg=f"normalized={normalized}: Controls placebo estimates mismatch",
    )
    np.testing.assert_allclose(
        py_result.placebos.std_errors,
        np.array(r_result["placebo_se"]),
        rtol=1e-10,
        err_msg=f"normalized={normalized}: Controls placebo standard errors mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_controls_ate(favara_imbs_data, favara_imbs_csv_path):
    r_result = r_did_multiplegt(favara_imbs_csv_path, effects=3, controls=["Dl_hpi"])

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    if "ate_estimate" not in r_result:
        pytest.fail("R did not return ATE estimate")

    py_result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=3,
        xformla="~ Dl_hpi",
    )

    np.testing.assert_allclose(
        py_result.ate.estimate,
        r_result["ate_estimate"],
        rtol=1e-10,
        err_msg="Controls ATE estimate mismatch",
    )
    np.testing.assert_allclose(
        py_result.ate.std_error,
        r_result["ate_se"],
        rtol=1e-10,
        err_msg="Controls ATE standard error mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_controls_collinear(favara_imbs_data, favara_imbs_csv_path):
    r_result = r_did_multiplegt(favara_imbs_csv_path, effects=3, placebo=2, controls=["Dl_hpi", "w1"])

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    with pytest.warns(UserWarning, match="collinear"):
        py_result = did_multiplegt(
            favara_imbs_data,
            yname="Dl_vloans_b",
            idname="county",
            tname="year",
            dname="inter_bra",
            effects=3,
            placebo=2,
            xformla="~ Dl_hpi + w1",
        )

    np.testing.assert_allclose(
        py_result.effects.estimates,
        np.array(r_result["effect_estimates"]),
        rtol=1e-10,
        err_msg="Collinear controls: Effect estimates mismatch",
    )
    np.testing.assert_allclose(
        py_result.effects.std_errors,
        np.array(r_result["effect_se"]),
        rtol=1e-10,
        err_msg="Collinear controls: Effect standard errors mismatch",
    )
    np.testing.assert_allclose(
        py_result.placebos.estimates,
        np.array(r_result["placebo_estimates"]),
        rtol=1e-10,
        err_msg="Collinear controls: Placebo estimates mismatch",
    )
    np.testing.assert_allclose(
        py_result.placebos.std_errors,
        np.array(r_result["placebo_se"]),
        rtol=1e-10,
        err_msg="Collinear controls: Placebo standard errors mismatch",
    )


def _generate_synthetic_het_data(seed=315):
    """Generate a synthetic panel dataset where HC2 standard errors are well-defined.

    R's ``vcovHC(type="HC2")`` returns NaN SEs on Favara & Imbs for predict_het
    because the hat matrix has leverage values near 1.  This synthetic dataset
    provides enough group/time variation (100 groups, 8 periods, staggered
    adoption across 3 cohorts) to keep leverage values small, so both R and
    Python produce finite HC2 and HC2-BM standard errors for cross-validation.

    The DGP includes a treatment-covariate interaction (``0.3 * D * covariate``)
    so the heterogeneity regression has a real signal to detect.
    """
    rng = np.random.default_rng(seed)
    n_groups = 100
    n_periods = 8
    groups = np.repeat(np.arange(1, n_groups + 1), n_periods)
    times = np.tile(np.arange(1, n_periods + 1), n_groups)

    switch_time = np.full(n_groups + 1, n_periods + 1)
    switch_time[1:31] = 4
    switch_time[31:51] = 5
    switch_time[51:66] = 6
    treatment = (times >= switch_time[groups]).astype(float)
    covariate = (groups % 3).astype(float)
    cluster = ((groups - 1) // 10 + 1).astype(int)
    group_fe = rng.standard_normal(n_groups)
    time_fe = rng.standard_normal(n_periods)
    y = (
        group_fe[groups - 1]
        + time_fe[times - 1]
        + 0.5 * covariate
        + 2.0 * treatment
        + 0.3 * treatment * covariate
        + rng.standard_normal(len(groups)) * 0.5
    )

    return pl.DataFrame(
        {
            "group": groups,
            "time": times,
            "outcome": y,
            "treatment": treatment,
            "covariate": covariate,
            "cluster_id": cluster,
        }
    )


def _r_did_multiplegt_synthetic(data_path, predict_het_hc2bm=False):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    hc2bm_str = "TRUE" if predict_het_hc2bm else "FALSE"
    cluster_str = '"cluster_id"' if predict_het_hc2bm else "NULL"

    r_script = f"""
options(rgl.useNULL = TRUE)
suppressPackageStartupMessages(library(DIDmultiplegtDYN))
library(polars)
library(jsonlite)

data <- read.csv("{data_path}")

result <- tryCatch(
  suppressWarnings(did_multiplegt_dyn(
    df = data,
    outcome = "outcome",
    group = "group",
    time = "time",
    treatment = "treatment",
    effects = 3,
    cluster = {cluster_str},
    predict_het = list(c("covariate"), -1),
    predict_het_hc2bm = {hc2bm_str},
    graph_off = TRUE
  )),
  error = function(e) NULL
)

if (is.null(result)) {{
    write_json(list(error = "R estimation failed"), "{result_path}")
    quit(status = 0)
}}

r <- result$results
out <- list()

if (!is.null(r$Effects)) {{
    out$effect_estimates <- as.numeric(r$Effects[, "Estimate"])
    out$effect_se <- as.numeric(r$Effects[, "SE"])
}}

if (!is.null(r$predict_het)) {{
    het <- r$predict_het
    out$het_effects <- het$effect
    out$het_covariates <- het$covariate
    out$het_estimates <- het$Estimate
    out$het_se <- het$SE
    out$het_t <- het$t
    out$het_pf <- het$pF
}}

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path, timeout=300)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_predict_het_estimates(favara_imbs_data, favara_imbs_csv_path):
    r_result = r_did_multiplegt(
        favara_imbs_csv_path,
        effects=2,
        predict_het=(["state_n"], [-1]),
    )

    if r_result is None or "error" in r_result:
        pytest.skip("R predict_het not supported in installed R package version")

    if "het_estimates" not in r_result:
        pytest.skip("R did not return predict_het results")

    r_estimates = np.array(r_result["het_estimates"], dtype=float)
    r_se = np.array(r_result.get("het_se", []), dtype=float)

    with pytest.warns(UserWarning, match="fit a group exactly"):
        py_result = did_multiplegt(
            favara_imbs_data,
            yname="Dl_vloans_b",
            idname="county",
            tname="year",
            dname="inter_bra",
            effects=2,
            predict_het=(["state_n"], [-1]),
        )

    assert py_result.heterogeneity is not None
    assert len(py_result.heterogeneity) > 0

    py_estimates = np.concatenate([h.estimates for h in py_result.heterogeneity])

    np.testing.assert_allclose(
        py_estimates,
        r_estimates,
        rtol=1e-10,
        err_msg="predict_het: estimates mismatch",
    )

    py_se = np.concatenate([h.std_errors for h in py_result.heterogeneity])
    np.testing.assert_array_equal(np.isnan(py_se), np.isnan(r_se))
    np.testing.assert_allclose(py_se, r_se, rtol=1e-10, err_msg="predict_het: SE mismatch")


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_predict_het_estimates_when_cluster_is_the_covariate(favara_imbs_data, favara_imbs_csv_path):
    r_result = r_did_multiplegt(
        favara_imbs_csv_path,
        effects=3,
        cluster="state_n",
        predict_het=(["state_n"], [-1]),
    )

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=3,
        cluster="state_n",
        predict_het=(["state_n"], [-1]),
    )

    py_estimates = np.concatenate([h.estimates for h in py_result.heterogeneity])
    np.testing.assert_allclose(py_estimates, np.array(r_result["het_estimates"], dtype=float), rtol=1e-10)
    np.testing.assert_allclose(py_result.effects.std_errors, r_result["effect_se"], rtol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_predict_het_hc2bm_warns_without_cluster(synthetic_het_data):
    py_hc2 = did_multiplegt(
        synthetic_het_data,
        yname="outcome",
        idname="group",
        tname="time",
        dname="treatment",
        effects=2,
        predict_het=(["covariate"], [-1]),
    )

    with pytest.warns(UserWarning, match="predict_het_hc2bm has no effect"):
        py_hc2bm = did_multiplegt(
            synthetic_het_data,
            yname="outcome",
            idname="group",
            tname="time",
            dname="treatment",
            effects=2,
            predict_het=(["covariate"], [-1]),
            predict_het_hc2bm=True,
        )

    assert py_hc2.heterogeneity is not None
    assert py_hc2bm.heterogeneity is not None

    py_hc2_se = np.concatenate([h.std_errors for h in py_hc2.heterogeneity])
    py_hc2bm_se = np.concatenate([h.std_errors for h in py_hc2bm.heterogeneity])

    assert np.all(np.isfinite(py_hc2_se))
    assert np.all(np.isfinite(py_hc2bm_se))
    np.testing.assert_allclose(py_hc2_se, py_hc2bm_se, rtol=1e-10)


@pytest.fixture(scope="module")
def synthetic_het_data():
    return _generate_synthetic_het_data()


@pytest.fixture(scope="module")
def synthetic_het_csv_path(synthetic_het_data):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        synthetic_het_data.write_csv(f.name)
        return f.name


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_predict_het_synthetic_estimates(synthetic_het_data, synthetic_het_csv_path):
    r_result = _r_did_multiplegt_synthetic(synthetic_het_csv_path, predict_het_hc2bm=False)

    if r_result is None or "error" in r_result:
        pytest.skip("R predict_het failed on synthetic data")

    if "het_estimates" not in r_result:
        pytest.skip("R did not return predict_het results for synthetic data")

    r_estimates = np.array(r_result["het_estimates"], dtype=float)
    r_se = np.array(r_result.get("het_se", []), dtype=float)

    if len(r_se) == 0:
        pytest.fail("R returned het estimates but no het SEs for synthetic data")

    assert np.all(np.isfinite(r_se)), f"R HC2 SEs should be finite on synthetic data, got: {r_se}"

    py_result = did_multiplegt(
        synthetic_het_data,
        yname="outcome",
        idname="group",
        tname="time",
        dname="treatment",
        effects=3,
        predict_het=(["covariate"], [-1]),
    )

    assert py_result.heterogeneity is not None
    assert len(py_result.heterogeneity) > 0

    py_estimates = np.concatenate([h.estimates for h in py_result.heterogeneity])
    py_se = np.concatenate([h.std_errors for h in py_result.heterogeneity])

    assert np.all(np.isfinite(py_se)), f"Python HC2 SEs should be finite on synthetic data, got: {py_se}"

    assert len(py_estimates) == len(r_estimates), (
        f"Length mismatch: Python returned {len(py_estimates)} estimates, R returned {len(r_estimates)}"
    )

    np.testing.assert_allclose(
        py_estimates,
        r_estimates,
        rtol=1e-6,
        err_msg="predict_het synthetic: estimates mismatch",
    )
    np.testing.assert_allclose(
        py_se,
        r_se,
        rtol=1e-6,
        err_msg="predict_het synthetic: SE mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_predict_het_hc2bm_synthetic(synthetic_het_data, synthetic_het_csv_path):
    r_hc2 = _r_did_multiplegt_synthetic(synthetic_het_csv_path, predict_het_hc2bm=False)
    r_hc2bm = _r_did_multiplegt_synthetic(synthetic_het_csv_path, predict_het_hc2bm=True)

    if r_hc2 is None or "error" in r_hc2 or "het_se" not in r_hc2:
        pytest.skip("R predict_het HC2 failed on synthetic data")
    if r_hc2bm is None or "error" in r_hc2bm or "het_se" not in r_hc2bm:
        pytest.skip("R predict_het_hc2bm failed on synthetic data")

    r_hc2_se = np.array(r_hc2["het_se"], dtype=float)
    r_hc2bm_se = np.array(r_hc2bm["het_se"], dtype=float)

    assert np.all(np.isfinite(r_hc2_se)), f"R HC2 SEs not finite on synthetic data: {r_hc2_se}"
    assert np.all(np.isfinite(r_hc2bm_se)), f"R HC2-BM SEs not finite on synthetic data: {r_hc2bm_se}"
    assert not np.allclose(r_hc2_se, r_hc2bm_se, rtol=1e-4), (
        "R: HC2 and HC2-BM should produce different SEs on synthetic data"
    )

    py_hc2 = did_multiplegt(
        synthetic_het_data,
        yname="outcome",
        idname="group",
        tname="time",
        dname="treatment",
        effects=3,
        predict_het=(["covariate"], [-1]),
    )
    py_hc2bm = did_multiplegt(
        synthetic_het_data,
        yname="outcome",
        idname="group",
        tname="time",
        dname="treatment",
        effects=3,
        cluster="cluster_id",
        predict_het=(["covariate"], [-1]),
        predict_het_hc2bm=True,
    )

    assert py_hc2.heterogeneity is not None
    assert py_hc2bm.heterogeneity is not None

    py_hc2_se = np.concatenate([h.std_errors for h in py_hc2.heterogeneity])
    py_hc2bm_se = np.concatenate([h.std_errors for h in py_hc2bm.heterogeneity])

    assert np.all(np.isfinite(py_hc2_se)), f"Python HC2 SEs not finite on synthetic data: {py_hc2_se}"
    assert np.all(np.isfinite(py_hc2bm_se)), f"Python HC2-BM SEs not finite on synthetic data: {py_hc2bm_se}"
    assert not np.allclose(py_hc2_se, py_hc2bm_se, rtol=1e-4), (
        "Python: HC2 and HC2-BM should produce different SEs on synthetic data"
    )

    np.testing.assert_allclose(
        py_hc2_se,
        r_hc2_se,
        rtol=1e-6,
        err_msg="predict_het synthetic HC2: R vs Python SE mismatch",
    )
    np.testing.assert_allclose(
        py_hc2bm_se,
        r_hc2bm_se,
        rtol=1e-6,
        err_msg="predict_het synthetic HC2-BM: R vs Python SE mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_clustered_inference(favara_imbs_data, favara_imbs_csv_path):
    r_result = r_did_multiplegt(favara_imbs_csv_path, effects=5, placebo=3, cluster="state_n")

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=5,
        placebo=3,
        cluster="state_n",
    )

    np.testing.assert_allclose(py_result.effects.std_errors, r_result["effect_se"], rtol=1e-10)
    np.testing.assert_allclose(py_result.placebos.std_errors, r_result["placebo_se"], rtol=1e-10)
    np.testing.assert_allclose(py_result.ate.std_error, r_result["ate_se"], rtol=1e-10)
    np.testing.assert_allclose(py_result.placebo_joint_test["p_value"], r_result["placebo_joint_pvalue"], rtol=1e-10)
    np.testing.assert_array_equal(py_result.ate.n_observations, r_result["ate_n"])
    np.testing.assert_array_equal(py_result.ate.n_switchers, r_result["ate_switchers"])


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_clustered_normalized_tests(favara_imbs_data, favara_imbs_csv_path):
    kwargs = {
        "effects": 5,
        "placebo": 3,
        "cluster": "state_n",
        "normalized": True,
        "same_switchers": True,
        "effects_equal": True,
    }
    r_result = r_did_multiplegt(favara_imbs_csv_path, **kwargs)

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        **kwargs,
    )

    np.testing.assert_allclose(py_result.effects.std_errors, r_result["effect_se"], rtol=1e-10)
    np.testing.assert_allclose(py_result.placebos.std_errors, r_result["placebo_se"], rtol=1e-10)
    np.testing.assert_allclose(py_result.effects_equal_test["p_value"], r_result["effects_equal_pvalue"], rtol=1e-10)
    np.testing.assert_allclose(py_result.placebo_joint_test["p_value"], r_result["placebo_joint_pvalue"], rtol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_less_conservative_se_clustered(favara_imbs_data, favara_imbs_csv_path):
    r_result = r_did_multiplegt(
        favara_imbs_csv_path,
        effects=5,
        placebo=3,
        cluster="state_n",
        less_conservative_se=True,
    )

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    kwargs = {
        "yname": "Dl_vloans_b",
        "idname": "county",
        "tname": "year",
        "dname": "inter_bra",
        "effects": 5,
        "placebo": 3,
        "cluster": "state_n",
    }
    py_result = did_multiplegt(favara_imbs_data, **kwargs, less_conservative_se=True)
    py_default = did_multiplegt(favara_imbs_data, **kwargs)

    np.testing.assert_allclose(py_result.effects.std_errors, r_result["effect_se"], rtol=1e-10)
    np.testing.assert_allclose(py_result.placebos.std_errors, r_result["placebo_se"], rtol=1e-10)
    np.testing.assert_allclose(py_result.ate.std_error, r_result["ate_se"], rtol=1e-10)
    np.testing.assert_array_equal(py_result.placebos.std_errors, py_default.placebos.std_errors)


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_weighted_clustered_inference(favara_imbs_data, favara_imbs_csv_path):
    r_result = r_did_multiplegt(favara_imbs_csv_path, effects=5, placebo=3, cluster="state_n", weight="w1")

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        weightsname="w1",
        effects=5,
        placebo=3,
        cluster="state_n",
    )

    np.testing.assert_allclose(py_result.effects.estimates, r_result["effect_estimates"], rtol=1e-10)
    np.testing.assert_allclose(py_result.effects.std_errors, r_result["effect_se"], rtol=1e-10)
    np.testing.assert_allclose(py_result.placebos.std_errors, r_result["placebo_se"], rtol=1e-10)
    np.testing.assert_allclose(py_result.ate.std_error, r_result["ate_se"], rtol=1e-10)
    np.testing.assert_allclose(py_result.placebo_joint_test["p_value"], r_result["placebo_joint_pvalue"], rtol=1e-10)
    np.testing.assert_array_equal(py_result.ate.n_switchers, r_result["ate_switchers"])


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_only_never_switchers_clustered(favara_imbs_data, favara_imbs_csv_path):
    r_result = r_did_multiplegt(favara_imbs_csv_path, effects=5, cluster="state_n", only_never_switchers=True)

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=5,
        cluster="state_n",
        only_never_switchers=True,
    )

    np.testing.assert_allclose(py_result.effects.std_errors, r_result["effect_se"], rtol=1e-10)
    np.testing.assert_allclose(py_result.ate.std_error, r_result["ate_se"], rtol=1e-10)
    np.testing.assert_array_equal(py_result.ate.n_observations, r_result["ate_n"])


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_trends_lin_reports_no_ate(favara_imbs_data, favara_imbs_csv_path):
    r_result = r_did_multiplegt(favara_imbs_csv_path, effects=3, cluster="state_n", trends_lin=True)

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    with pytest.warns(UserWarning, match="ATE"):
        py_result = did_multiplegt(
            favara_imbs_data,
            yname="Dl_vloans_b",
            idname="county",
            tname="year",
            dname="inter_bra",
            effects=3,
            cluster="state_n",
            trends_lin=True,
        )

    assert r_result["ate_missing"] == [True]
    assert py_result.ate is None
    np.testing.assert_allclose(py_result.effects.std_errors, r_result["effect_se"], rtol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
@pytest.mark.parametrize(
    ("cluster", "less_conservative_se", "weight"),
    [(None, False, None), ("cl", False, None), (None, True, None), ("cl", True, None), ("cl", False, "w")],
)
def test_unbalanced_panel_inference(
    didinter_unbalanced_data, didinter_unbalanced_csv_path, cluster, less_conservative_se, weight
):
    kwargs = {"effects": 3, "placebo": 2, "cluster": cluster, "less_conservative_se": less_conservative_se}
    r_result = r_did_multiplegt(
        didinter_unbalanced_csv_path,
        **kwargs,
        weight=weight,
        outcome="y",
        group="g",
        time="t",
        treatment="d",
    )

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(
        didinter_unbalanced_data,
        yname="y",
        idname="g",
        tname="t",
        dname="d",
        weightsname=weight,
        **kwargs,
    )

    np.testing.assert_allclose(py_result.effects.estimates, r_result["effect_estimates"], rtol=1e-10)
    np.testing.assert_allclose(py_result.effects.std_errors, r_result["effect_se"], rtol=1e-10)
    np.testing.assert_allclose(py_result.placebos.std_errors, r_result["placebo_se"], rtol=1e-10)
    np.testing.assert_array_equal(py_result.placebos.n_switchers, r_result["placebo_n_switchers"])
    np.testing.assert_allclose(py_result.placebo_joint_test["p_value"], r_result["placebo_joint_pvalue"], rtol=1e-10)
    np.testing.assert_allclose(py_result.ate.estimate, r_result["ate_estimate"], rtol=1e-10)
    np.testing.assert_allclose(py_result.ate.std_error, r_result["ate_se"], rtol=1e-10)
    np.testing.assert_array_equal(py_result.ate.n_observations, r_result["ate_n"])


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
@pytest.mark.filterwarnings("ignore:did_multiplegt computes analytical standard errors:UserWarning")
@pytest.mark.parametrize("cluster", ["state_n", None])
def test_bootstrap_matches_reference_on_its_draws(
    favara_imbs_data, favara_imbs_csv_path, fixed_draws, reference_draws, tmp_path, cluster
):
    r_result = r_did_multiplegt_bootstrap(favara_imbs_csv_path, tmp_path, reps=10, seed=7, cluster=cluster)

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    draws = reference_draws(tmp_path, favara_imbs_data[cluster or "county"])
    py_result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=3,
        placebo=2,
        cluster=cluster,
        boot=True,
        biters=len(draws),
        random_state=fixed_draws(draws),
    )

    assert len(draws) == 10
    np.testing.assert_allclose(py_result.effects.std_errors, r_result["effect_se"], rtol=1e-10)
    np.testing.assert_allclose(py_result.placebos.std_errors, r_result["placebo_se"], rtol=1e-10)
    np.testing.assert_allclose(py_result.ate.std_error, r_result["ate_se"], rtol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
@pytest.mark.filterwarnings("ignore:did_multiplegt computes analytical standard errors:UserWarning")
@pytest.mark.filterwarnings("ignore:Dropped:UserWarning")
def test_bootstrap_matches_reference_on_its_draws_with_missing_states(
    favara_missing_states, favara_missing_states_csv_path, fixed_draws, reference_draws, tmp_path
):
    r_result = r_did_multiplegt_bootstrap(favara_missing_states_csv_path, tmp_path, reps=10, seed=11, cluster="state_n")

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    draws = reference_draws(tmp_path, favara_missing_states["state_n"])
    py_result = did_multiplegt(
        favara_missing_states,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=3,
        placebo=2,
        cluster="state_n",
        boot=True,
        biters=len(draws),
        random_state=fixed_draws(draws),
    )

    assert len(draws) == 10
    np.testing.assert_allclose(py_result.effects.std_errors, r_result["effect_se"], rtol=1e-10)
    np.testing.assert_allclose(py_result.placebos.std_errors, r_result["placebo_se"], rtol=1e-10)
    np.testing.assert_allclose(py_result.ate.std_error, r_result["ate_se"], rtol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
@pytest.mark.filterwarnings("ignore:did_multiplegt computes analytical standard errors:UserWarning")
@pytest.mark.parametrize("switchers", ["", "out"])
def test_bootstrap_matches_reference_on_its_draws_in_both_directions(
    didinter_two_way_data, didinter_two_way_csv_path, fixed_draws, reference_draws, tmp_path, switchers
):
    columns = {"outcome": "y", "group": "g", "time": "t", "treatment": "d"}
    r_result = r_did_multiplegt_bootstrap(
        didinter_two_way_csv_path, tmp_path, reps=10, seed=5, cluster="cl", switchers=switchers, **columns
    )

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    draws = reference_draws(tmp_path, didinter_two_way_data["cl"])
    py_result = did_multiplegt(
        didinter_two_way_data,
        yname="y",
        idname="g",
        tname="t",
        dname="d",
        effects=3,
        placebo=2,
        switchers=switchers,
        cluster="cl",
        boot=True,
        biters=len(draws),
        random_state=fixed_draws(draws),
    )

    assert len(draws) == 10
    np.testing.assert_allclose(py_result.effects.std_errors, r_result["effect_se"], rtol=1e-10)
    np.testing.assert_allclose(py_result.placebos.std_errors, r_result["placebo_se"], rtol=1e-10)
    np.testing.assert_allclose(py_result.ate.std_error, r_result["ate_se"], rtol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
@pytest.mark.filterwarnings("ignore:Dropped:UserWarning")
@pytest.mark.parametrize("cluster", ["state_n", None])
def test_rows_with_a_missing_cluster_leave_the_sample(favara_missing_states, favara_missing_states_csv_path, cluster):
    kwargs = {"effects": 5, "placebo": 3, "cluster": cluster}
    r_result = r_did_multiplegt(favara_missing_states_csv_path, **kwargs)

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(
        favara_missing_states, yname="Dl_vloans_b", idname="county", tname="year", dname="inter_bra", **kwargs
    )

    np.testing.assert_allclose(py_result.effects.estimates, r_result["effect_estimates"], rtol=1e-10)
    np.testing.assert_allclose(py_result.effects.std_errors, r_result["effect_se"], rtol=1e-10)
    np.testing.assert_array_equal(py_result.effects.n_switchers, r_result["effect_n_switchers"])
    np.testing.assert_array_equal(py_result.effects.n_observations, r_result["effect_n"])
    np.testing.assert_allclose(py_result.placebos.estimates, r_result["placebo_estimates"], rtol=1e-10)
    np.testing.assert_allclose(py_result.placebos.std_errors, r_result["placebo_se"], rtol=1e-10)
    np.testing.assert_allclose(py_result.ate.estimate, r_result["ate_estimate"], rtol=1e-10)
    np.testing.assert_allclose(py_result.ate.std_error, r_result["ate_se"], rtol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_groups_in_more_than_one_cluster_stop_the_estimation(favara_moved_county, favara_moved_county_csv_path):
    r_result = r_did_multiplegt(favara_moved_county_csv_path, effects=2, cluster="state_n")
    r_unclustered = r_did_multiplegt(favara_moved_county_csv_path, effects=2)

    with pytest.raises(ValueError, match="Some groups belong to more than one cluster in 'state_n'"):
        did_multiplegt(
            favara_moved_county,
            yname="Dl_vloans_b",
            idname="county",
            tname="year",
            dname="inter_bra",
            effects=2,
            cluster="state_n",
        )
    assert "error" in r_result
    assert "error" not in r_unclustered


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
@pytest.mark.parametrize("same_switchers", [False, True])
def test_placebos_stop_at_the_number_of_effects(favara_imbs_data, favara_imbs_csv_path, same_switchers):
    kwargs = {
        "effects": 1,
        "placebo": 3,
        "cluster": "state_n",
        "same_switchers": same_switchers,
        "same_switchers_pl": same_switchers,
    }
    r_result = r_did_multiplegt(favara_imbs_csv_path, **kwargs)

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    with pytest.warns(UserWarning, match="the number of placebos cannot exceed the number of effects"):
        py_result = did_multiplegt(
            favara_imbs_data, yname="Dl_vloans_b", idname="county", tname="year", dname="inter_bra", **kwargs
        )

    np.testing.assert_allclose(py_result.placebos.estimates, r_result["placebo_estimates"], rtol=1e-10)
    np.testing.assert_allclose(py_result.placebos.std_errors, r_result["placebo_se"], rtol=1e-10)
    np.testing.assert_array_equal(py_result.placebos.n_switchers, r_result["placebo_n_switchers"])


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
@pytest.mark.parametrize(
    ("switchers", "cluster", "normalized", "weight"),
    [
        ("", None, False, None),
        ("in", None, False, None),
        ("out", None, False, None),
        ("", "cl", False, "w"),
        ("", None, True, None),
        ("out", "cl", True, None),
    ],
)
def test_switchers_in_and_out(didinter_two_way_data, didinter_two_way_csv_path, switchers, cluster, normalized, weight):
    kwargs = {"effects": 3, "placebo": 2, "switchers": switchers, "cluster": cluster, "normalized": normalized}
    r_result = r_did_multiplegt(
        didinter_two_way_csv_path, **kwargs, weight=weight, outcome="y", group="g", time="t", treatment="d"
    )

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(
        didinter_two_way_data, yname="y", idname="g", tname="t", dname="d", weightsname=weight, **kwargs
    )

    np.testing.assert_allclose(py_result.effects.estimates, r_result["effect_estimates"], rtol=1e-10)
    np.testing.assert_allclose(py_result.effects.std_errors, r_result["effect_se"], rtol=1e-10)
    np.testing.assert_array_equal(py_result.effects.n_switchers, r_result["effect_n_switchers"])
    np.testing.assert_array_equal(py_result.effects.n_observations, r_result["effect_n"])
    np.testing.assert_allclose(py_result.placebos.estimates, r_result["placebo_estimates"], rtol=1e-10)
    np.testing.assert_allclose(py_result.placebos.std_errors, r_result["placebo_se"], rtol=1e-10)
    np.testing.assert_array_equal(py_result.placebos.n_switchers, r_result["placebo_n_switchers"])
    np.testing.assert_allclose(py_result.placebo_joint_test["p_value"], r_result["placebo_joint_pvalue"], rtol=1e-10)
    np.testing.assert_allclose(py_result.ate.estimate, r_result["ate_estimate"], rtol=1e-10)
    np.testing.assert_allclose(py_result.ate.std_error, r_result["ate_se"], rtol=1e-10)
    np.testing.assert_array_equal(py_result.ate.n_observations, r_result["ate_n"])


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_reference_pooled_placebo_counts_cover_only_switchers_in(didinter_two_way_data, didinter_two_way_csv_path):
    kwargs = {"effects": 3, "placebo": 2}
    columns = {"outcome": "y", "group": "g", "time": "t", "treatment": "d"}
    r_pooled = r_did_multiplegt(didinter_two_way_csv_path, **kwargs, **columns)
    r_rises = r_did_multiplegt(didinter_two_way_csv_path, **kwargs, switchers="in", **columns)

    if r_pooled is None or "error" in r_pooled or r_rises is None or "error" in r_rises:
        pytest.fail("R estimation failed")

    py_kwargs = {"yname": "y", "idname": "g", "tname": "t", "dname": "d", **kwargs}
    py_pooled = did_multiplegt(didinter_two_way_data, **py_kwargs)
    py_rises = did_multiplegt(didinter_two_way_data, **py_kwargs, switchers="in")

    np.testing.assert_array_equal(py_rises.placebos.n_observations, r_rises["placebo_n"])
    np.testing.assert_array_equal(py_pooled.effects.n_observations, r_pooled["effect_n"])
    np.testing.assert_array_equal(r_pooled["placebo_n"], py_rises.placebos.n_observations)


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
@pytest.mark.filterwarnings("ignore:Requested:UserWarning")
@pytest.mark.parametrize(
    ("switchers", "cluster", "effects", "placebo"),
    [("", None, 4, 2), ("", "cl", 4, 2), ("", None, 7, 5), ("out", "cl", 5, 4)],
)
def test_all_groups_switch(
    didinter_all_switch_data, didinter_all_switch_csv_path, switchers, cluster, effects, placebo
):
    kwargs = {"effects": effects, "placebo": placebo, "switchers": switchers, "cluster": cluster}
    r_result = r_did_multiplegt(didinter_all_switch_csv_path, **kwargs, outcome="y", group="g", time="t", treatment="d")

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(didinter_all_switch_data, yname="y", idname="g", tname="t", dname="d", **kwargs)

    np.testing.assert_allclose(py_result.effects.estimates, r_result["effect_estimates"], rtol=1e-10)
    np.testing.assert_allclose(py_result.effects.std_errors, r_result["effect_se"], rtol=1e-10)
    np.testing.assert_array_equal(py_result.effects.n_switchers, r_result["effect_n_switchers"])
    np.testing.assert_array_equal(py_result.effects.n_observations, r_result["effect_n"])
    np.testing.assert_allclose(py_result.placebos.estimates, r_result["placebo_estimates"], rtol=1e-10)
    np.testing.assert_allclose(py_result.placebos.std_errors, r_result["placebo_se"], rtol=1e-10)
    np.testing.assert_array_equal(py_result.placebos.n_switchers, r_result["placebo_n_switchers"])
    np.testing.assert_allclose(py_result.placebo_joint_test["p_value"], r_result["placebo_joint_pvalue"], rtol=1e-10)
    np.testing.assert_allclose(py_result.ate.estimate, r_result["ate_estimate"], rtol=1e-10)
    np.testing.assert_allclose(py_result.ate.std_error, r_result["ate_se"], rtol=1e-10)
    assert py_result.n_never_switchers == 0


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
@pytest.mark.filterwarnings("ignore:When trends_lin=True:UserWarning")
@pytest.mark.parametrize(
    "kwargs",
    [
        {"effects": 5, "placebo": 3, "cluster": "state_n"},
        {"effects": 3, "placebo": 1, "cluster": "state_n", "trends_lin": True},
    ],
    ids=["page", "trends-lin"],
)
def test_unevenly_spaced_periods(favara_imbs_data, favara_gapped_years, favara_gapped_years_csv_path, kwargs):
    r_result = r_did_multiplegt(favara_gapped_years_csv_path, **kwargs)

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    columns = {"yname": "Dl_vloans_b", "idname": "county", "tname": "year", "dname": "inter_bra"}
    py_result = did_multiplegt(favara_gapped_years, **columns, **kwargs)
    py_consecutive = did_multiplegt(favara_imbs_data, **columns, **kwargs)

    np.testing.assert_allclose(py_result.effects.estimates, r_result["effect_estimates"], rtol=1e-10)
    np.testing.assert_allclose(py_result.effects.std_errors, r_result["effect_se"], rtol=1e-10)
    np.testing.assert_array_equal(py_result.effects.n_switchers, r_result["effect_n_switchers"])
    np.testing.assert_allclose(py_result.placebos.estimates, r_result["placebo_estimates"], rtol=1e-10)
    np.testing.assert_allclose(py_result.placebos.std_errors, r_result["placebo_se"], rtol=1e-10)
    np.testing.assert_allclose(py_result.effects.estimates, py_consecutive.effects.estimates, rtol=1e-12)
    np.testing.assert_allclose(py_result.effects.std_errors, py_consecutive.effects.std_errors, rtol=1e-12)


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_trends_nonparam_with_a_region_without_never_switchers(favara_census_regions, favara_census_regions_csv_path):
    kwargs = {"effects": 5, "placebo": 3, "cluster": "state_n", "trends_nonparam": ["region"]}
    r_result = r_did_multiplegt(favara_census_regions_csv_path, **kwargs)

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(
        favara_census_regions, yname="Dl_vloans_b", idname="county", tname="year", dname="inter_bra", **kwargs
    )

    np.testing.assert_allclose(py_result.effects.estimates, r_result["effect_estimates"], rtol=1e-10)
    np.testing.assert_allclose(py_result.effects.std_errors, r_result["effect_se"], rtol=1e-10)
    np.testing.assert_array_equal(py_result.effects.n_switchers, r_result["effect_n_switchers"])
    np.testing.assert_array_equal(py_result.effects.n_observations, r_result["effect_n"])
    np.testing.assert_allclose(py_result.placebos.estimates, r_result["placebo_estimates"], rtol=1e-10)
    np.testing.assert_allclose(py_result.placebos.std_errors, r_result["placebo_se"], rtol=1e-10)
    np.testing.assert_array_equal(py_result.placebos.n_switchers, r_result["placebo_n_switchers"])


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
@pytest.mark.filterwarnings("ignore:When continuous > 0:UserWarning")
@pytest.mark.parametrize(
    ("kwargs", "weight", "controls"),
    [
        ({"continuous": 1}, None, None),
        ({"continuous": 2, "cluster": "cl"}, None, None),
        ({"continuous": 1, "normalized": True}, None, None),
        ({"continuous": 1}, "w", None),
        ({"continuous": 1}, None, ["x"]),
        ({"continuous": 1, "switchers": "in"}, None, None),
        ({"continuous": 3, "less_conservative_se": True}, None, None),
    ],
    ids=["degree-one", "degree-two-clustered", "normalized", "weighted", "controls", "rises", "degree-three-lcs"],
)
def test_continuous_baselines(didinter_continuous_data, didinter_continuous_csv_path, kwargs, weight, controls):
    columns = {"outcome": "y", "group": "g", "time": "t", "treatment": "d"}
    r_result = r_did_multiplegt(
        didinter_continuous_csv_path, effects=3, placebo=2, weight=weight, controls=controls, **columns, **kwargs
    )

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(
        didinter_continuous_data,
        yname="y",
        idname="g",
        tname="t",
        dname="d",
        effects=3,
        placebo=2,
        weightsname=weight,
        xformla="~ " + " + ".join(controls) if controls else "~1",
        **kwargs,
    )

    np.testing.assert_allclose(py_result.effects.estimates, r_result["effect_estimates"], rtol=1e-10)
    np.testing.assert_allclose(py_result.effects.std_errors, r_result["effect_se"], rtol=1e-10)
    np.testing.assert_array_equal(py_result.effects.n_switchers, r_result["effect_n_switchers"])
    np.testing.assert_array_equal(py_result.effects.n_observations, r_result["effect_n"])
    np.testing.assert_allclose(py_result.placebos.estimates, r_result["placebo_estimates"], rtol=1e-10)
    np.testing.assert_allclose(py_result.placebos.std_errors, r_result["placebo_se"], rtol=1e-10)
    np.testing.assert_array_equal(py_result.placebos.n_switchers, r_result["placebo_n_switchers"])
    np.testing.assert_allclose(py_result.placebo_joint_test["p_value"], r_result["placebo_joint_pvalue"], rtol=1e-10)
    np.testing.assert_allclose(py_result.ate.estimate, r_result["ate_estimate"], rtol=1e-10)
    np.testing.assert_allclose(py_result.ate.std_error, r_result["ate_se"], rtol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
@pytest.mark.filterwarnings("ignore:When continuous > 0:UserWarning")
def test_continuous_absent_rows_match_reference_with_those_rows_empty(
    didinter_continuous_gaps, didinter_continuous_gaps_csv_paths
):
    absent, _ = didinter_continuous_gaps
    absent_path, empty_path = didinter_continuous_gaps_csv_paths
    columns = {"outcome": "y", "group": "g", "time": "t", "treatment": "d"}
    r_absent = r_did_multiplegt(absent_path, effects=3, placebo=2, continuous=1, **columns)
    r_empty = r_did_multiplegt(empty_path, effects=3, placebo=2, continuous=1, **columns)

    if r_absent is None or "error" in r_absent or r_empty is None or "error" in r_empty:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(absent, yname="y", idname="g", tname="t", dname="d", effects=3, placebo=2, continuous=1)

    np.testing.assert_allclose(py_result.effects.estimates, r_empty["effect_estimates"], rtol=1e-10)
    np.testing.assert_allclose(py_result.effects.std_errors, r_empty["effect_se"], rtol=1e-10)
    np.testing.assert_allclose(py_result.placebos.estimates, r_empty["placebo_estimates"], rtol=1e-10)
    np.testing.assert_allclose(py_result.placebos.std_errors, r_empty["placebo_se"], rtol=1e-10)
    assert not np.allclose(r_absent["effect_estimates"], r_empty["effect_estimates"], rtol=1e-6)


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
@pytest.mark.parametrize(
    ("cluster", "weight", "same_switchers_pl"),
    [(None, None, False), (None, None, True), ("cl", None, True), ("cl", "w", False)],
)
def test_same_switchers_on_an_unbalanced_panel(
    didinter_unbalanced_data, didinter_unbalanced_csv_path, cluster, weight, same_switchers_pl
):
    kwargs = {
        "effects": 3,
        "placebo": 2,
        "cluster": cluster,
        "same_switchers": True,
        "same_switchers_pl": same_switchers_pl,
    }
    r_result = r_did_multiplegt(
        didinter_unbalanced_csv_path, **kwargs, weight=weight, outcome="y", group="g", time="t", treatment="d"
    )

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(
        didinter_unbalanced_data, yname="y", idname="g", tname="t", dname="d", weightsname=weight, **kwargs
    )

    np.testing.assert_allclose(py_result.effects.estimates, r_result["effect_estimates"], rtol=1e-10)
    np.testing.assert_allclose(py_result.effects.std_errors, r_result["effect_se"], rtol=1e-10)
    np.testing.assert_array_equal(py_result.effects.n_switchers, r_result["effect_n_switchers"])
    np.testing.assert_array_equal(py_result.effects.n_observations, r_result["effect_n"])
    np.testing.assert_allclose(py_result.placebos.estimates, r_result["placebo_estimates"], rtol=1e-10)
    np.testing.assert_allclose(py_result.placebos.std_errors, r_result["placebo_se"], rtol=1e-10)
    np.testing.assert_array_equal(py_result.placebos.n_switchers, r_result["placebo_n_switchers"])
    np.testing.assert_array_equal(py_result.placebos.n_observations, r_result["placebo_n"])
    np.testing.assert_allclose(py_result.placebo_joint_test["p_value"], r_result["placebo_joint_pvalue"], rtol=1e-10)
    np.testing.assert_allclose(py_result.ate.estimate, r_result["ate_estimate"], rtol=1e-10)
    np.testing.assert_allclose(py_result.ate.std_error, r_result["ate_se"], rtol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_same_switchers_with_only_never_switchers(didinter_missing_outcome_data, didinter_missing_outcome_csv_path):
    kwargs = {
        "effects": 4,
        "placebo": 3,
        "only_never_switchers": True,
        "same_switchers": True,
        "same_switchers_pl": True,
    }
    r_result = r_did_multiplegt(
        didinter_missing_outcome_csv_path, **kwargs, outcome="y", group="g", time="t", treatment="d"
    )

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(didinter_missing_outcome_data, yname="y", idname="g", tname="t", dname="d", **kwargs)

    np.testing.assert_allclose(py_result.effects.estimates, r_result["effect_estimates"], rtol=1e-10)
    np.testing.assert_allclose(py_result.effects.std_errors, r_result["effect_se"], rtol=1e-10)
    np.testing.assert_array_equal(py_result.effects.n_switchers, r_result["effect_n_switchers"])
    np.testing.assert_allclose(py_result.placebos.estimates, r_result["placebo_estimates"], rtol=1e-10)
    np.testing.assert_allclose(py_result.placebos.std_errors, r_result["placebo_se"], rtol=1e-10)
    np.testing.assert_array_equal(py_result.placebos.n_switchers, r_result["placebo_n_switchers"])
    np.testing.assert_allclose(py_result.ate.estimate, r_result["ate_estimate"], rtol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_same_switchers_pl_leaves_the_effects_alone(favara_imbs_data, favara_imbs_csv_path):
    kwargs = {"effects": 5, "placebo": 3, "cluster": "state_n", "same_switchers": True}
    r_result = r_did_multiplegt(favara_imbs_csv_path, **kwargs, same_switchers_pl=True)

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    columns = {"yname": "Dl_vloans_b", "idname": "county", "tname": "year", "dname": "inter_bra"}
    py_result = did_multiplegt(favara_imbs_data, **columns, **kwargs, same_switchers_pl=True)
    py_same = did_multiplegt(favara_imbs_data, **columns, **kwargs)

    np.testing.assert_allclose(py_result.effects.estimates, r_result["effect_estimates"], rtol=1e-10)
    np.testing.assert_array_equal(py_result.effects.estimates, py_same.effects.estimates)
    np.testing.assert_allclose(py_result.placebos.estimates, r_result["placebo_estimates"], rtol=1e-10)
    np.testing.assert_allclose(py_result.placebos.std_errors, r_result["placebo_se"], rtol=1e-10)
    np.testing.assert_array_equal(py_result.placebos.n_switchers, r_result["placebo_n_switchers"])
    np.testing.assert_allclose(py_result.placebo_joint_test["p_value"], r_result["placebo_joint_pvalue"], rtol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
@pytest.mark.filterwarnings("ignore:When trends_lin=True:UserWarning")
@pytest.mark.parametrize(
    ("kwargs", "weight", "controls"),
    [
        ({}, None, None),
        ({"cluster": "cl", "normalized": True}, None, None),
        ({"switchers": "out"}, None, None),
        ({"cluster": "cl"}, None, ["x"]),
        ({"less_conservative_se": True, "effects_equal": True}, None, None),
        ({}, "w", None),
    ],
    ids=["default", "clustered-normalized", "falls", "controls", "lcs-equal-effects", "weighted"],
)
def test_trends_lin_with_missing_outcomes(
    didinter_missing_outcome_data, didinter_missing_outcome_csv_path, kwargs, weight, controls
):
    columns = {"outcome": "y", "group": "g", "time": "t", "treatment": "d"}
    r_result = r_did_multiplegt(
        didinter_missing_outcome_csv_path,
        effects=4,
        placebo=2,
        trends_lin=True,
        weight=weight,
        controls=controls,
        **columns,
        **kwargs,
    )

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(
        didinter_missing_outcome_data,
        yname="y",
        idname="g",
        tname="t",
        dname="d",
        effects=4,
        placebo=2,
        trends_lin=True,
        weightsname=weight,
        xformla="~ " + " + ".join(controls) if controls else "~1",
        **kwargs,
    )

    np.testing.assert_allclose(py_result.effects.estimates, r_result["effect_estimates"], rtol=1e-10)
    np.testing.assert_allclose(py_result.effects.std_errors, r_result["effect_se"], rtol=1e-10)
    np.testing.assert_array_equal(py_result.effects.n_observations, r_result["effect_n"])
    np.testing.assert_allclose(py_result.placebos.estimates, r_result["placebo_estimates"], rtol=1e-10)
    np.testing.assert_allclose(py_result.placebos.std_errors, r_result["placebo_se"], rtol=1e-10)
    np.testing.assert_allclose(py_result.placebo_joint_test["p_value"], r_result["placebo_joint_pvalue"], rtol=1e-10)
    assert py_result.ate is None
    assert r_result["ate_missing"] == [True]
    if weight is None:
        np.testing.assert_array_equal(py_result.effects.n_switchers, r_result["effect_n_switchers_w"])
        np.testing.assert_array_equal(py_result.placebos.n_switchers, r_result["placebo_n_switchers_w"])
    if kwargs.get("effects_equal"):
        np.testing.assert_allclose(
            py_result.effects_equal_test["p_value"], r_result["effects_equal_pvalue"], rtol=1e-10
        )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
@pytest.mark.parametrize(
    ("cluster", "weight", "hc2bm"),
    [(None, None, False), ("cl", None, True), (None, "w", False), ("cl", "w", True)],
    ids=["hc2", "hc2bm-clustered", "weighted", "hc2bm-clustered-weighted"],
)
def test_predict_het_placebo_regressions(didinter_het_data, didinter_het_csv_path, cluster, weight, hc2bm):
    kwargs = {"effects": 3, "placebo": 2, "cluster": cluster, "predict_het": (["x"], [-1]), "predict_het_hc2bm": hc2bm}
    r_result = r_did_multiplegt(
        didinter_het_csv_path, **kwargs, weight=weight, outcome="y", group="g", time="t", treatment="d"
    )

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    py_result = did_multiplegt(
        didinter_het_data, yname="y", idname="g", tname="t", dname="d", weightsname=weight, **kwargs
    )
    het = py_result.heterogeneity

    assert [h.horizon for h in het] == r_result["het_effects"] == [-2, -1, 1, 2, 3]
    np.testing.assert_allclose([h.estimates[0] for h in het], r_result["het_estimates"], rtol=1e-10)
    np.testing.assert_allclose([h.std_errors[0] for h in het], r_result["het_se"], rtol=1e-10)
    np.testing.assert_allclose([h.f_pvalue for h in het], r_result["het_pf"], rtol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
@pytest.mark.parametrize("hc2bm", [False, True])
def test_predict_het_standard_errors_are_nan_when_a_group_is_fit_exactly(favara_imbs_data, favara_imbs_csv_path, hc2bm):
    kwargs = {"effects": 2, "placebo": 1, "predict_het": (["state_n"], [-1]), "predict_het_hc2bm": hc2bm}
    r_result = r_did_multiplegt(favara_imbs_csv_path, **kwargs)

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    with pytest.warns(UserWarning, match="horizons 1, 2 fit a group exactly"):
        py_result = did_multiplegt(
            favara_imbs_data, yname="Dl_vloans_b", idname="county", tname="year", dname="inter_bra", **kwargs
        )
    py_se = np.array([h.std_errors[0] for h in py_result.heterogeneity])
    r_se = np.array(r_result["het_se"], dtype=float)

    np.testing.assert_allclose([h.estimates[0] for h in py_result.heterogeneity], r_result["het_estimates"], rtol=1e-10)
    np.testing.assert_array_equal(np.isnan(py_se), [False, True, True])
    np.testing.assert_allclose(py_se, r_se, rtol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
@pytest.mark.filterwarnings("ignore:When trends_lin=True:UserWarning")
def test_predict_het_with_trends_lin_regresses_effects_only(didinter_het_data, didinter_het_csv_path):
    kwargs = {"effects": 3, "placebo": 2, "trends_lin": True, "predict_het": (["x"], [-1])}
    r_result = r_did_multiplegt(didinter_het_csv_path, **kwargs, outcome="y", group="g", time="t", treatment="d")

    if r_result is None or "error" in r_result:
        pytest.fail("R estimation failed")

    with pytest.warns(UserWarning, match="no placebo regressions"):
        py_result = did_multiplegt(didinter_het_data, yname="y", idname="g", tname="t", dname="d", **kwargs)
    effects = np.array(r_result["het_effects"]) > 0

    assert [h.horizon for h in py_result.heterogeneity] == [1, 2, 3]
    np.testing.assert_allclose(
        [h.estimates[0] for h in py_result.heterogeneity], np.array(r_result["het_estimates"])[effects], rtol=1e-10
    )
    np.testing.assert_allclose(
        [h.std_errors[0] for h in py_result.heterogeneity], np.array(r_result["het_se"])[effects], rtol=1e-10
    )
