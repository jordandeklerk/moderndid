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
    less_conservative_se=False,
    more_granular_demeaning=False,
    controls=None,
    predict_het=None,
    predict_het_hc2bm=False,
):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    normalized_str = "TRUE" if normalized else "FALSE"
    trends_lin_str = "TRUE" if trends_lin else "FALSE"
    only_never_str = "TRUE" if only_never_switchers else "FALSE"
    same_switchers_str = "TRUE" if same_switchers else "FALSE"
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
    outcome = "Dl_vloans_b",
    group = "county",
    time = "year",
    treatment = "inter_bra",
    effects = {effects},
    placebo = {placebo},
    normalized = {normalized_str},
    cluster = {cluster_str},
    effects_equal = {effects_equal_str},
    trends_lin = {trends_lin_str},
    switchers = {switchers_str},
    only_never_switchers = {only_never_str},
    same_switchers = {same_switchers_str},
    less_conservative_se = {less_conservative_str},
    more_granular_demeaning = {more_granular_str},
    controls = {controls_str},
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
}}

if (!is.null(r$Placebos)) {{
    out$placebo_estimates <- as.numeric(r$Placebos[, "Estimate"])
    out$placebo_se <- as.numeric(r$Placebos[, "SE"])
    out$placebo_n_switchers <- as.integer(r$Placebos[, "Switchers"])
}}

if (!is.null(r$ATE)) {{
    out$ate_estimate <- r$ATE[1, "Estimate"]
    out$ate_se <- r$ATE[1, "SE"]
}}

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
        rtol=2e-4,
        atol=1e-5,
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
        rtol=1e-6,
        atol=1e-10,
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
        rtol=2e-1,
        atol=1e-2,
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
        rtol=1e-4,
        atol=1e-5,
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
        rtol=1e-2,
        atol=1e-4,
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
        rtol=1e-4,
        atol=1e-5,
        err_msg="trends_lin: Effect estimates mismatch",
    )
    np.testing.assert_allclose(
        py_result.effects.std_errors,
        np.array(r_result["effect_se"]),
        rtol=2e-4,
        atol=1e-5,
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
        rtol=3e-3,
        atol=1e-4,
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
        rtol=3e-3,
        atol=1e-4,
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
        rtol=2e-4,
        atol=1e-5,
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
        rtol=2e-1,
        atol=1e-2,
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
            rtol=3e-3,
            atol=1e-4,
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
        rtol=1e-2,
        atol=1e-4,
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
    r_has_finite_se = len(r_se) > 0 and np.all(np.isfinite(r_se))

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
        rtol=1e-4,
        err_msg="predict_het: estimates mismatch",
    )

    if r_has_finite_se:
        py_se = np.concatenate([h.std_errors for h in py_result.heterogeneity])
        np.testing.assert_allclose(
            py_se,
            r_se,
            rtol=1e-4,
            err_msg="predict_het: SE mismatch",
        )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DIDmultiplegtDYN package not available")
def test_predict_het_hc2bm_warns_without_cluster(favara_imbs_data):
    py_hc2 = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=2,
        predict_het=(["state_n"], [-1]),
    )

    with pytest.warns(UserWarning, match="predict_het_hc2bm has no effect"):
        py_hc2bm = did_multiplegt(
            favara_imbs_data,
            yname="Dl_vloans_b",
            idname="county",
            tname="year",
            dname="inter_bra",
            effects=2,
            predict_het=(["state_n"], [-1]),
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
