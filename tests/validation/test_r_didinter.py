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
    controls=None,
):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    normalized_str = "TRUE" if normalized else "FALSE"
    effects_equal_str = "TRUE" if effects_equal else "FALSE"
    trends_lin_str = "TRUE" if trends_lin else "FALSE"
    only_never_str = "TRUE" if only_never_switchers else "FALSE"
    same_switchers_str = "TRUE" if same_switchers else "FALSE"
    less_conservative_str = "TRUE" if less_conservative_se else "FALSE"

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
    controls = {controls_str},
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
        rtol=3e-3,
        atol=1e-4,
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
        rtol=3e-3,
        atol=1e-4,
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
