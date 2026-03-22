"""Validation tests comparing Python etwfe implementation with R etwfe package."""

import json
import re
import subprocess
import tempfile

import pytest

pytestmark = pytest.mark.slow

from tests.helpers import importorskip

pl = importorskip("polars")
np = importorskip("numpy")
importorskip("pyfixest")

from moderndid import emfx, etwfe, load_mpdta


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
            input='library(etwfe); library(did); library(jsonlite); cat("OK")',
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return result.returncode == 0 and "OK" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


R_AVAILABLE = check_r_available()


def _parse_r_gt_pairs(coef_names):
    pattern = re.compile(r"\.Dtreat:.*?::(\d+):.*?::(\d+)")
    pairs = []
    for name in coef_names:
        m = pattern.search(name)
        if m:
            pairs.append((float(m.group(1)), float(m.group(2))))
    return pairs


def r_etwfe_coefficients(controls="0", cgroup="notyet", family=None, vcov="~countyreal", ivar=True, fe=None):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    fml_rhs = controls
    yvar = "lemp"
    family_arg = ""
    if family is not None:
        family_arg = f', family = "{family}"'
        if family == "poisson":
            yvar = "emp"

    ivar_arg = ", ivar = countyreal" if ivar else ""
    fe_arg = f', fe = "{fe}"' if fe else ""

    r_script = f"""
library(etwfe)
library(did)
library(jsonlite)

data("mpdta", package = "did")
mpdta$emp <- exp(mpdta$lemp)

mod <- etwfe(
  fml = {yvar} ~ {fml_rhs},
  tvar = year,
  gvar = first.treat,
  data = mpdta,
  vcov = {vcov},
  cgroup = "{cgroup}"{family_arg}{ivar_arg}{fe_arg}
)

ct <- fixest::coeftable(mod)
treat_rows <- grepl("^\\\\.Dtreat:", rownames(ct))

all_names <- rownames(ct)
treat_names <- all_names[treat_rows]
treat_est <- unname(ct[treat_rows, "Estimate"])
treat_se <- unname(ct[treat_rows, "Std. Error"])

out <- list(
  coef_names = treat_names,
  estimates = treat_est,
  std_errors = treat_se,
  n_treat_coefs = sum(treat_rows),
  n_obs = mod$nobs
)

write_json(out, "{result_path}", auto_unbox = TRUE, digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path, timeout=120)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_emfx(
    controls="0",
    cgroup="notyet",
    agg_type="simple",
    family=None,
    post_only=True,
    window=None,
    vcov="~countyreal",
    ivar=True,
):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    fml_rhs = controls
    yvar = "lemp"
    family_arg = ""
    if family is not None:
        family_arg = f', family = "{family}"'
        if family == "poisson":
            yvar = "emp"

    post_only_str = "TRUE" if post_only else "FALSE"
    window_str = "NULL"
    if window is not None:
        window_str = f"c({window[0]}, {window[1]})"

    ivar_arg = ", ivar = countyreal" if ivar else ""

    r_script = f"""
library(etwfe)
library(did)
library(jsonlite)

data("mpdta", package = "did")
mpdta$emp <- exp(mpdta$lemp)

mod <- etwfe(
  fml = {yvar} ~ {fml_rhs},
  tvar = year,
  gvar = first.treat,
  data = mpdta,
  vcov = {vcov},
  cgroup = "{cgroup}"{family_arg}{ivar_arg}
)

window_val <- {window_str}
mfx <- emfx(mod, type = "{agg_type}", post_only = {post_only_str},
             window = window_val)

if ("{agg_type}" == "simple") {{
    out <- list(
        estimate = mfx$estimate,
        std_error = mfx$std.error,
        conf_low = mfx$conf.low,
        conf_high = mfx$conf.high,
        n_results = nrow(mfx)
    )
}} else if ("{agg_type}" == "event") {{
    out <- list(
        event = mfx$event,
        estimate = mfx$estimate,
        std_error = mfx$std.error,
        conf_low = mfx$conf.low,
        conf_high = mfx$conf.high,
        n_results = nrow(mfx)
    )
}} else if ("{agg_type}" == "group") {{
    out <- list(
        group = mfx$first.treat,
        estimate = mfx$estimate,
        std_error = mfx$std.error,
        conf_low = mfx$conf.low,
        conf_high = mfx$conf.high,
        n_results = nrow(mfx)
    )
}} else if ("{agg_type}" == "calendar") {{
    out <- list(
        calendar = mfx$year,
        estimate = mfx$estimate,
        std_error = mfx$std.error,
        conf_low = mfx$conf.low,
        conf_high = mfx$conf.high,
        n_results = nrow(mfx)
    )
}}

write_json(out, "{result_path}", auto_unbox = TRUE, digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path, timeout=180)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_baker_emfx():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        data_path = f.name

    r_script = f"""
library(etwfe)
library(jsonlite)

set.seed(1234)
baker <- expand.grid(n = 1:30, id = 1:1000)
baker <- within(baker, {{
    year <- n + 1980 - 1
    state <- 1 + (id - 1) %/% 25
    firms <- runif(id * year, 0, 5)
    grp <- 1 + (state - 1) %/% 10
    treat_date <- 1980 + grp * 6
    time_til <- year - treat_date
    treat <- time_til >= 0
    e <- rnorm(id * year, 0, 0.5^2)
    te <- rnorm(id * year, 10 - 2 * (grp - 1), 0.2^2)
    y <- firms + n + treat * te * (year - treat_date + 1) + e
}})

data_path <- "{data_path}"
write.csv(baker[, c("id", "year", "treat_date", "y")], data_path, row.names = FALSE)

mod <- etwfe(
  fml = y ~ 0,
  tvar = year,
  gvar = treat_date,
  data = baker,
  vcov = ~id
)

mfx <- emfx(mod, type = "event")

out <- list(
    event = mfx$event,
    estimate = mfx$estimate,
    std_error = mfx$std.error,
    conf_low = mfx$conf.low,
    conf_high = mfx$conf.high,
    n_results = nrow(mfx),
    data_path = data_path
)

write_json(out, "{result_path}", auto_unbox = TRUE, digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path, timeout=300)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_etwfe_xvar():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    r_script = f"""
library(etwfe)
library(did)
library(jsonlite)

data("mpdta", package = "did")

gls_fips <- c("IL" = 17, "IN" = 18, "MI" = 26, "MN" = 27,
              "NY" = 36, "OH" = 39, "PA" = 42, "WI" = 55)
mpdta$gls <- substr(mpdta$countyreal, 1, 2) %in% gls_fips

mod <- etwfe(
  fml = lemp ~ lpop,
  tvar = year,
  gvar = first.treat,
  data = mpdta,
  vcov = ~countyreal,
  xvar = gls,
  ivar = countyreal
)

mfx_simple <- emfx(mod, type = "simple", by_xvar = FALSE)
mfx_event <- emfx(mod, type = "event", by_xvar = FALSE)
mfx_het <- emfx(mod, type = "simple", by_xvar = TRUE)

ct <- fixest::coeftable(mod)
treat_rows <- grepl("^\\\\.Dtreat:", rownames(ct))
treat_names <- rownames(ct)[treat_rows]
treat_est <- unname(ct[treat_rows, "Estimate"])
treat_se <- unname(ct[treat_rows, "Std. Error"])

out <- list(
    coef_names = treat_names,
    estimates = treat_est,
    std_errors = treat_se,
    simple_estimate = mfx_simple$estimate,
    simple_se = mfx_simple$std.error,
    event_times = mfx_event$event,
    event_estimates = mfx_event$estimate,
    event_se = mfx_event$std.error,
    het_gls_values = mfx_het$gls,
    het_estimates = mfx_het$estimate,
    het_se = mfx_het$std.error,
    n_treat_coefs = sum(treat_rows)
)

write_json(out, "{result_path}", auto_unbox = TRUE, digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path, timeout=180)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_emfx_fe_variants(fe_type="feo"):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    r_script = f"""
library(etwfe)
library(did)
library(jsonlite)

data("mpdta", package = "did")

mod <- etwfe(
  fml = lemp ~ lpop,
  tvar = year,
  gvar = first.treat,
  data = mpdta,
  vcov = ~countyreal,
  fe = "{fe_type}",
  ivar = countyreal
)

mfx_simple <- emfx(mod, type = "simple")
mfx_event <- emfx(mod, type = "event")

ct <- fixest::coeftable(mod)
treat_rows <- grepl("^\\\\.Dtreat:", rownames(ct))
treat_names <- rownames(ct)[treat_rows]
treat_est <- unname(ct[treat_rows, "Estimate"])

out <- list(
    coef_names = treat_names,
    estimates = treat_est,
    simple_estimate = mfx_simple$estimate,
    simple_se = mfx_simple$std.error,
    event_times = mfx_event$event,
    event_estimates = mfx_event$estimate,
    event_se = mfx_event$std.error
)

write_json(out, "{result_path}", auto_unbox = TRUE, digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path, timeout=180)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_emfx_ivar():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    r_script = f"""
library(etwfe)
library(did)
library(jsonlite)

data("mpdta", package = "did")

mod <- etwfe(
  fml = lemp ~ lpop,
  tvar = year,
  gvar = first.treat,
  data = mpdta,
  ivar = countyreal
)

mfx_simple <- emfx(mod, type = "simple")
mfx_event <- emfx(mod, type = "event")

ct <- fixest::coeftable(mod)
treat_rows <- grepl("^\\\\.Dtreat:", rownames(ct))
treat_names <- rownames(ct)[treat_rows]
treat_est <- unname(ct[treat_rows, "Estimate"])

out <- list(
    coef_names = treat_names,
    estimates = treat_est,
    simple_estimate = mfx_simple$estimate,
    simple_se = mfx_simple$std.error,
    event_times = mfx_event$event,
    event_estimates = mfx_event$estimate,
    event_se = mfx_event$std.error
)

write_json(out, "{result_path}", auto_unbox = TRUE, digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path, timeout=180)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


@pytest.fixture(scope="module")
def mpdta_data():
    return load_mpdta()


def _match_gt_coefficients(py_gt_pairs, py_coefs, r_gt_pairs, r_coefs, rtol, atol, label):
    assert len(r_gt_pairs) > 0, f"{label}: R returned no treatment coefficients"
    assert len(py_gt_pairs) > 0, f"{label}: Python returned no treatment coefficients"
    assert len(r_gt_pairs) == len(py_gt_pairs), (
        f"{label}: (g,t) count differs: R={len(r_gt_pairs)}, Python={len(py_gt_pairs)}"
    )

    matched = 0
    for r_idx, r_gt in enumerate(r_gt_pairs):
        for py_idx, py_gt in enumerate(py_gt_pairs):
            if abs(r_gt[0] - py_gt[0]) < 1e-6 and abs(r_gt[1] - py_gt[1]) < 1e-6:
                np.testing.assert_allclose(
                    py_coefs[py_idx],
                    r_coefs[r_idx],
                    rtol=rtol,
                    atol=atol,
                    err_msg=f"{label}: mismatch at g={r_gt[0]}, t={r_gt[1]}",
                )
                matched += 1
                break

    assert matched == len(r_gt_pairs), f"{label}: only matched {matched}/{len(r_gt_pairs)} (g,t) pairs"


def _extract_r_base_treatment(r_result):
    base_names = [n for n in r_result["coef_names"] if "_dm" not in n and ".Dtreat:" in n]
    base_gt = _parse_r_gt_pairs(base_names)
    base_est = np.array(
        [r_result["estimates"][i] for i, n in enumerate(r_result["coef_names"]) if "_dm" not in n and ".Dtreat:" in n]
    )
    return base_gt, base_est


def _assert_r_values_valid(r_values, label):
    arr = np.array(r_values)
    assert len(arr) > 0, f"{label}: R returned empty array"
    assert not np.all(np.isnan(arr)), f"{label}: R returned all NaN"
    assert not np.all(arr == 0), f"{label}: R returned all zeros"


def _match_agg_by_key(py_times, py_att, r_keys, r_est, rtol, atol, label, min_matches=2):
    assert len(r_keys) >= min_matches, f"{label}: R returned too few results ({len(r_keys)})"
    assert py_times is not None, f"{label}: Python returned None for aggregation keys"
    assert len(py_times) >= min_matches, f"{label}: Python returned too few results ({len(py_times)})"

    matched = 0
    for r_idx, r_k in enumerate(r_keys):
        py_mask = np.abs(py_times - r_k) < 0.5
        if np.any(py_mask):
            py_idx = np.where(py_mask)[0][0]
            np.testing.assert_allclose(
                py_att[py_idx],
                r_est[r_idx],
                rtol=rtol,
                atol=atol,
                err_msg=f"{label}: mismatch at key={r_k}",
            )
            matched += 1

    assert matched == len(r_keys), f"{label}: only matched {matched}/{len(r_keys)} keys"


@pytest.mark.skipif(not R_AVAILABLE, reason="R etwfe package not available")
class TestEtwfeCoefficients:
    def test_no_controls_notyet(self, mpdta_data):
        r_result = r_etwfe_coefficients(controls="0", cgroup="notyet")
        if r_result is None:
            pytest.fail("R etwfe estimation failed")

        _assert_r_values_valid(r_result["estimates"], "R notyet coefficients")

        py_result = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            vcov={"CRV1": "countyreal"},
        )

        r_gt = _parse_r_gt_pairs(r_result["coef_names"])
        r_est = np.array(r_result["estimates"])
        _match_gt_coefficients(
            py_result.gt_pairs, py_result.coefficients, r_gt, r_est, rtol=1e-7, atol=1e-8, label="notyet no controls"
        )

    def test_no_controls_never(self, mpdta_data):
        r_result = r_etwfe_coefficients(controls="0", cgroup="never")
        if r_result is None:
            pytest.fail("R etwfe estimation failed")

        py_result = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            vcov={"CRV1": "countyreal"},
            cgroup="never",
        )

        r_gt = _parse_r_gt_pairs(r_result["coef_names"])
        r_est = np.array(r_result["estimates"])

        assert len(r_gt) >= 10, f"R returned too few never-treated coefficients ({len(r_gt)})"
        _match_gt_coefficients(
            py_result.gt_pairs, py_result.coefficients, r_gt, r_est, rtol=1e-7, atol=1e-8, label="never no controls"
        )

    def test_with_controls(self, mpdta_data):
        r_result = r_etwfe_coefficients(controls="lpop", cgroup="notyet", fe="feo")
        if r_result is None:
            pytest.fail("R etwfe estimation failed")

        py_result = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            xformla="~ lpop",
            vcov={"CRV1": "countyreal"},
            fe="feo",
        )

        r_base_gt, r_base_est = _extract_r_base_treatment(r_result)
        _match_gt_coefficients(
            py_result.gt_pairs,
            py_result.coefficients,
            r_base_gt,
            r_base_est,
            rtol=1e-7,
            atol=1e-8,
            label="notyet with controls",
        )

    def test_with_controls_never(self, mpdta_data):
        r_result = r_etwfe_coefficients(controls="lpop", cgroup="never", fe="feo")
        if r_result is None:
            pytest.fail("R etwfe estimation failed")

        py_result = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            xformla="~ lpop",
            vcov={"CRV1": "countyreal"},
            cgroup="never",
            fe="feo",
        )

        r_base_gt, r_base_est = _extract_r_base_treatment(r_result)
        assert len(r_base_gt) >= 10
        _match_gt_coefficients(
            py_result.gt_pairs,
            py_result.coefficients,
            r_base_gt,
            r_base_est,
            rtol=1e-7,
            atol=1e-8,
            label="never with controls",
        )

    def test_poisson_no_controls(self, mpdta_data):
        r_result = r_etwfe_coefficients(controls="0", family="poisson")
        if r_result is None:
            pytest.fail("R etwfe Poisson estimation failed")

        mpdta_pois = mpdta_data.with_columns(pl.col("lemp").exp().alias("emp"))

        py_result = etwfe(
            data=mpdta_pois,
            yname="emp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            vcov={"CRV1": "countyreal"},
            family="poisson",
        )

        r_base_gt, r_base_est = _extract_r_base_treatment(r_result)
        _match_gt_coefficients(
            py_result.gt_pairs,
            py_result.coefficients,
            r_base_gt,
            r_base_est,
            rtol=1e-6,
            atol=1e-7,
            label="poisson no controls",
        )

    def test_poisson_with_controls(self, mpdta_data):
        r_result = r_etwfe_coefficients(controls="lpop", family="poisson")
        if r_result is None:
            pytest.fail("R etwfe Poisson with controls failed")

        mpdta_pois = mpdta_data.with_columns(pl.col("lemp").exp().alias("emp"))

        py_result = etwfe(
            data=mpdta_pois,
            yname="emp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            xformla="~ lpop",
            vcov={"CRV1": "countyreal"},
            family="poisson",
        )

        r_base_gt, r_base_est = _extract_r_base_treatment(r_result)
        _match_gt_coefficients(
            py_result.gt_pairs,
            py_result.coefficients,
            r_base_gt,
            r_base_est,
            rtol=1e-6,
            atol=1e-7,
            label="poisson with controls",
        )

    def test_standard_errors_notyet(self, mpdta_data):
        r_result = r_etwfe_coefficients(controls="0", cgroup="notyet")
        if r_result is None:
            pytest.fail("R etwfe estimation failed")

        py_result = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            vcov={"CRV1": "countyreal"},
        )

        r_gt = _parse_r_gt_pairs(r_result["coef_names"])
        r_se = np.array(r_result["std_errors"])
        _match_gt_coefficients(
            py_result.gt_pairs, py_result.std_errors, r_gt, r_se, rtol=2e-3, atol=1e-4, label="SE notyet no controls"
        )

    def test_n_obs_matches(self, mpdta_data):
        r_result = r_etwfe_coefficients(controls="0", cgroup="notyet")
        if r_result is None:
            pytest.fail("R etwfe estimation failed")

        py_result = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            vcov={"CRV1": "countyreal"},
        )

        assert py_result.n_obs == r_result["n_obs"], f"N obs mismatch: Python={py_result.n_obs}, R={r_result['n_obs']}"


@pytest.mark.skipif(not R_AVAILABLE, reason="R etwfe package not available")
class TestEmfxSimple:
    def test_simple_no_controls(self, mpdta_data):
        r_result = r_emfx(controls="0", agg_type="simple")
        if r_result is None:
            pytest.fail("R emfx estimation failed")

        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            vcov={"CRV1": "countyreal"},
        )
        py_result = emfx(py_mod, type="simple")

        r_est = r_result["estimate"] if isinstance(r_result["estimate"], float) else r_result["estimate"][0]
        r_se = r_result["std_error"] if isinstance(r_result["std_error"], float) else r_result["std_error"][0]

        np.testing.assert_allclose(py_result.overall_att, r_est, rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(py_result.overall_se, r_se, rtol=5e-3, atol=1e-4)

    def test_simple_with_controls(self, mpdta_data):
        r_result = r_emfx(controls="lpop", agg_type="simple")
        if r_result is None:
            pytest.fail("R emfx estimation failed")

        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            xformla="~ lpop",
            vcov={"CRV1": "countyreal"},
        )
        py_result = emfx(py_mod, type="simple")

        r_est = r_result["estimate"] if isinstance(r_result["estimate"], float) else r_result["estimate"][0]
        r_se = r_result["std_error"] if isinstance(r_result["std_error"], float) else r_result["std_error"][0]

        np.testing.assert_allclose(py_result.overall_att, r_est, rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(py_result.overall_se, r_se, rtol=5e-3, atol=1e-4)

    def test_simple_never(self, mpdta_data):
        r_result = r_emfx(controls="lpop", cgroup="never", agg_type="simple")
        if r_result is None:
            pytest.fail("R emfx estimation failed")

        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            xformla="~ lpop",
            vcov={"CRV1": "countyreal"},
            cgroup="never",
        )
        py_result = emfx(py_mod, type="simple")

        r_est = r_result["estimate"] if isinstance(r_result["estimate"], float) else r_result["estimate"][0]
        r_se = r_result["std_error"] if isinstance(r_result["std_error"], float) else r_result["std_error"][0]
        np.testing.assert_allclose(py_result.overall_att, r_est, rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(py_result.overall_se, r_se, rtol=5e-3, atol=1e-4)


@pytest.mark.skipif(not R_AVAILABLE, reason="R etwfe package not available")
class TestEmfxEvent:
    def test_event_with_controls(self, mpdta_data):
        r_result = r_emfx(controls="lpop", agg_type="event")
        if r_result is None:
            pytest.fail("R emfx estimation failed")

        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            xformla="~ lpop",
            vcov={"CRV1": "countyreal"},
        )
        py_result = emfx(py_mod, type="event")

        _match_agg_by_key(
            py_result.event_times,
            py_result.att_by_event,
            np.array(r_result["event"]),
            np.array(r_result["estimate"]),
            rtol=1e-7,
            atol=1e-8,
            label="event with controls",
        )

    def test_event_no_controls(self, mpdta_data):
        r_result = r_emfx(controls="0", agg_type="event")
        if r_result is None:
            pytest.fail("R emfx estimation failed")

        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            vcov={"CRV1": "countyreal"},
        )
        py_result = emfx(py_mod, type="event")

        _match_agg_by_key(
            py_result.event_times,
            py_result.att_by_event,
            np.array(r_result["event"]),
            np.array(r_result["estimate"]),
            rtol=1e-7,
            atol=1e-8,
            label="event no controls",
        )

    def test_event_se(self, mpdta_data):
        r_result = r_emfx(controls="lpop", agg_type="event")
        if r_result is None:
            pytest.fail("R emfx estimation failed")

        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            xformla="~ lpop",
            vcov={"CRV1": "countyreal"},
        )
        py_result = emfx(py_mod, type="event")

        r_events = np.array(r_result["event"])
        r_se = np.array(r_result["std_error"])

        matched = 0
        for r_idx, r_e in enumerate(r_events):
            py_mask = np.abs(py_result.event_times - r_e) < 0.5
            if np.any(py_mask):
                py_idx = np.where(py_mask)[0][0]
                if not np.isnan(r_se[r_idx]):
                    np.testing.assert_allclose(
                        py_result.se_by_event[py_idx],
                        r_se[r_idx],
                        rtol=5e-3,
                        atol=1e-4,
                        err_msg=f"Event SE mismatch at e={r_e}",
                    )
                matched += 1

        assert matched == len(r_events)

    def test_event_never(self, mpdta_data):
        r_result = r_emfx(controls="lpop", cgroup="never", agg_type="event")
        if r_result is None:
            pytest.fail("R emfx estimation failed")

        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            xformla="~ lpop",
            vcov={"CRV1": "countyreal"},
            cgroup="never",
        )
        py_result = emfx(py_mod, type="event")

        r_events = np.array(r_result["event"])
        r_est = np.array(r_result["estimate"])

        assert len(r_events) >= 4, f"R never-treated returned too few events ({len(r_events)})"
        assert py_result.event_times is not None

        r_post_mask = r_events >= 0
        r_post_events = r_events[r_post_mask]
        r_post_est = r_est[r_post_mask]

        matched = 0
        for r_idx, r_e in enumerate(r_post_events):
            py_mask = np.abs(py_result.event_times - r_e) < 0.5
            if np.any(py_mask):
                py_idx = np.where(py_mask)[0][0]
                np.testing.assert_allclose(
                    py_result.att_by_event[py_idx],
                    r_post_est[r_idx],
                    rtol=1e-7,
                    atol=1e-8,
                    err_msg=f"Never-treated event ATT mismatch at e={r_e}",
                )
                matched += 1

        assert matched >= 1

    def test_event_never_window(self, mpdta_data):
        r_result = r_emfx(controls="lpop", cgroup="never", agg_type="event", window=(-2, 3))
        if r_result is None:
            pytest.fail("R emfx estimation failed")

        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            xformla="~ lpop",
            vcov={"CRV1": "countyreal"},
            cgroup="never",
        )
        py_result = emfx(py_mod, type="event", window=(-2, 3))

        r_events = np.array(r_result["event"])
        r_est = np.array(r_result["estimate"])

        assert len(r_events) >= 2
        assert py_result.event_times is not None
        assert np.all(r_events >= -2) and np.all(r_events <= 3)
        assert np.all(py_result.event_times >= -2) and np.all(py_result.event_times <= 3)

        r_post_mask = r_events >= 0
        matched = 0
        for r_idx, r_e in enumerate(r_events[r_post_mask]):
            py_mask = np.abs(py_result.event_times - r_e) < 0.5
            if np.any(py_mask):
                py_idx = np.where(py_mask)[0][0]
                np.testing.assert_allclose(
                    py_result.att_by_event[py_idx],
                    r_est[r_post_mask][r_idx],
                    rtol=1e-7,
                    atol=1e-8,
                    err_msg=f"Windowed event ATT mismatch at e={r_e}",
                )
                matched += 1

        assert matched >= 1


@pytest.mark.skipif(not R_AVAILABLE, reason="R etwfe package not available")
class TestEmfxGroup:
    def test_group_with_controls(self, mpdta_data):
        r_result = r_emfx(controls="lpop", agg_type="group")
        if r_result is None:
            pytest.fail("R emfx estimation failed")

        _assert_r_values_valid(r_result["estimate"], "R group estimates")

        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            xformla="~ lpop",
            vcov={"CRV1": "countyreal"},
        )
        py_result = emfx(py_mod, type="group")

        _match_agg_by_key(
            py_result.event_times,
            py_result.att_by_event,
            np.array(r_result["group"]),
            np.array(r_result["estimate"]),
            rtol=1e-7,
            atol=1e-8,
            label="group with controls",
        )
        _match_agg_by_key(
            py_result.event_times,
            py_result.se_by_event,
            np.array(r_result["group"]),
            np.array(r_result["std_error"]),
            rtol=5e-3,
            atol=1e-4,
            label="group SE with controls",
        )

    def test_group_no_controls(self, mpdta_data):
        r_result = r_emfx(controls="0", agg_type="group")
        if r_result is None:
            pytest.fail("R emfx estimation failed")

        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            vcov={"CRV1": "countyreal"},
        )
        py_result = emfx(py_mod, type="group")

        _match_agg_by_key(
            py_result.event_times,
            py_result.att_by_event,
            np.array(r_result["group"]),
            np.array(r_result["estimate"]),
            rtol=1e-7,
            atol=1e-8,
            label="group no controls",
        )


@pytest.mark.skipif(not R_AVAILABLE, reason="R etwfe package not available")
class TestEmfxCalendar:
    def test_calendar_with_controls(self, mpdta_data):
        r_result = r_emfx(controls="lpop", agg_type="calendar")
        if r_result is None:
            pytest.fail("R emfx estimation failed")

        _assert_r_values_valid(r_result["estimate"], "R calendar estimates")

        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            xformla="~ lpop",
            vcov={"CRV1": "countyreal"},
        )
        py_result = emfx(py_mod, type="calendar")

        _match_agg_by_key(
            py_result.event_times,
            py_result.att_by_event,
            np.array(r_result["calendar"]),
            np.array(r_result["estimate"]),
            rtol=1e-7,
            atol=1e-8,
            label="calendar with controls",
        )
        _match_agg_by_key(
            py_result.event_times,
            py_result.se_by_event,
            np.array(r_result["calendar"]),
            np.array(r_result["std_error"]),
            rtol=5e-3,
            atol=1e-4,
            label="calendar SE with controls",
        )

    def test_calendar_no_controls(self, mpdta_data):
        r_result = r_emfx(controls="0", agg_type="calendar")
        if r_result is None:
            pytest.fail("R emfx estimation failed")

        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            vcov={"CRV1": "countyreal"},
        )
        py_result = emfx(py_mod, type="calendar")

        _match_agg_by_key(
            py_result.event_times,
            py_result.att_by_event,
            np.array(r_result["calendar"]),
            np.array(r_result["estimate"]),
            rtol=1e-7,
            atol=1e-8,
            label="calendar no controls",
        )


@pytest.mark.skipif(not R_AVAILABLE, reason="R etwfe package not available")
class TestEmfxPoisson:
    def test_poisson_event_response_scale(self, mpdta_data):
        r_result = r_emfx(controls="lpop", agg_type="event", family="poisson")
        if r_result is None:
            pytest.fail("R Poisson emfx estimation failed")

        mpdta_pois = mpdta_data.with_columns(pl.col("lemp").exp().alias("emp"))

        py_mod = etwfe(
            data=mpdta_pois,
            yname="emp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            xformla="~ lpop",
            vcov={"CRV1": "countyreal"},
            family="poisson",
        )
        py_result = emfx(py_mod, type="event")

        r_events = np.array(r_result["event"])
        r_est = np.array(r_result["estimate"])

        assert len(r_events) >= 2
        assert py_result.event_times is not None
        assert len(py_result.event_times) >= 2

        matched = 0
        for r_idx, r_e in enumerate(r_events):
            py_mask = np.abs(py_result.event_times - r_e) < 0.5
            if np.any(py_mask):
                py_idx = np.where(py_mask)[0][0]
                np.testing.assert_allclose(
                    py_result.att_by_event[py_idx],
                    r_est[r_idx],
                    rtol=1e-5,
                    atol=1e-4,
                    err_msg=f"Poisson event ATT mismatch at e={r_e}",
                )
                matched += 1

        assert matched >= 2

    def test_poisson_simple(self, mpdta_data):
        r_result = r_emfx(controls="lpop", agg_type="simple", family="poisson")
        if r_result is None:
            pytest.fail("R Poisson simple emfx failed")

        mpdta_pois = mpdta_data.with_columns(pl.col("lemp").exp().alias("emp"))

        py_mod = etwfe(
            data=mpdta_pois,
            yname="emp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            xformla="~ lpop",
            vcov={"CRV1": "countyreal"},
            family="poisson",
        )
        py_result = emfx(py_mod, type="simple")

        r_est = r_result["estimate"] if isinstance(r_result["estimate"], float) else r_result["estimate"][0]
        np.testing.assert_allclose(py_result.overall_att, r_est, rtol=1e-5, atol=1e-4)

    def test_poisson_no_controls(self, mpdta_data):
        r_result = r_emfx(controls="0", agg_type="event", family="poisson")
        if r_result is None:
            pytest.fail("R Poisson emfx failed")

        mpdta_pois = mpdta_data.with_columns(pl.col("lemp").exp().alias("emp"))

        py_mod = etwfe(
            data=mpdta_pois,
            yname="emp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            vcov={"CRV1": "countyreal"},
            family="poisson",
        )
        py_result = emfx(py_mod, type="event")

        r_events = np.array(r_result["event"])
        r_est = np.array(r_result["estimate"])

        assert len(r_events) >= 2
        assert py_result.event_times is not None
        assert len(py_result.event_times) >= 2

        matched = 0
        for r_idx, r_e in enumerate(r_events):
            py_mask = np.abs(py_result.event_times - r_e) < 0.5
            if np.any(py_mask):
                py_idx = np.where(py_mask)[0][0]
                np.testing.assert_allclose(
                    py_result.att_by_event[py_idx],
                    r_est[r_idx],
                    rtol=1e-5,
                    atol=1e-4,
                    err_msg=f"Poisson event ATT mismatch at e={r_e}",
                )
                matched += 1

        assert matched >= 2


@pytest.mark.skipif(not R_AVAILABLE, reason="R etwfe package not available")
class TestFESpecifications:
    @pytest.mark.parametrize("fe_type", ["feo", "none"])
    def test_fe_variants_coefficients(self, mpdta_data, fe_type):
        r_result = r_emfx_fe_variants(fe_type=fe_type)
        if r_result is None:
            pytest.fail(f"R etwfe with fe={fe_type} failed")

        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            xformla="~ lpop",
            vcov={"CRV1": "countyreal"},
            fe=fe_type,
        )

        r_base_names = [n for n in r_result["coef_names"] if "_dm" not in n]
        r_base_gt = _parse_r_gt_pairs(r_base_names)
        r_base_est = np.array(
            [r_result["estimates"][i] for i, n in enumerate(r_result["coef_names"]) if "_dm" not in n]
        )

        _match_gt_coefficients(
            py_mod.gt_pairs,
            py_mod.coefficients,
            r_base_gt,
            r_base_est,
            rtol=1e-7,
            atol=1e-8,
            label=f"fe={fe_type} coefficients",
        )

    @pytest.mark.parametrize("fe_type", ["feo", "none"])
    def test_fe_variants_emfx_simple(self, mpdta_data, fe_type):
        r_result = r_emfx_fe_variants(fe_type=fe_type)
        if r_result is None:
            pytest.fail(f"R etwfe with fe={fe_type} failed")

        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            xformla="~ lpop",
            vcov={"CRV1": "countyreal"},
            fe=fe_type,
        )
        py_result = emfx(py_mod, type="simple")

        r_est = r_result["simple_estimate"]
        if isinstance(r_est, list):
            r_est = r_est[0]

        np.testing.assert_allclose(py_result.overall_att, r_est, rtol=1e-7, atol=1e-8)

    @pytest.mark.parametrize("fe_type", ["feo", "none"])
    def test_fe_variants_emfx_event(self, mpdta_data, fe_type):
        r_result = r_emfx_fe_variants(fe_type=fe_type)
        if r_result is None:
            pytest.fail(f"R etwfe with fe={fe_type} failed")

        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            xformla="~ lpop",
            vcov={"CRV1": "countyreal"},
            fe=fe_type,
        )
        py_result = emfx(py_mod, type="event")

        _match_agg_by_key(
            py_result.event_times,
            py_result.att_by_event,
            np.array(r_result["event_times"]),
            np.array(r_result["event_estimates"]),
            rtol=1e-7,
            atol=1e-8,
            label=f"fe={fe_type} event",
        )


@pytest.mark.skipif(not R_AVAILABLE, reason="R etwfe package not available")
class TestIvar:
    def test_ivar_coefficients(self, mpdta_data):
        r_result = r_emfx_ivar()
        if r_result is None:
            pytest.fail("R etwfe with ivar failed")

        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            xformla="~ lpop",
        )

        r_base_names = [n for n in r_result["coef_names"] if "_dm" not in n]
        r_base_gt = _parse_r_gt_pairs(r_base_names)
        r_base_est = np.array(
            [r_result["estimates"][i] for i, n in enumerate(r_result["coef_names"]) if "_dm" not in n]
        )

        _match_gt_coefficients(
            py_mod.gt_pairs, py_mod.coefficients, r_base_gt, r_base_est, rtol=1e-7, atol=1e-8, label="ivar coefficients"
        )

    def test_ivar_emfx_event(self, mpdta_data):
        r_result = r_emfx_ivar()
        if r_result is None:
            pytest.fail("R etwfe with ivar failed")

        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            xformla="~ lpop",
        )
        py_result = emfx(py_mod, type="event")

        _match_agg_by_key(
            py_result.event_times,
            py_result.att_by_event,
            np.array(r_result["event_times"]),
            np.array(r_result["event_estimates"]),
            rtol=1e-7,
            atol=1e-8,
            label="ivar event",
        )


@pytest.mark.skipif(not R_AVAILABLE, reason="R etwfe package not available")
class TestXvar:
    def test_xvar_coefficients(self, mpdta_data):
        r_result = r_etwfe_xvar()
        if r_result is None:
            pytest.fail("R etwfe with xvar failed")

        _assert_r_values_valid(r_result["estimates"], "R xvar coefficients")

        gls_fips = [17, 18, 26, 27, 36, 39, 42, 55]
        mpdta_xvar = mpdta_data.with_columns(
            pl.col("countyreal").cast(pl.Utf8).str.slice(0, 2).cast(pl.Int64).is_in(gls_fips).alias("gls")
        )

        py_mod = etwfe(
            data=mpdta_xvar,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            xformla="~ lpop",
            xvar="gls",
            vcov={"CRV1": "countyreal"},
        )

        assert len(py_mod.gt_pairs) > 0
        assert r_result["n_treat_coefs"] > 0
        assert len(py_mod.gt_pairs) == 7

        r_base_gt, r_base_est = _extract_r_base_treatment(r_result)
        _match_gt_coefficients(
            py_mod.gt_pairs, py_mod.coefficients, r_base_gt, r_base_est, rtol=0.25, atol=0.02, label="xvar coefficients"
        )

    def test_xvar_event_study(self, mpdta_data):
        r_result = r_etwfe_xvar()
        if r_result is None:
            pytest.fail("R etwfe with xvar failed")

        gls_fips = [17, 18, 26, 27, 36, 39, 42, 55]
        mpdta_xvar = mpdta_data.with_columns(
            pl.col("countyreal").cast(pl.Utf8).str.slice(0, 2).cast(pl.Int64).is_in(gls_fips).alias("gls")
        )

        py_mod = etwfe(
            data=mpdta_xvar,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            xformla="~ lpop",
            xvar="gls",
            vcov={"CRV1": "countyreal"},
        )
        py_result = emfx(py_mod, type="event")

        r_events = np.array(r_result["event_times"])
        r_est = np.array(r_result["event_estimates"])

        assert py_result.event_times is not None
        assert len(r_events) >= 2
        assert len(py_result.event_times) >= 2

        matched = 0
        for r_idx, r_e in enumerate(r_events):
            py_mask = np.abs(py_result.event_times - r_e) < 0.5
            if np.any(py_mask):
                py_idx = np.where(py_mask)[0][0]
                np.testing.assert_allclose(
                    py_result.att_by_event[py_idx],
                    r_est[r_idx],
                    rtol=0.05,
                    atol=1e-2,
                    err_msg=f"xvar event ATT mismatch at e={r_e}",
                )
                matched += 1

        assert matched >= 2


@pytest.mark.skipif(not R_AVAILABLE, reason="R etwfe package not available")
class TestBaker:
    def test_baker_event_study(self):
        r_result = r_baker_emfx()
        if r_result is None:
            pytest.fail("R Baker event study failed")

        r_events = np.array(r_result["event"])
        r_est = np.array(r_result["estimate"])

        assert len(r_events) >= 15
        assert r_result["n_results"] >= 15

        data_path = r_result.get("data_path")
        if data_path is None:
            pytest.fail("R did not return baker data path")

        baker_data = pl.read_csv(data_path)

        py_mod = etwfe(
            data=baker_data,
            yname="y",
            tname="year",
            gname="treat_date",
            vcov={"CRV1": "id"},
        )
        py_result = emfx(py_mod, type="event")

        assert py_result.event_times is not None
        assert len(py_result.event_times) >= 15

        matched = 0
        for r_idx, r_e in enumerate(r_events):
            py_mask = np.abs(py_result.event_times - r_e) < 0.5
            if np.any(py_mask):
                py_idx = np.where(py_mask)[0][0]
                np.testing.assert_allclose(
                    py_result.att_by_event[py_idx],
                    r_est[r_idx],
                    rtol=1e-5,
                    atol=1e-4,
                    err_msg=f"Baker event ATT mismatch at e={r_e}",
                )
                matched += 1

        assert matched >= 15

    def test_baker_event_study_se(self):
        r_result = r_baker_emfx()
        if r_result is None:
            pytest.fail("R Baker event study failed")

        r_events = np.array(r_result["event"])
        r_se = np.array(r_result["std_error"])

        data_path = r_result.get("data_path")
        if data_path is None:
            pytest.fail("R did not return baker data path")

        baker_data = pl.read_csv(data_path)

        py_mod = etwfe(
            data=baker_data,
            yname="y",
            tname="year",
            gname="treat_date",
            vcov={"CRV1": "id"},
        )
        py_result = emfx(py_mod, type="event")

        assert py_result.event_times is not None
        matched = 0
        for r_idx, r_e in enumerate(r_events):
            py_mask = np.abs(py_result.event_times - r_e) < 0.5
            if np.any(py_mask):
                py_idx = np.where(py_mask)[0][0]
                if not np.isnan(r_se[r_idx]) and not np.isnan(py_result.se_by_event[py_idx]):
                    np.testing.assert_allclose(
                        py_result.se_by_event[py_idx],
                        r_se[r_idx],
                        rtol=5e-3,
                        atol=1e-4,
                        err_msg=f"Baker event SE mismatch at e={r_e}",
                    )
                matched += 1

        assert matched >= 15


@pytest.mark.skipif(not R_AVAILABLE, reason="R etwfe package not available")
class TestConfidenceIntervals:
    def test_event_ci_ordering(self, mpdta_data):
        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            vcov={"CRV1": "countyreal"},
        )
        py_result = emfx(py_mod, type="event")

        assert py_result.ci_lower is not None
        assert py_result.ci_upper is not None
        assert py_result.att_by_event is not None

        assert np.all(py_result.ci_lower <= py_result.att_by_event + 1e-10)
        assert np.all(py_result.ci_upper >= py_result.att_by_event - 1e-10)

    def test_event_ci_vs_r(self, mpdta_data):
        r_result = r_emfx(controls="lpop", agg_type="event")
        if r_result is None:
            pytest.fail("R emfx estimation failed")

        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            xformla="~ lpop",
            vcov={"CRV1": "countyreal"},
        )
        py_result = emfx(py_mod, type="event")

        r_events = np.array(r_result["event"])
        r_cl = np.array(r_result["conf_low"])
        r_ch = np.array(r_result["conf_high"])

        assert py_result.ci_lower is not None
        assert py_result.ci_upper is not None

        matched = 0
        for r_idx, r_e in enumerate(r_events):
            py_mask = np.abs(py_result.event_times - r_e) < 0.5
            if np.any(py_mask):
                py_idx = np.where(py_mask)[0][0]
                np.testing.assert_allclose(
                    py_result.ci_lower[py_idx],
                    r_cl[r_idx],
                    rtol=1e-4,
                    atol=1e-5,
                    err_msg=f"CI lower mismatch at e={r_e}",
                )
                np.testing.assert_allclose(
                    py_result.ci_upper[py_idx],
                    r_ch[r_idx],
                    rtol=1e-4,
                    atol=1e-5,
                    err_msg=f"CI upper mismatch at e={r_e}",
                )
                matched += 1

        assert matched >= 2


@pytest.mark.skipif(not R_AVAILABLE, reason="R etwfe package not available")
class TestNonEmptyGuards:
    def test_etwfe_has_coefficients(self, mpdta_data):
        py_result = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
        )
        assert len(py_result.coefficients) >= 7
        assert len(py_result.gt_pairs) >= 7
        assert py_result.n_obs > 0
        assert not np.all(np.isnan(py_result.coefficients))
        assert not np.all(py_result.coefficients == 0)

    def test_etwfe_never_has_more_coefficients(self, mpdta_data):
        py_notyet = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
        )
        py_never = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            cgroup="never",
        )
        assert len(py_never.gt_pairs) > len(py_notyet.gt_pairs)

    def test_emfx_event_has_multiple_periods(self, mpdta_data):
        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
        )
        py_result = emfx(py_mod, type="event")
        assert py_result.event_times is not None
        assert len(py_result.event_times) >= 2
        assert py_result.att_by_event is not None
        assert py_result.se_by_event is not None
        assert not np.all(np.isnan(py_result.att_by_event))
        assert not np.all(py_result.att_by_event == 0)

    def test_emfx_group_has_treated_cohorts(self, mpdta_data):
        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
        )
        py_result = emfx(py_mod, type="group")
        assert py_result.event_times is not None
        assert len(py_result.event_times) >= 2

    def test_emfx_calendar_has_treatment_periods(self, mpdta_data):
        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
        )
        py_result = emfx(py_mod, type="calendar")
        assert py_result.event_times is not None
        assert len(py_result.event_times) >= 2

    def test_r_and_python_produce_nonzero_att(self, mpdta_data):
        r_result = r_emfx(controls="0", agg_type="simple")
        if r_result is None:
            pytest.fail("R emfx estimation failed")

        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
        )
        py_result = emfx(py_mod, type="simple")

        r_est = r_result["estimate"] if isinstance(r_result["estimate"], float) else r_result["estimate"][0]
        assert abs(r_est) > 1e-6, f"R ATT is essentially zero ({r_est})"
        assert abs(py_result.overall_att) > 1e-6, f"Python ATT is essentially zero ({py_result.overall_att})"

    def test_r_and_python_event_same_length(self, mpdta_data):
        r_result = r_emfx(controls="0", agg_type="event")
        if r_result is None:
            pytest.fail("R emfx estimation failed")

        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
        )
        py_result = emfx(py_mod, type="event")

        r_n = len(r_result["event"]) if isinstance(r_result["event"], list) else 1

        assert py_result.event_times is not None
        assert len(py_result.event_times) == r_n

    def test_r_and_python_group_same_length(self, mpdta_data):
        r_result = r_emfx(controls="0", agg_type="group")
        if r_result is None:
            pytest.fail("R emfx estimation failed")

        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
        )
        py_result = emfx(py_mod, type="group")

        r_n = len(r_result["group"]) if isinstance(r_result["group"], list) else 1

        assert py_result.event_times is not None
        assert len(py_result.event_times) == r_n


@pytest.mark.skipif(not R_AVAILABLE, reason="R etwfe package not available")
class TestCrossConsistency:
    def test_simple_vs_event_overall(self, mpdta_data):
        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            vcov={"CRV1": "countyreal"},
        )
        simple = emfx(py_mod, type="simple")
        event = emfx(py_mod, type="event")

        np.testing.assert_allclose(simple.overall_att, event.overall_att, rtol=1e-6)

    def test_simple_vs_group_overall(self, mpdta_data):
        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            vcov={"CRV1": "countyreal"},
        )
        simple = emfx(py_mod, type="simple")
        group = emfx(py_mod, type="group")

        np.testing.assert_allclose(simple.overall_att, group.overall_att, rtol=1e-6)

    def test_simple_vs_calendar_overall(self, mpdta_data):
        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            vcov={"CRV1": "countyreal"},
        )
        simple = emfx(py_mod, type="simple")
        calendar = emfx(py_mod, type="calendar")

        np.testing.assert_allclose(simple.overall_att, calendar.overall_att, rtol=1e-6)

    @pytest.mark.parametrize("cgroup", ["notyet", "never"])
    def test_all_aggregation_types_run(self, mpdta_data, cgroup):
        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            cgroup=cgroup,
        )
        for agg_type in ["simple", "event", "group", "calendar"]:
            result = emfx(py_mod, type=agg_type)
            assert result.overall_att is not None
            assert not np.isnan(result.overall_att)


@pytest.mark.skipif(not R_AVAILABLE, reason="R etwfe package not available")
class TestKnownValues:
    def test_notyet_no_controls_known_coefficients(self, mpdta_data):
        known_coefs = {
            (2004, 2004): -0.0193723636759116,
            (2004, 2005): -0.0783190990620529,
            (2004, 2006): -0.136078114440309,
            (2004, 2007): -0.104707471576594,
            (2006, 2006): 0.00251386194191313,
            (2006, 2007): -0.0391927355917248,
            (2007, 2007): -0.0431060328087001,
        }

        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            vcov={"CRV1": "countyreal"},
        )

        assert len(py_mod.gt_pairs) == len(known_coefs)

        for i, (g, t) in enumerate(py_mod.gt_pairs):
            key = (int(g), int(t))
            assert key in known_coefs, f"Unexpected (g,t)={key}"
            np.testing.assert_allclose(
                py_mod.coefficients[i],
                known_coefs[key],
                rtol=1e-7,
                atol=1e-8,
                err_msg=f"Known value mismatch at g={g}, t={t}",
            )

    def test_emfx_simple_known_att(self, mpdta_data):
        known_att = -0.0506270331228907

        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            xformla="~ lpop",
            vcov={"CRV1": "countyreal"},
        )
        py_result = emfx(py_mod, type="simple")

        np.testing.assert_allclose(py_result.overall_att, known_att, rtol=1e-7, atol=1e-8)

    def test_emfx_event_known_atts(self, mpdta_data):
        known_events = {
            0: -0.0332122037837542,
            1: -0.0573456479257483,
            2: -0.137870386660852,
            3: -0.10953945536511,
        }

        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            xformla="~ lpop",
            vcov={"CRV1": "countyreal"},
        )
        py_result = emfx(py_mod, type="event")

        assert py_result.event_times is not None
        assert len(py_result.event_times) == len(known_events)

        for i, e in enumerate(py_result.event_times):
            key = int(e)
            assert key in known_events, f"Unexpected event time={key}"
            np.testing.assert_allclose(
                py_result.att_by_event[i],
                known_events[key],
                rtol=1e-7,
                atol=1e-8,
                err_msg=f"Known event ATT mismatch at e={key}",
            )

    def test_emfx_group_known_atts(self, mpdta_data):
        known_groups = {
            2004: -0.0876269608795944,
            2006: -0.0212783329358987,
            2007: -0.0459545277368072,
        }

        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            xformla="~ lpop",
            vcov={"CRV1": "countyreal"},
        )
        py_result = emfx(py_mod, type="group")

        assert py_result.event_times is not None
        assert len(py_result.event_times) == len(known_groups)

        for i, g in enumerate(py_result.event_times):
            key = int(g)
            assert key in known_groups, f"Unexpected group={key}"
            np.testing.assert_allclose(
                py_result.att_by_event[i],
                known_groups[key],
                rtol=1e-7,
                atol=1e-8,
                err_msg=f"Known group ATT mismatch at g={key}",
            )

    def test_emfx_calendar_known_atts(self, mpdta_data):
        known_calendar = {
            2004: -0.0212480022225509,
            2005: -0.0818499992698648,
            2006: -0.0442655912990223,
            2007: -0.0524323095862384,
        }

        py_mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            xformla="~ lpop",
            vcov={"CRV1": "countyreal"},
        )
        py_result = emfx(py_mod, type="calendar")

        assert py_result.event_times is not None
        assert len(py_result.event_times) == len(known_calendar)

        for i, t in enumerate(py_result.event_times):
            key = int(t)
            assert key in known_calendar, f"Unexpected calendar time={key}"
            np.testing.assert_allclose(
                py_result.att_by_event[i],
                known_calendar[key],
                rtol=1e-7,
                atol=1e-8,
                err_msg=f"Known calendar ATT mismatch at t={key}",
            )
