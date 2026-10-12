"""Validation tests comparing Python drdid implementation with R DRDID package."""

import json
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

from tests.helpers import importorskip

pl = importorskip("polars")
np = importorskip("numpy")

from moderndid.core.preprocess import preprocess_drdid
from moderndid.drdid import drdid_imp_local_rc, drdid_imp_rc, drdid_rc, drdid_trad_rc
from moderndid.drdid.drdid import drdid
from moderndid.drdid.ipwdid import ipwdid
from moderndid.drdid.ordid import ordid

DATA_DIR = Path(__file__).parent / "data"


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
            input='library(DRDID); library(jsonlite); cat("OK")',
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return "OK" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


R_AVAILABLE = check_r_available()


def r_drdid(data_path, est_method="imp", panel=True, xformla=None):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    xformla_str = f"xformla = {xformla}," if xformla else ""

    r_script = f"""
library(DRDID)
library(jsonlite)

data <- read.csv("{data_path}")

result <- drdid(
  yname = "re",
  tname = "year",
  idname = "id",
  dname = "experimental",
  {xformla_str}
  data = data,
  panel = {str(panel).upper()},
  estMethod = "{est_method}"
)

out <- list(
  ATT = result$ATT,
  se = result$se,
  lci = result$lci,
  uci = result$uci
)

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_drdid_rc_estimator(data_path, estimator, xformla=None):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    xformla_str = xformla if xformla else "~ 1"

    r_script = f"""
library(DRDID)
library(jsonlite)

data <- read.csv("{data_path}")

tlist <- sort(unique(data$year))
pre_period <- tlist[1]
post_period <- tlist[2]

data$post <- as.numeric(data$year == post_period)

covariates <- model.matrix({xformla_str}, data = data)

i.weights <- rep(1, nrow(data))
i.weights <- i.weights / mean(i.weights)

result <- {estimator}(
  y = data$re,
  post = data$post,
  D = data$experimental,
  covariates = covariates,
  i.weights = i.weights
)

out <- list(
  ATT = result$ATT,
  se = result$se,
  lci = result$lci,
  uci = result$uci
)

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_ipwdid(data_path, normalized=True, panel=True, xformla=None):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    xformla_str = f"xformla = {xformla}," if xformla else ""

    r_script = f"""
library(DRDID)
library(jsonlite)

data <- read.csv("{data_path}")

result <- ipwdid(
  yname = "re",
  tname = "year",
  idname = "id",
  dname = "experimental",
  {xformla_str}
  data = data,
  panel = {str(panel).upper()},
  normalized = {str(normalized).upper()}
)

out <- list(
  ATT = result$ATT,
  se = result$se,
  lci = result$lci,
  uci = result$uci
)

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_ordid(data_path, panel=True, xformla=None):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    xformla_str = f"xformla = {xformla}," if xformla else ""

    r_script = f"""
library(DRDID)
library(jsonlite)

data <- read.csv("{data_path}")

result <- ordid(
  yname = "re",
  tname = "year",
  idname = "id",
  dname = "experimental",
  {xformla_str}
  data = data,
  panel = {str(panel).upper()}
)

out <- list(
  ATT = result$ATT,
  se = result$se,
  lci = result$lci,
  uci = result$uci
)

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_estimator_error(data_path, estimator):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    r_script = f"""
library(DRDID)
library(jsonlite)

data <- read.csv("{data_path}")

message <- tryCatch(
  {{
    {estimator}(yname = "re", tname = "year", idname = "id", dname = "experimental", data = data, panel = TRUE)
    ""
  }},
  error = function(e) conditionMessage(e)
)

write_json(list(message = message), "{result_path}", auto_unbox = TRUE)
"""
    try:
        return _run_r_script(r_script, result_path)["message"]
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_rc_estimator_on_design(data_path, estimator, covariates):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    r_script = f"""
library(DRDID)
library(jsonlite)

data <- read.csv("{data_path}")
covariates <- model.matrix(~ {" + ".join(covariates)}, data = data)

out <- tryCatch(
  {{
    result <- {estimator}(y = data$y, post = as.numeric(data$year == 2001), D = data$d, covariates = covariates)
    list(ATT = result$ATT, se = result$se, message = "")
  }},
  error = function(e) list(ATT = NA, se = NA, message = conditionMessage(e))
)

write_json(out, "{result_path}", auto_unbox = TRUE, digits = NA)
"""
    try:
        return _run_r_script(r_script, result_path)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def unidentified_treated_cell(design):
    if design == "married":
        rng = np.random.default_rng(6)
        n = 4000
        post = (rng.uniform(size=n) < 0.5).astype(int)
        x0 = rng.normal(size=n)
        married = (rng.uniform(size=n) < 0.5).astype(int)
        d = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.5 * x0))).astype(int)
        married[(d == 1) & (post == 0)] = 1
        y = 1 + x0 + 0.5 * married + d * post + rng.normal(size=n)
        return pl.DataFrame({"year": 2000 + post, "y": y, "d": d, "x0": x0, "married": married})
    sizes = [4, 140, 60, 140] if design == "small_pre" else [60, 140, 4, 140]
    rng = np.random.default_rng(3)
    post = np.repeat([0, 0, 1, 1], sizes)
    d = np.repeat([1, 0, 1, 0], sizes)
    x = rng.normal(size=(post.size, 5))
    y = x @ np.arange(1, 6) + d * post + rng.normal(size=post.size)
    return pl.DataFrame({"year": 2000 + post, "y": y, "d": d, **{f"x{j}": x[:, j] for j in range(5)}})


def rc_arrays(data, covariates):
    return {
        "y": data["y"].to_numpy(),
        "post": (data["year"] == 2001).cast(pl.Int64).to_numpy(),
        "d": data["d"].to_numpy(),
        "covariates": np.column_stack([np.ones(data.height), data.select(covariates).to_numpy()]),
    }


def python_drdid(data, est_method="imp", panel=True, xformla=None):
    return drdid(
        data=data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id" if panel else None,
        xformla=xformla,
        panel=panel,
        est_method=est_method,
    )


def python_drdid_rc_estimator(data, estimator, xformla=None):
    dp = preprocess_drdid(data=data, yname="re", tname="year", treat_col="experimental", xformla=xformla, panel=False)
    return estimator(y=dp.y, post=dp.post, d=dp.D, covariates=dp.covariates, i_weights=dp.weights)


def python_ipwdid(data, est_method="std_ipw", panel=True, xformla=None):
    return ipwdid(
        data=data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id" if panel else None,
        xformla=xformla,
        panel=panel,
        est_method=est_method,
    )


def python_ordid(data, panel=True, xformla=None):
    return ordid(
        data=data,
        yname="re",
        tname="year",
        treatname="experimental",
        idname="id" if panel else None,
        xformla=xformla,
        panel=panel,
    )


@pytest.fixture(scope="module")
def nsw_data():
    data_path = DATA_DIR / "nsw_long_r.csv.gz"
    if not data_path.exists():
        pytest.fail("R NSW data file not found")
    return pl.read_csv(data_path)


@pytest.fixture(scope="module")
def data_path():
    return DATA_DIR / "nsw_long_r.csv.gz"


@pytest.mark.skipif(not R_AVAILABLE, reason="R DRDID package not available")
@pytest.mark.parametrize("est_method", ["imp", "trad"])
def test_drdid_panel_no_covariates(nsw_data, data_path, est_method):
    r_result = r_drdid(data_path, est_method=est_method, panel=True)

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = python_drdid(nsw_data, est_method=est_method, panel=True)

    np.testing.assert_allclose(
        py_result.att,
        r_result["ATT"],
        rtol=1e-6,
        err_msg=f"Panel {est_method}: ATT mismatch",
    )
    np.testing.assert_allclose(
        py_result.se,
        r_result["se"],
        rtol=1e-4,
        err_msg=f"Panel {est_method}: SE mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DRDID package not available")
@pytest.mark.parametrize("est_method", ["imp", "trad"])
def test_drdid_panel_with_covariates(nsw_data, data_path, est_method):
    xformla = "~ age + educ + black + married + nodegree + hisp"

    r_result = r_drdid(data_path, est_method=est_method, panel=True, xformla=xformla)

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = python_drdid(nsw_data, est_method=est_method, panel=True, xformla=xformla)

    np.testing.assert_allclose(
        py_result.att,
        r_result["ATT"],
        rtol=1e-6,
        err_msg=f"Panel {est_method} with covariates: ATT mismatch",
    )
    np.testing.assert_allclose(
        py_result.se,
        r_result["se"],
        rtol=1e-4,
        err_msg=f"Panel {est_method} with covariates: SE mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DRDID package not available")
@pytest.mark.parametrize("est_method", ["imp", "trad"])
def test_drdid_rc_no_covariates(nsw_data, data_path, est_method):
    r_result = r_drdid(data_path, est_method=est_method, panel=False)

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = python_drdid(nsw_data, est_method=est_method, panel=False)

    np.testing.assert_allclose(
        py_result.att,
        r_result["ATT"],
        rtol=1e-6,
        err_msg=f"RC {est_method}: ATT mismatch",
    )
    np.testing.assert_allclose(
        py_result.se,
        r_result["se"],
        rtol=1e-4,
        err_msg=f"RC {est_method}: SE mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DRDID package not available")
@pytest.mark.parametrize("est_method", ["imp", "trad"])
def test_drdid_rc_with_covariates(nsw_data, data_path, est_method):
    xformla = "~ age + educ + black + married"

    r_result = r_drdid(data_path, est_method=est_method, panel=False, xformla=xformla)

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = python_drdid(nsw_data, est_method=est_method, panel=False, xformla=xformla)

    np.testing.assert_allclose(
        py_result.att,
        r_result["ATT"],
        rtol=1e-6,
        err_msg=f"RC {est_method} with covariates: ATT mismatch",
    )
    np.testing.assert_allclose(
        py_result.se,
        r_result["se"],
        rtol=1e-4,
        err_msg=f"RC {est_method} with covariates: SE mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DRDID package not available")
@pytest.mark.parametrize("xformla", [None, "~ age + educ + black + married"], ids=["no_covariates", "covariates"])
@pytest.mark.parametrize(
    ("py_estimator", "r_estimator"),
    [(drdid_imp_rc, "drdid_imp_rc1"), (drdid_trad_rc, "drdid_rc1")],
    ids=["drdid_imp_rc", "drdid_trad_rc"],
)
def test_rc_estimators_that_are_not_locally_efficient_match_r_low_level_estimators(
    nsw_data, data_path, py_estimator, r_estimator, xformla
):
    r_result = r_drdid_rc_estimator(data_path, r_estimator, xformla=xformla)

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = python_drdid_rc_estimator(nsw_data, py_estimator, xformla=xformla)

    np.testing.assert_allclose(py_result.att, r_result["ATT"], rtol=1e-6, err_msg=f"{r_estimator}: ATT mismatch")
    np.testing.assert_allclose(py_result.se, r_result["se"], rtol=1e-4, err_msg=f"{r_estimator}: SE mismatch")


@pytest.mark.skipif(not R_AVAILABLE, reason="R DRDID package not available")
@pytest.mark.parametrize("normalized,est_method", [(True, "std_ipw"), (False, "ipw")])
def test_ipwdid_panel_no_covariates(nsw_data, data_path, normalized, est_method):
    r_result = r_ipwdid(data_path, normalized=normalized, panel=True)

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = python_ipwdid(nsw_data, est_method=est_method, panel=True)

    np.testing.assert_allclose(
        py_result.att,
        r_result["ATT"],
        rtol=1e-6,
        err_msg=f"Panel IPW normalized={normalized}: ATT mismatch",
    )
    np.testing.assert_allclose(
        py_result.se,
        r_result["se"],
        rtol=1e-4,
        err_msg=f"Panel IPW normalized={normalized}: SE mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DRDID package not available")
@pytest.mark.parametrize("normalized,est_method", [(True, "std_ipw"), (False, "ipw")])
def test_ipwdid_panel_with_covariates(nsw_data, data_path, normalized, est_method):
    xformla = "~ age + educ + black + married + nodegree + hisp"

    r_result = r_ipwdid(data_path, normalized=normalized, panel=True, xformla=xformla)

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = python_ipwdid(nsw_data, est_method=est_method, panel=True, xformla=xformla)

    np.testing.assert_allclose(
        py_result.att,
        r_result["ATT"],
        rtol=1e-6,
        err_msg=f"Panel IPW normalized={normalized} with covariates: ATT mismatch",
    )
    np.testing.assert_allclose(
        py_result.se,
        r_result["se"],
        rtol=1e-4,
        err_msg=f"Panel IPW normalized={normalized} with covariates: SE mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DRDID package not available")
@pytest.mark.parametrize("normalized,est_method", [(True, "std_ipw"), (False, "ipw")])
def test_ipwdid_rc_no_covariates(nsw_data, data_path, normalized, est_method):
    r_result = r_ipwdid(data_path, normalized=normalized, panel=False)

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = python_ipwdid(nsw_data, est_method=est_method, panel=False)

    np.testing.assert_allclose(
        py_result.att,
        r_result["ATT"],
        rtol=1e-6,
        err_msg=f"RC IPW normalized={normalized}: ATT mismatch",
    )
    np.testing.assert_allclose(
        py_result.se,
        r_result["se"],
        rtol=1e-4,
        err_msg=f"RC IPW normalized={normalized}: SE mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DRDID package not available")
@pytest.mark.parametrize("normalized,est_method", [(True, "std_ipw"), (False, "ipw")])
def test_ipwdid_rc_with_covariates(nsw_data, data_path, normalized, est_method):
    xformla = "~ age + educ + black + married"

    r_result = r_ipwdid(data_path, normalized=normalized, panel=False, xformla=xformla)

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = python_ipwdid(nsw_data, est_method=est_method, panel=False, xformla=xformla)

    np.testing.assert_allclose(
        py_result.att,
        r_result["ATT"],
        rtol=1e-6,
        err_msg=f"RC IPW normalized={normalized} with covariates: ATT mismatch",
    )
    np.testing.assert_allclose(
        py_result.se,
        r_result["se"],
        rtol=1e-4,
        err_msg=f"RC IPW normalized={normalized} with covariates: SE mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DRDID package not available")
def test_ordid_panel_no_covariates(nsw_data, data_path):
    r_result = r_ordid(data_path, panel=True)

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = python_ordid(nsw_data, panel=True)

    np.testing.assert_allclose(
        py_result.att,
        r_result["ATT"],
        rtol=1e-6,
        err_msg="Panel OR: ATT mismatch",
    )
    np.testing.assert_allclose(
        py_result.se,
        r_result["se"],
        rtol=1e-4,
        err_msg="Panel OR: SE mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DRDID package not available")
def test_ordid_panel_with_covariates(nsw_data, data_path):
    xformla = "~ age + educ + black + married + nodegree + hisp"

    r_result = r_ordid(data_path, panel=True, xformla=xformla)

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = python_ordid(nsw_data, panel=True, xformla=xformla)

    np.testing.assert_allclose(
        py_result.att,
        r_result["ATT"],
        rtol=1e-6,
        err_msg="Panel OR with covariates: ATT mismatch",
    )
    np.testing.assert_allclose(
        py_result.se,
        r_result["se"],
        rtol=1e-4,
        err_msg="Panel OR with covariates: SE mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DRDID package not available")
def test_ordid_rc_no_covariates(nsw_data, data_path):
    r_result = r_ordid(data_path, panel=False)

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = python_ordid(nsw_data, panel=False)

    np.testing.assert_allclose(
        py_result.att,
        r_result["ATT"],
        rtol=1e-6,
        err_msg="RC OR: ATT mismatch",
    )
    np.testing.assert_allclose(
        py_result.se,
        r_result["se"],
        rtol=1e-4,
        err_msg="RC OR: SE mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DRDID package not available")
def test_ordid_rc_with_covariates(nsw_data, data_path):
    xformla = "~ age + educ + black + married"

    r_result = r_ordid(data_path, panel=False, xformla=xformla)

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = python_ordid(nsw_data, panel=False, xformla=xformla)

    np.testing.assert_allclose(
        py_result.att,
        r_result["ATT"],
        rtol=1e-6,
        err_msg="RC OR with covariates: ATT mismatch",
    )
    np.testing.assert_allclose(
        py_result.se,
        r_result["se"],
        rtol=1e-4,
        err_msg="RC OR with covariates: SE mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DRDID package not available")
def test_drdid_confidence_interval(nsw_data, data_path):
    xformla = "~ age + educ + black"

    r_result = r_drdid(data_path, est_method="imp", panel=True, xformla=xformla)

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = python_drdid(nsw_data, est_method="imp", panel=True, xformla=xformla)

    np.testing.assert_allclose(
        py_result.lci,
        r_result["lci"],
        rtol=1e-4,
        err_msg="LCI mismatch",
    )
    np.testing.assert_allclose(
        py_result.uci,
        r_result["uci"],
        rtol=1e-4,
        err_msg="UCI mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DRDID package not available")
def test_ipwdid_confidence_interval(nsw_data, data_path):
    xformla = "~ age + educ + black"

    r_result = r_ipwdid(data_path, normalized=True, panel=True, xformla=xformla)

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = python_ipwdid(nsw_data, est_method="std_ipw", panel=True, xformla=xformla)

    np.testing.assert_allclose(
        py_result.lci,
        r_result["lci"],
        rtol=1e-4,
        err_msg="LCI mismatch",
    )
    np.testing.assert_allclose(
        py_result.uci,
        r_result["uci"],
        rtol=1e-4,
        err_msg="UCI mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DRDID package not available")
def test_ordid_confidence_interval(nsw_data, data_path):
    xformla = "~ age + educ + black"

    r_result = r_ordid(data_path, panel=True, xformla=xformla)

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = python_ordid(nsw_data, panel=True, xformla=xformla)

    np.testing.assert_allclose(
        py_result.lci,
        r_result["lci"],
        rtol=1e-4,
        err_msg="LCI mismatch",
    )
    np.testing.assert_allclose(
        py_result.uci,
        r_result["uci"],
        rtol=1e-4,
        err_msg="UCI mismatch",
    )


def test_invalid_panel_est_method(nsw_data):
    with pytest.raises(ValueError, match=re.escape("est_method='bad' is not valid. Must be 'imp' or 'trad'.")):
        drdid(
            data=nsw_data,
            yname="re",
            tname="year",
            treatname="experimental",
            idname="id",
            panel=True,
            est_method="bad",
        )


def test_invalid_rc_est_method(nsw_data):
    with pytest.raises(ValueError, match=re.escape("est_method='bad' is not valid. Must be 'imp' or 'trad'.")):
        drdid(
            data=nsw_data,
            yname="re",
            tname="year",
            treatname="experimental",
            panel=False,
            est_method="bad",
        )


@pytest.mark.parametrize("trim_level", [0, 1, -0.5, 1.5])
def test_invalid_trim_level(nsw_data, trim_level):
    with pytest.raises(ValueError, match="trim_level=.*is not valid"):
        drdid(
            data=nsw_data,
            yname="re",
            tname="year",
            treatname="experimental",
            idname="id",
            panel=True,
            est_method="imp",
            trim_level=trim_level,
        )


@pytest.mark.skipif(not R_AVAILABLE, reason="R DRDID package not available")
@pytest.mark.parametrize("panel", [True, False])
def test_drdid_covariates_without_formulaic_match_r(nsw_data, data_path, without_formulaic, panel):
    xformla = "~ age + educ + black + married"

    r_result = r_drdid(data_path, est_method="trad", panel=panel, xformla=xformla)

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = python_drdid(nsw_data, est_method="trad", panel=panel, xformla=xformla)

    np.testing.assert_allclose(py_result.att, r_result["ATT"], rtol=1e-8)
    np.testing.assert_allclose(py_result.se, r_result["se"], rtol=1e-8)


@pytest.mark.skipif(not R_AVAILABLE, reason="R DRDID package not available")
@pytest.mark.parametrize("panel", [True, False])
def test_ipwdid_covariates_without_formulaic_match_r(nsw_data, data_path, without_formulaic, panel):
    xformla = "~ age + educ + black + married"

    r_result = r_ipwdid(data_path, normalized=True, panel=panel, xformla=xformla)

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = python_ipwdid(nsw_data, est_method="std_ipw", panel=panel, xformla=xformla)

    np.testing.assert_allclose(py_result.att, r_result["ATT"], rtol=1e-8)
    np.testing.assert_allclose(py_result.se, r_result["se"], rtol=1e-8)


@pytest.mark.skipif(not R_AVAILABLE, reason="R DRDID package not available")
@pytest.mark.parametrize("panel", [True, False])
def test_ordid_covariates_without_formulaic_match_r(nsw_data, data_path, without_formulaic, panel):
    xformla = "~ age + educ + black + married"

    r_result = r_ordid(data_path, panel=panel, xformla=xformla)

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = python_ordid(nsw_data, panel=panel, xformla=xformla)

    np.testing.assert_allclose(py_result.att, r_result["ATT"], rtol=1e-8)
    np.testing.assert_allclose(py_result.se, r_result["se"], rtol=1e-8)


@pytest.mark.skipif(not R_AVAILABLE, reason="R DRDID package not available")
@pytest.mark.parametrize(
    ("estimator", "py_estimator"), [("drdid", drdid), ("ipwdid", ipwdid), ("ordid", ordid)], ids=["dr", "ipw", "or"]
)
def test_repeated_unit_periods_raise_like_r(nsw_repeated_row, nsw_repeated_row_csv_path, estimator, py_estimator):
    r_message = r_estimator_error(nsw_repeated_row_csv_path, estimator)

    assert r_message == "The value of idname must be the unique (by tname)"
    with pytest.raises(ValueError, match=re.escape("The value of idname must be unique (by tname).")):
        py_estimator(data=nsw_repeated_row, yname="re", tname="year", treatname="experimental", idname="id", panel=True)


@pytest.mark.skipif(not R_AVAILABLE, reason="R DRDID package not available")
@pytest.mark.parametrize(
    ("design", "covariates"),
    [
        ("small_pre", ["x0", "x1", "x2", "x3", "x4"]),
        ("small_post", ["x0", "x1", "x2", "x3", "x4"]),
        ("married", ["x0", "married"]),
    ],
)
@pytest.mark.parametrize(
    ("py_estimator", "r_estimator", "rtol"),
    [(drdid_imp_rc, "drdid_imp_rc1", 1e-8), (drdid_trad_rc, "drdid_rc1", 1e-12)],
    ids=["drdid_imp_rc", "drdid_trad_rc"],
)
def test_rc_estimators_without_treated_regressions_match_r_on_an_unidentified_treated_cell(
    tmp_path, design, covariates, py_estimator, r_estimator, rtol
):
    data = unidentified_treated_cell(design)
    data.write_csv(tmp_path / "data.csv")

    r_result = r_rc_estimator_on_design(tmp_path / "data.csv", r_estimator, covariates)
    py_result = py_estimator(**rc_arrays(data, covariates))

    if r_result is None:
        pytest.fail("R estimation failed")
    assert r_result["message"] == ""
    np.testing.assert_allclose([py_result.att, py_result.se], [r_result["ATT"], r_result["se"]], rtol=rtol)


@pytest.mark.skipif(not R_AVAILABLE, reason="R DRDID package not available")
@pytest.mark.parametrize(
    ("design", "covariates", "period"),
    [
        ("small_pre", ["x0", "x1", "x2", "x3", "x4"], "pre"),
        ("small_post", ["x0", "x1", "x2", "x3", "x4"], "post"),
        ("married", ["x0", "married"], "pre"),
    ],
)
def test_drdid_rc_rejects_an_unidentified_treated_cell_like_r(tmp_path, design, covariates, period):
    data = unidentified_treated_cell(design)
    data.write_csv(tmp_path / "data.csv")

    r_result = r_rc_estimator_on_design(tmp_path / "data.csv", "drdid_rc", covariates)

    if r_result is None:
        pytest.fail("R estimation failed")
    assert r_result["message"] == (
        f"The regression design matrix for {period}-treatment is singular. Consider removing some covariates."
    )
    with pytest.raises(ValueError, match=f"treated group in the {period}-period"):
        drdid_rc(**rc_arrays(data, covariates))


@pytest.mark.skipif(not R_AVAILABLE, reason="R DRDID package not available")
def test_drdid_imp_local_rc_rejects_a_constant_treated_covariate_where_r_depends_on_column_order(tmp_path):
    data = unidentified_treated_cell("married")
    data.write_csv(tmp_path / "data.csv")
    message = "The regression design matrix of the treated group in the pre-period is singular."

    given = r_rc_estimator_on_design(tmp_path / "data.csv", "drdid_imp_rc", ["x0", "married"])
    swapped = r_rc_estimator_on_design(tmp_path / "data.csv", "drdid_imp_rc", ["married", "x0"])

    if given is None or swapped is None:
        pytest.fail("R estimation failed")
    assert abs(given["ATT"] - swapped["ATT"]) > 0.1
    with pytest.raises(ValueError, match=f"^{re.escape(message)}"):
        drdid_imp_local_rc(**rc_arrays(data, ["x0", "married"]))
