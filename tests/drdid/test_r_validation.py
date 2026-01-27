# pylint: disable=redefined-outer-name
"""Validation tests comparing Python drdid implementation with R DRDID package."""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

from tests.helpers import importorskip

pl = importorskip("polars")
np = importorskip("numpy")

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


def r_drdid_imp_rc1(data_path, xformla=None):
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

result <- drdid_imp_rc1(
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
        pytest.skip("R NSW data file not found")
    return pl.read_csv(data_path)


@pytest.fixture(scope="module")
def data_path():
    return DATA_DIR / "nsw_long_r.csv.gz"


@pytest.mark.skipif(not R_AVAILABLE, reason="R DRDID package not available")
@pytest.mark.parametrize("est_method", ["imp", "trad"])
def test_drdid_panel_no_covariates(nsw_data, data_path, est_method):
    r_result = r_drdid(data_path, est_method=est_method, panel=True)

    if r_result is None:
        pytest.skip("R estimation failed")

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
        pytest.skip("R estimation failed")

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
    if est_method == "imp":
        r_result = r_drdid_imp_rc1(data_path)
    else:
        r_result = r_drdid(data_path, est_method=est_method, panel=False)

    if r_result is None:
        pytest.skip("R estimation failed")

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

    if est_method == "imp":
        r_result = r_drdid_imp_rc1(data_path, xformla=xformla)
    else:
        r_result = r_drdid(data_path, est_method=est_method, panel=False, xformla=xformla)

    if r_result is None:
        pytest.skip("R estimation failed")

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
@pytest.mark.parametrize("normalized,est_method", [(True, "std_ipw"), (False, "ipw")])
def test_ipwdid_panel_no_covariates(nsw_data, data_path, normalized, est_method):
    r_result = r_ipwdid(data_path, normalized=normalized, panel=True)

    if r_result is None:
        pytest.skip("R estimation failed")

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
        pytest.skip("R estimation failed")

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
        pytest.skip("R estimation failed")

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
        pytest.skip("R estimation failed")

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
        pytest.skip("R estimation failed")

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
        pytest.skip("R estimation failed")

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
        pytest.skip("R estimation failed")

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
        pytest.skip("R estimation failed")

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
        pytest.skip("R estimation failed")

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
        pytest.skip("R estimation failed")

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
        pytest.skip("R estimation failed")

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
