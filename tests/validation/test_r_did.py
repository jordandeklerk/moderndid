"""Validation tests comparing Python did implementation with R did package."""

import json
import re
import subprocess
import tempfile

import pytest

pytestmark = pytest.mark.slow

from tests.helpers import importorskip

pl = importorskip("polars")
np = importorskip("numpy")

from moderndid import aggte, att_gt, load_mpdta, mboot
from moderndid.core.numba_utils import multiplier_bootstrap


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
            input='library(did); library(jsonlite); cat("OK")',
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return "OK" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


R_AVAILABLE = check_r_available()


def r_att_gt(
    data_path,
    est_method="dr",
    control_group="nevertreated",
    base_period="varying",
    anticipation=0,
    xformla="~1",
    panel=True,
    weightsname=None,
):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    weightsname_str = "NULL" if weightsname is None else f'"{weightsname}"'

    r_script = f"""
library(did)
library(jsonlite)

data <- read.csv("{data_path}")

result <- att_gt(
  yname = "lemp",
  tname = "year",
  idname = "countyreal",
  gname = "first.treat",
  xformla = {xformla},
  data = data,
  est_method = "{est_method}",
  control_group = "{control_group}",
  base_period = "{base_period}",
  anticipation = {anticipation},
  panel = {str(panel).upper()},
  weightsname = {weightsname_str},
  bstrap = FALSE
)

out <- list(
  groups = result$group,
  times = result$t,
  att_gt = result$att,
  se_gt = result$se,
  critical_value = result$c,
  n_units = result$n
)

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_att_gt_preprocessing_path(data_path, clustervars=None, faster_mode=True):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    clustervars_str = "NULL" if clustervars is None else f'"{clustervars}"'

    r_script = f"""
library(did)
library(jsonlite)

data <- read.csv("{data_path}")

result <- att_gt(
  yname = "lemp",
  tname = "year",
  idname = "countyreal",
  gname = "first.treat",
  data = data,
  est_method = "reg",
  clustervars = {clustervars_str},
  bstrap = FALSE,
  faster_mode = {str(faster_mode).upper()}
)

out <- list(
  groups = result$group,
  times = result$t,
  att_gt = result$att,
  se_gt = result$se,
  n_units = result$n
)

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_att_gt_error(data_path, weightsname=None):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    weightsname_str = "NULL" if weightsname is None else f'"{weightsname}"'

    r_script = f"""
library(did)
library(jsonlite)

data <- read.csv("{data_path}")

message <- tryCatch(
  {{
    att_gt(yname = "lemp", tname = "year", idname = "countyreal", gname = "first.treat", data = data,
           est_method = "reg", weightsname = {weightsname_str}, bstrap = FALSE)
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


def r_att_gt_bootstrap(data_path, est_method="dr", biters=100, cband=True, random_state=42):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    r_script = f"""
library(did)
library(jsonlite)

set.seed({random_state})

data <- read.csv("{data_path}")

result <- att_gt(
  yname = "lemp",
  tname = "year",
  idname = "countyreal",
  gname = "first.treat",
  xformla = ~1,
  data = data,
  est_method = "{est_method}",
  control_group = "nevertreated",
  bstrap = TRUE,
  biters = {biters},
  cband = {str(cband).upper()}
)

out <- list(
  groups = result$group,
  times = result$t,
  att_gt = result$att,
  se_gt = result$se,
  critical_value = result$c
)

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path, timeout=300)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_aggte(
    data_path,
    agg_type="simple",
    est_method="dr",
    balance_e=None,
    min_e=None,
    max_e=None,
    na_rm=False,
    weightsname=None,
    allow_unbalanced_panel=False,
):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    balance_e_str = "NULL" if balance_e is None else str(balance_e)
    min_e_str = "-Inf" if min_e is None else str(min_e)
    max_e_str = "Inf" if max_e is None else str(max_e)
    na_rm_str = "TRUE" if na_rm else "FALSE"
    weightsname_str = "NULL" if weightsname is None else f'"{weightsname}"'
    allow_unbalanced_panel_str = "TRUE" if allow_unbalanced_panel else "FALSE"

    r_script = f"""
library(did)
library(jsonlite)

data <- read.csv("{data_path}")

mp_result <- att_gt(
  yname = "lemp",
  tname = "year",
  idname = "countyreal",
  gname = "first.treat",
  xformla = ~1,
  data = data,
  est_method = "{est_method}",
  control_group = "nevertreated",
  weightsname = {weightsname_str},
  allow_unbalanced_panel = {allow_unbalanced_panel_str},
  bstrap = FALSE
)

agg_result <- aggte(
  mp_result,
  type = "{agg_type}",
  balance_e = {balance_e_str},
  min_e = {min_e_str},
  max_e = {max_e_str},
  na.rm = {na_rm_str},
  bstrap = FALSE
)

if ("{agg_type}" == "simple") {{
    out <- list(
        overall_att = agg_result$overall.att,
        overall_se = agg_result$overall.se
    )
}} else {{
    out <- list(
        overall_att = agg_result$overall.att,
        overall_se = agg_result$overall.se,
        egt = agg_result$egt,
        att_egt = agg_result$att.egt,
        se_egt = agg_result$se.egt
    )
}}

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_aggte_bootstrap(data_path, agg_type="simple", biters=100, cband=True, random_state=42):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    r_script = f"""
library(did)
library(jsonlite)

set.seed({random_state})

data <- read.csv("{data_path}")

mp_result <- att_gt(
  yname = "lemp",
  tname = "year",
  idname = "countyreal",
  gname = "first.treat",
  xformla = ~1,
  data = data,
  est_method = "dr",
  control_group = "nevertreated",
  bstrap = TRUE,
  biters = {biters},
  cband = {str(cband).upper()}
)

agg_result <- aggte(
  mp_result,
  type = "{agg_type}",
  bstrap = TRUE,
  biters = {biters},
  cband = {str(cband).upper()}
)

if ("{agg_type}" == "simple") {{
    out <- list(
        overall_att = agg_result$overall.att,
        overall_se = agg_result$overall.se,
        critical_value = agg_result$crit.val
    )
}} else {{
    out <- list(
        overall_att = agg_result$overall.att,
        overall_se = agg_result$overall.se,
        critical_value = agg_result$crit.val,
        egt = agg_result$egt,
        att_egt = agg_result$att.egt,
        se_egt = agg_result$se.egt
    )
}}

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path, timeout=300)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


@pytest.fixture(scope="module")
def mpdta_data():
    return load_mpdta()


@pytest.fixture(scope="module")
def mpdta_csv_path(mpdta_data):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        mpdta_data.write_csv(f.name)
        return f.name


@pytest.fixture(scope="module")
def mpdta_small(mpdta_data):
    unique_counties = mpdta_data["countyreal"].unique().sort()[:100].to_list()
    return mpdta_data.filter(pl.col("countyreal").is_in(unique_counties))


@pytest.fixture(scope="module")
def mpdta_small_csv_path(mpdta_small):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        mpdta_small.write_csv(f.name)
        return f.name


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
def test_att_gt_estimation_methods(mpdta_data, mpdta_csv_path, est_method):
    r_result = r_att_gt(mpdta_csv_path, est_method=est_method)

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        est_method=est_method,
        control_group="nevertreated",
        boot=False,
    )

    r_groups = np.array(r_result["groups"])
    r_times = np.array(r_result["times"])
    r_att = np.array(r_result["att_gt"])

    assert len(py_result.groups) == len(r_groups), f"{est_method}: Number of group-time pairs mismatch"

    for i, (g, t) in enumerate(zip(py_result.groups, py_result.times)):
        r_mask = (r_groups == g) & (r_times == t)
        if np.any(r_mask):
            r_idx = np.where(r_mask)[0][0]
            py_att = py_result.att_gt[i]
            r_att_val = r_att[r_idx]

            if np.isnan(py_att) and np.isnan(r_att_val):
                continue
            if not np.isnan(py_att) and not np.isnan(r_att_val):
                np.testing.assert_allclose(
                    py_att,
                    r_att_val,
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"{est_method}: ATT mismatch at g={g}, t={t}",
                )


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
def test_att_gt_standard_errors(mpdta_data, mpdta_csv_path, est_method):
    r_result = r_att_gt(mpdta_csv_path, est_method=est_method)

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        est_method=est_method,
        control_group="nevertreated",
        boot=False,
    )

    r_groups = np.array(r_result["groups"])
    r_times = np.array(r_result["times"])
    r_se = np.array(r_result["se_gt"])

    for i, (g, t) in enumerate(zip(py_result.groups, py_result.times)):
        r_mask = (r_groups == g) & (r_times == t)
        if np.any(r_mask):
            r_idx = np.where(r_mask)[0][0]
            py_se = py_result.se_gt[i]
            r_se_val = r_se[r_idx]

            if np.isnan(py_se) and np.isnan(r_se_val):
                continue
            if not np.isnan(py_se) and not np.isnan(r_se_val):
                np.testing.assert_allclose(
                    py_se,
                    r_se_val,
                    rtol=1e-3,
                    atol=1e-4,
                    err_msg=f"{est_method}: SE mismatch at g={g}, t={t}",
                )


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
def test_att_gt_control_group_nevertreated(mpdta_data, mpdta_csv_path):
    r_result = r_att_gt(mpdta_csv_path, est_method="reg", control_group="nevertreated")

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        est_method="reg",
        control_group="nevertreated",
        boot=False,
    )

    r_groups = np.array(r_result["groups"])
    r_times = np.array(r_result["times"])
    r_att = np.array(r_result["att_gt"])

    matches = 0
    total = 0
    for i, (g, t) in enumerate(zip(py_result.groups, py_result.times)):
        r_mask = (r_groups == g) & (r_times == t)
        if np.any(r_mask):
            r_idx = np.where(r_mask)[0][0]
            py_att = py_result.att_gt[i]
            r_att_val = r_att[r_idx]
            total += 1

            if np.isnan(py_att) and np.isnan(r_att_val):
                matches += 1
            elif not np.isnan(py_att) and not np.isnan(r_att_val):
                if np.allclose(py_att, r_att_val, rtol=1e-5, atol=1e-6):
                    matches += 1

    match_rate = matches / total if total > 0 else 0
    assert match_rate > 0.95, f"nevertreated: Only {match_rate:.1%} of ATT(g,t) estimates match"


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
def test_att_gt_control_group_notyettreated(mpdta_data, mpdta_csv_path):
    r_result = r_att_gt(mpdta_csv_path, est_method="reg", control_group="notyettreated")

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        est_method="reg",
        control_group="notyettreated",
        boot=False,
    )

    r_groups = np.array(r_result["groups"])
    r_times = np.array(r_result["times"])
    r_att = np.array(r_result["att_gt"])

    matches = 0
    total = 0
    for i, (g, t) in enumerate(zip(py_result.groups, py_result.times)):
        r_mask = (r_groups == g) & (r_times == t)
        if np.any(r_mask):
            r_idx = np.where(r_mask)[0][0]
            py_att = py_result.att_gt[i]
            r_att_val = r_att[r_idx]
            total += 1

            if np.isnan(py_att) and np.isnan(r_att_val):
                matches += 1
            elif not np.isnan(py_att) and not np.isnan(r_att_val):
                if np.allclose(py_att, r_att_val, rtol=1e-5, atol=1e-6):
                    matches += 1

    match_rate = matches / total if total > 0 else 0
    assert match_rate > 0.95, f"notyettreated: Only {match_rate:.1%} of ATT(g,t) estimates match"


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
@pytest.mark.parametrize("base_period", ["varying", "universal"])
def test_att_gt_base_periods(mpdta_data, mpdta_csv_path, base_period):
    r_result = r_att_gt(mpdta_csv_path, est_method="reg", base_period=base_period)

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        est_method="reg",
        base_period=base_period,
        boot=False,
    )

    r_groups = np.array(r_result["groups"])
    r_times = np.array(r_result["times"])
    r_att = np.array(r_result["att_gt"])

    matches = 0
    total = 0
    for i, (g, t) in enumerate(zip(py_result.groups, py_result.times)):
        r_mask = (r_groups == g) & (r_times == t)
        if np.any(r_mask):
            r_idx = np.where(r_mask)[0][0]
            py_att = py_result.att_gt[i]
            r_att_val = r_att[r_idx]
            total += 1

            if np.isnan(py_att) and np.isnan(r_att_val):
                matches += 1
            elif not np.isnan(py_att) and not np.isnan(r_att_val):
                if np.allclose(py_att, r_att_val, rtol=1e-5, atol=1e-6):
                    matches += 1

    match_rate = matches / total if total > 0 else 0
    assert match_rate > 0.95, f"{base_period}: Only {match_rate:.1%} of ATT(g,t) estimates match"


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
@pytest.mark.parametrize("anticipation", [0, 1])
def test_att_gt_anticipation(mpdta_data, mpdta_csv_path, anticipation):
    r_result = r_att_gt(mpdta_csv_path, est_method="reg", anticipation=anticipation)

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        est_method="reg",
        anticipation=anticipation,
        boot=False,
    )

    r_groups = np.array(r_result["groups"])
    r_times = np.array(r_result["times"])
    r_att = np.array(r_result["att_gt"])

    matches = 0
    total = 0
    for i, (g, t) in enumerate(zip(py_result.groups, py_result.times)):
        r_mask = (r_groups == g) & (r_times == t)
        if np.any(r_mask):
            r_idx = np.where(r_mask)[0][0]
            py_att = py_result.att_gt[i]
            r_att_val = r_att[r_idx]
            total += 1

            if np.isnan(py_att) and np.isnan(r_att_val):
                matches += 1
            elif not np.isnan(py_att) and not np.isnan(r_att_val):
                if np.allclose(py_att, r_att_val, rtol=1e-5, atol=1e-6):
                    matches += 1

    match_rate = matches / total if total > 0 else 0
    assert match_rate > 0.90, f"anticipation={anticipation}: Only {match_rate:.1%} of ATT(g,t) estimates match"


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
def test_att_gt_with_covariates(mpdta_data, mpdta_csv_path):
    r_result = r_att_gt(mpdta_csv_path, est_method="dr", xformla="~lpop")

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        est_method="dr",
        control_group="nevertreated",
        boot=False,
    )

    r_groups = np.array(r_result["groups"])
    r_times = np.array(r_result["times"])
    r_att = np.array(r_result["att_gt"])

    matches = 0
    total = 0
    for i, (g, t) in enumerate(zip(py_result.groups, py_result.times)):
        r_mask = (r_groups == g) & (r_times == t)
        if np.any(r_mask):
            r_idx = np.where(r_mask)[0][0]
            py_att = py_result.att_gt[i]
            r_att_val = r_att[r_idx]
            total += 1

            if np.isnan(py_att) and np.isnan(r_att_val):
                matches += 1
            elif not np.isnan(py_att) and not np.isnan(r_att_val):
                if np.allclose(py_att, r_att_val, rtol=1e-4, atol=1e-5):
                    matches += 1

    match_rate = matches / total if total > 0 else 0
    assert match_rate > 0.95, f"With covariates: Only {match_rate:.1%} of ATT(g,t) estimates match"


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
@pytest.mark.parametrize(
    "r_xformla, py_xformla",
    [("~log.pop", "~ log.pop"), ("~I(lpop^2)", "~ lpop_sq"), ("~log(lpop)", "~ log_lpop")],
)
def test_att_gt_formula_terms_as_columns(mpdta_formula_columns, mpdta_formula_columns_csv_path, r_xformla, py_xformla):
    r_result = r_att_gt(mpdta_formula_columns_csv_path, est_method="dr", xformla=r_xformla)

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = att_gt(
        data=mpdta_formula_columns,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla=py_xformla,
        est_method="dr",
        boot=False,
    )

    np.testing.assert_array_equal(py_result.groups, r_result["groups"])
    np.testing.assert_array_equal(py_result.times, r_result["times"])
    np.testing.assert_allclose(py_result.att_gt, r_result["att_gt"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(py_result.se_gt, r_result["se_gt"], rtol=0, atol=1e-8)


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
def test_att_gt_bootstrap_se(mpdta_small, mpdta_small_csv_path):
    r_result = r_att_gt_bootstrap(mpdta_small_csv_path, est_method="dr", biters=100, cband=False)

    if r_result is None:
        pytest.fail("R bootstrap estimation failed")

    py_result = att_gt(
        data=mpdta_small,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        est_method="dr",
        control_group="nevertreated",
        boot=True,
        biters=100,
        cband=False,
        random_state=42,
    )

    r_groups = np.array(r_result["groups"])
    r_times = np.array(r_result["times"])
    r_se = np.array(r_result["se_gt"])

    se_ratios = []
    for i, (g, t) in enumerate(zip(py_result.groups, py_result.times)):
        r_mask = (r_groups == g) & (r_times == t)
        if np.any(r_mask):
            r_idx = np.where(r_mask)[0][0]
            py_se = py_result.se_gt[i]
            r_se_val = r_se[r_idx]

            if not np.isnan(py_se) and not np.isnan(r_se_val) and r_se_val > 0:
                se_ratios.append(py_se / r_se_val)

    assert len(se_ratios) > 0, "No valid SE pairs to compare between Python and R"
    mean_ratio = np.mean(se_ratios)
    assert 0.7 < mean_ratio < 1.3, f"Bootstrap SE ratio outside reasonable range: {mean_ratio:.2f}"


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
@pytest.mark.parametrize("agg_type", ["simple", "dynamic", "group", "calendar"])
def test_aggte_overall_att(mpdta_data, mpdta_csv_path, agg_type):
    r_result = r_aggte(mpdta_csv_path, agg_type=agg_type, est_method="reg")

    if r_result is None:
        pytest.fail("R aggregation failed")

    py_mp_result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        est_method="reg",
        control_group="nevertreated",
        boot=False,
    )

    py_agg_result = aggte(py_mp_result, type=agg_type)

    np.testing.assert_allclose(
        py_agg_result.overall_att,
        r_result["overall_att"],
        rtol=1e-5,
        atol=1e-6,
        err_msg=f"{agg_type}: Overall ATT mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
@pytest.mark.parametrize("agg_type", ["simple", "dynamic", "group", "calendar"])
def test_aggte_overall_se(mpdta_data, mpdta_csv_path, agg_type):
    r_result = r_aggte(mpdta_csv_path, agg_type=agg_type, est_method="reg")

    if r_result is None:
        pytest.fail("R aggregation failed")

    py_mp_result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        est_method="reg",
        control_group="nevertreated",
        boot=False,
    )

    py_agg_result = aggte(py_mp_result, type=agg_type)

    np.testing.assert_allclose(
        py_agg_result.overall_se,
        r_result["overall_se"],
        rtol=1e-3,
        atol=1e-4,
        err_msg=f"{agg_type}: Overall SE mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
@pytest.mark.parametrize("agg_type", ["dynamic", "group", "calendar"])
def test_aggte_disaggregated_effects(mpdta_data, mpdta_csv_path, agg_type):
    r_result = r_aggte(mpdta_csv_path, agg_type=agg_type, est_method="reg")

    if r_result is None:
        pytest.fail("R aggregation failed")

    if "egt" not in r_result or r_result["egt"] is None:
        pytest.fail("R result missing disaggregated effects")

    py_mp_result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        est_method="reg",
        control_group="nevertreated",
        boot=False,
    )

    py_agg_result = aggte(py_mp_result, type=agg_type)

    r_egt = np.array(r_result["egt"])
    r_att_egt = np.array(r_result["att_egt"])

    common_egt = set(py_agg_result.event_times) & set(r_egt)
    assert len(common_egt) > 0, f"{agg_type}: No common event times between Python and R"

    compared = 0
    for e in common_egt:
        py_idx = np.where(py_agg_result.event_times == e)[0][0]
        r_idx = np.where(r_egt == e)[0][0]

        py_att = py_agg_result.att_by_event[py_idx]
        r_att = r_att_egt[r_idx]

        if np.isnan(py_att) and np.isnan(r_att):
            compared += 1
            continue
        assert not (np.isnan(py_att) ^ np.isnan(r_att)), f"{agg_type} e={e}: NaN mismatch (Python={py_att}, R={r_att})"
        np.testing.assert_allclose(
            py_att,
            r_att,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"{agg_type} e={e}: ATT mismatch",
        )
        compared += 1
    assert compared > 0, f"{agg_type}: No event times were compared"


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
def test_aggte_dynamic_with_balance_e(mpdta_data, mpdta_csv_path):
    r_result = r_aggte(mpdta_csv_path, agg_type="dynamic", est_method="reg", balance_e=1)

    if r_result is None:
        pytest.fail("R aggregation failed")

    py_mp_result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        est_method="reg",
        control_group="nevertreated",
        boot=False,
    )

    py_agg_result = aggte(py_mp_result, type="dynamic", balance_e=1)

    np.testing.assert_allclose(
        py_agg_result.overall_att,
        r_result["overall_att"],
        rtol=1e-4,
        atol=1e-5,
        err_msg="balance_e=1: Overall ATT mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
def test_aggte_dynamic_with_min_max_e(mpdta_data, mpdta_csv_path):
    r_result = r_aggte(mpdta_csv_path, agg_type="dynamic", est_method="reg", min_e=-1, max_e=2)

    if r_result is None:
        pytest.fail("R aggregation failed")

    py_mp_result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        est_method="reg",
        control_group="nevertreated",
        boot=False,
    )

    py_agg_result = aggte(py_mp_result, type="dynamic", min_e=-1, max_e=2)

    assert all(-1 <= e <= 2 for e in py_agg_result.event_times), "Python event times outside [min_e, max_e]"

    np.testing.assert_allclose(
        py_agg_result.overall_att,
        r_result["overall_att"],
        rtol=1e-4,
        atol=1e-5,
        err_msg="min_e=-1, max_e=2: Overall ATT mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
def test_aggte_estimation_methods(mpdta_data, mpdta_csv_path, est_method):
    r_result = r_aggte(mpdta_csv_path, agg_type="simple", est_method=est_method)

    if r_result is None:
        pytest.fail("R aggregation failed")

    py_mp_result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        est_method=est_method,
        control_group="nevertreated",
        boot=False,
    )

    py_agg_result = aggte(py_mp_result, type="simple")

    np.testing.assert_allclose(
        py_agg_result.overall_att,
        r_result["overall_att"],
        rtol=1e-5,
        atol=1e-6,
        err_msg=f"{est_method}: Overall ATT mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
@pytest.mark.parametrize("agg_type", ["simple", "dynamic", "group", "calendar"])
def test_aggte_weights_column_not_named_weights(mpdta_pop_weighted, mpdta_pop_weighted_csv_path, agg_type):
    r_result = r_aggte(mpdta_pop_weighted_csv_path, agg_type=agg_type, weightsname="pop")

    if r_result is None:
        pytest.fail("R aggregation failed")

    py_mp_result = att_gt(
        data=mpdta_pop_weighted,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        control_group="nevertreated",
        weightsname="pop",
        boot=False,
    )

    py_agg_result = aggte(py_mp_result, type=agg_type, cband=False)

    np.testing.assert_allclose(py_agg_result.overall_att, r_result["overall_att"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(py_agg_result.overall_se, r_result["overall_se"], rtol=1e-9, atol=1e-12)
    if agg_type != "simple":
        np.testing.assert_allclose(py_agg_result.att_by_event, r_result["att_egt"], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(py_agg_result.se_by_event, r_result["se_egt"], rtol=1e-9, atol=1e-12)


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
@pytest.mark.parametrize("agg_type", ["simple", "dynamic", "group", "calendar"])
def test_aggte_unbalanced_panel_se(mpdta_unbalanced, mpdta_unbalanced_csv_path, agg_type):
    r_result = r_aggte(mpdta_unbalanced_csv_path, agg_type=agg_type, allow_unbalanced_panel=True)

    if r_result is None:
        pytest.fail("R aggregation failed")

    py_mp_result = att_gt(
        data=mpdta_unbalanced,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        control_group="nevertreated",
        allow_unbalanced_panel=True,
        boot=False,
    )

    py_agg_result = aggte(py_mp_result, type=agg_type, cband=False)

    np.testing.assert_allclose(py_agg_result.overall_att, r_result["overall_att"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(py_agg_result.overall_se, r_result["overall_se"], rtol=1e-9, atol=1e-12)
    if agg_type != "simple":
        np.testing.assert_allclose(py_agg_result.att_by_event, r_result["att_egt"], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(py_agg_result.se_by_event, r_result["se_egt"], rtol=1e-9, atol=1e-12)


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
@pytest.mark.parametrize("agg_type", ["simple", "dynamic", "group", "calendar"])
def test_aggte_unbalanced_panel_time_varying_weights(
    mpdta_unbalanced_varying_weights, mpdta_unbalanced_varying_weights_csv_path, agg_type
):
    r_result = r_aggte(
        mpdta_unbalanced_varying_weights_csv_path, agg_type=agg_type, weightsname="w", allow_unbalanced_panel=True
    )

    if r_result is None:
        pytest.fail("R aggregation failed")

    py_mp_result = att_gt(
        data=mpdta_unbalanced_varying_weights,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        control_group="nevertreated",
        weightsname="w",
        allow_unbalanced_panel=True,
        boot=False,
    )

    py_agg_result = aggte(py_mp_result, type=agg_type, cband=False)

    np.testing.assert_allclose(py_agg_result.overall_att, r_result["overall_att"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(py_agg_result.overall_se, r_result["overall_se"], rtol=1e-9, atol=1e-12)
    if agg_type != "simple":
        np.testing.assert_allclose(py_agg_result.att_by_event, r_result["att_egt"], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(py_agg_result.se_by_event, r_result["se_egt"], rtol=1e-9, atol=1e-12)


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
@pytest.mark.parametrize("base_period", ["varying", "universal"])
@pytest.mark.parametrize(("est_method", "xformla"), [("reg", "~1"), ("dr", "~lpop")])
def test_att_gt_balanced_panel_time_varying_weights(
    mpdta_varying_weights, mpdta_varying_weights_csv_path, est_method, xformla, base_period
):
    r_result = r_att_gt(
        mpdta_varying_weights_csv_path,
        est_method=est_method,
        base_period=base_period,
        xformla=xformla,
        weightsname="w",
    )

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = att_gt(
        data=mpdta_varying_weights,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla=xformla,
        est_method=est_method,
        base_period=base_period,
        weightsname="w",
        boot=False,
    )
    r_se = np.array([np.nan if se == "NA" else se for se in r_result["se_gt"]], dtype=float)

    np.testing.assert_array_equal(py_result.groups, r_result["groups"])
    np.testing.assert_array_equal(py_result.times, r_result["times"])
    np.testing.assert_allclose(py_result.att_gt, r_result["att_gt"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(py_result.se_gt, r_se, rtol=1e-8, atol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
@pytest.mark.parametrize("agg_type", ["simple", "dynamic", "group", "calendar"])
def test_aggte_balanced_panel_time_varying_weights(mpdta_varying_weights, mpdta_varying_weights_csv_path, agg_type):
    r_result = r_aggte(mpdta_varying_weights_csv_path, agg_type=agg_type, weightsname="w")

    if r_result is None:
        pytest.fail("R aggregation failed")

    py_mp_result = att_gt(
        data=mpdta_varying_weights,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        control_group="nevertreated",
        weightsname="w",
        boot=False,
    )

    py_agg_result = aggte(py_mp_result, type=agg_type, cband=False)

    np.testing.assert_allclose(py_agg_result.overall_att, r_result["overall_att"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(py_agg_result.overall_se, r_result["overall_se"], rtol=1e-9, atol=1e-12)
    if agg_type != "simple":
        np.testing.assert_allclose(py_agg_result.att_by_event, r_result["att_egt"], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(py_agg_result.se_by_event, r_result["se_egt"], rtol=1e-9, atol=1e-12)


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
@pytest.mark.parametrize("agg_type", ["simple", "dynamic", "group", "calendar"])
def test_aggte_bootstrap_se(mpdta_small, mpdta_small_csv_path, agg_type):
    r_result = r_aggte_bootstrap(mpdta_small_csv_path, agg_type=agg_type, biters=100, cband=False)

    if r_result is None:
        pytest.fail("R bootstrap aggregation failed")

    py_mp_result = att_gt(
        data=mpdta_small,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        est_method="dr",
        control_group="nevertreated",
        boot=True,
        biters=100,
        cband=False,
        random_state=42,
    )

    py_agg_result = aggte(py_mp_result, type=agg_type, boot=True, biters=100, cband=False, random_state=42)

    np.testing.assert_allclose(
        py_agg_result.overall_att,
        r_result["overall_att"],
        rtol=1e-4,
        atol=1e-5,
        err_msg=f"{agg_type}: Overall ATT mismatch (bootstrap)",
    )

    assert not np.isnan(py_agg_result.overall_se), f"{agg_type}: Python returned NaN for overall SE"
    assert not np.isnan(r_result["overall_se"]), f"{agg_type}: R returned NaN for overall SE"
    se_ratio = py_agg_result.overall_se / r_result["overall_se"]
    assert 0.5 < se_ratio < 2.0, f"{agg_type}: Bootstrap SE ratio outside reasonable range: {se_ratio:.2f}"


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
def test_full_pipeline_consistency(mpdta_data, mpdta_csv_path):
    r_gt_result = r_att_gt(mpdta_csv_path, est_method="dr")
    r_agg_result = r_aggte(mpdta_csv_path, agg_type="simple", est_method="dr")

    if r_gt_result is None or r_agg_result is None:
        pytest.fail("R estimation failed")

    py_mp_result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        est_method="dr",
        control_group="nevertreated",
        boot=False,
    )

    py_agg_result = aggte(py_mp_result, type="simple")

    r_groups = np.array(r_gt_result["groups"])
    r_times = np.array(r_gt_result["times"])
    r_att = np.array(r_gt_result["att_gt"])

    matches = 0
    total = 0
    for i, (g, t) in enumerate(zip(py_mp_result.groups, py_mp_result.times)):
        r_mask = (r_groups == g) & (r_times == t)
        if np.any(r_mask):
            r_idx = np.where(r_mask)[0][0]
            py_att = py_mp_result.att_gt[i]
            r_att_val = r_att[r_idx]
            total += 1

            if np.isnan(py_att) and np.isnan(r_att_val):
                matches += 1
            elif not np.isnan(py_att) and not np.isnan(r_att_val):
                if np.allclose(py_att, r_att_val, rtol=1e-5, atol=1e-6):
                    matches += 1

    match_rate = matches / total if total > 0 else 0
    assert match_rate > 0.95, f"Pipeline: Only {match_rate:.1%} of ATT(g,t) estimates match"

    np.testing.assert_allclose(
        py_agg_result.overall_att,
        r_agg_result["overall_att"],
        rtol=1e-5,
        atol=1e-6,
        err_msg="Pipeline: Overall ATT mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
def test_repeated_cross_section(mpdta_data, mpdta_csv_path):
    r_result = r_att_gt(mpdta_csv_path, est_method="reg", panel=False)

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        gname="first.treat",
        xformla="~1",
        est_method="reg",
        control_group="nevertreated",
        panel=False,
        boot=False,
    )

    py_cells = list(zip(py_result.groups.tolist(), py_result.times.tolist()))
    assert py_cells == list(zip(r_result["groups"], r_result["times"]))
    assert py_result.n_units == r_result["n_units"][0]
    np.testing.assert_allclose(py_result.att_gt, r_result["att_gt"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(py_result.se_gt, r_result["se_gt"], rtol=1e-8, atol=1e-10)


def r_att_gt_repeated_cross_section(data_path, est_method="reg", allow_unbalanced_panel=False):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    r_script = f"""
library(did)
library(jsonlite)

data <- read.csv("{data_path}")

mp_result <- att_gt(
  yname = "lemp",
  tname = "year",
  idname = "countyreal",
  gname = "first.treat",
  xformla = ~1,
  data = data,
  est_method = "{est_method}",
  control_group = "nevertreated",
  panel = FALSE,
  allow_unbalanced_panel = {str(allow_unbalanced_panel).upper()},
  bstrap = FALSE,
  cband = FALSE
)

out <- list(
  groups = mp_result$group,
  times = mp_result$t,
  att_gt = mp_result$att,
  se_gt = mp_result$se,
  n_units = mp_result$n
)

for (agg_type in c("simple", "group", "dynamic")) {{
  agg_result <- aggte(mp_result, type = agg_type, bstrap = FALSE, cband = FALSE)
  out[[paste0("aggte_", agg_type)]] <- list(
    overall_att = agg_result$overall.att,
    overall_se = agg_result$overall.se,
    egt = agg_result$egt,
    att_egt = agg_result$att.egt,
    se_egt = agg_result$se.egt
  )
}}

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
@pytest.mark.filterwarnings("ignore:panel=False was specified:UserWarning")
@pytest.mark.parametrize(
    ("data_name", "est_method", "allow_unbalanced_panel"),
    [
        ("mpdta_data", "reg", False),
        ("mpdta_data", "dr", False),
        ("mpdta_data", "dr", True),
        ("mpdta_rotating", "reg", False),
    ],
)
def test_repeated_cross_section_with_idname(request, data_name, est_method, allow_unbalanced_panel):
    data = request.getfixturevalue(data_name)
    csv_path = request.getfixturevalue("mpdta_csv_path" if data_name == "mpdta_data" else f"{data_name}_csv_path")
    r_result = r_att_gt_repeated_cross_section(
        csv_path, est_method=est_method, allow_unbalanced_panel=allow_unbalanced_panel
    )

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = att_gt(
        data=data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        est_method=est_method,
        control_group="nevertreated",
        panel=False,
        allow_unbalanced_panel=allow_unbalanced_panel,
        boot=False,
        cband=False,
    )

    py_cells = list(zip(py_result.groups.tolist(), py_result.times.tolist()))
    r_cells = list(zip(r_result["groups"], r_result["times"]))
    assert set(py_cells) == set(r_cells)
    assert py_cells == r_cells
    assert py_result.n_units == r_result["n_units"][0] == data.height
    np.testing.assert_allclose(py_result.att_gt, r_result["att_gt"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(py_result.se_gt, r_result["se_gt"], rtol=1e-8, atol=1e-10)

    for agg_type in ("simple", "group", "dynamic"):
        py_agg = aggte(py_result, type=agg_type, cband=False)
        r_agg = r_result[f"aggte_{agg_type}"]
        np.testing.assert_allclose(py_agg.overall_att, r_agg["overall_att"], rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(py_agg.overall_se, r_agg["overall_se"], rtol=1e-8, atol=1e-10)
        if agg_type != "simple":
            np.testing.assert_array_equal(py_agg.event_times, r_agg["egt"])
            np.testing.assert_allclose(py_agg.att_by_event, r_agg["att_egt"], rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(py_agg.se_by_event, r_agg["se_egt"], rtol=1e-8, atol=1e-10)


def r_att_gt_clustered(data_path, est_method="dr", biters=100, random_state=42):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    r_script = f"""
library(did)
library(jsonlite)

set.seed({random_state})

data <- read.csv("{data_path}")

result <- att_gt(
  yname = "lemp",
  tname = "year",
  idname = "countyreal",
  gname = "first.treat",
  xformla = ~1,
  data = data,
  est_method = "{est_method}",
  control_group = "nevertreated",
  bstrap = TRUE,
  biters = {biters},
  cband = FALSE,
  clustervars = "cluster"
)

out <- list(
  groups = result$group,
  times = result$t,
  att_gt = result$att,
  se_gt = result$se,
  critical_value = result$c
)

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path, timeout=300)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_aggte_clustered(data_path, agg_type="simple", biters=100, random_state=42):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    r_script = f"""
library(did)
library(jsonlite)

set.seed({random_state})

data <- read.csv("{data_path}")

mp_result <- att_gt(
  yname = "lemp",
  tname = "year",
  idname = "countyreal",
  gname = "first.treat",
  xformla = ~1,
  data = data,
  est_method = "dr",
  control_group = "nevertreated",
  bstrap = TRUE,
  biters = {biters},
  cband = FALSE,
  clustervars = "cluster"
)

agg_result <- aggte(
  mp_result,
  type = "{agg_type}",
  bstrap = TRUE,
  biters = {biters},
  cband = FALSE
)

if ("{agg_type}" == "simple") {{
    out <- list(
        overall_att = agg_result$overall.att,
        overall_se = agg_result$overall.se
    )
}} else {{
    out <- list(
        overall_att = agg_result$overall.att,
        overall_se = agg_result$overall.se,
        egt = agg_result$egt,
        att_egt = agg_result$att.egt,
        se_egt = agg_result$se.egt
    )
}}

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path, timeout=300)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_att_gt_wald(data_path, est_method="dr", clustervars="NULL"):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    if clustervars == "NULL":
        clustervars_str = "NULL"
    else:
        clustervars_str = f'"{clustervars}"'

    bstrap_str = "TRUE" if clustervars != "NULL" else "FALSE"

    r_script = f"""
library(did)
library(jsonlite)

data <- read.csv("{data_path}")

result <- att_gt(
  yname = "lemp",
  tname = "year",
  idname = "countyreal",
  gname = "first.treat",
  xformla = ~1,
  data = data,
  est_method = "{est_method}",
  control_group = "nevertreated",
  bstrap = {bstrap_str},
  biters = 100,
  clustervars = {clustervars_str}
)

wald_stat <- result$Wpval
if (is.null(result$W)) {{
    W <- NA
}} else {{
    W <- as.numeric(result$W)
}}
if (is.null(result$Wpval)) {{
    Wpval <- NA
}} else {{
    Wpval <- as.numeric(result$Wpval)
}}

out <- list(
  wald_stat = W,
  wald_pvalue = Wpval
)

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_att_gt_unbalanced_clustered(data_path):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    r_script = f"""
library(did)
library(jsonlite)

data <- read.csv("{data_path}")

result <- att_gt(
  yname = "lemp",
  tname = "year",
  idname = "countyreal",
  gname = "first.treat",
  xformla = ~1,
  data = data,
  control_group = "nevertreated",
  allow_unbalanced_panel = TRUE,
  bstrap = FALSE,
  cband = FALSE,
  clustervars = "cluster"
)

out <- list(
  groups = result$group,
  times = result$t,
  att_gt = result$att,
  se_gt = result$se
)

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


@pytest.fixture(scope="module")
def mpdta_clustered(mpdta_data):
    return mpdta_data.with_columns((pl.col("countyreal") % 10).alias("cluster"))


@pytest.fixture(scope="module")
def mpdta_clustered_csv_path(mpdta_clustered):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        mpdta_clustered.write_csv(f.name)
        return f.name


@pytest.fixture(scope="module")
def mpdta_small_clustered(mpdta_small):
    return mpdta_small.with_columns((pl.col("countyreal") % 10).alias("cluster"))


@pytest.fixture(scope="module")
def mpdta_small_clustered_csv_path(mpdta_small_clustered):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        mpdta_small_clustered.write_csv(f.name)
        return f.name


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
def test_att_gt_clustered_att_unchanged(mpdta_clustered, mpdta_clustered_csv_path, mpdta_csv_path):
    r_unclustered = r_att_gt(mpdta_csv_path, est_method="dr")
    r_clustered = r_att_gt_clustered(mpdta_clustered_csv_path, est_method="dr", biters=100)

    if r_unclustered is None or r_clustered is None:
        pytest.fail("R estimation failed")

    py_result = att_gt(
        data=mpdta_clustered,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        est_method="dr",
        control_group="nevertreated",
        boot=True,
        biters=100,
        clustervars=["cluster"],
        random_state=42,
    )

    r_groups = np.array(r_clustered["groups"])
    r_times = np.array(r_clustered["times"])
    r_att = np.array(r_clustered["att_gt"])

    for i, (g, t) in enumerate(zip(py_result.groups, py_result.times)):
        r_mask = (r_groups == g) & (r_times == t)
        if np.any(r_mask):
            r_idx = np.where(r_mask)[0][0]
            py_att = py_result.att_gt[i]
            r_att_val = r_att[r_idx]

            if np.isnan(py_att) and np.isnan(r_att_val):
                continue
            if not np.isnan(py_att) and not np.isnan(r_att_val):
                np.testing.assert_allclose(
                    py_att,
                    r_att_val,
                    rtol=1e-10,
                    atol=1e-12,
                    err_msg=f"Clustered ATT mismatch at g={g}, t={t}",
                )


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
def test_att_gt_clustered_bootstrap_se(mpdta_small_clustered, mpdta_small_clustered_csv_path):
    r_result = r_att_gt_clustered(mpdta_small_clustered_csv_path, est_method="dr", biters=100)

    if r_result is None:
        pytest.fail("R clustered bootstrap estimation failed")

    py_result = att_gt(
        data=mpdta_small_clustered,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        est_method="dr",
        control_group="nevertreated",
        boot=True,
        biters=100,
        clustervars=["cluster"],
        random_state=42,
    )

    r_groups = np.array(r_result["groups"])
    r_times = np.array(r_result["times"])
    r_se = np.array(r_result["se_gt"])

    se_ratios = []
    for i, (g, t) in enumerate(zip(py_result.groups, py_result.times)):
        r_mask = (r_groups == g) & (r_times == t)
        if np.any(r_mask):
            r_idx = np.where(r_mask)[0][0]
            py_se = py_result.se_gt[i]
            r_se_val = r_se[r_idx]

            if not np.isnan(py_se) and not np.isnan(r_se_val) and r_se_val > 0:
                se_ratios.append(py_se / r_se_val)

    assert len(se_ratios) > 0, "No valid SE pairs to compare"
    mean_ratio = np.mean(se_ratios)
    assert 0.3 < mean_ratio < 1.7, f"Clustered bootstrap SE ratio outside reasonable range: {mean_ratio:.2f}"


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
@pytest.mark.parametrize("agg_type", ["simple", "dynamic", "group", "calendar"])
def test_aggte_clustered_overall_att(mpdta_clustered, mpdta_clustered_csv_path, agg_type):
    r_result = r_aggte_clustered(mpdta_clustered_csv_path, agg_type=agg_type, biters=100)

    if r_result is None:
        pytest.fail("R clustered aggregation failed")

    py_mp_result = att_gt(
        data=mpdta_clustered,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        est_method="dr",
        control_group="nevertreated",
        boot=True,
        biters=100,
        clustervars=["cluster"],
        random_state=42,
    )

    py_agg_result = aggte(
        py_mp_result,
        type=agg_type,
        boot=True,
        biters=100,
        clustervars=["cluster"],
        random_state=42,
    )

    r_overall_att = float(np.asarray(r_result["overall_att"]).flat[0])

    np.testing.assert_allclose(
        py_agg_result.overall_att,
        r_overall_att,
        rtol=1e-10,
        atol=1e-12,
        err_msg=f"{agg_type}: Clustered overall ATT mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
def test_clustering_changes_se(mpdta_small_clustered, mpdta_small_clustered_csv_path, mpdta_small_csv_path):
    r_unclustered = r_att_gt_bootstrap(mpdta_small_csv_path, est_method="dr", biters=100, cband=False)
    r_clustered = r_att_gt_clustered(mpdta_small_clustered_csv_path, est_method="dr", biters=100)

    if r_unclustered is None or r_clustered is None:
        pytest.fail("R estimation failed")

    py_unclustered = att_gt(
        data=mpdta_small_clustered,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        est_method="dr",
        control_group="nevertreated",
        boot=True,
        biters=100,
        random_state=42,
    )

    py_clustered = att_gt(
        data=mpdta_small_clustered,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        est_method="dr",
        control_group="nevertreated",
        boot=True,
        biters=100,
        clustervars=["cluster"],
        random_state=42,
    )

    r_se_uncl = np.array(r_unclustered["se_gt"])
    r_se_cl = np.array(r_clustered["se_gt"])
    valid_r = ~np.isnan(r_se_uncl) & ~np.isnan(r_se_cl) & (r_se_uncl > 0) & (r_se_cl > 0)
    assert not np.allclose(r_se_uncl[valid_r], r_se_cl[valid_r], rtol=0.01), "R: Clustering did not change SEs"

    valid_py = ~np.isnan(py_unclustered.se_gt) & ~np.isnan(py_clustered.se_gt)
    valid_py &= (py_unclustered.se_gt > 0) & (py_clustered.se_gt > 0)
    assert not np.allclose(py_unclustered.se_gt[valid_py], py_clustered.se_gt[valid_py], rtol=0.01), (
        "Python: Clustering did not change SEs"
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
@pytest.mark.filterwarnings("ignore:Clustering the standard errors requires using the bootstrap:UserWarning")
@pytest.mark.filterwarnings("ignore:The Wald pre-test is not reported:UserWarning")
def test_att_gt_unbalanced_cluster_sums_match_analytic_clustered_se(
    mpdta_unbalanced_clustered, mpdta_unbalanced_clustered_csv_path
):
    r_result = r_att_gt_unbalanced_clustered(mpdta_unbalanced_clustered_csv_path)

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = att_gt(
        data=mpdta_unbalanced_clustered,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        control_group="nevertreated",
        clustervars=["cluster"],
        allow_unbalanced_panel=True,
        boot=False,
        cband=False,
    )

    _, cluster_index = np.unique(py_result.estimation_params["cluster"], return_inverse=True)
    cluster_sums = np.zeros((cluster_index.max() + 1, py_result.influence_func.shape[1]))
    np.add.at(cluster_sums, cluster_index, py_result.influence_func)
    clustered_se = np.sqrt((cluster_sums**2).sum(axis=0)) / py_result.n_units

    assert list(zip(py_result.groups, py_result.times)) == list(zip(r_result["groups"], r_result["times"]))
    np.testing.assert_allclose(py_result.att_gt, r_result["att_gt"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(clustered_se, r_result["se_gt"], rtol=1e-9, atol=1e-12)


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
@pytest.mark.parametrize("est_method", ["dr", "reg"])
def test_wald_pretest_matches_r(mpdta_data, mpdta_csv_path, est_method):
    r_result = r_att_gt_wald(mpdta_csv_path, est_method=est_method)

    if r_result is None:
        pytest.fail("R Wald estimation failed")

    py_result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        est_method=est_method,
        control_group="nevertreated",
        boot=False,
    )

    r_wald = r_result["wald_stat"]
    r_pval = r_result["wald_pvalue"]

    r_wald_is_na = r_wald is None or (isinstance(r_wald, float) and np.isnan(r_wald))
    py_wald_is_na = py_result.wald_stat is None

    assert r_wald_is_na == py_wald_is_na, (
        f"{est_method}: Wald availability mismatch (R NA={r_wald_is_na}, Python None={py_wald_is_na})"
    )

    if not r_wald_is_na and not py_wald_is_na:
        np.testing.assert_allclose(
            py_result.wald_stat,
            r_wald,
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"{est_method}: Wald statistic mismatch",
        )
        np.testing.assert_allclose(
            py_result.wald_pvalue,
            r_pval,
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"{est_method}: Wald p-value mismatch",
        )


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
@pytest.mark.parametrize("agg_type", ["dynamic", "group", "calendar"])
def test_aggte_bootstrap_critical_value(mpdta_small, mpdta_small_csv_path, agg_type):
    r_result = r_aggte_bootstrap(mpdta_small_csv_path, agg_type=agg_type, biters=100, cband=True)

    if r_result is None:
        pytest.fail("R bootstrap aggregation failed")

    py_mp_result = att_gt(
        data=mpdta_small,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        est_method="dr",
        control_group="nevertreated",
        boot=True,
        biters=100,
        cband=True,
        random_state=42,
    )

    py_agg_result = aggte(
        py_mp_result,
        type=agg_type,
        boot=True,
        biters=100,
        cband=True,
        random_state=42,
    )

    r_cv_raw = r_result.get("critical_value")
    r_cv = float(np.asarray(r_cv_raw).flat[0]) if r_cv_raw is not None else np.nan
    if np.isfinite(r_cv) and r_cv > 0:
        py_cv_arr = py_agg_result.critical_values
        assert py_cv_arr is not None, f"{agg_type}: Python critical values should not be None"
        py_cv = float(py_cv_arr[0])
        assert py_cv > 0, f"{agg_type}: Python critical value should be positive"
        cv_ratio = py_cv / r_cv
        assert 0.5 < cv_ratio < 2.0, (
            f"{agg_type}: Bootstrap critical value ratio outside reasonable range: {cv_ratio:.2f}"
        )


def r_mboot(draws_path, n_units, alpha):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    r_script = f"""
library(did)
library(jsonlite)

raw <- as.matrix(read.csv("{draws_path}", header = FALSE))
dimnames(raw) <- NULL
assignInNamespace("run_multiplier_bootstrap", function(...) raw, ns = "did")

params <- list(
  idname = "id", clustervars = NULL, biters = nrow(raw), tname = "period", alp = {alpha},
  panel = TRUE, true_repeated_cross_sections = FALSE, allow_unbalanced_panel = FALSE,
  cluster_vector = NULL, faster_mode = TRUE
)
out <- mboot(matrix(0, nrow = {n_units}, ncol = ncol(raw)), params, return_V = FALSE)

write_json(
  list(crit_val = as.numeric(out$crit.val), se = as.numeric(out$se)),
  "{result_path}", digits = NA, na = "string", auto_unbox = TRUE
)
"""
    try:
        return _run_r_script(r_script, result_path)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("noise", [0.0, 1e-12])
def test_mboot_zero_and_tiny_scale_columns_match_reference(inf_func_with_zero_scale_column, tmp_path, noise):
    inf_func = inf_func_with_zero_scale_column
    inf_func[2:, 1] = noise * np.random.default_rng(1).standard_normal(len(inf_func) - 2)
    draws_path = tmp_path / "draws.csv"
    np.savetxt(draws_path, multiplier_bootstrap(inf_func, 999, 7), delimiter=",", fmt="%.17g")

    r_result = r_mboot(draws_path, len(inf_func), 0.05)

    if r_result is None:
        pytest.fail("R multiplier bootstrap failed")

    py_result = mboot(inf_func, n_units=len(inf_func), biters=999, alp=0.05, random_state=7)

    assert np.isfinite(py_result["crit_val"])
    np.testing.assert_allclose(py_result["crit_val"], r_result["crit_val"], rtol=1e-9)
    np.testing.assert_allclose(py_result["se"], r_result["se"], rtol=1e-9)


def r_att_gt_critical_value(data_path, draws_path, biters):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    r_script = f"""
library(did)
library(jsonlite)

raw <- as.matrix(read.csv("{draws_path}", header = FALSE))
dimnames(raw) <- NULL
assignInNamespace("run_multiplier_bootstrap", function(...) raw, ns = "did")

data <- read.csv("{data_path}")

result <- att_gt(
  yname = "y",
  tname = "t",
  idname = "id",
  gname = "g",
  xformla = ~1,
  data = data,
  est_method = "dr",
  control_group = "nevertreated",
  bstrap = TRUE,
  biters = {biters},
  cband = TRUE
)

write_json(list(critical_value = as.numeric(result$c)), "{result_path}", digits = NA, auto_unbox = TRUE)
"""
    try:
        return _run_r_script(r_script, result_path)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_att_gt_critical_value_with_offsetting_cohort_matches_reference(did_offsetting_cohort_data, tmp_path):
    data_path = tmp_path / "data.csv"
    did_offsetting_cohort_data.write_csv(data_path)
    kwargs = {
        "data": did_offsetting_cohort_data,
        "yname": "y",
        "tname": "t",
        "idname": "id",
        "gname": "g",
        "xformla": "~1",
        "est_method": "dr",
        "control_group": "nevertreated",
    }
    cells = att_gt(**kwargs)
    draws_path = tmp_path / "draws.csv"
    np.savetxt(draws_path, multiplier_bootstrap(cells.influence_func, 999, 7), delimiter=",", fmt="%.17g")

    r_result = r_att_gt_critical_value(data_path, draws_path, 999)

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = att_gt(**kwargs, boot=True, biters=999, cband=True, random_state=7)

    np.testing.assert_allclose(py_result.critical_value, r_result["critical_value"], rtol=1e-9)


def r_att_gt_cohort_coding(data_path, control_group, anticipation, base_period, est_method):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    r_script = f"""
library(did)
library(jsonlite)

data <- read.csv("{data_path}")

result <- suppressWarnings(att_gt(
  yname = "lemp",
  tname = "year",
  idname = "countyreal",
  gname = "first.treat",
  xformla = ~1,
  data = data,
  est_method = "{est_method}",
  control_group = "{control_group}",
  base_period = "{base_period}",
  anticipation = {anticipation},
  bstrap = FALSE,
  cband = FALSE
))
dynamic <- aggte(result, type = "dynamic", bstrap = FALSE, cband = FALSE, na.rm = TRUE)

out <- list(
  groups = result$group,
  times = result$t,
  att_gt = result$att,
  se_gt = result$se,
  n_units = result$n,
  wald = as.numeric(result$W),
  wald_pvalue = as.numeric(result$Wpval),
  event_times = dynamic$egt,
  att_by_event = dynamic$att.egt,
  se_by_event = dynamic$se.egt
)

write_json(out, "{result_path}", digits = 16, na = "null")
"""
    try:
        return _run_r_script(r_script, result_path)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def _assert_att_gt_matches_r(py_result, r_result):
    py_cells = list(zip(py_result.groups.tolist(), py_result.times.tolist()))
    assert py_cells == list(zip(r_result["groups"], r_result["times"]))
    assert py_result.n_units == r_result["n_units"][0]
    np.testing.assert_allclose(py_result.att_gt, np.asarray(r_result["att_gt"], dtype=float), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(py_result.se_gt, np.asarray(r_result["se_gt"], dtype=float), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(py_result.wald_stat, r_result["wald"][0], rtol=1e-8)
    assert py_result.wald_pvalue == r_result["wald_pvalue"][0]

    dynamic = aggte(py_result, type="dynamic", cband=False)
    np.testing.assert_array_equal(dynamic.event_times, r_result["event_times"])
    np.testing.assert_allclose(
        dynamic.att_by_event, np.asarray(r_result["att_by_event"], dtype=float), rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(
        dynamic.se_by_event, np.asarray(r_result["se_by_event"], dtype=float), rtol=1e-8, atol=1e-10
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
@pytest.mark.parametrize(
    ("est_method", "base_period"),
    [("reg", "varying"), ("dr", "varying"), ("ipw", "varying"), ("reg", "universal")],
)
def test_att_gt_without_never_treated_matches_r(
    mpdta_without_never_treated, mpdta_without_never_treated_csv_path, est_method, base_period
):
    r_result = r_att_gt_cohort_coding(mpdta_without_never_treated_csv_path, "notyettreated", 0, base_period, est_method)

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = att_gt(
        data=mpdta_without_never_treated,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        est_method=est_method,
        control_group="notyettreated",
        base_period=base_period,
        boot=False,
        cband=False,
    )

    assert set(py_result.groups.tolist()) == {2004.0, 2006.0}
    _assert_att_gt_matches_r(py_result, r_result)


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
@pytest.mark.filterwarnings("ignore:anticipation = :UserWarning")
@pytest.mark.filterwarnings("ignore:Dropped 20 units that were already treated:UserWarning")
@pytest.mark.parametrize(
    ("mpdta_cohort_after_panel", "anticipation", "control_group", "groups"),
    [
        (2008, 1, "nevertreated", {2006.0, 2008.0}),
        (2008, 1, "notyettreated", {2006.0, 2008.0}),
        (2009, 2, "nevertreated", {2006.0, 2009.0}),
        (2009, 1, "nevertreated", {2006.0}),
    ],
    indirect=["mpdta_cohort_after_panel"],
)
def test_att_gt_cohort_after_panel_matches_r(
    mpdta_cohort_after_panel, mpdta_cohort_after_panel_csv_path, anticipation, control_group, groups
):
    r_result = r_att_gt_cohort_coding(mpdta_cohort_after_panel_csv_path, control_group, anticipation, "varying", "reg")

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = att_gt(
        data=mpdta_cohort_after_panel,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        est_method="reg",
        control_group=control_group,
        anticipation=anticipation,
        boot=False,
        cband=False,
    )

    assert set(py_result.groups.tolist()) == groups
    _assert_att_gt_matches_r(py_result, r_result)


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
def test_att_gt_repeated_unit_periods_raise_like_r(mpdta_repeated_row, mpdta_repeated_row_csv_path):
    r_message = r_att_gt_error(mpdta_repeated_row_csv_path)

    assert r_message == (
        "The value of idname must be unique (by tname). Some units are observed more than once in a period."
    )
    with pytest.raises(ValueError, match=re.escape(f"{r_message} Rows repeat for the (countyreal, year) pair")):
        att_gt(
            data=mpdta_repeated_row,
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
            est_method="reg",
        )


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
@pytest.mark.filterwarnings("ignore:Dropped 2 rows from original data due to missing values:UserWarning")
def test_att_gt_rows_without_a_year_match_r(mpdta_without_years, mpdta_without_years_csv_path):
    r_result = r_att_gt(mpdta_without_years_csv_path, est_method="reg")

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = att_gt(
        data=mpdta_without_years,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        est_method="reg",
        boot=False,
    )

    assert list(zip(py_result.groups.tolist(), py_result.times.tolist())) == list(
        zip(r_result["groups"], r_result["times"])
    )
    assert py_result.n_units == r_result["n_units"][0]
    np.testing.assert_allclose(py_result.att_gt, np.asarray(r_result["att_gt"], dtype=float), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(py_result.se_gt, np.asarray(r_result["se_gt"], dtype=float), rtol=1e-8, atol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
@pytest.mark.filterwarnings("ignore:Dropped 1 rows from original data due to missing values:UserWarning")
@pytest.mark.parametrize(
    ("mpdta_one_infinite", "spec"),
    [
        (("lemp", float("inf")), {"est_method": "reg"}),
        (("lemp", float("-inf")), {"est_method": "dr"}),
        (("lpop", float("inf")), {"est_method": "dr", "xformla": "~lpop"}),
        (("pop", float("inf")), {"est_method": "reg", "weightsname": "pop"}),
        (("year", float("-inf")), {"est_method": "reg"}),
        (("countyreal", float("inf")), {"est_method": "reg"}),
        (("lemp", float("inf")), {"est_method": "reg", "panel": False}),
    ],
    indirect=["mpdta_one_infinite"],
    ids=["outcome", "outcome-dr", "covariate", "weights", "time", "unit", "outcome-rcs"],
)
def test_att_gt_drops_non_finite_rows_like_r(mpdta_one_infinite, mpdta_one_infinite_csv_path, spec):
    r_result = r_att_gt(mpdta_one_infinite_csv_path, **spec)

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = att_gt(
        data=mpdta_one_infinite,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        boot=False,
        **spec,
    )

    assert list(zip(py_result.groups.tolist(), py_result.times.tolist())) == list(
        zip(r_result["groups"], r_result["times"])
    )
    assert py_result.n_units == r_result["n_units"][0]
    np.testing.assert_allclose(py_result.att_gt, np.asarray(r_result["att_gt"], dtype=float), rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(py_result.se_gt, np.asarray(r_result["se_gt"], dtype=float), rtol=1e-9, atol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
@pytest.mark.filterwarnings("ignore:Dropped 1 rows from original data due to missing values:UserWarning")
@pytest.mark.filterwarnings("ignore:Dropped 1 units while converting to balanced panel:UserWarning")
@pytest.mark.parametrize(
    "mpdta_one_missing",
    [("first.treat", float("nan")), ("first.treat", None)],
    indirect=True,
    ids=["nan", "null"],
)
def test_att_gt_drops_a_row_without_a_cohort_before_checking_the_panel_like_r(
    mpdta_one_missing, mpdta_one_missing_csv_path
):
    r_result = r_att_gt_preprocessing_path(mpdta_one_missing_csv_path)

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = att_gt(
        data=mpdta_one_missing,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        est_method="reg",
        boot=False,
    )

    assert list(zip(py_result.groups.tolist(), py_result.times.tolist())) == list(
        zip(r_result["groups"], r_result["times"])
    )
    assert py_result.n_units == r_result["n_units"][0] == 499
    np.testing.assert_allclose(py_result.att_gt, np.asarray(r_result["att_gt"], dtype=float), rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(py_result.se_gt, np.asarray(r_result["se_gt"], dtype=float), rtol=1e-9, atol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
@pytest.mark.filterwarnings("ignore:Dropped 1 rows from original data due to missing values:UserWarning")
@pytest.mark.filterwarnings("ignore:Dropped 1 units while converting to balanced panel:UserWarning")
@pytest.mark.filterwarnings("ignore:Clustering the standard errors requires using the bootstrap:UserWarning")
@pytest.mark.filterwarnings("ignore:The Wald pre-test is not reported:UserWarning")
@pytest.mark.parametrize("mpdta_one_missing", [("cluster", float("nan"))], indirect=True)
def test_att_gt_drops_a_row_without_a_cluster_before_checking_the_panel_like_r(
    mpdta_one_missing, mpdta_one_missing_csv_path
):
    r_result = r_att_gt_preprocessing_path(mpdta_one_missing_csv_path, clustervars="cluster", faster_mode=False)

    if r_result is None:
        pytest.fail("R estimation failed")

    py_result = att_gt(
        data=mpdta_one_missing,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        clustervars=["cluster"],
        est_method="reg",
        boot=False,
    )

    assert list(zip(py_result.groups.tolist(), py_result.times.tolist())) == list(
        zip(r_result["groups"], r_result["times"])
    )
    assert py_result.n_units == r_result["n_units"][0] == 499
    np.testing.assert_allclose(py_result.att_gt, np.asarray(r_result["att_gt"], dtype=float), rtol=1e-9, atol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
@pytest.mark.parametrize("mpdta_bad_weights", ["zero", "zero outside an infinite row", "one negative"], indirect=True)
def test_att_gt_weights_without_positive_mean_raise_like_r(mpdta_bad_weights, mpdta_bad_weights_csv_path):
    r_message = r_att_gt_error(mpdta_bad_weights_csv_path, weightsname="w")

    assert r_message == "The weights variable 'w' must be non-negative with a positive mean."
    with pytest.raises(ValueError, match=f"^{re.escape(r_message)}$"):
        att_gt(
            data=mpdta_bad_weights,
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
            weightsname="w",
            est_method="reg",
        )
