"""Validation tests comparing Python did implementation with R did package."""

import json
import subprocess
import tempfile

import pytest

pytestmark = pytest.mark.slow

from tests.helpers import importorskip

pl = importorskip("polars")
np = importorskip("numpy")

from moderndid import aggte, att_gt, load_mpdta


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
):
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
  xformla = {xformla},
  data = data,
  est_method = "{est_method}",
  control_group = "{control_group}",
  base_period = "{base_period}",
  anticipation = {anticipation},
  panel = {str(panel).upper()},
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


def r_aggte(data_path, agg_type="simple", est_method="dr", balance_e=None, min_e=None, max_e=None, na_rm=False):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    balance_e_str = "NULL" if balance_e is None else str(balance_e)
    min_e_str = "-Inf" if min_e is None else str(min_e)
    max_e_str = "Inf" if max_e is None else str(max_e)
    na_rm_str = "TRUE" if na_rm else "FALSE"

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
        pytest.skip("R estimation failed")

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
        pytest.skip("R estimation failed")

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
        pytest.skip("R estimation failed")

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
        pytest.skip("R estimation failed")

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
        pytest.skip("R estimation failed")

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
        pytest.skip("R estimation failed")

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
        pytest.skip("R estimation failed")

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
def test_att_gt_bootstrap_se(mpdta_small, mpdta_small_csv_path):
    r_result = r_att_gt_bootstrap(mpdta_small_csv_path, est_method="dr", biters=100, cband=False)

    if r_result is None:
        pytest.skip("R bootstrap estimation failed")

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

    if len(se_ratios) > 0:
        mean_ratio = np.mean(se_ratios)
        assert 0.7 < mean_ratio < 1.3, f"Bootstrap SE ratio outside reasonable range: {mean_ratio:.2f}"


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
@pytest.mark.parametrize("agg_type", ["simple", "dynamic", "group", "calendar"])
def test_aggte_overall_att(mpdta_data, mpdta_csv_path, agg_type):
    r_result = r_aggte(mpdta_csv_path, agg_type=agg_type, est_method="reg")

    if r_result is None:
        pytest.skip("R aggregation failed")

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
        pytest.skip("R aggregation failed")

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
        pytest.skip("R aggregation failed")

    if "egt" not in r_result or r_result["egt"] is None:
        pytest.skip("R result missing disaggregated effects")

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

    for e in common_egt:
        py_idx = np.where(py_agg_result.event_times == e)[0][0]
        r_idx = np.where(r_egt == e)[0][0]

        py_att = py_agg_result.att_by_event[py_idx]
        r_att = r_att_egt[r_idx]

        if np.isnan(py_att) and np.isnan(r_att):
            continue
        if not np.isnan(py_att) and not np.isnan(r_att):
            np.testing.assert_allclose(
                py_att,
                r_att,
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"{agg_type} e={e}: ATT mismatch",
            )


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
def test_aggte_dynamic_with_balance_e(mpdta_data, mpdta_csv_path):
    r_result = r_aggte(mpdta_csv_path, agg_type="dynamic", est_method="reg", balance_e=1)

    if r_result is None:
        pytest.skip("R aggregation failed")

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
        pytest.skip("R aggregation failed")

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
        pytest.skip("R aggregation failed")

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
def test_aggte_bootstrap_se(mpdta_small, mpdta_small_csv_path, agg_type):
    r_result = r_aggte_bootstrap(mpdta_small_csv_path, agg_type=agg_type, biters=100, cband=False)

    if r_result is None:
        pytest.skip("R bootstrap aggregation failed")

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

    if not np.isnan(py_agg_result.overall_se) and not np.isnan(r_result["overall_se"]):
        se_ratio = py_agg_result.overall_se / r_result["overall_se"]
        assert 0.5 < se_ratio < 2.0, f"{agg_type}: Bootstrap SE ratio outside reasonable range: {se_ratio:.2f}"


@pytest.mark.skipif(not R_AVAILABLE, reason="R did package not available")
def test_full_pipeline_consistency(mpdta_data, mpdta_csv_path):
    r_gt_result = r_att_gt(mpdta_csv_path, est_method="dr")
    r_agg_result = r_aggte(mpdta_csv_path, agg_type="simple", est_method="dr")

    if r_gt_result is None or r_agg_result is None:
        pytest.skip("R estimation failed")

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
        pytest.skip("R estimation failed")

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
    assert match_rate > 0.90, f"RC mode: Only {match_rate:.1%} of ATT(g,t) estimates match"
