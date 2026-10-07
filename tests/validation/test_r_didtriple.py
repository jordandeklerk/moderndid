"""Validation tests comparing Python DDD implementation with R triplediff package."""

import json
import re
import subprocess
import tempfile
from pathlib import Path

import pytest
from polars.testing import assert_frame_equal

pytestmark = pytest.mark.slow

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid import (
    agg_ddd,
    ddd,
    ddd_mp,
    ddd_mp_rc,
    ddd_panel,
    ddd_rc,
    gen_ddd_2periods,
    gen_ddd_mult_periods,
)
from moderndid.core.preprocessing import preprocess_ddd_2periods

np = importorskip("numpy")
pd = importorskip("pandas")


def python_estimate_2period(data, est_method="dr"):
    ddd_data = preprocess_ddd_2periods(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
        xformla="~ cov1 + cov2 + cov3 + cov4",
        est_method=est_method,
    )

    covariates = np.column_stack([np.ones(ddd_data.n_units), ddd_data.covariates])

    return ddd_panel(
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
        covariates=covariates,
        i_weights=ddd_data.weights,
        est_method=est_method,
        boot=False,
        influence_func=True,
    )


def r_estimate_2period(data, est_method="dr"):
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "data.csv"
        result_path = Path(tmpdir) / "result.json"

        data.write_csv(data_path)

        r_script = f"""
library(triplediff)
library(jsonlite)

data <- read.csv("{data_path}")

result <- ddd(
    yname = "y",
    tname = "time",
    idname = "id",
    gname = "state",
    pname = "partition",
    xformla = ~ cov1 + cov2 + cov3 + cov4,
    data = data,
    est_method = "{est_method}",
    boot = FALSE,
    inffunc = TRUE
)

output <- list(
    att = result$ATT,
    se = result$se,
    lci = result$lci,
    uci = result$uci
)

write_json(output, "{result_path}", auto_unbox = TRUE)
"""
        try:
            return _run_r_script(r_script, result_path, timeout=60)
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
            return None


def r_estimate_multiperiod(data, control_group="nevertreated", base_period="universal", est_method="dr"):
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "data.csv"
        result_path = Path(tmpdir) / "result.json"

        data.write_csv(data_path)

        r_script = f"""
library(triplediff)
library(jsonlite)

data <- read.csv("{data_path}")

result <- ddd(
    yname = "y",
    tname = "time",
    idname = "id",
    gname = "group",
    pname = "partition",
    xformla = ~1,
    data = data,
    control_group = "{control_group}",
    base_period = "{base_period}",
    est_method = "{est_method}",
    boot = FALSE
)

output <- list(
    att = result$ATT,
    se = result$se,
    groups = result$groups,
    times = result$periods
)

write_json(output, "{result_path}", auto_unbox = TRUE)
"""
        try:
            return _run_r_script(r_script, result_path, timeout=120)
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
            return None


def r_estimate_multiperiod_agg(
    data, agg_type="eventstudy", boot=False, balance_e=None, min_e=None, max_e=None, alpha=0.05
):
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "data.csv"
        result_path = Path(tmpdir) / "result.json"

        data.write_csv(data_path)

        balance_e_str = "NULL" if balance_e is None else str(balance_e)
        min_e_str = "-Inf" if min_e is None else str(min_e)
        max_e_str = "Inf" if max_e is None else str(max_e)
        boot_str = "TRUE" if boot else "FALSE"
        ddd_boot_str = "TRUE" if boot else "FALSE"

        r_script = f"""
library(triplediff)
library(jsonlite)

set.seed(42)
data <- read.csv("{data_path}")

mp_result <- ddd(
    yname = "y",
    tname = "time",
    idname = "id",
    gname = "group",
    pname = "partition",
    xformla = ~1,
    data = data,
    control_group = "nevertreated",
    base_period = "universal",
    est_method = "reg",
    boot = {ddd_boot_str},
    nboot = 100
)

agg_result <- agg_ddd(
    mp_result,
    type = "{agg_type}",
    boot = {boot_str},
    nboot = 100,
    balance_e = {balance_e_str},
    min_e = {min_e_str},
    max_e = {max_e_str},
    alpha = {alpha}
)

agg_data <- agg_result$aggte_ddd

if ("{agg_type}" == "simple") {{
    output <- list(
        overall_att = as.numeric(agg_data$overall.att),
        overall_se = as.numeric(agg_data$overall.se),
        overall_lci = as.numeric(agg_data$overall.att - agg_data$crit.val * agg_data$overall.se),
        overall_uci = as.numeric(agg_data$overall.att + agg_data$crit.val * agg_data$overall.se),
        crit_val = as.numeric(agg_data$crit.val)
    )
}} else {{
    output <- list(
        overall_att = as.numeric(agg_data$overall.att),
        overall_se = as.numeric(agg_data$overall.se),
        overall_lci = as.numeric(agg_data$overall.att - agg_data$crit.val * agg_data$overall.se),
        overall_uci = as.numeric(agg_data$overall.att + agg_data$crit.val * agg_data$overall.se),
        crit_val = as.numeric(agg_data$crit.val),
        egt = agg_data$egt,
        att_egt = agg_data$att.egt,
        se_egt = agg_data$se.egt
    )
}}

write_json(output, "{result_path}", auto_unbox = TRUE)
"""
        try:
            return _run_r_script(r_script, result_path, timeout=120)
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
            return None


def r_estimate_2period_bootstrap(data, est_method="dr", biters=100):
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "data.csv"
        result_path = Path(tmpdir) / "result.json"

        data.write_csv(data_path)

        r_script = f"""
library(triplediff)
library(jsonlite)

set.seed(42)
data <- read.csv("{data_path}")

result <- ddd(
    yname = "y",
    tname = "time",
    idname = "id",
    gname = "state",
    pname = "partition",
    xformla = ~ cov1 + cov2 + cov3 + cov4,
    data = data,
    est_method = "{est_method}",
    boot = TRUE,
    nboot = {biters},
    inffunc = TRUE
)

output <- list(
    att = result$ATT,
    se = result$se,
    lci = result$lci,
    uci = result$uci
)

write_json(output, "{result_path}", auto_unbox = TRUE)
"""
        try:
            return _run_r_script(r_script, result_path, timeout=120)
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
            return None


def r_estimate_2period_clustered(data, panel=True, biters=99999):
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "data.csv"
        result_path = Path(tmpdir) / "result.json"

        data.write_csv(data_path)

        panel_str = "TRUE" if panel else "FALSE"

        r_script = f"""
library(triplediff)
library(jsonlite)

set.seed(42)
data <- read.csv("{data_path}")

result <- ddd(
    yname = "y",
    tname = "time",
    idname = "id",
    gname = "state",
    pname = "partition",
    xformla = ~ cov1 + cov2 + cov3 + cov4,
    data = data,
    est_method = "dr",
    panel = {panel_str},
    boot = TRUE,
    nboot = {biters},
    cluster = "cluster"
)

output <- list(
    att = result$ATT,
    se = result$se,
    lci = result$lci,
    uci = result$uci
)

write_json(output, "{result_path}", auto_unbox = TRUE, digits = NA)
"""
        try:
            return _run_r_script(r_script, result_path, timeout=120)
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
            return None


def r_estimate_multiperiod_clustered(data, biters=99999):
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "data.csv"
        result_path = Path(tmpdir) / "result.json"

        data.write_csv(data_path)

        r_script = f"""
library(triplediff)
library(jsonlite)

set.seed(42)
data <- read.csv("{data_path}")

result <- ddd(
    yname = "y",
    tname = "time",
    idname = "id",
    gname = "group",
    pname = "partition",
    xformla = ~1,
    data = data,
    control_group = "nevertreated",
    base_period = "universal",
    est_method = "dr",
    boot = TRUE,
    nboot = {biters},
    cluster = "cluster"
)

output <- list(
    att = result$ATT,
    se = result$se,
    groups = result$groups,
    times = result$periods
)

write_json(output, "{result_path}", auto_unbox = TRUE, digits = NA)
"""
        try:
            return _run_r_script(r_script, result_path, timeout=120)
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
            return None


def r_ddd_wrapper(data, is_multiperiod=False, est_method="dr", control_group="nevertreated", base_period="universal"):
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "data.csv"
        result_path = Path(tmpdir) / "result.json"

        data.write_csv(data_path)

        if is_multiperiod:
            r_script = f"""
library(triplediff)
library(jsonlite)

data <- read.csv("{data_path}")

result <- ddd(
    yname = "y",
    tname = "time",
    idname = "id",
    gname = "group",
    pname = "partition",
    xformla = ~1,
    data = data,
    control_group = "{control_group}",
    base_period = "{base_period}",
    est_method = "{est_method}",
    boot = FALSE
)

output <- list(
    att = result$ATT,
    se = result$se,
    groups = result$groups,
    times = result$periods,
    is_multiperiod = TRUE
)

write_json(output, "{result_path}", auto_unbox = TRUE)
"""
        else:
            r_script = f"""
library(triplediff)
library(jsonlite)

data <- read.csv("{data_path}")

result <- ddd(
    yname = "y",
    tname = "time",
    idname = "id",
    gname = "state",
    pname = "partition",
    xformla = ~ cov1 + cov2 + cov3 + cov4,
    data = data,
    est_method = "{est_method}",
    boot = FALSE
)

output <- list(
    att = result$ATT,
    se = result$se,
    lci = result$lci,
    uci = result$uci,
    is_multiperiod = FALSE
)

write_json(output, "{result_path}", auto_unbox = TRUE)
"""
        try:
            return _run_r_script(r_script, result_path, timeout=120)
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
            return None


def check_r_available():
    try:
        result = subprocess.run(
            ["R", "--vanilla", "--quiet"],
            input='library(triplediff); library(jsonlite); cat("OK")',
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return "OK" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


R_AVAILABLE = check_r_available()


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_2period_point_estimates_match(two_period_dgp_result, est_method):
    data, _, _ = two_period_dgp_result

    py_result = python_estimate_2period(data, est_method)
    r_result = r_estimate_2period(data, est_method)

    if r_result is None:
        pytest.fail("R estimation failed")

    np.testing.assert_allclose(
        py_result.att,
        r_result["att"],
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"{est_method}: ATT mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_2period_standard_errors_match(two_period_dgp_result, est_method):
    data, _, _ = two_period_dgp_result

    py_result = python_estimate_2period(data, est_method)
    r_result = r_estimate_2period(data, est_method)

    if r_result is None:
        pytest.fail("R estimation failed")

    np.testing.assert_allclose(
        py_result.se,
        r_result["se"],
        rtol=1e-2,
        atol=1e-3,
        err_msg=f"{est_method}: SE mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
def test_2period_confidence_intervals_match(two_period_dgp_result):
    data, _, _ = two_period_dgp_result

    py_result = python_estimate_2period(data, "dr")
    r_result = r_estimate_2period(data, "dr")

    if r_result is None:
        pytest.fail("R estimation failed")

    np.testing.assert_allclose(
        py_result.lci,
        r_result["lci"],
        rtol=1e-4,
        atol=1e-4,
        err_msg="LCI mismatch",
    )
    np.testing.assert_allclose(
        py_result.uci,
        r_result["uci"],
        rtol=1e-4,
        atol=1e-4,
        err_msg="UCI mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
@pytest.mark.parametrize("dgp_type", [1, 2, 3, 4])
def test_2period_dgp_types_match(dgp_type):
    result = gen_ddd_2periods(n=1000, dgp_type=dgp_type, random_state=42)
    data = result["data"]

    py_result = python_estimate_2period(data, "dr")
    r_result = r_estimate_2period(data, "dr")

    if r_result is None:
        pytest.fail("R estimation failed")

    np.testing.assert_allclose(
        py_result.att,
        r_result["att"],
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"DGP type {dgp_type}: ATT mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
def test_2period_bootstrap_se_reasonable(two_period_dgp_result):
    data, _, _ = two_period_dgp_result

    py_result = ddd(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
        xformla="~ cov1 + cov2 + cov3 + cov4",
        est_method="dr",
        boot=True,
        biters=100,
        random_state=42,
    )

    r_result = r_estimate_2period_bootstrap(data, "dr", biters=100)

    if r_result is None:
        pytest.fail("R bootstrap estimation failed")

    np.testing.assert_allclose(
        py_result.se,
        r_result["se"],
        rtol=0.2,
        atol=0.05,
        err_msg="Bootstrap SE mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
@pytest.mark.parametrize(
    "data_fixture, panel", [("two_period_clustered_data", True), ("two_period_rcs_clustered_data", False)]
)
def test_2period_clustered_bootstrap_matches(request, data_fixture, panel):
    data = request.getfixturevalue(data_fixture)

    py_result = ddd(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
        xformla="~ cov1 + cov2 + cov3 + cov4",
        est_method="dr",
        panel=panel,
        boot=True,
        biters=99999,
        cluster="cluster",
        random_state=42,
    )
    r_result = r_estimate_2period_clustered(data, panel=panel)

    if r_result is None:
        pytest.fail("R clustered bootstrap estimation failed")

    np.testing.assert_allclose(py_result.att, r_result["att"], rtol=1e-6)
    np.testing.assert_allclose(py_result.se, r_result["se"], rtol=0.06)
    np.testing.assert_allclose(py_result.uci - py_result.lci, r_result["uci"] - r_result["lci"], rtol=0.06)


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_mp_att_gt_estimates_match(mp_ddd_data, est_method):
    data = mp_ddd_data

    py_result = ddd_mp(
        data=data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        est_method=est_method,
    )

    r_result = r_estimate_multiperiod(data, est_method=est_method)

    if r_result is None:
        pytest.fail("R estimation failed")

    r_att = np.atleast_1d(r_result["att"])
    r_groups = np.atleast_1d(r_result["groups"])
    r_times = np.atleast_1d(r_result["times"])

    assert len(r_att) == len(r_groups) == len(r_times), (
        f"{est_method}: R returned mismatched lengths: att={len(r_att)}, groups={len(r_groups)}, times={len(r_times)}"
    )

    matches = 0
    for i, (g, t) in enumerate(zip(py_result.groups, py_result.times)):
        r_mask = (r_groups == g) & (r_times == t)
        if np.any(r_mask):
            r_idx = np.where(r_mask)[0][0]
            py_att = py_result.att[i]
            r_att_val = r_att[r_idx]

            if np.isnan(py_att) and np.isnan(r_att_val):
                matches += 1
            elif not np.isnan(py_att) and not np.isnan(r_att_val):
                if np.allclose(py_att, r_att_val, rtol=1e-4, atol=1e-4):
                    matches += 1

    match_rate = matches / len(py_result.att) if len(py_result.att) > 0 else 0
    assert match_rate > 0.95, f"{est_method}: Only {match_rate:.1%} of ATT(g,t) estimates match"


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
@pytest.mark.parametrize("control_group", ["nevertreated", "notyettreated"])
def test_mp_control_group_options(mp_ddd_data, control_group):
    data = mp_ddd_data

    py_result = ddd_mp(
        data=data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        control_group=control_group,
        est_method="reg",
    )

    r_result = r_estimate_multiperiod(data, control_group=control_group, est_method="reg")

    if r_result is None:
        pytest.fail("R estimation failed")

    r_att = np.atleast_1d(r_result["att"])
    assert len(py_result.att) > 0, f"Python returned no ATTs for {control_group}"
    assert len(r_att) > 0, f"R returned no ATTs for {control_group}"

    matches = 0
    r_groups = np.atleast_1d(r_result["groups"])
    r_times = np.atleast_1d(r_result["times"])
    for i, (g, t) in enumerate(zip(py_result.groups, py_result.times)):
        r_mask = (r_groups == g) & (r_times == t)
        if np.any(r_mask):
            r_idx = np.where(r_mask)[0][0]
            py_att = py_result.att[i]
            r_att_val = r_att[r_idx]
            if np.isnan(py_att) and np.isnan(r_att_val):
                matches += 1
            elif not np.isnan(py_att) and not np.isnan(r_att_val):
                if np.allclose(py_att, r_att_val, rtol=1e-4, atol=1e-4):
                    matches += 1
    match_rate = matches / len(py_result.att) if len(py_result.att) > 0 else 0
    assert match_rate > 0.95, f"{control_group}: Only {match_rate:.1%} of ATT(g,t) estimates match"


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
@pytest.mark.parametrize("base_period", ["universal", "varying"])
def test_mp_base_period_options(mp_ddd_data, base_period):
    data = mp_ddd_data

    py_result = ddd_mp(
        data=data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        base_period=base_period,
        est_method="reg",
    )

    r_result = r_estimate_multiperiod(data, base_period=base_period, est_method="reg")

    if r_result is None:
        pytest.fail("R estimation failed")

    r_att = np.atleast_1d(r_result["att"])
    assert len(py_result.att) > 0, f"Python returned no ATTs for {base_period}"
    assert len(r_att) > 0, f"R returned no ATTs for {base_period}"

    matches = 0
    r_groups = np.atleast_1d(r_result["groups"])
    r_times = np.atleast_1d(r_result["times"])
    for i, (g, t) in enumerate(zip(py_result.groups, py_result.times)):
        r_mask = (r_groups == g) & (r_times == t)
        if np.any(r_mask):
            r_idx = np.where(r_mask)[0][0]
            py_att = py_result.att[i]
            r_att_val = r_att[r_idx]
            if np.isnan(py_att) and np.isnan(r_att_val):
                matches += 1
            elif not np.isnan(py_att) and not np.isnan(r_att_val):
                if np.allclose(py_att, r_att_val, rtol=1e-4, atol=1e-4):
                    matches += 1
    match_rate = matches / len(py_result.att) if len(py_result.att) > 0 else 0
    assert match_rate > 0.95, f"{base_period}: Only {match_rate:.1%} of ATT(g,t) estimates match"


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
def test_mp_clustered_bootstrap_matches(mp_ddd_clustered_data):
    py_result = ddd(
        data=mp_ddd_clustered_data,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        control_group="nevertreated",
        base_period="universal",
        est_method="dr",
        boot=True,
        biters=99999,
        cluster="cluster",
        random_state=42,
    )
    r_result = r_estimate_multiperiod_clustered(mp_ddd_clustered_data)

    if r_result is None:
        pytest.fail("R clustered bootstrap estimation failed")

    np.testing.assert_array_equal(py_result.groups, np.atleast_1d(r_result["groups"]))
    np.testing.assert_array_equal(py_result.times, np.atleast_1d(r_result["times"]))
    np.testing.assert_allclose(py_result.att, _convert_r_array(r_result["att"]), rtol=0, atol=1e-10)
    np.testing.assert_allclose(py_result.se, _convert_r_array(r_result["se"]), rtol=0.06)


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
def test_ddd_wrapper_2period(two_period_dgp_result):
    data, _, _ = two_period_dgp_result

    py_result = ddd(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
        xformla="~ cov1 + cov2 + cov3 + cov4",
        est_method="dr",
    )

    r_result = r_ddd_wrapper(data, is_multiperiod=False, est_method="dr")

    if r_result is None:
        pytest.fail("R estimation failed")

    np.testing.assert_allclose(
        py_result.att,
        r_result["att"],
        rtol=1e-4,
        atol=1e-4,
        err_msg="2-period wrapper ATT mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
def test_ddd_wrapper_multiperiod(mp_ddd_data):
    data = mp_ddd_data

    py_result = ddd(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method="reg",
    )

    r_result = r_ddd_wrapper(data, is_multiperiod=True, est_method="reg")

    if r_result is None:
        pytest.fail("R estimation failed")

    assert len(py_result.att) > 0, "Python wrapper returned no ATTs"
    assert len(r_result["att"]) > 0, "R wrapper returned no ATTs"


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
@pytest.mark.parametrize("agg_type", ["simple", "eventstudy", "group", "calendar"])
def test_agg_overall_att_matches(mp_ddd_data, agg_type):
    data = mp_ddd_data

    py_mp_result = ddd_mp(
        data=data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        est_method="reg",
    )
    py_agg = agg_ddd(py_mp_result, type=agg_type, boot=False, cband=False)

    r_result = r_estimate_multiperiod_agg(data, agg_type)

    if r_result is None:
        pytest.fail("R aggregation failed")

    np.testing.assert_allclose(
        py_agg.overall_att,
        r_result["overall_att"],
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"{agg_type}: Overall ATT mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
@pytest.mark.parametrize("agg_type", ["simple", "eventstudy", "group", "calendar"])
def test_agg_overall_se_matches(mp_ddd_data, agg_type):
    data = mp_ddd_data

    py_mp_result = ddd_mp(
        data=data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        est_method="reg",
    )
    py_agg = agg_ddd(py_mp_result, type=agg_type, boot=False, cband=False)

    r_result = r_estimate_multiperiod_agg(data, agg_type)

    if r_result is None:
        pytest.fail("R aggregation failed")

    np.testing.assert_allclose(
        py_agg.overall_se,
        r_result["overall_se"],
        rtol=0.05,
        atol=1e-2,
        err_msg=f"{agg_type}: Overall SE mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
@pytest.mark.parametrize("agg_type", ["eventstudy", "group", "calendar"])
def test_agg_disaggregated_effects_match(mp_ddd_data, agg_type):
    data = mp_ddd_data

    py_mp_result = ddd_mp(
        data=data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        est_method="reg",
    )
    py_agg = agg_ddd(py_mp_result, type=agg_type, boot=False, cband=False)

    r_result = r_estimate_multiperiod_agg(data, agg_type)

    if r_result is None:
        pytest.fail("R aggregation failed")

    if "egt" not in r_result or r_result["egt"] is None:
        pytest.fail("R result missing disaggregated effects")

    r_egt = np.array(r_result["egt"])
    r_att_egt = np.array(r_result["att_egt"])

    assert len(py_agg.egt) > 0, f"{agg_type}: Python egt is empty"
    assert len(r_egt) > 0, f"{agg_type}: R egt is empty"

    common_egt = set(py_agg.egt) & set(r_egt)
    assert len(common_egt) > 0, f"{agg_type}: No common event times between Python and R"

    for e in common_egt:
        py_idx = np.where(py_agg.egt == e)[0][0]
        r_idx = np.where(r_egt == e)[0][0]

        py_att = py_agg.att_egt[py_idx]
        r_att = r_att_egt[r_idx]

        np.testing.assert_allclose(
            py_att,
            r_att,
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"{agg_type} e={e}: ATT mismatch",
        )


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
@pytest.mark.parametrize("agg_type", ["simple", "eventstudy", "group", "calendar"])
def test_agg_with_bootstrap_matches(mp_ddd_data, agg_type):
    data = mp_ddd_data

    py_mp_result = ddd_mp(
        data=data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        est_method="reg",
    )
    py_agg = agg_ddd(
        py_mp_result,
        type=agg_type,
        boot=True,
        biters=100,
        cband=True,
        random_state=42,
    )

    r_result = r_estimate_multiperiod_agg(data, agg_type, boot=True)

    if r_result is None:
        pytest.fail("R aggregation with bootstrap failed")

    np.testing.assert_allclose(
        py_agg.overall_att,
        r_result["overall_att"],
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"{agg_type} boot: Overall ATT mismatch",
    )

    np.testing.assert_allclose(
        py_agg.overall_se,
        r_result["overall_se"],
        rtol=0.2,
        atol=0.05,
        err_msg=f"{agg_type} boot: Overall SE mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
def test_eventstudy_balance_e_matches(mp_ddd_data):
    data = mp_ddd_data

    py_mp_result = ddd_mp(
        data=data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        est_method="reg",
    )
    py_agg = agg_ddd(
        py_mp_result,
        type="eventstudy",
        balance_e=1,
        boot=False,
        cband=False,
    )

    r_result = r_estimate_multiperiod_agg(data, "eventstudy", balance_e=1)

    if r_result is None:
        pytest.fail("R aggregation with balance_e failed")

    np.testing.assert_allclose(
        py_agg.overall_att,
        r_result["overall_att"],
        rtol=1e-4,
        atol=1e-4,
        err_msg="balance_e=1: Overall ATT mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
def test_eventstudy_min_max_e_matches(mp_ddd_data):
    data = mp_ddd_data

    py_mp_result = ddd_mp(
        data=data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        est_method="reg",
    )
    py_agg = agg_ddd(
        py_mp_result,
        type="eventstudy",
        min_e=-1,
        max_e=2,
        boot=False,
        cband=False,
    )

    r_result = r_estimate_multiperiod_agg(data, "eventstudy", min_e=-1, max_e=2)

    if r_result is None:
        pytest.fail("R aggregation with min_e/max_e failed")

    assert "egt" in r_result and r_result["egt"] is not None, "R result missing egt field"
    r_egt = np.array(r_result["egt"])
    assert all(-1 <= e <= 2 for e in py_agg.egt), "Python egt outside [min_e, max_e]"
    assert all(-1 <= e <= 2 for e in r_egt), "R egt outside [min_e, max_e]"
    np.testing.assert_array_equal(np.sort(py_agg.egt), np.sort(r_egt), err_msg="Event times mismatch")


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
@pytest.mark.parametrize("alpha", [0.01, 0.10])
def test_agg_alpha_levels_match(mp_ddd_data, alpha):
    data = mp_ddd_data

    py_mp_result = ddd_mp(
        data=data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        est_method="reg",
    )
    py_agg = agg_ddd(
        py_mp_result,
        type="simple",
        boot=False,
        cband=False,
        alpha=alpha,
    )

    r_result = r_estimate_multiperiod_agg(data, "simple", alpha=alpha)

    if r_result is None:
        pytest.fail(f"R aggregation with alpha={alpha} failed")

    np.testing.assert_allclose(
        py_agg.overall_att,
        r_result["overall_att"],
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"alpha={alpha}: Overall ATT mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
def test_group_agg_all_groups_match(mp_ddd_data):
    data = mp_ddd_data

    py_mp_result = ddd_mp(
        data=data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        est_method="reg",
    )
    py_agg = agg_ddd(
        py_mp_result,
        type="group",
        boot=False,
        cband=False,
    )

    r_result = r_estimate_multiperiod_agg(data, "group")

    if r_result is None:
        pytest.fail("R group aggregation failed")

    if "egt" not in r_result or r_result["egt"] is None:
        pytest.fail("R result missing group effects")

    r_groups = np.array(r_result["egt"])
    r_att = np.array(r_result["att_egt"])

    py_groups = py_agg.egt

    common_groups = set(py_groups) & set(r_groups)
    assert len(common_groups) > 0, "No common groups between Python and R"

    for g in common_groups:
        py_idx = np.where(py_groups == g)[0][0]
        r_idx = np.where(r_groups == g)[0][0]

        py_group_att = py_agg.att_egt[py_idx]
        r_group_att = r_att[r_idx]

        np.testing.assert_allclose(
            py_group_att,
            r_group_att,
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"Group {g}: ATT mismatch",
        )


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
def test_calendar_agg_all_times_match(mp_ddd_data):
    data = mp_ddd_data

    py_mp_result = ddd_mp(
        data=data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        est_method="reg",
    )
    py_agg = agg_ddd(
        py_mp_result,
        type="calendar",
        boot=False,
        cband=False,
    )

    r_result = r_estimate_multiperiod_agg(data, "calendar")

    if r_result is None:
        pytest.fail("R calendar aggregation failed")

    if "egt" not in r_result or r_result["egt"] is None:
        pytest.fail("R result missing calendar effects")

    r_times = np.array(r_result["egt"])
    r_att = np.array(r_result["att_egt"])

    py_times = py_agg.egt

    common_times = set(py_times) & set(r_times)
    assert len(common_times) > 0, "No common calendar times between Python and R"

    for t in common_times:
        py_idx = np.where(py_times == t)[0][0]
        r_idx = np.where(r_times == t)[0][0]

        py_time_att = py_agg.att_egt[py_idx]
        r_time_att = r_att[r_idx]

        np.testing.assert_allclose(
            py_time_att,
            r_time_att,
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"Calendar time {t}: ATT mismatch",
        )


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
@pytest.mark.parametrize("agg_type", ["eventstudy", "group", "calendar"])
def test_agg_se_egt_matches(mp_ddd_data, agg_type):
    data = mp_ddd_data

    py_mp_result = ddd_mp(
        data=data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        est_method="reg",
    )
    py_agg = agg_ddd(py_mp_result, type=agg_type, boot=False, cband=False)

    r_result = r_estimate_multiperiod_agg(data, agg_type)

    if r_result is None:
        pytest.fail("R aggregation failed")

    if "se_egt" not in r_result or r_result["se_egt"] is None:
        pytest.fail("R result missing se_egt")

    r_egt = _convert_r_array(r_result["egt"])
    r_se_egt = _convert_r_array(r_result["se_egt"])

    common_egt = set(py_agg.egt[~np.isnan(py_agg.egt)]) & set(r_egt[~np.isnan(r_egt)])

    for e in common_egt:
        py_idx = np.where(py_agg.egt == e)[0][0]
        r_idx = np.where(r_egt == e)[0][0]

        py_se = float(py_agg.se_egt[py_idx])
        r_se = float(r_se_egt[r_idx])

        if np.isnan(py_se) and np.isnan(r_se):
            continue
        if np.isnan(py_se) or np.isnan(r_se):
            continue

        np.testing.assert_allclose(
            py_se,
            r_se,
            rtol=0.05,
            atol=1e-2,
            err_msg=f"{agg_type} e={e}: SE mismatch",
        )


def test_dgp_produces_valid_structure():
    py_result = gen_ddd_2periods(n=5000, dgp_type=1, random_state=42)
    py_data = py_result["data"]

    required_cols = ["id", "time", "y", "state", "partition", "cov1", "cov2", "cov3", "cov4"]
    for col in required_cols:
        assert col in py_data.columns, f"Missing column: {col}"

    assert len(py_data) == 10000, f"Expected 10000 rows, got {len(py_data)}"


def test_dgp_subgroup_proportions_reasonable():
    py_result = gen_ddd_2periods(n=5000, dgp_type=1, random_state=42)
    py_data = py_result["data"]

    units = py_data.unique(subset=["id"])
    subgroup_counts = units.group_by(["state", "partition"]).len()

    min_count = subgroup_counts["len"].min()
    assert min_count > 200, f"Subgroup too small: {min_count}"


@pytest.mark.parametrize("dgp_type", [1, 2, 3, 4])
def test_dgp_types_work(dgp_type):
    result = gen_ddd_2periods(n=1000, dgp_type=dgp_type, random_state=42)
    data = result["data"]

    assert len(data) == 2000, f"Expected 2000 rows, got {len(data)}"
    assert result["true_att"] == 0.0, f"Expected true ATT=0, got {result['true_att']}"
    assert "efficiency_bound" in result, "Missing efficiency_bound"


def test_mp_dgp_produces_valid_structure():
    result = gen_ddd_mult_periods(n=500, dgp_type=1, random_state=42)
    data = result["data"]

    required_cols = ["id", "time", "y", "group", "partition"]
    for col in required_cols:
        assert col in data.columns, f"Missing column: {col}"

    n_obs = len(data)
    n_units = data["id"].n_unique()
    n_times = data["time"].n_unique()
    assert n_obs == n_units * n_times, f"Unbalanced panel: {n_obs} != {n_units} * {n_times}"


def test_mp_dgp_group_structure():
    result = gen_ddd_mult_periods(n=500, dgp_type=1, random_state=42)
    data = result["data"]

    groups = data["group"].unique()
    assert 0 in groups, "Missing never-treated group (0)"
    assert len(groups) >= 2, "Need at least 2 groups"


def test_mp_dgp_partition_binary():
    result = gen_ddd_mult_periods(n=500, dgp_type=1, random_state=42)
    data = result["data"]

    assert set(data["partition"].unique()) == {0, 1}, "Partition should be binary {0, 1}"


@pytest.mark.parametrize("dgp_type", [1, 2, 3, 4])
def test_mp_dgp_types_work(dgp_type):
    result = gen_ddd_mult_periods(n=500, dgp_type=dgp_type, random_state=42)
    data = result["data"]

    assert len(data) > 0, f"DGP type {dgp_type} produced empty data"
    assert "data_wide" in result, "Missing data_wide"


def test_mp_dgp_wide_format():
    result = gen_ddd_mult_periods(n=500, dgp_type=1, random_state=42)
    data_wide = result["data_wide"]

    assert "id" in data_wide.columns, "Missing id in wide format"
    assert "group" in data_wide.columns, "Missing group in wide format"
    assert "partition" in data_wide.columns, "Missing partition in wide format"

    y_cols = [c for c in data_wide.columns if c.startswith("y_")]
    assert len(y_cols) > 0, "Missing outcome columns in wide format"


def test_mp_dgp_reproducibility():
    result1 = gen_ddd_mult_periods(n=500, dgp_type=1, random_state=42)
    result2 = gen_ddd_mult_periods(n=500, dgp_type=1, random_state=42)

    assert_frame_equal(result1["data"], result2["data"])


def test_did_components_sum_correctly(two_period_dgp_result):
    data, _, _ = two_period_dgp_result

    py_result = python_estimate_2period(data, "dr")

    computed_ddd = py_result.did_atts["att_4v3"] + py_result.did_atts["att_4v2"] - py_result.did_atts["att_4v1"]
    np.testing.assert_almost_equal(py_result.att, computed_ddd, decimal=10, err_msg="DDD formula mismatch")


def test_subgroup_counts_reasonable(two_period_dgp_result):
    data, _, _ = two_period_dgp_result

    py_result = python_estimate_2period(data, "dr")

    for sg, count in py_result.subgroup_counts.items():
        assert count >= 50, f"Subgroup {sg} too small: {count}"


def test_influence_function_properties(two_period_dgp_result):
    data, _, _ = two_period_dgp_result

    py_result = python_estimate_2period(data, "dr")

    assert py_result.att_inf_func is not None, "Influence function is None"

    inf_func = py_result.att_inf_func
    se_from_if = np.sqrt(np.var(inf_func) / len(inf_func))

    np.testing.assert_almost_equal(py_result.se, se_from_if, decimal=4, err_msg="SE from IF doesn't match reported SE")


@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_estimation_methods_valid(two_period_dgp_result, est_method):
    data, _, _ = two_period_dgp_result

    py_result = python_estimate_2period(data, est_method)

    assert hasattr(py_result, "att"), "Missing att attribute"
    assert hasattr(py_result, "se"), "Missing se attribute"
    assert hasattr(py_result, "lci"), "Missing lci attribute"
    assert hasattr(py_result, "uci"), "Missing uci attribute"

    assert py_result.se > 0, f"SE must be positive, got {py_result.se}"
    assert py_result.lci < py_result.att < py_result.uci, "ATT not within CI"


@pytest.mark.parametrize("agg_type", ["simple", "eventstudy", "group", "calendar"])
def test_aggregation_produces_valid_output(mp_ddd_result, agg_type):
    result = agg_ddd(mp_ddd_result, type=agg_type, boot=False, cband=False)

    assert hasattr(result, "overall_att"), "Missing overall_att"
    assert hasattr(result, "overall_se"), "Missing overall_se"
    assert hasattr(result, "aggregation_type"), "Missing aggregation_type"

    assert result.aggregation_type == agg_type, f"Wrong agg type: {result.aggregation_type}"
    assert isinstance(result.overall_att, float | np.floating), "overall_att not float"
    assert isinstance(result.overall_se, float | np.floating), "overall_se not float"


@pytest.mark.parametrize("agg_type", ["eventstudy", "group", "calendar"])
def test_disaggregated_effects_structure(mp_ddd_result, agg_type):
    result = agg_ddd(mp_ddd_result, type=agg_type, boot=False, cband=False)

    assert result.egt is not None, "egt is None"
    assert result.att_egt is not None, "att_egt is None"
    assert result.se_egt is not None, "se_egt is None"

    assert len(result.egt) == len(result.att_egt), "egt and att_egt length mismatch"
    assert len(result.egt) == len(result.se_egt), "egt and se_egt length mismatch"


def test_simple_has_no_disaggregated(mp_ddd_result):
    result = agg_ddd(mp_ddd_result, type="simple", boot=False, cband=False)

    assert result.egt is None, "simple should have egt=None"
    assert result.att_egt is None, "simple should have att_egt=None"
    assert result.se_egt is None, "simple should have se_egt=None"


def test_influence_function_overall(mp_ddd_result):
    result = agg_ddd(mp_ddd_result, type="simple", boot=False, cband=False)

    assert result.inf_func_overall is not None, "inf_func_overall is None"
    assert isinstance(result.inf_func_overall, np.ndarray), "inf_func_overall not ndarray"
    assert len(result.inf_func_overall) == mp_ddd_result.n, "inf_func_overall wrong length"


def test_att_gt_structure(mp_ddd_data):
    data = mp_ddd_data

    result = ddd_mp(
        data=data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        est_method="reg",
    )

    assert len(result.att) == len(result.se), "ATT and SE length mismatch"
    assert len(result.att) == len(result.groups), "ATT and groups length mismatch"
    assert len(result.att) == len(result.times), "ATT and times length mismatch"


def test_glist_tlist_consistency(mp_ddd_data):
    data = mp_ddd_data

    result = ddd_mp(
        data=data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        est_method="reg",
    )

    assert all(g in result.glist for g in np.unique(result.groups)), "groups not in glist"
    assert all(t in result.tlist for t in np.unique(result.times)), "times not in tlist"


def test_inf_func_mat_shape(mp_ddd_data):
    data = mp_ddd_data

    result = ddd_mp(
        data=data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        est_method="reg",
    )

    assert result.inf_func_mat.shape[0] == result.n, "inf_func_mat rows != n"
    assert result.inf_func_mat.shape[1] == len(result.att), "inf_func_mat cols != len(att)"


@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_mp_all_methods_work(mp_ddd_data, est_method):
    data = mp_ddd_data

    result = ddd_mp(
        data=data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        est_method=est_method,
    )

    assert len(result.att) > 0, f"{est_method} produced no ATTs"
    valid_atts = result.att[~np.isnan(result.att)]
    assert len(valid_atts) > 0, f"{est_method} produced all NaN ATTs"


def _convert_r_array(arr):
    result = []
    for val in arr:
        if val == "NA" or val is None:
            result.append(np.nan)
        else:
            result.append(float(val))
    return np.array(result)


def r_estimate_2period_rcs(data, est_method="dr"):
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "data.csv"
        result_path = Path(tmpdir) / "result.json"

        data.write_csv(data_path)

        r_script = f"""
library(triplediff)
library(jsonlite)

data <- read.csv("{data_path}")

result <- ddd(
    yname = "y",
    tname = "time",
    idname = "id",
    gname = "state",
    pname = "partition",
    xformla = ~ cov1 + cov2 + cov3 + cov4,
    data = data,
    est_method = "{est_method}",
    panel = FALSE,
    boot = FALSE,
    inffunc = TRUE
)

output <- list(
    att = result$ATT,
    se = result$se,
    lci = result$lci,
    uci = result$uci
)

write_json(output, "{result_path}", auto_unbox = TRUE)
"""
        try:
            return _run_r_script(r_script, result_path, timeout=60)
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
            return None


def r_estimate_multiperiod_rcs(data, control_group="nevertreated", base_period="universal", est_method="dr"):
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "data.csv"
        result_path = Path(tmpdir) / "result.json"

        data.write_csv(data_path)

        r_script = f"""
library(triplediff)
library(jsonlite)

data <- read.csv("{data_path}")

result <- ddd(
    yname = "y",
    tname = "time",
    idname = "id",
    gname = "group",
    pname = "partition",
    xformla = ~1,
    data = data,
    control_group = "{control_group}",
    base_period = "{base_period}",
    est_method = "{est_method}",
    panel = FALSE,
    boot = FALSE
)

output <- list(
    att = result$ATT,
    se = result$se,
    groups = result$groups,
    times = result$periods
)

write_json(output, "{result_path}", auto_unbox = TRUE)
"""
        try:
            return _run_r_script(r_script, result_path, timeout=120)
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
            return None


def python_estimate_2period_rcs(data, est_method="dr"):
    return ddd(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
        xformla="~ cov1 + cov2 + cov3 + cov4",
        est_method=est_method,
        panel=False,
        boot=False,
    )


def python_estimate_multiperiod_rcs(data, control_group="nevertreated", base_period="universal", est_method="dr"):
    return ddd(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        control_group=control_group,
        base_period=base_period,
        est_method=est_method,
        panel=False,
        boot=False,
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_2period_rcs_point_estimates_match(two_period_rcs_data, est_method):
    data = two_period_rcs_data

    py_result = python_estimate_2period_rcs(data, est_method)
    r_result = r_estimate_2period_rcs(data, est_method)

    if r_result is None:
        pytest.fail("R RCS estimation failed")

    np.testing.assert_allclose(
        py_result.att,
        r_result["att"],
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"RCS {est_method}: ATT mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_2period_rcs_standard_errors_match(two_period_rcs_data, est_method):
    data = two_period_rcs_data

    py_result = python_estimate_2period_rcs(data, est_method)
    r_result = r_estimate_2period_rcs(data, est_method)

    if r_result is None:
        pytest.fail("R RCS estimation failed")

    np.testing.assert_allclose(
        py_result.se,
        r_result["se"],
        rtol=0.05,
        atol=0.02,
        err_msg=f"RCS {est_method}: SE mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
def test_2period_rcs_confidence_intervals_match(two_period_rcs_data):
    data = two_period_rcs_data

    py_result = python_estimate_2period_rcs(data, "dr")
    r_result = r_estimate_2period_rcs(data, "dr")

    if r_result is None:
        pytest.fail("R RCS estimation failed")

    np.testing.assert_allclose(
        py_result.lci,
        r_result["lci"],
        rtol=1e-4,
        atol=1e-4,
        err_msg="RCS LCI mismatch",
    )
    np.testing.assert_allclose(
        py_result.uci,
        r_result["uci"],
        rtol=1e-4,
        atol=1e-4,
        err_msg="RCS UCI mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_mp_rcs_att_gt_estimates_match(mp_rcs_data, est_method):
    data = mp_rcs_data

    py_result = python_estimate_multiperiod_rcs(data, est_method=est_method)
    r_result = r_estimate_multiperiod_rcs(data, est_method=est_method)

    if r_result is None:
        pytest.fail("R RCS estimation failed")

    r_att = np.atleast_1d(r_result["att"])
    r_groups = np.atleast_1d(r_result["groups"])
    r_times = np.atleast_1d(r_result["times"])

    assert len(r_att) == len(r_groups) == len(r_times), (
        f"RCS {est_method}: R returned mismatched lengths: "
        f"att={len(r_att)}, groups={len(r_groups)}, times={len(r_times)}"
    )

    matches = 0
    for i, (g, t) in enumerate(zip(py_result.groups, py_result.times)):
        r_mask = (r_groups == g) & (r_times == t)
        if np.any(r_mask):
            r_idx = np.where(r_mask)[0][0]
            py_att = py_result.att[i]
            r_att_val = r_att[r_idx]

            if np.isnan(py_att) and np.isnan(r_att_val):
                matches += 1
            elif not np.isnan(py_att) and not np.isnan(r_att_val):
                if np.allclose(py_att, r_att_val, rtol=1e-4, atol=1e-4):
                    matches += 1

    match_rate = matches / len(py_result.att) if len(py_result.att) > 0 else 0
    assert match_rate > 0.95, f"RCS {est_method}: Only {match_rate:.1%} of ATT(g,t) estimates match"


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
@pytest.mark.parametrize("control_group", ["nevertreated", "notyettreated"])
def test_mp_rcs_control_group_options(mp_rcs_data, control_group):
    data = mp_rcs_data

    py_result = python_estimate_multiperiod_rcs(data, control_group=control_group, est_method="reg")
    r_result = r_estimate_multiperiod_rcs(data, control_group=control_group, est_method="reg")

    if r_result is None:
        pytest.fail("R RCS estimation failed")

    r_att = np.atleast_1d(r_result["att"])
    r_groups = np.atleast_1d(r_result["groups"])
    r_times = np.atleast_1d(r_result["times"])
    assert len(py_result.att) > 0, f"RCS Python returned no ATTs for {control_group}"
    assert len(r_att) > 0, f"RCS R returned no ATTs for {control_group}"

    matches = 0
    for i, (g, t) in enumerate(zip(py_result.groups, py_result.times)):
        r_mask = (r_groups == g) & (r_times == t)
        if np.any(r_mask):
            r_idx = np.where(r_mask)[0][0]
            py_att = py_result.att[i]
            r_att_val = r_att[r_idx]
            if np.isnan(py_att) and np.isnan(r_att_val):
                matches += 1
            elif not np.isnan(py_att) and not np.isnan(r_att_val):
                if np.allclose(py_att, r_att_val, rtol=1e-4, atol=1e-4):
                    matches += 1
    match_rate = matches / len(py_result.att) if len(py_result.att) > 0 else 0
    assert match_rate > 0.95, f"RCS {control_group}: Only {match_rate:.1%} of ATT(g,t) match"


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
@pytest.mark.parametrize("base_period", ["universal", "varying"])
def test_mp_rcs_base_period_options(mp_rcs_data, base_period):
    data = mp_rcs_data

    py_result = python_estimate_multiperiod_rcs(data, base_period=base_period, est_method="reg")
    r_result = r_estimate_multiperiod_rcs(data, base_period=base_period, est_method="reg")

    if r_result is None:
        pytest.fail("R RCS estimation failed")

    r_att = np.atleast_1d(r_result["att"])
    r_groups = np.atleast_1d(r_result["groups"])
    r_times = np.atleast_1d(r_result["times"])
    assert len(py_result.att) > 0, f"RCS Python returned no ATTs for {base_period}"
    assert len(r_att) > 0, f"RCS R returned no ATTs for {base_period}"

    matches = 0
    for i, (g, t) in enumerate(zip(py_result.groups, py_result.times)):
        r_mask = (r_groups == g) & (r_times == t)
        if np.any(r_mask):
            r_idx = np.where(r_mask)[0][0]
            py_att = py_result.att[i]
            r_att_val = r_att[r_idx]
            if np.isnan(py_att) and np.isnan(r_att_val):
                matches += 1
            elif not np.isnan(py_att) and not np.isnan(r_att_val):
                if np.allclose(py_att, r_att_val, rtol=1e-4, atol=1e-4):
                    matches += 1
    match_rate = matches / len(py_result.att) if len(py_result.att) > 0 else 0
    assert match_rate > 0.95, f"RCS {base_period}: Only {match_rate:.1%} of ATT(g,t) match"


def test_ddd_rc_basic_functionality(two_period_rcs_data):
    data = two_period_rcs_data

    post = (data["time"] == 1).cast(pl.Int64).to_numpy()
    y = data["y"].to_numpy()
    state = data["state"].to_numpy()
    partition = data["partition"].to_numpy()
    subgroup = 1 + state + 2 * partition

    covariates = data.select(["cov1", "cov2", "cov3", "cov4"]).to_numpy()
    covariates = np.column_stack([np.ones(len(data)), covariates])

    result = ddd_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        est_method="dr",
        boot=False,
        influence_func=True,
    )

    assert hasattr(result, "att"), "Missing att attribute"
    assert hasattr(result, "se"), "Missing se attribute"
    assert result.se > 0, f"SE must be positive, got {result.se}"


def test_ddd_mp_rc_basic_functionality(mp_rcs_data):
    data = mp_rcs_data

    result = ddd_mp_rc(
        data=data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        est_method="reg",
    )

    assert len(result.att) > 0, "ddd_mp_rc produced no ATTs"
    assert len(result.groups) == len(result.att), "groups and att length mismatch"
    assert len(result.times) == len(result.att), "times and att length mismatch"


@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_ddd_rc_all_methods_work(two_period_rcs_data, est_method):
    data = two_period_rcs_data

    post = (data["time"] == 1).cast(pl.Int64).to_numpy()
    y = data["y"].to_numpy()
    state = data["state"].to_numpy()
    partition = data["partition"].to_numpy()
    subgroup = 1 + state + 2 * partition

    covariates = data.select(["cov1", "cov2", "cov3", "cov4"]).to_numpy()
    covariates = np.column_stack([np.ones(len(data)), covariates])

    result = ddd_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        est_method=est_method,
        boot=False,
    )

    assert result.se > 0, f"{est_method}: SE must be positive"
    assert result.lci < result.att < result.uci, f"{est_method}: ATT not within CI"


@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_ddd_mp_rc_all_methods_work(mp_rcs_data, est_method):
    data = mp_rcs_data

    result = ddd_mp_rc(
        data=data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        est_method=est_method,
    )

    assert len(result.att) > 0, f"{est_method} produced no ATTs"
    valid_atts = result.att[~np.isnan(result.att)]
    assert len(valid_atts) > 0, f"{est_method} produced all NaN ATTs"


def test_ddd_wrapper_rcs_mode(two_period_rcs_data):
    data = two_period_rcs_data

    result = ddd(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
        xformla="~ cov1 + cov2 + cov3 + cov4",
        est_method="dr",
        panel=False,
    )

    assert hasattr(result, "att"), "Missing att attribute"
    assert hasattr(result, "se"), "Missing se attribute"


def test_ddd_wrapper_mp_rcs_mode(mp_rcs_data):
    data = mp_rcs_data

    result = ddd(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method="reg",
        panel=False,
    )

    assert len(result.att) > 0, "Multi-period RCS produced no ATTs"


def test_rcs_influence_function_properties(two_period_rcs_data):
    data = two_period_rcs_data

    post = (data["time"] == 1).cast(pl.Int64).to_numpy()
    y = data["y"].to_numpy()
    state = data["state"].to_numpy()
    partition = data["partition"].to_numpy()
    subgroup = 1 + state + 2 * partition

    covariates = data.select(["cov1", "cov2", "cov3", "cov4"]).to_numpy()
    covariates = np.column_stack([np.ones(len(data)), covariates])

    result = ddd_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        est_method="dr",
        boot=False,
        influence_func=True,
    )

    assert result.att_inf_func is not None, "Influence function is None"
    assert len(result.att_inf_func) == len(data), "IF length should match number of observations"

    se_from_if = np.sqrt(np.var(result.att_inf_func) / len(result.att_inf_func))
    np.testing.assert_almost_equal(result.se, se_from_if, decimal=3, err_msg="SE from IF doesn't match reported SE")


def test_rcs_ddd_formula_holds(two_period_rcs_data):
    data = two_period_rcs_data

    post = (data["time"] == 1).cast(pl.Int64).to_numpy()
    y = data["y"].to_numpy()
    state = data["state"].to_numpy()
    partition = data["partition"].to_numpy()
    subgroup = 1 + state + 2 * partition

    covariates = data.select(["cov1", "cov2", "cov3", "cov4"]).to_numpy()
    covariates = np.column_stack([np.ones(len(data)), covariates])

    result = ddd_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        est_method="dr",
        boot=False,
        influence_func=True,
    )

    computed_ddd = result.did_atts["att_4v3"] + result.did_atts["att_4v2"] - result.did_atts["att_4v1"]
    np.testing.assert_almost_equal(result.att, computed_ddd, decimal=10, err_msg="DDD formula mismatch for RCS")


def r_estimate_with_eventstudy(
    data,
    yname="y",
    tname="time",
    idname="id",
    gname="group",
    pname="partition",
    xformla="~1",
    control_group="nevertreated",
    allow_unbalanced_panel=False,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "data.csv"
        result_path = Path(tmpdir) / "result.json"

        data.write_csv(data_path)

        unbalanced_str = "TRUE" if allow_unbalanced_panel else "FALSE"

        r_script = f"""
library(triplediff)
library(jsonlite)

data <- read.csv("{data_path}")

result <- ddd(
    yname = "{yname}",
    tname = "{tname}",
    idname = "{idname}",
    gname = "{gname}",
    pname = "{pname}",
    xformla = {xformla},
    data = data,
    control_group = "{control_group}",
    base_period = "universal",
    est_method = "dr",
    allow_unbalanced_panel = {unbalanced_str},
    boot = FALSE
)

es <- agg_ddd(result, type = "eventstudy", boot = FALSE)$aggte_ddd

output <- list(
    att = result$ATT,
    se = result$se,
    groups = result$groups,
    times = result$periods,
    es_egt = es$egt,
    es_att = es$att.egt,
    es_se = es$se.egt,
    es_overall_att = es$overall.att,
    es_overall_se = es$overall.se
)

write_json(output, "{result_path}", auto_unbox = TRUE, digits = NA)
"""
        try:
            return _run_r_script(r_script, result_path, timeout=300)
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
            return None


def python_estimate_cai(data, xformla=None, allow_unbalanced_panel=False):
    return ddd(
        data=data,
        yname="checksaving_ratio",
        tname="year",
        idname="hhno",
        gname="group",
        pname="sector",
        xformla=xformla,
        control_group="nevertreated",
        base_period="universal",
        est_method="dr",
        allow_unbalanced_panel=allow_unbalanced_panel,
    )


def r_estimate_cai(data, xformla=None, allow_unbalanced_panel=False):
    return r_estimate_with_eventstudy(
        data,
        yname="checksaving_ratio",
        tname="year",
        idname="hhno",
        gname="group",
        pname="sector",
        xformla=xformla or "~1",
        allow_unbalanced_panel=allow_unbalanced_panel,
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
@pytest.mark.parametrize("xformla", [None, "~ hhsize + age"])
def test_cai_unbalanced_att_gt_match(cai_data, xformla):
    py_result = python_estimate_cai(cai_data, xformla=xformla, allow_unbalanced_panel=True)
    r_result = r_estimate_cai(cai_data, xformla=xformla, allow_unbalanced_panel=True)

    if r_result is None:
        pytest.fail("R estimation failed")

    assert py_result.n == cai_data["hhno"].n_unique()
    np.testing.assert_array_equal(py_result.times, np.atleast_1d(r_result["times"]))
    np.testing.assert_allclose(py_result.att, _convert_r_array(r_result["att"]), rtol=0, atol=1e-10)
    np.testing.assert_allclose(py_result.se, _convert_r_array(r_result["se"]), rtol=0, atol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
def test_cai_unbalanced_eventstudy_matches(cai_data):
    py_result = python_estimate_cai(cai_data, xformla="~ hhsize + age", allow_unbalanced_panel=True)
    py_agg = agg_ddd(py_result, type="eventstudy", boot=False, cband=False)
    r_result = r_estimate_cai(cai_data, xformla="~ hhsize + age", allow_unbalanced_panel=True)

    if r_result is None:
        pytest.fail("R estimation failed")

    np.testing.assert_array_equal(py_agg.egt, np.atleast_1d(r_result["es_egt"]))
    np.testing.assert_allclose(py_agg.att_egt, _convert_r_array(r_result["es_att"]), rtol=0, atol=1e-10)
    np.testing.assert_allclose(py_agg.se_egt, _convert_r_array(r_result["es_se"]), rtol=0, atol=1e-10)
    np.testing.assert_allclose(py_agg.overall_att, r_result["es_overall_att"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(py_agg.overall_se, r_result["es_overall_se"], rtol=0, atol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
def test_cai_balanced_covariates_post_treatment_match(cai_balanced_data):
    py_result = python_estimate_cai(cai_balanced_data, xformla="~ hhsize + age")
    r_result = r_estimate_cai(cai_balanced_data, xformla="~ hhsize + age")

    if r_result is None:
        pytest.fail("R estimation failed")

    post = py_result.times >= py_result.groups
    np.testing.assert_array_equal(py_result.times, np.atleast_1d(r_result["times"]))
    np.testing.assert_allclose(py_result.att[post], _convert_r_array(r_result["att"])[post], rtol=0, atol=1e-10)
    np.testing.assert_allclose(py_result.se[post], _convert_r_array(r_result["se"])[post], rtol=0, atol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
def test_cai_balanced_covariates_pre_treatment_close(cai_balanced_data):
    py_result = python_estimate_cai(cai_balanced_data, xformla="~ hhsize + age")
    r_result = r_estimate_cai(cai_balanced_data, xformla="~ hhsize + age")

    if r_result is None:
        pytest.fail("R estimation failed")

    pre = py_result.times < py_result.groups
    np.testing.assert_allclose(py_result.att[pre], _convert_r_array(r_result["att"])[pre], rtol=0, atol=5e-5)
    np.testing.assert_allclose(py_result.se[pre], _convert_r_array(r_result["se"])[pre], rtol=0, atol=1e-5)


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
@pytest.mark.parametrize("control_group", ["nevertreated", "notyettreated"])
def test_mp_unbalanced_panel_matches(mp_ddd_unbalanced_data, control_group):
    data = mp_ddd_unbalanced_data

    py_result = ddd(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        control_group=control_group,
        base_period="universal",
        est_method="dr",
        allow_unbalanced_panel=True,
    )
    py_agg = agg_ddd(py_result, type="eventstudy", boot=False, cband=False)
    r_result = r_estimate_with_eventstudy(data, control_group=control_group, allow_unbalanced_panel=True)

    if r_result is None:
        pytest.fail("R estimation failed")

    np.testing.assert_allclose(py_result.att, _convert_r_array(r_result["att"]), rtol=0, atol=1e-10)
    np.testing.assert_allclose(py_result.se, _convert_r_array(r_result["se"]), rtol=0, atol=1e-10)
    np.testing.assert_allclose(py_agg.att_egt, _convert_r_array(r_result["es_att"]), rtol=0, atol=1e-10)
    np.testing.assert_allclose(py_agg.se_egt, _convert_r_array(r_result["es_se"]), rtol=0, atol=1e-10)
    np.testing.assert_allclose(py_agg.overall_se, r_result["es_overall_se"], rtol=0, atol=1e-10)


def r_estimate_ddd(
    data,
    gname="state",
    panel=True,
    xformla="~ cov1 + cov2 + cov3 + cov4",
    est_method="dr",
    alpha=0.05,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "data.csv"
        result_path = Path(tmpdir) / "result.json"

        data.write_csv(data_path)

        panel_str = "TRUE" if panel else "FALSE"

        r_script = f"""
library(triplediff)
library(jsonlite)

data <- read.csv("{data_path}")

result <- ddd(
    yname = "y",
    tname = "time",
    idname = "id",
    gname = "{gname}",
    pname = "partition",
    xformla = {xformla},
    data = data,
    control_group = "nevertreated",
    base_period = "universal",
    est_method = "{est_method}",
    panel = {panel_str},
    alpha = {alpha},
    boot = FALSE
)

output <- list(
    att = result$ATT,
    se = result$se,
    lci = result$lci,
    uci = result$uci,
    alpha = result$argu$alpha
)

write_json(output, "{result_path}", auto_unbox = TRUE, digits = NA)
"""
        try:
            return _run_r_script(r_script, result_path, timeout=120)
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
            return None


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
@pytest.mark.parametrize(
    "data_fixture, gname, panel, xformla",
    [
        ("two_period_clustered_data", "state", True, "~ cov1 + cov2 + cov3 + cov4"),
        ("two_period_rcs_data", "state", False, "~ cov1 + cov2 + cov3 + cov4"),
        ("mp_ddd_data", "group", True, "~1"),
    ],
)
def test_alpha_above_tenth_matches_reference(request, data_fixture, gname, panel, xformla):
    data = request.getfixturevalue(data_fixture)

    with pytest.warns(UserWarning, match="alpha=0.2 is above 0.10. Using alpha=0.05."):
        py_result = ddd(
            data=data,
            yname="y",
            tname="time",
            idname="id",
            gname=gname,
            pname="partition",
            xformla=xformla,
            est_method="reg",
            panel=panel,
            alpha=0.2,
        )
    r_result = r_estimate_ddd(data, gname=gname, panel=panel, xformla=xformla, est_method="reg", alpha=0.2)

    if r_result is None:
        pytest.fail("R estimation failed")

    assert py_result.args["alpha"] == r_result["alpha"] == 0.05
    for key in ["att", "se", "lci", "uci"]:
        np.testing.assert_allclose(
            np.atleast_1d(getattr(py_result, key)),
            _convert_r_array(np.atleast_1d(r_result[key])),
            rtol=0,
            atol=1e-8,
            err_msg=key,
        )


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
def test_2period_unbalanced_panel_matches_reference_on_complete_units(two_period_dgp_result):
    data, _, _ = two_period_dgp_result
    unbalanced = data.filter(~((pl.col("id") % 25 == 0) & (pl.col("time") == 2)))
    complete = unbalanced.filter(pl.len().over("id") == 2)

    with pytest.warns(UserWarning, match="Dropped 40 units while converting to balanced panel"):
        py_result = ddd(
            data=unbalanced,
            yname="y",
            tname="time",
            idname="id",
            gname="state",
            pname="partition",
            xformla="~ cov1 + cov2 + cov3 + cov4",
            est_method="dr",
        )
    r_result = r_estimate_ddd(complete)

    if r_result is None:
        pytest.fail("R estimation failed")

    for key in ["att", "se", "lci", "uci"]:
        np.testing.assert_allclose(getattr(py_result, key), r_result[key], rtol=0, atol=1e-8, err_msg=key)


def _run_r_script(r_script, result_path, timeout=60):
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


def r_estimate_ddd_mp(
    data,
    idname="id",
    panel=True,
    allow_unbalanced_panel=False,
    weightsname=None,
    xformla="~1",
    control_group="nevertreated",
    base_period="universal",
    est_method="dr",
    agg_types=(),
    cluster=None,
    boot=False,
    nboot=999,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "data.csv"
        result_path = Path(tmpdir) / "result.json"

        data.write_csv(data_path)

        weights_str = "NULL" if weightsname is None else f'"{weightsname}"'
        panel_str = "TRUE" if panel else "FALSE"
        unbalanced_str = "TRUE" if allow_unbalanced_panel else "FALSE"
        agg_str = ", ".join(f'"{agg_type}"' for agg_type in agg_types)
        cluster_str = "NULL" if cluster is None else f'"{cluster}"'
        boot_str = "TRUE" if boot else "FALSE"

        r_script = f"""
library(triplediff)
library(jsonlite)

set.seed(42)
data <- read.csv("{data_path}")

result <- ddd(
    yname = "y",
    tname = "time",
    idname = "{idname}",
    gname = "group",
    pname = "partition",
    xformla = {xformla},
    data = data,
    control_group = "{control_group}",
    base_period = "{base_period}",
    est_method = "{est_method}",
    weightsname = {weights_str},
    panel = {panel_str},
    allow_unbalanced_panel = {unbalanced_str},
    boot = {boot_str},
    nboot = {nboot},
    cluster = {cluster_str}
)

output <- list(att = result$ATT, se = result$se, groups = result$groups, times = result$periods, n = result$n)
for (agg_type in c({agg_str})) {{
    agg <- agg_ddd(result, type = agg_type, boot = FALSE, cband = FALSE)$aggte_ddd
    output[[agg_type]] <- list(
        overall_att = agg$overall.att,
        overall_se = agg$overall.se,
        egt = agg$egt,
        att_egt = agg$att.egt,
        se_egt = agg$se.egt
    )
}}

write_json(output, "{result_path}", auto_unbox = TRUE, digits = NA)
"""
        try:
            return _run_r_script(r_script, result_path, timeout=300)
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
            return None


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
@pytest.mark.parametrize("base_period", ["universal", "varying"])
def test_mp_first_period_cohort_matches_reference(mp_first_period_cohort_data, base_period):
    with pytest.warns(UserWarning, match="^Dropped 80 units that were already treated in the first period$"):
        py_result = ddd(
            data=mp_first_period_cohort_data,
            yname="y",
            tname="time",
            idname="id",
            gname="group",
            pname="partition",
            base_period=base_period,
            est_method="reg",
        )
    r_result = r_estimate_ddd_mp(
        mp_first_period_cohort_data, base_period=base_period, est_method="reg", agg_types=["simple", "eventstudy"]
    )

    if r_result is None:
        pytest.fail("R estimation failed")

    assert py_result.n == r_result["n"]
    np.testing.assert_array_equal(py_result.groups, np.atleast_1d(r_result["groups"]))
    np.testing.assert_array_equal(py_result.times, np.atleast_1d(r_result["times"]))
    np.testing.assert_allclose(py_result.att, _convert_r_array(np.atleast_1d(r_result["att"])), rtol=0, atol=1e-10)
    np.testing.assert_allclose(py_result.se, _convert_r_array(np.atleast_1d(r_result["se"])), rtol=0, atol=1e-10)
    for agg_type in ["simple", "eventstudy"]:
        py_agg = agg_ddd(py_result, type=agg_type, boot=False, cband=False)
        np.testing.assert_allclose(py_agg.overall_att, r_result[agg_type]["overall_att"], rtol=0, atol=1e-10)
        np.testing.assert_allclose(py_agg.overall_se, r_result[agg_type]["overall_se"], rtol=0, atol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
@pytest.mark.parametrize("base_period", ["universal", "varying"])
def test_mp_no_never_treated_matches_reference_on_shared_cells(mp_no_never_treated_data, base_period):
    py_result = ddd(
        data=mp_no_never_treated_data,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        control_group="notyettreated",
        base_period=base_period,
        est_method="reg",
    )
    r_result = r_estimate_ddd_mp(
        mp_no_never_treated_data, control_group="notyettreated", base_period=base_period, est_method="reg"
    )

    if r_result is None:
        pytest.fail("R estimation failed")

    r_cells = list(zip(np.atleast_1d(r_result["groups"]), np.atleast_1d(r_result["times"])))
    py_cells = {cell: i for i, cell in enumerate(zip(py_result.groups, py_result.times))}
    rows = [py_cells[cell] for cell in r_cells]
    r_att = _convert_r_array(np.atleast_1d(r_result["att"]))
    r_se = _convert_r_array(np.atleast_1d(r_result["se"]))
    assert {int(g) for g, _ in set(py_cells) - set(r_cells)} == {4}
    np.testing.assert_allclose(py_result.att[rows], r_att, rtol=0, atol=1e-10)
    np.testing.assert_allclose(py_result.se[rows], r_se, rtol=0, atol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
@pytest.mark.parametrize(("est_method", "atol"), [("dr", 1e-6), ("reg", 1e-10)])
@pytest.mark.parametrize("control_group", ["nevertreated", "notyettreated"])
@pytest.mark.parametrize("base_period", ["universal", "varying"])
def test_mp_weights_match_reference(mp_ddd_weighted_data, est_method, atol, control_group, base_period):
    py_result = ddd(
        data=mp_ddd_weighted_data,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        xformla="~ cov1 + cov2",
        control_group=control_group,
        base_period=base_period,
        est_method=est_method,
        weightsname="w",
    )
    r_result = r_estimate_ddd_mp(
        mp_ddd_weighted_data,
        weightsname="w",
        xformla="~ cov1 + cov2",
        control_group=control_group,
        base_period=base_period,
        est_method=est_method,
    )

    if r_result is None:
        pytest.fail("R estimation failed")

    np.testing.assert_array_equal(py_result.times, np.atleast_1d(r_result["times"]))
    np.testing.assert_allclose(py_result.att, _convert_r_array(np.atleast_1d(r_result["att"])), rtol=0, atol=atol)
    np.testing.assert_allclose(py_result.se, _convert_r_array(np.atleast_1d(r_result["se"])), rtol=0, atol=atol)


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
@pytest.mark.parametrize(
    ("data_fixture", "idname", "panel", "allow_unbalanced_panel"),
    [
        ("mp_ddd_weighted_data", "id", True, False),
        ("mp_ddd_weighted_unbalanced_data", "id", True, True),
        ("mp_rcs_weighted_data", "rid", False, False),
    ],
)
def test_mp_weighted_aggregations_match_reference(request, data_fixture, idname, panel, allow_unbalanced_panel):
    data = request.getfixturevalue(data_fixture)
    agg_types = ["simple", "eventstudy", "group", "calendar"]
    py_result = ddd(
        data=data,
        yname="y",
        tname="time",
        idname=idname,
        gname="group",
        pname="partition",
        xformla="~ cov1 + cov2",
        est_method="dr",
        weightsname="w",
        panel=panel,
        allow_unbalanced_panel=allow_unbalanced_panel,
    )
    r_result = r_estimate_ddd_mp(
        data,
        idname=idname,
        panel=panel,
        allow_unbalanced_panel=allow_unbalanced_panel,
        weightsname="w",
        xformla="~ cov1 + cov2",
        agg_types=agg_types,
    )

    if r_result is None:
        pytest.fail("R estimation failed")

    assert py_result.n == r_result["n"]
    np.testing.assert_allclose(py_result.att, _convert_r_array(np.atleast_1d(r_result["att"])), rtol=0, atol=1e-6)
    np.testing.assert_allclose(py_result.se, _convert_r_array(np.atleast_1d(r_result["se"])), rtol=0, atol=1e-6)
    for agg_type in agg_types:
        py_agg = agg_ddd(py_result, type=agg_type, boot=False, cband=False)
        r_agg = r_result[agg_type]
        np.testing.assert_allclose(py_agg.overall_att, r_agg["overall_att"], rtol=0, atol=1e-6)
        if agg_type != "simple":
            np.testing.assert_allclose(py_agg.att_egt, _convert_r_array(np.atleast_1d(r_agg["att_egt"])), atol=1e-6)
            np.testing.assert_allclose(py_agg.se_egt, _convert_r_array(np.atleast_1d(r_agg["se_egt"])), atol=1e-6)
        if agg_type != "group":
            np.testing.assert_allclose(py_agg.overall_se, r_agg["overall_se"], rtol=0, atol=1e-6)


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
@pytest.mark.parametrize("est_method", ["dr", "reg"])
def test_mp_rcs_pooled_cells_match_reference(mp_rcs_data, est_method):
    py_result = ddd(
        data=mp_rcs_data,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        control_group="notyettreated",
        est_method=est_method,
        panel=False,
    )
    r_result = r_estimate_ddd_mp(mp_rcs_data, panel=False, control_group="notyettreated", est_method=est_method)

    if r_result is None:
        pytest.fail("R estimation failed")

    assert py_result.n == r_result["n"]
    np.testing.assert_array_equal(py_result.times, np.atleast_1d(r_result["times"]))
    np.testing.assert_allclose(py_result.att, _convert_r_array(np.atleast_1d(r_result["att"])), rtol=0, atol=1e-10)
    np.testing.assert_allclose(py_result.se, _convert_r_array(np.atleast_1d(r_result["se"])), rtol=0, atol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
@pytest.mark.parametrize(
    ("data_fixture", "panel", "allow_unbalanced_panel", "control_group"),
    [
        ("mp_ddd_clustered_data", True, False, "nevertreated"),
        ("mp_ddd_clustered_data", True, False, "notyettreated"),
        ("mp_ddd_unbalanced_clustered_data", True, True, "notyettreated"),
        ("mp_rcs_clustered_data", False, False, "notyettreated"),
    ],
)
def test_mp_analytic_clustered_se_matches_reference(
    request, data_fixture, panel, allow_unbalanced_panel, control_group
):
    data = request.getfixturevalue(data_fixture)
    agg_types = ["simple", "eventstudy", "group", "calendar"]
    py_result = ddd(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        control_group=control_group,
        est_method="dr",
        panel=panel,
        allow_unbalanced_panel=allow_unbalanced_panel,
        cluster="cluster",
    )
    r_result = r_estimate_ddd_mp(
        data,
        panel=panel,
        allow_unbalanced_panel=allow_unbalanced_panel,
        control_group=control_group,
        cluster="cluster",
        agg_types=agg_types,
    )

    if r_result is None:
        pytest.fail("R estimation failed")

    np.testing.assert_allclose(py_result.att, _convert_r_array(np.atleast_1d(r_result["att"])), rtol=0, atol=1e-8)
    np.testing.assert_allclose(py_result.se, _convert_r_array(np.atleast_1d(r_result["se"])), rtol=0, atol=1e-8)
    for agg_type in agg_types:
        py_agg = agg_ddd(py_result, type=agg_type, boot=False, cband=False)
        r_agg = r_result[agg_type]
        if agg_type != "simple":
            np.testing.assert_allclose(py_agg.se_egt, _convert_r_array(np.atleast_1d(r_agg["se_egt"])), atol=1e-8)
        if agg_type != "group":
            np.testing.assert_allclose(py_agg.overall_se, r_agg["overall_se"], rtol=0, atol=1e-8)


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
def test_mp_bootstrap_pooled_cells_match_reference(mp_rcs_data):
    py_result = ddd(
        data=mp_rcs_data,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        control_group="notyettreated",
        est_method="dr",
        panel=False,
        boot=True,
        biters=20000,
        random_state=42,
    )
    r_result = r_estimate_ddd_mp(mp_rcs_data, panel=False, control_group="notyettreated", boot=True, nboot=20000)

    if r_result is None:
        pytest.fail("R bootstrap estimation failed")

    np.testing.assert_allclose(py_result.att, _convert_r_array(np.atleast_1d(r_result["att"])), rtol=0, atol=1e-10)
    np.testing.assert_allclose(py_result.se, _convert_r_array(np.atleast_1d(r_result["se"])), rtol=0.05)


def r_ddd_mp_error(data, control_group="nevertreated"):
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "data.csv"
        result_path = Path(tmpdir) / "result.json"

        data.write_csv(data_path)

        r_script = f"""
library(triplediff)
library(jsonlite)

data <- read.csv("{data_path}")

message <- tryCatch({{
    ddd(
        yname = "y",
        tname = "time",
        idname = "id",
        gname = "group",
        pname = "partition",
        xformla = ~1,
        data = data,
        control_group = "{control_group}",
        base_period = "universal",
        est_method = "reg",
        boot = FALSE
    )
    NA
}}, error = function(e) conditionMessage(e))

write_json(list(error = message), "{result_path}", auto_unbox = TRUE)
"""
        try:
            return _run_r_script(r_script, result_path, timeout=120)
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
            return None


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
def test_mp_without_never_treated_units_stops_like_reference(mp_no_never_treated_data):
    r_result = r_ddd_mp_error(mp_no_never_treated_data)

    if r_result is None:
        pytest.fail("R estimation failed")

    assert r_result["error"] == "There is no available never-treated group"
    with pytest.raises(ValueError, match=re.escape(r_result["error"])):
        ddd(
            data=mp_no_never_treated_data,
            yname="y",
            tname="time",
            idname="id",
            gname="group",
            pname="partition",
            control_group="nevertreated",
            est_method="reg",
        )


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
@pytest.mark.parametrize("control_group", ["nevertreated", "notyettreated"])
def test_mp_unbalanced_panel_drops_incomplete_units_like_reference(mp_ddd_unbalanced_data, control_group):
    agg_types = ["simple", "eventstudy"]
    py_result = ddd(
        data=mp_ddd_unbalanced_data,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        control_group=control_group,
        est_method="dr",
    )
    r_result = r_estimate_ddd_mp(mp_ddd_unbalanced_data, control_group=control_group, agg_types=agg_types)

    if r_result is None:
        pytest.fail("R estimation failed")

    assert py_result.n == r_result["n"] == mp_ddd_unbalanced_data.filter(pl.len().over("id") == 5)["id"].n_unique()
    np.testing.assert_allclose(py_result.att, _convert_r_array(np.atleast_1d(r_result["att"])), rtol=0, atol=1e-10)
    np.testing.assert_allclose(py_result.se, _convert_r_array(np.atleast_1d(r_result["se"])), rtol=0, atol=1e-10)
    for agg_type in agg_types:
        py_agg = agg_ddd(py_result, type=agg_type, boot=False, cband=False)
        np.testing.assert_allclose(py_agg.overall_att, r_result[agg_type]["overall_att"], rtol=0, atol=1e-10)
        np.testing.assert_allclose(py_agg.overall_se, r_result[agg_type]["overall_se"], rtol=0, atol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
@pytest.mark.parametrize("mp_ddd_missing_value_data", ["cohort", "repeated_row"], indirect=True)
def test_mp_drops_rows_with_missing_values_before_checking_the_panel_like_reference(mp_ddd_missing_value_data):
    py_result = ddd(
        data=mp_ddd_missing_value_data,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method="reg",
    )
    r_result = r_estimate_ddd_mp(mp_ddd_missing_value_data, est_method="reg")

    if r_result is None:
        pytest.fail("R estimation failed")

    assert py_result.n == r_result["n"]
    np.testing.assert_allclose(py_result.att, _convert_r_array(np.atleast_1d(r_result["att"])), rtol=0, atol=1e-10)
    np.testing.assert_allclose(py_result.se, _convert_r_array(np.atleast_1d(r_result["se"])), rtol=0, atol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
def test_mp_balanced_panel_allowing_unbalanced_panels_matches_reference_panel_path(mp_ddd_weighted_data):
    py_result = ddd(
        data=mp_ddd_weighted_data,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        xformla="~ cov1 + cov2",
        est_method="dr",
        allow_unbalanced_panel=True,
    )
    r_result = r_estimate_ddd_mp(mp_ddd_weighted_data, xformla="~ cov1 + cov2", allow_unbalanced_panel=False)

    if r_result is None:
        pytest.fail("R estimation failed")

    np.testing.assert_allclose(py_result.att, _convert_r_array(np.atleast_1d(r_result["att"])), rtol=0, atol=1e-8)
    np.testing.assert_allclose(py_result.se, _convert_r_array(np.atleast_1d(r_result["se"])), rtol=0, atol=1e-8)


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
@pytest.mark.parametrize(
    ("mp_no_never_treated_layout_data", "panel", "allow_unbalanced_panel"),
    [
        ("late_gap", True, False),
        ("cohort_gap", True, False),
        ("late_only", True, True),
        ("one_row_per_id", False, False),
    ],
    indirect=["mp_no_never_treated_layout_data"],
)
def test_mp_no_never_treated_leaves_out_periods_without_comparisons_like_reference(
    mp_no_never_treated_layout_data, panel, allow_unbalanced_panel
):
    data = mp_no_never_treated_layout_data
    agg_types = ["simple", "eventstudy"]
    py_result = ddd(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        control_group="notyettreated",
        est_method="reg",
        panel=panel,
        allow_unbalanced_panel=allow_unbalanced_panel,
    )
    r_result = r_estimate_ddd_mp(
        data,
        panel=panel,
        allow_unbalanced_panel=allow_unbalanced_panel,
        control_group="notyettreated",
        est_method="reg",
        agg_types=agg_types,
    )

    if r_result is None:
        pytest.fail("R estimation failed")

    r_cells = list(zip(np.atleast_1d(r_result["groups"]), np.atleast_1d(r_result["times"])))
    py_cells = {cell: i for i, cell in enumerate(zip(py_result.groups, py_result.times))}
    rows = [py_cells[cell] for cell in r_cells]
    assert py_result.n == r_result["n"]
    assert {int(g) for g, _ in set(py_cells) - set(r_cells)} == {4}
    r_att = _convert_r_array(np.atleast_1d(r_result["att"]))
    r_se = _convert_r_array(np.atleast_1d(r_result["se"]))
    np.testing.assert_allclose(py_result.att[rows], r_att, rtol=0, atol=1e-10)
    np.testing.assert_allclose(py_result.se[rows], r_se, rtol=0, atol=1e-10)
    for agg_type in agg_types:
        py_agg = agg_ddd(py_result, type=agg_type, boot=False, cband=False)
        np.testing.assert_allclose(py_agg.overall_att, r_result[agg_type]["overall_att"], rtol=0, atol=1e-10)
        np.testing.assert_allclose(py_agg.overall_se, r_result[agg_type]["overall_se"], rtol=0, atol=1e-10)
