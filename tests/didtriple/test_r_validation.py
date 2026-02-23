"""Validation tests comparing Python DDD implementation with R triplediff package."""

import json
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
    gen_dgp_2periods,
    gen_dgp_mult_periods,
)
from moderndid.core.preprocessing import preprocess_ddd_2periods

from ..helpers import importorskip

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
    groups = result$group,
    times = result$t
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
    groups = result$group,
    times = result$t,
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
    result = gen_dgp_2periods(n=1000, dgp_type=dgp_type, random_state=42)
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

    if len(r_att) != len(r_groups) or len(r_att) != len(r_times):
        r_pairs = []
        for g in r_groups:
            for t in r_times:
                if 0 < g <= t:
                    r_pairs.append((g, t))

        assert len(py_result.att) > 0, f"{est_method}: Python returned no ATTs"
        assert len(r_att) > 0, f"{est_method}: R returned no ATTs"
        return

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

    assert len(py_result.att) > 0, f"Python returned no ATTs for {control_group}"


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

    assert len(py_result.att) > 0, f"Python returned no ATTs for {base_period}"


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

    if "egt" in r_result and r_result["egt"] is not None:
        r_egt = np.array(r_result["egt"])
        assert all(-1 <= e <= 2 for e in py_agg.egt), "Python egt outside [min_e, max_e]"
        assert all(-1 <= e <= 2 for e in r_egt), "R egt outside [min_e, max_e]"


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
    py_result = gen_dgp_2periods(n=5000, dgp_type=1, random_state=42)
    py_data = py_result["data"]

    required_cols = ["id", "time", "y", "state", "partition", "cov1", "cov2", "cov3", "cov4"]
    for col in required_cols:
        assert col in py_data.columns, f"Missing column: {col}"

    assert len(py_data) == 10000, f"Expected 10000 rows, got {len(py_data)}"


def test_dgp_subgroup_proportions_reasonable():
    py_result = gen_dgp_2periods(n=5000, dgp_type=1, random_state=42)
    py_data = py_result["data"]

    units = py_data.unique(subset=["id"])
    subgroup_counts = units.group_by(["state", "partition"]).len()

    min_count = subgroup_counts["len"].min()
    assert min_count > 200, f"Subgroup too small: {min_count}"


@pytest.mark.parametrize("dgp_type", [1, 2, 3, 4])
def test_dgp_types_work(dgp_type):
    result = gen_dgp_2periods(n=1000, dgp_type=dgp_type, random_state=42)
    data = result["data"]

    assert len(data) == 2000, f"Expected 2000 rows, got {len(data)}"
    assert result["true_att"] == 0.0, f"Expected true ATT=0, got {result['true_att']}"
    assert "efficiency_bound" in result, "Missing efficiency_bound"


def test_mp_dgp_produces_valid_structure():
    result = gen_dgp_mult_periods(n=500, dgp_type=1, random_state=42)
    data = result["data"]

    required_cols = ["id", "time", "y", "group", "partition"]
    for col in required_cols:
        assert col in data.columns, f"Missing column: {col}"

    n_obs = len(data)
    n_units = data["id"].n_unique()
    n_times = data["time"].n_unique()
    assert n_obs == n_units * n_times, f"Unbalanced panel: {n_obs} != {n_units} * {n_times}"


def test_mp_dgp_group_structure():
    result = gen_dgp_mult_periods(n=500, dgp_type=1, random_state=42)
    data = result["data"]

    groups = data["group"].unique()
    assert 0 in groups, "Missing never-treated group (0)"
    assert len(groups) >= 2, "Need at least 2 groups"


def test_mp_dgp_partition_binary():
    result = gen_dgp_mult_periods(n=500, dgp_type=1, random_state=42)
    data = result["data"]

    assert set(data["partition"].unique()) == {0, 1}, "Partition should be binary {0, 1}"


@pytest.mark.parametrize("dgp_type", [1, 2, 3, 4])
def test_mp_dgp_types_work(dgp_type):
    result = gen_dgp_mult_periods(n=500, dgp_type=dgp_type, random_state=42)
    data = result["data"]

    assert len(data) > 0, f"DGP type {dgp_type} produced empty data"
    assert "data_wide" in result, "Missing data_wide"


def test_mp_dgp_wide_format():
    result = gen_dgp_mult_periods(n=500, dgp_type=1, random_state=42)
    data_wide = result["data_wide"]

    assert "id" in data_wide.columns, "Missing id in wide format"
    assert "group" in data_wide.columns, "Missing group in wide format"
    assert "partition" in data_wide.columns, "Missing partition in wide format"

    y_cols = [c for c in data_wide.columns if c.startswith("y_")]
    assert len(y_cols) > 0, "Missing outcome columns in wide format"


def test_mp_dgp_reproducibility():
    result1 = gen_dgp_mult_periods(n=500, dgp_type=1, random_state=42)
    result2 = gen_dgp_mult_periods(n=500, dgp_type=1, random_state=42)

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
    groups = result$group,
    times = result$t
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

    if len(r_att) != len(r_groups) or len(r_att) != len(r_times):
        assert len(py_result.att) > 0, f"RCS {est_method}: Python returned no ATTs"
        assert len(r_att) > 0, f"RCS {est_method}: R returned no ATTs"
        return

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

    assert len(py_result.att) > 0, f"RCS Python returned no ATTs for {control_group}"


@pytest.mark.skipif(not R_AVAILABLE, reason="R triplediff package not available")
@pytest.mark.parametrize("base_period", ["universal", "varying"])
def test_mp_rcs_base_period_options(mp_rcs_data, base_period):
    data = mp_rcs_data

    py_result = python_estimate_multiperiod_rcs(data, base_period=base_period, est_method="reg")
    r_result = r_estimate_multiperiod_rcs(data, base_period=base_period, est_method="reg")

    if r_result is None:
        pytest.fail("R RCS estimation failed")

    assert len(py_result.att) > 0, f"RCS Python returned no ATTs for {base_period}"


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
