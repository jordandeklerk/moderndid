"""Validation tests comparing Python cont_did implementation with R contdid package."""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

from tests.helpers import importorskip

pl = importorskip("polars")
np = importorskip("numpy")

from moderndid import cont_did, simulate_cont_did_data


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
            input='library(contdid); library(jsonlite); cat("OK")',
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return "OK" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


R_AVAILABLE = check_r_available()


def python_estimate_dose(data, target_parameter="level", control_group="notyettreated", degree=3, num_knots=1):
    return cont_did(
        data=data,
        yname="Y",
        tname="time_period",
        idname="id",
        gname="G",
        dname="D",
        target_parameter=target_parameter,
        aggregation="dose",
        treatment_type="continuous",
        dose_est_method="parametric",
        control_group=control_group,
        biters=100,
        cband=True,
        degree=degree,
        num_knots=num_knots,
        random_state=42,
    )


def python_estimate_eventstudy(data, target_parameter="level", control_group="notyettreated", degree=3, num_knots=1):
    return cont_did(
        data=data,
        yname="Y",
        tname="time_period",
        idname="id",
        gname="G",
        dname="D",
        target_parameter=target_parameter,
        aggregation="eventstudy",
        treatment_type="continuous",
        dose_est_method="parametric",
        control_group=control_group,
        biters=100,
        cband=True,
        degree=degree,
        num_knots=num_knots,
        random_state=42,
    )


def python_estimate_cck(data):
    return cont_did(
        data=data,
        yname="Y",
        tname="time_period",
        idname="id",
        gname="G",
        dname="D",
        target_parameter="level",
        aggregation="dose",
        treatment_type="continuous",
        dose_est_method="cck",
        control_group="notyettreated",
        biters=100,
        cband=True,
        random_state=42,
    )


def r_estimate_dose(data, target_parameter="level", control_group="notyettreated", degree=3, num_knots=1):
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "data.csv"
        result_path = Path(tmpdir) / "result.json"

        data.write_csv(data_path)

        r_script = f"""
library(contdid)
library(jsonlite)

set.seed(42)
data <- read.csv("{data_path}")

result <- cont_did(
    yname = "Y",
    dname = "D",
    gname = "G",
    tname = "time_period",
    idname = "id",
    data = data,
    target_parameter = "{target_parameter}",
    aggregation = "dose",
    treatment_type = "continuous",
    control_group = "{control_group}",
    bstrap = FALSE,
    degree = {degree},
    num_knots = {num_knots}
)

output <- list(
    overall_att = result$overall_att,
    overall_att_se = result$overall_att_se,
    overall_acrt = result$overall_acrt,
    overall_acrt_se = result$overall_acrt_se
)

write_json(output, "{result_path}", auto_unbox = TRUE, digits = 16)
"""
        try:
            return _run_r_script(r_script, result_path, timeout=180)
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
            return None


def r_estimate_eventstudy(data, target_parameter="level", control_group="notyettreated", degree=3, num_knots=1):
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "data.csv"
        result_path = Path(tmpdir) / "result.json"

        data.write_csv(data_path)

        r_script = f"""
library(contdid)
library(jsonlite)

set.seed(42)
data <- read.csv("{data_path}")

result <- cont_did(
    yname = "Y",
    dname = "D",
    gname = "G",
    tname = "time_period",
    idname = "id",
    data = data,
    target_parameter = "{target_parameter}",
    aggregation = "eventstudy",
    treatment_type = "continuous",
    control_group = "{control_group}",
    bstrap = FALSE,
    degree = {degree},
    num_knots = {num_knots}
)

output <- list(
    overall_att = result$event_study$overall.att,
    overall_se = result$event_study$overall.se,
    egt = as.list(result$event_study$egt),
    att_egt = as.list(result$event_study$att.egt),
    se_egt = as.list(result$event_study$se.egt)
)

write_json(output, "{result_path}", auto_unbox = TRUE, digits = 16)
"""
        try:
            return _run_r_script(r_script, result_path, timeout=180)
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
            return None


def r_estimate_cck(data):
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "data.csv"
        result_path = Path(tmpdir) / "result.json"

        data.write_csv(data_path)

        r_script = f"""
library(contdid)
library(jsonlite)

set.seed(42)
data <- read.csv("{data_path}")

result <- cont_did(
    yname = "Y",
    dname = "D",
    gname = "G",
    tname = "time_period",
    idname = "id",
    data = data,
    target_parameter = "level",
    aggregation = "dose",
    treatment_type = "continuous",
    dose_est_method = "cck",
    control_group = "notyettreated",
    bstrap = FALSE
)

output <- list(
    overall_att = result$overall_att,
    overall_att_se = result$overall_att_se,
    overall_acrt = result$overall_acrt,
    overall_acrt_se = result$overall_acrt_se
)

write_json(output, "{result_path}", auto_unbox = TRUE, digits = 16)
"""
        try:
            return _run_r_script(r_script, result_path, timeout=180)
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
            return None


def load_r_data():
    df = pl.read_csv("tests/didcont/data/cont_test_data.csv.gz")
    df = df.with_columns(pl.when(pl.col("G") == 0).then(0).otherwise(pl.col("D")).alias("D"))
    return df


def load_cck_data():
    df = pl.read_csv("tests/didcont/data/cont_test_data_cck.csv.gz")
    df = df.with_columns(pl.when(pl.col("G") == 0).then(0).otherwise(pl.col("D")).alias("D"))
    return df


@pytest.fixture
def r_test_data():
    return load_r_data()


@pytest.fixture
def r_test_data_cck():
    return load_cck_data()


@pytest.fixture
def cont_did_data():
    return simulate_cont_did_data(
        n=500,
        num_time_periods=4,
        dose_linear_effect=0.5,
        dose_quadratic_effect=0.1,
        seed=42,
    )


@pytest.fixture
def cont_did_data_cck():
    return simulate_cont_did_data(
        n=500,
        num_time_periods=2,
        dose_linear_effect=0.5,
        dose_quadratic_effect=0,
        seed=42,
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R contdid package not available")
@pytest.mark.parametrize("target_parameter", ["level", "slope"])
def test_dose_overall_att_matches(r_test_data, target_parameter):
    py_result = python_estimate_dose(r_test_data, target_parameter=target_parameter)
    r_result = r_estimate_dose(r_test_data, target_parameter=target_parameter)

    if r_result is None:
        pytest.skip("R estimation failed")

    np.testing.assert_allclose(
        py_result.overall_att,
        r_result["overall_att"],
        rtol=1e-6,
        atol=1e-6,
        err_msg=f"{target_parameter}: Overall ATT mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R contdid package not available")
@pytest.mark.parametrize("target_parameter", ["level", "slope"])
def test_dose_overall_se_matches(r_test_data, target_parameter):
    py_result = python_estimate_dose(r_test_data, target_parameter=target_parameter)
    r_result = r_estimate_dose(r_test_data, target_parameter=target_parameter)

    if r_result is None:
        pytest.skip("R estimation failed")

    np.testing.assert_allclose(
        py_result.overall_att_se,
        r_result["overall_att_se"],
        rtol=1.0,
        atol=0.05,
        err_msg=f"{target_parameter}: Overall SE mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R contdid package not available")
def test_dose_acrt_matches(r_test_data):
    py_result = python_estimate_dose(r_test_data, target_parameter="slope")
    r_result = r_estimate_dose(r_test_data, target_parameter="slope")

    if r_result is None:
        pytest.skip("R estimation failed")

    np.testing.assert_allclose(
        py_result.overall_acrt,
        r_result["overall_acrt"],
        rtol=0.01,
        atol=1e-3,
        err_msg="Overall ACRT mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R contdid package not available")
def test_dose_acrt_se_matches(r_test_data):
    py_result = python_estimate_dose(r_test_data, target_parameter="slope")
    r_result = r_estimate_dose(r_test_data, target_parameter="slope")

    if r_result is None:
        pytest.skip("R estimation failed")

    np.testing.assert_allclose(
        py_result.overall_acrt_se,
        r_result["overall_acrt_se"],
        rtol=0.1,
        atol=0.01,
        err_msg="Overall ACRT SE mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R contdid package not available")
@pytest.mark.parametrize("control_group", ["notyettreated", "nevertreated"])
def test_dose_control_group_options(r_test_data, control_group):
    py_result = python_estimate_dose(r_test_data, control_group=control_group)
    r_result = r_estimate_dose(r_test_data, control_group=control_group)

    if r_result is None:
        pytest.skip("R estimation failed")

    rtol = 1e-6 if control_group == "notyettreated" else 0.01
    atol = 1e-6 if control_group == "notyettreated" else 1e-3

    np.testing.assert_allclose(
        py_result.overall_att,
        r_result["overall_att"],
        rtol=rtol,
        atol=atol,
        err_msg=f"{control_group}: ATT mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R contdid package not available")
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_dose_degree_options(r_test_data, degree):
    py_result = python_estimate_dose(r_test_data, degree=degree, num_knots=0)
    r_result = r_estimate_dose(r_test_data, degree=degree, num_knots=0)

    if r_result is None:
        pytest.skip("R estimation failed")

    np.testing.assert_allclose(
        py_result.overall_att,
        r_result["overall_att"],
        rtol=1e-6,
        atol=1e-6,
        err_msg=f"degree={degree}: ATT mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R contdid package not available")
@pytest.mark.parametrize("num_knots", [0, 1, 2])
def test_dose_knots_options(r_test_data, num_knots):
    py_result = python_estimate_dose(r_test_data, degree=3, num_knots=num_knots)
    r_result = r_estimate_dose(r_test_data, degree=3, num_knots=num_knots)

    if r_result is None:
        pytest.skip("R estimation failed")

    np.testing.assert_allclose(
        py_result.overall_att,
        r_result["overall_att"],
        rtol=1e-6,
        atol=1e-6,
        err_msg=f"num_knots={num_knots}: ATT mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R contdid package not available")
@pytest.mark.parametrize("target_parameter", ["level", "slope"])
def test_eventstudy_overall_att_matches(r_test_data, target_parameter):
    py_result = python_estimate_eventstudy(r_test_data, target_parameter=target_parameter)
    r_result = r_estimate_eventstudy(r_test_data, target_parameter=target_parameter)

    if r_result is None:
        pytest.skip("R estimation failed")

    py_att = (
        py_result.overall_att.overall_att if hasattr(py_result.overall_att, "overall_att") else py_result.overall_att
    )

    rtol = 1e-6 if target_parameter == "level" else 0.01
    atol = 1e-6 if target_parameter == "level" else 1e-3

    np.testing.assert_allclose(
        py_att,
        r_result["overall_att"],
        rtol=rtol,
        atol=atol,
        err_msg=f"{target_parameter} eventstudy: Overall ATT mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R contdid package not available")
@pytest.mark.parametrize("target_parameter", ["level", "slope"])
def test_eventstudy_overall_se_matches(r_test_data, target_parameter):
    py_result = python_estimate_eventstudy(r_test_data, target_parameter=target_parameter)
    r_result = r_estimate_eventstudy(r_test_data, target_parameter=target_parameter)

    if r_result is None:
        pytest.skip("R estimation failed")

    py_se = (
        py_result.overall_att.overall_se if hasattr(py_result.overall_att, "overall_se") else py_result.overall_att_se
    )

    np.testing.assert_allclose(
        py_se,
        r_result["overall_se"],
        rtol=0.5,
        atol=0.1,
        err_msg=f"{target_parameter} eventstudy: Overall SE mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R contdid package not available")
def test_eventstudy_event_times_match(r_test_data):
    py_result = python_estimate_eventstudy(r_test_data, target_parameter="level")
    r_result = r_estimate_eventstudy(r_test_data, target_parameter="level")

    if r_result is None:
        pytest.skip("R estimation failed")

    if "egt" not in r_result or len(r_result["egt"]) == 0:
        pytest.skip("R result missing event times")

    py_event_times = set(py_result.event_study.event_times)
    r_event_times = set(r_result["egt"])

    assert py_event_times == r_event_times, f"Event times mismatch: Python={py_event_times}, R={r_event_times}"


@pytest.mark.skipif(not R_AVAILABLE, reason="R contdid package not available")
@pytest.mark.parametrize("target_parameter", ["level", "slope"])
def test_eventstudy_dynamic_effects_match(r_test_data, target_parameter):
    py_result = python_estimate_eventstudy(r_test_data, target_parameter=target_parameter)
    r_result = r_estimate_eventstudy(r_test_data, target_parameter=target_parameter)

    if r_result is None:
        pytest.skip("R estimation failed")

    if "egt" not in r_result or len(r_result["egt"]) == 0:
        pytest.skip("R result missing event times")

    r_event_times = np.array(r_result["egt"])
    r_att_by_event = np.array(r_result["att_egt"])

    py_event_times = py_result.event_study.event_times
    py_att_by_event = py_result.event_study.att_by_event

    rtol = 1e-6 if target_parameter == "level" else 0.1
    atol = 1e-6 if target_parameter == "level" else 0.01

    for e in set(py_event_times) & set(r_event_times):
        py_idx = np.where(py_event_times == e)[0]
        r_idx = np.where(r_event_times == e)[0]

        if len(py_idx) > 0 and len(r_idx) > 0:
            py_att = py_att_by_event[py_idx[0]]
            r_att = r_att_by_event[r_idx[0]]

            np.testing.assert_allclose(
                py_att,
                r_att,
                rtol=rtol,
                atol=atol,
                err_msg=f"{target_parameter} e={e}: ATT mismatch",
            )


@pytest.mark.skipif(not R_AVAILABLE, reason="R contdid package not available")
@pytest.mark.parametrize("target_parameter", ["level", "slope"])
def test_eventstudy_dynamic_se_match(r_test_data, target_parameter):
    py_result = python_estimate_eventstudy(r_test_data, target_parameter=target_parameter)
    r_result = r_estimate_eventstudy(r_test_data, target_parameter=target_parameter)

    if r_result is None:
        pytest.skip("R estimation failed")

    if "se_egt" not in r_result or len(r_result["se_egt"]) == 0:
        pytest.skip("R result missing se_egt")

    r_event_times = np.array(r_result["egt"])
    r_se_by_event = np.array(r_result["se_egt"])

    py_event_times = py_result.event_study.event_times
    py_se_by_event = py_result.event_study.se_by_event

    for e in set(py_event_times) & set(r_event_times):
        py_idx = np.where(py_event_times == e)[0]
        r_idx = np.where(r_event_times == e)[0]

        if len(py_idx) > 0 and len(r_idx) > 0:
            py_se = py_se_by_event[py_idx[0]]
            r_se = r_se_by_event[r_idx[0]]

            np.testing.assert_allclose(
                py_se,
                r_se,
                rtol=0.5,
                atol=0.1,
                err_msg=f"{target_parameter} e={e}: SE mismatch",
            )


@pytest.mark.skipif(not R_AVAILABLE, reason="R contdid package not available")
@pytest.mark.parametrize("control_group", ["notyettreated", "nevertreated"])
def test_eventstudy_control_group_options(r_test_data, control_group):
    py_result = python_estimate_eventstudy(r_test_data, control_group=control_group)
    r_result = r_estimate_eventstudy(r_test_data, control_group=control_group)

    if r_result is None:
        pytest.skip("R estimation failed")

    py_att = (
        py_result.overall_att.overall_att if hasattr(py_result.overall_att, "overall_att") else py_result.overall_att
    )

    rtol = 1e-6 if control_group == "notyettreated" else 0.05
    atol = 1e-6 if control_group == "notyettreated" else 0.01

    np.testing.assert_allclose(
        py_att,
        r_result["overall_att"],
        rtol=rtol,
        atol=atol,
        err_msg=f"{control_group}: ATT mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R contdid package not available")
def test_cck_overall_att_matches(r_test_data_cck):
    py_result = python_estimate_cck(r_test_data_cck)
    r_result = r_estimate_cck(r_test_data_cck)

    if r_result is None:
        pytest.skip("R CCK estimation failed")

    np.testing.assert_allclose(
        py_result.overall_att,
        r_result["overall_att"],
        rtol=0.05,
        atol=0.01,
        err_msg="CCK: Overall ATT mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R contdid package not available")
def test_cck_overall_att_se_matches(r_test_data_cck):
    py_result = python_estimate_cck(r_test_data_cck)
    r_result = r_estimate_cck(r_test_data_cck)

    if r_result is None:
        pytest.skip("R CCK estimation failed")

    np.testing.assert_allclose(
        py_result.overall_att_se,
        r_result["overall_att_se"],
        rtol=0.2,
        atol=0.05,
        err_msg="CCK: Overall ATT SE mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R contdid package not available")
def test_cck_overall_acrt_matches(r_test_data_cck):
    py_result = python_estimate_cck(r_test_data_cck)
    r_result = r_estimate_cck(r_test_data_cck)

    if r_result is None:
        pytest.skip("R CCK estimation failed")

    np.testing.assert_allclose(
        py_result.overall_acrt,
        r_result["overall_acrt"],
        rtol=0.1,
        atol=0.05,
        err_msg="CCK: Overall ACRT mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R contdid package not available")
def test_cck_overall_acrt_se_matches(r_test_data_cck):
    py_result = python_estimate_cck(r_test_data_cck)
    r_result = r_estimate_cck(r_test_data_cck)

    if r_result is None:
        pytest.skip("R CCK estimation failed")

    np.testing.assert_allclose(
        py_result.overall_acrt_se,
        r_result["overall_acrt_se"],
        rtol=0.2,
        atol=0.05,
        err_msg="CCK: Overall ACRT SE mismatch",
    )


def test_cont_did_returns_valid_structure(cont_did_data):
    result = python_estimate_dose(cont_did_data)

    assert hasattr(result, "overall_att"), "Missing overall_att"
    assert hasattr(result, "overall_att_se"), "Missing overall_att_se"
    assert hasattr(result, "overall_acrt"), "Missing overall_acrt"
    assert hasattr(result, "overall_acrt_se"), "Missing overall_acrt_se"
    assert hasattr(result, "att_d"), "Missing att_d"
    assert hasattr(result, "acrt_d"), "Missing acrt_d"


def test_cont_did_se_positive(cont_did_data):
    result = python_estimate_dose(cont_did_data)

    assert result.overall_att_se > 0, f"ATT SE must be positive, got {result.overall_att_se}"
    assert result.overall_acrt_se > 0, f"ACRT SE must be positive, got {result.overall_acrt_se}"


def test_cont_did_eventstudy_structure(cont_did_data):
    result = python_estimate_eventstudy(cont_did_data, target_parameter="level")

    assert hasattr(result, "event_study"), "Missing event_study"
    assert hasattr(result.event_study, "event_times"), "Missing event_times"
    assert hasattr(result.event_study, "att_by_event"), "Missing att_by_event"
    assert hasattr(result.event_study, "se_by_event"), "Missing se_by_event"


def test_cont_did_cck_requires_two_periods():
    data = simulate_cont_did_data(n=200, num_time_periods=4, seed=42)

    with pytest.raises(ValueError, match="2 groups and 2 time periods"):
        cont_did(
            data=data,
            yname="Y",
            tname="time_period",
            idname="id",
            gname="G",
            dname="D",
            dose_est_method="cck",
        )


def test_cont_did_missing_dname_raises():
    data = simulate_cont_did_data(n=100, seed=42)

    with pytest.raises(ValueError, match="dname is required"):
        cont_did(
            data=data,
            yname="Y",
            tname="time_period",
            idname="id",
            gname="G",
        )


def test_cont_did_invalid_data_type_raises():
    with pytest.raises(TypeError, match="Expected object implementing '__arrow_c_stream__'"):
        cont_did(
            data=[[1, 2, 3]],
            yname="Y",
            tname="time_period",
            idname="id",
            gname="G",
            dname="D",
        )


def test_cont_did_missing_columns_raises():
    data = pl.DataFrame({"id": [1, 2], "time": [1, 2], "y": [1.0, 2.0]})

    with pytest.raises(ValueError, match="Missing columns"):
        cont_did(
            data=data,
            yname="Y",
            tname="time_period",
            idname="id",
            gname="G",
            dname="D",
        )


@pytest.mark.parametrize("control_group", ["notyettreated", "nevertreated"])
def test_cont_did_control_group_options_work(cont_did_data, control_group):
    result = python_estimate_dose(cont_did_data, control_group=control_group)
    assert result.overall_att is not None


@pytest.mark.parametrize("target_parameter", ["level", "slope"])
def test_cont_did_target_parameter_options_work(cont_did_data, target_parameter):
    result = python_estimate_dose(cont_did_data, target_parameter=target_parameter)
    assert result.overall_att is not None


def test_cont_did_reproducible_with_seed(cont_did_data):
    result1 = cont_did(
        data=cont_did_data,
        yname="Y",
        tname="time_period",
        idname="id",
        gname="G",
        dname="D",
        biters=50,
        random_state=42,
    )
    result2 = cont_did(
        data=cont_did_data,
        yname="Y",
        tname="time_period",
        idname="id",
        gname="G",
        dname="D",
        biters=50,
        random_state=42,
    )

    np.testing.assert_allclose(result1.overall_att, result2.overall_att)


def test_simulate_cont_did_produces_valid_structure():
    data = simulate_cont_did_data(n=100, num_time_periods=4, seed=42)

    required_cols = ["id", "time_period", "Y", "G", "D"]
    for col in required_cols:
        assert col in data.columns, f"Missing column: {col}"


def test_simulate_cont_did_balanced_panel():
    n = 100
    num_periods = 4
    data = simulate_cont_did_data(n=n, num_time_periods=num_periods, seed=42)

    expected_rows = n * num_periods
    assert len(data) == expected_rows, f"Expected {expected_rows} rows, got {len(data)}"

    obs_per_unit = data.group_by("id").len()
    assert (obs_per_unit["len"] == num_periods).all(), "Panel is not balanced"


def test_simulate_cont_did_group_structure():
    data = simulate_cont_did_data(n=500, num_time_periods=4, seed=42)

    groups = data["G"].unique().to_numpy()

    assert 0 in groups, "Missing never-treated group (G=0)"
    assert len(groups) >= 2, "Need at least 2 groups"


def test_simulate_cont_did_dose_structure():
    data = simulate_cont_did_data(n=500, num_time_periods=4, seed=42)

    never_treated = data.filter(pl.col("G") == 0)
    assert (never_treated["D"] == 0).all(), "Never-treated units should have D=0"


def test_simulate_cont_did_reproducibility():
    data1 = simulate_cont_did_data(n=100, seed=42)
    data2 = simulate_cont_did_data(n=100, seed=42)

    assert data1.equals(data2), "Data should be identical with same seed"


@pytest.mark.parametrize("num_time_periods", [2, 4, 6])
def test_simulate_cont_did_different_periods(num_time_periods):
    data = simulate_cont_did_data(n=100, num_time_periods=num_time_periods, seed=42)

    actual_periods = data["time_period"].n_unique()
    assert actual_periods == num_time_periods, f"Expected {num_time_periods} periods, got {actual_periods}"


def test_cont_did_small_sample():
    data = simulate_cont_did_data(n=50, num_time_periods=3, seed=42)

    result = cont_did(
        data=data,
        yname="Y",
        tname="time_period",
        idname="id",
        gname="G",
        dname="D",
        biters=10,
    )
    assert result is not None


def test_cont_did_handles_zero_doses(cont_did_data):
    result = python_estimate_dose(cont_did_data)

    assert result.overall_att is not None
    assert not np.isnan(result.overall_att)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_cont_did_different_degree_options(degree):
    data = simulate_cont_did_data(n=200, num_time_periods=3, seed=42)

    result = cont_did(
        data=data,
        yname="Y",
        tname="time_period",
        idname="id",
        gname="G",
        dname="D",
        degree=degree,
        num_knots=0,
        biters=10,
    )
    assert result is not None, f"Failed with degree={degree}"


@pytest.mark.parametrize("num_knots", [0, 1, 2])
def test_cont_did_different_knot_options(num_knots):
    data = simulate_cont_did_data(n=200, num_time_periods=3, seed=42)

    result = cont_did(
        data=data,
        yname="Y",
        tname="time_period",
        idname="id",
        gname="G",
        dname="D",
        degree=3,
        num_knots=num_knots,
        biters=10,
    )
    assert result is not None, f"Failed with num_knots={num_knots}"


@pytest.mark.parametrize(
    "param,value",
    [
        ("aggregation", "invalid"),
        ("target_parameter", "invalid"),
        ("dose_est_method", "invalid"),
        ("control_group", "invalid"),
    ],
)
def test_cont_did_invalid_params(param, value):
    data = simulate_cont_did_data(n=100, seed=42)
    with pytest.raises(ValueError, match=f"{param}='invalid' is not valid"):
        cont_did(
            data=data,
            yname="Y",
            tname="time_period",
            idname="id",
            gname="G",
            dname="D",
            **{param: value},
        )
