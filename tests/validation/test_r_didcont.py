"""Validation tests comparing Python cont_did implementation with R contdid package."""

import functools
import json
import subprocess
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

from tests.helpers import importorskip

pl = importorskip("polars")
np = importorskip("numpy")

from moderndid import cont_did, gen_cont_did_data
from moderndid.didcont.estimation import pte_default


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


def check_r_available(package):
    try:
        result = subprocess.run(
            ["R", "--vanilla", "--quiet"],
            input=f'library({package}); library(jsonlite); cat("OK")',
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return "OK" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


R_AVAILABLE = check_r_available("contdid")
R_DID_AVAILABLE = check_r_available("did")


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


def python_estimate_two_period_dose(data, dvals=None):
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
        dose_est_method="parametric",
        control_group="notyettreated",
        biters=100,
        cband=True,
        degree=3,
        num_knots=0,
        dvals=dvals,
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
    knots = as.list(as.numeric(result$pte_params$knots)),
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
    ids = unique(result$pte_params$data$id),
    overall_att = result$overall_att,
    overall_att_se = result$overall_att_se,
    overall_att_inf_func = as.numeric(result$overall_att_inffunc),
    overall_acrt = result$overall_acrt,
    overall_acrt_se = result$overall_acrt_se
)

write_json(output, "{result_path}", auto_unbox = TRUE, digits = NA)
"""
        try:
            return _run_r_script(r_script, result_path, timeout=180)
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
            return None


@functools.cache
def r_binary_att_reference(control_group):
    data = load_r_data()
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "data.csv"
        result_path = Path(tmpdir) / "result.json"

        data.write_csv(data_path)

        r_script = f"""
library(ptetools)
library(jsonlite)

set.seed(42)
data <- read.csv("{data_path}")

result <- pte_default(
    yname = "Y",
    gname = "G",
    tname = "time_period",
    idname = "id",
    data = data,
    d_outcome = TRUE,
    control_group = "{control_group}",
    biters = 10
)

first <- data[data$time_period == min(data$time_period), ]
glist <- sort(unique(result$att_gt$group))

output <- list(
    ids = unique(result$ptep$data$id),
    overall_att = result$overall_att$overall.att,
    inf_func = as.numeric(result$overall_att$inf.function$selective.inf.func),
    inf_func_by_group = result$overall_att$inf.function$selective.inf.func.g,
    pg = sapply(glist, function(g) mean(first$G == g))
)

write_json(output, "{result_path}", auto_unbox = TRUE, digits = NA)
"""
        try:
            return _run_r_script(r_script, result_path, timeout=180)
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
            return None


@functools.cache
def r_eventstudy_reference(target_parameter, control_group="notyettreated"):
    data = load_r_data()
    attgt_fun = "contdid::cont_did_acrt" if target_parameter == "slope" else "ptetools::did_attgt"
    subset_fun = "contdid::cont_two_by_two_subset" if target_parameter == "slope" else "ptetools::two_by_two_subset"
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "data.csv"
        result_path = Path(tmpdir) / "result.json"

        data.write_csv(data_path)

        r_script = f"""
library(contdid)
library(ptetools)
library(jsonlite)

set.seed(42)
data <- read.csv("{data_path}")

setup_with_control_group <- function(...) {{
    ptep <- contdid::setup_pte_cont(...)
    ptep$control_group <- "{control_group}"
    ptep
}}

result <- ptetools::pte(
    yname = "Y",
    gname = "G",
    tname = "time_period",
    idname = "id",
    data = data,
    setup_pte_fun = setup_with_control_group,
    subset_fun = {subset_fun},
    attgt_fun = {attgt_fun},
    xformla = ~1,
    target_parameter = "{target_parameter}",
    aggregation = "eventstudy",
    treatment_type = "continuous",
    dose_est_method = "parametric",
    anticipation = 0,
    gt_type = "att",
    cband = FALSE,
    alp = 0.05,
    boot_type = "multiplier",
    biters = 10,
    cl = 1,
    dname = "D",
    degree = 3,
    num_knots = 0,
    dvals = NULL,
    control_group = "{control_group}"
)

event_study <- result$event_study
output <- list(
    ids = unique(result$ptep$data$id),
    groups = result$att_gt$group,
    times = result$att_gt$t,
    att_gt = result$att_gt$att,
    cell_inf_func = result$att_gt$inf_func,
    egt = event_study$egt,
    att_egt = event_study$att.egt,
    inf_func_by_event = event_study$inf.function$dynamic.inf.func.e,
    overall_att = event_study$overall.att,
    overall_inf_func = as.numeric(event_study$inf.function$dynamic.inf.func)
)

write_json(output, "{result_path}", auto_unbox = TRUE, digits = NA)
"""
        try:
            return _run_r_script(r_script, result_path, timeout=300)
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
            return None


@functools.cache
def r_two_period_dose_reference(dvals):
    data = load_cck_data()
    dvals_r = ", ".join(repr(float(value)) for value in dvals)
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
    control_group = "notyettreated",
    degree = 3,
    num_knots = 0,
    dvals = c({dvals_r}),
    biters = 10
)

output <- list(
    ids = unique(result$pte_params$data$id),
    overall_att = result$overall_att,
    overall_att_inf_func = as.numeric(result$overall_att_inffunc),
    overall_acrt = result$overall_acrt,
    overall_acrt_inf_func = as.numeric(result$overall_acrt_inffunc),
    att_d = as.numeric(result$att.d),
    att_d_inf_func = result$att.d_inffunc,
    acrt_d = as.numeric(result$acrt.d),
    acrt_d_inf_func = result$acrt.d_inffunc
)

write_json(output, "{result_path}", auto_unbox = TRUE, digits = NA)
"""
        try:
            return _run_r_script(r_script, result_path, timeout=300)
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
            return None


@functools.cache
def r_att_gt_reference(variant, anticipation=0, base_period="varying"):
    data = r_test_data_variant(variant)
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "data.csv"
        result_path = Path(tmpdir) / "result.json"

        data.write_csv(data_path)

        r_script = f"""
library(did)
library(jsonlite)

data <- read.csv("{data_path}")

result <- att_gt(
    yname = "Y",
    tname = "time_period",
    idname = "id",
    gname = "G",
    data = data,
    control_group = "notyettreated",
    anticipation = {anticipation},
    base_period = "{base_period}",
    bstrap = FALSE,
    cband = FALSE
)
dynamic <- aggte(result, type = "dynamic", bstrap = FALSE, cband = FALSE, na.rm = TRUE)
group <- aggte(result, type = "group", bstrap = FALSE, cband = FALSE, na.rm = TRUE)

output <- list(
    groups = result$group,
    times = result$t,
    att_gt = result$att,
    egt = dynamic$egt,
    att_egt = dynamic$att.egt,
    se_egt = dynamic$se.egt,
    group_overall = group$overall.att,
    group_overall_se = group$overall.se
)

write_json(output, "{result_path}", auto_unbox = TRUE, digits = NA, na = "null")
"""
        try:
            return _run_r_script(r_script, result_path, timeout=180)
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
            return None


def in_sorted_id_order(r_result, key):
    values = np.asarray(r_result[key], dtype=float)
    return values[np.argsort(np.asarray(r_result["ids"]))]


def first_period_groups(data):
    return data.filter(pl.col("time_period") == data["time_period"].min()).sort("id")["G"].to_numpy()


def analytic_se(inf_func):
    inf_func = np.asarray(inf_func, dtype=float).reshape(len(inf_func), -1)
    return np.sqrt(np.sum(inf_func**2, axis=0)) / inf_func.shape[0]


def reference_event_study_inf_func(r_result, groups, target_parameter):
    cells = in_sorted_id_order(r_result, "cell_inf_func")
    by_event = in_sorted_id_order(r_result, "inf_func_by_event")
    cell_groups = np.asarray(r_result["groups"], dtype=float)
    cell_times = np.asarray(r_result["times"], dtype=float)
    pg = {g: np.mean(groups == g) for g in np.unique(cell_groups)}
    p_treated = np.mean(groups > 0)

    scale = np.ones(len(cell_groups))
    if target_parameter == "slope":
        n_treated = np.array([np.sum(groups == g) for g in cell_groups])
        n_cell = np.array(
            [np.sum((groups == g) | (groups == 0) | (groups > t)) for g, t in zip(cell_groups, cell_times)]
        )
        scale = n_cell / n_treated

    expected = np.zeros_like(by_event)
    for j, e in enumerate(np.asarray(r_result["egt"], dtype=float)):
        in_event = np.flatnonzero(cell_times - cell_groups == e)
        weights = np.array([pg[g] for g in cell_groups[in_event]])
        weights = weights / weights.sum()
        reference_main = cells[:, in_event] @ weights
        expected[:, j] = (cells[:, in_event] * scale[in_event]) @ weights + (
            by_event[:, j] - reference_main
        ) / p_treated
    return expected


def load_r_data():
    df = pl.read_csv("tests/didcont/data/cont_test_data.csv.gz")
    df = df.with_columns(pl.when(pl.col("G") == 0).then(0).otherwise(pl.col("D")).alias("D"))
    return df


def r_test_data_variant(variant):
    data = load_r_data()
    if variant == "years":
        return data.with_columns(
            (pl.col("time_period") + 2000).alias("time_period"),
            pl.when(pl.col("G") > 0).then(pl.col("G") + 2000).otherwise(0).alias("G"),
        )
    if variant == "doubled":
        return data.with_columns((pl.col("time_period") * 2).alias("time_period"), (pl.col("G") * 2).alias("G"))
    if variant == "treated_only":
        return data.filter(pl.col("G") > 0)
    return data


def cells(groups, times, att):
    return dict(zip(zip(np.asarray(groups, dtype=float).tolist(), np.asarray(times, dtype=float).tolist()), att))


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
    return gen_cont_did_data(
        n=500,
        num_time_periods=4,
        dose_linear_effect=0.5,
        dose_quadratic_effect=0.1,
        seed=42,
    )


@pytest.fixture
def cont_did_data_cck():
    return gen_cont_did_data(
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
        pytest.fail("R estimation failed")

    np.testing.assert_allclose(
        py_result.overall_att,
        r_result["overall_att"],
        rtol=1e-6,
        atol=1e-6,
        err_msg=f"{target_parameter}: Overall ATT mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R contdid package not available")
@pytest.mark.parametrize("control_group", ["notyettreated", "nevertreated"])
def test_dose_overall_se_matches(r_test_data, control_group):
    py_result = python_estimate_dose(r_test_data, control_group=control_group)
    r_result = r_binary_att_reference(control_group)

    if r_result is None:
        pytest.fail("R estimation failed")

    total = in_sorted_id_order(r_result, "inf_func")
    by_group = in_sorted_id_order(r_result, "inf_func_by_group")
    pg = np.asarray(r_result["pg"])
    main = by_group @ (pg / pg.sum())
    expected = main + (total - main) / pg.sum()

    np.testing.assert_allclose(py_result.overall_att_inf_func, expected, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(analytic_se(py_result.overall_att_inf_func), analytic_se(expected), rtol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R contdid package not available")
def test_dose_acrt_matches(r_test_data):
    py_result = python_estimate_dose(r_test_data, target_parameter="slope")
    r_result = r_estimate_dose(r_test_data, target_parameter="slope")

    if r_result is None:
        pytest.fail("R estimation failed")

    np.testing.assert_allclose(py_result.pte_params.knots, r_result["knots"], rtol=1e-12)
    np.testing.assert_allclose(
        py_result.overall_acrt,
        r_result["overall_acrt"],
        rtol=1e-8,
        atol=1e-10,
        err_msg="Overall ACRT mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R contdid package not available")
def test_dose_acrt_se_matches(r_test_data_cck):
    py_result = python_estimate_two_period_dose(r_test_data_cck)
    r_result = r_two_period_dose_reference(tuple(py_result.dose))

    if r_result is None:
        pytest.fail("R estimation failed")

    groups = first_period_groups(r_test_data_cck)
    scale = len(groups) / np.sum(groups > 0)
    expected = scale * in_sorted_id_order(r_result, "overall_acrt_inf_func")

    np.testing.assert_allclose(py_result.overall_acrt, r_result["overall_acrt"], rtol=1e-10)
    np.testing.assert_allclose(py_result.overall_acrt_inf_func, expected, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(analytic_se(py_result.overall_acrt_inf_func), analytic_se(expected), rtol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R contdid package not available")
def test_dose_acrt_inf_func_combines_reference_cells(r_test_data):
    py_result = python_estimate_dose(r_test_data, target_parameter="slope", num_knots=0)
    r_result = r_eventstudy_reference("slope")

    if r_result is None:
        pytest.fail("R estimation failed")

    groups = first_period_groups(r_test_data)
    cell_groups = np.asarray(r_result["groups"], dtype=float)
    cell_times = np.asarray(r_result["times"], dtype=float)
    cell_acrt = np.asarray(r_result["att_gt"], dtype=float)
    n_treated = np.array([np.sum(groups == g) for g in cell_groups])
    n_cell = np.array([np.sum((groups == g) | (groups == 0) | (groups > t)) for g, t in zip(cell_groups, cell_times)])
    cell_inf_func = in_sorted_id_order(r_result, "cell_inf_func") * (n_cell / n_treated)

    post = cell_times >= cell_groups
    cohorts = np.unique(cell_groups[post])
    in_cohort = (groups[:, None] == cohorts).astype(float)
    pg = in_cohort.mean(axis=0)
    n_post = np.array([np.sum(post & (cell_groups == g)) for g in cohorts])
    cohort_of_cell = np.searchsorted(cohorts, cell_groups[post])
    weights = (pg / pg.sum())[cohort_of_cell] / n_post[cohort_of_cell]
    share_inf_func = (in_cohort - pg) / pg.sum() - (in_cohort - pg).sum(axis=1, keepdims=True) * pg / pg.sum() ** 2
    share_term = (share_inf_func[:, cohort_of_cell] / n_post[cohort_of_cell]) @ cell_acrt[post]
    expected = cell_inf_func[:, post] @ weights + share_term

    np.testing.assert_allclose(py_result.overall_acrt, weights @ cell_acrt[post], rtol=1e-10)
    np.testing.assert_allclose(py_result.overall_acrt_inf_func, expected, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(analytic_se(py_result.overall_acrt_inf_func), analytic_se(expected), rtol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R contdid package not available")
def test_two_period_dose_influence_functions_match(r_test_data_cck):
    py_result = python_estimate_two_period_dose(r_test_data_cck)
    r_result = r_two_period_dose_reference(tuple(py_result.dose))

    if r_result is None:
        pytest.fail("R estimation failed")

    treated = first_period_groups(r_test_data_cck) > 0
    r_att_d_inf_func = in_sorted_id_order(r_result, "att_d_inf_func")

    np.testing.assert_allclose(py_result.overall_att, r_result["overall_att"], rtol=1e-10)
    np.testing.assert_allclose(
        py_result.overall_att_inf_func, in_sorted_id_order(r_result, "overall_att_inf_func"), rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(py_result.att_d, r_result["att_d"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(py_result.acrt_d, r_result["acrt_d"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(py_result.att_d_inf_func[treated], r_att_d_inf_func[treated], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(py_result.att_d_inf_func[~treated], -r_att_d_inf_func[~treated], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(analytic_se(py_result.att_d_inf_func), analytic_se(r_att_d_inf_func), rtol=1e-10)
    np.testing.assert_allclose(
        py_result.acrt_d_inf_func, in_sorted_id_order(r_result, "acrt_d_inf_func"), rtol=1e-8, atol=1e-10
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R contdid package not available")
@pytest.mark.parametrize("control_group", ["notyettreated", "nevertreated"])
def test_dose_control_group_options(r_test_data, control_group):
    py_result = python_estimate_dose(r_test_data, control_group=control_group)
    r_result = r_binary_att_reference(control_group)

    if r_result is None:
        pytest.fail("R estimation failed")

    np.testing.assert_allclose(
        py_result.overall_att,
        r_result["overall_att"],
        rtol=1e-10,
        atol=1e-12,
        err_msg=f"{control_group}: ATT mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R contdid package not available")
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_dose_degree_options(r_test_data, degree):
    py_result = python_estimate_dose(r_test_data, degree=degree, num_knots=0)
    r_result = r_estimate_dose(r_test_data, degree=degree, num_knots=0)

    if r_result is None:
        pytest.fail("R estimation failed")

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
        pytest.fail("R estimation failed")

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
    if target_parameter == "slope":
        r_result = r_estimate_eventstudy(r_test_data, target_parameter=target_parameter)
    else:
        r_result = r_eventstudy_reference("level")

    if r_result is None:
        pytest.fail("R estimation failed")

    np.testing.assert_allclose(
        py_result.overall_att.overall_att,
        r_result["overall_att"],
        rtol=1e-8,
        atol=1e-10,
        err_msg=f"{target_parameter} eventstudy: Overall ATT mismatch",
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R contdid package not available")
@pytest.mark.parametrize("target_parameter", ["level", "slope"])
def test_eventstudy_overall_se_matches(r_test_data, target_parameter):
    py_result = python_estimate_eventstudy(r_test_data, target_parameter=target_parameter, num_knots=0)
    r_result = r_eventstudy_reference(target_parameter)

    if r_result is None:
        pytest.fail("R estimation failed")

    expected_by_event = reference_event_study_inf_func(r_result, first_period_groups(r_test_data), target_parameter)
    post = np.asarray(r_result["egt"]) >= 0
    expected = expected_by_event[:, post].mean(axis=1)

    np.testing.assert_allclose(py_result.event_study.overall_att, r_result["overall_att"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(py_result.event_study.influence_func["overall"], expected, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        analytic_se(py_result.event_study.influence_func["overall"]), analytic_se(expected), rtol=1e-10
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R contdid package not available")
def test_eventstudy_event_times_match(r_test_data):
    py_result = python_estimate_eventstudy(r_test_data, target_parameter="level")
    r_result = r_estimate_eventstudy(r_test_data, target_parameter="level")

    if r_result is None:
        pytest.fail("R estimation failed")

    if "egt" not in r_result or len(r_result["egt"]) == 0:
        pytest.fail("R result missing event times")

    py_event_times = set(py_result.event_study.event_times)
    r_event_times = set(r_result["egt"])

    assert py_event_times == r_event_times, f"Event times mismatch: Python={py_event_times}, R={r_event_times}"


@pytest.mark.skipif(not R_AVAILABLE, reason="R contdid package not available")
@pytest.mark.parametrize("target_parameter", ["level", "slope"])
def test_eventstudy_dynamic_effects_match(r_test_data, target_parameter):
    py_result = python_estimate_eventstudy(r_test_data, target_parameter=target_parameter)
    if target_parameter == "slope":
        r_result = r_estimate_eventstudy(r_test_data, target_parameter=target_parameter)
    else:
        r_result = r_eventstudy_reference("level")

    if r_result is None:
        pytest.fail("R estimation failed")

    if "egt" not in r_result or len(r_result["egt"]) == 0:
        pytest.fail("R result missing event times")

    r_event_times = np.array(r_result["egt"])
    r_att_by_event = np.array(r_result["att_egt"])

    py_event_times = py_result.event_study.event_times
    py_att_by_event = py_result.event_study.att_by_event

    rtol = 1e-8
    atol = 1e-10

    np.testing.assert_array_equal(py_event_times, r_event_times)
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
    py_result = python_estimate_eventstudy(r_test_data, target_parameter=target_parameter, num_knots=0)
    r_result = r_eventstudy_reference(target_parameter)

    if r_result is None:
        pytest.fail("R estimation failed")

    expected = reference_event_study_inf_func(r_result, first_period_groups(r_test_data), target_parameter)

    np.testing.assert_array_equal(py_result.event_study.event_times, np.asarray(r_result["egt"]))
    np.testing.assert_allclose(py_result.event_study.att_by_event, r_result["att_egt"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(py_result.event_study.influence_func["by_event"], expected, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        analytic_se(py_result.event_study.influence_func["by_event"]), analytic_se(expected), rtol=1e-10
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R contdid package not available")
def test_eventstudy_slope_cell_inf_func_scaled_by_treated_count(r_test_data):
    py_result = python_estimate_eventstudy(r_test_data, target_parameter="slope", num_knots=0)
    r_result = r_eventstudy_reference("slope")

    if r_result is None:
        pytest.fail("R estimation failed")

    groups = first_period_groups(r_test_data)
    cell_groups = np.asarray(r_result["groups"], dtype=float)
    cell_times = np.asarray(r_result["times"], dtype=float)
    n_treated = np.array([np.sum(groups == g) for g in cell_groups])
    n_cell = np.array([np.sum((groups == g) | (groups == 0) | (groups > t)) for g, t in zip(cell_groups, cell_times)])
    expected = in_sorted_id_order(r_result, "cell_inf_func") * (n_cell / n_treated)

    np.testing.assert_array_equal(py_result.att_gt.groups, cell_groups)
    np.testing.assert_array_equal(py_result.att_gt.times, cell_times)
    np.testing.assert_allclose(py_result.att_gt.att, r_result["att_gt"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(py_result.att_gt.influence_func, expected, rtol=1e-8, atol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R contdid package not available")
@pytest.mark.parametrize("control_group", ["notyettreated", "nevertreated"])
def test_eventstudy_control_group_options(r_test_data, control_group):
    py_result = python_estimate_eventstudy(r_test_data, control_group=control_group)
    r_result = r_eventstudy_reference("level", control_group)

    if r_result is None:
        pytest.fail("R estimation failed")

    np.testing.assert_allclose(
        py_result.event_study.overall_att,
        r_result["overall_att"],
        rtol=1e-10,
        atol=1e-12,
        err_msg=f"{control_group}: ATT mismatch",
    )
    np.testing.assert_allclose(py_result.event_study.att_by_event, r_result["att_egt"], rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not R_DID_AVAILABLE, reason="R did package not available")
@pytest.mark.parametrize("variant", ["years", "doubled"])
def test_eventstudy_level_period_coding_matches_att_gt(variant):
    data = r_test_data_variant(variant)
    py_result = python_estimate_eventstudy(data, target_parameter="level")
    r_result = r_att_gt_reference(variant)

    if r_result is None:
        pytest.fail("R estimation failed")

    expected = cells(r_result["groups"], r_result["times"], r_result["att_gt"])
    observed = cells(py_result.att_gt.groups, py_result.att_gt.times, py_result.att_gt.att)

    assert observed.keys() == expected.keys()
    np.testing.assert_allclose([observed[c] for c in expected], list(expected.values()), rtol=1e-8, atol=1e-10)
    np.testing.assert_array_equal(py_result.event_study.event_times, np.asarray(r_result["egt"]))
    np.testing.assert_allclose(py_result.event_study.att_by_event, r_result["att_egt"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        analytic_se(py_result.event_study.influence_func["by_event"]), r_result["se_egt"], rtol=1e-8
    )


@pytest.mark.skipif(not R_DID_AVAILABLE, reason="R did package not available")
@pytest.mark.filterwarnings("ignore:Simultaneous band smaller than pointwise:UserWarning")
@pytest.mark.parametrize("variant", ["years", "doubled"])
def test_pte_default_group_column_named_g_matches_att_gt(variant):
    py_result = pte_default(
        yname="Y",
        gname="G",
        tname="time_period",
        idname="id",
        data=r_test_data_variant(variant),
        d_outcome=True,
        biters=10,
        random_state=42,
    )
    r_result = r_att_gt_reference(variant)

    if r_result is None:
        pytest.fail("R estimation failed")

    np.testing.assert_array_equal(py_result.event_study.event_times, np.asarray(r_result["egt"]))
    np.testing.assert_allclose(py_result.event_study.att_by_event, r_result["att_egt"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        analytic_se(py_result.event_study.influence_func["by_event"]), r_result["se_egt"], rtol=1e-8
    )


@pytest.mark.skipif(not R_DID_AVAILABLE, reason="R did package not available")
@pytest.mark.filterwarnings("ignore:Dropped 1 groups treated before period:UserWarning")
def test_eventstudy_level_anticipation_matches_att_gt(r_test_data):
    py_result = cont_did(
        data=r_test_data,
        yname="Y",
        tname="time_period",
        idname="id",
        gname="G",
        dname="D",
        aggregation="eventstudy",
        anticipation=1,
        biters=100,
        random_state=42,
    )
    r_result = r_att_gt_reference("base", anticipation=1)

    if r_result is None:
        pytest.fail("R estimation failed")

    expected = cells(r_result["groups"], r_result["times"], r_result["att_gt"])
    observed = cells(py_result.att_gt.groups, py_result.att_gt.times, py_result.att_gt.att)

    assert observed.keys() == expected.keys()
    np.testing.assert_allclose([observed[c] for c in expected], list(expected.values()), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(py_result.event_study.att_by_event, r_result["att_egt"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        analytic_se(py_result.event_study.influence_func["by_event"]), r_result["se_egt"], rtol=1e-8
    )


@pytest.mark.skipif(not R_DID_AVAILABLE, reason="R did package not available")
@pytest.mark.filterwarnings("ignore:Dropped 1 groups treated before period:UserWarning")
def test_dose_overall_att_with_anticipation_matches_att_gt(r_test_data):
    py_result = cont_did(
        data=r_test_data,
        yname="Y",
        tname="time_period",
        idname="id",
        gname="G",
        dname="D",
        anticipation=1,
        degree=3,
        num_knots=0,
        biters=100,
        random_state=42,
    )
    r_result = r_att_gt_reference("base", anticipation=1)

    if r_result is None:
        pytest.fail("R estimation failed")

    np.testing.assert_allclose(py_result.overall_att, r_result["group_overall"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(analytic_se(py_result.overall_att_inf_func), r_result["group_overall_se"], rtol=1e-8)


@pytest.mark.skipif(not R_DID_AVAILABLE, reason="R did package not available")
def test_eventstudy_level_universal_base_matches_att_gt(r_test_data):
    py_result = cont_did(
        data=r_test_data,
        yname="Y",
        tname="time_period",
        idname="id",
        gname="G",
        dname="D",
        aggregation="eventstudy",
        base_period="universal",
        biters=100,
        random_state=42,
    )
    r_result = r_att_gt_reference("base", base_period="universal")

    if r_result is None:
        pytest.fail("R estimation failed")

    expected_se = np.asarray(r_result["se_egt"], dtype=float)
    observed_se = analytic_se(py_result.event_study.influence_func["by_event"])
    reference_period = np.isnan(expected_se)

    np.testing.assert_array_equal(py_result.event_study.event_times, np.asarray(r_result["egt"]))
    np.testing.assert_allclose(py_result.event_study.att_by_event, r_result["att_egt"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(observed_se[~reference_period], expected_se[~reference_period], rtol=1e-8)
    np.testing.assert_array_equal(observed_se[reference_period], 0.0)


@pytest.mark.skipif(not R_DID_AVAILABLE, reason="R did package not available")
@pytest.mark.filterwarnings("ignore:The data has no never-treated units:UserWarning")
def test_dose_without_never_treated_units_matches_att_gt():
    data = r_test_data_variant("treated_only")
    py_result = python_estimate_dose(data, num_knots=0)
    r_result = r_att_gt_reference("treated_only")

    if r_result is None:
        pytest.fail("R estimation failed")

    np.testing.assert_allclose(py_result.overall_att, r_result["group_overall"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(analytic_se(py_result.overall_att_inf_func), r_result["group_overall_se"], rtol=1e-8)


@pytest.mark.skipif(not R_AVAILABLE, reason="R contdid package not available")
def test_cck_overall_att_matches(r_test_data_cck):
    py_result = python_estimate_cck(r_test_data_cck)
    r_result = r_estimate_cck(r_test_data_cck)

    if r_result is None:
        pytest.fail("R CCK estimation failed")

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
        pytest.fail("R CCK estimation failed")

    expected = in_sorted_id_order(r_result, "overall_att_inf_func")

    np.testing.assert_allclose(py_result.overall_att_inf_func, expected, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(analytic_se(py_result.overall_att_inf_func), analytic_se(expected), rtol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R contdid package not available")
def test_cck_overall_acrt_matches(r_test_data_cck):
    py_result = python_estimate_cck(r_test_data_cck)
    r_result = r_estimate_cck(r_test_data_cck)

    if r_result is None:
        pytest.fail("R CCK estimation failed")

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
        pytest.fail("R CCK estimation failed")

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
    data = gen_cont_did_data(n=200, num_time_periods=4, seed=42)

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
    data = gen_cont_did_data(n=100, seed=42)

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
    data = gen_cont_did_data(n=100, num_time_periods=4, seed=42)

    required_cols = ["id", "time_period", "Y", "G", "D"]
    for col in required_cols:
        assert col in data.columns, f"Missing column: {col}"


def test_simulate_cont_did_balanced_panel():
    n = 100
    num_periods = 4
    data = gen_cont_did_data(n=n, num_time_periods=num_periods, seed=42)

    expected_rows = n * num_periods
    assert len(data) == expected_rows, f"Expected {expected_rows} rows, got {len(data)}"

    obs_per_unit = data.group_by("id").len()
    assert (obs_per_unit["len"] == num_periods).all(), "Panel is not balanced"


def test_simulate_cont_did_group_structure():
    data = gen_cont_did_data(n=500, num_time_periods=4, seed=42)

    groups = data["G"].unique().to_numpy()

    assert 0 in groups, "Missing never-treated group (G=0)"
    assert len(groups) >= 2, "Need at least 2 groups"


def test_simulate_cont_did_dose_structure():
    data = gen_cont_did_data(n=500, num_time_periods=4, seed=42)

    never_treated = data.filter(pl.col("G") == 0)
    assert (never_treated["D"] == 0).all(), "Never-treated units should have D=0"


def test_simulate_cont_did_reproducibility():
    data1 = gen_cont_did_data(n=100, seed=42)
    data2 = gen_cont_did_data(n=100, seed=42)

    assert data1.equals(data2), "Data should be identical with same seed"


@pytest.mark.parametrize("num_time_periods", [2, 4, 6])
def test_simulate_cont_did_different_periods(num_time_periods):
    data = gen_cont_did_data(n=100, num_time_periods=num_time_periods, seed=42)

    actual_periods = data["time_period"].n_unique()
    assert actual_periods == num_time_periods, f"Expected {num_time_periods} periods, got {actual_periods}"


def test_cont_did_small_sample():
    data = gen_cont_did_data(n=50, num_time_periods=3, seed=42)

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
    data = gen_cont_did_data(n=200, num_time_periods=3, seed=42)

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
    data = gen_cont_did_data(n=200, num_time_periods=3, seed=42)

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
    data = gen_cont_did_data(n=100, seed=42)
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
