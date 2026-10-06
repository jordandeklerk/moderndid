"""Validation tests comparing Python didhonest implementation with R HonestDiD package."""

import json
import subprocess
import tempfile
from functools import partial

import pytest
from scipy import stats

pytestmark = pytest.mark.slow

from tests.helpers import importorskip

pl = importorskip("polars")
np = importorskip("numpy")

from moderndid.didhonest import (
    basis_vector,
    compute_conditional_cs_rm,
    compute_conditional_cs_rmb,
    compute_conditional_cs_rmm,
    compute_conditional_cs_sd,
    compute_conditional_cs_sdb,
    compute_conditional_cs_sdm,
    compute_conditional_cs_sdrm,
    compute_conditional_cs_sdrmb,
    compute_conditional_cs_sdrmm,
    compute_delta_sd_lowerbound_m,
    compute_delta_sd_upperbound_m,
    compute_flci,
    compute_identified_set_rm,
    compute_identified_set_rmb,
    compute_identified_set_rmm,
    compute_identified_set_sd,
    compute_identified_set_sdrm,
    compute_identified_set_sdrmb,
    compute_identified_set_sdrmm,
    compute_vlo_vup_dual,
    construct_original_cs,
    create_sensitivity_results_rm,
    create_sensitivity_results_sm,
)
from moderndid.didhonest.arp_nuisance import _construct_gamma, _test_delta_lp
from moderndid.didhonest.conditional import _norminvp_generalized
from moderndid.didhonest.delta.rm.rmb import _create_relative_magnitudes_bias_constraint_matrix
from moderndid.didhonest.numba import find_rows_with_post_period_values


def _run_r_script(r_script, result_path, timeout=300):
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


def check_r_honestdid_available():
    try:
        result = subprocess.run(
            ["R", "--vanilla", "--quiet"],
            input='library(HonestDiD); library(jsonlite); cat("OK")',
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return "OK" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


R_HONESTDID_AVAILABLE = check_r_honestdid_available()


def r_get_bc_data():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    r_script = f"""
library(HonestDiD)
library(jsonlite)

data(BCdata_EventStudy)

out <- list(
    betahat = BCdata_EventStudy$betahat,
    sigma = BCdata_EventStudy$sigma,
    numPrePeriods = length(BCdata_EventStudy$prePeriodIndices),
    numPostPeriods = length(BCdata_EventStudy$postPeriodIndices)
)

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_construct_original_cs(betahat, sigma, num_pre_periods, num_post_periods, l_vec=None, alpha=0.05):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    betahat_str = "c(" + ",".join(map(str, betahat)) + ")"
    sigma_rows = ["c(" + ",".join(map(str, row)) + ")" for row in sigma]
    sigma_str = "matrix(c(" + ",".join(sigma_rows) + "), nrow=" + str(len(sigma)) + ", byrow=TRUE)"

    if l_vec is None:
        l_vec_str = f"basisVector(index = 1, size = {num_post_periods})"
    else:
        l_vec_str = "c(" + ",".join(map(str, l_vec)) + ")"

    r_script = f"""
library(HonestDiD)
library(jsonlite)

betahat <- {betahat_str}
sigma <- {sigma_str}
l_vec <- {l_vec_str}

result <- constructOriginalCS(
    betahat = betahat,
    sigma = sigma,
    numPrePeriods = {num_pre_periods},
    numPostPeriods = {num_post_periods},
    l_vec = l_vec,
    alpha = {alpha}
)

out <- list(
    lb = result$lb,
    ub = result$ub
)

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_compute_flci(betahat, sigma, num_pre_periods, num_post_periods, m_bar=0.0, alpha=0.05, l_vec=None):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    betahat_str = "c(" + ",".join(map(str, betahat)) + ")"
    sigma_rows = ["c(" + ",".join(map(str, row)) + ")" for row in sigma]
    sigma_str = "matrix(c(" + ",".join(sigma_rows) + "), nrow=" + str(len(sigma)) + ", byrow=TRUE)"

    if l_vec is None:
        l_vec_str = f"basisVector(index = 1, size = {num_post_periods})"
    else:
        l_vec_str = "c(" + ",".join(map(str, l_vec)) + ")"

    r_script = f"""
library(HonestDiD)
library(jsonlite)

betahat <- {betahat_str}
sigma <- {sigma_str}
l_vec <- {l_vec_str}

result <- findOptimalFLCI(
    betahat = betahat,
    sigma = sigma,
    numPrePeriods = {num_pre_periods},
    numPostPeriods = {num_post_periods},
    l_vec = l_vec,
    M = {m_bar},
    alpha = {alpha}
)

out <- list(
    lb = result$FLCI[1],
    ub = result$FLCI[2],
    halflength = result$optimalHalfLength,
    optimal_l = result$optimalVec
)

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_sensitivity_sm(
    betahat,
    sigma,
    num_pre_periods,
    num_post_periods,
    m_vec,
    method="FLCI",
    l_vec=None,
    alpha=0.05,
    monotonicity_direction=None,
    bias_direction=None,
):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    betahat_str = "c(" + ",".join(map(str, betahat)) + ")"
    sigma_rows = ["c(" + ",".join(map(str, row)) + ")" for row in sigma]
    sigma_str = "matrix(c(" + ",".join(sigma_rows) + "), nrow=" + str(len(sigma)) + ", byrow=TRUE)"
    m_vec_str = "c(" + ",".join(map(str, m_vec)) + ")"

    if l_vec is None:
        l_vec_str = f"basisVector(index = 1, size = {num_post_periods})"
    else:
        l_vec_str = "c(" + ",".join(map(str, l_vec)) + ")"

    monotonicity_str = "NULL" if monotonicity_direction is None else f'"{monotonicity_direction}"'
    bias_str = "NULL" if bias_direction is None else f'"{bias_direction}"'

    r_script = f"""
library(HonestDiD)
library(jsonlite)

betahat <- {betahat_str}
sigma <- {sigma_str}
l_vec <- {l_vec_str}
Mvec <- {m_vec_str}

result <- createSensitivityResults(
    betahat = betahat,
    sigma = sigma,
    numPrePeriods = {num_pre_periods},
    numPostPeriods = {num_post_periods},
    l_vec = l_vec,
    Mvec = Mvec,
    method = "{method}",
    alpha = {alpha},
    monotonicityDirection = {monotonicity_str},
    biasDirection = {bias_str}
)

out <- list(
    lb = result$lb,
    ub = result$ub,
    method = result$method,
    delta = result$Delta,
    m = result$M
)

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


# This helper loads BCdata_EventStudy directly inside R and
# calls computeConditionalCS_DeltaSDM with explicit gridPoints to match
# the Python side. R's CVXR solves an underdetermined QP for vbar whose
# solution depends on the active QP backend (OSQP, ECOS, SCS, etc.).
# Different conda/R environments ship different solver stacks, so the
# live R result is not reproducible across environments.  The
# corresponding test therefore compares Python against hardcoded
# reference values from a known-good R installation (CVXR 1.0.15, OSQP)
# rather than the live R output.
def r_sensitivity_sm_bcdata(m_vec, method="C-F", monotonicity_direction=None, alpha=0.05, grid_points=100):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    m_vec_str = "c(" + ",".join(map(str, m_vec)) + ")"
    monotonicity_str = "NULL" if monotonicity_direction is None else f'"{monotonicity_direction}"'
    hybrid_flag = {"C-F": "FLCI", "Conditional": "ARP", "C-LF": "LF"}.get(method, "FLCI")

    r_script = f"""
library(HonestDiD)
library(jsonlite)

data(BCdata_EventStudy)
betahat <- BCdata_EventStudy$betahat
sigma <- BCdata_EventStudy$sigma
numPre <- length(BCdata_EventStudy$prePeriodIndices)
numPost <- length(BCdata_EventStudy$postPeriodIndices)
l_vec <- basisVector(index = 1, size = numPost)
Mvec <- {m_vec_str}
monoDir <- {monotonicity_str}

lbs <- numeric(length(Mvec))
ubs <- numeric(length(Mvec))

for (i in seq_along(Mvec)) {{
    temp <- computeConditionalCS_DeltaSDM(
        betahat = betahat,
        sigma = sigma,
        numPrePeriods = numPre,
        numPostPeriods = numPost,
        l_vec = l_vec,
        alpha = {alpha},
        M = Mvec[i],
        monotonicityDirection = monoDir,
        hybrid_flag = "{hybrid_flag}",
        gridPoints = {grid_points}
    )
    accepted <- temp$grid[temp$accept == 1]
    lbs[i] <- min(accepted)
    ubs[i] <- max(accepted)
}}

out <- list(lb = lbs, ub = ubs, m = Mvec)
write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_sensitivity_rm(
    betahat,
    sigma,
    num_pre_periods,
    num_post_periods,
    m_bar_vec,
    method="C-LF",
    bound="deviation from parallel trends",
    l_vec=None,
    alpha=0.05,
    grid_points=100,
    monotonicity_direction=None,
    bias_direction=None,
    grid_lb=None,
    grid_ub=None,
):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    betahat_str = "c(" + ",".join(map(str, betahat)) + ")"
    sigma_rows = ["c(" + ",".join(map(str, row)) + ")" for row in sigma]
    sigma_str = "matrix(c(" + ",".join(sigma_rows) + "), nrow=" + str(len(sigma)) + ", byrow=TRUE)"
    m_bar_vec_str = "c(" + ",".join(map(str, m_bar_vec)) + ")"

    if l_vec is None:
        l_vec_str = f"basisVector(index = 1, size = {num_post_periods})"
    else:
        l_vec_str = "c(" + ",".join(map(str, l_vec)) + ")"

    method_str = "NULL" if method is None else f'"{method}"'
    monotonicity_str = "NULL" if monotonicity_direction is None else f'"{monotonicity_direction}"'
    bias_str = "NULL" if bias_direction is None else f'"{bias_direction}"'

    grid_str = ""
    if grid_lb is not None:
        grid_str = f",\n    grid.lb = {grid_lb},\n    grid.ub = {grid_ub}"

    r_script = f"""
library(HonestDiD)
library(jsonlite)

betahat <- {betahat_str}
sigma <- {sigma_str}
l_vec <- {l_vec_str}
Mbarvec <- {m_bar_vec_str}

result <- createSensitivityResults_relativeMagnitudes(
    betahat = betahat,
    sigma = sigma,
    numPrePeriods = {num_pre_periods},
    numPostPeriods = {num_post_periods},
    l_vec = l_vec,
    Mbarvec = Mbarvec,
    method = {method_str},
    bound = "{bound}",
    alpha = {alpha},
    gridPoints = {grid_points},
    monotonicityDirection = {monotonicity_str},
    biasDirection = {bias_str}{grid_str}
)

out <- list(
    lb = result$lb,
    ub = result$ub,
    method = result$method,
    delta = result$Delta,
    mbar = result$Mbar
)

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path, timeout=600)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_identified_set_sd(betahat, sigma, num_pre_periods, num_post_periods, m_bar, l_vec=None):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    betahat_str = "c(" + ",".join(map(str, betahat)) + ")"
    sigma_rows = ["c(" + ",".join(map(str, row)) + ")" for row in sigma]
    sigma_str = "matrix(c(" + ",".join(sigma_rows) + "), nrow=" + str(len(sigma)) + ", byrow=TRUE)"

    if l_vec is None:
        l_vec_str = f"basisVector(index = 1, size = {num_post_periods})"
    else:
        l_vec_str = "c(" + ",".join(map(str, l_vec)) + ")"

    r_script = f"""
library(HonestDiD)
library(jsonlite)

betahat <- {betahat_str}
sigma <- {sigma_str}
l_vec <- {l_vec_str}

id_set <- HonestDiD:::.compute_IDset_DeltaSD(
    M = {m_bar},
    trueBeta = betahat,
    l_vec = l_vec,
    numPrePeriods = {num_pre_periods},
    numPostPeriods = {num_post_periods}
)

out <- list(
    id_lb = id_set$id.lb,
    id_ub = id_set$id.ub
)

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_identified_set_rm(betahat, sigma, num_pre_periods, num_post_periods, m_bar, l_vec=None):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    betahat_str = "c(" + ",".join(map(str, betahat)) + ")"
    sigma_rows = ["c(" + ",".join(map(str, row)) + ")" for row in sigma]
    sigma_str = "matrix(c(" + ",".join(sigma_rows) + "), nrow=" + str(len(sigma)) + ", byrow=TRUE)"

    if l_vec is None:
        l_vec_str = f"basisVector(index = 1, size = {num_post_periods})"
    else:
        l_vec_str = "c(" + ",".join(map(str, l_vec)) + ")"

    r_script = f"""
library(HonestDiD)
library(jsonlite)

betahat <- {betahat_str}
sigma <- {sigma_str}
l_vec <- {l_vec_str}

id_set <- HonestDiD:::.compute_IDset_DeltaRM(
    Mbar = {m_bar},
    trueBeta = betahat,
    l_vec = l_vec,
    numPrePeriods = {num_pre_periods},
    numPostPeriods = {num_post_periods}
)

out <- list(
    id_lb = id_set$id.lb,
    id_ub = id_set$id.ub
)

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_identified_set_rmb(
    betahat, sigma, num_pre_periods, num_post_periods, m_bar, bias_direction="positive", l_vec=None
):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    betahat_str = "c(" + ",".join(map(str, betahat)) + ")"
    sigma_rows = ["c(" + ",".join(map(str, row)) + ")" for row in sigma]
    sigma_str = "matrix(c(" + ",".join(sigma_rows) + "), nrow=" + str(len(sigma)) + ", byrow=TRUE)"

    if l_vec is None:
        l_vec_str = f"basisVector(index = 1, size = {num_post_periods})"
    else:
        l_vec_str = "c(" + ",".join(map(str, l_vec)) + ")"

    r_script = f"""
library(HonestDiD)
library(jsonlite)

betahat <- {betahat_str}
sigma <- {sigma_str}
l_vec <- {l_vec_str}

id_set <- HonestDiD:::.compute_IDset_DeltaRMB(
    Mbar = {m_bar},
    trueBeta = betahat,
    l_vec = l_vec,
    numPrePeriods = {num_pre_periods},
    numPostPeriods = {num_post_periods},
    biasDirection = "{bias_direction}"
)

out <- list(
    id_lb = id_set$id.lb,
    id_ub = id_set$id.ub
)

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_identified_set_rmm(
    betahat, sigma, num_pre_periods, num_post_periods, m_bar, monotonicity_direction="increasing", l_vec=None
):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    betahat_str = "c(" + ",".join(map(str, betahat)) + ")"
    sigma_rows = ["c(" + ",".join(map(str, row)) + ")" for row in sigma]
    sigma_str = "matrix(c(" + ",".join(sigma_rows) + "), nrow=" + str(len(sigma)) + ", byrow=TRUE)"

    if l_vec is None:
        l_vec_str = f"basisVector(index = 1, size = {num_post_periods})"
    else:
        l_vec_str = "c(" + ",".join(map(str, l_vec)) + ")"

    r_script = f"""
library(HonestDiD)
library(jsonlite)

betahat <- {betahat_str}
sigma <- {sigma_str}
l_vec <- {l_vec_str}

id_set <- HonestDiD:::.compute_IDset_DeltaRMM(
    Mbar = {m_bar},
    trueBeta = betahat,
    l_vec = l_vec,
    numPrePeriods = {num_pre_periods},
    numPostPeriods = {num_post_periods},
    monotonicityDirection = "{monotonicity_direction}"
)

out <- list(
    id_lb = id_set$id.lb,
    id_ub = id_set$id.ub
)

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_identified_set_sdrm(betahat, sigma, num_pre_periods, num_post_periods, m_bar, l_vec=None):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    betahat_str = "c(" + ",".join(map(str, betahat)) + ")"
    sigma_rows = ["c(" + ",".join(map(str, row)) + ")" for row in sigma]
    sigma_str = "matrix(c(" + ",".join(sigma_rows) + "), nrow=" + str(len(sigma)) + ", byrow=TRUE)"

    if l_vec is None:
        l_vec_str = f"basisVector(index = 1, size = {num_post_periods})"
    else:
        l_vec_str = "c(" + ",".join(map(str, l_vec)) + ")"

    r_script = f"""
library(HonestDiD)
library(jsonlite)

betahat <- {betahat_str}
sigma <- {sigma_str}
l_vec <- {l_vec_str}

id_set <- HonestDiD:::.compute_IDset_DeltaSDRM(
    Mbar = {m_bar},
    trueBeta = betahat,
    l_vec = l_vec,
    numPrePeriods = {num_pre_periods},
    numPostPeriods = {num_post_periods}
)

out <- list(
    id_lb = id_set$id.lb,
    id_ub = id_set$id.ub
)

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_identified_set_sdrmb(
    betahat, sigma, num_pre_periods, num_post_periods, m_bar, bias_direction="positive", l_vec=None
):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    betahat_str = "c(" + ",".join(map(str, betahat)) + ")"
    sigma_rows = ["c(" + ",".join(map(str, row)) + ")" for row in sigma]
    sigma_str = "matrix(c(" + ",".join(sigma_rows) + "), nrow=" + str(len(sigma)) + ", byrow=TRUE)"

    if l_vec is None:
        l_vec_str = f"basisVector(index = 1, size = {num_post_periods})"
    else:
        l_vec_str = "c(" + ",".join(map(str, l_vec)) + ")"

    r_script = f"""
library(HonestDiD)
library(jsonlite)

betahat <- {betahat_str}
sigma <- {sigma_str}
l_vec <- {l_vec_str}

id_set <- HonestDiD:::.compute_IDset_DeltaSDRMB(
    Mbar = {m_bar},
    trueBeta = betahat,
    l_vec = l_vec,
    numPrePeriods = {num_pre_periods},
    numPostPeriods = {num_post_periods},
    biasDirection = "{bias_direction}"
)

out <- list(
    id_lb = id_set$id.lb,
    id_ub = id_set$id.ub
)

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_identified_set_sdrmm(
    betahat, sigma, num_pre_periods, num_post_periods, m_bar, monotonicity_direction="increasing", l_vec=None
):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    betahat_str = "c(" + ",".join(map(str, betahat)) + ")"
    sigma_rows = ["c(" + ",".join(map(str, row)) + ")" for row in sigma]
    sigma_str = "matrix(c(" + ",".join(sigma_rows) + "), nrow=" + str(len(sigma)) + ", byrow=TRUE)"

    if l_vec is None:
        l_vec_str = f"basisVector(index = 1, size = {num_post_periods})"
    else:
        l_vec_str = "c(" + ",".join(map(str, l_vec)) + ")"

    r_script = f"""
library(HonestDiD)
library(jsonlite)

betahat <- {betahat_str}
sigma <- {sigma_str}
l_vec <- {l_vec_str}

id_set <- HonestDiD:::.compute_IDset_DeltaSDRMM(
    Mbar = {m_bar},
    trueBeta = betahat,
    l_vec = l_vec,
    numPrePeriods = {num_pre_periods},
    numPostPeriods = {num_post_periods},
    monotonicityDirection = "{monotonicity_direction}"
)

out <- list(
    id_lb = id_set$id.lb,
    id_ub = id_set$id.ub
)

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_conditional_cs_rmb(
    betahat,
    sigma,
    num_pre_periods,
    num_post_periods,
    m_bar,
    bias_direction="positive",
    method="LF",
    l_vec=None,
    alpha=0.05,
    grid_points=None,
    grid_lb=None,
    grid_ub=None,
):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    betahat_str = "c(" + ",".join(map(str, betahat)) + ")"
    sigma_rows = ["c(" + ",".join(map(str, row)) + ")" for row in sigma]
    sigma_str = "matrix(c(" + ",".join(sigma_rows) + "), nrow=" + str(len(sigma)) + ", byrow=TRUE)"

    if l_vec is None:
        l_vec_str = f"basisVector(index = 1, size = {num_post_periods})"
    else:
        l_vec_str = "c(" + ",".join(map(str, l_vec)) + ")"

    grid_str = ""
    if grid_points is not None:
        grid_str = f",\n    gridPoints = {grid_points}, grid.lb = {grid_lb}, grid.ub = {grid_ub}"

    r_script = f"""
library(HonestDiD)
library(jsonlite)

betahat <- {betahat_str}
sigma <- {sigma_str}
l_vec <- {l_vec_str}

result <- computeConditionalCS_DeltaRMB(
    betahat = betahat,
    sigma = sigma,
    numPrePeriods = {num_pre_periods},
    numPostPeriods = {num_post_periods},
    l_vec = l_vec,
    Mbar = {m_bar},
    alpha = {alpha},
    hybrid_flag = "{method}",
    biasDirection = "{bias_direction}"{grid_str}
)

accept_idx <- which(result$accept == 1)
if (length(accept_idx) > 0) {{
    lb <- result$grid[min(accept_idx)]
    ub <- result$grid[max(accept_idx)]
}} else {{
    lb <- NA
    ub <- NA
}}

out <- list(
    lb = lb,
    ub = ub,
    grid = result$grid,
    accept = result$accept
)

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path, timeout=600)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_conditional_cs_rmm(
    betahat,
    sigma,
    num_pre_periods,
    num_post_periods,
    m_bar,
    monotonicity_direction="increasing",
    method="LF",
    l_vec=None,
    alpha=0.05,
    grid_points=None,
    grid_lb=None,
    grid_ub=None,
):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    betahat_str = "c(" + ",".join(map(str, betahat)) + ")"
    sigma_rows = ["c(" + ",".join(map(str, row)) + ")" for row in sigma]
    sigma_str = "matrix(c(" + ",".join(sigma_rows) + "), nrow=" + str(len(sigma)) + ", byrow=TRUE)"

    if l_vec is None:
        l_vec_str = f"basisVector(index = 1, size = {num_post_periods})"
    else:
        l_vec_str = "c(" + ",".join(map(str, l_vec)) + ")"

    grid_str = ""
    if grid_points is not None:
        grid_str = f",\n    gridPoints = {grid_points}, grid.lb = {grid_lb}, grid.ub = {grid_ub}"

    r_script = f"""
library(HonestDiD)
library(jsonlite)

betahat <- {betahat_str}
sigma <- {sigma_str}
l_vec <- {l_vec_str}

result <- computeConditionalCS_DeltaRMM(
    betahat = betahat,
    sigma = sigma,
    numPrePeriods = {num_pre_periods},
    numPostPeriods = {num_post_periods},
    l_vec = l_vec,
    Mbar = {m_bar},
    alpha = {alpha},
    hybrid_flag = "{method}",
    monotonicityDirection = "{monotonicity_direction}"{grid_str}
)

accept_idx <- which(result$accept == 1)
if (length(accept_idx) > 0) {{
    lb <- result$grid[min(accept_idx)]
    ub <- result$grid[max(accept_idx)]
}} else {{
    lb <- NA
    ub <- NA
}}

out <- list(
    lb = lb,
    ub = ub,
    grid = result$grid,
    accept = result$accept
)

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path, timeout=600)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_conditional_cs_sdrm(
    betahat, sigma, num_pre_periods, num_post_periods, m_bar, method="LF", l_vec=None, alpha=0.05
):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    betahat_str = "c(" + ",".join(map(str, betahat)) + ")"
    sigma_rows = ["c(" + ",".join(map(str, row)) + ")" for row in sigma]
    sigma_str = "matrix(c(" + ",".join(sigma_rows) + "), nrow=" + str(len(sigma)) + ", byrow=TRUE)"

    if l_vec is None:
        l_vec_str = f"basisVector(index = 1, size = {num_post_periods})"
    else:
        l_vec_str = "c(" + ",".join(map(str, l_vec)) + ")"

    r_script = f"""
library(HonestDiD)
library(jsonlite)

betahat <- {betahat_str}
sigma <- {sigma_str}
l_vec <- {l_vec_str}

result <- computeConditionalCS_DeltaSDRM(
    betahat = betahat,
    sigma = sigma,
    numPrePeriods = {num_pre_periods},
    numPostPeriods = {num_post_periods},
    l_vec = l_vec,
    Mbar = {m_bar},
    alpha = {alpha},
    hybrid_flag = "{method}"
)

accept_idx <- which(result$accept == 1)
if (length(accept_idx) > 0) {{
    lb <- result$grid[min(accept_idx)]
    ub <- result$grid[max(accept_idx)]
}} else {{
    lb <- NA
    ub <- NA
}}

out <- list(
    lb = lb,
    ub = ub,
    grid = result$grid,
    accept = result$accept
)

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path, timeout=600)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_conditional_cs_sdrmb(
    betahat,
    sigma,
    num_pre_periods,
    num_post_periods,
    m_bar,
    bias_direction="positive",
    method="LF",
    l_vec=None,
    alpha=0.05,
):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    betahat_str = "c(" + ",".join(map(str, betahat)) + ")"
    sigma_rows = ["c(" + ",".join(map(str, row)) + ")" for row in sigma]
    sigma_str = "matrix(c(" + ",".join(sigma_rows) + "), nrow=" + str(len(sigma)) + ", byrow=TRUE)"

    if l_vec is None:
        l_vec_str = f"basisVector(index = 1, size = {num_post_periods})"
    else:
        l_vec_str = "c(" + ",".join(map(str, l_vec)) + ")"

    r_script = f"""
library(HonestDiD)
library(jsonlite)

betahat <- {betahat_str}
sigma <- {sigma_str}
l_vec <- {l_vec_str}

result <- computeConditionalCS_DeltaSDRMB(
    betahat = betahat,
    sigma = sigma,
    numPrePeriods = {num_pre_periods},
    numPostPeriods = {num_post_periods},
    l_vec = l_vec,
    Mbar = {m_bar},
    alpha = {alpha},
    hybrid_flag = "{method}",
    biasDirection = "{bias_direction}"
)

accept_idx <- which(result$accept == 1)
if (length(accept_idx) > 0) {{
    lb <- result$grid[min(accept_idx)]
    ub <- result$grid[max(accept_idx)]
}} else {{
    lb <- NA
    ub <- NA
}}

out <- list(
    lb = lb,
    ub = ub,
    grid = result$grid,
    accept = result$accept
)

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path, timeout=600)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_conditional_cs_sdrmm(
    betahat,
    sigma,
    num_pre_periods,
    num_post_periods,
    m_bar,
    monotonicity_direction="increasing",
    method="LF",
    l_vec=None,
    alpha=0.05,
):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    betahat_str = "c(" + ",".join(map(str, betahat)) + ")"
    sigma_rows = ["c(" + ",".join(map(str, row)) + ")" for row in sigma]
    sigma_str = "matrix(c(" + ",".join(sigma_rows) + "), nrow=" + str(len(sigma)) + ", byrow=TRUE)"

    if l_vec is None:
        l_vec_str = f"basisVector(index = 1, size = {num_post_periods})"
    else:
        l_vec_str = "c(" + ",".join(map(str, l_vec)) + ")"

    r_script = f"""
library(HonestDiD)
library(jsonlite)

betahat <- {betahat_str}
sigma <- {sigma_str}
l_vec <- {l_vec_str}

result <- computeConditionalCS_DeltaSDRMM(
    betahat = betahat,
    sigma = sigma,
    numPrePeriods = {num_pre_periods},
    numPostPeriods = {num_post_periods},
    l_vec = l_vec,
    Mbar = {m_bar},
    alpha = {alpha},
    hybrid_flag = "{method}",
    monotonicityDirection = "{monotonicity_direction}"
)

accept_idx <- which(result$accept == 1)
if (length(accept_idx) > 0) {{
    lb <- result$grid[min(accept_idx)]
    ub <- result$grid[max(accept_idx)]
}} else {{
    lb <- NA
    ub <- NA
}}

out <- list(
    lb = lb,
    ub = ub,
    grid = result$grid,
    accept = result$accept
)

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path, timeout=600)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def r_conditional_cs_sd(
    betahat,
    sigma,
    num_pre_periods,
    num_post_periods,
    m_bar,
    method="FLCI",
    l_vec=None,
    alpha=0.05,
    grid_points=None,
    grid_lb=None,
    grid_ub=None,
    bias_direction=None,
):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    betahat_str = "c(" + ",".join(map(str, betahat)) + ")"
    sigma_rows = ["c(" + ",".join(map(str, row)) + ")" for row in sigma]
    sigma_str = "matrix(c(" + ",".join(sigma_rows) + "), nrow=" + str(len(sigma)) + ", byrow=TRUE)"

    if l_vec is None:
        l_vec_str = f"basisVector(index = 1, size = {num_post_periods})"
    else:
        l_vec_str = "c(" + ",".join(map(str, l_vec)) + ")"

    grid_str = ""
    if grid_points is not None:
        grid_str = f",\n    gridPoints = {grid_points}, grid.lb = {grid_lb}, grid.ub = {grid_ub}"

    r_function = "computeConditionalCS_DeltaSD"
    if bias_direction is not None:
        r_function = "computeConditionalCS_DeltaSDB"
        grid_str = f',\n    biasDirection = "{bias_direction}"' + grid_str

    r_script = f"""
library(HonestDiD)
library(jsonlite)

betahat <- {betahat_str}
sigma <- {sigma_str}
l_vec <- {l_vec_str}

result <- {r_function}(
    betahat = betahat,
    sigma = sigma,
    numPrePeriods = {num_pre_periods},
    numPostPeriods = {num_post_periods},
    l_vec = l_vec,
    M = {m_bar},
    alpha = {alpha},
    hybrid_flag = "{method}"{grid_str}
)

accept_idx <- which(result$accept == 1)
if (length(accept_idx) > 0) {{
    lb <- result$grid[min(accept_idx)]
    ub <- result$grid[max(accept_idx)]
}} else {{
    lb <- NA
    ub <- NA
}}

out <- list(
    lb = lb,
    ub = ub,
    grid = result$grid,
    accept = result$accept
)

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path, timeout=600)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def _extract_scalar(val):
    if isinstance(val, list):
        return val[0]
    return val


@pytest.fixture(scope="module")
def bc_data():
    r_data = r_get_bc_data()
    if r_data is None:
        pytest.fail("Could not load BC data from R")

    betahat = np.array(r_data["betahat"]).flatten()
    sigma_list = r_data["sigma"]
    sigma = np.array([row if isinstance(row, list) else [row] for row in sigma_list])
    if sigma.ndim == 1:
        sigma = sigma.reshape(-1, 1)

    return {
        "betahat": betahat,
        "sigma": sigma,
        "num_pre_periods": int(_extract_scalar(r_data["numPrePeriods"])),
        "num_post_periods": int(_extract_scalar(r_data["numPostPeriods"])),
    }


@pytest.fixture(scope="module")
def synthetic_event_study():
    np.random.seed(42)
    num_pre = 3
    num_post = 4
    total = num_pre + num_post

    betahat = np.array([-0.02, -0.01, 0.01, 0.05, 0.08, 0.12, 0.15])
    sigma = np.eye(total) * 0.01
    sigma = sigma + np.ones((total, total)) * 0.002

    return {
        "betahat": betahat,
        "sigma": sigma,
        "num_pre_periods": num_pre,
        "num_post_periods": num_post,
    }


@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
def test_original_cs_bc_data(bc_data):
    r_result = r_construct_original_cs(
        betahat=bc_data["betahat"].tolist(),
        sigma=bc_data["sigma"].tolist(),
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
    )

    if r_result is None:
        pytest.fail("R construct original CS failed")

    l_vec = basis_vector(1, bc_data["num_post_periods"])
    py_result = construct_original_cs(
        betahat=bc_data["betahat"],
        sigma=bc_data["sigma"],
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        l_vec=l_vec,
    )

    np.testing.assert_allclose(py_result.lb, r_result["lb"], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(py_result.ub, r_result["ub"], rtol=1e-4, atol=1e-6)


@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
def test_original_cs_synthetic(synthetic_event_study):
    r_result = r_construct_original_cs(
        betahat=synthetic_event_study["betahat"].tolist(),
        sigma=synthetic_event_study["sigma"].tolist(),
        num_pre_periods=synthetic_event_study["num_pre_periods"],
        num_post_periods=synthetic_event_study["num_post_periods"],
    )

    if r_result is None:
        pytest.fail("R construct original CS failed")

    l_vec = basis_vector(1, synthetic_event_study["num_post_periods"])
    py_result = construct_original_cs(
        betahat=synthetic_event_study["betahat"],
        sigma=synthetic_event_study["sigma"],
        num_pre_periods=synthetic_event_study["num_pre_periods"],
        num_post_periods=synthetic_event_study["num_post_periods"],
        l_vec=l_vec,
    )

    np.testing.assert_allclose(py_result.lb, r_result["lb"], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(py_result.ub, r_result["ub"], rtol=1e-4, atol=1e-6)


@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
@pytest.mark.parametrize("m_bar", [0.0, 0.1, 0.2])
def test_flci_bc_data(bc_data, m_bar):
    r_result = r_compute_flci(
        betahat=bc_data["betahat"].tolist(),
        sigma=bc_data["sigma"].tolist(),
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        m_bar=m_bar,
    )

    if r_result is None:
        pytest.fail("R FLCI computation failed")

    l_vec = basis_vector(1, bc_data["num_post_periods"])
    py_result = compute_flci(
        beta_hat=bc_data["betahat"],
        sigma=bc_data["sigma"],
        n_pre_periods=bc_data["num_pre_periods"],
        n_post_periods=bc_data["num_post_periods"],
        smoothness_bound=m_bar,
        post_period_weights=l_vec.flatten(),
    )

    np.testing.assert_allclose(py_result.flci[0], r_result["lb"], rtol=0.05, atol=1e-3)
    np.testing.assert_allclose(py_result.flci[1], r_result["ub"], rtol=0.05, atol=1e-3)
    np.testing.assert_allclose(py_result.optimal_half_length, r_result["halflength"], rtol=0.05, atol=1e-3)


@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
@pytest.mark.parametrize("m_bar", [0.0, 0.05, 0.1])
def test_flci_synthetic(synthetic_event_study, m_bar):
    r_result = r_compute_flci(
        betahat=synthetic_event_study["betahat"].tolist(),
        sigma=synthetic_event_study["sigma"].tolist(),
        num_pre_periods=synthetic_event_study["num_pre_periods"],
        num_post_periods=synthetic_event_study["num_post_periods"],
        m_bar=m_bar,
    )

    if r_result is None:
        pytest.fail("R FLCI computation failed")

    l_vec = basis_vector(1, synthetic_event_study["num_post_periods"])
    py_result = compute_flci(
        beta_hat=synthetic_event_study["betahat"],
        sigma=synthetic_event_study["sigma"],
        n_pre_periods=synthetic_event_study["num_pre_periods"],
        n_post_periods=synthetic_event_study["num_post_periods"],
        smoothness_bound=m_bar,
        post_period_weights=l_vec.flatten(),
    )

    np.testing.assert_allclose(py_result.flci[0], r_result["lb"], rtol=0.05, atol=1e-3)
    np.testing.assert_allclose(py_result.flci[1], r_result["ub"], rtol=0.05, atol=1e-3)


@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
@pytest.mark.parametrize("m_bar", [0.0, 0.1, 0.2])
def test_identified_set_sd_bc_data(bc_data, m_bar):
    r_result = r_identified_set_sd(
        betahat=bc_data["betahat"].tolist(),
        sigma=bc_data["sigma"].tolist(),
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        m_bar=m_bar,
    )

    if r_result is None:
        pytest.fail("R identified set SD computation failed")

    l_vec = basis_vector(1, bc_data["num_post_periods"])
    py_result = compute_identified_set_sd(
        m_bar=m_bar,
        true_beta=bc_data["betahat"],
        l_vec=l_vec,
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
    )

    np.testing.assert_allclose(py_result.id_lb, r_result["id_lb"], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(py_result.id_ub, r_result["id_ub"], rtol=1e-4, atol=1e-6)


@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
@pytest.mark.parametrize("m_bar", [0.0, 0.5, 1.0])
def test_identified_set_rm_bc_data(bc_data, m_bar):
    r_result = r_identified_set_rm(
        betahat=bc_data["betahat"].tolist(),
        sigma=bc_data["sigma"].tolist(),
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        m_bar=m_bar,
    )

    if r_result is None:
        pytest.fail("R identified set RM computation failed")

    l_vec = basis_vector(1, bc_data["num_post_periods"])
    py_result = compute_identified_set_rm(
        m_bar=m_bar,
        true_beta=bc_data["betahat"],
        l_vec=l_vec,
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
    )

    np.testing.assert_allclose(py_result.id_lb, r_result["id_lb"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(py_result.id_ub, r_result["id_ub"], rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
def test_sensitivity_sm_flci_bc_data(bc_data):
    m_vec = [0.0, 0.1, 0.2, 0.3]

    r_result = r_sensitivity_sm(
        betahat=bc_data["betahat"].tolist(),
        sigma=bc_data["sigma"].tolist(),
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        m_vec=m_vec,
        method="FLCI",
    )

    if r_result is None:
        pytest.fail("R sensitivity SM FLCI failed")

    l_vec = basis_vector(1, bc_data["num_post_periods"])
    py_result = create_sensitivity_results_sm(
        betahat=bc_data["betahat"],
        sigma=bc_data["sigma"],
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        method="FLCI",
        m_vec=np.array(m_vec),
        l_vec=l_vec,
    )

    compared = 0
    for i, m in enumerate(m_vec):
        py_row = py_result.filter(pl.col("m") == m)
        if len(py_row) == 0:
            continue

        py_lb = py_row["lb"][0]
        py_ub = py_row["ub"][0]
        r_lb = r_result["lb"][i]
        r_ub = r_result["ub"][i]

        if not np.isnan(py_lb) and not np.isnan(r_lb):
            np.testing.assert_allclose(py_lb, r_lb, rtol=0.05, atol=1e-2)
            compared += 1
        if not np.isnan(py_ub) and not np.isnan(r_ub):
            np.testing.assert_allclose(py_ub, r_ub, rtol=0.05, atol=1e-2)
            compared += 1
    assert compared > 0, "SM FLCI: No non-NaN values to compare"


@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
@pytest.mark.parametrize("method", ["Conditional", "C-F", "C-LF"])
def test_sensitivity_sm_methods_bc_data(bc_data, method):
    m_vec = [0.0, 0.1, 0.2]
    hybrid_flag = {"Conditional": "ARP", "C-F": "FLCI", "C-LF": "LF"}[method]

    r_results = [
        r_conditional_cs_sd(
            betahat=bc_data["betahat"].tolist(),
            sigma=bc_data["sigma"].tolist(),
            num_pre_periods=bc_data["num_pre_periods"],
            num_post_periods=bc_data["num_post_periods"],
            m_bar=m,
            method=hybrid_flag,
            grid_points=141,
            grid_lb=-0.1,
            grid_ub=0.6,
        )
        for m in m_vec
    ]

    if any(r_result is None for r_result in r_results):
        pytest.fail(f"R sensitivity SM {method} failed")

    l_vec = basis_vector(1, bc_data["num_post_periods"])
    py_result = create_sensitivity_results_sm(
        betahat=bc_data["betahat"],
        sigma=bc_data["sigma"],
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        method=method,
        m_vec=np.array(m_vec),
        l_vec=l_vec,
        grid_points=141,
        grid_lb=-0.1,
        grid_ub=0.6,
    )

    compared = 0
    for m, r_result in zip(m_vec, r_results, strict=True):
        py_row = py_result.filter(pl.col("m") == m)
        if len(py_row) == 0:
            continue

        py_lb = py_row["lb"][0]
        py_ub = py_row["ub"][0]
        r_lb = _extract_scalar(r_result["lb"])
        r_ub = _extract_scalar(r_result["ub"])

        if not np.isnan(py_lb) and not np.isnan(r_lb):
            np.testing.assert_allclose(py_lb, r_lb, rtol=0, atol=0.005 + 1e-12)
            compared += 1
        if not np.isnan(py_ub) and not np.isnan(r_ub):
            np.testing.assert_allclose(py_ub, r_ub, rtol=0, atol=0.005 + 1e-12)
            compared += 1
    assert compared > 0, f"SM {method}: No non-NaN values to compare"


@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
def test_sensitivity_rm_clf_bc_data(bc_data):
    m_bar_vec = [0.0, 0.5, 1.0]

    r_result = r_sensitivity_rm(
        betahat=bc_data["betahat"].tolist(),
        sigma=bc_data["sigma"].tolist(),
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        m_bar_vec=m_bar_vec,
        method="C-LF",
        grid_points=101,
        grid_lb=-0.3,
        grid_ub=0.7,
    )

    if r_result is None:
        pytest.fail("R sensitivity RM C-LF failed")

    l_vec = basis_vector(1, bc_data["num_post_periods"])
    py_result = create_sensitivity_results_rm(
        betahat=bc_data["betahat"],
        sigma=bc_data["sigma"],
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        method="C-LF",
        m_bar_vec=np.array(m_bar_vec),
        l_vec=l_vec,
        grid_points=101,
        grid_lb=-0.3,
        grid_ub=0.7,
    )

    compared = 0
    for i, mbar in enumerate(m_bar_vec):
        py_row = py_result.filter(pl.col("Mbar") == mbar)
        if len(py_row) == 0:
            continue

        py_lb = py_row["lb"][0]
        py_ub = py_row["ub"][0]
        r_lb = r_result["lb"][i]
        r_ub = r_result["ub"][i]

        if not np.isnan(py_lb) and not np.isnan(r_lb):
            np.testing.assert_allclose(py_lb, r_lb, rtol=0, atol=0.01 + 1e-12)
            compared += 1
        if not np.isnan(py_ub) and not np.isnan(r_ub):
            np.testing.assert_allclose(py_ub, r_ub, rtol=0, atol=0.01 + 1e-12)
            compared += 1
    assert compared > 0, "RM C-LF: No non-NaN values to compare"


@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
@pytest.mark.parametrize("monotonicity_direction", ["increasing", "decreasing"])
def test_sensitivity_sm_monotonicity(bc_data, monotonicity_direction):
    m_vec = [0.0, 0.1, 0.2]
    grid_points = 100

    # Reference values from R's HonestDiD with original BCdata_EventStudy
    # and gridPoints=100.
    ref = {
        "increasing": {
            "lb": [0.193975, 0.119858, 0.024793],
            "ub": [0.233122, 0.231247, 0.226876],
        },
        "decreasing": {
            "lb": [0.206209, 0.166570, 0.165128],
            "ub": [0.234345, 0.418094, 0.513159],
        },
    }[monotonicity_direction]

    l_vec = basis_vector(1, bc_data["num_post_periods"])
    py_result = create_sensitivity_results_sm(
        betahat=bc_data["betahat"],
        sigma=bc_data["sigma"],
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        method="C-F",
        m_vec=np.array(m_vec),
        l_vec=l_vec,
        monotonicity_direction=monotonicity_direction,
        grid_points=grid_points,
    )

    compared = 0
    for i, m in enumerate(m_vec):
        py_row = py_result.filter(pl.col("m") == m)
        if len(py_row) == 0:
            continue
        py_lb = py_row["lb"][0]
        py_ub = py_row["ub"][0]
        r_lb = ref["lb"][i]
        r_ub = ref["ub"][i]
        if not np.isnan(py_lb):
            np.testing.assert_allclose(py_lb, r_lb, rtol=0.01, atol=0.005)
            compared += 1
        if not np.isnan(py_ub):
            np.testing.assert_allclose(py_ub, r_ub, rtol=0.01, atol=0.005)
            compared += 1
    assert compared > 0, f"monotonicity={monotonicity_direction}: No non-NaN values to compare"


@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
@pytest.mark.parametrize("bias_direction", ["positive", "negative"])
def test_sensitivity_sm_bias_direction(bc_data, bias_direction):
    m_vec = [0.0, 0.1, 0.2]

    r_results = [
        r_conditional_cs_sd(
            betahat=bc_data["betahat"].tolist(),
            sigma=bc_data["sigma"].tolist(),
            num_pre_periods=bc_data["num_pre_periods"],
            num_post_periods=bc_data["num_post_periods"],
            m_bar=m,
            method="FLCI",
            grid_points=141,
            grid_lb=-0.1,
            grid_ub=0.6,
            bias_direction=bias_direction,
        )
        for m in m_vec
    ]

    if any(r_result is None for r_result in r_results):
        pytest.fail(f"R sensitivity SM with bias {bias_direction} failed")

    l_vec = basis_vector(1, bc_data["num_post_periods"])
    py_result = create_sensitivity_results_sm(
        betahat=bc_data["betahat"],
        sigma=bc_data["sigma"],
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        method="C-F",
        m_vec=np.array(m_vec),
        l_vec=l_vec,
        bias_direction=bias_direction,
        grid_points=141,
        grid_lb=-0.1,
        grid_ub=0.6,
    )

    compared = 0
    for m, r_result in zip(m_vec, r_results, strict=True):
        py_row = py_result.filter(pl.col("m") == m)
        if len(py_row) == 0:
            continue
        py_lb = py_row["lb"][0]
        py_ub = py_row["ub"][0]
        r_lb = _extract_scalar(r_result["lb"])
        r_ub = _extract_scalar(r_result["ub"])
        if not np.isnan(py_lb) and not np.isnan(r_lb):
            np.testing.assert_allclose(py_lb, r_lb, rtol=0, atol=0.005 + 1e-12)
            compared += 1
        if not np.isnan(py_ub) and not np.isnan(r_ub):
            np.testing.assert_allclose(py_ub, r_ub, rtol=0, atol=0.005 + 1e-12)
            compared += 1
    assert compared > 0, f"bias={bias_direction}: No non-NaN values to compare"


@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
def test_conditional_cs_sd_flci_bc_data(bc_data):
    m_bar = 0.1

    r_result = r_conditional_cs_sd(
        betahat=bc_data["betahat"].tolist(),
        sigma=bc_data["sigma"].tolist(),
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        m_bar=m_bar,
        method="FLCI",
        grid_points=141,
        grid_lb=-0.1,
        grid_ub=0.6,
    )

    if r_result is None:
        pytest.fail("R conditional CS SD FLCI failed")

    l_vec = basis_vector(1, bc_data["num_post_periods"])
    py_result = compute_conditional_cs_sd(
        betahat=bc_data["betahat"],
        sigma=bc_data["sigma"],
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        l_vec=l_vec,
        m_bar=m_bar,
        hybrid_flag="FLCI",
        grid_points=141,
        grid_lb=-0.1,
        grid_ub=0.6,
    )

    accept_idx = np.where(py_result["accept"])[0]
    if len(accept_idx) > 0:
        py_lb = py_result["grid"][accept_idx[0]]
        py_ub = py_result["grid"][accept_idx[-1]]
    else:
        py_lb = np.nan
        py_ub = np.nan

    r_lb = _extract_scalar(r_result["lb"])
    r_ub = _extract_scalar(r_result["ub"])

    compared = 0
    np.testing.assert_allclose(py_result["grid"], np.array(r_result["grid"]), rtol=0, atol=1e-12)
    if not np.isnan(py_lb) and not np.isnan(r_lb):
        np.testing.assert_allclose(py_lb, r_lb, rtol=0, atol=0.005 + 1e-12)
        compared += 1
    if not np.isnan(py_ub) and not np.isnan(r_ub):
        np.testing.assert_allclose(py_ub, r_ub, rtol=0, atol=0.005 + 1e-12)
        compared += 1
    assert compared > 0, "Conditional CS SD FLCI: No non-NaN values to compare"


@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
@pytest.mark.parametrize("method", ["LF", "ARP"])
def test_conditional_cs_sd_methods(bc_data, method):
    m_bar = 0.1

    r_result = r_conditional_cs_sd(
        betahat=bc_data["betahat"].tolist(),
        sigma=bc_data["sigma"].tolist(),
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        m_bar=m_bar,
        method=method,
        grid_points=141,
        grid_lb=-0.1,
        grid_ub=0.6,
    )

    if r_result is None:
        pytest.fail(f"R conditional CS SD {method} failed")

    l_vec = basis_vector(1, bc_data["num_post_periods"])
    py_result = compute_conditional_cs_sd(
        betahat=bc_data["betahat"],
        sigma=bc_data["sigma"],
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        l_vec=l_vec,
        m_bar=m_bar,
        hybrid_flag=method,
        grid_points=141,
        grid_lb=-0.1,
        grid_ub=0.6,
    )

    accept_idx = np.where(py_result["accept"])[0]
    if len(accept_idx) > 0:
        py_lb = py_result["grid"][accept_idx[0]]
        py_ub = py_result["grid"][accept_idx[-1]]
    else:
        py_lb = np.nan
        py_ub = np.nan

    r_lb = _extract_scalar(r_result["lb"])
    r_ub = _extract_scalar(r_result["ub"])

    compared = 0
    np.testing.assert_allclose(py_result["grid"], np.array(r_result["grid"]), rtol=0, atol=1e-12)
    if not np.isnan(py_lb) and not np.isnan(r_lb):
        np.testing.assert_allclose(py_lb, r_lb, rtol=0, atol=0.005 + 1e-12)
        compared += 1
    if not np.isnan(py_ub) and not np.isnan(r_ub):
        np.testing.assert_allclose(py_ub, r_ub, rtol=0, atol=0.005 + 1e-12)
        compared += 1
    assert compared > 0, f"conditional CS SD {method}: No non-NaN values to compare"


@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
def test_conditional_cs_rm_bc_data(bc_data):
    m_bar = 1.0

    l_vec = basis_vector(1, bc_data["num_post_periods"])
    py_result = compute_conditional_cs_rm(
        betahat=bc_data["betahat"],
        sigma=bc_data["sigma"],
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        l_vec=l_vec,
        m_bar=m_bar,
        hybrid_flag="LF",
        grid_points=100,
    )

    assert py_result is not None
    assert "grid" in py_result
    assert "accept" in py_result


def test_flci_produces_valid_ci(synthetic_event_study):
    l_vec = basis_vector(1, synthetic_event_study["num_post_periods"])
    result = compute_flci(
        beta_hat=synthetic_event_study["betahat"],
        sigma=synthetic_event_study["sigma"],
        n_pre_periods=synthetic_event_study["num_pre_periods"],
        n_post_periods=synthetic_event_study["num_post_periods"],
        smoothness_bound=0.05,
        post_period_weights=l_vec.flatten(),
    )

    assert result.flci[0] < result.flci[1]
    assert result.optimal_half_length > 0
    assert result.status == "optimal"


def test_identified_set_sd_contains_point_estimate(synthetic_event_study):
    l_vec = basis_vector(1, synthetic_event_study["num_post_periods"])
    point_est = l_vec.flatten() @ synthetic_event_study["betahat"][synthetic_event_study["num_pre_periods"] :]

    result = compute_identified_set_sd(
        m_bar=0.1,
        true_beta=synthetic_event_study["betahat"],
        l_vec=l_vec,
        num_pre_periods=synthetic_event_study["num_pre_periods"],
        num_post_periods=synthetic_event_study["num_post_periods"],
    )

    assert result.id_lb <= point_est <= result.id_ub


def test_identified_set_rm_contains_point_estimate(synthetic_event_study):
    l_vec = basis_vector(1, synthetic_event_study["num_post_periods"])
    point_est = l_vec.flatten() @ synthetic_event_study["betahat"][synthetic_event_study["num_pre_periods"] :]

    result = compute_identified_set_rm(
        m_bar=1.0,
        true_beta=synthetic_event_study["betahat"],
        l_vec=l_vec,
        num_pre_periods=synthetic_event_study["num_pre_periods"],
        num_post_periods=synthetic_event_study["num_post_periods"],
    )

    assert result.id_lb <= point_est <= result.id_ub


def test_sensitivity_sm_monotone_in_m(synthetic_event_study):
    m_vec = np.array([0.0, 0.05, 0.1, 0.15, 0.2])
    l_vec = basis_vector(1, synthetic_event_study["num_post_periods"])

    result = create_sensitivity_results_sm(
        betahat=synthetic_event_study["betahat"],
        sigma=synthetic_event_study["sigma"],
        num_pre_periods=synthetic_event_study["num_pre_periods"],
        num_post_periods=synthetic_event_study["num_post_periods"],
        method="FLCI",
        m_vec=m_vec,
        l_vec=l_vec,
    )

    ci_lengths = (result["ub"] - result["lb"]).to_numpy()
    for i in range(len(ci_lengths) - 1):
        if not np.isnan(ci_lengths[i]) and not np.isnan(ci_lengths[i + 1]):
            assert ci_lengths[i] <= ci_lengths[i + 1] + 1e-6


def test_sensitivity_rm_monotone_in_mbar(synthetic_event_study):
    m_bar_vec = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    l_vec = basis_vector(1, synthetic_event_study["num_post_periods"])

    result = create_sensitivity_results_rm(
        betahat=synthetic_event_study["betahat"],
        sigma=synthetic_event_study["sigma"],
        num_pre_periods=synthetic_event_study["num_pre_periods"],
        num_post_periods=synthetic_event_study["num_post_periods"],
        method="C-LF",
        m_bar_vec=m_bar_vec,
        l_vec=l_vec,
        grid_points=50,
    )

    ci_lengths = (result["ub"] - result["lb"]).to_numpy()
    for i in range(len(ci_lengths) - 1):
        if not np.isnan(ci_lengths[i]) and not np.isnan(ci_lengths[i + 1]):
            assert ci_lengths[i] <= ci_lengths[i + 1] + 1e-2


def test_original_cs_is_subset_of_robust_cs(synthetic_event_study):
    l_vec = basis_vector(1, synthetic_event_study["num_post_periods"])

    original = construct_original_cs(
        betahat=synthetic_event_study["betahat"],
        sigma=synthetic_event_study["sigma"],
        num_pre_periods=synthetic_event_study["num_pre_periods"],
        num_post_periods=synthetic_event_study["num_post_periods"],
        l_vec=l_vec,
    )

    robust = create_sensitivity_results_sm(
        betahat=synthetic_event_study["betahat"],
        sigma=synthetic_event_study["sigma"],
        num_pre_periods=synthetic_event_study["num_pre_periods"],
        num_post_periods=synthetic_event_study["num_post_periods"],
        method="FLCI",
        m_vec=np.array([0.1]),
        l_vec=l_vec,
    )

    robust_lb = robust["lb"][0]
    robust_ub = robust["ub"][0]

    assert robust_lb <= original.lb + 1e-4
    assert robust_ub >= original.ub - 1e-4


def test_conditional_cs_sd_with_sdm(synthetic_event_study):
    l_vec = basis_vector(1, synthetic_event_study["num_post_periods"])

    result = compute_conditional_cs_sdm(
        betahat=synthetic_event_study["betahat"],
        sigma=synthetic_event_study["sigma"],
        num_pre_periods=synthetic_event_study["num_pre_periods"],
        num_post_periods=synthetic_event_study["num_post_periods"],
        l_vec=l_vec,
        m_bar=0.1,
        monotonicity_direction="increasing",
        hybrid_flag="LF",
        grid_points=50,
    )

    assert result is not None
    assert "grid" in result
    assert "accept" in result


def test_conditional_cs_sd_with_sdb(synthetic_event_study):
    l_vec = basis_vector(1, synthetic_event_study["num_post_periods"])

    result = compute_conditional_cs_sdb(
        betahat=synthetic_event_study["betahat"],
        sigma=synthetic_event_study["sigma"],
        num_pre_periods=synthetic_event_study["num_pre_periods"],
        num_post_periods=synthetic_event_study["num_post_periods"],
        l_vec=l_vec,
        m_bar=0.1,
        bias_direction="positive",
        hybrid_flag="LF",
        grid_points=50,
    )

    assert result is not None
    assert "grid" in result
    assert "accept" in result


def test_different_l_vec_values(synthetic_event_study):
    for post_idx in range(1, synthetic_event_study["num_post_periods"] + 1):
        l_vec = basis_vector(post_idx, synthetic_event_study["num_post_periods"])

        result = compute_flci(
            beta_hat=synthetic_event_study["betahat"],
            sigma=synthetic_event_study["sigma"],
            n_pre_periods=synthetic_event_study["num_pre_periods"],
            n_post_periods=synthetic_event_study["num_post_periods"],
            smoothness_bound=0.05,
            post_period_weights=l_vec.flatten(),
        )

        assert result.flci[0] < result.flci[1]
        assert result.status == "optimal"


def test_sensitivity_sm_all_valid_ses(synthetic_event_study):
    m_vec = np.array([0.0, 0.05, 0.1])
    l_vec = basis_vector(1, synthetic_event_study["num_post_periods"])

    result = create_sensitivity_results_sm(
        betahat=synthetic_event_study["betahat"],
        sigma=synthetic_event_study["sigma"],
        num_pre_periods=synthetic_event_study["num_pre_periods"],
        num_post_periods=synthetic_event_study["num_post_periods"],
        method="FLCI",
        m_vec=m_vec,
        l_vec=l_vec,
    )

    for i in range(len(result)):
        lb = result["lb"][i]
        ub = result["ub"][i]
        if not np.isnan(lb) and not np.isnan(ub):
            assert lb < ub


@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
def test_full_pipeline_consistency(bc_data):
    m_vec = [0.0, 0.1, 0.2]
    l_vec = basis_vector(1, bc_data["num_post_periods"])

    r_original = r_construct_original_cs(
        betahat=bc_data["betahat"].tolist(),
        sigma=bc_data["sigma"].tolist(),
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
    )

    r_robust = r_sensitivity_sm(
        betahat=bc_data["betahat"].tolist(),
        sigma=bc_data["sigma"].tolist(),
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        m_vec=m_vec,
        method="FLCI",
    )

    if r_original is None or r_robust is None:
        pytest.fail("R full pipeline failed")

    py_original = construct_original_cs(
        betahat=bc_data["betahat"],
        sigma=bc_data["sigma"],
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        l_vec=l_vec,
    )

    py_robust = create_sensitivity_results_sm(
        betahat=bc_data["betahat"],
        sigma=bc_data["sigma"],
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        method="FLCI",
        m_vec=np.array(m_vec),
        l_vec=l_vec,
    )

    np.testing.assert_allclose(py_original.lb, r_original["lb"], rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(py_original.ub, r_original["ub"], rtol=1e-3, atol=1e-4)

    compared = 0
    for i, m in enumerate(m_vec):
        py_row = py_robust.filter(pl.col("m") == m)
        if len(py_row) > 0:
            py_lb = py_row["lb"][0]
            py_ub = py_row["ub"][0]
            r_lb = r_robust["lb"][i]
            r_ub = r_robust["ub"][i]

            if not np.isnan(py_lb) and not np.isnan(r_lb):
                np.testing.assert_allclose(py_lb, r_lb, rtol=0.1, atol=0.02)
                compared += 1
            if not np.isnan(py_ub) and not np.isnan(r_ub):
                np.testing.assert_allclose(py_ub, r_ub, rtol=0.1, atol=0.02)
                compared += 1
    assert compared > 0, "Full pipeline: No non-NaN robust values to compare"


# Tests for RM with bias direction (RMB)
@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
@pytest.mark.parametrize("m_bar", [0.5, 1.0])
@pytest.mark.parametrize("bias_direction", ["positive", "negative"])
def test_identified_set_rmb_bc_data(bc_data, m_bar, bias_direction):
    r_result = r_identified_set_rmb(
        betahat=bc_data["betahat"].tolist(),
        sigma=bc_data["sigma"].tolist(),
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        m_bar=m_bar,
        bias_direction=bias_direction,
    )

    if r_result is None:
        pytest.fail("R identified set RMB computation failed")

    l_vec = basis_vector(1, bc_data["num_post_periods"]).flatten()
    py_result = compute_identified_set_rmb(
        m_bar=m_bar,
        true_beta=bc_data["betahat"],
        l_vec=l_vec,
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        bias_direction=bias_direction,
    )

    r_lb = _extract_scalar(r_result["id_lb"])
    r_ub = _extract_scalar(r_result["id_ub"])

    np.testing.assert_allclose(py_result.id_lb, r_lb, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(py_result.id_ub, r_ub, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
@pytest.mark.parametrize("bias_direction", ["positive", "negative"])
def test_conditional_cs_rmb_bc_data(bc_data, bias_direction):
    r_result = r_conditional_cs_rmb(
        betahat=bc_data["betahat"].tolist(),
        sigma=bc_data["sigma"].tolist(),
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        m_bar=0.5,
        bias_direction=bias_direction,
        method="LF",
        grid_points=101,
        grid_lb=-0.3,
        grid_ub=0.7,
    )

    if r_result is None:
        pytest.fail("R conditional CS RMB computation failed")

    l_vec = basis_vector(1, bc_data["num_post_periods"]).flatten()
    py_result = compute_conditional_cs_rmb(
        betahat=bc_data["betahat"],
        sigma=bc_data["sigma"],
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        l_vec=l_vec,
        m_bar=0.5,
        bias_direction=bias_direction,
        hybrid_flag="LF",
        grid_points=101,
        grid_lb=-0.3,
        grid_ub=0.7,
    )

    accept_idx = np.where(py_result["accept"])[0]
    if len(accept_idx) > 0:
        py_lb = py_result["grid"][accept_idx[0]]
        py_ub = py_result["grid"][accept_idx[-1]]
    else:
        py_lb, py_ub = np.nan, np.nan

    r_lb = _extract_scalar(r_result["lb"])
    r_ub = _extract_scalar(r_result["ub"])

    compared = 0
    np.testing.assert_allclose(py_result["grid"], np.array(r_result["grid"]), rtol=0, atol=1e-12)
    if not np.isnan(py_lb) and not np.isnan(r_lb):
        np.testing.assert_allclose(py_lb, r_lb, rtol=0, atol=0.01 + 1e-12)
        compared += 1
    if not np.isnan(py_ub) and not np.isnan(r_ub):
        np.testing.assert_allclose(py_ub, r_ub, rtol=0, atol=0.01 + 1e-12)
        compared += 1
    assert compared > 0, f"Conditional CS RMB {bias_direction}: No non-NaN values to compare"


# Tests for RM with monotonicity (RMM)
@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
@pytest.mark.parametrize("m_bar", [0.5, 1.0])
@pytest.mark.parametrize("monotonicity_direction", ["increasing", "decreasing"])
def test_identified_set_rmm_bc_data(bc_data, m_bar, monotonicity_direction):
    r_result = r_identified_set_rmm(
        betahat=bc_data["betahat"].tolist(),
        sigma=bc_data["sigma"].tolist(),
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        m_bar=m_bar,
        monotonicity_direction=monotonicity_direction,
    )

    if r_result is None:
        pytest.fail("R identified set RMM computation failed")

    l_vec = basis_vector(1, bc_data["num_post_periods"]).flatten()
    py_result = compute_identified_set_rmm(
        m_bar=m_bar,
        true_beta=bc_data["betahat"],
        l_vec=l_vec,
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        monotonicity_direction=monotonicity_direction,
    )

    r_lb = _extract_scalar(r_result["id_lb"])
    r_ub = _extract_scalar(r_result["id_ub"])

    np.testing.assert_allclose(py_result.id_lb, r_lb, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(py_result.id_ub, r_ub, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
@pytest.mark.parametrize("monotonicity_direction", ["increasing", "decreasing"])
def test_conditional_cs_rmm_bc_data(bc_data, monotonicity_direction):
    r_result = r_conditional_cs_rmm(
        betahat=bc_data["betahat"].tolist(),
        sigma=bc_data["sigma"].tolist(),
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        m_bar=0.5,
        monotonicity_direction=monotonicity_direction,
        method="LF",
        grid_points=101,
        grid_lb=-0.3,
        grid_ub=0.7,
    )

    if r_result is None:
        pytest.fail("R conditional CS RMM computation failed")

    l_vec = basis_vector(1, bc_data["num_post_periods"]).flatten()
    py_result = compute_conditional_cs_rmm(
        betahat=bc_data["betahat"],
        sigma=bc_data["sigma"],
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        l_vec=l_vec,
        m_bar=0.5,
        monotonicity_direction=monotonicity_direction,
        hybrid_flag="LF",
        grid_points=101,
        grid_lb=-0.3,
        grid_ub=0.7,
    )

    accept_idx = np.where(py_result["accept"])[0]
    if len(accept_idx) > 0:
        py_lb = py_result["grid"][accept_idx[0]]
        py_ub = py_result["grid"][accept_idx[-1]]
    else:
        py_lb, py_ub = np.nan, np.nan

    r_lb = _extract_scalar(r_result["lb"])
    r_ub = _extract_scalar(r_result["ub"])

    compared = 0
    np.testing.assert_allclose(py_result["grid"], np.array(r_result["grid"]), rtol=0, atol=1e-12)
    if not np.isnan(py_lb) and not np.isnan(r_lb):
        np.testing.assert_allclose(py_lb, r_lb, rtol=0, atol=0.01 + 1e-12)
        compared += 1
    if not np.isnan(py_ub) and not np.isnan(r_ub):
        np.testing.assert_allclose(py_ub, r_ub, rtol=0, atol=0.01 + 1e-12)
        compared += 1
    assert compared > 0, f"Conditional CS RMM m_bar=0.5 {monotonicity_direction}: No non-NaN values to compare"


# Tests for SD + RM combined (SDRM)
@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
@pytest.mark.parametrize("m_bar", [0.5, 1.0])
def test_identified_set_sdrm_bc_data(bc_data, m_bar):
    r_result = r_identified_set_sdrm(
        betahat=bc_data["betahat"].tolist(),
        sigma=bc_data["sigma"].tolist(),
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        m_bar=m_bar,
    )

    if r_result is None:
        pytest.fail("R identified set SDRM computation failed")

    l_vec = basis_vector(1, bc_data["num_post_periods"]).flatten()
    py_result = compute_identified_set_sdrm(
        m_bar=m_bar,
        true_beta=bc_data["betahat"],
        l_vec=l_vec,
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
    )

    r_lb = _extract_scalar(r_result["id_lb"])
    r_ub = _extract_scalar(r_result["id_ub"])

    np.testing.assert_allclose(py_result.id_lb, r_lb, rtol=0.05, atol=1e-3)
    np.testing.assert_allclose(py_result.id_ub, r_ub, rtol=0.05, atol=1e-3)


@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
def test_conditional_cs_sdrm_bc_data(bc_data):
    r_result = r_conditional_cs_sdrm(
        betahat=bc_data["betahat"].tolist(),
        sigma=bc_data["sigma"].tolist(),
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        m_bar=0.5,
        method="LF",
    )

    if r_result is None:
        pytest.fail("R conditional CS SDRM computation failed")

    l_vec = basis_vector(1, bc_data["num_post_periods"])
    py_result = compute_conditional_cs_sdrm(
        betahat=bc_data["betahat"],
        sigma=bc_data["sigma"],
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        l_vec=l_vec,
        m_bar=0.5,
        hybrid_flag="LF",
        grid_points=100,
    )

    accept_idx = np.where(py_result["accept"])[0]
    if len(accept_idx) > 0:
        py_lb = py_result["grid"][accept_idx[0]]
        py_ub = py_result["grid"][accept_idx[-1]]
    else:
        py_lb, py_ub = np.nan, np.nan

    r_lb = _extract_scalar(r_result["lb"])
    r_ub = _extract_scalar(r_result["ub"])

    compared = 0
    if not np.isnan(py_lb) and not np.isnan(r_lb):
        np.testing.assert_allclose(py_lb, r_lb, rtol=0.15, atol=0.05)
        compared += 1
    if not np.isnan(py_ub) and not np.isnan(r_ub):
        np.testing.assert_allclose(py_ub, r_ub, rtol=0.15, atol=0.05)
        compared += 1
    assert compared > 0, "Conditional CS SDRM m_bar=0.5: No non-NaN values to compare"


# Tests for SD + RM with bias direction (SDRMB)
@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
@pytest.mark.parametrize("m_bar", [0.5, 1.0])
@pytest.mark.parametrize("bias_direction", ["positive", "negative"])
def test_identified_set_sdrmb_bc_data(bc_data, m_bar, bias_direction):
    r_result = r_identified_set_sdrmb(
        betahat=bc_data["betahat"].tolist(),
        sigma=bc_data["sigma"].tolist(),
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        m_bar=m_bar,
        bias_direction=bias_direction,
    )

    if r_result is None:
        pytest.fail("R identified set SDRMB computation failed")

    l_vec = basis_vector(1, bc_data["num_post_periods"]).flatten()
    py_result = compute_identified_set_sdrmb(
        m_bar=m_bar,
        true_beta=bc_data["betahat"],
        l_vec=l_vec,
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        bias_direction=bias_direction,
    )

    r_lb = _extract_scalar(r_result["id_lb"])
    r_ub = _extract_scalar(r_result["id_ub"])

    np.testing.assert_allclose(py_result.id_lb, r_lb, rtol=0.05, atol=1e-3)
    np.testing.assert_allclose(py_result.id_ub, r_ub, rtol=0.05, atol=1e-3)


@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
@pytest.mark.parametrize("bias_direction", ["positive", "negative"])
def test_conditional_cs_sdrmb_bc_data(bc_data, bias_direction):
    r_result = r_conditional_cs_sdrmb(
        betahat=bc_data["betahat"].tolist(),
        sigma=bc_data["sigma"].tolist(),
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        m_bar=0.5,
        bias_direction=bias_direction,
        method="LF",
    )

    if r_result is None:
        pytest.fail("R conditional CS SDRMB computation failed")

    l_vec = basis_vector(1, bc_data["num_post_periods"])
    py_result = compute_conditional_cs_sdrmb(
        betahat=bc_data["betahat"],
        sigma=bc_data["sigma"],
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        l_vec=l_vec,
        m_bar=0.5,
        bias_direction=bias_direction,
        hybrid_flag="LF",
        grid_points=100,
    )

    accept_idx = np.where(py_result["accept"])[0]
    if len(accept_idx) > 0:
        py_lb = py_result["grid"][accept_idx[0]]
        py_ub = py_result["grid"][accept_idx[-1]]
    else:
        py_lb, py_ub = np.nan, np.nan

    r_lb = _extract_scalar(r_result["lb"])
    r_ub = _extract_scalar(r_result["ub"])

    compared = 0
    if not np.isnan(py_lb) and not np.isnan(r_lb):
        np.testing.assert_allclose(py_lb, r_lb, rtol=0.15, atol=0.05)
        compared += 1
    if not np.isnan(py_ub) and not np.isnan(r_ub):
        np.testing.assert_allclose(py_ub, r_ub, rtol=0.15, atol=0.05)
        compared += 1
    assert compared > 0, f"Conditional CS SDRMB m_bar=0.5 {bias_direction}: No non-NaN values to compare"


# Tests for SD + RM with monotonicity (SDRMM)
@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
@pytest.mark.parametrize("m_bar", [0.5, 1.0])
@pytest.mark.parametrize("monotonicity_direction", ["increasing", "decreasing"])
def test_identified_set_sdrmm_bc_data(bc_data, m_bar, monotonicity_direction):
    r_result = r_identified_set_sdrmm(
        betahat=bc_data["betahat"].tolist(),
        sigma=bc_data["sigma"].tolist(),
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        m_bar=m_bar,
        monotonicity_direction=monotonicity_direction,
    )

    if r_result is None:
        pytest.fail("R identified set SDRMM computation failed")

    l_vec = basis_vector(1, bc_data["num_post_periods"]).flatten()
    py_result = compute_identified_set_sdrmm(
        m_bar=m_bar,
        true_beta=bc_data["betahat"],
        l_vec=l_vec,
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        monotonicity_direction=monotonicity_direction,
    )

    r_lb = _extract_scalar(r_result["id_lb"])
    r_ub = _extract_scalar(r_result["id_ub"])

    np.testing.assert_allclose(py_result.id_lb, r_lb, rtol=0.05, atol=1e-3)
    np.testing.assert_allclose(py_result.id_ub, r_ub, rtol=0.05, atol=1e-3)


@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
@pytest.mark.parametrize("monotonicity_direction", ["increasing", "decreasing"])
def test_conditional_cs_sdrmm_bc_data(bc_data, monotonicity_direction):
    r_result = r_conditional_cs_sdrmm(
        betahat=bc_data["betahat"].tolist(),
        sigma=bc_data["sigma"].tolist(),
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        m_bar=0.5,
        monotonicity_direction=monotonicity_direction,
        method="LF",
    )

    if r_result is None:
        pytest.fail("R conditional CS SDRMM computation failed")

    l_vec = basis_vector(1, bc_data["num_post_periods"])
    py_result = compute_conditional_cs_sdrmm(
        betahat=bc_data["betahat"],
        sigma=bc_data["sigma"],
        num_pre_periods=bc_data["num_pre_periods"],
        num_post_periods=bc_data["num_post_periods"],
        l_vec=l_vec,
        m_bar=0.5,
        monotonicity_direction=monotonicity_direction,
        hybrid_flag="LF",
        grid_points=100,
    )

    accept_idx = np.where(py_result["accept"])[0]
    if len(accept_idx) > 0:
        py_lb = py_result["grid"][accept_idx[0]]
        py_ub = py_result["grid"][accept_idx[-1]]
    else:
        py_lb, py_ub = np.nan, np.nan

    r_lb = _extract_scalar(r_result["lb"])
    r_ub = _extract_scalar(r_result["ub"])

    compared = 0
    if not np.isnan(py_lb) and not np.isnan(r_lb):
        np.testing.assert_allclose(py_lb, r_lb, rtol=0.15, atol=0.05)
        compared += 1
    if not np.isnan(py_ub) and not np.isnan(r_ub):
        np.testing.assert_allclose(py_ub, r_ub, rtol=0.15, atol=0.05)
        compared += 1
    assert compared > 0, f"Conditional CS SDRMM m_bar=0.5 {monotonicity_direction}: No non-NaN values to compare"


def _r_vector(x):
    return "c(" + ",".join(repr(float(v)) for v in np.ravel(x)) + ")"


def _r_matrix(m):
    m = np.asarray(m, dtype=float)
    return f"matrix({_r_vector(m)}, nrow={m.shape[0]}, byrow=TRUE)"


def r_honestdid_json(body):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    r_script = f"""
suppressPackageStartupMessages({{library(HonestDiD); library(jsonlite)}})
{body}
write_json(out, "{result_path}", digits = NA, auto_unbox = TRUE)
"""
    try:
        return _run_r_script(r_script, result_path, timeout=600)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
@pytest.mark.parametrize(
    "case,restriction,m_bar",
    [
        ("C", "RM", 1.0),
        ("A", "RMBpos", 1.0),
        ("A", "RMI", 1.0),
        ("B", "RMBpos", 0.5),
        ("A", "RMI", 0.5),
    ],
)
def test_conditional_cs_rm_family_matches_reference(rm_cases, case, restriction, m_bar):
    betahat, sigma = rm_cases[case], rm_cases["sigma"]
    r_call = {
        "RM": "computeConditionalCS_DeltaRM(",
        "RMBpos": "computeConditionalCS_DeltaRMB(biasDirection = 'positive', ",
        "RMI": "computeConditionalCS_DeltaRMM(monotonicityDirection = 'increasing', ",
    }[restriction]
    r_result = r_honestdid_json(
        f"ci <- {r_call}betahat = {_r_vector(betahat)}, sigma = {_r_matrix(sigma)}, numPrePeriods = 3, "
        f"numPostPeriods = 2, l_vec = c(1, 0), Mbar = {m_bar}, hybrid_flag = 'ARP', gridPoints = 61, "
        "grid.lb = -0.1, grid.ub = 0.2)\nout <- list(grid = ci$grid, accept = ci$accept)"
    )
    if r_result is None:
        pytest.fail("R relative magnitudes confidence set failed")

    py_fn = {
        "RM": compute_conditional_cs_rm,
        "RMBpos": partial(compute_conditional_cs_rmb, bias_direction="positive"),
        "RMI": partial(compute_conditional_cs_rmm, monotonicity_direction="increasing"),
    }[restriction]
    py_result = py_fn(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=3,
        num_post_periods=2,
        l_vec=np.array([1.0, 0.0]),
        m_bar=m_bar,
        hybrid_flag="ARP",
        grid_points=61,
        grid_lb=-0.1,
        grid_ub=0.2,
    )

    np.testing.assert_allclose(py_result["grid"], np.array(r_result["grid"]), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(py_result["accept"], np.array(r_result["accept"], dtype=float))


@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
@pytest.mark.parametrize("theta", [0.0, 0.03, 0.06, 0.09])
def test_dual_truncation_bounds_match_reference(rm_cases, theta):
    a_matrix = _create_relative_magnitudes_bias_constraint_matrix(
        num_pre_periods=3, num_post_periods=2, m_bar=0.5, s=-2, max_positive=False, bias_direction="positive"
    )
    rows = find_rows_with_post_period_values(a_matrix, [3, 4])
    a_gamma_inv = a_matrix[:, 3:5] @ np.linalg.inv(_construct_gamma(np.array([1.0, 0.0])))
    y_t = (a_matrix @ rm_cases["A"] - a_gamma_inv[:, 0] * theta)[rows]
    x_t = a_gamma_inv[rows][:, 1:]
    sigma = (a_matrix @ rm_cases["sigma"] @ a_matrix.T)[np.ix_(rows, rows)]
    lin_soln = _test_delta_lp(y_t, x_t, sigma)
    gamma_tilde = lin_soln["lambda"]
    s_t = y_t - sigma @ gamma_tilde * (gamma_tilde @ y_t) / (gamma_tilde @ sigma @ gamma_tilde)
    w_t = np.column_stack([np.sqrt(np.diag(sigma)), x_t])

    r_result = r_honestdid_json(
        f"v <- HonestDiD:::.vlo_vup_dual_fn(eta = {float(lin_soln['eta_star'])!r}, s_T = {_r_vector(s_t)}, "
        f"gamma_tilde = {_r_vector(gamma_tilde)}, sigma = {_r_matrix(sigma)}, W_T = {_r_matrix(w_t)})\n"
        "out <- list(vlo = v$vlo, vup_infinite = is.infinite(v$vup))"
    )
    if r_result is None:
        pytest.fail("R dual truncation bounds failed")

    py_result = compute_vlo_vup_dual(lin_soln["eta_star"], s_t, gamma_tilde, sigma, w_t)

    assert py_result["vlo"] == pytest.approx(r_result["vlo"], abs=1e-12)
    assert r_result["vup_infinite"]
    assert py_result["vup"] == np.inf


@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
def test_identified_set_rm_earliest_difference_matches_reference():
    betahat = np.array([0.10, 0.0, 0.01, 0.0, 0.05, 0.06])
    r_result = r_honestdid_json(
        f"b <- {_r_vector(betahat)}\n"
        "rm <- HonestDiD:::.compute_IDset_DeltaRM(Mbar = 1, trueBeta = b, l_vec = c(1, 0), numPrePeriods = 4, "
        "numPostPeriods = 2)\n"
        "rmb <- HonestDiD:::.compute_IDset_DeltaRMB(Mbar = 1, trueBeta = b, l_vec = c(1, 0), numPrePeriods = 4, "
        "numPostPeriods = 2, biasDirection = 'negative')\n"
        "out <- list(rm = c(rm$id.lb, rm$id.ub), rmb = c(rmb$id.lb, rmb$id.ub))"
    )
    if r_result is None:
        pytest.fail("R relative magnitudes identified set failed")

    l_vec = np.array([1.0, 0.0])
    rm = compute_identified_set_rm(1.0, betahat, l_vec, 4, 2)
    rmb = compute_identified_set_rmb(1.0, betahat, l_vec, 4, 2, bias_direction="negative")

    np.testing.assert_allclose([rm.id_lb, rm.id_ub], r_result["rm"], rtol=0, atol=1e-10)
    np.testing.assert_allclose([rmb.id_lb, rmb.id_ub], r_result["rmb"], rtol=0, atol=1e-10)
    assert rm.id_lb == pytest.approx(-0.05)


@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
@pytest.mark.parametrize(
    "restriction,hybrid_flag",
    [
        ("SD", "ARP"),
        ("SD", "FLCI"),
        ("SDBpos", "ARP"),
        ("SDBpos", "FLCI"),
        ("SDI", "ARP"),
        ("RM", "ARP"),
        ("RMBpos", "ARP"),
        ("RMI", "ARP"),
    ],
)
def test_single_post_period_matches_reference(bc_data, restriction, hybrid_flag):
    betahat = bc_data["betahat"][:5]
    sigma = bc_data["sigma"][:5, :5]
    bound_name, bound = ("Mbar", 1.0) if restriction.startswith("RM") else ("M", 0.2)
    r_call = {
        "SD": "computeConditionalCS_DeltaSD(",
        "SDBpos": "computeConditionalCS_DeltaSDB(biasDirection = 'positive', ",
        "SDI": "computeConditionalCS_DeltaSDM(monotonicityDirection = 'increasing', ",
        "RM": "computeConditionalCS_DeltaRM(",
        "RMBpos": "computeConditionalCS_DeltaRMB(biasDirection = 'positive', ",
        "RMI": "computeConditionalCS_DeltaRMM(monotonicityDirection = 'increasing', ",
    }[restriction]
    r_result = r_honestdid_json(
        f"ci <- suppressWarnings({r_call}betahat = {_r_vector(betahat)}, sigma = {_r_matrix(sigma)}, "
        f"numPrePeriods = 4, numPostPeriods = 1, l_vec = 1, {bound_name} = {bound}, hybrid_flag = '{hybrid_flag}', "
        "gridPoints = 81, grid.lb = -0.2, grid.ub = 0.6))\nout <- list(grid = ci$grid, accept = ci$accept)"
    )
    if r_result is None:
        pytest.fail("R single post-period confidence set failed")

    py_fn = {
        "SD": compute_conditional_cs_sd,
        "SDBpos": partial(compute_conditional_cs_sdb, bias_direction="positive"),
        "SDI": partial(compute_conditional_cs_sdm, monotonicity_direction="increasing"),
        "RM": compute_conditional_cs_rm,
        "RMBpos": partial(compute_conditional_cs_rmb, bias_direction="positive"),
        "RMI": partial(compute_conditional_cs_rmm, monotonicity_direction="increasing"),
    }[restriction]
    py_result = py_fn(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=4,
        num_post_periods=1,
        l_vec=np.array([1.0]),
        m_bar=bound,
        hybrid_flag=hybrid_flag,
        grid_points=81,
        grid_lb=-0.2,
        grid_ub=0.6,
    )

    np.testing.assert_allclose(py_result["grid"], np.array(r_result["grid"]), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(
        np.asarray(py_result["accept"], dtype=float), np.array(r_result["accept"], dtype=float)
    )


@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
def test_delta_sd_m_bounds_match_reference(bc_data):
    betahat = bc_data["betahat"]
    sigma = bc_data["sigma"]
    r_result = r_honestdid_json(
        f"b <- {_r_vector(betahat)}\ns <- {_r_matrix(sigma)}\n"
        "out <- list(ub = DeltaSD_upperBound_Mpre(b, s, 4), ub_two = DeltaSD_upperBound_Mpre(b[3:8], s[3:8, 3:8], 2), "
        "lb = suppressWarnings(DeltaSD_lowerBound_Mpre(b, s, 4, grid.ub = 0.3, gridPoints = 301)))"
    )
    if r_result is None:
        pytest.fail("R bounds on M failed")

    upper = compute_delta_sd_upperbound_m(betahat, sigma, 4)
    upper_two = compute_delta_sd_upperbound_m(betahat[2:], sigma[2:, 2:], 2)
    lower = compute_delta_sd_lowerbound_m(betahat, sigma, 4, grid_ub=0.3, grid_points=301)

    assert upper == pytest.approx(r_result["ub"], rel=1e-10)
    assert upper_two == pytest.approx(r_result["ub_two"], rel=1e-10)
    assert lower == pytest.approx(r_result["lb"], abs=1e-12)


@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
def test_flci_zero_smoothness_matches_reference(bc_data):
    betahat = bc_data["betahat"]
    sigma = bc_data["sigma"]
    r_result = r_honestdid_json(
        f"f <- findOptimalFLCI(betahat = {_r_vector(betahat)}, sigma = {_r_matrix(sigma)}, M = 0, "
        "numPrePeriods = 4, numPostPeriods = 4)\n"
        "out <- list(ci = f$FLCI, half = f$optimalHalfLength, q = HonestDiD:::.qfoldednormal(p = 0.95, mu = 0))"
    )
    if r_result is None:
        pytest.fail("R FLCI failed")

    py_result = compute_flci(betahat, sigma, 0.0, 4, 4)
    rescaled_half = py_result.optimal_half_length * r_result["q"] / stats.norm.ppf(0.975)

    assert np.mean(py_result.flci) == pytest.approx(np.mean(r_result["ci"]), abs=1e-5)
    assert rescaled_half == pytest.approx(r_result["half"], rel=1e-6)


@pytest.mark.skipif(not R_HONESTDID_AVAILABLE, reason="R HonestDiD package not available")
def test_truncated_normal_quantile_matches_reference():
    lower = [5.0, 8.0, 8.284, 9.0, 12.0, 30.0]
    r_result = r_honestdid_json(
        f"out <- sapply({_r_vector(lower)}, function(l) HonestDiD:::.norminvp_generalized(0.95, l, Inf))"
    )
    if r_result is None:
        pytest.fail("R truncated normal quantile failed")

    py_result = [_norminvp_generalized(0.95, value, np.inf) for value in lower]

    np.testing.assert_allclose(py_result, r_result, rtol=1e-12)
