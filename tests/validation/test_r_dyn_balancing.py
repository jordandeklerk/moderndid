"""Validation tests comparing Python dyn_balancing with R DynBalancing package."""

import json
import subprocess
import tempfile

import pytest
from scipy.stats import chi2

pytestmark = pytest.mark.slow

from tests.helpers import importorskip

pl = importorskip("polars")
np = importorskip("numpy")

from moderndid.dev.diddynamic import dyn_balancing

R_PKG_PATH = "DynBalancing"


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
            input='library(quadprog); library(glmnet); library(jsonlite); cat("OK")',
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return "OK" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


R_AVAILABLE = check_r_available()


def r_dyn_balancing_ate(
    data_path,
    covariates,
    ds1,
    ds2,
    method="lasso_plain",
    ub=10,
    fixed_effects=None,
    pooled=False,
    cluster_se=None,
    alpha=0.05,
):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    cov_str = 'c("' + '", "'.join(covariates) + '")'
    ds1_str = "c(" + ", ".join(str(d) for d in ds1) + ")"
    ds2_str = "c(" + ", ".join(str(d) for d in ds2) + ")"

    fe_str = "NA"
    if fixed_effects:
        fe_str = 'c("' + '", "'.join(fixed_effects) + '")'

    pooled_str = "TRUE" if pooled else "FALSE"

    params_parts = [
        f'method = "{method}"',
        "open_source = TRUE",
        "fast_adaptive = FALSE",
        f"alpha = {alpha}",
        f"ub = {ub}",
    ]
    if cluster_se:
        params_parts.append(f'cluster_SE = "{cluster_se}"')
    params_str = "list(" + ", ".join(params_parts) + ")"

    r_script = f"""
load("{R_PKG_PATH}/data/params_default.rda")
fpath <- "{R_PKG_PATH}/R"
source_files <- list.files(fpath, full.names=TRUE, pattern="[.]R$")
for(f in source_files) source(f)

library(jsonlite)

panel <- read.csv("{data_path}", check.names=FALSE)

result <- DynBalancing_ATE(
    panel,
    covariates_names = {cov_str},
    Time_name = "Time",
    unit_name = "Unit",
    outcome_name = "Y",
    treatment_name = "D",
    ds1 = {ds1_str},
    ds2 = {ds2_str},
    fixed_effects = {fe_str},
    pooled = {pooled_str},
    params = {params_str}
)

s <- result$summaries
out <- list(
    ATE = as.numeric(s[['ATE']]),
    Var_ATE = as.numeric(s[['Var_ATE']]),
    Mu1 = as.numeric(s[['Mu1']]),
    Mu2 = as.numeric(s[['Mu2']]),
    Var_mu1 = as.numeric(s[['Var_mu1']]),
    Var_mu2 = as.numeric(s[['Var_mu2']]),
    Robust_Quantile_ATE = as.numeric(s[['Robust_Quantile_ATE']]),
    Robust_Quantile_mu = as.numeric(s[['Robust_Quantile_mu']])
)

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path, timeout=300)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


def _prepare_panel_csv(covariates, extra_cols=None):
    """Write the Acemoglu panel data to a temporary CSV with numeric Unit column."""
    from moderndid.core.data import load_acemoglu

    df = load_acemoglu()
    units = sorted(df["Unit"].unique().to_list())
    unit_map = {u: i for i, u in enumerate(units)}
    df = df.with_columns(pl.col("Unit").replace(unit_map).cast(pl.Int64))

    keep_cols = ["Y", "D", "Unit", "Time"] + covariates
    if extra_cols:
        keep_cols.extend(extra_cols)
    df = df.select([c for c in keep_cols if c in df.columns])

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
    df.write_csv(path)
    return path


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_ate_lasso_plain_2periods():
    covariates = ["V1", "V2", "V3", "V4", "V5"]
    data_path = _prepare_panel_csv(covariates)

    r_result = r_dyn_balancing_ate(data_path, covariates, ds1=[1, 1], ds2=[0, 0])
    if r_result is None:
        pytest.skip("R estimation failed")

    df = pl.read_csv(data_path)
    py_result = dyn_balancing(
        data=df,
        yname="Y",
        tname="Time",
        idname="Unit",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        xformla="~ " + " + ".join(covariates),
        ub=10.0,
    )

    np.testing.assert_allclose(py_result.att, r_result["ATE"], atol=0.1)
    np.testing.assert_allclose(py_result.mu1, r_result["Mu1"], atol=0.1)
    np.testing.assert_allclose(py_result.mu2, r_result["Mu2"], atol=0.1)


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_ate_with_synthetic_data():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        data_path = f.name

    rng = np.random.default_rng(42)
    n = 100
    n_periods = 2
    unit = np.repeat(np.arange(1, n + 1), n_periods)
    time = np.tile(np.arange(1, n_periods + 1), n)
    x1 = rng.standard_normal(n * n_periods)
    x2 = rng.standard_normal(n * n_periods)
    d = rng.integers(0, 2, size=n * n_periods).astype(float)
    y = 1.0 + x1 + 0.5 * x2 + 2.0 * d + rng.standard_normal(n * n_periods) * 0.3

    df = pl.DataFrame({"Unit": unit, "Time": time, "Y": y, "D": d, "V1": x1, "V2": x2})
    df.write_csv(data_path)

    r_result = r_dyn_balancing_ate(data_path, ["V1", "V2"], ds1=[1, 1], ds2=[0, 0])
    if r_result is None:
        pytest.skip("R estimation failed")

    py_result = dyn_balancing(
        data=df,
        yname="Y",
        tname="Time",
        idname="Unit",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        xformla="~ V1 + V2",
        ub=10.0,
    )

    np.testing.assert_allclose(py_result.att, r_result["ATE"], atol=0.1)
    np.testing.assert_allclose(py_result.mu1, r_result["Mu1"], atol=0.1)
    np.testing.assert_allclose(py_result.mu2, r_result["Mu2"], atol=0.1)


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_variance_close_to_r():
    covariates = ["V1", "V2", "V3", "V4", "V5"]
    data_path = _prepare_panel_csv(covariates)

    r_result = r_dyn_balancing_ate(data_path, covariates, ds1=[1, 1], ds2=[0, 0])
    if r_result is None:
        pytest.skip("R estimation failed")

    df = pl.read_csv(data_path)
    py_result = dyn_balancing(
        data=df,
        yname="Y",
        tname="Time",
        idname="Unit",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        xformla="~ " + " + ".join(covariates),
        ub=10.0,
    )

    np.testing.assert_allclose(py_result.var_att, r_result["Var_ATE"], rtol=0.5)
    np.testing.assert_allclose(py_result.var_mu1, r_result["Var_mu1"], rtol=0.5)
    np.testing.assert_allclose(py_result.var_mu2, r_result["Var_mu2"], rtol=0.5)


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_ate_tight_tolerance():
    covariates = ["V1", "V2", "V3", "V4", "V5"]
    data_path = _prepare_panel_csv(covariates)

    r_result = r_dyn_balancing_ate(data_path, covariates, ds1=[1, 1], ds2=[0, 0])
    if r_result is None:
        pytest.skip("R estimation failed")

    df = pl.read_csv(data_path)
    py_result = dyn_balancing(
        data=df,
        yname="Y",
        tname="Time",
        idname="Unit",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        xformla="~ " + " + ".join(covariates),
        ub=10.0,
    )

    np.testing.assert_allclose(py_result.att, r_result["ATE"], atol=1e-3)
    np.testing.assert_allclose(py_result.mu1, r_result["Mu1"], atol=1e-3)
    np.testing.assert_allclose(py_result.mu2, r_result["Mu2"], atol=1e-3)


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_transition_into_treatment():
    covariates = ["V1", "V2", "V3"]
    data_path = _prepare_panel_csv(covariates)

    r_result = r_dyn_balancing_ate(data_path, covariates, ds1=[0, 1], ds2=[0, 0])
    if r_result is None:
        pytest.skip("R estimation failed")

    df = pl.read_csv(data_path)
    py_result = dyn_balancing(
        data=df,
        yname="Y",
        tname="Time",
        idname="Unit",
        treatment_name="D",
        ds1=[0, 1],
        ds2=[0, 0],
        xformla="~ " + " + ".join(covariates),
        ub=10.0,
    )

    np.testing.assert_allclose(py_result.att, r_result["ATE"], atol=0.1)
    np.testing.assert_allclose(py_result.mu1, r_result["Mu1"], atol=0.1)
    np.testing.assert_allclose(py_result.mu2, r_result["Mu2"], atol=0.1)


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_more_covariates():
    covariates = [f"V{i}" for i in range(1, 11)]
    data_path = _prepare_panel_csv(covariates)

    r_result = r_dyn_balancing_ate(data_path, covariates, ds1=[1, 1], ds2=[0, 0])
    if r_result is None:
        pytest.skip("R estimation failed")

    df = pl.read_csv(data_path)
    py_result = dyn_balancing(
        data=df,
        yname="Y",
        tname="Time",
        idname="Unit",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        xformla="~ " + " + ".join(covariates),
        ub=10.0,
    )

    np.testing.assert_allclose(py_result.att, r_result["ATE"], atol=0.1)


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_non_adaptive_balancing():
    covariates = ["V1", "V2", "V3", "V4", "V5"]

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        data_path = f.name

    from moderndid.core.data import load_acemoglu

    df = load_acemoglu()
    units = sorted(df["Unit"].unique().to_list())
    unit_map = {u: i for i, u in enumerate(units)}
    df = df.with_columns(pl.col("Unit").replace(unit_map).cast(pl.Int64))
    df.select(["Y", "D", "Unit", "Time"] + covariates).write_csv(data_path)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    cov_str = 'c("' + '", "'.join(covariates) + '")'

    r_script = f"""
load("{R_PKG_PATH}/data/params_default.rda")
fpath <- "{R_PKG_PATH}/R"
source_files <- list.files(fpath, full.names=TRUE, pattern="[.]R$")
for(sf in source_files) source(sf)

library(jsonlite)

panel <- read.csv("{data_path}", check.names=FALSE)

result <- DynBalancing_ATE(
    panel,
    covariates_names = {cov_str},
    Time_name = "Time",
    unit_name = "Unit",
    outcome_name = "Y",
    treatment_name = "D",
    ds1 = c(1, 1),
    ds2 = c(0, 0),
    params = list(
        method = "lasso_plain",
        open_source = TRUE,
        fast_adaptive = FALSE,
        alpha = 0.05,
        ub = 10,
        adaptive_balancing = FALSE
    )
)

out <- list(
    ATE = as.numeric(result$summaries$ATE),
    Mu1 = as.numeric(result$summaries$Mu1),
    Mu2 = as.numeric(result$summaries$Mu2)
)

write_json(out, "{result_path}", digits = 16)
"""
    try:
        r_result = _run_r_script(r_script, result_path, timeout=180)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        pytest.skip("R estimation failed")

    df = pl.read_csv(data_path)
    py_result = dyn_balancing(
        data=df,
        yname="Y",
        tname="Time",
        idname="Unit",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        xformla="~ " + " + ".join(covariates),
        ub=10.0,
        adaptive_balancing=False,
    )

    np.testing.assert_allclose(py_result.att, r_result["ATE"], atol=0.1)
    np.testing.assert_allclose(py_result.mu1, r_result["Mu1"], atol=0.1)


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_pooled_with_time_fe():
    covariates = ["V1", "V2", "V3", "V4", "V5"]
    data_path = _prepare_panel_csv(covariates, extra_cols=["region"])

    r_result = r_dyn_balancing_ate(
        data_path,
        covariates,
        ds1=[1, 1],
        ds2=[0, 0],
        fixed_effects=["region", "Time"],
        pooled=True,
    )
    if r_result is None:
        pytest.skip("R estimation failed")

    df = pl.read_csv(data_path)
    py_result = dyn_balancing(
        data=df,
        yname="Y",
        tname="Time",
        idname="Unit",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        xformla="~ " + " + ".join(covariates),
        fixed_effects=["region", "Time"],
        pooled=True,
        ub=10.0,
    )

    np.testing.assert_allclose(py_result.att, r_result["ATE"], atol=0.1)
    np.testing.assert_allclose(py_result.mu1, r_result["Mu1"], atol=0.1)
    np.testing.assert_allclose(py_result.mu2, r_result["Mu2"], atol=0.1)


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_clustered_se():
    covariates = ["V1", "V2", "V3", "V4", "V5"]
    data_path = _prepare_panel_csv(covariates, extra_cols=["region"])

    r_result = r_dyn_balancing_ate(
        data_path,
        covariates,
        ds1=[1, 1],
        ds2=[0, 0],
        fixed_effects=["region"],
        cluster_se="region",
    )
    if r_result is None:
        pytest.skip("R estimation failed")

    df = pl.read_csv(data_path)
    py_result = dyn_balancing(
        data=df,
        yname="Y",
        tname="Time",
        idname="Unit",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        xformla="~ " + " + ".join(covariates),
        fixed_effects=["region"],
        clustervars=["region"],
        ub=10.0,
    )

    np.testing.assert_allclose(py_result.att, r_result["ATE"], atol=0.15)
    np.testing.assert_allclose(py_result.mu1, r_result["Mu1"], atol=0.15)
    np.testing.assert_allclose(py_result.mu2, r_result["Mu2"], atol=0.15)
    # Cluster-robust variance diverges more than non-clustered due to
    # sklearn LassoCV vs glmnet penalty.factor differences cascading through
    # DCB weights into the 7-cluster sandwich estimator.
    np.testing.assert_allclose(py_result.var_att, r_result["Var_ATE"], rtol=0.65)


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_all_158_covariates():
    covariates = [f"V{i}" for i in range(1, 159)]
    data_path = _prepare_panel_csv(covariates)

    r_result = r_dyn_balancing_ate(
        data_path,
        covariates,
        ds1=[1, 1],
        ds2=[0, 0],
    )
    if r_result is None:
        pytest.skip("R estimation failed")

    df = pl.read_csv(data_path)
    py_result = dyn_balancing(
        data=df,
        yname="Y",
        tname="Time",
        idname="Unit",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        xformla="~ " + " + ".join(covariates),
        ub=10.0,
    )

    np.testing.assert_allclose(py_result.att, r_result["ATE"], atol=0.15)
    np.testing.assert_allclose(py_result.mu1, r_result["Mu1"], atol=0.15)
    np.testing.assert_allclose(py_result.mu2, r_result["Mu2"], atol=0.15)


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_non_default_alpha():
    covariates = ["V1", "V2", "V3", "V4", "V5"]
    data_path = _prepare_panel_csv(covariates)

    r_result = r_dyn_balancing_ate(
        data_path,
        covariates,
        ds1=[1, 1],
        ds2=[0, 0],
        alpha=0.1,
    )
    if r_result is None:
        pytest.skip("R estimation failed")

    df = pl.read_csv(data_path)
    py_result = dyn_balancing(
        data=df,
        yname="Y",
        tname="Time",
        idname="Unit",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        xformla="~ " + " + ".join(covariates),
        alp=0.1,
        ub=10.0,
    )

    np.testing.assert_allclose(py_result.att, r_result["ATE"], atol=1e-3)
    np.testing.assert_allclose(py_result.mu1, r_result["Mu1"], atol=1e-3)
    np.testing.assert_allclose(py_result.mu2, r_result["Mu2"], atol=1e-3)
    expected_robust_q = np.sqrt(chi2.isf(0.1, 2 * 2))
    np.testing.assert_allclose(py_result.robust_quantile, expected_robust_q, rtol=1e-6)


def r_dyn_balancing_history(
    data_path,
    covariates,
    ds1,
    ds2,
    histories_length,
    fixed_effects=None,
    pooled=False,
    initial_period=None,
    impulse_response=False,
    ub=10,
    final_period=None,
):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    cov_str = 'c("' + '", "'.join(covariates) + '")'
    ds1_str = "c(" + ", ".join(str(d) for d in ds1) + ")"
    ds2_str = "c(" + ", ".join(str(d) for d in ds2) + ")"
    hl_str = "c(" + ", ".join(str(h) for h in histories_length) + ")"

    fe_str = "NA"
    if fixed_effects:
        fe_str = 'c("' + '", "'.join(fixed_effects) + '")'

    pooled_str = "TRUE" if pooled else "FALSE"
    ir_str = "TRUE" if impulse_response else "FALSE"

    params_parts = [
        'method = "lasso_plain"',
        "open_source = TRUE",
        "fast_adaptive = FALSE",
        "alpha = 0.05",
        f"ub = {ub}",
        "numcores = 1",
    ]
    if initial_period is not None:
        params_parts.append(f"initial_period = {initial_period}")
    if final_period is not None:
        params_parts.append(f"final_period = {final_period}")
    params_parts.append(f"impulse_response = {ir_str}")
    params_str = "list(" + ", ".join(params_parts) + ")"

    r_script = f"""
load("{R_PKG_PATH}/data/params_default.rda")
fpath <- "{R_PKG_PATH}/R"
source_files <- list.files(fpath, full.names=TRUE, pattern="[.]R$")
for(f in source_files) source(f)

library(jsonlite)
library(doParallel)
library(foreach)

panel <- read.csv("{data_path}", check.names=FALSE)

result <- DynBalancing_History(
    panel,
    covariates_names = {cov_str},
    Time_name = "Time",
    unit_name = "Unit",
    outcome_name = "Y",
    treatment_name = "D",
    ds1 = {ds1_str},
    ds2 = {ds2_str},
    histories_length = {hl_str},
    fixed_effects = {fe_str},
    pooled = {pooled_str},
    params = {params_str}
)

m <- result$all_results
out <- list(
    ATEs = as.numeric(m$ATE),
    Mu1s = as.numeric(m$mu1),
    Mu2s = as.numeric(m$mu2),
    Period_lengths = as.numeric(m$Period_length)
)

write_json(out, "{result_path}", digits = 16)
"""
    try:
        return _run_r_script(r_script, result_path, timeout=600)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return None


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_history_ates_match_r():
    covariates = ["V1", "V2", "V3", "V4", "V5"]
    data_path = _prepare_panel_csv(covariates, extra_cols=["region"])

    r_result = r_dyn_balancing_history(
        data_path,
        covariates,
        ds1=[1, 1],
        ds2=[0, 0],
        histories_length=[1, 2],
        fixed_effects=["region"],
    )
    if r_result is None:
        pytest.skip("R estimation failed")

    df = pl.read_csv(data_path)
    py_result = dyn_balancing(
        data=df,
        yname="Y",
        tname="Time",
        idname="Unit",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        xformla="~ " + " + ".join(covariates),
        fixed_effects=["region"],
        histories_length=[1, 2],
        ub=10.0,
    )

    r_ates = r_result["ATEs"]
    py_ates = py_result.summary["att"].to_list()
    for i in range(len(r_ates)):
        np.testing.assert_allclose(py_ates[i], r_ates[i], atol=0.15)


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_history_mu_values_match_r():
    covariates = ["V1", "V2", "V3", "V4", "V5"]
    data_path = _prepare_panel_csv(covariates, extra_cols=["region"])

    r_result = r_dyn_balancing_history(
        data_path,
        covariates,
        ds1=[1, 1],
        ds2=[0, 0],
        histories_length=[1, 2],
        fixed_effects=["region"],
    )
    if r_result is None:
        pytest.skip("R estimation failed")

    df = pl.read_csv(data_path)
    py_result = dyn_balancing(
        data=df,
        yname="Y",
        tname="Time",
        idname="Unit",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        xformla="~ " + " + ".join(covariates),
        fixed_effects=["region"],
        histories_length=[1, 2],
        ub=10.0,
    )

    py_mu1s = py_result.summary["mu1"].to_list()
    py_mu2s = py_result.summary["mu2"].to_list()
    for i in range(len(r_result["Mu1s"])):
        np.testing.assert_allclose(py_mu1s[i], r_result["Mu1s"][i], atol=0.15)
        np.testing.assert_allclose(py_mu2s[i], r_result["Mu2s"][i], atol=0.15)


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_impulse_response_ates_match_r():
    covariates = ["V1", "V2", "V3", "V4", "V5"]
    data_path = _prepare_panel_csv(covariates, extra_cols=["region"])

    r_result = r_dyn_balancing_history(
        data_path,
        covariates,
        ds1=[1, 1],
        ds2=[0, 0],
        histories_length=[2],
        fixed_effects=["region"],
        impulse_response=True,
        ub=50,
        final_period=4,
    )
    if r_result is None:
        pytest.skip("R estimation failed")

    df = pl.read_csv(data_path)
    py_result = dyn_balancing(
        data=df,
        yname="Y",
        tname="Time",
        idname="Unit",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        xformla="~ " + " + ".join(covariates),
        fixed_effects=["region"],
        histories_length=[2],
        impulse_response=True,
        ub=50.0,
        final_period=4,
    )

    r_ates = r_result["ATEs"]
    py_ates = py_result.summary["att"].to_list()
    for i in range(len(r_ates)):
        np.testing.assert_allclose(py_ates[i], r_ates[i], atol=0.15)
