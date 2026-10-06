"""Validation tests comparing Python dyn_balancing with R DynBalancing package.

DynBalancing is not on CRAN. Set DYNBALANCING_PATH to a checkout of
https://github.com/dviviano/DynBalancing to run these tests. Since the reference
runs with moderndid's contiguous cross-validation folds and first-period
balancing tolerance, both sides are deterministic and agree closely.
"""

import json
import os
import subprocess
import tempfile

import pytest
from scipy.stats import chi2, norm

R_PKG_PATH = os.environ.get("DYNBALANCING_PATH", "")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.path.isfile(os.path.join(R_PKG_PATH, "data", "params_default.rda")),
        reason="Set DYNBALANCING_PATH to a DynBalancing checkout to compare with the reference",
    ),
]

from tests.helpers import importorskip

pl = importorskip("polars")
np = importorskip("numpy")

from moderndid.core.converters import dynbalancingresult_to_polars
from moderndid.core.data import load_acemoglu
from moderndid.diddynamic import dyn_balancing

R_PREAMBLE = """
load("{path}/data/params_default.rda")
for (sf in list.files("{path}/R", full.names = TRUE, pattern = "[.]R$")) source(sf)
library(jsonlite)

contiguous_cv <- function(y, x, nfolds, penalty.factor = rep(1, ncol(x))) {{
    n <- nrow(x)
    k <- min(nfolds, n)
    sizes <- rep(n %/% k, k) + c(rep(1, n %% k), rep(0, k - n %% k))
    glmnet::cv.glmnet(y = y, x = x, foldid = rep(seq_len(k), times = sizes),
                      penalty.factor = penalty.factor, thresh = 1e-12)
}}
coef_src <- gsub("glmnet::cv.glmnet(", "contiguous_cv(", deparse(compute_coefficients), fixed = TRUE)
compute_coefficients <- eval(parse(text = coef_src))
gamma1_src <- gsub("sqrt(log(p)/(n))", "sqrt(log(p)/sqrt(n))", deparse(compute_gamma1_os), fixed = TRUE)
gamma1_src <- gsub("sqrt(log(p)/((n)))", "sqrt(log(p)/sqrt(n))", gamma1_src, fixed = TRUE)
compute_gamma1_os <- eval(parse(text = gamma1_src))
pool_rows <- create_pooled_matrix
create_pooled_matrix <- function(...) {{
    out <- pool_rows(...)
    out[order(as.character(out$new_name), method = "radix"), ]
}}
"""


def _run_r_script(r_script, result_path, timeout=600):
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


def _r_call_args(covariates, ds1, ds2, fixed_effects, params):
    cov_str = 'c("' + '", "'.join(covariates) + '")'
    fe_str = 'c("' + '", "'.join(fixed_effects) + '")' if fixed_effects else "NA"
    ds1_str = "c(" + ", ".join(str(d) for d in ds1) + ")"
    ds2_str = "c(" + ", ".join(str(d) for d in ds2) + ")"
    merged = {"method": '"lasso_plain"', "open_source": "TRUE", "alpha": 0.05, "ub": 2, "lb": 0.0005, **(params or {})}
    params_str = "list(" + ", ".join(f"{key} = {value}" for key, value in merged.items()) + ")"
    return f"""covariates_names = {cov_str},
    Time_name = "Time",
    unit_name = "Unit",
    outcome_name = "Y",
    treatment_name = "D",
    ds1 = {ds1_str},
    ds2 = {ds2_str},
    fixed_effects = {fe_str},
    params = {params_str}"""


def r_dyn_balancing_matched(data_path, covariates, ds1, ds2, fixed_effects=("region",), pooled=False, params=None):
    """Run the reference with moderndid's contiguous folds and first-period tolerance."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    pooled_str = "TRUE" if pooled else "FALSE"
    r_script = (
        R_PREAMBLE.format(path=R_PKG_PATH)
        + f"""
panel <- read.csv("{data_path}", check.names = FALSE)
result <- suppressWarnings(DynBalancing_ATE(
    panel,
    pooled = {pooled_str},
    {_r_call_args(covariates, ds1, ds2, fixed_effects, params)}
))
s <- result$summaries
imbalance <- function(x) list(
    log_imbalance = x$LogImbalance,
    period = as.numeric(as.character(x$Period)),
    covariate = x$Covariates
)
out <- list(
    ATE = s$ATE, Var_ATE = s$Var_ATE, Mu1 = s$Mu1, Mu2 = s$Mu2, Var_mu1 = s$Var_mu1, Var_mu2 = s$Var_mu2,
    Robust_Quantile_ATE = s$Robust_Quantile_ATE, Gaussian_Quantile_ATE = s[[4]],
    Robust_Quantile_mu = s$Robust_Quantile_mu,
    imbalance_ds1 = imbalance(result$imbalances_summaries$po_1),
    imbalance_ds2 = imbalance(result$imbalances_summaries$po2)
)
write_json(out, "{result_path}", digits = 16, auto_unbox = TRUE)
"""
    )
    return _run_r_script(r_script, result_path)


def r_dyn_balancing_history_matched(
    data_path, covariates, ds1, ds2, histories_length, fixed_effects=("region",), params=None
):
    """Run the reference history wrapper with moderndid's contiguous folds and first-period tolerance."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    lengths = "c(" + ", ".join(str(h) for h in histories_length) + ")"
    r_script = (
        R_PREAMBLE.format(path=R_PKG_PATH)
        + f"""
library(doParallel)
library(foreach)
panel <- read.csv("{data_path}", check.names = FALSE)
result <- suppressWarnings(DynBalancing_History(
    panel,
    histories_length = {lengths},
    pooled = FALSE,
    {_r_call_args(covariates, ds1, ds2, fixed_effects, {"numcores": 1, **(params or {})})}
))
m <- result$all_results
out <- list(ATE = m$ATE, Var_ATE = m$Var_ATE, Mu1 = m$mu1, Mu2 = m$mu2, Period_length = m$Period_length)
write_json(out, "{result_path}", digits = 16)
"""
    )
    return _run_r_script(r_script, result_path)


def _prepare_sorted_panel_csv(covariates, extra_cols=None, missing=None, complete=False):
    """Write the Acemoglu panel sorted by unit and time, with one column nulled at one period for ten units."""
    df = load_acemoglu().sort(["Unit", "Time"]).select(["Y", "D", "Unit", "Time", *covariates, *(extra_cols or [])])
    if complete:
        df = df.filter(pl.col("Y").is_not_null().all().over("Unit"))
    if missing is not None:
        col, time = missing
        nulled = pl.col("Unit").is_in([5, 11, 12, 25, 34, 81, 106, 109, 119, 135]) & (pl.col("Time") == time)
        df = df.with_columns(pl.when(nulled).then(None).otherwise(pl.col(col)).alias(col))
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
    df.write_csv(path)
    return path


def _prepare_synthetic_panel_csv():
    """Write a two-period panel with random treatments and two covariates."""
    rng = np.random.default_rng(42)
    n = 100
    n_periods = 2
    x1 = rng.standard_normal(n * n_periods)
    x2 = rng.standard_normal(n * n_periods)
    d = rng.integers(0, 2, size=n * n_periods).astype(float)
    df = pl.DataFrame(
        {
            "Unit": np.repeat(np.arange(1, n + 1), n_periods),
            "Time": np.tile(np.arange(1, n_periods + 1), n),
            "Y": 1.0 + x1 + 0.5 * x2 + 2.0 * d + rng.standard_normal(n * n_periods) * 0.3,
            "D": d,
            "V1": x1,
            "V2": x2,
        }
    )
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
    df.write_csv(path)
    return path


def _py_matched(data_path, covariates, ds1, ds2, fixed_effects=("region",), **kwargs):
    """Fit moderndid on a panel written by one of the CSV helpers."""
    return dyn_balancing(
        data=pl.read_csv(data_path),
        yname="Y",
        tname="Time",
        idname="Unit",
        treatment_name="D",
        ds1=ds1,
        ds2=ds2,
        xformla="~ " + " + ".join(covariates),
        fixed_effects=list(fixed_effects) if fixed_effects else None,
        **kwargs,
    )


def _assert_matches(py_result, r_result, atol=1e-4):
    np.testing.assert_allclose(py_result.att, r_result["ATE"], atol=atol)
    np.testing.assert_allclose(py_result.mu1, r_result["Mu1"], atol=atol)
    np.testing.assert_allclose(py_result.mu2, r_result["Mu2"], atol=atol)
    np.testing.assert_allclose(py_result.var_att, r_result["Var_ATE"], rtol=1e-3)


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_no_fixed_effects_matches_r():
    covariates = ["V1", "V2", "V3", "V4", "V5"]
    data_path = _prepare_sorted_panel_csv(covariates)

    r_result = r_dyn_balancing_matched(data_path, covariates, [1, 1], [0, 0], fixed_effects=None, params={"ub": 10})
    py_result = _py_matched(data_path, covariates, [1, 1], [0, 0], fixed_effects=None, ub=10.0)
    _assert_matches(py_result, r_result)


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_synthetic_panel_matches_r():
    data_path = _prepare_synthetic_panel_csv()

    r_result = r_dyn_balancing_matched(data_path, ["V1", "V2"], [1, 1], [0, 0], fixed_effects=None, params={"ub": 10})
    py_result = _py_matched(data_path, ["V1", "V2"], [1, 1], [0, 0], fixed_effects=None, ub=10.0)
    _assert_matches(py_result, r_result)


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_potential_outcome_variances_match_r():
    covariates = ["V1", "V2", "V3", "V4", "V5"]
    data_path = _prepare_sorted_panel_csv(covariates, extra_cols=["region"])

    r_result = r_dyn_balancing_matched(data_path, covariates, [1, 1], [0, 0])
    py_result = _py_matched(data_path, covariates, [1, 1], [0, 0])
    np.testing.assert_allclose(py_result.var_mu1, r_result["Var_mu1"], rtol=1e-3)
    np.testing.assert_allclose(py_result.var_mu2, r_result["Var_mu2"], rtol=1e-3)


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_ridge_path_matches_r():
    covariates = ["V1", "V2", "V3", "V4", "V5"]
    data_path = _prepare_sorted_panel_csv(covariates, extra_cols=["region"])

    r_result = r_dyn_balancing_matched(data_path, covariates, [1, 1], [0, 0], params={"regularization": "FALSE"})
    py_result = _py_matched(data_path, covariates, [1, 1], [0, 0], regularization=False)
    _assert_matches(py_result, r_result)


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_transition_into_treatment_matches_r():
    covariates = ["V1", "V2", "V3"]
    data_path = _prepare_sorted_panel_csv(covariates)

    r_result = r_dyn_balancing_matched(data_path, covariates, [0, 1], [0, 0], fixed_effects=None, params={"ub": 10})
    py_result = _py_matched(data_path, covariates, [0, 1], [0, 0], fixed_effects=None, ub=10.0)
    _assert_matches(py_result, r_result)


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_more_covariates_match_r():
    covariates = [f"V{i}" for i in range(1, 11)]
    data_path = _prepare_sorted_panel_csv(covariates)

    r_result = r_dyn_balancing_matched(data_path, covariates, [1, 1], [0, 0], fixed_effects=None, params={"ub": 10})
    py_result = _py_matched(data_path, covariates, [1, 1], [0, 0], fixed_effects=None, ub=10.0)
    _assert_matches(py_result, r_result)


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_non_adaptive_balancing_matches_r():
    covariates = ["V1", "V2", "V3", "V4", "V5"]
    data_path = _prepare_sorted_panel_csv(covariates, extra_cols=["region"])

    r_result = r_dyn_balancing_matched(data_path, covariates, [1, 1], [0, 0], params={"adaptive_balancing": "FALSE"})
    py_result = _py_matched(data_path, covariates, [1, 1], [0, 0], adaptive_balancing=False)
    _assert_matches(py_result, r_result)


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_pooled_with_time_fe():
    covariates = ["V1", "V2", "V3", "V4", "V5"]
    data_path = _prepare_sorted_panel_csv(covariates, extra_cols=["region"])

    r_result = r_dyn_balancing_matched(
        data_path, covariates, ds1=[1, 1], ds2=[0, 0], fixed_effects=("region", "Time"), pooled=True
    )
    py_result = _py_matched(
        data_path, covariates, [1, 1], [0, 0], fixed_effects=("region", "Time"), pooled=True, initial_period=3
    )
    _assert_matches(py_result, r_result, atol=2e-4)


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_pooled_all_windows_match_r():
    covariates = ["V1", "V2", "V3", "V4", "V5"]
    data_path = _prepare_sorted_panel_csv(covariates, extra_cols=["region"])

    r_result = r_dyn_balancing_matched(
        data_path,
        covariates,
        ds1=[1, 1],
        ds2=[0, 0],
        fixed_effects=("region", "Time"),
        pooled=True,
        params={"initial_period": 1},
    )
    py_result = _py_matched(data_path, covariates, [1, 1], [0, 0], fixed_effects=("region", "Time"), pooled=True)
    _assert_matches(py_result, r_result)
    assert py_result.estimation_params["n_units"] == 141


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_clustered_se():
    covariates = ["V1", "V2", "V3", "V4", "V5"]
    data_path = _prepare_sorted_panel_csv(covariates, extra_cols=["region"])

    r_result = r_dyn_balancing_matched(
        data_path, covariates, ds1=[1, 1], ds2=[0, 0], params={"ub": 10, "cluster_SE": '"region"'}
    )
    py_result = _py_matched(data_path, covariates, [1, 1], [0, 0], clustervars=["region"], ub=10.0)
    _assert_matches(py_result, r_result)


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_lasso_plain_matches_r_with_matched_folds():
    covariates = ["V1", "V2", "V3", "V4", "V5"]
    data_path = _prepare_sorted_panel_csv(covariates, extra_cols=["region"])

    r_result = r_dyn_balancing_matched(data_path, covariates, ds1=[1, 1], ds2=[0, 0])
    py_result = _py_matched(data_path, covariates, [1, 1], [0, 0])
    _assert_matches(py_result, r_result)


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_dotted_lagged_outcome_names_match_r():
    covariates = [f"lag{i}.Value1" for i in range(1, 5)]
    data_path = _prepare_sorted_panel_csv(covariates, extra_cols=["region"])

    r_result = r_dyn_balancing_matched(data_path, covariates, ds1=[1, 1], ds2=[0, 0])
    py_result = _py_matched(data_path, covariates, [1, 1], [0, 0])
    _assert_matches(py_result, r_result)


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
@pytest.mark.parametrize("lags", [0, 1])
def test_lags_match_r(lags):
    covariates = ["V1", "V2", "V3", "V4", "V5"]
    data_path = _prepare_sorted_panel_csv(covariates, extra_cols=["region"])

    r_result = r_dyn_balancing_matched(data_path, covariates, ds1=[1, 1, 1], ds2=[0, 0, 0], params={"lags": lags})
    py_result = _py_matched(data_path, covariates, [1, 1, 1], [0, 0, 0], lags=lags)
    _assert_matches(py_result, r_result)


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_long_history_matches_r():
    covariates = ["V1", "V2", "V3", "V4", "V5"]
    data_path = _prepare_sorted_panel_csv(covariates, extra_cols=["region"])

    r_result = r_dyn_balancing_matched(data_path, covariates, ds1=[1] * 5, ds2=[0] * 5)
    py_result = _py_matched(data_path, covariates, [1] * 5, [0] * 5)
    _assert_matches(py_result, r_result)


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
@pytest.mark.parametrize("missing", [("Y", 0), ("Y", 4), ("D", 0), ("V1", 4)])
def test_missing_values_match_r(missing):
    covariates = ["V1", "V2", "V3", "V4", "V5"]
    data_path = _prepare_sorted_panel_csv(covariates, extra_cols=["region"], missing=missing)

    r_result = r_dyn_balancing_matched(data_path, covariates, ds1=[1, 1], ds2=[0, 0])
    py_result = _py_matched(data_path, covariates, [1, 1], [0, 0])
    _assert_matches(py_result, r_result)
    assert py_result.estimation_params["n_units"] == 137


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_earlier_final_period_matches_r():
    covariates = ["V1", "V2", "V3", "V4", "V5"]
    data_path = _prepare_sorted_panel_csv(covariates, extra_cols=["region"])

    r_result = r_dyn_balancing_matched(data_path, covariates, ds1=[1, 1], ds2=[0, 0], params={"final_period": 3})
    py_result = _py_matched(data_path, covariates, [1, 1], [0, 0], final_period=3)
    _assert_matches(py_result, r_result)
    assert py_result.estimation_params["n_units"] == 141


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_all_158_covariates_match_r():
    covariates = [f"V{i}" for i in range(1, 159)]
    data_path = _prepare_sorted_panel_csv(covariates)

    r_result = r_dyn_balancing_matched(data_path, covariates, [1, 1], [0, 0], fixed_effects=None, params={"ub": 10})
    py_result = _py_matched(data_path, covariates, [1, 1], [0, 0], fixed_effects=None, ub=10.0)
    _assert_matches(py_result, r_result)


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
@pytest.mark.parametrize("robust", [None, False, True])
def test_quantiles_match_r(robust):
    covariates = ["V1", "V2", "V3", "V4", "V5"]
    data_path = _prepare_sorted_panel_csv(covariates, extra_cols=["region"])

    r_params = {"alpha": 0.1}
    py_kwargs = {"alp": 0.1}
    if robust is not None:
        r_params["robust_quantile"] = "TRUE" if robust else "FALSE"
        py_kwargs["robust_quantile"] = robust
    r_result = r_dyn_balancing_matched(data_path, covariates, [1, 1], [0, 0], params=r_params)
    py_result = _py_matched(data_path, covariates, [1, 1], [0, 0], **py_kwargs)

    mu_row = dynbalancingresult_to_polars(py_result).filter(pl.col("parameter") == "mu(ds1)").row(0, named=True)
    mu_quantile = (mu_row["ci_upper_robust"] - mu_row["estimate"]) / mu_row["se"]
    expected_ate = np.sqrt(chi2.ppf(0.9, 4)) if robust else norm.ppf(0.95)
    np.testing.assert_allclose(py_result.robust_quantile, r_result["Robust_Quantile_ATE"], rtol=1e-10)
    np.testing.assert_allclose(py_result.gaussian_quantile, r_result["Gaussian_Quantile_ATE"], rtol=1e-10)
    np.testing.assert_allclose(mu_quantile, r_result["Robust_Quantile_mu"], rtol=1e-10)
    np.testing.assert_allclose(py_result.robust_quantile, expected_ate, rtol=1e-10)
    _assert_matches(py_result, r_result)


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_imbalances_match_r():
    covariates = ["V1", "V2", "V3", "V4", "V5"]
    data_path = _prepare_sorted_panel_csv(covariates, extra_cols=["region"], complete=True)

    r_result = r_dyn_balancing_matched(data_path, covariates, [1, 1], [0, 0])
    py_result = _py_matched(data_path, covariates, [1, 1], [0, 0])
    for history in ("ds1", "ds2"):
        r_table = pl.DataFrame(r_result[f"imbalance_{history}"]).sort("period", "covariate")
        py_table = (
            py_result.imbalances[history]
            .filter(pl.col("covariate").is_in(covariates))
            .with_columns((pl.col("imbalance").abs() + 1).log().alias("log_imbalance"))
            .sort("period", "covariate")
        )
        assert py_table["covariate"].to_list() == r_table["covariate"].to_list()
        np.testing.assert_allclose(py_table["log_imbalance"].to_numpy(), r_table["log_imbalance"].to_numpy(), atol=1e-5)


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_history_matches_r():
    covariates = ["V1", "V2", "V3", "V4", "V5"]
    data_path = _prepare_sorted_panel_csv(covariates, extra_cols=["region"])

    r_result = r_dyn_balancing_history_matched(data_path, covariates, [1, 1], [0, 0], histories_length=[1, 2])
    py_result = _py_matched(data_path, covariates, [1, 1], [0, 0], histories_length=[1, 2])
    np.testing.assert_allclose(py_result.summary["att"].to_numpy(), r_result["ATE"], atol=1e-4)
    np.testing.assert_allclose(py_result.summary["mu1"].to_numpy(), r_result["Mu1"], atol=1e-4)
    np.testing.assert_allclose(py_result.summary["mu2"].to_numpy(), r_result["Mu2"], atol=1e-4)
    np.testing.assert_allclose(py_result.summary["var_att"].to_numpy(), r_result["Var_ATE"], rtol=1e-3)


@pytest.mark.skipif(not R_AVAILABLE, reason="R or required R packages not available")
def test_impulse_response_matches_r():
    covariates = ["V1", "V2", "V3", "V4", "V5"]
    data_path = _prepare_sorted_panel_csv(covariates, extra_cols=["region"])

    r_result = r_dyn_balancing_history_matched(
        data_path,
        covariates,
        [1, 1],
        [0, 0],
        histories_length=[2],
        params={"impulse_response": "TRUE", "ub": 50, "final_period": 4},
    )
    py_result = _py_matched(
        data_path, covariates, [1, 1], [0, 0], histories_length=[2], impulse_response=True, ub=50.0, final_period=4
    )
    np.testing.assert_allclose(py_result.summary["att"].to_numpy(), r_result["ATE"], atol=1e-4)
    np.testing.assert_allclose(py_result.summary["var_att"].to_numpy(), r_result["Var_ATE"], rtol=1e-3)
