"""Validation tests comparing Python NPIV implementation with R npiv package."""

import json
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.slow

from tests.helpers import importorskip

np = importorskip("numpy")

from moderndid.npiv import gsl_bs, npiv, prodspline
from moderndid.npiv.selection import npiv_choose_j

R_NAMES = {
    "j_x_degree": "J.x.degree",
    "j_x_segments": "J.x.segments",
    "k_w_degree": "K.w.degree",
    "k_w_segments": "K.w.segments",
    "k_w_smooth": "K.w.smooth",
    "knots": "knots",
    "basis": "basis",
    "alpha": "alpha",
    "deriv_index": "deriv.index",
    "deriv_order": "deriv.order",
    "ucb_h": "ucb.h",
    "ucb_deriv": "ucb.deriv",
    "x_min": "X.min",
    "x_max": "X.max",
    "w_min": "W.min",
    "w_max": "W.max",
}


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


def check_r_available():
    try:
        result = subprocess.run(
            ["R", "--vanilla", "--quiet"],
            input='library(npiv); library(jsonlite); cat("OK")',
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return "OK" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


R_AVAILABLE = check_r_available()


def _r_value(value):
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, str):
        return f'"{value}"'
    return repr(value)


def _r_matrix_code(name, path, n_cols):
    if n_cols == 1:
        return f'{name} <- as.numeric(as.matrix(read.csv("{path}", header = FALSE))[, 1])'
    return f'{name} <- unname(as.matrix(read.csv("{path}", header = FALSE)))'


def _r_npiv(y, x, w, x_eval=None, biters=99, selection=False, timeout=300, **kwargs):
    """Run R npiv with set.seed(42) on the given arrays and return its results and bootstrap draws."""
    x = np.asarray(x, dtype=float).reshape(len(y), -1)
    w = np.asarray(w, dtype=float).reshape(len(y), -1)
    is_regression = np.array_equal(x, w)
    r_args = ", ".join(f"{R_NAMES[k]} = {_r_value(v)}" for k, v in kwargs.items())
    sel_args = ", ".join(
        f"{R_NAMES[k]} = {_r_value(v)}"
        for k, v in kwargs.items()
        if k not in ("alpha", "deriv_index", "deriv_order", "ucb_h", "ucb_deriv")
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        np.savetxt(tmp / "y.csv", np.asarray(y, dtype=float), fmt="%.17g")
        np.savetxt(tmp / "x.csv", x, delimiter=",", fmt="%.17g")
        np.savetxt(tmp / "w.csv", w, delimiter=",", fmt="%.17g")
        eval_code = "X.eval <- NULL"
        if x_eval is not None:
            x_eval = np.asarray(x_eval, dtype=float).reshape(-1, x.shape[1])
            np.savetxt(tmp / "eval.csv", x_eval, delimiter=",", fmt="%.17g")
            eval_code = _r_matrix_code("X.eval", tmp / "eval.csv", x.shape[1])
        w_code = "W <- X" if is_regression else _r_matrix_code("W", tmp / "w.csv", w.shape[1])
        selection_code = ""
        if selection:
            selection_code = f"""
set.seed(42)
cj <- npiv:::npiv_choose_J(Y, X, W, {sel_args}{", " if sel_args else ""}boot.num = {biters}, progress = FALSE)
output$selection <- cj[c("J.hat.max", "J.hat.n", "J.hat", "J.tilde", "J.x.seg", "K.w.seg", "theta.star")]
"""
        r_script = f"""
library(npiv)
library(jsonlite)

Y <- as.numeric(read.csv("{tmp / "y.csv"}", header = FALSE)[, 1])
{_r_matrix_code("X", tmp / "x.csv", x.shape[1])}
{w_code}
{eval_code}

set.seed(42)
writeBin(rnorm(length(Y) * {biters}), "{tmp / "draws.bin"}")

set.seed(42)
result <- npiv(Y = Y, X = X, W = W, X.eval = X.eval, {r_args}{", " if r_args else ""}boot.num = {biters},
               progress = FALSE)

output <- list(
    h = as.numeric(result$h),
    deriv = as.numeric(result$deriv),
    se = as.numeric(result$asy.se),
    se_deriv = as.numeric(result$deriv.asy.se),
    beta = as.numeric(result$beta),
    J_x_segments = result$J.x.segments,
    K_w_segments = result$K.w.segments,
    K_w_degree = result$K.w.degree
)
if (!is.null(result$cv)) output$cv <- result$cv
if (!is.null(result$cv.deriv)) output$cv_deriv <- result$cv.deriv
if (!is.null(result$h.lower)) output$h_lower <- as.numeric(result$h.lower)
if (!is.null(result$h.upper)) output$h_upper <- as.numeric(result$h.upper)
if (!is.null(result$h.lower.deriv)) output$h_lower_deriv <- as.numeric(result$h.lower.deriv)
if (!is.null(result$h.upper.deriv)) output$h_upper_deriv <- as.numeric(result$h.upper.deriv)
{selection_code}
write_json(output, "{tmp / "result.json"}", auto_unbox = TRUE, digits = NA)
"""
        output = _run_r_script(r_script, tmp / "result.json", timeout=timeout)
        output["draws"] = np.fromfile(tmp / "draws.bin", dtype="<f8").reshape(biters, len(y))
    return output


def _shared_draws(draws):
    """Return a stand-in for np.random.default_rng that hands out the reference's bootstrap draws."""

    def make_rng(seed=None):
        rows = iter(draws)

        def normal(loc=0.0, scale=1.0, size=None):
            if isinstance(size, tuple):
                return draws[: size[0], : size[1]].copy()
            return next(rows)[:size].copy()

        return SimpleNamespace(normal=normal)

    return make_rng


def _python_npiv(y, x, w, r_result=None, monkeypatch=None, x_eval=None, biters=99, **kwargs):
    if r_result is not None:
        monkeypatch.setattr(np.random, "default_rng", _shared_draws(r_result["draws"]))
    return npiv(y=y, x=x, w=w, x_eval=x_eval, biters=biters, seed=42, **kwargs)


def _engel_args(engel_data, w_key="logwages"):
    return engel_data["food"], engel_data["logexp"], engel_data[w_key]


ENGEL_GRID = np.linspace(4.5, 6.5, 100)


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_iv_h_matches(engel_data):
    y, x, w = _engel_args(engel_data)
    py = _python_npiv(y, x, w, x_eval=ENGEL_GRID, j_x_segments=1, k_w_segments=4)
    r = _r_npiv(y, x, w, x_eval=ENGEL_GRID, j_x_segments=1, k_w_segments=4)

    assert len(py.h) == 100
    np.testing.assert_allclose(py.h, r["h"], rtol=1e-10, atol=1e-12, err_msg="IV function estimates (h) don't match R")
    np.testing.assert_allclose(py.asy_se, r["se"], rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_iv_deriv_matches(engel_data):
    y, x, w = _engel_args(engel_data)
    py = _python_npiv(y, x, w, x_eval=ENGEL_GRID, j_x_segments=1, k_w_segments=4)
    r = _r_npiv(y, x, w, x_eval=ENGEL_GRID, j_x_segments=1, k_w_segments=4)

    np.testing.assert_allclose(py.deriv, r["deriv"], rtol=1e-10, atol=1e-12, err_msg="IV derivatives don't match R")
    np.testing.assert_allclose(py.deriv_asy_se, r["se_deriv"], rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_iv_confidence_bands_match(engel_data, monkeypatch):
    y, x, w = _engel_args(engel_data)
    r = _r_npiv(y, x, w, x_eval=ENGEL_GRID, j_x_segments=5, k_w_segments=20, biters=199)
    py = _python_npiv(y, x, w, r, monkeypatch, x_eval=ENGEL_GRID, j_x_segments=5, k_w_segments=20, biters=199)

    assert py.cv == pytest.approx(r["cv"], rel=1e-10)
    np.testing.assert_allclose(py.h_lower, r["h_lower"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(py.h_upper, r["h_upper"], rtol=1e-10, atol=1e-12)
    assert np.all(py.h_lower <= py.h)
    assert np.all(py.h <= py.h_upper)


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_iv_deriv_confidence_bands_match(engel_data, monkeypatch):
    y, x, w = _engel_args(engel_data)
    r = _r_npiv(y, x, w, x_eval=ENGEL_GRID, j_x_segments=1, k_w_segments=4, biters=199)
    py = _python_npiv(y, x, w, r, monkeypatch, x_eval=ENGEL_GRID, j_x_segments=1, k_w_segments=4, biters=199)

    assert py.cv_deriv == pytest.approx(r["cv_deriv"], rel=1e-10)
    np.testing.assert_allclose(py.h_lower_deriv, r["h_lower_deriv"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(py.h_upper_deriv, r["h_upper_deriv"], rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_default_k_matches_r_explicit_k(engel_data, monkeypatch):
    y, x, w = _engel_args(engel_data)
    r = _r_npiv(y, x, w, x_eval=ENGEL_GRID, j_x_segments=5, k_w_segments=20, biters=99)
    py = _python_npiv(y, x, w, r, monkeypatch, x_eval=ENGEL_GRID, j_x_segments=5)

    assert py.k_w_segments == 20
    np.testing.assert_allclose(py.h, r["h"], rtol=1e-10, atol=1e-12)
    assert py.cv == pytest.approx(r["cv"], rel=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_k_below_j_raises_like_r(engel_data):
    y, x, w = _engel_args(engel_data)

    with pytest.raises(RuntimeError, match="K.w.degree\\+K.w.segments must be >= J.x.degree\\+J.x.segments"):
        _r_npiv(y, x, w, j_x_segments=5, k_w_segments=3, biters=5)
    with pytest.raises(ValueError, match="not identified"):
        _python_npiv(y, x, w, j_x_segments=5, k_w_segments=3, biters=5)


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_regression_h_matches(engel_data):
    y, x, _ = _engel_args(engel_data)
    py = _python_npiv(y, x, x, x_eval=ENGEL_GRID, j_x_segments=64, k_w_degree=3, k_w_segments=64)
    r = _r_npiv(y, x, x, x_eval=ENGEL_GRID, j_x_segments=64, k_w_degree=3, k_w_segments=64)

    np.testing.assert_allclose(py.h, r["h"], rtol=1e-8, atol=1e-10, err_msg="Regression h don't match R")
    np.testing.assert_allclose(py.deriv, r["deriv"], rtol=1e-8, atol=1e-10, err_msg="Regression deriv don't match R")


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_regression_confidence_bands_match(engel_data, monkeypatch):
    y, x, _ = _engel_args(engel_data)
    r = _r_npiv(y, x, x, x_eval=ENGEL_GRID, j_x_segments=4, biters=199)
    py = _python_npiv(y, x, x, r, monkeypatch, x_eval=ENGEL_GRID, j_x_segments=4, biters=199)

    assert py.k_w_segments == r["K_w_segments"] == 4
    assert py.cv == pytest.approx(r["cv"], rel=1e-9)
    assert py.cv_deriv == pytest.approx(r["cv_deriv"], rel=1e-9)
    np.testing.assert_allclose(py.h_lower, r["h_lower"], rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(py.h_upper_deriv, r["h_upper_deriv"], rtol=1e-9, atol=1e-11)


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
@pytest.mark.parametrize("x_eval", [ENGEL_GRID, None])
def test_npiv_data_driven_iv_matches(engel_data, monkeypatch, x_eval):
    y, x, w = _engel_args(engel_data)
    r = _r_npiv(y, x, w, x_eval=x_eval, biters=199, timeout=600)
    py = _python_npiv(y, x, w, r, monkeypatch, x_eval=x_eval, biters=199)

    assert py.j_x_segments == r["J_x_segments"]
    assert py.k_w_segments == r["K_w_segments"]
    np.testing.assert_allclose(py.h, r["h"], rtol=1e-10, atol=1e-12, err_msg="Data-driven IV h don't match R")
    np.testing.assert_allclose(
        py.deriv, r["deriv"], rtol=1e-10, atol=1e-12, err_msg="Data-driven IV deriv don't match R"
    )
    assert py.cv == pytest.approx(r["cv"], rel=1e-9)
    assert py.cv_deriv == pytest.approx(r["cv_deriv"], rel=1e-9)
    np.testing.assert_allclose(py.h_lower, r["h_lower"], rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(py.h_upper, r["h_upper"], rtol=1e-9, atol=1e-11)


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
@pytest.mark.parametrize("knots", ["uniform", "quantiles"])
def test_npiv_data_driven_selection_matches(engel_data, monkeypatch, knots):
    y, x, w = _engel_args(engel_data)
    r = _r_npiv(y, x, w, x_eval=ENGEL_GRID, knots=knots, biters=199, selection=True, timeout=600)
    monkeypatch.setattr(np.random, "default_rng", _shared_draws(r["draws"]))
    sel = npiv_choose_j(y=y, x=x, w=w, knots=knots, biters=199, seed=42)

    assert sel["j_hat_max"] == r["selection"]["J.hat.max"]
    assert sel["j_tilde"] == r["selection"]["J.tilde"]
    assert sel["j_x_seg"] == r["selection"]["J.x.seg"]
    assert sel["theta_star"] == pytest.approx(r["selection"]["theta.star"], rel=1e-9)


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_data_driven_regression_matches(engel_data, monkeypatch):
    y, x, _ = _engel_args(engel_data)
    r = _r_npiv(y, x, x, x_eval=ENGEL_GRID, biters=199, timeout=600)
    py = _python_npiv(y, x, x, r, monkeypatch, x_eval=ENGEL_GRID, biters=199)

    assert py.j_x_segments == r["J_x_segments"]
    assert py.k_w_segments == r["K_w_segments"]
    assert py.k_w_degree == r["K_w_degree"]
    np.testing.assert_allclose(py.h, r["h"], rtol=1e-8, atol=1e-10, err_msg="Data-driven regression h don't match R")
    assert py.cv == pytest.approx(r["cv"], rel=1e-8)
    assert py.cv_deriv == pytest.approx(r["cv_deriv"], rel=1e-8)


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_data_driven_derivative_order_two_matches(engel_data, monkeypatch):
    y, x, w = _engel_args(engel_data)
    spec = {"j_x_degree": 4, "k_w_degree": 5, "k_w_smooth": 1, "deriv_order": 2, "alpha": 0.1}
    r = _r_npiv(y, x, w, x_eval=ENGEL_GRID, biters=199, timeout=600, **spec)
    py = _python_npiv(y, x, w, r, monkeypatch, x_eval=ENGEL_GRID, biters=199, **spec)

    assert py.j_x_segments == r["J_x_segments"]
    np.testing.assert_allclose(py.deriv, r["deriv"], rtol=1e-9, atol=1e-11)
    assert py.cv == pytest.approx(r["cv"], rel=1e-9)
    assert py.cv_deriv == pytest.approx(r["cv_deriv"], rel=1e-9)


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
@pytest.mark.parametrize("basis", ["additive", "glp"])
def test_npiv_data_driven_one_regressor_bases_match(engel_data, monkeypatch, basis):
    y, x, w = _engel_args(engel_data)
    r = _r_npiv(y, x, w, x_eval=ENGEL_GRID, basis=basis, biters=199, selection=True, timeout=600)
    py = _python_npiv(y, x, w, r, monkeypatch, x_eval=ENGEL_GRID, basis=basis, biters=199)

    assert py.j_x_segments == r["J_x_segments"]
    assert py.args["theta_star"] == pytest.approx(r["selection"]["theta.star"], rel=1e-9)
    assert py.args["j_tilde"] == r["selection"]["J.tilde"]
    np.testing.assert_allclose(py.h, r["h"], rtol=1e-9, atol=1e-11)
    assert py.cv == pytest.approx(r["cv"], rel=1e-9)


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_zero_support_bounds_match(monkeypatch):
    rng = np.random.default_rng(5)
    n = 600
    w = rng.uniform(size=n)
    v = rng.uniform(size=n)
    x = 0.6 * w + 0.4 * v
    y = np.sin(2 * np.pi * x) + 0.5 * (v - 0.5) + 0.2 * rng.normal(size=n)
    bounds = {"x_min": 0.0, "x_max": 1.0, "w_min": 0.0, "w_max": 1.0}
    x_eval = np.linspace(0.05, 0.95, 50)
    r = _r_npiv(y, x, w, x_eval=x_eval, biters=99, selection=True, **bounds)
    py = _python_npiv(y, x, w, r, monkeypatch, x_eval=x_eval, **bounds)

    assert py.j_x_segments == r["J_x_segments"]
    assert py.args["theta_star"] == pytest.approx(r["selection"]["theta.star"], rel=1e-9)
    assert py.cv == pytest.approx(r["cv"], rel=1e-9)
    np.testing.assert_allclose(py.h, r["h"], rtol=1e-9, atol=1e-11)


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_extrapolation_outside_support_matches(engel_data):
    y, x, w = _engel_args(engel_data)
    x_eval = np.linspace(3, 8, 101)
    with pytest.warns(UserWarning, match="beyond boundary knots"):
        py = _python_npiv(y, x, w, x_eval=x_eval, j_x_segments=5, k_w_segments=20, ucb_h=False, ucb_deriv=False)
    r = _r_npiv(y, x, w, x_eval=x_eval, j_x_segments=5, k_w_segments=20, ucb_h=False, ucb_deriv=False)

    np.testing.assert_allclose(py.h, r["h"], rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(py.deriv, r["deriv"], rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(py.asy_se, r["se"], rtol=1e-9, atol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
@pytest.mark.parametrize(
    "deriv,xs",
    [
        (0, np.array([-0.3, -0.1, -1e-9, 0.0, 0.25, 0.5, 1.0, 1 + 1e-9, 1.2, 1.5])),
        (1, np.array([-0.3, -0.1, -1e-9, 0.0, 0.25, 0.5, 1.0, 1 + 1e-9, 1.2, 1.5])),
        (2, np.array([-0.3, -0.1, -1e-9, 0.0, 0.25, 0.5, 1.0, 1 + 1e-9, 1.2, 1.5])),
        (3, np.array([-0.3, 0.0, 0.25, 0.5, 1.0, 1.5])),
    ],
)
def test_gsl_bs_extrapolation_matches_r(deriv, xs):
    with tempfile.TemporaryDirectory() as tmpdir:
        result_path = Path(tmpdir) / "result.json"
        r_script = f"""
library(npiv)
library(jsonlite)
xs <- c({", ".join(repr(float(v)) for v in xs)})
B <- suppressWarnings(
    npiv:::gsl.bs(xs, degree = 3, nbreak = 4, deriv = {deriv}, x.min = 0, x.max = 1, intercept = TRUE)
)
write_json(unname(matrix(unclass(B), nrow = length(xs))), "{result_path}", digits = NA)
"""
        r_basis = np.array(_run_r_script(r_script, result_path))

    with pytest.warns(UserWarning, match="beyond boundary knots"):
        py_basis = gsl_bs(xs, degree=3, nbreak=4, deriv=deriv, x_min=0.0, x_max=1.0, intercept=True).basis

    np.testing.assert_allclose(py_basis, r_basis, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
@pytest.mark.parametrize(
    "basis,segments,deriv_index",
    [("tensor", (2, 4), 2), ("glp", (2, 4), 2), ("glp", (3, 6), 1)],
)
def test_npiv_two_regressors_match(two_regressor_iv_data, monkeypatch, basis, segments, deriv_index):
    y, x, w, x_eval = two_regressor_iv_data
    spec = {"basis": basis, "j_x_segments": segments[0], "k_w_segments": segments[1], "deriv_index": deriv_index}
    r = _r_npiv(y, x, w, x_eval=x_eval, **spec)
    py = _python_npiv(y, x, w, r, monkeypatch, x_eval=x_eval, **spec)

    np.testing.assert_allclose(py.h, r["h"], rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(py.deriv, r["deriv"], rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(py.asy_se, r["se"], rtol=1e-9, atol=1e-11)
    assert py.cv == pytest.approx(r["cv"], rel=1e-9)
    assert py.cv_deriv == pytest.approx(r["cv_deriv"], rel=1e-9)


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_additive_derivative_drops_other_regressors(two_regressor_iv_data, monkeypatch):
    y, x, w, x_eval = two_regressor_iv_data
    spec = {"basis": "additive", "j_x_segments": 3, "k_w_segments": 6, "deriv_index": 2}
    r = _r_npiv(y, x, w, x_eval=x_eval, **spec)
    py = _python_npiv(y, x, w, r, monkeypatch, x_eval=x_eval, **spec)
    x1_levels = prodspline(x, np.array([[3, 2], [3, 2]]), xeval=x_eval, knots="uniform", basis="additive").basis[:, :5]

    np.testing.assert_allclose(py.h, r["h"], rtol=1e-9, atol=1e-11)
    assert py.cv == pytest.approx(r["cv"], rel=1e-9)
    np.testing.assert_allclose(py.deriv + x1_levels @ py.beta[1:6], r["deriv"], rtol=1e-9, atol=1e-10)
    assert np.max(np.abs(x1_levels @ py.beta[1:6])) > 0.01


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_degree_zero_derivative_matches(engel_data):
    y, x, w = _engel_args(engel_data)
    spec = {"j_x_degree": 0, "j_x_segments": 1, "k_w_segments": 4, "ucb_h": False, "ucb_deriv": False}
    r = _r_npiv(y, x, w, x_eval=ENGEL_GRID, **spec)
    with pytest.warns(UserWarning, match="deriv order too large"):
        py = _python_npiv(y, x, w, x_eval=ENGEL_GRID, **spec)

    np.testing.assert_allclose(r["h"], r["h"][0], rtol=1e-12)
    np.testing.assert_allclose(py.h, r["h"][0], rtol=1e-10)
    np.testing.assert_array_equal(r["deriv"], 0.0)
    np.testing.assert_array_equal(py.deriv, 0.0)


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_quantile_knots_match(engel_data, monkeypatch):
    y, x, w = _engel_args(engel_data)
    r = _r_npiv(y, x, w, x_eval=ENGEL_GRID, knots="quantiles", j_x_segments=2, k_w_segments=5, biters=199)
    py = _python_npiv(
        y, x, w, r, monkeypatch, x_eval=ENGEL_GRID, knots="quantiles", j_x_segments=2, k_w_segments=5, biters=199
    )

    np.testing.assert_allclose(py.h, r["h"], rtol=1e-10, atol=1e-12, err_msg="Quantile knots h don't match R")
    np.testing.assert_allclose(
        py.deriv, r["deriv"], rtol=1e-10, atol=1e-12, err_msg="Quantile knots deriv don't match R"
    )
    assert py.cv == pytest.approx(r["cv"], rel=1e-10)


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_quantile_knots_regression_h_matches(engel_data):
    y, x, _ = _engel_args(engel_data)
    py = _python_npiv(y, x, x, x_eval=ENGEL_GRID, knots="quantiles", j_x_segments=4, k_w_degree=3, k_w_segments=4)
    r = _r_npiv(y, x, x, x_eval=ENGEL_GRID, knots="quantiles", j_x_segments=4, k_w_degree=3, k_w_segments=4)

    np.testing.assert_allclose(py.h, r["h"], rtol=1e-8, atol=1e-10, err_msg="Quantile knots regression h don't match R")


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_no_ucb_h_matches(engel_data):
    y, x, w = _engel_args(engel_data)
    py = _python_npiv(
        y, x, w, x_eval=ENGEL_GRID, j_x_segments=1, k_w_segments=4, ucb_h=False, ucb_deriv=False, biters=1
    )
    r = _r_npiv(y, x, w, x_eval=ENGEL_GRID, j_x_segments=1, k_w_segments=4, ucb_h=False, ucb_deriv=False, biters=1)

    assert py.cv is None
    np.testing.assert_allclose(py.h, r["h"], rtol=1e-10, atol=1e-12, err_msg="No-UCB h don't match R")
    np.testing.assert_allclose(py.deriv, r["deriv"], rtol=1e-10, atol=1e-12, err_msg="No-UCB deriv don't match R")


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
@pytest.mark.parametrize(
    "j_deg,j_seg,k_deg,k_seg",
    [
        (2, 2, 3, 3),
        (4, 1, 5, 3),
        (3, 3, 4, 5),
    ],
)
def test_npiv_iv_degree_segment_combos(engel_data, monkeypatch, j_deg, j_seg, k_deg, k_seg):
    y, x, w = _engel_args(engel_data)
    spec = {"j_x_degree": j_deg, "j_x_segments": j_seg, "k_w_degree": k_deg, "k_w_segments": k_seg}
    r = _r_npiv(y, x, w, x_eval=ENGEL_GRID, **spec)
    py = _python_npiv(y, x, w, r, monkeypatch, x_eval=ENGEL_GRID, **spec)

    np.testing.assert_allclose(py.h, r["h"], rtol=1e-9, atol=1e-11, err_msg=f"h mismatch for deg=({j_deg},{k_deg})")
    np.testing.assert_allclose(
        py.deriv, r["deriv"], rtol=1e-9, atol=1e-11, err_msg=f"deriv mismatch for deg=({j_deg},{k_deg})"
    )
    assert py.cv == pytest.approx(r["cv"], rel=1e-9)
