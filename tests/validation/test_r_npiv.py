"""Validation tests comparing Python NPIV implementation with R npiv package."""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

from tests.helpers import importorskip

np = importorskip("numpy")

from moderndid.didcont.npiv import npiv


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


def _r_npiv(
    w_var="logwages",
    j_x_degree=3,
    j_x_segments=1,
    k_w_degree=4,
    k_w_segments=4,
    knots="uniform",
    ucb_h=True,
    ucb_deriv=True,
    boot_num=99,
    timeout=300,
):
    """Run R npiv on the Engel95 dataset and return JSON results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result_path = Path(tmpdir) / "result.json"

        j_seg = f"J.x.segments={j_x_segments}" if j_x_segments is not None else "J.x.segments=NULL"
        k_seg = f"K.w.segments={k_w_segments}" if k_w_segments is not None else "K.w.segments=NULL"
        ucb_h_r = "TRUE" if ucb_h else "FALSE"
        ucb_d_r = "TRUE" if ucb_deriv else "FALSE"

        r_script = f"""
library(npiv)
library(jsonlite)

data("Engel95", package = "npiv")
Engel95 <- Engel95[order(Engel95$logexp),]

x.eval <- seq(4.5, 6.5, length = 100)

set.seed(42)
result <- npiv(
    Y = Engel95$food,
    X = Engel95$logexp,
    W = Engel95${w_var},
    X.eval = x.eval,
    J.x.degree = {j_x_degree},
    {j_seg},
    K.w.degree = {k_w_degree},
    {k_seg},
    knots = "{knots}",
    deriv.index = 1,
    deriv.order = 1,
    ucb.h = {ucb_h_r},
    ucb.deriv = {ucb_d_r},
    boot.num = {boot_num},
    progress = FALSE
)

output <- list(
    h = as.numeric(result$h),
    deriv = as.numeric(result$deriv)
)
if (!is.null(result$h.lower)) output$h_lower <- as.numeric(result$h.lower)
if (!is.null(result$h.upper)) output$h_upper <- as.numeric(result$h.upper)
if (!is.null(result$h.lower.deriv)) output$h_lower_deriv <- as.numeric(result$h.lower.deriv)
if (!is.null(result$h.upper.deriv)) output$h_upper_deriv <- as.numeric(result$h.upper.deriv)
output$J_x_segments <- result$J.x.segments
output$K_w_segments <- result$K.w.segments

write_json(output, "{result_path}", auto_unbox = TRUE, digits = 16)
"""
        try:
            return _run_r_script(r_script, result_path, timeout=timeout)
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, RuntimeError):
            return None


def _python_npiv(
    engel_data,
    w_key="logwages",
    j_x_degree=3,
    j_x_segments=1,
    k_w_degree=4,
    k_w_segments=4,
    knots="uniform",
    ucb_h=True,
    ucb_deriv=True,
    boot_num=99,
):
    """Run Python npiv on the Engel dataset."""
    x_eval = np.linspace(4.5, 6.5, 100)
    return npiv(
        y=engel_data["food"],
        x=engel_data["logexp"],
        w=engel_data[w_key],
        x_eval=x_eval.reshape(-1, 1),
        j_x_degree=j_x_degree,
        j_x_segments=j_x_segments,
        k_w_degree=k_w_degree,
        k_w_segments=k_w_segments,
        knots=knots,
        deriv_index=1,
        deriv_order=1,
        ucb_h=ucb_h,
        ucb_deriv=ucb_deriv,
        boot_num=boot_num,
        seed=42,
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_iv_h_matches(engel_data):
    py = _python_npiv(engel_data)
    r = _r_npiv()

    if r is None:
        pytest.fail("R npiv estimation failed")

    np.testing.assert_allclose(
        py.h, np.array(r["h"]), rtol=1e-8, atol=1e-8, err_msg="IV function estimates (h) don't match R"
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_iv_deriv_matches(engel_data):
    py = _python_npiv(engel_data)
    r = _r_npiv()

    if r is None:
        pytest.fail("R npiv estimation failed")

    np.testing.assert_allclose(
        py.deriv, np.array(r["deriv"]), rtol=1e-8, atol=1e-8, err_msg="IV derivative estimates don't match R"
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_iv_confidence_bands_match(engel_data):
    py = _python_npiv(engel_data)
    r = _r_npiv()

    if r is None:
        pytest.fail("R npiv estimation failed")

    assert py.h_lower is not None
    assert py.h_upper is not None

    np.testing.assert_allclose(py.h_lower, np.array(r["h_lower"]), atol=0.02, err_msg="IV h_lower don't match R")
    np.testing.assert_allclose(py.h_upper, np.array(r["h_upper"]), atol=0.02, err_msg="IV h_upper don't match R")
    assert np.all(py.h_lower <= py.h), "Lower band should be <= estimate"
    assert np.all(py.h <= py.h_upper), "Estimate should be <= upper band"


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_iv_deriv_confidence_bands_structure(engel_data):
    py = _python_npiv(engel_data)
    r = _r_npiv()

    if r is None:
        pytest.fail("R npiv estimation failed")

    assert py.h_lower_deriv is not None
    assert py.h_upper_deriv is not None
    assert py.h_lower_deriv.shape == (100,)
    assert py.h_upper_deriv.shape == (100,)
    assert np.all(py.h_lower_deriv <= py.deriv), "Lower deriv <= estimate"
    assert np.all(py.deriv <= py.h_upper_deriv), "Estimate <= upper deriv"

    assert py.h_lower_deriv.shape == np.array(r["h_lower_deriv"]).shape
    assert py.h_upper_deriv.shape == np.array(r["h_upper_deriv"]).shape


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_regression_h_matches(engel_data):
    py = _python_npiv(engel_data, w_key="logexp", j_x_segments=64, k_w_degree=3, k_w_segments=64)
    r = _r_npiv(w_var="logexp", j_x_segments=64, k_w_degree=3, k_w_segments=64)

    if r is None:
        pytest.fail("R npiv regression estimation failed")

    np.testing.assert_allclose(py.h, np.array(r["h"]), rtol=1e-8, atol=1e-8, err_msg="Regression h don't match R")


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_regression_deriv_matches(engel_data):
    py = _python_npiv(engel_data, w_key="logexp", j_x_segments=64, k_w_degree=3, k_w_segments=64)
    r = _r_npiv(w_var="logexp", j_x_segments=64, k_w_degree=3, k_w_segments=64)

    if r is None:
        pytest.fail("R npiv regression estimation failed")

    np.testing.assert_allclose(
        py.deriv, np.array(r["deriv"]), rtol=1e-8, atol=1e-8, err_msg="Regression deriv don't match R"
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_regression_confidence_bands_match(engel_data):
    py = _python_npiv(engel_data, w_key="logexp", j_x_segments=64, k_w_degree=3, k_w_segments=64)
    r = _r_npiv(w_var="logexp", j_x_segments=64, k_w_degree=3, k_w_segments=64)

    if r is None:
        pytest.fail("R npiv regression estimation failed")

    if py.h_lower is not None:
        assert np.median(np.abs(py.h_lower - np.array(r["h_lower"]))) < 0.01
    if py.h_upper is not None:
        assert np.median(np.abs(py.h_upper - np.array(r["h_upper"]))) < 0.01

    if py.h_lower is not None and py.h_upper is not None:
        assert np.all(py.h_lower <= py.h)
        assert np.all(py.h <= py.h_upper)

    if py.h_lower_deriv is not None and py.h_upper_deriv is not None:
        assert np.all(py.h_lower_deriv <= py.deriv)
        assert np.all(py.deriv <= py.h_upper_deriv)


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_data_driven_iv_h_matches(engel_data):
    py = _python_npiv(engel_data, j_x_segments=None, k_w_segments=None)
    r = _r_npiv(j_x_segments=None, k_w_segments=None, timeout=600)

    if r is None:
        pytest.fail("R npiv data-driven estimation failed")

    np.testing.assert_allclose(py.h, np.array(r["h"]), rtol=1e-8, atol=1e-8, err_msg="Data-driven IV h don't match R")


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_data_driven_iv_deriv_matches(engel_data):
    py = _python_npiv(engel_data, j_x_segments=None, k_w_segments=None)
    r = _r_npiv(j_x_segments=None, k_w_segments=None, timeout=600)

    if r is None:
        pytest.fail("R npiv data-driven estimation failed")

    np.testing.assert_allclose(
        py.deriv, np.array(r["deriv"]), rtol=1e-8, atol=1e-8, err_msg="Data-driven IV deriv don't match R"
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_data_driven_iv_confidence_bands(engel_data):
    py = _python_npiv(engel_data, j_x_segments=None, k_w_segments=None)
    r = _r_npiv(j_x_segments=None, k_w_segments=None, timeout=600)

    if r is None:
        pytest.fail("R npiv data-driven estimation failed")

    assert py.h_lower is not None
    assert py.h_upper is not None
    assert np.all(py.h_lower <= py.h)
    assert np.all(py.h <= py.h_upper)

    np.testing.assert_allclose(
        py.h_lower, np.array(r["h_lower"]), atol=0.04, err_msg="Data-driven h_lower don't match R"
    )
    np.testing.assert_allclose(
        py.h_upper, np.array(r["h_upper"]), atol=0.04, err_msg="Data-driven h_upper don't match R"
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_data_driven_regression_h_matches(engel_data):
    py = _python_npiv(engel_data, w_key="logexp", j_x_segments=None, k_w_degree=3, k_w_segments=None)
    r = _r_npiv(w_var="logexp", j_x_segments=None, k_w_degree=3, k_w_segments=None, timeout=600)

    if r is None:
        pytest.fail("R npiv data-driven regression failed")

    np.testing.assert_allclose(
        py.h, np.array(r["h"]), rtol=1e-8, atol=1e-8, err_msg="Data-driven regression h don't match R"
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_quantile_knots_h_matches(engel_data):
    py = _python_npiv(engel_data, knots="quantiles", j_x_segments=2, k_w_segments=4)
    r = _r_npiv(knots="quantiles", j_x_segments=2, k_w_segments=4)

    if r is None:
        pytest.fail("R npiv quantile knots estimation failed")

    np.testing.assert_allclose(py.h, np.array(r["h"]), rtol=1e-8, atol=1e-8, err_msg="Quantile knots h don't match R")


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_quantile_knots_deriv_matches(engel_data):
    py = _python_npiv(engel_data, knots="quantiles", j_x_segments=2, k_w_segments=4)
    r = _r_npiv(knots="quantiles", j_x_segments=2, k_w_segments=4)

    if r is None:
        pytest.fail("R npiv quantile knots estimation failed")

    np.testing.assert_allclose(
        py.deriv, np.array(r["deriv"]), rtol=1e-8, atol=1e-8, err_msg="Quantile knots deriv don't match R"
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_quantile_knots_regression_h_matches(engel_data):
    py = _python_npiv(engel_data, w_key="logexp", knots="quantiles", j_x_segments=4, k_w_degree=3, k_w_segments=4)
    r = _r_npiv(w_var="logexp", knots="quantiles", j_x_segments=4, k_w_degree=3, k_w_segments=4)

    if r is None:
        pytest.fail("R npiv quantile knots regression failed")

    np.testing.assert_allclose(
        py.h, np.array(r["h"]), rtol=1e-8, atol=1e-8, err_msg="Quantile knots regression h don't match R"
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
def test_npiv_no_ucb_h_matches(engel_data):
    py = _python_npiv(engel_data, ucb_h=False, ucb_deriv=False, boot_num=1)
    r = _r_npiv(ucb_h=False, ucb_deriv=False, boot_num=1)

    if r is None:
        pytest.fail("R npiv no-UCB estimation failed")

    np.testing.assert_allclose(py.h, np.array(r["h"]), rtol=1e-8, atol=1e-8, err_msg="No-UCB h don't match R")
    np.testing.assert_allclose(
        py.deriv, np.array(r["deriv"]), rtol=1e-8, atol=1e-8, err_msg="No-UCB deriv don't match R"
    )


@pytest.mark.skipif(not R_AVAILABLE, reason="R npiv package not available")
@pytest.mark.parametrize(
    "j_deg,j_seg,k_deg,k_seg",
    [
        (2, 2, 3, 3),
        (4, 1, 5, 3),
        (3, 3, 4, 5),
    ],
)
def test_npiv_iv_degree_segment_combos(engel_data, j_deg, j_seg, k_deg, k_seg):
    py = _python_npiv(engel_data, j_x_degree=j_deg, j_x_segments=j_seg, k_w_degree=k_deg, k_w_segments=k_seg)
    r = _r_npiv(j_x_degree=j_deg, j_x_segments=j_seg, k_w_degree=k_deg, k_w_segments=k_seg)

    if r is None:
        pytest.fail(f"R npiv failed for deg=({j_deg},{k_deg}) seg=({j_seg},{k_seg})")

    np.testing.assert_allclose(
        py.h, np.array(r["h"]), rtol=1e-8, atol=1e-8, err_msg=f"h mismatch for deg=({j_deg},{k_deg})"
    )
    np.testing.assert_allclose(
        py.deriv, np.array(r["deriv"]), rtol=1e-8, atol=1e-8, err_msg=f"deriv mismatch for deg=({j_deg},{k_deg})"
    )
