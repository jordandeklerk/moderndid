"""Tests for the GPU least squares and logistic regression helpers run on the strict CuPy stand-in."""

import numpy as np
import pytest
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

from moderndid.cupy.backend import use_backend
from moderndid.cupy.regression import cupy_logistic_irls, cupy_wls


def test_cupy_wls_gives_a_rank_deficient_design_the_minimum_norm_coefficients(strict_cupy, weighted_bootstrap_inputs):
    inputs = weighted_bootstrap_inputs["panel"]
    design = np.column_stack([inputs["x"], 2 * inputs["x"][:, 1]])

    with use_backend("cupy"):
        beta, fitted = cupy_wls(
            strict_cupy.asarray(inputs["delta_y"]),
            strict_cupy.asarray(design),
            strict_cupy.asarray(inputs["i_weights"]),
        )
    expected = sm.WLS(inputs["delta_y"], design, weights=inputs["i_weights"]).fit()

    assert np.linalg.matrix_rank(design) == 2
    assert isinstance(beta, strict_cupy.ndarray)
    assert isinstance(fitted, strict_cupy.ndarray)
    np.testing.assert_allclose(beta.get(), expected.params, rtol=1e-10)
    np.testing.assert_allclose(beta.get()[2], 2 * beta.get()[1], rtol=1e-10)
    np.testing.assert_allclose(fitted.get(), expected.fittedvalues, rtol=1e-10)


@pytest.mark.parametrize(
    "scale,band",
    [(1.0, (1.0, 1e10)), (1e-11, (1e10, 1e15))],
    ids=["well_conditioned", "ill_conditioned"],
)
def test_cupy_wls_condition_number_matches_statsmodels(strict_cupy, weighted_bootstrap_inputs, scale, band):
    inputs = weighted_bootstrap_inputs["panel"]
    covariate = inputs["x"][:, 1]
    design = np.column_stack([inputs["x"], covariate + scale * np.random.default_rng(3).normal(size=covariate.size)])

    with use_backend("cupy"):
        _, _, condition_number = cupy_wls(
            strict_cupy.asarray(inputs["delta_y"]),
            strict_cupy.asarray(design),
            strict_cupy.asarray(inputs["i_weights"]),
            return_condition_number=True,
        )
    expected = sm.WLS(inputs["delta_y"], design, weights=inputs["i_weights"]).fit().condition_number

    assert condition_number == pytest.approx(expected, rel=1e-10)
    assert band[0] < condition_number < band[1]


def test_cupy_logistic_irls_matches_statsmodels_glm(strict_cupy, weighted_bootstrap_inputs):
    inputs = weighted_bootstrap_inputs["panel"]

    with use_backend("cupy"):
        beta, mu, converged = cupy_logistic_irls(
            strict_cupy.asarray(inputs["d"]),
            strict_cupy.asarray(inputs["x"]),
            strict_cupy.asarray(inputs["i_weights"]),
            return_converged=True,
        )
    expected = sm.GLM(inputs["d"], inputs["x"], family=sm.families.Binomial(), freq_weights=inputs["i_weights"]).fit()

    assert converged
    assert expected.converged
    assert isinstance(beta, strict_cupy.ndarray)
    assert isinstance(mu, strict_cupy.ndarray)
    np.testing.assert_allclose(beta.get(), expected.params, rtol=1e-10)
    np.testing.assert_allclose(mu.get(), expected.predict(inputs["x"]), rtol=1e-10)


def test_cupy_logistic_irls_reports_no_convergence_at_the_iteration_limit(strict_cupy, weighted_bootstrap_inputs):
    inputs = weighted_bootstrap_inputs["panel"]
    arrays = [strict_cupy.asarray(inputs[name]) for name in ("d", "x", "i_weights")]
    model = sm.GLM(inputs["d"], inputs["x"], family=sm.families.Binomial(), freq_weights=inputs["i_weights"])

    with use_backend("cupy"):
        beta, _, converged = cupy_logistic_irls(*arrays, max_iter=2, return_converged=True)
        _, _, converged_without_limit = cupy_logistic_irls(*arrays, return_converged=True)
    expected = model.fit(maxiter=2)

    assert not converged
    assert not expected.converged
    assert converged_without_limit
    np.testing.assert_allclose(beta.get(), expected.params, rtol=1e-10)


def test_cupy_logistic_irls_warns_on_perfect_separation_like_statsmodels(strict_cupy, weighted_bootstrap_inputs):
    inputs = weighted_bootstrap_inputs["panel"]
    treated = (inputs["x"][:, 1] > 0.8).astype(float)
    message = "Perfect separation or prediction detected, parameter may not be identified"

    with pytest.warns(PerfectSeparationWarning, match=message):
        expected = sm.GLM(treated, inputs["x"], family=sm.families.Binomial(), freq_weights=inputs["i_weights"]).fit()
    with use_backend("cupy"), pytest.warns(PerfectSeparationWarning, match=message):
        _, _, converged = cupy_logistic_irls(
            strict_cupy.asarray(treated),
            strict_cupy.asarray(inputs["x"]),
            strict_cupy.asarray(inputs["i_weights"]),
            return_converged=True,
        )

    assert converged == expected.converged
