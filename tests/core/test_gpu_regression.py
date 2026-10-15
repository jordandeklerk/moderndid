"""Tests for the GPU least squares and logistic regression helpers against statsmodels."""

import numpy as np
import pytest
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

from moderndid.cupy.backend import to_numpy, use_backend
from moderndid.cupy.regression import cupy_logistic_irls, cupy_wls


@pytest.mark.parametrize("seed,n,k,weighted", [(42, 200, 3, False), (123, 300, 4, True), (77, 250, 2, True)])
def test_cupy_wls_matches_statsmodels_wls_and_a_gaussian_glm(xp, seed, n, k, weighted):
    rng = np.random.default_rng(seed)
    design = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
    y = design @ rng.standard_normal(k) + 0.3 * rng.standard_normal(n)
    weights = rng.uniform(0.5, 3.0, n) if weighted else np.ones(n)

    beta, _ = cupy_wls(xp.asarray(y), xp.asarray(design), xp.asarray(weights))
    wls = sm.WLS(y, design, weights=weights).fit()
    glm = sm.GLM(y, design, family=sm.families.Gaussian(), var_weights=weights).fit()

    assert isinstance(beta, xp.ndarray)
    np.testing.assert_allclose(to_numpy(beta), wls.params, rtol=1e-10)
    np.testing.assert_allclose(to_numpy(beta), glm.params, rtol=1e-10)


def test_cupy_wls_gives_a_rank_deficient_design_the_minimum_norm_coefficients(xp, weighted_bootstrap_inputs):
    inputs = weighted_bootstrap_inputs["panel"]
    design = np.column_stack([inputs["x"], inputs["x"][:, 1] / 1000])

    beta, fitted = cupy_wls(xp.asarray(inputs["delta_y"]), xp.asarray(design), xp.asarray(inputs["i_weights"]))
    expected = sm.WLS(inputs["delta_y"], design, weights=inputs["i_weights"]).fit()

    assert np.linalg.matrix_rank(design) == 2
    assert isinstance(beta, xp.ndarray)
    assert isinstance(fitted, xp.ndarray)
    np.testing.assert_allclose(to_numpy(beta), expected.params, rtol=1e-10)
    np.testing.assert_allclose(to_numpy(beta)[2], to_numpy(beta)[1] / 1000, rtol=1e-10)
    np.testing.assert_allclose(to_numpy(fitted), expected.fittedvalues, rtol=1e-10)


@pytest.mark.parametrize(
    "scale,band,rel",
    [(1.0, (1.0, 1e10), 1e-8), (1e-11, (1e10, 1e15), 1e-2)],
    ids=["well_conditioned", "ill_conditioned"],
)
def test_cupy_wls_condition_number_matches_statsmodels(xp, weighted_bootstrap_inputs, scale, band, rel):
    inputs = weighted_bootstrap_inputs["panel"]
    covariate = inputs["x"][:, 1]
    design = np.column_stack([inputs["x"], covariate + scale * np.random.default_rng(3).normal(size=covariate.size)])

    _, _, condition_number = cupy_wls(
        xp.asarray(inputs["delta_y"]),
        xp.asarray(design),
        xp.asarray(inputs["i_weights"]),
        return_condition_number=True,
    )
    expected = sm.WLS(inputs["delta_y"], design, weights=inputs["i_weights"]).fit().condition_number

    assert condition_number == pytest.approx(expected, rel=rel)
    assert band[0] < condition_number < band[1]


@pytest.mark.parametrize("seed,n,k,weighted", [(42, 500, 2, False), (99, 400, 3, True), (55, 600, 6, True)])
def test_cupy_logistic_irls_matches_statsmodels_glm(xp, seed, n, k, weighted):
    rng = np.random.default_rng(seed)
    design = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
    probability = 1 / (1 + np.exp(-(design @ (0.5 * rng.standard_normal(k)))))
    y = rng.binomial(1, probability).astype(float)
    weights = rng.uniform(0.3, 2.5, n) if weighted else np.ones(n)

    beta, mu, converged = cupy_logistic_irls(
        xp.asarray(y), xp.asarray(design), xp.asarray(weights), return_converged=True
    )
    expected = sm.GLM(y, design, family=sm.families.Binomial(), freq_weights=weights).fit()

    assert converged
    assert expected.converged
    assert isinstance(beta, xp.ndarray)
    assert isinstance(mu, xp.ndarray)
    np.testing.assert_allclose(to_numpy(beta), expected.params, rtol=1e-8)
    np.testing.assert_allclose(to_numpy(mu), expected.predict(design), rtol=1e-8)


def test_cupy_logistic_irls_reports_no_convergence_at_the_iteration_limit(xp, weighted_bootstrap_inputs):
    inputs = weighted_bootstrap_inputs["panel"]
    arrays = [xp.asarray(inputs[name]) for name in ("d", "x", "i_weights")]
    model = sm.GLM(inputs["d"], inputs["x"], family=sm.families.Binomial(), freq_weights=inputs["i_weights"])

    beta, _, converged = cupy_logistic_irls(*arrays, max_iter=2, return_converged=True)
    _, _, converged_without_limit = cupy_logistic_irls(*arrays, return_converged=True)
    expected = model.fit(maxiter=2)

    assert not converged
    assert not expected.converged
    assert converged_without_limit
    np.testing.assert_allclose(to_numpy(beta), expected.params, rtol=1e-10)


def test_cupy_logistic_irls_warns_on_perfect_separation_like_statsmodels(xp, weighted_bootstrap_inputs):
    inputs = weighted_bootstrap_inputs["panel"]
    treated = (inputs["x"][:, 1] > 0.8).astype(float)
    message = "Perfect separation or prediction detected, parameter may not be identified"

    with pytest.warns(PerfectSeparationWarning, match=message):
        expected = sm.GLM(treated, inputs["x"], family=sm.families.Binomial(), freq_weights=inputs["i_weights"]).fit()
    with pytest.warns(PerfectSeparationWarning, match=message):
        _, _, converged = cupy_logistic_irls(
            xp.asarray(treated), xp.asarray(inputs["x"]), xp.asarray(inputs["i_weights"]), return_converged=True
        )

    assert converged == expected.converged


@pytest.mark.gpu
def test_cupy_wls_raises_memory_error_when_the_gpu_runs_out_of_memory(tiny_gpu_memory_pool):
    with use_backend("cupy"), pytest.raises(MemoryError, match="GPU out of memory during WLS"):
        cupy_wls(np.zeros(1000), np.zeros((1000, 2)), np.ones(1000))


@pytest.mark.gpu
def test_cupy_logistic_irls_raises_memory_error_when_the_gpu_runs_out_of_memory(request, cp):
    arrays = [cp.zeros(1000), cp.zeros((1000, 2)), cp.ones(1000)]
    request.getfixturevalue("tiny_gpu_memory_pool")

    with use_backend("cupy"), pytest.raises(MemoryError, match="GPU out of memory during logistic IRLS"):
        cupy_logistic_irls(*arrays)
