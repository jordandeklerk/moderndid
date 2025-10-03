"""Tests for the locally efficient doubly robust DiD estimator for repeated cross-section data."""

import numpy as np
import pytest

from moderndid import drdid_rc


def dgp_rc_for_test(n=2000, seed=42):
    np.random.seed(seed)
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    x3 = np.random.normal(0, 1, n)
    x4 = np.random.normal(0, 1, n)
    post = np.random.binomial(1, 0.5, n)
    d_propensity = 1 / (1 + np.exp(-(x1 + x3)))
    d = (np.random.uniform(size=n) < d_propensity).astype(int)
    y = 1 + 2 * x1 + 3 * x2 + 4 * x3 + 5 * x4 + post + d * post + np.random.normal(0, 1, n)
    covariates = np.column_stack((np.ones(n), x1, x2, x3, x4))
    return y, post, d, covariates


def get_test_data():
    y, post, d, covariates = dgp_rc_for_test()
    i_weights = np.ones(len(y))
    return {
        "y": y,
        "post": post,
        "d": d,
        "covariates": covariates,
        "i_weights": i_weights,
        "n": len(y),
    }


def test_analytical_inference():
    setup = get_test_data()
    result = drdid_rc(
        y=setup["y"],
        post=setup["post"],
        d=setup["d"],
        covariates=setup["covariates"],
        i_weights=setup["i_weights"],
        boot=False,
    )
    assert result.att is not None
    assert result.se is not None
    assert np.isclose(result.att, 1.0, atol=0.2)
    assert result.boots is None


def test_weighted_bootstrap():
    setup = get_test_data()
    result = drdid_rc(
        y=setup["y"],
        post=setup["post"],
        d=setup["d"],
        covariates=setup["covariates"],
        i_weights=setup["i_weights"],
        boot=True,
        boot_type="weighted",
        nboot=20,
    )
    assert result.att is not None
    assert result.se is not None
    assert np.isclose(result.att, 1.0, atol=0.2)
    assert result.boots is not None
    assert len(result.boots) == result.args["nboot"]


def test_multiplier_bootstrap():
    setup = get_test_data()
    result = drdid_rc(
        y=setup["y"],
        post=setup["post"],
        d=setup["d"],
        covariates=setup["covariates"],
        i_weights=setup["i_weights"],
        boot=True,
        boot_type="multiplier",
        nboot=20,
    )
    assert result.att is not None
    assert result.se is not None
    assert np.isclose(result.att, 1.0, atol=0.2)
    assert result.boots is not None
    assert len(result.boots) == result.args["nboot"]


def test_influence_function():
    setup = get_test_data()
    result = drdid_rc(
        y=setup["y"],
        post=setup["post"],
        d=setup["d"],
        covariates=setup["covariates"],
        i_weights=setup["i_weights"],
        influence_func=True,
    )
    assert result.att_inf_func is not None
    assert len(result.att_inf_func) == setup["n"]
    assert np.abs(np.mean(result.att_inf_func)) < 0.1


def test_no_covariates():
    setup = get_test_data()
    result = drdid_rc(
        y=setup["y"],
        post=setup["post"],
        d=setup["d"],
        covariates=None,
        i_weights=setup["i_weights"],
        boot=False,
    )
    assert result.att is not None
    assert result.se is not None
    assert not np.isnan(result.att)
    assert not np.isnan(result.se)


def test_with_weights():
    n_units = 200
    rng = np.random.RandomState(42)

    post = rng.binomial(1, 0.5, n_units)
    d = rng.binomial(1, 0.5, n_units)
    x = np.column_stack([np.ones(n_units), rng.randn(n_units, 2)])
    y = rng.randn(n_units) + 2 * d * post
    i_weights = rng.uniform(0.5, 1.5, n_units)

    result = drdid_rc(y=y, post=post, d=d, covariates=x, i_weights=i_weights)

    assert isinstance(result.att, float)
    assert result.se > 0


def test_no_treated():
    n_units = 100
    rng = np.random.RandomState(42)

    post = rng.binomial(1, 0.5, n_units)
    d = np.zeros(n_units)
    x = np.column_stack([np.ones(n_units), rng.randn(n_units, 2)])
    y = rng.randn(n_units)

    with pytest.raises(ValueError, match="No effectively treated units"):
        drdid_rc(y=y, post=post, d=d, covariates=x)


def test_all_treated():
    n_units = 100
    rng = np.random.RandomState(42)

    post = rng.binomial(1, 0.5, n_units)
    d = np.ones(n_units)
    x = np.column_stack([np.ones(n_units), rng.randn(n_units, 2)])
    y = rng.randn(n_units) + 2 * post

    with pytest.raises(ValueError, match="No control units"):
        drdid_rc(y=y, post=post, d=d, covariates=x)


def test_no_post_treatment():
    n_units = 100
    rng = np.random.RandomState(42)

    post = np.zeros(n_units)
    d = rng.binomial(1, 0.5, n_units)
    x = np.column_stack([np.ones(n_units), rng.randn(n_units, 2)])
    y = rng.randn(n_units)

    with pytest.raises(ValueError, match="No post-treatment observations"):
        drdid_rc(y=y, post=post, d=d, covariates=x)


def test_no_pre_treatment():
    n_units = 100
    rng = np.random.RandomState(42)

    post = np.ones(n_units)
    d = rng.binomial(1, 0.5, n_units)
    x = np.column_stack([np.ones(n_units), rng.randn(n_units, 2)])
    y = rng.randn(n_units) + 2 * d

    with pytest.raises(ValueError, match="No pre-treatment observations"):
        drdid_rc(y=y, post=post, d=d, covariates=x)


def test_trimming():
    n_units = 200
    rng = np.random.RandomState(42)

    post = rng.binomial(1, 0.5, n_units)
    d = rng.binomial(1, 0.05, n_units)
    x = np.column_stack([np.ones(n_units), rng.randn(n_units, 2)])
    y = rng.randn(n_units) + 2 * d * post

    result = drdid_rc(y=y, post=post, d=d, covariates=x, trim_level=0.9)

    assert isinstance(result.att, float)
    assert result.se > 0


def test_singular_covariate_matrix():
    n_units = 100
    rng = np.random.RandomState(42)

    post = rng.binomial(1, 0.5, n_units)
    d = rng.binomial(1, 0.5, n_units)
    x1 = rng.randn(n_units)
    x = np.column_stack([np.ones(n_units), x1, x1])
    y = rng.randn(n_units) + 2 * d * post

    with pytest.raises(ValueError, match="[Ss]ingular"):
        drdid_rc(y=y, post=post, d=d, covariates=x)


def test_negative_weights():
    n_units = 100
    rng = np.random.RandomState(42)

    post = rng.binomial(1, 0.5, n_units)
    d = rng.binomial(1, 0.5, n_units)
    x = np.column_stack([np.ones(n_units), rng.randn(n_units, 2)])
    y = rng.randn(n_units) + 2 * d * post
    i_weights = rng.randn(n_units)

    with pytest.raises(ValueError, match="non-negative"):
        drdid_rc(y=y, post=post, d=d, covariates=x, i_weights=i_weights)


def test_perfect_separation():
    n_units = 100
    rng = np.random.RandomState(42)

    post = rng.binomial(1, 0.5, n_units)
    d = np.zeros(n_units)
    x = np.column_stack([np.ones(n_units), rng.randn(n_units, 2)])
    x[0:10, 1] = 100
    d[0:10] = 1
    y = rng.randn(n_units) + 2 * d * post

    with pytest.raises(np.linalg.LinAlgError, match="Singular matrix"):
        drdid_rc(y=y, post=post, d=d, covariates=x)


def test_bootstrap_with_zero_se():
    n_units = 50
    rng = np.random.RandomState(42)

    post = rng.binomial(1, 0.5, n_units)
    d = rng.binomial(1, 0.5, n_units)
    x = np.column_stack([np.ones(n_units)])
    y = np.ones(n_units) * (1 + d * post)

    result = drdid_rc(y=y, post=post, d=d, covariates=x, boot=True, boot_type="multiplier", nboot=10)

    assert result.att is not None
    if result.se == 0:
        assert result.lci == result.att
        assert result.uci == result.att


@pytest.mark.parametrize("covariates", [None, "with_covariates"])
def test_basic_functionality(covariates):
    n_units = 200
    rng = np.random.RandomState(42)

    post = rng.binomial(1, 0.5, n_units)
    d = rng.binomial(1, 0.5, n_units)

    if covariates == "with_covariates":
        x = rng.randn(n_units, 3)
        x = np.column_stack([np.ones(n_units), x])
    else:
        x = None

    y = rng.randn(n_units) + 2 * d * post

    result = drdid_rc(y=y, post=post, d=d, covariates=x)

    assert isinstance(result.att, float)
    assert isinstance(result.se, float)
    assert result.se > 0
    assert result.lci < result.uci
    assert result.boots is None
    assert result.att_inf_func is None


@pytest.mark.parametrize("boot_type", ["weighted", "multiplier"])
def test_bootstrap_types(boot_type):
    n_units = 100
    rng = np.random.RandomState(42)

    post = rng.binomial(1, 0.5, n_units)
    d = rng.binomial(1, 0.5, n_units)
    x = np.column_stack([np.ones(n_units), rng.randn(n_units, 2)])
    y = rng.randn(n_units) + 2 * d * post

    result = drdid_rc(y=y, post=post, d=d, covariates=x, boot=True, boot_type=boot_type, nboot=20)

    assert result.boots is not None
    assert len(result.boots) == 100
    assert result.se > 0
    assert result.lci < result.uci
