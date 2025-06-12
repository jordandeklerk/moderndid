import numpy as np

from pydid import drdid_imp_rc


def dgp_rc_for_test(n=2000, seed=42):
    np.random.seed(seed)
    # covariates
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    x3 = np.random.normal(0, 1, n)
    x4 = np.random.normal(0, 1, n)
    # Post-treatment dummy
    post = np.random.binomial(1, 0.5, n)
    # Treatment indicator
    d_propensity = 1 / (1 + np.exp(-(x1 + x3)))
    d = (np.random.uniform(size=n) < d_propensity).astype(int)
    # outcomes
    y = 1 + 2 * x1 + 3 * x2 + 4 * x3 + 5 * x4 + post + d * post + np.random.normal(0, 1, n)
    # matrix of covariates
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
    result = drdid_imp_rc(
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
    result = drdid_imp_rc(
        y=setup["y"],
        post=setup["post"],
        d=setup["d"],
        covariates=setup["covariates"],
        i_weights=setup["i_weights"],
        boot=True,
        boot_type="weighted",
        nboot=100,
    )
    assert result.att is not None
    assert result.se is not None
    assert np.isclose(result.att, 1.0, atol=0.2)
    assert result.boots is not None
    assert len(result.boots) == 100


def test_multiplier_bootstrap():
    setup = get_test_data()
    result = drdid_imp_rc(
        y=setup["y"],
        post=setup["post"],
        d=setup["d"],
        covariates=setup["covariates"],
        i_weights=setup["i_weights"],
        boot=True,
        boot_type="multiplier",
        nboot=100,
    )
    assert result.att is not None
    assert result.se is not None
    assert np.isclose(result.att, 1.0, atol=0.2)
    assert result.boots is not None
    assert len(result.boots) == 100


def test_influence_function():
    setup = get_test_data()
    result = drdid_imp_rc(
        y=setup["y"],
        post=setup["post"],
        d=setup["d"],
        covariates=setup["covariates"],
        i_weights=setup["i_weights"],
        influence_func=True,
    )
    assert result.att_inf_func is not None
    assert len(result.att_inf_func) == setup["n"]


def test_no_covariates():
    setup = get_test_data()
    result = drdid_imp_rc(
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
