"""Tests for IPT repeated cross-section bootstrap inference functions."""

import re
import warnings

import numpy as np
import pytest

from moderndid import (
    drdid_imp_local_rc,
    drdid_imp_rc,
    wboot_drdid_ipt_rc1,
    wboot_drdid_ipt_rc2,
)


@pytest.mark.parametrize("data_fixture", ["trimmed_two_period_data", "wide_rc_two_period_data"])
@pytest.mark.parametrize(
    ("bootstrap", "locally_efficient"),
    [(wboot_drdid_ipt_rc1, False), (wboot_drdid_ipt_rc2, True)],
    ids=["ipt_rc1", "ipt_rc2"],
)
def test_ipt_rc_bootstrap_draws_refit_the_estimate_at_the_drawn_weights(
    request, preset_bootstrap_weights, tilting_scores, rc_dr_att, data_fixture, bootstrap, locally_efficient
):
    data = request.getfixturevalue(data_fixture)
    y, post, d, x = data["y"], data["post"], data["d"], data["x"]
    i_weights = np.random.default_rng(3).uniform(0.5, 1.5, len(d))
    draws = np.random.default_rng(4).uniform(0.5, 1.5, (3, len(d)))
    expected = [
        rc_dr_att(
            y,
            post,
            d,
            x,
            weights,
            tilting_scores(d, x, weights),
            improved=True,
            locally_efficient=locally_efficient,
            trim_level=0.9,
        )
        for weights in i_weights * draws
    ]
    preset_bootstrap_weights(draws)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        boot_estimates = bootstrap(y=y, post=post, d=d, x=x, i_weights=i_weights, biters=3, trim_level=0.9)

    assert isinstance(boot_estimates, np.ndarray)
    assert np.all(np.isfinite(boot_estimates))
    np.testing.assert_allclose(boot_estimates, expected, rtol=1e-8)
    assert [str(w.message) for w in record] == []


@pytest.mark.parametrize("bootstrap", [wboot_drdid_ipt_rc1, wboot_drdid_ipt_rc2], ids=["ipt_rc1", "ipt_rc2"])
def test_ipt_rc_bootstraps_reject_invalid_inputs(bootstrap):
    rng = np.random.default_rng(42)
    n = 50
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = rng.standard_normal(n)
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    weights = np.ones(n)

    with pytest.raises(TypeError, match=f"^{re.escape('y must be a NumPy array.')}$"):
        bootstrap(list(y), post, d, x, weights)

    with pytest.raises(ValueError, match=f"^{re.escape('All arrays must have the same number of observations.')}$"):
        bootstrap(y[:-1], post, d, x, weights)

    with pytest.raises(ValueError, match=f"^{re.escape('biters must be a positive integer.')}$"):
        bootstrap(y, post, d, x, weights, biters=0)

    with pytest.raises(ValueError, match=f"^{re.escape('trim_level must be between 0 and 1.')}$"):
        bootstrap(y, post, d, x, weights, trim_level=1.5)


@pytest.mark.parametrize("bootstrap", [wboot_drdid_ipt_rc1, wboot_drdid_ipt_rc2], ids=["ipt_rc1", "ipt_rc2"])
def test_ipt_rc_bootstraps_mark_every_draw_nan_without_comparison_rows(bootstrap):
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = rng.standard_normal(n)
    post = rng.binomial(1, 0.5, n)
    reason = (
        "No units found in the comparison group in the pre-period. Pass data with comparison units in both periods."
    )
    summary = (
        "2 out of 2 bootstrap iterations failed and resulted in NaN. This might be due to issues in propensity score "
        "estimation, outcome regression, or the AIPW calculation itself, such as perfect prediction, collinearity, or "
        "small effective sample sizes after weighting or trimming."
    )

    with pytest.warns(UserWarning) as record:
        boot_estimates = bootstrap(y=y, post=post, d=np.ones(n), x=x, i_weights=np.ones(n), biters=2, random_state=42)
    messages = [str(w.message) for w in record]

    assert [m for m in messages if m.startswith("Outcome regression failed")] == [
        f"Outcome regression failed in bootstrap {b}: {reason}" for b in range(2)
    ]
    assert messages[-1] == summary
    np.testing.assert_array_equal(boot_estimates, np.full(2, np.nan))


@pytest.mark.parametrize("bootstrap", [wboot_drdid_ipt_rc1, wboot_drdid_ipt_rc2], ids=["ipt_rc1", "ipt_rc2"])
def test_ipt_rc_bootstraps_mark_every_draw_nan_when_a_covariate_repeats_the_others(trimmed_two_period_data, bootstrap):
    data = trimmed_two_period_data
    x = np.column_stack([data["x"], 1 - 2 * data["x"][:, 1]])
    reason = (
        "The regression design matrix of the comparison group in the pre-period is singular. Consider removing "
        "covariates that are constant or collinear within that group and period."
    )
    summary = (
        "3 out of 3 bootstrap iterations failed and resulted in NaN. This might be due to issues in propensity score "
        "estimation, outcome regression, or the AIPW calculation itself, such as perfect prediction, collinearity, or "
        "small effective sample sizes after weighting or trimming."
    )

    with pytest.warns(UserWarning) as record:
        boot_estimates = bootstrap(
            y=data["y"], post=data["post"], d=data["d"], x=x, i_weights=np.ones(len(x)), biters=3, random_state=42
        )
    messages = [str(w.message) for w in record]

    assert [m for m in messages if m.startswith("Outcome regression failed")] == [
        f"Outcome regression failed in bootstrap {b}: {reason}" for b in range(3)
    ]
    assert messages[-1] == summary
    np.testing.assert_array_equal(boot_estimates, np.full(3, np.nan))


@pytest.mark.parametrize("bootstrap", [wboot_drdid_ipt_rc1, wboot_drdid_ipt_rc2], ids=["ipt_rc1", "ipt_rc2"])
def test_ipt_rc_bootstraps_mark_a_draw_nan_when_its_tilting_fit_fails(trimmed_two_period_data, bootstrap):
    data = trimmed_two_period_data
    x = data["x"].copy()
    x[0, 1] = np.nan
    summary = (
        "2 out of 2 bootstrap iterations failed and resulted in NaN. This might be due to issues in propensity score "
        "estimation, outcome regression, or the AIPW calculation itself, such as perfect prediction, collinearity, or "
        "small effective sample sizes after weighting or trimming."
    )

    with pytest.warns(UserWarning) as record:
        boot_estimates = bootstrap(
            y=data["y"], post=data["post"], d=data["d"], x=x, i_weights=np.ones(len(x)), biters=2, random_state=42
        )
    messages = [str(w.message) for w in record]

    assert [m.partition(": ")[0] for m in messages[:-1]] == [
        f"Propensity score estimation (IPT) failed in bootstrap {b}" for b in range(2)
    ]
    assert messages[-1] == summary
    np.testing.assert_array_equal(boot_estimates, np.full(2, np.nan))


@pytest.mark.parametrize("bootstrap", [wboot_drdid_ipt_rc1, wboot_drdid_ipt_rc2], ids=["ipt_rc1", "ipt_rc2"])
def test_ipt_rc_bootstraps_repeat_their_draws_for_a_seed(bootstrap):
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + rng.standard_normal(n)
    spec = {"y": y, "post": post, "d": d, "x": x, "i_weights": np.ones(n), "biters": 10}

    first = bootstrap(**spec, random_state=123)
    second = bootstrap(**spec, random_state=123)
    other = bootstrap(**spec, random_state=124)

    assert np.all(np.isfinite(first))
    np.testing.assert_array_equal(first, second)
    assert np.all(first != other)


@pytest.mark.parametrize("bootstrap", [wboot_drdid_ipt_rc1, wboot_drdid_ipt_rc2], ids=["ipt_rc1", "ipt_rc2"])
def test_ipt_rc_bootstraps_warn_once_when_x_has_no_intercept(bootstrap):
    rng = np.random.default_rng(42)
    n = 50
    x_no_intercept = rng.standard_normal((n, 2))
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = x_no_intercept @ [0.5, -0.3] + 2 * d * post + rng.standard_normal(n)
    message = (
        "The first column of the covariate matrix 'x' does not appear to be an intercept (all ones). "
        "IPT propensity score estimation typically requires an intercept."
    )

    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$") as record:
        bootstrap(y=y, post=post, d=d, x=x_no_intercept, i_weights=np.ones(n), biters=10, random_state=42)

    assert len(record) == 1


@pytest.mark.parametrize(
    ("estimator", "bootstrap", "locally_efficient"),
    [(drdid_imp_rc, wboot_drdid_ipt_rc1, False), (drdid_imp_local_rc, wboot_drdid_ipt_rc2, True)],
    ids=["drdid_imp_rc", "drdid_imp_local_rc"],
)
def test_ipt_rc_bootstraps_fit_regressions_on_untrimmed_weights(
    unit_bootstrap_weights, trimmed_two_period_data, tilting_scores, rc_dr_att, estimator, bootstrap, locally_efficient
):
    data = trimmed_two_period_data
    weights = np.ones(len(data["d"]))
    scores = tilting_scores(data["d"], data["x"], weights)
    expected = rc_dr_att(
        data["y"],
        data["post"],
        data["d"],
        data["x"],
        weights,
        scores,
        improved=True,
        locally_efficient=locally_efficient,
        trim_level=0.9,
    )
    estimate = estimator(data["y"], data["post"], data["d"], data["x"], trim_level=0.9).att

    draws = bootstrap(
        y=data["y"], post=data["post"], d=data["d"], x=data["x"], i_weights=weights, biters=2, trim_level=0.9
    )

    assert np.sum((data["d"] == 0) & (scores >= 0.9)) == 23
    assert np.all(np.isfinite(draws))
    np.testing.assert_allclose(draws, [expected, expected], rtol=1e-8)
    np.testing.assert_allclose(draws, estimate, rtol=1e-8)
