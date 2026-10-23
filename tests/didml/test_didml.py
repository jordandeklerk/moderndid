"""Tests for the ML doubly robust DiD estimator."""

import re
import warnings

import numpy as np
import pytest
from scipy import stats

from tests.helpers import importorskip

pl = importorskip("polars")
importorskip("moderndid.didml")

from moderndid import (
    aggte_didml,
    amle_weights,
    att_gt,
    blp_eventtimes,
    clan_glhtest,
    clan_ttest,
    didml,
    dynamic_cates,
    het_prep,
    lnw_did,
    summary_didml,
    summary_didml_agg,
)
from moderndid.core.preprocess.config import DIDMLConfig
from moderndid.didml.compute import simultaneous_critical_value
from moderndid.didml.nuisance import fit_delta, fit_rlearner
from moderndid.didml.nuisance.cv_lasso import cv_lasso_with_oof


def test_didml_balanced_panel(mpdta_data, mpdta_spec, mpdta_analytic_fit):
    result = mpdta_analytic_fit
    reference = att_gt(
        mpdta_data, est_method="dr", control_group="notyettreated", boot=False, cband=False, **mpdta_spec
    )

    assert result.n_units == 500
    assert result.estimation_params["n_obs"] == 2500
    assert result.estimation_params["cohort_counts"][2007.0] == 131
    np.testing.assert_array_equal(result.groups, reference.groups)
    np.testing.assert_array_equal(result.times, reference.times)
    np.testing.assert_allclose(result.drdid_benchmark, reference.att_gt, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.drdid_benchmark_se, reference.se_gt, rtol=1e-10)
    assert np.isfinite(result.att_gt).all()
    assert np.isfinite(result.se_gt).all()
    np.testing.assert_array_equal(result.unit_periods, np.zeros(500))
    assert result.aggregate_effects is None
    assert result.att_gt_draws is None
    np.testing.assert_array_equal(result.weights_ind, np.ones(500))
    assert np.sum(result.G == 2007.0) == 131


def test_didml_influence_functions_give_the_analytic_standard_errors(mpdta_analytic_fit):
    result = mpdta_analytic_fit
    inf = result.influence_func

    np.testing.assert_allclose(inf.sum(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(result.se_gt, np.sqrt((inf**2).sum(axis=0)) / 500, rtol=1e-12)
    pre = result.groups > result.times
    vcov = inf[:, pre].T @ inf[:, pre] / 500**2
    att = result.att_gt[pre]
    assert result.wald_stat == pytest.approx(att @ np.linalg.solve(vcov, att), rel=1e-10)


def test_didml_att_averages_the_treated_catt_and_the_residual_correction(mpdta_analytic_fit):
    result = mpdta_analytic_fit
    cates = result.cates.toarray()
    corrections = result.scores.toarray() - cates
    in_cell = result.influence_func != 0
    treated = result.G[:, None] == result.groups[None, :]

    expected = [
        cates[in_cell[:, j] & treated[:, j], j].mean() + corrections[in_cell[:, j], j].mean()
        for j in range(len(result.groups))
    ]
    np.testing.assert_allclose(result.att_gt, expected, rtol=1e-10)


def test_didml_bootstrap_refits_every_cell(mpdta_short_bootstrap_fit):
    result = mpdta_short_bootstrap_fit
    draws = result.att_gt_draws

    assert draws.shape == (2, 4)
    assert result.estimation_params["bootstrap"] is True
    np.testing.assert_allclose(result.se_gt, draws.std(axis=0, ddof=1), rtol=1e-12)
    assert result.cohort_share_draws.shape == (2, 2)
    assert np.all(result.cohort_share_draws.sum(axis=1) < 1)
    assert not np.allclose(draws, result.att_gt)


def test_didml_bootstrap_band_and_pretest_use_the_draws(mpdta_bootstrap_fit):
    result = mpdta_bootstrap_fit
    draws = result.att_gt_draws
    pre = result.groups > result.times
    vcov = np.cov(draws[:, pre], rowvar=False)
    n_pre, n_draws = int(pre.sum()), draws.shape[0]
    wald = result.att_gt[pre] @ np.linalg.solve(vcov, result.att_gt[pre])
    hotelling = stats.f.sf(wald * (n_draws - n_pre) / (n_pre * (n_draws - 1)), n_pre, n_draws - n_pre)

    expected = simultaneous_critical_value(draws, result.att_gt, result.se_gt, 0.05)
    assert result.critical_value == pytest.approx(expected)
    assert result.critical_value > stats.norm.ppf(0.975)
    assert result.wald_stat == pytest.approx(wald)
    assert result.wald_pvalue == pytest.approx(hotelling, abs=5e-6)
    assert abs(hotelling - stats.chi2.sf(wald, n_pre)) > 1e-2


def test_didml_bootstrap_pretest_needs_more_draws_than_pre_treatment_cells(mpdta_2007_cohort, mpdta_spec):
    message = (
        "Pre-test Wald not returned because 2 complete bootstrap draws cannot estimate the covariance of 2 "
        "pre-treatment cells. Pass biters above 2 or boot=False."
    )

    with pytest.warns(UserWarning, match=re.escape(message)):
        result = didml(mpdta_2007_cohort, **mpdta_spec, k_folds=2, random_state=0, biters=2)

    assert result.wald_stat is None
    assert result.wald_pvalue is None


def test_didml_bootstrap_with_a_tiny_cohort_raises_no_convergence_warning(mpdta_tiny_cohort, mpdta_spec):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = didml(mpdta_tiny_cohort, **mpdta_spec, k_folds=2, random_state=7, biters=4)

    assert [str(w.message) for w in caught if w.category.__name__ == "ConvergenceWarning"] == []
    assert np.all(np.isfinite(result.se_gt[result.groups == 2004]))


def test_didml_bootstrap_draws_repeat_with_the_seed_and_worker_processes(
    mpdta_short, mpdta_spec, mpdta_short_bootstrap_fit
):
    again = didml(mpdta_short, **mpdta_spec, k_folds=2, random_state=0, biters=2, n_jobs=2)

    np.testing.assert_array_equal(again.att_gt_draws, mpdta_short_bootstrap_fit.att_gt_draws)
    np.testing.assert_array_equal(again.cohort_share_draws, mpdta_short_bootstrap_fit.cohort_share_draws)


def test_didml_maketables_name_no_cluster_for_ignored_clustervars(mpdta_with_state, mpdta_spec):
    message = "Cluster-robust standard errors are not yet implemented for didml."

    with (
        pytest.warns(UserWarning, match=re.escape(message)),
        pytest.warns(UserWarning, match="^Pre-test Wald not returned because 2 complete bootstrap draws"),
    ):
        result = didml(mpdta_with_state, clustervars=["state"], k_folds=2, random_state=0, biters=2, **mpdta_spec)
    agg = aggte_didml(result)

    assert result.estimation_params["clustervars"] == ["state"]
    assert result.__maketables_vcov_info__ == {"vcov_type": "bootstrap", "clustervar": None}
    assert agg.__maketables_vcov_info__ == {"vcov_type": "bootstrap", "clustervar": None}


def test_simultaneous_critical_value_is_the_quantile_of_the_largest_deviation():
    draws = np.array([[1.0, 2.0], [3.0, 0.0], [2.0, 5.0], [0.0, 1.0]])
    deviations = np.max(np.abs(draws - [1.0, 1.0]) / [2.0, 4.0], axis=1)

    assert simultaneous_critical_value(draws, np.array([1.0, 1.0]), np.array([2.0, 4.0]), 0.5) == pytest.approx(
        np.quantile(deviations, 0.5)
    )


def test_didml_constant_weights_reproduce_the_unweighted_fit(mpdta_short, mpdta_spec, didml_options):
    weighted = didml(mpdta_short.with_columns(pl.lit(2.0).alias("w")), weightsname="w", **mpdta_spec, **didml_options)
    unweighted = didml(mpdta_short, **mpdta_spec, **didml_options)

    np.testing.assert_allclose(weighted.att_gt, unweighted.att_gt, rtol=1e-6)
    np.testing.assert_allclose(weighted.se_gt, unweighted.se_gt, rtol=1e-6)


def test_didml_unbalanced_panel_matches_complete_units(
    mpdta_short_unbalanced, mpdta_short_without_county_8001, mpdta_spec, didml_options
):
    with pytest.warns(UserWarning) as record:
        result = didml(mpdta_short_unbalanced, **mpdta_spec, **didml_options)
    expected = didml(mpdta_short_without_county_8001, **mpdta_spec, **didml_options)

    assert [str(warning.message) for warning in record if "units" in str(warning.message)] == [
        "Dropped 1 units while converting to balanced panel"
    ]
    assert result.n_units == 479
    assert result.estimation_params["n_obs"] == 1437
    assert result.estimation_params["cohort_counts"][2007.0] == 130
    np.testing.assert_array_equal(result.unit_ids, expected.unit_ids)
    np.testing.assert_allclose(result.att_gt, expected.att_gt, rtol=1e-10, atol=0)
    np.testing.assert_allclose(result.se_gt, expected.se_gt, rtol=1e-10, atol=0)
    np.testing.assert_allclose(result.influence_func, expected.influence_func, rtol=1e-10, atol=0)
    np.testing.assert_allclose(result.drdid_benchmark, expected.drdid_benchmark, rtol=1e-10, atol=0)


def test_didml_unbalanced_panel_benchmark_values(mpdta_unbalanced, mpdta_spec, didml_options):
    with pytest.warns(UserWarning, match="^Dropped 1 units while converting to balanced panel$"):
        result = didml(mpdta_unbalanced, **mpdta_spec, **didml_options)

    np.testing.assert_allclose(
        result.drdid_benchmark,
        [
            -0.0214476431,
            -0.0819451635,
            -0.1384672740,
            -0.1069038981,
            -0.0078795766,
            -0.0046391636,
            0.0088056750,
            -0.0412938656,
            0.0278802898,
            -0.0039929236,
            -0.0288987794,
            -0.0294653715,
        ],
        rtol=0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        result.drdid_benchmark_se,
        [
            0.0216370386,
            0.0283087981,
            0.0342008245,
            0.0328864930,
            0.0218314053,
            0.0182913483,
            0.0168826750,
            0.0197211441,
            0.0140059112,
            0.0156736865,
            0.0182171165,
            0.0163382674,
        ],
        rtol=0,
        atol=1e-9,
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"tname": "yeer"}, "tname='yeer' is not a column in the data. Did you mean 'year'?"),
        ({"idname": "countyrel"}, "idname='countyrel' is not a column in the data. Did you mean 'countyreal'?"),
    ],
)
def test_didml_names_misspelled_columns(mpdta_data, mpdta_spec, didml_options, changes, message):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        didml(mpdta_data, **(mpdta_spec | changes), **didml_options)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"control_group": None}, "control_group=None is not valid. Must be 'nevertreated' or 'notyettreated'."),
        ({"control_group": 3}, "control_group=3 is not valid. Must be 'nevertreated' or 'notyettreated'."),
        ({"base_period": None}, "base_period=None is not valid. Must be 'varying' or 'universal'."),
        ({"anticipation": -1}, "anticipation=-1 is not valid. Must be a non-negative number."),
        ({"n_jobs": 0}, "n_jobs=0 is not valid. Must be a positive integer or -1 for all cores."),
        ({"alpha": 1.5}, "alpha=1.5 is not valid. Must be between 0 and 1 (exclusive)."),
        (
            {"tname": "yeer", "control_group": None},
            "control_group=None is not valid. Must be 'nevertreated' or 'notyettreated'.",
        ),
    ],
)
def test_didml_rejects_invalid_options_like_att_gt(mpdta_data, mpdta_spec, didml_options, changes, message):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        didml(mpdta_data, **(mpdta_spec | changes), **didml_options)
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        att_gt(mpdta_data, **(mpdta_spec | changes))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"alpha": 1.5}, "alpha=1.5 is not valid. Must be between 0 and 1 (exclusive)."),
        ({"boot": True, "biters": 1}, "biters=1 is not valid. Must be an integer of at least 2 when boot=True."),
    ],
)
def test_didml_rejects_invalid_inference_options(mpdta_data, mpdta_spec, didml_options, changes, message):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        didml(mpdta_data, **mpdta_spec, **(didml_options | changes))


def test_didml_no_longer_accepts_tune_penalty(mpdta_data, mpdta_spec, didml_options):
    with pytest.raises(TypeError, match="tune_penalty"):
        didml(mpdta_data, tune_penalty=True, **mpdta_spec, **didml_options)
    with pytest.raises(TypeError, match="tune_penalty"):
        DIDMLConfig(**mpdta_spec, tune_penalty=True)


def test_didml_without_amle_weights_warns_that_analytic_errors_are_undefined(mpdta_short, mpdta_spec, didml_options):
    message = "With use_gamma=False the influence-function standard errors are undefined and se_gt holds NaN."

    with (
        pytest.warns(UserWarning, match=re.escape(message)),
        pytest.warns(UserWarning, match="^Pre-test Wald not returned due to NaN entries in pre-treatment variance.$"),
    ):
        result = didml(mpdta_short, use_gamma=False, **mpdta_spec, **didml_options)

    assert np.isnan(result.se_gt).all()
    assert np.isfinite(result.att_gt).all()


def test_didml_without_never_treated_leaves_latest_cohort_out(mpdta_without_never_treated, mpdta_spec, didml_options):
    result = didml(mpdta_without_never_treated, **mpdta_spec, **didml_options)
    reference = att_gt(
        mpdta_without_never_treated,
        est_method="dr",
        control_group="notyettreated",
        boot=False,
        cband=False,
        **mpdta_spec,
    )

    assert set(result.groups.tolist()) == {2004.0, 2006.0}
    assert result.n_units == 191
    np.testing.assert_array_equal(result.groups, reference.groups)
    np.testing.assert_array_equal(result.times, reference.times)
    np.testing.assert_allclose(result.drdid_benchmark, reference.att_gt, rtol=0, atol=1e-12)
    assert np.isfinite(result.att_gt).all()


@pytest.mark.filterwarnings("error:.*unbalanced:UserWarning")
def test_didml_rejects_repeated_unit_periods(mpdta_repeated_row, mpdta_spec, didml_options):
    message = (
        "The value of idname must be unique (by tname). Some units are observed more than once in a period. "
        "Rows repeat for the (countyreal, year) pair (17005, 2005)."
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        didml(mpdta_repeated_row, **mpdta_spec, **didml_options)


def test_didml_drops_infinite_outcome_like_a_missing_one(
    mpdta_short_infinite_outcome, mpdta_short_without_county_8001, mpdta_spec, didml_options
):
    with (
        pytest.warns(UserWarning, match="^Dropped 1 rows from original data due to missing values$"),
        pytest.warns(UserWarning, match="^Dropped 1 units while converting to balanced panel$"),
    ):
        result = didml(mpdta_short_infinite_outcome, **mpdta_spec, **didml_options)
    expected = didml(mpdta_short_without_county_8001, **mpdta_spec, **didml_options)

    assert result.n_units == 479
    np.testing.assert_array_equal(result.unit_ids, expected.unit_ids)
    np.testing.assert_allclose(result.att_gt, expected.att_gt, rtol=1e-10, atol=0)
    np.testing.assert_allclose(result.se_gt, expected.se_gt, rtol=1e-10, atol=0)
    np.testing.assert_allclose(result.drdid_benchmark, expected.drdid_benchmark, rtol=1e-10, atol=0)


def test_didml_reads_decimal_outcome_as_doubles(mpdta_short, mpdta_spec, didml_options):
    decimal = mpdta_short.with_columns(pl.col("lemp").cast(pl.Decimal(18, 6)))
    floats = decimal.with_columns(pl.col("lemp").cast(pl.Float64))

    result = didml(decimal, **mpdta_spec, **didml_options)
    expected = didml(floats, **mpdta_spec, **didml_options)

    np.testing.assert_array_equal(result.groups, expected.groups)
    np.testing.assert_array_equal(result.times, expected.times)
    np.testing.assert_allclose(result.att_gt, expected.att_gt, rtol=1e-12)
    np.testing.assert_allclose(result.se_gt, expected.se_gt, rtol=1e-12)


def test_didml_rejects_weights_without_positive_mean(mpdta_zero_weights, mpdta_spec, didml_options):
    message = "The weights variable 'w' must be non-negative with a positive mean."

    with pytest.raises(ValueError, match=re.escape(message)):
        didml(mpdta_zero_weights, weightsname="w", **mpdta_spec, **didml_options)


def test_didml_critical_value_stays_pointwise_without_the_bootstrap(mpdta_short, mpdta_spec, didml_options):
    result = didml(mpdta_short, cband=True, **mpdta_spec, **didml_options)

    assert result.critical_value == pytest.approx(stats.norm.ppf(0.975))


def test_aggte_didml_weights_cohorts_by_their_units(two_cohort_result):
    agg = aggte_didml(two_cohort_result)

    np.testing.assert_array_equal(agg.event_times, [-1, 0, 1, 2, 3])
    np.testing.assert_allclose(agg.att_by_event, [10.0, 5.75, 9.0, 3.0, 4.0])
    assert agg.overall_att == pytest.approx(5.4375)


def test_aggte_didml_falls_back_to_cohort_counts(two_cohort_counts_only):
    agg = aggte_didml(two_cohort_counts_only)

    np.testing.assert_allclose(agg.att_by_event, [10.0, 5.75, 9.0, 3.0, 4.0])
    np.testing.assert_allclose(agg.se_by_event, 0.0)


def test_aggte_didml_adds_the_cohort_share_term(two_cohort_result):
    agg = aggte_didml(two_cohort_result)
    share_se = np.sqrt(0.75 * 0.25 / 4)

    np.testing.assert_allclose(agg.se_by_event, [0.0, 19 * share_se, 28 * share_se, 0.0, 0.0], atol=1e-12)
    assert agg.overall_se == pytest.approx(11.75 * share_se)


def test_aggte_didml_weights_the_cohort_shares(two_cohort_weighted):
    agg = aggte_didml(two_cohort_weighted)
    share_se = np.sqrt((3 * (1 / 3) ** 2 + 1) / 16)

    np.testing.assert_allclose(agg.att_by_event, [10.0, 10.5, 16.0, 3.0, 4.0])
    np.testing.assert_allclose(agg.se_by_event, [0.0, 19 * share_se, 28 * share_se, 0.0, 0.0], atol=1e-12)


def test_aggte_didml_aggregates_each_bootstrap_draw_with_its_shares(two_cohort_draws):
    agg = aggte_didml(two_cohort_draws)
    theta = np.array(
        [
            [10.0, 5.75, 9.0, 3.0, 4.0],
            [10.0, 12.0, 16.0, 3.0, 4.0],
            [12.0, 5.75, 11.5, 3.0, 5.0],
        ]
    )

    np.testing.assert_allclose(agg.att_by_event, [10.0, 5.75, 9.0, 3.0, 4.0])
    np.testing.assert_allclose(agg.se_by_event, theta.std(axis=0, ddof=1))
    assert agg.overall_se == pytest.approx(theta[:, 1:].mean(axis=1).std(ddof=1))


def test_aggte_didml_drops_a_missing_cohort_from_each_bootstrap_draw(two_cohort_draws_missing_a_cohort):
    theta = np.array(
        [
            [10.0, 5.75, 9.0, 3.0, 4.0],
            [10.0, 12.0, 16.0, 3.0, 4.0],
            [12.0, 5.75, 11.5, 3.0, 5.0],
            [11.0, 21.0, 31.0, np.nan, np.nan],
        ]
    )
    expected_se = [np.std(column[np.isfinite(column)], ddof=1) for column in theta.T]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agg = aggte_didml(two_cohort_draws_missing_a_cohort)

    assert [str(w.message) for w in caught] == []
    np.testing.assert_allclose(agg.se_by_event, expected_se, rtol=1e-12)
    assert agg.overall_se == pytest.approx(theta[:3, 1:].mean(axis=1).std(ddof=1), rel=1e-12)


def test_aggte_didml_keeps_the_aggte_layout_with_empty_call_info(two_cohort_result):
    agg = aggte_didml(two_cohort_result)

    assert agg.call_info == {}
    assert agg.balanced_event_threshold is None


def test_summaries_return_the_printout_without_printing(two_cohort_result, capsys):
    agg = aggte_didml(two_cohort_result)

    assert summary_didml(two_cohort_result) == str(two_cohort_result)
    assert summary_didml_agg(agg) == str(agg)
    assert capsys.readouterr().out == ""


def test_het_prep_keeps_treated_rows_with_period_and_event_dummies(het_panel, het_cates):
    merged = het_prep(het_panel, het_cates, idname="id", tname="time", gname="group").sort(["id", "time"])

    assert merged["id"].to_list() == [1, 1, 1, 2, 2, 2]
    assert merged["period"].to_list() == [2004, 2005, 2006, 2004, 2005, 2006]
    assert merged["dynamic_cate"].to_list() == [0.0, 0.3, 0.4, 0.1, 0.5, 0.6]
    assert merged["e_1"].to_list() == [1, 0, 0, 1, 0, 0]
    assert merged["e1"].to_list() == [0, 0, 1, 0, 0, 1]


def test_blp_eventtimes_reads_decimal_covariates_from_het_prep_as_doubles(mpdta_data, mpdta_analytic_fit):
    cates = dynamic_cates(mpdta_analytic_fit)
    decimal = mpdta_data.with_columns(pl.col("lpop").cast(pl.Decimal(18, 6)))
    floats = decimal.with_columns(pl.col("lpop").cast(pl.Float64))
    keys = {"idname": "countyreal", "tname": "year", "gname": "first.treat"}

    result = blp_eventtimes(het_prep(decimal, cates, **keys), n_periods=1, rhs_formula="lpop")
    expected = blp_eventtimes(het_prep(floats, cates, **keys), n_periods=1, rhs_formula="lpop")

    assert list(result.coefs) == list(expected.coefs) == ["Intercept", "lpop"]
    np.testing.assert_allclose(
        np.vstack(list(result.coefs.values())), np.vstack(list(expected.coefs.values())), rtol=1e-12
    )
    np.testing.assert_allclose(np.vstack(list(result.ses.values())), np.vstack(list(expected.ses.values())), rtol=1e-12)


@pytest.mark.parametrize("event_time", [0, 1])
def test_clan_ttest_runs_welch_tests_on_the_quantile_groups(het_data, event_time):
    result = clan_ttest(het_data, affected=["x"], threshold=0.2)[event_time]
    x = het_data.filter(pl.col("event_time") == event_time)["x"].to_numpy()

    np.testing.assert_allclose(result.diffs, [x[16:].mean() - x[:4].mean()])
    np.testing.assert_allclose(result.pvalues, [stats.ttest_ind(x[16:], x[:4], equal_var=False).pvalue])
    assert result.test_type == "t"


@pytest.mark.parametrize("event_time", [0, 1])
def test_clan_glhtest_pools_group_variances_and_ignores_alpha(het_data, event_time):
    result = clan_glhtest(het_data, affected=["x"], threshold=0.2, alpha=0.01)[event_time]
    other = clan_glhtest(het_data, affected=["x"], threshold=0.2, alpha=0.5)[event_time]
    x = het_data.filter(pl.col("event_time") == event_time)["x"].to_numpy()

    np.testing.assert_allclose(result.high_means, [x[16:].mean()])
    np.testing.assert_allclose(result.low_means, [x[:4].mean()])
    np.testing.assert_allclose(result.pvalues, [stats.ttest_ind(x[16:], x[:4]).pvalue])
    np.testing.assert_array_equal(result.pvalues, other.pvalues)
    assert result.test_type == "glh"


@pytest.mark.parametrize("clan", [clan_ttest, clan_glhtest])
def test_clan_tests_skip_pre_treatment_and_all_null_event_times(het_data, het_data_padded, clan):
    padded = clan(het_data_padded, affected=["x"], threshold=0.2)
    plain = clan(het_data, affected=["x"], threshold=0.2)

    assert len(padded) == 2
    np.testing.assert_array_equal([r.pvalues for r in padded], [r.pvalues for r in plain])


def test_fit_delta_returns_the_out_of_fold_lasso_predictions(stacked_cell):
    X = stacked_cell["X"]
    response = (stacked_cell["post_indicator"] - 0.5) * (stacked_cell["cohort_indicator"] - 0.5)
    fit = fit_delta(X, response, model="glm", k_folds=3, random_state=0)
    standardized = (X - X.mean(axis=0)) / X.std(axis=0)
    expected = cv_lasso_with_oof(standardized, response, k_folds=3, random_state=0, standardize=False)

    assert set(fit) == {"delta_hat", "best_l1_ratio"}
    np.testing.assert_allclose(fit["delta_hat"], expected["oof_predictions"], rtol=1e-10)


def test_fit_rlearner_no_longer_reports_a_penalty_factor(stacked_cell):
    fit = fit_rlearner(
        stacked_cell["X"], stacked_cell["Y"], stacked_cell["cohort_indicator"], k_folds=3, random_state=0
    )

    assert set(fit) == {"tau_hat", "p_hat", "m_hat", "tau_coef"}


def test_amle_weights_with_equal_sample_weights_match_the_default(stacked_cell):
    args = (stacked_cell["X"], stacked_cell["post_indicator"], stacked_cell["cohort_indicator"])
    plain = amle_weights(*args)
    weighted = amle_weights(*args, sample_weights=np.full(len(stacked_cell["Y"]), 3.0))

    np.testing.assert_allclose(weighted, plain, rtol=1e-6, atol=1e-8)


def test_amle_weights_target_the_weighted_covariate_mean_of_the_treated_cell(stacked_cell):
    args = (stacked_cell["X"], stacked_cell["post_indicator"], stacked_cell["cohort_indicator"])
    sample_weights = np.where(stacked_cell["X"][:, 0] > 0, 4.0, 1.0)
    plain = amle_weights(*args)
    weighted = amle_weights(*args, sample_weights=sample_weights)
    treated = stacked_cell["post_indicator"] * stacked_cell["cohort_indicator"]
    target = np.average(stacked_cell["X"], axis=0, weights=sample_weights * treated)
    n = len(stacked_cell["Y"])

    weighted_gap = np.abs(stacked_cell["X"].T @ (treated * weighted) / n - target)
    plain_gap = np.abs(stacked_cell["X"].T @ (treated * plain) / n - target)
    assert weighted_gap.max() == pytest.approx(0.0545, abs=1e-3)
    assert plain_gap.max() == pytest.approx(0.4348, abs=1e-3)


def test_amle_weights_balance_the_intercepts(stacked_cell):
    post, cohort = stacked_cell["post_indicator"], stacked_cell["cohort_indicator"]
    gamma = amle_weights(stacked_cell["X"], post, cohort)
    n = len(stacked_cell["Y"])

    np.testing.assert_allclose(
        [gamma.sum() / n, (post * gamma).sum() / n, (cohort * gamma).sum() / n, (post * cohort * gamma).sum() / n],
        [0.0, 0.0, 0.0, 1.0],
        atol=1e-7,
    )


def test_lnw_did_averages_the_catt_over_the_treated_cell(stacked_cell):
    post, cohort = stacked_cell["post_indicator"], stacked_cell["cohort_indicator"]
    gamma = amle_weights(stacked_cell["X"], post, cohort)
    fit = lnw_did(**stacked_cell, gamma=gamma, k_folds=3, random_state=0)
    treated = post * cohort == 1

    expected = fit["tau_hat"][treated].mean() + np.mean(gamma * (stacked_cell["Y"] - fit["y_hat"]))
    assert fit["TAU_hat"] == pytest.approx(expected, rel=1e-12)
    assert abs(fit["TAU_hat"] - np.mean(fit["score"])) > 1e-6


def test_lnw_did_rejects_a_cell_without_treated_rows(stacked_cell):
    cohort_before_treatment = stacked_cell["cohort_indicator"] * (1 - stacked_cell["post_indicator"])
    no_treated = {**stacked_cell, "cohort_indicator": cohort_before_treatment}

    with pytest.raises(ValueError, match="never 1 on the same row"):
        lnw_did(**no_treated, k_folds=3, random_state=0)


def test_lnw_did_returns_the_linear_catt_coefficients(stacked_cell):
    fit = lnw_did(**stacked_cell, k_folds=3, random_state=0)
    design = np.column_stack([np.ones(len(stacked_cell["Y"])), stacked_cell["X"]])

    assert fit["tau_coef"].shape == (3,)
    np.testing.assert_allclose(fit["tau_hat"], design @ fit["tau_coef"])


def test_lnw_did_constant_effect_has_no_fitted_values(stacked_cell):
    fit = lnw_did(**stacked_cell, constant_eff="constant", k_folds=3, random_state=0)

    assert fit["y_hat"] is None
    assert fit["tau_hat"] is None
    assert "tau_coef" not in fit
