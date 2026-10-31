"""Tests for Aggregated Treatment Effect."""

import warnings
from importlib import import_module

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid import compute_aggte, mp


@pytest.mark.parametrize("data_fixture", ["weighted_cohorts_mp_result", "mpdta_mp_result"])
def test_simple_aggregation_weights_post_treatment_cells_by_cohort_share(request, data_fixture):
    fit = request.getfixturevalue(data_fixture)
    share = np.array([np.mean(fit.weights_ind * (g == fit.G)) for g in fit.groups])
    post = fit.groups <= fit.times

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_aggte(fit, aggregation_type="simple")

    assert result.aggregation_type == "simple"
    np.testing.assert_allclose(result.overall_att, share[post] @ fit.att_gt[post] / share[post].sum(), rtol=1e-12)
    assert [str(w.message) for w in caught] == []


@pytest.mark.parametrize("data_fixture", ["weighted_cohorts_mp_result", "mpdta_mp_result"])
def test_dynamic_aggregation_weights_each_event_time_by_cohort_share(request, data_fixture):
    fit = request.getfixturevalue(data_fixture)
    share = np.array([np.mean(fit.weights_ind * (g == fit.G)) for g in fit.groups])
    cells = [fit.times - fit.groups == e for e in range(-3, 4)]
    expected = np.array([share[cell] @ fit.att_gt[cell] / share[cell].sum() for cell in cells])

    result = compute_aggte(fit, aggregation_type="dynamic")

    assert result.aggregation_type == "dynamic"
    np.testing.assert_array_equal(result.event_times, [-3, -2, -1, 0, 1, 2, 3])
    np.testing.assert_allclose(result.att_by_event, expected, rtol=1e-12)
    np.testing.assert_allclose(result.overall_att, expected[3:].mean(), rtol=1e-12)


@pytest.mark.parametrize("data_fixture", ["weighted_cohorts_mp_result", "mpdta_mp_result"])
def test_group_aggregation_weights_cohort_effects_by_cohort_share(request, data_fixture):
    fit = request.getfixturevalue(data_fixture)
    share = np.array([np.mean(fit.weights_ind * (g == fit.G)) for g in (2004, 2006, 2007)])
    by_cohort = np.array([fit.att_gt[(fit.groups == g) & (fit.times >= g)].mean() for g in (2004, 2006, 2007)])

    result = compute_aggte(fit, aggregation_type="group")

    assert result.aggregation_type == "group"
    np.testing.assert_array_equal(result.event_times, [2004, 2006, 2007])
    np.testing.assert_allclose(result.att_by_event, by_cohort, rtol=1e-12)
    np.testing.assert_allclose(result.overall_att, share @ by_cohort / share.sum(), rtol=1e-12)


@pytest.mark.parametrize("data_fixture", ["weighted_cohorts_mp_result", "mpdta_mp_result"])
def test_calendar_aggregation_weights_treated_cohorts_by_share_in_each_period(request, data_fixture):
    fit = request.getfixturevalue(data_fixture)
    share = np.array([np.mean(fit.weights_ind * (g == fit.G)) for g in fit.groups])
    cells = [(fit.groups <= fit.times) & (fit.times == t) for t in (2004, 2005, 2006, 2007)]
    expected = np.array([share[cell] @ fit.att_gt[cell] / share[cell].sum() for cell in cells])

    result = compute_aggte(fit, aggregation_type="calendar")

    assert result.aggregation_type == "calendar"
    np.testing.assert_array_equal(result.event_times, [2004, 2005, 2006, 2007])
    np.testing.assert_allclose(result.att_by_event, expected, rtol=1e-12)
    np.testing.assert_allclose(result.overall_att, expected.mean(), rtol=1e-12)


@pytest.mark.parametrize("data_fixture", ["weighted_cohorts_mp_result", "mpdta_mp_result"])
def test_simple_influence_function_matches_weight_derivative(
    request, weight_derivative, reweighted_mp_result, data_fixture
):
    fit = request.getfixturevalue(data_fixture)

    result = compute_aggte(fit, aggregation_type="simple")
    expected = weight_derivative(
        lambda multipliers: (
            compute_aggte(reweighted_mp_result(fit, multipliers), aggregation_type="simple").overall_att
        ),
        fit.n_units,
        range(fit.n_units),
    )

    np.testing.assert_allclose(result.influence_func, expected, rtol=1e-6, atol=1e-12)
    np.testing.assert_array_equal(result.influence_func_overall, result.influence_func)


@pytest.mark.parametrize("data_fixture", ["weighted_cohorts_mp_result", "mpdta_mp_result"])
@pytest.mark.parametrize("aggregation_type", ["dynamic", "group", "calendar"])
def test_disaggregated_influence_functions_match_weight_derivatives(
    request, weight_derivative, reweighted_mp_result, data_fixture, aggregation_type
):
    fit = request.getfixturevalue(data_fixture)
    rows = np.concatenate([np.flatnonzero(g == fit.G)[:2] for g in (2004, 2006, 2007, np.inf)])

    result = compute_aggte(fit, aggregation_type=aggregation_type)
    by_event = weight_derivative(
        lambda multipliers: (
            compute_aggte(reweighted_mp_result(fit, multipliers), aggregation_type=aggregation_type).att_by_event
        ),
        fit.n_units,
        rows,
    )
    overall = weight_derivative(
        lambda multipliers: (
            compute_aggte(reweighted_mp_result(fit, multipliers), aggregation_type=aggregation_type).overall_att
        ),
        fit.n_units,
        rows,
    )

    assert result.influence_func.shape == (fit.n_units, len(result.event_times))
    np.testing.assert_allclose(result.influence_func[rows], by_event, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(result.influence_func_overall[rows], overall, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("data_fixture", ["weighted_cohorts_mp_result", "mpdta_mp_result"])
def test_simple_standard_error_is_the_root_mean_square_of_the_influence_function(request, data_fixture):
    fit = request.getfixturevalue(data_fixture)

    result = compute_aggte(fit, aggregation_type="simple")
    second_moment = np.mean(result.influence_func**2)

    np.testing.assert_allclose(result.overall_se, np.sqrt(second_moment / fit.n_units), rtol=1e-12)
    np.testing.assert_allclose(result.vcov_analytical, [[second_moment]], rtol=1e-12)


@pytest.mark.parametrize("data_fixture", ["weighted_cohorts_mp_result", "mpdta_mp_result"])
@pytest.mark.parametrize("aggregation_type", ["dynamic", "group", "calendar"])
def test_standard_errors_are_root_mean_squares_of_the_influence_functions(request, data_fixture, aggregation_type):
    fit = request.getfixturevalue(data_fixture)

    result = compute_aggte(fit, aggregation_type=aggregation_type)
    influence = result.influence_func

    np.testing.assert_allclose(result.se_by_event, np.sqrt(np.mean(influence**2, axis=0) / fit.n_units), rtol=1e-12)
    np.testing.assert_allclose(
        result.overall_se, np.sqrt(np.mean(result.influence_func_overall**2) / fit.n_units), rtol=1e-12
    )
    np.testing.assert_allclose(result.vcov_analytical, influence.T @ influence / fit.n_units, rtol=1e-12)


@pytest.mark.parametrize("aggregation_type", ["dynamic", "group", "calendar"])
def test_aggregated_effects_without_variation_report_nan_standard_errors(weighted_cohorts_mp_result, aggregation_type):
    fit = weighted_cohorts_mp_result._replace(att_gt=np.full(12, 0.5), influence_func=np.zeros((200, 12)))

    result = compute_aggte(fit, aggregation_type=aggregation_type)

    np.testing.assert_allclose(result.overall_att, 0.5, rtol=1e-12)
    assert np.all(np.isnan(result.se_by_event))
    assert np.isnan(result.overall_se)


def test_bootstrap_standard_error_of_zero_is_reported_as_nan(
    monkeypatch, draws_from_weights, offsetting_spike_weights, inf_func_with_zero_scale_column
):
    signs = offsetting_spike_weights(200, 0, 1)
    monkeypatch.setattr(import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", draws_from_weights(signs))
    fit = mp(
        groups=np.array([2004, 2006, 2007]),
        times=np.array([2004, 2006, 2007]),
        att_gt=np.full(3, 0.5),
        vcov_analytical=np.eye(3),
        se_gt=np.ones(3),
        critical_value=1.96,
        influence_func=inf_func_with_zero_scale_column,
        n_units=200,
        G=np.repeat([2004.0, 2006.0, 2007.0, np.inf], 50),
        weights_ind=np.ones(200),
    )

    result = compute_aggte(fit, aggregation_type="group", bootstrap=True, biters=999, random_state=7)
    q75, q25 = np.percentile(signs @ inf_func_with_zero_scale_column / 200, [75, 25], axis=0, method="inverted_cdf")

    assert q75[1] - q25[1] == 0
    assert np.isnan(result.se_by_event[1])
    np.testing.assert_allclose(result.se_by_event[[0, 2]], (q75 - q25)[[0, 2]] / 1.3489795, rtol=1e-12)


def test_unit_level_groups_without_weights_weigh_every_unit_once(weighted_cohorts_mp_result):
    fit = weighted_cohorts_mp_result
    share = np.array([np.mean(g == fit.G) for g in fit.groups])
    post = fit.groups <= fit.times

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_aggte(fit._replace(weights_ind=None), aggregation_type="simple")
    equal_weights = compute_aggte(fit._replace(weights_ind=np.ones(fit.n_units)), aggregation_type="simple")

    np.testing.assert_allclose(result.overall_att, share[post] @ fit.att_gt[post] / share[post].sum(), rtol=1e-12)
    np.testing.assert_array_equal(result.influence_func, equal_weights.influence_func)
    assert [str(w.message) for w in caught] == []


def test_calendar_aggregation_leaves_out_periods_without_treated_effects(synthetic_mp_result):
    failed = (synthetic_mp_result.groups == 2004) & (synthetic_mp_result.times >= 2004)
    with_failed_cells = synthetic_mp_result._replace(att_gt=np.where(failed, np.nan, synthetic_mp_result.att_gt))

    result = compute_aggte(with_failed_cells, aggregation_type="calendar", na_rm=True)

    np.testing.assert_array_equal(result.event_times, [2006, 2007])
    np.testing.assert_allclose(result.att_by_event, [0.05, 0.055], rtol=1e-12)
    np.testing.assert_allclose(result.overall_att, 0.0525, rtol=1e-12)


def test_group_aggregation_leaves_out_cohorts_without_post_treatment_effects(synthetic_mp_result):
    failed = (synthetic_mp_result.groups == 2004) & (synthetic_mp_result.times >= 2004)
    dropped = synthetic_mp_result._replace(
        groups=synthetic_mp_result.groups[~failed],
        times=synthetic_mp_result.times[~failed],
        att_gt=synthetic_mp_result.att_gt[~failed],
        influence_func=synthetic_mp_result.influence_func[:, ~failed],
    )
    with_nan = synthetic_mp_result._replace(att_gt=np.where(failed, np.nan, synthetic_mp_result.att_gt))

    from_dropped = compute_aggte(dropped, aggregation_type="group")
    from_nan = compute_aggte(with_nan, aggregation_type="group", na_rm=True)

    np.testing.assert_array_equal(from_dropped.event_times, [2006, 2007])
    np.testing.assert_array_equal(from_nan.event_times, [2006, 2007])
    np.testing.assert_allclose(from_dropped.att_by_event, [0.055, 0.05], rtol=1e-12)
    np.testing.assert_allclose(from_nan.att_by_event, [0.055, 0.05], rtol=1e-12)
    np.testing.assert_allclose(from_dropped.overall_att, 0.0525, rtol=1e-12)
    np.testing.assert_allclose(from_nan.overall_att, 0.0525, rtol=1e-12)
    np.testing.assert_allclose(from_dropped.se_by_event, from_nan.se_by_event, rtol=1e-12)
    np.testing.assert_allclose(from_dropped.overall_se, from_nan.overall_se, rtol=1e-12)


@pytest.mark.parametrize("aggregation_type", ["simple", "dynamic", "group", "calendar"])
def test_aggregation_with_only_pre_treatment_cells_left_reports_nan(synthetic_mp_result, aggregation_type):
    post_treatment = synthetic_mp_result.groups <= synthetic_mp_result.times
    with_failed_cells = synthetic_mp_result._replace(
        att_gt=np.where(post_treatment, np.nan, synthetic_mp_result.att_gt)
    )

    with pytest.warns(
        UserWarning,
        match="^The overall effect is NaN because no post-treatment group-time effect is left to average\\. ",
    ):
        result = compute_aggte(with_failed_cells, aggregation_type=aggregation_type, na_rm=True)

    assert np.isnan(result.overall_att)
    assert np.isnan(result.overall_se)


@pytest.mark.parametrize(("min_e", "max_e"), [(-2, 2), (-1, 1), (0, 3)])
def test_min_e_and_max_e_keep_the_event_times_of_the_unrestricted_event_study(weighted_cohorts_mp_result, min_e, max_e):
    full = compute_aggte(weighted_cohorts_mp_result, aggregation_type="dynamic")

    result = compute_aggte(weighted_cohorts_mp_result, aggregation_type="dynamic", min_e=min_e, max_e=max_e)
    kept = (full.event_times >= min_e) & (full.event_times <= max_e)

    np.testing.assert_array_equal(result.event_times, np.arange(min_e, max_e + 1))
    np.testing.assert_allclose(result.att_by_event, full.att_by_event[kept], rtol=1e-12)
    np.testing.assert_allclose(result.se_by_event, full.se_by_event[kept], rtol=1e-12)
    np.testing.assert_allclose(result.overall_att, full.att_by_event[kept & (full.event_times >= 0)].mean(), rtol=1e-12)
    assert (result.min_event_time, result.max_event_time) == (min_e, max_e)


@pytest.mark.parametrize("max_e", [0, 1])
def test_max_e_limits_the_cells_of_the_simple_and_group_aggregations(weighted_cohorts_mp_result, max_e):
    fit = weighted_cohorts_mp_result
    share = np.array([np.mean(fit.weights_ind * (g == fit.G)) for g in fit.groups])
    window = (fit.groups <= fit.times) & (fit.times <= fit.groups + max_e)
    cohort_share = np.array([np.mean(fit.weights_ind * (g == fit.G)) for g in (2004, 2006, 2007)])
    by_cohort = np.array([fit.att_gt[window & (fit.groups == g)].mean() for g in (2004, 2006, 2007)])

    simple = compute_aggte(fit, aggregation_type="simple", max_e=max_e)
    group = compute_aggte(fit, aggregation_type="group", max_e=max_e)

    np.testing.assert_allclose(simple.overall_att, share[window] @ fit.att_gt[window] / share[window].sum(), rtol=1e-12)
    np.testing.assert_array_equal(group.event_times, [2004, 2006, 2007])
    np.testing.assert_allclose(group.att_by_event, by_cohort, rtol=1e-12)
    np.testing.assert_allclose(group.overall_att, cohort_share @ by_cohort / cohort_share.sum(), rtol=1e-12)


def test_balance_e_keeps_only_cohorts_observed_through_that_event_time(weighted_cohorts_mp_result):
    fit = weighted_cohorts_mp_result
    share = np.array([np.mean(fit.weights_ind * (g == fit.G)) for g in fit.groups])
    cells = [(fit.groups < 2007) & (fit.times - fit.groups == e) for e in (-2, -1, 0, 1)]
    expected = np.array([share[cell] @ fit.att_gt[cell] / share[cell].sum() for cell in cells])

    result = compute_aggte(fit, aggregation_type="dynamic", balance_e=1)

    np.testing.assert_array_equal(result.event_times, [-2, -1, 0, 1])
    np.testing.assert_allclose(result.att_by_event, expected, rtol=1e-12)
    np.testing.assert_allclose(result.overall_att, expected[2:].mean(), rtol=1e-12)
    assert result.balanced_event_threshold == 1


def test_na_rm_drops_nan_cells_and_their_influence_functions(weighted_cohorts_mp_result):
    fit = weighted_cohorts_mp_result
    failed = np.isin(np.arange(12), [5, 7])
    with_nan = fit._replace(att_gt=np.where(failed, np.nan, fit.att_gt))
    dropped = fit._replace(
        groups=fit.groups[~failed],
        times=fit.times[~failed],
        att_gt=fit.att_gt[~failed],
        influence_func=fit.influence_func[:, ~failed],
    )
    share = np.array([np.mean(fit.weights_ind * (g == fit.G)) for g in fit.groups])
    post = (fit.groups <= fit.times) & ~failed

    with pytest.raises(
        ValueError, match="^Missing values at att_gt found\\. If you want to remove these, set na_rm=True in aggte\\.$"
    ):
        compute_aggte(with_nan, aggregation_type="simple")
    result = compute_aggte(with_nan, aggregation_type="simple", na_rm=True)
    from_dropped = compute_aggte(dropped, aggregation_type="simple")

    np.testing.assert_allclose(result.overall_att, share[post] @ fit.att_gt[post] / share[post].sum(), rtol=1e-12)
    np.testing.assert_array_equal(result.influence_func, from_dropped.influence_func)
    assert result.overall_se == from_dropped.overall_se


@pytest.mark.parametrize(("bootstrap", "confidence_band"), [(True, True), (True, False), (False, True), (False, False)])
def test_bootstrap_and_confidence_band_choose_the_standard_errors_and_critical_values(
    monkeypatch, draws_from_weights, weighted_cohorts_mp_result, bootstrap, confidence_band
):
    fit = weighted_cohorts_mp_result
    signs = np.where(np.random.default_rng(7).random((199, fit.n_units)) < 0.5, 1.0, -1.0)
    monkeypatch.setattr(import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", draws_from_weights(signs))
    spec = {"aggregation_type": "group", "biters": 199, "alpha": 0.1, "random_state": 3}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_aggte(fit, bootstrap=bootstrap, confidence_band=confidence_band, **spec)
    influence = np.column_stack([result.influence_func, result.influence_func_overall])
    draws = signs @ influence / fit.n_units
    q75, q25 = np.percentile(draws, [75, 25], axis=0, method="inverted_cdf")
    scale = (q75 - q25) / 1.3489795
    band = np.percentile(np.max(np.abs(draws[:, :-1]) / scale[:-1], axis=1), 90, method="inverted_cdf")
    expected_se = scale if bootstrap else np.sqrt(np.mean(influence**2, axis=0) / fit.n_units)
    expected_messages = ["Used bootstrap procedure to compute simultaneous confidence band"]

    np.testing.assert_allclose(result.se_by_event, expected_se[:-1], rtol=1e-12)
    np.testing.assert_allclose(result.overall_se, expected_se[-1], rtol=1e-12)
    np.testing.assert_allclose(
        result.critical_values, np.full(3, band if confidence_band else 1.6448536269514722), rtol=1e-12
    )
    assert [str(w.message) for w in caught] == (expected_messages if confidence_band and not bootstrap else [])
    assert (result.estimation_params["bootstrap"], result.estimation_params["uniform_bands"]) == (
        bootstrap,
        confidence_band,
    )


@pytest.mark.parametrize(
    ("alpha", "critical_value"), [(0.01, 2.5758293035489004), (0.05, 1.959963984540054), (0.1, 1.6448536269514722)]
)
def test_alpha_sets_the_pointwise_critical_values(weighted_cohorts_mp_result, alpha, critical_value):
    result = compute_aggte(weighted_cohorts_mp_result, aggregation_type="dynamic", alpha=alpha)

    np.testing.assert_allclose(result.critical_values, np.full(7, critical_value), rtol=1e-12)
    assert result.estimation_params.get("alpha") == alpha


def test_alpha_left_out_takes_the_critical_values_at_the_level_of_the_fit(weighted_cohorts_mp_result):
    fit = weighted_cohorts_mp_result._replace(alpha=0.1)

    result = compute_aggte(fit, aggregation_type="dynamic")

    np.testing.assert_allclose(result.critical_values, np.full(7, 1.6448536269514722), rtol=1e-12)


def test_compute_aggte_passes_biters_to_the_bootstrap(weighted_cohorts_mp_result):
    spec = {"aggregation_type": "group", "bootstrap": True, "random_state": 4}

    few = compute_aggte(weighted_cohorts_mp_result, biters=77, **spec)
    many = compute_aggte(weighted_cohorts_mp_result, biters=999, **spec)

    assert few.estimation_params.get("biters") == 77
    assert np.max(np.abs(few.se_by_event - many.se_by_event)) > 1e-4
    assert abs(few.overall_se - many.overall_se) > 1e-4


def test_compute_aggte_rejects_the_removed_bootstrap_iterations_argument(synthetic_mp_result):
    with pytest.raises(TypeError, match="unexpected keyword argument 'bootstrap_iterations'"):
        compute_aggte(synthetic_mp_result, aggregation_type="simple", bootstrap_iterations=99)


def test_compute_aggte_rejects_the_removed_dropna_argument(synthetic_mp_result):
    with pytest.raises(TypeError, match="unexpected keyword argument 'dropna'"):
        compute_aggte(synthetic_mp_result, aggregation_type="simple", dropna=True)


def test_invalid_aggregation_type(synthetic_mp_result):
    with pytest.raises(
        ValueError,
        match="^`aggregation_type` must be one of \\['simple', 'dynamic', 'group', 'calendar'\\], got invalid$",
    ):
        compute_aggte(synthetic_mp_result, aggregation_type="invalid")


def test_single_group_edge_case():
    groups = np.array([2004, 2004, 2004])
    times = np.array([2003, 2004, 2005])
    att_gt = np.array([0.0, 0.05, 0.06])

    inf_func = np.random.default_rng(0).normal(0, 0.01, size=(100, 3))
    vcov = np.eye(3) * 0.01
    se_gt = np.array([0.01, 0.01, 0.01])

    mp_result = mp(
        groups=groups,
        times=times,
        att_gt=att_gt,
        vcov_analytical=vcov,
        se_gt=se_gt,
        critical_value=1.96,
        influence_func=inf_func,
        n_units=100,
    )

    with pytest.warns(
        UserWarning, match="^Unit-level data \\(unit_level_groups and unit_level_weights\\) not provided"
    ):
        result = compute_aggte(mp_result, aggregation_type="group")

    np.testing.assert_array_equal(result.event_times, [2004])
    np.testing.assert_allclose(result.att_by_event, [0.055], rtol=1e-12)


def test_no_post_treatment_edge_case():
    groups = np.array([2010, 2010, 2010])
    times = np.array([2003, 2004, 2005])
    att_gt = np.array([0.0, 0.0, 0.0])

    inf_func = np.zeros((100, 3))
    vcov = np.eye(3) * 0.01
    se_gt = np.array([0.01, 0.01, 0.01])

    mp_result = mp(
        groups=groups,
        times=times,
        att_gt=att_gt,
        vcov_analytical=vcov,
        se_gt=se_gt,
        critical_value=1.96,
        influence_func=inf_func,
        n_units=100,
        G=np.r_[np.full(40, 2010), np.zeros(60)],
        weights_ind=np.ones(100),
    )

    with pytest.warns(
        UserWarning,
        match="^The overall effect is NaN because no post-treatment group-time effect is left to average\\. ",
    ):
        result = compute_aggte(mp_result, aggregation_type="simple")

    assert np.isnan(result.overall_att)
    assert np.isnan(result.overall_se)


def test_clustered_standard_errors_sum_the_influence_functions_within_clusters(weighted_cohorts_mp_result):
    fit = weighted_cohorts_mp_result
    cluster = np.arange(fit.n_units) % 20
    clustered_fit = fit._replace(
        estimation_params={**fit.estimation_params, "clustervars": ["state"], "cluster": cluster}
    )

    clustered = compute_aggte(clustered_fit, aggregation_type="group")
    unclustered = compute_aggte(fit, aggregation_type="group")
    influence = np.column_stack([clustered.influence_func, clustered.influence_func_overall])
    sums = np.column_stack([np.bincount(cluster, weights=column) for column in influence.T])
    expected = np.sqrt(np.sum(sums**2, axis=0)) / fit.n_units

    np.testing.assert_allclose(clustered.se_by_event, expected[:-1], rtol=1e-12)
    np.testing.assert_allclose(clustered.overall_se, expected[-1], rtol=1e-12)
    np.testing.assert_allclose(clustered.vcov_analytical, sums[:, :-1].T @ sums[:, :-1] / fit.n_units, rtol=1e-12)
    np.testing.assert_array_equal(clustered.influence_func, unclustered.influence_func)
    assert clustered.overall_att == unclustered.overall_att
    assert clustered.overall_se > 1.5 * unclustered.overall_se
    assert clustered.estimation_params["clustervars"] == ["state"]
    np.testing.assert_array_equal(clustered.estimation_params["cluster"], cluster)


def test_hand_built_result_resolves_clusters_from_its_data(mpdta_mp_result):
    draws = {"aggregation_type": "group", "bootstrap": True, "biters": 99, "random_state": 42}

    by_unit = compute_aggte(mpdta_mp_result, clustervars=["countyreal"], **draws)
    unclustered = compute_aggte(mpdta_mp_result, clustervars=[], **draws)

    assert by_unit.estimation_params["clustervars"] == []
    assert by_unit.estimation_params["cluster"] is None
    assert by_unit.overall_se == unclustered.overall_se
    np.testing.assert_array_equal(by_unit.se_by_event, unclustered.se_by_event)
    with pytest.raises(
        ValueError,
        match="^The att_gt fit did not cluster on 'state'\\. Since the aggregation reuses the clusters of the att_gt "
        "fit, pass the cluster variable to att_gt instead\\. To aggregate this fit, pass clustervars=\\['cluster'\\] "
        "to reuse its clusters, or an empty list for unit-level inference\\.$",
    ):
        compute_aggte(mpdta_mp_result, clustervars=["state"], **draws)


def test_hand_built_result_clusters_on_the_cluster_stored_with_its_data(mpdta_mp_result):
    stored = mpdta_mp_result.estimation_params["data"].cluster
    codes = np.unique(stored, return_inverse=True)[1]

    result = compute_aggte(mpdta_mp_result, aggregation_type="group", clustervars=["cluster"])
    unclustered = compute_aggte(mpdta_mp_result, aggregation_type="group")
    influence = np.column_stack([result.influence_func, result.influence_func_overall])
    sums = np.column_stack([np.bincount(codes, weights=column) for column in influence.T])
    expected = np.sqrt(np.sum(sums**2, axis=0)) / mpdta_mp_result.n_units

    assert result.estimation_params["clustervars"] == ["cluster"]
    np.testing.assert_array_equal(result.estimation_params["cluster"], stored)
    np.testing.assert_allclose(result.se_by_event, expected[:-1], rtol=1e-12)
    np.testing.assert_allclose(result.overall_se, expected[-1], rtol=1e-12)
    assert unclustered.estimation_params["cluster"] is None
    assert len(np.unique(stored)) < mpdta_mp_result.n_units
