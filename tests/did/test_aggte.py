"""Tests for aggregate treatment effects."""

import re
import warnings
from importlib import import_module

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid import aggte, att_gt


def test_aggte_simple_weighs_each_post_treatment_cell_by_the_size_of_its_cohort(mp_result, mpdta_cohort_sizes):
    post = mp_result.groups <= mp_result.times
    sizes = [mpdta_cohort_sizes[g] for g in mp_result.groups[post]]

    result = aggte(mp_result, type="simple", cband=False)

    assert result.aggregation_type == "simple"
    assert result.event_times is None
    assert result.att_by_event is None
    assert result.se_by_event is None
    np.testing.assert_allclose(result.overall_att, np.average(mp_result.att_gt[post], weights=sizes), rtol=1e-12)


def test_aggte_group_averages_each_cohort_and_weighs_the_cohorts_by_size(mp_result, mpdta_cohort_sizes):
    cohorts = [2004, 2006, 2007]
    expected = [mp_result.att_gt[(mp_result.groups == g) & (mp_result.times >= g)].mean() for g in cohorts]

    result = aggte(mp_result, type="group", cband=False)

    assert result.aggregation_type == "group"
    np.testing.assert_array_equal(result.event_times, cohorts)
    np.testing.assert_allclose(result.att_by_event, expected, rtol=1e-12)
    np.testing.assert_allclose(
        result.overall_att, np.average(expected, weights=[mpdta_cohort_sizes[g] for g in cohorts]), rtol=1e-12
    )


def test_aggte_dynamic_weighs_the_cohorts_at_each_event_time_by_size(mp_result, mpdta_cohort_sizes):
    event = mp_result.times - mp_result.groups
    sizes = np.array([mpdta_cohort_sizes[g] for g in mp_result.groups])
    expected = [np.average(mp_result.att_gt[event == e], weights=sizes[event == e]) for e in range(-3, 4)]

    result = aggte(mp_result, type="dynamic", cband=False)

    assert result.aggregation_type == "dynamic"
    np.testing.assert_array_equal(result.event_times, np.arange(-3, 4))
    np.testing.assert_allclose(result.att_by_event, expected, rtol=1e-12)
    np.testing.assert_allclose(result.overall_att, np.mean(expected[3:]), rtol=1e-12)


def test_aggte_calendar_weighs_the_treated_cohorts_of_each_period_by_size(mp_result, mpdta_cohort_sizes):
    treated = mp_result.groups <= mp_result.times
    sizes = np.array([mpdta_cohort_sizes[g] for g in mp_result.groups])
    in_period = [treated & (mp_result.times == t) for t in range(2004, 2008)]
    expected = [np.average(mp_result.att_gt[cells], weights=sizes[cells]) for cells in in_period]

    result = aggte(mp_result, type="calendar", cband=False)

    assert result.aggregation_type == "calendar"
    np.testing.assert_array_equal(result.event_times, np.arange(2004, 2008))
    np.testing.assert_allclose(result.att_by_event, expected, rtol=1e-12)
    np.testing.assert_allclose(result.overall_att, np.mean(expected), rtol=1e-12)


@pytest.mark.parametrize("agg_type", ["simple", "dynamic", "group", "calendar"])
def test_aggte_analytic_overall_standard_error_is_the_root_mean_square_of_its_influence_function(mp_result, agg_type):
    result = aggte(mp_result, type=agg_type, cband=False)

    np.testing.assert_allclose(
        result.overall_se, np.sqrt(np.mean(result.influence_func_overall**2) / mp_result.n_units), rtol=1e-12
    )


@pytest.mark.parametrize("agg_type", ["dynamic", "group", "calendar"])
def test_aggte_analytic_standard_errors_are_the_root_mean_square_of_the_influence_functions(mp_result, agg_type):
    result = aggte(mp_result, type=agg_type, cband=False)
    influence = result.influence_func

    np.testing.assert_allclose(
        result.se_by_event, np.sqrt(np.mean(influence**2, axis=0) / mp_result.n_units), rtol=1e-12
    )
    np.testing.assert_allclose(result.vcov_analytical, influence.T @ influence / mp_result.n_units, rtol=1e-12)


def test_aggte_influence_functions_match_the_weight_derivative(weighted_staggered_panel, weight_derivative):
    spec = {"yname": "y", "tname": "t", "idname": "id", "gname": "g", "weightsname": "w", "est_method": "reg"}
    agg_types = ("simple", "dynamic", "group", "calendar")

    def effects(factors):
        data = weighted_staggered_panel.with_columns(pl.col("w") * pl.Series(np.repeat(factors, 5)))
        fit = att_gt(data, cband=False, **spec)
        results = [aggte(fit, type=agg_type, cband=False) for agg_type in agg_types]
        overall = [result.overall_att for result in results]
        return np.concatenate([overall] + [result.att_by_event for result in results[1:]])

    fit = att_gt(weighted_staggered_panel, cband=False, **spec)
    results = [aggte(fit, type=agg_type, cband=False) for agg_type in agg_types]
    influence = np.column_stack(
        [result.influence_func_overall for result in results] + [result.influence_func for result in results[1:]]
    )

    np.testing.assert_allclose(influence, weight_derivative(effects, 30, range(30)), rtol=0, atol=1e-5)


def test_aggte_invalid_type(mp_result):
    with pytest.raises(ValueError, match="Must be one of"):
        aggte(mp_result, type="invalid")


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"alpha": 1.0}, "alpha=1.0 is not valid. Must be between 0 and 1 (exclusive)."),
        ({"alpha": 0.0}, "alpha=0.0 is not valid. Must be between 0 and 1 (exclusive)."),
        ({"biters": 0}, "biters=0 is not valid. Must be a positive integer."),
        ({"biters": 99.0}, "biters=99.0 is not valid. Must be a positive integer."),
        ({"balance_e": -1}, "balance_e=-1 is not valid. Must be a non-negative integer."),
        ({"balance_e": 1.5}, "balance_e=1.5 is not valid. Must be a non-negative integer."),
        ({"min_e": 2, "max_e": 1}, "min_e=2 must be less than or equal to max_e=1."),
    ],
)
def test_aggte_rejects_invalid_arguments(mp_result, arguments, message):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        aggte(mp_result, type="dynamic", **arguments)


def test_aggte_rejects_a_string_for_clustervars(mp_result):
    message = "clustervars must be a list of strings, not a string. Use clustervars=['state']."

    with pytest.raises(TypeError, match=f"^{re.escape(message)}$"):
        aggte(mp_result, type="dynamic", clustervars="state")


def test_aggte_dynamic_accepts_a_window_of_one_event_time_balanced_on_it(mp_result, mpdta_cohort_sizes):
    on_impact = mp_result.times == mp_result.groups
    sizes = [mpdta_cohort_sizes[g] for g in mp_result.groups[on_impact]]
    expected = np.average(mp_result.att_gt[on_impact], weights=sizes)

    result = aggte(mp_result, type="dynamic", balance_e=0, min_e=0, max_e=0, cband=False)

    np.testing.assert_array_equal(result.event_times, [0])
    np.testing.assert_allclose(result.att_by_event, [expected], rtol=1e-12)


@pytest.mark.parametrize(
    ("balance_e", "cells"),
    [
        (1, {-2: [(2006, 2004)], -1: [(2006, 2005)], 0: [(2004, 2004), (2006, 2006)], 1: [(2004, 2005), (2006, 2007)]}),
        (2, {0: [(2004, 2004)], 1: [(2004, 2005)], 2: [(2004, 2006)]}),
    ],
)
def test_aggte_dynamic_with_balance(mp_result, mpdta_cohort_sizes, balance_e, cells):
    att = dict(zip(zip(mp_result.groups, mp_result.times), mp_result.att_gt))
    expected = [
        np.average([att[cell] for cell in kept], weights=[mpdta_cohort_sizes[g] for g, _ in kept])
        for kept in cells.values()
    ]

    result = aggte(mp_result, type="dynamic", balance_e=balance_e, cband=False)

    assert result.aggregation_type == "dynamic"
    assert result.balanced_event_threshold == balance_e
    np.testing.assert_array_equal(result.event_times, list(cells))
    np.testing.assert_allclose(result.att_by_event, expected, rtol=1e-12)
    np.testing.assert_allclose(
        result.overall_att, np.mean([value for e, value in zip(cells, expected) if e >= 0]), rtol=1e-12
    )


def test_aggte_dynamic_with_min_max_e(mp_result, mpdta_cohort_sizes):
    event = mp_result.times - mp_result.groups
    sizes = np.array([mpdta_cohort_sizes[g] for g in mp_result.groups])
    expected = [np.average(mp_result.att_gt[event == e], weights=sizes[event == e]) for e in range(-1, 3)]

    result = aggte(mp_result, type="dynamic", min_e=-1, max_e=2, cband=False)

    assert result.aggregation_type == "dynamic"
    assert (result.min_event_time, result.max_event_time) == (-1, 2)
    np.testing.assert_array_equal(result.event_times, [-1, 0, 1, 2])
    np.testing.assert_allclose(result.att_by_event, expected, rtol=1e-12)
    np.testing.assert_allclose(result.overall_att, np.mean(expected[1:]), rtol=1e-12)


def test_aggte_max_e_limits_the_cells_that_the_simple_and_group_aggregations_average(mp_result, mpdta_cohort_sizes):
    event = mp_result.times - mp_result.groups
    kept = (event >= 0) & (event <= 1)
    sizes = np.array([mpdta_cohort_sizes[g] for g in mp_result.groups])
    cohorts = [2004, 2006, 2007]
    group_effects = [mp_result.att_gt[kept & (mp_result.groups == g)].mean() for g in cohorts]

    simple = aggte(mp_result, type="simple", max_e=1, cband=False)
    group = aggte(mp_result, type="group", max_e=1, cband=False)

    np.testing.assert_allclose(simple.overall_att, np.average(mp_result.att_gt[kept], weights=sizes[kept]), rtol=1e-12)
    np.testing.assert_allclose(group.att_by_event, group_effects, rtol=1e-12)
    np.testing.assert_allclose(
        group.overall_att, np.average(group_effects, weights=[mpdta_cohort_sizes[g] for g in cohorts]), rtol=1e-12
    )


@pytest.mark.parametrize("agg_type", ["simple", "group", "calendar", "dynamic"])
def test_aggte_with_na_rm(mp_result, agg_type):
    failed = (mp_result.groups == 2006) & (mp_result.times == 2007)
    with_nan = mp_result._replace(att_gt=np.where(failed, np.nan, mp_result.att_gt))
    without = mp_result._replace(
        groups=mp_result.groups[~failed],
        times=mp_result.times[~failed],
        att_gt=mp_result.att_gt[~failed],
        influence_func=mp_result.influence_func[:, ~failed],
    )

    result = aggte(with_nan, type=agg_type, na_rm=True, cband=False)
    expected = aggte(without, type=agg_type, cband=False)

    assert result.aggregation_type == agg_type
    np.testing.assert_allclose(result.overall_att, expected.overall_att, rtol=1e-12)
    np.testing.assert_allclose(result.overall_se, expected.overall_se, rtol=1e-12)
    np.testing.assert_allclose(result.vcov_analytical, expected.vcov_analytical, rtol=1e-12)


@pytest.mark.parametrize("agg_type", ["dynamic", "group", "calendar"])
def test_aggte_bootstrap_takes_standard_errors_and_band_from_the_requested_draws(
    monkeypatch, draws_from_leading_weights, mp_result, agg_type
):
    signs = np.where(np.random.default_rng(5).random((400, mp_result.n_units)) < 0.5, 1.0, -1.0)
    monkeypatch.setattr(
        import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", draws_from_leading_weights(signs)
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = aggte(mp_result, type=agg_type, boot=True, cband=True, biters=299, alpha=0.1, random_state=1)
    params = result.estimation_params
    influence = np.column_stack([result.influence_func, result.influence_func_overall])
    draws = signs[:299] @ influence / mp_result.n_units
    q75, q25 = np.percentile(draws, [75, 25], axis=0, method="inverted_cdf")
    scale = (q75 - q25) / 1.3489795
    band = np.percentile(np.max(np.abs(draws[:, :-1]) / scale[:-1], axis=1), 90, method="inverted_cdf")

    assert [str(w.message) for w in caught] == []
    assert (params["bootstrap"], params["biters"], params["uniform_bands"], params["alpha"]) == (True, 299, True, 0.1)
    np.testing.assert_allclose(result.se_by_event, scale[:-1], rtol=1e-12)
    np.testing.assert_allclose(result.overall_se, scale[-1], rtol=1e-12)
    np.testing.assert_allclose(result.critical_values, band, rtol=1e-12)


def test_aggte_bootstrap_defaults_to_the_settings_of_the_fit(monkeypatch, draws_from_leading_weights, mpdta_data):
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "est_method": "reg"}
    fit = att_gt(mpdta_data, boot=True, biters=123, cband=True, alpha=0.1, random_state=3, **spec)
    signs = np.where(np.random.default_rng(5).random((400, fit.n_units)) < 0.5, 1.0, -1.0)
    monkeypatch.setattr(
        import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", draws_from_leading_weights(signs)
    )

    result = aggte(fit, type="dynamic", random_state=3)
    draws = signs[:123] @ result.influence_func / fit.n_units
    q75, q25 = np.percentile(draws, [75, 25], axis=0, method="inverted_cdf")
    scale = (q75 - q25) / 1.3489795

    np.testing.assert_allclose(result.se_by_event, scale, rtol=1e-12)
    np.testing.assert_allclose(
        result.critical_values,
        np.percentile(np.max(np.abs(draws) / scale, axis=1), 90, method="inverted_cdf"),
        rtol=1e-12,
    )


def test_aggte_rejects_the_removed_alp_argument(mp_result):
    with pytest.raises(TypeError, match="unexpected keyword argument 'alp'"):
        aggte(mp_result, type="simple", alp=0.1)


@pytest.mark.parametrize(
    ("recorded", "request_args"),
    [({}, {"clustervars": ["countyreal"]}), ({"clustervars": ["countyreal"], "cluster": None}, {})],
)
def test_aggte_clustering_on_unit_id_records_no_clusters(mp_result, recorded, request_args):
    fit = mp_result._replace(estimation_params={**mp_result.estimation_params, **recorded})
    draws = {"type": "simple", "boot": True, "biters": 99, "random_state": 5}

    result = aggte(fit, **request_args, **draws)
    plain = aggte(mp_result, **draws)

    assert result.estimation_params["clustervars"] == []
    assert result.estimation_params["cluster"] is None
    assert result.overall_se == plain.overall_se


@pytest.mark.parametrize(
    ("fit_clusters", "requested", "names", "advice"),
    [
        (None, ["state"], "'state'", "pass an empty list for unit-level inference"),
        (
            ["state"],
            ["countyreal", "region", "county"],
            "'region' or 'county'",
            "pass clustervars=['state'] to reuse its clusters, or an empty list for unit-level inference",
        ),
    ],
)
def test_aggte_rejects_clusters_the_fit_did_not_use(mp_result, fit_clusters, requested, names, advice):
    cluster = None if fit_clusters is None else np.arange(mp_result.n_units) % 29
    fit = mp_result._replace(
        estimation_params={**mp_result.estimation_params, "clustervars": fit_clusters, "cluster": cluster}
    )

    with pytest.raises(
        ValueError,
        match=f"^The att_gt fit did not cluster on {re.escape(names)}\\. Since the aggregation reuses the clusters "
        "of the att_gt fit, pass the cluster variable to att_gt instead\\. To aggregate this fit, "
        f"{re.escape(advice)}\\.$",
    ):
        aggte(fit, type="dynamic", clustervars=requested)


@pytest.mark.parametrize("request_args", [{}, {"clustervars": ["state"]}])
def test_aggte_rejects_recorded_clusters_without_a_cluster_vector(mp_result, request_args):
    fit = mp_result._replace(
        estimation_params={**mp_result.estimation_params, "clustervars": ["state"], "cluster": None}
    )

    with pytest.raises(
        ValueError,
        match="^The att_gt result records clustervars=\\['state'\\] but no cluster for its units\\. Rerun att_gt or "
        "store the cluster of each unit in estimation_params\\['cluster'\\]\\. To aggregate the result as it is, pass "
        "an empty list for unit-level inference\\.$",
    ):
        aggte(fit, type="dynamic", **request_args)


@pytest.mark.parametrize(
    "fit_args", [{"clustervars": ["state"]}, {"clustervars": ["countyreal", "state"], "panel": False}]
)
def test_aggte_unit_id_request_on_clustered_fit_gives_unit_level_inference(mpdta_with_states, fit_args, recwarn):
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "est_method": "reg"}
    with pytest.warns(UserWarning, match="^The units fall in only 29 clusters\\. "):
        fit = att_gt(mpdta_with_states, boot=True, biters=99, random_state=7, **spec, **fit_args)
    draws = {"type": "dynamic", "boot": True, "biters": 199, "cband": False, "random_state": 3}
    recwarn.clear()

    by_unit = aggte(fit, clustervars=["countyreal"], **draws)
    unclustered = aggte(fit, clustervars=[], **draws)
    by_state = aggte(fit, clustervars=["countyreal", "state"], **draws)
    inherited = aggte(fit, **draws)
    aggte(fit, type="dynamic", clustervars=["countyreal"], boot=False, cband=False)

    assert len(recwarn) == 0
    assert by_unit.estimation_params["clustervars"] == []
    assert by_unit.estimation_params["cluster"] is None
    assert by_unit.overall_se == unclustered.overall_se
    np.testing.assert_array_equal(by_unit.se_by_event, unclustered.se_by_event)
    assert by_state.estimation_params["clustervars"] == ["state"]
    np.testing.assert_array_equal(by_state.estimation_params["cluster"], fit.estimation_params["cluster"])
    assert by_state.overall_se == inherited.overall_se
    assert inherited.overall_se != pytest.approx(by_unit.overall_se, rel=0.05)


@pytest.mark.parametrize("boot", [True, False], ids=["bootstrap", "analytic"])
def test_aggte_leaves_the_few_clusters_warning_to_att_gt(mpdta_with_states, boot):
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "est_method": "reg"}
    agg_types = ("simple", "dynamic", "group", "calendar")
    requests = [(agg_type, agg_boot) for agg_type in agg_types for agg_boot in (True, False)]
    kinds = ["Bootstrap" if agg_boot else "Analytical" for _, agg_boot in requests]

    with warnings.catch_warnings(record=True) as at_fit:
        warnings.simplefilter("always")
        fit = att_gt(mpdta_with_states, clustervars=["state"], boot=boot, biters=99, random_state=7, **spec)
    with warnings.catch_warnings(record=True) as at_aggregation:
        warnings.simplefilter("always")
        aggregations = [
            aggte(fit, type=agg_type, boot=agg_boot, biters=99, random_state=7) for agg_type, agg_boot in requests
        ]

    assert len([w for w in at_fit if "very few clusters" in str(w.message)]) == 1
    assert not [w for w in at_aggregation if "very few clusters" in str(w.message)]
    assert [[line for line in str(result).splitlines() if "standard errors" in line] for result in aggregations] == [
        [f" {kind} standard errors clustered by state"] for kind in kinds
    ]
    assert [result.__maketables_stat__("se_type") for result in aggregations] == [
        f"{kind} clustered by state" for kind in kinds
    ]
    assert [result.__maketables_vcov_info__ for result in aggregations] == [
        {"vcov_type": kind.lower(), "clustervar": "state"} for kind in kinds
    ]


@pytest.mark.parametrize("agg_type", ["simple", "dynamic", "group", "calendar"])
def test_aggte_without_the_bootstrap_sums_the_overall_influence_function_within_clusters(mpdta_with_states, agg_type):
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "est_method": "reg"}
    with pytest.warns(UserWarning, match="^The units fall in only 29 clusters\\. "):
        fit = att_gt(mpdta_with_states, clustervars=["state"], **spec)
    plain_fit = att_gt(mpdta_with_states, **spec)

    result = aggte(fit, type=agg_type, cband=False)
    plain = aggte(plain_fit, type=agg_type, cband=False)
    _, cluster_idx = np.unique(fit.estimation_params["cluster"], return_inverse=True)
    sums = np.bincount(cluster_idx, weights=result.influence_func_overall)

    np.testing.assert_allclose(result.overall_se, np.sqrt(np.sum(sums**2)) / fit.n_units, rtol=1e-12)
    assert result.overall_att == plain.overall_att
    assert abs(result.overall_se - plain.overall_se) > 1e-4


@pytest.mark.parametrize("agg_type", ["dynamic", "group", "calendar"])
def test_aggte_without_the_bootstrap_sums_each_disaggregated_influence_function_within_clusters(
    mpdta_with_states, agg_type
):
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "est_method": "reg"}
    with pytest.warns(UserWarning, match="^The units fall in only 29 clusters\\. "):
        fit = att_gt(mpdta_with_states, clustervars=["state"], **spec)
    plain_fit = att_gt(mpdta_with_states, **spec)

    result = aggte(fit, type=agg_type, cband=False)
    plain = aggte(plain_fit, type=agg_type, cband=False)
    _, cluster_idx = np.unique(fit.estimation_params["cluster"], return_inverse=True)
    sums = np.stack([np.bincount(cluster_idx, weights=column) for column in result.influence_func.T], axis=1)
    expected = np.sqrt(np.sum(sums**2, axis=0)) / fit.n_units

    np.testing.assert_allclose(result.se_by_event, expected, rtol=1e-12)
    np.testing.assert_array_equal(result.att_by_event, plain.att_by_event)
    assert np.max(np.abs(result.se_by_event - plain.se_by_event)) > 1e-3


@pytest.mark.parametrize("agg_type", ["simple", "dynamic", "group", "calendar"])
def test_aggte_call_info_names_the_requested_aggregation(mp_result, agg_type):
    result = aggte(mp_result, type=agg_type, cband=False)

    assert result.call_info == {"function": f"aggte(MP, type='{agg_type}')", "type": agg_type}


def test_aggte_with_missing_values_no_na_rm(mp_result):
    mp_result.att_gt[0] = np.nan

    with pytest.raises(ValueError, match="Missing values at att_gt found"):
        aggte(mp_result, type="simple", na_rm=False)


def test_aggte_all_nan_values(mp_result):
    mp_result.att_gt[:] = np.nan

    with pytest.raises(
        ValueError,
        match="^All att_gt estimates are NaN\\. The att_gt warnings name the cells that failed to estimate\\. Fewer "
        "covariates or control_group='notyettreated' can recover them\\.$",
    ):
        aggte(mp_result, type="simple", na_rm=True)


@pytest.mark.parametrize(
    ("window", "message"),
    [
        ({"min_e": 100, "max_e": 101}, "min_e=100 and max_e=101"),
        ({"balance_e": 10}, "min_e=-inf and max_e=inf with balance_e=10"),
    ],
)
def test_aggte_dynamic_rejects_window_without_event_times(mp_result, window, message):
    with pytest.raises(ValueError, match=f"^No group-time effect has an event time within {re.escape(message)}\\. "):
        aggte(mp_result, type="dynamic", **window)


def test_aggte_infinite_values_in_att(mp_result):
    mp_result.att_gt[0] = np.inf

    result = aggte(mp_result, type="simple", cband=False)

    assert result.overall_att == np.inf


def test_aggte_reports_nan_standard_errors_when_the_influence_functions_vanish(mp_result):
    flat = mp_result._replace(att_gt=np.full(12, 0.1), influence_func=np.zeros_like(mp_result.influence_func))

    result = aggte(flat, type="dynamic", cband=False)

    np.testing.assert_allclose(result.att_by_event, np.full(7, 0.1), rtol=1e-12)
    assert np.isnan(result.overall_se)
    assert np.isnan(result.se_by_event).all()


def test_aggte_reports_nan_bootstrap_standard_errors_when_the_draws_have_no_spread(
    monkeypatch, draws_from_leading_weights, mp_result
):
    weights = np.zeros((999, mp_result.n_units))
    weights[:, 0] = np.r_[np.zeros(799), np.full(100, 1.0), np.full(100, -1.0)]
    monkeypatch.setattr(
        import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", draws_from_leading_weights(weights)
    )

    result = aggte(mp_result, type="group", boot=True, cband=False, biters=999, random_state=7)

    assert np.isnan(result.overall_se)
    assert np.isnan(result.se_by_event).all()


def test_aggte_extreme_event_times(mp_result):
    agg_result = aggte(mp_result, type="dynamic", min_e=-1e10, max_e=1e10, cband=False)

    assert agg_result.aggregation_type == "dynamic"
    assert agg_result.min_event_time == int(-1e10)
    assert agg_result.max_event_time == int(1e10)


def test_aggte_calendar_with_shifted_periods(mpdta_data, mp_result):
    df = mpdta_data.with_columns(
        pl.when(pl.col("first.treat") != 0).then(pl.col("first.treat") + 1000).otherwise(np.inf).alias("first.treat")
    )
    df = df.with_columns((pl.col("year") + 1000).alias("year"))

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first.treat",
        idname="countyreal",
        xformla="~ 1",
        est_method="reg",
    )

    shifted = aggte(result, type="calendar", cband=False)
    original = aggte(mp_result, type="calendar", cband=False)

    np.testing.assert_array_equal(shifted.event_times, [3004, 3005, 3006, 3007])
    np.testing.assert_allclose(shifted.att_by_event, original.att_by_event, rtol=1e-12)
    np.testing.assert_allclose(shifted.se_by_event, original.se_by_event, rtol=1e-12)


def test_aggte_all_treated_same_time(mpdta_data):
    df = mpdta_data.with_columns(pl.when(pl.col("first.treat") != 0).then(2004).otherwise(np.inf).alias("first.treat"))

    with pytest.warns(
        UserWarning, match="^No pre-treatment periods available for the Wald pre-test of parallel trends\\."
    ):
        result = att_gt(
            data=df,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            xformla="~ 1",
            est_method="reg",
        )

    agg_result = aggte(result, type="group", cband=False)

    np.testing.assert_array_equal(agg_result.event_times, [2004])
    np.testing.assert_allclose(agg_result.att_by_event, [result.att_gt[result.times >= 2004].mean()], rtol=1e-12)
    np.testing.assert_allclose(agg_result.overall_att, agg_result.att_by_event[0], rtol=1e-12)


@pytest.mark.parametrize("band_args", [{}, {"cband": True, "boot": False}])
@pytest.mark.parametrize("agg_type", ["dynamic", "group", "calendar"])
def test_aggte_uniform_bands_without_bootstrap(mp_result, agg_type, band_args):
    with pytest.warns(UserWarning, match="^Used bootstrap procedure to compute simultaneous confidence band$"):
        agg_result = aggte(mp_result, type=agg_type, biters=999, random_state=3, **band_args)

    assert agg_result.estimation_params["uniform_bands"]
    assert np.all(agg_result.critical_values > 1.96)


def test_aggte_critical_value_nan_handling(mp_result):
    mp_result.influence_func[:] = 0

    with pytest.warns(
        UserWarning,
        match="^Simultaneous critical value is NaN, likely because the standard errors could not be computed\\. ",
    ):
        result = aggte(mp_result, type="group", cband=True, boot=True, biters=99, random_state=7)

    assert not result.estimation_params["uniform_bands"]
    np.testing.assert_allclose(result.critical_values, np.full(3, 1.959963984540054), rtol=1e-12)


def test_aggte_dynamic_no_post_treatment(mp_result, mpdta_cohort_sizes):
    event = mp_result.times - mp_result.groups
    sizes = np.array([mpdta_cohort_sizes[g] for g in mp_result.groups])
    expected = [np.average(mp_result.att_gt[event == e], weights=sizes[event == e]) for e in (-3, -2, -1)]

    with pytest.warns(
        UserWarning,
        match="^The overall effect is NaN because max_e=-1 leaves out every post-treatment group-time effect\\. "
        "Pass a larger max_e to estimate it\\.$",
    ):
        agg_result = aggte(mp_result, type="dynamic", max_e=-1, cband=False)

    np.testing.assert_array_equal(agg_result.event_times, [-3, -2, -1])
    np.testing.assert_allclose(agg_result.att_by_event, expected, rtol=1e-12)
    assert np.isnan(agg_result.overall_att)
    assert np.isnan(agg_result.overall_se)


@pytest.mark.parametrize(("aggregation", "event_times"), [("simple", None), ("group", [])])
def test_aggte_max_e_before_treatment_reports_nan(mp_result, aggregation, event_times):
    with pytest.warns(
        UserWarning,
        match="^The overall effect is NaN because max_e=-1 leaves out every post-treatment group-time effect\\. ",
    ):
        result = aggte(mp_result, type=aggregation, max_e=-1, cband=False)

    assert np.isnan(result.overall_att)
    assert np.isnan(result.overall_se)
    np.testing.assert_array_equal(result.event_times, event_times)


def test_aggte_simple_bootstrap_standard_error_uses_the_requested_number_of_draws(
    monkeypatch, draws_from_leading_weights, mp_result
):
    signs = np.where(np.random.default_rng(5).random((400, mp_result.n_units)) < 0.5, 1.0, -1.0)
    monkeypatch.setattr(
        import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", draws_from_leading_weights(signs)
    )

    agg_result = aggte(mp_result, type="simple", boot=True, biters=10, random_state=2)
    draws = signs[:10] @ agg_result.influence_func / mp_result.n_units
    q75, q25 = np.percentile(draws, [75, 25], method="inverted_cdf")

    assert agg_result.estimation_params["biters"] == 10
    np.testing.assert_allclose(agg_result.overall_se, (q75 - q25) / 1.3489795, rtol=1e-12)


def test_aggte_cband_narrower_than_pointwise_fallback(monkeypatch, draws_from_leading_weights, mp_result):
    weights = np.zeros((999, mp_result.n_units))
    weights[:, 0] = np.linspace(-1.0, 1.0, 999)
    monkeypatch.setattr(
        import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", draws_from_leading_weights(weights)
    )

    with pytest.warns(
        UserWarning,
        match="^The simultaneous confidence band is narrower than the pointwise confidence interval\\. ",
    ):
        agg_result = aggte(mp_result, type="group", cband=True, boot=True, biters=999, random_state=7)

    assert not agg_result.estimation_params["uniform_bands"]
    np.testing.assert_allclose(agg_result.critical_values, np.full(3, 1.959963984540054), rtol=1e-12)


def test_aggte_cband_very_large_cv_warning(monkeypatch, draws_from_leading_weights, mp_result):
    spikes = np.r_[np.linspace(-1.0, 1.0, 899), np.full(50, 100.0), np.full(50, -100.0)]
    weights = np.zeros((999, mp_result.n_units))
    weights[:, 0] = spikes
    monkeypatch.setattr(
        import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", draws_from_leading_weights(weights)
    )
    q75, q25 = np.percentile(spikes, [75, 25], method="inverted_cdf")

    with pytest.warns(UserWarning, match="^Simultaneous critical value is very large and may be unreliable\\. "):
        agg_result = aggte(mp_result, type="group", cband=True, boot=True, biters=999, random_state=7)

    assert agg_result.estimation_params["uniform_bands"]
    np.testing.assert_allclose(
        agg_result.critical_values,
        np.full(3, np.percentile(np.abs(spikes) / ((q75 - q25) / 1.3489795), 95, method="inverted_cdf")),
        rtol=1e-12,
    )


@pytest.mark.parametrize("agg_type", ["simple", "dynamic", "group", "calendar"])
def test_aggte_bootstrap_reproducibility(mp_result, agg_type):
    spec = {"type": agg_type, "boot": True, "biters": 50}

    result1 = aggte(mp_result, **spec, random_state=42)
    result2 = aggte(mp_result, **spec, random_state=42)
    other = aggte(mp_result, **spec, random_state=43)

    assert result1.estimation_params["random_state"] == 42
    assert result1.overall_se == result2.overall_se
    np.testing.assert_array_equal(result1.se_by_event, result2.se_by_event)
    np.testing.assert_array_equal(result1.critical_values, result2.critical_values)
    assert other.overall_se != result1.overall_se


def test_aggte_band_without_bootstrap_draws_one_multiplier_per_cluster(
    monkeypatch, draws_from_leading_weights, mpdta_with_states
):
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "est_method": "reg"}
    with pytest.warns(UserWarning, match="^The units fall in only 29 clusters\\. "):
        clustered = att_gt(mpdta_with_states, clustervars=["state"], **spec)
    signs = np.where(np.random.default_rng(11).random((999, clustered.n_units)) < 0.5, 1.0, -1.0)
    monkeypatch.setattr(
        import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", draws_from_leading_weights(signs)
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from_clustered = aggte(clustered, type="dynamic", boot=False, cband=True, biters=599, random_state=11)
    _, codes = np.unique(clustered.estimation_params["cluster"], return_inverse=True)
    sums = np.column_stack([np.bincount(codes, weights=column) for column in from_clustered.influence_func.T])
    draws = signs[:599, :29] @ sums
    q75, q25 = np.percentile(draws, [75, 25], axis=0, method="inverted_cdf")
    band = np.percentile(np.max(np.abs(draws) / ((q75 - q25) / 1.3489795), axis=1), 95, method="inverted_cdf")

    assert [str(w.message) for w in caught] == ["Used bootstrap procedure to compute simultaneous confidence band"]
    assert from_clustered.estimation_params["uniform_bands"]
    assert sums.shape == (29, 7)
    np.testing.assert_allclose(from_clustered.critical_values, np.full(7, band), rtol=1e-12)


def test_aggte_unit_id_clusters_give_unit_level_inference(mpdta_data, mp_result):
    unit_clustered = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        gname="first.treat",
        idname="countyreal",
        xformla="~ 1",
        est_method="reg",
        clustervars=["countyreal"],
    )
    draws = {"type": "dynamic", "cband": True, "biters": 999, "random_state": 11}

    with pytest.warns(UserWarning) as analytic_warnings:
        aggte(unit_clustered, clustervars=["countyreal"], boot=False, **draws)
    from_unit_clustered = aggte(unit_clustered, clustervars=["countyreal"], boot=True, **draws)
    from_plain = aggte(mp_result, boot=True, **draws)

    assert [str(w.message) for w in analytic_warnings] == [
        "Used bootstrap procedure to compute simultaneous confidence band"
    ]
    np.testing.assert_array_equal(from_unit_clustered.se_by_event, from_plain.se_by_event)
    np.testing.assert_array_equal(from_unit_clustered.critical_values, from_plain.critical_values)
    assert from_unit_clustered.overall_se == from_plain.overall_se


def test_aggte_cluster_variable_unique_to_each_unit_stays_unit_level(mpdta_data, mp_result):
    data = mpdta_data.with_columns(pl.col("countyreal").alias("county_copy"))
    copy_fit = att_gt(
        data,
        yname="lemp",
        tname="year",
        gname="first.treat",
        idname="countyreal",
        xformla="~ 1",
        est_method="reg",
        clustervars=["county_copy"],
        boot=True,
        biters=99,
        random_state=7,
    )
    draws = {"type": "dynamic", "cband": True, "biters": 999, "random_state": 11}

    with pytest.warns(UserWarning) as analytic_warnings:
        aggte(copy_fit, boot=False, **draws)
    from_copy = aggte(copy_fit, boot=True, **draws)
    from_plain = aggte(mp_result, boot=True, **draws)

    assert [str(w.message) for w in analytic_warnings] == [
        "Used bootstrap procedure to compute simultaneous confidence band"
    ]
    assert from_copy.estimation_params["clustervars"] == ["county_copy"]
    np.testing.assert_array_equal(from_copy.se_by_event, from_plain.se_by_event)
    np.testing.assert_array_equal(from_copy.critical_values, from_plain.critical_values)
    assert from_copy.overall_se == from_plain.overall_se


@pytest.mark.parametrize("agg_type", ["simple", "dynamic", "group", "calendar"])
def test_aggte_unbalanced_se_invariant_to_unit_labels(mpdta_unbalanced, agg_type):
    spec = dict(
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        allow_unbalanced_panel=True,
        boot=False,
        cband=False,
    )
    reversed_ids = mpdta_unbalanced.with_columns((100000 - pl.col("countyreal")).alias("countyreal"))

    original = aggte(att_gt(data=mpdta_unbalanced, **spec), type=agg_type)
    relabeled = aggte(att_gt(data=reversed_ids, **spec), type=agg_type)

    np.testing.assert_allclose(relabeled.overall_att, original.overall_att, rtol=1e-12)
    np.testing.assert_allclose(relabeled.overall_se, original.overall_se, rtol=1e-12)
    if agg_type != "simple":
        np.testing.assert_allclose(relabeled.se_by_event, original.se_by_event, rtol=1e-12)


@pytest.mark.parametrize("agg_type", ["simple", "dynamic", "group", "calendar"])
def test_aggte_cross_section_allow_unbalanced_matches_cross_section(mpdta_data, agg_type):
    spec = dict(yname="lemp", tname="year", gname="first.treat", panel=False, boot=False, cband=False)

    plain = aggte(att_gt(data=mpdta_data, **spec), type=agg_type)
    flagged = aggte(att_gt(data=mpdta_data, allow_unbalanced_panel=True, **spec), type=agg_type)

    np.testing.assert_allclose(flagged.overall_att, plain.overall_att, rtol=1e-12)
    np.testing.assert_allclose(flagged.overall_se, plain.overall_se, rtol=1e-12)
    if agg_type != "simple":
        np.testing.assert_allclose(flagged.se_by_event, plain.se_by_event, rtol=1e-12)


@pytest.mark.filterwarnings("ignore:panel=False was specified:UserWarning")
@pytest.mark.parametrize("agg_type", ["simple", "dynamic", "group", "calendar"])
@pytest.mark.parametrize("allow_unbalanced_panel", [False, True])
def test_aggte_cross_section_with_idname_matches_cross_section(mpdta_data, agg_type, allow_unbalanced_panel):
    spec = dict(yname="lemp", tname="year", gname="first.treat", panel=False, boot=False, cband=False)

    plain = aggte(att_gt(data=mpdta_data, **spec), type=agg_type)
    with_id = aggte(
        att_gt(data=mpdta_data, idname="countyreal", allow_unbalanced_panel=allow_unbalanced_panel, **spec),
        type=agg_type,
    )

    np.testing.assert_allclose(with_id.overall_att, plain.overall_att, rtol=1e-12)
    np.testing.assert_allclose(with_id.overall_se, plain.overall_se, rtol=1e-12)
    if agg_type != "simple":
        np.testing.assert_allclose(with_id.se_by_event, plain.se_by_event, rtol=1e-12)


@pytest.mark.parametrize("aggregation", ["simple", "dynamic", "group", "calendar"])
def test_aggte_without_post_treatment_effects_reports_nan(no_post_treatment_result, aggregation):
    with pytest.warns(
        UserWarning,
        match="^The overall effect is NaN because no post-treatment group-time effect is left to average\\. The "
        "att_gt warnings name the cells that failed to estimate\\. Fewer covariates or "
        "control_group='notyettreated' can recover them\\.$",
    ):
        result = aggte(no_post_treatment_result, type=aggregation, cband=False)

    assert np.isnan(result.overall_att)
    assert np.isnan(result.overall_se)


@pytest.mark.parametrize("aggregation", ["simple", "group"])
def test_aggte_names_the_failed_cells_when_max_e_leaves_out_no_post_treatment_effect(
    no_post_treatment_result, aggregation
):
    with pytest.warns(
        UserWarning,
        match="^The overall effect is NaN because no post-treatment group-time effect is left to average\\. ",
    ):
        result = aggte(no_post_treatment_result, type=aggregation, max_e=-2, cband=False)

    assert np.isnan(result.overall_att)


def test_aggte_without_post_treatment_effects_keeps_pre_treatment_event_times(no_post_treatment_result):
    with pytest.warns(UserWarning, match="^The overall effect is NaN because no post-treatment group-time effect is "):
        result = aggte(no_post_treatment_result, type="dynamic", cband=False)

    np.testing.assert_array_equal(result.event_times, [-1])
    np.testing.assert_allclose(result.att_by_event, no_post_treatment_result.att_gt, rtol=1e-12)


@pytest.mark.parametrize("na_rm", [False, True])
@pytest.mark.parametrize("aggregation", ["group", "calendar"])
def test_aggte_leaves_out_cohorts_and_periods_without_post_treatment_effects(
    no_post_treatment_result, aggregation, na_rm
):
    with pytest.warns(UserWarning, match="^The overall effect is NaN because no post-treatment group-time effect is "):
        result = aggte(no_post_treatment_result, type=aggregation, na_rm=na_rm, cband=False)

    assert len(result.event_times) == 0
    assert len(result.att_by_event) == 0


@pytest.mark.parametrize("aggregation", ["simple", "dynamic", "group", "calendar"])
def test_aggte_rejects_result_without_cells(thin_weighted_controls_panel, aggregation):
    with (
        pytest.warns(UserWarning, match="^No pre-treatment periods available for the Wald pre-test"),
        pytest.warns(UserWarning, match="^Overlap condition violated for 3 in time period [23]$"),
    ):
        result = att_gt(thin_weighted_controls_panel, yname="y", tname="t", idname="id", gname="g", weightsname="w")

    with pytest.raises(
        ValueError,
        match="^The att_gt result holds no group-time effects to aggregate\\. The att_gt warnings name the cells "
        "that failed to estimate\\. Fewer covariates or control_group='notyettreated' can recover them\\.$",
    ):
        aggte(result, type=aggregation)


def test_aggte_calendar_leaves_out_periods_without_treated_estimates(mpdta_separated_2004):
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "xformla": "~ size"}

    with pytest.warns(UserWarning, match="^Overlap condition violated for 2004 in time period 200[4-7]$"):
        universal = att_gt(mpdta_separated_2004, **spec, base_period="universal")
        varying = att_gt(mpdta_separated_2004, **spec)
    from_universal = aggte(universal, type="calendar", na_rm=True, cband=False)
    from_varying = aggte(varying, type="calendar", cband=False)

    np.testing.assert_array_equal(from_universal.event_times, [2006, 2007])
    np.testing.assert_array_equal(from_varying.event_times, [2006, 2007])
    np.testing.assert_allclose(from_universal.att_by_event, from_varying.att_by_event, rtol=1e-12)
    np.testing.assert_allclose(from_universal.se_by_event, from_varying.se_by_event, rtol=1e-12)
    np.testing.assert_allclose(from_universal.overall_att, from_varying.overall_att, rtol=1e-12)
