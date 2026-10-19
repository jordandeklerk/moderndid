"""Tests for Aggregated Treatment Effect."""

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid import compute_aggte, mp


@pytest.mark.parametrize("agg_type", ["simple", "dynamic", "group", "calendar"])
def test_aggregation_types(mpdta_mp_result, agg_type):
    result = compute_aggte(mpdta_mp_result, aggregation_type=agg_type)

    assert result.aggregation_type == agg_type
    assert result.overall_att is not None
    assert result.overall_se is not None
    assert result.influence_func is not None
    if agg_type == "simple":
        assert result.influence_func.shape == (mpdta_mp_result.n_units,)
    else:
        assert result.influence_func.shape[0] == mpdta_mp_result.n_units
        assert result.influence_func.shape[1] == len(result.event_times)

    if agg_type in ["dynamic", "group", "calendar"]:
        assert result.event_times is not None
        assert result.att_by_event is not None
        assert result.se_by_event is not None
        assert len(result.event_times) == len(result.att_by_event)
        assert len(result.event_times) == len(result.se_by_event)


def test_simple_aggregation_details(mpdta_mp_result):
    result = compute_aggte(mpdta_mp_result, aggregation_type="simple")

    post_treatment = mpdta_mp_result.groups <= mpdta_mp_result.times
    if post_treatment.any():
        assert not np.isnan(result.overall_att)
        assert result.overall_se > 0


def test_dynamic_aggregation_event_times(mpdta_mp_result):
    result = compute_aggte(mpdta_mp_result, aggregation_type="dynamic")

    assert len(result.event_times) == 7
    assert np.allclose(np.sort(result.event_times), np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))


def test_group_aggregation_groups(mpdta_mp_result):
    result = compute_aggte(mpdta_mp_result, aggregation_type="group")

    unique_groups = np.unique(mpdta_mp_result.groups)
    assert len(result.event_times) == len(unique_groups)
    assert np.allclose(result.event_times, unique_groups)


def test_calendar_aggregation_periods(mpdta_mp_result):
    result = compute_aggte(mpdta_mp_result, aggregation_type="calendar")

    np.testing.assert_array_equal(result.event_times, [2004, 2005, 2006, 2007])


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


@pytest.mark.parametrize("min_e,max_e", [(-2, 2), (-1, 1), (0, 3)])
def test_event_time_restrictions(mpdta_mp_result, min_e, max_e):
    result = compute_aggte(mpdta_mp_result, aggregation_type="dynamic", min_e=min_e, max_e=max_e)

    assert result.min_event_time == min_e
    assert result.max_event_time == max_e
    assert np.all((result.event_times >= min_e) & (result.event_times <= max_e))


def test_balance_e_option(synthetic_mp_result):
    result = compute_aggte(synthetic_mp_result, aggregation_type="dynamic", balance_e=1)

    assert result.balanced_event_threshold == 1
    assert np.all(result.event_times <= 1)


def test_na_rm_functionality(synthetic_mp_result):
    synthetic_mp_result.att_gt[0] = np.nan
    synthetic_mp_result.att_gt[5] = np.nan

    with pytest.raises(ValueError, match="Missing values"):
        compute_aggte(synthetic_mp_result, aggregation_type="simple", na_rm=False)

    result = compute_aggte(synthetic_mp_result, aggregation_type="simple", na_rm=True)
    assert not np.isnan(result.overall_att)
    assert not np.isnan(result.overall_se)


@pytest.mark.parametrize("bstrap,cband", [(True, True), (True, False), (False, False)])
def test_bootstrap_options(synthetic_mp_result, bstrap, cband):
    result = compute_aggte(
        synthetic_mp_result,
        aggregation_type="group",
        bootstrap=bstrap,
        biters=999,
        confidence_band=cband,
        random_state=42,
    )

    assert result.estimation_params["bootstrap"] == bstrap
    assert result.estimation_params["uniform_bands"] == cband
    assert len(result.critical_values) == len(result.event_times)
    if cband:
        assert np.all(result.critical_values > 1.96)
    else:
        np.testing.assert_allclose(result.critical_values, 1.959964, rtol=1e-6)


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
def test_significance_levels(synthetic_mp_result, alpha):
    result = compute_aggte(synthetic_mp_result, aggregation_type="simple", alpha=alpha)

    stored_alpha = result.estimation_params.get("alpha", synthetic_mp_result.alpha)
    assert stored_alpha == alpha


def test_compute_aggte_stores_biters(synthetic_mp_result):
    result = compute_aggte(synthetic_mp_result, aggregation_type="group", bootstrap=True, biters=77, random_state=4)

    assert result.estimation_params["biters"] == 77


def test_compute_aggte_rejects_the_removed_bootstrap_iterations_argument(synthetic_mp_result):
    with pytest.raises(TypeError, match="unexpected keyword argument 'bootstrap_iterations'"):
        compute_aggte(synthetic_mp_result, aggregation_type="simple", bootstrap_iterations=99)


def test_compute_aggte_rejects_the_removed_dropna_argument(synthetic_mp_result):
    with pytest.raises(TypeError, match="unexpected keyword argument 'dropna'"):
        compute_aggte(synthetic_mp_result, aggregation_type="simple", dropna=True)


def test_invalid_aggregation_type(synthetic_mp_result):
    with pytest.raises(ValueError, match="must be one of"):
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


@pytest.mark.parametrize("agg_type", ["simple", "group", "dynamic", "calendar"])
def test_all_positive_standard_errors(mpdta_mp_result, agg_type):
    result = compute_aggte(mpdta_mp_result, aggregation_type=agg_type)

    if not np.isnan(result.overall_se):
        assert result.overall_se > 0

    if result.se_by_event is not None:
        valid_se = result.se_by_event[~np.isnan(result.se_by_event)]
        assert np.all(valid_se > 0)


def test_influence_function_properties(mpdta_mp_result):
    result = compute_aggte(mpdta_mp_result, aggregation_type="simple")

    assert np.abs(np.mean(result.influence_func)) < 0.01

    var_if = np.var(result.influence_func)
    se_from_if = np.sqrt(var_if / mpdta_mp_result.n_units)

    if not np.isnan(result.overall_se):
        assert np.abs(se_from_if - result.overall_se) / result.overall_se < 0.1


@pytest.mark.parametrize("agg_type", ["simple", "dynamic", "group", "calendar"])
def test_aggregation_correctness(synthetic_mp_result, agg_type):
    result = compute_aggte(synthetic_mp_result, aggregation_type=agg_type)

    groups = synthetic_mp_result.groups
    times = synthetic_mp_result.times
    att_gt = synthetic_mp_result.att_gt

    post_treatment = times >= groups

    if agg_type == "simple":
        expected_att = np.mean(att_gt[post_treatment])
    elif agg_type == "dynamic":
        event_times = times - groups
        post_treatment_mask = (times >= groups) & np.isfinite(groups)
        valid_event_times = event_times[post_treatment_mask]
        unique_event_times = np.unique(valid_event_times[np.isfinite(valid_event_times)])
        att_sum = 0
        for e in unique_event_times:
            mask = (event_times == e) & post_treatment_mask
            if mask.any():
                att_sum += np.mean(att_gt[mask])
        expected_att = att_sum / len(unique_event_times) if len(unique_event_times) > 0 else 0
    elif agg_type == "group":
        unique_groups = np.unique(groups)
        group_att = []
        for g in unique_groups:
            mask = (groups == g) & (g <= times) & (times <= g + np.inf)
            if mask.any():
                group_att.append(np.mean(att_gt[mask]))
        expected_att = np.mean(group_att)
    elif agg_type == "calendar":
        unique_times = np.unique(times)
        att_sum = 0
        count = 0
        for t in unique_times:
            if np.any(post_treatment & (times == t)):
                att_sum += np.mean(att_gt[(times == t) & post_treatment])
                count += 1
        expected_att = att_sum / count
    else:
        raise ValueError(f"Unexpected aggregation type: {agg_type}")

    assert np.allclose(result.overall_att, expected_att)


def test_clustered_standard_errors(mpdta_mp_result):
    result_no_cluster = compute_aggte(
        mpdta_mp_result,
        aggregation_type="group",
        bootstrap=True,
        biters=50,
        random_state=42,
    )

    result_with_cluster = compute_aggte(
        mpdta_mp_result,
        aggregation_type="group",
        bootstrap=True,
        biters=50,
        clustervars=["cluster"],
        random_state=42,
    )

    assert np.allclose(result_no_cluster.overall_att, result_with_cluster.overall_att)
    assert result_with_cluster.overall_se > 0
    assert not np.isnan(result_with_cluster.overall_se)
    assert result_no_cluster.overall_se != result_with_cluster.overall_se
    assert result_no_cluster.estimation_params["cluster"] is None
    assert result_with_cluster.estimation_params["clustervars"] == ["cluster"]
    np.testing.assert_array_equal(
        result_with_cluster.estimation_params["cluster"], mpdta_mp_result.estimation_params["data"].cluster
    )


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
