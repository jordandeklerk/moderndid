# pylint: disable=redefined-outer-name
"""Tests for Aggregated Treatment Effect."""

import numpy as np
import pytest

from pydid import compute_aggte, compute_att_gt, load_mpdta, mp, preprocess_did


@pytest.fixture
def mpdta_mp_result():
    mpdta = load_mpdta()
    mpdta["first_treat"] = mpdta["first.treat"].replace(0, np.inf)
    mpdta["cluster"] = mpdta["countyreal"] % 10

    data = preprocess_did(
        mpdta,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first_treat",
        xformla="~lpop",
        panel=True,
        control_group="nevertreated",
        anticipation=0,
        clustervars=["cluster"],
    )

    att_result = compute_att_gt(data)

    groups = np.array([r.group for r in att_result.attgt_list])
    times = np.array([r.year for r in att_result.attgt_list])
    att_gt = np.array([r.att for r in att_result.attgt_list])

    inf_func = att_result.influence_functions.toarray()
    n_units = data.config.id_count

    vcov = np.cov(inf_func.T)
    se_gt = np.sqrt(np.diag(vcov) / n_units)

    mp_result = mp(
        groups=groups,
        times=times,
        att_gt=att_gt,
        vcov_analytical=vcov,
        se_gt=se_gt,
        critical_value=1.96,
        influence_func=inf_func,
        n_units=n_units,
        alpha=0.05,
        estimation_params={
            "bootstrap": False,
            "uniform_bands": False,
            "control_group": "nevertreated",
            "anticipation_periods": 0,
            "estimation_method": "dr",
            "data": data,
            "panel": True,
        },
    )

    return mp_result


@pytest.fixture
def synthetic_mp_result():
    groups = []
    times = []
    att_gt = []

    for g in [2004, 2006, 2007]:
        for t in [2003, 2004, 2005, 2006, 2007]:
            groups.append(g)
            times.append(t)
            if g <= t:
                att_gt.append(0.05 + 0.01 * (t - g))
            else:
                att_gt.append(0.0)

    groups = np.array(groups)
    times = np.array(times)
    att_gt = np.array(att_gt)

    n_units = 500
    n_gt = len(groups)
    inf_func = np.random.normal(0, 0.01, size=(n_units, n_gt))

    vcov = np.cov(inf_func.T)
    se_gt = np.sqrt(np.diag(vcov) / n_units)

    return mp(
        groups=groups,
        times=times,
        att_gt=att_gt,
        vcov_analytical=vcov,
        se_gt=se_gt,
        critical_value=1.96,
        influence_func=inf_func,
        n_units=n_units,
        alpha=0.05,
        estimation_params={
            "bootstrap": False,
            "uniform_bands": False,
            "control_group": "nevertreated",
            "anticipation_periods": 0,
            "estimation_method": "dr",
        },
    )


@pytest.mark.parametrize("agg_type", ["simple", "dynamic", "group", "calendar"])
def test_aggregation_types(mpdta_mp_result, agg_type):
    result = compute_aggte(mpdta_mp_result, aggregation_type=agg_type)

    assert result.aggregation_type == agg_type
    assert result.overall_att is not None
    assert result.overall_se is not None
    assert result.influence_func is not None
    assert result.influence_func.shape == (mpdta_mp_result.n_units,)

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

    expected_event_times = np.unique(mpdta_mp_result.times - mpdta_mp_result.groups + 1)
    expected_event_times = expected_event_times[np.isfinite(expected_event_times)]

    assert len(result.event_times) == len(expected_event_times)
    assert np.allclose(np.sort(result.event_times), np.sort(expected_event_times))


def test_group_aggregation_groups(mpdta_mp_result):
    result = compute_aggte(mpdta_mp_result, aggregation_type="group")

    unique_groups = np.unique(mpdta_mp_result.groups)
    assert len(result.event_times) == len(unique_groups)
    assert np.allclose(result.event_times, unique_groups)


def test_calendar_aggregation_periods(mpdta_mp_result):
    result = compute_aggte(mpdta_mp_result, aggregation_type="calendar")

    min_group = mpdta_mp_result.groups.min()
    expected_times = np.unique(mpdta_mp_result.times)
    expected_times = expected_times[expected_times >= min_group]

    assert len(result.event_times) == len(expected_times)
    assert np.allclose(result.event_times, expected_times)


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
        compute_aggte(synthetic_mp_result, aggregation_type="simple", dropna=False)

    result = compute_aggte(synthetic_mp_result, aggregation_type="simple", dropna=True)
    assert not np.isnan(result.overall_att)
    assert not np.isnan(result.overall_se)


@pytest.mark.parametrize("bstrap,cband", [(True, True), (True, False), (False, False)])
def test_bootstrap_options(synthetic_mp_result, bstrap, cband):
    result = compute_aggte(
        synthetic_mp_result,
        aggregation_type="group",
        bootstrap=bstrap,
        bootstrap_iterations=99,
        confidence_band=cband,
    )

    assert result.estimation_params["bootstrap"] == bstrap
    assert result.estimation_params["uniform_bands"] == cband

    if cband and result.critical_values is not None:
        assert len(result.critical_values) == len(result.event_times)


@pytest.mark.parametrize("alp", [0.01, 0.05, 0.10])
def test_significance_levels(synthetic_mp_result, alp):
    result = compute_aggte(synthetic_mp_result, aggregation_type="simple", alpha=alp)

    stored_alpha = result.estimation_params.get("alpha", synthetic_mp_result.alpha)
    assert stored_alpha == alp


def test_invalid_aggregation_type(synthetic_mp_result):
    with pytest.raises(ValueError, match="must be one of"):
        compute_aggte(synthetic_mp_result, aggregation_type="invalid")


def test_single_group_edge_case():
    groups = np.array([2004, 2004, 2004])
    times = np.array([2003, 2004, 2005])
    att_gt = np.array([0.0, 0.05, 0.06])

    inf_func = np.random.normal(0, 0.01, size=(100, 3))
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

    result = compute_aggte(mp_result, aggregation_type="group")
    assert len(result.event_times) == 1
    assert result.event_times[0] == 2004


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
    )

    result = compute_aggte(mp_result, aggregation_type="simple")
    assert np.isnan(result.overall_att) or result.overall_att == 0


def test_all_positive_standard_errors(mpdta_mp_result):
    for agg_type in ["simple", "group", "dynamic", "calendar"]:
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
        event_times = times - groups + 1
        post_treatment_events = event_times >= 0
        unique_event_times = np.unique(event_times[post_treatment_events])
        att_sum = 0
        for e in unique_event_times:
            att_sum += np.mean(att_gt[event_times == e])
        expected_att = att_sum / len(unique_event_times)
    elif agg_type == "group":
        unique_groups = np.unique(groups)
        group_att = []
        for g in unique_groups:
            mask = (groups == g) & ((g - 1) <= times) & (times <= g + np.inf)
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
        bootstrap_iterations=100,
    )

    result_with_cluster = compute_aggte(
        mpdta_mp_result,
        aggregation_type="group",
        bootstrap=True,
        bootstrap_iterations=100,
        clustervars=["cluster"],
    )

    assert np.allclose(result_no_cluster.overall_att, result_with_cluster.overall_att)

    assert result_no_cluster.overall_se != result_with_cluster.overall_se

    assert result_with_cluster.overall_se > 0
    assert not np.isnan(result_with_cluster.overall_se)
