"""Tests for DDD aggregate treatment effects."""

import re
import warnings

import numpy as np
import polars as pl
import pytest
from scipy import stats

from moderndid import agg_ddd, ddd, mboot_ddd


@pytest.mark.parametrize("agg_type", ["simple", "eventstudy", "group", "calendar"])
def test_agg_ddd_basic(mp_ddd_result, agg_type):
    result = agg_ddd(mp_ddd_result, type=agg_type, boot=False, cband=False)

    assert result.aggregation_type == agg_type
    assert isinstance(result.overall_att, float | np.floating)
    assert isinstance(result.overall_se, float | np.floating)


@pytest.mark.parametrize("agg_type", ["simple", "eventstudy", "group", "calendar"])
def test_agg_ddd_result_structure(mp_ddd_result, agg_type):
    result = agg_ddd(mp_ddd_result, type=agg_type, boot=False, cband=False)

    assert hasattr(result, "overall_att")
    assert hasattr(result, "overall_se")
    assert hasattr(result, "aggregation_type")
    assert hasattr(result, "egt")
    assert hasattr(result, "att_egt")
    assert hasattr(result, "se_egt")
    assert hasattr(result, "crit_val")
    assert hasattr(result, "inf_func")
    assert hasattr(result, "inf_func_overall")
    assert hasattr(result, "args")

    if agg_type == "simple":
        assert result.egt is None
        assert result.att_egt is None
        assert result.se_egt is None
    else:
        assert result.egt is not None
        assert result.att_egt is not None
        assert result.se_egt is not None
        assert len(result.egt) == len(result.att_egt)
        assert len(result.egt) == len(result.se_egt)


def test_agg_ddd_invalid_type(mp_ddd_result):
    with pytest.raises(ValueError, match="Must be one of"):
        agg_ddd(mp_ddd_result, type="invalid")


def test_agg_ddd_cband_requires_boot(mp_ddd_result):
    with pytest.raises(ValueError, match="cband=True requires boot=True"):
        agg_ddd(mp_ddd_result, type="group", cband=True, boot=False)


def test_agg_ddd_eventstudy_balance(mp_ddd_result):
    result = agg_ddd(mp_ddd_result, type="eventstudy", balance_e=1, boot=False, cband=False)

    assert result.aggregation_type == "eventstudy"
    assert result.args.get("balance_e") == 1
    assert isinstance(result.overall_att, float | np.floating)


def test_agg_ddd_eventstudy_min_max_e(mp_ddd_result):
    result = agg_ddd(mp_ddd_result, type="eventstudy", min_e=-1, max_e=2, boot=False, cband=False)

    assert result.aggregation_type == "eventstudy"
    assert result.args.get("min_e") == -1
    assert result.args.get("max_e") == 2


@pytest.mark.parametrize("agg_type", ["simple", "eventstudy", "group", "calendar"])
def test_agg_ddd_na_rm_drops_missing_cells(mp_ddd_result, agg_type):
    missing = (mp_ddd_result.groups == 3) & (mp_ddd_result.times == 4)
    with_nan = mp_ddd_result._replace(att=np.where(missing, np.nan, mp_ddd_result.att))
    hand_dropped = mp_ddd_result._replace(
        att=mp_ddd_result.att[~missing],
        groups=mp_ddd_result.groups[~missing],
        times=mp_ddd_result.times[~missing],
        inf_func_mat=mp_ddd_result.inf_func_mat[:, ~missing],
    )

    result = agg_ddd(with_nan, type=agg_type, na_rm=True, boot=False, cband=False)
    expected = agg_ddd(hand_dropped, type=agg_type, boot=False, cband=False)
    complete = agg_ddd(mp_ddd_result, type=agg_type, boot=False, cband=False)

    assert result.aggregation_type == agg_type
    np.testing.assert_allclose(result.overall_att, expected.overall_att, rtol=1e-12)
    np.testing.assert_allclose(result.overall_se, expected.overall_se, rtol=1e-12)
    assert abs(result.overall_att - complete.overall_att) > 1e-3
    if agg_type != "simple":
        np.testing.assert_array_equal(result.egt, expected.egt)
        np.testing.assert_allclose(result.att_egt, expected.att_egt, rtol=1e-12)
        np.testing.assert_allclose(result.se_egt, expected.se_egt, rtol=1e-12)


def test_agg_ddd_influence_function(mp_ddd_result):
    result = agg_ddd(mp_ddd_result, type="simple", boot=False, cband=False)

    assert result.inf_func_overall is not None
    assert isinstance(result.inf_func_overall, np.ndarray)
    assert len(result.inf_func_overall) == mp_ddd_result.n


def test_agg_ddd_inf_func_eventstudy(mp_ddd_result):
    result = agg_ddd(mp_ddd_result, type="eventstudy", boot=False, cband=False)

    if result.inf_func is not None:
        assert result.inf_func.shape[0] == mp_ddd_result.n
        assert result.inf_func.shape[1] == len(result.egt)


def test_agg_ddd_inf_func_group(mp_ddd_result):
    result = agg_ddd(mp_ddd_result, type="group", boot=False, cband=False)

    if result.inf_func is not None:
        assert result.inf_func.shape[0] == mp_ddd_result.n
        assert result.inf_func.shape[1] == len(result.egt)


def test_agg_ddd_inf_func_calendar(mp_ddd_result):
    result = agg_ddd(mp_ddd_result, type="calendar", boot=False, cband=False)

    if result.inf_func is not None:
        assert result.inf_func.shape[0] == mp_ddd_result.n
        assert result.inf_func.shape[1] == len(result.egt)


@pytest.mark.parametrize("agg_type", ["eventstudy", "group", "calendar"])
def test_agg_ddd_bootstrap(mp_ddd_result, agg_type):
    result = agg_ddd(mp_ddd_result, type=agg_type, boot=True, biters=50, cband=False)

    assert result.aggregation_type == agg_type
    assert result.args.get("boot") is True
    assert result.args.get("biters") == 50


@pytest.mark.parametrize("agg_type", ["eventstudy", "group", "calendar"])
def test_agg_ddd_cband(mp_ddd_result, agg_type):
    result = agg_ddd(mp_ddd_result, type=agg_type, boot=True, biters=50, cband=True)

    assert result.args.get("cband") is True
    assert result.crit_val is not None


@pytest.mark.parametrize("agg_type", ["simple", "eventstudy", "group", "calendar"])
def test_agg_ddd_reproducibility(mp_ddd_result, agg_type):
    result1 = agg_ddd(
        mp_ddd_result,
        type=agg_type,
        boot=True,
        biters=50,
        cband=False,
        random_state=42,
    )
    result2 = agg_ddd(
        mp_ddd_result,
        type=agg_type,
        boot=True,
        biters=50,
        cband=False,
        random_state=42,
    )

    assert result1.overall_se == result2.overall_se

    if agg_type != "simple" and result1.se_egt is not None:
        np.testing.assert_array_equal(result1.se_egt, result2.se_egt)


def test_agg_ddd_args_stored(mp_ddd_result):
    result = agg_ddd(
        mp_ddd_result,
        type="eventstudy",
        boot=True,
        biters=100,
        cband=True,
        alpha=0.05,
    )

    assert result.args["aggregation_type"] == "eventstudy"
    assert result.args["boot"] is True
    assert result.args["biters"] == 100
    assert result.args["cband"] is True
    assert result.args["alpha"] == 0.05


def test_agg_ddd_alpha(mp_ddd_result):
    result = agg_ddd(mp_ddd_result, type="simple", alpha=0.10, boot=False, cband=False)

    assert result.args.get("alpha") == 0.10


def test_agg_ddd_group_structure(mp_ddd_result):
    result = agg_ddd(mp_ddd_result, type="group", boot=False, cband=False)

    assert result.aggregation_type == "group"
    assert result.egt is not None

    for g in result.egt:
        assert g in mp_ddd_result.glist


def test_agg_ddd_calendar_structure(mp_ddd_result):
    result = agg_ddd(mp_ddd_result, type="calendar", boot=False, cband=False)

    assert result.aggregation_type == "calendar"
    assert result.egt is not None


def test_agg_ddd_eventstudy_structure(mp_ddd_result):
    result = agg_ddd(mp_ddd_result, type="eventstudy", boot=False, cband=False)

    assert result.aggregation_type == "eventstudy"
    assert result.egt is not None

    if len(result.egt) > 0:
        sorted_egt = np.sort(result.egt)
        np.testing.assert_array_equal(result.egt, sorted_egt)


def test_agg_ddd_simple_no_disaggregated(mp_ddd_result):
    result = agg_ddd(mp_ddd_result, type="simple", boot=False, cband=False)

    assert result.egt is None
    assert result.att_egt is None
    assert result.se_egt is None
    assert result.inf_func is None
    assert result.inf_func_overall is not None


def test_agg_ddd_missing_values_error(mp_ddd_result):
    mp_ddd_result.att[0] = np.nan

    with pytest.raises(ValueError, match="Missing values"):
        agg_ddd(mp_ddd_result, type="simple", na_rm=False, boot=False, cband=False)


def test_agg_ddd_without_na_rm_names_the_argument(mp_ddd_result):
    with_nan = mp_ddd_result._replace(att=np.where(mp_ddd_result.times == 4, np.nan, mp_ddd_result.att))

    with pytest.raises(ValueError, match=r"na_rm=True"):
        agg_ddd(with_nan, boot=False, cband=False)


def test_agg_ddd_rejects_the_removed_dropna_argument(mp_ddd_result):
    with pytest.raises(TypeError, match="unexpected keyword argument 'dropna'"):
        agg_ddd(mp_ddd_result, type="simple", dropna=True, boot=False, cband=False)


def test_agg_ddd_eventstudy_no_post(mp_ddd_result):
    with pytest.warns(
        UserWarning,
        match="^The overall effect is NaN because max_e=-1 leaves out every post-treatment group-time effect\\. "
        "Pass a larger max_e to estimate it\\.$",
    ):
        result = agg_ddd(mp_ddd_result, type="eventstudy", max_e=-1, boot=False, cband=False)

    assert np.isnan(result.overall_att)
    assert np.isnan(result.overall_se)
    np.testing.assert_array_equal(result.egt, [-3, -2, -1])


@pytest.mark.parametrize(
    "agg_type,expected_text",
    [
        ("simple", "Aggregate DDD Treatment Effects"),
        ("eventstudy", "Event Study"),
        ("group", "Group/Cohort"),
        ("calendar", "Calendar Time"),
    ],
)
def test_agg_ddd_print(mp_ddd_result, agg_type, expected_text):
    result = agg_ddd(mp_ddd_result, type=agg_type, boot=False, cband=False)
    output = str(result)

    assert "Aggregate DDD Treatment Effects" in output
    assert expected_text in output


@pytest.mark.parametrize("agg_type", ["simple", "eventstudy", "group", "calendar"])
def test_agg_ddd_weighted_shares_match_replicated_units(mp_weighted_df, mp_weighted_replicated_df, agg_type):
    spec = {"yname": "y", "tname": "time", "idname": "id", "gname": "group", "pname": "partition", "est_method": "reg"}
    weighted = agg_ddd(ddd(data=mp_weighted_df, weightsname="w", **spec), type=agg_type, boot=False, cband=False)
    replicated = agg_ddd(ddd(data=mp_weighted_replicated_df, **spec), type=agg_type, boot=False, cband=False)

    np.testing.assert_allclose(weighted.overall_att, replicated.overall_att, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("agg_type", ["simple", "eventstudy", "group", "calendar"])
def test_agg_ddd_analytic_standard_errors_sum_within_clusters(mp_three_cohort_df, agg_type):
    result = ddd(
        data=mp_three_cohort_df,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        cluster="cluster",
    )
    agg = agg_ddd(result, type=agg_type, boot=False, cband=False)
    clusters = mp_three_cohort_df.filter(pl.col("time") == 1).sort("id")["cluster"].to_numpy()
    _, cluster_idx = np.unique(clusters, return_inverse=True)
    columns = agg.inf_func_overall[:, None] if agg.inf_func is None else agg.inf_func
    sums = np.stack([np.bincount(cluster_idx, weights=column) for column in columns.T], axis=1)
    expected = np.sqrt(np.sum(sums**2, axis=0)) / result.n
    expected[expected < 1e-7] = np.nan
    overall_sums = np.bincount(cluster_idx, weights=agg.inf_func_overall)

    np.testing.assert_allclose(agg.overall_se, np.sqrt(np.sum(overall_sums**2)) / result.n, rtol=1e-12)
    if agg.se_egt is not None:
        np.testing.assert_allclose(agg.se_egt, expected, rtol=1e-12)
    assert agg.args["cluster"] == "cluster"


def test_agg_ddd_bootstrap_draws_one_multiplier_per_cluster(mp_three_cohort_df):
    result = ddd(
        data=mp_three_cohort_df,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        cluster="cluster",
    )
    agg = agg_ddd(result, type="eventstudy", biters=199, cband=False, random_state=3)
    clusters = mp_three_cohort_df.filter(pl.col("time") == 1).sort("id")["cluster"].to_numpy()
    expected = mboot_ddd(agg.inf_func, 199, 0.05, cluster=clusters, random_state=3)

    np.testing.assert_array_equal(agg.se_egt, expected.se)


def test_agg_ddd_leaves_the_few_clusters_warning_to_ddd(mp_clustered_df):
    with warnings.catch_warnings(record=True) as at_fit:
        warnings.simplefilter("always")
        result = ddd(
            data=mp_clustered_df,
            yname="y",
            tname="time",
            idname="id",
            gname="group",
            pname="partition",
            est_method="reg",
            cluster="cluster",
        )
    with warnings.catch_warnings(record=True) as at_aggregation:
        warnings.simplefilter("always")
        booted = [agg_ddd(result, type=agg_type, biters=99, random_state=3) for agg_type in ("simple", "eventstudy")]
        analytical = agg_ddd(result, type="group", boot=False, cband=False)

    assert len([w for w in at_fit if "very few clusters" in str(w.message)]) == 1
    assert not [w for w in at_aggregation if "very few clusters" in str(w.message)]
    for agg in booted:
        assert " Bootstrap standard errors clustered by cluster" in str(agg).splitlines()
        assert agg.__maketables_stat__("se_type") == "Bootstrap clustered by cluster"
    assert " Analytical standard errors clustered by cluster" in str(analytical).splitlines()
    assert analytical.__maketables_vcov_info__ == {"vcov_type": "analytical", "clustervar": "cluster"}


@pytest.mark.parametrize("boot", [False, True])
def test_agg_ddd_calendar_overall_se_averages_periods_with_cells(mp_ddd_result, boot):
    empty = (mp_ddd_result.groups == 3) & (mp_ddd_result.times == 3)
    gapped = mp_ddd_result._replace(att=np.where(empty, np.nan, mp_ddd_result.att))
    agg = agg_ddd(gapped, type="calendar", na_rm=True, boot=boot, cband=False, biters=199, random_state=3)
    valid = np.isfinite(agg.att_egt)
    if boot:
        draws = mboot_ddd(agg.inf_func, 199, 0.05, random_state=3).bres[:, valid].mean(axis=1)
        quartiles = np.percentile(draws, [75, 25], method="inverted_cdf")
        expected = (quartiles[0] - quartiles[1]) / 1.3489795 / np.sqrt(mp_ddd_result.n)
    else:
        expected = np.sqrt(np.mean(agg.inf_func[:, valid].mean(axis=1) ** 2) / mp_ddd_result.n)

    np.testing.assert_array_equal(agg.egt, [4, 5])
    np.testing.assert_array_equal(valid, [True, True])
    np.testing.assert_allclose(agg.overall_att, np.mean(agg.att_egt[valid]), rtol=1e-12)
    np.testing.assert_allclose(agg.overall_se, expected, rtol=1e-12)


@pytest.mark.parametrize("agg_type", ["simple", "eventstudy", "group", "calendar"])
@pytest.mark.parametrize("base_period", ["universal", "varying"])
def test_agg_ddd_without_never_treated_units_matches_the_trimmed_panel(mp_no_never_treated_df, agg_type, base_period):
    spec = {
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "group",
        "pname": "partition",
        "control_group": "notyettreated",
        "base_period": base_period,
        "est_method": "reg",
    }
    result = agg_ddd(ddd(data=mp_no_never_treated_df, **spec), type=agg_type, boot=False, cband=False)
    trimmed = ddd(data=mp_no_never_treated_df.filter(pl.col("time") < 4), **spec)
    expected = agg_ddd(trimmed, type=agg_type, boot=False, cband=False)

    np.testing.assert_allclose(result.overall_att, expected.overall_att, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(result.overall_se, expected.overall_se, rtol=1e-10, atol=1e-10)
    if expected.egt is not None:
        np.testing.assert_array_equal(result.egt, expected.egt)
        np.testing.assert_allclose(result.att_egt, expected.att_egt, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(result.se_egt, expected.se_egt, rtol=1e-10, atol=1e-10)
    assert str(result) == str(expected)


def test_agg_ddd_event_study_weights_cohorts_by_eligible_treated_units(
    mp_three_cohort_reg_result, mp_three_cohort_units
):
    result = mp_three_cohort_reg_result
    cohorts, eligible = mp_three_cohort_units
    agg = agg_ddd(result, type="eventstudy", boot=False, cband=False)
    shares = np.array([np.mean(eligible & (cohorts == g)) for g in result.groups])
    pooled = np.array([np.mean(cohorts == g) for g in result.groups])
    event = result.times - result.groups
    expected = np.array([np.average(result.att[event == e], weights=shares[event == e]) for e in agg.egt])
    pooled_es = np.array([np.average(result.att[event == e], weights=pooled[event == e]) for e in agg.egt])

    np.testing.assert_allclose(agg.att_egt, expected, rtol=1e-12)
    np.testing.assert_allclose(agg.overall_att, expected[agg.egt >= 0].mean(), rtol=1e-12)
    assert abs(pooled_es[agg.egt >= 0].mean() - agg.overall_att) > 1


def test_agg_ddd_simple_weights_cells_by_eligible_treated_units(mp_three_cohort_reg_result, mp_three_cohort_units):
    result = mp_three_cohort_reg_result
    cohorts, eligible = mp_three_cohort_units
    agg = agg_ddd(result, type="simple", boot=False, cband=False)
    post = result.times >= result.groups
    shares = np.array([np.mean(eligible & (cohorts == g)) for g in result.groups[post]])
    pooled = np.array([np.mean(cohorts == g) for g in result.groups[post]])

    np.testing.assert_allclose(agg.overall_att, np.average(result.att[post], weights=shares), rtol=1e-12)
    assert abs(np.average(result.att[post], weights=pooled) - agg.overall_att) > 1


def test_agg_ddd_group_overall_weights_cohorts_by_eligible_treated_units(
    mp_three_cohort_reg_result, mp_three_cohort_units
):
    result = mp_three_cohort_reg_result
    cohorts, eligible = mp_three_cohort_units
    agg = agg_ddd(result, type="group", boot=False, cband=False)
    post = result.times >= result.groups
    means = np.array([result.att[post & (result.groups == g)].mean() for g in result.glist])
    shares = np.array([np.mean(eligible & (cohorts == g)) for g in result.glist])
    pooled = np.array([np.mean(cohorts == g) for g in result.glist])

    np.testing.assert_array_equal(agg.egt, result.glist)
    np.testing.assert_allclose(agg.att_egt, means, rtol=1e-12)
    np.testing.assert_allclose(agg.overall_att, np.average(means, weights=shares), rtol=1e-12)
    assert abs(np.average(means, weights=pooled) - agg.overall_att) > 1


def test_agg_ddd_calendar_weights_cohorts_by_eligible_treated_units(mp_three_cohort_reg_result, mp_three_cohort_units):
    result = mp_three_cohort_reg_result
    cohorts, eligible = mp_three_cohort_units
    agg = agg_ddd(result, type="calendar", boot=False, cband=False)
    shares = np.array([np.mean(eligible & (cohorts == g)) for g in result.groups])
    pooled = np.array([np.mean(cohorts == g) for g in result.groups])
    cells = [(result.times == t) & (result.groups <= t) for t in agg.egt]
    expected = np.array([np.average(result.att[cell], weights=shares[cell]) for cell in cells])
    pooled_calendar = np.array([np.average(result.att[cell], weights=pooled[cell]) for cell in cells])

    np.testing.assert_array_equal(agg.egt, [2, 3, 4, 5])
    np.testing.assert_allclose(agg.att_egt, expected, rtol=1e-12)
    np.testing.assert_allclose(agg.overall_att, expected.mean(), rtol=1e-12)
    assert abs(pooled_calendar.mean() - agg.overall_att) > 1


def test_agg_ddd_repeated_cross_sections_weight_cohorts_by_eligible_observations(mp_rcs_data):
    result = ddd(
        data=mp_rcs_data,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method="reg",
        panel=False,
    )
    agg = agg_ddd(result, type="eventstudy", boot=False, cband=False)
    cohorts = mp_rcs_data["group"].to_numpy()
    eligible = mp_rcs_data["partition"].to_numpy() == 1
    shares = np.array([np.mean(eligible & (cohorts == g)) for g in result.groups])
    event = result.times - result.groups
    expected = np.array([np.average(result.att[event == e], weights=shares[event == e]) for e in agg.egt])

    np.testing.assert_allclose(agg.att_egt, expected, rtol=1e-12)
    np.testing.assert_allclose(agg.overall_att, expected[agg.egt >= 0].mean(), rtol=1e-12)


def test_agg_ddd_requires_unit_partition(ddd_mp_result_fixture):
    with pytest.raises(ValueError, match="ddd_result has no unit_partition"):
        agg_ddd(ddd_mp_result_fixture, boot=False, cband=False)


@pytest.mark.parametrize("na_rm", [False, True])
@pytest.mark.parametrize("agg_type", ["simple", "eventstudy", "group", "calendar"])
def test_agg_ddd_without_eligible_cohort_units_raises(ddd_mp_share_result, agg_type, na_rm):
    base_periods_only = ddd_mp_share_result._replace(
        att=np.zeros(2),
        groups=np.array([2, 3]),
        times=np.array([1, 2]),
        inf_func_mat=np.zeros((10, 2)),
        unit_partition=np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
    )

    with pytest.raises(ValueError, match="No cohort in ddd_result has an eligible unit"):
        agg_ddd(base_periods_only, type=agg_type, na_rm=na_rm, boot=False, cband=False)


@pytest.mark.parametrize("na_rm", [False, True])
def test_agg_ddd_group_leaves_out_cohorts_without_post_treatment_effects(ddd_mp_share_result, na_rm):
    keep = np.array([True, True, True, True, True, False])
    inf_func_mat = np.random.default_rng(0).standard_normal((10, 5))
    gapped = ddd_mp_share_result._replace(
        att=ddd_mp_share_result.att[keep],
        groups=ddd_mp_share_result.groups[keep],
        times=ddd_mp_share_result.times[keep],
        inf_func_mat=inf_func_mat,
    )
    group = agg_ddd(gapped, type="group", na_rm=na_rm, boot=False, cband=False)
    event_study = agg_ddd(gapped, type="eventstudy", boot=False, cband=False)
    simple = agg_ddd(gapped, type="simple", boot=False, cband=False)
    expected_inf_func = inf_func_mat[:, [1, 2]].mean(axis=1)

    np.testing.assert_array_equal(group.egt, [2])
    np.testing.assert_allclose(group.att_egt, [1.25], rtol=1e-12)
    np.testing.assert_allclose(group.inf_func[:, 0], expected_inf_func, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(group.se_egt, [np.sqrt(np.mean(expected_inf_func**2) / 10)], rtol=1e-12)
    np.testing.assert_allclose(group.overall_att, 1.25, rtol=1e-12)
    np.testing.assert_allclose(event_study.overall_att, 1.25, rtol=1e-12)
    np.testing.assert_allclose(simple.overall_att, 1.25, rtol=1e-12)
    np.testing.assert_allclose(group.overall_se, group.se_egt[0], rtol=1e-12)
    np.testing.assert_allclose(group.overall_se, event_study.overall_se, rtol=1e-12)


@pytest.mark.parametrize("na_rm", [False, True])
def test_agg_ddd_group_overall_without_post_treatment_window_is_nan(mp_ddd_result, na_rm):
    with pytest.warns(UserWarning, match="^The overall effect is NaN because max_e=-1 leaves out every "):
        result = agg_ddd(mp_ddd_result, type="group", max_e=-1, na_rm=na_rm, boot=False, cband=False)

    assert np.isnan(result.overall_att)
    assert np.isnan(result.overall_se)
    np.testing.assert_array_equal(result.inf_func_overall, np.zeros(mp_ddd_result.n))
    assert len(result.egt) == 0


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_agg_ddd_with_only_pre_treatment_cells_reports_nan_overall_effects_without_runtime_warnings(
    ddd_mp_share_result,
):
    pre_only = ddd_mp_share_result._replace(
        att=np.array([0.2, 0.0]),
        groups=np.array([3, 3]),
        times=np.array([1, 2]),
        glist=np.array([3]),
        inf_func_mat=np.zeros((10, 2)),
    )
    with pytest.warns(UserWarning, match="^The overall effect is NaN because no post-treatment ") as record:
        aggs = {kind: agg_ddd(pre_only, type=kind, boot=False, cband=False) for kind in ("simple", "group", "calendar")}
        event_study = agg_ddd(pre_only, type="eventstudy", boot=False, cband=False)

    assert len(record) == 4
    assert all(np.isnan(agg.overall_att) for agg in aggs.values())
    assert all(np.isnan(agg.overall_se) for agg in aggs.values())
    assert len(aggs["group"].egt) == 0
    assert len(aggs["calendar"].egt) == 0
    np.testing.assert_array_equal(event_study.egt, [-2, -1])
    np.testing.assert_allclose(event_study.att_egt, [0.2, 0.0], rtol=1e-12)
    assert np.isnan(event_study.overall_att)


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize(
    ("unit_partition", "overall", "cells"),
    [
        ([0, 1, 0, 1, 1, 1, 0, 0, 0, 0], 1.25, [1, 2]),
        ([0, 1, 0, 1, 0, 0, 0, 1, 1, 0], 4.0, [5]),
    ],
)
@pytest.mark.parametrize("agg_type", ["simple", "eventstudy", "group", "calendar"])
def test_agg_ddd_overall_effect_ignores_cohort_without_eligible_units_without_runtime_warnings(
    ddd_mp_share_result, agg_type, unit_partition, overall, cells
):
    inf_func_mat = np.random.default_rng(0).standard_normal((10, 6))
    result = ddd_mp_share_result._replace(inf_func_mat=inf_func_mat, unit_partition=np.array(unit_partition))
    agg = agg_ddd(result, type=agg_type, boot=False, cband=False)
    expected_inf_func = inf_func_mat[:, cells].mean(axis=1)

    np.testing.assert_allclose(agg.overall_att, overall, rtol=1e-12)
    np.testing.assert_allclose(agg.inf_func_overall, expected_inf_func, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(agg.overall_se, np.sqrt(np.mean(expected_inf_func**2) / 10), rtol=1e-12)


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize(
    ("unit_partition", "agg_type", "effects"),
    [
        ([0, 1, 0, 1, 1, 1, 0, 0, 0, 0], "eventstudy", [np.nan, 0.0, 1.0, 1.5]),
        ([0, 1, 0, 1, 1, 1, 0, 0, 0, 0], "group", [1.25, np.nan]),
        ([0, 1, 0, 1, 0, 0, 0, 1, 1, 0], "eventstudy", [0.2, 0.0, 4.0, np.nan]),
        ([0, 1, 0, 1, 0, 0, 0, 1, 1, 0], "group", [np.nan, 4.0]),
        ([0, 1, 0, 1, 0, 0, 0, 1, 1, 0], "calendar", [np.nan, 4.0]),
    ],
)
def test_agg_ddd_effect_of_only_cohorts_without_eligible_units_is_nan_without_runtime_warnings(
    ddd_mp_share_result, unit_partition, agg_type, effects
):
    result = ddd_mp_share_result._replace(
        inf_func_mat=np.random.default_rng(0).standard_normal((10, 6)), unit_partition=np.array(unit_partition)
    )
    agg = agg_ddd(result, type=agg_type, boot=False, cband=False)
    missing = np.isnan(effects)

    np.testing.assert_allclose(agg.att_egt, effects, rtol=1e-12)
    np.testing.assert_array_equal(agg.inf_func[:, missing], 0.0)
    assert np.all(np.isnan(agg.se_egt[missing]))


def test_agg_ddd_event_study_bootstrap_overall_se_ignores_event_time_without_eligible_units(ddd_mp_share_result):
    result = ddd_mp_share_result._replace(
        inf_func_mat=np.random.default_rng(0).standard_normal((10, 6)),
        unit_partition=np.array([0, 1, 0, 1, 0, 0, 0, 1, 1, 0]),
    )
    agg = agg_ddd(result, type="eventstudy", boot=True, biters=99, cband=False, random_state=7)

    np.testing.assert_allclose(agg.overall_att, 4.0, rtol=1e-12)
    np.testing.assert_allclose(agg.overall_se, agg.se_egt[agg.egt == 0][0], rtol=1e-12)


@pytest.mark.parametrize("na_rm", [False, True])
@pytest.mark.parametrize("agg_type", ["simple", "eventstudy", "group", "calendar"])
def test_agg_ddd_without_post_treatment_effects_warns_and_reports_nan(
    ddd_mp_share_pre_treatment_result, agg_type, na_rm
):
    with pytest.warns(
        UserWarning,
        match="^The overall effect is NaN because no post-treatment group-time effect is left to average\\. The ddd "
        "warnings name the cells that failed to estimate\\. Fewer covariates or control_group='notyettreated' can "
        "recover them\\.$",
    ):
        agg = agg_ddd(ddd_mp_share_pre_treatment_result, type=agg_type, na_rm=na_rm, boot=False, cband=False)

    assert np.isnan(agg.overall_att)
    assert np.isnan(agg.overall_se)
    np.testing.assert_array_equal(agg.inf_func_overall, np.zeros(10))
    if agg_type == "eventstudy":
        np.testing.assert_array_equal(agg.egt, [-2, -1])
        np.testing.assert_allclose(agg.att_egt, [0.2, 0.0], rtol=1e-12)
    elif agg_type != "simple":
        assert len(agg.egt) == len(agg.att_egt) == len(agg.se_egt) == 0
        assert agg.inf_func.shape == (10, 0)


@pytest.mark.parametrize("agg_type", ["simple", "eventstudy", "group"])
def test_agg_ddd_max_e_before_treatment_warns_and_reports_nan(ddd_mp_share_result, agg_type):
    result = ddd_mp_share_result._replace(inf_func_mat=np.random.default_rng(0).standard_normal((10, 6)))

    with pytest.warns(
        UserWarning,
        match="^The overall effect is NaN because max_e=-1 leaves out every post-treatment group-time effect\\. "
        "Pass a larger max_e to estimate it\\.$",
    ):
        agg = agg_ddd(result, type=agg_type, max_e=-1, boot=False, cband=False)

    assert np.isnan(agg.overall_att)
    assert np.isnan(agg.overall_se)
    if agg_type == "eventstudy":
        np.testing.assert_array_equal(agg.egt, [-2, -1])
    elif agg_type == "group":
        assert len(agg.egt) == 0


@pytest.mark.parametrize("na_rm", [False, True])
def test_agg_ddd_group_table_leaves_out_cohorts_without_effects_up_to_max_e(ddd_mp_share_result, na_rm):
    keep = np.array([True, False, True, True, True, True])
    late_start = ddd_mp_share_result._replace(
        att=ddd_mp_share_result.att[keep],
        groups=ddd_mp_share_result.groups[keep],
        times=ddd_mp_share_result.times[keep],
        inf_func_mat=np.random.default_rng(0).standard_normal((10, 5)),
    )
    agg = agg_ddd(late_start, type="group", max_e=0, na_rm=na_rm, boot=False, cband=False)

    np.testing.assert_array_equal(agg.egt, [3])
    np.testing.assert_allclose(agg.att_egt, [4.0], rtol=1e-12)
    np.testing.assert_allclose(agg.overall_att, 4.0, rtol=1e-12)
    np.testing.assert_allclose(agg.overall_se, agg.se_egt[0], rtol=1e-12)


@pytest.mark.parametrize("na_rm", [False, True])
def test_agg_ddd_group_max_e_window_matches_other_types_when_a_cohort_adopts_between_observed_periods(
    ddd_mp_share_result, na_rm
):
    inf_func_mat = np.random.default_rng(0).standard_normal((10, 6))
    off_grid = ddd_mp_share_result._replace(
        att=np.array([0.1, 1.0, 2.0, 3.0, 0.2, 0.0]),
        groups=np.array([2, 2, 2, 2, 3, 3]),
        times=np.array([1, 2, 4, 5, 1, 2]),
        tlist=np.array([1, 2, 4, 5]),
        inf_func_mat=inf_func_mat,
    )
    group = agg_ddd(off_grid, type="group", max_e=1, na_rm=na_rm, boot=False, cband=False)
    simple = agg_ddd(off_grid, type="simple", max_e=1, na_rm=na_rm, boot=False, cband=False)
    event_study = agg_ddd(off_grid, type="eventstudy", max_e=1, na_rm=na_rm, boot=False, cband=False)
    expected_se = np.sqrt(np.mean(inf_func_mat[:, 1] ** 2) / 10)

    np.testing.assert_array_equal(group.egt, [2])
    np.testing.assert_allclose(group.att_egt, [1.0], rtol=1e-12)
    np.testing.assert_allclose(group.se_egt, [expected_se], rtol=1e-12)
    np.testing.assert_allclose([group.overall_att, simple.overall_att, event_study.overall_att], 1.0, rtol=1e-12)
    np.testing.assert_allclose([group.overall_se, simple.overall_se, event_study.overall_se], expected_se, rtol=1e-12)


@pytest.mark.parametrize("na_rm", [False, True])
def test_agg_ddd_group_table_of_a_fit_leaves_out_a_cohort_whose_post_treatment_cells_failed(mp_ddd_result, na_rm):
    failed = (mp_ddd_result.groups == 3) & (mp_ddd_result.times >= 3)
    result = mp_ddd_result._replace(
        att=mp_ddd_result.att[~failed],
        groups=mp_ddd_result.groups[~failed],
        times=mp_ddd_result.times[~failed],
        inf_func_mat=mp_ddd_result.inf_func_mat[:, ~failed],
    )
    agg = agg_ddd(result, type="group", na_rm=na_rm, boot=False, cband=False)
    complete = agg_ddd(mp_ddd_result, type="group", boot=False, cband=False)

    np.testing.assert_array_equal(agg.egt, [4])
    np.testing.assert_allclose(agg.att_egt, complete.att_egt[complete.egt == 4], rtol=1e-12)
    np.testing.assert_allclose(agg.se_egt, complete.se_egt[complete.egt == 4], rtol=1e-12)
    np.testing.assert_allclose(agg.overall_att, agg.att_egt[0], rtol=1e-12)
    np.testing.assert_allclose(agg.overall_se, agg.se_egt[0], rtol=1e-12)


@pytest.mark.parametrize("na_rm", [False, True])
def test_agg_ddd_calendar_leaves_out_periods_without_treated_effects(ddd_mp_share_result, na_rm):
    keep = (ddd_mp_share_result.groups == 3) | (ddd_mp_share_result.times == 1)
    inf_func_mat = np.random.default_rng(0).standard_normal((10, 4))
    early_gap = ddd_mp_share_result._replace(
        att=ddd_mp_share_result.att[keep],
        groups=ddd_mp_share_result.groups[keep],
        times=ddd_mp_share_result.times[keep],
        inf_func_mat=inf_func_mat,
    )
    agg = agg_ddd(early_gap, type="calendar", na_rm=na_rm, boot=False, cband=False)

    np.testing.assert_array_equal(agg.egt, [3])
    np.testing.assert_allclose(agg.att_egt, [4.0], rtol=1e-12)
    np.testing.assert_allclose(agg.se_egt, [np.sqrt(np.mean(inf_func_mat[:, 3] ** 2) / 10)], rtol=1e-12)
    np.testing.assert_allclose(agg.overall_att, 4.0, rtol=1e-12)
    np.testing.assert_allclose(agg.overall_se, agg.se_egt[0], rtol=1e-12)


@pytest.mark.parametrize("agg_type", ["group", "calendar"])
def test_agg_ddd_empty_table_with_bands_warns_only_about_the_overall_effect(
    ddd_mp_share_pre_treatment_result, agg_type
):
    with pytest.warns(UserWarning) as record:
        agg = agg_ddd(ddd_mp_share_pre_treatment_result, type=agg_type, biters=99, random_state=7)

    assert [str(warning.message) for warning in record] == [
        "The overall effect is NaN because no post-treatment group-time effect is left to average. The ddd warnings "
        "name the cells that failed to estimate. Fewer covariates or control_group='notyettreated' can recover them."
    ]
    np.testing.assert_allclose(agg.crit_val, stats.norm.ppf(0.975), rtol=1e-12)


@pytest.mark.parametrize("agg_type", ["simple", "eventstudy", "group", "calendar"])
def test_agg_ddd_rejects_result_whose_effects_are_all_nan(ddd_mp_share_result, agg_type):
    all_nan = ddd_mp_share_result._replace(att=np.full(6, np.nan))

    with pytest.raises(
        ValueError,
        match="^Every ATT\\(g,t\\) in ddd_result is NaN\\. The ddd warnings name the cells that failed to estimate\\. "
        "Fewer covariates or control_group='notyettreated' can recover them\\.$",
    ):
        agg_ddd(all_nan, type=agg_type, na_rm=True, boot=False, cband=False)


@pytest.mark.parametrize("agg_type", ["simple", "eventstudy", "group", "calendar"])
def test_agg_ddd_rejects_result_without_group_time_effects(ddd_mp_share_result, agg_type):
    empty = ddd_mp_share_result._replace(
        att=np.array([]), groups=np.array([], dtype=int), times=np.array([], dtype=int), inf_func_mat=np.zeros((10, 0))
    )

    with pytest.raises(
        ValueError,
        match="^ddd_result holds no group-time effects to aggregate\\. The ddd warnings name the cells that failed to "
        "estimate\\. Fewer covariates or control_group='notyettreated' can recover them\\.$",
    ):
        agg_ddd(empty, type=agg_type, boot=False, cband=False)


@pytest.mark.parametrize(
    ("window", "message"),
    [
        ({"min_e": 5, "max_e": 6}, "min_e=5 and max_e=6"),
        ({"balance_e": 2}, "min_e=-inf and max_e=inf with balance_e=2"),
    ],
)
def test_agg_ddd_rejects_event_study_window_without_event_times(ddd_mp_share_result, window, message):
    with pytest.raises(
        ValueError,
        match=f"^No group-time effect has an event time within {re.escape(message)}\\. Widen the window or check "
        "the ddd warnings for cells that failed to estimate\\.$",
    ):
        agg_ddd(ddd_mp_share_result, type="eventstudy", boot=False, cband=False, **window)
