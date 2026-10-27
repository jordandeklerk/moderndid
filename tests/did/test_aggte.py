"""Tests for aggregate treatment effects."""

import re
import warnings
from unittest.mock import patch

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid import aggte, att_gt
from moderndid.did.mboot import mboot as real_mboot


@pytest.mark.parametrize("agg_type", ["simple", "dynamic", "group", "calendar"])
def test_aggte_basic_structure(mp_result, agg_type):
    result = aggte(mp_result, type=agg_type, cband=False)

    assert result.aggregation_type == agg_type
    assert isinstance(result.overall_att, float | np.floating)
    assert isinstance(result.overall_se, float | np.floating)

    if agg_type == "simple":
        assert result.event_times is None
        assert result.att_by_event is None
        assert result.se_by_event is None
    else:
        assert result.event_times is not None
        assert result.att_by_event is not None
        assert result.se_by_event is not None
        assert len(result.event_times) == len(result.att_by_event)
        assert len(result.event_times) == len(result.se_by_event)


def test_aggte_invalid_type(mp_result):
    with pytest.raises(ValueError, match="Must be one of"):
        aggte(mp_result, type="invalid")


def test_aggte_dynamic_with_balance(mp_result):
    result = aggte(mp_result, type="dynamic", balance_e=1, cband=False)

    assert result.aggregation_type == "dynamic"
    assert result.balanced_event_threshold == 1
    assert isinstance(result.overall_att, float)
    assert isinstance(result.overall_se, float)


def test_aggte_dynamic_with_min_max_e(mp_result):
    result = aggte(mp_result, type="dynamic", min_e=-1, max_e=2, cband=False)

    assert result.aggregation_type == "dynamic"
    assert result.min_event_time == -1
    assert result.max_event_time == 2
    assert np.all(result.event_times >= -1)
    assert np.all(result.event_times <= 2)


@pytest.mark.parametrize("agg_type", ["simple", "group", "calendar", "dynamic"])
def test_aggte_with_na_rm(mp_result, agg_type):
    result = aggte(mp_result, type=agg_type, na_rm=True, cband=False)

    assert result.aggregation_type == agg_type
    assert isinstance(result.overall_att, float)
    assert isinstance(result.overall_se, float)


@pytest.mark.parametrize(
    "params,expected",
    [
        ({"boot": True, "biters": 99}, {"bootstrap": True, "biters": 99}),
        ({"cband": True, "boot": True, "biters": 99}, {"uniform_bands": True, "bootstrap": True}),
        ({"alpha": 0.1}, {"alpha": 0.1}),
    ],
)
def test_aggte_parameter_passing(mp_result, params, expected):
    result = aggte(mp_result, type="simple", **params)

    for key, value in expected.items():
        assert result.estimation_params.get(key) == value


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
        match=f"^The att_gt fit did not cluster on {re.escape(names)}\\. The aggregation reuses the clusters of "
        "the att_gt fit\\. Pass the cluster variable to att_gt instead\\. To aggregate this fit, "
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
def test_aggte_unit_id_request_on_clustered_fit_gives_unit_level_inference(mpdta_data, fit_args, recwarn):
    data = mpdta_data.with_columns((pl.col("countyreal") // 1000).alias("state"))
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "est_method": "reg"}
    with pytest.warns(UserWarning, match="^The units fall in only 29 clusters\\. "):
        fit = att_gt(data, boot=True, biters=99, random_state=7, **spec, **fit_args)
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
def test_aggte_leaves_the_few_clusters_warning_to_att_gt(mpdta_data, boot):
    data = mpdta_data.with_columns((pl.col("countyreal") // 1000).alias("state"))
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "est_method": "reg"}

    with warnings.catch_warnings(record=True) as at_fit:
        warnings.simplefilter("always")
        fit = att_gt(data, clustervars=["state"], boot=boot, biters=99, random_state=7, **spec)
    with warnings.catch_warnings(record=True) as at_aggregation:
        warnings.simplefilter("always")
        aggregations = [
            aggte(fit, type=agg_type, boot=agg_boot, biters=99, random_state=7)
            for agg_type in ("simple", "dynamic", "group", "calendar")
            for agg_boot in (True, False)
        ]

    assert len([w for w in at_fit if "very few clusters" in str(w.message)]) == 1
    assert not [w for w in at_aggregation if "very few clusters" in str(w.message)]
    for result in aggregations:
        kind = "Bootstrap" if result.estimation_params["bootstrap"] else "Analytical"
        assert f" {kind} standard errors clustered by state" in str(result).splitlines()
        assert result.__maketables_stat__("se_type") == f"{kind} clustered by state"
        assert result.__maketables_vcov_info__ == {"vcov_type": kind.lower(), "clustervar": "state"}


@pytest.mark.parametrize("agg_type", ["simple", "dynamic", "group", "calendar"])
def test_aggte_without_the_bootstrap_sums_the_overall_influence_function_within_clusters(mpdta_data, agg_type):
    data = mpdta_data.with_columns((pl.col("countyreal") // 1000).alias("state"))
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "est_method": "reg"}
    fit = att_gt(data, clustervars=["state"], **spec)
    plain_fit = att_gt(data, **spec)

    result = aggte(fit, type=agg_type, cband=False)
    plain = aggte(plain_fit, type=agg_type, cband=False)
    _, cluster_idx = np.unique(fit.estimation_params["cluster"], return_inverse=True)
    sums = np.bincount(cluster_idx, weights=result.influence_func_overall)

    np.testing.assert_allclose(result.overall_se, np.sqrt(np.sum(sums**2)) / fit.n_units, rtol=1e-12)
    assert result.overall_att == plain.overall_att
    assert abs(result.overall_se - plain.overall_se) > 1e-4


@pytest.mark.parametrize("agg_type", ["dynamic", "group", "calendar"])
def test_aggte_without_the_bootstrap_sums_each_disaggregated_influence_function_within_clusters(mpdta_data, agg_type):
    data = mpdta_data.with_columns((pl.col("countyreal") // 1000).alias("state"))
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "est_method": "reg"}
    fit = att_gt(data, clustervars=["state"], **spec)
    plain_fit = att_gt(data, **spec)

    result = aggte(fit, type=agg_type, cband=False)
    plain = aggte(plain_fit, type=agg_type, cband=False)
    _, cluster_idx = np.unique(fit.estimation_params["cluster"], return_inverse=True)
    sums = np.stack([np.bincount(cluster_idx, weights=column) for column in result.influence_func.T], axis=1)
    expected = np.sqrt(np.sum(sums**2, axis=0)) / fit.n_units

    np.testing.assert_allclose(result.se_by_event, expected, rtol=1e-12)
    np.testing.assert_array_equal(result.att_by_event, plain.att_by_event)
    assert np.max(np.abs(result.se_by_event - plain.se_by_event)) > 1e-3


@pytest.mark.parametrize("agg_type", ["simple", "dynamic", "group", "calendar"])
def test_aggte_result_attributes(mp_result, agg_type):
    result = aggte(mp_result, type=agg_type, cband=False)

    required_attrs = [
        "overall_att",
        "overall_se",
        "aggregation_type",
        "event_times",
        "att_by_event",
        "se_by_event",
        "critical_values",
        "influence_func",
        "estimation_params",
        "call_info",
    ]

    for attr in required_attrs:
        assert hasattr(result, attr)

    assert "function" in result.call_info
    assert "aggte" in result.call_info["function"]


def test_aggte_influence_function(mp_result):
    result = aggte(mp_result, type="simple")

    assert result.influence_func is not None
    assert isinstance(result.influence_func, np.ndarray)
    assert len(result.influence_func) == mp_result.n_units


@pytest.mark.parametrize(
    "agg_type,expected_text",
    [
        ("simple", "Aggregate Treatment Effects"),
        ("dynamic", "Event Study"),
        ("group", "Group/Cohort"),
        ("calendar", "Calendar Time"),
    ],
)
def test_aggte_print_output(mp_result, agg_type, expected_text):
    result = aggte(mp_result, type=agg_type, cband=False)
    output = str(result)
    assert "Aggregate Treatment Effects" in output
    assert expected_text in output


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

    result = aggte(mp_result, type="simple")
    assert not np.isnan(result.overall_att)


def test_aggte_very_small_standard_errors(mp_result):
    mp_result.influence_func[:] = 1e-15

    result = aggte(mp_result, type="simple")
    assert np.isnan(result.overall_se) or result.overall_se > 0


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
    np.testing.assert_allclose(shifted.att_by_event, original.att_by_event, rtol=1e-10)
    np.testing.assert_allclose(shifted.se_by_event, original.se_by_event, rtol=1e-10)


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

    with pytest.warns(UserWarning, match="critical value is NaN"):
        result = aggte(
            mp_result,
            type="group",
            cband=True,
            boot=True,
            biters=99,
        )

    assert result.aggregation_type == "group"
    assert not result.estimation_params.get("uniform_bands", True)


def test_aggte_dynamic_no_post_treatment(mp_result):
    with pytest.warns(
        UserWarning,
        match="^The overall effect is NaN because max_e=-1 leaves out every post-treatment group-time effect\\. "
        "Pass a larger max_e to estimate it\\.$",
    ):
        agg_result = aggte(mp_result, type="dynamic", max_e=-1, cband=False)
    full_result = aggte(mp_result, type="dynamic", cband=False)

    np.testing.assert_array_equal(agg_result.event_times, [-3, -2, -1])
    np.testing.assert_allclose(agg_result.att_by_event, full_result.att_by_event[:3], rtol=1e-12)
    assert np.isnan(agg_result.overall_att)
    assert np.isnan(agg_result.overall_se)


@pytest.mark.parametrize("aggregation", ["simple", "group"])
def test_aggte_max_e_before_treatment_reports_nan(mp_result, aggregation):
    with pytest.warns(
        UserWarning,
        match="^The overall effect is NaN because max_e=-1 leaves out every post-treatment group-time effect\\. ",
    ):
        result = aggte(mp_result, type=aggregation, max_e=-1, cband=False)

    assert np.isnan(result.overall_att)
    assert np.isnan(result.overall_se)
    if aggregation == "group":
        assert len(result.event_times) == 0


def test_aggte_dynamic_with_extreme_balance(mp_result):
    agg_result = aggte(mp_result, type="dynamic", balance_e=2, cband=False)

    assert agg_result.aggregation_type == "dynamic"
    assert agg_result.balanced_event_threshold == 2


def test_aggte_very_large_bootstrap_iterations(mp_result):
    agg_result = aggte(
        mp_result,
        type="simple",
        boot=True,
        biters=10,
    )

    assert agg_result.aggregation_type == "simple"
    assert agg_result.estimation_params.get("biters") == 10


def test_aggte_cband_narrower_than_pointwise_fallback(mp_result):
    def mock_mboot(*args, **kwargs):
        real_result = real_mboot(*args, **kwargs)
        real_result["crit_val"] = 1.0
        return real_result

    with (
        patch("moderndid.did.compute_aggte.mboot", side_effect=mock_mboot),
        pytest.warns(UserWarning, match="simultaneous confidence band is narrower than the pointwise"),
    ):
        agg_result = aggte(mp_result, type="group", cband=True, boot=True, biters=99)

    assert not agg_result.estimation_params.get("uniform_bands", True)


def test_aggte_cband_very_large_cv_warning(mp_result):
    def mock_mboot(*args, **kwargs):
        real_result = real_mboot(*args, **kwargs)
        real_result["crit_val"] = 8.0
        return real_result

    with (
        patch("moderndid.did.compute_aggte.mboot", side_effect=mock_mboot),
        pytest.warns(UserWarning, match="Simultaneous critical value is very large"),
    ):
        agg_result = aggte(mp_result, type="group", cband=True, boot=True, biters=99)

    assert agg_result.estimation_params.get("uniform_bands", False)
    assert np.all(agg_result.critical_values == 8.0)


@pytest.mark.parametrize("agg_type", ["simple", "dynamic", "group", "calendar"])
def test_aggte_bootstrap_reproducibility(mp_result, agg_type):
    result1 = aggte(mp_result, type=agg_type, boot=True, biters=50, random_state=42)
    result2 = aggte(mp_result, type=agg_type, boot=True, biters=50, random_state=42)

    assert result1.overall_se == result2.overall_se
    assert result1.estimation_params.get("random_state") == 42

    if agg_type != "simple":
        np.testing.assert_array_equal(result1.se_by_event, result2.se_by_event)


def test_aggte_band_without_bootstrap_draws_one_multiplier_per_cluster(mpdta_data):
    data = mpdta_data.with_columns((pl.col("countyreal") // 1000).alias("state"))
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "est_method": "reg"}
    clustered = att_gt(data, clustervars=["state"], **spec)
    plain = att_gt(data, **spec)
    band = {"type": "dynamic", "boot": False, "cband": True, "biters": 999, "random_state": 11}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from_clustered = aggte(clustered, **band)
    from_plain = aggte(plain, **band)
    draws = real_mboot(
        from_clustered.influence_func,
        n_units=clustered.n_units,
        biters=999,
        alpha=0.05,
        cluster=clustered.estimation_params["cluster"],
        random_state=11,
    )

    assert [str(w.message) for w in caught] == ["Used bootstrap procedure to compute simultaneous confidence band"]
    assert from_clustered.estimation_params["uniform_bands"]
    np.testing.assert_allclose(from_clustered.critical_values, draws["crit_val"], rtol=1e-12)
    assert from_clustered.critical_values[0] != from_plain.critical_values[0]


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


def test_aggte_without_post_treatment_effects_keeps_pre_treatment_event_times(no_post_treatment_result):
    with pytest.warns(UserWarning, match="^The overall effect is NaN because no post-treatment group-time effect is "):
        result = aggte(no_post_treatment_result, type="dynamic", cband=False)

    np.testing.assert_array_equal(result.event_times, [-1])
    np.testing.assert_allclose(result.att_by_event, no_post_treatment_result.att_gt)


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
    np.testing.assert_allclose(from_universal.att_by_event, from_varying.att_by_event, rtol=1e-10)
    np.testing.assert_allclose(from_universal.se_by_event, from_varying.se_by_event, rtol=1e-10)
    np.testing.assert_allclose(from_universal.overall_att, from_varying.overall_att, rtol=1e-10)
