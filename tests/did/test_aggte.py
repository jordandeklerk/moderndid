"""Tests for aggregate treatment effects."""

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid import aggte, att_gt, load_mpdta


@pytest.fixture
def mp_result():
    df = load_mpdta()

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first.treat",
        idname="countyreal",
        xformla="~ 1",
        est_method="reg",
    )

    return result


@pytest.mark.parametrize("agg_type", ["simple", "dynamic", "group", "calendar"])
def test_aggte_basic_structure(mp_result, agg_type):
    result = aggte(mp_result, type=agg_type)

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
    result = aggte(mp_result, type="dynamic", balance_e=1)

    assert result.aggregation_type == "dynamic"
    assert result.balanced_event_threshold == 1
    assert isinstance(result.overall_att, float)
    assert isinstance(result.overall_se, float)


def test_aggte_dynamic_with_min_max_e(mp_result):
    result = aggte(mp_result, type="dynamic", min_e=-1, max_e=2)

    assert result.aggregation_type == "dynamic"
    assert result.min_event_time == -1
    assert result.max_event_time == 2
    assert np.all(result.event_times >= -1)
    assert np.all(result.event_times <= 2)


@pytest.mark.parametrize("agg_type", ["simple", "group", "calendar", "dynamic"])
def test_aggte_with_na_rm(mp_result, agg_type):
    result = aggte(mp_result, type=agg_type, na_rm=True)

    assert result.aggregation_type == agg_type
    assert isinstance(result.overall_att, float)
    assert isinstance(result.overall_se, float)


@pytest.mark.parametrize(
    "params,expected",
    [
        ({"boot": True, "biters": 99}, {"bootstrap": True, "biters": 99}),
        ({"cband": True, "boot": True, "biters": 99}, {"uniform_bands": True, "bootstrap": True}),
        ({"alp": 0.1}, {"alpha": 0.1}),
    ],
)
def test_aggte_parameter_passing(mp_result, params, expected):
    result = aggte(mp_result, type="simple", **params)

    for key, value in expected.items():
        assert result.estimation_params.get(key) == value


@pytest.mark.filterwarnings("ignore:Clustering requested.*but data not available:UserWarning")
def test_aggte_clustering(mp_result):
    result = aggte(
        mp_result,
        type="simple",
        clustervars=["countyreal"],
        boot=True,
        biters=99,
    )

    assert result.aggregation_type == "simple"
    assert result.estimation_params.get("clustervars") == ["countyreal"]


@pytest.mark.parametrize("agg_type", ["simple", "dynamic", "group", "calendar"])
def test_aggte_result_attributes(mp_result, agg_type):
    result = aggte(mp_result, type=agg_type)

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
    result = aggte(mp_result, type=agg_type)
    output = str(result)
    assert "Aggregate Treatment Effects" in output
    assert expected_text in output


def test_aggte_with_missing_values_no_na_rm(mp_result):
    mp_result.att_gt[0] = np.nan

    with pytest.raises(ValueError, match="Missing values at att_gt found"):
        aggte(mp_result, type="simple", na_rm=False)


def test_aggte_all_nan_values():
    df = load_mpdta()

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first.treat",
        idname="countyreal",
        xformla="~ 1",
        est_method="reg",
    )

    result.att_gt[:] = np.nan

    agg_result = aggte(result, type="simple", na_rm=True)
    assert agg_result.overall_att == 0.0
    assert np.isnan(agg_result.overall_se)


def test_aggte_empty_keepers_after_filtering():
    df = load_mpdta()

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first.treat",
        idname="countyreal",
        xformla="~ 1",
        est_method="reg",
    )

    with pytest.raises(ValueError, match="need at least one array"):
        aggte(result, type="dynamic", min_e=100, max_e=101)


def test_aggte_infinite_values_in_att(mp_result):
    mp_result.att_gt[0] = np.inf

    result = aggte(mp_result, type="simple")
    assert not np.isnan(result.overall_att)


def test_aggte_very_small_standard_errors(mp_result):
    mp_result.influence_func[:] = 1e-15

    result = aggte(mp_result, type="simple")
    assert np.isnan(result.overall_se) or result.overall_se > 0


def test_aggte_extreme_event_times():
    df = load_mpdta()

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first.treat",
        idname="countyreal",
        xformla="~ 1",
        est_method="reg",
    )

    agg_result = aggte(result, type="dynamic", min_e=-1e10, max_e=1e10)
    assert agg_result.aggregation_type == "dynamic"
    assert agg_result.min_event_time == int(-1e10)
    assert agg_result.max_event_time == int(1e10)


@pytest.mark.filterwarnings("ignore:Setting an item of incompatible dtype:FutureWarning")
def test_aggte_non_sequential_time_periods():
    df = load_mpdta()
    df = df.with_columns(
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

    agg_result = aggte(result, type="calendar")
    assert agg_result.aggregation_type == "calendar"
    assert agg_result.event_times is not None


@pytest.mark.filterwarnings("ignore:Setting an item of incompatible dtype:FutureWarning")
@pytest.mark.filterwarnings("ignore:Not returning pre-test Wald statistic:UserWarning")
def test_aggte_all_treated_same_time():
    df = load_mpdta()
    df = df.with_columns(pl.when(pl.col("first.treat") != 0).then(2004).otherwise(np.inf).alias("first.treat"))

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first.treat",
        idname="countyreal",
        xformla="~ 1",
        est_method="reg",
    )

    agg_result = aggte(result, type="group")
    treated_groups = [g for g in agg_result.event_times if np.isfinite(g)]
    assert len(treated_groups) == 1


def test_aggte_clustering_multiple_vars_error(mp_result):
    with pytest.raises(NotImplementedError, match="multiple variables"):
        aggte(
            mp_result,
            type="simple",
            clustervars=["var1", "var2"],
            boot=True,
            biters=99,
        )


def test_aggte_uniform_bands_without_bootstrap():
    df = load_mpdta()

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first.treat",
        idname="countyreal",
        xformla="~ 1",
        est_method="reg",
    )

    with pytest.warns(UserWarning, match="bootstrap procedure"):
        agg_result = aggte(
            result,
            type="group",
            cband=True,
            boot=False,
        )

    assert agg_result.aggregation_type == "group"


def test_aggte_critical_value_nan_handling(mp_result):
    mp_result.influence_func[:] = 0

    with pytest.warns(UserWarning, match="critical value is NA"):
        result = aggte(
            mp_result,
            type="group",
            cband=True,
            boot=True,
            biters=99,
        )

    assert result.aggregation_type == "group"
    assert not result.estimation_params.get("uniform_bands", True)


def test_aggte_dynamic_no_post_treatment():
    df = load_mpdta()

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first.treat",
        idname="countyreal",
        xformla="~ 1",
        est_method="reg",
    )

    agg_result = aggte(result, type="dynamic", max_e=-1)

    assert np.isnan(agg_result.overall_att)
    assert np.isnan(agg_result.overall_se)


def test_aggte_dynamic_with_extreme_balance():
    df = load_mpdta()

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first.treat",
        idname="countyreal",
        xformla="~ 1",
        est_method="reg",
    )

    agg_result = aggte(result, type="dynamic", balance_e=2)

    assert agg_result.aggregation_type == "dynamic"
    assert agg_result.balanced_event_threshold == 2


def test_aggte_conflicting_cband_bootstrap():
    df = load_mpdta()

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first.treat",
        idname="countyreal",
        xformla="~ 1",
        est_method="reg",
    )

    with pytest.warns(UserWarning, match="bootstrap procedure"):
        agg_result = aggte(
            result,
            type="dynamic",
            cband=True,
            boot=False,
        )

    assert agg_result.aggregation_type == "dynamic"


def test_aggte_parameter_inheritance_from_mp():
    df = load_mpdta()

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first.treat",
        idname="countyreal",
        xformla="~ 1",
        est_method="reg",
    )

    agg_result = aggte(result, type="simple", alp=0.1)

    assert agg_result.estimation_params.get("alpha") == 0.1


def test_aggte_very_large_bootstrap_iterations():
    df = load_mpdta()

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first.treat",
        idname="countyreal",
        xformla="~ 1",
        est_method="reg",
    )

    agg_result = aggte(
        result,
        type="simple",
        boot=True,
        biters=10,
    )

    assert agg_result.aggregation_type == "simple"
    assert agg_result.estimation_params.get("biters") == 10


@pytest.mark.parametrize("agg_type", ["simple", "dynamic", "group", "calendar"])
def test_aggte_bootstrap_reproducibility(mp_result, agg_type):
    result1 = aggte(mp_result, type=agg_type, boot=True, biters=50, random_state=42)
    result2 = aggte(mp_result, type=agg_type, boot=True, biters=50, random_state=42)

    assert result1.overall_se == result2.overall_se
    assert result1.estimation_params.get("random_state") == 42

    if agg_type != "simple":
        np.testing.assert_array_equal(result1.se_by_event, result2.se_by_event)
