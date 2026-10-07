"""Tests for DDD aggregate treatment effects."""

import numpy as np
import polars as pl
import pytest

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
def test_agg_ddd_dropna(mp_ddd_result, agg_type):
    result = agg_ddd(mp_ddd_result, type=agg_type, dropna=True, boot=False, cband=False)

    assert result.aggregation_type == agg_type
    assert isinstance(result.overall_att, float | np.floating)


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
        agg_ddd(mp_ddd_result, type="simple", dropna=False, boot=False, cband=False)


def test_agg_ddd_eventstudy_no_post(mp_ddd_result):
    result = agg_ddd(mp_ddd_result, type="eventstudy", max_e=-1, boot=False, cband=False)

    assert np.isnan(result.overall_att)
    assert np.isnan(result.overall_se)


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


@pytest.mark.parametrize("boot", [False, True])
def test_agg_ddd_calendar_overall_se_averages_periods_with_cells(mp_ddd_result, boot):
    empty = (mp_ddd_result.groups == 3) & (mp_ddd_result.times == 3)
    gapped = mp_ddd_result._replace(att=np.where(empty, np.nan, mp_ddd_result.att))
    agg = agg_ddd(gapped, type="calendar", dropna=True, boot=boot, cband=False, biters=199, random_state=3)
    valid = np.isfinite(agg.att_egt)
    if boot:
        draws = mboot_ddd(agg.inf_func, 199, 0.05, random_state=3).bres[:, valid].mean(axis=1)
        expected = (np.percentile(draws, 75) - np.percentile(draws, 25)) / 1.3489795 / np.sqrt(mp_ddd_result.n)
    else:
        expected = np.sqrt(np.mean(agg.inf_func[:, valid].mean(axis=1) ** 2) / mp_ddd_result.n)

    np.testing.assert_array_equal(agg.egt, [3, 4, 5])
    np.testing.assert_array_equal(valid, [False, True, True])
    np.testing.assert_allclose(agg.overall_att, np.mean(agg.att_egt[valid]), rtol=1e-12)
    np.testing.assert_allclose(agg.overall_se, expected, rtol=1e-12)
