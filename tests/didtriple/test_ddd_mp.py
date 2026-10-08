"""Tests for the multi-period DDD estimator."""

import re
import traceback
import warnings

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid import ddd, ddd_mp
from moderndid.didtriple.container import DDDMultiPeriodResult


@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_ddd_mp_basic(mp_ddd_data, est_method):
    result = ddd_mp(
        data=mp_ddd_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        est_method=est_method,
    )

    assert len(result.att) > 0
    assert len(result.att) == len(result.se)
    assert len(result.att) == len(result.groups)
    assert len(result.att) == len(result.times)
    assert result.n == mp_ddd_data["id"].n_unique()


def test_ddd_mp_confidence_intervals(mp_ddd_data):
    result = ddd_mp(
        data=mp_ddd_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
    )

    valid_mask = ~np.isnan(result.se)
    assert np.all(result.lci[valid_mask] < result.att[valid_mask])
    assert np.all(result.att[valid_mask] < result.uci[valid_mask])


def test_ddd_mp_influence_functions(mp_ddd_data):
    result = ddd_mp(
        data=mp_ddd_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
    )

    assert result.inf_func_mat is not None
    assert result.inf_func_mat.shape[0] == result.n
    assert result.inf_func_mat.shape[1] == len(result.att)


@pytest.mark.parametrize("control_group", ["nevertreated", "notyettreated"])
def test_ddd_mp_control_group(mp_ddd_data, control_group):
    result = ddd_mp(
        data=mp_ddd_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        control_group=control_group,
    )

    assert len(result.att) > 0
    assert result.args["control_group"] == control_group


@pytest.mark.parametrize("base_period", ["universal", "varying"])
def test_ddd_mp_base_period(mp_ddd_data, base_period):
    result = ddd_mp(
        data=mp_ddd_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        base_period=base_period,
    )

    assert len(result.att) > 0
    assert result.args["base_period"] == base_period


def test_ddd_mp_glist_tlist(mp_ddd_data):
    result = ddd_mp(
        data=mp_ddd_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
    )

    assert 3 in result.glist
    assert 4 in result.glist
    assert set(result.tlist) == {1, 2, 3, 4, 5}


def test_ddd_mp_post_treatment_effects(mp_ddd_data):
    result = ddd_mp(
        data=mp_ddd_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
    )

    post_atts = [
        result.att[i]
        for i, (g, t) in enumerate(zip(result.groups, result.times))
        if t >= g and np.isfinite(result.att[i])
    ]
    assert len(post_atts) > 0
    assert all(0.0 < att < 5.0 for att in post_atts)


def test_ddd_mp_never_treated_as_inf():
    rng = np.random.default_rng(42)
    n_units = 300
    time_periods = [1, 2, 3]

    records = []
    for unit in range(n_units):
        g = rng.choice([np.inf, 3], p=[0.5, 0.5])
        p = rng.choice([0, 1])
        for t in time_periods:
            y = rng.normal(0, 1)
            if np.isfinite(g) and t >= g and p == 1:
                y += 1.5
            records.append({"id": unit, "time": t, "y": y, "group": g, "partition": p})

    data = pl.DataFrame(records)

    result = ddd_mp(
        data=data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
    )

    assert len(result.att) > 0
    assert 3 in result.glist


@pytest.mark.parametrize("cband", [False, True])
def test_ddd_mp_bootstrap(mp_ddd_data, cband):
    result = ddd_mp(
        data=mp_ddd_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        boot=True,
        biters=50,
        cband=cband,
        random_state=42,
    )

    valid_mask = ~np.isnan(result.se)
    assert np.sum(valid_mask) > 0
    assert np.all(result.lci[valid_mask] < result.att[valid_mask])
    assert np.all(result.att[valid_mask] < result.uci[valid_mask])


def test_ddd_mp_reproducibility(mp_ddd_data):
    result1 = ddd_mp(
        data=mp_ddd_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        boot=True,
        biters=20,
        random_state=123,
    )

    result2 = ddd_mp(
        data=mp_ddd_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        boot=True,
        biters=20,
        random_state=123,
    )

    np.testing.assert_allclose(result1.se, result2.se, equal_nan=True)


def _ddd_mp_kwargs(data, n_jobs):
    return {
        "data": data,
        "y_col": "y",
        "time_col": "time",
        "id_col": "id",
        "group_col": "group",
        "partition_col": "partition",
        "n_jobs": n_jobs,
    }


def _check_ddd_mp_result(result, data):
    assert len(result.att) > 0
    assert len(result.att) == len(result.se)
    assert len(result.att) == len(result.groups)
    assert len(result.att) == len(result.times)
    assert result.n == data["id"].n_unique()
    assert result.inf_func_mat.shape == (result.n, len(result.att))
    valid = ~np.isnan(result.se)
    assert np.all(result.lci[valid] < result.att[valid])
    assert np.all(result.att[valid] < result.uci[valid])


@pytest.mark.benchmark
def test_benchmark_ddd_mp_sequential(benchmark, mp_ddd_data):
    result = benchmark.pedantic(ddd_mp, kwargs=_ddd_mp_kwargs(mp_ddd_data, n_jobs=1), rounds=3, warmup_rounds=1)
    _check_ddd_mp_result(result, mp_ddd_data)


@pytest.mark.benchmark
def test_benchmark_ddd_mp_parallel(benchmark, mp_ddd_data):
    result = benchmark.pedantic(ddd_mp, kwargs=_ddd_mp_kwargs(mp_ddd_data, n_jobs=-1), rounds=3, warmup_rounds=1)
    _check_ddd_mp_result(result, mp_ddd_data)


@pytest.mark.parametrize("base_period", ["universal", "varying"])
def test_ddd_mp_parallel_matches_sequential(mp_ddd_data, base_period):
    result_seq = ddd_mp(
        data=mp_ddd_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        base_period=base_period,
        n_jobs=1,
    )

    result_par = ddd_mp(
        data=mp_ddd_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        base_period=base_period,
        n_jobs=2,
    )

    np.testing.assert_allclose(result_seq.att, result_par.att, rtol=1e-10)
    np.testing.assert_array_equal(result_seq.groups, result_par.groups)
    np.testing.assert_array_equal(result_seq.times, result_par.times)
    np.testing.assert_allclose(result_seq.se, result_par.se, rtol=1e-10, equal_nan=True)
    np.testing.assert_allclose(result_seq.inf_func_mat, result_par.inf_func_mat, rtol=1e-10)


def test_ddd_unbalanced_panel_returns_unit_level_result(mp_ddd_data):
    rng = np.random.default_rng(99)
    n_rows = len(mp_ddd_data)
    drop_mask = rng.random(n_rows) < 0.05
    keep_idx = [i for i in range(n_rows) if not drop_mask[i]]
    unbalanced = mp_ddd_data[keep_idx]

    assert unbalanced["id"].n_unique() == mp_ddd_data["id"].n_unique()

    result = ddd(
        data=unbalanced,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        allow_unbalanced_panel=True,
    )

    assert isinstance(result, DDDMultiPeriodResult)
    assert len(result.att) > 0
    assert len(result.att) == len(result.se)
    assert len(result.att) == len(result.groups)
    assert len(result.att) == len(result.times)
    assert result.n == unbalanced["id"].n_unique()
    assert result.inf_func_mat.shape == (result.n, len(result.att))
    assert all(np.isfinite(result.att))
    assert result.args["panel"] is True


@pytest.mark.parametrize(
    ("data_fixture", "ddd_options", "direct_options"),
    [
        ("mp_unbalanced_df", {"xformla": "~ cov1 + cov2"}, {"covariate_cols": ["cov1", "cov2"]}),
        ("mp_unbalanced_df", {"xformla": "~ cov1"}, {"covariate_cols": "cov1"}),
        ("mp_unbalanced_df", {"allow_unbalanced_panel": True}, {"allow_unbalanced_panel": True}),
        (
            "mp_clustered_df",
            {"cluster": "cluster", "boot": True, "biters": 49, "random_state": 1},
            {"cluster": "cluster", "boot": True, "biters": 49, "random_state": 1},
        ),
        ("mp_first_period_cohort_df", {}, {}),
        ("mp_weighted_df", {"weightsname": "w", "est_method": "dr"}, {"weights_col": "w", "est_method": "dr"}),
        ("multi_period_df", {"alpha": 0.2}, {"alpha": 0.2}),
    ],
)
def test_ddd_mp_matches_ddd_on_the_same_data(request, data_fixture, ddd_options, direct_options):
    data = request.getfixturevalue(data_fixture)
    spec = {"yname": "y", "tname": "time", "idname": "id", "gname": "group", "pname": "partition", "est_method": "reg"}
    direct = {
        "y_col": "y",
        "time_col": "time",
        "id_col": "id",
        "group_col": "group",
        "partition_col": "partition",
        "est_method": "reg",
    }

    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter("always")
        expected = ddd(data=data, **(spec | ddd_options))
    with warnings.catch_warnings(record=True) as direct_warnings:
        warnings.simplefilter("always")
        result = ddd_mp(data=data, **(direct | direct_options))

    assert result.n == expected.n
    np.testing.assert_array_equal(result.glist, expected.glist)
    np.testing.assert_array_equal(result.groups, expected.groups)
    np.testing.assert_array_equal(result.times, expected.times)
    np.testing.assert_array_equal(result.att, expected.att)
    np.testing.assert_array_equal(result.se, expected.se)
    np.testing.assert_array_equal(result.lci, expected.lci)
    np.testing.assert_array_equal(result.uci, expected.uci)
    np.testing.assert_array_equal(result.inf_func_mat, expected.inf_func_mat)
    np.testing.assert_array_equal(result.unit_groups, expected.unit_groups)
    np.testing.assert_array_equal(result.unit_weights, expected.unit_weights)
    np.testing.assert_array_equal(result.unit_clusters, expected.unit_clusters)
    assert [str(w.message) for w in direct_warnings] == [str(w.message) for w in expected_warnings]


@pytest.mark.parametrize("mp_missing_outcome_df", [None, float("nan")], indirect=True)
def test_ddd_mp_drops_rows_with_missing_values_like_ddd(mp_missing_outcome_df):
    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter("always")
        expected = ddd(
            data=mp_missing_outcome_df, yname="y", tname="time", idname="id", gname="group", pname="partition"
        )
    with warnings.catch_warnings(record=True) as direct_warnings:
        warnings.simplefilter("always")
        result = ddd_mp(
            data=mp_missing_outcome_df,
            y_col="y",
            time_col="time",
            id_col="id",
            group_col="group",
            partition_col="partition",
        )

    assert result.n == expected.n
    np.testing.assert_array_equal(result.att, expected.att)
    np.testing.assert_array_equal(result.se, expected.se)
    np.testing.assert_array_equal(result.inf_func_mat, expected.inf_func_mat)
    assert [str(w.message) for w in direct_warnings] == [str(w.message) for w in expected_warnings]
    assert [str(w.message) for w in direct_warnings] == [
        "Dropped 3 rows from original data due to missing values",
        "Dropped 3 units while converting to balanced panel",
    ]


@pytest.mark.parametrize("mp_no_never_treated_gap_df", [2, 5], indirect=True)
@pytest.mark.parametrize("base_period", ["universal", "varying"])
def test_ddd_mp_without_never_treated_units_matches_ddd(mp_no_never_treated_gap_df, base_period):
    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter("always")
        expected = ddd(
            data=mp_no_never_treated_gap_df,
            yname="y",
            tname="time",
            idname="id",
            gname="group",
            pname="partition",
            control_group="notyettreated",
            base_period=base_period,
            est_method="reg",
        )
    with warnings.catch_warnings(record=True) as direct_warnings:
        warnings.simplefilter("always")
        result = ddd_mp(
            data=mp_no_never_treated_gap_df,
            y_col="y",
            time_col="time",
            id_col="id",
            group_col="group",
            partition_col="partition",
            control_group="notyettreated",
            base_period=base_period,
            est_method="reg",
        )

    assert result.n == expected.n
    np.testing.assert_array_equal(result.glist, [2, 3])
    np.testing.assert_array_equal(result.glist, expected.glist)
    np.testing.assert_array_equal(result.tlist, expected.tlist)
    np.testing.assert_array_equal(result.groups, expected.groups)
    np.testing.assert_array_equal(result.times, expected.times)
    np.testing.assert_array_equal(result.att, expected.att)
    np.testing.assert_array_equal(result.se, expected.se)
    np.testing.assert_array_equal(result.inf_func_mat, expected.inf_func_mat)
    assert [str(w.message) for w in direct_warnings] == [str(w.message) for w in expected_warnings]


@pytest.mark.parametrize("base_period", ["universal", "varying"])
def test_ddd_mp_without_never_treated_units_needs_a_cohort_besides_the_latest(mp_no_never_treated_df, base_period):
    with pytest.raises(ValueError, match=re.escape("No cohort is left to estimate.")):
        ddd_mp(
            data=mp_no_never_treated_df.filter(pl.col("group") == 4),
            y_col="y",
            time_col="time",
            id_col="id",
            group_col="group",
            partition_col="partition",
            control_group="notyettreated",
            base_period=base_period,
            est_method="reg",
        )


@pytest.mark.parametrize(
    ("change", "ddd_options", "direct_options", "ddd_message", "direct_message"),
    [
        (
            pl.col("y"),
            {"yname": "yy"},
            {"y_col": "yy"},
            "yname='yy' is not a column in the data. Did you mean 'y'?",
            "y_col='yy' is not a column in the data. Did you mean 'y'?",
        ),
        (
            pl.col("y"),
            {"xformla": "~ cov1 + covv2"},
            {"covariate_cols": ["cov1", "covv2"]},
            "'covv2' in xformla is not a column in the data.",
            "'covv2' in covariate_cols is not a column in the data.",
        ),
        (
            pl.col("y").alias(".w"),
            {"yname": ".w"},
            {"y_col": ".w"},
            "yname names the column '.w'.",
            "y_col names the column '.w'.",
        ),
        (
            pl.col("id").cast(pl.String),
            {},
            {},
            "idname='id' is not numeric. Please convert it.",
            "id_col='id' is not numeric. Please convert it.",
        ),
        (
            pl.col("partition") + 1,
            {},
            {},
            "pname='partition' must be 1 for eligible units and 0 for ineligible units",
            "partition_col='partition' must be 1 for eligible units and 0 for ineligible units",
        ),
        (
            pl.when((pl.col("id") == 3) & (pl.col("time") == 2)).then(1).otherwise(pl.col("time")).alias("time"),
            {},
            {},
            "The value of idname must be unique (by tname).",
            "The value of id_col must be unique (by time_col).",
        ),
        (
            pl.when(pl.col("id") == 3).then(-1).otherwise(pl.col("group")).alias("group"),
            {},
            {},
            "gname = 'group' holds negative values such as -1.",
            "group_col = 'group' holds negative values such as -1.",
        ),
        (
            pl.int_range(pl.len()).alias("id"),
            {},
            {},
            "panel=True was specified, but no units appear in multiple time periods.",
            "No unit in id_col='id' appears in more than one period. For repeated cross-sections, use ddd_mp_rc.",
        ),
        (
            [
                pl.when(pl.int_range(pl.len()) == 1).then(0).otherwise(pl.int_range(pl.len())).alias("id"),
                pl.when(pl.int_range(pl.len()) == 1).then(None).otherwise(pl.col("y")).alias("y"),
            ],
            {},
            {},
            "panel=True was specified, but no units appear in multiple time periods.",
            "No unit in id_col='id' appears in more than one period. For repeated cross-sections, use ddd_mp_rc.",
        ),
        (
            pl.when(pl.col("time") == pl.col("id") % 3 + 1).then(None).otherwise(pl.col("y")).alias("y"),
            {},
            {},
            "Consider setting allow_unbalanced_panel=True and/or revisiting 'idname'",
            "Consider setting allow_unbalanced_panel=True and/or revisiting 'id_col'",
        ),
        (
            pl.col("cluster") + (pl.col("time") == 3),
            {"cluster": "cluster"},
            {"cluster": "cluster"},
            "Cluster variable must be time-invariant within units.",
            "Cluster variable must be time-invariant within units.",
        ),
        (
            pl.when(pl.col("group") == 0).then(2).otherwise(pl.col("group")).alias("group"),
            {},
            {},
            "There is no available never-treated group.",
            "There is no available never-treated group.",
        ),
        (
            pl.col("time").min().alias("group"),
            {},
            {},
            "Every unit was already treated in the first period.",
            "Every unit was already treated in the first period.",
        ),
        (
            pl.col("y"),
            {"est_method": "foo"},
            {"est_method": "foo"},
            "est_method='foo' is not valid.",
            "est_method='foo' is not valid.",
        ),
        (
            pl.col("y"),
            {"idname": None},
            {"id_col": None},
            "idname must be provided when panel=True.",
            "id_col must be provided for panel data.",
        ),
    ],
)
def test_ddd_mp_raises_the_errors_of_ddd_with_its_own_argument_names(
    multi_period_df, change, ddd_options, direct_options, ddd_message, direct_message
):
    data = multi_period_df.with_columns(change)
    spec = {"yname": "y", "tname": "time", "idname": "id", "gname": "group", "pname": "partition", "est_method": "reg"}
    direct = {
        "y_col": "y",
        "time_col": "time",
        "id_col": "id",
        "group_col": "group",
        "partition_col": "partition",
        "est_method": "reg",
    }

    with pytest.raises(ValueError, match=re.escape(ddd_message)):
        ddd(data=data, **(spec | ddd_options))
    with pytest.raises(ValueError, match=re.escape(direct_message)):
        ddd_mp(data=data, **(direct | direct_options))


def test_ddd_mp_error_keeps_the_frames_of_the_preparation(multi_period_df):
    data = multi_period_df.with_columns(pl.col("partition") + 1)

    with pytest.raises(ValueError, match="partition_col='partition' must be 1") as caught:
        ddd_mp(data=data, y_col="y", time_col="time", id_col="id", group_col="group", partition_col="partition")

    frames = [frame.name for frame in traceback.extract_tb(caught.value.__traceback__)]
    assert "_preprocess_multiple_periods" in frames
    assert caught.value.__context__ is None
