"""Tests for att_gt, aggte, and honest_did on a panel held in a Spark DataFrame."""

import re

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from scipy import stats

from moderndid import aggte, att_gt, drdid_panel, honest_did, to_df
from tests.helpers import importorskip

pyspark = importorskip("pyspark")

from pyspark.sql import functions as F

pytestmark = pytest.mark.spark


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
@pytest.mark.parametrize("control_group", ["nevertreated", "notyettreated"])
@pytest.mark.parametrize("base_period", ["varying", "universal"])
def test_att_gt_on_spark_matches_att_gt(
    mpdta_data, to_spark, record_warnings, assert_same_fit, est_method, control_group, base_period
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "est_method": est_method,
        "control_group": control_group,
        "base_period": base_period,
        "random_state": 7,
    }
    frame = to_spark(mpdta_data)

    memory, memory_warnings = record_warnings(att_gt, mpdta_data, **spec)
    distributed, spark_warnings = record_warnings(att_gt, frame, **spec)

    assert spark_warnings == memory_warnings
    assert_same_fit(distributed, memory, 1e-12)


@pytest.mark.parametrize(
    "data_fixture, spec, messages",
    [
        (
            "mpdta_pop_weighted",
            {
                "yname": "lemp",
                "tname": "year",
                "idname": "countyreal",
                "gname": "first.treat",
                "xformla": "~lpop",
                "weightsname": "pop",
            },
            [],
        ),
        (
            "mpdta_with_weights_and_states",
            {
                "yname": "lemp",
                "tname": "year",
                "idname": "countyreal",
                "gname": "first.treat",
                "xformla": "~lpop",
                "weightsname": "w",
            },
            [
                "Time-varying weights detected. For balanced panel data, each 2x2 comparison uses the weights from "
                "the earlier of its two periods, the base period for post-treatment cells."
            ],
        ),
        (
            "scalable_panel",
            {
                "yname": "y",
                "tname": "time",
                "idname": "id",
                "gname": "group",
                "xformla": "~cov1 + cov2 + cov3 + cov4",
                "clustervars": ["cluster"],
            },
            [],
        ),
        (
            "mpdta_with_state_names",
            {
                "yname": "lemp",
                "tname": "year",
                "idname": "countyreal",
                "gname": "first.treat",
                "xformla": "~lpop",
                "clustervars": ["countyreal", "state"],
            },
            [
                "The units fall in only 29 clusters. The treated units of cohort 2004 fall in 1 cluster and those of "
                "cohort 2006 in 3 clusters. Cluster-robust inference rests on very few clusters and can give "
                "confidence intervals and bands that are too narrow."
            ],
        ),
        (
            "mpdta_data",
            {
                "yname": "lemp",
                "tname": "year",
                "idname": "countyreal",
                "gname": "first.treat",
                "xformla": "~lpop",
                "control_group": "notyettreated",
                "anticipation": 1,
            },
            [
                "anticipation = 1. Never-treated units (with group status 0 or inf) are assumed to never anticipate "
                "treatment. Anticipation only applies to eventually-treated units.",
                "Dropped 20 units that were already treated in the first period",
            ],
        ),
        (
            "mpdta_data",
            {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "xformla": "~1"},
            [],
        ),
    ],
    ids=["weights", "time_varying_weights", "numeric_clusters", "text_clusters", "anticipation", "no_covariates"],
)
def test_att_gt_on_spark_matches_att_gt_with_each_option(
    request, to_spark, record_warnings, assert_same_fit, data_fixture, spec, messages
):
    data = request.getfixturevalue(data_fixture)
    frame = to_spark(data)

    memory, memory_warnings = record_warnings(att_gt, data, random_state=7, **spec)
    distributed, spark_warnings = record_warnings(att_gt, frame, random_state=7, **spec)

    assert [text for _, text in memory_warnings] == messages
    assert spark_warnings == memory_warnings
    assert_same_fit(distributed, memory, 1e-12)


def test_att_gt_on_spark_with_two_periods_reports_the_bootstrap_band(
    mpdta_data, to_spark, att_gt_on_shards, record_warnings
):
    data = mpdta_data.filter(pl.col("year") >= 2006)
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "boot": True,
        "cband": True,
        "biters": 999,
        "random_state": 1,
    }
    frame = to_spark(data)

    memory, memory_warnings = record_warnings(att_gt, data, **spec)
    distributed, spark_warnings = record_warnings(att_gt, frame, **spec)
    local, _ = record_warnings(att_gt_on_shards, data, 3, **spec)

    assert spark_warnings == memory_warnings
    np.testing.assert_array_equal(distributed.groups, memory.groups)
    np.testing.assert_array_equal(distributed.times, memory.times)
    np.testing.assert_allclose(distributed.att_gt, memory.att_gt, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(distributed.vcov_analytical, memory.vcov_analytical, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(distributed.se_gt, local.se_gt, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(distributed.critical_value, local.critical_value, rtol=1e-12)
    assert abs(distributed.critical_value - stats.norm.ppf(0.975)) > 1e-3


@pytest.mark.parametrize("id_dtype", [pl.Int64, pl.Float64], ids=["integer_ids", "float_ids"])
def test_att_gt_on_spark_bootstraps_like_the_shard_engine_however_the_input_is_partitioned(
    mpdta_data, to_spark, att_gt_on_shards, assert_same_fit, id_dtype
):
    data = mpdta_data.with_columns(pl.col("countyreal").cast(id_dtype))
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "boot": True,
        "biters": 299,
        "random_state": 5,
    }
    frame = to_spark(data)

    one = att_gt(frame.repartition(1), **spec)
    seven = att_gt(frame.repartition(7), **spec)
    local = att_gt_on_shards(mpdta_data, 3, **spec)
    partitions = frame.repartition(7).select(F.spark_partition_id()).distinct().count()

    assert partitions == 7
    assert_same_fit(seven, one, 1e-12)
    assert_same_fit(one, local, 1e-12)
    np.testing.assert_allclose(one.influence_moments.gram, local.influence_moments.gram, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(
        one.influence_moments.bootstrap_sums, local.influence_moments.bootstrap_sums, rtol=1e-12, atol=1e-10
    )


def test_att_gt_on_spark_keeps_each_cluster_whole_on_a_unit_table_of_several_partitions(
    large_clustered_panel, to_spark, unit_table_of, att_gt_on_shards, assert_same_fit
):
    columns = {
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "group",
        "xformla": "~cov1",
        "clustervars": ["cluster"],
    }
    spec = {**columns, "boot": True, "biters": 199, "random_state": 3}

    unit_table, *_ = unit_table_of(large_clustered_panel, **columns)
    partitions = unit_table.select(F.spark_partition_id().alias("part"), "cluster")
    n_partitions = partitions.select("part").distinct().count()
    parts_per_cluster = partitions.groupBy("cluster").agg(F.count_distinct("part").alias("parts"))
    split_clusters = parts_per_cluster.filter(F.col("parts") > 1).count()
    distributed = att_gt(to_spark(large_clustered_panel), **spec)
    memory = att_gt(large_clustered_panel, **spec)
    local = att_gt_on_shards(large_clustered_panel, 3, **spec)
    moments, expected = distributed.influence_moments, local.influence_moments

    assert n_partitions >= 2
    assert split_clusters == 0
    assert parts_per_cluster.count() == moments.n_clusters == 50
    np.testing.assert_array_equal(distributed.groups, memory.groups)
    np.testing.assert_array_equal(distributed.times, memory.times)
    np.testing.assert_allclose(distributed.att_gt, memory.att_gt, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(distributed.vcov_analytical, memory.vcov_analytical, rtol=1e-12, atol=1e-12)
    assert_same_fit(distributed, local, 1e-12)
    np.testing.assert_allclose(moments.cluster_gram, expected.cluster_gram, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(moments.bootstrap_sums, expected.bootstrap_sums, rtol=1e-12, atol=1e-10)


def test_att_gt_on_spark_bootstrap_draws_one_multiplier_for_each_cluster_like_the_shard_engine(
    scalable_panel_with_cluster_names, to_spark, att_gt_on_shards, assert_same_fit
):
    spec = {
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "group",
        "xformla": "~cov1 + cov2 + cov3 + cov4",
        "clustervars": ["cluster"],
        "boot": True,
        "biters": 299,
        "random_state": 3,
    }
    frame = to_spark(scalable_panel_with_cluster_names)

    distributed = att_gt(frame, **spec)
    local = att_gt_on_shards(scalable_panel_with_cluster_names, 3, **spec)
    moments, expected = distributed.influence_moments, local.influence_moments
    analytic = np.sqrt(np.diag(distributed.vcov_analytical) / distributed.n_units)

    assert_same_fit(distributed, local, 1e-12)
    np.testing.assert_allclose(moments.gram, expected.gram, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(moments.cluster_gram, expected.cluster_gram, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(moments.bootstrap_sums, expected.bootstrap_sums, rtol=1e-12, atol=1e-10)
    assert (moments.n_clusters, moments.random_state) == (50, 3)
    assert np.nanmax(np.abs(distributed.se_gt / analytic - 1)) > 1e-3


@pytest.mark.parametrize("aggregation", ["group", "dynamic", "calendar"])
def test_aggte_on_a_spark_fit_matches_aggte_in_memory(universal_base_fits, aggregation):
    on_spark, in_memory = universal_base_fits

    distributed = aggte(on_spark, type=aggregation, cband=False)
    memory = aggte(in_memory, type=aggregation, cband=False)

    np.testing.assert_array_equal(distributed.event_times, memory.event_times)
    np.testing.assert_allclose(distributed.att_by_event, memory.att_by_event, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(distributed.se_by_event, memory.se_by_event, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(distributed.vcov_analytical, memory.vcov_analytical, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(distributed.overall_att, memory.overall_att, rtol=1e-12)
    np.testing.assert_allclose(distributed.overall_se, memory.overall_se, rtol=1e-12)


def test_aggte_simple_on_a_spark_fit_matches_aggte_in_memory(universal_base_fits):
    on_spark, in_memory = universal_base_fits

    distributed = aggte(on_spark, type="simple", cband=False)
    memory = aggte(in_memory, type="simple", cband=False)

    np.testing.assert_allclose(distributed.overall_att, memory.overall_att, rtol=1e-12)
    np.testing.assert_allclose(distributed.overall_se, memory.overall_se, rtol=1e-12)
    np.testing.assert_allclose(distributed.vcov_analytical, memory.vcov_analytical, rtol=1e-12)


@pytest.mark.parametrize(
    "sensitivity_type, options",
    [
        ("smoothness", {"m_vec": np.array([0.0, 0.01])}),
        ("relative_magnitude", {"m_bar_vec": np.array([0.0, 1.0]), "grid_points": 20}),
    ],
    ids=["smoothness", "relative_magnitude"],
)
def test_honest_did_on_an_event_study_of_a_spark_fit_matches_the_in_memory_fit(
    universal_base_fits, sensitivity_type, options
):
    on_spark, in_memory = universal_base_fits
    distributed = aggte(on_spark, type="dynamic", cband=False)
    memory = aggte(in_memory, type="dynamic", cband=False)

    from_spark = honest_did(distributed, event_time=0, sensitivity_type=sensitivity_type, **options)
    from_memory = honest_did(memory, event_time=0, sensitivity_type=sensitivity_type, **options)

    assert distributed.influence_func is None
    np.testing.assert_allclose(from_spark.robust_ci["lb"], from_memory.robust_ci["lb"], rtol=1e-10)
    np.testing.assert_allclose(from_spark.robust_ci["ub"], from_memory.robust_ci["ub"], rtol=1e-10)
    np.testing.assert_allclose(
        [from_spark.original_ci.lb, from_spark.original_ci.ub],
        [from_memory.original_ci.lb, from_memory.original_ci.ub],
        rtol=1e-12,
    )


def test_to_df_of_a_spark_fit_matches_the_in_memory_fit(universal_base_fits):
    on_spark, in_memory = universal_base_fits

    cells = to_df(on_spark)
    expected_cells = to_df(in_memory)
    event_study = to_df(aggte(on_spark, type="dynamic", cband=False))
    expected_event_study = to_df(aggte(in_memory, type="dynamic", cband=False))

    assert_frame_equal(cells, expected_cells, rel_tol=1e-12, abs_tol=1e-12)
    assert_frame_equal(event_study, expected_event_study, rel_tol=1e-12, abs_tol=1e-12)


def test_report_of_a_spark_fit_matches_the_in_memory_report(universal_base_fits):
    on_spark, in_memory = universal_base_fits

    report = str(on_spark)
    expected_report = str(in_memory)
    event_study_report = str(aggte(on_spark, type="dynamic", cband=False))
    expected_event_study_report = str(aggte(in_memory, type="dynamic", cband=False))

    assert "Group-Time Average Treatment Effects" in report
    assert report == expected_report
    assert event_study_report == expected_event_study_report


@pytest.mark.parametrize(
    "option, message",
    [
        (
            {"biters": 500},
            "biters=500 is more than the 499 bootstrap draws that this att_gt fit on a Spark DataFrame stored. "
            "Pass biters=499 or fewer. For more draws, rerun att_gt with biters=500.",
        ),
        (
            {"random_state": 8},
            "random_state=8 is not the seed 7 of the bootstrap draws that this att_gt fit on a Spark DataFrame "
            "stored. Pass random_state=7 or leave it out. For other draws, rerun att_gt with that random_state.",
        ),
    ],
    ids=["more_draws", "another_seed"],
)
def test_aggte_on_a_spark_fit_rejects_draws_the_fit_did_not_store(universal_base_fits, option, message):
    on_spark, _ = universal_base_fits

    with pytest.raises(ValueError, match=re.escape(message)):
        aggte(on_spark, type="dynamic", **option)


def test_aggte_on_a_clustered_spark_fit_rejects_unit_level_bootstrap_draws(scalable_panel, to_spark):
    spec = {"yname": "y", "tname": "time", "idname": "id", "gname": "group", "clustervars": ["cluster"]}
    fit = att_gt(to_spark(scalable_panel), random_state=3, **spec)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "clustervars=[] asks for unit-level bootstrap draws. Since this att_gt fit on a Spark DataFrame drew "
            "one multiplier for each cluster, it holds no draws for single units."
        ),
    ):
        aggte(fit, type="dynamic", clustervars=[])


@pytest.mark.parametrize(
    "option, message",
    [
        (
            {"panel": False},
            "panel=False is not supported for a Spark DataFrame. Pass panel=True to fit the data as a balanced panel.",
        ),
        (
            {"allow_unbalanced_panel": True},
            "allow_unbalanced_panel=True is not supported for a Spark DataFrame. "
            "Pass allow_unbalanced_panel=False to keep the units observed in every period.",
        ),
        (
            {"est_method": drdid_panel},
            "est_method=drdid_panel is not supported for a Spark DataFrame. Pass est_method='dr', 'ipw', or 'reg'.",
        ),
        (
            {"backend": "numpy"},
            "backend='numpy' is not supported for a Spark DataFrame. Pass backend=None to fit on the Spark executors.",
        ),
        (
            {"n_jobs": 2},
            "n_jobs=2 is not supported for a Spark DataFrame. "
            "Pass n_jobs=1 and let the Spark executors run the fit in parallel.",
        ),
        (
            {"random_state": "7"},
            "random_state='7' is not valid for a Spark DataFrame. Pass None, a non-negative integer, a SeedSequence, "
            "a BitGenerator, or a Generator.",
        ),
    ],
    ids=["repeated_cross_sections", "unbalanced_panel", "custom_estimator", "backend", "parallel_jobs", "text_seed"],
)
def test_att_gt_on_spark_rejects_options_it_cannot_fit(mpdta_data, to_spark, option, message):
    frame = to_spark(mpdta_data)

    with pytest.raises(ValueError, match=re.escape(message)):
        att_gt(frame, yname="lemp", tname="year", idname="countyreal", gname="first.treat", **option)


def test_att_gt_on_spark_rejects_a_pyspark_older_than_3_5(mpdta_data, to_spark, monkeypatch):
    frame = to_spark(mpdta_data)
    monkeypatch.setattr(pyspark, "__version__", "3.4.1")

    with pytest.raises(
        ValueError,
        match=re.escape(
            "pyspark 3.4.1 is installed. A fit of att_gt on a Spark DataFrame needs pyspark 3.5 or later. Upgrade "
            "pyspark or, on Databricks, choose Databricks Runtime 16 or later, as the Spark guide of the moderndid "
            "docs describes."
        ),
    ):
        att_gt(frame, yname="lemp", tname="year", idname="countyreal", gname="first.treat")
