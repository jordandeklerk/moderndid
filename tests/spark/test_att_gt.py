"""Tests for att_gt, aggte, and honest_did on data held in a Spark DataFrame."""

import re
from functools import partial

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from scipy import stats

from moderndid import aggte, att_gt, drdid_panel, honest_did, preprocess_did, to_df
from moderndid.core.bootstrap import canonical_keys, keyed_rademacher
from moderndid.did.shards.layout import row_keys, row_shard_from_arrow
from moderndid.did.shards.row_driver import fit_att_gt_on_row_shards
from tests.helpers import importorskip

pyspark = importorskip("pyspark")

from pyspark.sql import functions as F

from moderndid.spark.executor import SparkExecutor

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


@pytest.mark.parametrize(
    "data_fixture, spec, messages, n_rows",
    [
        (
            "mpdta_data",
            {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "xformla": "~lpop"},
            [
                "panel=False was specified, but units appear across all time periods. Your data appears to be panel "
                "data. Consider setting panel=True to use the panel estimator."
            ],
            2500,
        ),
        (
            "mpdta_data",
            {
                "yname": "lemp",
                "tname": "year",
                "gname": "first.treat",
                "xformla": "~lpop",
                "est_method": "ipw",
                "control_group": "notyettreated",
                "base_period": "universal",
            },
            [],
            2500,
        ),
        (
            "mpdta_data",
            {
                "yname": "lemp",
                "tname": "year",
                "gname": "first.treat",
                "xformla": "~lpop",
                "est_method": "reg",
                "control_group": "notyettreated",
                "anticipation": 1,
            },
            [
                "anticipation = 1. Never-treated units (with group status 0 or inf) are assumed to never anticipate "
                "treatment. Anticipation only applies to eventually-treated units.",
                "Dropped 100 observations that were already treated in the first period",
            ],
            2400,
        ),
        (
            "mpdta_with_weights_and_states",
            {
                "yname": "lemp",
                "tname": "year",
                "gname": "first.treat",
                "xformla": "~lpop",
                "weightsname": "w",
                "clustervars": ["state"],
            },
            [
                "The units fall in only 29 clusters. The treated units of cohort 2004 fall in 1 cluster and those of "
                "cohort 2006 in 3 clusters. Cluster-robust inference rests on very few clusters and can give "
                "confidence intervals and bands that are too narrow."
            ],
            2500,
        ),
        (
            "scalable_cross_sections",
            {
                "yname": "y",
                "tname": "time",
                "gname": "group",
                "xformla": "~cov1 + cov2 + cov3 + cov4",
                "weightsname": "w",
                "clustervars": ["cluster"],
            },
            [],
            400,
        ),
    ],
    ids=["identifiers_dr", "ipw_universal_base", "reg_anticipation", "weights_and_text_clusters", "four_covariates"],
)
def test_att_gt_on_spark_matches_att_gt_on_repeated_cross_sections(
    request, to_spark, record_warnings, assert_same_fit, data_fixture, spec, messages, n_rows
):
    data = request.getfixturevalue(data_fixture)
    frame = to_spark(data)

    memory, memory_warnings = record_warnings(att_gt, data, panel=False, random_state=7, **spec)
    distributed, spark_warnings = record_warnings(att_gt, frame, panel=False, random_state=7, **spec)

    assert [text for _, text in memory_warnings] == messages
    assert spark_warnings == memory_warnings
    assert distributed.n_units == n_rows
    assert_same_fit(distributed, memory, 1e-12)


@pytest.mark.parametrize(
    "data_fixture, spec, messages",
    [
        (
            "mpdta_unbalanced",
            {
                "yname": "lemp",
                "tname": "year",
                "idname": "countyreal",
                "gname": "first.treat",
                "xformla": "~lpop",
                "base_period": "universal",
            },
            [],
        ),
        (
            "mpdta_unbalanced",
            {
                "yname": "lemp",
                "tname": "year",
                "idname": "countyreal",
                "gname": "first.treat",
                "xformla": "~lpop",
                "est_method": "ipw",
                "control_group": "notyettreated",
                "weightsname": "w",
                "clustervars": ["countyreal", "state"],
            },
            [
                "Time-varying weights detected. For balanced panel data, each 2x2 comparison uses the weights from "
                "the earlier of its two periods, the base period for post-treatment cells.",
                "The units fall in only 29 clusters. The treated units of cohort 2004 fall in 1 cluster and those of "
                "cohort 2006 in 3 clusters. Cluster-robust inference rests on very few clusters and can give "
                "confidence intervals and bands that are too narrow.",
            ],
        ),
        (
            "scalable_unbalanced_panel",
            {
                "yname": "y",
                "tname": "time",
                "idname": "id",
                "gname": "group",
                "xformla": "~cov1 + cov2 + cov3 + cov4",
                "est_method": "reg",
                "control_group": "notyettreated",
                "clustervars": ["cluster"],
            },
            [],
        ),
    ],
    ids=["dr_universal_base", "ipw_time_varying_weights_and_text_clusters", "reg_four_covariates_and_clusters"],
)
def test_att_gt_on_spark_matches_att_gt_on_an_unbalanced_panel(
    request, to_spark, record_warnings, assert_same_fit, data_fixture, spec, messages
):
    data = request.getfixturevalue(data_fixture)
    frame = to_spark(data)

    memory, memory_warnings = record_warnings(att_gt, data, allow_unbalanced_panel=True, random_state=7, **spec)
    distributed, spark_warnings = record_warnings(att_gt, frame, allow_unbalanced_panel=True, random_state=7, **spec)

    assert [text for _, text in memory_warnings] == messages
    assert spark_warnings == memory_warnings
    assert distributed.n_units == memory.n_units == data[spec["idname"]].n_unique()
    assert data.group_by(spec["idname"]).len()["len"].min() < data[spec["tname"]].n_unique()
    assert_same_fit(distributed, memory, 1e-12)


def test_att_gt_on_spark_draws_a_multiplier_for_each_duplicate_row_however_the_rows_are_partitioned(
    mpdta_with_repeated_rows, to_spark, row_table_of, assert_same_fit
):
    data = mpdta_with_repeated_rows.with_columns(pl.col("lpop").exp().alias("pop"))
    columns = {
        "yname": "lemp",
        "tname": "year",
        "gname": "first.treat",
        "xformla": "~lpop",
        "weightsname": "pop",
        "panel": False,
    }
    spec = {**columns, "est_method": "reg", "boot": True, "biters": 299}
    frame = to_spark(data)

    one = att_gt(frame.repartition(1), random_state=5, **spec)
    table, config, cohort_counts, *_ = row_table_of(frame.repartition(9), draws=True, **spec)
    spread = table.repartition(9)
    nine = fit_att_gt_on_row_shards(
        SparkExecutor(spread, partial(row_shard_from_arrow, n_covariates=1)),
        config,
        cohort_counts,
        5,
        {"bootstrap": True, "uniform_bands": True},
    )
    memory = att_gt(data, random_state=5, **spec)
    n_tasks = spread.select(F.spark_partition_id()).distinct().count()
    rows = preprocess_did(data, **columns).data
    values = rows.select("year", "first.treat", "lemp", "pop", "lpop")
    copy = values.select(pl.int_range(pl.len()).over(values.columns)).to_series().to_numpy()
    fields = (
        np.unique(rows["year"].to_numpy(), return_inverse=True)[1],
        rows["first.treat"].to_numpy().astype(float),
        rows["lemp"].to_numpy(),
        rows["pop"].to_numpy(),
        rows.select("lpop").to_numpy(),
    )
    expected = keyed_rademacher(row_keys(*fields, copy), 5, 299).T @ memory.influence_func
    shared = keyed_rademacher(row_keys(*fields, np.zeros_like(copy)), 5, 299).T @ memory.influence_func
    sums = one.influence_moments.bootstrap_sums[:, : len(memory.att_gt)]

    assert n_tasks == 9
    assert copy.max() == 1
    assert one.influence_moments.random_state == 5
    assert_same_fit(nine, one, 1e-12)
    np.testing.assert_allclose(one.att_gt, memory.att_gt, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(one.vcov_analytical, memory.vcov_analytical, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        nine.influence_moments.bootstrap_sums, one.influence_moments.bootstrap_sums, rtol=1e-12, atol=1e-10
    )
    np.testing.assert_allclose(sums, expected, rtol=1e-12, atol=1e-10)
    assert np.max(np.abs(shared - expected)) > 1e-3


def test_att_gt_on_spark_draws_one_multiplier_for_each_unit_of_an_unbalanced_panel(mpdta_unbalanced, to_spark):
    columns = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "allow_unbalanced_panel": True,
    }
    spec = {**columns, "boot": True, "biters": 299, "random_state": 5}

    distributed = att_gt(to_spark(mpdta_unbalanced), **spec)
    memory = att_gt(mpdta_unbalanced, **spec)
    units = preprocess_did(mpdta_unbalanced, **columns).time_invariant_data["countyreal"].to_numpy()
    expected = keyed_rademacher(canonical_keys(units), 5, 299).T @ memory.influence_func
    analytic = np.sqrt(np.diag(distributed.vcov_analytical) / distributed.n_units)

    assert distributed.n_units == len(units) == 500
    np.testing.assert_allclose(distributed.att_gt, memory.att_gt, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(distributed.vcov_analytical, memory.vcov_analytical, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        distributed.influence_moments.bootstrap_sums[:, : len(memory.att_gt)], expected, rtol=1e-12, atol=1e-10
    )
    assert np.nanmax(np.abs(distributed.se_gt / analytic - 1)) > 1e-3


@pytest.mark.parametrize(
    "data_fixture, table_fixture, options, cluster_dtype, reference_dtype",
    [
        ("large_clustered_panel", "row_table_of", {"panel": False}, pl.String, pl.String),
        (
            "large_clustered_unbalanced_panel",
            "unit_table_of",
            {"idname": "id", "allow_unbalanced_panel": True},
            pl.Decimal(18, 0),
            pl.Int64,
        ),
    ],
    ids=["repeated_cross_sections_with_text_clusters", "unbalanced_panel_with_decimal_clusters"],
)
def test_att_gt_on_spark_spreads_the_clusters_or_keeps_them_whole_in_either_layout(
    request,
    to_spark,
    assert_same_fit,
    monkeypatch,
    data_fixture,
    table_fixture,
    options,
    cluster_dtype,
    reference_dtype,
):
    data = request.getfixturevalue(data_fixture)
    table_of = request.getfixturevalue(table_fixture)
    columns = {
        "yname": "y",
        "tname": "time",
        "gname": "group",
        "xformla": "~cov1",
        "clustervars": ["cluster"],
        **options,
    }
    spec = {**columns, "est_method": "reg", "boot": True, "biters": 199, "random_state": 3}
    spark_data = data.with_columns(pl.col("cluster").cast(cluster_dtype))
    reference_data = data.with_columns(pl.col("cluster").cast(reference_dtype))
    frame = to_spark(spark_data).repartition(4)

    spread_table, *_, spread_is_whole = table_of(frame, **columns)
    spread_pairs = spread_table.select(F.spark_partition_id(), "cluster").distinct().collect()
    spread = att_gt(frame, **spec)
    monkeypatch.setattr("moderndid.spark.preprocess._spread_clusters", lambda n_clusters, n_columns: False)
    whole_table, *_, whole_is_whole = table_of(frame, **columns)
    whole_pairs = whole_table.select(F.spark_partition_id(), "cluster").distinct().collect()
    whole = att_gt(frame, **spec)
    memory = att_gt(reference_data, **spec)
    labels, codes = np.unique(memory.estimation_params["cluster"], return_inverse=True)
    cluster_sums = np.zeros((len(labels), len(memory.att_gt)))
    np.add.at(cluster_sums, codes.ravel(), memory.influence_func)
    expected = keyed_rademacher(canonical_keys(labels), 3, 199).T @ cluster_sums

    assert (spread_is_whole, whole_is_whole) == (False, True)
    assert len({part for part, _ in spread_pairs}) >= 2
    assert len({cluster for _, cluster in spread_pairs}) < len(spread_pairs)
    assert len({part for part, _ in whole_pairs}) >= 2
    assert len({cluster for _, cluster in whole_pairs}) == len(whole_pairs)
    assert spread.influence_moments.n_clusters == whole.influence_moments.n_clusters == len(labels) == 50
    np.testing.assert_allclose(spread.att_gt, memory.att_gt, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(spread.vcov_analytical, memory.vcov_analytical, rtol=1e-12, atol=1e-12)
    assert_same_fit(whole, spread, 1e-12)
    np.testing.assert_allclose(
        whole.influence_moments.bootstrap_sums, spread.influence_moments.bootstrap_sums, rtol=1e-12, atol=1e-10
    )
    np.testing.assert_allclose(
        spread.influence_moments.bootstrap_sums[:, : len(memory.att_gt)], expected, rtol=1e-12, atol=1e-10
    )


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


def test_att_gt_on_spark_spreads_few_clusters_over_the_partitions_of_the_unit_table(
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

    unit_table, *_, whole_clusters = unit_table_of(large_clustered_panel, **columns)
    partitions = unit_table.select(F.spark_partition_id().alias("part"), "cluster")
    n_partitions = partitions.select("part").distinct().count()
    parts_per_cluster = partitions.groupBy("cluster").agg(F.count_distinct("part").alias("parts"))
    split_clusters = parts_per_cluster.filter(F.col("parts") > 1).count()
    distributed = att_gt(to_spark(large_clustered_panel), **spec)
    memory = att_gt(large_clustered_panel, **spec)
    local = att_gt_on_shards(large_clustered_panel, 3, **spec)
    moments, expected = distributed.influence_moments, local.influence_moments

    assert n_partitions >= 2
    assert not whole_clusters
    assert split_clusters > 0
    assert parts_per_cluster.count() == moments.n_clusters == 50
    np.testing.assert_array_equal(distributed.groups, memory.groups)
    np.testing.assert_array_equal(distributed.times, memory.times)
    np.testing.assert_allclose(distributed.att_gt, memory.att_gt, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(distributed.vcov_analytical, memory.vcov_analytical, rtol=1e-12, atol=1e-12)
    assert_same_fit(distributed, local, 1e-12)
    np.testing.assert_allclose(moments.cluster_gram, expected.cluster_gram, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(moments.bootstrap_sums, expected.bootstrap_sums, rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize(
    "cluster_dtype",
    [pl.Int64, pl.String, pl.Decimal(18, 0)],
    ids=["integer_clusters", "text_clusters", "decimal_clusters"],
)
def test_att_gt_on_spark_keeps_each_cluster_whole_once_the_clusters_are_many(
    large_clustered_panel, to_spark, unit_table_of, assert_same_fit, monkeypatch, cluster_dtype
):
    data = large_clustered_panel.with_columns(pl.col("cluster").cast(cluster_dtype))
    columns = {
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "group",
        "xformla": "~cov1",
        "clustervars": ["cluster"],
    }
    spec = {**columns, "boot": True, "biters": 199, "random_state": 3}
    frame = to_spark(data)

    spread_table, *_, spread_is_whole = unit_table_of(data, **columns)
    spread_parts = spread_table.select(F.spark_partition_id().alias("part"), "cluster")
    spread_parts_per_cluster = spread_parts.groupBy("cluster").agg(F.count_distinct("part").alias("parts"))
    spread_split_clusters = spread_parts_per_cluster.filter(F.col("parts") > 1).count()
    spread = att_gt(frame, **spec)
    monkeypatch.setattr("moderndid.spark.preprocess._spread_clusters", lambda n_clusters, n_columns: False)
    whole_table, *_, whole_is_whole = unit_table_of(data, **columns)
    whole_parts = whole_table.select(F.spark_partition_id().alias("part"), "cluster")
    n_whole_partitions = whole_parts.select("part").distinct().count()
    whole_parts_per_cluster = whole_parts.groupBy("cluster").agg(F.count_distinct("part").alias("parts"))
    whole_split_clusters = whole_parts_per_cluster.filter(F.col("parts") > 1).count()
    whole = att_gt(frame, **spec)
    moments, expected = whole.influence_moments, spread.influence_moments

    assert (spread_is_whole, whole_is_whole) == (False, True)
    assert spread_split_clusters > 0
    assert n_whole_partitions >= 2
    assert whole_split_clusters == 0
    assert moments.n_clusters == expected.n_clusters == 50
    np.testing.assert_allclose(moments.gram, expected.gram, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(moments.cluster_gram, expected.cluster_gram, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(moments.bootstrap_sums, expected.bootstrap_sums, rtol=1e-12, atol=1e-10)
    assert_same_fit(whole, spread, 1e-12)


def test_att_gt_on_spark_keeps_clusters_whole_once_spreading_them_would_pass_the_budget(
    finely_clustered_panel, to_spark, unit_table_of, assert_same_fit
):
    spec = {
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "group",
        "xformla": "~cov1",
        "clustervars": ["cluster"],
        "est_method": "reg",
    }

    unit_table, *_, whole_clusters = unit_table_of(finely_clustered_panel, **spec)
    partitions = unit_table.select(F.spark_partition_id().alias("part"), "cluster")
    n_partitions = partitions.select("part").distinct().count()
    parts_per_cluster = partitions.groupBy("cluster").agg(F.count_distinct("part").alias("parts"))
    split_clusters = parts_per_cluster.filter(F.col("parts") > 1).count()
    distributed = att_gt(to_spark(finely_clustered_panel), cband=False, **spec)
    memory = att_gt(finely_clustered_panel, cband=False, **spec)

    assert whole_clusters
    assert n_partitions >= 2
    assert split_clusters == 0
    assert distributed.influence_moments.n_clusters == finely_clustered_panel["cluster"].n_unique() == 10000
    assert_same_fit(distributed, memory, 1e-12)


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
@pytest.mark.parametrize(
    "fits",
    ["universal_base_fits", "repeated_cross_section_fits", "unbalanced_panel_fits"],
    ids=["balanced_panel", "repeated_cross_sections", "unbalanced_panel"],
)
def test_aggte_on_a_spark_fit_matches_aggte_in_memory(request, fits, aggregation):
    on_spark, in_memory = request.getfixturevalue(fits)

    distributed = aggte(on_spark, type=aggregation, cband=False)
    memory = aggte(in_memory, type=aggregation, cband=False)

    np.testing.assert_array_equal(distributed.event_times, memory.event_times)
    np.testing.assert_allclose(distributed.att_by_event, memory.att_by_event, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(distributed.se_by_event, memory.se_by_event, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(distributed.vcov_analytical, memory.vcov_analytical, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(distributed.overall_att, memory.overall_att, rtol=1e-12)
    np.testing.assert_allclose(distributed.overall_se, memory.overall_se, rtol=1e-12)


@pytest.mark.parametrize(
    "fits",
    ["universal_base_fits", "repeated_cross_section_fits", "unbalanced_panel_fits"],
    ids=["balanced_panel", "repeated_cross_sections", "unbalanced_panel"],
)
def test_aggte_simple_on_a_spark_fit_matches_aggte_in_memory(request, fits):
    on_spark, in_memory = request.getfixturevalue(fits)

    distributed = aggte(on_spark, type="simple", cband=False)
    memory = aggte(in_memory, type="simple", cband=False)

    np.testing.assert_allclose(distributed.overall_att, memory.overall_att, rtol=1e-12)
    np.testing.assert_allclose(distributed.overall_se, memory.overall_se, rtol=1e-12)
    np.testing.assert_allclose(distributed.vcov_analytical, memory.vcov_analytical, rtol=1e-12)


@pytest.mark.parametrize(
    "fits, sensitivity_type, options",
    [
        ("universal_base_fits", "smoothness", {"m_vec": np.array([0.0, 0.01])}),
        ("universal_base_fits", "relative_magnitude", {"m_bar_vec": np.array([0.0, 1.0]), "grid_points": 20}),
        ("repeated_cross_section_fits", "smoothness", {"m_vec": np.array([0.0, 0.01])}),
        ("unbalanced_panel_fits", "smoothness", {"m_vec": np.array([0.0, 0.01])}),
    ],
    ids=[
        "balanced_panel_smoothness",
        "balanced_panel_relative_magnitude",
        "repeated_cross_sections_smoothness",
        "unbalanced_panel_smoothness",
    ],
)
def test_honest_did_on_an_event_study_of_a_spark_fit_matches_the_in_memory_fit(
    request, fits, sensitivity_type, options
):
    on_spark, in_memory = request.getfixturevalue(fits)
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


@pytest.mark.parametrize(
    "fits",
    ["universal_base_fits", "repeated_cross_section_fits", "unbalanced_panel_fits"],
    ids=["balanced_panel", "repeated_cross_sections", "unbalanced_panel"],
)
def test_to_df_of_a_spark_fit_matches_the_in_memory_fit(request, fits):
    on_spark, in_memory = request.getfixturevalue(fits)

    cells = to_df(on_spark)
    expected_cells = to_df(in_memory)
    event_study = to_df(aggte(on_spark, type="dynamic", cband=False))
    expected_event_study = to_df(aggte(in_memory, type="dynamic", cband=False))

    assert_frame_equal(cells, expected_cells, rel_tol=1e-12, abs_tol=1e-12)
    assert_frame_equal(event_study, expected_event_study, rel_tol=1e-12, abs_tol=1e-12)


@pytest.mark.parametrize(
    "fits",
    ["universal_base_fits", "repeated_cross_section_fits", "unbalanced_panel_fits"],
    ids=["balanced_panel", "repeated_cross_sections", "unbalanced_panel"],
)
def test_report_of_a_spark_fit_matches_the_in_memory_report(request, fits):
    on_spark, in_memory = request.getfixturevalue(fits)

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
            "clustervars=[] asks for unit-level bootstrap draws. This att_gt fit on a Spark DataFrame drew "
            "one multiplier for each cluster and holds no draws for single units."
        ),
    ):
        aggte(fit, type="dynamic", clustervars=[])


@pytest.mark.parametrize(
    "option, message",
    [
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
    ids=["custom_estimator", "backend", "parallel_jobs", "text_seed"],
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
            "pyspark or, on Databricks, choose Databricks Runtime 16.4 LTS or later. The Spark setup guide of the "
            "moderndid docs says which runtimes to choose."
        ),
    ):
        att_gt(frame, yname="lemp", tname="year", idname="countyreal", gname="first.treat")
