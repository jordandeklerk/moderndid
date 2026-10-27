"""Tests for the checks and the tables that prepare a Spark DataFrame for att_gt."""

import re

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from moderndid import aggte, att_gt, preprocess_did
from moderndid.core.dataframe import to_polars
from moderndid.core.preprocess import DIDConfig
from moderndid.did.shards.layout import row_keys, row_shard_from_arrow, shard_from_arrow
from moderndid.did.shards.local import shards_from_did_data
from tests.helpers import importorskip

importorskip("pyspark")

from pyspark.sql import functions as F

from moderndid.spark.preprocess import _long_facts, _spread_clusters

pytestmark = pytest.mark.spark


@pytest.mark.parametrize(
    "data_fixture, options, message",
    [
        ("mpdta_data", {"xformla": "~lpopp"}, "'lpopp' in xformla is not a column in the data. Did you mean 'lpop'?"),
        ("mpdta_with_text_outcome", {}, "Validation failed:\nyname = 'lemp' is not numeric. Please convert it"),
        (
            "mpdta_with_repeated_rows",
            {},
            "Validation failed:\nThe value of idname must be unique (by tname). Some units are observed more than "
            "once in a period. Rows repeat for 4 (countyreal, year) pairs, such as (8001, 2006), (8001, 2007), and "
            "(8019, 2006).",
        ),
        (
            "mpdta_with_varying_cohort",
            {},
            "Validation failed:\nThe value of gname (treatment variable) must be the same across all periods for "
            "each particular unit. The treatment must be irreversible.",
        ),
    ],
    ids=["missing_column", "text_outcome", "repeated_rows", "varying_cohort"],
)
def test_att_gt_on_spark_raises_the_errors_of_att_gt(request, to_spark, data_fixture, options, message):
    data = request.getfixturevalue(data_fixture)
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", **options}
    frame = to_spark(data)

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        att_gt(data, **spec)
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        att_gt(frame, **spec)


@pytest.mark.parametrize(
    "data_fixture, options, spread",
    [
        ("mpdta_data", {}, True),
        ("mpdta_data", {}, False),
        ("mpdta_missing_a_period", {"allow_unbalanced_panel": True}, True),
    ],
    ids=["spread_clusters", "whole_clusters", "unbalanced_panel"],
)
def test_att_gt_on_spark_rejects_a_cluster_that_varies_within_a_unit_in_either_layout(
    request, to_spark, monkeypatch, data_fixture, options, spread
):
    moved = (pl.col("countyreal") == 8001) & (pl.col("year") == 2007)
    states = request.getfixturevalue(data_fixture).with_columns((pl.col("countyreal") // 1000).alias("state"))
    data = states.with_columns(pl.when(moved).then(99).otherwise(pl.col("state")).alias("state"))
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "clustervars": ["state"],
        **options,
    }
    message = (
        "Validation failed:\nDiD cannot handle time-varying cluster variables at the moment. "
        "Please check your cluster variable."
    )
    frame = to_spark(data)
    monkeypatch.setattr("moderndid.spark.preprocess._spread_clusters", lambda n_clusters, n_columns: spread)

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        att_gt(data, **spec)
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        att_gt(frame, **spec)


@pytest.mark.parametrize(
    "data_fixture, options, message",
    [
        ("mpdta_missing_a_period", {}, "Dropped 45 units while converting to balanced panel"),
        ("mpdta_treated_in_the_first_period", {}, "Dropped 78 units that were already treated in the first period"),
        (
            "mpdta_with_states",
            {"clustervars": ["state"]},
            "The units fall in only 29 clusters. The treated units of cohort 2004 fall in 1 cluster and those of "
            "cohort 2006 in 3 clusters. Cluster-robust inference rests on very few clusters and can give confidence "
            "intervals and bands that are too narrow.",
        ),
    ],
    ids=["units_missing_a_period", "units_treated_in_the_first_period", "few_clusters"],
)
def test_att_gt_on_spark_warns_like_att_gt(request, to_spark, assert_same_fit, data_fixture, options, message):
    data = request.getfixturevalue(data_fixture)
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "random_state": 7,
        **options,
    }
    frame = to_spark(data)

    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$"):
        memory = att_gt(data, **spec)
    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$"):
        distributed = att_gt(frame, **spec)

    assert_same_fit(distributed, memory, 1e-12)


def test_att_gt_on_spark_drops_every_row_of_an_identifier_with_a_row_treated_in_the_first_period_like_att_gt(
    mpdta_with_early_rows, to_spark, record_warnings, assert_same_fit
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "panel": False,
        "random_state": 7,
    }
    frame = to_spark(mpdta_with_early_rows)
    early = mpdta_with_early_rows.filter(pl.col("first.treat") == 2003)["countyreal"].unique()
    kept = mpdta_with_early_rows.filter(~pl.col("countyreal").is_in(early.to_list()))

    memory, memory_warnings = record_warnings(att_gt, mpdta_with_early_rows, **spec)
    distributed, spark_warnings = record_warnings(att_gt, frame, **spec)

    assert [text for _, text in memory_warnings] == [
        "panel=False was specified, but units appear across all time periods. Your data appears to be panel data. "
        "Consider setting panel=True to use the panel estimator.",
        f"Dropped {len(early)} units that were already treated in the first period",
    ]
    assert spark_warnings == memory_warnings
    assert distributed.n_units == len(kept)
    assert len(kept) < len(mpdta_with_early_rows) - len(early)
    assert_same_fit(distributed, memory, 1e-12)


def test_att_gt_on_spark_fits_an_unbalanced_panel_that_turns_out_balanced_as_a_balanced_panel(
    mpdta_data, to_spark, record_warnings, assert_same_fit
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "random_state": 7,
    }
    frame = to_spark(mpdta_data)

    memory, memory_warnings = record_warnings(att_gt, mpdta_data, allow_unbalanced_panel=True, **spec)
    distributed, spark_warnings = record_warnings(att_gt, frame, allow_unbalanced_panel=True, **spec)
    balanced = att_gt(mpdta_data, **spec)
    with pytest.warns(UserWarning, match="^panel=False was specified"):
        rows = att_gt(mpdta_data, panel=False, **spec)

    assert spark_warnings == memory_warnings == []
    assert_same_fit(distributed, memory, 1e-12)
    assert_same_fit(distributed, balanced, 1e-12)
    assert np.max(np.abs(rows.se_gt - balanced.se_gt)) > 1e-3


def test_att_gt_on_spark_reads_decimal_periods_and_cohorts_as_doubles_and_decimal_ids_as_integers(
    mpdta_with_decimal_columns, mpdta_with_float_periods, to_spark, att_gt_on_shards, assert_same_fit
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "boot": True,
        "biters": 99,
        "random_state": 7,
    }
    frame = to_spark(mpdta_with_decimal_columns)

    distributed = att_gt(frame, **spec)
    memory = att_gt(mpdta_with_float_periods, **spec)
    local = att_gt_on_shards(mpdta_with_float_periods, 3, **spec)
    event_study = aggte(distributed, type="dynamic", cband=False)
    expected_event_study = aggte(memory, type="dynamic", cband=False)

    assert [frame.schema[name].dataType.simpleString() for name in ("year", "countyreal", "first.treat")] == [
        "decimal(10,0)"
    ] * 3
    np.testing.assert_allclose(distributed.att_gt, memory.att_gt, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(distributed.vcov_analytical, memory.vcov_analytical, rtol=1e-12, atol=1e-12)
    assert_same_fit(distributed, local, 1e-12)
    np.testing.assert_array_equal(event_study.event_times, expected_event_study.event_times)
    np.testing.assert_allclose(event_study.att_by_event, expected_event_study.att_by_event, rtol=1e-12, atol=1e-12)


def test_att_gt_on_spark_keeps_decimal_ids_above_two_to_the_53_apart_like_integer_ids(
    mpdta_with_large_ids, mpdta_with_large_decimal_ids, to_spark, att_gt_on_shards, assert_same_fit
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "boot": True,
        "biters": 99,
        "random_state": 7,
    }
    frame = to_spark(mpdta_with_large_decimal_ids)
    ids = np.unique(mpdta_with_large_ids["countyreal"].to_numpy())

    distributed = att_gt(frame, **spec)
    memory = att_gt(mpdta_with_large_ids, **spec)
    local = att_gt_on_shards(mpdta_with_large_ids, 3, **spec)

    assert frame.schema["countyreal"].dataType.simpleString() == "decimal(18,0)"
    assert ids[1] - ids[0] == 1
    assert float(ids[0]) == float(ids[1])
    assert distributed.n_units == 500
    np.testing.assert_allclose(distributed.att_gt, memory.att_gt, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(distributed.vcov_analytical, memory.vcov_analytical, rtol=1e-12, atol=1e-12)
    assert_same_fit(distributed, local, 1e-12)


def test_att_gt_on_spark_draws_for_decimal_clusters_as_for_the_same_integer_codes(
    scalable_panel, scalable_panel_with_decimal_clusters, to_spark, att_gt_on_shards, assert_same_fit
):
    spec = {
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "group",
        "xformla": "~cov1",
        "clustervars": ["cluster"],
        "boot": True,
        "biters": 299,
        "random_state": 3,
    }
    frame = to_spark(scalable_panel_with_decimal_clusters)

    distributed = att_gt(frame, **spec)
    local = att_gt_on_shards(scalable_panel, 3, **spec)
    moments, expected = distributed.influence_moments, local.influence_moments
    analytic = np.sqrt(np.diag(distributed.vcov_analytical) / distributed.n_units)

    assert frame.schema["cluster"].dataType.simpleString() == "decimal(18,0)"
    assert_same_fit(distributed, local, 1e-12)
    np.testing.assert_allclose(moments.cluster_gram, expected.cluster_gram, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(moments.bootstrap_sums, expected.bootstrap_sums, rtol=1e-12, atol=1e-10)
    assert moments.n_clusters == 50
    assert np.nanmax(np.abs(distributed.se_gt / analytic - 1)) > 1e-3


@pytest.mark.parametrize("collector", ["to_pandas", "to_arrow"])
def test_att_gt_on_spark_matches_att_gt_on_collected_decimal_outcome_covariate_and_weights(
    request, mpdta_with_decimal_values, to_spark, assert_same_fit, collector
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "weightsname": "pop",
        "random_state": 7,
    }
    frame = to_spark(mpdta_with_decimal_values)
    collected = request.getfixturevalue(collector)(frame)

    distributed = att_gt(frame, **spec)
    memory = att_gt(collected, **spec)

    assert [frame.schema[name].dataType.simpleString() for name in ("lemp", "lpop", "pop")] == ["decimal(18,6)"] * 3
    assert [to_polars(collected).schema[name].is_decimal() for name in ("lemp", "lpop", "pop")] == [True] * 3
    assert_same_fit(distributed, memory, 1e-12)


@pytest.mark.parametrize(
    "data_fixture, options, message",
    [
        (
            "mpdta_with_decimal_ids_of_19_digits",
            {},
            "idname='countyreal' is a decimal(19,0) column. A Spark fit reads decimal identifiers and clusters "
            "as 64-bit integers. They must have no fraction digits and at most 18 digits. Cast the column to bigint or "
            "code the units as integers first.",
        ),
        (
            "mpdta_with_fractional_decimal_ids",
            {},
            "idname='countyreal' is a decimal(12,2) column. A Spark fit reads decimal identifiers and clusters "
            "as 64-bit integers. They must have no fraction digits and at most 18 digits. Cast the column to bigint or "
            "code the units as integers first.",
        ),
        (
            "mpdta_with_fractional_decimal_states",
            {"clustervars": ["countyreal", "state"]},
            "'state' in clustervars is a decimal(10,2) column. A Spark fit reads decimal identifiers and "
            "clusters as 64-bit integers. They must have no fraction digits and at most 18 digits. Cast the column to "
            "bigint or code the clusters as integers first.",
        ),
        (
            "mpdta_with_fractional_decimal_ids",
            {"panel": False},
            "idname='countyreal' is a decimal(12,2) column. A Spark fit reads decimal identifiers and clusters "
            "as 64-bit integers. They must have no fraction digits and at most 18 digits. Cast the column to bigint or "
            "code the units as integers first.",
        ),
        (
            "mpdta_with_fractional_decimal_states",
            {"panel": False, "clustervars": ["state"]},
            "'state' in clustervars is a decimal(10,2) column. A Spark fit reads decimal identifiers and "
            "clusters as 64-bit integers. They must have no fraction digits and at most 18 digits. Cast the column to "
            "bigint or code the clusters as integers first.",
        ),
        (
            "mpdta_with_decimal_ids_of_19_digits",
            {"allow_unbalanced_panel": True},
            "idname='countyreal' is a decimal(19,0) column. A Spark fit reads decimal identifiers and clusters "
            "as 64-bit integers. They must have no fraction digits and at most 18 digits. Cast the column to bigint or "
            "code the units as integers first.",
        ),
    ],
    ids=[
        "decimal_ids_of_19_digits",
        "fractional_decimal_ids",
        "fractional_decimal_clusters",
        "fractional_decimal_ids_of_repeated_cross_sections",
        "fractional_decimal_clusters_of_repeated_cross_sections",
        "decimal_ids_of_19_digits_of_an_unbalanced_panel",
    ],
)
def test_att_gt_on_spark_rejects_decimal_ids_and_clusters_that_64_bit_integers_cannot_hold(
    request, to_spark, data_fixture, options, message
):
    frame = to_spark(request.getfixturevalue(data_fixture))

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        att_gt(frame, yname="lemp", tname="year", idname="countyreal", gname="first.treat", **options)


def test_build_unit_table_holds_each_unit_in_the_layout_that_shards_read(
    mpdta_with_weights_and_states, unit_table_of, to_arrow
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "weightsname": "w",
        "clustervars": ["state"],
    }

    did_data = preprocess_did(mpdta_with_weights_and_states, panel=True, allow_unbalanced_panel=False, **spec)
    expected = shards_from_did_data(did_data, 1)[0]
    unit_table, config, *_ = unit_table_of(mpdta_with_weights_and_states, **spec)
    shard = shard_from_arrow(to_arrow(unit_table), config.time_periods_count, 1)
    order = np.argsort(shard.unit_keys)
    expected_order = np.argsort(expected.unit_keys)

    assert unit_table.columns == ["unit", "cohort", "y", "w", "x", "cluster"]
    np.testing.assert_array_equal(shard.unit_keys[order], expected.unit_keys[expected_order])
    np.testing.assert_array_equal(shard.cohort[order], expected.cohort[expected_order])
    np.testing.assert_array_equal(shard.y[order], expected.y[expected_order])
    np.testing.assert_allclose(shard.weights[order], expected.weights[expected_order], rtol=1e-14)
    np.testing.assert_array_equal(shard.covariates[order], expected.covariates[expected_order])
    np.testing.assert_array_equal(shard.cluster_keys[order], expected.cluster_keys[expected_order])


@pytest.mark.parametrize(
    "data_fixture, options",
    [
        ("mpdta_missing_a_period", {"xformla": "~lpop"}),
        ("mpdta_without_never_treated", {"control_group": "notyettreated", "anticipation": 1}),
        ("mpdta_with_float_periods", {"xformla": "~lpop"}),
    ],
    ids=["units_missing_a_period", "without_never_treated", "float_periods"],
)
def test_build_unit_table_sets_the_facts_of_the_in_memory_preprocessing(
    request, unit_table_of, record_warnings, data_fixture, options
):
    data = request.getfixturevalue(data_fixture)
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", **options}

    did_data, memory_warnings = record_warnings(preprocess_did, data, panel=True, allow_unbalanced_panel=False, **spec)
    (_, config, cohort_counts, n_obs, cluster_counts, whole_clusters), spark_warnings = record_warnings(
        unit_table_of, data, **spec
    )
    expected = did_data.config

    assert spark_warnings == memory_warnings
    np.testing.assert_array_equal(config.time_periods, expected.time_periods)
    np.testing.assert_array_equal(config.treated_groups, expected.treated_groups)
    assert (config.time_periods.dtype, config.treated_groups.dtype) == (
        expected.time_periods.dtype,
        expected.treated_groups.dtype,
    )
    assert (config.time_periods_count, config.treated_groups_count, config.id_count) == (
        expected.time_periods_count,
        expected.treated_groups_count,
        expected.id_count,
    )
    assert (config.cband, config.data_format) == (expected.cband, expected.data_format)
    assert_frame_equal(cohort_counts, did_data.cohort_counts)
    assert n_obs == len(did_data.data)
    assert cluster_counts is None
    assert whole_clusters is False


def test_build_unit_table_counts_the_clusters_of_all_units_and_of_each_cohort(mpdta_with_state_names, unit_table_of):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "clustervars": ["countyreal", "state"],
    }

    units = preprocess_did(mpdta_with_state_names, panel=True, allow_unbalanced_panel=False, **spec).time_invariant_data
    per_cohort = units.group_by("first.treat").agg(pl.col("state").n_unique())
    *_, cluster_counts, _ = unit_table_of(mpdta_with_state_names, **spec)

    assert cluster_counts == (units["state"].n_unique(), dict(per_cohort.iter_rows()))
    assert cluster_counts[0] == 29


@pytest.mark.parametrize(
    "data_fixture, options, messages",
    [
        (
            "mpdta_data",
            {"idname": "countyreal", "xformla": "~lpop"},
            [
                "panel=False was specified, but units appear across all time periods. Your data appears to be panel "
                "data. Consider setting panel=True to use the panel estimator."
            ],
        ),
        (
            "mpdta_with_early_rows",
            {"idname": "countyreal", "xformla": "~lpop", "control_group": "notyettreated"},
            [
                "panel=False was specified, but units appear across all time periods. Your data appears to be panel "
                "data. Consider setting panel=True to use the panel estimator.",
                "Dropped 74 units that were already treated in the first period",
            ],
        ),
        (
            "mpdta_without_never_treated",
            {"xformla": "~lpop"},
            ["No never-treated group is available. The last treated cohort is being coerced as 'never-treated' units."],
        ),
        (
            "mpdta_with_a_small_cohort",
            {"xformla": "~lpop"},
            ["Be aware that there are some small groups in your dataset.\nCheck groups: 2004"],
        ),
    ],
    ids=["identifiers", "early_identifiers", "without_never_treated", "small_cohort"],
)
def test_build_row_table_sets_the_facts_of_the_in_memory_preprocessing(
    request, row_table_of, record_warnings, data_fixture, options, messages
):
    data = request.getfixturevalue(data_fixture)
    spec = {"yname": "lemp", "tname": "year", "gname": "first.treat", **options}

    did_data, memory_warnings = record_warnings(preprocess_did, data, panel=False, **spec)
    (_, config, cohort_counts, n_obs, cluster_counts, whole_clusters), spark_warnings = record_warnings(
        row_table_of, data, **spec
    )
    expected = did_data.config

    assert [text for _, text in memory_warnings] == messages
    assert spark_warnings == memory_warnings
    np.testing.assert_array_equal(config.time_periods, expected.time_periods)
    np.testing.assert_array_equal(config.treated_groups, expected.treated_groups)
    assert (config.time_periods.dtype, config.treated_groups.dtype) == (
        expected.time_periods.dtype,
        expected.treated_groups.dtype,
    )
    assert (config.panel, config.true_repeated_cross_sections, config.idname, config.data_format) == (
        expected.panel,
        expected.true_repeated_cross_sections,
        expected.idname,
        expected.data_format,
    )
    assert (config.time_periods_count, config.treated_groups_count, config.id_count, config.cband) == (
        expected.time_periods_count,
        expected.treated_groups_count,
        expected.id_count,
        expected.cband,
    )
    assert_frame_equal(cohort_counts, did_data.cohort_counts.sort("cohort"), check_dtypes=False)
    assert n_obs == config.id_count == len(did_data.data)
    assert cluster_counts is None
    assert whole_clusters is False


def test_build_row_table_counts_the_clusters_of_all_rows_and_of_each_cohort(
    mpdta_with_weights_and_states, row_table_of
):
    data = mpdta_with_weights_and_states.with_columns(((pl.col("countyreal") + pl.col("year")) % 7).alias("cell"))
    spec = {"yname": "lemp", "tname": "year", "gname": "first.treat", "weightsname": "w", "clustervars": ["cell"]}

    rows = preprocess_did(data, panel=False, **spec).time_invariant_data
    per_cohort = rows.group_by("first.treat").agg(pl.col("cell").n_unique())
    *_, cluster_counts, whole_clusters = row_table_of(data, **spec)

    assert cluster_counts == (rows["cell"].n_unique(), dict(per_cohort.iter_rows()))
    assert cluster_counts[0] == 7
    assert data.group_by("countyreal").agg(pl.col("cell").n_unique())["cell"].min() > 1
    assert whole_clusters is False


@pytest.mark.parametrize(
    "data_fixture, options, messages",
    [
        ("mpdta_missing_a_period", {"xformla": "~lpop"}, []),
        (
            "mpdta_without_never_treated",
            {"xformla": "~lpop"},
            ["No never-treated group is available. The last treated cohort is being coerced as 'never-treated' units."],
        ),
        ("mpdta_unbalanced", {"xformla": "~lpop", "weightsname": "w", "control_group": "notyettreated"}, []),
    ],
    ids=["units_missing_a_period", "balanced_without_never_treated", "time_varying_weights"],
)
def test_build_unit_table_sets_the_facts_of_an_unbalanced_panel_like_the_in_memory_preprocessing(
    request, unit_table_of, record_warnings, data_fixture, options, messages
):
    data = request.getfixturevalue(data_fixture)
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "allow_unbalanced_panel": True,
        **options,
    }

    did_data, memory_warnings = record_warnings(preprocess_did, data, **spec)
    (_, config, cohort_counts, n_obs, cluster_counts, whole_clusters), spark_warnings = record_warnings(
        unit_table_of, data, **spec
    )
    expected = did_data.config

    assert [text for _, text in memory_warnings] == messages
    assert spark_warnings == memory_warnings
    np.testing.assert_array_equal(config.time_periods, expected.time_periods)
    np.testing.assert_array_equal(config.treated_groups, expected.treated_groups)
    assert (config.time_periods.dtype, config.treated_groups.dtype) == (
        expected.time_periods.dtype,
        expected.treated_groups.dtype,
    )
    assert (
        config.panel,
        config.allow_unbalanced_panel,
        config.true_repeated_cross_sections,
        config.data_format,
    ) == (
        expected.panel,
        expected.allow_unbalanced_panel,
        expected.true_repeated_cross_sections,
        expected.data_format,
    )
    assert (config.time_periods_count, config.treated_groups_count, config.id_count, config.cband) == (
        expected.time_periods_count,
        expected.treated_groups_count,
        expected.id_count,
        expected.cband,
    )
    assert_frame_equal(cohort_counts, did_data.cohort_counts.sort("cohort"))
    assert n_obs == len(did_data.data)
    assert cluster_counts is None
    assert whole_clusters is False


def test_build_row_table_gives_each_copy_of_a_row_its_own_key_whatever_the_sign_of_its_zeros(
    mpdta_data, row_table_of, to_arrow
):
    zeros = mpdta_data.filter(pl.col("countyreal") % 50 == 0).with_columns(pl.lit(0.0).alias("lemp"))
    data = pl.concat([mpdta_data, zeros, zeros.with_columns(pl.lit(-0.0).alias("lemp"))])
    spec = {"yname": "lemp", "tname": "year", "gname": "first.treat", "xformla": "~lpop"}

    table, *_ = row_table_of(data, draws=True, **spec)
    shard = row_shard_from_arrow(to_arrow(table), 1)
    rows = preprocess_did(data, panel=False, **spec).data
    values = rows.select("year", "first.treat", pl.col("lemp") + 0.0, "lpop")
    copy = values.select(pl.int_range(pl.len()).over(values.columns)).to_series().to_numpy()
    expected = row_keys(
        np.unique(rows["year"].to_numpy(), return_inverse=True)[1],
        rows["first.treat"].to_numpy().astype(float),
        rows["lemp"].to_numpy(),
        None,
        rows.select("lpop").to_numpy(),
        copy,
    )

    assert table.columns == ["period", "cohort", "y", "x", "copy"]
    assert copy.max() == 1
    assert np.signbit(rows["lemp"].to_numpy()).sum() == len(zeros)
    assert len(np.unique(shard.unit_keys)) == len(data)
    np.testing.assert_array_equal(np.sort(shard.unit_keys), np.sort(expected))


def test_long_facts_counts_each_cluster_of_the_kept_rows_once_across_periods_and_cohorts(
    large_clustered_panel, to_spark
):
    config = DIDConfig(yname="y", tname="time", idname="id", gname="group", clustervars=["cluster"])
    frame = to_spark(large_clustered_panel)
    kept = (F.col("cluster") > 10 * (F.col("time") - 1)) & (F.col("cluster") <= 10 * (F.col("time") + 1))
    window = (pl.col("cluster") > 10 * (pl.col("time") - 1)) & (pl.col("cluster") <= 10 * (pl.col("time") + 1))
    n_kept_clusters = large_clustered_panel.filter(window)["cluster"].n_unique()

    facts = _long_facts(frame, config, [~kept], kept, "cluster")
    unclustered = _long_facts(frame, config, [~kept], kept, None)

    np.testing.assert_allclose(facts["clusters"], n_kept_clusters, rtol=0.1)
    assert unclustered["clusters"] is None
    assert {**facts, "clusters": None} == unclustered


def test_spread_clusters_bounds_the_sums_of_every_cluster_in_a_task():
    assert _spread_clusters(2**17 // 9, 9)
    assert not _spread_clusters(2**17 // 9 + 1, 9)
