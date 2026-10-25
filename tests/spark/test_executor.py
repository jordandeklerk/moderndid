"""Tests for the Spark executor that runs the passes of a sharded att_gt fit."""

import re
import sys

import numpy as np
import pytest

from moderndid import _version, att_gt, preprocess_did
from moderndid.did.shards.container import SummaryState
from moderndid.did.shards.driver import fit_att_gt_on_shards
from moderndid.did.shards.kernels import summary_kernel
from moderndid.did.shards.local import LocalExecutor, shards_from_did_data
from moderndid.did.shards.plan import plan_cells
from tests.helpers import importorskip

importorskip("pyspark")

from pyspark.sql import functions as F

from moderndid.spark.executor import SparkExecutor

pytestmark = pytest.mark.spark


def _by_name(statistics):
    return [statistics[name] for name in sorted(statistics)]


def _stacked(statistics):
    return np.concatenate([np.ravel(value) for value in _by_name(statistics)])


def _factor_grams(factors):
    return np.concatenate([np.ravel(np.hstack([r.T @ r, r.T @ rotated])) for r, rotated in _by_name(factors)])


def test_spark_executor_combines_a_pass_like_the_local_executor(mpdta_pop_weighted, unit_table_of):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "weightsname": "pop",
    }

    did_data = preprocess_did(mpdta_pop_weighted, panel=True, allow_unbalanced_panel=False, **spec)
    plan = plan_cells(did_data.config, did_data.cohort_counts)
    state = SummaryState(fits=plan.fits, cells=plan.cells, logit=True, regression=True)
    expected = LocalExecutor(shards_from_did_data(did_data, 3, random_state=0)).run(summary_kernel, state)
    unit_table, config, *_ = unit_table_of(mpdta_pop_weighted, **spec)
    executor = SparkExecutor(unit_table.repartition(5), config.time_periods_count, 1)
    result = executor.run(summary_kernel, state)

    assert executor.n_passes == 1
    assert result.sums["units"] == 500
    assert (sorted(result.sums), sorted(result.maxima), sorted(result.factors)) == (
        sorted(expected.sums),
        sorted(expected.maxima),
        sorted(expected.factors),
    )
    assert result.keyed == {}
    np.testing.assert_allclose(_stacked(result.sums), _stacked(expected.sums), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(_stacked(result.maxima), _stacked(expected.maxima), rtol=1e-12)
    np.testing.assert_allclose(_factor_grams(result.factors), _factor_grams(expected.factors), rtol=1e-12, atol=1e-12)


def test_spark_executor_fits_a_unit_table_with_empty_and_single_unit_partitions(
    small_scalable_panel, unit_table_of, assert_same_fit
):
    spec = {"yname": "y", "tname": "time", "idname": "id", "gname": "group", "xformla": "~cov1"}

    memory = att_gt(small_scalable_panel, cband=False, **spec)
    unit_table, config, cohort_counts, _, cluster_counts, _ = unit_table_of(small_scalable_panel, **spec)
    table = unit_table.repartition(32, "unit")
    sizes = [row["count"] for row in table.groupBy(F.spark_partition_id()).count().collect()]
    result = fit_att_gt_on_shards(
        SparkExecutor(table, config.time_periods_count, 1),
        config,
        cohort_counts,
        0,
        {"bootstrap": False, "uniform_bands": False},
        cluster_counts=cluster_counts,
    )

    assert sum(sizes) == 50
    assert 16 < len(sizes) < 32
    assert 1 in sizes
    assert_same_fit(result, memory, 1e-12)


def test_spark_executor_combines_the_rows_of_many_partitions_in_a_second_stage(
    scalable_panel, unit_table_of, att_gt_on_shards, assert_same_fit
):
    spec = {
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "group",
        "xformla": "~cov1 + cov2",
        "clustervars": ["cluster"],
        "est_method": "reg",
        "biters": 199,
    }

    local = att_gt_on_shards(scalable_panel, 3, boot=True, random_state=11, **spec)
    unit_table, config, cohort_counts, _, cluster_counts, _ = unit_table_of(scalable_panel, **spec)
    table = unit_table.repartition(24, "cluster")
    sizes = [row["count"] for row in table.groupBy(F.spark_partition_id()).count().collect()]
    result = fit_att_gt_on_shards(
        SparkExecutor(table, config.time_periods_count, 2),
        config,
        cohort_counts,
        11,
        {"bootstrap": True, "uniform_bands": True},
        cluster_counts=cluster_counts,
    )
    moments, expected = result.influence_moments, local.influence_moments

    assert len(sizes) > 16
    assert_same_fit(result, local, 1e-12)
    np.testing.assert_allclose(moments.gram, expected.gram, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(moments.cluster_gram, expected.cluster_gram, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(moments.bootstrap_sums, expected.bootstrap_sums, rtol=1e-12, atol=1e-10)
    assert moments.n_clusters == 50


def test_spark_executor_finalizes_keyed_sums_in_tasks_and_on_the_driver_alike(
    scalable_panel, unit_table_of, att_gt_on_shards, assert_same_fit
):
    spec = {
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "group",
        "xformla": "~cov1 + cov2",
        "clustervars": ["cluster"],
        "est_method": "reg",
        "biters": 199,
    }
    settings = {"bootstrap": True, "uniform_bands": True}

    local = att_gt_on_shards(scalable_panel, 3, boot=True, random_state=11, **spec)
    unit_table, config, cohort_counts, _, cluster_counts, _ = unit_table_of(scalable_panel, **spec)
    by_cluster = unit_table.repartition(24, "cluster")
    by_row = unit_table.repartition(24)
    partitions = by_row.select(F.spark_partition_id().alias("part"), "cluster")
    n_tasks = partitions.select("part").distinct().count()
    parts_per_cluster = partitions.groupBy("cluster").agg(F.count_distinct("part").alias("parts"))
    split_clusters = parts_per_cluster.filter(F.col("parts") > 1).count()
    in_tasks = fit_att_gt_on_shards(
        SparkExecutor(by_cluster, config.time_periods_count, 2, whole_clusters=True),
        config,
        cohort_counts,
        11,
        settings,
        cluster_counts=cluster_counts,
    )
    on_driver = fit_att_gt_on_shards(
        SparkExecutor(by_row, config.time_periods_count, 2, whole_clusters=False),
        config,
        cohort_counts,
        11,
        settings,
        cluster_counts=cluster_counts,
    )
    expected = local.influence_moments

    assert n_tasks > 16
    assert split_clusters > 0
    assert in_tasks.influence_moments.n_clusters == on_driver.influence_moments.n_clusters == 50
    assert_same_fit(in_tasks, local, 1e-12)
    assert_same_fit(on_driver, local, 1e-12)
    np.testing.assert_allclose(in_tasks.influence_moments.gram, expected.gram, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(on_driver.influence_moments.gram, expected.gram, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(in_tasks.influence_moments.cluster_gram, expected.cluster_gram, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(on_driver.influence_moments.cluster_gram, expected.cluster_gram, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(
        in_tasks.influence_moments.bootstrap_sums, expected.bootstrap_sums, rtol=1e-12, atol=1e-10
    )
    np.testing.assert_allclose(
        on_driver.influence_moments.bootstrap_sums, expected.bootstrap_sums, rtol=1e-12, atol=1e-10
    )


def test_spark_executor_finalizing_in_tasks_counts_every_piece_of_a_split_cluster(scalable_panel, unit_table_of):
    spec = {
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "group",
        "clustervars": ["cluster"],
        "est_method": "reg",
    }
    settings = {"bootstrap": False, "uniform_bands": False}
    n_clusters = scalable_panel["cluster"].n_unique()

    unit_table, config, cohort_counts, _, cluster_counts, _ = unit_table_of(scalable_panel, **spec)
    by_row = unit_table.repartition(24)
    in_tasks = fit_att_gt_on_shards(
        SparkExecutor(by_row, config.time_periods_count, 0, whole_clusters=True),
        config,
        cohort_counts,
        0,
        settings,
        cluster_counts=cluster_counts,
    )
    on_driver = fit_att_gt_on_shards(
        SparkExecutor(by_row, config.time_periods_count, 0, whole_clusters=False),
        config,
        cohort_counts,
        0,
        settings,
        cluster_counts=cluster_counts,
    )

    assert on_driver.influence_moments.n_clusters == n_clusters
    assert in_tasks.influence_moments.n_clusters > n_clusters


def test_spark_executor_reports_executors_whose_moderndid_differs_from_the_driver(
    mpdta_data, unit_table_of, monkeypatch
):
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat"}
    version = _version.__version__
    python = "{}.{}".format(*sys.version_info[:2])
    unit_table, config, cohort_counts, _, _, _ = unit_table_of(mpdta_data, **spec)
    monkeypatch.setattr(_version, "__version__", "0.0.0-test")

    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"The Spark executors run moderndid {version} on Python {python}. The driver runs moderndid 0.0.0-test "
            f"on Python {python}. Install the same moderndid version and Python minor version on the driver and on "
            "every executor."
        ),
    ):
        fit_att_gt_on_shards(
            SparkExecutor(unit_table, config.time_periods_count, 0),
            config,
            cohort_counts,
            0,
            {"bootstrap": False, "uniform_bands": False},
        )
