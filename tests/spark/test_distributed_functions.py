"""Tests for distributed functions requiring a Spark session."""

import numpy as np
import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")

from moderndid.spark._bootstrap import distributed_mboot_ddd
from moderndid.spark._gram import partition_gram
from moderndid.spark._inf_func import compute_variance_distributed
from moderndid.spark._regression import (
    _irls_local_stats_with_y,
    distributed_logistic_irls,
    distributed_logistic_irls_from_partitions,
    distributed_wls,
    distributed_wls_from_partitions,
)
from moderndid.spark._utils import get_default_partitions, get_or_create_spark, prepare_cohort_wide_pivot


@pytest.fixture
def wls_partitions(rng):
    partitions = []
    beta_true = np.array([1.0, -0.5, 0.3])
    for _ in range(4):
        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
        W = np.ones(n)
        y = X @ beta_true + rng.standard_normal(n) * 0.01
        partitions.append((X, W, y))
    return partitions, beta_true


@pytest.fixture
def logistic_partitions(rng):
    partitions = []
    beta_true = np.array([0.0, 1.0])
    for _ in range(4):
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        prob = 1.0 / (1.0 + np.exp(-X @ beta_true))
        y = rng.binomial(1, prob).astype(np.float64)
        W = np.ones(n)
        partitions.append((X, W, y))
    return partitions, beta_true


def test_distributed_wls_recovers_coefficients(spark_session, wls_partitions):
    partitions, beta_true = wls_partitions
    beta_hat = distributed_wls(spark_session, partitions)
    np.testing.assert_allclose(beta_hat, beta_true, atol=0.1)


def test_distributed_logistic_irls_converges(spark_session, logistic_partitions):
    partitions, beta_true = logistic_partitions
    beta_hat = distributed_logistic_irls(spark_session, partitions)
    np.testing.assert_allclose(beta_hat, beta_true, atol=0.5)
    assert np.all(np.isfinite(beta_hat))


def test_distributed_logistic_irls_max_iter(spark_session):
    local_rng = np.random.default_rng(777)
    beta_true = np.array([0.0, 1.0])
    partitions = []
    for _i in range(4):
        n = 200
        X = np.column_stack([np.ones(n), local_rng.standard_normal(n)])
        prob = 1.0 / (1.0 + np.exp(-X @ beta_true))
        y = local_rng.binomial(1, prob).astype(np.float64)
        W = local_rng.uniform(0.9, 1.1, size=n)
        partitions.append((X, W, y))
    beta = distributed_logistic_irls(spark_session, partitions, max_iter=2)
    assert np.all(np.isfinite(beta))
    assert beta.shape == (2,)


def test_distributed_wls_from_partitions(spark_session, rng):
    beta_true = np.array([2.0, -1.0])
    parts = []
    for _ in range(3):
        n = 80
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        W = np.ones(n)
        y = X @ beta_true + rng.standard_normal(n) * 0.01
        parts.append({"X": X, "W": W, "y": y})

    def gram_fn(part_data):
        return partition_gram(part_data["X"], part_data["W"], part_data["y"])

    beta_hat = distributed_wls_from_partitions(spark_session, parts, gram_fn)
    np.testing.assert_allclose(beta_hat, beta_true, atol=0.1)


def test_distributed_logistic_irls_from_partitions(spark_session, rng):
    beta_true = np.array([0.0, 0.5])
    parts = []
    for _ in range(3):
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        prob = 1.0 / (1.0 + np.exp(-X @ beta_true))
        y = rng.binomial(1, prob).astype(np.float64)
        parts.append({"X": X, "W": np.ones(n), "y": y})

    def gram_fn(part_data, beta):
        return _irls_local_stats_with_y(part_data["X"], part_data["W"], part_data["y"], beta)

    beta_hat = distributed_logistic_irls_from_partitions(
        spark_session,
        parts,
        gram_fn,
        k=2,
    )
    np.testing.assert_allclose(beta_hat, beta_true, atol=0.5)


def test_get_default_partitions(spark_session):
    n = get_default_partitions(spark_session)
    assert isinstance(n, int)
    assert n >= 1
    assert n == 2


def test_get_or_create_spark_with_existing(spark_session):
    result = get_or_create_spark(spark_session)
    assert result is spark_session


def test_get_or_create_spark_none_uses_active(spark_session):
    result = get_or_create_spark(None)
    assert result is not None


def test_distributed_mboot_ddd_shapes(spark_session, rng):
    n, k, biters = 100, 4, 50
    parts = [rng.standard_normal((25, k)) for _ in range(4)]
    bres, se, crit_val = distributed_mboot_ddd(
        spark=spark_session,
        inf_func_partitions=parts,
        n_total=n,
        biters=biters,
        alpha=0.05,
        random_state=42,
    )
    assert bres.shape == (biters, k)
    assert se.shape == (k,)
    assert np.isfinite(crit_val) or np.isnan(crit_val)


def test_distributed_mboot_ddd_different_seeds(spark_session):
    n, k, biters = 80, 3, 30
    rng_a = np.random.default_rng(200)
    parts_a = [rng_a.standard_normal((20, k)) for _ in range(4)]
    a = distributed_mboot_ddd(spark_session, parts_a, n, biters, random_state=1)

    rng_b = np.random.default_rng(300)
    parts_b = [rng_b.standard_normal((20, k)) for _ in range(4)]
    b = distributed_mboot_ddd(spark_session, parts_b, n, biters, random_state=2)

    assert not np.array_equal(a[0], b[0])


def test_distributed_mboot_ddd_se_positive(spark_session, rng):
    n, k, biters = 200, 3, 100
    parts = [rng.standard_normal((50, k)) * 5 for _ in range(4)]
    _, se, _ = distributed_mboot_ddd(spark_session, parts, n, biters, random_state=7)
    finite_se = se[np.isfinite(se)]
    assert len(finite_se) > 0
    assert np.all(finite_se > 0)


def test_compute_variance_distributed_shapes(spark_session):
    local_rng = np.random.default_rng(500)
    n, k = 100, 4
    parts = [local_rng.standard_normal((25, k)) for _ in range(4)]
    se = compute_variance_distributed(spark_session, parts, n, k)
    assert se.shape == (k,)
    assert np.all(np.isfinite(se))
    assert np.all(se > 0)


def test_compute_variance_distributed_matches_local(spark_session):
    local_rng = np.random.default_rng(501)
    n, k = 80, 3
    parts = [local_rng.standard_normal((20, k)) for _ in range(4)]
    se_dist = compute_variance_distributed(spark_session, parts, n, k)

    full = np.vstack(parts)
    V = full.T @ full / n
    se_local = np.sqrt(np.diag(V) / n)
    np.testing.assert_allclose(se_dist, se_local, atol=1e-10)


def test_prepare_cohort_wide_pivot_basic(spark_session):
    pdf = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "time": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "group": [5, 5, 5, 0, 0, 0, 5, 5, 5],
            "y": [1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 0.8, 1.8, 2.8],
            "x1": [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3],
        }
    )
    sdf = spark_session.createDataFrame(pdf).cache()
    sdf.count()

    cells = [
        (0, 5, 2, 1, True, "compute"),
        (1, 5, 3, 2, True, "compute"),
    ]
    wide, n_wide = prepare_cohort_wide_pivot(
        spark=spark_session,
        sdf=sdf,
        g=5,
        cells=cells,
        time_col="time",
        group_col="group",
        id_col="id",
        y_col="y",
        covariate_cols=["x1"],
        n_partitions=2,
    )
    assert n_wide == 3
    result = wide.toPandas()
    assert "_y_1" in result.columns
    assert "_y_2" in result.columns
    assert "_y_3" in result.columns
    assert "x1" in result.columns
    assert len(result) == 3
    sdf.unpersist()


def test_prepare_cohort_wide_pivot_no_compute_cells(spark_session):
    pdf = pd.DataFrame(
        {
            "id": [1, 1],
            "time": [1, 2],
            "group": [5, 5],
            "y": [1.0, 2.0],
        }
    )
    sdf = spark_session.createDataFrame(pdf).cache()
    sdf.count()

    cells = [(0, 5, 2, 1, True, "skip")]
    wide, n_wide = prepare_cohort_wide_pivot(
        spark=spark_session,
        sdf=sdf,
        g=5,
        cells=cells,
        time_col="time",
        group_col="group",
        id_col="id",
        y_col="y",
        covariate_cols=None,
        n_partitions=1,
    )
    assert wide is None
    assert n_wide == 0
    sdf.unpersist()


def test_prepare_cohort_wide_pivot_extra_cols(spark_session):
    pdf = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "time": [1, 2, 1, 2],
            "group": [5, 5, 0, 0],
            "partition": [1, 1, 1, 1],
            "y": [1.0, 2.0, 0.5, 1.5],
        }
    )
    sdf = spark_session.createDataFrame(pdf).cache()
    sdf.count()

    cells = [(0, 5, 2, 1, True, "compute")]
    wide, n_wide = prepare_cohort_wide_pivot(
        spark=spark_session,
        sdf=sdf,
        g=5,
        cells=cells,
        time_col="time",
        group_col="group",
        id_col="id",
        y_col="y",
        covariate_cols=None,
        n_partitions=1,
        extra_cols=["partition"],
    )
    assert n_wide == 2
    result = wide.toPandas()
    assert "partition" in result.columns
    sdf.unpersist()
