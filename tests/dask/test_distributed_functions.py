"""Tests for distributed functions requiring a Dask client."""

import numpy as np
import pandas as pd
import pytest

distributed = pytest.importorskip("distributed")
dd = pytest.importorskip("dask.dataframe")

from moderndid.dask._bootstrap import distributed_mboot_ddd
from moderndid.dask._gram import _sum_gram_pair, distributed_gram, tree_reduce
from moderndid.dask._inf_func import compute_variance_distributed
from moderndid.dask._regression import (
    _irls_local_stats_with_y,
    distributed_logistic_irls,
    distributed_logistic_irls_from_futures,
    distributed_wls,
    distributed_wls_from_futures,
    partition_gram,
)
from moderndid.dask._utils import get_default_partitions, get_or_create_client, prepare_cohort_wide_pivot


@pytest.fixture
def rng():
    return np.random.default_rng(42)


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


def test_tree_reduce_single_future(dask_client):
    f = dask_client.submit(lambda: (np.eye(2), np.ones(2), 10))
    result = tree_reduce(dask_client, [f], _sum_gram_pair)
    np.testing.assert_array_equal(result[0], np.eye(2))
    assert result[2] == 10


def test_tree_reduce_multiple_futures(dask_client):
    futures = [dask_client.submit(lambda i=i: (np.eye(2) * i, np.ones(2) * i, i)) for i in range(1, 9)]
    result = tree_reduce(dask_client, futures, _sum_gram_pair)
    expected_sum = sum(range(1, 9))
    np.testing.assert_allclose(result[0], np.eye(2) * expected_sum)
    np.testing.assert_allclose(result[1], np.ones(2) * expected_sum)
    assert result[2] == expected_sum


@pytest.mark.parametrize("split_every", [2, 4, 8])
def test_tree_reduce_split_every(dask_client, split_every):
    futures = [dask_client.submit(lambda i=i: (np.array([[i]]), np.array([i]), i)) for i in range(1, 17)]
    result = tree_reduce(dask_client, futures, _sum_gram_pair, split_every=split_every)
    expected = sum(range(1, 17))
    assert result[2] == expected


def test_distributed_gram(dask_client, wls_partitions):
    partitions, _ = wls_partitions
    XtWX, XtWy, n_total = distributed_gram(dask_client, partitions)
    k = partitions[0][0].shape[1]
    assert XtWX.shape == (k, k)
    assert XtWy.shape == (k,)
    assert n_total == sum(p[0].shape[0] for p in partitions)


def test_distributed_gram_matches_local(dask_client, wls_partitions):
    partitions, _ = wls_partitions
    XtWX_dist, XtWy_dist, _ = distributed_gram(dask_client, partitions)

    XtWX_local = sum(partition_gram(X, W, y)[0] for X, W, y in partitions)
    XtWy_local = sum(partition_gram(X, W, y)[1] for X, W, y in partitions)

    np.testing.assert_allclose(XtWX_dist, XtWX_local, atol=1e-10)
    np.testing.assert_allclose(XtWy_dist, XtWy_local, atol=1e-10)


def test_distributed_wls_recovers_coefficients(dask_client, wls_partitions):
    partitions, beta_true = wls_partitions
    beta_hat = distributed_wls(dask_client, partitions)
    np.testing.assert_allclose(beta_hat, beta_true, atol=0.1)


def test_distributed_logistic_irls_converges(dask_client, logistic_partitions):
    partitions, beta_true = logistic_partitions
    beta_hat = distributed_logistic_irls(dask_client, partitions)
    np.testing.assert_allclose(beta_hat, beta_true, atol=0.5)
    assert np.all(np.isfinite(beta_hat))


def test_distributed_logistic_irls_max_iter(dask_client):
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
    beta = distributed_logistic_irls(dask_client, partitions, max_iter=2)
    assert np.all(np.isfinite(beta))
    assert beta.shape == (2,)


def test_distributed_wls_from_futures(dask_client, rng):
    beta_true = np.array([2.0, -1.0])
    parts = []
    for _ in range(3):
        n = 80
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        W = np.ones(n)
        y = X @ beta_true + rng.standard_normal(n) * 0.01
        parts.append({"X": X, "W": W, "y": y})

    scattered = dask_client.scatter(parts)

    def gram_fn(part_data):
        return partition_gram(part_data["X"], part_data["W"], part_data["y"])

    beta_hat = distributed_wls_from_futures(dask_client, scattered, gram_fn)
    np.testing.assert_allclose(beta_hat, beta_true, atol=0.1)


def test_distributed_wls_from_futures_all_none_raises(dask_client):
    scattered = dask_client.scatter([None, None])

    def gram_fn(part_data):
        return None

    with pytest.raises((ValueError, TypeError)):
        distributed_wls_from_futures(dask_client, scattered, gram_fn)


def test_distributed_logistic_irls_from_futures(dask_client, rng):
    beta_true = np.array([0.0, 0.5])
    parts = []
    for _ in range(3):
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        prob = 1.0 / (1.0 + np.exp(-X @ beta_true))
        y = rng.binomial(1, prob).astype(np.float64)
        parts.append({"X": X, "W": np.ones(n), "y": y})

    scattered = dask_client.scatter(parts)

    def gram_fn(part_data, beta):
        return _irls_local_stats_with_y(part_data["X"], part_data["W"], part_data["y"], beta)

    beta_hat = distributed_logistic_irls_from_futures(
        dask_client,
        scattered,
        gram_fn,
        k=2,
    )
    np.testing.assert_allclose(beta_hat, beta_true, atol=0.5)


def test_distributed_logistic_irls_from_futures_all_none(dask_client):
    scattered = dask_client.scatter([None, None])

    def gram_fn(part_data, beta):
        return None

    with pytest.raises(TypeError):
        distributed_logistic_irls_from_futures(dask_client, scattered, gram_fn, k=2)


def test_get_default_partitions(dask_client):
    n = get_default_partitions(dask_client)
    assert isinstance(n, int)
    assert n >= 1
    assert n == 2


def test_get_or_create_client_with_existing(dask_client):
    result = get_or_create_client(dask_client)
    assert result is dask_client


def test_get_or_create_client_none_uses_current(dask_client):
    result = get_or_create_client(None)
    assert result is not None


def test_distributed_mboot_ddd_shapes(dask_client, rng):
    n, k, biters = 100, 4, 50
    parts = [rng.standard_normal((25, k)) for _ in range(4)]
    bres, se, crit_val = distributed_mboot_ddd(
        client=dask_client,
        inf_func_partitions=parts,
        n_total=n,
        biters=biters,
        alpha=0.05,
        random_state=42,
    )
    assert bres.shape == (biters, k)
    assert se.shape == (k,)
    assert np.isfinite(crit_val) or np.isnan(crit_val)


def test_distributed_mboot_ddd_different_seeds(dask_client):
    n, k, biters = 80, 3, 30
    rng_a = np.random.default_rng(200)
    parts_a = [rng_a.standard_normal((20, k)) for _ in range(4)]
    a = distributed_mboot_ddd(dask_client, parts_a, n, biters, random_state=1)

    rng_b = np.random.default_rng(300)
    parts_b = [rng_b.standard_normal((20, k)) for _ in range(4)]
    b = distributed_mboot_ddd(dask_client, parts_b, n, biters, random_state=2)

    assert not np.array_equal(a[0], b[0])


def test_distributed_mboot_ddd_se_positive(dask_client, rng):
    n, k, biters = 200, 3, 100
    parts = [rng.standard_normal((50, k)) * 5 for _ in range(4)]
    _, se, _ = distributed_mboot_ddd(dask_client, parts, n, biters, random_state=7)
    finite_se = se[np.isfinite(se)]
    assert len(finite_se) > 0
    assert np.all(finite_se > 0)


def test_compute_variance_distributed_shapes(dask_client):
    local_rng = np.random.default_rng(500)
    n, k = 100, 4
    parts = [local_rng.standard_normal((25, k)) for _ in range(4)]
    se = compute_variance_distributed(dask_client, parts, n, k)
    assert se.shape == (k,)
    assert np.all(np.isfinite(se))
    assert np.all(se > 0)


def test_compute_variance_distributed_matches_local(dask_client):
    local_rng = np.random.default_rng(501)
    n, k = 80, 3
    parts = [local_rng.standard_normal((20, k)) for _ in range(4)]
    se_dist = compute_variance_distributed(dask_client, parts, n, k)

    full = np.vstack(parts)
    V = full.T @ full / n
    se_local = np.sqrt(np.diag(V) / n)
    np.testing.assert_allclose(se_dist, se_local, atol=1e-10)


def test_prepare_cohort_wide_pivot_basic(dask_client):
    pdf = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "time": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "group": [5, 5, 5, 0, 0, 0, 5, 5, 5],
            "y": [1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 0.8, 1.8, 2.8],
            "x1": [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3],
        }
    )
    ddf = dd.from_pandas(pdf, npartitions=2).persist()

    cells = [
        (0, 5, 2, 1, True, "compute"),
        (1, 5, 3, 2, True, "compute"),
    ]
    wide, n_wide = prepare_cohort_wide_pivot(
        client=dask_client,
        dask_data=ddf,
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
    result = wide.compute()
    assert "_y_1" in result.columns
    assert "_y_2" in result.columns
    assert "_y_3" in result.columns
    assert "x1" in result.columns
    assert len(result) == 3


def test_prepare_cohort_wide_pivot_no_compute_cells(dask_client):
    pdf = pd.DataFrame(
        {
            "id": [1, 1],
            "time": [1, 2],
            "group": [5, 5],
            "y": [1.0, 2.0],
        }
    )
    ddf = dd.from_pandas(pdf, npartitions=1).persist()

    cells = [(0, 5, 2, 1, True, "skip")]
    wide, n_wide = prepare_cohort_wide_pivot(
        client=dask_client,
        dask_data=ddf,
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


def test_prepare_cohort_wide_pivot_extra_cols(dask_client):
    pdf = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "time": [1, 2, 1, 2],
            "group": [5, 5, 0, 0],
            "partition": [1, 1, 1, 1],
            "y": [1.0, 2.0, 0.5, 1.5],
        }
    )
    ddf = dd.from_pandas(pdf, npartitions=1).persist()

    cells = [(0, 5, 2, 1, True, "compute")]
    wide, n_wide = prepare_cohort_wide_pivot(
        client=dask_client,
        dask_data=ddf,
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
    result = wide.compute()
    assert "partition" in result.columns
