"""Distributed influence function computation and variance estimation for Spark."""

from __future__ import annotations

import numpy as np

from moderndid.distributed._inf_func import _compute_did

from ._gram import _sum_gram_pair


def compute_did_distributed(
    spark,
    subgroup,
    covariates,
    weights,
    pscores,
    or_results,
    est_method,
    n_total,
    n_partitions,
):
    r"""Compute all DiD components and combine into DDD estimate.

    Computes the three pairwise DiD comparisons and combines them into
    the triple-difference estimand:

    .. math::

        \text{ATT}^{DDD} = \text{DiD}(4, 3) + \text{DiD}(4, 2) - \text{DiD}(4, 1)

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Spark session.
    subgroup : ndarray
        Subgroup indicators (1, 2, 3, or 4).
    covariates : ndarray
        Covariates including intercept, shape :math:`(n, k)`.
    weights : ndarray
        Observation weights.
    pscores : list of DistPScoreResult
        Propensity score results for comparisons [3, 2, 1].
    or_results : list of DistOutcomeRegResult
        Outcome regression results for comparisons [3, 2, 1].
    est_method : {"dr", "reg", "ipw"}
        Estimation method.
    n_total : int
        Total number of units.
    n_partitions : int
        Number of partitions.

    Returns
    -------
    did_results : list of tuple[float, ndarray]
        Per-comparison DiD ATT estimates and influence functions.
    ddd_att : float
        The DDD ATT point estimate.
    inf_func : ndarray of shape :math:`(n,)`
        Combined influence function.
    """
    did_results = []
    for i, comp_subgroup in enumerate([3, 2, 1]):
        did_result = _compute_did(
            subgroup=subgroup,
            covariates=covariates,
            weights=weights,
            comparison_subgroup=comp_subgroup,
            pscore_result=pscores[i],
            or_result=or_results[i],
            est_method=est_method,
            n_total=n_total,
        )
        did_results.append(did_result)

    dr_att_3, inf_func_3 = did_results[0]
    dr_att_2, inf_func_2 = did_results[1]
    dr_att_1, inf_func_1 = did_results[2]

    ddd_att = dr_att_3 + dr_att_2 - dr_att_1

    n = n_total
    n3 = int(np.sum((subgroup == 4) | (subgroup == 3)))
    n2 = int(np.sum((subgroup == 4) | (subgroup == 2)))
    n1 = int(np.sum((subgroup == 4) | (subgroup == 1)))

    w3 = n / n3 if n3 > 0 else 0
    w2 = n / n2 if n2 > 0 else 0
    w1 = n / n1 if n1 > 0 else 0

    inf_func = w3 * inf_func_3 + w2 * inf_func_2 - w1 * inf_func_1

    return did_results, float(ddd_att), inf_func


def compute_variance_distributed(spark, inf_func_partitions, n_total, k):
    r"""Compute variance from distributed influence function partitions.

    Uses Spark RDD ``treeReduce`` to compute
    :math:`V = (1/n) \, \Psi^\top \Psi` without materializing the full
    matrix on one node.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Spark session.
    inf_func_partitions : list of ndarray
        Per-partition influence function arrays.
    n_total : int
        Total number of observations.
    k : int
        Number of influence function columns.

    Returns
    -------
    se : ndarray of shape :math:`(k,)`
        Standard errors.
    """
    sc = spark.sparkContext
    rdd = sc.parallelize(inf_func_partitions, numSlices=len(inf_func_partitions))
    rdd = rdd.map(lambda chunk: (chunk.T @ chunk, np.zeros(chunk.shape[1]), chunk.shape[0]))

    PtP, _, _ = rdd.treeReduce(_sum_gram_pair, depth=3)
    V = PtP / n_total
    return np.sqrt(np.diag(V) / n_total)
