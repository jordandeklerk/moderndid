"""Distributed nuisance estimation for Spark estimators."""

from __future__ import annotations

from functools import partial

from moderndid.distributed._nuisance import (
    DistOutcomeRegResult,  # noqa: F401
    DistPScoreResult,  # noqa: F401
    _build_partitions_for_subset,  # noqa: F401
    _compute_outcome_regression_null,  # noqa: F401
    _compute_pscore_null,  # noqa: F401
)
from moderndid.distributed._nuisance import (
    compute_all_nuisances_distributed as _compute_all_nuisances,
)

from ._regression import distributed_logistic_irls, distributed_wls
from ._utils import get_default_partitions


def compute_all_nuisances_distributed(
    spark,
    y1,
    y0,
    subgroup,
    covariates,
    weights,
    est_method="dr",
    trim_level=0.995,
    n_partitions=None,
):
    r"""Compute all nuisance parameters using distributed regression.

    Estimates propensity scores and outcome regressions for the three
    subgroup comparisons :math:`(4, 3)`, :math:`(4, 2)`, and
    :math:`(4, 1)` required by the DDD estimator.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Spark session.
    y1 : ndarray
        Post-treatment outcomes.
    y0 : ndarray
        Pre-treatment outcomes.
    subgroup : ndarray
        Subgroup indicators (1, 2, 3, or 4).
    covariates : ndarray
        Covariates including intercept, shape :math:`(n, k)`.
    weights : ndarray
        Observation weights.
    est_method : {"dr", "reg", "ipw"}, default "dr"
        Estimation method.
    trim_level : float, default 0.995
        Trimming level for propensity scores.
    n_partitions : int or None
        Number of partitions for distributed computation.

    Returns
    -------
    pscores : list of DistPScoreResult
        Propensity score results for comparisons [3, 2, 1].
    or_results : list of DistOutcomeRegResult
        Outcome regression results for comparisons [3, 2, 1].
    """
    if n_partitions is None:
        n_partitions = get_default_partitions(spark)

    return _compute_all_nuisances(
        logistic_irls_fn=partial(distributed_logistic_irls, spark),
        wls_fn=partial(distributed_wls, spark),
        y1=y1,
        y0=y0,
        subgroup=subgroup,
        covariates=covariates,
        weights=weights,
        est_method=est_method,
        trim_level=trim_level,
        n_partitions=n_partitions,
    )
