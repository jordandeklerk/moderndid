"""Distributed 2-period panel DDD estimator for Spark."""

from __future__ import annotations

from functools import partial

from moderndid.distributed._ddd_panel import ddd_panel_core
from moderndid.distributed._validate import _validate_inputs  # noqa: F401

from ._bootstrap import distributed_mboot_ddd
from ._inf_func import compute_variance_distributed
from ._nuisance import compute_all_nuisances_distributed
from ._utils import get_default_partitions


def spark_ddd_panel(
    spark,
    y1,
    y0,
    subgroup,
    covariates,
    i_weights=None,
    est_method="dr",
    boot=False,
    biters=1000,
    influence_func=False,
    alpha=0.05,
    random_state=None,
    n_partitions=None,
):
    r"""Distributed 2-period doubly robust DDD estimator for panel data.

    Computes the triple-difference ATT for a two-period panel using
    distributed nuisance estimation and influence-function-based
    inference. The DDD estimand is:

    .. math::

        \text{ATT}^{DDD} = \text{DiD}(4, 3) + \text{DiD}(4, 2) - \text{DiD}(4, 1)

    where subgroup 4 is treated-eligible, 3 is treated-ineligible,
    2 is control-eligible, and 1 is control-ineligible.

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
    i_weights : ndarray or None
        Observation weights.
    est_method : {"dr", "reg", "ipw"}, default "dr"
        Estimation method.
    boot : bool, default False
        Whether to use multiplier bootstrap for inference.
    biters : int, default 1000
        Number of bootstrap iterations.
    influence_func : bool, default False
        Whether to return the influence function.
    alpha : float, default 0.05
        Significance level.
    random_state : int or None
        Random seed for bootstrap.
    n_partitions : int or None
        Number of partitions. Defaults to Spark default parallelism.

    Returns
    -------
    DDDPanelResult
        Result containing ATT, standard error, confidence intervals,
        bootstrap draws (if requested), and influence function.
    """
    if n_partitions is None:
        n_partitions = get_default_partitions(spark)

    return ddd_panel_core(
        nuisance_fn=partial(compute_all_nuisances_distributed, spark),
        variance_fn=partial(compute_variance_distributed, spark),
        bootstrap_fn=partial(distributed_mboot_ddd, spark),
        y1=y1,
        y0=y0,
        subgroup=subgroup,
        covariates=covariates,
        i_weights=i_weights,
        est_method=est_method,
        boot=boot,
        biters=biters,
        influence_func=influence_func,
        alpha=alpha,
        random_state=random_state,
        n_partitions=n_partitions,
    )
