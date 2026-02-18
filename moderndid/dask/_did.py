"""Entry point for distributed DiD estimation."""

from __future__ import annotations

import logging

from ._utils import get_or_create_client, validate_dask_input


def dask_att_gt(
    data,
    yname,
    tname,
    idname,
    gname,
    xformla=None,
    client=None,
    control_group="nevertreated",
    base_period="varying",
    anticipation=0,
    est_method="dr",
    panel=True,
    weightsname=None,
    boot=False,
    biters=1000,
    cband=False,
    alp=0.05,
    clustervars=None,
    allow_unbalanced_panel=False,
    trim_level=0.995,
    random_state=None,
    n_partitions=None,
    max_cohorts=None,
    progress_bar=False,
):
    r"""Compute the distributed group-time average treatment effects.

    Distributed implementation of the DiD estimator from [1]_ for datasets
    that exceed single-machine memory. This function accepts a Dask DataFrame
    and dispatches to the distributed multi-period estimator.

    The underlying methodology is identical to :func:`~moderndid.att_gt`: for
    each group-time cell :math:`(g,t)`, the estimator computes

    .. math::

        ATT(g,t) = \mathbb{E}[Y_t(g) - Y_t(0) \mid G = g].

    The distributed backend computes sufficient statistics (Gram matrices,
    cross-products) partition-by-partition and aggregates via tree-reduce,
    avoiding full materialization of the dataset on any single worker.

    Users do not need to call this function directly. Passing a Dask DataFrame
    to :func:`~moderndid.att_gt` will automatically dispatch here.

    Parameters
    ----------
    data : dask.dataframe.DataFrame
        Data in long format as a Dask DataFrame.
    yname : str
        Name of the outcome variable column.
    tname : str
        Name of the column containing time periods.
    idname : str
        Name of the unit identifier column.
    gname : str
        Name of the treatment group column. Should be 0 for never-treated
        units and a positive value indicating the first treatment period.
    xformla : str, optional
        Formula for covariates in the form ``"~ x1 + x2"``.
    client : distributed.Client, optional
        Dask distributed client. If None, a local client is created.
    control_group : {"nevertreated", "notyettreated"}, default="nevertreated"
        Which units to use as controls.
    base_period : {"varying", "universal"}, default="varying"
        Base period selection.
    anticipation : int, default=0
        Number of anticipation periods.
    est_method : {"dr", "reg", "ipw"}, default="dr"
        Estimation method.
    boot : bool, default=False
        Whether to use the multiplier bootstrap.
    biters : int, default=1000
        Number of bootstrap iterations.
    cband : bool, default=False
        Whether to compute uniform confidence bands.
    alp : float, default=0.05
        Significance level.
    random_state : int, optional
        Random seed for bootstrap.
    n_partitions : int, optional
        Number of Dask partitions per cell.
    max_cohorts : int or None, default None
        Maximum number of treatment cohorts to process in parallel.
    progress_bar : bool, default False
        Whether to display a tqdm progress bar.

    Returns
    -------
    MPResult
        Group-time average treatment effect results.

    See Also
    --------
    att_gt : Local (non-distributed) DiD estimator. Automatically dispatches
        to ``dask_att_gt`` when passed a Dask DataFrame.

    References
    ----------

    .. [1] Callaway, B., & Sant'Anna, P. H. (2021). "Difference-in-differences
           with multiple time periods." Journal of Econometrics, 225(2), 200-230.
    """
    if callable(est_method):
        raise ValueError("Callable est_method is not supported for Dask inputs. Use 'dr', 'reg', or 'ipw'.")

    client = get_or_create_client(client)
    logging.getLogger("distributed.shuffle").setLevel(logging.ERROR)

    if clustervars is not None and isinstance(clustervars, str):
        raise TypeError(f"clustervars must be a list of strings, not a string. Use clustervars=['{clustervars}'].")

    required_cols = [yname, tname, gname]
    if idname is not None:
        required_cols.append(idname)
    if weightsname is not None:
        required_cols.append(weightsname)
    if clustervars is not None:
        for cv in clustervars:
            if cv not in required_cols:
                required_cols.append(cv)
    validate_dask_input(data, required_cols)

    from moderndid.core.preprocess.utils import get_covariate_names_from_formula

    from ._did_mp import dask_att_gt_mp

    covariate_cols = get_covariate_names_from_formula(xformla)

    return dask_att_gt_mp(
        client=client,
        data=data,
        y_col=yname,
        time_col=tname,
        id_col=idname,
        group_col=gname,
        covariate_cols=covariate_cols,
        control_group=control_group,
        base_period=base_period,
        anticipation=anticipation,
        est_method=est_method,
        weightsname=weightsname,
        boot=boot,
        biters=biters,
        cband=cband,
        alp=alp,
        clustervars=clustervars,
        allow_unbalanced_panel=allow_unbalanced_panel,
        trim_level=trim_level,
        random_state=random_state,
        n_partitions=n_partitions,
        max_cohorts=max_cohorts,
        progress_bar=progress_bar,
        panel=panel,
    )
