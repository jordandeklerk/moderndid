"""Entry point for distributed intertemporal DiD estimation via Dask."""

from __future__ import annotations

import logging

from ._utils import get_or_create_client, validate_dask_input


def dask_did_multiplegt(
    data,
    yname,
    tname,
    idname,
    dname,
    client=None,
    cluster=None,
    weightsname=None,
    xformla="~1",
    effects=1,
    placebo=0,
    normalized=False,
    effects_equal=False,
    predict_het=None,
    switchers="",
    only_never_switchers=False,
    same_switchers=False,
    same_switchers_pl=False,
    trends_lin=False,
    trends_nonparam=None,
    continuous=0,
    ci_level=95.0,
    less_conservative_se=False,
    keep_bidirectional_switchers=False,
    drop_missing_preswitch=False,
    boot=False,
    biters=1000,
    random_state=None,
    n_partitions=None,
):
    r"""Compute distributed intertemporal treatment effects via Dask.

    Distributed implementation of the intertemporal DiD estimator from [1]_
    for datasets that exceed single-machine memory. This function accepts a
    Dask DataFrame, collects it to the driver, and runs the full estimation
    pipeline.

    The underlying methodology is identical to :func:`~moderndid.did_multiplegt`.
    The didinter estimator computes horizon-level effects by comparing
    switchers to non-switchers with matching baseline treatment.

    Users do not need to call this function directly. Passing a Dask
    DataFrame to :func:`~moderndid.did_multiplegt` will automatically
    dispatch here.

    Parameters
    ----------
    data : dask.dataframe.DataFrame
        Data in long format as a Dask DataFrame.
    yname : str
        Name of the outcome variable.
    tname : str
        Name of the time period variable.
    idname : str
        Name of the unit identifier variable.
    dname : str
        Name of the treatment variable.
    client : distributed.Client, optional
        Dask distributed client. If None, a local client is created.
    cluster : str, optional
        Cluster variable for standard errors.
    effects : int, default=1
        Number of post-treatment horizons.
    placebo : int, default=0
        Number of pre-treatment placebo horizons.
    normalized : bool, default=False
        If True, normalize by cumulative treatment change.
    boot : bool, default=False
        Whether to use bootstrap inference.
    biters : int, default=1000
        Number of bootstrap iterations.
    random_state : int, optional
        Random seed.
    n_partitions : int, optional
        Not used for didinter (reserved for API consistency).

    Returns
    -------
    DIDInterResult
        Treatment effect results.

    See Also
    --------
    did_multiplegt : Local (non-distributed) intertemporal DiD estimator.

    References
    ----------

    .. [1] de Chaisemartin, C., & D'Haultfoeuille, X. (2024).
           "Difference-in-Differences Estimators of Intertemporal Treatment
           Effects." *Review of Economics and Statistics*, 106(6), 1723-1736.
    """
    client = get_or_create_client(client)
    logging.getLogger("distributed.shuffle").setLevel(logging.ERROR)

    required_cols = [yname, tname, idname, dname]
    if cluster is not None:
        required_cols.append(cluster)
    if weightsname is not None:
        required_cols.append(weightsname)
    validate_dask_input(data, required_cols)

    from ._didinter_mp import dask_did_multiplegt_mp

    return dask_did_multiplegt_mp(
        client=client,
        data=data,
        yname=yname,
        tname=tname,
        idname=idname,
        dname=dname,
        cluster=cluster,
        weightsname=weightsname,
        xformla=xformla,
        effects=effects,
        placebo=placebo,
        normalized=normalized,
        effects_equal=effects_equal,
        predict_het=predict_het,
        switchers=switchers,
        only_never_switchers=only_never_switchers,
        same_switchers=same_switchers,
        same_switchers_pl=same_switchers_pl,
        trends_lin=trends_lin,
        trends_nonparam=trends_nonparam,
        continuous=continuous,
        ci_level=ci_level,
        less_conservative_se=less_conservative_se,
        keep_bidirectional_switchers=keep_bidirectional_switchers,
        drop_missing_preswitch=drop_missing_preswitch,
        boot=boot,
        biters=biters,
        random_state=random_state,
    )
