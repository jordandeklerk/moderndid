"""Main wrapper for Triple Difference-in-Differences estimation."""

import warnings

import numpy as np
import polars as pl

from moderndid.core.dataframe import to_polars
from moderndid.core.preprocess.constants import ROW_ID_COLUMN, WEIGHTS_COLUMN
from moderndid.core.preprocess.utils import get_column_terms
from moderndid.core.preprocess.validators import _reserved_name_errors, check_columns
from moderndid.core.preprocessing import preprocess_ddd_2periods

from .estimators.ddd_mp import _check_options, _ddd_mp, _preprocess_multiple_periods
from .estimators.ddd_mp_rc import _ddd_mp_rc
from .estimators.ddd_panel import ddd_panel
from .estimators.ddd_rc import _ddd_rc_2period
from .utils import _complete_rows, add_intercept, detect_multiple_periods, detect_rcs_mode, get_covariate_names


def ddd(
    data,
    yname,
    tname,
    idname=None,
    gname=None,
    pname=None,
    xformla=None,
    control_group="nevertreated",
    base_period="universal",
    est_method="dr",
    weightsname=None,
    boot=False,
    boot_type="multiplier",
    biters=1000,
    cluster=None,
    alpha=0.05,
    trim_level=0.995,
    panel=True,
    allow_unbalanced_panel=False,
    random_state=None,
    n_jobs=1,
    backend=None,
):
    r"""Compute the doubly robust Triple Difference-in-Differences estimator for the ATT.

    Estimates the effect of a treatment that reaches only the eligible units
    (:math:`Q = 1`) of the groups that enable it [1]_. With :math:`S` the period in
    which a unit's group enables treatment, the group-time average treatment effect on
    the eligible units of cohort :math:`g` in period :math:`t` is

    .. math::

        ATT(g,t) = \mathbb{E}[Y_{i,t}(g) - Y_{i,t}(\infty) \mid S_i = g, Q_i = 1].

    The ineligible units measure the trend gap that an ordinary DiD would leave in the
    estimate. Trends may therefore differ across groups and between eligible and
    ineligible units, provided the eligible-minus-ineligible gap is the same in every
    group. With two periods the function returns one effect. With more it returns an
    :math:`ATT(g,t)` for each cohort and period for :func:`~moderndid.agg_ddd` to
    aggregate.

    See the :ref:`triple differences example <example_triple_did>` for a full analysis
    of the crop insurance data.

    Parameters
    ----------
    data : DataFrame
        Data in long format. Accepts any object implementing the Arrow PyCapsule
        Interface (``__arrow_c_stream__``), including polars, pandas, pyarrow Table, and
        cudf DataFrames.
    yname : str
        The name of the outcome column.
    tname : str
        The name of the time period column.
    idname : str, optional
        The name of the unit identifier column. Required for panel data, where each unit
        appears once per period.
    gname : str
        The name of the column holding the period in which each unit's group enables
        treatment, 0 or ``np.inf`` if it never does. With two periods, any positive
        value marks a treated group. A unit enabled after the last period counts as
        never treated. Units already treated in the first period leave the data with a
        warning.
    pname : str
        The name of the eligibility column, 1 for eligible units and 0 for the rest.
    xformla : str, optional
        A formula for the covariates, such as ``"~ x1 + x2"``.
    control_group : {"nevertreated", "notyettreated"}, default="nevertreated"
        Whether the comparison group holds only the never-treated units or also the
        units not yet treated. Ignored with two periods. Without never-treated units,
        ``"nevertreated"`` raises an error. ``"notyettreated"`` instead leaves out the
        periods from the latest cohort's first period on and uses that cohort only as a
        comparison.
    base_period : {"universal", "varying"}, default="universal"
        ``"universal"`` compares every period with the last period before treatment.
        ``"varying"`` compares each pre-treatment period with the one before it.
        Ignored with two periods.
    est_method : {"dr", "reg", "ipw"}, default="dr"
        The estimator, doubly robust, outcome regression, or inverse probability
        weighting.
    weightsname : str, optional
        The name of the sampling weights column. With panel data, a unit must keep the
        same weight in every period.
    boot : bool, default=False
        Whether to compute the standard errors with the bootstrap. Its intervals are
        pointwise. :func:`~moderndid.agg_ddd` gives simultaneous bands.
    boot_type : {"multiplier", "weighted"}, default="multiplier"
        The bootstrap with two periods. More periods always use the multiplier
        bootstrap.
    biters : int, default=1000
        The number of bootstrap draws. Only applicable if ``boot=True``.
    cluster : str, optional
        The name of the column that assigns each unit to a cluster for the standard
        errors, such as its county. A unit must keep the same cluster in every period.
        With two periods, a cluster sets ``boot=True`` with a warning and requires
        ``boot_type="multiplier"``.
    alpha : float, default=0.05
        The significance level. A value above 0.10 becomes 0.05 with a warning.
    trim_level : float, default=0.995
        The trimming level of the propensity scores. Used only with repeated
        cross-sections and with unbalanced panels under ``allow_unbalanced_panel=True``.
    panel : bool, default=True
        Whether the data is a panel (True) or repeated cross-sections (False).
    allow_unbalanced_panel : bool, default=False
        Whether to keep the units of a panel that miss some periods. If True, an
        unbalanced panel takes the repeated cross-section estimator. If False, the units
        that miss one of the periods kept for estimation leave the data with a warning.
        A row with a missing value counts as a missed period.
    random_state : int, Generator, optional
        Controls the randomness of the bootstrap. Pass an int for reproducible results
        across calls.
    n_jobs : int, default=1
        The number of parallel jobs for the group-time cells. Pass -1 to use all cores.
        Ignored with two periods.
    backend : {"numpy", "cupy"}, optional
        The array backend for this call only. None uses the backend that
        :func:`~moderndid.set_backend` made active.

    Returns
    -------
    DDDPanelResult, DDDRCResult, DDDMultiPeriodResult, or DDDMultiPeriodRCResult
        With two periods, a DDDPanelResult for a panel or a DDDRCResult for
        repeated cross-sections, each containing:

        - **att**: The DDD point estimate
        - **se**: Standard error
        - **uci**: Upper bound of the confidence interval
        - **lci**: Lower bound of the confidence interval
        - **boots**: Bootstrap draws (if requested)
        - **att_inf_func**: Influence function
        - **did_atts**: Individual DiD ATT estimates
        - **subgroup_counts**: Number of units per subgroup
        - **args**: Estimation arguments

        With more periods, a DDDMultiPeriodResult for a panel or a
        DDDMultiPeriodRCResult for repeated cross-sections, each containing:

        - **att**: Array of ATT(g,t) point estimates
        - **se**: Standard errors for each ATT(g,t)
        - **uci**: Upper confidence interval bounds
        - **lci**: Lower confidence interval bounds
        - **groups**: Treatment cohort of each estimate
        - **times**: Time period of each estimate
        - **glist**: Cohorts with ATT(g,t) estimates
        - **tlist**: Unique periods
        - **inf_func_mat**: Influence function matrix
        - **n**: Number of units
        - **args**: Estimation arguments
        - **unit_groups**: Enabling cohort of each unit, or 0 for a unit whose group never enables treatment
        - **unit_weights**: Normalized sampling weight of each unit, or None without weights
        - **unit_clusters**: Cluster of each unit, or None without a cluster
        - **unit_partition**: Eligibility of each unit, 1 if eligible and 0 if not

        The unit-level arrays of a DDDMultiPeriodRCResult hold one entry per
        observation.

    See Also
    --------
    ddd_panel : Two-period DDD estimator for panel data.
    ddd_rc : Two-period DDD estimator for repeated cross-section data.
    ddd_mp : Multi-period DDD estimator for staggered adoption with panel data.
    ddd_mp_rc : Multi-period DDD estimator for staggered adoption with RCS data.
    agg_ddd : Aggregate group-time DDD effects.

    Notes
    -----
    Identification rests on DDD conditional parallel trends. For cohorts :math:`g` and
    :math:`g' > \max\{g,t\}`,

    .. math::

        &\mathbb{E}[\Delta Y(\infty) \mid S=g, Q=1, X]
        - \mathbb{E}[\Delta Y(\infty) \mid S=g, Q=0, X] \\
        &= \mathbb{E}[\Delta Y(\infty) \mid S=g', Q=1, X]
        - \mathbb{E}[\Delta Y(\infty) \mid S=g', Q=0, X],

    where :math:`\Delta Y(\infty) = Y_t(\infty) - Y_{t-1}(\infty)` is the change in
    untreated potential outcomes. The :ref:`triple differences background
    <background-tripledid>` derives the doubly robust estimator, its combination of
    several comparison cohorts, and its influence function.

    With a cluster, a warning appears when the eligible units of an enabling cohort fall
    in fewer than 5 clusters or all units fall in fewer than 30. On an unbalanced panel
    with ``allow_unbalanced_panel=True``, the standard errors sum each unit's influence
    function over its observations.

    References
    ----------

    .. [1] Ortiz-Villavicencio, M., & Sant'Anna, P. H. C. (2025).
        *Better Understanding Triple Differences Estimators.*
        arXiv preprint arXiv:2505.09942. https://arxiv.org/abs/2505.09942
    """
    if backend is not None:
        from moderndid.cupy.backend import use_backend

        with use_backend(backend):
            return ddd(
                data=data,
                yname=yname,
                tname=tname,
                idname=idname,
                gname=gname,
                pname=pname,
                xformla=xformla,
                control_group=control_group,
                base_period=base_period,
                est_method=est_method,
                weightsname=weightsname,
                boot=boot,
                boot_type=boot_type,
                biters=biters,
                cluster=cluster,
                alpha=alpha,
                trim_level=trim_level,
                panel=panel,
                allow_unbalanced_panel=allow_unbalanced_panel,
                random_state=random_state,
                n_jobs=n_jobs,
                backend=None,
            )

    if gname is None:
        raise ValueError("gname is required. Please specify the treatment group column.")
    if pname is None:
        raise ValueError("pname is required. Please specify the partition/eligibility column.")
    if panel and idname is None:
        raise ValueError("idname must be provided when panel=True.")
    _check_options(est_method, control_group, base_period, alpha, biters, trim_level, n_jobs, boot_type)
    named_columns = {
        "yname": yname,
        "tname": tname,
        "idname": idname,
        "gname": gname,
        "pname": pname,
        "xformla": xformla,
        "weightsname": weightsname,
        "cluster": cluster,
    }
    data = to_polars(data)
    check_columns(data, **named_columns)
    covariate_terms = get_column_terms(xformla) if xformla else []
    # Since the cross-section routes number the rows in an internal column, no named column may take its name.
    reserved = _reserved_name_errors(named_columns | {"xformla": covariate_terms}, (ROW_ID_COLUMN,))
    if reserved:
        raise ValueError("\n".join(reserved))

    # Since a missing value must not pick the route, the routing reads the rows that the missing-data step keeps.
    # The route's own missing-data step still drops the rows with a missing value and warns once.
    complete = _complete_rows(data, [yname, tname, idname, gname, pname, cluster, weightsname, *covariate_terms], gname)
    is_rcs = detect_rcs_mode(complete, tname, idname, panel, allow_unbalanced_panel)

    if is_rcs and idname is None:
        data = data.with_columns(pl.Series(ROW_ID_COLUMN, np.arange(len(data))))
        idname = ROW_ID_COLUMN

    multiple_periods = detect_multiple_periods(complete, tname, gname)

    # Settling the inference options before routing gives every data layout the same interval level.
    if alpha > 0.10:
        warnings.warn(f"alpha={alpha} is above 0.10. Using alpha=0.05.", UserWarning, stacklevel=2)
        alpha = 0.05
    if cluster is not None and not multiple_periods:
        if boot_type != "multiplier":
            raise ValueError("cluster requires boot_type='multiplier' with two periods.")
        if not boot:
            warnings.warn("Clustered SEs require bootstrap. Setting boot=True.", UserWarning, stacklevel=2)
            boot = True
    # Every route reports pointwise intervals. agg_ddd builds the simultaneous bands.
    cband = False

    if multiple_periods:
        covariate_cols = get_covariate_names(xformla)
        data, glist = _preprocess_multiple_periods(
            data,
            yname,
            tname,
            idname,
            gname,
            pname,
            covariate_cols,
            weightsname,
            cluster,
            panel,
            allow_unbalanced_panel,
            control_group,
        )
        weights_col = None if weightsname is None else WEIGHTS_COLUMN
        if not panel:
            return _ddd_mp_rc(
                data=data,
                glist=glist,
                y_col=yname,
                time_col=tname,
                id_col=idname,
                group_col=gname,
                partition_col=pname,
                covariate_cols=covariate_cols,
                control_group=control_group,
                base_period=base_period,
                est_method=est_method,
                boot=boot,
                biters=biters,
                cband=cband,
                cluster=cluster,
                alpha=alpha,
                trim_level=trim_level,
                random_state=random_state,
                n_jobs=n_jobs,
                weights_col=weights_col,
            )
        return _ddd_mp(
            data=data,
            glist=glist,
            y_col=yname,
            time_col=tname,
            id_col=idname,
            group_col=gname,
            partition_col=pname,
            covariate_cols=covariate_cols,
            control_group=control_group,
            base_period=base_period,
            est_method=est_method,
            boot=boot,
            biters=biters,
            cband=cband,
            cluster=cluster,
            alpha=alpha,
            allow_unbalanced_panel=allow_unbalanced_panel,
            random_state=random_state,
            n_jobs=n_jobs,
            weights_col=weights_col,
            trim_level=trim_level,
        )

    if is_rcs:
        result = _ddd_rc_2period(
            data=data,
            yname=yname,
            tname=tname,
            gname=gname,
            pname=pname,
            xformla=xformla,
            weightsname=weightsname,
            est_method=est_method,
            boot=boot,
            boot_type=boot_type,
            biters=biters,
            alpha=alpha,
            trim_level=trim_level,
            random_state=random_state,
            cluster=cluster,
            idname=idname,
            panel=panel,
        )
    else:
        ddd_data = preprocess_ddd_2periods(
            data=data,
            yname=yname,
            tname=tname,
            idname=idname,
            gname=gname,
            pname=pname,
            xformla=xformla,
            est_method=est_method,
            weightsname=weightsname,
            boot=boot,
            boot_type=boot_type,
            biters=biters,
            cluster=cluster,
            cband=cband,
            alpha=alpha,
            inf_func=True,
        )
        result = ddd_panel(
            y1=ddd_data.y1,
            y0=ddd_data.y0,
            subgroup=ddd_data.subgroup,
            covariates=add_intercept(ddd_data.covariates),
            i_weights=ddd_data.weights,
            est_method=est_method,
            boot=boot,
            boot_type=boot_type,
            biters=biters,
            influence_func=True,
            alpha=alpha,
            random_state=random_state,
            cluster=ddd_data.cluster,
        )

    # Since the two-period estimators work on arrays, the report learns the column names here.
    named_cluster = cluster if result.args.get("cluster") else None
    return result._replace(args=result.args | {"yname": yname, "pname": pname, "cluster": named_cluster})
