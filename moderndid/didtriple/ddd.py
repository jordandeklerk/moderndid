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

    Implements triple difference-in-differences (DDD) estimation following [1]_. DDD
    extends standard DiD by incorporating a partition variable :math:`Q` that identifies
    eligible units within treatment-enabling groups :math:`S`, allowing for violations
    of traditional DiD parallel trends as long as these violations are stable across
    groups.

    Let :math:`S_i` denote the period when treatment is enabled for unit :math:`i`'s
    group, and :math:`Q_i \in \{0,1\}` indicate eligibility within that group. The
    group-time average treatment effect measures the effect among eligible units in
    group :math:`g` at time :math:`t`

    .. math::

        ATT(g,t) = \mathbb{E}[Y_{i,t}(g) - Y_{i,t}(\infty) \mid S_i = g, Q_i = 1].

    Identification relies on a DDD conditional parallel trends assumption that allows
    for differential trends between eligible and ineligible units, provided these
    differentials are stable across treatment-enabling groups. For groups :math:`g`
    and :math:`g'` where :math:`g' > \max\{g,t\}`

    .. math::

        &\mathbb{E}[\Delta Y(\infty) \mid S=g, Q=1, X]
        - \mathbb{E}[\Delta Y(\infty) \mid S=g, Q=0, X] \\
        &= \mathbb{E}[\Delta Y(\infty) \mid S=g', Q=1, X]
        - \mathbb{E}[\Delta Y(\infty) \mid S=g', Q=0, X],

    where :math:`\Delta Y(\infty) = Y_t(\infty) - Y_{t-1}(\infty)` denotes the change
    in untreated potential outcomes. This assumption does not impose standard DiD
    parallel trends within or across groups, making DDD appealing when such assumptions
    are implausible.

    See the :ref:`triple differences example <example_triple_did>` for a full analysis of
    the crop insurance data.

    Parameters
    ----------
    data : DataFrame
        Data in long format. Accepts any object implementing the Arrow
        PyCapsule Interface (``__arrow_c_stream__``), including polars, pandas,
        pyarrow Table, and cudf DataFrames.
    yname : str
        Name of outcome variable column.
    tname : str
        Name of time period column.
    idname : str, optional
        Name of unit identifier column. Required for panel data, where each
        unit may appear only once per period. For repeated cross-section data
        (panel=False), this can be omitted and a row index will be used
        automatically.
    gname : str
        Name of treatment group column. For 2-period data, this should be
        0 for never-treated and a positive value for treated units. For
        multi-period data, this is the first period when treatment is enabled
        for the unit's group (use 0 or np.inf for never-treated units). A unit
        first treated after the last period counts as never treated. Since a
        unit already treated in the first period has no earlier period to
        compare with, ddd drops it with a warning.
    pname : str
        Name of partition/eligibility column (1=eligible, 0=ineligible).
        This identifies which units within a treatment group are actually
        eligible to receive the treatment effect.
    xformla : str, optional
        Formula for covariates in the form "~ x1 + x2 + x3". If None, only an
        intercept is used.
    control_group : {"nevertreated", "notyettreated"}, default="nevertreated"
        Which units to use as controls in multi-period settings.
        This parameter is ignored for 2-period data. "nevertreated" requires
        at least one never-treated unit. Without never-treated units,
        "notyettreated" leaves out the periods from the first period of the
        latest cohort on. That cohort then serves only as a comparison and
        gets no ATT(g,t) of its own.
    base_period : {"universal", "varying"}, default="universal"
        Base period selection for multi-period settings.
        This parameter is ignored for 2-period data.
    est_method : {"dr", "reg", "ipw"}, default="dr"
        Estimation method: doubly robust, regression, or IPW.
    weightsname : str, optional
        Name of the column of sampling weights. With panel data, a unit must
        keep the same weight in every period.
    boot : bool, default=False
        Whether to use bootstrap for inference. Its intervals are pointwise.
        For simultaneous bands, pass the result to :func:`~moderndid.agg_ddd`.
    boot_type : {"multiplier", "weighted"}, default="multiplier"
        Type of bootstrap for 2-period data (only used if boot=True).
        Multi-period data always uses multiplier bootstrap.
    biters : int, default=1000
        Number of bootstrap repetitions (only used if boot=True).
    cluster : str, optional
        Name of the column that assigns each unit to a cluster, such as its
        county. A unit must keep the same cluster in every period. The standard
        errors then sum the influence function within clusters. The bootstrap
        draws one multiplier per cluster. With two periods, a cluster sets
        boot=True and requires boot_type="multiplier". The aggregations of
        :func:`~moderndid.agg_ddd` cluster by the same column.
    alpha : float, default=0.05
        Significance level for confidence intervals. A value above 0.10 is
        replaced by 0.05 with a warning.
    trim_level : float, default=0.995
        Trimming level for propensity scores. Only used for repeated
        cross-section data (panel=False) and for unbalanced panels with
        allow_unbalanced_panel=True.
    panel : bool, default=True
        Whether the data is panel data (True) or repeated cross-section data (False).
        Panel data has the same units observed across time periods. Repeated
        cross-section data has different samples in each period.
    allow_unbalanced_panel : bool, default=False
        Whether to keep the units of a panel that miss some periods. If True,
        an unbalanced panel takes the repeated cross-section estimator. If
        False, the units that miss one of the remaining periods leave the data
        with a warning. The periods that "notyettreated" leaves out without
        never-treated units don't count. A balanced panel takes the panel
        estimators either way. Since rows with a missing value leave the data
        first, a unit that loses a row this way misses that period.
    random_state : int, Generator, optional
        Random seed for reproducibility of bootstrap.
    n_jobs : int, default=1
        Number of parallel jobs for group-time estimation in multi-period
        settings. 1 = sequential (default), -1 = all cores, >1 = that many
        workers. Ignored for 2-period data.
    backend : {"numpy", "cupy"} or None, default=None
        Array backend to use for this call only. When set, the backend is
        activated for the duration of this call and reverted automatically
        when the call returns. ``None`` (the default) uses whatever backend
        is currently active (see :func:`~moderndid.set_backend`).

    Returns
    -------
    DDDPanelResult, DDDRCResult, DDDMultiPeriodResult, or DDDMultiPeriodRCResult
        For 2-period panel data (panel=True), returns DDDPanelResult containing:

        - **att**: The DDD point estimate
        - **se**: Standard error
        - **uci**, **lci**: Confidence interval bounds
        - **boots**: Bootstrap draws (if requested)
        - **att_inf_func**: Influence function
        - **did_atts**: Individual DiD ATT estimates
        - **subgroup_counts**: Number of units per subgroup
        - **args**: Estimation arguments

        For 2-period repeated cross-section data (panel=False), returns DDDRCResult
        with the same structure.

        For multi-period panel data, returns DDDMultiPeriodResult containing:

        - **att**: Array of ATT(g,t) point estimates
        - **se**: Standard errors for each ATT(g,t)
        - **uci**, **lci**: Confidence interval bounds
        - **groups**, **times**: Treatment cohort and time for each estimate
        - **glist**: Cohorts with ATT(g,t) estimates
        - **tlist**: Unique periods
        - **inf_func_mat**: Influence function matrix
        - **n**: Number of units
        - **args**: Estimation arguments

        For multi-period repeated cross-section data, returns DDDMultiPeriodRCResult
        with the same structure.

    Notes
    -----
    The DDD estimator identifies treatment effects in settings where units must satisfy
    two criteria to be treated: belonging to a group that enables treatment (e.g., a state
    that passes a policy) and being in an eligible partition (e.g., women eligible for
    maternity benefits). This allows for violations of standard DiD parallel trends
    assumptions, as long as these violations are stable across groups.

    When ``est_method="dr"`` (the default), the function implements doubly robust
    DDD estimators that combine outcome regression and inverse probability weighting.
    These estimators are consistent if either the outcome model or the propensity
    score model is correctly specified.

    With ``allow_unbalanced_panel=True`` on an unbalanced panel, the standard
    errors come from each unit's influence function summed over its observations.

    See Also
    --------
    ddd_panel : Two-period DDD estimator for panel data.
    ddd_rc : Two-period DDD estimator for repeated cross-section data.
    ddd_mp : Multi-period DDD estimator for staggered adoption with panel data.
    ddd_mp_rc : Multi-period DDD estimator for staggered adoption with RCS data.
    agg_ddd : Aggregate group-time DDD effects.

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
        return _ddd_rc_2period(
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
        n_boot=biters,
        cluster=cluster,
        cband=cband,
        alp=alpha,
        inf_func=True,
    )

    covariates_with_intercept = add_intercept(ddd_data.covariates)

    return ddd_panel(
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
        covariates=covariates_with_intercept,
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
