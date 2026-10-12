"""Continuous treatment difference-in-differences estimation."""

import warnings
from functools import partial

import numpy as np
import polars as pl
from scipy import stats

from moderndid.core.dataframe import to_polars
from moderndid.core.preprocess import (
    get_first_difference as _get_first_difference,
)
from moderndid.core.preprocess import (
    make_balanced_panel as _make_balanced_panel,
)
from moderndid.core.preprocess.validators import _reserved_name_errors, check_columns
from moderndid.core.preprocessing import preprocess_cont_did
from moderndid.cupy.backend import get_backend, to_device, to_numpy, use_backend
from moderndid.npiv import gsl_bs, npiv

from .estimation import (
    AttgtResult,
    _build_pte_params,
    pte,
    pte_default,
)
from .estimation.estimators import pte_attgt
from .estimation.process_dose import DoseResult
from .estimation.process_panel import (
    _and_list,
    _check_event_time_window,
    _event_time_options,
    _two_by_two_subset,
)
from .spline import BSpline


def cont_did(
    data,
    yname,
    tname,
    idname,
    gname=None,
    dname=None,
    xformla="~1",
    target_parameter="level",
    aggregation="dose",
    balance_e=None,
    min_e=-np.inf,
    max_e=np.inf,
    treatment_type="continuous",
    dose_est_method="parametric",
    dvals=None,
    degree=3,
    num_knots=0,
    allow_unbalanced_panel=False,
    control_group="notyettreated",
    anticipation=0,
    weightsname=None,
    alpha=0.05,
    cband=False,
    boot_type="multiplier",
    biters=1000,
    clustervars=None,
    base_period="varying",
    random_state=None,
    backend=None,
):
    r"""Compute difference-in-differences with a continuous treatment.

    Implements the difference-in-differences estimator of [1]_ for a treatment
    that comes in different amounts, or doses, across units. Units may start
    treatment in different periods but keep one dose over time.

    Two effects arise at each dose :math:`d`. The level effect
    :math:`ATT(d \mid d)` compares outcomes under dose :math:`d` with outcomes
    without treatment among the units that received dose :math:`d`. The
    average causal response :math:`ACRT(d \mid d)` measures how the outcome of
    those units would respond to a slightly larger dose. Parallel trends
    identifies the level effects and their local average :math:`ATT^{loc}`
    over treated doses. The curve's derivative also changes the dose group
    being compared. A causal interpretation therefore requires restrictions
    on treated potential outcomes. In two periods, strong parallel trends
    identifies the response for all treated units rather than necessarily
    the response local to the observed dose group.

    With ``aggregation="dose"``, the estimator fits a curve in the dose for
    each cohort in each period after treatment starts. It then averages those
    curves using fixed cohort shares and averages each cell's effects over
    its own observed doses for the overall summaries. With
    ``aggregation="eventstudy"``, it reports the level effects or the average
    slopes by time since treatment started.

    ``min_e``, ``max_e``, and ``balance_e`` trim or balance the event study.
    A window that keeps no event time raises an error. When it keeps no
    event time from the start of treatment on, the overall effect is NaN and
    a warning says why.

    Each cohort's spline uses the interior knots that fall inside its own
    dose range. It has ``degree + 1`` coefficients plus one for each of those
    knots. A unique fit needs at least that many distinct positive doses
    spread across the knots. Doses that nearly coincide count as one. A
    warning names each cohort whose doses can't give a unique fit. Such a
    cohort has no curve or slope. Its cells leave the curves, the overall
    ACRT, and the event study of slopes. Its binary ATT stays in the overall
    ATT. If no cohort is left, ``cont_did`` raises an error.

    See the :ref:`continuous treatment example <example_cont_did>` for a full
    analysis of county employment and geological exposure to fracking.
    The :ref:`continuous treatment background <background-didcont>` explains
    the distinction between local and global effects and the additional
    restrictions needed to interpret the reported slopes.

    Parameters
    ----------
    data : DataFrame
        Panel data in long format. Accepts any object implementing the Arrow
        PyCapsule Interface (``__arrow_c_stream__``), including polars, pandas,
        pyarrow Table, and cudf DataFrames.
    yname : str
        Name of the column containing the outcome variable.
    tname : str
        Name of the column containing the time period variable.
    idname : str
        Name of the column containing the unit ID variable.
    gname : str, optional
        Name of the column containing the timing-group variable indicating
        when treatment starts for each unit. Each group must be one of the
        observed periods, or 0 for never-treated units. If None, each unit's
        group is the first period in which its dose is positive. The dose
        must then be 0 before treatment starts. Because the group is stored
        in a column named ``.G``, no column that the other arguments name may
        be called ``.G``.
    dname : str
        Name of the column containing the continuous treatment variable,
        the "dose" or amount of treatment received. Each unit's dose must
        stay the same over time. Before treatment starts, it may also be
        recorded as 0. Never-treated units get a dose of 0 whatever the column
        holds.
    xformla : str, default="~1"
        Formula for the covariates. Since covariates aren't supported yet, it
        must be ``"~1"``.
    target_parameter : {"level", "slope"}, default="level"
        Effect that an event study averages. ``"level"`` treats every unit in
        a cohort as treated whatever its dose. ``"slope"`` averages the slope
        of each cohort's curve. With ``aggregation="dose"``, both curves are
        returned whatever this says.
    aggregation : {"dose", "eventstudy"}, default="dose"
        How to average the cohort-period effects. ``"dose"`` reports the curves
        over doses along with the overall ATT and ACRT. ``"eventstudy"``
        reports the effects by time since treatment started. It counts that
        time in units of ``tname``.
    balance_e : float, optional
        Event time to balance the event study on. With ``balance_e=2``, the
        event study keeps only the cohorts observed through event time 2 and
        reports event times up to 2. The same cohorts then make up every event
        time from 0 to 2. It needs ``aggregation="eventstudy"``.
    min_e : float, default=-inf
        Smallest event time to include in the event study. It needs
        ``aggregation="eventstudy"``.
    max_e : float, default=inf
        Largest event time to include in the event study. It needs
        ``aggregation="eventstudy"``.
    treatment_type : {"continuous", "discrete"}, default="continuous"
        Type of treatment. Only ``"continuous"`` is supported.
    dose_est_method : {"parametric", "cck"}, default="parametric"
        Estimator of the curves. ``"parametric"`` fits the B-spline that
        ``degree`` and ``num_knots`` set. ``"cck"`` chooses the sieve from the
        data following [2]_ and needs two periods with a single treated cohort.
        It also needs at least four distinct treated doses. The sieve that it
        selects can need more.
    dvals : array-like, optional
        One or more finite doses at which to evaluate the curves. Defaults to
        50 evenly spaced doses between the smallest and largest treated dose.
    degree : int, default=3
        Degree of the B-spline in the dose. With ``num_knots=0`` the spline is
        a single polynomial of this degree.
    num_knots : int, default=0
        Number of interior knots of the B-spline, placed at quantiles of the
        treated doses. More knots make the curve more flexible and its
        estimates noisier. Ties among the treated doses must not put more
        than ``degree`` knots at one dose.
    allow_unbalanced_panel : bool, default=False
        Whether to allow an unbalanced panel. Only False is supported.
    control_group : {"notyettreated", "nevertreated"}, default="notyettreated"
        Units to compare with. ``"notyettreated"`` uses never-treated units
        together with cohorts whose treatment, including any anticipation,
        starts after both periods of a comparison. ``"nevertreated"`` uses
        never-treated units alone.
    anticipation : int, default=0
        Number of observed periods before treatment in which outcomes may
        already react. It counts the periods in the data rather than units of
        ``tname``.
    weightsname : str, optional
        Name of the column containing sampling weights. Sampling weights
        are not supported yet. Only None is accepted.
    alpha : float, default=0.05
        Significance level of the intervals and bands, such as 0.05 for 95
        percent coverage.
    cband : bool, default=False
        Whether each band covers all doses, or all event times, at once.
        With ``dose_est_method="cck"``, the fitted level band uses a
        conservative approximation.
    boot_type : {"multiplier", "weighted", "empirical"}, default="multiplier"
        Bootstrap for event-study standard errors and bands.
        ``"multiplier"`` draws a random weight for each unit and applies it to
        that unit's influence function. ``"weighted"`` is another name for
        ``"multiplier"``. ``"empirical"`` resamples units and reruns the
        estimation in each draw. Since the dose curves always use the
        multiplier bootstrap, ``"empirical"`` needs
        ``aggregation="eventstudy"``. It doesn't support the event-time
        options ``min_e``, ``max_e``, and ``balance_e``.
    biters : int, default=1000
        Number of bootstrap draws.
    clustervars : str or list of str, optional
        Variables for clustering standard errors. The standard errors always
        cluster by ``idname``. Any other variable is ignored with a warning
        because clustering on it isn't supported yet.
    base_period : {"varying", "universal"}, default="varying"
        Base period of each comparison. ``"universal"`` measures every period
        from the one before the cohort starts. ``"varying"`` measures each
        period before treatment from the period just before it.
    random_state : int, Generator, optional
        Controls the randomness of the bootstrap. Pass an int for reproducible
        results across multiple function calls. Can also accept a NumPy
        ``Generator`` instance.
    backend : {"numpy", "cupy"} or None, default=None
        Array backend to use for this call only. When set, the backend is
        activated before estimation and the previous backend is restored
        when the call returns. ``None`` (the default) uses whatever backend
        is currently active (see :func:`~moderndid.set_backend`).

    Returns
    -------
    DoseResult or PTEResult
        With ``aggregation="dose"``, a DoseResult containing:

        - **dose**: Doses at which the curves are evaluated
        - **att_d**: Fitted level contrasts at each dose
        - **att_d_se**: Standard errors of the fitted level contrasts
        - **att_d_crit_val**: Critical value of the level band
        - **acrt_d**: Fitted dose derivatives at each dose
        - **acrt_d_se**: Standard errors of the fitted dose derivatives
        - **acrt_d_crit_val**: Critical value of the derivative band
        - **overall_att**: Overall ATT
        - **overall_att_se**: Standard error of the overall ATT
        - **overall_acrt**: Overall ACRT
        - **overall_acrt_se**: Standard error of the overall ACRT

        With ``aggregation="eventstudy"``, a PTEResult that holds the event
        study in ``event_study``. Its overall effect averages the event-study
        effects at the event times e >= 0 that the event study reports.

    Notes
    -----
    The level effect and the average causal response at dose :math:`d` are

    .. math::

        ATT(d \mid d) = \mathbb{E}[Y_{t}(d) - Y_{t}(0) \mid D = d], \qquad
        ACRT(d \mid d) = \left.\frac{\partial}{\partial l}
        \mathbb{E}[Y_{t}(l) \mid D = d]\right|_{l=d}.

    Under parallel trends, a comparison of the outcome changes of units at
    dose :math:`d` with those of untreated units identifies the level effect,

    .. math::

        ATT(d \mid d) = \mathbb{E}[\Delta Y \mid D = d] - \mathbb{E}[\Delta Y \mid D = 0].

    Averaging over the doses of the treated units gives the summaries

    .. math::

        ATT^{loc} = \mathbb{E}[ATT(D \mid D) \mid D > 0], \qquad
        ACRT^{loc} = \mathbb{E}[ACRT(D \mid D) \mid D > 0].

    The slope of the observed level comparison includes selection across
    dose groups under ordinary parallel trends. The two-period strong
    parallel trends assumption identifies :math:`ACRT(d)` for all treated
    units and its global average. The stronger multi-period restriction in
    Appendix C of [1]_ supports a local response interpretation as well.

    With staggered adoption, these quantities are estimated for each cohort in
    each period after treatment starts. The dose aggregation gives each cohort
    its share of the treated units and splits that share evenly over the
    cohort's periods after treatment starts. When a cohort has no curve, the
    other cohorts share its weight in the curves and the overall ACRT.

    References
    ----------

    .. [1] Callaway, B., Goodman-Bacon, A., & Sant'Anna, P. H. C. (2025).
           "Difference-in-differences with a continuous treatment."
           American Economic Review, forthcoming.
           December 31, 2025 manuscript.
           https://psantanna.com/files/CGBS_v4.pdf

    .. [2] Chen, X., Christensen, T. M., & Kankanala, S. (2025).
           "Adaptive Estimation and Uniform Confidence Bands for Nonparametric
           Structural Functions and Elasticities."
           The Review of Economic Studies, 92(1), 162-196.
           https://doi.org/10.1093/restud/rdae025
    """
    if backend is not None:
        with use_backend(backend):
            return cont_did(
                data=data,
                yname=yname,
                tname=tname,
                idname=idname,
                gname=gname,
                dname=dname,
                xformla=xformla,
                target_parameter=target_parameter,
                aggregation=aggregation,
                balance_e=balance_e,
                min_e=min_e,
                max_e=max_e,
                treatment_type=treatment_type,
                dose_est_method=dose_est_method,
                dvals=dvals,
                degree=degree,
                num_knots=num_knots,
                allow_unbalanced_panel=allow_unbalanced_panel,
                control_group=control_group,
                anticipation=anticipation,
                weightsname=weightsname,
                alpha=alpha,
                cband=cband,
                boot_type=boot_type,
                biters=biters,
                clustervars=clustervars,
                base_period=base_period,
                random_state=random_state,
                backend=None,
            )

    if dname is None:
        raise ValueError("dname is required. Please specify the dose/treatment column.")

    if aggregation not in ("dose", "eventstudy"):
        raise ValueError(f"aggregation={aggregation!r} is not valid. Must be 'dose' or 'eventstudy'.")
    if target_parameter not in ("level", "slope"):
        raise ValueError(f"target_parameter={target_parameter!r} is not valid. Must be 'level' or 'slope'.")
    if dose_est_method not in ("parametric", "cck"):
        raise ValueError(f"dose_est_method={dose_est_method!r} is not valid. Must be 'parametric' or 'cck'.")
    if control_group not in ("notyettreated", "nevertreated"):
        raise ValueError(f"control_group={control_group!r} is not valid. Must be 'notyettreated' or 'nevertreated'.")
    if base_period not in ("varying", "universal"):
        raise ValueError(f"base_period={base_period!r} is not valid. Must be 'varying' or 'universal'.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha={alpha} is not valid. Must be between 0 and 1 (exclusive).")
    if not isinstance(biters, int) or biters < 1:
        raise ValueError(f"biters={biters} is not valid. Must be a positive integer.")
    if boot_type not in ("weighted", "multiplier", "empirical"):
        raise ValueError(f"boot_type={boot_type!r} is not valid. Must be 'weighted', 'multiplier', or 'empirical'.")
    if boot_type == "empirical" and aggregation == "dose":
        raise ValueError(
            "boot_type='empirical' needs aggregation='eventstudy'. Since the dose curves always use the multiplier "
            "bootstrap, use boot_type='multiplier' with aggregation='dose'."
        )
    _check_event_time_window(min_e, max_e, balance_e)
    event_time_options = [f"{name}={value}" for name, value in _event_time_options(min_e, max_e, balance_e).items()]
    if aggregation == "dose" and event_time_options:
        listed = _and_list(event_time_options)
        need = "needs" if len(event_time_options) == 1 else "need"
        if dose_est_method == "cck":
            them = "it" if len(event_time_options) == 1 else "them"
            raise ValueError(
                f"{listed} {need} an event study. Since dose_est_method='cck' has no event study, leave {them} out."
            )
        raise ValueError(f"{listed} {need} aggregation='eventstudy'.")
    if not isinstance(anticipation, int | float) or anticipation < 0:
        raise ValueError(f"anticipation={anticipation} is not valid. Must be a non-negative number.")
    if degree < 1:
        raise ValueError(f"degree={degree} is not valid. Must be at least 1.")
    if num_knots < 0:
        raise ValueError(f"num_knots={num_knots} is not valid. Must be non-negative.")
    if dvals is not None:
        dvals = np.asarray(dvals, dtype=float).ravel()
        if dvals.size == 0:
            raise ValueError(
                "dvals is empty. Pass the doses at which to evaluate the curves, or None for 50 evenly spaced "
                "treated doses."
            )
        if not np.all(np.isfinite(dvals)):
            raise ValueError(
                "dvals holds missing or infinite values. Pass finite doses at which to evaluate the curves."
            )
    if treatment_type not in ("continuous", "discrete"):
        raise ValueError(f"treatment_type={treatment_type!r} is not valid. Must be 'continuous' or 'discrete'.")

    data = to_polars(data)

    if xformla != "~1":
        raise NotImplementedError("Covariates not currently supported, use xformla='~1'")

    if treatment_type == "discrete":
        raise NotImplementedError("Discrete treatment not yet supported")

    if allow_unbalanced_panel:
        raise NotImplementedError("Unbalanced panel not currently supported")

    if weightsname is not None:
        raise NotImplementedError("Sampling weights are not supported yet. Use weightsname=None.")

    if clustervars is not None:
        cluster_names = [clustervars] if isinstance(clustervars, str) else list(clustervars)
        # Since the standard errors treat each unit as an independent observation, clustering by idname changes nothing.
        other_clusters = [name for name in cluster_names if name != idname]
        if other_clusters:
            warnings.warn(
                f"cont_did clusters standard errors by unit only. It ignores clustervars={other_clusters!r} since "
                "clustering on other variables isn't supported yet.",
                UserWarning,
            )
        clustervars = None

    if dose_est_method == "cck" and aggregation != "dose":
        raise ValueError("Event study not supported with CCK estimator yet, use aggregation='dose'")

    check_columns(data, yname=yname, tname=tname, idname=idname, gname=gname, dname=dname)

    if gname is None:
        errors = _reserved_name_errors({"yname": yname, "tname": tname, "idname": idname, "dname": dname}, (".G",))
        if errors:
            raise ValueError("\n".join(errors))
        # Writing the group straight to .G leaves every column of the data as it was.
        first_dose = pl.col(tname).filter(pl.col(dname) > 0).min().over(idname)
        data = data.with_columns(first_dose.fill_null(0).cast(pl.Int64).alias(".G"))
        gname = ".G"
        treated_starts = data.filter(pl.col(gname) > 0)[gname]
        if treated_starts.len() > 0 and (treated_starts == data[tname].min()).all():
            raise ValueError(
                "With gname=None, a unit's group is the first period in which its dose is positive. Since every "
                "unit with a positive dose already has one in the first period, no unit is observed before "
                "treatment. Pass gname or record the dose as 0 before treatment starts."
            )

    req_pre_periods = 0 if dose_est_method == "cck" else 1

    cont_did_data = preprocess_cont_did(
        data=data,
        yname=yname,
        tname=tname,
        gname=gname,
        dname=dname,
        idname=idname,
        xformla=xformla,
        panel=True,
        allow_unbalanced_panel=allow_unbalanced_panel,
        control_group=control_group,
        anticipation=anticipation,
        weightsname=weightsname,
        alpha=alpha,
        cband=cband,
        biters=biters,
        clustervars=clustervars,
        degree=degree,
        num_knots=num_knots,
        dvals=dvals,
        target_parameter=target_parameter,
        aggregation=aggregation,
        base_period=base_period,
        boot_type=boot_type,
        required_pre_periods=req_pre_periods,
        dose_est_method=dose_est_method,
    )

    if dose_est_method == "cck":
        return _estimate_cck(
            cont_did_data=cont_did_data,
            original_data=data,
            random_state=random_state,
        )

    subset_fun = cont_two_by_two_subset
    if aggregation == "eventstudy":
        if target_parameter == "slope":
            attgt_fun = cont_did_acrt
            gt_type = "dose"
        else:
            # A level effect averaged over doses compares the cohort with untreated units. Its cells
            # take the binary treatment indicator in place of the dose.
            attgt_fun = pte_attgt
            subset_fun = _two_by_two_subset
            gt_type = "att"
    elif target_parameter in ["level", "slope"]:
        attgt_fun = cont_did_acrt
        gt_type = "dose"
    else:
        raise ValueError(f"Invalid combination of parameters: {target_parameter}, {aggregation}, {treatment_type}")

    setup_fn = partial(_build_pte_params, cont_did_data, gt_type=gt_type)

    return pte(
        yname=cont_did_data.config.yname,
        gname=cont_did_data.config.gname,
        tname=cont_did_data.config.tname,
        idname=cont_did_data.config.idname,
        data=cont_did_data.data,
        setup_pte_fun=setup_fn,
        subset_fun=subset_fun,
        attgt_fun=attgt_fun,
        xformla=xformla,
        target_parameter=target_parameter,
        aggregation=aggregation,
        treatment_type=treatment_type,
        dose_est_method=dose_est_method,
        anticipation=anticipation,
        gt_type=gt_type,
        cband=cband,
        alpha=alpha,
        boot_type=boot_type,
        biters=biters,
        dname=dname,
        degree=degree,
        num_knots=num_knots,
        dvals=dvals,
        control_group=control_group,
        base_period=base_period,
        weightsname=weightsname,
        random_state=random_state,
        d_outcome=aggregation == "eventstudy",
        balance_e=balance_e,
        min_e=min_e,
        max_e=max_e,
    )


def cont_did_acrt(gt_data, dvals=None, degree=3, knots=None, **kwargs):
    """Compute the average causal response on the treated for one cohort and period.

    Regresses the outcome changes of the cell's treated units on a B-spline in
    the dose. The spline's boundary knots are the smallest and largest treated
    dose. Since a knot outside that range adds no breakpoint among the cell's
    doses, the spline keeps only the interior knots strictly between them.

    When the cell's doses can't give the spline a unique fit, the cell has no
    curve and no slope. The estimate and its influence function are then NaN.
    The extra returns hold only the cell's binary ATT, the influence function of
    that ATT, the indicator of treated units, and the numbers of distinct
    positive doses and of spline coefficients.

    Parameters
    ----------
    gt_data : pl.DataFrame
        Data of one group-time cell in long format with the columns ``id``,
        ``Y``, ``D``, ``period``, and ``name``. The ``name`` column marks the
        base period as ``"pre"`` and the cell's period as ``"post"``.
    dvals : array-like, optional
        Doses at which to evaluate the curves. If None, the curves are not
        evaluated.
    degree : int, default=3
        Degree of the B-spline basis.
    knots : array-like, optional
        Interior knots of the B-spline. If None, the spline has no interior
        knots.
    **kwargs
        Options of the panel routine that this function doesn't use.

    Returns
    -------
    AttgtResult
        NamedTuple containing:

        - **attgt**: Overall ACRT of the cell, or NaN without a unique spline fit
        - **inf_func**: Influence function of the overall ACRT on the cell's units
        - **extra_gt_returns**: Dictionary with the dose-specific ATT and ACRT,
          the cell's binary ATT and its influence function, and the pieces of the
          dose-specific influence functions
    """
    # Under a universal base, a pre-treatment cell's base period comes after the cell's own period. The
    # change in outcomes therefore runs from the named base period rather than from the previous row.
    pre_outcomes = gt_data.filter(pl.col("name") == "pre").select("id", pl.col("Y").alias(".y_pre"))
    post_data = (
        gt_data.filter(pl.col("name") == "post")
        .join(pre_outcomes, on="id", how="left")
        .with_columns((pl.col("Y") - pl.col(".y_pre")).alias("dy"))
        .sort("id")
    )
    dose = post_data["D"].to_numpy()
    dy = post_data["dy"].to_numpy()
    n_cell = len(post_data)
    dvals = np.asarray([] if dvals is None else dvals, dtype=float).ravel()

    treated_mask = dose > 0
    if not np.any(treated_mask):
        return AttgtResult(attgt=np.nan, inf_func=np.full(n_cell, np.nan), extra_gt_returns=None)

    positive_doses = dose[treated_mask]
    y_treated = dy[treated_mask]
    n_treated = len(y_treated)
    control_mean = np.mean(dy[dose == 0])

    comparison_mask = ~treated_mask
    n_comparison = int(np.sum(comparison_mask))
    att_inf_func = np.zeros(n_cell)
    att_inf_func[treated_mask] = (n_cell / n_treated) * (y_treated - np.mean(y_treated))
    if n_comparison > 0:
        att_inf_func[comparison_mask] = -(n_cell / n_comparison) * (dy[comparison_mask] - control_mean)

    boundary_knots = [np.min(positive_doses), np.max(positive_doses)]
    cell_knots = np.asarray([] if knots is None else knots, dtype=float).ravel()
    cell_knots = cell_knots[(cell_knots > boundary_knots[0]) & (cell_knots < boundary_knots[1])]
    knot_values, knot_counts = np.unique(cell_knots, return_counts=True)
    if degree > 0 and knot_counts.size > 0 and knot_counts.max() > degree:
        raise ValueError(
            f"Ties among the treated doses put {knot_counts.max()} interior knots at the dose "
            f"{knot_values[np.argmax(knot_counts)]:g}. More than degree={degree} knots at one dose break the curve "
            "there and leave its slope undefined. Lower num_knots."
        )
    distinct_doses = len(np.unique(positive_doses))
    coefficients = degree + len(cell_knots) + 1

    # Since the binary ATT needs no spline, a cell without a unique spline fit still reports it. The two counts
    # let the warning tell a cohort with one dose from a cohort whose doses nearly coincide.
    no_curve = AttgtResult(
        attgt=np.nan,
        inf_func=np.full(n_cell, np.nan),
        extra_gt_returns={
            "att_overall": np.mean(y_treated) - control_mean,
            "att_inf_func": att_inf_func,
            "treated": treated_mask,
            "distinct_doses": distinct_doses,
            "coefficients": coefficients,
        },
    )
    if distinct_doses < coefficients:
        return no_curve

    bspline_treated = BSpline(x=positive_doses, degree=degree, internal_knots=cell_knots, boundary_knots=boundary_knots)
    x_treated = to_numpy(bspline_treated.basis(complete_basis=False))
    x_treated = np.column_stack([np.ones(n_treated), x_treated])

    # Enough distinct doses can still leave the basis short of full rank if some nearly coincide or if they don't
    # spread across the knots. The cutoff is the one that least squares uses to decide the rank.
    u, singular_values, vt = np.linalg.svd(x_treated, full_matrices=False)
    tolerance = singular_values[0] * max(x_treated.shape) * np.finfo(float).eps
    if np.sum(singular_values > tolerance) < x_treated.shape[1]:
        return no_curve

    # Working from the singular values keeps the bread accurate where inverting X'X would square the condition
    # number of a basis with nearly tied doses.
    coef = vt.T @ ((u.T @ y_treated) / singular_values)
    resid = y_treated - x_treated @ coef
    bread = n_treated * (vt.T / singular_values**2) @ vt

    att_d = np.array([])
    acrt_d = np.array([])
    if dvals.size > 0:
        bspline_grid = BSpline(x=dvals, degree=degree, internal_knots=cell_knots, boundary_knots=boundary_knots)
        x_grid = to_numpy(bspline_grid.basis(complete_basis=False))
        x_grid = np.column_stack([np.ones(x_grid.shape[0]), x_grid])
        att_d = x_grid @ coef - control_mean

        x_deriv = to_numpy(bspline_grid.derivative(derivs=1, complete_basis=False))
        acrt_d = x_deriv @ coef[1:]

    att_overall = np.mean(x_treated @ coef) - control_mean

    x_deriv_overall = to_numpy(bspline_treated.derivative(derivs=1, complete_basis=False))
    acrt_overall = np.mean(x_deriv_overall @ coef[1:])

    inf_func1 = x_deriv_overall @ coef[1:] - acrt_overall

    score = resid[:, None] * x_treated
    avg_deriv = np.mean(x_deriv_overall, axis=0)
    inf_func2 = score @ bread @ np.concatenate([np.zeros(1), avg_deriv])

    # The ACRT uses treated units only. Scaling by n1/n_treated here turns the n/n1 factor that
    # compute_pte applies into n/n_treated.
    inf_func = np.zeros(n_cell)
    inf_func[treated_mask] = (n_cell / n_treated) * (inf_func1 + inf_func2)

    extra_gt_returns = {
        "att_d": att_d,
        "acrt_d": acrt_d,
        "att_overall": att_overall,
        "acrt_overall": acrt_overall,
        "dvals": dvals,
        "coef": coef,
        "bread": bread,
        "x_expanded": score,
        "score": score,
        "att_inf_func": att_inf_func,
        "treated": treated_mask,
        "knots": cell_knots,
        "boundary_knots": boundary_knots,
    }

    return AttgtResult(attgt=acrt_overall, inf_func=inf_func, extra_gt_returns=extra_gt_returns)


def cont_two_by_two_subset(
    data,
    g,
    tp,
    control_group="notyettreated",
    anticipation=0,
    base_period="varying",
    **kwargs,
):
    """Keep the units and periods that estimate one cohort and period of the dose curves.

    The subset holds period ``tp`` and its base period for the units of
    cohort ``g`` and their comparison units. Never-treated units have ``G``
    equal to 0 and are always comparison units. With
    ``control_group="notyettreated"``, so are the cohorts that are untreated,
    and not yet anticipating treatment, in both periods. Outside cohort
    ``g``, the dose is set to 0.

    Parameters
    ----------
    data : pl.DataFrame
        Panel with the columns ``G``, ``period``, ``id``, and ``D``. ``G`` and
        ``period`` hold period positions, and ``G`` is 0 for never-treated
        units.
    g : numeric
        Position of the period in which the cohort starts treatment.
    tp : numeric
        Position of the period of the cell.
    control_group : {"notyettreated", "nevertreated"}, default="notyettreated"
        Which units serve as comparison units.
    anticipation : int, default=0
        Number of periods before treatment in which units may respond to it.
    base_period : {"varying", "universal"}, default="varying"
        Whether the base period before treatment is the period just before
        ``tp`` or always the period ``g - anticipation - 1``.
    **kwargs
        Options of the panel routine that this function doesn't use.

    Returns
    -------
    dict
        Dictionary containing the cell's data:

        - **gt_data**: The subset with the period label ``name`` and the cohort's dose ``D``
        - **n1**: Number of units in the subset
        - **disidx**: Mask over the sorted unit ids that marks the units in the subset
    """
    main_base_period = g - anticipation - 1

    if base_period == "varying":
        base_period_val = tp - 1 if tp < g - anticipation else main_base_period
    else:
        base_period_val = main_base_period

    if control_group == "notyettreated":
        # A comparison unit must be untreated, and not yet anticipating treatment, in both periods of the cell.
        latest_untreated = max(tp, base_period_val) + anticipation
        unit_mask = (pl.col("G") == g) | (pl.col("G") > latest_untreated) | (pl.col("G") == 0)
    else:
        unit_mask = (pl.col("G") == g) | (pl.col("G") == 0)

    subset_data = data.filter(unit_mask)
    time_mask = (pl.col("period") == tp) | (pl.col("period") == base_period_val)
    subset_data = subset_data.filter(time_mask)
    subset_data = subset_data.with_columns(
        pl.when(pl.col("period") == tp).then(pl.lit("post")).otherwise(pl.lit("pre")).alias("name")
    )

    subset_data = subset_data.with_columns((pl.col("D") * (pl.col("G") == g).cast(pl.Float64)).alias("D"))

    n1 = subset_data["id"].n_unique()
    all_ids = np.unique(data["id"].to_numpy())
    subset_ids = subset_data["id"].unique().to_numpy()
    disidx = np.isin(all_ids, subset_ids)

    return {"gt_data": subset_data, "n1": n1, "disidx": disidx}


def _estimate_cck(cont_did_data, original_data, random_state=None):
    """Compute the CCK non-parametric estimator."""
    config = cont_did_data.config
    data = cont_did_data.data.clone()

    unique_groups = config.treated_groups
    unique_times = config.time_periods

    n_groups = len(unique_groups) + 1
    if n_groups != 2 or len(unique_times) != 2:
        raise ValueError(
            f"CCK estimator requires exactly 2 groups and 2 time periods "
            f"(found {n_groups} groups and {len(unique_times)} periods)"
        )

    data = _make_balanced_panel(data, config.idname, config.tname)
    data = _get_first_difference(data, config.idname, config.yname, config.tname)
    data = data.rename({"dy": ".dy"})

    max_t = data[config.tname].max()
    post_data = data.filter(pl.col(config.tname) == max_t)

    dose = post_data[config.dname].to_numpy()
    dy = post_data[".dy"].to_numpy()

    m0 = np.mean(dy[dose == 0])
    dy_centered = dy - m0

    positive_doses = dose[dose > 0]
    if len(positive_doses) == 0:
        raise ValueError("No treated units found")
    # npiv's smallest cubic sieve has 4 terms. Checking it here spares npiv's bootstrap when the doses can't pin
    # down any sieve.
    distinct_doses = len(np.unique(positive_doses))
    if distinct_doses == 1:
        raise ValueError(
            "Since every treated unit takes the same dose, the CCK estimator has no curve to fit. Pass "
            "dose_est_method='parametric', aggregation='eventstudy', and target_parameter='level' to compare treated "
            "and untreated units without the dose."
        )
    if distinct_doses < 4:
        raise ValueError(
            f"The smallest sieve of the CCK estimator is a cubic with 4 terms. The {distinct_doses} distinct treated "
            f"doses can't pin it down. Use dose_est_method='parametric' with a degree below {distinct_doses}."
        )

    dvals = config.dvals
    if dvals is None:
        dvals = np.linspace(positive_doses.min(), positive_doses.max(), 50)

    dvals = np.asarray(dvals).reshape(-1, 1)

    alpha = config.alpha
    cband = config.cband

    cck_res = npiv(
        y=dy_centered[dose > 0],
        x=dose[dose > 0].reshape(-1, 1),
        w=dose[dose > 0].reshape(-1, 1),
        x_grid=dvals,
        alpha=alpha,
        knots="quantiles",
        biters=999,
        j_x_degree=3,
        k_w_degree=3,
        seed=random_state,
    )

    att_d = cck_res.h
    # npiv treats the comparison mean as known. Since ATT(d) subtracts the estimated mean, the variance
    # of that mean adds to npiv's.
    n_control = int(np.sum(dose == 0))
    se_m0 = np.sqrt(np.sum((dy[dose == 0] - m0) ** 2)) / n_control
    att_d_se = np.sqrt(to_numpy(cck_res.asy_se) ** 2 + se_m0**2)

    pointwise_crit_val = stats.norm.ppf(1 - alpha / 2)
    # A band that covers every dose at once can't be narrower than the pointwise intervals.
    if cband and cck_res.cv is not None and np.isfinite(cck_res.cv):
        att_d_crit_val = max(float(cck_res.cv), pointwise_crit_val)
    else:
        att_d_crit_val = pointwise_crit_val

    acrt_d = cck_res.deriv if hasattr(cck_res, "deriv") else np.gradient(att_d, dvals.flatten())
    acrt_d_se = cck_res.deriv_asy_se if hasattr(cck_res, "deriv_asy_se") else np.full_like(acrt_d, np.nan)

    if cband and cck_res.cv_deriv is not None and np.isfinite(cck_res.cv_deriv):
        acrt_d_crit_val = max(float(cck_res.cv_deriv), pointwise_crit_val)
    else:
        acrt_d_crit_val = pointwise_crit_val

    ptep = _build_pte_params(cont_did_data, gt_type="att")

    overall_att_res = pte_default(
        yname=config.yname,
        gname=config.gname,
        tname=config.tname,
        idname=config.idname,
        data=original_data,
        d_outcome=True,
        anticipation=config.anticipation,
        base_period=config.base_period.value if hasattr(config.base_period, "value") else config.base_period,
        control_group=config.control_group.value if hasattr(config.control_group, "value") else config.control_group,
        weightsname=config.weightsname,
        boot_type="multiplier",
        biters=config.biters,
        alpha=config.alpha,
        random_state=random_state,
    )

    w_treated = dose[dose > 0]
    n_breaks = cck_res.j_x_segments + 1

    # Since npiv separates tied quantile knots by a tiny amount, the basis here does the same to match its coefficients.
    knots = np.quantile(w_treated, np.linspace(0, 1, n_breaks))
    knots = knots + np.linspace(0, 1e-10 * (np.max(w_treated) - np.min(w_treated)), n_breaks)
    spline_dosage_result = gsl_bs(
        w_treated,
        degree=cck_res.j_x_degree,
        knots=knots,
        nbreak=n_breaks,
        deriv=0,
        intercept=True,
    )
    spline_dosage = spline_dosage_result.basis
    if np.linalg.matrix_rank(to_numpy(spline_dosage)) < spline_dosage.shape[1]:
        raise ValueError(
            f"The CCK estimator selected a sieve with {spline_dosage.shape[1]} terms that the "
            f"{len(np.unique(w_treated))} distinct treated doses can't pin down. Its slope isn't identified. Use "
            "dose_est_method='parametric' with a low degree."
        )

    y_treated = dy_centered[dose > 0]
    n_treated_val = len(y_treated)

    xp = get_backend()

    spline_dosage = to_device(spline_dosage)
    y_treated_dev = to_device(y_treated)
    beta_array = to_device(cck_res.beta.flatten() if cck_res.beta.ndim > 1 else cck_res.beta)

    h_hat_w_treated = spline_dosage @ beta_array
    infl_reg = (y_treated_dev - h_hat_w_treated.flatten())[:, None] * (
        spline_dosage @ xp.linalg.pinv(spline_dosage.T @ spline_dosage / n_treated_val)
    )

    deriv_spline_basis_w = to_device(
        gsl_bs(
            w_treated,
            degree=cck_res.j_x_degree,
            knots=knots,
            nbreak=n_breaks,
            deriv=1,
            intercept=True,
        ).basis
    )

    average_spline_deriv = xp.mean(deriv_spline_basis_w, axis=0)
    deriv_at_w = (deriv_spline_basis_w @ beta_array).flatten()

    average_acr = float(xp.mean(deriv_at_w))
    infl_avg_acr = (deriv_at_w - average_acr) + infl_reg @ average_spline_deriv
    se_avg_acr = float(xp.std(infl_avg_acr) / xp.sqrt(xp.asarray(n_treated_val, dtype=float)))

    overall_att = overall_att_res.overall_att.overall_att
    overall_att_se = overall_att_res.overall_att.overall_se
    overall_att_inf_func = overall_att_res.overall_att.influence_func["overall"]

    result = DoseResult(
        dose=to_numpy(dvals.flatten() if dvals.ndim > 1 else dvals),
        overall_att=overall_att,
        overall_att_se=overall_att_se,
        overall_att_inf_func=overall_att_inf_func,
        overall_acrt=average_acr,
        overall_acrt_se=se_avg_acr,
        overall_acrt_inf_func=to_numpy(infl_avg_acr),
        att_d=to_numpy(att_d),
        att_d_se=to_numpy(att_d_se),
        att_d_crit_val=att_d_crit_val,
        att_d_inf_func=None,
        acrt_d=to_numpy(acrt_d),
        acrt_d_se=to_numpy(acrt_d_se),
        acrt_d_crit_val=acrt_d_crit_val,
        acrt_d_inf_func=None,
        pte_params=ptep,
    )

    return result
