"""Dynamic ATT estimation for intertemporal treatment effects."""

import warnings

from moderndid.core.preprocess import PreprocessDataBuilder
from moderndid.core.preprocess.config import DIDInterConfig
from moderndid.core.preprocess.validators import check_columns

from .compute_did_multiplegt import compute_did_multiplegt


def did_multiplegt(
    data,
    yname,
    tname,
    idname,
    dname,
    cluster=None,
    weightsname=None,
    xformla="~1",
    effects=1,
    placebo=0,
    normalized=False,
    effects_equal=False,
    predict_het=None,
    predict_het_hc2bm=False,
    switchers="",
    only_never_switchers=False,
    same_switchers=False,
    same_switchers_pl=False,
    trends_lin=False,
    trends_nonparam=None,
    continuous=0,
    ci_level=95.0,
    less_conservative_se=False,
    more_granular_demeaning=False,
    keep_bidirectional_switchers=False,
    drop_missing_preswitch=False,
    boot=False,
    biters=1000,
    random_state=None,
):
    r"""Estimate event-study effects of a treatment whose level changes over time.

    Implements the difference-in-differences estimators of [3]_ for treatments that
    can take many values, change more than once, and affect the outcome through their
    lags. Each group whose treatment changes is compared with the groups that had the
    same treatment in the first period and have not changed yet. Restricting the
    comparison to the same starting treatment keeps it valid when past treatments
    still move the outcome.

    The effect at horizon :math:`\ell` is the average effect of having been exposed
    for :math:`\ell` periods to a treatment at least as high as the first-period one.
    A group whose first change lowered its treatment enters with a minus sign.
    Placebos run the same comparison over the periods before each group's first
    change.

    With ``normalized=True``, each effect is divided by the average cumulative change
    in treatment up to its horizon. It then becomes a weighted average of the effects
    of the current treatment and its lags. The average total effect in ``ate`` adds up
    the effects over the estimated horizons and divides them by the treatment changes
    over the same horizons.

    The estimator adds its own columns next to the ones the call names. A column that
    an argument names can't start with a dot or use one of the names ``F_g``, ``S_g``,
    ``L_g``, ``T_g``, ``d_sq``, ``d_sq_int``, ``d_fg``, ``weight_gt``,
    ``first_obs_by_gp``, or ``t_max_by_group``.

    See the :ref:`intertemporal treatment example <example_inter_did>` for a full
    analysis of the banking deregulation data.

    Parameters
    ----------
    data : DataFrame
        Panel data in long format. Accepts any object implementing the Arrow
        PyCapsule Interface (``__arrow_c_stream__``), including polars, pandas,
        pyarrow Table, and cudf DataFrames.
    yname : str
        Name of the outcome column.
    tname : str
        Name of the time period column. Periods enter by their rank and need not be
        evenly spaced.
    idname : str
        Name of the group identifier column.
    dname : str
        Name of the treatment column. The treatment must be non-negative and can be
        binary or take many values that change more than once.
    cluster : str, optional
        Name of the column to cluster standard errors by. Each group must belong
        to a single cluster. Rows whose cluster is missing are dropped with a
        warning. If None, groups are treated as independent.
    weightsname : str, optional
        Name of the column of sampling weights. If None, every observation has the
        same weight.
    xformla : str, default="~1"
        Formula for time-varying covariates, such as ``"~ X1 + X2"``. Their changes
        are netted out of each group's outcome changes. The coefficients come from
        the not-yet-switched groups that share its first-period treatment. ``"~1"``
        uses no covariates.
    effects : int, default=1
        Number of horizons to estimate after each group's first change. A request
        beyond the last horizon that any switcher reaches is cut back to that horizon
        with a warning.
    placebo : int, default=0
        Number of placebo horizons to estimate before each group's first change. A
        request beyond what the data allow or beyond ``effects`` is cut back with a
        warning.
    normalized : bool, default=False
        Whether to divide each effect by the average cumulative change in treatment
        up to its horizon.
    effects_equal : bool or str or tuple, default=False
        Whether to test that the effects are equal across horizons. True or
        ``"all"`` tests every horizon. A string ``"lb, ub"`` or a tuple
        ``(lb, ub)`` tests the horizons from ``lb`` to ``ub``.
    predict_het : tuple[list[str], list[int]], optional
        Time-invariant covariates and horizons for regressions that test whether
        the effects vary with those covariates. Passing ``[-1]`` as the horizons
        selects every estimated horizon. When placebos are estimated, each listed
        horizon up to the number of placebos also gets a placebo regression unless
        ``trends_lin=True``. Even when ``switchers`` is ``"in"`` or ``"out"``, the
        regressions pool the switchers of both directions.
    predict_het_hc2bm : bool, default=False
        Whether the ``predict_het`` regressions use HC2 standard errors clustered by
        ``cluster`` [1]_. Requires ``predict_het`` and has no effect without
        ``cluster``.
    switchers : {"", "in", "out"}, default=""
        Which switchers to estimate effects for. ``""`` pools the groups whose
        treatment first rises with the groups whose treatment first falls. The falls
        enter with a minus sign. ``"in"`` keeps only the rises and ``"out"`` only the
        falls. Groups that switch the other way serve as controls until they switch.
    only_never_switchers : bool, default=False
        Whether to use only the groups whose treatment never changes as controls.
        If False, groups that have not switched yet also serve as controls.
    same_switchers : bool, default=False
        If True, every horizon uses only the switchers that reach all the requested
        effects. A switcher reaches an effect when its outcome change and a
        not-yet-switched group with the same baseline treatment are observed at that
        horizon. This ensures comparability across horizons but may reduce sample size.
    same_switchers_pl : bool, default=False
        If True, the placebos also use only the switchers that reach all the
        requested placebos. Requires ``same_switchers=True``.
    trends_lin : bool, default=False
        If True, include group-specific linear time trends in the estimation. The
        effect at horizon :math:`\ell` then sums the estimates of horizons 1 to
        :math:`\ell` on the switchers that reach horizon :math:`\ell`.
    trends_nonparam : list[str], optional
        Names of time-invariant columns whose values split the groups into sets with
        their own trends. Each switcher is then compared only with groups in its own
        set.
    continuous : int, default=0
        Degree of a polynomial in the baseline treatment, for baseline treatments
        that are continuous. A positive degree compares each switcher with all
        not-yet-switched groups and lets the outcome evolution of every period
        depend on that polynomial.
    ci_level : float, default=95.0
        Confidence level of the intervals in percent, such as 95.0.
    less_conservative_se : bool, default=False
        Whether the effect standard errors demean each switcher's outcome change
        among the switchers that share its treatment path up to the horizon, instead
        of among those that share only its first-period treatment and switch period.
        A switcher alone on its path falls back to the coarser groups. Placebo
        standard errors don't change.
    more_granular_demeaning : bool, default=False
        Alias that sets ``less_conservative_se=True``.
    keep_bidirectional_switchers : bool, default=False
        Whether to keep a group's periods after its treatment has been both above
        and below its first-period value. By default these periods are dropped.
    drop_missing_preswitch : bool, default=False
        Whether to drop the observations whose treatment is missing before the
        group's first switch.
    boot : bool, default=False
        Whether to replace the analytical standard errors of the effects, the
        placebos, and ``ate`` with bootstrap ones. Each draw resamples clusters with
        replacement and reruns the estimation. Groups take the place of clusters
        when ``cluster`` is None. The joint placebo test and the test of equal
        effects keep the analytical variance.
    biters : int, default=1000
        Number of bootstrap draws when ``boot=True``.
    random_state : int, Generator, optional
        Seed or generator for the bootstrap draws.

    Returns
    -------
    DIDInterResult
        Result object containing:

        - **effects**: EffectsResult with the estimate, standard error, confidence
          interval, switchers, and sample size at each horizon
        - **placebos**: PlacebosResult with the same fields for each placebo, or None
        - **ate**: ATEResult with the average total effect, or None when
          ``trends_lin=True``
        - **n_units**: Number of groups in the sample
        - **n_switchers**: Number of groups that switch in the requested directions. Groups
          that no effect uses also count.
        - **n_never_switchers**: Number of groups whose treatment never changes
        - **ci_level**: Confidence level of the intervals
        - **effects_equal_test**: Chi-squared test that the effects are equal, if requested
        - **placebo_joint_test**: Chi-squared test that all placebos are zero
        - **influence_effects**: Influence functions of the effects
        - **influence_placebos**: Influence functions of the placebos
        - **heterogeneity**: Heterogeneity regressions, if ``predict_het`` is set
        - **estimation_params**: Dictionary of the estimation settings

    Notes
    -----
    Let :math:`F_g` be the first period in which group :math:`g`'s treatment changes
    and :math:`D_{g,1}` its first-period treatment. The actual-versus-status-quo
    effect

    .. math::

        \delta_{g,\ell} = \mathbb{E}\left[Y_{g,F_g-1+\ell} -
        Y_{g,F_g-1+\ell}(D_{g,1}, \ldots, D_{g,1}) \mid \boldsymbol{D}\right]

    compares the group's outcome at :math:`F_g - 1 + \ell` with the outcome it would
    have had if its treatment had stayed at :math:`D_{g,1}`. Under no anticipation
    and parallel trends among groups with the same first-period treatment, it is
    estimated by

    .. math::

        \text{DID}_{g,\ell} = Y_{g,F_g-1+\ell} - Y_{g,F_g-1} -
        \frac{1}{N_{F_g-1+\ell}^g} \sum_{g': D_{g',1}=D_{g,1}, F_{g'}>F_g-1+\ell}
        \left(Y_{g',F_g-1+\ell} - Y_{g',F_g-1}\right),

    where :math:`N_{F_g-1+\ell}^g` counts the groups in the sum. The effect at horizon
    :math:`\ell` averages :math:`S_g \text{DID}_{g,\ell}` over the :math:`N_\ell`
    switchers observed at that horizon. Here :math:`S_g` is 1 when the first change
    raises the treatment and -1 when it lowers it. The normalized effect
    divides that average by the average of
    :math:`\left|\sum_{k=1}^{\ell} (D_{g,F_g-1+k} - D_{g,1})\right|` over the same
    switchers. Over the requested horizons :math:`\ell = 1, \ldots, L`, the average
    total effect is

    .. math::

        \hat{\delta} = \frac{\sum_{\ell=1}^{L} \sum_{g} S_g \text{DID}_{g,\ell}}
        {\sum_{\ell=1}^{L} \sum_{g} S_g (D_{g,F_g-1+\ell} - D_{g,1})},

    where each inner sum runs over the switchers observed at horizon :math:`\ell`.
    It is a total effect per unit of treatment. Each change in treatment contributes
    its effect in the period it happens and in every later period up to horizon
    :math:`L`. Comparing it with the cost of a unit of treatment gives a cost-benefit
    analysis.

    With a binary treatment, comparing groups that start treated with groups that
    start untreated would require the effect of being treated for :math:`t` periods
    to equal that of being treated for :math:`t - 1` periods. That rules out effects
    of lagged treatments and effects that vary over time. Every comparison therefore
    keeps to one first-period treatment.

    With a binary treatment that groups adopt at most once from a common
    first-period value, the effects equal the event-study estimates of
    :func:`~moderndid.att_gt` [2]_ with not-yet-treated controls, a universal base
    period, and no covariates. The placebos agree only at the first horizon, since
    each placebo keeps the comparison groups of the effect at the same horizon.

    See Also
    --------
    att_gt : Group-time ATT for binary, staggered adoption designs.
    cont_did : Continuous treatment DID with dose-response estimation.

    References
    ----------

    .. [1] Bell, R., & McCaffrey, D. (2002). Bias Reduction in Standard
           Errors for Linear Regression with Multi-Stage Samples.
           *Survey Methodology*, 28(2), 169-181.

    .. [2] Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-Differences
           with Multiple Time Periods. *Journal of Econometrics*, 225(2),
           200-230. https://doi.org/10.1016/j.jeconom.2020.12.001

    .. [3] de Chaisemartin, C., & D'Haultfoeuille, X. (2024). Difference-in-
           Differences Estimators of Intertemporal Treatment Effects.
           *Review of Economics and Statistics*, 106(6), 1723-1736.
           https://doi.org/10.1162/rest_a_01414
    """
    if continuous > 0 and not boot:
        warnings.warn(
            "When continuous > 0, variance estimators are not backed by proven asymptotic "
            "normality. Bootstrap inference (boot=True) is recommended.",
            UserWarning,
        )

    if trends_lin:
        warnings.warn(
            "When trends_lin=True, the average total effect (ATE) is not computed.",
            UserWarning,
        )

    if keep_bidirectional_switchers:
        warnings.warn(
            "Keeping bidirectional switchers (units with both treatment increases and decreases) "
            "may violate the no-sign-reversal property. The default behavior of dropping these "
            "units is recommended.",
            UserWarning,
        )

    if not isinstance(effects, int) or effects < 1:
        raise ValueError(f"effects={effects} is not valid. Must be a positive integer.")
    if not isinstance(placebo, int) or placebo < 0:
        raise ValueError(f"placebo={placebo} is not valid. Must be a non-negative integer.")
    if not isinstance(continuous, int) or continuous < 0:
        raise ValueError(f"continuous={continuous} is not valid. Must be a non-negative integer.")
    if switchers not in ("", "in", "out"):
        raise ValueError(f"switchers='{switchers}' is not valid. Must be '', 'in', or 'out'.")
    if not 0 < ci_level < 100:
        raise ValueError(f"ci_level={ci_level} is not valid. Must be between 0 and 100 (exclusive).")
    if not isinstance(biters, int) or biters < 1:
        raise ValueError(f"biters={biters} is not valid. Must be a positive integer.")
    if predict_het is not None:
        if not isinstance(predict_het, tuple) or len(predict_het) != 2:
            raise ValueError("predict_het must be a tuple of (covariate_names, horizons).")
        covs, horizons = predict_het
        if not isinstance(covs, list) or not all(isinstance(c, str) for c in covs):
            raise ValueError("predict_het[0] must be a list of covariate name strings.")
        if not isinstance(horizons, list) or not all(isinstance(h, int) for h in horizons):
            raise ValueError("predict_het[1] must be a list of integer horizons.")
    if predict_het_hc2bm and predict_het is None:
        raise ValueError("predict_het_hc2bm=True requires predict_het to be specified.")
    if same_switchers_pl and not same_switchers:
        raise ValueError("same_switchers_pl=True requires same_switchers=True.")
    if more_granular_demeaning:
        less_conservative_se = True

    effects_equal_lb = None
    effects_equal_ub = None
    if isinstance(effects_equal, str) and effects_equal != "all":
        parts = [p.strip() for p in effects_equal.split(",")]
        if len(parts) != 2:
            raise ValueError(
                f"effects_equal='{effects_equal}' is not valid. Use True, 'all', 'lb,ub', or a (lb, ub) tuple."
            )
        effects_equal_lb, effects_equal_ub = int(parts[0]), int(parts[1])
        effects_equal = True
    elif isinstance(effects_equal, tuple):
        if len(effects_equal) != 2:
            raise ValueError("effects_equal tuple must have exactly 2 elements (lb, ub).")
        effects_equal_lb, effects_equal_ub = int(effects_equal[0]), int(effects_equal[1])
        effects_equal = True
    elif effects_equal == "all":
        effects_equal = True

    if effects_equal_lb is not None and effects_equal_ub is not None:
        if effects_equal_lb < 1:
            raise ValueError(f"effects_equal lower bound must be >= 1, got {effects_equal_lb}.")
        if effects_equal_ub <= effects_equal_lb:
            raise ValueError(
                f"effects_equal upper bound ({effects_equal_ub}) must be greater than lower bound ({effects_equal_lb})."
            )

    if trends_nonparam is not None and (
        not isinstance(trends_nonparam, list) or not all(isinstance(v, str) for v in trends_nonparam)
    ):
        raise ValueError("trends_nonparam must be a list of variable name strings.")
    check_columns(
        data,
        yname=yname,
        tname=tname,
        idname=idname,
        dname=dname,
        cluster=cluster,
        weightsname=weightsname,
        xformla=xformla,
        trends_nonparam=trends_nonparam,
        predict_het=None if predict_het is None else predict_het[0],
    )

    config = DIDInterConfig(
        yname=yname,
        tname=tname,
        gname=idname,
        dname=dname,
        cluster=cluster,
        weightsname=weightsname,
        xformla=xformla,
        trends_nonparam=trends_nonparam,
        effects=effects,
        placebo=placebo,
        normalized=normalized,
        effects_equal=effects_equal,
        predict_het=predict_het,
        predict_het_hc2bm=predict_het_hc2bm,
        more_granular_demeaning=more_granular_demeaning,
        effects_equal_lb=effects_equal_lb,
        effects_equal_ub=effects_equal_ub,
        switchers=switchers,
        only_never_switchers=only_never_switchers,
        same_switchers=same_switchers,
        same_switchers_pl=same_switchers_pl,
        trends_lin=trends_lin,
        continuous=continuous,
        ci_level=ci_level,
        less_conservative_se=less_conservative_se,
        keep_bidirectional_switchers=keep_bidirectional_switchers,
        drop_missing_preswitch=drop_missing_preswitch,
        boot=boot,
        biters=biters,
        random_state=random_state,
    )

    builder = PreprocessDataBuilder()
    preprocessed = builder.with_data(data).with_config(config).validate().transform().build()

    return compute_did_multiplegt(preprocessed, data)
