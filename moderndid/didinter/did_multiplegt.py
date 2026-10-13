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
    alpha=0.05,
    less_conservative_se=False,
    more_granular_demeaning=False,
    keep_bidirectional_switchers=False,
    drop_missing_preswitch=False,
    boot=False,
    biters=1000,
    random_state=None,
):
    r"""Estimate event-study effects of a treatment whose level changes over time.

    Implements the difference-in-differences estimators of [3]_ for a treatment that can
    take many values, change more than once, and affect the outcome through its lags.
    Each group whose treatment changes is compared with the groups that had the same
    first-period treatment and have not changed yet. The effect at horizon :math:`\ell`
    is the average effect of having been exposed for :math:`\ell` periods to a treatment
    at least as high as the first-period one. Placebos run the same comparison over the
    periods before each group's first change.

    See the :ref:`intertemporal treatment example <example_inter_did>` for a full
    analysis of the banking deregulation data.

    Parameters
    ----------
    data : DataFrame
        Panel data in long format. Accepts any object implementing the Arrow
        PyCapsule Interface (``__arrow_c_stream__``), including polars, pandas,
        pyarrow Table, and cudf DataFrames.
    yname : str
        The name of the outcome column.
    tname : str
        The name of the time period column. Periods enter by their rank.
    idname : str
        The name of the group identifier column.
    dname : str
        The name of the treatment column. The treatment must be non-negative.
    cluster : str, optional
        The name of the column to cluster standard errors by. Each group must belong to
        one cluster. Rows whose cluster is missing are dropped with a warning.
    weightsname : str, optional
        The name of the sampling weights column.
    xformla : str, default="~1"
        Formula for time-varying covariates, such as ``"~ X1 + X2"``.
    effects : int, default=1
        Number of horizons after each group's first change. A request beyond the last
        horizon that any switcher reaches is cut back with a warning.
    placebo : int, default=0
        Number of placebo horizons before each group's first change. A request beyond
        what the data allow or beyond ``effects`` is cut back with a warning.
    normalized : bool, default=False
        Whether to divide each effect by the average cumulative change in treatment.
    effects_equal : bool or str or tuple, default=False
        Whether to test that the effects are equal. True or ``"all"`` tests every horizon.
        ``"lb, ub"`` or ``(lb, ub)`` tests the horizons from ``lb`` to ``ub``.
    predict_het : tuple[list[str], list[int]], optional
        Time-invariant covariates and horizons for regressions that test whether the
        effects vary with the covariates. Horizons ``[-1]`` select every horizon. The
        regressions pool both switcher directions and also cover the placebos at the
        listed horizons unless ``trends_lin=True``.
    predict_het_hc2bm : bool, default=False
        Whether the ``predict_het`` regressions use HC2 standard errors clustered by
        ``cluster`` [1]_. Requires ``predict_het`` and has no effect without ``cluster``.
    switchers : {"", "in", "out"}, default=""
        Which switchers to estimate effects for. ``"in"`` keeps first rises in treatment
        and ``"out"`` first falls. ``""`` pools both.
    only_never_switchers : bool, default=False
        Whether only the groups whose treatment never changes serve as controls.
    same_switchers : bool, default=False
        Whether every horizon uses only the switchers that reach all requested effects.
    same_switchers_pl : bool, default=False
        Whether the placebos use only the switchers that reach all requested placebos.
        Requires ``same_switchers=True``.
    trends_lin : bool, default=False
        Whether to add group-specific linear trends. The effect at horizon :math:`\ell`
        then sums the estimates of horizons 1 to :math:`\ell` on the switchers that reach
        horizon :math:`\ell`. It warns and leaves out ``ate``.
    trends_nonparam : list[str], optional
        Time-invariant columns whose values split the groups into sets with their own
        trends. Each switcher is compared only with groups in its own set.
    continuous : int, default=0
        Degree of a polynomial in a continuous baseline treatment. A positive degree
        compares switchers with all not-yet-switched groups and warns unless
        ``boot=True``.
    alpha : float, default=0.05
        The significance level of the confidence intervals.
    less_conservative_se : bool, default=False
        Whether the effect standard errors demean each switcher's outcome change among
        the switchers with the same treatment path up to the horizon.
    more_granular_demeaning : bool, default=False
        Alias that sets ``less_conservative_se=True``.
    keep_bidirectional_switchers : bool, default=False
        Whether to keep a group's periods after its treatment has been both above and
        below its first-period value. Keeping them warns.
    drop_missing_preswitch : bool, default=False
        Whether to drop observations whose treatment is missing before the first switch.
    boot : bool, default=False
        Whether to bootstrap the standard errors by resampling clusters, or groups
        without ``cluster``. The chi-squared tests keep the analytical variance.
    biters : int, default=1000
        The number of bootstrap draws. Only applicable if ``boot=True``.
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
        - **alpha**: Significance level of the intervals
        - **effects_equal_test**: Chi-squared test that the effects are equal, if requested
        - **placebo_joint_test**: Chi-squared test that all placebos are zero
        - **influence_effects**: Influence functions of the effects
        - **influence_placebos**: Influence functions of the placebos
        - **heterogeneity**: Heterogeneity regressions, if ``predict_het`` is set
        - **estimation_params**: Dictionary of the estimation settings

    See Also
    --------
    att_gt : Group-time ATT for binary, staggered adoption designs.
    cont_did : Continuous treatment DID with dose-response estimation.

    Notes
    -----
    Let :math:`F_g` be the first period in which group :math:`g`'s treatment changes
    and :math:`D_{g,1}` its first-period treatment. The actual-versus-status-quo effect

    .. math::

        \delta_{g,\ell} = \mathbb{E}\left[Y_{g,F_g-1+\ell} -
        Y_{g,F_g-1+\ell}(D_{g,1}, \ldots, D_{g,1}) \mid \boldsymbol{D}\right]

    compares the group's outcome at :math:`F_g - 1 + \ell` with the outcome it would
    have had if its treatment had stayed at :math:`D_{g,1}`. Under no anticipation and
    parallel trends among groups with the same first-period treatment, it is estimated
    by

    .. math::

        \text{DID}_{g,\ell} = Y_{g,F_g-1+\ell} - Y_{g,F_g-1} -
        \frac{1}{N_{F_g-1+\ell}^g} \sum_{g': D_{g',1}=D_{g,1}, F_{g'}>F_g-1+\ell}
        \left(Y_{g',F_g-1+\ell} - Y_{g',F_g-1}\right),

    where :math:`N_{F_g-1+\ell}^g` counts the groups in the sum. The effect at horizon
    :math:`\ell` averages :math:`S_g \text{DID}_{g,\ell}` over the :math:`N_\ell`
    switchers observed at that horizon. Here :math:`S_g` is 1 when the first change
    raises the treatment and -1 when it lowers it. With ``normalized=True``, that
    average is divided by the average of
    :math:`\left|\sum_{k=1}^{\ell} (D_{g,F_g-1+k} - D_{g,1})\right|` over the same
    switchers. Over the requested horizons :math:`\ell = 1, \ldots, L`, the average
    total effect in ``ate`` is

    .. math::

        \hat{\delta} = \frac{\sum_{\ell=1}^{L} \sum_{g} S_g \text{DID}_{g,\ell}}
        {\sum_{\ell=1}^{L} \sum_{g} S_g (D_{g,F_g-1+\ell} - D_{g,1})},

    where each inner sum runs over the switchers observed at horizon :math:`\ell`. It is
    a total effect per unit of treatment. Comparing it with the cost of a unit of
    treatment gives a cost-benefit analysis.

    The :ref:`intertemporal treatment background <background-didinter>` derives these
    quantities. With a binary treatment that groups adopt at most once from a common
    first-period value, the effects equal the event-study estimates of
    :func:`~moderndid.att_gt` [2]_ with not-yet-treated controls, a universal base
    period, and no covariates.

    The estimator adds its own columns to the data. A column that an argument names
    can't start with a dot or be one of ``F_g``, ``S_g``, ``L_g``, ``T_g``, ``d_sq``,
    ``d_sq_int``, ``d_fg``, ``weight_gt``, ``first_obs_by_gp``, or ``t_max_by_group``.

    References
    ----------

    .. [1] Bell, R., & McCaffrey, D. (2002). Bias Reduction in Standard
           Errors for Linear Regression with Multi-Stage Samples.
           *Survey Methodology*, 28(2), 169-181.

    .. [2] Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-Differences
           with Multiple Time Periods. *Journal of Econometrics*, 225(2),
           200-230. https://doi.org/10.1016/j.jeconom.2020.12.001

    .. [3] de Chaisemartin, C., & D'Haultfoeuille, X. (2024).
           Difference-in-Differences Estimators of Intertemporal Treatment
           Effects. *Review of Economics and Statistics*, 106(6), 1723-1736.
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
    if not 0 < alpha < 1:
        raise ValueError(f"alpha={alpha} is not valid. Must be between 0 and 1 (exclusive).")
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
        alpha=alpha,
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
