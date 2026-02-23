"""Dynamic ATT estimation for intertemporal treatment effects."""

import warnings

from moderndid.core.preprocess import PreprocessDataBuilder
from moderndid.core.preprocess.config import DIDInterConfig

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
    r"""Estimate intertemporal treatment effects with non-binary, non-absorbing treatments.

    Implements difference-in-differences estimation for settings where treatment
    may be non-binary, non-absorbing (time-varying), and where lagged treatments
    may affect the outcome, following [1]_. Unlike standard DID which assumes
    binary absorbing treatment, this estimator handles complex treatment patterns
    where units can experience treatment increases, decreases, or multiple changes
    over time.

    Let :math:`F_g` denote the first period when group :math:`g`'s treatment changes,
    and let :math:`D_{g,1}` be its baseline (period-1) treatment. The key parameter
    of interest is the actual-versus-status-quo (AVSQ) effect

    .. math::

        \delta_{g,\ell} = \mathbb{E}\left[Y_{g,F_g-1+\ell} -
        Y_{g,F_g-1+\ell}(D_{g,1}, \ldots, D_{g,1}) \mid \boldsymbol{D}\right]

    which measures the expected difference between group :math:`g`'s actual outcome
    at :math:`F_g - 1 + \ell` and the counterfactual "status quo" outcome it would
    have obtained if its treatment had remained equal to its period-one value.

    The estimator computes

    .. math::

        \text{DID}_{g,\ell} = Y_{g,F_g-1+\ell} - Y_{g,F_g-1} -
        \frac{1}{N_{F_g-1+\ell}^g} \sum_{g': D_{g',1}=D_{g,1}, F_{g'}>F_g-1+\ell}
        \left(Y_{g',F_g-1+\ell} - Y_{g',F_g-1}\right)

    comparing the outcome evolution of switchers to that of groups with the same
    baseline treatment that have not yet switched. These are aggregated into
    event-study effects :math:`\delta_\ell`, the average effect of having been
    exposed to a weakly higher treatment dose for :math:`\ell` periods.

    When ``normalized=True``, the estimator computes :math:`\delta_\ell^n`,
    which normalizes by the cumulative treatment change and can be interpreted
    as a weighted average of the effects of the current treatment and its
    :math:`\ell - 1` first lags on the outcome.

    Parameters
    ----------
    data : DataFrame
        Panel data in long format. Accepts any object implementing the Arrow
        PyCapsule Interface (``__arrow_c_stream__``), including polars, pandas,
        pyarrow Table, and cudf DataFrames.
    yname : str
        Name of the outcome variable.
    tname : str
        Name of the time period variable.
    idname : str
        Name of the unit identifier variable.
    dname : str
        Name of the treatment variable. Can be binary or continuous, and
        can vary over time for the same unit (non-absorbing). Must be
        non-negative.
    cluster : str, optional
        Name of the cluster variable for clustered standard errors.
        If None, standard errors are computed using the influence function
        at the unit level.
    weightsname : str, optional
        Name of the sampling weights column. If None, all observations
        have equal weight.
    xformla : str, default="~1"
        A formula for the covariates to include in the model.
        Should be of the form "~ X1 + X2" (intercept is always included).
        Use "~1" for no covariates.
    effects : int, default=1
        Number of post-treatment horizons to estimate (1, 2, ..., effects).
        :math:`\delta_\ell` estimates the effect of :math:`\ell` periods of
        exposure to changed treatment.
    placebo : int, default=0
        Number of pre-treatment horizons to estimate for placebo tests
        (-1, -2, ..., -placebo). These compare outcome trends of switchers
        and non-switchers before switching occurs, testing the parallel
        trends assumption.
    normalized : bool, default=False
        If True, compute normalized effects :math:`\delta_\ell^n` by dividing
        by the average cumulative treatment change. The normalized effect is
        a weighted average of the effects of the current treatment and its
        lags, useful when treatment magnitudes vary across units.
    effects_equal : bool, default=False
        If True, perform a joint test of whether all treatment effects
        :math:`\delta_1 = \delta_2 = \cdots = \delta_L` are equal. Returns
        a chi-squared test statistic and p-value.
    predict_het : tuple[list[str], list[int]], optional
        Analyze heterogeneous effects by covariates. A tuple of (covariates, horizons)
        where covariates is a list of time-invariant covariate names and horizons
        is a list of effect horizons to analyze (use [-1] for all horizons).
        Runs WLS regressions to test whether effects vary by covariates.
    switchers : {"", "in", "out"}, default=""
        Which switchers to include in estimation:

        - ``""``: All switchers (treatment increases and decreases)
        - ``"in"``: Only treatment increases (:math:`D_{g,F_g} > D_{g,1}`)
        - ``"out"``: Only treatment decreases (:math:`D_{g,F_g} < D_{g,1}`)
    only_never_switchers : bool, default=False
        If True, use only never-switchers as controls. If False (default),
        also use not-yet-switchers as controls.
    same_switchers : bool, default=False
        If True, use the same set of switchers across all effect horizons.
        This ensures comparability across horizons but may reduce sample size.
    same_switchers_pl : bool, default=False
        If True, use the same set of switchers across all placebo horizons.
    trends_lin : bool, default=False
        If True, include unit-specific linear time trends in the estimation.
    trends_nonparam : list[str], optional
        Variables for non-parametric group-specific trends.
    continuous : int, default=0
        Polynomial degree for continuous treatment. If > 0, treatment is
        modeled as continuous with polynomial terms of the specified degree.
    ci_level : float, default=95.0
        Confidence level for confidence intervals (e.g., 95.0 for 95% CI).
    less_conservative_se : bool, default=False
        If True, use less conservative standard error estimation with
        degrees-of-freedom adjustment based on the number of clusters or
        switchers.
    keep_bidirectional_switchers : bool, default=False
        If True, keep units that experience both treatment increases AND
        decreases over time. By default, these units are dropped because
        their :math:`\delta_{g,\ell}` may not satisfy the no-sign-reversal
        property.
    drop_missing_preswitch : bool, default=False
        If True, drop observations where treatment is missing before the
        first switch time.
    boot : bool, default=False
        If True, compute standard errors using the multiplier bootstrap
        instead of asymptotic influence function-based inference. The
        bootstrap resamples at the cluster level when ``cluster`` is
        specified.
    biters : int, default=1000
        Number of bootstrap iterations when ``boot=True``.
    random_state : int, Generator, optional
        Random seed for reproducibility of bootstrap.
    n_partitions : int, optional
        Number of partitions for distributed computation when ``data`` is a
        Dask or Spark DataFrame. If ``None``, defaults to the framework's
        default parallelism.

    Returns
    -------
    DIDInterResult
        Result object containing:

        - **effects**: EffectsResult with treatment effects at each horizon,
          including point estimates, standard errors, confidence intervals,
          and sample sizes
        - **placebos**: PlacebosResult with placebo effects (if placebo > 0)
        - **ate**: ATEResult with the average total effect :math:`\delta`,
          which can be used for cost-benefit analysis
        - **n_units**: Total number of units in the sample
        - **n_switchers**: Number of switching units
        - **n_never_switchers**: Number of never-switching units
        - **ci_level**: Confidence level used for intervals
        - **effects_equal_test**: Chi-squared test for equal effects (if requested)
        - **placebo_joint_test**: Joint test that all placebo effects are zero
        - **influence_effects**: Influence functions for effects (for custom inference)
        - **influence_placebos**: Influence functions for placebos
        - **heterogeneity**: Heterogeneous effects analysis (if predict_het specified)
        - **estimation_params**: Dictionary of estimation parameters used

    Examples
    --------
    Estimate intertemporal treatment effects using the Favara and Imbs (2015)
    banking deregulation data, where treatment (interstate branching) is
    non-binary and potentially non-absorbing.

    .. ipython::
        :okwarning:

        In [1]: import moderndid as md
           ...: df = md.load_favara_imbs()
           ...: df.head()

    Estimate effects at multiple horizons with placebo tests.

    .. ipython::
        :okwarning:

        In [2]: result = md.did_multiplegt(
           ...:     data=df,
           ...:     yname="Dl_vloans_b",
           ...:     idname="county",
           ...:     tname="year",
           ...:     dname="inter_bra",
           ...:     effects=8,
           ...:     placebo=3,
           ...:     cluster="state_n",
           ...:     normalized=True,
           ...:     same_switchers=True,
           ...:     effects_equal=True,
           ...: )
           ...: result

    Notes
    -----
    Identification relies on a parallel trends assumption for groups with the
    same baseline treatment. If two groups have the same period-one treatment,
    they have the same expected evolution of their status-quo outcome. This is
    weaker than standard parallel trends across all groups, which would rule out
    both dynamic treatment effects and time-varying effects.

    With binary staggered treatment and uniform baseline, this is equivalent
    to the :func:`att_gt` event-study estimator. With varying baseline treatments,
    the estimators differ because this method compares switchers only to non-switchers
    with the same baseline, preserving validity under a conditional parallel
    trends assumption that allows for lagged and time-varying effects.

    By default, units that experience both treatment increases and decreases
    are dropped (``keep_bidirectional_switchers=False``) because their
    :math:`\delta_{g,\ell}` can be written as a linear combination with negative
    weights of effects of different treatment lags, potentially violating the
    no-sign-reversal property.

    The ATE parameter :math:`\delta` measures the average total effect per
    unit of treatment, where total effect includes both contemporaneous
    and lagged effects. It can be compared to the average treatment cost
    to assess whether treatment changes were beneficial.

    See Also
    --------
    att_gt : Group-time ATT for binary, staggered adoption designs.
    cont_did : Continuous treatment DID with dose-response estimation.

    References
    ----------

    .. [1] Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-Differences
           with Multiple Time Periods. *Journal of Econometrics*, 225(2),
           200-230. https://doi.org/10.1016/j.jeconom.2020.12.001

    .. [2] de Chaisemartin, C., & D'Haultfoeuille, X. (2024). Difference-in-
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
    if trends_nonparam is not None and (
        not isinstance(trends_nonparam, list) or not all(isinstance(v, str) for v in trends_nonparam)
    ):
        raise ValueError("trends_nonparam must be a list of variable name strings.")

    from moderndid.dask._utils import is_dask_collection

    if is_dask_collection(data):
        from moderndid.dask._didinter import dask_did_multiplegt

        return dask_did_multiplegt(
            data,
            yname,
            tname,
            idname,
            dname,
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
            n_partitions=n_partitions,
        )

    from moderndid.spark._utils import is_spark_dataframe

    if is_spark_dataframe(data):
        from moderndid.spark._didinter import spark_did_multiplegt

        return spark_did_multiplegt(
            data,
            yname,
            tname,
            idname,
            dname,
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
            n_partitions=n_partitions,
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

    return compute_did_multiplegt(preprocessed)
