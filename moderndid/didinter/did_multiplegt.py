"""Dynamic ATT estimation for heterogeneous and dynamic treatment effects."""

from moderndid.core.preprocess import PreprocessDataBuilder
from moderndid.core.preprocess.config import DIDInterConfig

from .compute_did_multiplegt import compute_effects


def did_multiplegt(
    data,
    yname,
    tname,
    idname,
    dname,
    cluster=None,
    weightsname=None,
    controls=None,
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
):
    r"""Estimate heterogeneous and dynamic treatment effects.

    Parameters
    ----------
    data : pd.DataFrame | pl.DataFrame
        Panel data in long format.
    yname : str
        Name of the outcome variable.
    tname : str
        Name of the time period variable.
    idname : str
        Name of the unit identifier variable.
    dname : str
        Name of the treatment variable. Can be binary or continuous.
        Treatment can vary over time for the same unit.
    cluster : str, optional
        Name of the cluster variable for clustered standard errors.
    weightsname : str, optional
        Name of the sampling weights column.
    controls : list[str], optional
        List of control variable names to include in the estimation.
    effects : int, default=1
        Number of post-treatment horizons to estimate (1, 2, ..., effects).
    placebo : int, default=0
        Number of pre-treatment horizons to estimate for placebo tests.
    normalized : bool, default=False
        If True, normalize effects by the average treatment change.
    effects_equal : bool, default=False
        If True, test whether all treatment effects are equal.
    predict_het : tuple[list[str], list[int]], optional
        Analyze heterogeneous effects by covariates. A tuple of (covariates, horizons)
        where covariates is a list of time-invariant covariate names and horizons
        is a list of effect horizons to analyze (use [-1] for all horizons).
        Runs WLS regressions to test whether effects vary by covariates.
    switchers : {"", "in", "out"}, default=""
        Which switchers to include. "" for all, "in" for treatment increases,
        "out" for treatment decreases.
    only_never_switchers : bool, default=False
        If True, use only never-switchers as controls (exclude not-yet-switchers).
    same_switchers : bool, default=False
        If True, use the same set of switchers across all effect horizons.
    same_switchers_pl : bool, default=False
        If True, use the same set of switchers across all placebo horizons.
    trends_lin : bool, default=False
        If True, include unit-specific linear time trends.
    trends_nonparam : list[str], optional
        Variables for non-parametric group-specific trends.
    continuous : int, default=0
        Polynomial degree for continuous treatment. If > 0, treatment is
        treated as continuous with polynomial terms of the specified degree.
    ci_level : float, default=95.0
        Confidence level for confidence intervals (e.g., 95.0 for 95% CI).
    less_conservative_se : bool, default=False
        If True, use less conservative standard error estimation with
        degrees-of-freedom adjustment.
    keep_bidirectional_switchers : bool, default=False
        If True, keep units that both increase AND decrease treatment over
        time. By default, these units are dropped.
    drop_missing_preswitch : bool, default=False
        If True, drop observations where treatment is missing before the
        first switch time.

    Returns
    -------
    DIDInterResult
        Result object containing:

        - *effects*: EffectsResult with treatment effects at each horizon
        - *placebos*: PlacebosResult with placebo effects (if placebo > 0)
        - *ate*: ATEResult with average total effect
        - *n_units*: Total number of units
        - *n_switchers*: Number of switching units
        - *n_never_switchers*: Number of never-switching units
        - *ci_level*: Confidence level used
        - *effects_equal_test*: Test results for equal effects (if requested)
        - *placebo_joint_test*: Joint test that all placebos are zero
        - *influence_effects*: Influence functions for effects
        - *influence_placebos*: Influence functions for placebos
        - *heterogeneity*: Heterogeneous effects analysis (if predict_het specified)
        - *estimation_params*: Dictionary of estimation parameters

    See Also
    --------
    att_gt : Group-time ATT estimation
    cont_did : Continuous treatment DID

    References
    ----------

    .. [1] de Chaisemartin, C., & D'Haultfoeuille, X. (2024). Difference-in-
           Differences Estimators of Intertemporal Treatment Effects.
           *Review of Economics and Statistics*, 106(6), 1723-1736.
           https://doi.org/10.1162/rest_a_01414
    """
    config = DIDInterConfig(
        yname=yname,
        tname=tname,
        gname=idname,
        dname=dname,
        cluster=cluster,
        weightsname=weightsname,
        controls=controls,
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
    )

    builder = PreprocessDataBuilder()
    preprocessed = builder.with_data(data).with_config(config).validate().transform().build()

    return compute_effects(preprocessed)
