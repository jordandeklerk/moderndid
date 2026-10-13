"""Marginal effects aggregation for ETWFE cell-level treatment effects."""

import warnings

from scipy import stats

from .compute import _format_cells, compute_emfx
from .container import EmfxResult, EtwfeResult


def emfx(result, type="simple", post_only=True, window=None):
    r"""Aggregate ETWFE cell-level treatment effects.

    Averages the cohort-time effects :math:`\hat{\tau}_{g,t}` of :func:`~moderndid.etwfe` into
    an overall effect or into effects by cohort, calendar time, or exposure time [1]_. Each
    effect weights a cell by its number of observations. For nonlinear families, the effect of
    an observation is the difference between its predicted outcomes with and without
    treatment [2]_.

    Every type also reports a summary effect of the post-treatment cells. Standard errors come
    from the delta method with the regression's variance-covariance matrix and treat the weights
    as fixed. A cell that the regression dropped as collinear draws a warning. Its effect and
    every average that includes it are NaN.

    See the :ref:`extended TWFE example <example_etwfe>` for overall, group, and event study
    aggregations of ``etwfe`` estimates.

    Parameters
    ----------
    result : EtwfeResult
        The result of :func:`~moderndid.etwfe`.
    type : {"simple", "group", "calendar", "event"}, default="simple"
        How to aggregate the cells. ``"simple"`` averages all post-treatment cells, ``"group"``
        averages within each treatment cohort, ``"calendar"`` within each calendar time, and
        ``"event"`` within each exposure time :math:`e = t - g`.
    post_only : bool, default=True
        Whether the event study keeps only the event times from zero on. With ``False``, the
        never-treated design also reports its placebo cells and the reference period
        :math:`e = -1` at zero. The other types never use pre-treatment cells.
    window : tuple[int, int], optional
        For the event study, keep only the event times from ``window[0]`` to ``window[1]``.

    Returns
    -------
    EmfxResult
        Aggregated treatment effects with delta-method standard errors:

        - **overall_att**: summary effect for the aggregation type
        - **overall_se**: standard error of the summary effect
        - **aggregation_type**: the aggregation type
        - **event_times**: event times, cohorts, or calendar times, None for the simple type
        - **att_by_event**: effect at each value of ``event_times``
        - **se_by_event**: standard error of each effect, NaN at the reference period
        - **ci_lower**: lower bound of each pointwise confidence interval
        - **ci_upper**: upper bound of each pointwise confidence interval
        - **critical_value**: normal critical value of the intervals
        - **n_obs**: number of observations in the regression
        - **estimation_params**: estimation details carried over from the regression

    See Also
    --------
    etwfe : Estimate the saturated ETWFE regression.
    aggte : Aggregation for Callaway and Sant'Anna (2021) group-time ATTs.

    Notes
    -----
    With :math:`N_g` the number of units in cohort :math:`g`, the simple effect averages the
    post-treatment cells with weights proportional to cohort size,

    .. math::

        \hat{\bar{\tau}}_\omega
        = \sum_g \sum_{t=g}^{T} \hat{\omega}_g \, \hat{\tau}_{g,t},
        \qquad
        \hat{\omega}_g = \frac{N_g}{\sum_{g'} (T - g' + 1) \, N_{g'}}.

    The effect at exposure time :math:`e = t - g` averages the cohorts observed at that exposure,

    .. math::

        \hat{\tau}_{\omega,e}
        = \sum_{g=q}^{T-e} \hat{\omega}_{ge} \, \hat{\tau}_{g,\,g+e},
        \qquad
        \hat{\omega}_{ge} = \frac{N_g}{N_q + \cdots + N_{T-e}}.

    With :math:`\hat{\tau}_g` the effect of cohort :math:`g` and :math:`\hat{\tau}_t` the effect
    at calendar time :math:`t`, the summaries of the other types are

    .. math::

        \hat{\theta}_{\mathrm{group}}
        = \frac{\sum_g N_g \, \hat{\tau}_g}{\sum_g N_g},
        \qquad
        \hat{\theta}_{\mathrm{event}}
        = \frac{1}{|E_+|} \sum_{e \in E_+} \hat{\tau}_{\omega,e},
        \qquad
        \hat{\theta}_{\mathrm{calendar}}
        = \frac{1}{|C|} \sum_{t \in C} \hat{\tau}_t,

    where :math:`E_+` holds the reported event times from zero on and :math:`C` the reported
    calendar times. Without ``idname``, the group summary takes :math:`N_g` as the cohort's
    average number of observations per post-treatment period. The
    :ref:`extended TWFE background <background-etwfe>` explains what each average represents.

    References
    ----------

    .. [1] Wooldridge, J. M. (2025). "Two-Way Fixed Effects, the Two-Way
       Mundlak Regression, and Difference-in-Differences Estimators."
       Empirical Economics, 69(5), 2545-2587.
       https://doi.org/10.1007/s00181-025-02807-z

    .. [2] Wooldridge, J. M. (2023). "Simple Approaches to Nonlinear
       Difference-in-Differences with Panel Data." The Econometrics
       Journal, 26(3), C31-C66.
       https://doi.org/10.1093/ectj/utad016

    """
    if not isinstance(result, EtwfeResult):
        raise TypeError(f"Expected EtwfeResult, got {result.__class__.__name__}")
    if result.model_coefficients is None:
        raise ValueError("result has no model_coefficients. Refit the model with etwfe before calling emfx.")

    valid_types = ("simple", "group", "calendar", "event")
    if type not in valid_types:
        raise ValueError(f"type must be one of {valid_types}, got '{type}'")

    alpha = result.estimation_params.get("alpha", 0.05)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    mfx = compute_emfx(
        fit_data=result.data,
        config=result.config,
        coef_names=result.coef_names,
        beta=result.model_coefficients,
        vcov_matrix=result.vcov,
        agg_type=type,
        post_only=post_only,
        window=window,
    )
    if mfx["dropped_cells"]:
        warnings.warn(
            f"The regression dropped the treatment cells {_format_cells(mfx['dropped_cells'])} as collinear. "
            "Their effects and every aggregate that includes them are NaN.",
            UserWarning,
            stacklevel=2,
        )

    overall_att = mfx["overall_att"]
    overall_se = mfx["overall_se"]

    event_times = mfx["event_times"]
    att_by_event = mfx["att_by_event"]
    se_by_event = mfx["se_by_event"]

    ci_lower = ci_upper = None
    if att_by_event is not None and se_by_event is not None:
        ci_lower = att_by_event - z_crit * se_by_event
        ci_upper = att_by_event + z_crit * se_by_event

    return EmfxResult(
        overall_att=overall_att,
        overall_se=overall_se,
        aggregation_type=type,
        event_times=event_times,
        att_by_event=att_by_event,
        se_by_event=se_by_event,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        critical_value=z_crit,
        n_obs=result.n_obs,
        estimation_params={**result.estimation_params, "alpha": alpha},
    )
