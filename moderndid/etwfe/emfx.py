"""Marginal effects aggregation for ETWFE cell-level treatment effects."""

from __future__ import annotations

from scipy import stats

from .compute import compute_emfx, run_etwfe_regression
from .container import EmfxResult, EtwfeResult


def emfx(
    result: EtwfeResult,
    type: str = "simple",
    post_only: bool = True,
    window: tuple[int, int] | None = None,
) -> EmfxResult:
    r"""Aggregate ETWFE cell-level treatment effects.

    Computes weighted averages of the cohort-time ATTs
    :math:`\hat{\tau}_{g,t}` from :func:`~moderndid.etwfe.etwfe.etwfe`
    into overall, group, calendar-time, or event-study summaries [1]_.
    For a simple overall effect, the weighted average is

    .. math::

        \hat{\bar{\tau}}_\omega
        = \sum_g \sum_{t=g}^{T} \hat{\omega}_g \, \hat{\tau}_{g,t},
        \qquad
        \hat{\omega}_g = \frac{N_g}{\sum_{g'} (T - g' + 1) \, N_{g'}},

    where :math:`N_g` is the number of units in cohort :math:`g`.
    For event-study aggregation, effects are averaged by exposure time
    :math:`e = t - g` with cohort-share weights within each exposure
    level,

    .. math::

        \hat{\tau}_{\omega,e}
        = \sum_{g=q}^{T-e} \hat{\omega}_{ge} \, \hat{\tau}_{g,\,g+e},
        \qquad
        \hat{\omega}_{ge} = \frac{N_g}{N_q + \cdots + N_{T-e}}.

    Standard errors are obtained via the delta method using the model's
    variance-covariance matrix.

    Parameters
    ----------
    result : EtwfeResult
        Output from :func:`~moderndid.etwfe.etwfe.etwfe`.
    type : {'simple', 'group', 'calendar', 'event'}, default='simple'
        Aggregation type:

        - ``"simple"``: overall weighted average across all post-treatment
          (g, t) cells
        - ``"group"``: average within each treatment cohort g
        - ``"calendar"``: average within each calendar time t
        - ``"event"``: average within each exposure time e = t - g
    post_only : bool, default=True
        If True, only include post-treatment cells (t >= g) in aggregation.
    window : tuple[int, int] or None, default=None
        For event-study aggregation, restrict to event times within
        ``[window[0], window[1]]``.

    Returns
    -------
    EmfxResult
        Aggregated treatment effects with delta-method standard errors.

    See Also
    --------
    etwfe : Estimate the saturated ETWFE regression.
    aggte : Aggregation for Callaway and Sant'Anna (2021) group-time ATTs.

    References
    ----------
    .. [1] Wooldridge, J. M. (2025). "Two-Way Fixed Effects, the Two-Way
       Mundlak Regression, and Difference-in-Differences Estimators."
       Empirical Economics.

    Examples
    --------
    .. ipython::
        :okwarning:

        In [1]: from moderndid import etwfe, emfx, load_mpdta
           ...:
           ...: df = load_mpdta()
           ...: mod = etwfe(
           ...:     data=df,
           ...:     yname="lemp",
           ...:     tname="year",
           ...:     gname="first.treat",
           ...:     idname="countyreal",
           ...: )

    Simple overall ATT:

    .. ipython::
        :okwarning:

        In [2]: print(emfx(mod, type="simple"))

    Event-study aggregation by exposure time:

    .. ipython::
        :okwarning:

        In [3]: print(emfx(mod, type="event"))

    Group-level aggregation:

    .. ipython::
        :okwarning:

        In [4]: print(emfx(mod, type="group"))
    """
    if not isinstance(result, EtwfeResult):
        raise TypeError(f"Expected EtwfeResult, got {result.__class__.__name__}")

    valid_types = ("simple", "group", "calendar", "event")
    if type not in valid_types:
        raise ValueError(f"type must be one of {valid_types}, got '{type}'")

    config = result.config
    alpha = result.estimation_params.get("alpha", 0.05)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    reg = run_etwfe_regression(
        config._formula,
        result.data,
        config,
        vcov=result.estimation_params.get("vcov_type", "hetero"),
        backend=result.estimation_params.get("backend"),
    )
    model = reg["model"]

    mfx = compute_emfx(
        model=model,
        fit_data=result.data,
        config=config,
        agg_type=type,
        post_only=post_only,
        window=window,
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
