.. _causal_inference:

============================
The idea behind a comparison
============================

An outcome changing after a policy begins does not tell you how much of that
change the policy caused. In the minimum wage data used in our
:doc:`quickstart`, teen employment could change because of the wage increase,
the business cycle, or other developments in a county. Difference-in-differences
(DiD) uses an untreated comparison group to estimate the change that the
treated counties would have experienced without the policy.

We'll use the employment question to work through what makes that comparison
causal, starting with one treatment date before considering staggered
adoption. You can then connect the identifying assumptions to the comparison
groups and summaries you choose in a ModernDiD analysis. The
:doc:`background pages <../background/index>` give the formal assumptions and
derivations for each method once you want to work through them in detail.

The outcome you cannot observe
------------------------------

For a unit observed before and after treatment, write :math:`Y_s(0)` for the
outcome it would have in period :math:`s` without treatment and
:math:`Y_s(1)` for its outcome with treatment. A unit can be a county, a firm,
or another entity whose outcomes you follow. Let :math:`D=1` indicate that
it belongs to the treated group and :math:`D=0` indicate the comparison group.

The average treatment effect on the treated (ATT) asks how treatment changed
the outcome for the units that received it. It compares their treated and
untreated potential outcomes in the same post-treatment period :math:`t`,

.. math::

   ATT = \mathbb{E}[Y_t(1)-Y_t(0)\mid D=1].

In the minimum wage study, this target averages the employment effect across
counties whose states raised the wage. It does not describe how the policy
would affect every county or imply that all treated counties share the same
effect. Although you observe :math:`Y_t(1)` for the treated counties, their
counterfactual employment under no increase, :math:`Y_t(0)`, is missing.
A before-and-after comparison does not recover it because employment might
have changed even without the policy.

Use an untreated change to construct the comparison
---------------------------------------------------

Suppose no units are treated in period :math:`t-1` and only the treated group
receives treatment in period :math:`t`. Under no anticipation, the earlier
outcomes have not already responded to the future policy. We use that
untreated period as a baseline and subtract the comparison group's outcome
change from the treated group's change over the same periods.

For that subtraction to recover the ATT, the groups must have the same
average change in outcomes under no treatment. This is the parallel trends
assumption,

.. math::

   \mathbb{E}[Y_t(0)-Y_{t-1}(0)\mid D=1]
   = \mathbb{E}[Y_t(0)-Y_{t-1}(0)\mid D=0].

Parallel trends allows the groups to have different employment levels
before treatment because it concerns how those levels would change without
the policy. Choosing counties with similar employment levels alone does not
establish the assumption; their untreated employment changes must also be
comparable during the period you study.

If parallel trends and no anticipation hold, the unobserved trend for the
treated group equals the observed trend for the comparison group. Substituting
that trend into the ATT gives the two-period DiD formula,

.. math::

   ATT = \mathbb{E}[Y_t-Y_{t-1}\mid D=1]
   - \mathbb{E}[Y_t-Y_{t-1}\mid D=0].

This interpretation also requires that the comparison units remain
untreated and that their outcomes are not changed by spillovers from the
treated units. The estimate cannot distinguish a treatment effect from an
unrelated shock that affects only the treated group at the same time.

.. admonition:: The comparison needs a substantive reason
   :class: important

   A fitted model cannot establish why the untreated group represents the
   treated group's missing outcome path. That argument comes from the policy,
   how treatment was assigned and what else changed during your study.

ModernDiD's :func:`~moderndid.drdid` estimates this two-period ATT with panel
data or repeated cross-sections. The :doc:`two-period background
<../background/drdid>` explains the additional conditions each data structure
requires.

.. _conditional-parallel-trends:

Make the comparison conditional on observed characteristics
-----------------------------------------------------------

Treated and comparison units may differ in characteristics that predict
outcome growth. In the county data, population is one possible characteristic
to consider. If employment trends vary with population, a comparison that
adjusts for pre-policy population can be more plausible than one that treats
all counties as comparable.

Conditional parallel trends asks for the same untreated outcome change among
treated and comparison units with the same covariates :math:`X`. To make that
comparison, the analysis also needs overlap so the data contains comparison
units with the characteristics represented among treated units. Covariate
adjustment cannot supply those comparisons when they are absent from the
sample.

You pass the covariates through ``xformla`` in estimators such as
:func:`~moderndid.att_gt`. Its default ``est_method="dr"`` combines an outcome
regression with a propensity score model. Under the identifying assumptions,
the doubly robust estimator is consistent when either working model is
correctly specified. You still need parallel trends and overlap because
double robustness concerns estimation of an already identified effect. The
:doc:`data guide <data>` explains how to choose covariates measured before
treatment can affect them.

Keep adoption dates separate
----------------------------

When units adopt in different periods and remain treated afterward, an
earlier adopter is already exposed to treatment when a later adopter begins.
Using the earlier adopter's outcome change as the later adopter's comparison
can then subtract a treatment response. A conventional two-way fixed effects
regression includes such comparisons. Its coefficient can be difficult to
interpret when effects differ across cohorts or change with exposure.

The comparison problem also affects event-study regressions when treatment
effects differ across cohorts or exposure lengths. Their lead and lag
coefficients can mix effects from other event times, so apparent
pre-treatment differences can arise from that mixing.
`Sun and Abraham (2021) <https://doi.org/10.1016/j.jeconom.2020.09.006>`_
explain this problem for conventional event-study regressions. It is a reason
to choose the estimator before interpreting the shape of an event study.

The approach of `Callaway and Sant'Anna (2021)
<https://doi.org/10.1016/j.jeconom.2020.12.001>`_ instead estimates effects
separately for each adoption cohort and period. If :math:`G=g` identifies units
first treated in period :math:`g`, its target is

.. math::

   ATT(g,t) = \mathbb{E}[Y_t(g)-Y_t(0)\mid G=g].

Here :math:`Y_t(g)` describes the outcome under adoption in period :math:`g`,
so :math:`ATT(2004,2006)` concerns the 2004 cohort's employment in 2006
relative to its own employment under no wage increase. We use untreated
comparison counties to learn about that missing outcome through a parallel
trends assumption.

The :func:`~moderndid.att_gt` function estimates these group-time effects.
Its ``control_group`` argument chooses never-treated units or units that
remain untreated during the comparison. To use later adopters as comparison
units, you need a credible parallel trends assumption for those comparisons
and attention to anticipation rather than randomly assigned adoption dates.

Decide which average answers your question
------------------------------------------

A collection of group-time effects lets you choose the average that answers
your question about the policy. If you want to know how effects change with
exposure, an event study averages effects at a given time since adoption.
You can instead summarize one cohort's experience over its observed treated
periods or average the effects for all cohorts already treated in a
particular calendar year.

Because those averages can describe different populations and exposure
lengths, they need not give the same answer. In an event study, later adopters may
contribute to the first-year effect but have no observations several years
after treatment. A changing curve can therefore reflect a changing mix of
cohorts as well as changing effects within cohorts.

ModernDiD's :func:`~moderndid.aggte` computes these summaries from an
``att_gt`` result. The :doc:`results guide <results>` explains their weights
and how to restrict an event window when you want to hold its cohort
composition fixed.

Assess the assumptions behind the result
-----------------------------------------

Pre-treatment comparisons can reveal differences in outcome changes before
the policy begins. This provides evidence about the research design without
verifying the treated group's unobserved post-treatment outcomes.
A failure to detect a pre-treatment difference may also reflect imprecise
estimates rather than close agreement between the groups. For the same
reason, confidence intervals quantify sampling uncertainty under the design's
assumptions without measuring how far those assumptions might be from holding.
The :doc:`results guide <results>` explains how to read that uncertainty for
an individual effect and for an entire event study.

The :ref:`sensitivity analysis example <example_honest_did>` shows another way
to assess the conclusion. It uses :func:`~moderndid.honest_did` to construct
confidence intervals under specified bounds on departures from parallel
trends. You still need to justify those bounds in the context of the study.

With that interpretation in place, the :doc:`quickstart` takes these choices
into a first fit. If your treatment can reverse, varies in dose, or applies
only to an eligible subgroup, start with :doc:`estimator_overview` to find
the assumptions and comparisons that match that design.
