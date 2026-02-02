.. _causal_inference:

************************************************
Introduction to Difference-in-Differences
************************************************

This page introduces the core concepts and terminology from the
difference-in-differences literature.

Difference-in-differences is a method for estimating causal effects when
randomized experiments are not possible. The idea is to compare how outcomes
change over time for units that receive a treatment to how outcomes change for
units that do not. If both groups would have evolved similarly absent treatment,
then any divergence after treatment can be attributed to the treatment itself.

We start with the canonical two-period case to build intuition, then extend
to multiple periods and staggered treatment timing.


Notation and Potential Outcomes
===============================

The potential outcomes framework provides precise definitions of causal effects.
For unit :math:`i` in period :math:`s`, we define :math:`Y_{is}(0)` as the
*untreated potential outcome*, representing what unit :math:`i` would experience
in period :math:`s` if it did *not* participate in treatment. Similarly,
:math:`Y_{is}(1)` is the *treated potential outcome*, representing what unit
:math:`i` would experience in period :math:`s` if it *did* participate in
treatment. The indicator :math:`D_i` denotes group membership, taking value 1
for units in the treated group and 0 for units in the untreated group.

In the canonical two-period setup with periods :math:`t-1` and :math:`t`, no
one is treated in the first period, and units in the treated group become
treated in the second period. This means observed outcomes are given by

.. math::

   Y_{i,t-1} = Y_{i,t-1}(0) \quad \text{and} \quad Y_{it} = D_i Y_{it}(1) + (1-D_i) Y_{it}(0).

In the first period, we observe untreated potential outcomes for everyone since
no treatment has occurred yet. There is an implicit no-anticipation assumption
here, meaning units do not change their behavior in anticipation of future
treatment. In the second period, we observe treated potential outcomes for
units that actually participate in the treatment and untreated potential
outcomes for units that do not participate.


The Average Treatment Effect on the Treated
===========================================

The primary parameter of interest in DiD designs is the Average Treatment
Effect on the Treated, or ATT. It is defined as

.. math::

   ATT = \mathbb{E}[Y_t(1) - Y_t(0) \mid D=1].

This quantity represents the average difference between treated and untreated
potential outcomes for units in the treated group. It answers a specific causal
question: what was the average effect of the treatment on those who actually
received it?

The fundamental challenge of causal inference is that we never observe
:math:`Y_t(0)` for treated units. We see what happened to them after treatment,
but we cannot observe what *would have* happened to them in the absence of
treatment. This unobserved quantity is called the counterfactual. DiD solves
this problem by using the untreated group to construct an estimate of this
counterfactual.


The Parallel Trends Assumption
==============================

The key identifying assumption in DiD is called parallel trends. Formally,
this assumption states that

.. math::

   \mathbb{E}[Y_t(0) - Y_{t-1}(0) \mid D=1] = \mathbb{E}[Y_t(0) - Y_{t-1}(0) \mid D=0].

This says that the change in untreated potential outcomes over time
that units in the treated group *would have experienced* if they had not
participated in treatment is the same as the change in outcomes that units in
the untreated group *actually* experienced.

This assumption is potentially useful because the left-hand side of the
equation, the path of untreated potential outcomes for the treated group, is
fundamentally unobservable. We cannot see what would have happened to treated
units without treatment. However, the right-hand side, the path of untreated
potential outcomes for the untreated group, is directly observable since these
units never received treatment. The parallel trends assumption allows us to use
this observable quantity to stand in for the unobservable counterfactual.

Under parallel trends, the ATT is identified and given by

.. math::

   ATT = \mathbb{E}[Y_t - Y_{t-1} \mid D=1] - \mathbb{E}[Y_t - Y_{t-1} \mid D=0].

The ATT equals the mean change in outcomes over time experienced by units in
the treated group, adjusted by the mean change in outcomes over time experienced
by units in the untreated group. This latter term, under the parallel trends
assumption, represents what the path of outcomes for treated units would have
been if they had not participated in treatment. The resulting
"difference-in-differences" removes both time-invariant differences between
groups and common time trends affecting all units.

It is important to note that parallel trends allows the *level* of untreated
potential outcomes to differ across groups. The treated group could have
systematically higher or lower outcomes than the untreated group in both
periods. What matters is that the *changes* over time would have been the same
in the absence of treatment. This is consistent with fixed effects models where
the mean of unobserved unit characteristics can differ across groups.


Two-Way Fixed Effects and Its Limitations
=========================================

Now consider a more general setting with :math:`\mathcal{T}` total time periods
where different units become treated at different times. This is common in
policy evaluation where reforms roll out to different regions over time.


The TWFE Regression
-------------------

The traditional approach to estimating treatment effects in this setting is
the two-way fixed effects (TWFE) linear regression

.. math::

   Y_{it} = \theta_t + \eta_i + \alpha D_{it} + \varepsilon_{it},

where :math:`\theta_t` is a time fixed effect, :math:`\eta_i` is a unit fixed
effect, :math:`D_{it}` is a treatment dummy variable equal to 1 if unit
:math:`i` has been treated by time :math:`t`, :math:`\varepsilon_{it}` represents
time-varying unobservables, and :math:`\alpha` is presumably the parameter of
interest. Researchers typically interpret :math:`\alpha` as the average effect
of participating in treatment.

When there are only two time periods, this approach works well. The coefficient
:math:`\alpha` is numerically equal to the ATT under parallel trends and
exhibits robustness to treatment effect heterogeneity. Even if the effect
varies across individual units, the TWFE estimate correctly captures the
average treatment effect on the treated.


The Problem with Staggered Adoption
-----------------------------------

This robustness does not extend to settings with multiple time periods and
variation in treatment timing. The problem is that in a TWFE regression, units
whose treatment status does not change over time serve as the comparison group
for units whose treatment status does change. With staggered adoption, some of
these comparisons are problematic.

Consider the three types of comparisons that TWFE implicitly makes. First,
it compares newly treated units to never-treated units, which is exactly in
the spirit of DiD. We adjust the path of outcomes for newly treated units
by the path of outcomes for units that never participate in treatment.
Second, it compares newly treated units to not-yet-treated units, which is
also reasonable since these units have not yet been affected by treatment
and can serve as valid comparisons for the current period.

The third comparison, however, is problematic. TWFE also compares newly
treated units to already-treated units, those that received treatment in
earlier periods. But already-treated units do not represent untreated
potential outcomes. Their outcomes in later periods reflect the ongoing
effects of treatment, including any treatment effect dynamics. Using them
as controls means that treatment effect dynamics from earlier-treated groups
contaminate the estimate of :math:`\alpha`.


Consequences of Contamination
-----------------------------

This contamination can have severe consequences. It is possible to construct
examples where the effect of treatment is positive for all units in all time
periods, yet the TWFE estimate is negative. Effects can appear smaller than
they actually are, and spurious "pre-trends" can appear in the data even when
the parallel trends assumption genuinely holds. The estimated coefficient
:math:`\alpha` does not correspond to any clearly interpretable causal
parameter.

These problems arise even when treatment effects are homogeneous across groups.
The issues are structural, stemming from which comparisons TWFE makes, not from
treatment effect heterogeneity per se. Heterogeneity makes the problems worse,
but homogeneity does not eliminate them.


Event-Study Regressions
-----------------------

A common extension of TWFE is the event-study regression

.. math::

   Y_{it} = \alpha_i + \alpha_t + \sum_{k \neq -1} \gamma_k D_{it}^k + \varepsilon_{it}.

where :math:`D_{it}^k = \mathbf{1}\{t - G_i = k\}` is an indicator for unit
:math:`i` being exactly :math:`k` periods from initial treatment at time
:math:`t`, and :math:`G_i` denotes the period when unit :math:`i` first receives
treatment. For instance, :math:`D_{it}^0` equals one if unit :math:`i` is first
treated at time :math:`t`, while :math:`D_{it}^{-2}` equals one if unit
:math:`i` will be treated in two periods.

Researchers typically interpret the coefficients :math:`\gamma_k` for
:math:`k \geq 0` as dynamic treatment effects, showing how the impact of
treatment evolves over time since implementation. The coefficients
:math:`\gamma_k` for :math:`k < 0` are interpreted as pre-trends, serving as
placebo tests of the parallel trends assumption. If these pre-treatment
coefficients are close to zero, it suggests the treated and comparison groups
were evolving similarly before treatment occurred.

Unfortunately, these interpretations can be severely misleading. The estimated
post-treatment effects :math:`\hat{\gamma}_k` for :math:`k \geq 0` are biased
for the true dynamic effects, even when treatment effects are homogeneous
across groups. The pre-treatment coefficients :math:`\hat{\gamma}_k` for
:math:`k < 0` can appear statistically significant even when parallel trends
genuinely holds, making pre-trend tests unreliable. These problems occur
because event-study regressions suffer from the same fundamental issue as
TWFE, implicitly using already-treated units as part of the comparison group.


Group-Time Average Treatment Effects
====================================

The solution to these problems is conceptually straightforward. Rather than
pooling all units into a single regression that makes problematic comparisons,
we estimate treatment effects separately for each combination of treatment
cohort and time period, using only valid comparisons.

Let :math:`G_i` denote the time period when unit :math:`i` first receives
treatment. If a unit is never treated, we set :math:`G_i = \infty`. Units
with the same treatment timing form a group or cohort. For example, if some
states raised their minimum wage in 2010 and others in 2012, there are two
treatment groups, the 2010 cohort and the 2012 cohort. Units that never
receive treatment form the comparison group.

The group-time average treatment effect is defined as

.. math::

   ATT(g, t) = \mathbb{E}[Y_t(g) - Y_t(0) \mid G = g].

This is the average effect of participating in treatment for units in group
:math:`g` at time period :math:`t`. The notation :math:`Y_t(g)` denotes the
potential outcome at time :math:`t` if a unit were first treated in period
:math:`g`. This parameter is flexible and does not impose homogeneity across
groups or time. When there are only two time periods and two groups, the ATT
from the canonical case equals :math:`ATT(g=2, t=2)`.

To give a concrete example, suppose a researcher has access to data from 2010
to 2015, with some units first treated in 2012 and others first treated in
2014. Then :math:`ATT(g=2012, t=2014)` is the average effect of participating
in treatment for the group of units that became treated in 2012, measured in
2014, which is two years after their treatment began. Similarly,
:math:`ATT(g=2014, t=2014)` is the effect for units treated in 2014, measured
in that same year.


Parallel Trends with Multiple Periods
=====================================

The parallel trends assumption extends naturally to the multi-period setting.
When using never-treated units as the comparison group, the assumption states
that for all groups :math:`g` and post-treatment periods :math:`t \geq g`,

.. math::

   \mathbb{E}[Y_t(0) - Y_{t-1}(0) \mid G = g] = \mathbb{E}[Y_t(0) - Y_{t-1}(0) \mid C = 1],

where :math:`C = 1` indicates units in the never-treated group. This says that,
in the absence of treatment, the average path of untreated potential outcomes
for group :math:`g` would have been the same as the path for never-treated
units.

An alternative version uses not-yet-treated units as the comparison group.
For all groups :math:`g` and periods :math:`t \geq g`

.. math::

   \mathbb{E}[Y_t(0) - Y_{t-1}(0) \mid G = g] = \mathbb{E}[Y_t(0) - Y_{t-1}(0) \mid D_s = 0, G \neq g],

where :math:`s \geq t` and :math:`D_s = 0` indicates units not yet treated by
period :math:`s`. This version uses units that will be treated later, but have
not yet been treated, as valid comparisons. It can improve precision when the
never-treated group is small, but requires assuming that treatment timing is
unrelated to potential outcomes.

Under either version of parallel trends, each :math:`ATT(g, t)` is identified.
With never-treated comparisons

.. math::

   ATT(g, t) = \mathbb{E}[Y_t - Y_{g-1} \mid G = g] - \mathbb{E}[Y_t - Y_{g-1} \mid C = 1],

this compares the change in outcomes for group :math:`g` from just before
treatment (period :math:`g-1`) to period :math:`t`, with the corresponding
change for never-treated units over the same time span. Each group-time
effect is a clean causal comparison that never uses already-treated units
as controls, avoiding the problems that plague TWFE.


Aggregating Group-Time Effects
==============================

Group-time average treatment effects are the fundamental building blocks of
causal inference in staggered adoption settings. However, with many groups
and time periods, the set of all :math:`ATT(g, t)` can be large. There are
both benefits and costs to working with these disaggregated effects. The
main benefit is that it is relatively straightforward to examine heterogeneous
effects across groups and time. The cost is that summarizing many parameters
into interpretable conclusions can be challenging. Several aggregation schemes
address this challenge.


Event-Study Aggregation
-----------------------

Event-study aggregation answers the question of how treatment effects evolve
with time since treatment. Define :math:`e = t - g` as the event time or
relative time, representing the number of periods since treatment started.
The event-study parameter is

.. math::

   \theta_D(e) = \sum_g \mathbf{1}\{g + e \leq \mathcal{T}\} \cdot ATT(g, g + e) \cdot P(G = g \mid G + e \leq \mathcal{T}).

This averages :math:`ATT(g, t)` across all groups that are observed :math:`e`
periods from treatment, weighted by the relative size of each group. Event
time :math:`e = 0` is the period of first treatment. Positive event times
show how treatment effects evolve in the periods following implementation,
revealing whether effects strengthen, weaken, or remain stable over time.

Negative event times represent pre-treatment periods. Since treatment has
not yet occurred, any non-zero effects in these periods would indicate that
groups were already diverging before the policy change, which would violate
parallel trends. Effects near zero for negative event times support the
identifying assumption.


Group Aggregation
-----------------

Group aggregation addresses whether early adopters experience different
treatment effects than late adopters. For each group :math:`g`, we average
effects over all post-treatment periods

.. math::

   \theta_S(g) = \frac{1}{\mathcal{T} - g + 1} \sum_{t=g}^{\mathcal{T}} ATT(g, t).

This reveals treatment effect heterogeneity across cohorts. Differences
might reflect how the policy was implemented differently over time,
compositional differences between early and late adopters, or genuine
variation in treatment effectiveness.


Calendar-Time Aggregation
-------------------------

Calendar-time aggregation shows how the aggregate treatment effect evolves
over calendar time, accounting for the staggered rollout of treatment. For
each calendar period :math:`t`,

.. math::

   \theta_C(t) = \sum_g \mathbf{1}\{t \geq g\} \cdot ATT(g, t) \cdot P(G = g \mid G \leq t).

This weights each group's contribution by its relative size among all groups
treated by time :math:`t`. It is useful for understanding the total impact
of a policy at each point in time, particularly when aggregate effects may
vary with macroeconomic conditions or concurrent policy changes.


Overall Average Treatment Effect
--------------------------------

When a single summary measure is needed, the overall effect aggregates across
all groups and post-treatment periods

.. math::

   \theta_S^O = \sum_g \theta_S(g) \cdot P(G = g \mid G \leq \mathcal{T}).

This is the average effect of participating in treatment across the entire
treated population, properly accounting for the staggered adoption pattern.
It is the natural multi-period analogue of the ATT in the two-period case.
If a researcher must report a single treatment effect summary, this is the
recommended parameter.


Conditional Parallel Trends
===========================

In many applications, the parallel trends assumption is more credible when
conditioning on observed pre-treatment covariates. For example, if treatment
timing correlates with observable characteristics that also affect outcome
trends, controlling for these characteristics strengthens identification.

The conditional parallel trends assumption states that, within levels of
covariates :math:`X`, treated and comparison groups would have followed the
same path

.. math::

   \mathbb{E}[Y_t(0) - Y_{t-1}(0) \mid X, G = g] = \mathbb{E}[Y_t(0) - Y_{t-1}(0) \mid X, C = 1].

This allows covariate-specific trends. Different covariate values can have
different outcome trends, as long as treated and comparison units with the
same covariate values would have followed the same trend absent treatment.

The target parameter remains the unconditional :math:`ATT(g, t)`. To recover
it, we estimate the covariate-conditional effect and then average over the
distribution of covariates for units in group :math:`g`.


Using ModernDiD
===============

ModernDiD implements the methods described above. The :mod:`~moderndid.did`
module estimates group-time average treatment effects using the
`Callaway and Sant'Anna (2021) <https://arxiv.org/abs/1803.09015>`_ framework,
with options for never-treated or not-yet-treated comparison groups, outcome
regression, inverse probability weighting, or doubly robust estimation.

Key parameters map directly to the concepts introduced here:

- ``yname``: the outcome variable :math:`Y`
- ``tname``: the time period variable :math:`t`
- ``idname``: the unit identifier :math:`i`
- ``gname``: the group (first treatment period) variable :math:`G`
- ``xformla``: covariates :math:`X` for conditional parallel trends
- ``control_group``: whether to use never-treated or not-yet-treated comparisons

The ``att_gt`` function returns all group-time effects :math:`ATT(g,t)`.
The ``aggte`` function aggregates these into event-study, group, calendar-time,
or overall summaries as described above.

For the classic two-period, two-group setting, the ``drdid`` function provides
the `Sant'Anna and Zhao (2020) <https://arxiv.org/abs/1812.01723>`_ estimators
directly without requiring the multi-period machinery.

See the :doc:`/user_guide/index` for detailed examples.
