.. _background-tripledid:

Triple Difference-in-Differences
================================

The ``didtriple`` module implements the triple difference-in-differences (DDD) methodology from
`Ortiz-Villavicencio and Sant'Anna (2025) <https://arxiv.org/abs/2505.09942>`_. This approach extends
standard DiD to settings where treatment requires satisfying two criteria, enabling researchers to relax
parallel trends assumptions that may be implausible in traditional DiD designs.

When Triple DiD is Appropriate
------------------------------

Standard difference-in-differences compares treated and untreated units under the assumption that both
groups would have followed parallel outcome paths absent treatment. In many applications, this assumption may
be difficult to defend. Triple DiD offers an alternative when the treatment structure has a specific form.

Consider a policy that only affects units meeting two conditions simultaneously. For example, a state-level
program might target women specifically, a trade policy might affect particular crops in certain countries,
or an insurance program might be available only to farmers in participating regions. In these cases, a unit
receives treatment only if it belongs to both a group that enables treatment (such as a state that adopted
the policy) and a population partition that qualifies for treatment (such as women, specific crops, or
farmers).

This two-dimensional treatment structure creates four types of units. In the treated group
(policy-enabling region), some units qualify for treatment while others do not. In the comparison group
(non-enabling region), we observe the same partition. The triple difference exploits this structure
by comparing how the gap between qualified and non-qualified units evolves differently across enabling and
non-enabling groups.

The key advantage is that DDD allows for violations of standard parallel trends that affect either
dimension separately. Trends may differ between enabling and non-enabling groups, and trends may differ
between qualified and non-qualified units. What DDD requires is that these differential trends are stable
across the other dimension.

Setup and Notation
------------------

We consider a panel with :math:`T` time periods and units indexed by :math:`i`. Each unit belongs to a
group :math:`S_i` indicating when treatment was enabled. Let :math:`S_i \in \{2, \ldots, T, \infty\}`,
where :math:`S_i = g` means unit :math:`i` belongs to a group that first enabled treatment in period
:math:`g`, and :math:`S_i = \infty` indicates units in groups that never enable treatment.

Additionally, each unit belongs to a partition determining eligibility. Let :math:`Q_i = 1` if unit
:math:`i` qualifies for treatment and :math:`Q_i = 0` otherwise. This eligibility is assumed
time-invariant.

A unit is treated in period :math:`t` if and only if both conditions hold, namely that the group has enabled
treatment by period :math:`t` (i.e., :math:`t \geq S_i`) and that the unit qualifies (:math:`Q_i = 1`).
The treatment indicator is thus

.. math::

   D_{i,t} = \mathbf{1}\{t \geq S_i, Q_i = 1\}.

Treatment is an absorbing state, so once treated, units remain treated. Let :math:`G_i` denote the first
period in which unit :math:`i` is treated. For qualified units, :math:`G_i = S_i`. For non-qualified
units, :math:`G_i = \infty` since they are never treated regardless of their group's enabling status.

Using the potential outcomes framework, let :math:`Y_{i,t}(g)` denote the potential outcome for unit
:math:`i` at time :math:`t` if first treated in period :math:`g`, and :math:`Y_{i,t}(\infty)` denote
the never-treated potential outcome. The observed outcome combines these based on treatment status.

Parameters of Interest
----------------------

The fundamental parameter is the group-time average treatment effect for units first treated in period
:math:`g`, evaluated at time :math:`t`

.. math::

   ATT(g, t) = \mathbb{E}[Y_t(g) - Y_t(\infty) \mid G = g] = \mathbb{E}[Y_t(g) - Y_t(\infty) \mid S = g, Q = 1].

This parameter captures how treatment affects the average outcome for a specific cohort at a specific time.
The collection of all :math:`ATT(g,t)` parameters provides a complete picture of treatment effect
heterogeneity across cohorts and over time.

For summarizing results, the event-study aggregation averages effects by time elapsed since treatment

.. math::

   ES(e) = \sum_{g \in \mathcal{G}_{\text{trt}}} \mathbb{P}(G = g \mid G + e \in [2, T]) \cdot ATT(g, g+e),

where :math:`e = t - g` represents event time. This aggregation weights each cohort's contribution by its
relative size among cohorts observed at that event time. A simple overall summary averages across all
post-treatment event times

.. math::

   ES_{\text{avg}} = \frac{1}{N_E} \sum_{e \in \mathcal{E}} ES(e).

Identification Assumptions
--------------------------

Identification relies on several assumptions that parallel those in standard DiD but are adapted to the DDD
structure.

.. admonition:: Assumption (Strong Overlap)

   For every combination of enabling group :math:`g` and eligibility status :math:`q`, and for some
   :math:`\epsilon > 0`,

   .. math::

      \mathbb{P}[S = g, Q = q \mid X] > \epsilon

   with probability one.

This ensures that for any covariate value, we can find units in all four cells of the group-by-eligibility
matrix. Without overlap, certain comparisons would rely on extrapolation rather than observed data.

.. admonition:: Assumption (No Anticipation)

   For all :math:`g` in the set of treated cohorts and all pre-treatment periods :math:`t < g`,

   .. math::

      \mathbb{E}[Y_t(g) \mid S = g, Q = 1, X] = \mathbb{E}[Y_t(\infty) \mid S = g, Q = 1, X]

   with probability one.

This rules out anticipatory behavior. Units do not alter their outcomes before treatment actually begins. If
treatment is announced in advance, the treatment date should be adjusted accordingly.

.. admonition:: Assumption (DDD Conditional Parallel Trends)

   For each treated cohort :math:`g`, comparison cohort :math:`g'` with :math:`g' > \max\{g, t\}`, and
   post-treatment period :math:`t \geq g`, the following holds with probability one. Let

   .. math::

      \Delta Y_t(\infty) = Y_t(\infty) - Y_{t-1}(\infty)

   denote the change in untreated potential outcomes. Then

   .. math::

      \begin{aligned}
      &\mathbb{E}[\Delta Y_t(\infty) \mid S = g, Q = 1, X]
      - \mathbb{E}[\Delta Y_t(\infty) \mid S = g, Q = 0, X] \\
      &= \mathbb{E}[\Delta Y_t(\infty) \mid S = g', Q = 1, X]
      - \mathbb{E}[\Delta Y_t(\infty) \mid S = g', Q = 0, X].
      \end{aligned}

This is the core identifying assumption. It states that the gap in outcome trends between qualified and
non-qualified units is the same across enabling groups, conditional on covariates. Importantly, this
assumption does not require that qualified and non-qualified units follow parallel trends within a group,
nor that the same eligibility group follows parallel trends across enabling groups. It only requires that
the difference-in-trends is stable.

Why Standard Approaches Fail
----------------------------

Several common empirical strategies for DDD analysis can lead to biased or misleading results.
Understanding these pitfalls helps motivate the more careful estimation approaches described below.

Three-Way Fixed Effects Regressions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A natural starting point for DDD analysis is the three-way fixed effects (3WFE) regression

.. math::

   Y_{i,t} = \gamma_i + \gamma_{s,t} + \gamma_{q,t}
   + \beta_{3\text{wfe}} D_{i,t} + \varepsilon_{i,t},

where :math:`\gamma_i` are unit fixed effects, :math:`\gamma_{s,t}` are enabling-group-by-time fixed
effects, :math:`\gamma_{q,t}` are eligibility-group-by-time fixed effects, and :math:`D_{i,t}` is the
treatment indicator.

In the simplest setting with two periods, two enabling groups, and no covariates, the OLS estimate of
:math:`\beta_{3\text{wfe}}` does recover the :math:`ATT(2,2)`. The estimate can be written as the
difference of two DiD estimands

.. math::

   \begin{aligned}
   \beta_{3\text{wfe}}
   &= \underbrace{(\mathbb{E}[Y_2 - Y_1 \mid S=2, Q=1]
   - \mathbb{E}[Y_2 - Y_1 \mid S=2, Q=0])}_{\text{DiD among } S=2} \\
   &- \underbrace{(\mathbb{E}[Y_2 - Y_1 \mid S=\infty, Q=1]
   - \mathbb{E}[Y_2 - Y_1 \mid S=\infty, Q=0])}_{\text{DiD among } S=\infty}.
   \end{aligned}

This equivalence has led many researchers to view DDD as simply "the difference between two DiDs."
Unfortunately, this interpretation breaks down in more realistic settings.

When Covariates Matter
~~~~~~~~~~~~~~~~~~~~~~

When identification requires conditioning on covariates, computing DDD as the difference of two DiD
estimates produces biased results. The issue is that each DiD estimate integrates over covariates using its
own group's covariate distribution. But for the DDD parameter, we need to integrate using the covariate
distribution of the treated units (those with :math:`S = g` and :math:`Q = 1`).

Taking the difference of two separately computed DiD estimates averages over the wrong distributions. The
resulting estimator does not target the causal parameter of interest. Monte Carlo evidence demonstrates
that this bias can be substantial, leading to incorrect signs or magnitudes even in moderately sized
samples.

Similarly, adding covariates linearly to the 3WFE specification does not solve the problem. Specifications
like

.. math::

   Y_{i,t} = \gamma_i + \gamma_{s,t} + \gamma_{q,t}
   + \tilde{\beta}_{3\text{wfe}} D_{i,t} + X_i' \theta \cdot \mathbf{1}\{t=2\} + u_{i,t}

remain "too rigid" to properly account for covariate-specific trends in DDD designs.

The key insight is that valid DDD estimation with covariates requires *three* DiD-type comparisons, not
two. Each comparison uses a different subset of untreated units as controls, and all three are combined
while integrating over the treated units' covariate distribution.

Why Pooling Not-Yet-Treated Units Fails
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In standard staggered DiD, a common and valid practice is to pool all not-yet-treated units and use them as
an aggregate comparison group. One might expect this approach to carry over to DDD. It does not.

The DDD parallel trends assumption allows both group-specific and eligibility-specific deviations from
standard parallel trends. If the fraction of qualified units varies across enabling groups, pooling
not-yet-treated groups conflates populations with different trend structures. The deviations do not average
out, and the resulting estimator is biased.

Formally, using the shorthand :math:`\Delta Y_t(\infty)` for the change in untreated potential outcomes,
the DDD-CPT assumption does not imply that

.. math::

   \begin{aligned}
   &\mathbb{E}[\Delta Y_t(\infty) \mid S = g, Q = 1, X]
   - \mathbb{E}[\Delta Y_t(\infty) \mid S = g, Q = 0, X] \\
   &= \mathbb{E}[\Delta Y_t(\infty) \mid S > t, Q = 1, X]
   - \mathbb{E}[\Delta Y_t(\infty) \mid S > t, Q = 0, X].
   \end{aligned}

The pooled not-yet-treated group :math:`S > t` mixes units from different enabling cohorts, each potentially
with different eligibility compositions. Monte Carlo simulations show that pooling not-yet-treated units can
produce estimates with the wrong sign even when the underlying assumptions are satisfied. The solution is to
use comparison groups one at a time and combine estimates appropriately.

Identification and Estimation
-----------------------------

Under the identifying assumptions above, the group-time average treatment effects :math:`ATT(g,t)` can be
recovered using three approaches. Regression adjustment (RA) models the counterfactual outcome evolution
directly. Inverse probability weighting (IPW) reweights comparison units to match the treated group's
covariate distribution. Doubly robust (DR) estimation combines both. Each approach uses a specific
not-yet-treated cohort :math:`g_c > t` as the comparison group. All three identify the same parameter, but
they differ in their robustness properties and efficiency.

Notation
~~~~~~~~

Let :math:`\Delta Y = Y_t - Y_{g-1}` denote the outcome change from the baseline period to the current
period. Define the outcome regression function for units in enabling group :math:`g'` with eligibility
:math:`q'` as

.. math::

   m^{S=g', Q=q'}(X) = \mathbb{E}[\Delta Y \mid S = g', Q = q', X].

The generalized propensity score for being in the treated cell :math:`(S=g, Q=1)` versus comparison cell
:math:`(S=g', Q=q')` is

.. math::

   p_{g',q'}^{S=g,Q=1}(X) = \mathbb{P}[S=g, Q=1 \mid X, (S=g, Q=1) \cup (S=g', Q=q')].

Regression Adjustment Estimand
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The RA estimand adjusts for covariates by modeling the counterfactual outcome evolution directly

.. math::

   ATT_{\text{ra}, g_c}(g, t) =
   \mathbb{E}\left[w_{\text{trt}}^{S=g,Q=1}
   \left(\Delta Y - m^{S=g, Q=0}(X)
   - m^{S=g_c, Q=1}(X)
   + m^{S=g_c, Q=0}(X)\right)\right],

where the weight on treated units is

.. math::

   w_{\text{trt}}^{S=g,Q=1} =
   \frac{\mathbf{1}\{S=g, Q=1\}}{\mathbb{E}[\mathbf{1}\{S=g, Q=1\}]}.

This estimand is consistent when all three outcome regression models are correctly specified.

Inverse Probability Weighting Estimand
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The IPW estimand reweights comparison units to match the covariate distribution of treated units

.. math::

   \begin{aligned}
   ATT_{\text{ipw}, g_c}(g, t)
   &= \mathbb{E}\left[
   \left(w_{\text{trt}}^{S=g,Q=1}
   - w_{g,0}^{S=g,Q=1}(X)\right)\Delta Y\right] \\
   &- \mathbb{E}\left[
   \left(w_{g_c,1}^{S=g,Q=1}(X)
   - w_{g_c,0}^{S=g,Q=1}(X)\right)\Delta Y\right],
   \end{aligned}

where the comparison weights :math:`w_{g',q'}^{S=g,Q=1}(X)` use the generalized propensity scores to
reweight units from cell :math:`(g', q')` to match the treated cell's covariate distribution.

This estimand is consistent when all generalized propensity score models are correctly specified.

Doubly Robust Estimand
~~~~~~~~~~~~~~~~~~~~~~

The DR estimand combines both approaches

.. math::

   \begin{aligned}
   ATT_{\text{dr}, g_c}(g, t)
   &= \mathbb{E}\left[
   \left(w_{\text{trt}} - w_{g,0}\right)
   \left(\Delta Y - m^{S=g, Q=0}(X)\right)\right] \\
   &+ \mathbb{E}\left[
   \left(w_{\text{trt}} - w_{g_c,1}\right)
   \left(\Delta Y - m^{S=g_c, Q=1}(X)\right)\right] \\
   &- \mathbb{E}\left[
   \left(w_{\text{trt}} - w_{g_c,0}\right)
   \left(\Delta Y - m^{S=g_c, Q=0}(X)\right)\right].
   \end{aligned}

This estimand has a structure of three DiD-type terms. Each term compares treated units to a different
subset of untreated units. The first uses non-qualified units in the enabling group, the second uses
qualified units in the comparison group, and the third uses non-qualified units in the comparison group.
The combination ensures proper integration over the treated units' covariate distribution.

Multiply Robust Property
~~~~~~~~~~~~~~~~~~~~~~~~

The DR DDD estimator enjoys a particularly strong form of robustness. For each of the three DiD-type
components, consistency requires that either the propensity score model or the outcome regression model is
correctly specified. Since there are three components and two model types per component, this creates eight
possible combinations of correct specifications that all lead to consistent estimation.

This "multiply robust" property provides substantial protection against model misspecification. Even if
some models are wrong, as long as at least one model per component is correct, the estimator remains
consistent for the :math:`ATT(g,t)`.

Three DiDs, Not Two
~~~~~~~~~~~~~~~~~~~

An important conceptual point is that the DDD estimand with covariates cannot be written as the difference
of two DiD estimands. It requires three DiD-type terms, each using a different comparison group:

1. Treated units vs. non-qualified units in the same enabling group
2. Treated units vs. qualified units in the comparison enabling group
3. Treated units vs. non-qualified units in the comparison enabling group

When all units are integrated over the treated units' covariate distribution, these three terms combine to
identify the :math:`ATT(g,t)`.

Combining Multiple Comparison Groups
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since any not-yet-treated cohort :math:`g_c > t` provides a valid comparison, the model is
over-identified. Rather than choosing a single comparison group, we can combine all available estimators to
improve precision.

For a given :math:`(g, t)` pair, let

.. math::

   \mathcal{G}_c^{g,t} = \{g_c \in \mathcal{S} : g_c > \max\{g, t\}\}

denote the set of valid comparison groups. Any weighted combination of estimators using different comparison
groups identifies the same :math:`ATT(g,t)`, as long as the weights sum to one.

Let :math:`\widehat{ATT}_{\text{dr}}(g,t)` be the vector of estimates using each valid comparison group,
and let :math:`\widehat{\Omega}_{g,t}` be their estimated variance-covariance matrix. The optimal
combination that minimizes asymptotic variance is

.. math::

   \widehat{ATT}_{\text{dr,gmm}}(g, t) =
   \frac{\mathbf{1}' \widehat{\Omega}_{g,t}^{-1}}
   {\mathbf{1}' \widehat{\Omega}_{g,t}^{-1} \mathbf{1}}
   \widehat{ATT}_{\text{dr}}(g, t).

This GMM-style estimator has an interpretation as an optimal generalized method of moments estimator based
on recentered influence functions. The weights are chosen to minimize variance subject to the constraint
that they sum to one.

In practice, using this combination can yield substantial precision gains. Monte Carlo simulations show that
confidence intervals based on the GMM combination can be more than 50% narrower than those using only the
never-treated comparison group. This is particularly valuable when the never-treated group is small.

Asymptotic Properties and Inference
-----------------------------------

Asymptotic Normality
~~~~~~~~~~~~~~~~~~~~

Under the identifying assumptions and standard regularity conditions, the DR DDD estimators are
asymptotically normal. For each :math:`(g, t)` pair and comparison group :math:`g_c`,

.. math::

   \sqrt{n}\left(\widehat{ATT}_{\text{dr}, g_c}(g, t) - ATT(g, t)\right)
   \xrightarrow{d} N(0, \Omega_{g,t,g_c}),

where the asymptotic variance :math:`\Omega_{g,t,g_c}` can be consistently estimated using influence
function methods.

The GMM-combined estimator achieves the minimum asymptotic variance among all weighted combinations

.. math::

   \Omega_{g,t,\text{gmm}} = \left(\mathbf{1}'
   \Omega_{g,t}^{-1} \mathbf{1}\right)^{-1} \leq \Omega_{g,t,g_c}

for any single comparison group :math:`g_c`.

Simultaneous Inference
~~~~~~~~~~~~~~~~~~~~~~

When examining multiple :math:`ATT(g,t)` parameters or event-study coefficients, pointwise confidence
intervals can be misleading due to multiple testing concerns. The methodology provides simultaneous
confidence bands that cover all parameters with a pre-specified probability.

These bands are constructed using a multiplier bootstrap procedure that accounts for the dependence across
different group-time estimates. The resulting simultaneous bands are wider than pointwise intervals but
provide valid coverage for the entire collection of parameters.

Event-Study Aggregation
-----------------------

While the group-time parameters :math:`ATT(g,t)` provide complete flexibility, researchers often want to
summarize results by event time. The event-study aggregation computes weighted averages across cohorts for
each elapsed time since treatment.

For post-treatment event times :math:`e \geq 0`, the event-study parameter is

.. math::

   ES(e) = \sum_{g \in \mathcal{G}_{\text{trt}}}
   \mathbb{P}(G = g \mid G + e \in [2, T]) \cdot ATT(g, g+e).

The natural estimator replaces each :math:`ATT(g, g+e)` with its DR GMM estimate and uses sample proportions
for the weights

.. math::

   \widehat{ES}_{\text{dr,gmm}}(e) = \sum_{g \in \mathcal{G}_{\text{trt}}}
   \mathbb{P}_n(G = g \mid G + e \in [2, T]) \cdot
   \widehat{ATT}_{\text{dr,gmm}}(g, g+e).

This estimator is asymptotically normal, and inference follows from the delta method combined with the
asymptotic theory for the underlying :math:`ATT(g,t)` estimates.

An overall summary parameter that averages across all post-treatment event times is

.. math::

   ES_{\text{avg}} = \frac{1}{N_E} \sum_{e \in \mathcal{E}} ES(e),

where :math:`\mathcal{E}` is the set of post-treatment event times and :math:`N_E` is its cardinality.

Pre-Treatment Testing
---------------------

The availability of pre-treatment periods allows assessment of the DDD parallel trends assumption. By
computing the DDD estimand for periods :math:`t < g`, researchers can check whether pre-treatment
"effects" are close to zero.

Pre-treatment event-study coefficients :math:`ES(e)` for :math:`e < 0` should equal zero under the
identification assumptions. Note that :math:`ES(-1) = 0` by construction since the baseline period is fixed
at :math:`g-1`. For earlier pre-treatment periods, significant coefficients suggest the identifying
assumption may be violated.

Event-study plots that include both pre-treatment and post-treatment periods provide visual diagnostics.
Under the identification assumptions, pre-treatment coefficients should be statistically
indistinguishable from zero, while post-treatment coefficients reveal the dynamic treatment effects.

When pre-treatment coefficients suggest potential violations, researchers can apply sensitivity analysis
methods following Rambachan and Roth (2023) to assess robustness of conclusions to departures from parallel
trends.

.. note::

   For complete theoretical details including proofs, regularity conditions, and additional Monte Carlo
   evidence, see the original paper by `Ortiz-Villavicencio and Sant'Anna (2025) <https://arxiv.org/abs/2505.09942>`_.
