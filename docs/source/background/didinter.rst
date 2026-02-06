.. _background-didinter:

DiD with Intertemporal Treatment Effects
========================================

The ``didinter`` module implements the difference-in-differences methodology for settings where lagged
treatments may affect current outcomes, based on the work of `de Chaisemartin and D'Haultfœuille (2024)
<https://arxiv.org/abs/2007.04267>`_. This approach addresses the challenges of estimating treatment effects
when the treatment is potentially non-binary, non-absorbing, and when treatment history matters for
current outcomes.

When Intertemporal Effects Matter
---------------------------------

Standard DiD methods typically assume that only contemporaneous treatment affects outcomes. However, many
real-world treatments have effects that persist and accumulate over time. A policy implemented today may
continue to affect outcomes for months or years afterward. When such dynamic effects are present, failing
to account for them can lead to biased estimates of treatment effects.

Consider a panel of groups observed over multiple time periods where treatment can vary in intensity and
can increase or decrease over time. Examples include state-level policy changes where intensity varies,
dosage effects in medical treatments, or regulatory changes that can be strengthened or relaxed. In these
settings, a group's outcome at time :math:`t` may depend not only on its current treatment but also on its
entire treatment history.

This methodology is particularly valuable when the treatment is non-binary (taking on multiple values
rather than just 0 or 1), non-absorbing (groups can leave treatment or have treatment intensity change in
either direction), or when there is reason to believe that treatment effects accumulate or decay over
time. The DID estimators developed in this framework are applicable to any design where some groups
maintain their initial treatment level for at least a few periods, providing valid comparison units.

Setup and Notation
------------------

We consider a panel of :math:`G` groups observed at :math:`T` time periods. Let :math:`D_{g,t}` denote
the treatment of group :math:`g` at period :math:`t`, where :math:`D_{g,t} \geq 0`. Groups may represent
states, counties, firms, or even individuals depending on the application. The treatment may be binary or
take on multiple values, and may increase or decrease over time.

Using the dynamic potential outcome framework, let :math:`Y_{g,t}(d_1, \ldots, d_t)` denote the potential
outcome of group :math:`g` at time :math:`t` if its treatments from period 1 to :math:`t` were equal to
:math:`(d_1, \ldots, d_t)`. This framework explicitly allows for the possibility that a group's outcome
at :math:`t` depends on its entire treatment history, not just its current treatment. The observed outcome
is

.. math::

   Y_{g,t} = Y_{g,t}(D_{g,1}, \ldots, D_{g,t}).

A key quantity is the first period at which a group's treatment changes from its initial value. Let

.. math::

   F_g = \min\{t : t \geq 2, D_{g,t} \neq D_{g,t-1}\}

denote the first treatment change for group :math:`g`, with the convention that :math:`F_g = T + 1` if
the treatment never changes. The DID estimators use groups whose treatment has not yet changed as
comparison groups for those whose treatment has changed.

Design Requirements
-------------------

The DID estimators apply to any design satisfying a minimal requirement on treatment variation.

.. admonition:: Design Restriction 1 (Common Baseline Treatment)

   There exist groups :math:`g` and :math:`g'` such that :math:`D_{g,1} = D_{g',1}` and :math:`F_g \neq
   F_{g'}`.

This restriction has two parts. First, there must exist at least two groups with the same period-one
treatment. Second, among groups with the same baseline treatment, there must be variation in when they
first change treatment. The restriction rules out designs where treatment is extremely non-persistent
(all groups change in period 2) or where there is a universal treatment change affecting all groups
simultaneously.

Several common designs automatically satisfy this requirement. Binary staggered designs where all groups
start untreated and some eventually receive treatment satisfy the restriction. Designs with group-specific
treatment intensities where all groups start at zero also satisfy it. More complex designs with non-zero
baseline treatments and bidirectional treatment changes can also satisfy the requirement as long as some
groups share the same initial treatment level.

A second design restriction rules out cases where groups cross their baseline treatment in both
directions.

.. admonition:: Design Restriction 2 (No Crossing)

   For all groups :math:`g`, either :math:`D_{g,t} \geq D_{g,1}` for all :math:`t`, or :math:`D_{g,t}
   \leq D_{g,1}` for all :math:`t`.

This restriction ensures that treatment effects have a clear interpretation. If a group experiences both
higher and lower treatments than its baseline, the resulting effect parameter can be written as a
difference between effects of increasing and decreasing treatment, which may have opposite signs. This
makes interpretation difficult and violates a "no sign reversal" property. When this restriction fails in
the data, one can simply exclude the problematic observations and apply the DID estimators to the
remaining sample.

Identifying Assumptions
-----------------------

Identification relies on two key assumptions that generalize standard DiD assumptions to the dynamic
setting.

.. admonition:: Assumption 1 (No Anticipation)

   A group's current outcome does not depend on its future treatments. For all groups :math:`g` and all
   treatment sequences :math:`(d_1, \ldots, d_T)`,

   .. math::

      Y_{g,t}(d_1, \ldots, d_T) = Y_{g,t}(d_1, \ldots, d_t).

This assumption rules out anticipatory behavior where units change their outcomes in response to expected
future treatment changes. If treatment changes are announced in advance, the treatment timing should be
redefined accordingly.

.. admonition:: Assumption 2 (Parallel Trends for Same Baseline Treatment)

   Groups with the same period-one treatment have the same expected evolution of their status-quo
   potential outcome. If :math:`D_{g,1} = D_{g',1}`, then for all :math:`t \geq 2`,

   .. math::

      \begin{aligned}
      &\mathbb{E}[Y_{g,t}(D_{g,1}, \ldots, D_{g,1}) - Y_{g,t-1}(D_{g,1}, \ldots, D_{g,1}) \mid
      \boldsymbol{D}] \\
      &= \mathbb{E}[Y_{g',t}(D_{g',1}, \ldots, D_{g',1}) - Y_{g',t-1}(D_{g',1}, \ldots, D_{g',1})
      \mid \boldsymbol{D}].
      \end{aligned}

The status-quo outcome :math:`Y_{g,t}(D_{g,1}, \ldots, D_{g,1})` is the counterfactual outcome that would
have been observed if the group had maintained its period-one treatment throughout. The assumption
requires that this counterfactual outcome evolves in parallel across groups with the same baseline
treatment.

This assumption is weaker than requiring parallel trends across all groups regardless of their baseline
treatment. Comparing groups with different baseline treatments would require ruling out both dynamic
effects and time-varying treatment effects, which is typically implausible. By restricting comparisons to
groups with the same baseline treatment, this identification strategy accommodates both dynamic effects
and heterogeneous effects across groups.

Parameters of Interest
----------------------

The fundamental parameter is the actual-versus-status-quo (AVSQ) effect, which compares a group's actual
outcome to what it would have been under the status-quo counterfactual of maintaining the period-one
treatment.

Actual-Versus-Status-Quo Effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a group :math:`g` whose treatment first changes at period :math:`F_g`, the AVSQ effect at
:math:`\ell` periods after :math:`F_g - 1` is

.. math::

   \delta_{g,\ell} = \mathbb{E}[Y_{g,F_g-1+\ell} - Y_{g,F_g-1+\ell}(D_{g,1}, \ldots, D_{g,1}) \mid
   \boldsymbol{D}].

This parameter captures the expected difference between the group's actual outcome and its status-quo
outcome at period :math:`F_g - 1 + \ell`. When :math:`\ell = 1`, this is the effect one period after
the first treatment change. When :math:`\ell = 2`, it is the effect two periods after, and so on.

The AVSQ effect captures the combined impact of all treatment changes from period :math:`F_g` through
period :math:`F_g - 1 + \ell`. In simple designs where treatment changes only once, interpretation is
straightforward. In more complex designs where treatment continues to change, the AVSQ effect captures
the cumulative impact of the entire treatment trajectory relative to the status quo.

Event-Study Effects
~~~~~~~~~~~~~~~~~~~

To summarize results across groups, we aggregate the group-specific effects into event-study parameters.
Let :math:`S_g = 1` if the treatment increases at the first change (:math:`D_{g,F_g} > D_{g,1}`) and
:math:`S_g = -1` if it decreases (:math:`D_{g,F_g} < D_{g,1}`). The event-study effect at event time
:math:`\ell` is

.. math::

   \delta_\ell = \frac{1}{N_\ell} \sum_{g: F_g - 1 + \ell \leq T_g} S_g \delta_{g,\ell},

where :math:`N_\ell` is the number of groups for which :math:`\delta_{g,\ell}` can be estimated, and
:math:`T_g` is the last period where valid comparison groups exist for group :math:`g`.

Multiplying by :math:`S_g` ensures that :math:`\delta_\ell` can be interpreted as an average effect
of having been exposed to a weakly higher treatment dose for :math:`\ell` periods. For groups whose
treatment increased, their effect enters positively. For groups whose treatment decreased, their effect
is negated so that the overall parameter still captures the effect of higher treatment.

Estimation
----------

The :math:`\text{DID}_{g,\ell}` and :math:`\text{DID}_\ell` estimators compare the outcome evolution of
groups that change treatment to groups that have not yet changed and share the same baseline treatment.

Group-Specific Estimator
~~~~~~~~~~~~~~~~~~~~~~~~

For group :math:`g` at event time :math:`\ell`, the DID estimator is

.. math::

   \text{DID}_{g,\ell} = Y_{g,F_g-1+\ell} - Y_{g,F_g-1} - \frac{1}{N_{F_g-1+\ell}^g} \sum_{g': D_{g',1}
   = D_{g,1}, F_{g'} > F_g-1+\ell} (Y_{g',F_g-1+\ell} - Y_{g',F_g-1}),

where :math:`N_{F_g-1+\ell}^g` is the number of groups with the same baseline treatment as :math:`g` that
have not changed treatment by period :math:`F_g - 1 + \ell`.

This estimator compares the :math:`(F_g - 1)`-to-:math:`(F_g - 1 + \ell)` outcome change for group
:math:`g` against the average outcome change for groups with the same baseline treatment that have not
yet experienced any treatment change. Under the identifying assumptions, this comparison identifies the
causal effect :math:`\delta_{g,\ell}`.

Event-Study Estimator
~~~~~~~~~~~~~~~~~~~~~

Aggregating across groups yields the event-study estimator

.. math::

   \text{DID}_\ell = \frac{1}{N_\ell} \sum_{g: F_g - 1 + \ell \leq T_g} S_g \text{DID}_{g,\ell}.

Under assumptions 1 and 2, this estimator is unbiased for the event-study effect :math:`\delta_\ell`.
The estimator can be computed for any :math:`\ell` from 1 up to the maximum horizon where valid
comparison groups exist.

Normalized Effects
------------------

While the event-study effects :math:`\delta_\ell` provide reduced-form evidence on treatment effects,
they can be difficult to interpret in complex designs where treatment trajectories vary across groups.
The framework addresses this by defining normalized versions of these parameters.

Definition of Normalized Effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let

.. math::

   \delta_{g,\ell}^D = \sum_{k=0}^{\ell-1} (D_{g,F_g+k} - D_{g,1})

denote the total treatment dose received by group :math:`g` from :math:`F_g` to :math:`F_g - 1 + \ell`
relative to the status-quo counterfactual. The normalized AVSQ effect is

.. math::

   \delta_{g,\ell}^n = \frac{\delta_{g,\ell}}{\delta_{g,\ell}^D}.

The normalized event-study effect is a weighted average across groups

.. math::

   \delta_\ell^n = \frac{\delta_\ell}{\delta_\ell^D},

where :math:`\delta_\ell^D = \frac{1}{N_\ell} \sum_{g: F_g - 1 + \ell \leq T_g} |\delta_{g,\ell}^D|`.

Interpretation as Average of Lag Effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The normalized effect has a structural interpretation as a weighted average of the effects of different
treatment lags on the outcome. Specifically,

.. math::

   \delta_{g,\ell}^n = \sum_{k=0}^{\ell-1} w_{g,\ell,k} s_{g,\ell,k},

where :math:`s_{g,\ell,k}` is the slope of the potential outcome function with respect to the
:math:`k`-th treatment lag, and

.. math::

   w_{g,\ell,k} = \frac{D_{g,F_g-1+\ell-k} - D_{g,1}}{\delta_{g,\ell}^D}

are weights that sum to one and are non-negative under Design Restriction 2.

In binary staggered designs, the normalized effect simplifies to the simple average of the effects of the
current treatment and its :math:`\ell - 1` first lags. In designs with group-specific treatment
intensities, the normalized effect averages the effects of different lags, with each lag's effect scaled
by the treatment intensity. This interpretation makes :math:`\delta_\ell^n` more comparable across
different values of :math:`\ell` than the non-normalized :math:`\delta_\ell`.

Testing for Constant Effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The normalized effects can be used to test whether the current and lagged treatments have the same effect
on outcomes. If effects are constant across lags, then :math:`\ell \mapsto \delta_\ell^n` should be
constant. A test of this null hypothesis provides evidence on whether treatment effects are stable or
dynamic over time.

Cost-Benefit Analysis
---------------------

Beyond event-study parameters, the framework defines a cost-benefit parameter useful for policy
evaluation. Consider a planner comparing the actual treatment allocation to a counterfactual where all
groups maintained their period-one treatment. This parameter is

.. math::

   \delta = \frac{\sum_{g: F_g \leq T_g} \sum_{\ell=1}^{T_g - F_g + 1} \delta_{g,\ell}}{\sum_{g: F_g
   \leq T_g} \sum_{\ell=1}^{T_g - F_g + 1} (D_{g,F_g-1+\ell} - D_{g,1})}.

This parameter represents the average benefit per unit of treatment administered relative to the status
quo. If the treatment cost per unit is :math:`c`, then the treatment changes were beneficial in monetary
terms if :math:`\delta > c`.

The parameter :math:`\delta` has an interpretation as an average total effect per unit of treatment. The
numerator sums all the outcome gains (or losses) from treatment changes across all groups and time
periods. The denominator sums all the incremental treatment doses administered. The ratio gives the
average return per unit of treatment, accounting for both immediate and delayed effects.

The cost-benefit parameter connects to the event-study effects through the relation

.. math::

   \delta = \sum_{\ell=1}^{L} w_\ell \delta_\ell,

where the weights :math:`w_\ell` are non-negative. This shows that :math:`\delta` is a weighted average
of the event-study effects.

Pre-Treatment Testing
---------------------

The identifying assumptions have testable implications that can be assessed using placebo estimators. For
a group :math:`g` with :math:`F_g \geq 3`, we can compute

.. math::

   \text{DID}_{g,\ell}^{pl} = Y_{g,F_g-1-\ell} - Y_{g,F_g-1} - \frac{1}{N_{F_g-1+\ell}^g} \sum_{g':
   D_{g',1} = D_{g,1}, F_{g'} > F_g-1+\ell} (Y_{g',F_g-1-\ell} - Y_{g',F_g-1}).

This placebo estimator mimics the actual estimator but compares outcome changes in the pre-treatment
period, from :math:`F_g - 1 - \ell` to :math:`F_g - 1`, before group :math:`g`'s treatment changes.
Under the identifying assumptions, the expected value of this placebo is zero. Significant pre-treatment
effects suggest potential violations of parallel trends.

The placebo estimators assess whether groups that will change treatment at different times have similar
outcome trends before any treatment changes occur. This tests the same parallel trends assumption over
the same time horizon that is required for :math:`\text{DID}_{g,\ell}` and :math:`\text{DID}_\ell` to be
unbiased.

Why Standard Approaches Fail
----------------------------

The :math:`\text{DID}_\ell` and :math:`\text{DID}_\ell^n` estimators provide valid inference where
several commonly-used approaches can produce misleading results.

Two-Way Fixed Effects with Treatment Intensity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In designs without variation in treatment timing, researchers often estimate TWFE regressions with
treatment intensity interacted with period fixed effects. While intuitive, these regressions can assign
negative weights to some treatment effects. The coefficient :math:`\hat{\beta}_{fe,\ell}` from such
regressions identifies

.. math::

   \mathbb{E}[\hat{\beta}_{fe,\ell} \mid \boldsymbol{D}] = \sum_{g: I_g \neq 0} w_g^{fe}
   \frac{\delta_{g,\ell}}{I_g},

where :math:`I_g` is group :math:`g`'s treatment intensity and the weights :math:`w_g^{fe}` can be
negative for groups with intensity below the mean. When weights are negative, the estimator can have the
wrong sign even when all underlying effects are positive.

Local-Projection Panel Regressions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Local-projection regressions of :math:`Y_{g,t-1+\ell}` on :math:`D_{g,t}` with group and period fixed
effects suffer from multiple problems. First, the coefficient :math:`\hat{\beta}_{lp,\ell}` is
contaminated by effects of other exposure lengths. Effects supposed to measure :math:`\ell` periods of
exposure are actually mixtures of effects from different exposure durations.

Second, some weights in the decomposition of :math:`\hat{\beta}_{lp,\ell}` are always negative for
:math:`\ell \geq 2` in binary staggered designs. Third, and perhaps most strikingly, the weights can sum
to less than one or even to a negative number. This means that even with perfectly constant treatment
effects, local-projection coefficients can be biased toward zero or even have the wrong sign. The
regression is fundamentally misspecified because it treats groups with :math:`D_{g,t} = 0` as untreated
controls when some of them may become treated before period :math:`t - 1 + \ell`.

Distributed-Lag Regressions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Distributed-lag regressions of :math:`Y_{g,t}` on the current treatment and its first :math:`K` lags with
group and period fixed effects also face problems. The coefficient on the :math:`l`-th lag estimates a
weighted sum where weights may be negative. Additionally, even under the strong assumption that the
functional form is correctly specified (only the first :math:`K` lags matter and they enter additively),
each coefficient is contaminated by effects of other lags whenever treatment effects are heterogeneous
across groups or time periods.

Asymptotic Properties and Inference
-----------------------------------

Asymptotic Normality
~~~~~~~~~~~~~~~~~~~~

Under standard regularity conditions (independent groups, bounded moments, non-degenerate variances), the
DID estimators are asymptotically normal. For each :math:`\ell`,

.. math::

   \sqrt{N_\ell}(\text{DID}_\ell - \delta_\ell) \xrightarrow{d} N(0, \sigma_\ell^2),

where the asymptotic variance :math:`\sigma_\ell^2` can be consistently estimated using cohort-specific
variance estimators.

Confidence Intervals
~~~~~~~~~~~~~~~~~~~~

Confidence intervals of the form

.. math::

   CI_{1-\alpha} = \left[\text{DID}_\ell \pm z_{1-\alpha/2}
   \frac{\hat{\sigma}_\ell}{\sqrt{N_\ell}}\right]

are asymptotically valid. In general, inference is conservative due to the heterogeneity across groups in
the i.n.i.d. (independent but not identically distributed) setup. When groups are identically distributed
and the treatment path is determined by the baseline treatment and switching behavior, the confidence
intervals achieve their nominal coverage asymptotically.

Extensions
----------

Several extensions to the basic framework are available.

**Covariates.** The DID estimators accommodate time-varying covariates by replacing the standard
parallel trends assumption with a version that allows differential trends fully explained by covariate
changes. The estimators adjust outcomes for covariate effects before computing the comparisons.

**Group-Specific Trends.** When units may have different underlying trends, the DID estimators can be
extended to allow for group-specific linear trends, estimated from the pre-treatment period.

**Heterogeneous Effects.** Treatment effects can be estimated separately for subgroups defined by
time-invariant covariates, allowing researchers to examine treatment effect heterogeneity across different
types of units.

**Fuzzy Designs.** When treatment varies within groups (e.g., at the individual level within states),
the DID estimators extend to fuzzy designs using appropriate aggregation.

.. note::

   For complete theoretical details including proofs, regularity conditions, and additional extensions,
   see the original paper by `de Chaisemartin and D'Haultfœuille (2024)
   <https://arxiv.org/abs/2007.04267>`_.
