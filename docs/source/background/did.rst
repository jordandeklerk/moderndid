.. _background-did:

DiD with Multiple Time Periods
==============================

The ``did`` module implements the difference-in-differences (DiD) methodology for settings with multiple time periods and variation in treatment timing from the work of `Callaway and Sant'Anna (2020) <https://psantanna.com/files/Callaway_SantAnna_2020.pdf>`_.
This approach addresses the challenges of staggered DiD designs by providing flexible estimators for group-time average treatment effects and various aggregation schemes to summarize treatment effect heterogeneity.

Setup and Notation
------------------

We consider a setup with :math:`\mathcal{T}` time periods. Let :math:`D_{it}` be a binary variable indicating if unit :math:`i` is treated in period :math:`t`.
The treatment adoption process follows two key assumptions:

* **No treatment in the first period.** :math:`D_{i1} = 0` for all units.
* **Irreversibility of Treatment (Staggered Adoption).** Once a unit is treated, it remains treated. Formally, :math:`D_{it-1} = 1` implies :math:`D_{it} = 1`.

Let :math:`G_i` be the time period when unit :math:`i` is first treated. If a unit is never treated, we set :math:`G_i = \infty`.
Units are thus partitioned into groups based on their treatment adoption time. Let :math:`C_i` be an indicator for units that are never treated (:math:`G_i = \infty`).

We use the potential outcomes framework and let :math:`Y_{it}(g)` be the potential outcome for unit :math:`i` at time :math:`t` if it were first treated in period :math:`g`.
The potential outcome under no treatment is :math:`Y_{it}(0)`. The observed outcome is a combination of these potential outcomes, determined by the group to which unit :math:`i` belongs

.. math::

   Y_{it} = Y_{it}(0) + \sum_{g=2}^{\mathcal{T}} (Y_{it}(g) - Y_{it}(0)) \cdot \mathbf{1}\{G_i = g\}.

The Group-Time Average Treatment Effect
----------------------------------------

A key insight of the Callaway and Sant'Anna approach is that rather than estimating a single aggregate treatment effect, we can instead target more disaggregated parameters that capture treatment effect heterogeneity. In staggered adoption designs, effects may differ both across groups (early versus late adopters may respond differently to treatment) and over time (effects may grow, fade, or evolve as units accumulate exposure). The group-time framework provides the flexibility to capture all of these patterns.

The fundamental parameter of interest is the **group-time average treatment effect**, :math:`ATT(g, t)`, which is the average treatment effect for group :math:`g` at time :math:`t` given by

.. math::

   ATT(g, t) = \mathbb{E}[Y_t(g) - Y_t(0) | G = g].

This parameter is flexible and does not impose homogeneity across groups or time. The set of all :math:`ATT(g, t)`'s can be used to understand treatment effect dynamics and heterogeneity.

Identifying Assumptions
-----------------------

Like all causal inference methods, identification of treatment effects from observational data requires assumptions. The DiD framework replaces the strong assumption of unconfoundedness (that treatment assignment is as good as random conditional on observables) with assumptions about how outcomes would have evolved in the absence of treatment. The key assumptions below formalize conditions under which we can use comparison groups to construct valid counterfactuals for what would have happened to treated units had they not been treated.

The identification of :math:`ATT(g, t)` relies on the following key assumptions.

.. admonition:: Assumption 1 (Limited Treatment Anticipation)

   Potential outcomes are not affected by the treatment in periods far enough before it is implemented. For a known anticipation horizon :math:`\delta \ge 0`,

   .. math::

      \mathbb{E}[Y_t(g) | X, G_g = 1] = \mathbb{E}[Y_t(0) | X, G_g = 1] \quad \text{for all } t < g - \delta.

When :math:`\delta = 0`, this is a "no anticipation" assumption.

.. admonition:: Assumption 2 (Conditional Parallel Trends)

   The average evolution of untreated potential outcomes is the same for a treatment group and a comparison group, conditional on a set of pre-treatment covariates :math:`X`. Two alternative formulations are available.

   *Based on a "Never-Treated" Group.* For each group :math:`g` and for periods :math:`t \ge g - \delta`,

   .. math::

      \mathbb{E}[Y_t(0) - Y_{t-1}(0) | X, G_g = 1] = \mathbb{E}[Y_t(0) - Y_{t-1}(0) | X, C = 1].

   *Based on "Not-Yet-Treated" Groups.* For each group :math:`g` and for periods :math:`t \ge g - \delta`,

   .. math::

      \mathbb{E}[Y_t(0) - Y_{t-1}(0) | X, G_g = 1] = \mathbb{E}[Y_t(0) - Y_{t-1}(0) | X, D_s = 0, G_g = 0],

   where :math:`s` is a time period such that :math:`t + \delta \le s`.

.. admonition:: Assumption 3 (Overlap)

   For any given covariates, there is a positive probability of being in a treatment group and in the comparison group. Formally, for some :math:`\varepsilon > 0`, :math:`P(G_g = 1) > \varepsilon` and the generalized propensity score

   .. math::

      p_{g,t}(X) = P(G_g = 1 | X, G_g + (1 - D_t)(1 - G_g) = 1)

   is bounded away from 1.

Nonparametric Identification of ATT(g,t)
-----------------------------------------

Under the assumptions above, :math:`ATT(g, t)` is non-parametrically identified. The paper provides three types of estimands that can be used to identify these effects, each with different strengths and properties. For what follows, let :math:`\Delta Y_{t,g,\delta} = Y_t - Y_{g-\delta-1}` denote the change in outcomes from the pre-treatment base period to the current period.

Never-Treated Comparison Group Estimands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using never-treated units as the comparison group, we first define the following key quantities. The propensity score for being in group :math:`g` conditional on being either in group :math:`g` or never-treated is

.. math::

   p_g(X) = P(G_g = 1 | X, G_g + C = 1),

and the expected outcome change for never-treated units is

.. math::

   m_{g,t,\delta}^{nev}(X) = \mathbb{E}[\Delta Y_{t,g,\delta} | X, C = 1].

**Inverse Probability Weighting (IPW) Estimand**

The IPW estimand reweights observations to balance the covariate distributions between the treatment and comparison groups and is given by

.. math::

   ATT_{ipw}^{nev}(g, t; \delta) = \mathbb{E}\left[\left(\frac{G_g}{\mathbb{E}[G_g]} - \frac{\frac{p_g(X) C}{1 - p_g(X)}}{\mathbb{E}\left[\frac{p_g(X) C}{1 - p_g(X)}\right]}\right) \Delta Y_{t,g,\delta} \right].

This estimand is consistent when the propensity score model is correctly specified.

**Outcome Regression (OR) Estimand**

The OR estimand uses regression adjustment to control for differences in covariates and is given by

.. math::

   ATT_{or}^{nev}(g, t; \delta) = \mathbb{E}\left[\frac{G_g}{\mathbb{E}[G_g]} \left( \Delta Y_{t,g,\delta} - m_{g,t,\delta}^{nev}(X) \right) \right].

This approach is consistent when the outcome regression model is correctly specified.

**Doubly Robust (DR) Estimand**

The DR estimand combines both IPW and OR approaches, providing consistency if either the propensity score or outcome regression model is correctly specified, but not necessarily both. The DR estimand is given by

.. math::

   ATT_{dr}^{nev}(g, t; \delta) = \mathbb{E}\left[\left(\frac{G_g}{\mathbb{E}[G_g]} - \frac{\frac{p_g(X) C}{1 - p_g(X)}}{\mathbb{E}\left[\frac{p_g(X) C}{1 - p_g(X)}\right]}\right) \left( \Delta Y_{t,g,\delta} - m_{g,t,\delta}^{nev}(X) \right) \right].

This estimand offers the best of both worlds, providing robustness against model mis-specification and improved efficiency properties.

Not-Yet-Treated Comparison Group Estimands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using not-yet-treated units as the comparison group, we work with different propensity score and outcome regression functions given by

.. math::

   p_{g,t+\delta}(X) = P(G_g = 1 | X, G_g + (1 - D_{t+\delta})(1 - G_g) = 1)

and

.. math::

   m_{g,t,\delta}^{ny}(X) = \mathbb{E}[\Delta Y_{t,g,\delta} | X, D_{t+\delta} = 0, G_g = 0].

**Inverse Probability Weighting (IPW) Estimand**

The IPW estimand for the not-yet-treated comparison adapts the weighting scheme to account for units that have not been treated by time :math:`t + \delta` and is given by

.. math::

   ATT_{ipw}^{ny}(g, t; \delta) = \mathbb{E}\left[\left(\frac{G_g}{\mathbb{E}[G_g]} - \frac{\frac{p_{g,t+\delta}(X)(1 - D_{t+\delta})(1 - G_g)}{1 - p_{g,t+\delta}(X)}}{\mathbb{E}\left[\frac{p_{g,t+\delta}(X)(1 - D_{t+\delta})(1 - G_g)}{1 - p_{g,t+\delta}(X)}\right]}\right) \Delta Y_{t,g,\delta} \right].

**Outcome Regression (OR) Estimand**

The OR estimand adjusts for differences using the expected outcomes of not-yet-treated units is given by

.. math::

   ATT_{or}^{ny}(g, t; \delta) = \mathbb{E}\left[\frac{G_g}{\mathbb{E}[G_g]} \left( \Delta Y_{t,g,\delta} - m_{g,t,\delta}^{ny}(X) \right) \right].

**Doubly Robust (DR) Estimand**

The DR estimand for not-yet-treated comparisons combines both approaches and is given by

.. math::

   ATT_{dr}^{ny}(g, t; \delta) = \mathbb{E}\left[\left(\frac{G_g}{\mathbb{E}[G_g]} - \frac{\frac{p_{g,t+\delta}(X)(1 - D_{t+\delta})(1 - G_g)}{1 - p_{g,t+\delta}(X)}}{\mathbb{E}\left[\frac{p_{g,t+\delta}(X)(1 - D_{t+\delta})(1 - G_g)}{1 - p_{g,t+\delta}(X)}\right]}\right) \left( \Delta Y_{t,g,\delta} - m_{g,t,\delta}^{ny}(X) \right) \right].

The choice between never-treated and not-yet-treated comparison groups depends on the specific empirical context. Never-treated comparisons may be more stable but require the existence of a sufficiently large never-treated group. Not-yet-treated comparisons can utilize more data but may be less appropriate when treatment timing is endogenous.

Unconditional Estimands
~~~~~~~~~~~~~~~~~~~~~~~

When pre-treatment covariates play no role in identification (i.e., the parallel trends assumption holds unconditionally on :math:`X`), the estimands simplify considerably. For the never-treated comparison group, the unconditional estimand is

.. math::

   ATT_{unc}^{nev}(g, t; \delta) = \mathbb{E}[Y_t - Y_{g-\delta-1} | G_g = 1] - \mathbb{E}[Y_t - Y_{g-\delta-1} | C = 1].

For the not-yet-treated comparison group, the unconditional estimand is

.. math::

   ATT_{unc}^{ny}(g, t; \delta) = \mathbb{E}[Y_t - Y_{g-\delta-1} | G_g = 1] - \mathbb{E}[Y_t - Y_{g-\delta-1} | D_{t+\delta} = 0].

These expressions clearly resemble the canonical two-period, two-group DiD estimand. The average effect for group :math:`g` is identified by comparing the outcome path experienced by that group to the path experienced by the comparison group. Under parallel trends, this latter path represents the counterfactual outcome path that group :math:`g` would have experienced without treatment.

Doubly Robust Estimation
~~~~~~~~~~~~~~~~~~~~~~~~

Although the IPW, OR, and DR estimands are equivalent from an identification standpoint, the DR approach has important advantages for estimation and inference. The DR estimators are consistent if *either* the propensity score model *or* the outcome regression model is correctly specified, but not necessarily both. This double robustness provides important protection against model misspecification.

Additionally, DR estimators allow for the use of flexible estimation methods, including those involving regularization and model selection, making them particularly attractive when the number of covariates is moderate or large relative to the sample size.

Aggregation of Effects
----------------------

While the group-time average treatment effects :math:`ATT(g, t)` provide a complete characterization of treatment effect heterogeneity, the number of such parameters can be large in applications with many groups and time periods. Researchers often want to summarize these effects to answer specific policy questions. For example, do effects grow or fade over time? Do early adopters experience different effects than late adopters? What is the overall average effect of the policy?

A key feature of this methodology is the ability to aggregate the :math:`ATT(g, t)`'s into meaningful summary measures. This allows researchers to answer specific policy questions and understand different dimensions of treatment effect heterogeneity.

Event-Study Aggregation
~~~~~~~~~~~~~~~~~~~~~~~

Event-study plots aggregate effects by length of exposure to treatment, where :math:`e = t - g` represents the time elapsed since treatment adoption. This aggregation reveals how treatment effects evolve dynamically after implementation. The event-study parameter is

.. math::

   \theta_{es}(e) = \sum_{g \in \mathcal{G}} \mathbf{1}\{g + e \le \mathcal{T}\} P(G = g | G + e \le \mathcal{T}) ATT(g, g + e).

This parameter weights the group-time effects by the relative size of each group among those observed :math:`e` periods after treatment, providing insights into whether effects strengthen, weaken, or remain stable over time.

**Compositional Changes and Balanced Event-Study**

When comparing :math:`\theta_{es}(e)` across different values of :math:`e`, one must be aware that compositional changes can complicate interpretation. For example, comparing :math:`\theta_{es}(e_2)` and :math:`\theta_{es}(e_1)` includes not only the dynamic effect of treatment but also two additional terms arising from different compositions of groups at different event times.

To address this, the paper proposes a "balanced" event-study parameter that uses a fixed set of groups across all event times:

.. math::

   \theta_{es}^{bal}(e; e') = \sum_{g \in \mathcal{G}} \mathbf{1}\{g + e' \le \mathcal{T}\} ATT(g, g + e) P(G = g | G + e' \le \mathcal{T}).

This calculates the average treatment effect for units whose event time is :math:`e` among those observed for at least :math:`e'` periods. Since the composition of groups is the same across all values of :math:`e \le e'`, differences in :math:`\theta_{es}^{bal}(e; e')` across event times cannot be attributed to compositional changes. The trade-off is that fewer groups are used, potentially leading to less precise inference.

Group-Specific Effects
~~~~~~~~~~~~~~~~~~~~~~

To understand whether treatment timing matters, we can average effects over time for each group. This allows us to understand whether early adopters experience different treatment effects compared to late adopters. For a specific group :math:`\tilde{g}`, the average effect is

.. math::

   \theta_{sel}(\tilde{g}) = \frac{1}{\mathcal{T} - \tilde{g} + 1} \sum_{t=\tilde{g}}^{\mathcal{T}} ATT(\tilde{g}, t).

This measure helps identify whether there are advantages or disadvantages to adopting treatment earlier versus later in the sample period.

Calendar-Time Effects
~~~~~~~~~~~~~~~~~~~~~

Calendar-time aggregation averages effects across all treated groups for each time period, revealing how treatment effects vary with time-specific factors such as macroeconomic conditions or concurrent policy changes. For a specific time period :math:`\tilde{t}`, the calendar-time effect is

.. math::

   \theta_{c}(\tilde{t}) = \sum_{g \in \mathcal{G}} \mathbf{1}\{\tilde{t} \ge g\} P(G = g | G \le \tilde{t}) ATT(g, \tilde{t}).

This aggregation weights each group's contribution by its relative size among all groups treated by time :math:`\tilde{t}`.

Overall Average Treatment Effect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a single summary measure is needed, we can compute an overall average that aggregates across all groups and post-treatment time periods. One such measure weights group-specific effects by the distribution of treatment timing

.. math::

   \theta_{sel}^O = \sum_{g \in \mathcal{G}} \theta_{sel}(g) P(G = g | G \le \mathcal{T}).

This provides a single number summarizing the average treatment effect across the entire treated population, properly accounting for the staggered adoption pattern.

These aggregations provide transparent and interpretable ways to summarize treatment effect heterogeneity, with researcher-specified non-negative weights that directly reflect the policy questions of interest.

Inference and Pre-Treatment Testing
-----------------------------------

Asymptotic Properties
~~~~~~~~~~~~~~~~~~~~~

The DR estimators for :math:`ATT(g, t)` are asymptotically normal. Under the doubly robust condition (either the propensity score or outcome regression model correctly specified), the estimators admit an influence function representation

.. math::

   \sqrt{n}(\widehat{ATT}_{dr}(g, t) - ATT(g, t)) = \frac{1}{\sqrt{n}} \sum_{i=1}^{n} \psi_{g,t}(W_i) + o_p(1),

where :math:`\psi_{g,t}` is the influence function. This representation enables straightforward computation of standard errors and forms the basis for bootstrap inference procedures.

Simultaneous Confidence Bands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The paper proposes a multiplier bootstrap procedure for constructing simultaneous confidence bands that cover all :math:`ATT(g, t)` with probability :math:`1 - \alpha`. Unlike pointwise confidence intervals, simultaneous bands account for the dependency across different group-time average treatment effect estimators and avoid multiple testing problems. This is particularly important when visualizing the overall estimation uncertainty across all group-time effects.

Pre-Treatment Placebo Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Although the limited anticipation assumption implies :math:`ATT(g, t) = 0` for all :math:`t < g - \delta`, it is common practice to estimate these pre-treatment parameters and use them to assess the credibility of the parallel trends assumption. If the estimated pre-treatment effects are significantly different from zero, this provides evidence against the identifying assumptions. The DiD estimands can be adjusted to include pre-treatment periods by replacing the "long differences" :math:`(Y_t - Y_{g-\delta-1})` with "short differences" :math:`(Y_t - Y_{t-1})` for :math:`t < g - \delta`.

.. note::

   For the full theoretical details, including efficiency bounds, asymptotic properties, and the multiplier bootstrap algorithm, please refer to the original paper by `Callaway and Sant'Anna (2020) <https://psantanna.com/files/Callaway_SantAnna_2020.pdf>`_.
