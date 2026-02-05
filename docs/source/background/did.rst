.. _background-did:

DiD with Multiple Time Periods
==============================

The ``did`` module implements the difference-in-differences (DiD) methodology for settings with multiple time periods and variation in treatment timing from the work of `Callaway and Sant'Anna (2020) <https://psantanna.com/files/Callaway_SantAnna_2020.pdf>`_.
This approach addresses the challenges of staggered DiD designs by providing flexible estimators for group-time average treatment effects and various aggregation schemes to summarize treatment effect heterogeneity.

Setup and Notation
------------------

We consider a setup with :math:`\mathcal{T}` time periods. Let :math:`D_{it}` be a binary variable indicating if unit :math:`i` is treated in period :math:`t`.
The treatment adoption process follows two key assumptions:

* **No treatment in the first period**: :math:`D_{i1} = 0` for all units.
* **Irreversibility of Treatment (Staggered Adoption)**: Once a unit is treated, it remains treated. Formally, :math:`D_{it-1} = 1` implies :math:`D_{it} = 1`.

Let :math:`G_i` be the time period when unit :math:`i` is first treated. If a unit is never treated, we set :math:`G_i = \infty`.
Units are thus partitioned into groups based on their treatment adoption time. Let :math:`C_i` be an indicator for units that are never treated (:math:`G_i = \infty`).

We use the potential outcomes framework and let :math:`Y_{it}(g)` be the potential outcome for unit :math:`i` at time :math:`t` if it were first treated in period :math:`g`.
The potential outcome under no treatment is :math:`Y_{it}(0)`. The observed outcome is a combination of these potential outcomes, determined by the group to which unit :math:`i` belongs

.. math::

   Y_{it} = Y_{it}(0) + \sum_{g=2}^{\mathcal{T}} (Y_{it}(g) - Y_{it}(0)) \cdot \mathbf{1}\{G_i = g\}.

The Group-Time Average Treatment Effect
----------------------------------------

The fundamental parameter of interest is the **group-time average treatment effect**, :math:`ATT(g, t)`, which is the average treatment effect for group :math:`g` at time :math:`t` given by

.. math::

   ATT(g, t) = \mathbb{E}[Y_t(g) - Y_t(0) | G = g].

This parameter is flexible and does not impose homogeneity across groups or time. The set of all :math:`ATT(g, t)`'s can be used to understand treatment effect dynamics and heterogeneity.

Identifying Assumptions
-----------------------

The identification of :math:`ATT(g, t)` relies on the following key assumptions.

.. admonition:: Assumption 1 (Limited Treatment Anticipation)

   Potential outcomes are not affected by the treatment in periods far enough before it is implemented. For a known anticipation horizon :math:`\delta \ge 0`,

   .. math::

      \mathbb{E}[Y_t(g) | X, G_g = 1] = \mathbb{E}[Y_t(0) | X, G_g = 1] \quad \text{for all } t < g - \delta.

When :math:`\delta = 0`, this is a "no anticipation" assumption.

.. admonition:: Assumption 2 (Conditional Parallel Trends)

   The average evolution of untreated potential outcomes is the same for a treatment group and a comparison group, conditional on a set of pre-treatment covariates :math:`X`. Two alternative formulations are available.

   *Based on a "Never-Treated" Group*: For each group :math:`g` and for periods :math:`t \ge g - \delta`,

   .. math::

      \mathbb{E}[Y_t(0) - Y_{t-1}(0) | X, G_g = 1] = \mathbb{E}[Y_t(0) - Y_{t-1}(0) | X, C = 1].

   *Based on "Not-Yet-Treated" Groups*: For each group :math:`g` and for periods :math:`t \ge g - \delta`,

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

Aggregation of Effects
----------------------

A key feature of this methodology is the ability to aggregate the :math:`ATT(g, t)`'s into meaningful summary measures. This allows researchers to answer specific policy questions and understand different dimensions of treatment effect heterogeneity.

Event-Study Aggregation
~~~~~~~~~~~~~~~~~~~~~~~

Event-study plots aggregate effects by length of exposure to treatment, where :math:`e = t - g` represents the time elapsed since treatment adoption. This aggregation reveals how treatment effects evolve dynamically after implementation. The event-study parameter is

.. math::

   \theta_{es}(e) = \sum_{g \in \mathcal{G}} \mathbf{1}\{g + e \le \mathcal{T}\} P(G = g | G + e \le \mathcal{T}) ATT(g, g + e).

This parameter weights the group-time effects by the relative size of each group among those observed :math:`e` periods after treatment, providing insights into whether effects strengthen, weaken, or remain stable over time.

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

These aggregations provide transparent and interpretable ways to summarize treatment effect heterogeneity, avoiding the pitfalls of standard two-way fixed effects (TWFE) regressions, which can produce misleading estimates when treatment effects vary across groups or over time.

.. note::

   For the full theoretical details, including efficiency bounds and asymptotic properties, please refer to the original paper by `Callaway and Sant'Anna (2020) <https://psantanna.com/files/Callaway_SantAnna_2020.pdf>`_.
