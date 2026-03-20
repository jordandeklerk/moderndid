.. _background-etwfe:

Extended Two-Way Fixed Effects
==============================

The ``etwfe`` module implements the Extended Two-Way Fixed Effects (ETWFE) methodology for
difference-in-differences with staggered treatment adoption and heterogeneous treatment effects,
based on the work of `Wooldridge (2023) <https://doi.org/10.1093/ectj/utad016>`_ and
`Wooldridge (2025) <https://doi.org/10.1007/s00181-025-02807-z>`_. Rather than discarding the
TWFE estimator, this approach shows that TWFE produces unbiased estimates of the ATTs when
applied to a suitably flexible model that saturates the regression with cohort-by-time interaction
terms.

Why Conventional TWFE Fails
---------------------------

A conventional TWFE regression for staggered interventions estimates

.. math::

   y_{it} = \tau \cdot w_{it} + \sum_{s=2}^{T} \gamma_s f_{s,t} + c_i + u_{it},

where :math:`w_{it}` is the time-varying treatment indicator, :math:`f_{s,t}` are time dummies, and
:math:`c_i` are unit fixed effects. This specification imposes a constant treatment effect
:math:`\tau` regardless of when a unit entered treatment and how long it has been exposed.
As shown by de Chaisemartin and D'Haultfoeuille (2020), Goodman-Bacon (2021), and others, the
resulting TWFE estimate depends on :math:`2 \times 2` difference-in-differences comparisons,
several of which use already-treated units as controls. These "forbidden comparisons" can
produce negative weights on some group-time ATTs, causing the estimated effect to have the
wrong sign even when all underlying effects are positive.

The ETWFE approach resolves this by saturating the model with the full set of cohort-time
interaction dummies, allowing separate treatment effects for every (cohort, time period) cell.

Setup and Notation
------------------

Consider a panel of :math:`N` units observed over :math:`T` time periods. Let :math:`q` be the
first period in which any unit receives treatment. Define treatment cohort dummies
:math:`d_g, \; g = q, \ldots, T`, where :math:`d_{g,i} = 1` if unit :math:`i` is first treated in
period :math:`g`. Units never treated during the sample window belong to the "never-treated"
group (:math:`d_{\infty,i} = 1`).

The potential outcome for unit :math:`i` at time :math:`t` if first treated in period :math:`g` is
:math:`y_{it}(g)`, and the potential outcome under no treatment is :math:`y_{it}(\infty)`. The
parameter of interest is the cohort-time-specific average treatment effect on the treated

.. math::

   \tau_{g,t} \equiv E[y_t(g) - y_t(\infty) \mid d_g = 1], \quad g = q, \ldots, T; \; t = g, \ldots, T.

The time-varying binary treatment indicator is

.. math::

   w_{it} = \sum_{g=q}^{T} d_{g,i} \cdot p_{g,t},

where :math:`p_{g,t} = f_{g,t} + f_{g+1,t} + \cdots + f_{T,t}` is a post-intervention indicator.
Because treatment is irreversible, :math:`\{w_{it}\}_{t=1}^{T}` is a sequence of zeros followed by
ones, with the first one appearing when :math:`t = g`. The treatment indicator :math:`w_{it}` is
the sum of mutually exclusive dummies :math:`d_{g,i} \cdot f_{s,t}` indicating cohort :math:`g`
in period :math:`s`. Including :math:`w_{it}` explicitly in the regression is redundant for point
estimation but useful for computing aggregated effects and their standard errors.

Identifying Assumptions
-----------------------

Identification of the ATTs relies on five assumptions.

.. admonition:: Assumption SUTVA (Stable Unit Treatment Value)

   The potential outcome of each unit does not depend on the treatment assignment of other units.

.. admonition:: Assumption NBC (No Bad Controls)

   Letting :math:`\mathbf{x}(g)` be the covariates when the treatment cohort is :math:`g`, assume
   :math:`\mathbf{x}(g) = \mathbf{x}(\infty)` for :math:`g = q, \ldots, T`. This ensures that
   the covariates are not affected by the treatment, so we can estimate
   :math:`E[\mathbf{x}(\infty) \mid d_g = 1]` from the observed data.

.. admonition:: Assumption NA (No Anticipation)

   For treatment cohorts :math:`g = q, \ldots, T` and time-constant covariates :math:`\mathbf{x}`,

   .. math::

      E[y_t(g) - y_t(\infty) \mid \mathbf{d}, \mathbf{x}] = 0, \quad t < g.

   Units do not alter their behavior in anticipation of future treatment.

.. admonition:: Assumption CPT (Conditional Parallel Trends)

   For :math:`t = 2, \ldots, T` and time-constant controls :math:`\mathbf{x}`,

   .. math::

      E[y_t(\infty) - y_1(\infty) \mid \mathbf{d}, \mathbf{x}]
      = E[y_t(\infty) - y_1(\infty) \mid \mathbf{x}].

   Conditional on covariates, the cohort assignment :math:`\mathbf{d}` is not systematically related
   to the trend in the never-treated state. Selection into treatment based on levels is permitted;
   only selection based on trends is ruled out. An equivalent characterization is that
   :math:`\mathbf{d}` is unconfounded with respect to :math:`y_t(\infty) - y_1(\infty)` conditional
   on :math:`\mathbf{x}`.

.. admonition:: Assumption LIN (Linearity)

   The conditional expectations satisfy

   .. math::

      E[y_1(\infty) \mid \mathbf{d}, \mathbf{x}]
      = \alpha + \sum_{g} \beta_g d_g
      + \mathbf{x}\boldsymbol{\kappa}
      + \sum_{g} (d_g \cdot \mathbf{x})\boldsymbol{\xi}_g,

   .. math::

      E[y_t(\infty) \mid \mathbf{d}, \mathbf{x}]
      - E[y_1(\infty) \mid \mathbf{d}, \mathbf{x}]
      = \sum_{s=2}^{T} \gamma_s f_{s,t}
      + \sum_{s=2}^{T} (f_{s,t} \cdot \mathbf{x})\boldsymbol{\pi}_s.

   The first equation is definitional when there are no covariates. With covariates, linearity
   in parameters is assumed. The second equation implies CPT because :math:`\mathbf{d}` does not
   appear on the right-hand side. When :math:`\mathbf{x}` consists only of exhaustive indicator
   variables, both expressions are nonparametric.

The Conditional Expectation in the Never-Treated State
------------------------------------------------------

Combining the assumptions above yields the conditional expectation of the outcome in the
never-treated state across all time periods

.. math::

   E[y_t(\infty) \mid \mathbf{d}, \mathbf{x}]
   = \alpha + \sum_g \beta_g d_g + \mathbf{x}\boldsymbol{\kappa}
   + \sum_g (d_g \cdot \mathbf{x})\boldsymbol{\xi}_g
   + \sum_{s=2}^{T} \gamma_s f_{s,t}
   + \sum_{s=2}^{T} (f_{s,t} \cdot \mathbf{x})\boldsymbol{\pi}_s.

This equation allows for selection into treatment through nonzero :math:`\beta_g` and
:math:`\boldsymbol{\xi}_g`, and for heterogeneous trends in the never-treated state through
nonzero :math:`\boldsymbol{\pi}_s`. The exclusion of :math:`d_g \cdot f_{s,t}` interactions from
this equation is precisely the parallel trends assumption.

Identification and Imputation
-----------------------------

By the no-anticipation assumption, the observed outcome for control observations
(:math:`w_{it} = 0`) equals the never-treated potential outcome. OLS on the control observations
identifies all parameters in the conditional expectation above. The ATTs are then identified as

.. math::

   \tau_{g,t}
   = E(y_t \mid d_g = 1)
   - \bigl[(\alpha + \beta_g + \gamma_t) + E(\mathbf{x} \mid d_g = 1)
   \cdot (\boldsymbol{\kappa} + \boldsymbol{\xi}_g + \boldsymbol{\pi}_t)\bigr],

where Assumption NBC ensures that :math:`E(\mathbf{x} \mid d_g = 1)` can be estimated from the
observed covariate means for cohort :math:`g`.

This motivates a three-step *cohort imputation* procedure (Procedure 4.1 in Wooldridge, 2025).

Step 1: Estimate the never-treated conditional expectation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using only the control observations (:math:`w_{it} = 0`), run an OLS regression

.. math::

   y_{it} \text{ on } 1, \; d_{q,i}, \ldots, d_{T,i}, \;
   \mathbf{x}_i, \; d_{q,i} \cdot \mathbf{x}_i, \ldots, d_{T,i} \cdot \mathbf{x}_i, \;
   f_{2,t}, \ldots, f_{T,t}, \; f_{2,t} \cdot \mathbf{x}_i, \ldots, f_{T,t} \cdot \mathbf{x}_i

to obtain the parameter estimates

.. math::

   (\tilde{\alpha}, \; \tilde{\beta}_q, \ldots, \tilde{\beta}_T, \;
   \tilde{\boldsymbol{\kappa}}, \; \tilde{\boldsymbol{\xi}}_q, \ldots, \tilde{\boldsymbol{\xi}}_T, \;
   \tilde{\gamma}_2, \ldots, \tilde{\gamma}_T, \;
   \tilde{\boldsymbol{\pi}}_2, \ldots, \tilde{\boldsymbol{\pi}}_T).

Step 2: Impute the counterfactual outcomes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Impute the missing outcomes in the never-treated state for all observations (including
treated ones):

.. math::

   \tilde{y}_{it}(\infty)
   = \tilde{\alpha} + \sum_{g=q}^{T} \tilde{\beta}_g d_{g,i}
   + \mathbf{x}_i \tilde{\boldsymbol{\kappa}}
   + \sum_{g=q}^{T} (d_{g,i} \cdot \mathbf{x}_i) \tilde{\boldsymbol{\xi}}_g
   + \sum_{s=2}^{T} \tilde{\gamma}_s f_{s,t}
   + \sum_{s=2}^{T} (f_{s,t} \cdot \mathbf{x}_i) \tilde{\boldsymbol{\pi}}_s.

The unit-specific treatment effects are then

.. math::

   \widetilde{te}_{it} = y_{it} - \tilde{y}_{it}(\infty).

Step 3: Average over cohorts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Obtain the estimated ATT for cohort :math:`g` in period :math:`t` by averaging the
unit-specific treatment effects over cohort :math:`g`:

.. math::

   \tilde{\tau}_{g,t}
   = N_g^{-1} \sum_{i=1}^{N} d_{g,i} \, \widetilde{te}_{it}
   = \bar{y}_{g,t}
   - \bigl[(\tilde{\alpha} + \tilde{\beta}_g + \tilde{\gamma}_t)
   + \bar{\mathbf{x}}_g
   \cdot (\tilde{\boldsymbol{\kappa}} + \tilde{\boldsymbol{\xi}}_g
   + \tilde{\boldsymbol{\pi}}_t)\bigr],

where

.. math::

   \bar{\mathbf{x}}_g = N_g^{-1} \sum_i d_{g,i} \mathbf{x}_i

is the vector of cohort-specific covariate averages.

A practical drawback of this multi-step procedure is that analytical standard errors must account
for the covariance between the first-step OLS estimates and the cohort means. The pooled OLS
approach described next avoids this complication entirely.

The ETWFE Regression
--------------------

Rather than implementing the three-step procedure, the ETWFE approach specifies a single pooled
regression over all observations

.. math::

   E(y_{it} \mid \mathbf{d}_i, \mathbf{x}_i)
   = \alpha + \sum_g \beta_g d_{g,i} + \mathbf{x}_i\boldsymbol{\kappa}
   + \sum_g (d_{g,i} \cdot \mathbf{x}_i)\boldsymbol{\xi}_g
   + \sum_s \gamma_s f_{s,t} + \sum_s (f_{s,t} \cdot \mathbf{x}_i)\boldsymbol{\pi}_s
   + \sum_g \sum_{s \geq g} \tau_{gs} (w_{it} \cdot d_{g,i} \cdot f_{s,t})
   + \sum_g \sum_{s \geq g} (w_{it} \cdot d_{g,i} \cdot f_{s,t} \cdot \dot{\mathbf{x}}_{ig})\boldsymbol{\delta}_{gs},

where :math:`\dot{\mathbf{x}}_{ig} = \mathbf{x}_i - \bar{\mathbf{x}}_g` are covariates demeaned
about their cohort means. The coefficients :math:`\tau_{gs}` on the treatment interaction terms
are the cohort-time ATTs. The coefficients :math:`\boldsymbol{\delta}_{gs}` on the demeaned
covariate interactions capture how the ATTs vary with the covariates (moderating effects).

The covariate demeaning is essential: without centering, the main-effect coefficients would
estimate ATTs at :math:`\mathbf{x} = \mathbf{0}`, which is unlikely to be meaningful. With
centering, :math:`\tau_{gs}` estimates the ATT evaluated at the cohort-average covariate values.

A key result (Proposition 5.2 in Wooldridge, 2025) is that the pooled OLS estimates of
:math:`\tau_{g,t}` from this saturated regression are numerically identical to the cohort
imputation estimates :math:`\tilde{\tau}_{g,t}`. The same equivalence holds for the coefficients
on the control variables.

TWFE-POLS Equivalence
~~~~~~~~~~~~~~~~~~~~~

When the pooled regression includes unit dummies :math:`c_{1,i}, \ldots, c_{N,i}` in place of
the cohort dummies :math:`d_{g,i}` and their interactions, the resulting two-way fixed effects
estimator produces identical coefficients on the treatment interaction terms. This follows from
the Two-Way Mundlak (TWM) theorem (Theorem 3.1 in Wooldridge, 2025): including unit-specific
time averages :math:`\bar{\mathbf{x}}_{i \cdot}` and period-specific cross-sectional averages
:math:`\bar{\mathbf{x}}_{\cdot t}` in a pooled OLS regression reproduces the TWFE estimates.
Since all the necessary Mundlak terms are already present in the saturated regression, the POLS
and TWFE estimates of :math:`\tau_{g,t}` coincide. The same holds for random effects (RE)
estimation.

The full equivalence chain is: cohort imputation = POLS on cohort dummies = TWFE = RE.
Additionally, the Borusyak, Jaravel and Spiess (2024) imputation estimator (which uses unit
dummies rather than cohort dummies in the first step) produces different unit-level residuals but
the same cohort-time ATT estimates.

This equivalence has a practical implication: controlling for cohort dummies rather than unit
dummies yields the same ATT estimates with far fewer parameters. With :math:`N = 1000` units,
five treatment cohorts, and ten controls, the cohort-based regression includes roughly 60
time-constant controls versus 1000 unit dummies.

Not-Yet-Treated Control Groups
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When all units are eventually treated (no never-treated group exists), the last-treated cohort
serves as the reference group. The ATTs are redefined as

.. math::

   \tau_{(g:T),t} = E[y_t(g) - y_t(T) \mid d_g = 1],
   \quad g = q, \ldots, T-1; \; t = g, \ldots, T,

measuring the effect of earlier treatment relative to the last-treated cohort. Under no
anticipation, for :math:`t = g, \ldots, T-1` these equal the standard ATTs
:math:`\tau_{g,t}`. No ATT is identified for the last-treated cohort :math:`T` because it has
no control group. Mechanically, all variables involving :math:`d_T` are dropped, making the
last-treated cohort act as the never-treated group.

Event Study and Pre-Treatment Testing
-------------------------------------

The framework extends naturally to event-study ("leads and lags") estimation by including
pre-treatment cohort-time interactions alongside the post-treatment ones. For each cohort
:math:`g`, a reference period :math:`g - 1` is excluded, producing pre-treatment coefficients
:math:`\theta_{g,s}` for :math:`s \leq g - 2` and post-treatment coefficients
:math:`\tau_{g,s}` for :math:`s \geq g`.

The pre-treatment coefficients estimate

.. math::

   E[y_s(\infty) - y_{g-1}(\infty) \mid d_g = 1]
   - E[y_s(\infty) - y_{g-1}(\infty) \mid \text{control}],

which equals zero under parallel trends. Testing whether the pre-treatment coefficients are
jointly zero provides a placebo test for the identifying assumptions. All equivalences
(POLS = TWFE = imputation) continue to hold for the leads-and-lags specification.

The event-study estimator with flexible covariates is equivalent to the Sun and Abraham (2021)
interaction-weighted estimator and to the Callaway and Sant'Anna (2021) regression-based
:math:`2 \times 2` DiD estimators applied to long differences.

Heterogeneous Cohort Trends
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the unconditional parallel trends assumption is suspect, the model can be extended by
adding cohort-specific linear trends :math:`\eta_g (d_{g,i} \cdot t)` to the conditional
expectation. This requires at least two pre-treatment periods per cohort. All equivalences
continue to hold with the additional trend terms. Higher-order polynomial trends in :math:`t`
are possible with more pre-treatment periods.

Nonlinear Extensions
--------------------

Wooldridge (2023) extends the framework to nonlinear models by replacing the identity link with
a known, strictly increasing function :math:`G(\cdot)`. The index version of parallel trends
requires

.. math::

   G^{-1}\bigl(E[y_t(\infty) \mid \mathbf{d}, \mathbf{x}]\bigr)
   - G^{-1}\bigl(E[y_1(\infty) \mid \mathbf{d}, \mathbf{x}]\bigr)
   = \gamma_t + \mathbf{x}\boldsymbol{\pi}_t,

so that parallel trends holds for the linear index inside :math:`G(\cdot)` rather than for the
mean directly. When :math:`G(\cdot) = \exp(\cdot)`, the assumption becomes parallel trends in
growth rates,

.. math::

   \frac{E[y_t(\infty) \mid D]}{E[y_1(\infty) \mid D]} = \exp(\gamma_t),

which is arguably more natural for nonnegative outcomes. When :math:`G(\cdot)` is the logistic
function, the assumption holds on the log-odds scale, which is more plausible for binary or
fractional outcomes than linear PT.

Estimation proceeds by pooled quasi-maximum likelihood in the linear exponential family (LEF).
The leading cases are:

- **Logistic/Bernoulli**: for binary and fractional outcomes
- **Exponential/Poisson**: for nonnegative outcomes (counts, corner solutions)
- **Linear/Normal**: the standard OLS case

The resulting QMLE is consistent for the conditional mean parameters regardless of the true
distribution, requiring only correct specification of the conditional mean function.

In the nonlinear case, the ATT is no longer simply the coefficient on the interaction term.
Using the imputation approach, the estimated ATT for cohort :math:`g` in period :math:`r` is

.. math::

   \hat{\tau}_{g,r}
   = \bar{Y}_{g,r}
   - N_g^{-1} \sum_{i} d_{g,i} \, G\bigl(
   \hat{\alpha} + \hat{\beta}_g
   + \mathbf{x}_i\hat{\boldsymbol{\kappa}}
   + \mathbf{x}_i\hat{\boldsymbol{\eta}}_g
   + \hat{\gamma}_r
   + \mathbf{x}_i\hat{\boldsymbol{\pi}}_r
   \bigr),

where the argument of :math:`G(\cdot)` is the estimated never-treated linear predictor for unit
:math:`i` in period :math:`r`. This subtracts the imputed counterfactual from the observed
cohort mean, averaging over the treated subsample. For the pooled QMLE approach using the
canonical link function, the ATT estimates are numerically identical to the imputation estimates
(Proposition 3.1 in Wooldridge, 2023). Standard errors are obtained via the delta method.

The pooled regression also produces index-scale treatment effects
:math:`\hat{\delta}_{g,r}`, the coefficients on the treatment interactions inside
:math:`G(\cdot)`. For the exponential model,

.. math::

   \exp(\hat{\delta}_{g,r}) - 1

is an approximate proportional effect. These index-scale parameters are often of independent
interest.

For nonnegative outcomes, Poisson fixed effects (with unit dummies) is the one nonlinear
estimator that avoids the incidental parameters problem. Without covariates, Poisson FE and
pooled Poisson QMLE produce numerically identical estimates. With covariates, they may differ.

Aggregation of Cohort-Time Effects
----------------------------------

With many cohorts and time periods, the number of individual :math:`\tau_{g,t}` estimates can
be large. Several aggregation schemes reduce these to interpretable summaries.

Overall Weighted Average
~~~~~~~~~~~~~~~~~~~~~~~~

A single summary effect weights each cohort-time ATT by the cohort share in the treated
population

.. math::

   \bar{\tau}_\omega = \sum_g \sum_{t=g}^{T} \hat{\omega}_g \hat{\tau}_{g,t}, \qquad
   \hat{\omega}_g = \frac{N_g}{\sum_{g'} (T - g' + 1) N_{g'}},

where :math:`N_g` is the number of units in cohort :math:`g`. This corresponds to averaging the
marginal effect of :math:`w_{it}` over all treated observations (:math:`w_{it} = 1`).

Event-Study Aggregation
~~~~~~~~~~~~~~~~~~~~~~~

Effects are averaged by exposure time :math:`e = t - g`, with cohort-share weights within each
exposure level

.. math::

   \hat{\tau}_{\omega,e} = \sum_{g=q}^{T-e} \hat{\omega}_{ge} \hat{\tau}_{g,g+e}, \qquad
   \hat{\omega}_{ge} = \frac{N_g}{N_q + \cdots + N_{T-e}}.

Since the weights are positive and sum to one within each exposure time, these estimates are
free of negative weighting. The exposure-time effects are commonly plotted along with 95%
confidence intervals to visualize treatment dynamics. Pre-treatment exposure times
(:math:`e < 0`) from the leads-and-lags specification can be included in the same plot,
with the :math:`e = -1` reference point normalized to zero.

Group and Calendar-Time Aggregation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Averaging across time periods for each cohort yields group-specific effects, revealing whether
early versus late adopters respond differently. Averaging across cohorts for each calendar period
yields calendar-time effects that capture how the aggregate treatment effect evolves with
macroeconomic conditions or concurrent policies.

Inference
~~~~~~~~~

Because the ATT estimates come from a single pooled regression, inference is straightforward.
Standard cluster-robust variance estimators at the unit level account for arbitrary serial
correlation and heteroskedasticity. Aggregated effects and their standard errors are computed
from the estimated coefficients and variance-covariance matrix using the delta method, with
proper accounting for sampling variation in the cohort-mean covariates and cohort shares.

.. note::

   For the complete theoretical development, including proofs of the equivalence results,
   the event-study extension, heterogeneous trends, and handling of unbalanced panels, see
   `Wooldridge (2025) <https://doi.org/10.1007/s00181-025-02807-z>`_. For the nonlinear
   extension including Poisson, logit, and probit models, see
   `Wooldridge (2023) <https://doi.org/10.1093/ectj/utad016>`_.
