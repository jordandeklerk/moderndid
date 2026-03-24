.. _background-drdid:

Doubly Robust DiD
=================

The ``drdid`` module provides doubly robust estimators for the average treatment effect on the treated (ATT) in
difference-in-differences (DID) designs, based on the work of `Sant'Anna and Zhao (2020)
<https://psantanna.com/files/SantAnna_Zhao_DRDID.pdf>`_. These estimators are consistent if either a propensity score
model or an outcome regression model is correctly specified, but not necessarily both, offering robustness against model
misspecification.

In many empirical applications, researchers want to control for pre-treatment covariates that may be associated with
both treatment assignment and outcome dynamics. The challenge is that standard DiD estimators can be sensitive to how
these covariates are incorporated. If the propensity score model (which predicts treatment status) is misspecified,
inverse probability weighting estimators may be biased. If the outcome regression model (which predicts counterfactual
outcomes) is misspecified, regression-adjusted estimators may be biased. Doubly robust methods address this concern by
providing consistency as long as at least one of these two models is correct.

Setup and Notation
------------------

The canonical difference-in-differences setup involves comparing outcomes before and after treatment between a treated
group and an untreated comparison group. We consider a setting with two groups and two time periods. Let :math:`Y_{it}`
be the outcome for unit :math:`i` at time :math:`t`, where :math:`t=0` is the pre-treatment period and :math:`t=1` is
the post-treatment period. Let :math:`D_i` be an indicator for treatment status, where :math:`D_i=1` if the unit is in
the treatment group and :math:`D_i=0` for the comparison group. We assume treatment happens between :math:`t=0` and
:math:`t=1`, so :math:`D_{i0}=0` for all :math:`i`. We also observe a vector of pre-treatment covariates :math:`X_i`.

Using the potential outcomes framework, :math:`Y_{it}(d)` is the outcome that would be observed for unit :math:`i` at
time :math:`t` under treatment status :math:`d`. The observed outcome is

.. math::
   Y_{it} = D_i Y_{it}(1) + (1-D_i) Y_{it}(0).

The parameter of interest is the Average Treatment Effect on the Treated (ATT)

.. math::
   \tau = \mathbb{E}[Y_{1}(1) - Y_{1}(0) | D=1].

Since :math:`Y_1(1)` is observed for the treated group, we can write

.. math::
   \tau = \mathbb{E}[Y_1 | D=1] - \mathbb{E}[Y_1(0) | D=1].

The main identification challenge is to estimate the counterfactual term :math:`\mathbb{E}[Y_1(0) | D=1]`.

Identification
--------------

The identification of treatment effects in DiD settings relies on assumptions that allow us to construct valid
counterfactuals. In the covariate-adjusted setting, these assumptions are stated conditionally on covariates,
recognizing that parallel trends may only hold after controlling for observable differences between treatment and
comparison groups. The following assumptions formalize the conditions under which the ATT can be recovered from the
observed data.

Beyond the negative weighting problems of TWFE in staggered settings (see
:ref:`background-did`), there is an additional issue in the two-period case with covariates.
Running a two-way fixed effects regression

.. math::

   Y_{it} = \alpha_1 + \alpha_2 T_i + \alpha_3 D_i
   + \tau^{fe}(T_i \cdot D_i) + \theta' X_i + \varepsilon_{it}

and interpreting :math:`\hat{\tau}^{fe}` as the ATT implicitly imposes two restrictions beyond
the identifying assumptions. First, treatment effects must be homogeneous in :math:`X`,

.. math::

   E[Y_1(1) - Y_1(0) \mid X, D=1] = \tau^{fe} \quad \text{a.s.}

Second, there can be no covariate-specific trends,

.. math::

   E[Y_1 - Y_0 \mid X, D=d] = E[Y_1 - Y_0 \mid D=d] \quad \text{a.s.}
When either restriction fails, :math:`\tau^{fe}` generally differs from the ATT. The doubly
robust estimators below do not require these restrictions.

The key identifying assumptions are as follows.

.. admonition:: Assumption 1 (Data Structure)

   The data are assumed to be either

   (a) A panel dataset where :math:`\{Y_{i0}, Y_{i1}, D_i, X_i\}_{i=1}^n` are independent and identically distributed
   (i.i.d.).

   (b) A pooled repeated cross-section where :math:`\{Y_i, D_i, X_i, T_i\}_{i=1}^n` are i.i.d. draws from a mixture
   distribution, and the joint distribution of :math:`(D, X)` is invariant to the time period :math:`T`.

.. admonition:: Assumption 2 (Conditional Parallel Trends)

   The average outcomes for the treated and comparison groups would have evolved in parallel, conditional on
   covariates,

   .. math::

      \mathbb{E}[Y_1(0) - Y_0(0) | D=1, X] = \mathbb{E}[Y_1(0) - Y_0(0) | D=0, X].

.. admonition:: Assumption 3 (Overlap)

   For all values of covariates :math:`X`, there is a positive probability of being in either the treatment or
   comparison group,

   .. math::

      \mathbb{P}(D=1|X) < 1-\varepsilon \text{ for some } \varepsilon > 0.

Individual OR and IPW Approaches
---------------------------------

Under Assumptions 1-3, there are two standard approaches for estimating the ATT with
covariates, each relying on correctly specifying a different nuisance function.

The outcome regression (OR) approach models the counterfactual outcome evolution of the
comparison group, :math:`m_{0,t}(X) = \mathbb{E}[Y_t \mid D=0, X]`, and estimates the ATT as
the difference between the treated group's observed outcome change and the predicted
counterfactual change, averaged over treated units. Consistency requires the outcome model to
be correctly specified.

The inverse probability weighting (IPW) approach avoids modeling outcomes and instead reweights
the comparison group's outcome change to match the covariate distribution of the treated group,
using the propensity score :math:`p(X) = \mathbb{P}(D=1 \mid X)`. When panel data are available,
the IPW estimand is

.. math::

   \tau = \frac{1}{\mathbb{E}[D]} \mathbb{E}\left[\frac{D - p(X)}{1 - p(X)}
   (Y_1 - Y_0)\right].

Consistency requires the propensity score model to be correctly specified.

These two approaches depend on different, non-nested conditions. In practice, it is hard to
know which nuisance model is correctly specified, making it difficult to choose between them.

Doubly Robust Estimands
-----------------------

The doubly robust (DR) approach combines both the OR and IPW methods so that the resulting
estimand identifies the ATT if either (but not necessarily both) the propensity score or the
outcome regression is correctly specified.

Let :math:`p(X) = \mathbb{P}(D=1|X)` be the propensity score and :math:`\pi(X)` be a working model for the propensity
score.

**Panel Data**

When panel data are available, we observe :math:`(Y_{i0}, Y_{i1})` for each unit. Let :math:`\Delta Y = Y_1 - Y_0`
and :math:`\mu_{0, \Delta}^p(X)` be a working model for the outcome evolution of the comparison group,
:math:`\mathbb{E}[\Delta Y | D=0, X]`.

The DR estimand for panel data is given by

.. math::
   \tau^{dr, p} = \mathbb{E}\left[ (w_1^p(D) - w_0^p(D, X; \pi)) (\Delta Y - \mu_{0, \Delta}^p(X)) \right],

where the weights are defined as

.. math::
   w_1^p(D) = \frac{D}{\mathbb{E}[D]} \quad \text{and} \quad w_0^p(D, X; \pi) = \frac{\frac{\pi(X)(1-D)}{1-\pi(X)}}{\mathbb{E}\left[\frac{\pi(X)(1-D)}{1-\pi(X)}\right]}.

This estimand is consistent for the ATT if either the propensity score model is correct, :math:`\pi(X) = p(X)`, or
the outcome model is correct, :math:`\mu_{0, \Delta}^p(X) = \mathbb{E}[\Delta Y | D=0, X]`.

**Repeated Cross-Sections**

When only repeated cross-sections are available, we do not observe the same units in both periods. Let :math:`T` be a
time indicator with :math:`T=1` for post-treatment and :math:`T=0` for pre-treatment. Let :math:`\mu_{d,t}^{rc}(X)`
be a working model for :math:`\mathbb{E}[Y | D=d, T=t, X]`, and define

.. math::
   \mu_{d, Y}^{rc}(T, X) = T \cdot \mu_{d, 1}^{rc}(X) + (1-T) \cdot \mu_{d, 0}^{rc}(X).

Two DR estimands are proposed. The first, :math:`\tau_1^{dr,rc}`, is given by

.. math::
   \tau_{1}^{dr, rc} = \mathbb{E}\left[ (w_{1}^{rc}(D, T) - w_{0}^{rc}(D, T, X; \pi)) (Y - \mu_{0, Y}^{rc}(T, X)) \right].

The second estimand, :math:`\tau_2^{dr,rc}`, which is locally efficient, is

.. math::
   \begin{aligned}
   \tau_{2}^{dr, rc} = \tau_{1}^{dr, rc} &+ \left(\mathbb{E}[\mu_{1,1}^{rc}(X)-\mu_{0,1}^{rc}(X) | D=1] - \mathbb{E}[\mu_{1,1}^{rc}(X)-\mu_{0,1}^{rc}(X) | D=1, T=1]\right) \\
   &- \left(\mathbb{E}[\mu_{1,0}^{rc}(X)-\mu_{0,0}^{rc}(X) | D=1] - \mathbb{E}[\mu_{1,0}^{rc}(X)-\mu_{0,0}^{rc}(X) | D=1, T=0]\right)
   \end{aligned}

The weights for the repeated cross-sections case are defined as

.. math::
   w_{1, t}^{rc}(D, T) = \frac{D \cdot \mathbf{1}\{T=t\}}{\mathbb{E}[D \cdot \mathbf{1}\{T=t\}]} \quad \text{and}
   \quad w_{0, t}^{rc}(D, T, X; \pi) = \frac{\frac{\pi(X)(1-D) \cdot \mathbf{1}\{T=t\}}{1-\pi(X)}}{\mathbb{E}\left[\frac{\pi(X)(1-D) \cdot \mathbf{1}\{T=t\}}{1-\pi(X)}\right]}.

Both estimands are consistent for the ATT under the same doubly robust conditions, namely if either the propensity score
model or the outcome model for the comparison group is correctly specified.

Although :math:`\tau_1^{dr,rc}` does not rely on outcome regression models for the treated group while
:math:`\tau_2^{dr,rc}` does, Theorem 1 of the paper shows that both identify the ATT under identical conditions. This
follows from the stationarity condition in Assumption 1(b), which implies that for any integrable function :math:`g`,
:math:`\mathbb{E}[g(X) | D=1] = \mathbb{E}[g(X) | D=1, T=t]` for :math:`t=0,1`. Thus, modeling the outcome regression
for the treated group can be "harmless" in terms of identification when incorporated appropriately.

Semiparametric Efficiency Bounds
--------------------------------

Beyond consistency, an important question is how precisely we can estimate the ATT given a fixed sample size.
Semiparametric efficiency theory provides a lower bound on the variance that any regular estimator can achieve.
Estimators that attain this bound are said to be efficient, meaning they make optimal use of the available information.
The efficiency bound depends on the data structure (panel versus repeated cross-sections) and provides guidance on
which estimators to prefer.

The paper derives the semiparametric efficiency bounds for the ATT under the DID framework. These bounds provide a
benchmark against which any regular semiparametric DID estimator can be compared.

Panel Data Efficiency Bound
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When panel data are available, the efficient influence function for the ATT is

.. math::

   \eta^{e,p}(Y_1, Y_0, D, X) &= w_1^p(D)(m_{1,\Delta}^p(X) - m_{0,\Delta}^p(X) - \tau) \\
   &\quad + w_1^p(D)(\Delta Y - m_{1,\Delta}^p(X)) - w_0^p(D, X; p)(\Delta Y - m_{0,\Delta}^p(X)),

where :math:`m_{d,\Delta}^p(X) = m_{d,1}^p(X) - m_{d,0}^p(X)` denotes the true conditional outcome evolution for
group :math:`d`. The semiparametric efficiency bound is :math:`\mathbb{E}[(\eta^{e,p})^2]`.

An important observation is that the efficient influence function for panel data does not depend on the outcome
regression for the treated group :math:`m_{1,t}^p(X)`. This can be seen by rewriting

.. math::

   \eta^{e,p}(Y_1, Y_0, D, X) = (w_1^p(D) - w_0^p(D, X; p))(\Delta Y - m_{0,\Delta}^p(X)) - w_1^p(D) \cdot \tau.

Repeated Cross-Section Efficiency Bound
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When only repeated cross-sections are available, the efficient influence function takes a different form that
explicitly depends on the outcome regressions for both treated and comparison groups in each time period. Let
:math:`\lambda = \mathbb{P}(T=1)`. The efficient influence function is

.. math::

   \eta^{e,rc}(Y, D, T, X) &= \frac{D}{\mathbb{E}[D]}(m_{1,\Delta}^{rc}(X) - m_{0,\Delta}^{rc}(X) - \tau) \\
   &\quad + (w_{1,1}^{rc}(D,T)(Y - m_{1,1}^{rc}(X)) - w_{1,0}^{rc}(D,T)(Y - m_{1,0}^{rc}(X))) \\
   &\quad - (w_{0,1}^{rc}(D,T,X;p)(Y - m_{0,1}^{rc}(X)) - w_{0,0}^{rc}(D,T,X;p)(Y - m_{0,0}^{rc}(X))).

Panel vs Repeated Cross-Section Efficiency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A natural question is whether there are efficiency gains from having panel data instead of repeated cross-sections.
The paper shows that panel data always enables more efficient estimation. Under the assumption that :math:`T` is
independent of :math:`(Y_1, Y_0, D, X)`,

.. math::

   \mathbb{E}[(\eta^{e,rc})^2] - \mathbb{E}[(\eta^{e,p})^2] \geq 0,

with the efficiency loss being convex in :math:`\lambda`. The loss is larger when the pre- and post-treatment sample
sizes are more imbalanced. When the conditional variances are equal across time periods, :math:`\lambda = 0.5`
minimizes the efficiency loss.

Efficiency of τ₁ vs τ₂
~~~~~~~~~~~~~~~~~~~~~~

The paper shows that when working models are correctly specified, :math:`\tau_2^{dr,rc}` attains the semiparametric
efficiency bound while :math:`\tau_1^{dr,rc}` does not. The efficiency loss from using estimators based on
:math:`\tau_1^{dr,rc}` instead of :math:`\tau_2^{dr,rc}` is

.. math::

   \begin{aligned}
   V_1^{rc} - V_2^{rc} &= \mathbb{E}[D]^{-1} \cdot \text{Var}\bigg[\sqrt{\frac{1-\lambda}{\lambda}}(m_{1,1}^{rc}(X) - m_{0,1}^{rc}(X)) \\
   &\qquad + \sqrt{\frac{\lambda}{1-\lambda}}(m_{1,0}^{rc}(X) - m_{0,0}^{rc}(X)) \,\bigg|\, D=1\bigg] \\
   &\geq 0.
   \end{aligned}

This loss is strictly positive whenever the conditional ATT varies with covariates, which is typically the case in
practice. Thus, estimators based on :math:`\tau_2^{dr,rc}` should generally be preferred.

Improved Doubly Robust Estimators
---------------------------------

Standard doubly robust estimators are consistent under misspecification of one nuisance model,
but their asymptotic variance can depend on which model is correct. In practice, the researcher
does not know which model is correct, making it unclear which variance formula to use for
inference. The improved estimators resolve this problem by achieving double robustness not only
for consistency but also for inference, meaning the exact form of the asymptotic variance does
not depend on which working models are correctly specified.

What makes this work is that for the first-stage estimation to have no effect on the limiting
distribution of the DR DID estimator, the first-order conditions of the nuisance parameter
estimators must satisfy specific moment conditions. When a logistic propensity score model
:math:`\pi(X;\gamma) = \Lambda(X'\gamma)` and linear outcome regression models
:math:`\mu_{d,t}(X;\beta) = X'\beta` are adopted, these moment conditions are automatically
satisfied by two specific estimation methods.

- **Inverse Probability Tilting (IPT)** for the propensity score (Graham et al., 2012). The
  IPT estimator solves

  .. math::

     \max_\gamma \mathbb{E}_n[D X'\gamma - (1-D)\exp(X'\gamma)],

  and its first-order conditions ensure the propensity-score-related estimation effect
  vanishes regardless of whether the propensity score model is correct.

- **Weighted Least Squares (WLS)** for the outcome regression of the comparison group, using
  propensity-score-derived weights

  .. math::

     \hat{w} = \frac{\Lambda(X'\hat{\gamma}^{ipt})}{1 - \Lambda(X'\hat{\gamma}^{ipt})}.

  The WLS first-order conditions ensure the outcome-regression-related estimation effect
  vanishes regardless of whether the outcome model is correct.

Because the first-stage estimation has no effect on the asymptotic distribution, the summands
of the DR DID estimator can be treated as if they were i.i.d., and the asymptotic variance
can be consistently estimated by the sample variance of the influence function evaluations.
This makes inference simple and avoids the need to account for estimation effects from the
first stage. When both working models are correctly specified, the improved estimators attain
the semiparametric efficiency bounds.

Practical Guidance
~~~~~~~~~~~~~~~~~~

.. tip::

   For repeated cross-sections, prefer :math:`\tau_2^{dr,rc}` over :math:`\tau_1^{dr,rc}`.
   Both are doubly robust for consistency and inference, but :math:`\tau_2^{dr,rc}` is locally
   semiparametrically efficient while :math:`\tau_1^{dr,rc}` is not. The efficiency loss is
   strictly positive whenever the conditional ATT varies with covariates, which is the typical
   case.

The outcome regressions for the treated group that :math:`\tau_2^{dr,rc}` requires can be
estimated by ordinary least squares (not weighted), since the pseudo-true parameters for the
treated group do not generate any estimation effect.

When designing a repeated cross-section study, the efficiency loss from not having panel data is
convex in :math:`\lambda = \mathbb{P}(T=1)`, the fraction of the sample observed in the
post-treatment period. When the conditional variances are equal across time periods,
:math:`\lambda = 0.5` minimizes the efficiency loss, making equal-sized pre- and post-treatment
samples a reasonable default.

.. note::
   For the full theoretical details, including proofs and regularity conditions, please refer to the original paper by
   `Sant'Anna and Zhao (2020) <https://psantanna.com/files/SantAnna_Zhao_DRDID.pdf>`_.
