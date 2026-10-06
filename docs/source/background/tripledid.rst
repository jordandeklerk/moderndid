.. _background-tripledid:

Triple differences with staggered adoption
==========================================

When a jurisdiction enables a policy in a given year for eligible units only, its ineligible
units never receive treatment. Those ineligible units can help measure local changes that an
ordinary difference-in-differences comparison would attribute to the policy.

By adding the eligibility comparison to the time and adoption comparisons, triple differences
asks whether the untreated eligible-minus-ineligible trend gap would have evolved similarly
across jurisdictions. This can be credible even when neither eligibility group satisfies an
ordinary parallel trends restriction on its own.

We follow `Ortiz-Villavicencio and Sant'Anna (2025)
<https://arxiv.org/abs/2505.09942>`_ to identify effects separately by
adoption cohort and period. Their `paper <https://arxiv.org/pdf/2505.09942v1>`_
shows why covariate adjustment and the choice of comparison cohorts
need special care in this design. ModernDiD implements the estimators
through :func:`~moderndid.ddd` and their summaries through
:func:`~moderndid.agg_ddd`.

Adoption and eligibility describe different groups
--------------------------------------------------

In a panel observed in periods :math:`1,\ldots,T`, let
:math:`S_i\in\{2,\ldots,T,\infty\}` be the first period when
unit :math:`i`'s jurisdiction enables treatment. Given its fixed eligibility status
:math:`Q_i\in\{0,1\}`, treatment requires both an enabled jurisdiction and an eligible unit,

.. math::

   D_{i,t}=\mathbf1\{t\geq S_i,\ Q_i=1\}.

With treatment remaining in place after adoption, the actual treatment cohort
:math:`G_i` equals :math:`S_i` for eligible units and :math:`\infty` for
ineligible units. By contrast, :math:`S_i` records the enabling date for
both eligibility groups. Write
:math:`\mathcal S` for its support and
:math:`\mathcal G_{trt}=\mathcal S\setminus\{\infty\}` for the
finite cohorts containing eligible treated units.

.. admonition:: Record the enabling date for everyone
   :class: tip

   Pass the enabling-cohort column :math:`S` as ``gname`` and eligibility
   :math:`Q` as ``pname`` in ``ddd``. Ineligible units in an adopting
   jurisdiction must keep that jurisdiction's enabling date. Coding
   them all as never enabled would remove the within-jurisdiction
   comparison the estimator needs.

Let :math:`Y_{i,t}(g)` be the potential outcome under first treatment
in period :math:`g`, and let :math:`Y_{i,t}(\infty)` be the outcome
without treatment. Under consistency, :math:`Y_{i,t}=Y_{i,t}(G_i)` in notation that presumes another unit's
treatment does not change unit :math:`i`'s outcome.

The effect for an eligible cohort in a particular period is

.. math::

   ATT(g,t)
      =\mathbb E[Y_t(g)-Y_t(\infty)\mid S=g,Q=1],
   \qquad g\in\mathcal G_{trt},\quad t\geq g.

This target averages over eligible units in the adopting cohort rather than the ineligible
units used to estimate the counterfactual. Keeping that target population fixed determines
how covariate adjustment enters the estimator.

The paper assumes a never-enabled cohort is available. Without one,
comparisons using later adopters are possible only while those adopters
remain untreated. In particular, periods at or after the last enabling
date have no such comparison cohort.

The untreated trend gap that must be stable
-------------------------------------------

A standard DiD would compare outcome changes across enabling cohorts
for eligible units alone. DDD allows that comparison to have a bias
if the same bias can be measured among ineligible units. We state
that restriction conditional on pre-treatment covariates :math:`X`,
along with the sampling, support, and timing conditions it needs.

.. admonition:: Assumption S Random sampling
   :class: assumption

   The vectors
   :math:`W_i=(Y_{i,1},\ldots,Y_{i,T},X_i',G_i,S_i,Q_i)'`,
   for :math:`i=1,\ldots,n`, are independent and identically
   distributed draws from their population law.

This sampling statement permits unrestricted dependence across periods within each independent
unit. Clustered inference requires a sampling justification at the cluster level rather than
interpreting this assumption as independence within clusters.

.. admonition:: Assumption SO Strong overlap
   :class: assumption

   There exists :math:`\varepsilon>0` such that, for every
   :math:`(s,q)\in\mathcal S\times\{0,1\}`,

   .. math::

      P(S=s,Q=q\mid X)>\varepsilon
      \qquad\text{almost surely}.

Strong overlap supplies all relevant cohort-by-eligibility cells at
covariate values represented in the target population. A covariate
profile observed only among eligible adopters cannot support the
DDD counterfactual without additional extrapolation assumptions.

.. admonition:: Assumption NA No anticipation
   :class: assumption

   For every :math:`g\in\mathcal G_{trt}` and every :math:`t<g`,

   .. math::

      \mathbb E[Y_t(g)\mid S=g,Q=1,X]
         =\mathbb E[Y_t(\infty)\mid S=g,Q=1,X]
      \qquad\text{almost surely}.

This restricts the conditional mean of the treated cohort's pre-treatment potential outcomes
rather than requiring every unit's individual anticipatory response to equal zero. A policy announcement
that changes those conditional means requires an earlier effective
adoption date or a different identifying design.

.. admonition:: Assumption DDD-CPT Conditional parallel trends
   :class: assumption

   Define :math:`\Delta Y_t(\infty)=Y_t(\infty)-Y_{t-1}(\infty)`.
   For every :math:`g\in\mathcal G_{trt}`,
   :math:`g'\in\mathcal S`, and :math:`t\geq g` with
   :math:`g'>\max\{g,t\}`,

   .. math::

      \begin{aligned}
      &\mathbb E[\Delta Y_t(\infty)\mid S=g,Q=1,X]
        -\mathbb E[\Delta Y_t(\infty)\mid S=g,Q=0,X]\\
      &\quad=\mathbb E[\Delta Y_t(\infty)\mid S=g',Q=1,X]
        -\mathbb E[\Delta Y_t(\infty)\mid S=g',Q=0,X]
      \qquad\text{almost surely}.
      \end{aligned}

Eligible and ineligible units may have different trends within a jurisdiction just as
jurisdictions may have different trends within either eligibility group. The restriction
equates differences in those untreated trends by requiring that subtracting the ineligible
trend removes the same difference across enabling cohorts.

Ineligible units remain untreated in this potential-outcome model.
If the policy also changes their outcomes through spillovers or a
separate policy component, their observed change need not measure
:math:`Y_t(\infty)-Y_{t-1}(\infty)`. Eligibility is therefore an
identifying feature of the design, rather than just a convenient
partition of the data.

Why the familiar regression needs more care
-------------------------------------------

In a two-period design without covariates, the DDD comparison is the
eligible-minus-ineligible change in the adopting cohort minus that
same change in a comparison cohort. Let :math:`g=2` and use a
never-enabled comparison. The population expression is

.. math::

   \begin{aligned}
   ATT(2,2)
      =&\bigl(\mathbb E[Y_2-Y_1\mid S=2,Q=1]
              -\mathbb E[Y_2-Y_1\mid S=2,Q=0]\bigr)\\
       &-\bigl(\mathbb E[Y_2-Y_1\mid S=\infty,Q=1]
              -\mathbb E[Y_2-Y_1\mid S=\infty,Q=0]\bigr).
   \end{aligned}

A saturated three-way fixed effects regression reproduces this
four-cell contrast in the two-period setting. Its staggered-adoption
analogue is often written

.. math::

   Y_{i,t}=\gamma_i+\gamma_{S_i,t}+\gamma_{Q_i,t}
           +\beta_{3wfe}D_{i,t}+\varepsilon_{i,t}.

With several adoption dates, a common coefficient can combine comparisons
using already-treated eligible units as controls. Treatment-effect
heterogeneity can then contaminate the coefficient's interpretation,
just as in the :ref:`staggered DiD setting <background-did>`.
The additional eligibility comparison does not remove that problem.

Even with two periods, covariate adjustment needs care if the DDD trend restriction holds only
conditional on :math:`X`. Subtracting two separately adjusted DiD estimates need not recover
the effect for eligible units in :math:`S=g` because those estimates can average their
conditional comparisons over different covariate distributions. The subtraction can work
under additional restrictions that make those averages coincide, though DDD-CPT alone does
not supply those restrictions.

The required adjustment instead evaluates all four conditional outcome
changes at the covariate distribution of the eligible adopting cohort.
Adding a common linear covariate-by-time term to the regression does
not generally represent that adjustment. The identification formulas
below show the common target population directly.

Identifying one cohort-period effect
-------------------------------------

Fix :math:`g\in\mathcal G_{trt}` and :math:`t\geq g`.
Choose a comparison cohort :math:`g_c>t` and use the last
pre-treatment period as the baseline,

.. math::

   \Delta Y=Y_t-Y_{g-1}.

We use one target cell and three untreated cells. Denote their
indicators by

.. math::

   \begin{aligned}
   T_g&=\mathbf1\{S=g,Q=1\},\\
   C_1&=\mathbf1\{S=g,Q=0\},\\
   C_2&=\mathbf1\{S=g_c,Q=1\},\\
   C_3&=\mathbf1\{S=g_c,Q=0\}.
   \end{aligned}

For each comparison :math:`j=1,2,3`, define its conditional outcome
change and its propensity score relative to the target cell,

.. math::

   m_j(x)=\mathbb E[\Delta Y\mid X=x,C_j=1],
   \qquad
   p_j(x)=P(T_g=1\mid X=x,T_g+C_j=1).

The score :math:`p_j` measures a conditional probability within two cells rather than the
unconditional probability of belonging to the target cohort. Its odds transport
comparison-cell observations to the target cell's covariate distribution,

.. math::

   r_j(X)=\frac{p_j(X)}{1-p_j(X)},\qquad
   w_T=\frac{T_g}{\mathbb E[T_g]},\qquad
   w_j=\frac{C_jr_j(X)}{\mathbb E[C_jr_j(X)]}.

Each normalized weight has expectation one. For any integrable
function :math:`f`, the true propensity scores imply
:math:`\mathbb E[w_jf(X)]=\mathbb E[f(X)\mid T_g=1]`.
That identity puts every comparison at the same covariate distribution.

Regression adjustment and weighting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Regression adjustment uses the three outcome regressions to predict
the target cohort's untreated change. Its population estimand is

.. math::

   ATT_{ra,g_c}(g,t)
      =\mathbb E[w_T\{\Delta Y-m_1(X)-m_2(X)+m_3(X)\}].

Restoring the comparison-cohort ineligible change through the positive sign on :math:`m_3`
after the other two subtractions recovers the missing untreated change for eligible adopters
under DDD-CPT.

Inverse probability weighting represents the same covariate adjustment
without specifying those outcome regressions,

.. math::

   ATT_{ipw,g_c}(g,t)
      =\mathbb E[(w_T-w_1-w_2+w_3)\Delta Y].

The weights align covariate distributions across the four cells before their changes are
combined, without requiring their outcome levels to match. Large odds can still make a sample estimator unstable
even when population overlap holds.

The doubly robust representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A doubly robust score combines the weighting and regression approaches.
Write :math:`s_1=s_2=1` and :math:`s_3=-1`. The resulting estimand is

.. math::

   ATT_{dr,g_c}(g,t)
      =\sum_{j=1}^{3}s_j\,
         \mathbb E[(w_T-w_j)\{\Delta Y-m_j(X)\}].

Although each term subtracts a different untreated cell from the same target cell, it need
not separately identify a causal DiD effect under ordinary parallel trends. Their signed
combination identifies the DDD effect under DDD-CPT.

.. admonition:: Theorem 4.1 Identification
   :class: theorem

   Under Assumptions S, SO, NA, and DDD-CPT, for every
   :math:`g\in\mathcal G_{trt}`, :math:`t\in\{2,\ldots,T\}`
   with :math:`t\geq g`, and comparison cohort
   :math:`g_c\in\mathcal S` with :math:`g_c>\max\{g,t\}`,

   .. math::

      ATT_{ra,g_c}(g,t)=ATT_{ipw,g_c}(g,t)
                     =ATT_{dr,g_c}(g,t)=ATT(g,t).

   Corollary 4.1 states that any combination over

   .. math::

      \mathcal G_c^{g,t}=\{g_c\in\mathcal S:
                             g_c>\max\{g,t\}\},

   with weights summing to one also identifies :math:`ATT(g,t)`.

Estimation replaces the population outcome regressions and propensity scores in the theorem
with fitted models. Regression
adjustment requires all three outcome models to be correct, whereas
IPW requires all three propensity models to be correct.

For the DR estimator, each comparison needs either its outcome model
or its propensity model to be correct at the fitted population limit,

.. math::

   \text{for every }j\in\{1,2,3\},\qquad
   p_j(X;\pi_j^*)=p_j(X)
      \quad\text{or}\quad
   m_j(X;\beta_j^*)=m_j(X)
   \qquad\text{almost surely}.

The eight choices of one correct model per comparison permit different comparisons to rely
on different model types. This property, called multiple robustness, still requires the causal
assumptions, overlap, and the estimation regularity conditions stated below.

In ``ddd``, ``est_method="dr"`` selects this estimator.
The alternatives ``"reg"`` and ``"ipw"`` select the regression and
weighting estimators. An intercept-only specification assumes the
trend-gap restriction holds without covariate adjustment. Including
covariates changes that identifying restriction rather than merely
adding predictors to improve precision.

Using several comparison cohorts
---------------------------------

When several untreated cohorts are available, each can provide an estimate of the same effect.
We can use their covariance to combine those estimates more precisely after keeping their
eligibility compositions separate.

DDD-CPT does not generally remain true after pooling all later cohorts
into a single comparison group. Within the pooled group, eligible and
ineligible units may represent different mixtures of enabling cohorts.
Group-specific untreated changes therefore need not cancel in the
eligible-minus-ineligible difference. This issue disappears under some
additional composition or trend restrictions. DDD-CPT does not itself
supply those additional identifying restrictions.

``control_group="nevertreated"`` uses the never-enabled cohort.
``control_group="notyettreated"`` constructs comparisons with individual
cohorts that remain untreated in the relevant periods and combines their
estimates. This retains the cohort-specific DDD contrasts rather than
assuming that a raw pooled contrast is valid.

For a fixed :math:`(g,t)`, stack the comparison-specific DR estimates
in :math:`\widehat{\boldsymbol{ATT}}_{dr}(g,t)`.
Let :math:`\Omega_{g,t}` be their asymptotic covariance and
:math:`\widehat\Omega_{g,t}` a consistent estimate. For noncollinear
comparisons, the variance-minimizing weights and estimator are

.. math::

   a_{g,t}
      =\frac{\Omega_{g,t}^{-1}\mathbf1}
             {\mathbf1'\Omega_{g,t}^{-1}\mathbf1},
   \qquad
   \widehat{ATT}_{dr,opt}(g,t)
      =\widehat a_{g,t}'\widehat{\boldsymbol{ATT}}_{dr}(g,t).

The weights sum to one but can be negative when combining these estimates of the same
:math:`ATT(g,t)`. Those negative values do not produce a negative-weight average of different
cohort treatment effects. The optimality claim concerns linear combinations of these valid
comparison-specific estimators rather than efficiency over all possible DDD estimators.

Sampling uncertainty and first-step estimation
----------------------------------------------

Standard errors must account for sampling uncertainty from the fitted propensity scores,
fitted outcome models, and estimated weight normalizations; consistency of the DR score
does not by itself provide them. The paper's influence function accounts for those
contributions under the following parametric estimation conditions.

.. admonition:: Assumption WM Working models
   :class: assumption

   For each propensity or outcome working model :math:`f(X;\gamma)`,
   the parameter space :math:`\Theta\subset\mathbb R^d` is compact
   and :math:`\gamma\mapsto f(X;\gamma)` is almost surely
   continuous. Its pseudo-true parameter :math:`\gamma^*` lies in
   :math:`\operatorname{int}(\Theta)`. An appropriate criterion
   :math:`\mathcal Q` identifies it uniquely. For every
   :math:`\varepsilon>0`, there is :math:`\delta>0` such that

   .. math::

      \inf_{\gamma\in\Theta:\|\gamma-\gamma^*\|\geq\varepsilon}
          \{\mathcal Q(\gamma)-\mathcal Q(\gamma^*)\}>\delta.

   The model is almost surely continuously differentiable on an open
   neighborhood :math:`\Theta_0\subset\Theta` of :math:`\gamma^*`.
   For each working propensity score, there is a common
   :math:`\varepsilon>0` such that

   .. math::

      0\leq p_j(X;\pi)\leq1-\varepsilon
      \quad\text{almost surely for every }
                       \pi\in\operatorname{int}(\Theta_{ps}).

The pseudo-true parameter is the probability limit of the fitted
model even when that model is misspecified. Multiple robustness requires
the corresponding fitted function to equal its population counterpart
for at least one model in each comparison. Merely including the correct
function somewhere in a model family does not establish that requirement.

.. admonition:: Assumption ALR Asymptotic linear representations
   :class: assumption

   Each first-step estimator :math:`\widehat\gamma` is strongly
   consistent for :math:`\gamma^*` and satisfies

   .. math::

      \sqrt n(\widehat\gamma-\gamma^*)
         =\frac1{\sqrt n}\sum_{i=1}^{n}
                    \ell_\gamma(W_i;\gamma^*)+o_P(1).

   The influence function has mean zero and a finite positive
   definite second-moment matrix. It also satisfies

   .. math::

      \lim_{\delta\downarrow0}
      \mathbb E\left[
       \sup_{\substack{\gamma\in\Theta_0\\
                  \|\gamma-\gamma^*\|\leq\delta}}
       \|\ell_\gamma(W;\gamma)-\ell_\gamma(W;\gamma^*)\|^2
      \right]=0.

Since the expansion uses the full panel sample size :math:`n`, a model fitted on a two-cell
subsample must have its influence function scaled to that convention before the DDD components
are combined.

.. admonition:: Assumption IC Integrability
   :class: assumption

   For every relevant :math:`g,t,g_c`, let
   :math:`\kappa^*` stack the first-step population limits and let
   :math:`h_{g,t,g_c}(W;\kappa)` be the signed DR score above,
   with model-dependent normalized population weights. On a small
   neighborhood :math:`\Gamma_0` of :math:`\kappa^*`, require

   .. math::

      \mathbb E[|h_{g,t,g_c}(W;\kappa^*)|^2]<\infty,
      \qquad
      \mathbb E\left[
         \sup_{\kappa\in\Gamma_0}
            \|\partial_\kappa h_{g,t,g_c}(W;\kappa)\|
      \right]<\infty.

These assumptions provide the paper's justification for parametric
first-step fitting. They do not automatically extend to arbitrary
machine-learning fits without a separate rate and inference argument.

The complete influence function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each :math:`j`, evaluate the working functions and normalized
weights at their population limits. Define the residual and its two
weighted means,

.. math::

   R_j=\Delta Y-m_j(X;\beta_j^*),\qquad
   \mu_{T,j}=\mathbb E[w_TR_j],\qquad
   \mu_{C,j}=\mathbb E[w_jR_j].

Dots denote derivatives with respect to the model parameters. The
first-step adjustment vectors are

.. math::

   \begin{aligned}
   b_j(W)&=\frac{C_j}
      {\{1-p_j(X;\pi_j^*)\}^2\,
           \mathbb E[C_jr_j(X;\pi_j^*)]},\\
   M_{\beta,j}&=\mathbb E[(w_T-w_j)\dot m_j(X;\beta_j^*)],\\
   M_{\pi,j}&=\mathbb E[b_j(W)\dot p_j(X;\pi_j^*)
                               (R_j-\mu_{C,j})].
   \end{aligned}

The influence function for component :math:`j` is

.. math::

   \begin{aligned}
   \psi_j(W)
      =&\ w_T(R_j-\mu_{T,j})-w_j(R_j-\mu_{C,j})\\
       &-\ell_{\beta_j}(W)'M_{\beta,j}
        -\ell_{\pi_j}(W)'M_{\pi,j}.
   \end{aligned}

The influence function accounts for estimated weight denominators through centering and for
first-step fitting through its final two terms. When both models
are correct for a component, those adjustment vectors vanish. Under
one correct model, the appropriate remaining fitting contribution
must still be included in the variance.

For comparison cohort :math:`g_c`, combine the three components,

.. math::

   \psi_{g,t,g_c}(W)=\psi_1(W)+\psi_2(W)-\psi_3(W).

Stack these functions over noncollinear comparison cohorts in
:math:`\boldsymbol\psi_{g,t}(W)`. Their covariance :math:`\Omega_{g,t}=\mathbb E[\boldsymbol\psi_{g,t}
\boldsymbol\psi_{g,t}']` determines the GMM weights as well as the uncertainty of the combined
estimate.

.. admonition:: Theorem 4.2 Consistency and asymptotic normality
   :class: theorem

   Suppose Assumptions S, SO, NA, DDD-CPT, WM, ALR, and IC hold.
   For every :math:`g\in\mathcal G_{trt}`, post-treatment
   :math:`t\in\{2,\ldots,T\}`, and
   :math:`g_c\in\mathcal G_c^{g,t}`, suppose each of the three
   components has a correct propensity or outcome model at its
   population limit. Then

   .. math::

      \begin{aligned}
      \sqrt n\{\widehat{ATT}_{dr,g_c}(g,t)-ATT(g,t)\}
         &=\frac1{\sqrt n}\sum_{i=1}^{n}\psi_{g,t,g_c}(W_i)+o_P(1)\\
         &\xrightarrow{d}N(0,\Omega_{g,t,g_c}),
      \end{aligned}

   where :math:`\Omega_{g,t,g_c}=\mathbb E[\psi_{g,t,g_c}^2]`.
   For noncollinear comparisons and consistently estimated GMM
   weights,

   .. math::

      \begin{aligned}
      \sqrt n\{\widehat{ATT}_{dr,opt}(g,t)-ATT(g,t)\}
         &=\frac1{\sqrt n}\sum_{i=1}^{n}
                     a_{g,t}'\boldsymbol\psi_{g,t}(W_i)+o_P(1)\\
         &\xrightarrow{d}N(0,\Omega_{g,t,opt}),\\
      \Omega_{g,t,opt}
         &=(\mathbf1'\Omega_{g,t}^{-1}\mathbf1)^{-1}
           \leq w'\Omega_{g,t}w
           \quad\text{for every }\mathbf1'w=1.
      \end{aligned}

   In particular, the combined asymptotic variance is no greater
   than that of any one comparison-cohort estimator.

Estimated GMM weights create no additional first-order contribution
when all comparison estimates share the same true effect and the
weights sum to one. For a singular covariance, the implementation uses a pseudoinverse when ordinary inversion
fails; the nonsingular inverse formula in the theorem does not by itself establish validity
for every singular case.

Inference for a collection of effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Although a pointwise interval covers one chosen effect, an event study often invites a
conclusion about several periods together, such as whether any post-treatment effect differs
from zero. Simultaneous bands target joint coverage of that reported collection.

For estimates with influence functions :math:`\widehat\psi_k`,
a multiplier draw uses the same mean-zero, variance-one multiplier
:math:`V_i^{(b)}` across all effects for unit :math:`i`,

.. math::

   \widehat{ATT}_k^{*(b)}-\widehat{ATT}_k
      =\frac1n\sum_{i=1}^{n}V_i^{(b)}\widehat\psi_k(W_i).

Sharing multipliers preserves the covariance across cohort-period
estimates. The empirical quantile of the maximum absolute standardized
perturbation supplies a simultaneous critical value through the joint-inference extension
described in the paper's Remark 4.6.

The ``ddd`` wrapper reports pointwise intervals even when
``boot=True`` supplies bootstrap standard errors. The lower-level
multi-period estimators expose ``cband`` for simultaneous cohort-period
bands. For aggregate effects, ``agg_ddd`` defaults to
``boot=True, cband=True`` and requires bootstrap inference when
simultaneous bands are requested.

.. admonition:: Check the scope of clustered inference
   :class: warning

   Multi-period ``ddd`` uses ``cluster`` only with ``boot=True``.
   Cells that combine several not-yet-treated cohorts retain their
   combination-specific standard errors without the requested cluster
   adjustment. ``agg_ddd`` does not currently propagate cluster labels
   into its own bootstrap. Its bands need a separate justification
   when units are dependent within clusters.

For cells that combine several comparison cohorts, the current reported
standard error uses the full sample size with a covariance calculated
on the comparison subsample. It can differ from a standard error
computed from the scaled full-sample influence functions. The normal
limit above requires a covariance estimate and sample-size convention
that represent the same sampling error.

The panel formulas above use observed within-unit outcome changes.
With ``panel=False``, the estimator works from repeated cross-sections
and needs stable population composition across the two samples,
including the joint distribution of :math:`S,Q,X`. Setting
``allow_unbalanced_panel=True`` also uses that repeated-cross-section
path. Keeping the default instead restricts each panel comparison to
units observed in both of its periods. The panel theorem alone does
not justify a change in sampling design or selective attrition.

Turning cohort-period effects into an event study
-------------------------------------------------

The collection of :math:`ATT(g,t)` separates adoption-cohort
heterogeneity from changes with exposure length. An event study averages
across the cohorts observed at a common event time :math:`e=t-g`.
Its weights therefore determine the population represented at each
point of the curve.

For post-treatment :math:`e\geq0`, define
:math:`\mathcal H_e=\{g\in\mathcal G_{trt}:2\leq g+e\leq T\}`.
The paper weights eligible treated units,

.. math::

   ES_G(e)
      =\sum_{g\in\mathcal H_e}
        \frac{P(S=g,Q=1)}
             {\sum_{h\in\mathcal H_e}P(S=h,Q=1)}\,
           ATT(g,g+e).

As :math:`e` grows, later adopters can leave the observable window.
An apparent change in the event-study curve can therefore reflect both
effect dynamics and changing cohort composition. ``balance_e=E`` in
``agg_ddd`` keeps cohorts observed through event time :math:`E` and
restricts the event window to
:math:`E-t_{max}+t_{min}\leq e\leq E`, where
:math:`t_{min},t_{max}` are the first and last sample periods.
Choosing ``min_e`` and
``max_e`` alone changes the displayed window without fixing its
cohort composition.

.. admonition:: Aggregation uses enabling-cohort shares
   :class: important

   The current ``agg_ddd`` weights count enabling cohorts across both
   eligibility groups. Its event-study target uses :math:`P(S=g)`
   rather than :math:`P(S=g,Q=1)`. It matches the paper's eligible-unit
   average when eligibility shares are equal across the included
   cohorts.

The targets can differ when eligibility shares vary across those cohorts.
Writing :math:`q_g=P(S=g)`, the package's event-study target is

.. math::

   ES_S(e)
      =\sum_{g\in\mathcal H_e}
           \frac{q_g}{\sum_{h\in\mathcal H_e}q_h}\,ATT(g,g+e).

The current multi-period ``ddd`` path does not pass ``weightsname``
into its cell estimators. Observation-weight support in the two-period
wrapper does not establish weighted estimation for the staggered design.
Since the cohort counts in ``agg_ddd`` also do not use that column, these distinctions matter
when the desired summary represents an observation-weighted eligible population rather than
enabling-cohort size.

Inference must account for estimated shares as well as estimated
effects. For either choice of cohort population, let :math:`I_g`
be its membership indicator, :math:`q_g=\mathbb E[I_g]`,
:math:`Q_e=\sum_{h\in\mathcal H_e}q_h`, and
:math:`\omega_{g,e}=q_g/Q_e`. The share influence function is

.. math::

   \zeta_{g,e}(W)
      =\frac{I_g-q_g}{Q_e}
       -\frac{q_g}{Q_e^2}
          \sum_{h\in\mathcal H_e}(I_h-q_h).

For :math:`\psi_{g,g+e}` denoting the influence function of the
chosen cohort-period estimator, the event-study influence function is

.. math::

   \psi_{ES(e)}(W)
      =\sum_{g\in\mathcal H_e}
        \{\omega_{g,e}\psi_{g,g+e}(W)
                           +ATT(g,g+e)\zeta_{g,e}(W)\}.

The delta-method representation behind the paper's Corollary 4.2 for eligible-unit shares also
gives the enabling-share version used by the package through the same differentiation.
Treating estimated cohort shares as fixed would omit the second term.

An overall event-study average gives equal weight to the selected
post-treatment event times,

.. math::

   ES_{avg}=\frac1{|\mathcal E|}\sum_{e\in\mathcal E}ES(e).

The package also provides ``type="group"`` and ``type="calendar"``
summaries. Its ``type="simple"`` summary weights the included
post-treatment cohort-period cells by cohort size. That summary need
not equal an equal-weight average of event-study coefficients because
cohorts contribute different numbers of observed periods.

What pre-treatment contrasts can tell you
-----------------------------------------

Pre-treatment DDD contrasts examine whether the untreated trend gap
was stable before adoption. With ``base_period="universal"``,
periods are measured relative to :math:`g-1` and event time
:math:`-1` is zero by normalization. With ``"varying"``,
pre-treatment contrasts compare adjacent periods instead.

DDD-CPT as stated above restricts periods at or after adoption.
To expect earlier contrasts to vanish in the population, extend that
trend-gap restriction to the corresponding pre-treatment periods.
No anticipation alone does not impose equal untreated pre-treatment
trend gaps across cohorts.

A nonzero pre-treatment contrast can challenge the credibility of
that extension. A failure to reject zero does not prove the
post-treatment counterfactual restriction, particularly when the
estimates are imprecise. Pre-treatment diagnostics should inform the
design and the interpretation of its assumptions.

The :ref:`triple differences example <example_triple_did>` estimates treatment effects in an
analysis of crop insurance adoption. For departures from a
common-reference trend-gap restriction, the
:ref:`sensitivity background <background-didhonest>` explains how to
state allowable violations explicitly. DDD estimates require a
compatible coefficient vector and joint covariance for that analysis;
the ``honest_did`` wrapper currently expects a dynamic ``aggte``
result rather than a ``DDDAggResult``.
