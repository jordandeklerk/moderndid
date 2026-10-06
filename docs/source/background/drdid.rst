.. _background-drdid:

Doubly robust DiD
=================

Doubly robust difference-in-differences estimates the average effect of treatment on the units
that receive it while adjusting for pre-treatment characteristics. When those characteristics
predict untreated outcome changes and differ between groups, an unadjusted comparison may
mistake differences in untreated trends for a treatment effect. The method combines an outcome
model with a treatment assignment model so that the estimate remains consistent if either model
is correct, provided the assumptions that identify the effect still hold.

This page develops the two-period estimators in `Sant'Anna and Zhao (2020)
<https://psantanna.com/files/SantAnna_Zhao_DRDID.pdf>`_. We will follow the missing counterfactual
through identification, estimation, and inference as we distinguish what you can learn from a
panel from what you can learn from repeated cross-sections. Those differences explain
the options in :func:`~moderndid.drdid` and the additional outcome regressions needed for efficient
estimation when the same units cannot be followed over time.

The counterfactual we need
--------------------------

There are two periods, :math:`t=0` before treatment and :math:`t=1` after treatment. The indicator
:math:`D_i` records whether unit :math:`i` belongs to the group treated in the second period.
The comparison group remains untreated in both periods. We observe pre-treatment covariates
:math:`X_i`, whose first component is a constant when we fit the parametric models below.

Let :math:`Y_{it}(d)` denote the potential outcome under treatment status :math:`d`. Treatment
starts between the two observations, after we have measured each unit's baseline outcome. We also
require that future treatment has no effect on that pre-treatment outcome. The observed outcomes therefore satisfy

.. math::

   Y_{i0}=Y_{i0}(0),
   \qquad
   Y_{i1}=D_iY_{i1}(1)+(1-D_i)Y_{i1}(0).

Our target is the average treatment effect on the treated, or ATT,

.. math::

   \tau=\mathbb{E}[Y_1(1)-Y_1(0)\mid D=1]
   =\mathbb{E}[Y_1\mid D=1]-\mathbb{E}[Y_1(0)\mid D=1].

The first term comes from treated units' observed post-treatment outcomes. The second term asks
how those same units would have fared without treatment. DiD recovers that missing mean by using
the comparison group's outcome change, after adjusting for the covariates.

The sampling design determines which changes we actually observe. A panel contains
:math:`(Y_{i0},Y_{i1},D_i,X_i)` for each of :math:`n` units. We can therefore form the
individual change :math:`\Delta Y_i=Y_{i1}-Y_{i0}` from each unit's observed outcomes. Repeated cross-sections instead contain different units in the two
periods. For these observations, :math:`T_i` indicates the sampled period and
:math:`Y_i=T_iY_{i1}+(1-T_i)Y_{i0}` is the single observed outcome. Write
:math:`\lambda=\mathbb{P}(T=1)` for the post-treatment sampling share.

.. admonition:: Assumption 1 (Sampling and stationarity)
   :class: assumption

   For panel data, the observations :math:`\{(Y_{i0},Y_{i1},D_i,X_i)\}_{i=1}^n` are
   independent and identically distributed across units. The two outcomes within a unit may be
   dependent.

   For repeated cross-sections, :math:`\{(Y_i,D_i,X_i,T_i)\}_{i=1}^n` are independent draws
   from a mixture of the pre-treatment and post-treatment populations. For any measurable set
   :math:`A` of outcome, group, and covariate values,

   .. math::

      \begin{aligned}
      \mathbb{P}((Y,D,X)\in A,T=1)
      &=\lambda\mathbb{P}((Y_1,D,X)\in A\mid T=1),\\
      \mathbb{P}((Y,D,X)\in A,T=0)
      &=(1-\lambda)\mathbb{P}((Y_0,D,X)\in A\mid T=0),
      \end{aligned}

   where :math:`0<\lambda<1`. The joint distribution of group membership and covariates is
   unchanged across the sampled periods,

   .. math::

      \mathcal{L}(D,X\mid T=1)=\mathcal{L}(D,X\mid T=0).

   The paper also accommodates separate random samples of fixed sizes :math:`n_1` and
   :math:`n_0`, provided their share approaches :math:`\lambda\in(0,1)`.

Stationarity concerns the joint distribution, rather than just each covariate's overall mean.
If the treated group's composition changes between surveys, a difference in observed means can
mix a treatment effect with a change in who was sampled. The repeated-cross-section results
below rely on ruling out that source of change.

What identifies the ATT
-----------------------

The comparison group can supply the missing trend only if its untreated change is informative
about the treated group's untreated change. We impose this restriction at each covariate value.
This permits different trends for different values of :math:`X`, rather than requiring the two
groups to have parallel trends before covariate adjustment.

.. admonition:: Assumption 2 (Conditional parallel trends)
   :class: assumption

   The untreated mean change is the same in both groups conditional on pre-treatment covariates,

   .. math::

      \mathbb{E}[Y_1(0)-Y_0(0)\mid D=1,X]
      =\mathbb{E}[Y_1(0)-Y_0(0)\mid D=0,X]

   almost surely on the covariate support relevant to treated units.

The groups may differ both in their outcome levels and in the covariates that predict their
trends. What we require is a comparison group with the same untreated mean change
after conditioning on those covariates. To make that comparison possible, define the propensity
score :math:`p(X)=\mathbb{P}(D=1\mid X)` and the treated share :math:`q=\mathbb{E}[D]`.

.. admonition:: Assumption 3 (Overlap for the ATT)
   :class: assumption

   There is an :math:`\varepsilon>0` such that

   .. math::

      q>\varepsilon,
      \qquad
      p(X)\leq 1-\varepsilon

   almost surely. The treated group's population share must exceed a positive constant.
   Comparison units are available at covariate values represented among treated units, as
   needed for estimating their untreated conditional trends. An ATT does not require a positive
   lower bound on :math:`p(X)` at every covariate value.

These restrictions are more flexible than the conditional-mean model behind a common two-period
regression. For a pooled observation in period :math:`t`, write that regression as

.. math::

   Y_{it}=\alpha_1+\alpha_2t+\alpha_3D_i
   +\tau^{fe}(tD_i)+\theta'X_i+\varepsilon_{it}.

If we interpret its right-hand side as the correctly specified conditional mean, it gives the
same covariate slope to both groups in both periods. Under conditional parallel trends, that
model implies homogeneous conditional treatment effects and no covariate-specific observed
trends,

.. math::

   \begin{aligned}
   \mathbb{E}[Y_1(1)-Y_1(0)\mid D=1,X]&=\tau^{fe},\\
   \mathbb{E}[Y_1-Y_0\mid D=d,X]
   &=\mathbb{E}[Y_1-Y_0\mid D=d],\qquad d\in\{0,1\}.
   \end{aligned}

Sant'Anna and Zhao's Remark 1 explains why the coefficient need not equal the ATT when those
additional restrictions fail. The doubly robust estimators allow covariate-specific trends and
treatment effect heterogeneity without imposing this regression's common-slope specification.

.. admonition:: Keep the identifying assumptions
   :class: important

   Double robustness protects against misspecifying one nuisance model only under conditional
   parallel trends, overlap, and the sampling conditions above. Adding covariates
   through ``xformla`` does not establish those assumptions for your application.

Two routes to the missing trend
-------------------------------

We can model the comparison group's outcomes or reweight its covariate distribution. Each route
recovers the ATT under the assumptions above through a different nuisance function.
Here, a nuisance function is an auxiliary feature of the observed-data distribution that helps
us estimate the treatment effect.

The outcome regression route uses
:math:`m_{d,t}^p(X)=\mathbb{E}[Y_t\mid D=d,X]` for a panel, and
:math:`m_{d,t}^{rc}(X)=\mathbb{E}[Y\mid D=d,T=t,X]` for repeated cross-sections.
The superscripts identify the sampling design. In either case, write
:math:`m_{d,\Delta}=m_{d,1}-m_{d,0}` for the corresponding conditional mean change.
Conditional parallel trends gives the panel representation

.. math::

   \tau=\mathbb{E}[\Delta Y-m_{0,\Delta}^p(X)\mid D=1].

For repeated cross-sections, stationarity lets us average the predicted comparison-group change
over the common covariate distribution of treated units,

.. math::

   \begin{aligned}
   \tau={}&\mathbb{E}[Y\mid D=1,T=1]
   -\mathbb{E}[Y\mid D=1,T=0]\\
   &-\mathbb{E}[m_{0,\Delta}^{rc}(X)\mid D=1].
   \end{aligned}

The inverse probability weighting route instead uses :math:`p(X)`. Multiplying comparison units
by the odds :math:`p(X)/(1-p(X))` reproduces the treated group's covariate distribution after
normalization. The resulting representations are

.. math::

   \tau=\frac{1}{q}\mathbb{E}\left[
   \frac{D-p(X)}{1-p(X)}\Delta Y\right]

for a panel, and

.. math::

   \tau=\frac{1}{q}\mathbb{E}\left[
   \frac{D-p(X)}{1-p(X)}
   \frac{T-\lambda}{\lambda(1-\lambda)}Y\right]

for repeated cross-sections. An outcome regression estimator needs a correct model of the
comparison group's mean change whereas IPW needs a correct propensity score model.
Since neither requirement implies the other, choosing either approach leaves the estimate
dependent on that modeling choice.

Combining the two models
------------------------

A doubly robust score adds a weighted correction to the outcome regression. If the outcome
model is wrong, a correct propensity score makes the correction recover its missing adjustment.
If the outcome model is correct, the remaining comparison-group residuals have mean zero, even
under misspecified weights. We will use :math:`\pi(X)` for a working propensity score and
:math:`\mu` for working outcome regressions to distinguish those models from the true functions.
For the weighted averages below to be defined, all their expectations must exist and each
normalizing denominator must be positive and finite.

Panel data
~~~~~~~~~~

With a panel, the outcome model can target the change directly. Let
:math:`\mu_{0,\Delta}^p(X)` approximate :math:`m_{0,\Delta}^p(X)` and define normalized weights

.. math::

   w_1^p(D)=\frac{D}{q},
   \qquad
   w_0^p(D,X;\pi)=
   \frac{\pi(X)(1-D)/(1-\pi(X))}
   {\mathbb{E}[\pi(X)(1-D)/(1-\pi(X))]}.

With both weights normalized to have expectation one over the population used for estimation,
the doubly robust estimand compares the groups after subtracting the predicted untreated change,

.. math::

   \tau^{dr,p}=\mathbb{E}\left[
   (w_1^p-w_0^p(\pi))(\Delta Y-\mu_{0,\Delta}^p(X))\right].

When :math:`\pi=p`, weighting balances any integrable function of :math:`X` between the two
groups, including a misspecified outcome prediction. When
:math:`\mu_{0,\Delta}^p=m_{0,\Delta}^p`, the weighted comparison residual has conditional mean
zero. Either argument recovers the ATT without requiring the other model to be correct.

Repeated cross-sections
~~~~~~~~~~~~~~~~~~~~~~~

Without individual outcome changes, we need a separate outcome model in each period. Let
:math:`\mu_{d,t}^{rc}(X)` approximate :math:`m_{d,t}^{rc}(X)` and define

.. math::

   \begin{aligned}
   \mu_{d,Y}^{rc}(T,X)
   &=T\mu_{d,1}^{rc}(X)+(1-T)\mu_{d,0}^{rc}(X),\\
   \mu_{d,\Delta}^{rc}(X)
   &=\mu_{d,1}^{rc}(X)-\mu_{d,0}^{rc}(X).
   \end{aligned}

Normalization now takes place separately in each group and sampled period. For
:math:`t\in\{0,1\}`, define

.. math::

   w_{1,t}^{rc}=
   \frac{D\mathbf{1}\{T=t\}}{\mathbb{E}[D\mathbf{1}\{T=t\}]},

.. math::

   w_{0,t}^{rc}(\pi)=
   \frac{\pi(X)(1-D)\mathbf{1}\{T=t\}/(1-\pi(X))}
   {\mathbb{E}[\pi(X)(1-D)\mathbf{1}\{T=t\}/(1-\pi(X))]}.

The signed weights :math:`w_j^{rc}=w_{j,1}^{rc}-w_{j,0}^{rc}` turn these period-specific
averages into a before-and-after comparison. The first repeated-cross-section score uses only
comparison-group outcome models,

.. math::

   \tau_1^{dr,rc}=\mathbb{E}\left[
   (w_1^{rc}-w_0^{rc}(\pi))(Y-\mu_{0,Y}^{rc}(T,X))\right].

To improve precision, the second score also uses treated-group regressions through the
adjustment defined by :math:`a_t(X)=\mu_{1,t}^{rc}(X)-\mu_{0,t}^{rc}(X)`,

.. math::

   \begin{aligned}
   \tau_2^{dr,rc}=\tau_1^{dr,rc}
   &+\mathbb{E}[a_1(X)\mid D=1]
   -\mathbb{E}[a_1(X)\mid D=1,T=1]\\
   &-\mathbb{E}[a_0(X)\mid D=1]
   +\mathbb{E}[a_0(X)\mid D=1,T=0].
   \end{aligned}

Stationarity makes each adjustment zero in the population for any integrable working models.
In a sample, the adjustment uses treated observations from both periods to reduce noise from
the period-specific covariate distributions. That is how treated-group regressions can improve
efficiency without adding a model correctness requirement for identification.

.. admonition:: Theorem 1 (Doubly robust identification)
   :class: theorem

   Under Assumptions 1-3 and the observation and no-anticipation conditions above, the
   well-defined, untrimmed estimands satisfy the following result from Sant'Anna and Zhao's
   Theorem 1. For a panel,

   .. math::

      \tau^{dr,p}=\tau
      \quad\text{if}\quad
      \pi(X)=p(X)
      \quad\text{or}\quad
      \mu_{0,\Delta}^p(X)=m_{0,\Delta}^p(X)

   almost surely. For repeated cross-sections,

   .. math::

      \tau_1^{dr,rc}=\tau_2^{dr,rc}=\tau
      \quad\text{if}\quad
      \pi(X)=p(X)
      \quad\text{or}\quad
      \mu_{0,\Delta}^{rc}(X)=m_{0,\Delta}^{rc}(X)

   almost surely. The result allows either model or both models to be correct. Correct
   specification of treated-group outcome regressions is not required for these equalities.

For repeated cross-sections, the outcome condition concerns the difference of the two control
regressions because errors in their levels can cancel when we take that difference. Requiring
each level model to be correct is therefore sufficient and stronger than the identification
result needs.

How much information the design contains
----------------------------------------

Once identification tells us which population quantity the score recovers, efficiency asks how
precisely a regular estimator can recover it from a given sample. An influence function
describes the first-order contribution of one observation to the estimator's sampling error
and its variance determines the limiting variance of :math:`\sqrt{n}(\hat\tau-\tau)`.

Panel outcomes contain information about within-unit changes that repeated cross-sections cannot
observe. We can see that difference directly in the efficient influence functions from
Sant'Anna and Zhao's Proposition 1.

.. admonition:: Efficient influence functions
   :class: theorem

   Under Assumptions 1-3, the efficient influence function in the panel model is

   .. math::

      \begin{aligned}
      \eta^{e,p}={}&w_1^p(m_{1,\Delta}^p-m_{0,\Delta}^p-\tau)\\
      &+w_1^p(\Delta Y-m_{1,\Delta}^p)
      -w_0^p(p)(\Delta Y-m_{0,\Delta}^p).
      \end{aligned}

   In the stationary repeated-cross-section model, the efficient influence function is

   .. math::

      \begin{aligned}
      \eta^{e,rc}={}&\frac{D}{q}
      (m_{1,\Delta}^{rc}-m_{0,\Delta}^{rc}-\tau)\\
      &+w_{1,1}^{rc}(Y-m_{1,1}^{rc})
      -w_{1,0}^{rc}(Y-m_{1,0}^{rc})\\
      &-w_{0,1}^{rc}(p)(Y-m_{0,1}^{rc})
      +w_{0,0}^{rc}(p)(Y-m_{0,0}^{rc}).
      \end{aligned}

   Outcome regressions in these expressions are evaluated at :math:`X`. For finite second
   moments, the semiparametric efficiency bounds are

   .. math::

      V^{e,p}=\mathbb{E}[(\eta^{e,p})^2],
      \qquad
      V^{e,rc}=\mathbb{E}[(\eta^{e,rc})^2].

   These bound the asymptotic variance of regular estimators scaled by :math:`\sqrt n`.
   The corresponding first-order variance bound for an ATT estimator is :math:`V^e/n`.

The treated-group regression terms in the panel influence function cancel algebraically.
Rewriting it makes the remaining nuisance functions easier to see,

.. math::

   \eta^{e,p}=(w_1^p-w_0^p(p))(\Delta Y-m_{0,\Delta}^p)-w_1^p\tau.

Thus, panel efficiency requires a propensity score and a control-group change regression.
Repeated-cross-section efficiency also uses the treated group's conditional outcome
levels in both periods. Correctly modeling the control-group change can identify the ATT without
being enough to attain the repeated-cross-section efficiency bound.

Comparing panels and repeated cross-sections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To compare the two designs, imagine sampling from the same underlying population and revealing
either both outcomes or just one. Assuming :math:`T` is independent of
:math:`(Y_1,Y_0,D,X)` makes their population distributions compatible for an efficiency
comparison. We can then suppress the design superscripts on the common conditional outcome
means and define weighted residuals

.. math::

   H_t=\frac{D}{q}(Y_t-m_{1,t}(X))
   -\frac{p(X)(1-D)}{q(1-p(X))}(Y_t-m_{0,t}(X)).

Corollary 1 gives the efficiency loss from observing just one outcome per unit,

.. math::

   V^{e,rc}-V^{e,p}
   =\mathbb{E}\left[
   \left(\sqrt{\frac{1-\lambda}{\lambda}}H_1
   +\sqrt{\frac{\lambda}{1-\lambda}}H_0\right)^2\right]
   \geq0.

The loss is convex in the post-treatment sampling share but need not be smallest at equal
sample sizes, because the weighted residual variances may differ across periods. If :math:`\sigma_t^2=\mathbb{E}[H_t^2]` is positive in both periods, the
minimizing share is

.. math::

   \lambda^*=\frac{\sigma_1}{\sigma_0+\sigma_1}.

Equal weighted residual variances give :math:`\lambda^*=1/2`; otherwise, the noisier period
receives a larger sample in this comparison. Since the post-treatment variance is usually
unknown when a study is designed, the formula describes the information trade-off without
providing an automatically available sampling rule.

The gain from treated-group regressions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two repeated-cross-section estimators share identification conditions while their variances
can differ under those conditions. When the propensity score and all four outcome level models are correct,
Corollary 2 gives

.. math::

   \begin{aligned}
   V_1^{rc}-V_2^{rc}
   =\frac{1}{q}\operatorname{Var}\Bigg[
   &\sqrt{\frac{1-\lambda}{\lambda}}
   (m_{1,1}^{rc}(X)-m_{0,1}^{rc}(X))\\
   &+\sqrt{\frac{\lambda}{1-\lambda}}
   (m_{1,0}^{rc}(X)-m_{0,0}^{rc}(X))
   \;\Bigm|\;D=1\Bigg]\geq0.
   \end{aligned}

The second estimator attains the efficiency bound under these conditions. The gain is strictly
positive when the weighted combination inside the variance varies among treated units.
Heterogeneous conditional treatment effects alone do not guarantee a strict gain, because the
two level differences can cancel in that combination.

Fitting models for reliable inference
-------------------------------------

Double robustness for the point estimate does not automatically justify a variance calculation
that treats estimated nuisance functions as known. Under misspecification of one model, the
first-step estimation error can contribute to the ATT's influence function. Traditional
estimators account for those contributions. Sant'Anna and Zhao's improved estimators instead
choose nuisance fitting methods whose first-order conditions remove them.

The package uses a logistic propensity score and linear outcome regressions. For a panel, the
working models are

.. math::

   \pi(X;\gamma)=\Lambda(X'\gamma)
   =\frac{e^{X'\gamma}}{1+e^{X'\gamma}},
   \qquad
   \mu_{0,\Delta}^p(X;\beta)=X'\beta.

Write :math:`\mathbb{E}_n[Z]=n^{-1}\sum_{i=1}^n Z_i` for a sample average. Inverse probability
tilting estimates the propensity parameter by

.. math::

   \hat\gamma^{ipt}
   =\arg\max_\gamma\mathbb{E}_n\left[
   DX'\gamma-(1-D)e^{X'\gamma}\right].

The outcome regression fits comparison-group changes by weighted least squares,

.. math::

   \hat\beta^{wls}
   =\arg\min_b\mathbb{E}_n\left[
   (1-D)e^{X'\hat\gamma^{ipt}}(\Delta Y-X'b)^2\right].

The logistic odds equal :math:`e^{X'\hat\gamma^{ipt}}`. At an interior solution, these
optimization problems imply

.. math::

   \begin{aligned}
   \mathbb{E}_n[(D-(1-D)e^{X'\hat\gamma^{ipt}})X]&=0,\\
   \mathbb{E}_n[(1-D)e^{X'\hat\gamma^{ipt}}
   X(\Delta Y-X'\hat\beta^{wls})]&=0.
   \end{aligned}

These fitting conditions balance covariate moments and make the weighted control residuals
orthogonal to those same covariates. Together, they remove the nuisance estimation terms from
the first-order ATT expansion, including when one working model is misspecified.

For repeated cross-sections, we fit the two control level regressions separately using the same
odds weights. The locally efficient version also fits treated level regressions separately by
ordinary least squares,

.. math::

   \begin{aligned}
   \hat\beta_{0,t}^{wls}
   &=\arg\min_b\mathbb{E}_n[
   (1-D)\mathbf{1}\{T=t\}e^{X'\hat\gamma^{ipt}}(Y-X'b)^2],\\
   \hat\beta_{1,t}^{ols}
   &=\arg\min_b\mathbb{E}_n[
   D\mathbf{1}\{T=t\}(Y-X'b)^2],\qquad t\in\{0,1\}.
   \end{aligned}

Using one control regression in place of two level regressions would ignore the fact that each
survey observes a different outcome period. The treated regressions serve the efficiency
adjustment rather than the control-group counterfactual model.

The regularity behind the limit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The causal assumptions identify the target; the asymptotic results also need smooth models and
enough moments to approximate estimation error. The following conditions spell out the
parametric requirements in Appendix A of the paper. A pseudo-true parameter is the population
limit of a fitted model, whether or not that model is correct.

Let :math:`g(X;\theta)` denote any nuisance model and let :math:`W` contain the observed data
for the relevant design. After stacking the nuisance parameters in :math:`\kappa`, define
:math:`h^p(W;\kappa)` and :math:`h^{rc,1}(W;\kappa)` as the integrands in the panel and first
repeated-cross-section estimands. For the second repeated-cross-section estimand, use

.. math::

   \begin{aligned}
   h^{rc,2}(W;\kappa)={}&\frac{D}{q}
   (\mu_{1,\Delta}^{rc}-\mu_{0,\Delta}^{rc})\\
   &+w_{1,1}^{rc}(Y-\mu_{1,1}^{rc})
   -w_{1,0}^{rc}(Y-\mu_{1,0}^{rc})\\
   &-w_{0,1}^{rc}(\pi)(Y-\mu_{0,1}^{rc})
   +w_{0,0}^{rc}(\pi)(Y-\mu_{0,0}^{rc}).
   \end{aligned}

The score has expectation :math:`\tau_2^{dr,rc}` and evaluates all its outcome models
at :math:`X`. A dot below denotes a derivative with respect to the stacked parameter.

.. admonition:: Assumption A (Parametric regularity)
   :class: assumption

   Each nuisance model has a finite-dimensional parameter in a compact parameter space
   :math:`\Theta`. It is almost surely continuous on :math:`\Theta` and twice continuously
   differentiable near :math:`\theta^*`, the unique interior pseudo-true parameter. Its estimator is
   strongly consistent for that value and admits

   .. math::

      \sqrt n(\hat\theta-\theta^*)
      =\frac{1}{\sqrt n}\sum_{i=1}^n l_g(W_i;\theta^*)+o_p(1),

   where :math:`\mathbb{E}[l_g(W;\theta^*)]=0` and
   :math:`\mathbb{E}[l_g(W;\theta^*)l_g(W;\theta^*)']` is finite and positive definite.
   In a neighborhood :math:`\Theta^*` of :math:`\theta^*`, the influence functions satisfy

   .. math::

      \lim_{\delta\downarrow0}\mathbb{E}\left[
      \sup_{\substack{\theta\in\Theta^*\\\|\theta-\theta^*\|\leq\delta}}
      \|l_g(W;\theta)-l_g(W;\theta^*)\|^2\right]=0.

   For some :math:`\varepsilon>0`, the working propensity score obeys
   :math:`0<\pi(X;\gamma)\leq1-\varepsilon` almost surely throughout the interior of its
   parameter space. For each score :math:`h` used in the relevant design and some neighborhood
   :math:`\Gamma^*` of :math:`\kappa^*`,

   .. math::

      \mathbb{E}[\|h(W;\kappa^*)\|^2]<\infty,
      \qquad
      \mathbb{E}\left[\sup_{\kappa\in\Gamma^*}
      \|\dot h(W;\kappa)\|\right]<\infty.

   These are Assumptions A.1-A.2 in the paper, collected here for both sampling designs.

The improved estimator's panel influence function has a particularly short form. Let
:math:`\pi^*` and :math:`\mu^*` denote the limits of the inverse probability tilting and weighted
least squares fits. Under either-correct specification, it is

.. math::

   \phi_p=(w_1^p-w_0^p(\pi^*))
   (\Delta Y-\mu_{0,\Delta}^{p,*}(X))-w_1^p\tau.

For the first improved repeated-cross-section estimator, write
:math:`r_t=Y-\mu_{0,t}^{rc,*}(X)` and
:math:`b_t=\mathbb{E}[w_{1,t}^{rc}r_t]`. The weighted control regression's intercept makes
its population weighted residual mean zero, giving

.. math::

   \begin{aligned}
   \phi_{rc,1}={}&w_{1,1}^{rc}(r_1-b_1)
   -w_{1,0}^{rc}(r_0-b_0)\\
   &-w_{0,1}^{rc}(\pi^*)r_1+w_{0,0}^{rc}(\pi^*)r_0.
   \end{aligned}

For the second improved estimator, fitting all four level models gives

.. math::

   \begin{aligned}
   \phi_{rc,2}={}&\frac{D}{q}
   (\mu_{1,\Delta}^{rc,*}-\mu_{0,\Delta}^{rc,*}-\tau)\\
   &+w_{1,1}^{rc}(Y-\mu_{1,1}^{rc,*})
   -w_{1,0}^{rc}(Y-\mu_{1,0}^{rc,*})\\
   &-w_{0,1}^{rc}(\pi^*)(Y-\mu_{0,1}^{rc,*})
   +w_{0,0}^{rc}(\pi^*)(Y-\mu_{0,0}^{rc,*}).
   \end{aligned}

Although this expression resembles the efficient influence function, its starred regressions
can be misspecified under double robustness. Under correct specification of all the nuisance
models, it becomes the efficient influence function and attains local efficiency.

.. admonition:: Improved estimation and inference
   :class: theorem

   Under Assumptions 1-3 and A, adopt the logistic and linear working models above and fit
   them by inverse probability tilting and the specified least squares procedures. In the
   repeated-cross-section design, also require :math:`n_1/n\to\lambda\in(0,1)` as both
   period-specific sample sizes grow.

   For the untrimmed panel estimator, either :math:`\pi^*=p` or
   :math:`\mu_{0,\Delta}^{p,*}=m_{0,\Delta}^p` gives consistency. For either untrimmed
   repeated-cross-section estimator, either :math:`\pi^*=p` or
   :math:`\mu_{0,\Delta}^{rc,*}=m_{0,\Delta}^{rc}` gives consistency. In each case,

   .. math::

      \sqrt n(\hat\tau-\tau)
      =\frac{1}{\sqrt n}\sum_{i=1}^n\phi(W_i)+o_p(1)
      \;\xrightarrow{d}\;N(0,V),
      \qquad V=\mathbb{E}[\phi(W)^2],

   using the corresponding influence function above. This is the content of Theorems 2-3
   for the improved estimators. The panel estimator attains :math:`V^{e,p}` when both its
   nuisance models are correct. The second repeated-cross-section estimator attains
   :math:`V^{e,rc}` when the propensity score and all four outcome level models are correct.
   The first repeated-cross-section estimator need not attain that bound.

We can estimate :math:`V` by the sample variance of fitted influence values and use
:math:`\sqrt{\hat V/n}` as the standard error. The same influence-function formula remains valid whichever nuisance model is correct even
though its numerical variance can change across data-generating processes. Double robustness
for inference does not mean a universal variance.

Choosing the package estimator
------------------------------

The arguments in :func:`~moderndid.drdid` select both the sampling design and the nuisance
fitting procedure. Set ``panel=True`` when the same units have outcomes in both periods, and
provide their identifiers through ``idname``. Set ``panel=False`` for stationary repeated
cross-sections. The ``treatname`` column identifies membership in the group treated in the
post-treatment period, including its members observed before treatment.

For panels, ``est_method="imp"`` uses the improved estimator developed above.
``est_method="trad"`` uses maximum likelihood for the logistic propensity score and ordinary
least squares for the control-group change. Its influence function retains the first-step
estimation contributions needed under misspecification.

For repeated cross-sections, ``est_method="imp"`` implements the first improved estimator
:math:`\hat\tau_1^{dr,rc}`. Choose ``est_method="imp_local"`` for the second improved estimator
:math:`\hat\tau_2^{dr,rc}` and its additional treated-group outcome regressions.
``est_method="trad"`` implements the second estimator with traditional nuisance fits.
``est_method="trad_local"`` implements the first estimator with traditional nuisance fits instead. Since local efficiency is a property of the second score under correct nuisance
models, the option names alone do not identify which repeated-cross-section score is being
fitted.

.. admonition:: Trimming changes the score
   :class: warning

   ``trim_level`` defaults to ``0.995`` and removes control contributions whose fitted
   propensity scores reach that threshold. The displayed identification and efficiency results
   concern untrimmed scores. They do not automatically apply when trimming removes relevant
   comparisons from the estimating equation. Inspect overlap rather than treating trimming as a substitute for Assumption 3.

The estimators also clip fitted propensity scores away from zero and one for numerical
stability. Sampling weights supplied through ``weightsname`` enter the nuisance fits and ATT
calculation; the unweighted equations above describe the equal-weight case. With ``boot=False``,
the package reports analytical standard errors. ``boot=True`` selects the requested weighted
or multiplier bootstrap. Setting ``inf_func=True`` returns the fitted influence values in
``att_inf_func`` for the selected estimator.

The two-period scores also supply building blocks for the staggered-adoption estimator. The
:ref:`staggered DiD background <background-did>` explains how those comparisons become
group-time effects and how their joint influence functions support aggregation and simultaneous
inference.
