.. _background-etwfe:

Extended two-way fixed effects
==============================

With staggered adoption, you rarely have reason to expect the same treatment effect for every
cohort in every period, even though a regression with one treatment coefficient asks the data
for one effect. Extended two-way fixed effects, or ETWFE, gives each treated cohort-time cell
its own coefficient within the regression. We can then estimate those effects before choosing
how to average them.

This page follows `Wooldridge (2025) <https://doi.org/10.1007/s00181-025-02807-z>`_ for the
linear estimator and `Wooldridge (2023) <https://doi.org/10.1093/ectj/utad016>`_ for nonlinear
outcomes. We will build the untreated outcome model that identifies each effect and use it to
show why imputation and a saturated regression can give the same estimates. That connection explains
the specification fitted by :func:`~moderndid.etwfe`. The nonlinear extension also explains why
you need :func:`~moderndid.emfx` to turn index coefficients into effects on the outcome scale.

Why a separate effect for each cell
-----------------------------------

The :ref:`staggered DiD background <background-did>` describes how a conventional TWFE
regression can mix comparisons across cohorts and exposure lengths. With heterogeneous effects,
its single treatment coefficient can place negative weights on some cohort-time ATTs. Adding
covariates to that regression does not remove the restriction that one coefficient summarize
all treated observations.

By replacing that coefficient with a full set of cohort-time treatment indicators, ETWFE
allows effects to differ both across a cohort's post-treatment periods and across cohorts
observed in the same calendar period. The remaining challenge is to specify an untreated
outcome model that makes those cell coefficients interpretable.

Cohorts, periods, and potential outcomes
----------------------------------------

Begin with a balanced panel of :math:`N` units observed in periods :math:`t=1,\ldots,T`.
Treatment is absorbing, meaning that a unit stays treated after its first adoption. Let :math:`q\geq2`
be the first adoption period and :math:`\mathcal{G}` the set of observed treated cohorts.
For a cohort :math:`g\in\mathcal{G}`, define
:math:`d_{g,i}=\mathbf{1}\{\text{unit }i\text{ first adopts in period }g\}`.
Never-treated units have :math:`d_{\infty,i}=1`. We initially assume that this comparison group
exists and that each included cohort has positive population probability.

Let :math:`y_{it}(g)` denote the outcome if unit :math:`i` first adopts in period :math:`g`.
The potential outcome :math:`y_{it}(\infty)` describes remaining untreated throughout the
sample. For a member of cohort :math:`g`, the observed outcome is :math:`y_{it}=y_{it}(g)`.
The cohort-time ATT is

.. math::

   \tau_{g,t}
   =\mathbb{E}[y_t(g)-y_t(\infty)\mid d_g=1],
   \qquad g\in\mathcal{G},\quad t=g,\ldots,T.

We will use :math:`f_{s,t}=\mathbf{1}\{t=s\}` for a period dummy and
:math:`p_{g,t}=\sum_{s=g}^T f_{s,t}=\mathbf{1}\{t\geq g\}` for a cohort's post-treatment
indicator. These definitions give the observed treatment status

.. math::

   w_{it}=\sum_{g\in\mathcal{G}}d_{g,i}p_{g,t}
   =\sum_{g\in\mathcal{G}}\sum_{s=g}^T d_{g,i}f_{s,t}.

Since the post-treatment cells partition the observations where :math:`w_{it}=1`, adding
:math:`w_{it}` alongside all those cell indicators supplies no new regression variation.
Keeping it in the notation will help us describe the change from untreated to treated
predictions when we reach nonlinear models.

The assumptions behind the untreated mean
-----------------------------------------

To estimate :math:`\tau_{g,t}`, we need the untreated mean for cohort :math:`g` in period
:math:`t`. We will build it from cohort differences in baseline levels and common conditional
trends. The covariate vector :math:`\mathbf{x}_i` is time-constant, excludes the intercept,
and can contain transformations created from pre-treatment characteristics. Write
:math:`\mathbf{d}_i=(d_{g,i})_{g\in\mathcal{G}}` for the vector of treated-cohort indicators.

.. admonition:: Assumption SUTVA (No interference)
   :class: assumption

   Each unit's potential outcome depends on its own adoption date and not on the treatment
   assignments of other units. The adoption-date potential outcomes refer to the intervention
   whose effects we want to estimate.

This rules out spillovers between units that would make an untreated unit's outcome depend on
other units' treatment. Covariate adjustment needs its own restriction, because conditioning
on a variable changed by the intervention can change the causal question.

.. admonition:: Assumption NBC (No bad controls)
   :class: assumption

   If :math:`\mathbf{x}(g)` denotes the covariates under adoption date :math:`g`, require

   .. math::

      \mathbf{x}(g)=\mathbf{x}(\infty),
      \qquad g\in\mathcal{G}.

   We write :math:`\mathbf{x}=\mathbf{x}(\infty)` for this common covariate vector.
   Its distribution among cohort members can therefore be learned from their observed data.

A covariate being constant in the dataset does not by itself establish NBC. A variable
constructed from post-treatment information can remain constant across its repeated rows
while still being affected by treatment. Pre-treatment measurement gives the assumption a more credible basis, though its justification
still comes from the application's timing and causal relationships.

.. admonition:: Assumption NA (Conditional no anticipation)
   :class: assumption

   For every treated cohort :math:`g` and period before its adoption,

   .. math::

      \mathbb{E}[y_t(g)-y_t(\infty)\mid\mathbf{d},\mathbf{x}]=0,
      \qquad t<g.

   This is a restriction on conditional means. It permits individual differences that average
   to zero and is weaker than requiring identical potential outcomes before adoption.

No anticipation allows pre-treatment observations of eventual adopters to inform the untreated
mean that parallel trends carries into later periods.

.. admonition:: Assumption CPT (Conditional parallel trends)
   :class: assumption

   For all :math:`t=2,\ldots,T`,

   .. math::

      \mathbb{E}[y_t(\infty)-y_1(\infty)\mid\mathbf{d},\mathbf{x}]
      =\mathbb{E}[y_t(\infty)-y_1(\infty)\mid\mathbf{x}].

   Cohort membership may predict untreated outcome levels. Conditional on the covariates, it
   cannot predict the mean untreated change from the common baseline period.

This formulation uses all pre-treatment periods, rather than only the period immediately before
each cohort's adoption. Trends may differ across covariate values, even if they are common across
cohorts at a given value. To turn that restriction into a regression, we also specify how the
conditional means depend on the covariates.

.. admonition:: Assumption LIN (Linear conditional means)
   :class: assumption

   The baseline untreated conditional mean is linear in the chosen covariate basis,

   .. math::

      \mathbb{E}[y_1(\infty)\mid\mathbf{d},\mathbf{x}]
      =\alpha+\sum_{g\in\mathcal{G}}\beta_gd_g
      +\mathbf{x}\boldsymbol{\kappa}
      +\sum_{g\in\mathcal{G}}d_g\mathbf{x}\boldsymbol{\xi}_g.

   Its change over time has the common conditional trend specification

   .. math::

      \begin{aligned}
      &\mathbb{E}[y_t(\infty)\mid\mathbf{d},\mathbf{x}]
      -\mathbb{E}[y_1(\infty)\mid\mathbf{d},\mathbf{x}]\\
      &\qquad=\sum_{s=2}^T\gamma_sf_{s,t}
      +\sum_{s=2}^Tf_{s,t}\mathbf{x}\boldsymbol{\pi}_s,
      \qquad t=2,\ldots,T.
      \end{aligned}

   Set :math:`\gamma_1=0` and :math:`\boldsymbol{\pi}_1=\mathbf{0}`. The trend equation
   implies CPT because cohort membership does not enter its right-hand side.

Without covariates, these equations record separate cohort means and state parallel trends
without imposing further restrictions on the period means. With covariates, both generally
impose functional form restrictions.
A complete set of mutually exclusive indicators for the covariates' joint support removes those
functional form restrictions within the represented strata. A short list of separate indicators
for several covariates need not be saturated in their joint distribution.

Recovering the missing outcome by imputation
--------------------------------------------

Combining the baseline and trend models gives the untreated conditional mean. Denote it by
:math:`b_{it}` to keep the following regressions readable,

.. math::

   \begin{aligned}
   b_{it}
   &\equiv\mathbb{E}[y_{it}(\infty)\mid\mathbf{d}_i,\mathbf{x}_i]\\
   &=\alpha+\sum_g\beta_gd_{g,i}
   +\mathbf{x}_i\boldsymbol{\kappa}
   +\sum_gd_{g,i}\mathbf{x}_i\boldsymbol{\xi}_g\\
   &\quad+\sum_{s=2}^T\gamma_sf_{s,t}
   +\sum_{s=2}^Tf_{s,t}\mathbf{x}_i\boldsymbol{\pi}_s.
   \end{aligned}

Here and below, sums over :math:`g` run over :math:`\mathcal{G}`. The cohort intercepts and slopes allow selection into treatment based on untreated levels
alongside the different covariate-specific trends captured by the time interactions. CPT
excludes cohort-specific untreated time shifts from this model.

For a control observation with :math:`w_{it}=0`, NA equates the observed conditional mean
with :math:`b_{it}`. Pooled OLS on those observations can therefore identify the untreated
model when the control design has sufficient rank and covariate variation. Averaging the fitted
model over a treated cohort supplies its missing mean. At the population level,

.. math::

   \tau_{g,t}
   =\mathbb{E}[y_t\mid d_g=1]
   -\bigl[\alpha+\beta_g+\gamma_t
   +\mathbb{E}[\mathbf{x}\mid d_g=1]
   (\boldsymbol{\kappa}+\boldsymbol{\xi}_g+\boldsymbol{\pi}_t)\bigr].

NBC is what allows the observed cohort covariate distribution to stand in for its untreated
distribution in this calculation. The regression also needs enough untreated observations to
estimate the relevant coefficients. Including a cell indicator cannot recover a counterfactual
whose control model is unidentified.

Fitting the control regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Procedure 4.1 in Wooldridge (2025) first regresses outcomes from :math:`w_{it}=0` observations
on the complete control design,

.. math::

   \begin{aligned}
   y_{it}\text{ on }&
   1,\ (d_{g,i})_g,\ \mathbf{x}_i,\ (d_{g,i}\mathbf{x}_i)_g,\\
   &(f_{s,t})_{s=2}^T,\ (f_{s,t}\mathbf{x}_i)_{s=2}^T.
   \end{aligned}

Write the resulting coefficient vector as

.. math::

   \widetilde{\boldsymbol{\theta}}_0
   =\bigl(\tilde\alpha,(\tilde\beta_g)_g,
   \widetilde{\boldsymbol{\kappa}},
   (\widetilde{\boldsymbol{\xi}}_g)_g,
   (\tilde\gamma_s)_{s=2}^T,
   (\widetilde{\boldsymbol{\pi}}_s)_{s=2}^T\bigr).

Pooling all admissible untreated observations under the common conditional trend model
imposes more structure across pre-treatment periods than separate two-period comparisons.
That structure is useful when it is credible; using more observations alone does not guarantee
a smaller variance under every pattern of serial dependence.

Imputing and averaging
~~~~~~~~~~~~~~~~~~~~~~

The second step predicts the untreated conditional mean for every treated observation,

.. math::

   \begin{aligned}
   \tilde y_{it}(\infty)
   ={}&\tilde\alpha+\sum_g\tilde\beta_gd_{g,i}
   +\mathbf{x}_i\widetilde{\boldsymbol{\kappa}}
   +\sum_gd_{g,i}\mathbf{x}_i\widetilde{\boldsymbol{\xi}}_g\\
   &+\sum_{s=2}^T\tilde\gamma_sf_{s,t}
   +\sum_{s=2}^Tf_{s,t}\mathbf{x}_i\widetilde{\boldsymbol{\pi}}_s.
   \end{aligned}

Subtracting this prediction from the observed outcome gives
:math:`\widetilde{te}_{it}=y_{it}-\tilde y_{it}(\infty)`, a residual that contains both the
treatment effect and outcome noise. You therefore do not observe an individual causal effect
merely by subtracting a conditional mean prediction.

The third step averages those residuals within a cohort-time cell. Define
:math:`N_g=\sum_i d_{g,i}`,
:math:`\bar y_{g,t}=N_g^{-1}\sum_i d_{g,i}y_{it}`, and
:math:`\bar{\mathbf{x}}_g=N_g^{-1}\sum_i d_{g,i}\mathbf{x}_i`. Then

.. math::

   \begin{aligned}
   \tilde\tau_{g,t}
   &=N_g^{-1}\sum_i d_{g,i}\widetilde{te}_{it}\\
   &=\bar y_{g,t}
   -\bigl[\tilde\alpha+\tilde\beta_g+\tilde\gamma_t
   +\bar{\mathbf{x}}_g
   (\widetilde{\boldsymbol{\kappa}}
   +\widetilde{\boldsymbol{\xi}}_g
   +\widetilde{\boldsymbol{\pi}}_t)\bigr].
   \end{aligned}

The uncertainty in this estimate includes uncertainty in the control regression and the cohort
averages. A single saturated regression reproduces the point estimate and puts the regression
coefficients in one covariance matrix. Sampling variation in estimated covariate means still
needs attention when the target is a population ATT.

The saturated ETWFE regression
------------------------------

The imputation argument suggests exactly what the full regression must contain. We keep the
untreated controls, add every post-treatment cohort-time indicator, and interact each treated
cell with covariates centered at that cohort's mean. In the population, let
:math:`\boldsymbol{\mu}_g=\mathbb{E}[\mathbf{x}\mid d_g=1]` and
:math:`\dot{\mathbf{x}}_{ig}=\mathbf{x}_i-\boldsymbol{\mu}_g`. The working regression is

.. math::

   \begin{aligned}
   y_{it}={}&b_{it}
   +\sum_g\sum_{s=g}^T\tau_{g,s}
   (w_{it}d_{g,i}f_{s,t})\\
   &+\sum_g\sum_{s=g}^T
   (w_{it}d_{g,i}f_{s,t}\dot{\mathbf{x}}_{ig})
   \boldsymbol{\delta}_{g,s}+u_{it}.
   \end{aligned}

Centering makes the average covariate interaction zero in its cohort. If conditional treatment
effects are linear in the chosen basis, the conditional effect is
:math:`\tau_{g,s}+\dot{\mathbf{x}}_{ig}\boldsymbol{\delta}_{g,s}` and its cohort average is
:math:`\tau_{g,s}`. Without that linearity condition, the slopes need not describe the true
conditional effects. The imputation equivalence still lets us estimate the cohort average under
the untreated mean model without requiring linear treatment effect heterogeneity.

In the fitted regression, replace population means by :math:`\bar{\mathbf{x}}_g` and use
:math:`\dot{\mathbf{x}}_{ig}=\mathbf{x}_i-\bar{\mathbf{x}}_g`. The coefficient on a cell
indicator then equals its imputation estimate. Without centering, that coefficient refers to
the treatment contrast at zero covariates rather than its cohort average.

.. admonition:: Proposition 5.2 (Imputation and pooled OLS)
   :class: theorem

   Use the same balanced sample, time-constant covariates, and control design in the
   control-only imputation regression and the full pooled regression above. Include each
   post-treatment cell and its cohort-centered covariate interactions. If the control design
   has full column rank after removing redundant columns, the estimates satisfy

   .. math::

      \hat\tau_{g,t}^{pols}=\tilde\tau_{g,t},
      \qquad g\in\mathcal{G},\quad t=g,\ldots,T.

   The estimated coefficients on the common control regressors are also identical.
   This algebraic equivalence from Wooldridge (2025) holds independently of whether the
   identifying assumptions are true for the data.

Although the algebra establishes that the procedures agree, the causal interpretation of
their common estimate still depends on the assumptions. Under SUTVA, NBC, NA, CPT, and LIN, independently sampled unit histories
with finite second moments and a nonsingular population control-regressor moment matrix give
consistent estimates as :math:`N` grows with :math:`T` fixed. Each included cohort must also
have positive probability so that its sample size grows. In the no-covariate balanced design, Wooldridge also establishes unbiasedness
under random sampling, NA, and unconditional parallel trends. That result does not imply that
every nonlinear or covariate-adjusted ETWFE estimate is unbiased in a finite sample.

.. admonition:: Keep the full covariate design
   :class: important

   ``xformla`` controls enter with cohort and time interactions, as well as centered
   treatment-cell interactions. These terms implement the conditional trend adjustment.
   Adding covariates only as main effects would not reproduce the imputation argument.

Why unit fixed effects can give the same answer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The pooled regression controls for cohort differences in the untreated outcome mean rather
than absorbing an intercept for every unit as the usual panel regression does. For the
balanced, time-constant specification above, the two-way Mundlak result explains why those
additional unit intercepts leave the cell effect estimates unchanged.

To state the general result, let :math:`\mathbf{z}_{it}` contain the regressors whose
coefficients we want to recover. Define their averages and double-demeaned values,

.. math::

   \begin{aligned}
   \bar{\mathbf{z}}_{i\cdot}&=T^{-1}\sum_t\mathbf{z}_{it},&
   \bar{\mathbf{z}}_{\cdot t}&=N^{-1}\sum_i\mathbf{z}_{it},\\
   \bar{\mathbf{z}}&=(NT)^{-1}\sum_i\sum_t\mathbf{z}_{it},&
   \ddot{\mathbf{z}}_{it}
   &=\mathbf{z}_{it}-\bar{\mathbf{z}}_{i\cdot}
   -\bar{\mathbf{z}}_{\cdot t}+\bar{\mathbf{z}}.
   \end{aligned}

.. admonition:: Theorem 3.1 (Two-way Mundlak equivalence)
   :class: theorem

   In a balanced panel, suppose
   :math:`\sum_i\sum_t\ddot{\mathbf{z}}_{it}'\ddot{\mathbf{z}}_{it}` is nonsingular.
   The coefficients on :math:`\mathbf{z}_{it}` from a regression with unit and period
   fixed effects equal those from pooled OLS on

   .. math::

      1,\ \mathbf{z}_{it},\
      \bar{\mathbf{z}}_{i\cdot},\
      \bar{\mathbf{z}}_{\cdot t},\
      \mathbf{r}_i,\ \mathbf{m}_t,

   where :math:`\mathbf{r}_i` contains any additional time-constant regressors and
   :math:`\mathbf{m}_t` contains any additional regressors constant across units in each
   period. Redundant columns can be omitted. Dropping any subset of
   :math:`(\mathbf{r}_i,\mathbf{m}_t)` leaves the coefficients on :math:`\mathbf{z}_{it}`
   unchanged. This is Theorem 3.1 in Wooldridge (2025).

In the saturated ETWFE design, cohort indicators and their covariate interactions span the
unit averages of the treatment regressors just as period indicators and their covariate
interactions span the period averages. These terms establish the equality between the
cohort-based pooled regression and the corresponding regression with unit fixed effects.
Random effects with the matching Mundlak controls also gives the same treatment coefficients.

For this matched design, the equivalence chain is

.. math::

   \hat\tau^{imputation}_{g,t}
   =\hat\tau^{pols}_{g,t}
   =\hat\tau^{twfe}_{g,t}
   =\hat\tau^{re}_{g,t}.

Unit-based imputation in the matched balanced specification likewise gives the same cohort-time
averages, even though its individual residuals differ. Wooldridge's Section 5.5 relates this to
`Borusyak, Jaravel, and Spiess (2024) <https://doi.org/10.1093/restud/rdae007>`_.
The equalities concern matched regressors, samples, and averaging schemes. They are not a claim
that arbitrary implementations bearing those estimator names must agree.

In :func:`~moderndid.etwfe`, supplying ``idname`` makes a linear model absorb unit and
period effects. Without an identifier, it absorbs cohort and period effects. ``fe="none"``
uses explicit cohort and period dummies. These specifications implement the same cell contrasts
in the balanced setting above; time-varying controls and missing observations can break that
equivalence.

Choosing which untreated rows contribute
----------------------------------------

With ``cgroup="notyet"``, the package uses the never-treated group and the untreated rows of
eventual adopters. This fits the model developed above using all available pre-treatment periods.
With ``cgroup="never"``, each treated cohort instead uses period :math:`g-1` as its reference
and receives a separate indicator for every other period. Earlier pre-treatment rows therefore
estimate placebo contrasts rather than contributing to a pooled pre-treatment baseline.

The two designs impose different restrictions on how the pre-treatment observations enter
the untreated model. Since a not-yet-treated fit has no estimated leads, an event graph cannot
create those placebo coefficients after estimation.

If no never-treated units exist, we can use the latest-treated cohort as the reference only
while it remains untreated. Let :math:`g_{\max}` denote that adoption date. A treatment
contrast relative to that path is

.. math::

   \tau_{(g:g_{\max}),t}
   =\mathbb{E}[y_t(g)-y_t(g_{\max})\mid d_g=1].

Under no anticipation, it equals the usual untreated ATT for
:math:`g\leq t<g_{\max}`. After the reference cohort adopts, its outcome no longer identifies
the never-treated counterfactual. The package's not-yet-treated design drops reference-cohort
rows from its adoption onward and drops those periods for the other cohorts as well. It does
not identify post-adoption effects by treating the reference cohort as if it were still untreated. Wooldridge's Section 5.4 also considers
earlier-versus-later adoption effects after :math:`g_{\max}` under a modified CPT assumption
stated for :math:`y_t(g_{\max})`. Those are effects relative to the later-adoption path,
rather than never-treated ATTs. The package excludes those periods from its estimation sample.

The ``gref`` argument selects the reference cohort. It defaults to never-treated units if
they exist, otherwise to the latest-treated cohort for ``cgroup="notyet"``.
A never-treated design requires a never-treated reference group. Cohorts with no usable
untreated observation leave the estimation sample, because their effects have no identified
baseline under the selected design.

What the pre-treatment coefficients test
----------------------------------------

The leads-and-lags version asks whether the cohort's earlier change differs from its comparison
group's change relative to :math:`g-1`. For each cohort, include pre-treatment cell coefficients
:math:`\theta_{g,s}` for :math:`s<g-1` and post-treatment coefficients for :math:`s\geq g`.
The omitted cell at :math:`s=g-1` fixes the reference contrast at zero,

.. math::

   \begin{aligned}
   y_{it}={}&b_{it}
   +\sum_g\sum_{s<g-1}\theta_{g,s}d_{g,i}f_{s,t}\\
   &+\sum_g\sum_{s<g-1}
   d_{g,i}f_{s,t}\dot{\mathbf{x}}_{ig}\boldsymbol{\nu}_{g,s}\\
   &+\sum_g\sum_{s\geq g}\tau_{g,s}d_{g,i}f_{s,t}\\
   &+\sum_g\sum_{s\geq g}
   d_{g,i}f_{s,t}\dot{\mathbf{x}}_{ig}\boldsymbol{\delta}_{g,s}+u_{it}.
   \end{aligned}

Here, :math:`\boldsymbol{\nu}_{g,s}` gives the pre-treatment covariate interaction slopes,
and :math:`\boldsymbol{\delta}_{g,s}` gives the post-treatment slopes. Without covariates,
a pre-treatment coefficient represents

.. math::

   \begin{aligned}
   &\mathbb{E}[y_s(\infty)-y_{g-1}(\infty)\mid d_g=1]\\
   &\qquad-\mathbb{E}[y_s(\infty)-y_{g-1}(\infty)\mid d_\infty=1].
   \end{aligned}

With covariates, the corresponding contrast is adjusted for the conditional mean model and
averaged over the treated cohort. A joint test of the leads examines restrictions on observed
pre-treatment trends. Failure to reject those restrictions does not establish parallel trends
after adoption.

Wooldridge's Section 6 shows that, in the saturated linear specification, fitting the leads
with all observations gives the same pre-treatment contrasts as fitting the matching regression
using only untreated observations. Flexible treatment-cell terms prevent post-treatment
heterogeneity from forcing itself into the pre-treatment fit. The matched no-covariate
leads-and-lags specification reproduces the `Sun and Abraham (2021)
<https://doi.org/10.1016/j.jeconom.2020.09.006>`_ interaction-weighted approach after aggregation.
The fully interacted regression adjustment can also reproduce the corresponding
`Callaway and Sant'Anna (2021) <https://doi.org/10.1016/j.jeconom.2020.12.001>`_ long-difference
outcome regression comparisons when the covariates, controls, and reference periods match.

Adding leads changes the restrictions imposed on the pre-treatment data. Under CPT, pooling
all pre-treatment periods can improve precision. The ranking depends on the outcome covariance
structure rather than the number of periods alone. Leads primarily supply a diagnostic comparison; they do not by themselves
model how a violation of parallel trends would continue after treatment.

In the package, fit ``cgroup="never"`` and call
``emfx(result, type="event", post_only=False)`` to report those pre-treatment contrasts.
The reference exposure :math:`e=-1` is shown at zero. Since ``cgroup="notyet"`` fits
post-treatment cells only, the same aggregation option does not supply a pre-treatment test
for that control design.

Allowing cohort-specific trends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the pre-treatment evidence suggests systematically different cohort trends, one possible
model adds :math:`\eta_gd_{g,i}t` to the untreated conditional mean. With at least two
pre-treatment periods and sufficient rank, a linear cohort trend can be estimated before adoption
and extrapolated into the post-treatment periods. That extrapolation replaces the common-trend
restriction with another substantive assumption.

For a single treated cohort observed in periods 1, 2, and 3, let
:math:`a_t=\mathbb{E}[y_t\mid D=1]-\mathbb{E}[y_t\mid D=0]`.
A linear untreated gap predicts :math:`a_3(0)=2a_2-a_1`, giving

.. math::

   \tau_3=a_3-2a_2+a_1.

This second difference of the between-group gap identifies the period-3 effect if the
untreated gap would have continued linearly. More pre-treatment periods permit higher-order
trend models whose post-treatment extrapolation still needs justification from the
application's untreated outcome dynamics. Trend terms
can also be highly correlated with the treatment indicators and substantially reduce precision.

The corresponding imputation and saturated-regression equivalences can be derived with the
matched trend design. The current ``etwfe`` interface does not supply a cohort-trend argument,
so this theoretical extension is not selected by changing ``xformla`` alone.

Nonlinear outcomes and index parallel trends
--------------------------------------------

An additive untreated mean model may be poorly suited to a binary outcome near zero or one
or give negative predictions for a nonnegative outcome. A nonlinear mean model respects the
outcome's range but also changes the parallel-trends assumption that we need to state before
interpreting its treatment coefficients.

Let :math:`G` be a known, strictly increasing mean function and let
:math:`a_{it}` have the same additive cohort, covariate, and period terms as :math:`b_{it}`.
The nonlinear untreated mean is :math:`G(a_{it})`.

.. admonition:: Assumption CIPT (Conditional index parallel trends)
   :class: assumption

   The untreated conditional mean satisfies

   .. math::

      \begin{aligned}
      \mathbb{E}[y_t(\infty)\mid\mathbf{d},\mathbf{x}]
      =G\Bigl(&\alpha+\sum_g\beta_gd_g
      +\mathbf{x}\boldsymbol{\kappa}
      +\sum_gd_g\mathbf{x}\boldsymbol{\xi}_g\\
      &+\gamma_t+\mathbf{x}\boldsymbol{\pi}_t\Bigr),
      \end{aligned}

   where :math:`\gamma_1=0` and :math:`\boldsymbol{\pi}_1=\mathbf{0}`.
   Equivalently, the change in the transformed untreated mean obeys

   .. math::

      \begin{aligned}
      &G^{-1}\!\left(\mathbb{E}[y_t(\infty)\mid\mathbf{d},\mathbf{x}]\right)
      -G^{-1}\!\left(\mathbb{E}[y_1(\infty)\mid\mathbf{d},\mathbf{x}]\right)\\
      &\qquad=\gamma_t+\mathbf{x}\boldsymbol{\pi}_t.
      \end{aligned}

   This is the staggered conditional index restriction in Wooldridge (2023), together with its
   specified baseline mean. It replaces the linear mean assumptions CPT and LIN.

The identity mean function gives the linear model whereas an exponential mean imposes common
conditional growth factors. Without covariates, the exponential model requires

.. math::

   \frac{\mathbb{E}[y_t(\infty)\mid\mathbf{d}]}
   {\mathbb{E}[y_1(\infty)\mid\mathbf{d}]}=e^{\gamma_t}.

A logistic mean instead makes the untreated log-odds change common across cohorts. Since
these mean functions impose different identifying restrictions, changing the family changes
more than the scale of a regression output.

Suppose the untreated latent outcome is
:math:`y_t^*(\infty)=\alpha+\beta D+\gamma_t+U_t` and the corresponding binary response is
:math:`y_t(\infty)=\mathbf{1}\{y_t^*(\infty)>0\}`. If :math:`U_t` is independent of
:math:`D` and has the same cumulative distribution :math:`F` in every period, then

.. math::

   \mathbb{E}[y_t(\infty)\mid D]
   =1-F(-(\alpha+\beta D+\gamma_t))
   \equiv G(\alpha+\beta D+\gamma_t).

The latent mean can have parallel additive trends while the observed response probabilities do
not. Index parallel trends describes this conditional response model without imposing additive
changes on probabilities near the boundary.

Quasi-likelihood and the canonical link
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The untreated nonlinear mean can be fitted by pooled quasi-maximum likelihood on untreated
observations. Correct specification of the conditional mean, identification, and the usual
sampling and moment conditions yield consistency as :math:`N` grows with fixed :math:`T`.
The quasi-likelihood density need not be the outcome's true conditional distribution.

The full pooled model adds treatment terms inside the index. Let
:math:`\ell_{g,t}(\mathbf{x}_i)=\delta_{g,t}
+\dot{\mathbf{x}}_{ig}\boldsymbol{\zeta}_{g,t}` denote a cell-specific index shift. Then

.. math::

   \mathbb{E}[y_{it}\mid\mathbf{d}_i,\mathbf{x}_i]
   =G\left(a_{it}
   +\sum_g\sum_{s=g}^T w_{it}d_{g,i}f_{s,t}
   \ell_{g,s}(\mathbf{x}_i)\right)

is the pooled working mean. Under a general link, this also models treated conditional means.
Correct specification in the untreated observations alone need not make the pooled response
contrasts consistent. A canonical link provides a useful exception through the same kind of
algebraic equivalence we saw for OLS.

.. admonition:: Proposition 3.1 (Canonical-link equivalence)
   :class: theorem

   In the staggered, absorbing-treatment design, fit the matching imputation and saturated
   pooled models in Wooldridge (2023). Suppose :math:`G^{-1}` is the canonical link for the
   chosen linear exponential family quasi-likelihood and the pooled solution is unique.
   Their common untreated-model parameter estimates agree. The estimated response-scale ATTs
   also agree across the two procedures,

   .. math::

      \hat\tau_{g,t}^{pooled}=\hat\tau_{g,t}^{imputation},
      \qquad g\in\mathcal{G},\quad t=g,\ldots,T.

   This is an algebraic result for the specified designs. The causal interpretation still
   requires no anticipation, the untreated index model, and enough variation to identify it.

The main pairings are an identity mean with a normal quasi-likelihood, a logistic mean with
a Bernoulli quasi-likelihood, and an exponential mean with a Poisson quasi-likelihood.
The paper also develops fractional responses and responses with known upper bounds through
appropriate logistic quasi-likelihoods. The package's ``logit`` and ``probit`` families
fit binary response models; that theoretical coverage does not establish support for fractional
or variable-upper-bound outcomes in the current interface.

The package supports ``family="poisson"``, ``family="logit"``, and
``family="probit"`` in addition to the linear default. Since probit has no corresponding canonical Bernoulli link, the imputation-pooled equality
above does not apply to this family even in the matched design.
Nonlinear fits use cohort and period dummies rather than absorbed unit effects.
Supplying ``idname`` selects clustering and unit counts for these fits.

Effects on the outcome scale
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For nonlinear means, the index coefficient is not an ATT in outcome units. We need to compare
treated and untreated response predictions for each cohort member and then average. The
imputation estimate is

.. math::

   \hat\tau_{g,t}^{imputation}
   =\bar y_{g,t}
   -N_g^{-1}\sum_i d_{g,i}G(\hat a_{it}),

where :math:`\hat a_{it}` predicts the untreated index. The pooled response contrast is

.. math::

   \hat\tau_{g,t}^{pooled}
   =N_g^{-1}\sum_i d_{g,i}
   \left[G(\hat a_{it}+\hat\ell_{g,t}(\mathbf{x}_i))
   -G(\hat a_{it})\right].

By equating the cell's average treated prediction with its observed mean, canonical-link
score equations make the two response contrast estimates agree in the matched design. For nonlinear families,
:func:`~moderndid.emfx` computes the second expression from observation-level predictions.

.. admonition:: Report response-scale effects
   :class: tip

   Nonlinear cell coefficients in an ``etwfe`` result describe index shifts. Use ``emfx``
   to obtain treatment contrasts in the outcome's units before comparing cohorts or averaging
   their effects. Centering a nonlinear index does not turn its coefficient into a response ATT.

For an exponential mean, the conditional proportional effect is exactly

.. math::

   \frac{G(a_{it}+\ell_{g,t}(\mathbf{x}_i))-G(a_{it})}{G(a_{it})}
   =e^{\ell_{g,t}(\mathbf{x}_i)}-1.

If the index shift is constant within a cell, :math:`e^{\delta_{g,t}}-1` gives its exact common
proportional effect. With covariate-dependent shifts, averaging
proportional effects and averaging effects in outcome units answer different questions,

.. math::

   \tau_{g,t}
   =\mathbb{E}\left[
   G(a_{it})\bigl(e^{\ell_{g,t}(\mathbf{x}_i)}-1\bigr)
   \mid d_g=1\right].

Wooldridge also discusses Poisson unit fixed effects, whose conditional approach avoids the
usual fixed-period incidental-parameter problem. The no-covariate saturated design gives the
same cell effects as pooled Poisson; adding covariates can break that equality. This is a
theoretical comparison, since the package's nonlinear ETWFE fits use cohort dummies.

Averaging the effects you want to report
----------------------------------------

Once we have cohort-time ATTs, aggregation lets us choose the population and time dimension
represented by a summary. :func:`~moderndid.emfx` implements overall, group, calendar, and
event-time summaries. The formulas below use a balanced panel and equal observation weights;
only cells identified in the selected control design enter an average.

The overall effect
~~~~~~~~~~~~~~~~~~

Let :math:`\mathcal{C}_+` contain the identified post-treatment cells and
:math:`K_g=|\{t:(g,t)\in\mathcal{C}_+\}|` count a cohort's included periods.
The overall effect averages over treated observations,

.. math::

   \hat{\bar\tau}_{\omega}
   =\sum_{(g,t)\in\mathcal{C}_+}\hat\omega_g\hat\tau_{g,t},
   \qquad
   \hat\omega_g=\frac{N_g}{\sum_hK_hN_h}.

If all post-treatment cells through :math:`T` are identified, :math:`K_g=T-g+1`. Because larger
cohorts have more weight in each cell and earlier cohorts contribute more periods when their
effects remain identified throughout the sample, the result describes an average treated
observation rather than an average cohort with equal weight on every adoption date.

Event time, cohorts, and calendar time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Event time :math:`e=t-g` records the number of periods since adoption. Let
:math:`\mathcal{G}_e=\{g:(g,g+e)\in\mathcal{C}_+\}`. The effect at that exposure length is

.. math::

   \hat\tau_{\omega,e}
   =\sum_{g\in\mathcal{G}_e}
   \frac{N_g}{\sum_{h\in\mathcal{G}_e}N_h}
   \hat\tau_{g,g+e}.

The nonnegative weights sum to one over the contributing cohorts, whose membership usually
changes with :math:`e`. Differences between exposure effects can therefore reflect both
treatment dynamics and changes in cohort composition. A reported event window restricts the exposures; it does not automatically
hold that composition fixed.

Group and calendar averages use the same cell estimates along different dimensions,

.. math::

   \hat\tau_g=\frac{1}{K_g}\sum_{t:(g,t)\in\mathcal{C}_+}\hat\tau_{g,t},
   \qquad
   \hat\tau_t=
   \frac{\sum_{g:(g,t)\in\mathcal{C}_+}N_g\hat\tau_{g,t}}
   {\sum_{g:(g,t)\in\mathcal{C}_+}N_g}.

The group effects compare cohorts' average post-treatment experiences. Calendar effects describe
the mean treatment effect among cohorts observed treated in a particular period. Their changes
do not by themselves isolate the influence of macroeconomic conditions or concurrent policies.

Each aggregation reports a one-number summary that weights cohorts by their unit counts in
the group case. The event and calendar summaries give equal weight to their reported
nonnegative event times and calendar periods, respectively,

.. math::

   \begin{aligned}
   \hat\theta_{group}
   &=\frac{\sum_gN_g\hat\tau_g}{\sum_gN_g},\\
   \hat\theta_{event}
   &=|\mathcal{E}_+|^{-1}\sum_{e\in\mathcal{E}_+}\hat\tau_{\omega,e},\\
   \hat\theta_{calendar}
   &=|\mathcal{T}_+|^{-1}\sum_{t\in\mathcal{T}_+}\hat\tau_t.
   \end{aligned}

Here, :math:`\mathcal{E}_+` and :math:`\mathcal{T}_+` are the reported post-treatment
exposures and calendar periods. These summaries need not equal the overall average, because
their period and cohort weights differ. In an unbalanced sample, ``emfx`` averages within
levels over their available observations. Its marginal-effect aggregation does not use
``weightsname`` as population weights, even when those weights enter the fitted regression.

Inference for a fitted effect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The joint covariance matrix supplied by the regression lets aggregation retain dependence
across the cell estimates. If :math:`\boldsymbol{\beta}` stacks the fitted coefficients and
:math:`m(\boldsymbol{\beta})` is a response contrast or aggregate, the delta method uses

.. math::

   \widehat{\operatorname{Var}}(m(\hat{\boldsymbol{\beta}}))
   =J(\hat{\boldsymbol{\beta}})
   \widehat{\operatorname{Var}}(\hat{\boldsymbol{\beta}})
   J(\hat{\boldsymbol{\beta}})',
   \qquad
   J=\frac{\partial m}{\partial\boldsymbol{\beta}'}.

For a linear fixed-weight average, this reduces to a quadratic form in the cell coefficient
covariance matrix. For nonlinear effects, the gradient also includes the derivative of the
inverse link. The package evaluates those gradients observation by observation before averaging.

Supplying ``idname`` makes the default covariance estimate cluster by unit. This allows
within-unit serial correlation and heteroskedasticity under the conditions for cluster inference,
including enough independent clusters. If treatment is assigned at a higher level, ``vcov``
can select that clustering level. Without an identifier, the default is heteroskedasticity-robust
and does not account for within-unit dependence.

.. admonition:: Estimated averages add uncertainty
   :class: warning

   ``emfx`` uses the regression covariance matrix and treats covariate means, empirical
   covariate distributions, and aggregation weights as fixed. Its standard errors leave out
   their additional sampling variation. Population-ATT inference that includes this variation
   requires an adjustment or a bootstrap that repeats fitting, centering, and aggregation.

The pointwise normal intervals from ``emfx`` do not give simultaneous coverage for an entire
event path, as you would need to assess several periods together rather than one reported
effect at a time.

When the balanced-panel argument changes
----------------------------------------

The earlier equalities describe the estimator's behavior in a balanced panel with time-constant
covariates. Missing observations, time-varying controls, or treatment exit
require additional assumptions and sometimes a different regression design.

Time-varying covariates
~~~~~~~~~~~~~~~~~~~~~~~

A time-varying covariate must remain unaffected by treatment at every date,

.. math::

   \mathbf{x}_t(g)=\mathbf{x}_t(\infty),
   \qquad g\in\mathcal{G},\quad t=1,\ldots,T.

That condition alone does not justify replacing every :math:`\mathbf{x}_i` in the derivation
by :math:`\mathbf{x}_{it}`. A panel regression also needs appropriate restrictions on its
relationship to outcome shocks across the full observed path. Wooldridge's Section 10.1
discusses strict exogeneity and expanded controls, including time averages
:math:`\bar{\mathbf{x}}_i=T^{-1}\sum_t\mathbf{x}_{it}`.

The imputation and pooled regression can still be matched under an expanded model. Their
automatic equality with unit fixed effects no longer follows from the time-constant design.
The package accepts varying control columns; their inclusion does not establish the
counterfactual restrictions required for a causal interpretation.

Unbalanced panels
~~~~~~~~~~~~~~~~~

Let :math:`s_{it}` indicate a complete observation and
:math:`T_i=\sum_t s_{it}` count the periods observed for unit :math:`i`. In an unbalanced
panel, a period dummy's unit average is

.. math::

   \bar f_{r,i}=T_i^{-1}\sum_t s_{it}f_{r,t}.

Those averages depend on each unit's observation pattern in the sample. The original cohort and
period controls therefore need not span the required Mundlak terms. In Wooldridge's common-timing example, the matching
pooled regression adds both :math:`\bar f_{r,i}` and the relevant treatment-group interactions
:math:`D_i\bar f_{r,i}`. Adding period averages alone is not a general correction for
staggered adoption with covariates.

Unit fixed effects can absorb selection related to additive unit heterogeneity under suitable
conditional mean assumptions. They do not remove selection related to unobserved time-varying
outcome shocks. The package fits available complete observations in an unbalanced sample.
A successful call establishes neither a missing-data adjustment nor the balanced-panel equivalence.

Treatment exit
~~~~~~~~~~~~~~

The adoption-date model above represents absorbing treatment. A theoretical extension can
index a treatment path by entry :math:`g` and last active period :math:`h`. Let
:math:`d_{g,h,i}` identify that path. The case :math:`h=\infty` denotes treatment through
the end of the observed sample. Its effect is

.. math::

   \tau_{g,h,r}
   =\mathbb{E}[y_r(g,h)-y_r(\infty)\mid d_{g,h}=1],
   \qquad r=g,\ldots,T.

Effects remain meaningful for :math:`r>h`, because an intervention may have lasting
consequences after it ends. A fully interacted regression would replace
:math:`d_{g,i}f_{s,t}` with :math:`d_{g,h,i}f_{s,t}` and require suitable no-anticipation
and untreated-trend assumptions for those paths. Exit triggered by time-varying untreated
outcome shocks would violate the corresponding strict exogeneity restriction.

The current ``etwfe`` function takes first adoption dates and constructs absorbing treatment
without implementing this entry-exit path extension. For estimators designed around treatment
changes and their subsequent effects, see the :ref:`intertemporal DiD background <background-didinter>`.

The :ref:`extended TWFE example <example_etwfe>` compares control groups and nonlinear
specifications so you can see how those choices change both the estimand and the evidence
you can report.
