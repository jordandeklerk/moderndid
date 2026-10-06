.. _background-did:

Staggered difference-in-differences
===================================

In the :ref:`minimum wage example <example_staggered_did>`, counties whose
states raised the minimum wage in 2004 have already had two years of
exposure by the time the 2006 cohort adopts. If employment
responds differently across cohorts or changes with exposure, a single
regression coefficient can hide the effects you want to measure.

We'll build the analysis around the effect for one adoption cohort in one
period. The approach of `Callaway and Sant'Anna (2021)
<https://doi.org/10.1016/j.jeconom.2020.12.001>`_ first identifies those effects
through comparisons with untreated units so you can then average them to answer
your question. This page works through the assumptions and
identification formulas behind :func:`~moderndid.att_gt`, the targets behind
:func:`~moderndid.aggte`, and the inference you need to interpret their results.
Sections 2 through 4 and Appendix B of the
`author manuscript <https://psantanna.com/files/Callaway_SantAnna_2020.pdf>`_
give the corresponding theoretical results.

The formal statements below keep the paper's numbering and use the notation
we develop here. You can read each condition alongside the explanation of
what it asks of your data and how the package uses it.

The effect for one cohort in one period
---------------------------------------

We start with a panel of :math:`n` units observed in periods
:math:`1,\ldots,T`. Let :math:`D_{it}` indicate whether unit :math:`i`
is treated in period :math:`t`, taking the value one for treated units and
zero otherwise. We omit the unit index :math:`i` when writing population
conditions, as in the treatment restriction below.

.. admonition:: Assumption 1 Irreversible treatment
   :class: assumption

   Every unit is untreated in the first period. Once treatment begins,
   it remains on in every subsequent observed period,

   .. math::

      \begin{aligned}
      D_1&=0\quad\text{almost surely},\\
      D_{r-1}=1&\Longrightarrow D_r=1\quad\text{almost surely},
      \quad r=2,\ldots,T.
      \end{aligned}

Write
:math:`G_i` for unit :math:`i`'s first treatment period and use
:math:`G_i=\infty` for a unit that remains untreated. The treatment indicator
and the indicator for membership in cohort :math:`g` are

.. math::

   D_{it}=\mathbf{1}\{G_i\leq t\},
   \qquad A_{ig}=\mathbf{1}\{G_i=g\}.

Because the data column passed as ``gname`` records :math:`G_i` rather than
:math:`D_{it}`, its value stays the same across a unit's rows. ModernDiD uses
``0`` to record never-treated units, even though the theory writes their
adoption time as infinity.

The derivation uses consecutive indices so horizons count observation
periods. If your numeric time labels have gaps, recode them before interpreting
``anticipation`` or event time as a number of observation periods.

Let :math:`Y_{it}(g)` be the outcome unit :math:`i` would have in period
:math:`t` if it first adopted in period :math:`g` and let :math:`Y_{it}(0)`
denote its outcome without treatment. Each unit follows only one observed path,

.. math::

   Y_{it}=Y_{it}(0)+\sum_{g=2}^{T}A_{ig}
   \bigl[Y_{it}(g)-Y_{it}(0)\bigr].

This setup treats adoption time as a description of the treatment path.
It also assumes that another unit's adoption does not change this unit's
potential outcomes. If the policy spills across county borders, that
assumption needs attention before a comparison county can stand in for an
untreated county.

For a cohort first treated in :math:`g`, the group-time average treatment
effect is

.. math::
   :label: did-group-time-att

   ATT(g,t)=\mathbb{E}\bigl[Y_t(g)-Y_t(0)\mid G=g\bigr].

For example, :math:`ATT(2004,2006)` compares employment in the 2004 cohort
in 2006 with its own employment in 2006 under no minimum wage increase.
Counties in another cohort supply information about that missing outcome
without changing whose treatment effect the parameter measures. Effects
can differ across cohorts, calendar periods, exposure lengths, and
pre-treatment characteristics without changing this definition.

The effect's definition does not by itself tell us which cohorts have
an untreated comparison. We will use the paper's cohort sets to state
where its identification results apply. Let :math:`\bar g` be the
largest supported adoption time, including infinity if never-treated
units are present. Write :math:`\delta` for a known nonnegative integer
that bounds how far before adoption conditional mean responses may begin.
Then set

.. math::

   \mathcal{G}=\operatorname{supp}(G)\setminus\{\bar g\},
   \qquad
   \mathcal{G}_\delta=\mathcal{G}\cap\{2+\delta,\ldots,T\}.

With never-treated units, :math:`\mathcal{G}` contains the finite adoption
cohorts. If everyone adopts, it excludes the last cohort because that
cohort has no untreated comparison after adoption. Membership in
:math:`\mathcal{G}_\delta` also leaves an observed period before the
anticipation window. We'll derive that base period when we state the
limited-anticipation assumption.

If treatment reverses or its dose changes over time, use a framework that represents
those paths, such as :ref:`intertemporal DiD <background-didinter>`.

Write :math:`X_i` for unit :math:`i`'s pre-treatment covariates, such as
county population measured before a minimum wage increase. Since the panel
sampling condition concerns whole units rather than individual rows,
repeated observations of one county can depend on each other.

.. admonition:: Assumption 2 Random panel sampling
   :class: assumption

   The observed unit-level vectors

   .. math::

      W_i=(Y_{i1},\ldots,Y_{iT},X_i,D_{i1},\ldots,D_{iT}),
      \qquad i=1,\ldots,n,

   are independent draws from a common population distribution. This
   permits arbitrary dependence within a unit over time and does not
   make treatment adoption independent of potential outcomes.

.. _background-did-twfe:

Why a pooled regression can mix the comparisons
-----------------------------------------------

A conventional two-way fixed effects regression summarizes treatment
across the panel with one coefficient. To see how that coefficient
relates to the cohort-time effects, write a unit effect and a
calendar-period effect alongside the treatment indicator,

.. math::

   Y_{it}=\alpha_i+\lambda_t+\beta D_{it}+\varepsilon_{it}.

A well-established result in the staggered adoption literature is that
this coefficient can be misleading when treatment effects vary across
cohorts or over time. To see why, we need to look at the comparisons used
to estimate it. These include earlier adopters measured against later
adopters while the later adopters are untreated. They also include later
adopters measured against earlier adopters after the earlier adopters are
treated. If the earlier cohort's effect changes during that second
comparison, its outcome change includes a treatment response. Subtracting
that change can distort the effect attributed to the later cohort.

`Goodman-Bacon (2021) <https://doi.org/10.1016/j.jeconom.2021.03.014>`_
decomposes the pooled coefficient into two-group, two-period DiD
comparisons whose weights are nonnegative. A different
decomposition into underlying treatment effects can have negative weights,
as `de Chaisemartin and D'Haultfoeuille (2020)
<https://doi.org/10.1257/aer.20181169>`_ show. Because the two decompositions weight
different objects, a negative ATT weight does not contradict the first
decomposition. Parallel trends and no anticipation are needed for a causal
interpretation of these comparisons. Even when cohort heterogeneity permits
an interpretable pooled estimate, the regression's weights need not answer
your question.

To follow the response over time, you might replace the single treatment
indicator with event-time leads and lags. For a treated unit, event time
:math:`e=t-G_i` indexes periods relative to adoption. Negative values
label periods before adoption and positive values label periods after it.
`Sun and Abraham (2021) <https://doi.org/10.1016/j.jeconom.2020.09.006>`_
show that a coefficient at one event time can contain effects from other
event times when cohort effects differ. Even a lead coefficient can reflect
post-treatment effects under no anticipation.

For a finite set of event times :math:`\mathcal{E}` and an omitted
reference period, the regression is

.. math::

   Y_{it}=\alpha_i+\lambda_t+
   \sum_{e\in\mathcal{E}}\beta_e\mathbf{1}\{G_i<\infty,\ t-G_i=e\}
   +v_{it}.

We'll instead estimate each
:math:`ATT(g,t)` using an explicitly chosen untreated comparison group.
That choice fixes the comparison before any effects are averaged.

.. _background-did-assumptions:

Recovering the untreated outcome path
-------------------------------------

The identification argument needs a period before treatment can affect the
cohort and a comparison group whose outcome change represents the cohort's
untreated change. We'll state those requirements conditional on
pre-treatment covariates :math:`X`. Conditional parallel trends permits
counties of different sizes to have different untreated trends. It asks
that treated and comparison counties with the same :math:`X` share those
trends.

Anticipation determines the base period
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose responses may begin up to a known nonnegative integer
:math:`\delta` periods before adoption. We allow responses inside that
window and require their conditional mean to be zero earlier.

.. admonition:: Assumption 3 Limited treatment anticipation
   :class: assumption

   There is a known nonnegative integer :math:`\delta` such that, for
   every :math:`g\in\mathcal{G}` and :math:`r\in\{1,\ldots,T\}`
   satisfying :math:`r<g-\delta`, the following equality holds
   almost surely,

   .. math::

      \mathbb{E}[Y_r(g)\mid X,G=g]=\mathbb{E}[Y_r(0)\mid X,G=g].

Setting ``anticipation=0`` rules out an average response before adoption,
whereas ``anticipation=1`` allows the period immediately before adoption
to contain a response. The last unaffected base period is therefore

.. math::

   b_g=g-\delta-1,
   \qquad \Delta Y_{g,t}=Y_t-Y_{b_g}.

An observed base period requires :math:`b_g\geq1` for every cohort
in the target. An early cohort can lose its usable base when you increase
``anticipation``. Moving the base backward also lengthens the span over
which parallel trends must hold. The theory can identify responses during
:math:`g-\delta\leq t<g` if you measure them relative to that clean base.
These are anticipation effects before adoption rather than effects after it.

This restriction must also hold for later cohorts supplying controls.
If everyone adopts, that includes the last cohort even though it is
excluded from the target set :math:`\mathcal{G}`.

.. _background-did-comparison-groups:

The comparison group determines parallel trends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``control_group="nevertreated"``, the comparison units remain
untreated. Their outcome changes supply the missing untreated changes
under the following conditional parallel trends assumption.

.. admonition:: Assumption 4 Never-treated parallel trends
   :class: assumption

   For the anticipation horizon in Assumption 3 and every
   :math:`g\in\mathcal{G}` and :math:`r\in\{2,\ldots,T\}`
   satisfying :math:`r\geq g-\delta`, the following equality holds
   almost surely,

   .. math::

      \begin{aligned}
      &\mathbb{E}[Y_r(0)-Y_{r-1}(0)\mid X,G=g]\\
      &\quad=\mathbb{E}[Y_r(0)-Y_{r-1}(0)\mid X,G=\infty].
      \end{aligned}

Because this assumption restricts outcome changes, cohorts may have
different untreated outcome levels. When :math:`\delta=0`, it restricts trends
beginning at adoption. It imposes no restriction on that cohort's earlier
observed trends. A pre-treatment diagnostic therefore checks an extension
of this assumption into earlier periods.

With ``control_group="notyettreated"``, later adopters can also supply
the comparison. The paper's alternative assumption matches the cohort's
untreated changes to those of units adopting beyond an eligible cutoff.

.. admonition:: Assumption 5 Not-yet-treated parallel trends
   :class: assumption

   For the anticipation horizon in Assumption 3 and every
   :math:`g\in\mathcal{G}` and
   :math:`(s,r)\in\{2,\ldots,T\}^2` satisfying
   :math:`r\geq g-\delta` and :math:`r+\delta\leq s<\bar g`,
   the following equality holds almost surely,

   .. math::

      \begin{aligned}
      &\mathbb{E}[Y_r(0)-Y_{r-1}(0)\mid X,G=g]\\
      &\quad=\mathbb{E}[Y_r(0)-Y_{r-1}(0)\mid X,G>s,G\ne g].
      \end{aligned}

To identify the effect at :math:`t`, take :math:`s=t+\delta`. A later adopter qualifies
only if it has not entered its anticipation window by :math:`t`.
Never-treated units also belong to this comparison group at every cutoff.
Because the assumption applies across the eligible cutoffs and cohorts,
using later adopters can restrict their pre-treatment trends as well.

To use one set of formulas for both choices, define the eligible-control
indicator

.. math::

   B_{g,t}=\begin{cases}
      \mathbf{1}\{G=\infty\}, & \text{never-treated comparison},\\
      \mathbf{1}\{G>t+\delta\}\mathbf{1}\{G\ne g\},
         & \text{not-yet-treated comparison}.
   \end{cases}

Although the second indicator in the not-yet-treated definition is redundant
for :math:`t\geq g-\delta`, retaining it makes clear that the target cohort
cannot also be a control. Choosing :math:`B_{g,t}` determines both the
counterfactual assumption and the available observations. Having more
controls can help precision without establishing that their untreated trends
match those of the cohort.

If every unit adopts by :math:`g_{\max}`, not-yet-treated comparisons
identify effects only while :math:`t<g_{\max}-\delta`. The last cohort
has no untreated comparison for its post-treatment effects, even if its
outcomes remain observed afterward.

When no never-treated cohort is available, the default control choice
warns and uses the latest cohort as the comparison. The package discards
periods at or after that cohort enters its anticipation window. Choosing
``control_group="notyettreated"`` also excludes those unsupported final
periods. Neither choice estimates the last cohort's post-treatment effects.

Nonadoption through :math:`T` also needs interpretation when
:math:`\delta>0`. A unit adopting just after the sample ends may already
anticipate treatment inside the sample. Theorem 1 restricts identification
to :math:`t\leq T-\delta` under the paper's finite-horizon setup. Remark 6
extends it through :math:`T` if the comparison units are known to remain
untreated afterward. ModernDiD treats a ``0`` in ``gname`` as never
anticipating treatment. Whether that description fits your controls is a
substantive assumption about the application.

Overlap makes the conditional comparison possible
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Matching untreated trends at the same :math:`X` also requires controls
at the covariate values observed in the treated cohort. The generalized
propensity score describes the probability of cohort membership within
the selected comparison. Since the paper indexes that comparison by a cutoff
:math:`s`, we distinguish the cutoff from the outcome period by writing

.. math::

   p^{\mathrm{cut}}_{g,s}(X)
   =P\bigl(A_g=1\mid X,A_g+\mathbf{1}\{G>s,G\ne g\}=1\bigr).

.. admonition:: Assumption 6 Overlap
   :class: assumption

   For each :math:`g\in\mathcal{G}` and
   :math:`s\in\{2,\ldots,T\}`, there is an
   :math:`\varepsilon>0` such that

   .. math::

      P(A_g=1)>\varepsilon,\qquad
      p^{\mathrm{cut}}_{g,s}(X)<1-\varepsilon
      \quad\text{almost surely}.

   The condition requires both positive cohort mass and control support
   at the cohort's covariate values.

The manuscript states this restriction across the observed cutoffs. A
cutoff after every unit adopts has no eligible controls and cannot satisfy
it. When no never-treated cohort exists, we impose Assumption 6 on the
eligible comparison cutoffs :math:`s<\bar g`. The not-yet-treated results
below use that scope, since their comparisons require an untreated group.

In the formulas below, :math:`p_{g,t}` instead indexes the outcome
comparison and its selected controls,

.. math::

   p_{g,t}(X)=P(A_g=1\mid X,A_g+B_{g,t}=1),
   \qquad A_g=\mathbf{1}\{G=g\}.

This probability is :math:`p^{\mathrm{cut}}_{g,T}` for never-treated controls
and :math:`p^{\mathrm{cut}}_{g,t+\delta}` for not-yet-treated controls.

Since the ATT targets treated units, it does not require treated
counterparts for every control unit.

The large control weights produced by propensity scores close to one are
evidence that a comparison rests on little support even if estimation returns
a number. Choosing ``est_method="reg"`` instead of weighting still requires
overlap because the regression would otherwise have to extrapolate the
missing untreated change into those regions.

Identification and estimation of a group-time effect
----------------------------------------------------

With the base and controls fixed, we can recover the missing untreated
change. We derive the post-treatment comparison for :math:`t\geq g`
using the clean base :math:`b_g`. The same theoretical
identities extend to :math:`g-\delta\leq t<g`. Use
``base_period="universal"`` to estimate those anticipation effects
relative to :math:`b_g`. Define the control outcome regression

.. math::

   m_{g,t}(X)=\mathbb{E}[\Delta Y_{g,t}\mid X,B_{g,t}=1].

The propensity score :math:`p_{g,t}` and control outcome regression
:math:`m_{g,t}` are nuisance functions, the auxiliary quantities needed
to recover :math:`ATT(g,t)`. They describe cohort membership and the
control outcome change, respectively.

Adding the one-period parallel trends equalities from :math:`g-\delta`
through :math:`t` gives equality of the untreated changes over the whole
comparison. Limited anticipation makes the base outcome untreated in conditional
mean. The controls remain unaffected in conditional mean at both
endpoints. Consequently,

.. math::

   \mathbb{E}[Y_t(0)-Y_{b_g}(0)\mid X,G=g]=m_{g,t}(X),

and the conditional treatment effect becomes

.. math::

   \mathbb{E}[Y_t(g)-Y_t(0)\mid X,G=g]
   =\mathbb{E}[\Delta Y_{g,t}\mid X,G=g]-m_{g,t}(X).

Averaging over the target cohort's covariates gives the outcome regression
identity,

.. math::
   :label: did-or-identity

   ATT(g,t)=\mathbb{E}\left[
      \frac{A_g}{\mathbb{E}[A_g]}
      \bigl(\Delta Y_{g,t}-m_{g,t}(X)\bigr)\right].

If treated counties are larger than controls, the averaging distribution
determines whose untreated change you recover. You predict changes at the treated counties'
sizes and average those predictions across the treated counties. Averaging
the fitted regression over the controls would answer a different question.

Reweighting the controls toward the cohort
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same identification argument can recover the untreated change by
reweighting controls. Write the propensity odds and the two normalized
weights as

.. math::

   r_{g,t}(X)=\frac{p_{g,t}(X)}{1-p_{g,t}(X)},
   \qquad w_g^1=\frac{A_g}{\mathbb{E}[A_g]},
   \qquad w_{g,t}^0=
   \frac{r_{g,t}(X)B_{g,t}}
        {\mathbb{E}[r_{g,t}(X)B_{g,t}]}.

With the true propensity score,
:math:`\mathbb{E}[r_{g,t}(X)B_{g,t}]=\mathbb{E}[A_g]`. For any integrable
function :math:`h` of the covariates, the weighted control and cohort
distributions satisfy

.. math::

   \mathbb{E}[w_{g,t}^0h(X)]
   =\mathbb{E}[w_g^1h(X)].

Taking :math:`h=m_{g,t}` replaces the cohort's average predicted untreated
change with a weighted average of observed control changes. This gives the
inverse probability weighting identity,

.. math::
   :label: did-ipw-identity

   ATT(g,t)=\mathbb{E}\bigl[
      (w_g^1-w_{g,t}^0)\Delta Y_{g,t}\bigr].

Because the cohort and control weights each sum to one in expectation,
their difference subtracts a control change from a treated change. The
negative signs implement this DiD comparison without assigning negative
weights to the treatment effects averaged later by :func:`~moderndid.aggte`.

Combining the regression and the weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An outcome regression estimate requires a correct model of the control
change, whereas an IPW estimate requires a correct model of cohort
membership. Under the identifying assumptions and suitable regularity
conditions, the untrimmed doubly robust estimator remains consistent
if either model is correct. We combine the two routes by applying
the same weights after subtracting the control regression from the
observed outcome change.

.. admonition:: Theorem 1 Group-time identification
   :class: theorem

   Under Assumptions 1, 2, 3, and 6, the outcome regression, IPW, and DR
   expressions identify the same effect using their population nuisance
   functions,

   .. math::
      :label: did-dr-identity

      \begin{aligned}
      ATT(g,t)
      &=\mathbb{E}\bigl[w_g^1(\Delta Y_{g,t}-m_{g,t}(X))\bigr]\\
      &=\mathbb{E}\bigl[(w_g^1-w_{g,t}^0)\Delta Y_{g,t}\bigr]\\
      &=\mathbb{E}\bigl[(w_g^1-w_{g,t}^0)
                       (\Delta Y_{g,t}-m_{g,t}(X))\bigr].
      \end{aligned}

   With Assumption 4, these equalities hold for every
   :math:`g\in\mathcal{G}_\delta` and
   :math:`t\in\{2,\ldots,T-\delta\}` satisfying
   :math:`t\geq g-\delta`, using :math:`B_{g,t}=\mathbf{1}\{G=\infty\}`.

   With Assumption 5, they hold for every
   :math:`g\in\mathcal{G}_\delta` and
   :math:`t\in\{2,\ldots,T-\delta\}` satisfying
   :math:`g-\delta\leq t<\bar g-\delta`, using
   :math:`B_{g,t}=\mathbf{1}\{G>t+\delta,G\ne g\}`.

   Each comparison uses its own :math:`p_{g,t}`, :math:`m_{g,t}`, and
   normalized control weights. The clean base is
   :math:`b_g=g-\delta-1` in both cases.

We call the weighted residual inside Theorem 1's last expectation the
DR score. Estimation replaces its population expectations, propensity
score, and outcome regression with sample counterparts. For an unweighted
panel of :math:`n` units, the
doubly robust estimate before numerical trimming has the form

.. math::

   \widehat{ATT}_{dr}(g,t)=\frac{1}{n}\sum_{i=1}^{n}
   \left[
      \frac{A_{ig}}{\overline{A_g}}-
      \frac{\widehat r_{g,t}(X_i)B_{i,g,t}}
           {\overline{\widehat r_{g,t}B_{g,t}}}
   \right]
   \bigl[\Delta Y_{i,g,t}-\widehat m_{g,t}(X_i)\bigr].

The bars denote sample means over the units in the estimation sample.
To see the double robustness, let :math:`\widetilde p` and
:math:`\widetilde m` be candidate nuisance functions and let
:math:`\widetilde w^0` use the candidate propensity odds. Write
:math:`S(\widetilde p,\widetilde m)` for the resulting population score
mean. Its bias satisfies

.. math::

   S(\widetilde p,\widetilde m)-ATT(g,t)
   =\mathbb{E}\bigl[(w_g^1-\widetilde w^0)
      (m_{g,t}(X)-\widetilde m(X))\bigr].

The bias vanishes when the outcome regression is correct because the
regression error on the right is zero. A correct propensity model also
removes the bias by balancing that error across treated units and controls.

.. admonition:: Keep the identifying assumptions
   :class: important

   Double robustness protects against misspecifying one nuisance model.
   It still requires the chosen parallel trends assumption, limited
   anticipation, and overlap. Misspecifying both nuisance models has no
   such consistency guarantee.

The built-in IPW and DR estimators omit control weighting contributions
when fitted propensity scores reach 0.995. The outer estimator can also
skip comparisons that fail its overlap or regression checks.

Those numerical safeguards do not establish population overlap or
restore unsupported effects. When trimming removes eligible controls,
correct propensities alone need not preserve the balancing identity
above.

With no covariates, the identities reduce to the familiar difference in
average changes,

.. math::

   ATT(g,t)=\mathbb{E}[\Delta Y_{g,t}\mid G=g]
            -\mathbb{E}[\Delta Y_{g,t}\mid B_{g,t}=1].

Setting ``xformla=None`` or ``xformla="~1"`` makes this unconditional
comparison under a parallel trends assumption without covariate adjustment.
Adding covariates to a pooled DiD regression generally gives a different
coefficient from the conditional identification formulas above. Remark 4
of Callaway and Sant'Anna discusses restrictions that make that regression
coefficient equal the ATT. Conditional identification here allows
covariate-specific untreated trends and effects that vary with :math:`X`.

What the package estimates
~~~~~~~~~~~~~~~~~~~~~~~~~~

The sample formula still leaves us to choose how to fit the nuisance
functions. The default ``est_method="dr"`` uses a logistic propensity
model and linear outcome regressions for each cohort-time comparison.
You can choose
``"ipw"`` for propensity weighting or ``"reg"`` for outcome regression.
For a balanced panel, the control regression models the change between the
base and current period. Covariates come from the earlier endpoint of
that comparison. Since ``xformla`` selects existing numeric columns,
create transformations or interaction columns before passing them.

Use covariates that can account for differences in untreated trends and
that treatment has not affected. A covariate measured after adoption can
contain part of the policy response you are trying to estimate. A correctly
fitted model for that variable does not repair the causal interpretation.
If you supply ``weightsname``, estimation and cohort shares use the
sampling weights rather than giving every observed unit equal weight.

The two-period estimators follow `Sant'Anna and Zhao (2020)
<https://doi.org/10.1016/j.jeconom.2020.06.003>`_. Their efficiency and
inference results depend on the estimator and nuisance-fitting procedure.
When both panel nuisance models are correct, the untrimmed DR estimator
attains the semiparametric efficiency bound under their regularity
conditions. This bound describes the lowest asymptotic variance available
to regular estimators under the two-period sampling design and identifying
assumptions. The default here uses
traditional propensity maximum likelihood and
least-squares regression. It does not use the improved inverse probability
tilting and weighted regression procedure described in the
:ref:`doubly robust DiD background <background-drdid>`. Supplying a custom
estimator also requires a valid influence function to describe its leading
sampling error. We work through that contribution in the inference section.
Double robustness alone does not justify arbitrary machine learning fits
or their standard errors.

When different units are observed in each period
------------------------------------------------

The panel derivation uses each unit's observed change between two periods.
When different units are sampled in each period, we cannot observe that
change and instead reconstruct each group's mean change from separate
outcome means at the two endpoints. Comparing those repeated-cross-section
means requires stable cohort composition.

Let :math:`S\in\{1,\ldots,T\}` denote the observation period and
:math:`\lambda_r=P(S=r)` its sampling probability. The repeated-cross-section
sampling condition replaces Assumption 2 rather than requiring repeated
measurements of each unit.

.. admonition:: Assumption B.1 Repeated-cross-section sampling
   :class: assumption

   For every :math:`r\in\{1,\ldots,T\}`, conditional on :math:`S=r`,
   observations are independent draws from the population law of
   :math:`(Y_r,A_2,\ldots,A_T,C,X)`, where
   :math:`C=\mathbf{1}\{G=\infty\}`. Cohort membership and covariates
   have the same joint distribution in every observation period,

   .. math::

      (A_2,\ldots,A_T,C,X)\perp S.

   The sample consists of independent draws from the corresponding
   mixture of period-specific distributions, with mixture probabilities
   :math:`\lambda_r`.

A change in who belongs to a cohort or in its covariate distribution needs
a different justification. Observing both conditional outcome distributions
requires :math:`\lambda_t>0` and :math:`\lambda_{b_g}>0`.

Write :math:`\mathbb{E}_M` for expectation under this mixture distribution.
For each endpoint :math:`u\in\{b_g,t\}`, define the treated and control
level regressions,

.. math::

   \begin{aligned}
   \mu_{1u}(X)&=\mathbb{E}_M[Y\mid X,A_g=1,S=u],\\
   \mu_{0u}(X)&=\mathbb{E}_M[Y\mid X,B_{g,t}=1,S=u],\\
   \Delta\mu_j(X)&=\mu_{jt}(X)-\mu_{jb_g}(X),\quad j\in\{0,1\}.
   \end{aligned}

The same :math:`B_{g,t}` selects controls at both endpoints. For
not-yet-treated comparisons, these are units still unaffected at
:math:`t`, rather than everyone untreated at the earlier endpoint.
Let :math:`J_u=\mathbf{1}\{S=u\}`. The period-specific weights are

.. math::

   \begin{aligned}
   \omega_u^1
      &=\frac{J_uA_g}{\mathbb{E}_M[J_uA_g]}
        =\frac{J_u}{\lambda_u}w_g^1,\\
   \omega_u^0
      &=\frac{J_ur_{g,t}(X)B_{g,t}}
              {\mathbb{E}_M[J_ur_{g,t}(X)B_{g,t}]}
        =\frac{J_u}{\lambda_u}w_{g,t}^0.
   \end{aligned}

Here the population weights :math:`w_g^1` and :math:`w_{g,t}^0` use
:math:`\mathbb{E}_M` in their normalizing denominators. Assumption B.1
justifies the factorization by :math:`\lambda_u` so these weights
reconstruct each group's outcome means at each endpoint. The outcome
regression expression averages the difference between the four
conditional means over the treated cohort's covariates, whereas the IPW
expression reconstructs that comparison using the period-specific weights.
The DR expression uses the regression expression as its starting point and
adds weighted residual adjustments. The three repeated-cross-section
estimands are

.. math::

   \begin{aligned}
   ATT_{\mathrm{or,rc}}(g,t)
      &=\mathbb{E}_M[w_g^1(\Delta\mu_1-\Delta\mu_0)],\\
   ATT_{\mathrm{ipw,rc}}(g,t)
      &=\mathbb{E}_M[(\omega_t^1-\omega_{b_g}^1
                     -\omega_t^0+\omega_{b_g}^0)Y],\\
   ATT_{\mathrm{dr,rc}}(g,t)
      &=ATT_{\mathrm{or,rc}}(g,t)\\
      &\quad+\mathbb{E}_M[\omega_t^1(Y-\mu_{1t})
                         -\omega_{b_g}^1(Y-\mu_{1b_g})]\\
      &\quad-\mathbb{E}_M[\omega_t^0(Y-\mu_{0t})
                         -\omega_{b_g}^0(Y-\mu_{0b_g})].
   \end{aligned}

.. admonition:: Theorem B.1 Repeated-cross-section identification
   :class: theorem

   Under Assumptions 1, 3, 4, 6, and B.1, use
   :math:`B_{g,t}=C`. For every :math:`g\in\mathcal{G}_\delta` and
   :math:`t\in\{2,\ldots,T-\delta\}` satisfying
   :math:`t\geq g-\delta`,

   .. math::

      ATT(g,t)=ATT_{\mathrm{or,rc}}(g,t)
              =ATT_{\mathrm{ipw,rc}}(g,t)
              =ATT_{\mathrm{dr,rc}}(g,t).

   With Assumption 5 in place of Assumption 4, use
   :math:`B_{g,t}=\mathbf{1}\{G>t+\delta,G\ne g\}`. The same three
   equalities hold on that cohort and time domain while
   :math:`t<\bar g-\delta`.

Correct treated-group regressions can improve the DR estimator's efficiency.
The untrimmed estimator attains the repeated-cross-section efficiency bound when
the propensity model and all four outcome regressions are correct. For
double robustness, the outcome-model requirement concerns the difference
between the two control regressions. Correctness of both control level
regressions is sufficient but stronger than correctness of their
difference, as Sant'Anna and Zhao's Theorem 1 shows.

Set ``panel=False`` for this sampling design. The defaults ``panel=True``
and ``allow_unbalanced_panel=False`` retain units observed in every period,
whereas ``allow_unbalanced_panel=True`` allows missing periods and uses the
repeated-cross-section estimation path. You still need a reason why the
observed samples represent the population in your target, because this setting
does not establish that attrition leaves the identifying comparison valid.

Choosing what the effects should average
----------------------------------------

Under these conditions, either sampling design gives us a collection
of cohort-time effects. We now choose how to average them to answer
your question. The weights of an aggregation determine whose effects
and which periods the new estimand describes. Write
:math:`\mathcal{I}` for the identified post-treatment
cohort-time pairs included in an average. Then

.. math::

   \theta=\sum_{(g,t)\in\mathcal{I}}w(g,t)ATT(g,t),
   \qquad w(g,t)\geq0,\quad
   \sum_{(g,t)\in\mathcal{I}}w(g,t)=1.

For the formulas below, take :math:`\delta=0`, a never-treated comparison
group, and all post-treatment periods through :math:`T` available for the
cohorts in :math:`\mathcal{G}`. Define
:math:`q_g=P(G=g\mid G\in\mathcal{G})`. These are shares among the treated
cohorts in the target population. Each average requires a nonempty set
of contributing cohorts so its denominator is positive. If you restrict
the sample, event window,
or identified cells, the weights and the population being averaged must
reflect that restriction. Removing missing effects with ``na_rm=True``
cannot recover their contribution to the original target.

Cohort averages and the overall effect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To summarize the effect for a typical treated unit, we first average
each cohort's post-treatment effects over the periods observed for that
cohort. Passing ``type="group"`` to :func:`~moderndid.aggte` then
weights those cohort averages by their shares,

.. math::

   \theta_{\mathrm{group}}(g)
   =\frac{1}{T-g+1}\sum_{t=g}^{T}ATT(g,t),
   \qquad
   \theta_{\mathrm{group}}^{O}
   =\sum_{g\in\mathcal{G}}q_g\theta_{\mathrm{group}}(g).

When sampling weights are equal, every treated unit receives the same total
weight in this overall target. Because early adopters are averaged over more
exposure periods than late adopters, a difference between cohort averages
combines cohort heterogeneity and different exposure windows. You therefore
cannot interpret that difference as what would happen if the same units
adopted earlier.

If you instead want an average over treated unit-periods in the
observation window, ``type="simple"`` weights each post-treatment cell
in proportion to its cohort's share,

.. math::

   \theta_{\mathrm{simple}}^{O}
   =\frac{\displaystyle\sum_{g\in\mathcal{G}}q_g
                   \sum_{t=g}^{T}ATT(g,t)}
          {\displaystyle\sum_{g\in\mathcal{G}}q_g(T-g+1)}.

A cohort's total weight is proportional to :math:`q_g(T-g+1)`.
Because early adopters contribute more treated periods, the simple
average gives them more total weight than the group average. Both targets
average causal effects over different distributions of cohorts and
exposure. The default ``type="group"`` uses the overall effect formed
from the cohort averages.

Event time and changing cohort composition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An overall average can hide how the effect changes with exposure.
We can follow that change by lining cohorts up at the same event time
:math:`e=t-g`.
For :math:`e\geq0`, define the cohorts observed at that exposure and their
normalized shares,

.. math::

   \mathcal{G}_e=\{g\in\mathcal{G}:g+e\leq T\},
   \qquad q_{g,e}=\frac{q_g}{\sum_{h\in\mathcal{G}_e}q_h}.

Passing ``type="dynamic"`` averages the effects within each event time,

.. math::

   \theta_{\mathrm{dynamic}}(e)
   =\sum_{g\in\mathcal{G}_e}q_{g,e}ATT(g,g+e).

Event time zero measures the adoption-period effect, whereas later event
times describe progressively longer exposure among the cohorts observed
that long. In the minimum wage example, all three cohorts contribute at
event time zero, whereas only the 2004 cohort contributes two and three
years after adoption. Growth between those points can therefore reflect a
change in the cohorts being averaged.

For :math:`0\leq e_1<e_2`, let :math:`a_g(e)=ATT(g,g+e)`.
Since :math:`\mathcal{G}_{e_2}\subseteq\mathcal{G}_{e_1}`, the change in
the event study decomposes into

.. math::

   \begin{aligned}
   \theta_{\mathrm{dynamic}}(e_2)-\theta_{\mathrm{dynamic}}(e_1)
   ={}&\sum_{g\in\mathcal{G}_{e_2}}q_{g,e_2}
            \bigl[a_g(e_2)-a_g(e_1)\bigr]\\
     &+\sum_{g\in\mathcal{G}_{e_2}}
            (q_{g,e_2}-q_{g,e_1})a_g(e_1)\\
     &-\sum_{g\in\mathcal{G}_{e_1}\setminus\mathcal{G}_{e_2}}
            q_{g,e_1}a_g(e_1).
   \end{aligned}

The first term measures change within cohorts observed at both event
times. The second and third terms describe composition changes as
surviving cohorts gain weight and other cohorts leave the average. You can interpret the difference as an average change
in effects within cohorts only when the composition terms cancel or
when you hold the cohorts and weights fixed.

.. _background-did-balanced:

.. admonition:: Hold cohort composition fixed
   :class: tip

   Set ``balance_e=E`` to keep cohorts observed through event time
   :math:`E` and cap the reported event times at :math:`E`. When all
   requested cells are estimable, cohort composition stays fixed over
   :math:`0\leq e\leq E` and changes compare the same cohorts.

For that fixed set of cohorts, the parameter is

.. math::

   \theta_{\mathrm{dynamic}}^{\mathrm{bal}}(e;E)
   =\sum_{g\in\mathcal{G}_E}q_{g,E}ATT(g,g+e),
   \qquad 0\leq e\leq E.

Balancing changes the target to cohorts observed for the whole exposure
window and can discard many later adopters as a result. Keeping this fixed set for
post-treatment effects also does not ensure identical cohort support at
negative event times, since early cohorts may lack the required
pre-treatment observations. Although ``min_e`` and ``max_e`` select the
reported event window, truncating that window alone does not balance the cohorts.

The dynamic result's ``overall_att`` averages its included nonnegative
event-time effects equally. It therefore gives equal weight to exposure
lengths, rather than equal total weight to treated units. That target
can differ substantially from the overall group effect even when both
use exactly the same estimated :math:`ATT(g,t)` values.

Calendar-period averages
~~~~~~~~~~~~~~~~~~~~~~~~

If your question concerns a particular calendar period, we can average
over the cohorts already treated in that period. Passing
``type="calendar"`` to :func:`~moderndid.aggte` computes that average
for period :math:`t`,

.. math::

   \theta_{\mathrm{calendar}}(t)
   =\sum_{\substack{g\in\mathcal{G}\\g\leq t}}
      \frac{q_g}{\sum_{h\in\mathcal{G}:h\leq t}q_h}ATT(g,t).

This measures the average effect among units treated by :math:`t`.
Its changes across calendar periods mix evolving effects, exposure
lengths, and the entry of new cohorts. They cannot isolate the effect
of a macroeconomic shock or another policy that happened in that year.
The calendar result's ``overall_att`` gives equal weight to the included
calendar-period averages. The group, simple, dynamic, and calendar
overall effects each answer a different averaging question.

Uncertainty across effects and averages
---------------------------------------

Many group-time estimates share the same counties and comparison units.
Their errors are related because each comparison reuses some of the
observations. We need that joint uncertainty both to average the effects
and to assess a whole event-study path.

We now derive inference for the untrimmed DR panel estimator with
never-treated controls. The paper takes the number of periods as fixed
and lets the number of independent units grow. We first state the
conditions on the nuisance fits and follow their estimation error
into the joint distribution. Section 4 explains that the
not-yet-treated version follows by replacing the comparison group and
using Assumption 5 in place of Assumption 4.

Conditions on the nuisance fits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To estimate the population probabilities and conditional means in the
identification formulas, we fit chosen function families such as logistic
and linear regressions to the sample. These function families are the
working models on which the estimation procedure relies.

Here expectations use the panel distribution in Assumption 2. The indicator
:math:`C=\mathbf{1}\{G=\infty\}` selects never-treated controls. Their
membership does not change across outcome periods. We therefore abbreviate
the population propensity score :math:`p_{g,t}(X)` as :math:`p_g(X)`.
Write the working models as :math:`p_g(X;\pi_g)` and
:math:`m_{g,t}(X;\beta_{g,t})`. Stack their parameters in
:math:`\kappa_{g,t}=(\pi_g',\beta_{g,t}')'`. A star denotes the
population limit of a fitted parameter, even when its model is wrong.
The corresponding normalized population score is

.. math::

   \begin{aligned}
   r_g(X;\pi)&=\frac{p_g(X;\pi)}{1-p_g(X;\pi)},\\
   w_g^0(W;\pi)&=\frac{r_g(X;\pi)C}{\mathbb{E}[r_g(X;\pi)C]},\\
   h_{g,t}(W;\kappa)&=
      \bigl(w_g^1-w_g^0(W;\pi)\bigr)
      \bigl(\Delta Y_{g,t}-m_{g,t}(X;\beta)\bigr).
   \end{aligned}

An influence function describes a unit's contribution to the leading
estimation error. An asymptotically linear expansion writes that error
as an average of these contributions plus a smaller remainder. The
:math:`\sqrt n` scaling tracks fluctuations at the :math:`n^{-1/2}`
rate and :math:`o_p(1)` denotes a remainder that converges to zero
in probability.

The following conditions apply to both working models and their fitting
procedures. They specify the smoothness, moments, and expansions needed
to carry that reasoning through to the estimator.

.. admonition:: Assumptions 7 and 8 Nuisance conditions
   :class: assumption

   For Assumption 7, represent either working model by
   :math:`f(X;\gamma)`, with :math:`\gamma\in\Theta\subset\mathbb{R}^k`.
   The parameter space is compact and the model is almost surely
   continuous at every parameter value. There is a unique population
   limit :math:`\gamma^*\in\operatorname{int}(\Theta)`. The model is
   almost surely twice continuously differentiable on a neighborhood
   :math:`\Theta^*\subset\Theta` of that limit.

   The fitted parameter converges almost surely to :math:`\gamma^*` and
   has an asymptotically linear expansion,

   .. math::

      \sqrt n(\widehat\gamma-\gamma^*)
      =\frac{1}{\sqrt n}\sum_{i=1}^n
         \ell_\gamma(W_i;\gamma^*)+o_p(1).

   Its influence function has mean zero and finite, positive definite
   covariance,

   .. math::

      \mathbb{E}[\ell_\gamma(W;\gamma^*)]=0,\qquad
      \mathbb{E}[\ell_\gamma(W;\gamma^*)
                 \ell_\gamma(W;\gamma^*)']
      \quad\text{finite and positive definite}.

   It is locally continuous in mean square in the following uniform
   sense,

   .. math::

      \lim_{a\downarrow0}\mathbb{E}\left[
         \sup_{\substack{\gamma\in\Theta^*\\
                         \|\gamma-\gamma^*\|\leq a}}
         \|\ell_\gamma(W;\gamma)-\ell_\gamma(W;\gamma^*)\|^2
      \right]=0.

   For some :math:`\varepsilon>0`, each cohort's working propensity
   model also satisfies

   .. math::

      0\leq p_g(X;\pi)\leq1-\varepsilon
      \quad\text{almost surely for every }
      \pi\in\operatorname{int}(\Theta^{ps}),\quad g\in\mathcal{G},

   where :math:`\Theta^{ps}` is its parameter space.

   For Assumption 8, each :math:`g\in\mathcal{G}` and
   :math:`t\in\{2,\ldots,T-\delta\}` satisfies

   .. math::

      \mathbb{E}[|h_{g,t}(W;\kappa_{g,t}^*)|^2]<\infty,\qquad
      \mathbb{E}\left[
         \sup_{\kappa\in\Gamma_{g,t}^*}
         \|\partial_\kappa h_{g,t}(W;\kappa)\|
      \right]<\infty,

   where :math:`\Gamma_{g,t}^*` is a neighborhood of
   :math:`\kappa_{g,t}^*` and :math:`\|\cdot\|` is the Euclidean norm.

These population and large-sample requirements cannot be established by
checking that a fitted propensity score lies below a numerical threshold. The overlap
bound on the working model also differs from Assumption 6's bound on the
true propensity score.

Following the fitting step into the influence function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To see what contributes to the standard error, evaluate the working
models and weights at :math:`\kappa_{g,t}^*`. Suppress that argument
temporarily and define

.. math::

   R_{g,t}=\Delta Y_{g,t}-m_{g,t}(X;\beta_{g,t}^*),\qquad
   \mu_{g,t}^1=\mathbb{E}[w_g^1R_{g,t}],\qquad
   \mu_{g,t}^0=\mathbb{E}[w_g^0R_{g,t}].

The treated and control parts of the influence function center the
residual within their own weighted populations,

.. math::

   \psi_{g,t}^1(W)=w_g^1(R_{g,t}-\mu_{g,t}^1),\qquad
   \psi_{g,t}^0(W)=w_g^0(R_{g,t}-\mu_{g,t}^0).

Estimating the regression and propensity parameters adds two further
terms. The :math:`M` vectors below measure the sensitivity of the
population score mean to those fitted coefficients. Write :math:`\dot m_{g,t}`
and :math:`\dot p_g` for their parameter
gradients at the population limits, and define

.. math::

   \begin{aligned}
   a_g^{ps}(W)
      &=\frac{C}{(1-p_g(X;\pi_g^*))^2
                 \mathbb{E}[r_g(X;\pi_g^*)C]},\\
   M_{g,t}^{\beta}
      &=\mathbb{E}[(w_g^1-w_g^0)\dot m_{g,t}],\\
   M_{g,t}^{\pi}
      &=\mathbb{E}[a_g^{ps}(W)\dot p_g
                   (R_{g,t}-\mu_{g,t}^0)].
   \end{aligned}

For the first-step influence functions :math:`\ell_{g,t}^{\beta}` and
:math:`\ell_g^{\pi}` in Assumption 7, the complete influence function is

.. math::

   \psi_{g,t}(W;\kappa_{g,t}^*)
   =\psi_{g,t}^1(W)-\psi_{g,t}^0(W)
      -\ell_{g,t}^{\beta}(W)'M_{g,t}^{\beta}
      -\ell_g^{\pi}(W)'M_{g,t}^{\pi}.

In this normalized-ratio form, differentiating both the control weights'
numerator and their normalizing mean keeps the term :math:`\mu_{g,t}^0`
inside the propensity contribution. The same normalization appears in
`Sant'Anna and Zhao's Appendix A, equation A.2
<https://arxiv.org/html/1812.01723v3#A1.E2>`_.
Both :math:`M` terms vanish when both nuisance models are correct, whereas
a fitting term can remain with only one correct model. The influence
function is therefore not generally the DR score minus the ATT.

The joint large-sample distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define the cells covered by the paper's never-treated result as

.. math::

   \mathcal{J}_\delta=
   \{(g,t):g\in\mathcal{G}_\delta,\
            t\in\{2,\ldots,T-\delta\},\ t\geq g-\delta\}.

Stack their effects in :math:`\boldsymbol{ATT}` and their complete
influence functions in :math:`\Psi(W)` in the same order.

.. admonition:: Theorem 2 Joint large-sample distribution
   :class: theorem

   Under Assumptions 1 through 4 and 6 through 8, impose condition 4.5
   for every
   :math:`(g,t)\in\mathcal{J}_\delta`. At least one working model must
   be correct at its population limit almost surely,

   .. math::

      p_g(X;\pi_g^*)=p_g(X),\;\text{or}\;
      m_{g,t}(X;\beta_{g,t}^*)=m_{g,t}(X).

   For the untrimmed DR panel estimator using never-treated controls,
   every cell then has the expansion

   .. math::

      \begin{aligned}
      &\sqrt n\bigl(\widehat{ATT}_{dr}(g,t)-ATT(g,t)\bigr)\\
      &\quad=\frac{1}{\sqrt n}\sum_{i=1}^n
         \psi_{g,t}(W_i;\kappa_{g,t}^*)+o_p(1).
      \end{aligned}

   With :math:`T` fixed and :math:`n\to\infty`, the stacked estimator
   has the joint limit

   .. math::

      \begin{gathered}
      \sqrt n(\widehat{\boldsymbol{ATT}}-\boldsymbol{ATT})
      \xrightarrow{d}N(0,\Sigma),\\
      \Sigma=\mathbb{E}[\Psi(W)\Psi(W)'].
      \end{gathered}

The diagonal entries of :math:`\Sigma` describe the variances of the
scaled estimation errors, whereas its off-diagonal entries describe how
errors in two effects move together. The covariance matrix of the ATT
estimates is approximately :math:`\Sigma/n` in large samples.

Although the one-correct-model condition gives consistency, valid inference
also needs a variance estimate that keeps the appropriate fitting terms.
The paper's cell set includes allowed anticipation periods measured
against the clean base. It does not automatically include the package's
adjacent-period contrasts from a varying pre-treatment base.

Accounting for estimated aggregation weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A different sample can contain a different mix of adoption cohorts.
The estimated shares can therefore change along with the estimated
effects. Corollary 2 considers
:math:`\delta=0` and an aggregate over the identified post-treatment
cells :math:`\mathcal{I}\subseteq\mathcal{J}_0`. In addition to the
conditions of Theorem 2, each estimated weight must have an expansion

.. math::

   \sqrt n\bigl(\widehat w(g,t)-w(g,t)\bigr)
   =\frac{1}{\sqrt n}\sum_{i=1}^n\xi^w_{g,t}(W_i)+o_p(1).

The weight influence functions have mean zero and finite variance.
The paper requires positive variance for each nondegenerate estimated
weight. Their joint covariance can still be singular, since normalized
weights sum to one. Fixed weights contribute
:math:`\xi^w_{g,t}=0`. A first-order expansion of the product of each
estimated weight and effect accounts for both sources of error. For
:math:`\widehat\theta=\sum_{\mathcal{I}}\widehat w(g,t)\widehat{ATT}(g,t)`,
the aggregate influence function is

.. math::

   \psi_\theta(W)=\sum_{(g,t)\in\mathcal{I}}
      \left[w(g,t)\psi_{g,t}(W)
            +ATT(g,t)\xi^w_{g,t}(W)\right].

The aggregate's estimation error includes both the error in the effects
through the first term and the error in their weights through the second.
For weights fixed in advance, the second term is zero. The resulting
expansion and limit are

.. math::

   \sqrt n(\widehat\theta-\theta)
   =\frac{1}{\sqrt n}\sum_{i=1}^n\psi_\theta(W_i)+o_p(1)
   \xrightarrow{d}N(0,\mathbb{E}[\psi_\theta(W)^2]).

:func:`~moderndid.aggte` combines the influence functions and accounts
for estimated shares. Averaging the standard
errors of the group-time effects would miss both their covariance and
this weight uncertainty.

Simultaneous bands and the multiplier bootstrap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pointwise intervals give a separate coverage statement for each effect,
whereas a simultaneous band gives one coverage statement for all selected
effects together.
To construct that band, we need a critical value for
the largest absolute estimation error measured in standard-error units. The
multiplier bootstrap constructs that joint fluctuation from estimated
influence functions. We keep the observed data fixed and perturb the
estimates using random multipliers. For independent units, draw
multipliers :math:`V_i` independently across units and independently of
the data. One draw is

.. math::

   \widehat{ATT}^{*}(g,t)=\widehat{ATT}(g,t)
      +\frac{1}{n}\sum_{i=1}^{n}V_i\widehat\psi_{g,t}(W_i).

The same multiplier applies to every effect for a given unit. This
preserves the estimated covariance across effects without refitting
the propensity model or outcome regressions. The paper's next result
justifies using these draws for joint inference.

.. admonition:: Theorem 3 Conditional multiplier-bootstrap limit
   :class: theorem

   Assume the conditions of Theorem 2. Let the multipliers be independent
   draws from a common distribution, independent of the data, with

   .. math::

      \mathbb{E}[V_i]=0,\qquad
      \mathbb{E}[V_i^2]=1,\qquad
      \mathbb{E}[|V_i|^3]<\infty.

   Use the sample analogue of the complete influence function to
   construct :math:`\widehat{\boldsymbol{ATT}}^*`. Conditional on the
   observed sample, its centered distribution converges in probability
   to the same Gaussian limit as the estimator,

   .. math::

      \sqrt n(\widehat{\boldsymbol{ATT}}^*
                    -\widehat{\boldsymbol{ATT}})
      \xrightarrow{d^*}Z,\qquad Z\sim N(0,\Sigma).

   For every continuous map :math:`\Gamma` between finite-dimensional
   Euclidean spaces, the corresponding transformed limit also holds,

   .. math::

      \Gamma\!\left[\sqrt n(\widehat{\boldsymbol{ATT}}^*
                             -\widehat{\boldsymbol{ATT}})\right]
      \xrightarrow{d^*}\Gamma(Z).

The symbol :math:`\xrightarrow{d^*}` describes this conditional bootstrap
convergence, rather than a new sample drawn from the population. To
implement Algorithm 1, write
:math:`R_{g,t}^*=\sqrt n(\widehat{ATT}^*(g,t)-\widehat{ATT}(g,t))`.
Let :math:`q_{a,g,t}^*` be its empirical bootstrap quantile and let
:math:`z_a` be the standard normal quantile. For a normal distribution,
the interquartile range is its standard deviation times
:math:`z_{0.75}-z_{0.25}`. Dividing the bootstrap interquartile range
by that normal reference estimates the limiting standard deviation,

.. math::

   \widehat\sigma_{g,t}
   =\frac{q_{0.75,g,t}^*-q_{0.25,g,t}^*}{z_{0.75}-z_{0.25}},
   \qquad \widehat{se}_{g,t}=\frac{\widehat\sigma_{g,t}}{\sqrt n}.

Dividing each perturbation by its standard error puts the effects on
a comparable scale. For a band over :math:`\mathcal{J}_\delta`, we
take the largest absolute value in each draw. This gives the maximum
studentized deviation,

.. math::

   M^*=\max_{(g,t)\in\mathcal{J}_\delta}
      \left|\frac{\widehat{ATT}^{*}(g,t)-\widehat{ATT}(g,t)}
                   {\widehat{se}_{g,t}}\right|.

Let :math:`\alpha` denote the desired noncoverage probability for
the family of effects. The empirical :math:`1-\alpha` quantile of
:math:`M^*`, written :math:`\widehat c_{1-\alpha}`, gives the band

.. math::

   \widehat{ATT}(g,t)\ \pm
      \widehat c_{1-\alpha}\widehat{se}_{g,t}.

For :math:`0<\alpha<1`, Corollary 1 gives simultaneous coverage of the
entire family. Under the theorem's conditions and with positive limiting
standard deviations,

.. math::

   \lim_{n\to\infty}P\left(
      ATT(g,t)\in
      [\widehat{ATT}(g,t)\pm\widehat c_{1-\alpha}\widehat{se}_{g,t}]
      \text{ for all }(g,t)\in\mathcal{J}_\delta
   \right)=1-\alpha.

These coverage statements concern large samples and provide no
finite-sample guarantee. The coverage result uses the conditional
bootstrap quantile. A finite ``biters`` introduces Monte Carlo error
into the simulated quantiles, including those used for bootstrap standard
errors. The same construction
applies to a family of aggregated effects after replacing the influence
functions with their aggregate counterparts.

For bootstrap standard errors and simultaneous bands from
:func:`~moderndid.att_gt`, set ``boot=True`` and ``cband=True``
explicitly. Since the defaults are ``boot=False`` and ``cband=True``,
the default call uses analytic standard errors and pointwise intervals.

``alp`` sets :math:`\alpha` and ``biters`` sets the number of multiplier
draws. Passing ``random_state`` makes those draws reproducible across
repeated calls.
:func:`~moderndid.aggte` inherits these settings unless you override them.
For group, dynamic, or calendar aggregates, ``cband=True`` and
``boot=False`` retain analytic standard errors but draw bootstrap
perturbations for the simultaneous critical value. Set ``cband=False``
if you want pointwise intervals for those aggregate families.

If units share shocks within a state, independent-unit inference can
understate uncertainty. The theoretical clustered extension assigns
one multiplier to the whole cluster and perturbs the estimate by
:math:`n^{-1}\sum_c V_c\sum_{i\in c}\widehat\psi_{g,t}(W_i)`.
Its justification requires an increasing number of independent
clusters and suitable restrictions on cluster sizes. In the package,
``clustervars`` requests clustered bootstrap inference and requires
``boot=True``. It accepts the unit ID and one additional time-invariant
cluster variable. Set that variable in ``att_gt`` so ``aggte`` inherits
the cluster assignments. More counties within a few states do not substitute for
more independent states.

.. admonition:: Unequal cluster sizes need care
   :class: warning

   The current clustered bootstrap averages influence functions within
   each cluster before applying equally weighted cluster multipliers.
   Since unequal cluster sizes make this differ from the unit-weighted
   cluster-sum construction above, its coverage does not follow
   directly from that argument.

What pre-treatment estimates can tell you
-----------------------------------------

The inference results take the identifying assumptions as given. We
now return to the periods before the allowed anticipation window to
examine the observed untreated changes. Before :math:`g-\delta`, limited
anticipation makes the causal :math:`ATT(g,t)` zero. The entries before that boundary in an
``att_gt`` result estimate placebo DiD contrasts. We use them to examine
whether observed untreated trends line up, rather than to measure
an identified pre-treatment causal response.

With the default ``base_period="varying"``, a pre-treatment contrast
uses the change from :math:`t-1` to :math:`t`. With
``base_period="universal"``, it compares :math:`t` with the fixed
base :math:`b_g=g-\delta-1`. Ignoring covariates for this display,
the two contrasts are

.. math::

   \begin{aligned}
   \pi_{\mathrm{varying}}(g,t)
      &=\mathbb{E}[Y_t-Y_{t-1}\mid G=g]
        -\mathbb{E}[Y_t-Y_{t-1}\mid B=1],\\
   \pi_{\mathrm{universal}}(g,t)
      &=\mathbb{E}[Y_t-Y_{b_g}\mid G=g]
        -\mathbb{E}[Y_t-Y_{b_g}\mid B=1].
   \end{aligned}

Here :math:`B` denotes controls unaffected at both comparison endpoints.
For not-yet-treated comparisons, ModernDiD selects other cohorts whose
adoption is later than the later endpoint plus the anticipation horizon.
Outcome regression and propensity weighting can adjust these placebo
contrasts for covariates just as they adjust post-treatment comparisons.
The varying base examines successive changes before adoption rather
than deviations from a fixed reference period. The universal base
expresses earlier outcomes relative to that reference period. The
reference-period contrast is zero by construction and the package
reports it as zero without an estimable standard error. Both choices
give the same post-treatment :math:`ATT(g,t)` estimates.

Inside the allowed anticipation window :math:`g-\delta\leq t<g`,
the universal base can identify the anticipation-level ATT. The varying
base still uses adjacent periods until adoption. Its contrasts can
therefore subtract an already affected outcome and measure changes in
anticipatory responses rather than their full levels.

Zero placebo contrasts require parallel trends in the pre-treatment
periods being compared. They do not follow from the post-treatment-only
never-treated assumption stated earlier. A rejection can reveal
differences in earlier trends or anticipation outside the allowed
window. A nonrejection can reflect low power and cannot establish what
the treated cohort's untreated path would have been after adoption.

The reported Wald pre-test jointly tests pre-treatment contrasts when
their covariance matrix permits it. When clustering beyond the unit
level, the package omits that Wald test because its analytic covariance
does not incorporate those cluster correlations.

.. admonition:: Anticipation enters the pre-test
   :class: warning

   Because the current test selects cells with :math:`t<g`,
   ``anticipation>0`` can put allowed anticipation effects into that test.
   Interpret the tested window before treating its p-value as evidence
   against your design.

The :ref:`staggered DiD example <example_staggered_did>` applies these
comparisons, changes the base period and anticipation horizon, and
shows how aggregation changes the minimum wage question being answered.
If your question instead concerns how large departures from parallel
trends could be before the evidence changes, the
:ref:`sensitivity analysis example <example_honest_did>` takes an
event study through that calculation.
