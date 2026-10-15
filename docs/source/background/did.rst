.. _background-did:

Staggered difference-in-differences
===================================

Staggered difference-in-differences studies treatment effects when units
adopt at different dates and remain treated afterward. To estimate an
effect for a cohort of units that adopt in the same period, you need to
reconstruct its untreated outcome path without assuming that every cohort
responds to treatment in the same way.

In the :ref:`minimum wage example <example_staggered_did>`, counties whose
states raised the minimum wage in 2004 have already had two years of
exposure by the time the 2006 cohort adopts. If employment
responds differently across cohorts or changes with exposure, a single
regression coefficient can hide the effects you want to measure.

We build the analysis around the effect for one adoption cohort in one
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
developed here. You can read each condition alongside the explanation of
what it asks of your data and how the package uses it.

The effect for one cohort in one period
---------------------------------------

Consider a panel of :math:`n` units observed in periods
:math:`1,\ldots,T`. Let :math:`D_{it}` equal one if unit :math:`i` is
treated in period :math:`t` and zero otherwise. Population conditions omit
the unit index :math:`i`, as in the treatment restriction below.

.. admonition:: Assumption 1 Irreversible treatment
   :class: assumption

   Every unit is untreated in the first period and stays treated in every
   observed period after its treatment begins,

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
:math:`D_{it}`, its value stays the same across a unit's rows. ModernDiD
records never-treated units with ``0`` in place of the infinite adoption
time the theory assigns them.

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

The effect's definition does not by itself say which cohorts have an
untreated comparison. We will use the paper's cohort sets to state where
its identification results apply. Counting never-treated units as
adopting at infinity, let :math:`\bar g` be the largest supported adoption
time. Write :math:`\delta` for a known nonnegative integer that bounds
how far before adoption conditional mean responses may begin. Then set

.. math::

   \mathcal{G}=\operatorname{supp}(G)\setminus\{\bar g\},
   \qquad
   \mathcal{G}_\delta=\mathcal{G}\cap\{2+\delta,\ldots,T\}.

The set :math:`\mathcal{G}` keeps every finite adoption cohort when
never-treated units exist. When everyone adopts, the last cohort has no
untreated comparison after adoption and drops out of :math:`\mathcal{G}`.
Membership in :math:`\mathcal{G}_\delta` also leaves an observed period
before the anticipation window, the base period that the
limited-anticipation assumption below defines.

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

When treatment effects vary across cohorts or over time, the comparisons
behind this coefficient can mix the effects of different treatment paths.
They include earlier adopters measured against later adopters while the
later adopters are untreated.
They also include later adopters measured against earlier adopters after
the earlier adopters are treated. If the earlier cohort's effect changes
during that second comparison, its outcome change includes a treatment
response. Subtracting that change can distort the effect attributed to the
later cohort.

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
:math:`e=t-G_i` counts periods from adoption and is negative before it.
For a finite set of event times :math:`\mathcal{E}` and an omitted
reference period, the regression is

.. math::

   Y_{it}=\alpha_i+\lambda_t+
   \sum_{e\in\mathcal{E}}\beta_e\mathbf{1}\{G_i<\infty,\ t-G_i=e\}
   +v_{it}.

`Sun and Abraham (2021) <https://doi.org/10.1016/j.jeconom.2020.09.006>`_
show that a coefficient at one event time can contain effects from other
event times when cohort effects differ. Even a lead coefficient can reflect
post-treatment effects under no anticipation. We instead estimate each
:math:`ATT(g,t)` using an explicitly chosen untreated comparison group.
That choice fixes the comparison before any effects are averaged.

.. _background-did-assumptions:

Recovering the untreated outcome path
-------------------------------------

The identification argument needs a period before treatment can affect the
cohort and a comparison group whose outcome change represents the cohort's
untreated change. We state those requirements conditional on the
pre-treatment covariates :math:`X` introduced above. Conditional parallel
trends permits counties of different sizes to have different untreated
trends. It asks that treated and comparison counties with the same
:math:`X` share those trends.

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

While ``anticipation=0`` rules out an average response before adoption,
``anticipation=1`` allows the period immediately before adoption to
contain a response. The last unaffected base period is therefore

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

With ``control_group="nevertreated"``, the outcome changes of units that
never adopt supply the missing untreated changes under the following
conditional parallel trends assumption.

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
only from adoption onward and leaves that cohort's earlier observed trends
unrestricted. A pre-treatment diagnostic therefore checks an extension
of this assumption into earlier periods.

Choosing ``control_group="notyettreated"`` lets later adopters supply the
comparison as well. The paper's alternative assumption matches the cohort's
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

To identify the effect at :math:`t`, take :math:`s=t+\delta` so that a later
adopter qualifies only if it has not entered its anticipation window by
:math:`t`. Never-treated units also belong to this comparison group at every cutoff.
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
match those of the cohort. Callaway and Sant'Anna suggest the never-treated
comparison when that group is sizable and resembles the eventual adopters.
The not-yet-treated comparison suits applications where the never-treated
group is missing or too small.

If every unit eventually adopts, not-yet-treated comparisons identify
effects only while :math:`t<\bar g-\delta`. Observing the last cohort's
outcomes after that point does not extend this window.

When no never-treated cohort is available, the default control choice
warns and uses the latest cohort as the comparison. The package discards
periods at or after that cohort enters its anticipation window. Since
``control_group="notyettreated"`` also excludes those unsupported final
periods, neither choice estimates the last cohort's post-treatment effects.

With :math:`\delta>0`, nonadoption through :math:`T` also needs
interpretation because a unit adopting just after the sample ends may
already anticipate treatment inside the sample. Theorem 1 restricts identification
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
:math:`s`, the label :math:`\mathrm{cut}` keeps that index apart from the
outcome period,

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

Although the manuscript states this restriction for every observed cutoff,
a cutoff after every unit adopts has no eligible controls and cannot
satisfy it. When no never-treated cohort exists, we impose Assumption 6 on the
eligible comparison cutoffs :math:`s<\bar g`. The not-yet-treated results
below use that scope because their comparisons require an untreated group.

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

Fixing the base and the controls makes the missing untreated change
recoverable. We derive the post-treatment comparison for :math:`t\geq g`
using the clean base :math:`b_g`. The same theoretical identities extend to
the anticipation window :math:`g-\delta\leq t<g`. With
``base_period="universal"``, :func:`~moderndid.att_gt` estimates those
anticipation effects relative to the same clean base :math:`b_g`.
Define the control outcome regression

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

   \mathbb{E}[Y_t(0)-Y_{b_g}(0)\mid X,G=g]=m_{g,t}(X).

The conditional treatment effect equals the cohort's observed change minus
this untreated change,

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
:math:`\mathbb{E}[r_{g,t}(X)B_{g,t}]=\mathbb{E}[A_g]` and, for any integrable
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
change. An IPW estimate instead requires a correct model of cohort
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

   With Assumption 4 and :math:`B_{g,t}=\mathbf{1}\{G=\infty\}`, these
   equalities hold for every :math:`g\in\mathcal{G}_\delta` and
   :math:`t\in\{2,\ldots,T-\delta\}` satisfying :math:`t\geq g-\delta`.

   With Assumption 5 and :math:`B_{g,t}=\mathbf{1}\{G>t+\delta,G\ne g\}`,
   they hold for every :math:`g\in\mathcal{G}_\delta` and
   :math:`t\in\{2,\ldots,T-\delta\}` satisfying
   :math:`g-\delta\leq t<\bar g-\delta`.

   Each comparison uses its own :math:`p_{g,t}`, :math:`m_{g,t}`, and
   normalized control weights around the clean base
   :math:`b_g=g-\delta-1`.

The weighted residual inside Theorem 1's last expectation is the DR
score. Estimation replaces its population expectations, propensity
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
:math:`\widetilde w^0` use the candidate propensity odds. With
:math:`ATT_{dr}(g,t;\widetilde p,\widetilde m)` denoting the resulting
population score mean, the bias satisfies

.. math::

   \begin{aligned}
   &ATT_{dr}(g,t;\widetilde p,\widetilde m)-ATT(g,t)\\
   &\quad=\mathbb{E}\bigl[(w_g^1-\widetilde w^0)
      (m_{g,t}(X)-\widetilde m(X))\bigr].
   \end{aligned}

The bias vanishes when the outcome regression is correct because the
regression error on the right is zero. A correct propensity model also
removes the bias by balancing that error across treated units and controls.

.. admonition:: Double robustness still requires parallel trends
   :class: warning

   Double robustness protects against misspecifying one nuisance model
   but still requires the chosen parallel trends assumption, limited
   anticipation, and overlap. With both nuisance models misspecified, the
   estimator has no such consistency guarantee.

The built-in IPW and DR estimators omit control weighting contributions
when fitted propensity scores reach 0.995. Before a built-in estimator
runs, :func:`~moderndid.att_gt` also screens each comparison with the
sampling weights that the estimators use. For the DR and IPW estimators,
it skips a comparison when a logit fitted with those weights gives a unit
with positive weight a propensity score of at least 0.999. For the DR and
regression estimators, it skips one whose weighted control covariates
leave the outcome regression numerically singular. It also skips a
comparison whose treated units or controls all have zero weight. With
repeated cross-sections, it applies that check within each of the
comparison's two periods. A warning names every comparison that fails
these checks or whose estimator returns no estimate. The cell of such a
comparison then drops out of the result under ``base_period="varying"``
and remains as NaN under ``base_period="universal"``.

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

The sample formula still leaves open how to fit the nuisance
functions. The default ``est_method="dr"`` uses a logistic propensity
model and linear outcome regressions for each cohort-time comparison.
You can choose
``"ipw"`` for propensity weighting or ``"reg"`` for outcome regression.
For a balanced panel, the control regression models the change between the
base and current period from covariates measured at the earlier endpoint.
Since ``xformla`` selects existing numeric columns, create transformations
or interaction columns before passing them. If you supply ``weightsname``,
estimation and cohort shares use the sampling weights rather than giving
every observed unit equal weight.

Choose covariates for their ability to account for differences in untreated
trends.

.. admonition:: Post-adoption covariates can absorb the effect
   :class: warning

   A covariate measured after adoption can contain part of the policy
   response you are trying to estimate. A correctly fitted model for that
   variable does not repair the causal interpretation.

The two-period estimators behind each comparison follow `Sant'Anna and Zhao
(2020) <https://doi.org/10.1016/j.jeconom.2020.06.003>`_. Their efficiency
and inference results depend on the estimator and the nuisance-fitting
procedure. When both panel nuisance models are correct, the untrimmed DR estimator
attains the semiparametric efficiency bound under their regularity
conditions. This bound describes the lowest asymptotic variance available
to regular estimators under the two-period sampling design and identifying
assumptions. The default here uses
traditional propensity maximum likelihood and
least-squares regression. It does not use the improved inverse probability
tilting and weighted regression procedure described in the
:ref:`doubly robust DiD background <background-drdid>`. Supplying a custom
estimator also requires a valid
:ref:`influence function <background-influence-functions>` for its leading
sampling error, as the inference section explains. Double robustness alone
does not justify arbitrary machine learning fits or their standard errors.

When different units are observed in each period
------------------------------------------------

The panel derivation uses each unit's observed change between two periods.
When different units are sampled in each period, that change is unobserved.
Since each group's mean change then comes from separate outcome means at
the two endpoints, the comparison requires stable cohort composition.

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
   mixture of period-specific distributions whose mixture probabilities
   are :math:`\lambda_r`.

A change in who belongs to a cohort or in its covariate distribution needs
a different justification. Observing both conditional outcome distributions
also requires positive sampling probabilities :math:`\lambda_t>0` and
:math:`\lambda_{b_g}>0` at the two endpoints.

Write :math:`\mathbb{E}_M` for expectation under this mixture distribution
and, for each endpoint :math:`u\in\{b_g,t\}`, define the treated and control
level regressions,

.. math::

   \begin{aligned}
   \mu_{1u}(X)&=\mathbb{E}_M[Y\mid X,A_g=1,S=u],\\
   \mu_{0u}(X)&=\mathbb{E}_M[Y\mid X,B_{g,t}=1,S=u],\\
   \Delta\mu_j(X)&=\mu_{jt}(X)-\mu_{jb_g}(X),\quad j\in\{0,1\}.
   \end{aligned}

Because the same :math:`B_{g,t}` selects controls at both endpoints,
not-yet-treated comparisons use units still unaffected at :math:`t` rather
than everyone untreated at the earlier endpoint. With
:math:`J_u=\mathbf{1}\{S=u\}`, the period-specific weights are

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
conditional means over the treated cohort's covariates. The IPW expression
instead reconstructs that comparison from the period-specific weights. The
DR expression starts from the regression expression and adds weighted
residual adjustments. The three repeated-cross-section estimands are

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

   Under Assumptions 1, 3, 4, 6, and B.1 with :math:`B_{g,t}=C`, every
   :math:`g\in\mathcal{G}_\delta` and :math:`t\in\{2,\ldots,T-\delta\}`
   satisfying :math:`t\geq g-\delta` has

   .. math::

      ATT(g,t)=ATT_{\mathrm{or,rc}}(g,t)
              =ATT_{\mathrm{ipw,rc}}(g,t)
              =ATT_{\mathrm{dr,rc}}(g,t).

   With Assumption 5 in place of Assumption 4 and
   :math:`B_{g,t}=\mathbf{1}\{G>t+\delta,G\ne g\}`, the same three
   equalities hold on that cohort and time domain while
   :math:`t<\bar g-\delta`.

Correct treated-group regressions can improve the efficiency of the DR
estimator. Its untrimmed version attains the repeated-cross-section
efficiency bound when the propensity model and all four outcome regressions
are correct. For double robustness, the outcome-model requirement concerns
the difference between the two control regressions. Correctness of both control level
regressions is sufficient but stronger than correctness of their
difference, as Sant'Anna and Zhao's Theorem 1 shows.

For this sampling design, set ``panel=False`` so that
:func:`~moderndid.att_gt` treats every row as its own observation. Under
the defaults ``panel=True`` and ``allow_unbalanced_panel=False``, it keeps
only units observed in every period. With ``allow_unbalanced_panel=True``,
an unbalanced panel keeps its incomplete units and goes through the
repeated-cross-section estimation path. This setting does not establish
that attrition leaves the identifying comparison valid. You still need a
reason why the observed samples represent the population in your target.

Choosing what the effects should average
----------------------------------------

Under these conditions, either sampling design gives a collection
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
cohorts in :math:`\mathcal{G}`. The shares of the treated cohorts in the
target population are :math:`q_g=P(G=g\mid G\in\mathcal{G})`. Each average
requires a nonempty set of contributing cohorts so its denominator is
positive. If you restrict the sample, event window, or identified cells,
the weights and the population being averaged must reflect that
restriction. Removing missing effects with ``na_rm=True`` cannot recover
their contribution to the original target.

Cohort averages and the overall effect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the effect on a typical treated unit, we first average
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

Under the simple average, a cohort's total weight is proportional to
:math:`q_g(T-g+1)`. Because early adopters contribute more treated
periods, they receive more total weight than under the group average.
Both targets average causal effects over different distributions of
cohorts and exposure. The default ``type="group"`` uses the overall effect
formed from the cohort averages.

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

Event time zero measures the effect in the period of adoption itself.
Later event times describe progressively longer exposure among the cohorts
observed that long. While all three cohorts in the minimum wage example
contribute at event time zero, only the 2004 cohort contributes two and
three years after adoption. Growth between those points can therefore
reflect a change in the cohorts being averaged.

For :math:`0\leq e_1<e_2` and :math:`a_g(e)=ATT(g,g+e)`, the nesting
:math:`\mathcal{G}_{e_2}\subseteq\mathcal{G}_{e_1}` splits the change in
the event study into

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
surviving cohorts gain weight and other cohorts leave the average. You can
interpret the difference as an average change in effects within cohorts
only when the composition terms cancel or when you hold the cohorts and
weights fixed.

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
window and can discard many later adopters as a result. Keeping this fixed
set for post-treatment effects also does not ensure identical cohort
support at negative event times. Early cohorts may lack the pre-treatment
observations that those event times require. Although ``min_e`` and
``max_e`` select the reported event window, truncating that window alone
does not balance the cohorts.

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
observations. Averaging the effects and assessing a whole event-study path
both need that joint uncertainty.

We now derive inference for the untrimmed DR panel estimator with
never-treated controls. The paper takes the number of periods as fixed
and lets the number of independent units grow. After the conditions on
the nuisance fits, the section traces their estimation error into the
joint distribution. Section 4 explains that the
not-yet-treated version follows by replacing the comparison group and
using Assumption 5 in place of Assumption 4.

Conditions on the nuisance fits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To estimate the population probabilities and conditional means in the
identification formulas, the procedure fits chosen function families such
as logistic and linear regressions to the sample. These function families
are the working models it relies on.

Expectations here use the panel distribution in Assumption 2 and the
never-treated comparison. Because its indicator
:math:`C=\mathbf{1}\{G=\infty\}` selects the same controls in every
outcome period, the population propensity score :math:`p_{g,t}(X)`
shortens to :math:`p_g(X)`. The working models :math:`p_g(X;\pi_g)` and
:math:`m_{g,t}(X;\beta_{g,t})` stack their parameters in
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
estimation error. The :ref:`influence functions background
<background-influence-functions>` develops that idea, the asymptotically
linear expansions the conditions below assume, and the terms that fitted
nuisance models add.

The following conditions apply to both working models and their fitting
procedures. They specify the smoothness, moments, and expansions needed
to derive the estimator's influence function below.

.. admonition:: Assumptions 7 and 8 Nuisance conditions
   :class: assumption

   For Assumption 7, represent either working model by
   :math:`f(X;\gamma)` for a parameter
   :math:`\gamma\in\Theta\subset\mathbb{R}^k`. The parameter space is
   compact and the model is almost surely continuous at every parameter
   value. The model also has a unique population limit
   :math:`\gamma^*\in\operatorname{int}(\Theta)` and is almost surely twice
   continuously differentiable on a neighborhood
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

.. _background-did-fitting:

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
population score mean to those fitted coefficients. With
:math:`\dot m_{g,t}` and :math:`\dot p_g` the parameter gradients at the
population limits, define

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
<https://arxiv.org/html/1812.01723v3#S7.E2>`_.
Although both :math:`M` terms vanish when both nuisance models are correct,
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
scaled estimation errors. Its off-diagonal entries describe how errors in
two effects move together. The covariance matrix of the ATT estimates is
approximately :math:`\Sigma/n` in large samples.

Although the one-correct-model condition gives consistency, valid inference
also needs a variance estimate that keeps the appropriate fitting terms.
The paper's cell set measures allowed anticipation periods against the
clean base. Its Remark 12 extends the results to the adjacent-period
placebo contrasts before :math:`g-\delta`, though not to the
adjacent-period contrasts that the package's varying base reports inside
the anticipation window.

.. _background-did-aggregation-weights:

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

The paper requires each weight influence function to have mean zero and a
finite variance that is positive for a nondegenerate estimated weight.
Since normalized weights sum to one, the joint covariance of these influence
functions can still be singular. A first-order expansion of the product of
each estimated weight and effect accounts for both sources of error. For
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

.. _background-did-bootstrap:

Simultaneous bands and the multiplier bootstrap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pointwise intervals give a separate coverage statement for each effect.
A simultaneous band instead gives one coverage statement for all selected
effects together. That band needs a critical value for the largest
absolute estimation error measured in standard-error units. The
multiplier bootstrap constructs that joint fluctuation from estimated
influence functions by holding the observed data fixed and perturbing the
estimates with random multipliers. For independent units, draw
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

   Assume the conditions of Theorem 2 and let the multipliers be
   independent draws from a common distribution that is independent of
   the data and satisfies

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
convergence, rather than a new sample drawn from the population.
Algorithm 1 works with the scaled draw
:math:`R_{g,t}^*=\sqrt n(\widehat{ATT}^*(g,t)-\widehat{ATT}(g,t))`, its
empirical bootstrap quantile :math:`q_{a,g,t}^*`, and the standard normal
quantile :math:`z_a`. For a normal distribution,
the interquartile range is its standard deviation times
:math:`z_{0.75}-z_{0.25}`. Dividing the bootstrap interquartile range
by that normal reference estimates the limiting standard deviation,

.. math::

   \widehat\sigma_{g,t}
   =\frac{q_{0.75,g,t}^*-q_{0.25,g,t}^*}{z_{0.75}-z_{0.25}},
   \qquad \widehat{se}_{g,t}=\frac{\widehat\sigma_{g,t}}{\sqrt n}.

Dividing each perturbation by its standard error puts the effects on
a comparable scale. A band over :math:`\mathcal{J}_\delta` takes the
largest absolute value in each draw. This gives the maximum studentized
deviation,

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
finite-sample guarantee. Since the coverage result uses the conditional
bootstrap quantile, a finite ``biters`` adds Monte Carlo error to the
simulated quantiles behind both the critical value and the bootstrap
standard errors. The same construction applies to a family of aggregated
effects after replacing the influence functions with their aggregate
counterparts.

Under its defaults ``boot=False`` and ``cband=True``,
:func:`~moderndid.att_gt` reports analytic standard errors and pointwise
intervals because its simultaneous band needs the bootstrap. Setting
``boot=True`` draws the multiplier bootstrap for both the standard errors
and the band.

``alpha`` sets :math:`\alpha` and ``biters`` sets the number of multiplier
draws. Passing ``random_state`` makes those draws reproducible across
repeated calls. Unless you override them, :func:`~moderndid.aggte`
inherits all three settings along with ``boot`` and ``cband``.
For group, dynamic, or calendar aggregates, ``cband=True`` and
``boot=False`` retain analytic standard errors but draw bootstrap
perturbations for the simultaneous critical value. Set ``cband=False``
if you want pointwise intervals for those aggregate families.

If units share shocks within a state, independent-unit inference can
understate uncertainty. Callaway and Sant'Anna's Remark 10 clusters the
bootstrap by drawing one multiplier :math:`V_c` for each cluster
:math:`c` instead of one for each unit. Since every unit in a cluster
shares that multiplier, a draw perturbs the estimate with the cluster sums
of the influence function,

.. math::

   \frac{1}{n}\sum_{c}V_c\sum_{i\in c}\widehat\psi_{g,t}(W_i).

The validity of this clustered bootstrap requires an increasing number
of independent clusters and suitable restrictions on cluster sizes. More
counties within a few states do not substitute for more independent states.

.. admonition:: Clustered bands need many effective clusters
   :class: warning

   When a few states hold most of a cohort's treated units, its
   cohort-time effects rest on few effective clusters even if the data
   contain many states. Simultaneous bands can then cover less often than
   their nominal level even where pointwise intervals come close to it.
   Callaway and Sant'Anna's Remark 13 notes that the many-clusters
   condition can be more plausible for aggregated parameters than for the
   :math:`ATT(g,t)` themselves.

In the package, ``clustervars`` accepts the unit ID and one additional
time-invariant cluster variable. Set that variable in ``att_gt`` so
``aggte`` inherits the cluster assignments. Since ``aggte`` has no data of
its own, it rejects any other cluster variable and gives unit-level
inference for an empty list or the unit ID alone. With ``boot=True``, the
cluster variable makes the bootstrap draw one multiplier per cluster.
Without the bootstrap, the analytic covariance sums the influence function
within each cluster before forming the outer products. The draws behind an
aggregate's simultaneous band still use one multiplier per cluster.
``att_gt`` warns when a treated cohort's units fall in fewer than 5 clusters
or all units fall in fewer than 30. Because the warning counts clusters
rather than effective clusters, it stays silent when a few of many states
hold most of a cohort's treated units.

What pre-treatment estimates can tell you
-----------------------------------------

With the identifying assumptions taken as given so far, we now return to
the periods before the allowed anticipation window to examine the observed
untreated changes. Since limited anticipation makes the causal
:math:`ATT(g,t)` zero before :math:`g-\delta`, the entries before that
boundary in an ``att_gt`` result estimate placebo DiD contrasts. They
serve to examine whether observed untreated trends line up rather than to
measure an identified pre-treatment causal response.

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

Here :math:`B` marks the controls unaffected at both comparison endpoints.
For not-yet-treated comparisons, ModernDiD selects other cohorts whose
adoption is later than the later endpoint plus the anticipation horizon.
Outcome regression and propensity weighting can adjust these placebo
contrasts for covariates just as they adjust post-treatment comparisons.
The varying base examines successive changes before adoption rather
than deviations from a fixed reference period. The universal base
expresses earlier outcomes relative to that reference period. Since the
reference-period contrast is zero by construction, the package reports it
as zero without an estimable standard error. Both choices
give the same post-treatment :math:`ATT(g,t)` estimates because both
measure those periods from :math:`b_g`.

Inside the allowed anticipation window :math:`g-\delta\leq t<g`,
the universal base can identify the anticipation-level ATT. Because the
varying base still uses adjacent periods until adoption, its contrasts
there can subtract an already affected outcome and measure changes in
anticipatory responses rather than their full levels.

Zero placebo contrasts require parallel trends in the pre-treatment
periods being compared. They do not follow from the post-treatment-only
never-treated assumption stated earlier. A rejection can reveal
differences in earlier trends or anticipation outside the allowed
window. A nonrejection can reflect low power and cannot establish what
the treated cohort's untreated path would have been after adoption.

The reported Wald pre-test jointly tests pre-treatment contrasts when
their covariance matrix permits it. Since a Wald test on a cluster-robust
covariance rejects too often at realistic numbers of clusters, the package
omits it when clustering beyond the unit level. The bootstrap's uniform
bands over the pre-treatment cells take its place.

.. admonition:: The pre-test can include allowed anticipation effects
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
