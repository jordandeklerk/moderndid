.. _background-didinter:

DiD with intertemporal treatment effects
========================================

Intertemporal difference-in-differences studies how changes in treatment affect
outcomes over the periods that follow. If treatment increases, decreases, or is
withdrawn, earlier exposure can still influence the outcome you observe.
Comparing groups only by their current treatment can therefore miss part of the
effect. The ``didinter`` module estimates the effect of each group's observed
treatment path relative to keeping treatment at its initial level.

The framework of `de Chaisemartin and D'Haultfœuille
<https://doi.org/10.1162/rest_a_01414>`_ reveals that missing outcome change using
groups that retain the same initial treatment. Its no-anticipation and parallel
trends restrictions concern the outcomes groups would have experienced if their
baseline treatment had continued. Treatment can be binary or a more general dose
without requiring homogeneous effects. A :ref:`staggered adoption analysis
<background-did>` also allows effects to evolve after adoption while treatment
remains in place; here treatment itself can change again.

We begin with the comparisons available in your panel and derive the effect of
the observed treatment path before examining what normalization by treatment
exposure changes. These targets guide the horizon, placebo, and inference choices
in :func:`~moderndid.did_multiplegt`. The :ref:`worked example
<example_inter_did>` puts those distinctions into practice in an analysis you can
run.

The comparison available in your panel
--------------------------------------

Before defining an event-study effect, we need to find a group that can reveal
the status-quo outcome change. Consider :math:`G` groups observed in
:math:`T` periods. Let :math:`D_{g,t}\geq0` be treatment and
:math:`Y_{g,t}(d_{1:t})` the potential outcome under assignments
:math:`d_{1:t}`. Groups can be states, firms, or individuals. Treatment can be
binary, discrete, or a varying dose.

Write :math:`b_g=D_{g,1}` for baseline treatment and define the first change
from it by

.. math::

   F_g=\min\{t\geq2:D_{g,t}\ne b_g\},
   \qquad F_g=T+1\text{ if treatment never changes}.

Before :math:`F_g`, treatment remains at :math:`b_g` and can take different
values afterward. The status-quo potential outcome is

.. math::

   Q_{g,t}=Y_{g,t}(b_g,\ldots,b_g).

This is the outcome if treatment never starts for an initially untreated
group or if it stays at its initial level for an initially treated group.
The latter counterfactual explains why baseline treatment must enter
the comparison.

.. admonition:: Design restriction 1 An available same-baseline comparison
   :class: assumption

   There are groups :math:`g,g'` such that

   .. math::

      b_g=b_{g'},\qquad F_g\ne F_{g'}.

This restriction guarantees at least one comparison at a first switch.
Identification at a later horizon requires a same-baseline group
that has not switched by that horizon. Binary staggered adoption, treatment
with exit, and heterogeneous doses starting at zero can satisfy the
restriction when their first-switch dates differ. Starting every group at
zero alone does not supply a comparison if they all switch simultaneously.

For each group, define the last period before every same-baseline group has
switched by

.. math::

   T_g=\max_{g':b_{g'}=b_g}F_{g'}-1.

At horizon :math:`\ell`, the outcome period is
:math:`\tau_{g,\ell}=F_g-1+\ell`. Thus :math:`\ell=1` is the switching
period itself. The eligible groups and their count are

.. math::

   \mathcal I_\ell=\{g:F_g-1+\ell\leq T_g\},
   \qquad N_\ell=|\mathcal I_\ell|.

These formulas describe the balanced, equally weighted group panel in the
paper. The implementation also requires observed outcome changes and valid
controls at the requested horizon. With multiple observations per group-period
or ``weightsname`` supplied, its aggregation uses the corresponding cell
weights. Since calendar periods enter by their rank, irregular calendar
spacing requires care when interpreting a horizon's duration.

What no anticipation and parallel trends require
------------------------------------------------

Observed controls reveal the status-quo trend only under substantive
restrictions on potential outcomes. We condition on the full treatment design
:math:`\boldsymbol D=(D_{g,t})_{g,t}` throughout. This permits the effects
and the target population to depend on the realized paths.

.. admonition:: Assumption 1 No anticipation
   :class: assumption

   For every group, period, and possible treatment path,

   .. math::

      Y_{g,t}(d_1,\ldots,d_T)=Y_{g,t}(d_1,\ldots,d_t).

No anticipation leaves outcomes free to depend on every past assignment
while ruling out a response to treatment that has not yet occurred. A
policy announcement can therefore require an earlier treatment date if
behavior changes before implementation.

Let :math:`\mathcal D_1^r` contain baseline treatment values shared by at
least two groups with different first-switch dates. Parallel trends is required
within these relevant baseline categories.

.. admonition:: Assumption 2 Same-baseline status-quo parallel trends
   :class: assumption

   For every :math:`t\geq2` and all groups :math:`g,g'` with
   :math:`b_g=b_{g'}\in\mathcal D_1^r`,

   .. math::

      \mathbb E[Q_{g,t}-Q_{g,t-1}\mid\boldsymbol D]
      =\mathbb E[Q_{g',t}-Q_{g',t-1}\mid\boldsymbol D].

Since the assumption restricts changes in the status-quo outcome, it permits
groups to have different outcome levels and treatment effects. Comparing
initially treated and initially untreated groups under a common status-quo
trend would impose an additional restriction beyond this assumption.
Matching baseline treatment avoids needing that cross-baseline comparison.

An effect of the observed path
------------------------------

The question at horizon :math:`\ell` is what the switcher's outcome would
have been if its initial treatment had continued. The group-specific
actual-versus-status-quo effect is

.. math::

   \delta_{g,\ell}
   =\mathbb E[Y_{g,\tau_{g,\ell}}-Q_{g,\tau_{g,\ell}}
              \mid\boldsymbol D],
   \qquad g\in\mathcal I_\ell.

This effect covers :math:`\ell` treated periods for a group that adopts
binary treatment permanently and includes the remaining effect of earlier
exposure for a group that later exits. Different groups can reach the same
horizon with different doses and different numbers of treated periods.

The comparison uses the period immediately before the first change as its
baseline. Let

.. math::

   \mathcal C_{g,\ell}
   =\{g':b_{g'}=b_g,\ F_{g'}>\tau_{g,\ell}\}.

Every control has retained :math:`b_g` from period one through the outcome
period. The corresponding DiD is

.. math::

   \begin{aligned}
   \operatorname{DID}_{g,\ell}
   ={}&Y_{g,\tau_{g,\ell}}-Y_{g,F_g-1}\\
   &-\frac1{|\mathcal C_{g,\ell}|}
      \sum_{g'\in\mathcal C_{g,\ell}}
       (Y_{g',\tau_{g,\ell}}-Y_{g',F_g-1}).
   \end{aligned}

.. admonition:: Lemma 1 Identification of the path effect
   :class: theorem

   Under Assumptions 1 and 2, for every group with :math:`F_g\leq T_g`
   and every :math:`1\leq\ell\leq T_g-F_g+1`,

   .. math::

      \mathbb E[\operatorname{DID}_{g,\ell}\mid\boldsymbol D]
      =\delta_{g,\ell}.

Because no anticipation makes the switcher's baseline outcome a status-quo
outcome, parallel trends can supply its missing status-quo change from the
controls. Subtracting that change leaves the effect of the whole observed
path through the horizon, including the effects of earlier post-switch
assignments.

``only_never_switchers=False`` permits controls that switch later. Setting
it to ``True`` restricts controls to groups whose treatment never changes.
Both choices still require controls to match the switcher's baseline treatment.

Orienting increases and decreases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If some groups increase treatment and others decrease it, the raw effects
answer interventions in opposite directions. We orient decreases with a minus
sign before averaging. Under the following design restriction, the direction
of the first switch also describes the direction of every deviation from
baseline.

.. admonition:: Design restriction 2 No crossing of baseline treatment
   :class: assumption

   For each group, either :math:`D_{g,t}\geq b_g` for every period, or
   :math:`D_{g,t}\leq b_g` for every period.

A path can return to baseline and vary repeatedly on the same side of it
without crossing to the other side. This restriction supports a same-direction
interpretation and the nonnegative normalized weights below. Lemma 1's
conditional unbiasedness itself uses Assumptions 1 and 2 rather than this
restriction.

Let :math:`S_g=\operatorname{sign}(D_{g,F_g}-b_g)` for switchers and
:math:`S_g=0` for never-switching groups. The
oriented event-study target and estimator are

.. math::

   \delta_\ell=\frac1{N_\ell}\sum_{g\in\mathcal I_\ell}S_g\delta_{g,\ell},
   \qquad
   \operatorname{DID}_\ell
   =\frac1{N_\ell}\sum_{g\in\mathcal I_\ell}S_g
       \operatorname{DID}_{g,\ell}.

Lemma 1 gives conditional unbiasedness for this average whether
``switchers="in"`` retains first increases or ``switchers="out"`` retains
first decreases. The default pools both directions after reversing the
sign of decreases. Groups excluded as switchers can still serve as controls
until they switch.

By default, the package discards a group's periods after its treatment has
been both above and below baseline. Earlier valid horizons can remain in the
analysis. If ``keep_bidirectional_switchers=True`` retains these periods,
the same-direction and nonnegative-weight interpretations need not hold.

Which groups enter each horizon
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A change between horizon estimates can reflect both the treatment path
and the groups being averaged. Longer horizons often use fewer switchers
because they require later outcomes and a comparison group that has not
yet switched.

``same_switchers=True`` keeps only switchers that reach every requested effect
horizon. Reaching a horizon requires both an observed outcome change and
an eligible same-baseline control. ``same_switchers_pl=True`` also
restricts the placebo sample to switchers reaching every requested placebo;
it requires ``same_switchers=True``. These restrictions stabilize membership
while potentially reducing precision.

In binary absorbing designs, this event-study comparison agrees with the
unconditional staggered-adoption comparison using not-yet-treated controls,
a universal pre-adoption baseline, and matching aggregation weights. Here
horizon :math:`\ell` corresponds to event time :math:`\ell-1` there.
The paper also explains a numerical equivalence after binarizing and
staggerizing treatment when all groups share a baseline. Such a relabeling
still estimates the effect of the underlying observed dose path. With
varying baselines, ignoring the baseline-treatment match changes the
identifying assumption.

Putting effects on a treatment-dose scale
-----------------------------------------

When groups receive different doses, you may want to convert the unnormalized
estimate from outcome units to an effect per unit of additional exposure.
We first define that denominator for a group and examine what it does to
the aggregate target.

The cumulative change in dose from the first switch through the horizon is

.. math::

   A_{g,\ell}
   =\sum_{k=0}^{\ell-1}(D_{g,F_g+k}-b_g),
   \qquad
   \delta_{g,\ell}^{n}=\frac{\delta_{g,\ell}}{A_{g,\ell}}.

Under design restriction 2, :math:`S_gA_{g,\ell}=|A_{g,\ell}|>0` for
an eligible switcher. The normalized effect can therefore be understood as
a weighted average of current and lagged treatment effects, even without
a differentiable dose-response function.

To define those effects, let :math:`\tau=\tau_{g,\ell}`. For
:math:`k=0,\ldots,\ell-1`, construct two histories through :math:`\tau`
that agree with the observed path through :math:`\tau-k-1` and have treatment
:math:`b_g` in every period after :math:`\tau-k`. Set the treatment at
:math:`\tau-k` to its observed value in :math:`P_{g,\ell,k}^1` and to
:math:`b_g` in :math:`P_{g,\ell,k}^0`. The secant slope is

.. math::

   s_{g,\ell,k}
   =\frac{\mathbb E[Y_{g,\tau}(P_{g,\ell,k}^1)
                     -Y_{g,\tau}(P_{g,\ell,k}^0)
                     \mid\boldsymbol D]}
          {D_{g,\tau-k}-b_g}.

Set the slope contribution to zero when the denominator is zero because
the two potential outcomes then coincide. Each slope changes one treatment
assignment while holding earlier assignments at observed values and later
assignments at baseline.

.. admonition:: Lemma 2 The normalized lag decomposition
   :class: theorem

   For every eligible :math:`(g,\ell)` with :math:`A_{g,\ell}\ne0`,
   define :math:`w_{g,\ell,k}=(D_{g,\tau_{g,\ell}-k}-b_g)/A_{g,\ell}`.
   Then

   .. math::

      \delta_{g,\ell}^{n}
      =\sum_{k=0}^{\ell-1}w_{g,\ell,k}s_{g,\ell,k},
      \qquad \sum_{k=0}^{\ell-1}w_{g,\ell,k}=1.

   Under design restriction 2, every :math:`w_{g,\ell,k}` is nonnegative.

The telescoping decomposition gives each lag weight :math:`1/\ell` under
binary absorbing treatment. With a varying dose, a lag receives more weight
when its dose differs more from baseline. The normalized estimate is
consequently an average of lag effects through that horizon, rather than
a separate estimate of its last lag's effect.

The aggregate target under ``normalized=True`` divides the oriented effect
by the average cumulative, oriented dose change,

.. math::

   \delta_\ell^D
   =\frac1{N_\ell}\sum_{g\in\mathcal I_\ell}|A_{g,\ell}|,
   \qquad
   \delta_\ell^n=\frac{\delta_\ell}{\delta_\ell^D},

.. math::

   \delta_\ell^n
   =\sum_{g\in\mathcal I_\ell}
      \frac{|A_{g,\ell}|}{\sum_{h\in\mathcal I_\ell}|A_{h,\ell}|}
       \delta_{g,\ell}^n.

Groups with larger cumulative dose changes receive more weight in this ratio.
Equal coefficients on current and lagged treatment, combined with comparable
effect composition, can produce constant normalized estimates. Heterogeneous
effects and changing switcher composition can also affect their pattern.
``effects_equal=True`` tests equality of the reported horizon effects. Its
rejection alone cannot isolate which of these mechanisms differs across
horizons.

A total effect per dose and cost-benefit comparison
---------------------------------------------------

A policy evaluation may ask whether the benefits across several outcome
periods justify the doses administered. Summing outcome effects and dividing
by treatment administered answers a different question from normalizing each
horizon by all earlier exposure. We keep those denominators separate.

Under the paper's design restriction 3, :math:`D_{g,t}\geq b_g` for all
groups and periods. Over the eligible group-periods, define

.. math::

   B=\sum_{g:F_g\leq T_g}\sum_{\ell=1}^{T_g-F_g+1}
       (D_{g,\tau_{g,\ell}}-b_g),

.. math::

   \delta^{\mathrm{total}}
   =\frac{\sum_{g:F_g\leq T_g}
                  \sum_{\ell=1}^{T_g-F_g+1}\delta_{g,\ell}}{B}.

Provided :math:`B>0`, replacing each group effect with its DiD gives a
conditionally unbiased estimate under Assumptions 1 and 2. Each period's
additional treatment enters the denominator once, even though its effects
can enter the numerator through subsequent observed outcome periods as well.

Lemma 3 in the paper states that, under design restrictions 1 and 3,

.. math::

   \delta^{\mathrm{total}}
   =\sum_{\ell=1}^L\frac{N_\ell}{B}\delta_\ell,
   \qquad L=\max_{g:F_g\leq T_g}(T_g-F_g+1).

These nonnegative factors convert the horizon effects into a total effect
per additional treatment dose. Since they generally do not sum to one,
the expression is not a convex average of the horizon effects.

If the outcome is expressed in monetary units, doses have a linear cost
:math:`c_{g,\ell}\geq0`, and the discount factor is one, the paper's
cost-benefit criterion is

.. math::

   \delta^{\mathrm{total}}>
   \frac{\sum_{g,\ell}c_{g,\ell}
                     (D_{g,\tau_{g,\ell}}-b_g)}{B}.

The sums cover the same eligible group-periods as :math:`B`. For outcomes
that have not been converted to monetary units, an effect estimate alone
cannot establish whether the intervention's benefits exceed its monetary cost.

The API's ``ate`` field uses the requested effect horizons and their available
switchers. It divides summed oriented effects by summed oriented treatment
changes at each outcome period using the package's cell weights for both
sums. This remains a total-effect ratio when ``normalized=True``.
Before applying the paper's increasing-treatment cost-benefit interpretation,
separate directions to account for the sign reversal when pooling treatment
decreases.
With ``trends_lin=True``, the package does not compute ``ate``.

Checking the pre-switch comparison
----------------------------------

You cannot observe post-switch status-quo outcomes for switchers. Earlier
periods can nevertheless reveal discrepancies in the outcome changes used
by the design. We construct placebos with the same comparison groups and
interval length as the corresponding effect horizon.

For :math:`3\leq F_g\leq T_g` and
:math:`1\leq\ell\leq\min(T_g-F_g+1,F_g-2)`, the group placebo is

.. math::

   \begin{aligned}
   \operatorname{DID}_{g,\ell}^{\mathrm{pl}}
   ={}&Y_{g,F_g-1-\ell}-Y_{g,F_g-1}\\
   &-\frac1{|\mathcal C_{g,\ell}|}
     \sum_{g'\in\mathcal C_{g,\ell}}
       (Y_{g',F_g-1-\ell}-Y_{g',F_g-1}).
   \end{aligned}

The outcome difference runs backward from the pre-switch reference period,
as in a conventional event study. The controls still must remain unchanged
through :math:`F_g-1+\ell`, even though the placebo outcomes precede the
switch. Under Assumptions 1 and 2, its conditional expectation is zero.
The package aggregates these comparisons across eligible switchers and
reports a joint test when multiple placebos are available.

A nonzero placebo can reflect anticipation or a failure of status-quo parallel
trends. A noisy placebo near zero cannot verify either assumption after the
switch. ``placebo`` requests the number of horizons; the package caps that
request at the available pre-switch periods and at ``effects``. With binary
absorbing treatment, the first placebo uses the same two pre-switch periods
as an adjacent staggered comparison. Its outcome change runs in the opposite
direction to a varying-base pseudo effect. Longer placebo horizons use a
different reference-period construction from adjacent pre-period pseudo effects.

Why common regressions can change the target
--------------------------------------------

Adding lags to a regression does not by itself establish that its coefficients
estimate the path effects above. The regression can use treated groups as
controls or pool treatment histories that have different lagged effects.
We can see the weighting issue even when every adoption occurs together.

Suppose :math:`D_{g,t}=I_g\mathbf1\{t\geq F\}`, where doses :math:`I_g`
vary across groups that all start untreated. A fixed-effects comparison of
outcome changes at a common horizon on :math:`I_g` can weight each group's
per-dose effect by

.. math::

   w_g^{\mathrm{FE}}
   =\frac{I_g(I_g-\overline I)}
          {\sum_h(I_h-\overline I)^2}.

Although these weights sum to one under the relevant common-trend restriction,
a positive dose below the mean dose receives a negative weight. The coefficient
therefore need not give a nonnegative weighted average of the heterogeneous
per-dose effects.

A local projection of later observed outcomes on current treatment can also
combine the current assignment's effect with the effects of later assignments.
When later treatment depends on current treatment or intermediate outcomes,
the coefficient does not generally hold the future path fixed. Distributed-lag
regressions impose additional restrictions on lag length and effect
heterogeneity. Under heterogeneous effects, the coefficient on one lag can
also contain contributions from other lags. These concerns motivate estimating
explicit path contrasts rather than interpreting each regression coefficient
as a separate causal lag effect. The paper's comparisons in Sections 4 and 5
work through the corresponding designs.

Sampling uncertainty across groups
-----------------------------------

Multiple periods for one group do not provide independent repetitions of a
switching event. The paper's asymptotic argument lets the number of groups
grow while keeping the number of periods fixed. We state its conditions for
the equally weighted group-panel estimators above before connecting them to
the package's inference options.

Let :math:`\boldsymbol Y_g=(Y_{g,1},\ldots,Y_{g,T})^\top` and
:math:`\Sigma_g=\operatorname{Var}(\boldsymbol Y_g\mid\boldsymbol D)`.
The relevant horizons have growing numbers of eligible switchers,

.. math::

   \mathcal L
   =\{\ell:N_\ell\longrightarrow\infty\text{ almost surely as }G\to\infty\}.

.. admonition:: Assumption 4 Independent outcome vectors across groups
   :class: assumption

   Conditional on the infinite sequence of treatment paths, the outcome
   vectors :math:`(\boldsymbol Y_g)_{g\geq1}` are mutually independent.

Conditioning on treatment paths permits dependence between those paths
as well as serial dependence in outcomes within a group. Cross-group
outcome dependence requires an additional sampling and inference argument,
such as independent clusters containing several groups.

.. admonition:: Assumption 5 Asymptotic design support
   :class: assumption

   Almost surely, the number of relevant baseline treatment values stays
   bounded as :math:`G\to\infty`, and :math:`\mathcal L` is nonempty.
   For each :math:`\ell\in\mathcal L`, define

   .. math::

      v_{d,s,\ell}^G
      =\#\{g\leq G:b_g=d,S_g=s,F_g-1+\ell\leq T_g\}.

   Whenever :math:`v_{d,s,\ell}^G\to\infty` almost surely,

   .. math::

      \liminf_{G\to\infty}
      \frac{\#\{g\leq G:b_g=d,
            F_g=\max_{h\leq G:b_h=d}F_h\}}
           {v_{d,s,\ell}^G}>0
      \quad\text{almost surely}.

The last condition prevents an expanding switcher population from relying
on a vanishingly small set of last-switching controls. Merely having one
control in each finite sample does not supply this asymptotic support.

.. admonition:: Assumption 6 Moments and nondegenerate outcome variation
   :class: assumption

   For some :math:`\eta>0`, almost surely,

   .. math::

      \sup_{g\geq1,\ t\leq T}
      \mathbb E[|Y_{g,t}|^{2+\eta}\mid\boldsymbol D]<\infty,
      \qquad
      \inf_{g\geq1}\lambda_{\min}(\Sigma_g)>0.

Alongside the moment bound on unusually large outcome contributions, the
eigenvalue condition rules out zero variance for nontrivial linear combinations
of a group's observed outcomes.

The variance and normal approximation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each outcome can enter its group's switcher comparison and other groups'
control comparisons. To account for that reuse, write the horizon estimator
as a sum of group contributions. Define :math:`r_g=F_g-1` and

.. math::

   \begin{aligned}
   a_{g,t,\ell}={}&S_g\mathbf1\{g\in\mathcal I_\ell\}
      (\mathbf1\{t=\tau_{g,\ell}\}-\mathbf1\{t=r_g\})\\
   &-\sum_{h\in\mathcal I_\ell}
      \frac{S_h\mathbf1\{g\in\mathcal C_{h,\ell}\}}
           {|\mathcal C_{h,\ell}|}
      (\mathbf1\{t=\tau_{h,\ell}\}-\mathbf1\{t=r_h\}),
   \end{aligned}

.. math::

   U_{g,\ell}^G=\sum_{t=1}^T a_{g,t,\ell}Y_{g,t},
   \qquad
   \operatorname{DID}_\ell=\frac1{N_\ell}\sum_{g=1}^G U_{g,\ell}^G.

For never-switching groups, take the first term in :math:`a_{g,t,\ell}`
as zero. This is the paper's linear group-contribution representation,
expressed directly using the comparison sets. The true variance scale is

.. math::

   \sigma_{\ell,G}^2
   =\frac1{N_\ell}\sum_{g=1}^G
       \operatorname{Var}(U_{g,\ell}^G\mid\boldsymbol D).

Since the group means of these contributions need not be equal or consistently
estimable, the paper uses their average within the cohort defined by
:math:`(b_g,F_g,S_g)` to estimate each mean. Write that average as
:math:`\widehat\theta_{g,\ell}`. The variance estimator and pointwise
interval are

.. math::

   \widehat\sigma_\ell^2
   =\frac1{N_\ell}\sum_{g=1}^G
      (U_{g,\ell}^G-\widehat\theta_{g,\ell})^2,

.. math::

   \operatorname{CI}_{1-\alpha,\ell}
   =\left[\operatorname{DID}_\ell
           \pm z_{1-\alpha/2}\frac{\widehat\sigma_\ell}{\sqrt{N_\ell}}\right],
   \qquad 0<\alpha<1.

.. admonition:: Theorem 1 Consistency, normality, and interval coverage
   :class: theorem

   Under Assumptions 1, 2, 4, 5, and 6, for each
   :math:`\ell\in\mathcal L`, conditional on the sequence of treatment
   paths and almost surely,

   .. math::

      \operatorname{DID}_\ell-\delta_\ell\xrightarrow{p}0,
      \qquad
      \frac{\sqrt{N_\ell}(\operatorname{DID}_\ell-\delta_\ell)}
           {\sigma_{\ell,G}}\xrightarrow{d}\mathcal N(0,1),

   .. math::

      \liminf_{G\to\infty}
      \Pr(\delta_\ell\in\operatorname{CI}_{1-\alpha,\ell}
           \mid\boldsymbol D)\geq1-\alpha.

   Coverage approaches exactly :math:`1-\alpha` if, in place of
   Assumption 4, the pairs :math:`(\boldsymbol D_g,\boldsymbol Y_g)` are
   i.i.d. and :math:`\boldsymbol D_g` is a function of
   :math:`(b_g,F_g,S_g)`.

Conservative coverage accommodates differences in expected group contributions
within a cohort. The additional condition fixes the entire path from the cohort
information, as it does for binary treatment that changes at most once.
For normalized effects, the studentized limit and interval coverage follow
by dividing by a positive dose denominator fixed conditional on
:math:`\boldsymbol D`. Consistency of the normalized effect also requires
the unnormalized estimation error divided by its dose denominator to converge
to zero in probability.

Because these intervals cover each horizon separately, their confidence
level does not promise simultaneous coverage of the full event-study curve.
The paper's fixed-period, discrete-baseline argument also does not establish
inference for every extension that the API accepts.

What the package reports
~~~~~~~~~~~~~~~~~~~~~~~~

By default, ``ci_level=95.0`` produces pointwise analytical intervals with
variance estimated from group contributions and cohort demeaning to account
for controls reused across comparisons. The default switcher cohorts use
baseline treatment, first-switch date, and treatment at that switch.
``cluster`` collects groups into clusters and sums their contributions
within clusters before computing the variance. Each group must belong to a
single cluster; reliable inference requires a suitable number of independent
clusters.

``less_conservative_se=True`` demeans switcher changes among groups sharing
the whole observed path through the horizon and falls back to a coarser
cohort for a group alone on that path. This option changes the variance
calculation without changing the effect target or the placebo variance.

With ``boot=True``, each draw resamples clusters with replacement and reruns
the estimation. Without a clustering column, the draws resample groups.
``biters`` and ``random_state`` control the number and reproducibility of
draws. Bootstrap standard errors replace the analytical standard errors for
the effects, placebos, and ``ate``. The joint placebo test and equality-of-
effects test continue to use the analytical covariance matrix.

Covariates and other trend restrictions
---------------------------------------

Different baseline-treatment groups can also differ in observed determinants
of their trends. We can adjust outcome changes when a specified covariate
model explains those trend differences. The adjustment changes the identifying
assumption. Choose a covariate for its role in the counterfactual trend
rather than for improving an observed-outcome regression.

For time-varying covariates :math:`X_{g,t}`, the extension requires the
conditional mean of

.. math::

   Q_{g,t}-Q_{g,t-1}
   -(X_{g,t}-X_{g,t-1})^\top\theta_{b_g}

conditional on :math:`(\boldsymbol D,\boldsymbol X)` to be equal across
groups with the same relevant baseline treatment. ``xformla`` supplies those
covariates. The coefficients are estimated from outcome changes among groups
that have not yet switched and share that baseline. The estimator subsequently
uses outcome changes after subtracting the covariate component.

A time-invariant characteristic :math:`X_g` can explain different linear
trends by entering as :math:`X_{g,t}=tX_g`. Interactions with calendar
indicators can instead allow its trend coefficient to vary by period.
Covariates affected by the intervention require care because this adjustment
can remove part of the effect you intend to measure.

``trends_nonparam`` restricts comparisons to sets defined by time-invariant
columns, such as counties within the same state. ``trends_lin=True`` allows
group-specific linear trends by working with first-differenced outcomes and
cumulating effect estimates through each horizon on the corresponding
switchers. It requires additional pre-switch information and omits the total
effect ``ate``. Neither option repairs an invalid comparison without its
own trend assumption.

``predict_het`` regresses group effect comparisons on time-invariant
characteristics to examine treatment effect heterogeneity. It is unavailable
for normalized effects. These regressions do not identify a separate structural
lag effect from an aggregate horizon comparison.

For a continuously distributed baseline treatment, exact baseline matches may
not exist. A positive ``continuous`` value specifies a polynomial degree
for modeling counterfactual outcome changes as a function of baseline
treatment. The package then permits comparisons across baselines under that
functional-form restriction.

.. warning::

   The discrete-baseline asymptotic theorem above does not establish normal
   inference for the continuous-baseline extension. The API warns when
   ``continuous>0`` is used without ``boot=True``. Resampling provides an
   inference procedure without itself proving that the extension's
   approximation or identifying assumptions hold.

Finally, the paper discusses fuzzy designs where treatment varies within
group-period cells. Aggregating such data requires a treatment and weighting
definition that matches the desired dose effect. An average treatment rate is
not automatically interchangeable with a homogeneous group-level assignment.
The `paper and its web appendix <https://arxiv.org/abs/2007.04267>`_ give the
additional restrictions and derivations for these extensions.
