.. _background-didcont:

Difference-in-differences with a continuous treatment
=====================================================

Difference-in-differences with a continuous treatment studies effects when
units receive different amounts of treatment. You might want to know what a
particular dose did for the units that received it or what increasing the
dose would do for those same units. Answering the second question requires
a comparison that holds the population fixed as the dose changes.

Under ordinary parallel trends, comparing each dose group with untreated
units identifies the group's effect relative to no treatment. Because
comparing effects at different doses also changes the units being studied,
the curve's slope can reflect differences in how those units respond to
treatment as well as the causal effect of receiving more of it.

We develop that distinction from the two-period setup through estimation
and inference before extending the argument to staggered adoption. The
results follow Sections 3 and 4 and Appendix C of the December 31, 2025
version of `Callaway, Goodman-Bacon, and Sant'Anna
<https://psantanna.com/files/CGBS_v4.pdf>`_, forthcoming in the American
Economic Review. The formal statements retain the numbering of the paper
and its `supplementary appendix
<https://psantanna.com/files/CGBS_supp_v4.pdf>`_ to help you check the
assumptions behind :func:`~moderndid.cont_did`.

Defining the dose and the outcome paths
---------------------------------------

The two-period design lets us separate differences in doses from differences
in treatment timing. Consider :math:`n` units observed before treatment in period 1
and afterward in period 2. Write :math:`D_i` for unit :math:`i`'s dose
in period 2 and use zero for a unit that remains untreated.
The support :math:`\mathcal D` contains zero and positive doses
:math:`\mathcal D_+`. We suppress the unit index in population expressions
and write :math:`\Delta Y=Y_2-Y_1` for the observed outcome change.

.. admonition:: Assumption 1 Random sampling
   :class: assumption

   The observed vectors :math:`(Y_{i2},Y_{i1},D_i)`,
   :math:`i=1,\ldots,n`, are independent draws from a common population
   distribution. Dependence between the two outcomes of a unit is unrestricted.

.. admonition:: Assumption 2 Treatment
   :class: assumption

   Every unit is untreated in period 1. In period 2, the dose has support
   :math:`\mathcal D=\{0\}\cup\mathcal D_+`, where
   :math:`\mathcal D_+\subseteq(0,\infty)` and :math:`P(D=0)>0`.

This support condition supplies an observed untreated comparison group
without yet requiring positive doses to have a particular distribution.
The continuous-dose model can allow a gap between zero and the smallest
positive dose. Smoothing across that gap would therefore impose an
additional restriction on the treatment effect function.

Let :math:`Y_{it}(d)` denote the outcome unit :math:`i` would have
in period :math:`t` under a period-2 dose of :math:`d`.
These potential outcomes describe the treatment amount received by that
unit and presume that another unit's dose does not change its outcome.
All expectations used below are finite and well defined.

.. admonition:: Assumption 3 No anticipation and observed outcomes
   :class: assumption

   For every unit and every :math:`d\in\mathcal D`,

   .. math::

      Y_{i1}=Y_{i1}(d)=Y_{i1}(0),\qquad Y_{i2}=Y_{i2}(D_i).

Although every unit has an untreated baseline in the first period, the
second period reveals only the potential outcome at the dose it received.
Treated units therefore still lack the untreated post-treatment outcome,
just as in a binary treatment design.

Separating level effects from causal responses
-----------------------------------------------

An effect measured against no treatment answers a different question from
an effect of changing a positive dose. We need both definitions before
reading the slope of a fitted dose curve.

Whose level effect the curve describes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The level effect of dose :math:`d` is :math:`Y_2(d)-Y_2(0)`.
We can average it among units that received a particular dose :math:`d'`
or among all units that received some positive dose,

.. math::

   \begin{aligned}
   ATT(d\mid d')&=\mathbb E[Y_2(d)-Y_2(0)\mid D=d'],\\
   ATT(d)&=\mathbb E[Y_2(d)-Y_2(0)\mid D>0].
   \end{aligned}

The first argument of :math:`ATT(d\mid d')` tells us which counterfactual
dose to evaluate for the population selected by the second argument.
Along the diagonal curve :math:`ATT(d\mid d)`, increasing :math:`d`
changes both the treatment dose and the units being averaged. A higher
point can therefore reflect a different response to treatment or a
different set of units receiving that dose.

In :math:`ATT(d)`, the population stays fixed as we vary the dose.
It describes what dose :math:`d` would do for all treated units,
including those whose observed dose differs from :math:`d`.
This is the treated population rather than the entire population of
treated and untreated units.

For a continuous dose, conditioning on :math:`D=d` describes a conditional
mean function rather than a subgroup with positive probability. Estimating
that function from finitely many observations requires information from
nearby doses and suitable restrictions on its smoothness.

What a marginal increase would change
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A causal response holds the population fixed while changing its dose.
For continuous treatment, define the responses through derivatives of the
corresponding mean potential outcome functions,

.. math::

   \begin{aligned}
   ACRT(d\mid d')
      &=\left.\frac{\partial ATT(l\mid d')}{\partial l}\right|_{l=d},\\
   ACRT(d)&=\frac{\partial ATT(d)}{\partial d}.
   \end{aligned}

These definitions require the mean functions to be differentiable without
requiring every individual potential outcome path to be differentiable.
Interchanging differentiation and expectation would additionally require
conditions that justify that interchange.

For ordered discrete doses, set :math:`d_0=0`. The corresponding
responses between adjacent doses are

.. math::

   \begin{aligned}
   ACRT(d_j\mid d_k)
      &=\frac{\mathbb E[Y_2(d_j)-Y_2(d_{j-1})\mid D=d_k]}{d_j-d_{j-1}},\\
   ACRT(d_j)
      &=\frac{\mathbb E[Y_2(d_j)-Y_2(d_{j-1})\mid D>0]}{d_j-d_{j-1}}.
   \end{aligned}

The denominator expresses the response per unit of treatment even when
adjacent doses are far apart. With a binary dose coded as zero or one,
the response from zero to treatment also equals the level effect.
For several doses, an adjacent response and an effect relative to zero
generally answer different questions.

Averages over the treated dose distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your question concerns an overall effect, the treated dose distribution
gives a natural set of weights. The paper distinguishes summaries that
retain each unit's observed dose group from summaries of effects for the
whole treated population,

.. math::

   \begin{aligned}
   ATT^{loc}&=\mathbb E[ATT(D\mid D)\mid D>0],\\
   ATT^{glob}&=\mathbb E[ATT(D)\mid D>0],\\
   ACRT^{loc}&=\mathbb E[ACRT(D\mid D)\mid D>0],\\
   ACRT^{glob}&=\mathbb E[ACRT(D)\mid D>0].
   \end{aligned}

The local level summary averages the effects that treated units experience
at their own doses. The global level summary instead averages effects of
counterfactual doses for the same treated population, using its observed
dose distribution as weights. Equivalently, its dose draw is independent
of which treated unit's potential outcomes are being evaluated.
The two summaries can differ when units select doses according to their
treatment gains.

Identifying the level curve
---------------------------

The missing untreated outcome can be recovered if untreated mean changes
agree across dose groups. We first impose that familiar DiD condition.
It makes comparisons with dose zero interpretable without equating
treatment effects across positive-dose groups.

.. admonition:: Assumption PT Parallel trends
   :class: assumption

   For every :math:`d\in\mathcal D_+`,

   .. math::

      \mathbb E[Y_2(0)-Y_1(0)\mid D=d]
      =\mathbb E[Y_2(0)-Y_1(0)\mid D=0].

By restricting only untreated potential outcomes, this condition leaves
the response to a positive dose free to differ across units and dose groups.
Adding and subtracting the pre-treatment outcome gives

.. math::

   \begin{aligned}
   ATT(d\mid d)
      &=\mathbb E[Y_2(d)-Y_1(0)\mid D=d]\\
      &\quad-\mathbb E[Y_2(0)-Y_1(0)\mid D=d]\\
      &=\mathbb E[\Delta Y\mid D=d]-\mathbb E[\Delta Y\mid D=0].
   \end{aligned}

Since the mean change within dose group :math:`d` is observed, parallel
trends is needed to replace its missing untreated change with the observed
change in the zero-dose group.

.. admonition:: Theorem 3.1 Level effects under parallel trends
   :class: theorem

   Under Assumptions 1, 2, 3, and PT, every :math:`d\in\mathcal D_+`
   satisfies

   .. math::

      ATT(d\mid d)=\mathbb E[\Delta Y\mid D=d]-\mathbb E[\Delta Y\mid D=0].

   Averaging over positive doses also identifies the local summary,

   .. math::

      ATT^{loc}=\mathbb E[\Delta Y\mid D>0]-\mathbb E[\Delta Y\mid D=0].

A binary indicator for receiving any positive dose can therefore estimate
the overall local ATT directly without fitting a dose curve. The curve is
needed when you want to describe how level effects differ across the units
receiving different amounts of treatment. Neither result identifies
:math:`ATT(d)` for the whole treated population under PT alone.

Why the slope needs another assumption
---------------------------------------

A common interpretation of a dose regression is that its slope measures
what a small increase in treatment would do. That interpretation needs
care because the observed level curve changes the dose and the conditioning
group together. We can expose the difference by separating those two changes.

The next assumption specifies the dose distribution and the observed
conditional mean needed for the continuous and discrete slope arguments.
It is separate from Assumption 2's requirement for untreated observations.

.. admonition:: Assumption 4 Continuous or multivalued discrete treatment
   :class: assumption

   The dose distribution satisfies one of the following two treatment specifications.

   In case (a), :math:`\mathcal D_+=\mathcal D_+^c=(d_L,d_U)`.
   The conditional distribution of :math:`D` among treated units has a
   Lebesgue density :math:`f_+` satisfying :math:`f_+(d)>0` throughout
   :math:`\mathcal D_+^c`. The function
   :math:`\mathbb E[\Delta Y\mid D=d]` is differentiable on this interval.

   In case (b), :math:`\mathcal D_+=\mathcal D_+^{mv}\subseteq\mathbb N_+`,
   where :math:`\mathbb N_+=\{1,2,\ldots\}`. Write :math:`d_j` for the
   ordered positive dose indexed by :math:`j`. Every supported positive dose has
   positive probability, :math:`P(D=d)>0` for all
   :math:`d\in\mathcal D_+^{mv}`.

The continuous specification permits positive doses arbitrarily close to
zero as well as a positive lower endpoint. The paper uses integer doses
for its discrete specification; the normalization by adjacent dose gaps
still matters when some integers are absent from the support.

.. admonition:: Theorem 3.2 Selection in comparisons across doses
   :class: theorem

   Under Assumptions 1, 2, 3, and PT, every
   :math:`(h,l)\in\mathcal D_+\times\mathcal D_+` satisfies

   .. math::

      \begin{aligned}
      \mathbb E[\Delta Y\mid D=h]-\mathbb E[\Delta Y\mid D=l]
         &=ATT(h\mid h)-ATT(l\mid l)\\
         &=\mathbb E[Y_2(h)-Y_2(l)\mid D=h]\\
         &\quad+ATT(l\mid h)-ATT(l\mid l).
      \end{aligned}

   With Assumption 4(a), the continuous-dose decomposition is

   .. math::

      \begin{aligned}
      \frac{d\,\mathbb E[\Delta Y\mid D=d]}{dd}
         &=\frac{d\,ATT(d\mid d)}{dd}\\
         &=ACRT(d\mid d)
           +\left.\frac{\partial ATT(d\mid l)}{\partial l}\right|_{l=d},
      \end{aligned}

   for :math:`d\in\mathcal D_+^c` wherever the component derivatives exist.
   With Assumption 4(b), adjacent discrete comparisons instead give

   .. math::

      \begin{aligned}
      &\frac{\mathbb E[\Delta Y\mid D=d_j]
               -\mathbb E[\Delta Y\mid D=d_{j-1}]}{d_j-d_{j-1}}\\
      &\quad=ACRT(d_j\mid d_j)
        +\frac{ATT(d_{j-1}\mid d_j)
                    -ATT(d_{j-1}\mid d_{j-1})}{d_j-d_{j-1}}.
      \end{aligned}

The final term compares the effect of the same lower dose across two
groups whose gains are unrestricted by parallel trends. If units at higher
doses would benefit more even at the lower dose, the observed difference
includes those different gains. At the first positive discrete dose,
this term vanishes because a level effect at dose zero is zero.

For continuous treatment, the observed slope mixes selection into the
conditioning group with the response to a marginal intervention on dose.
More precise estimation of the left-hand side cannot distinguish those
two contributions without a restriction on treated potential outcomes.

.. admonition:: Read a reported slope conditionally
   :class: important

   The ``acrt_d`` field contains the derivative of the estimated level
   curve. Under ordinary parallel trends, that derivative combines a causal
   response and selection across dose groups. Its field name alone does
   not establish a causal interpretation for the estimated slope.

Identifying effects for a common treated population
----------------------------------------------------

If you want to compare doses for the same population, the restriction must
also connect treated potential outcomes across dose groups. Strong parallel
trends does this by asking each dose group to represent the mean outcome
change for all treated units under that dose.

.. admonition:: Assumption SPT Strong parallel trends
   :class: assumption

   For every :math:`d\in\mathcal D`, including zero,

   .. math::

      \mathbb E[Y_2(d)-Y_1(0)\mid D>0]
      =\mathbb E[Y_2(d)-Y_1(0)\mid D=d].

At positive doses, this condition equates the observed mean change in
dose group :math:`d` with the counterfactual mean change for all treated
units under dose :math:`d`. At zero, it connects the treated population's
missing untreated change to the observed change in the untreated group.
It places restrictions on positive-dose potential outcomes that PT leaves
unrestricted.

PT and SPT are non-nested because SPT does not require untreated parallel
trends separately for every positive-dose group. If PT is also maintained,
SPT is equivalent to :math:`ATT(d\mid d)=ATT(d)` at each dose.
That equality lets a dose group's effect at its own dose represent the
whole treated population. It does not require every off-diagonal effect
:math:`ATT(l\mid h)` to equal :math:`ATT(l\mid l)`.

.. admonition:: Theorem 3.3 Treated-population effects under strong parallel trends
   :class: theorem

   Under Assumptions 1, 2, 3, and SPT, every :math:`d\in\mathcal D_+`
   satisfies

   .. math::

      ATT(d)=\mathbb E[\Delta Y\mid D=d]-\mathbb E[\Delta Y\mid D=0].

   For every :math:`(h,l)\in\mathcal D_+\times\mathcal D_+`,

   .. math::

      \begin{aligned}
      ATT(h)-ATT(l)
         &=\mathbb E[Y_2(h)-Y_2(l)\mid D>0]\\
         &=\mathbb E[\Delta Y\mid D=h]-\mathbb E[\Delta Y\mid D=l].
      \end{aligned}

   With Assumption 4(a), every :math:`d\in\mathcal D_+^c` satisfies

   .. math::

      ACRT(d)=\frac{d\,\mathbb E[\Delta Y\mid D=d]}{dd}.

   With Assumption 4(b), the discrete response is

   .. math::

      ACRT(d_j)=\frac{\mathbb E[\Delta Y\mid D=d_j]
                           -\mathbb E[\Delta Y\mid D=d_{j-1}]}{d_j-d_{j-1}}.

The comparison has the same observed-data form as it did under PT.
Its interpretation changes because the restriction on treated potential
outcomes changes whose counterfactual mean the dose group represents. In particular, the identified
slope is now :math:`ACRT(d)` for all treated units. It need not equal
:math:`ACRT(d\mid d)` without an additional restriction on selection.

.. admonition:: Corollary 3.1 Global summaries
   :class: theorem

   Under Assumptions 1, 2, 3, and SPT, the global level summary is

   .. math::

      ATT^{glob}=\mathbb E[\Delta Y\mid D>0]-\mathbb E[\Delta Y\mid D=0].

   With Assumption 4(a), its global response counterpart is

   .. math::

      ACRT^{glob}=\int_{d_L}^{d_U}
         \frac{d\,\mathbb E[\Delta Y\mid D=d]}{dd}f_+(d)\,dd.

   With Assumption 4(b), the response summary instead becomes

   .. math::

      ACRT^{glob}=\sum_{j=1}^J
         \frac{\mathbb E[\Delta Y\mid D=d_j]
                    -\mathbb E[\Delta Y\mid D=d_{j-1}]}{d_j-d_{j-1}}
         P(D=d_j\mid D>0).

The local and global level summaries thus have the same identification
formula under their respective assumptions. For responses, estimating an
average observed slope under PT alone still leaves an average selection
component. Averaging across doses cannot remove the identification problem
from Theorem 3.2.

What a linear dose regression averages
---------------------------------------

A linear regression on dose can produce a summary whose weights differ
from the effect you want to report even with only two periods. We can
examine this problem before adding staggered timing through the same
conditional change function :math:`m(d)=\mathbb E[\Delta Y\mid D=d]`.

Write :math:`\mu_D=\mathbb E[D]` and
:math:`v_D=\operatorname{Var}(D)>0`, assuming the required moments and
integrals exist. The two-period TWFE coefficient equals the slope in
a regression of outcome changes on dose,

.. math::

   \beta^{twfe}
   =\frac{\operatorname{Cov}(D,\Delta Y)}{v_D}
   =\frac{\mathbb E[(D-\mu_D)m(D)]}{v_D}.

Let :math:`f_D(d)=P(D>0)f_+(d)` denote the unconditional density
on the positive-dose interval and write :math:`p_0=P(D=0)`.
The paper's Table 1 defines the weights used in the four decompositions,

.. math::

   \begin{aligned}
   w_1^{acrt}(d)&=\frac{\mathbb E[(D-\mu_D)\mathbf1\{D\geq d\}]}{v_D},\\
   w_0^{acrt}&=\frac{d_L\mu_Dp_0}{v_D},\\
   w_1^{lev}(d)&=\frac{(d-\mu_D)f_D(d)}{v_D},
      &w_0^{lev}&=-\frac{\mu_Dp_0}{v_D},\\
   w^s(d)&=d\,w_1^{lev}(d),\\
   w_1^{2\times2}(l,h)&=\frac{(h-l)^2f_D(h)f_D(l)}{v_D},\\
   w_0^{2\times2}(d)&=\frac{d^2f_D(d)p_0}{v_D}.
   \end{aligned}

The endpoint term uses a finite one-sided limit

.. math::

   a_L=\lim_{d\downarrow d_L}\{m(d)-m(0)\}.

The proof also uses the integral representation
:math:`m(d)-m(0)=a_L+\int_{d_L}^d m'(s)\,ds` whenever the required
endpoint limits and derivative integrals exist. For a positive
:math:`d_L`, the endpoint contribution bridges the gap from zero to
the smallest positive dose. Its canceled form is
:math:`\mu_Dp_0a_L/v_D`. This form also specifies the limiting contribution when
:math:`d_L=0`. That contribution vanishes at zero only when
:math:`m(0+)=m(0)`; starting the positive-dose interval at zero does
not by itself impose continuity with the zero-dose group.

.. admonition:: Theorem 3.4 Four decompositions of the TWFE coefficient
   :class: theorem

   Under Assumptions 1, 2, 3, 4(a), and PT, the causal-response
   decomposition is

   .. math::

      \begin{aligned}
      \beta^{twfe}
         &=\int_{d_L}^{d_U}w_1^{acrt}(d)
           \left(ACRT(d\mid d)
             +\left.\frac{\partial ATT(d\mid l)}{\partial l}\right|_{l=d}\right)\,dd\\
         &\quad+w_0^{acrt}\frac{ATT(d_L\mid d_L)}{d_L}.
      \end{aligned}

   The weights are nonnegative and satisfy
   :math:`\int w_1^{acrt}(d)\,dd+w_0^{acrt}=1`.
   The level decomposition is

   .. math::

      \beta^{twfe}=\int_{d_L}^{d_U}w_1^{lev}(d)ATT(d\mid d)\,dd.

   Here :math:`w_1^{lev}(d)` is negative below :math:`\mu_D` and
   positive above it. The signed weights satisfy
   :math:`\int w_1^{lev}(d)\,dd+w_0^{lev}=0`.
   The scaled-level decomposition is

   .. math::

      \beta^{twfe}=\int_{d_L}^{d_U}w^s(d)\frac{ATT(d\mid d)}{d}\,dd.

   The scaled-level weights have the same sign pattern as the level
   weights and satisfy :math:`\int w^s(d)\,dd=1`.
   Finally, the scaled high-versus-low decomposition is

   .. math::

      \begin{aligned}
      \beta^{twfe}
         &=\int_{d_L}^{d_U}\int_l^{d_U}w_1^{2\times2}(l,h)\\
         &\quad\times\left(
            \frac{\mathbb E[Y_2(h)-Y_2(l)\mid D=h]}{h-l}
            +\frac{ATT(l\mid h)-ATT(l\mid l)}{h-l}\right)\,dh\,dl\\
         &\quad+\int_{d_L}^{d_U}w_0^{2\times2}(d)
                           \frac{ATT(d\mid d)}{d}\,dd.
      \end{aligned}

   The last pair of weights is nonnegative and has total integral one.
   Under SPT in place of PT, remove the selection terms, replace
   :math:`ATT(d\mid d)` with :math:`ATT(d)` and
   :math:`ACRT(d\mid d)` with :math:`ACRT(d)`, and condition the
   high-versus-low potential outcome contrast on :math:`D>0`.

The level weights sum to zero when their zero-dose atom is included.
Since the level effect at zero is zero, that atom drops out of the
coefficient's formula. It still determines the signed measure's total
mass. These signed weights therefore do not define a convex average of
level effects. Scaling level effects by dose changes the target and
normalizes the weights, although negative weights can remain wherever
positive doses lie below the unconditional mean dose.

Under PT, the slope representation assigns nonnegative weights to both
causal responses and selection terms. Under SPT,
the causal interpretation improves without making the weights equal to
:math:`f_+`. Their dependence on the entire dose distribution also makes
the coefficient sensitive to the proportion of untreated observations.
Choosing a density-weighted summary directly makes the target explicit
before estimation.

What weaker restrictions can support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Strong parallel trends may be implausible in an application. Section 5.1
and Appendix SE of the supplement describe restrictions that answer
narrower questions. For example, suppose that higher-dose groups would
experience at least as large an effect of every counterfactual dose,

.. math::

   ATT(d\mid l)\leq ATT(d\mid h),\qquad l<h.

Under PT and the differentiability conditions, Theorem 3.2 then makes
:math:`m'(d)` an upper bound on :math:`ACRT(d\mid d)`.
Reversing the restriction reverses the direction of the bound.

A restriction over a subset :math:`\mathcal D_s\subseteq\mathcal D_+`
can instead support causal comparisons only among doses in that subset,

.. math::

   \mathbb E[Y_2(d)-Y_1(0)\mid D\in\mathcal D_s]
   =\mathbb E[Y_2(d)-Y_1(0)\mid D=d],\qquad d\in\mathcal D_s.

For :math:`h,l\in\mathcal D_s`, this identifies
:math:`\mathbb E[Y_2(h)-Y_2(l)\mid D\in\mathcal D_s]`
from :math:`m(h)-m(l)`. It does not establish a level effect relative
to zero unless another restriction links that subset to an untreated
reference. A covariate-conditional version of SPT likewise changes the
population to the treated units with a given covariate value and needs
support for the corresponding conditional dose comparisons.

When no untreated units exist, Remark 3.1 and Appendix SD.1 show what
remains possible. In this setting, PT must be modified to require the
same untreated mean change across the positive-dose groups rather than
comparison with an absent zero-dose group. Comparing a dose with a
lowest-dose reference then identifies the difference between their
diagonal effects,

.. math::

   m(d)-m(d_L)=ATT(d\mid d)-ATT(d_L\mid d_L).

Under the corresponding SPT restriction within the positive-dose population,
it instead identifies :math:`ATT(d)-ATT(d_L)` for the common treated population.
Neither contrast supplies an absolute effect relative to zero without
additional information. ModernDiD's baseline continuous-treatment
estimator uses untreated comparison observations rather than implementing
these alternative identification strategies.

Fitting the curve in ModernDiD
-------------------------------

After choosing an identifying comparison, you still need a way to
estimate its conditional mean from finitely many observations at each dose.
Section 4 of `Callaway, Goodman-Bacon, and Sant'Anna (2025)
<https://psantanna.com/files/CGBS_v4.pdf>`_ approaches this as a regression
problem. We distinguish a spline specification you choose in advance from
a growing spline space whose dimension is selected from the data.

Write :math:`m(d)=\mathbb E[\Delta Y\mid D=d]`,
:math:`m_0=\mathbb E[\Delta Y\mid D=0]`, and :math:`h(d)=m(d)-m_0`.
Under parallel trends, :math:`h(d)=ATT(d\mid d)` describes the effect
for the units at dose :math:`d`. Strong parallel trends instead gives
:math:`h(d)=ATT(d)` and :math:`h'(d)=ACRT(d)` for the whole treated
population. The same numerical regression therefore supports different
interpretations depending on the assumption you maintain. Reading its
derivative as a causal response requires that assumption to remove the
selection term discussed above.

For ordered discrete doses :math:`0=d_0<d_1<\cdots<d_J`, the paper's
saturated regression estimates a separate comparison with zero at every
positive dose,

.. math::

   \Delta Y_i=\alpha+
      \sum_{j=1}^J\mathbf1\{D_i=d_j\}\beta_j+\varepsilon_i.

Under strong parallel trends, :math:`\widehat\beta_j` estimates
:math:`ATT(d_j)`. Set :math:`\widehat\beta_0=0` for the zero-dose
effect. This reference is distinct from the regression intercept
:math:`\alpha`. The corresponding scaled response is

.. math::

   \widehat{ACRT}(d_j)
      =\frac{\widehat\beta_j-\widehat\beta_{j-1}}{d_j-d_{j-1}}.

Dividing by the dose gap expresses the contrast per unit of treatment.
This is the paper's discrete-dose procedure. The current
:func:`~moderndid.cont_did` API instead accepts only
``treatment_type="continuous"``.

A spline for positive doses
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a continuous dose, splines let nearby observations inform a curve
without requiring a separate mean at every exact dose. Let :math:`n` be
the number of sampled units, :math:`n_+=\sum_i\mathbf1\{D_i>0\}`, and
:math:`n_0=\sum_i\mathbf1\{D_i=0\}`. Estimate the untreated mean by

.. math::

   \widehat m_0=\frac1{n_0}\sum_{i:D_i=0}\Delta Y_i.

Let :math:`\psi^K(d)` contain :math:`K` spline basis functions spanning
a constant. Write :math:`\partial\psi^K(d)` for the vector of their
first derivatives at dose :math:`d`. Least squares on the positive-dose sample gives

.. math::

   \widehat\beta_K
      =\left[\sum_{i:D_i>0}\psi^K(D_i)\psi^K(D_i)'\right]^{-}
       \sum_{i:D_i>0}\psi^K(D_i)(\Delta Y_i-\widehat m_0),

where :math:`{}^{-}` denotes the Moore-Penrose inverse.
The level curve and its derivative are then

.. math::

   \widehat h_K(d)=\psi^K(d)'\widehat\beta_K,
   \qquad
   \widehat h_K'(d)=\partial\psi^K(d)'\widehat\beta_K.

The default ``dose_est_method="parametric"`` uses ``degree=3`` and
``num_knots=0`` to fit a cubic polynomial on positive doses.
Interior knots allow its shape to vary across parts of the dose support.
For example, a correctly specified quadratic fit with ``degree=2`` and
``num_knots=0`` gives

.. math::

   \widehat h(d)=\widehat\beta_0+\widehat\beta_1d+\widehat\beta_2d^2,
   \qquad
   \widehat h'(d)=\widehat\beta_1+2\widehat\beta_2d.

A fixed spline space consistently estimates its population projection
under independent sampling, finite second moments, positive limiting
group shares, and a nonsingular population Gram matrix. Recovering the
unrestricted dose curve additionally requires that the true function
lies in that space or that the sieve dimension grows with the sample
under suitable approximation and variance conditions. Increasing the
number of evaluation points in ``dvals`` leaves the fitted spline space
unchanged.

The estimated untreated mean shifts the entire level curve together.
Since differentiating a constant gives zero, its estimation error affects
the level curve's uncertainty but disappears from the fitted derivative.
To see how the package accounts for this distinction, define

.. math::

   \widehat G_K=\frac1{n_+}\sum_{i:D_i>0}
       \psi^K(D_i)\psi^K(D_i)',
   \qquad
   \widehat u_{i,K}=\Delta Y_i-\widehat m_0-\widehat h_K(D_i).

For a fixed fitted basis, the estimated influence functions at a dose
:math:`d` are

.. math::

   \begin{aligned}
   \widehat\phi_{K,h}(W_i,d)
      &=\frac{\mathbf1\{D_i>0\}}{\widehat p_+}
        \psi^K(d)'\widehat G_K^{-}\psi^K(D_i)\widehat u_{i,K}
        -\frac{\mathbf1\{D_i=0\}}{\widehat p_0}
         (\Delta Y_i-\widehat m_0),\\
   \widehat\phi_{K,h'}(W_i,d)
      &=\frac{\mathbf1\{D_i>0\}}{\widehat p_+}
        \partial\psi^K(d)'\widehat G_K^{-}
        \psi^K(D_i)\widehat u_{i,K},
   \end{aligned}

where :math:`W_i=(Y_{i2},Y_{i1},D_i)`,
:math:`\widehat p_+=n_+/n`, and :math:`\widehat p_0=n_0/n`.
These expressions describe coefficient estimation and comparison-mean
uncertainty. They do not correct a misspecified fixed spline or establish
that approximation bias is negligible for inference on the unrestricted
curve.

Choosing the sieve from the data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A flexible curve needs enough basis functions to capture its shape
without letting sampling noise dominate the fit. Appendix B of the
paper adapts the data-driven procedure of `Chen, Christensen, and
Kankanala <https://arxiv.org/abs/2107.11869>`_ to this regression problem.
The procedure compares smaller spline fits with larger ones and keeps
the smallest fit whose differences remain below a bootstrap threshold.
Using the same multiplier for an observation across all fits preserves
their sampling covariance in those comparisons.

For cubic splines, the paper starts from the dyadic dimensions

.. math::

   \mathcal K=\{2^k+3:k=0,1,\ldots\},
   \qquad K^+=\min\{j\in\mathcal K:j>K\},
   \qquad v_n=\max\{1,(0.1\log n)^4\}.

Its feasible search set is

.. math::

   \widehat{\mathcal K}
      =\left\{K\in\mathcal K:
          0.1(\log\widehat K_{\max})^2\leq K\leq\widehat K_{\max}
        \right\},

where the upper dimension is determined by

.. math::

   \widehat K_{\max}
      =\min\left\{K\in\mathcal K:
        K\sqrt{\log K}\,v_n\leq10\sqrt n
        <K^+\sqrt{\log K^+}\,v_n
       \right\}.

The notation here follows Appendix B, whose :math:`n` counts the full
sample. Its regression coefficient still uses only positive-dose units.
Writing :math:`\mathbb E_n` for a full-sample average, define the
coefficient influence estimate by

.. math::

   \widehat\varphi_K(W_i)
      =\left[\mathbb E_n\{
         \mathbf1\{D>0\}\psi^K(D)\psi^K(D)'
        \}\right]^{-1}
        \mathbf1\{D_i>0\}\psi^K(D_i)\widehat u_{i,K}.

Appendix B uses its level and derivative projections,

.. math::

   \widehat\phi_K(W_i,d)=\psi^K(d)'\widehat\varphi_K(W_i),
   \qquad
   \widehat\phi_K^{acrt}(W_i,d)
      =\partial\psi^K(d)'\widehat\varphi_K(W_i).

These are the regression contributions from treated observations.
The appendix's nonparametric approximation omits the untreated-mean
contribution because its root-sample-size error is smaller than the
slower convergence rate of the growing-sieve curve.

For each pair :math:`K_2>K`, estimate the contrast variance by

.. math::

   \widehat\sigma_{K,K_2}^{\,2}(d)
      =\frac1n\sum_{i=1}^n
        \{\widehat\phi_K(W_i,d)-\widehat\phi_{K_2}(W_i,d)\}^2.

Independent standard normal multipliers :math:`\omega_i` generate the
studentized bootstrap contrast

.. math::

   Z_n^*(d,K,K_2)
      =\frac{n^{-1/2}\sum_{i=1}^n
          \{\widehat\phi_K(W_i,d)-\widehat\phi_{K_2}(W_i,d)\}\omega_i}
        {\widehat\sigma_{K,K_2}(d)}.

Let :math:`\widehat\alpha=\min\{0.5,
\sqrt{\log\widehat K_{\max}/\widehat K_{\max}}\}`.
The selection threshold :math:`\gamma_{1-\widehat\alpha}^*` is the
bootstrap :math:`1-\widehat\alpha` quantile of

.. math::

   \sup_{\substack{d\in\mathcal D_+^c,\ K,K_2\in\widehat{\mathcal K}\\
                    K_2>K}}
      |Z_n^*(d,K,K_2)|.

The selected dimension is

.. math::

   \widehat K=\inf\left\{K\in\widehat{\mathcal K}:
      \sup_{\substack{d\in\mathcal D_+^c,\ K_2\in\widehat{\mathcal K}\\
                      K_2>K}}
      \frac{\sqrt n\,|\widehat h_K(d)-\widehat h_{K_2}(d)|}
           {\widehat\sigma_{K,K_2}(d)}
      \leq1.1\gamma_{1-\widehat\alpha}^*
      \right\}.

This rule avoids asking you to supply the function's smoothness before
choosing the spline dimension. The resulting statistical guarantees
still require the error-moment, design, approximation, and smoothness
conditions of the CCK model. The :ref:`nonparametric IV background
<background-npiv>` states those conditions and its adaptive-rate results.
Passing dose as both the regressor and its own instrument reduces the
procedure to ordinary nonparametric regression.

For scalar regression and a smoothness class of order :math:`p`, the
corresponding sup-norm rates are

.. math::

   \|\widehat h-h\|_\infty
      =O_P\!\left((\log n_+/n_+)^{p/(2p+1)}\right),
   \qquad
   \|\widehat h'-h'\|_\infty
      =O_P\!\left((\log n_+/n_+)^{(p-1)/(2p+1)}\right).

The derivative rate requires :math:`p>1` and sufficient spline order.
A cubic spline does not give the stated adaptivity over arbitrarily
smooth classes. These rates concern the growing-sieve procedure under
the cited conditions rather than every numerical fit returned by the
package.

Bands over the dose curve
-------------------------

Looking across an entire fitted curve requires a band that accounts for
searching across doses. The adaptive procedure also allows for selecting
the spline dimension and its remaining approximation bias. Appendix B
uses a second bootstrap calibration for this purpose. Write
:math:`\alpha\in(0,1)` for the desired noncoverage probability,
corresponding to ``alp`` in the API. The nominal coverage of the band
is therefore :math:`1-\alpha`.

Define the level and derivative variance estimates by

.. math::

   \widehat\sigma_K^2(d)=\frac1n\sum_{i=1}^n
       \widehat\phi_K(W_i,d)^2,
   \qquad
   \widehat\sigma_K^{acrt,2}(d)=\frac1n\sum_{i=1}^n
       \widehat\phi_K^{acrt}(W_i,d)^2.

Their studentized bootstrap processes are

.. math::

   Z_n^*(d,K)
      =\frac{n^{-1/2}\sum_i\widehat\phi_K(W_i,d)\omega_i}
             {\widehat\sigma_K(d)},
   \qquad
   Z_n^{*,acrt}(d,K)
      =\frac{n^{-1/2}\sum_i\widehat\phi_K^{acrt}(W_i,d)\omega_i}
             {\widehat\sigma_K^{acrt}(d)}.

Let :math:`\widehat{\mathcal K}_-=
\{K\in\widehat{\mathcal K}:K<\widehat K\}`.
The appendix defines :math:`z_{1-\alpha}^*` and
:math:`z_{1-\alpha}^{*,acrt}` as the bootstrap :math:`1-\alpha`
quantiles of

.. math::

   \sup_{d\in\mathcal D_+^c,\ K\in\widehat{\mathcal K}_-}
      |Z_n^*(d,K)|,
   \qquad
   \sup_{d\in\mathcal D_+^c,\ K\in\widehat{\mathcal K}_-}
      |Z_n^{*,acrt}(d,K)|.

Using :math:`\widehat A=\log\log\widehat K`, the paper's bands are

.. math::

   C_n(d)=\left[\widehat h_{\widehat K}(d)
      \ \pm\
      \left(z_{1-\alpha}^*+
            \widehat A\gamma_{1-\widehat\alpha}^*\right)
      \frac{\widehat\sigma_{\widehat K}(d)}{\sqrt n}\right],

and

.. math::

   C_n^{acrt}(d)=\left[\widehat h_{\widehat K}'(d)
      \ \pm\
      \left(z_{1-\alpha}^{*,acrt}+
            \widehat A\gamma_{1-\widehat\alpha}^*\right)
      \frac{\widehat\sigma_{\widehat K}^{acrt}(d)}{\sqrt n}\right].

The added selection threshold allows for approximation bias at the
chosen dimension. Honest adaptive coverage means that asymptotic
coverage holds uniformly over the specified function class. CCK's
coverage classes impose restrictions on approximation bias through
self-similarity conditions rather than allowing every smooth function.
The :ref:`NPIV background <background-npiv>` gives their formal coverage
and width results. Parallel trends alone does not provide those
statistical guarantees.

What the CCK option implements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ModernDiD's ``dose_est_method="cck"`` requires two periods and one
treated timing cohort. It sends the centered positive-dose outcomes to
:func:`~moderndid.npiv` with dose as both regressor and instrument, cubic
splines at quantile knots, and 999 Gaussian multiplier draws internally.
The inner calculation therefore uses :math:`n_+` observations rather
than Appendix B's full-sample normalization. The wrapper uses
``random_state`` for its seed but fixes those inner draws independently
of the ``biters`` argument.

The numerical routines express their grid through counts of spline
segments. A cubic basis with :math:`s` segments has :math:`K=s+3`
basis functions. The segment counts below therefore refer to the same
spline spaces described by the basis dimensions above.

The numerical selection retains a finite dyadic set through its upper
cutoff without enforcing Appendix B's log-squared lower cutoff.
It truncates the selected dimension at the second-largest candidate.
For band calibration, it includes candidates through the larger of the
selected segment count and the third-largest count when the grid has
more than two entries. This differs from the appendix's strict set
:math:`K<\widehat K`. The implemented inflation is
:math:`\max\{0,\log\log\widetilde K\}\,\gamma^*` for the truncated
dimension :math:`\widetilde K`. If adaptive selection fails, the generic
estimator warns and returns a fallback fit rather than the selected
sieve's curve.

The wrapper also retains untreated-mean uncertainty in its level
standard errors,

.. math::

   \widehat{se}_{ATT}(d)
      =\sqrt{\widehat{se}_{npiv}(d)^2+
          \frac1{n_0^2}\sum_{i:D_i=0}(\Delta Y_i-\widehat m_0)^2}.

With ``cband=True``, it multiplies this standard error by the larger
of the inner level critical value and the normal pointwise critical
value. The derivative uses its corresponding inner critical value and
standard error. With ``cband=False``, both reported curves use the
normal pointwise critical value. This is a numerical adaptation of the
paper's construction rather than an exact implementation of every
candidate-set and variance calculation in Appendix B.

.. admonition:: Choose the doses your band covers
   :class: important

   The package takes bootstrap maxima over the finite ``dvals`` grid,
   which defaults to 50 doses. The paper's bands use a supremum over
   the continuous positive-dose support. A denser grid can better
   represent that region without establishing coverage between its
   evaluation points from grid coverage alone.

Summarizing the fitted effects
--------------------------------

An overall level effect and an overall causal response average different
quantities. The first can be estimated from mean outcome changes
without fitting the dose curve. The second requires an estimate of
how that curve changes with dose before averaging over treated units.

Section 4.2 estimates the binarized level comparison directly,

.. math::

   \widehat\theta_{bin}
      =\frac1{n_+}\sum_{i:D_i>0}\Delta Y_i
       -\frac1{n_0}\sum_{i:D_i=0}\Delta Y_i.

Under parallel trends, it estimates :math:`ATT^{loc}`, the average
effect of the doses units actually received. Under strong parallel
trends, Corollary 3.1 also identifies it with :math:`ATT^{glob}`.
The package's ``overall_att`` uses this binary comparison rather
than integrating ``att_d`` over the plotted dose grid.

For the response summary, the paper's plug-in estimator is

.. math::

   \widehat\theta=\widehat{ACRT}^{glob}
      =\frac1{n_+}\sum_{i:D_i>0}\widehat h_K'(D_i).

The observed positive-dose units determine the averaging weights
rather than equal spacing in ``dvals``. This is how the
two-period ``overall_acrt`` is computed for both the fixed-spline and
CCK paths. Under strong parallel trends and appropriate approximation
conditions, it targets :math:`ACRT^{glob}`. Under ordinary parallel
trends, averaging the derivative retains the selection term and does
not in general identify that causal response.

For a discrete dose, the analogous estimator weights the scaled
adjacent-dose contrasts by their treated-sample frequencies,

.. math::

   \widehat{ACRT}^{glob}
      =\sum_{j=1}^J
       \frac{\widehat\beta_j-\widehat\beta_{j-1}}{d_j-d_{j-1}}
       \frac{\sum_i\mathbf1\{D_i=d_j\}}{n_+}.

The continuous-dose package instead differentiates the fitted spline.
Its response-summary variance allows both the fitted coefficients and
the empirical dose distribution to vary across samples. Define

.. math::

   \widehat a_K=\frac1{n_+}\sum_{i:D_i>0}\partial\psi^K(D_i).

The treated-sample influence estimate used for the average derivative is

.. math::

   \widehat\eta_i
      =\widehat h_K'(D_i)-\widehat\theta
       +\widehat a_K'\widehat G_K^{-}
        \psi^K(D_i)\widehat u_{i,K},
   \qquad D_i>0.

The influence estimate combines variation in fitted derivatives across
sampled doses with coefficient estimation through the residual term. Neither contains an untreated-mean term because that
constant disappears from the derivative. In the two-period CCK path,
the reported standard error is

.. math::

   \widehat{se}_{overall\_acrt}
      =\sqrt{\frac1{n_+^2}
          \sum_{i:D_i>0}(\widehat\eta_i-\overline{\widehat\eta})^2},
   \qquad
   \overline{\widehat\eta}=\frac1{n_+}\sum_{i:D_i>0}\widehat\eta_i.

The fixed-spline path embeds this contribution in the full unit
influence function and computes its standard error by multiplier
bootstrap. Staggered-adoption summaries also account for estimated
cohort shares, as described below. These calculations use independent
units; the current continuous-treatment API does not provide cluster
inference, covariate adjustment, or sampling weights.

A scalar average derivative can have root-sample-size inference even
though estimating the entire derivative curve is slower.
Section 4.2 points to regularity conditions for series functionals
rather than stating a new theorem with complete conditions.
Appropriate smoothness, approximation bias, and variance conditions
remain necessary. A reported standard error alone does not establish
consistency or efficiency for an unrestricted response function.

The paper's orthogonal average-response moment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The paper also describes a route to estimating the average response
while reducing sensitivity to errors in nuisance estimates. Let
:math:`f_+(d)=f_{D\mid D>0}(d)` be the treated dose density and
:math:`s_+(d)=f_+'(d)/f_+(d)` its density score.
Equation (4.8) gives the representation

.. math::

   ACRT^{glob}
      =\mathbb E\!\left[
         m'(D)-\{\Delta Y-m(D)\}s_+(D)
         \,\middle|\,D>0\right].

At the true conditional mean, the residual term has mean zero given
dose. Its role in the orthogonal score is to offset the first-order
effect of estimating that mean. For a perturbation :math:`v` of
:math:`m` on :math:`[d_L,d_U]`, integration by parts gives

.. math::

   \mathbb E[v'(D)+v(D)s_+(D)\mid D>0]
      =[v(d)f_+(d)]_{d_L}^{d_U}.

The mean perturbation therefore cancels when the boundary product
vanishes and the derivatives are integrable. Perturbing the density
score has zero first-order effect because
:math:`\mathbb E[\Delta Y-m(D)\mid D]=0` on positive-dose support.
An orthogonality or efficiency argument must maintain the relevant
density, differentiability, boundary, and moment conditions rather
than infer them from parallel trends.

ModernDiD's ``overall_acrt`` uses the series plug-in calculation above.
It does not estimate :math:`s_+`, fit an orthogonal score separately,
or cross-fit nuisance functions. The paper identifies the orthogonal
representation as a possible basis for flexible nuisance estimation
and leaves development of a double machine learning procedure for
future research. Equation (4.8) thus explains a distinct estimation
approach without supplying an additional method in the current API.

Adding staggered adoption
---------------------------

When treatment starts at different dates, timing and dose must both
describe the counterfactual path. Let :math:`G_i` be the first treatment
period and let :math:`G_i=\infty` identify a never-treated unit.
The paper uses this infinite timing value; ModernDiD records never-treated
units with zero in the column specified by ``gname``. A unit's dose
:math:`D_i` stays fixed after
adoption. Its actual treatment amount in period :math:`t` is therefore

.. math::

   W_{it}=D_i\mathbf1\{t\geq G_i\}.

Write :math:`Y_{it}(g,d)` for the potential outcome under adoption time
:math:`g` and dose :math:`d`. Set
:math:`Y_{it}(0)=Y_{it}(\infty,0)` for the untreated path.
Let :math:`\mathcal G\subseteq\{2,\ldots,T,\infty\}` be the
timing support and :math:`\overline{\mathcal G}` its finite adoption dates.
We first follow the local-effect extension in Appendix C of the main paper.

.. admonition:: Assumptions 1-MP through 3-MP Panel treatment paths
   :class: assumption

   Assumption 1-MP requires independent draws of the observed vectors
   :math:`(Y_{i1},\ldots,Y_{iT},D_i,G_i)` from a common distribution.

   Assumption 2-MP(a) requires
   :math:`\mathcal D=\{0\}\cup\mathcal D_+` with
   :math:`\mathcal D_+\subseteq(0,\infty)` and :math:`P(D=0)>0`.
   Each :math:`(g,d)\in\overline{\mathcal G}\times\mathcal D_+`
   has positive conditional dose density or mass,
   :math:`dF_{D\mid G}(d\mid g)>0`.

   Assumption 2-MP(b) requires a continuous interval
   :math:`\mathcal D_+^c=(d_L,d_U)\subseteq\mathcal D_+` on which
   :math:`\mathbb E[Y_t-Y_{t-1}\mid G=g,D=d]` is continuously
   differentiable in :math:`d` for every finite cohort :math:`g`
   and :math:`t=2,\ldots,T`.

   Assumption 3-MP(a) requires no anticipation for every supported
   treatment path and every pre-treatment period,

   .. math::

      Y_{it}(g,d)=Y_{it}(0),\qquad t<g.

   Assumption 3-MP(b) requires an untreated first period and an
   absorbing positive dose almost surely for every :math:`t=2,\ldots,T`
   and :math:`d\in\mathcal D_+`,

   .. math::

      W_{i1}=0,\qquad W_{i,t-1}=d\Longrightarrow W_{it}=d.

   Observed outcomes follow the actual adoption time and dose,

   .. math::

      Y_{it}=Y_{it}(0)\mathbf1\{t<G_i\}
              +Y_{it}(G_i,D_i)\mathbf1\{t\geq G_i\}.

The local target now describes units sharing both an adoption date
and an observed dose. For a post-treatment period :math:`t\geq g`,
its level effect and marginal response are

.. math::

   \begin{aligned}
   ATT(g,t,d\mid g,d)
      &=\mathbb E[Y_t(g,d)-Y_t(0)\mid G=g,D=d],\\
   ACRT(g,t,d\mid g,d)
      &=\left.\frac{\partial ATT(g,t,l\mid g,d)}{\partial l}\right|_{l=d}.
   \end{aligned}

To identify the level effect, we need untreated trends to agree across
both timing and dose groups. The never-treated group supplies the
reference mean in the next assumption.

.. admonition:: Assumption PT-MP Parallel trends across timing and dose
   :class: assumption

   For every :math:`g\in\overline{\mathcal G}`,
   :math:`t=2,\ldots,T`, and :math:`d\in\mathcal D_+`,

   .. math::

      \mathbb E[Y_t(0)-Y_{t-1}(0)\mid G=g,D=d]
      =\mathbb E[Y_t(0)-Y_{t-1}(0)\mid G=\infty,D=0].

Summing those one-period restrictions from adoption through period
:math:`t` connects untreated long differences. Identifying a local
response additionally requires restrictions on treated potential outcomes
within the timing cohort.

.. admonition:: Assumption SPT-MP Strong parallel trends in Appendix C
   :class: assumption

   For every finite cohort :math:`g`, period :math:`t=2,\ldots,T`,
   and positive doses :math:`l,d\in\mathcal D_+`,

   .. math::

      \begin{aligned}
      &\mathbb E[Y_t(g,d)-Y_{t-1}(g,d)\mid G=g,D=l]\\
      &\quad=\mathbb E[Y_t(g,d)-Y_{t-1}(g,d)\mid G=g,D=d].
      \end{aligned}

   The untreated parallel trends equality in PT-MP also holds for
   every finite cohort, period, and positive dose.

This Appendix C assumption equates each potential treatment-path change
across all dose groups within a timing cohort. It is stronger than the
two-period SPT restriction that matches a dose group to a treated-population
average. The stronger version is what lets the theorem identify the
response local to units receiving dose :math:`d`.

.. admonition:: Theorem C.1 Local timing-dose identification
   :class: theorem

   Under Assumptions 1-MP, 2-MP(a), 3-MP, and PT-MP, for every
   finite cohort :math:`g`, period :math:`2\leq g\leq t\leq T`,
   and positive dose :math:`d\in\mathcal D_+`,

   .. math::

      \begin{aligned}
      ATT(g,t,d\mid g,d)
         &=\mathbb E[Y_t-Y_{g-1}\mid G=g,D=d]\\
         &\quad-\mathbb E[Y_t-Y_{g-1}\mid W_t=0].
      \end{aligned}

   If Assumptions 2-MP(b) and SPT-MP also hold, then every
   :math:`d\in\mathcal D_+^c` satisfies

   .. math::

      ACRT(g,t,d\mid g,d)
      =\frac{\partial\,\mathbb E[Y_t-Y_{g-1}\mid G=g,D=d]}{\partial d}.

The comparison units with :math:`W_t=0` remain untreated at both
endpoints of the long difference. The theorem also permits using only
never-treated units. ModernDiD defaults to
``control_group="notyettreated"``; with positive anticipation, it
moves the clean base backward and excludes comparison units affected
before the relevant endpoints. The :ref:`staggered adoption background
<background-did>` explains why those two adjustments must be made together.

Targets for the whole timing cohort
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The supplement also develops an extension closer to the two-period
SPT target. It averages over all positive-dose units within one timing
cohort rather than keeping the observed dose group fixed,

.. math::

   \begin{aligned}
   ATT(g,t,d)&=\mathbb E[Y_t(g,d)-Y_t(0)\mid G=g,D>0],\\
   ACRT(g,t,d)&=\frac{\partial ATT(g,t,d)}{\partial d}.
   \end{aligned}

The same Appendix C version of SPT-MP supports both local effects and
these cohort-wide effects. Because its restriction holds across every
conditioning dose group, we can average within the cohort without
changing the identified dose comparison.

.. admonition:: Theorem S1 Cohort-wide level identification
   :class: theorem

   Under Assumptions 1-MP, 2-MP(a), 3-MP, and SPT-MP, every
   finite cohort :math:`g`, period :math:`2\leq g\leq t\leq T`,
   and dose :math:`d\in\mathcal D_+` satisfies

   .. math::

      \begin{aligned}
      ATT(g,t,d)
         &=\mathbb E[Y_t-Y_{g-1}\mid G=g,D=d]\\
         &\quad-\mathbb E[Y_t-Y_{g-1}\mid W_t=0].
      \end{aligned}

To see what happens when we differentiate the long-difference mean, write
:math:`M_{g,t}(d)=\mathbb E[Y_t-Y_{g-1}\mid G=g,D=d]`.
The supplement's next result separates the local response from selection
before giving the cohort-wide interpretation under SPT-MP.

.. admonition:: Theorem S2 Selection in timing-dose derivatives
   :class: theorem

   Under Assumptions 1-MP, 2-MP, and 3-MP, consider a finite cohort
   :math:`g`, a period :math:`2\leq g\leq t\leq T`, and a dose
   :math:`d\in\mathcal D_+^c`. If PT-MP also holds,

   .. math::

      \begin{aligned}
      M_{g,t}'(d)
         &=\frac{d\,ATT(g,t,d\mid g,d)}{dd}\\
         &=ACRT(g,t,d\mid g,d)
           +\left.\frac{\partial ATT(g,t,d\mid g,l)}{\partial l}\right|_{l=d}.
      \end{aligned}

   If SPT-MP instead holds, the derivative identifies the cohort-wide
   response,

   .. math::

      M_{g,t}'(d)=\frac{\partial ATT(g,t,d)}{\partial d}=ACRT(g,t,d).

These are the multi-period counterparts of the distinction in Theorems
3.2 and 3.3. The spline derivative is computable under either assumption,
but its causal interpretation still depends on the potential outcome
restriction rather than on the fit's precision.

Averages across dose and exposure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once each timing-period curve is identified, we can choose how to
summarize it. Appendix C's dose-specific level curve averages over the
timing distribution among units receiving that dose,

.. math::

   \begin{aligned}
   ATT^{dose}(d\mid d)
      &=\sum_g\sum_{t=g}^T
          \omega^{dose}(g,t,d)ATT(g,t,d\mid g,d),\\
   \omega^{dose}(g,t,d)
      &=\frac{P(G=g\mid D=d,G\leq T)}{T-g+1}.
   \end{aligned}

The same weights average local causal responses. As :math:`d` changes,
this curve can reflect changes in the timing composition as well as
changes in effects within cohorts.

ModernDiD's dose aggregation instead fixes the timing mixture using
cohort shares among all treated units. For a complete post-treatment
grid, its weights are

.. math::

   w_{g,t}=\frac{q_g}{T-g+1},\qquad q_g=P(G=g\mid G\leq T).

The package distributes each cohort's share evenly over its retained
post-treatment cells. These fixed weights coincide with the paper's
conditional timing weights when the timing distribution does not vary
with dose. Otherwise, the reported dose curve describes a standardized
timing mixture. Common dose support matters because every contributing
cohort must supply a curve at the dose being averaged.

The overall summaries are computed separately by averaging each cell's
level effect or fitted response over its own treated dose distribution.
As a result, ``overall_att`` need not equal an integral of the reported
dose curve over the pooled dose distribution when timing and dose are
associated. The distinction follows from the populations being averaged
rather than from a failure of numerical integration.

An event study holds exposure length :math:`e=t-g` fixed and combines
cohort summaries among units observed at that exposure. For the level
target, the package estimates each cohort-period binary ATT for receiving
any positive dose. For the slope target, it averages the fitted
cohort-period derivatives over that cohort's positive doses before
combining cohorts.

For example, write :math:`\theta_{g,t}` for the cell's binary level
ATT and let :math:`\mathcal G_e` contain cohorts observed at exposure
:math:`e`. The event-study level target is

.. math::

   ATT_{loc}^{es}(e)=\sum_{g\in\mathcal G_e}
      P(G=g\mid G\in\mathcal G_e)\theta_{g,g+e}.

Since the contributing cohorts can change as exposure grows, a changing
event-study average can reflect composition as well as changing effects.
The ``balance_e`` option restricts the population to cohorts observed
through a chosen post-treatment horizon. Appendix C's Remark C.1 gives
a direct binary DiD route for the level event study and does not supply
a separate formal estimation theorem for the staggered setting.

.. admonition:: Match the supported estimation path
   :class: tip

   The current API requires a balanced panel and ``xformla="~1"``.
   It does not implement sampling weights or discrete-dose estimation.
   Since clustering arguments are ignored with a warning, inference
   assumes independent units. Choose ``aggregation="dose"`` for curves or
   ``aggregation="eventstudy"`` for exposure profiles.

What pre-treatment comparisons can establish
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extra pre-treatment periods let you examine untreated mean changes
across doses and timing cohorts. A violation challenges the corresponding
extension of parallel trends. A small estimated violation can also arise
when the comparison is imprecise. Failure to reject therefore does not
establish the identifying restriction needed for the post-treatment effects.

An aggregated event study can additionally conceal dose-specific
violations that cancel when averaged. Pre-treatment level and slope
comparisons examine untreated changes rather than the extra restrictions
that SPT places on treated potential outcomes. Those counterfactual
treated paths are not observed before adoption.

The :ref:`continuous treatment example <example_cont_did>` reproduces
the fracking application in the authors' separate 2024 paper,
`Event Studies with a Continuous Treatment
<https://doi.org/10.1257/pandp.20241047>`_. Its dose-group event studies
and pooled level curves provide an application of these identification
ideas without claiming to reproduce the Medicare application in the
December 2025 main paper. Comparing effects across recorded doses still
requires care about differences between the counties receiving those doses.
