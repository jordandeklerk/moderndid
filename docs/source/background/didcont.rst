.. _background-didcont:

Difference-in-differences with a continuous treatment
=====================================================

When a policy gives treated units different amounts of treatment, you may
want to know both what it did at a given dose and what a small increase in
that dose would do.

Ordinary parallel trends can identify the effect of a dose for the units
that received it. Comparing that effect with the effect at another dose
also changes the units being compared. The slope of the resulting curve
can therefore reflect selection into doses as well as a causal response
to treatment.

We will work through that distinction before choosing an estimator using
the results of `Callaway, Goodman-Bacon, and Sant'Anna (2024)
<https://arxiv.org/abs/2107.02637v4>`_. Sections 3 and 4 of their
`author manuscript <https://arxiv.org/pdf/2107.02637v4>`_ cover the
two-period argument and estimation, followed by the staggered-adoption
extension in Appendix D. The formal statements retain the paper's numbering
as we connect them to :func:`~moderndid.cont_did`.
The :ref:`continuous treatment example <example_cont_did>` shows how those
choices affect an analysis.

Defining the dose and the outcome paths
---------------------------------------

We begin with :math:`n` units observed before treatment in period 1
and afterward in period 2. Write :math:`D_i` for unit :math:`i`'s dose
in period 2 and use zero for a unit that remains untreated.
The support :math:`\mathcal D` contains zero and positive doses
:math:`\mathcal D_+`. We suppress the unit index in population expressions.
Write :math:`\Delta Y=Y_2-Y_1` for the observed outcome change.

.. admonition:: Assumption 1 Random sampling
   :class: assumption

   The observed vectors :math:`(Y_{i2},Y_{i1},D_i)`,
   :math:`i=1,\ldots,n`, are independent draws from a common population
   distribution. Dependence between the two outcomes of a unit is unrestricted.

.. admonition:: Assumption 2 Treatment support
   :class: assumption

   No unit is treated in period 1. In period 2,
   :math:`\mathcal D=\{0\}\cup\mathcal D_+` satisfies one of two alternatives.

   For continuous doses, :math:`\mathcal D_+^c=[d_L,d_U]`
   for :math:`0<d_L<d_U<\bar d<\infty`. There is positive untreated
   mass, :math:`P(D=0)>0`. The conditional density among treated units
   satisfies, for some finite :math:`a_f>0`,

   .. math::

      a_f^{-1}<f_{D\mid D>0}(d)<a_f,\qquad d\in\mathcal D_+^c.

   The conditional mean :math:`\mathbb E[\Delta Y\mid D=d]` is
   continuously differentiable on this interval.

   For ordered discrete doses,
   :math:`\mathcal D_+^{mv}=\{d_1,\ldots,d_J\}`
   for :math:`0<d_1<\cdots<d_J<\bar d<\infty`.
   Every dose, including zero, has positive probability.

The continuous-dose model permits a gap between its mass at zero and the
smallest positive dose. We fit the curve over the separate interval of
positive doses without smoothing across that gap.

Let :math:`Y_{it}(d)` denote the outcome unit :math:`i` would have
in period :math:`t` under dose :math:`d`. These potential outcomes
describe the treatment amount received by that unit. They presume that
another unit's dose does not change its outcome.

.. admonition:: Assumption 3 No anticipation and observed outcomes
   :class: assumption

   For every unit and every :math:`d\in\mathcal D`,

   .. math::

      Y_{i1}=Y_{i1}(d)=Y_{i1}(0),
      \qquad Y_{i2}=Y_{i2}(D_i).

   All relevant expectations are finite and well defined.

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
Its average within a group that received dose :math:`d'` and its
population average are

.. math::

   \begin{aligned}
   ATT(d\mid d')&=\mathbb E[Y_2(d)-Y_2(0)\mid D=d'],\\
   ATE(d)&=\mathbb E[Y_2(d)-Y_2(0)].
   \end{aligned}

The first argument tells us which counterfactual dose to evaluate for the
population selected by the second argument. Along the diagonal curve
:math:`ATT(d\mid d)`, increasing :math:`d` changes both the treatment dose
and the units being averaged. A higher point can therefore reflect a
different response to treatment or a different set of units receiving that
dose.

For a continuous dose, conditioning on :math:`D=d` describes a conditional
mean function rather than a subgroup with positive probability. The support
and smoothness conditions make that function meaningful for a regression
that estimates it using nearby doses in a finite sample.

What a marginal increase would change
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A causal response holds the population fixed while changing its dose.
For continuous treatment, define

.. math::

   ACRT(d\mid d')
   =\left.\frac{\partial ATT(l\mid d')}{\partial l}\right|_{l=d},
   \qquad
   ACR(d)=\frac{\partial ATE(d)}{\partial d}.

Defining responses through conditional means requires the corresponding
derivatives to exist without imposing differentiability on every unit's
path. To interchange unit-level differentiation and expectation, you
additionally need an integrability condition.

For discrete doses, set :math:`d_0=0`. The corresponding responses
between adjacent doses are

.. math::

   \begin{aligned}
   ACRT(d_j\mid d_k)
      &=\mathbb E[Y_2(d_j)-Y_2(d_{j-1})\mid D=d_k],\\
   ACR(d_j)
      &=\mathbb E[Y_2(d_j)-Y_2(d_{j-1})].
   \end{aligned}

These are finite changes rather than derivatives per unit of dose.
With binary treatment, the finite change from zero to treatment is also
the level effect. With several doses, that coincidence no longer holds.

Averages over the treated dose distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your question concerns an overall effect, the treated dose distribution
gives a natural set of weights. Define

.. math::

   \begin{aligned}
   ATT^o&=\mathbb E[ATT(D\mid D)\mid D>0],&
   ATE^o&=\mathbb E[ATE(D)\mid D>0],\\
   ACRT^o&=\mathbb E[ACRT(D\mid D)\mid D>0],&
   ACR^o&=\mathbb E[ACR(D)\mid D>0].
   \end{aligned}

The conditioning distribution determines how much weight each dose receives
without changing the population inside :math:`ATE(d)` or :math:`ACR(d)`.
For example, :math:`ATE^o` averages population effects at doses drawn
from the treated distribution. It need not equal the average effect actually
experienced by treated units.

Identifying the level curve
---------------------------

The missing untreated outcome can be recovered if untreated mean changes
agree across dose groups. We first impose that familiar DiD condition.
It makes comparisons with dose zero interpretable without equating
treatment effects across positive-dose groups.

.. admonition:: Assumption 4 Parallel trends
   :class: assumption

   For every :math:`d\in\mathcal D`,

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
      &=\mathbb E[\Delta Y\mid D=d]
         -\mathbb E[\Delta Y\mid D=0].
   \end{aligned}

Since the mean change within dose group :math:`d` is observed, parallel
trends is needed to replace its missing untreated change with the observed
change in the zero-dose group.

.. admonition:: Theorem 3.1 Level effects under parallel trends
   :class: theorem

   Under Assumptions 1 through 4, every :math:`d\in\mathcal D_+`
   satisfies

   .. math::

      ATT(d\mid d)
      =\mathbb E[\Delta Y\mid D=d]-\mathbb E[\Delta Y\mid D=0].

   Averaging over positive doses also identifies

   .. math::

      ATT^o
      =\mathbb E[\Delta Y\mid D>0]-\mathbb E[\Delta Y\mid D=0].

A binary indicator for receiving any positive dose can therefore estimate
the overall ATT directly without fitting a dose curve. The curve is needed
when you want to describe how level effects differ across the units receiving
different amounts of treatment.

Why the slope needs another assumption
---------------------------------------

A derivative of the observed level curve changes the dose and the
conditioning group together. The causal response changes only the dose.
We can see the difference by differentiating the two arguments separately.

.. admonition:: Theorem 3.2 Selection in comparisons across doses
   :class: theorem

   Under Assumptions 1 through 4, the continuous-dose case satisfies,
   wherever the component derivatives exist,

   .. math::

      \begin{aligned}
      \frac{d\,\mathbb E[\Delta Y\mid D=d]}{dd}
      &=\frac{d\,ATT(d\mid d)}{dd}\\
      &=ACRT(d\mid d)
        +\left.\frac{\partial ATT(d\mid l)}{\partial l}\right|_{l=d}.
      \end{aligned}

   For any :math:`(h,l)\in\mathcal D\times\mathcal D`,

   .. math::

      \begin{aligned}
      \mathbb E[\Delta Y\mid D=h]-\mathbb E[\Delta Y\mid D=l]
      &=ATT(h\mid h)-ATT(l\mid l)\\
      &=\mathbb E[Y_2(h)-Y_2(l)\mid D=h]\\
      &\quad+ATT(l\mid h)-ATT(l\mid l).
      \end{aligned}

   With discrete treatment and adjacent doses, the last expression becomes

   .. math::

      ACRT(d_j\mid d_j)
      +ATT(d_{j-1}\mid d_j)-ATT(d_{j-1}\mid d_{j-1}).

   Thus parallel trends alone does not identify the causal response
   from comparisons across doses.

The final term compares the effect of the same dose :math:`l` across two
groups whose gains are unrestricted by parallel trends. If units at higher
doses would benefit more even at dose :math:`l`, the observed difference
includes those different gains.

For the continuous-dose decomposition, the observed slope mixes selection
into the conditioning group with the response to a marginal intervention
on dose. More precise estimation of the left-hand side cannot distinguish
these two contributions.

.. admonition:: Read a reported slope conditionally
   :class: important

   The ``acrt_d`` field contains the derivative of the estimated level
   curve. Under ordinary parallel trends, that derivative combines a causal
   response and selection across dose groups. A causal interpretation
   requires an additional restriction on treated potential outcomes.

Identifying effects for a common population
--------------------------------------------

If you want to compare doses for the same population, the restriction must
also connect treated potential outcomes across dose groups. Strong parallel
trends does this through the population mean outcome path.

.. admonition:: Assumption 5 Strong parallel trends
   :class: assumption

   For every :math:`d\in\mathcal D`,

   .. math::

      \mathbb E[Y_2(d)-Y_1(0)]
      =\mathbb E[Y_2(d)-Y_1(0)\mid D=d].

Strong parallel trends asks dose group :math:`d` to represent the population
mean change if everyone received that dose. It equates those mean changes
separately at each dose, including zero.

Assumptions 4 and 5 are non-nested. Strong parallel trends alone identifies
population effects rather than automatically identifying the diagonal ATT.
If ordinary parallel trends is also maintained, Theorem C.1 gives

.. math::

   ATT(d\mid d)=ATE(d),\qquad d\in\mathcal D.

That equality concerns a dose group's effect at its own dose.
Even the two assumptions together do not imply
:math:`ATT(l\mid h)=ATT(l\mid l)` for every pair of different doses.
The stronger alternative assumption in Appendix C imposes equality
of potential mean changes across every conditioning dose group.

.. admonition:: Theorem 3.3 Population effects under strong parallel trends
   :class: theorem

   Under Assumptions 1 through 3 and 5, every
   :math:`d\in\mathcal D_+` satisfies

   .. math::

      ATE(d)=\mathbb E[\Delta Y\mid D=d]-\mathbb E[\Delta Y\mid D=0].

   For continuous treatment,

   .. math::

      ACR(d)=\frac{d\,\mathbb E[\Delta Y\mid D=d]}{dd}
            =\frac{d\,ATE(d)}{dd}.

   For every :math:`(h,l)\in\mathcal D\times\mathcal D`,

   .. math::

      ATE(h)-ATE(l)
      =\mathbb E[Y_2(h)-Y_2(l)]
      =\mathbb E[\Delta Y\mid D=h]-\mathbb E[\Delta Y\mid D=l].

   For discrete treatment, adjacent differences identify
   :math:`ACR(d_j)` by setting :math:`h=d_j` and :math:`l=d_{j-1}`.

The observed comparison has the same form as it did under parallel trends.
Its interpretation changes because the stronger restriction changes whose
counterfactual mean the dose group represents. In particular, the identified
slope is the population response :math:`ACR(d)`. It need not equal
:math:`ACRT(d\mid d)` without an additional restriction.

Corollary 3.1 identifies the corresponding summaries,

.. math::

   \begin{aligned}
   ATE^o
      &=\mathbb E[\Delta Y\mid D>0]-\mathbb E[\Delta Y\mid D=0],\\
   ACR^o
      &=\int_{d_L}^{d_U}
         \frac{d\,\mathbb E[\Delta Y\mid D=d]}{dd}
         f_{D\mid D>0}(d)\,dd.
   \end{aligned}

For discrete doses, replace the integral with

.. math::

   ACR^o
   =\sum_{j=1}^J
      \bigl(\mathbb E[\Delta Y\mid D=d_j]
            -\mathbb E[\Delta Y\mid D=d_{j-1}]\bigr)
      P(D=d_j\mid D>0).

By averaging over the observed treated dose distribution, these summaries
avoid the weights implicit in a regression of changes on one dose variable.

What a linear dose regression averages
---------------------------------------

A linear regression on dose can produce a summary whose weights differ
from the effect you want to report even with only two periods. We can
examine this problem before adding staggered timing through the same
conditional change function :math:`m(d)=\mathbb E[\Delta Y\mid D=d]`.

If :math:`\mu_D=\mathbb E[D]` and
:math:`v_D=\operatorname{Var}(D)>0`, the coefficient is

.. math::

   \beta^{twfe}
   =\frac{\operatorname{Cov}(D,\Delta Y)}{v_D}
   =\frac{\mathbb E[(D-\mu_D)m(D)]}{v_D}.

Under parallel trends, subtracting :math:`m(0)` replaces :math:`m(d)`
with :math:`ATT(d\mid d)` in this expression. Under strong parallel
trends, the same subtraction yields :math:`ATE(d)`. The level weights
on positive doses are

.. math::

   w^{lev}(d)=\frac{(d-\mu_D)f_D(d)}{v_D},\qquad d\in[d_L,d_U],

where :math:`f_D(d)=P(D>0)f_{D\mid D>0}(d)`.
Weights below the mean dose are negative. The full signed weighting measure,
including its zero-dose atom, has total mass zero,

.. math::

   \int_{d_L}^{d_U}w^{lev}(d)\,dd
   -\frac{\mu_D P(D=0)}{v_D}=0.

The zero-dose level effect drops out of the integral because it is zero,
even though its weight still contributes to the signed measure's total
mass. The coefficient therefore does not give a convex average of level
effects.

The slope representation has positive weights. Because the positive-dose
support starts above zero, it also includes a bridge from no treatment to
the smallest positive dose. Define

.. math::

   w^{acr}(d)
      =\frac{\mathbb E[(D-\mu_D)\mathbf 1\{D\geq d\}]}{v_D},
   \qquad
   w_0^{acr}=\frac{d_L\mu_D P(D=0)}{v_D}.

Then the continuous-dose decomposition in Theorem 3.4 gives

.. math::

   \beta^{twfe}
   =\int_{d_L}^{d_U}w^{acr}(d)m'(d)\,dd
     +w_0^{acr}\frac{m(d_L)-m(0)}{d_L},
   \qquad
   \int_{d_L}^{d_U}w^{acr}(d)\,dd+w_0^{acr}=1.

Under ordinary parallel trends, :math:`m'(d)` includes the selection
term from Theorem 3.2. Under strong parallel trends, it is :math:`ACR(d)`.
Even in that case, these weights generally differ from
:math:`f_{D\mid D>0}`. The weighting differences cease to matter if
the causal response is constant and the scaled level effect at
:math:`d_L` equals that same response. A constant slope on the
positive-dose interval alone does not establish that second condition.

The same coefficient also admits the paper's scaled-level and
high-versus-low dose decompositions. Dividing a level effect by its dose
changes the target to an effect per unit of treatment. Positive weights
on a high-versus-low contrast do not remove its selection term.
Choosing the level curve or the average response directly makes the target
explicit before estimation.

What weaker restrictions can support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Strong parallel trends may be implausible in an application. Section 5.1
describes restrictions that answer narrower questions. For example, if
:math:`\partial ATT(d\mid l)/\partial l\geq0` at :math:`l=d`,
Theorem 3.2 makes the observed slope an upper bound on
:math:`ACRT(d\mid d)`. Reversing that restriction reverses the bound.

A local version of strong parallel trends can identify population dose
contrasts and slopes within the interval where it holds. Identifying a
level effect relative to zero additionally requires a restriction linking
that interval to the untreated reference. A covariate-conditional version
instead compares mean paths within pre-treatment covariate values and
averages over a specified covariate distribution. That extension also
needs support for the relevant conditional dose comparisons.

If no untreated units exist, a common untreated mean trend across positive
dose groups still yields

.. math::

   \mathbb E[\Delta Y\mid D=d]-\mathbb E[\Delta Y\mid D=d_L]
   =ATT(d\mid d)-ATT(d_L\mid d_L).

This contrast retains the selection issue. If strong parallel trends holds
over the positive-dose support, it instead identifies
:math:`ATE(d)-ATE(d_L)`. Neither contrast fixes the absolute effect
relative to zero without more information. The formal support assumptions
above and the package's baseline comparison require untreated units.

Fitting the curve in ModernDiD
-------------------------------

Once identification supplies a conditional mean function, estimating it
from finitely many observed doses still requires a way to fit the curve.
We separate a fixed spline specification from a sieve chosen from the data.

For discrete doses, a saturated regression of :math:`\Delta Y_i`
on dose indicators has untreated units as its reference,

.. math::

   \Delta Y_i=\beta_0+
      \sum_{j=1}^J\mathbf 1\{D_i=d_j\}\beta_j+\varepsilon_i.

Each coefficient estimates the level comparison at that dose.
Under strong parallel trends, :math:`\beta_1` estimates :math:`ACR(d_1)`.
For :math:`j\geq2`, :math:`\beta_j-\beta_{j-1}` estimates
:math:`ACR(d_j)`. This is a theoretical discrete-dose estimator.
ModernDiD currently accepts only ``treatment_type="continuous"``.

A spline for positive doses
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let :math:`\psi^K(d)` be a vector of :math:`K` B-spline basis functions,
including an intercept. Write :math:`\overline{\Delta Y}_0` for the
sample mean change among zero-dose units. The positive-dose regression is

.. math::

   \widehat\beta_K
   =\left[\sum_{i:D_i>0}\psi^K(D_i)\psi^K(D_i)'\right]^{-}
      \sum_{i:D_i>0}\psi^K(D_i)(\Delta Y_i-\overline{\Delta Y}_0),

where :math:`(\cdot)^{-}` denotes the Moore-Penrose inverse.
The estimated level function and its slope are

.. math::

   \widehat h_K(d)=\psi^K(d)'\widehat\beta_K,
   \qquad \widehat h_K'(d)=\partial\psi^K(d)'\widehat\beta_K.

The curve has target :math:`h(d)=ATT(d\mid d)` under ordinary parallel
trends and :math:`h(d)=ATE(d)` under strong parallel trends. Every fitted
curve and summary must therefore be read under the identifying assumption
maintained for the analysis.

The default ``dose_est_method="parametric"`` uses ``degree=3``
and ``num_knots=0``. This is a single cubic polynomial on positive doses.
Increasing the number of interior knots permits a more flexible curve.
A fixed specification generally estimates a projection if the true function
lies outside its span. Consistency for an unrestricted dose function
requires an increasing sieve dimension under suitable regularity conditions.

Because the untreated mean is estimated rather than known, its sampling
error contributes to uncertainty in the level curve. The additive constant
disappears on differentiation and does not contribute to uncertainty in
the fitted slope.

Choosing the sieve from the data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``dose_est_method="cck"`` path uses the nonparametric regression
special case of `Chen, Christensen, and Kankanala (2024)
<https://arxiv.org/abs/2107.11869>`_. It compares fits across a dyadic set of
dimensions instead of choosing one polynomial specification in advance.
The :ref:`nonparametric IV background <background-npiv>` derives the
selection and confidence-band procedures.

For scalar dose and cubic splines, the theoretical grid is

.. math::

   \mathcal K=\{2^l+3:l=0,1,\ldots\},
   \qquad
   v_n=\max\{1,(0.1\log n_+)^4\},

where :math:`n_+` is the number of positive-dose observations.
The procedure bounds feasible dimensions using
:math:`K\sqrt{\log K}\,v_n` and compares larger fits with smaller ones.
It chooses the smallest dimension whose standardized differences from
larger candidates do not exceed a bootstrap threshold.
A final truncation protects against choosing the grid's largest dimension.

ModernDiD's CCK path requires two periods and one treated timing cohort.
It uses cubic splines at quantile knots, passes dose as both the regressor
and instrument, and uses 999 multiplier draws internally. The generic
``npiv`` API exposes broader selection controls than the single-cohort
``cont_did`` CCK path.

Conditions for the nonparametric guarantees
-------------------------------------------

Adaptive estimation and coverage depend on error moments and the growth of
uncertainty with sieve dimension as well as parallel trends. We state these
conditions before reporting the paper's nonparametric regression guarantees.

Let :math:`u=\Delta Y-\mathbb E[\Delta Y\mid D]` for treated observations.
For a candidate dimension :math:`K`, define

.. math::

   \begin{aligned}
   H_K&=\mathbb E[\psi^K(D)\psi^K(D)'\mid D>0],\\
   s_K^2(d)&=\psi^K(d)'H_K^{-1}\psi^K(d),\\
   \sigma_K^2(d)
      &=\psi^K(d)'H_K^{-1}
         \mathbb E[u^2\psi^K(D)\psi^K(D)'\mid D>0]
         H_K^{-1}\psi^K(d),\\
   s_{K,1}^2(d)&=\partial\psi^K(d)'H_K^{-1}\partial\psi^K(d).
   \end{aligned}

These are population variance measures for the regression approximation.
The derivative basis changes the growth rate of the last quantity.
All inverses in these population conditions require nonsingular
:math:`H_K`.

.. admonition:: Assumption 6 Nonparametric regression regularity
   :class: assumption

   There are finite positive constants
   :math:`c,C,\underline\sigma,\overline\sigma` and
   :math:`\rho\in(0,1)` such that, almost surely among treated units,

   .. math::

      \mathbb E[u^4\mid D]\leq\overline\sigma^2,
      \qquad \mathbb E[u^2\mid D]\geq\underline\sigma^2.

   For every :math:`K\in\mathcal K`,

   .. math::

      \begin{aligned}
      cK&\leq\inf_d s_K^2(d)\leq\sup_d s_K^2(d)\leq CK,\\
      cK^3&\leq\inf_d s_{K,1}^2(d)
               \leq\sup_d s_{K,1}^2(d)\leq CK^3.
      \end{aligned}

   Here extrema run over :math:`\mathcal D_+^c`. The variance ratios satisfy

   .. math::

      \limsup_{K\to\infty}
      \sup_{\substack{d\in\mathcal D_+^c\\K_2\in\mathcal K,\ K_2>K}}
      \frac{\sigma_K^2(d)}{\sigma_{K_2}^2(d)}<\rho.

   The derivative variance condition is needed for derivative bands.

To describe smoothness, let :math:`\mathcal H^p` be a Hölder ball of
radius :math:`M` and smoothness :math:`p` on the positive-dose interval.
Its members have uniformly bounded derivatives and the corresponding
Hölder continuity bound. The paper uses the Hölder-Zygmund notation
:math:`H_{\infty,\infty}^p(M)` for this class.
Take :math:`p\in[\underline p,\overline p]` for fixed
:math:`\overline p>\underline p>1/2`. Take spline order
:math:`r\geq\lfloor\overline p\rfloor+1` so the approximation space
can represent this smoothness range. The :ref:`NPIV background
<background-npiv>` defines the smoothness class and its approximation
conditions more fully.

.. admonition:: Theorem 4.1 Adaptive rates for the level and slope curves
   :class: theorem

   Under Assumptions 1, 2(a), 3, 5, and 6, the paper's data-driven
   estimator has constants :math:`C_1,C_1'>0` such that

   .. math::

      \sup_{p\in[\underline p,\overline p]}
      \sup_{h\in\mathcal H^p}
      P_h\!\left(
         \|\widehat h_{\widehat K}-h\|_\infty
         >C_1(\log n/n)^{p/(2p+1)}
      \right)\longrightarrow0.

   If :math:`\underline p>1`, its derivative also satisfies

   .. math::

      \sup_{p\in[\underline p,\overline p]}
      \sup_{h\in\mathcal H^p}
      P_h\!\left(
         \|\widehat h_{\widehat K}'-h'\|_\infty
         >C_1'(\log n/n)^{(p-1)/(2p+1)}
      \right)\longrightarrow0.

   Here :math:`h=ATE` and the norm is the largest absolute error over
   :math:`\mathcal D_+^c`. The probabilities range over distributions
   satisfying the stated conditions.

The increasing-sieve procedure attains these minimax rates for the specified
smoothness classes. Its derivative converges more slowly because small
changes in the function can produce larger changes in its slope. The
guarantee does not establish unrestricted-function consistency for the
default fixed cubic fit.

Which functions can have adaptive bands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To cover every dose at once while adapting band width to unknown smoothness,
the paper additionally restricts how the function's approximation error
behaves across dimensions. Let :math:`\Pi_Kh` be its population least-squares
projection onto the spline space. For fixed :math:`0<\underline B<\overline B`
and a fixed starting dimension :math:`K_*`, define

.. math::

   \begin{aligned}
   \mathcal G^p
      &=\left\{h\in\mathcal H^p:
          \|\Pi_Kh-h\|_\infty\geq\underline B K^{-p}
          \text{ for every }K\in\mathcal K,\ K\geq K_*
         \right\},\\
   \mathcal G&=\bigcup_{p\in[\underline p,\overline p]}\mathcal G^p.
   \end{aligned}

The upper approximation bound is
:math:`\|\Pi_Kh-h\|_\infty\leq\overline B K^{-p}`.
Together, these bounds describe the self-similar subclass used for honest
adaptive bands. A coverage statement over this subclass is more specific
than one over every smooth function.

Write :math:`\widehat{se}_K(d)` for the level standard error and
:math:`z_{1-\alpha}^*` for the bootstrap quantile over doses and candidate
dimensions. If :math:`\gamma^*` is the selection critical value, the
paper's level band is

.. math::

   C_n(d,A)=
      [\,\widehat h_{\widehat K}(d)
         \ \pm\ (z_{1-\alpha}^*+A\gamma^*)\widehat{se}_{\widehat K}(d)\,].

The derivative band :math:`C_n^1(d,A)` uses the fitted derivative,
its standard error, and the corresponding bootstrap quantile.
The parameter :math:`\alpha\in(0,1)` is the family's noncoverage
probability.

.. admonition:: Theorem 4.2 Coverage and width of adaptive bands
   :class: theorem

   Under the conditions of Theorem 4.1, sufficiently large fixed
   :math:`A` gives

   .. math::

      \liminf_{n\to\infty}\inf_{h\in\mathcal G}
      P_h\{h(d)\in C_n(d,A)\text{ for all }d\in\mathcal D_+^c\}
      \geq1-\alpha.

   For some universal :math:`C_2>0`,

   .. math::

      \inf_{p\in[\underline p,\overline p]}\inf_{h\in\mathcal G^p}
      P_h\!\left\{
         \sup_d|C_n(d,A)|
         \leq C_2(1+A)(\log n/n)^{p/(2p+1)}
      \right\}\longrightarrow1.

   If :math:`\underline p>1`, sufficiently large :math:`A` also gives

   .. math::

      \liminf_{n\to\infty}\inf_{h\in\mathcal G}
      P_h\{h'(d)\in C_n^1(d,A)\text{ for all }d\in\mathcal D_+^c\}
      \geq1-\alpha,

   with width bounded in the same uniform probability sense by
   :math:`C_2'(1+A)(\log n/n)^{(p-1)/(2p+1)}`.
   The required lower thresholds for :math:`A` are independent of
   :math:`\alpha` but can differ between levels and derivatives.

The recommended :math:`\widehat A=\log\log\widehat K` allows coverage
over the stated subclasses without choosing a fixed inflation constant.
Its band widths carry an additional :math:`\log\log n` factor.
These asymptotic coverage results do not supply a finite-sample guarantee
for a particular data set.

In the CCK implementation, the level standard error also incorporates
uncertainty in the untreated mean. Its simultaneous level band uses a
conservative approximation based on the NPIV critical value. Treat that
implementation as a numerical procedure informed by the theorem's
construction rather than an exact reproduction of every theoretical
candidate-set calculation.

Summarizing the fitted effects
--------------------------------

The overall level effect can be estimated directly from mean changes.
The slope summary instead averages a fitted derivative over positive doses.
We keep those estimation problems separate because their uncertainty has
different sources.

For :math:`ATT^o`, the regression

.. math::

   \Delta Y_i=\beta_0^{bin}+\mathbf1\{D_i>0\}\beta^{bin}+\varepsilon_i

estimates the binary DiD comparison from Theorem 3.1.
Under strong parallel trends, the same coefficient estimates
:math:`ATE^o`. ModernDiD's overall level result uses this direct
comparison rather than requiring the integral of the fitted level curve
to reproduce the binary estimate.

For :math:`ACR^o`, the plug-in estimator is

.. math::

   \widehat{ACR}^o
   =\frac1{n_+}\sum_{i:D_i>0}\widehat h'(D_i).

The regularity conditions for root-sample-size inference are stronger than
those needed to estimate a smooth curve.

.. admonition:: Assumption 7 Average derivative regularity
   :class: assumption

   The treated dose density is continuously differentiable and vanishes
   at the endpoints of :math:`[d_L,d_U]`. Its density
   score has finite second moment,

   .. math::

      \mathbb E\!\left[
         \left(\frac{f_{D\mid D>0}'(D)}{f_{D\mid D>0}(D)}\right)^2
         \,\middle|\,D>0
      \right]<\infty.

The boundary condition permits integration by parts without an endpoint
term. Appendix A adds it to Assumption 2(a)'s density conditions.
A density cannot remain uniformly bounded away from zero arbitrarily
close to an endpoint and also vanish continuously there.
The two conditions therefore conflict if read literally.
The theorem below retains the paper's original assumption references.
Applying it requires resolving this boundary issue rather than treating
Assumption 7 as an automatic consequence of the support condition.

The variance construction treats the fitted curve as a series regression
whose coefficients are estimated from the positive-dose sample.
For that sample, define

.. math::

   \begin{aligned}
   \widehat u_i
      &=\Delta Y_i-\overline{\Delta Y}_0-\widehat h_K(D_i),\\
   \widehat{\mathbf G}_K
      &=\frac1{n_+}\sum_{i:D_i>0}
         \psi^K(D_i)\psi^K(D_i)',\\
   \widehat{\mathbf a}_K'
      &=\frac1{n_+}\sum_{i:D_i>0}\partial\psi^K(D_i)'.
   \end{aligned}

The proposed influence-function estimate and variance estimate are

.. math::

   \begin{aligned}
   \widehat\eta_i
      &=\widehat h_K'(D_i)-\widehat{ACR}^o
        +\widehat{\mathbf a}_K'
         \widehat{\mathbf G}_K^-\psi^K(D_i)\widehat u_i,\\
   \widehat\sigma_{ACR^o}^{\,2}
      &=\frac1{n_+}\sum_{i:D_i>0}\widehat\eta_i^{\,2}.
   \end{aligned}

The influence function combines variation in fitted responses across sampled
doses with uncertainty in estimating the response curve through its weighted
residual term.

.. admonition:: Theorem 4.3 Efficient inference for the average response
   :class: theorem

   Under Assumptions 1, 2(a), 3, 5, 6, and 7, the paper's
   plug-in estimator and proposed variance estimator satisfy

   .. math::

      \frac{\sqrt{n_+}(\widehat{ACR}^o-ACR^o)}
           {\widehat\sigma_{ACR^o}}
      \xrightarrow{d}N(0,1),
      \qquad
      \widehat\sigma_{ACR^o}^2\xrightarrow{p}V_{ACR},

   where, for positive limiting variance,

   .. math::

      V_{ACR}
      =\operatorname{Var}\!\left[
         ACR(D)-u\,\frac{f_{D\mid D>0}'(D)}{f_{D\mid D>0}(D)}
         \,\middle|\,D>0
      \right].

   This is the semiparametric efficiency bound for :math:`ACR^o`
   under the paper's model.

The theorem concerns the paper's increasing-sieve estimator and variance
construction. A fixed polynomial slope average is a different estimator
unless its specification and approximation conditions justify the same
target. The package's ``overall_acrt`` summaries have a population
causal-response interpretation only under the relevant identifying
assumptions.

Adding staggered adoption
---------------------------

When treatment starts at different dates, timing and dose must both
describe the counterfactual path. Let :math:`G_i` be the first treatment
period and let :math:`G_i=\infty` identify a never-treated unit.
The paper instead codes never-treated timing with zero.
The dose :math:`D_i` stays fixed after adoption. Write

.. math::

   W_{it}=D_i\mathbf1\{t\geq G_i\},
   \qquad Y_{it}(g,d)

for the treatment received at time :math:`t` and the potential outcome
under adoption time :math:`g` and dose :math:`d`.
Here :math:`W_{it}` is a treatment amount rather than an instrument.
Set :math:`Y_{it}(0)=Y_{it}(\infty,0)`.

The Appendix D assumptions make the support and outcome-path restrictions
explicit before identifying each timing-dose comparison.

.. admonition:: Assumptions 1-MP through 3-MP Panel treatment paths
   :class: assumption

   Assumption 1-MP requires independent, identically distributed
   vectors :math:`(Y_{i1},\ldots,Y_{iT},D_i,G_i)`.

   Assumption 2-MP(a) requires compact dose support
   :math:`\mathcal D=\{0\}\cup\mathcal D_+\subset\mathbb R_+`,
   positive untreated mass, and common positive-dose support across
   finite timing cohorts. Every dose in :math:`\mathcal D_+` has
   positive conditional density or mass within each finite cohort.

   Assumption 2-MP(b), for derivatives, requires
   :math:`\mathcal D_+=[d_L,d_U]` for
   :math:`0<d_L<d_U<\infty` and continuous differentiability of
   :math:`\mathbb E[Y_t-Y_{t-1}\mid G=g,D=d]` in dose
   for each finite cohort and :math:`t=2,\ldots,T`.

   Assumption 3-MP requires no anticipation for every treatment path,

   .. math::

      Y_{it}(g,d)=Y_{it}(0),\qquad t<g,

   and an untreated first period followed by an absorbing positive dose,

   .. math::

      W_{i1}=0,\qquad
      W_{i,t-1}=d\Longrightarrow W_{it}=d,\quad t=2,\ldots,T,\quad d>0.

   Observed outcomes follow the unit's actual adoption time and dose.

The relevant target averages effects within one timing cohort,

.. math::

   ATE(g,t,d)
   =\mathbb E[Y_t(g,d)-Y_t(0)\mid G=g],
   \qquad
   ACR(g,t,d)=\frac{\partial ATE(g,t,d)}{\partial d}.

Unlike an effect conditional on both :math:`G=g` and :math:`D=d`,
the first target represents the whole timing cohort under dose :math:`d`.

.. admonition:: Assumption 5-MP Strong parallel trends across timing and dose
   :class: assumption

   For every finite timing cohort :math:`g`, period
   :math:`t=2,\ldots,T`, and supported positive dose :math:`d`,

   .. math::

      \begin{aligned}
      \mathbb E[Y_t(g,d)-Y_{t-1}(g,d)\mid G=g,D=d]
         &=\mathbb E[Y_t(g,d)-Y_{t-1}(g,d)\mid G=g],\\
      \mathbb E[Y_t(0)-Y_{t-1}(0)\mid G=g,D=d]
         &=\mathbb E[Y_t(0)-Y_{t-1}(0)\mid G=\infty,D=0].
      \end{aligned}

Assumption 5-MP connects dose groups within a timing cohort under the same
counterfactual treatment path and connects untreated changes across timing
and dose groups. Under no anticipation, summing these changes from
:math:`g` through :math:`t` gives the long-difference comparison.

.. admonition:: Theorem D.1 Timing-dose identification
   :class: theorem

   Under Assumptions 1-MP, 2-MP(a), 3-MP, and 5-MP,
   for every supported finite cohort :math:`g` and
   :math:`2\leq g\leq t\leq T` with an observed base :math:`g-1`,

   .. math::

      ATE(g,t,d)
      =\mathbb E[Y_t-Y_{g-1}\mid G=g,D=d]
       -\mathbb E[Y_t-Y_{g-1}\mid W_t=0],
      \qquad d\in\mathcal D_+.

   If Assumption 2-MP(b) also holds,

   .. math::

      ACR(g,t,d)
      =\frac{\partial\,\mathbb E[Y_t-Y_{g-1}\mid G=g,D=d]}
             {\partial d}.

   The zero-treatment comparison can instead use never-treated units.

The paper's theorem allows not-yet-treated units because their long
difference remains untreated under no anticipation. The package uses
``control_group="notyettreated"`` by default. With anticipation, it
requires comparison units to remain unaffected beyond the relevant
endpoints and moves the clean base backward. That is an extension of the
no-anticipation statement above. The :ref:`staggered adoption background
<background-did>` explains why the base and comparison cutoff must move
together.

Averages across dose and exposure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For dose aggregation, we combine timing-period curves at a fixed dose.
If all post-treatment cells are identified, define cohort shares
:math:`q_g=P(G=g\mid G<\infty)` and weights

.. math::

   w_{g,t}=\frac{q_g}{T-g+1},\qquad g\leq t\leq T,
   \qquad
   ATE^{dose}(d)=\sum_g\sum_{t=g}^T w_{g,t}ATE(g,t,d).

Using the same weights for :math:`ACR(g,t,d)` assigns each cohort its
population share in total, divided among its observed post-treatment periods.
This is the package's group-style dose aggregation when the target retains
all those cells. Common dose support matters because each contributing
cohort must supply a curve at the dose being averaged.

An event study instead holds exposure length :math:`e=t-g` fixed.
For the level target, the package first estimates a cohort-period binary
ATT for receipt of any positive dose. For the slope target, it averages
the fitted cohort-period slopes over that cohort's positive-dose
distribution. It combines those cell summaries across cohorts observed
at the requested event time.

Since the contributing cohorts can change as exposure grows, a changing
event-study average can reflect composition as well as changing effects.
The ``balance_e`` option holds post-treatment cohort support fixed over a
chosen horizon by restricting the target population to cohorts observed
for that whole horizon.

.. admonition:: Match the supported estimation path
   :class: tip

   The current API requires a balanced panel and ``xformla="~1"``.
   It does not implement sampling weights or discrete-dose estimation.
   Clustering arguments are ignored with a warning. These results therefore
   use independent-unit inference. Choose ``aggregation="dose"`` for curves
   or ``aggregation="eventstudy"`` for exposure profiles.

What pre-treatment comparisons can establish
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extra pre-treatment periods let you examine untreated mean changes
across doses and timing cohorts. A violation challenges the corresponding
extension of parallel trends. A small estimated violation can also arise
when the comparison is imprecise. A failure to reject therefore does not
establish the identifying restriction.

Pre-treatment level contrasts and dose-slope contrasts emphasize different
features of observed untreated changes. They cannot isolate the extra
restriction that strong parallel trends places on treated potential outcomes.
Those counterfactual treated paths are not observed before adoption.
The same limitation applies when a pre-treatment event-study slope looks
close to zero.

The :ref:`continuous treatment example <example_cont_did>` shows the
reported level and slope curves alongside their summaries. Read the slope
as a causal response only after deciding which population the maintained
parallel trends assumption lets each dose group represent.
