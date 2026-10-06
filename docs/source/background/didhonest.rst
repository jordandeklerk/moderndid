.. _background-didhonest:

Sensitivity to departures from parallel trends
==============================================

An event study can show little evidence of a pre-treatment difference
without ruling out a difference large enough to change your conclusion,
particularly when the estimates are imprecise. Even a precisely estimated
pre-treatment path leaves us to decide how that path would continue after
treatment.

The approach of `Rambachan and Roth (2023)
<https://doi.org/10.1093/restud/rdad018>`_ makes that decision explicit.
We restrict how the untreated difference between groups can evolve,
then ask which treatment effects remain compatible with the restriction.
The confidence sets account for uncertainty in both the estimated event
study and the counterfactual path. The
`published paper <https://www.jonathandroth.com/assets/files/HonestParallelTrends_Main.pdf>`_
provides the assumptions and results developed below.

In ModernDiD, :func:`~moderndid.honest_did` takes an estimated event
study into this calculation. To use the lower-level sensitivity functions,
you supply a coefficient vector and its covariance matrix from an existing
analysis. Choosing the original DiD design and deciding which departures
from parallel trends are plausible remain part of your application.

What the event-study coefficients measure
------------------------------------------

We begin with estimates that have a causal interpretation under parallel
trends, since sensitivity analysis cannot repair the mixing of effects
across cohorts and event times in a single-coefficient two-way fixed
effects event study. For staggered adoption, first use an estimator whose
effects and comparison groups match your target, such as the one in the
:ref:`staggered DiD background <background-did>`.

Let :math:`T_{pre}` and :math:`T_{post}` count the estimated pre-treatment
and post-treatment coefficients. Stack them in chronological order,

.. math::

   \widehat\beta=(\widehat\beta_{pre}',\widehat\beta_{post}')'
   \in\mathbb{R}^{T_{pre}+T_{post}}.

Every coefficient uses one common untreated reference period. Following
the paper, we label that omitted period :math:`0` and the first
post-treatment period :math:`1`. The estimated coefficients correspond
to :math:`-T_{pre},\ldots,-1,1,\ldots,T_{post}`. This numbering describes
the sensitivity model rather than the event-time labels in your original
analysis. With no anticipation, period :math:`1` here corresponds to
event time zero in a ModernDiD event study.

.. admonition:: Keep a common reference period
   :class: tip

   Estimate :func:`~moderndid.att_gt` with ``base_period="universal"``
   before forming the dynamic aggregation passed to
   :func:`~moderndid.honest_did`. Adjacent-period pre-treatment contrasts
   describe changes between successive periods rather than levels
   relative to the reference period used by these restrictions.

Write :math:`\beta` for the population coefficient vector. Its
post-treatment entries combine treatment effects and the untreated
difference that the comparison failed to remove.

.. admonition:: Assumption 1 Causal decomposition
   :class: assumption

   There are treatment-effect and untreated-difference vectors
   :math:`\tau` and :math:`\delta` such that

   .. math::

      \beta=\tau+\delta,
      \qquad
      \tau=\begin{pmatrix}0_{T_{pre}}\\\tau_{post}\end{pmatrix},
      \qquad
      \delta=\begin{pmatrix}\delta_{pre}\\\delta_{post}\end{pmatrix}.

   Thus :math:`\tau_{pre}=0`. Pre-treatment coefficients measure the
   untreated difference rather than a causal response to treatment.

In a two-group design, :math:`\delta_t` is the treated-minus-comparison
untreated outcome difference in period :math:`t`, relative to that
difference in the reference period. After we normalize :math:`\delta_0=0`,
its first difference describes a departure from parallel trends.
Exact post-treatment parallel trends would therefore give
:math:`\delta_{post}=0` and identify :math:`\tau_{post}=\beta_{post}`.

The zero pre-treatment causal response is a separate requirement.
If units respond before adoption, exclude the affected periods from
the pre-treatment part and choose an unaffected reference. The
``honest_did`` wrapper uses the reference recorded in the event study,
including the shift implied by its anticipation setting.

A pre-test examines whether the estimated pre-treatment differences
are distinguishable from zero without establishing how the counterfactual
post-treatment path would evolve. Selecting an analysis because its
pre-test did not reject can also change the distribution of subsequent
estimates and intervals. The direction of that distortion depends on the
design and the underlying departures from parallel trends.

From one effect to an identified set
------------------------------------

Suppose you want the first post-treatment effect or an average over
several post-treatment periods. Represent that choice by a fixed
nonzero vector :math:`\ell\in\mathbb{R}^{T_{post}}`,

.. math::

   \theta=\ell'\tau_{post}.

A basis vector selects one period and equal entries summing to one select
an average. The ``l_vec`` argument of
:func:`~moderndid.create_sensitivity_results_sm` and
:func:`~moderndid.create_sensitivity_results_rm` makes this choice
explicit. The ``event_time`` argument of ``honest_did`` selects a
single reported post-treatment event time.

Without a restriction on :math:`\delta_{post}`, the population
coefficients do not determine :math:`\theta`. Let :math:`\Delta`
contain the untreated-difference paths you are prepared to allow. The
identified set collects every treatment effect compatible with those
paths and the population coefficients,

.. math::

   \mathcal S(\beta,\Delta)
   =\left\{\ell'\beta_{post}-\ell'\delta_{post}:
       \delta\in\Delta,\ \delta_{pre}=\beta_{pre}\right\}.

The equality on the pre-treatment coordinates follows from Assumption 1
and uses population coefficients rather than treating noisy estimates
as the true pre-treatment path. Constructing a confidence set therefore
requires us to account for uncertainty in that estimated path.

For a nonempty polyhedral restriction with finite extrema, the bounds
come from two linear programs,

.. math::

   \begin{aligned}
   \theta^{lb}(\beta,\Delta)
      &=\ell'\beta_{post}
        -\max_{\delta\in\Delta:\delta_{pre}=\beta_{pre}}
             \ell'\delta_{post},\\
   \theta^{ub}(\beta,\Delta)
      &=\ell'\beta_{post}
        -\min_{\delta\in\Delta:\delta_{pre}=\beta_{pre}}
             \ell'\delta_{post}.
   \end{aligned}

For a convex restriction, every value between these bounds is also
compatible with the population coefficients. More general sets require
an infimum and supremum when extrema are not attained.
The identified set can be empty if the restriction contradicts the
population pre-treatment path, or unbounded if it leaves the target
unrestricted. Sampling uncertainty does not remove these distinctions.

Choosing a restriction on the untreated path
---------------------------------------------

The restriction should describe the threat to identification you have
in mind. A bound on changes may fit a concern about differential shocks.
A bound on curvature may fit a concern about a secular trend that
continues after treatment. We keep these choices separate because they
allow different counterfactual paths and use different units.

Relative magnitudes of changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A relative-magnitude restriction compares post-treatment departures from
parallel trends with the largest pre-treatment departure. It constrains
first differences of :math:`\delta`, rather than the levels of the
normalized event-study coefficients.

.. admonition:: Relative-magnitude restriction
   :class: assumption

   For a chosen :math:`\bar M\geq0`, assume
   :math:`\delta\in\Delta^{RM}(\bar M)`, where

   .. math::

      \begin{aligned}
      \Delta^{RM}(\bar M)=\biggl\{\delta:
      &|\delta_{t+1}-\delta_t|\\
      &\leq\bar M\max_{s=-T_{pre},\ldots,-1}
                    |\delta_{s+1}-\delta_s|,\quad
        t=0,\ldots,T_{post}-1\biggr\},
      \qquad \delta_0=0.
      \end{aligned}

At :math:`\bar M=1`, every allowed post-treatment slope is no larger
in absolute value than the largest pre-treatment slope. Increasing the
factor allows larger departures from the untreated trend.
Because this scale comes from the population pre-treatment path, inference
must account for uncertainty in that path rather than substitute the
largest estimated slope as known.

:func:`~moderndid.create_sensitivity_results_rm` uses this restriction
by default through ``bound="deviation from parallel trends"``.
The parameter ``m_bar_vec`` specifies the values of :math:`\bar M`
used in the sensitivity analysis.

Smoothness of the differential trend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A linear untreated difference can have a nonzero slope while remaining
perfectly smooth. If that is the plausible confounder, bounding the
change in its slope is more direct than bounding its level.

.. admonition:: Smoothness restriction
   :class: assumption

   For a chosen :math:`M\geq0`, assume
   :math:`\delta\in\Delta^{SD}(M)`, where

   .. math::

      \begin{aligned}
      \Delta^{SD}(M)=\bigl\{\delta:
      &|\delta_{t+1}-2\delta_t+\delta_{t-1}|\leq M,\\
      &t=-T_{pre}+1,\ldots,T_{post}-1\bigr\},
      \qquad \delta_0=0.
      \end{aligned}

Setting :math:`M=0` restricts the untreated difference to a linear
path that may still have a nonzero slope, rather than imposing parallel
trends. Positive :math:`M` allows that slope to change by at most
:math:`M` each period. Because the bound has the outcome's units per
squared observation period, its meaning depends on the spacing and scale
of your data.

:func:`~moderndid.create_sensitivity_results_sm` implements this
restriction. Its ``m_vec`` argument supplies the smoothness bounds
rather than the relative-magnitude factors used by ``m_bar_vec``.

Relative magnitudes of curvature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also allow curvature after treatment to scale with curvature
before treatment. This uses the same economic idea as a smoothness
restriction but calibrates its magnitude from the pre-treatment path.

.. admonition:: Relative-curvature restriction
   :class: assumption

   For :math:`\bar M\geq0`, assume
   :math:`\delta\in\Delta^{SDRM}(\bar M)`, where

   .. math::

      \begin{aligned}
      \Delta^{SDRM}(\bar M)=\biggl\{\delta:
      &|\delta_{t+1}-2\delta_t+\delta_{t-1}|\\
      &\leq\bar M\max_{s=-T_{pre}+1,\ldots,-1}
                    |\delta_{s+1}-2\delta_s+\delta_{s-1}|,\\
      &t=0,\ldots,T_{post}-1\biggr\},
      \qquad\delta_0=0.
      \end{aligned}

This restriction needs enough pre-treatment periods to measure
curvature. Passing ``bound="deviation from linear trend"`` to
``create_sensitivity_results_rm`` selects the relative-curvature
calculation. The current implementation requires at least three
estimated pre-treatment coefficients for that option.

Sign and monotonicity restrictions can further restrict the allowed
paths when your application supports them. For example, a positive
post-treatment bias imposes :math:`\delta_t\geq0` after treatment.
Because these restrictions add identifying information, they should follow
from the economic concern rather than from the sign of the estimate.
The ``bias_direction`` and ``monotonicity_direction`` arguments select
supported restrictions in the sensitivity functions.

Why the geometry determines the calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The smoothness class is a polyhedron because each absolute-value bound
can be written as two linear inequalities. Relative magnitudes also
require choosing which pre-treatment difference attains the maximum
and its sign. Enumerating those choices produces a finite union of
polyhedra,

.. math::

   \Delta=\{\delta:A\delta\leq d\},
   \qquad\text{or}\qquad
   \Delta=\bigcup_{k=1}^{K}\Delta_k.

For a union, identification and inference preserve every admissible
component. In particular,

.. math::

   \mathcal S(\beta,\Delta)
      =\bigcup_{k=1}^{K}\mathcal S(\beta,\Delta_k),
   \qquad
   \mathcal C_{\alpha,n}(\Delta)
      =\bigcup_{k=1}^{K}\mathcal C_{\alpha,n}(\Delta_k).

If each component confidence set covers the true target with probability
at least :math:`1-\alpha` whenever its component is correct, their
union has the same lower coverage bound whenever the union restriction
is correct. Since the true path belongs to at least one component, this
argument does not require a multiple-testing adjustment across components.

Confidence sets that include sampling uncertainty
-------------------------------------------------

Although the identified set is a population object, estimating its endpoints
introduces uncertainty from both the pre-treatment and post-treatment
coefficients. Plugging the estimated pre-treatment path into the bounds
and then attaching an ordinary standard error would ignore part of
that uncertainty.

For the finite-sample calculations, consider the Gaussian model

.. math::

   \widehat\beta_n\sim N(\delta+L_{post}\tau_{post},V_n),
   \qquad
   L_{post}=\begin{pmatrix}0_{T_{pre}\times T_{post}}\\I_{T_{post}}\end{pmatrix}.

Here :math:`V_n` is the covariance of the coefficient estimator. The
``sigma`` argument takes this covariance, rather than standard errors
or the covariance of a :math:`\sqrt n`-scaled estimator. Exact normality
with known :math:`V_n` supports finite-sample statements in this model.
With estimated coefficients and covariance, the asymptotic conditions
below supply the corresponding large-sample justification.

The ``honest_did`` wrapper reconstructs covariance from unit-level
influence-function outer products. It does not aggregate those values
by cluster or carry over a bootstrap covariance from the original
event study. For clustered sensitivity analysis, call a lower-level
sensitivity function with a valid joint covariance for the coefficient
vector. The original event study's cluster setting alone does not
provide that covariance to the wrapper.

The coverage requirement is uniform over the allowed paths and causal
effects,

.. math::

   \inf_{\substack{\delta\in\Delta\\
                   \tau_{post}\in\mathbb{R}^{T_{post}}}}
   P_{\delta,\tau_{post}}
      \bigl(\ell'\tau_{post}\in\mathcal C_{\alpha,n}\bigr)
   \geq1-\alpha.

Uniform coverage applies to each admissible true value, without requiring
one confidence set to contain every point of the identified set
simultaneously with that probability. In a partially identified model,
covering the true effect and covering the entire identified set are
different requirements.

Testing a candidate effect through moment inequalities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To decide whether a candidate value :math:`\theta_0` belongs in the
confidence set, test whether some allowed untreated path could produce
it. For a polyhedron, the nuisance parameters enter the resulting
inequalities linearly.

Choose an invertible :math:`T_{post}\times T_{post}` matrix
:math:`B` whose first row is :math:`\ell'`. Partition
:math:`B\tau_{post}=(\theta,\nu')'` and write
:math:`A L_{post}B^{-1}=(a_\theta,X)`. At the candidate value, define

.. math::

   Y=A\widehat\beta_n-d-a_\theta\theta_0,
   \qquad\Omega=A V_n A'.

The null hypothesis requires a :math:`\nu` satisfying
:math:`\mathbb E[Y]-X\nu\leq0`. Because the covariance :math:`\Omega`
does not depend on that nuisance parameter, the problem has the linear
structure used by `Andrews, Roth, and Pakes (2023)
<https://doi.org/10.1093/restud/rdac034>`_ and implemented by the
conditional inference routines.

Let :math:`\sigma_j=\sqrt{\Omega_{jj}}` and collect them in
:math:`\sigma`. The profiled maximum measures how far the inequalities
are from being jointly satisfied,

.. math::

   \widehat\eta
      =\min_\nu\max_j\frac{Y_j-(X\nu)_j}{\sigma_j}
      =\min_{\eta,\nu}\{\eta:Y-X\nu\leq\eta\sigma\}.

Large positive values contradict the candidate effect. When its dual
feasible set is nonempty, the same linear program has the representation

.. math::

   \widehat\eta=\max_{\gamma\in\mathcal G(\sigma)}\gamma'Y,
   \qquad
   \mathcal G(\sigma)
      =\{\gamma\geq0:\gamma'X=0,\ \gamma'\sigma=1\}.

The maximum occurs at a vertex of this polyhedron, whose vertices we
denote by :math:`\mathcal V(\sigma)`. An optimizing vertex
:math:`\widehat\gamma` selects a combination of moment inequalities
rather than an alternative treatment-effect estimator.

Least-favorable, conditional, and hybrid tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The least-favorable test calibrates the maximum as though all relevant
inequalities were binding. Its critical value is the quantile of a
random variable, rather than the random variable itself,

.. math::

   c_{\alpha,LF}
      =q_{1-\alpha}\left(
          \max_{\gamma\in\mathcal V(\sigma)}\gamma'Z\right),
   \qquad Z\sim N(0,\Omega).

This controls rejection under the null but can be conservative when
some inequalities are far from binding. The conditional test instead
conditions on the selected vertex and a residual that removes the
variation along that vertex,

.. math::

   S_\gamma
      =\left(I-\frac{\Omega\gamma\gamma'}{\gamma'\Omega\gamma}\right)Y.

For a selected vertex with positive variance, conditional on
:math:`\widehat\gamma=\gamma` and :math:`S_\gamma=s`, the Gaussian
model gives

.. math::

   \widehat\eta\mid\{\widehat\gamma=\gamma,S_\gamma=s\}
   \sim TN\bigl(\gamma'\mathbb E[Y],\gamma'\Omega\gamma,[L,U]\bigr).

Here :math:`TN` denotes a normal distribution truncated to the interval
:math:`[L,U]`. Selection of the maximizing vertex determines its bounds,

.. math::

   \begin{aligned}
   L&=\max_{\substack{\widetilde\gamma\in\mathcal V(\sigma)\\
              \gamma'\Omega\gamma>\gamma'\Omega\widetilde\gamma}}
      \frac{\gamma'\Omega\gamma\,\widetilde\gamma's}
           {\gamma'\Omega\gamma-\gamma'\Omega\widetilde\gamma},\\
   U&=\min_{\substack{\widetilde\gamma\in\mathcal V(\sigma)\\
              \gamma'\Omega\gamma<\gamma'\Omega\widetilde\gamma}}
      \frac{\gamma'\Omega\gamma\,\widetilde\gamma's}
           {\gamma'\Omega\gamma-\gamma'\Omega\widetilde\gamma}.
   \end{aligned}

Empty lower and upper index sets give :math:`-\infty` and
:math:`\infty`, respectively. Under the null, every feasible vertex
has :math:`\gamma'\mathbb E[Y]\leq0`. The conditional critical
value uses mean zero as the least-favorable mean of the truncated
normal distribution.

The LF hybrid first applies a size-:math:`\kappa` least-favorable test,
with :math:`0<\kappa<\alpha`. If that test does not reject, it applies
a conditional test at adjusted size :math:`(\alpha-\kappa)/(1-\kappa)`
and conditions on the first-stage nonrejection. Its upper truncation
point becomes

.. math::

   U_H=\min\{U,c_{\kappa,LF}\}.

The adjustment allocates the overall rejection probability across
these two stages. The paper uses :math:`\kappa=\alpha/10` as a
benchmark. ``method="Conditional"`` selects the conditional procedure
and ``method="C-LF"`` selects the LF hybrid in the sensitivity functions.

Conditions behind uniform validity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A Gaussian approximation at one particular data-generating process is
weaker than validity across every process allowed by the restriction.
We now state the conditions the paper uses for that uniform result.
Fix a nonempty :math:`\Delta=\{\delta:A\delta\leq d\}` with
nonzero rows of :math:`A` and a fixed :math:`\ell\ne0`. Let
:math:`\mathcal P` contain the laws satisfying Assumption 1 with
:math:`\delta_P\in\Delta`.

.. admonition:: Assumptions 2 through 5 Uniform inference
   :class: assumption

   Let :math:`BL_1` contain the functions bounded by one in absolute
   value with Lipschitz constant at most one. Assumption 2 requires

   .. math::

      \lim_{n\to\infty}\sup_{P\in\mathcal P}\sup_{f\in BL_1}
      \left|\mathbb E_P f\bigl(\sqrt n(\widehat\beta_n-\beta_P)\bigr)
           -\mathbb E f(\xi_P)\right|=0,
      \qquad\xi_P\sim N(0,\Sigma_P).

   Assumption 3 requires constants
   :math:`0<\underline\lambda\leq\overline\lambda<\infty` such
   that every eigenvalue of every :math:`\Sigma_P` lies between
   these constants. Denote that matrix class by :math:`\mathcal S`.

   For Assumption 4, the estimator :math:`\widehat\Sigma_n` of this
   limiting covariance satisfies, for every :math:`\varepsilon>0`,

   .. math::

      \lim_{n\to\infty}\sup_{P\in\mathcal P}
      P_P(\|\widehat\Sigma_n-\Sigma_P\|>\varepsilon)=0.

   Assumption 5 requires at least one of two conditions on
   :math:`A`. In part A, for :math:`k_1+k_2=\dim(\delta)`,
   write :math:`A=TQ` with :math:`Q` of full row rank and

   .. math::

      T=\begin{pmatrix}I_{k_1}&0\\-I_{k_1}&0\\0&I_{k_2}\end{pmatrix}.

   Zero-dimensional blocks are allowed. In part B, let
   :math:`\bar\gamma_1,\ldots,\bar\gamma_K` be the vertices of
   :math:`\mathcal G(\mathbf1)`. For every :math:`k`, require

   .. math::

      \bar\gamma_k'A=0
      \quad\text{or}\quad
      \inf_{a\geq0}\inf_{j\ne k}
         \|(\bar\gamma_k-a\bar\gamma_j)'A\|>0.

Assumption 5 controls degeneracy in the moment problem without requiring
a unique solution for the identified-set endpoints. In the
large-sample result, the covariance supplied to the test is
:math:`\widehat V_n=\widehat\Sigma_n/n`.

.. admonition:: Proposition 3.1 Uniform size control
   :class: theorem

   Under Assumptions 2 through 5, for :math:`0<\alpha<1/2` and
   :math:`0<\kappa<\alpha`, the conditional and LF-hybrid tests
   have rejection indicators :math:`\psi^C_\alpha` and
   :math:`\psi^{C-LF}_{\kappa,\alpha}` satisfying

   .. math::

      \begin{aligned}
      \limsup_{n\to\infty}\sup_{P\in\mathcal P}
      \mathbb E_P\psi^C_\alpha
         (\widehat\beta_n,A,d,\theta_P,\widehat\Sigma_n/n)
         &\leq\alpha,\\
      \limsup_{n\to\infty}\sup_{P\in\mathcal P}
      \mathbb E_P\psi^{C-LF}_{\kappa,\alpha}
         (\widehat\beta_n,A,d,\theta_P,\widehat\Sigma_n/n)
         &\leq\alpha.
      \end{aligned}

Inverting either test gives uniform asymptotic coverage of the true
:math:`\theta_P` over the class of laws whose untreated paths belong
to the restriction you chose. Whether that economic restriction is
credible remains a separate judgment about your application.

What consistency and local power add
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Size control limits false rejection of an admissible effect. Consistency
adds that a fixed effect outside the identified set will eventually be
rejected. The latter result needs stronger assumptions on the joint
behavior of the coefficient and covariance estimators.

.. admonition:: Assumptions 6 and 7 Joint estimation uncertainty
   :class: assumption

   Assumption 6 requires uniform Gaussian convergence of

   .. math::

      W_n=\begin{pmatrix}
         \widehat\beta_n-\beta_P\\
         \operatorname{vec}(\widehat\Sigma_n-\Sigma_P)
      \end{pmatrix}

   in the same bounded-Lipschitz metric,

   .. math::

      \lim_{n\to\infty}\sup_{P\in\mathcal P}\sup_{f\in BL_1}
      |\mathbb E_P f(\sqrt nW_n)-\mathbb E f(\xi_P^+)|=0,
      \qquad\xi_P^+\sim N(0,V_P),

   where

   .. math::

      V_P=\begin{pmatrix}
         \Sigma_P&V_{P,\beta\Sigma}\\
         V_{P,\Sigma\beta}&V_{P,\Sigma}
      \end{pmatrix}.

   For Assumption 7, :math:`\Sigma_P\in\mathcal S`, the matrices
   :math:`V_P` lie in a compact set. Every eigenvalue of

   .. math::

      \Sigma_P-V_{P,\beta\Sigma}V_{P,\Sigma}^{\dagger}
                   V_{P,\Sigma\beta}

   is at least a common positive constant. The dagger denotes the
   Moore--Penrose inverse.

The joint limit rules out a coefficient estimation error determined
entirely by the covariance estimation error. Under these conditions,
fixed departures beyond either identified-set endpoint become detectable.

.. admonition:: Proposition 3.2 Uniform consistency
   :class: theorem

   Under Assumptions 4 through 7, for every :math:`x>0` and
   :math:`0<\alpha<1/2`, both tests satisfy

   .. math::

      \begin{aligned}
      \lim_{n\to\infty}\inf_{P\in\mathcal P}
         \mathbb E_P\psi^C_\alpha
         (\widehat\beta_n,A,d,\theta_P^{ub}+x,\widehat\Sigma_n/n)&=1,\\
      \lim_{n\to\infty}\inf_{P\in\mathcal P}
         \mathbb E_P\psi^{C-LF}_{\kappa,\alpha}
         (\widehat\beta_n,A,d,\theta_P^{ub}+x,\widehat\Sigma_n/n)&=1.
      \end{aligned}

   The same limits hold for candidates :math:`\theta_P^{lb}-x`.

For local alternatives just :math:`x/\sqrt n` beyond an endpoint,
the geometry of the binding constraints determines power. Let
:math:`\tau_{post}^*` solve

.. math::

   \max_{\tau_{post}}\ell'\tau_{post}
   \quad\text{subject to}\quad
   -A_{(\cdot,post)}\tau_{post}\leq d-A\beta.

If :math:`B^*` indexes its binding constraints, the paper's linear
independence constraint qualification, or LICQ, requires that some
optimizer has :math:`-A_{(B^*,post)}` of full row rank. It is enough
for this condition to hold at one optimizer, rather than at every optimizer.

For :math:`\varepsilon>0`, let :math:`\mathcal P_\varepsilon`
contain laws satisfying LICQ in direction :math:`\ell` with the
nonbinding constraints slack by at least :math:`\varepsilon`.
Write :math:`\mathcal I_\alpha(\Delta,\Sigma_P/n)` for confidence
sets satisfying the Gaussian coverage requirement, and define the
local power envelope

.. math::

   \rho_\alpha^*(P,x)
      =\lim_{n\to\infty}\sup_{C\in\mathcal I_\alpha(\Delta,\Sigma_P/n)}
        P_{\widehat\beta_n\sim N(\beta_P,\Sigma_P/n)}
         (\theta_P^{ub}+x/\sqrt n\notin C).

.. admonition:: Proposition 3.3 Conditional local power
   :class: theorem

   Under Assumptions 2 through 4, for every
   :math:`\varepsilon>0`, :math:`x>0`, and :math:`0<\alpha<1/2`,

   .. math::

      \lim_{n\to\infty}\sup_{P\in\mathcal P_\varepsilon}
      \left|\mathbb E_P\psi^C_\alpha
         (\widehat\beta_n,A,d,\theta_P^{ub}+x/\sqrt n,
          \widehat\Sigma_n/n)-\rho_\alpha^*(P,x)\right|=0.

   The lower-endpoint result uses LICQ in direction :math:`-\ell`.
   Under these conditions, Corollary 3.1 gives the LF hybrid the
   lower local-power bound

   .. math::

      \liminf_{n\to\infty}\inf_{P\in\mathcal P_\varepsilon}
      \left[\mathbb E_P\psi^{C-LF}_{\kappa,\alpha}
       (\widehat\beta_n,A,d,\theta_P^{ub}+x/\sqrt n,
        \widehat\Sigma_n/n)
       -\rho^*_{(\alpha-\kappa)/(1-\kappa)}(P,x)\right]\geq0.

LICQ is needed for this power comparison, rather than for the earlier
uniform size result. A failure of LICQ does not by itself invalidate
the conditional or hybrid confidence set.

Fixed-length intervals and worst-case bias
------------------------------------------

The moment-inequality approach adapts to the estimated pre-treatment
path. A fixed-length confidence interval instead chooses its length
to cover the worst allowable bias before observing that path. This can
be attractive under smoothness restrictions when sampling uncertainty
is large relative to identification uncertainty.

For fixed covariance :math:`V_n`, consider the affine estimator
:math:`a+v'\widehat\beta_n` and the interval

.. math::

   C_{\alpha,n}(a,v,\chi)
      =[a+v'\widehat\beta_n-\chi,
        a+v'\widehat\beta_n+\chi].

Since the post-treatment effects are unrestricted, a finite worst-case
bias requires :math:`v_{post}=\ell`. Under that requirement,

.. math::

   \begin{aligned}
   \bar b(a,v)
      &=\sup_{\substack{\delta\in\Delta\\
                  \tau_{post}\in\mathbb R^{T_{post}}}}
          |a+v'(\delta+L_{post}\tau_{post})-\ell'\tau_{post}|\\
      &=\sup_{\delta\in\Delta}|a+v'\delta|,
      \qquad v_{post}=\ell.
   \end{aligned}

Both the pre-treatment adjustment and the post-treatment untreated
difference enter that bias. The restriction must bound their combined
contribution to the affine estimator's error.

Let :math:`cv_\alpha(t)` be the :math:`1-\alpha` quantile of
:math:`|N(t,1)|`. With :math:`\sigma_{v,n}=\sqrt{v'V_nv}`, the
smallest valid half-length for fixed :math:`a,v` is

.. math::

   \chi_n(a,v;\alpha)
      =\sigma_{v,n}\,
         cv_\alpha\!\left(\bar b(a,v)/\sigma_{v,n}\right).

Minimizing this expression over :math:`a,v` gives the optimal affine
fixed-length interval. :func:`~moderndid.compute_flci` performs this
calculation for the smoothness class. Its length is fixed conditional
on the supplied covariance and restriction, rather than identical
across applications or estimated covariance matrices.

Finite-sample length comparisons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The paper's near-optimality result compares this interval with every
confidence set satisfying Gaussian coverage. The comparison applies
under a specific symmetry condition at the true untreated path.

.. admonition:: Assumption 8 Symmetry at the true path
   :class: assumption

   The set :math:`\Delta` is convex and centrosymmetric,
   so :math:`\widetilde\delta\in\Delta` implies
   :math:`-\widetilde\delta\in\Delta`. The true
   :math:`\delta\in\Delta` also satisfies
   :math:`\widetilde\delta-\delta\in\Delta` for every
   :math:`\widetilde\delta\in\Delta`.

This symmetry allows the paper to compare the fixed-length interval
with all valid confidence sets, including sets whose lengths adapt
to the estimated pre-treatment path.

.. admonition:: Proposition 4.1 Near-optimal expected length
   :class: theorem

   Under Assumption 8, for any :math:`\tau` with
   :math:`\tau_{pre}=0` and positive definite :math:`V_n`, let
   :math:`\chi_n` be the optimal fixed half-length. Then

   .. math::

      \frac{\displaystyle\inf_{C\in\mathcal I_\alpha(\Delta,V_n)}
             \mathbb E_{\widehat\beta_n\sim N(\delta+\tau,V_n)}
                          [\lambda(C)]}{2\chi_n}
      \geq
      \frac{z_{1-\alpha}(1-\alpha)
             -\widetilde z_\alpha\Phi(\widetilde z_\alpha)
             +\phi(z_{1-\alpha})-\phi(\widetilde z_\alpha)}
           {z_{1-\alpha/2}},
      \qquad
      \widetilde z_\alpha=z_{1-\alpha}-z_{1-\alpha/2}.

   Here :math:`\lambda` is Lebesgue length, :math:`z_p` is a
   standard normal quantile, and :math:`\Phi,\phi` are its
   distribution function and density.

At :math:`\alpha=0.05`, the lower bound is about :math:`0.72`.
Under the stated conditions, no uniformly valid confidence set can
improve expected length by more than about 28 percent relative to
the optimal fixed-length interval. This is a result in the exact
Gaussian model with known covariance, rather than an unconditional
finite-sample guarantee for an estimated event study.

Although the smoothness class is convex and centrosymmetric, the true
path must also satisfy Assumption 8 for this comparison to apply. A linear
untreated path in that class satisfies the condition, including the zero
path. Sign restrictions and relative-magnitude classes do not inherit
the same comparison.

When a fixed-length interval cannot adapt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A uniformly valid fixed length must accommodate the largest identified
set allowed anywhere in the restriction class. If the true path produces
a shorter identified set, shrinking sampling error does not necessarily
make the interval shrink to that shorter set.

.. admonition:: Assumption 9 Maximal finite identified-set length
   :class: assumption

   Let :math:`L_{ID}(\delta_{pre},\Delta)` be the length of
   the identified interval at that pre-treatment path, and let

   .. math::

      \Delta_{pre}
         =\{\delta_{pre}:\exists\delta_{post},\
                          (\delta_{pre}',\delta_{post}')'\in\Delta\}.

   The true :math:`\delta\in\Delta` satisfies

   .. math::

      L_{ID}(\delta_{pre},\Delta)
         =\sup_{\widetilde\delta_{pre}\in\Delta_{pre}}
             L_{ID}(\widetilde\delta_{pre},\Delta)<\infty.

The next result characterizes consistency through that maximal-length
condition. It keeps the economic restriction fixed as sampling uncertainty
converges to zero.

.. admonition:: Proposition 4.2 Consistency of fixed-length intervals
   :class: theorem

   Suppose :math:`\Delta` is convex and :math:`0<\alpha<1/2`.
   Fix :math:`\delta\in\Delta` and :math:`\tau_{pre}=0` such
   that :math:`\mathcal S(\delta+\tau,\Delta)\ne\mathbb R`.
   For positive definite :math:`\Sigma^*` and
   :math:`V_n=\Sigma^*/n`, Assumption 9 holds if and only if
   the optimal fixed-length interval satisfies

   .. math::

      \lim_{n\to\infty}
      P_{\widehat\beta_n\sim N(\delta+\tau,V_n)}
      (\theta^{out}\in C^{FLCI}_{\alpha,n})=0
      \quad\text{for every }\theta^{out}\notin
                       \mathcal S(\delta+\tau,\Delta).

Constant finite identified-set length is sufficient for this consistency
condition to hold everywhere in the class. In the three-period
smoothness example for the first post-treatment effect, that length
is :math:`2M`. For relative magnitudes with :math:`\bar M>0` and
:math:`\theta=\tau_1`, every affine estimator has infinite worst-case
bias and the only uniformly valid fixed-length interval is the entire
real line. The relative-magnitude functions therefore use conditional or
hybrid inference rather than an ordinary finite FLCI.

Choosing the method and reading a sensitivity analysis
------------------------------------------------------

For smoothness without sign or monotonicity restrictions, the sensitivity
functions default to ``method="FLCI"``. With those restrictions, the
default switches to ``"C-F"``, the conditional FLCI hybrid. An explicitly
requested FLCI does not use the added sign or shape information. Relative
magnitudes default to ``"C-LF"`` and also support ``"Conditional"``.
The LF-hybrid results above concern ``"C-LF"``; ``"C-F"`` conditions
on a first-stage FLCI screening event instead.

The conditional methods invert tests over a finite grid of candidate
values. Because a returned lower or upper bound is a point on that grid,
its resolution and range remain part of the numerical calculation even
when the statistical procedure has valid theoretical coverage.

.. admonition:: Check the inversion grid
   :class: warning

   If accepted candidates reach ``grid_lb`` or ``grid_ub``, the reported
   endpoint may reflect the search limit. Widen the range before
   interpreting it as a bound on the effect. Increasing ``grid_points``
   refines resolution but does not widen a fixed range.

We use a sequence of :math:`M` or :math:`\bar M` values to show which
conclusions survive as the restriction relaxes. A breakdown value for
a null :math:`\theta_0` is

.. math::

   M^*=\inf\{M\geq0:\theta_0\in\mathcal C_\alpha(M)\}.

For relative magnitudes, replace :math:`M` by :math:`\bar M`.
A calculation on a finite parameter grid locates the crossing only to
that grid's resolution. If no crossing occurs, the evidence supports
nonrejection or rejection over the evaluated range rather than a known
breakdown value outside it. Different choices of :math:`\ell`,
restriction class, and confidence method can give different crossings.

To interpret the magnitude economically, read a smoothness bound as
an allowed change in slope in the outcome's units and a relative-magnitude
factor as a scale for the largest pre-treatment departure. Neither is a
probability that parallel trends fails or a data-estimated limit on
possible confounding.

The paper also considers restrictions conditional on covariates that
can sharpen bounds when those stronger restrictions are credible.
Covariate adjustment alone does not establish them and the
``honest_did`` wrapper does not infer them from unit-level covariate columns.
For a conditional analysis, the input coefficient vector and its joint
covariance must represent the target you intend to study.

The :ref:`sensitivity analysis example <example_honest_did>` takes
an event study through these choices, compares the restrictions, and
shows how the confidence intervals change. It provides a concrete
setting for deciding which departures would be consequential for the
question your DiD design answers.
