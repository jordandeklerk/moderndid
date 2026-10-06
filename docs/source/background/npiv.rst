.. _background-npiv:

Nonparametric instrumental variables
====================================

A flexible regression can describe how an outcome varies with a regressor
without recovering the structural relationship you want. When unobserved
determinants of the outcome also influence that regressor, the conditional
mean of the outcome can differ from the structural function.

An instrument connects the structural function to outcome variation
through a restriction that does not require the regressor to be exogenous.
Recovering a whole function from that restriction can be harder than
estimating a linear IV coefficient because the difficulty depends on which
features of the function the instruments reveal.

We examine how the instrument restriction identifies the function before
turning to sieve estimation, dimension selection, and uncertainty over the
function and its derivatives.
The methods behind :func:`~moderndid.npiv` follow
`Chen, Christensen, and Kankanala (2024)
<https://arxiv.org/abs/2107.11869>`_. Their
`author manuscript <https://arxiv.org/pdf/2107.11869>`_ gives the
procedures in Section 2 and the formal results in Section 4.
The assumption and theorem numbers below refer to that paper.
The :ref:`nonparametric IV example <example_npiv>` applies the method to
an Engel curve and a simulation.

What the instrument restriction identifies
------------------------------------------

Let :math:`Y` be a scalar outcome, :math:`X\in\mathbb R^d` a vector
of possibly endogenous regressors, and :math:`W\in\mathbb R^{d_w}`
a vector of instruments. We observe independent, identically distributed
vectors :math:`(X_i,Y_i,W_i)` for :math:`i=1,\ldots,n`.
The structural function :math:`h_0` satisfies

.. math::

   Y=h_0(X)+u,\qquad \mathbb E[u\mid W]=0\quad\text{almost surely}.

The error :math:`u` may have a nonzero conditional mean given
:math:`X`. Regressing :math:`Y` directly on :math:`X` would then
recover :math:`\mathbb E[Y\mid X]` rather than :math:`h_0`.
When :math:`W=X`, the restriction instead becomes ordinary
nonparametric regression.

To express identification, let :math:`L_X^2` and :math:`L_W^2`
be the spaces of functions with finite second moments under the
regressor and instrument distributions. Define the conditional expectation
operator

.. math::

   T:L_X^2\longrightarrow L_W^2,\qquad
   (Th)(w)=\mathbb E[h(X)\mid W=w].

The observed conditional mean gives :math:`Th_0=\mathbb E[Y\mid W]`
without distinguishing two different functions with the same conditional
expectation given the instrument. Identification therefore requires
injectivity to rule out that ambiguity.

.. admonition:: Assumption 1 Support and identification
   :class: assumption

   The support of :math:`X` is :math:`\mathcal X=[0,1]^d`.
   Its Lebesgue density satisfies, for some finite :math:`a_f>0`,

   .. math::

      a_f^{-1}<f_X(x)<a_f,\qquad x\in\mathcal X.

   The support of :math:`W` is :math:`\mathcal W=[0,1]^{d_w}`,
   and :math:`a_f^{-1}<f_W(w)<a_f` on :math:`\mathcal W`.
   Finally, :math:`T` is injective,

   .. math::

      Th=0\ \text{almost surely}
      \quad\Longrightarrow\quad
      h=0\ \text{almost surely},\qquad h\in L_X^2.

The unit-cube supports normalize the theory to a scale that rectangular
supports can reach through a change of units.
Neither rescaling nor including more instrument basis functions establishes
injectivity. It is a restriction on the conditional distribution of the
regressors given the instruments.

.. admonition:: Assumption 2 Error moments
   :class: assumption

   There are finite positive constants
   :math:`\underline\sigma,\overline\sigma` such that

   .. math::

      \mathbb E[u^4\mid W]\leq\overline\sigma^2,\qquad
      \mathbb E[u^2\mid W]\geq\underline\sigma^2
      \quad\text{almost surely}.

By bounding fourth moments and preventing conditional variance from
vanishing, these conditions still permit heteroskedasticity and
non-Gaussian errors.
The iid sampling setup concerns independent observations rather than a
clustered or serially dependent sample.

Approximating the function with a sieve
---------------------------------------

To make the infinite-dimensional recovery problem estimable, a sieve
replaces the unrestricted function with a growing space of finite-dimensional
approximations. We use :math:`J` basis functions of :math:`X`
and :math:`K` basis functions of :math:`W`,

.. math::

   \psi^J(x)=(\psi_{J1}(x),\ldots,\psi_{JJ}(x))',
   \qquad b^K(w)=(b_{K1}(w),\ldots,b_{KK}(w))'.

Using a prime to denote transpose, we write the candidate structural
function as :math:`\psi^J(x)'c_J` and substitute that approximation
into the model,

.. math::

   Y=\psi^J(X)'c_J+
      \bigl(h_0(X)-\psi^J(X)'c_J\bigr)+u.

Because the approximation error in the middle generally does not have
conditional mean zero given the instruments, the IV approximation becomes
accurate only as that term becomes sufficiently small.

How two-stage least squares estimates the coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate the bases at the sample observations to form matrices
:math:`\boldsymbol\Psi_J` of size :math:`n\times J`
and :math:`\mathbf B_K` of size :math:`n\times K`.
For the outcome vector :math:`\mathbf Y=(Y_1,\ldots,Y_n)'`, define

.. math::

   \mathbf P_K=\mathbf B_K(\mathbf B_K'\mathbf B_K)^{-}\mathbf B_K',
   \qquad
   \widehat c_J=
      (\boldsymbol\Psi_J'\mathbf P_K\boldsymbol\Psi_J)^{-}
      \boldsymbol\Psi_J'\mathbf P_K\mathbf Y.

The matrix :math:`\mathbf P_K` projects onto the instrument basis.
The superscript :math:`{}^{-}` denotes the Moore-Penrose inverse.
In the first stage, that projection determines the part of the regressor
basis explained by the instrument basis. The second stage estimates the
outcome relation from those projected basis functions.

At least :math:`K\geq J` is necessary for identifying all sieve
coefficients, although this dimension requirement alone does not ensure
full rank or useful instrument strength. A generalized inverse can return
a numerical solution when a matrix is singular without restoring information
missing from the data.

Write the coefficient map as

.. math::

   \mathbf M_J=
      (\boldsymbol\Psi_J'\mathbf P_{K(J)}\boldsymbol\Psi_J)^{-}
      \boldsymbol\Psi_J'\mathbf P_{K(J)}.

When we link the instrument dimension to :math:`J` through
:math:`K=K(J)`, the fitted function becomes
:math:`\widehat h_J(x)=\psi^J(x)'\mathbf M_J\mathbf Y`.
Because the same matrix maps outcome disturbances into estimation error,
it also appears in the variance and bootstrap calculations.

Derivatives and their units
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A derivative answers a question about changing a regressor while holding
the others fixed. For a multi-index :math:`a=(a_1,\ldots,a_d)`
of nonnegative integers, let :math:`|a|=\sum_j a_j` and define

.. math::

   \partial^a h(x)
      =\frac{\partial^{|a|}h(x)}
             {\partial x_1^{a_1}\cdots\partial x_d^{a_d}},
   \qquad
   \partial^a\widehat h_J(x)
      =\partial^a\psi^J(x)'\mathbf M_J\mathbf Y.

The estimator differentiates the basis rather than taking finite
differences of the plotted curve. Use ``deriv_index`` to select one
coordinate with one-based indexing and ``deriv_order`` to choose how
many times to differentiate with respect to that coordinate.
The general multi-index notation in the theory also covers mixed derivatives.

A derivative is an elasticity only when the variable transformations
justify that interpretation. If both outcome and regressor enter in logs,
the derivative of the log structural function with respect to the log
regressor is an elasticity. In levels, an elasticity instead involves
:math:`x_j\partial_jh_0(x)/h_0(x)` where the denominator is nonzero.
A confidence band for that ratio requires inference for the transformed
function rather than merely relabeling a derivative band.

Why B-splines and their dimensions matter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A B-spline of order :math:`r` has polynomial degree :math:`r-1`.
Its local basis functions describe the curve over a knot partition that
you can refine by adding segments. Too few segments leave approximation
bias and too many can make the inverse problem noisy.

The paper develops its sup-norm guarantees for B-splines and
Cohen-Daubechies-Vial wavelets. Their bounded projection norms control
how approximation errors behave uniformly over the support.
Because these basis conditions are part of the result, substituting
an arbitrary basis requires checking its properties before applying the
theorem.

For tensor-product splines at dyadic resolution :math:`l`, the theoretical
dimensions form

.. math::

   \mathcal T=\{(2^l+r-1)^d:l=0,1,\ldots\}.

For a scalar regressor and cubic splines, the dimensions are
:math:`4,5,7,11,19,\ldots`. The paper links instrument resolutions
to regressor resolutions and assumes :math:`J\leq K(J)\lesssim J`.
A dimension-adjusted linkage uses
:math:`l_w=\lceil(l+q)d/d_w\rceil`.
Instrument splines have sufficient order for the conditional mean
relationships being approximated.

ModernDiD defaults to ``j_x_degree=3`` and ``k_w_degree=4``.
Its ``k_w_smooth=2`` makes the instrument basis use four times as many
segments per coordinate as the regressor basis. This is a refinement
setting, not a direct estimate of instrument strength. If
:math:`d\ne d_w`, equal refinement per coordinate does not automatically
give the paper's proportional-dimension linkage.

How much information the instruments reveal
---------------------------------------------

Even when injectivity permits identification, inverting the instrument
restriction can magnify sampling noise. We measure that difficulty over
each sieve space before choosing how large a space the data can support.

Let :math:`\Psi_J` and :math:`B_K` be the function spaces spanned by
the regressor and instrument bases. For
:math:`\|h\|_{L_X^2}=(\mathbb E[h(X)^2])^{1/2}`, define

.. math::

   \tau_J=
      \sup_{\substack{h\in\Psi_J\\\|h\|_{L_X^2}\ne0}}
      \frac{\|h\|_{L_X^2}}{\|Th\|_{L_W^2}}.

A function can vary substantially with :math:`X` while its conditional
mean given :math:`W` varies very little. Large :math:`\tau_J`
describes that loss of information. Conditional expectation is contractive,
so :math:`\tau_J\geq1`.

In the mildly ill-posed regime,
:math:`\tau_J\asymp J^{\varsigma/d}` for :math:`\varsigma\geq0`.
In the severely ill-posed regime,
:math:`\tau_J\asymp\exp(CJ^{\varsigma/d})`
for :math:`C,\varsigma>0`. The notation :math:`\asymp` means
that the ratio is bounded above and below by positive constants.
For nonparametric regression, :math:`T` is the identity and
:math:`\tau_J=1`.

Conditions on the approximation spaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The instrument basis must approximate the conditional means generated
by the regressor basis. Bias must also remain controlled when mapped
through the inverse problem. We make those requirements precise with
population projections,

.. math::

   \begin{aligned}
   \Pi_Jf&=\arg\min_{h\in\Psi_J}\|f-h\|_{L_X^2},\\
   \Pi_{K(J)}f&=\arg\min_{b\in B_{K(J)}}\|f-b\|_{L_W^2},\\
   Q_Jf&=\arg\min_{h\in\Psi_J}
      \|\Pi_{K(J)}T(f-h)\|_{L_W^2}.
   \end{aligned}

The first two projections minimize least-squares distances and the last
is the population TSLS projection. Write
:math:`\|h\|_\infty=\sup_{x\in\mathcal X}|h(x)|`
for the largest absolute value over the regressor support.

.. admonition:: Assumption 3 Approximation and stability
   :class: assumption

   For every :math:`J\in\mathcal T`, there is
   :math:`v_J<1` with :math:`v_J\to0` such that

   .. math::

      \sup_{\substack{h\in\Psi_J\\\|h\|_{L_X^2}=1}}
      \tau_J\|\Pi_{K(J)}Th-Th\|_{L_W^2}\leq v_J.

   For finite positive constants :math:`C_T,C_Q`,

   .. math::

      \begin{aligned}
      \tau_J\|T(h_0-\Pi_Jh_0)\|_{L_W^2}
         &\leq C_T\|h_0-\Pi_Jh_0\|_{L_X^2},\\
      \|Q_J(h_0-\Pi_Jh_0)\|_\infty
         &\leq C_Q\|h_0-\Pi_Jh_0\|_\infty.
      \end{aligned}

   Both inequalities hold for every :math:`J\in\mathcal T`.

The first condition prevents the instrument approximation from losing
important sieve directions. The next two bound how approximation error
propagates through the structural fit, a requirement that holds trivially
for nonparametric regression with matching bases but needs justification
in an IV problem.

How uncertainty grows with dimension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dimension selection compares estimates at different resolutions.
The theory therefore needs a variance scale and a restriction ensuring
that larger dimensions provide a meaningfully noisier comparison.
Define population matrices

.. math::

   \begin{aligned}
   G_{b,J}&=\mathbb E[b^{K(J)}(W)b^{K(J)}(W)'],\\
   S_J&=\mathbb E[b^{K(J)}(W)\psi^J(X)'],\\
   H_J&=S_J'G_{b,J}^{-1}S_J,\\
   \Omega_J&=\mathbb E[u^2b^{K(J)}(W)b^{K(J)}(W)'].
   \end{aligned}

For nonsingular population matrices, define the information scale
and the variance scale,

.. math::

   \begin{aligned}
   s_J^2(x)&=\psi^J(x)'H_J^{-1}\psi^J(x),\\
   L_J(x)&=\psi^J(x)'H_J^{-1}S_J'G_{b,J}^{-1},\\
   \sigma_J^2(x)&=L_J(x)\Omega_JL_J(x)',\\
   s_{J,a}^2(x)&=\partial^a\psi^J(x)'H_J^{-1}\partial^a\psi^J(x).
   \end{aligned}

The error moment bounds make :math:`s_J(x)` and
:math:`\sigma_J(x)` comparable uniformly in :math:`x`.
Both describe population scales for sample-size-normalized fluctuations
rather than the standard error of a sample estimate.

.. admonition:: Assumption 4 Sieve variance growth
   :class: assumption

   For finite positive :math:`c,C` and every :math:`J\in\mathcal T`,

   .. math::

      c\tau_J^2J
      \leq\inf_x s_J^2(x)
      \leq\sup_x s_J^2(x)
      \leq C\tau_J^2J.

   For some :math:`\gamma\in(0,1)`,

   .. math::

      \limsup_{J\to\infty}
      \sup_{\substack{x\in\mathcal X\\J_2\in\mathcal T,\ J_2>J}}
      \frac{\sigma_J(x)}{\sigma_{J_2}(x)}<\gamma.

   For derivative bands, part (iii) additionally requires

   .. math::

      c\tau_J^2J^{1+2|a|/d}
      \leq\inf_x s_{J,a}^2(x)
      \leq\sup_x s_{J,a}^2(x)
      \leq C\tau_J^2J^{1+2|a|/d}

   for every :math:`J\in\mathcal T` and the derivative being studied.

These conditions describe how uncertainty changes across the theoretical
sieve sequence, beyond what a successful matrix inversion in one fitted
model can establish. The increasing variance is one reason the procedure
compares geometrically separated dimensions.

Choosing the dimension before constructing a band
--------------------------------------------------

A tuning rule for outcome prediction need not recover a structural function.
We first examine that issue before using comparisons across sieve dimensions
that respect the IV restriction.

Why ordinary prediction cross-validation can fail
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Leave-one-out squared-error cross-validation uses

.. math::

   CV(J)=\frac1n\sum_{i=1}^n
      (Y_i-\widehat h_{-i,J}(X_i))^2,

where the fit excludes observation :math:`i`.
Substituting :math:`Y_i=h_0(X_i)+u_i` gives

.. math::

   \begin{aligned}
   CV(J)
      &=\frac1n\sum_i(h_0(X_i)-\widehat h_{-i,J}(X_i))^2
        +\frac1n\sum_i u_i^2\\
      &\quad+\frac2n\sum_i
         u_i(h_0(X_i)-\widehat h_{-i,J}(X_i)).
   \end{aligned}

The last term can depend on :math:`J` when
:math:`\mathbb E[u\mid X]\ne0`. Minimizing this criterion can therefore
favor prediction of the endogenous conditional mean rather than recovery
of :math:`h_0`. This is a limitation of this ordinary prediction
criterion, not of every possible IV-specific validation method.

Even under exogeneity, squared prediction error targets an average-error
criterion. A confidence band requires control of the largest error over
the support. The paper instead uses a bootstrap Lepski procedure that
compares entire fitted functions at different resolutions.

Bounding the feasible search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To estimate how much inversion the sample can support, we compare each
dimension with the next larger one in the theoretical sequence. Denote
that next dimension by :math:`J^+` in :math:`\mathcal T` and let
:math:`\widehat s_J` be the smallest singular value of

.. math::

   (\mathbf B_{K(J)}'\mathbf B_{K(J)})^{-1/2}
   (\mathbf B_{K(J)}'\boldsymbol\Psi_J)
   (\boldsymbol\Psi_J'\boldsymbol\Psi_J)^{-1/2}.

The negative half power denotes the inverse of the positive-definite
square root. The reciprocal :math:`\widehat s_J^{-1}` estimates
the inversion difficulty. The paper chooses

.. math::

   \widehat J_{\max}
   =\min\left\{J\in\mathcal T:
      J\sqrt{\log J}\,\widehat s_J^{-1}\leq10\sqrt n
      <J^+\sqrt{\log J^+}\,\widehat s_{J^+}^{-1}
      \right\}.

Its search set and testing level are

.. math::

   \begin{aligned}
   \widehat{\mathcal J}
      &=\{J\in\mathcal T:
          0.1(\log\widehat J_{\max})^2\leq J\leq\widehat J_{\max}\},\\
   \widehat\alpha
      &=\min\{0.5,(\log\widehat J_{\max}/\widehat J_{\max})^{1/2}\}.
   \end{aligned}

The testing level :math:`\widehat\alpha` controls the dimension
comparison. It is different from the desired confidence-band
noncoverage probability :math:`\alpha`.

Comparing small and large fits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since the same observations produce every candidate fit, standardizing
their differences requires accounting for the covariance between their
estimation errors.
For fitted residuals
:math:`\widehat u_{i,J}=Y_i-\widehat h_J(X_i)`, write

.. math::

   \widehat U_{J,J_2}
      =\operatorname{diag}
         (\widehat u_{1,J}\widehat u_{1,J_2},\ldots,
          \widehat u_{n,J}\widehat u_{n,J_2}).

Define the estimated variance and cross-covariance,

.. math::

   \begin{aligned}
   \widehat\sigma_J^2(x)
      &=\psi^J(x)'\mathbf M_J\widehat U_{J,J}
          \mathbf M_J'\psi^J(x),\\
   \widetilde\sigma_{J,J_2}(x)
      &=\psi^J(x)'\mathbf M_J\widehat U_{J,J_2}
          \mathbf M_{J_2}'\psi^{J_2}(x),\\
   \widehat\sigma_{J,J_2}^2(x)
      &=\widehat\sigma_J^2(x)+\widehat\sigma_{J_2}^2(x)
          -2\widetilde\sigma_{J,J_2}(x).
   \end{aligned}

Because :math:`\widehat\sigma_J(x)` is already a standard error, a
band uses it without another division by :math:`\sqrt n`.
For :math:`J_2>J`, the observed comparison is

.. math::

   T_{J,J_2}
      =\sup_{x\in\mathcal X}
         \left|
         \frac{\widehat h_J(x)-\widehat h_{J_2}(x)}
              {\widehat\sigma_{J,J_2}(x)}
         \right|.

The bootstrap simulates the largest standardized difference under sampling
variation. Draw independent standard normal multipliers :math:`\varpi_i`,
independently of the data. Define

.. math::

   \begin{aligned}
   \widehat{\mathbf u}_J^*
      &=(\widehat u_{1,J}\varpi_1,\ldots,
          \widehat u_{n,J}\varpi_n)',\\
   D_J^*(x)&=\psi^J(x)'\mathbf M_J\widehat{\mathbf u}_J^*.
   \end{aligned}

Using the same multiplier for an observation across all candidate
dimensions preserves the estimated covariance between their fits.
Let :math:`\theta_{1-\widehat\alpha}^*` be the bootstrap quantile of

.. math::

   \sup_{\substack{x\in\mathcal X\\J,J_2\in\widehat{\mathcal J},\ J_2>J}}
      \left|
      \frac{D_J^*(x)-D_{J_2}^*(x)}
           {\widehat\sigma_{J,J_2}(x)}
      \right|.

The paper then selects

.. math::

   \begin{aligned}
   \widehat J
      &=\min\left\{J\in\widehat{\mathcal J}:
         \sup_{\substack{J_2\in\widehat{\mathcal J}\\J_2>J}}
            T_{J,J_2}
         \leq1.1\theta_{1-\widehat\alpha}^*
         \right\},\\
   \widehat J_n&=\max\{J\in\widehat{\mathcal J}:J<\widehat J_{\max}\},\\
   \widetilde J&=\min\{\widehat J,\widehat J_n\}.
   \end{aligned}

A small fit is retained only if all larger fits remain sufficiently close,
subject to truncation that prevents choosing the largest feasible dimension.
Those comparisons require a search set with enough candidates to be
meaningful.

The numerical search in the package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To choose the sieve dimension from the data, leave ``j_x_segments=None``
so the estimator calls :func:`~moderndid.npiv_choose_j`. Supplying a
number of segments instead fixes the structural sieve for your analysis.
The ``args`` field records selection quantities such as
``j_tilde``, ``j_hat``, and ``theta_star`` when selection succeeds.

The implementation uses a finite dyadic grid and retains its candidates
through the computed upper cutoff. It does not enforce the paper's
log-squared lower cutoff on that grid. Its upper-cutoff calculation also
protects the inverse singular value with :math:`(0.1\log n)^4`.
Those are numerical choices rather than additional identifying assumptions.

For multiple regressors, the default comparison grid has 50 rows in
which each coordinate runs through equally spaced values. Since these
rows do not form a full Cartesian product of coordinates, the grid can
miss differences between fitted functions in parts of joint support
that it does not cover.

.. admonition:: Set the region your band covers
   :class: important

   Supply ``x_grid`` to control where dimension selection compares
   fits and ``x_eval`` to control where bands are evaluated.
   The package takes maxima over those finite points.
   A numerical maximum approximates the paper's supremum over
   :math:`\mathcal X` only when the grid adequately represents that region.

If selection fails, the high-level estimator issues a warning and uses
fallback segment counts instead of the paper's selected sieve. Read the
warning and the recorded arguments before interpreting a reported band
as a result of adaptive selection.

What adaptivity means for estimation
------------------------------------

Although the procedure does not require you to supply the true smoothness
or ill-posedness exponent, its rate guarantees still depend on the
regularity conditions above and the class of functions being estimated.
Specifying that class gives the convergence result a precise scope.

Let :math:`B_{\infty,\infty}^p(M)` be the Hölder-Zygmund ball of
smoothness :math:`p` and radius :math:`M`.
One characterization uses an integer :math:`k>p` and the finite difference

.. math::

   \Delta_v^k h(x)
      =\sum_{j=0}^k(-1)^{k-j}\binom{k}{j}h(x+jv).

The norm defining the ball controls

.. math::

   \|h\|_\infty+
   \sup_{0<\|v\|\leq1}
      \frac{\|\Delta_v^k h\|_\infty}{\|v\|^p},

where the differences use points remaining in the support.
For noninteger :math:`p`, this is equivalent to bounding derivatives
through order :math:`\lfloor p\rfloor` and imposing Hölder continuity
of order :math:`p-\lfloor p\rfloor` on the highest derivatives.

Take a fixed smoothness range
:math:`\overline p>\underline p>d/2` and enough spline order,
:math:`r\geq\lfloor\overline p\rfloor+1`.
The paper's class :math:`\mathcal H^p` contains members of
:math:`B_{\infty,\infty}^p(M)` that satisfy the stability conditions
in Assumption 3(ii) and (iii) with the fixed constants.
Probabilities :math:`P_h` refer to iid data generated under
:math:`Y=h(X)+u` and the stated distributional assumptions.

.. admonition:: Theorem 4.1 and Corollary 4.1 Adaptive sup-norm rates
   :class: theorem

   Under Assumptions 1 through 3 and 4(i) and (ii), the paper's
   data-driven estimator satisfies, for some universal :math:`C_a>0`,

   .. math::

      \sup_{p\in[\underline p,\overline p]}
      \sup_{h\in\mathcal H^p}
      P_h\!\left(
         \|\partial^a\widehat h_{\widetilde J}-\partial^a h\|_\infty
         >C_a r_{n,a}(p)
      \right)\longrightarrow0.

   For :math:`a=0` this is Theorem 4.1. For derivatives with
   :math:`0<|a|<\underline p` it is Corollary 4.1. The rates are

   .. math::

      r_{n,a}(p)=
      \begin{cases}
      (\log n/n)^{(p-|a|)/(2(p+\varsigma)+d)},
         &\text{mildly ill-posed},\\
      (\log n)^{-(p-|a|)/\varsigma},
         &\text{severely ill-posed}.
      \end{cases}

   The sieve spaces and dimension-selection rule are those specified
   in the paper.

The estimator attains these minimax sup-norm rates for the stated classes
in both regimes without knowing the smoothness or inversion difficulty.
Estimating a derivative lowers the exponent because it magnifies
variation at small scales.

The order condition also limits what a low-degree spline can approximate.
A fixed cubic order does not give exact minimax adaptivity over arbitrarily
smooth classes. The paper's Remark 4.2 describes the slower rate when
the spline order is insufficient for the true smoothness.

Uncertainty over the function
-----------------------------

Whereas a pointwise interval targets one evaluation point, a uniform band
targets the entire selected region at once. Constructing that band requires
accounting for sampling variation, approximation bias, and the uncertainty
introduced by selecting a dimension.

Separating structural noise from approximation error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the true error vector :math:`\mathbf u=(u_1,\ldots,u_n)'`,
the fitted function has the exact decomposition

.. math::

   \begin{aligned}
   \widehat h_J(x)-h_0(x)
      &=\psi^J(x)'\mathbf M_J\mathbf u\\
      &\quad+\psi^J(x)'\mathbf M_J
         (h_0(X_1),\ldots,h_0(X_n))'-h_0(x).
   \end{aligned}

The decomposition separates the fluctuation caused by structural errors
from the sample approximation term. Although fitted residuals estimate
the error variance and enter the bootstrap, they cannot replace
:math:`\mathbf u` in this decomposition.

In particular, multiplying the fitted residual vector by
:math:`\mathbf M_J` gives zero under the TSLS normal equations.
The variance estimator instead uses squared residuals inside the sandwich
matrix defined above. The multiplier bootstrap changes their signs and
magnitudes observation by observation. Its fluctuation can therefore be
nonzero even though the unweighted fitted-residual term vanishes.

For derivatives, replace :math:`\psi^J(x)` by
:math:`\partial^a\psi^J(x)` in the error decomposition, variance,
and bootstrap expressions.

Bands when the dimension is fixed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a chosen :math:`J`, let :math:`z_{1-\alpha,J}^*` be the
bootstrap quantile of
:math:`\sup_x|D_J^*(x)/\widehat\sigma_J(x)|`.
The fixed-sieve band is

.. math::

   C_{n,J}(x)=
      [\,\widehat h_J(x)\ \pm\
         z_{1-\alpha,J}^*\widehat\sigma_J(x)\,].

The derivative counterpart uses its derivative bootstrap quantile and
standard error. These are the undersmoothing bands studied by
`Chen and Christensen (2018)
<https://arxiv.org/abs/1508.03365>`_.

For structural coverage, the approximation term must be negligible
relative to the band's sampling scale. Selecting a fixed number of
segments does not establish that condition.
The theory uses a sequence of dimensions that grows sufficiently quickly
to make bias negligible while respecting the rank, moment, and
growth requirements. Larger dimensions can reduce approximation bias
and increase uncertainty at the same time.

Adaptive bands and self-similar functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since bands cannot be both honest and adaptive uniformly over every Hölder
ball of unknown smoothness, the paper restricts approximation error to a
self-similar subclass. In this subclass, approximation error remains
detectable across resolutions rather than disappearing at selected scales.

For a fixed :math:`0<\underline B<\overline B` and starting
dimension :math:`J_*`, define

.. math::

   \begin{aligned}
   \mathcal G^p
      &=\left\{h\in\mathcal H^p:
         \|\Pi_Jh-h\|_\infty\geq\underline B J^{-p/d}
         \text{ for all }J\in\mathcal T,\ J\geq J_*
         \right\},\\
   \mathcal G&=\bigcup_{p\in[\underline p,\overline p]}\mathcal G^p.
   \end{aligned}

Spline approximation also gives the upper bound
:math:`\|\Pi_Jh-h\|_\infty\leq\overline B J^{-p/d}`
on the smoothness class. The lower bound prevents the bias from becoming
arbitrarily small at some resolutions while remaining large at others.

A band is honest over :math:`\mathcal G` if its simultaneous coverage
is at least :math:`1-\alpha` asymptotically, uniformly over that class.
It is adaptive if its width contracts at the rate associated with the
function's own smoothness rather than the least smooth member of the class.

The paper accounts for dimension selection by using the candidate set

.. math::

   \widehat{\mathcal J}_-=
   \begin{cases}
   \{J\in\widehat{\mathcal J}:J<\widehat J_n\},
      &\widehat J\leq\widehat J_n,\\
   \widehat{\mathcal J},
      &\widehat J>\widehat J_n.
   \end{cases}

Let :math:`z_{1-\alpha}^*` be the bootstrap quantile of

.. math::

   \sup_{\substack{x\in\mathcal X\\J\in\widehat{\mathcal J}_-}}
      \left|\frac{D_J^*(x)}{\widehat\sigma_J(x)}\right|.

This writes Procedure 2 with equality assigned to its first branch.
The convention makes the candidate set unambiguous when selection and
truncation choose the same dimension.
In the mild regime, a band with inflation constant :math:`A` is

.. math::

   C_n(x,A)=
      [\,\widehat h_{\widetilde J}(x)\ \pm\
         (z_{1-\alpha}^*+A\theta_{1-\widehat\alpha}^*)
         \widehat\sigma_{\widetilde J}(x)\,].

The derivative band :math:`C_n^a(x,A)` uses
:math:`\partial^a\widehat h_{\widetilde J}`, its standard error,
and a derivative bootstrap quantile over the same dimensions.

.. admonition:: Theorems 4.2 and 4.4 Mild-regime coverage and width
   :class: theorem

   Under Assumptions 1 through 4 and mild ill-posedness, there is
   :math:`A_*>0` independent of :math:`\alpha` such that
   every fixed :math:`A\geq A_*` gives

   .. math::

      \liminf_{n\to\infty}\inf_{h\in\mathcal G}
      P_h\{h(x)\in C_n(x,A)\text{ for all }x\in\mathcal X\}
      \geq1-\alpha.

   For a universal :math:`C>0`,

   .. math::

      \inf_{p\in[\underline p,\overline p]}
      \inf_{h\in\mathcal G^p}
      P_h\!\left\{
         \sup_x|C_n(x,A)|
         \leq C(1+A)(\log n/n)^{p/(2(p+\varsigma)+d)}
      \right\}\longrightarrow1.

   For :math:`0<|a|<\underline p`, Assumption 4(iii) gives the same
   coverage statement for :math:`\partial^a h` and
   :math:`C_n^a`. Its width bound replaces :math:`p` in the numerator
   of the exponent by :math:`p-|a|`.
   The thresholds and constants can differ for functions and derivatives.

These statements use the complete theoretical procedures and the
self-similar class. The recommended
:math:`\widehat A=\log\log\widetilde J` grows slowly enough that
coverage holds for the stated classes and width is within a
:math:`\log\log n` factor of the minimax rate.
The guarantee includes nonparametric regression as :math:`\varsigma=0`.

What changes in the severe regime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adaptive point estimation does not by itself imply adaptive bands.
When instruments reveal very little about high-frequency variation,
approximation bias can dominate the optimal estimator's sampling error.
The severe-regime coverage result adds an explicit bias envelope.

For the structural function, the paper's modified critical value is

.. math::

   cv_{\mathrm{sev}}^*(x,A)
      =z_{1-\alpha}^*
       +A\max\left\{
          \theta_{1-\widehat\alpha}^*,
          \frac{\widetilde J^{-\underline p/d}}
               {\widehat\sigma_{\widetilde J}(x)}
         \right\}.

For a derivative, replace the bias factor by
:math:`\widetilde J^{(|a|-\underline p)/d}`
and use its derivative standard error and quantile.
The least assumed smoothness :math:`\underline p` controls this
envelope. It can therefore produce conservative bands for smoother
functions in the stated class.

.. admonition:: Theorems 4.3 and 4.5 Severe-regime coverage
   :class: theorem

   Under Assumptions 1 through 4 and severe ill-posedness, the modified
   bands using the bias envelope have a constant :math:`A_*>0`
   independent of :math:`\alpha` such that, for every fixed
   :math:`A\geq A_*`,

   .. math::

      \liminf_{n\to\infty}\inf_{h\in\mathcal G}
      P_h\{h(x)\in C_{n,\mathrm{sev}}(x,A)
         \text{ for all }x\in\mathcal X\}
      \geq1-\alpha.

   Their widths satisfy, for a universal :math:`C>0`,

   .. math::

      \inf_{p\in[\underline p,\overline p]}
      \inf_{h\in\mathcal G^p}
      P_h\!\left\{
         \sup_x|C_{n,\mathrm{sev}}(x,A)|
         \leq C(1+A)(\log n)^{-\underline p/\varsigma}
      \right\}\longrightarrow1.

   If :math:`0<|a|<\underline p` and Assumption 4(iii) holds,
   the derivative band has the corresponding coverage guarantee.
   Its width bound is
   :math:`C_a(1+A)(\log n)^{-(\underline p-|a|)/\varsigma}`.

Because the width depends on the lower smoothness bound rather than
automatically on the true :math:`p`, these bands are not generally
rate-adaptive to smoother functions in the severe regime. This limitation
on band width does not change the point estimator's adaptive rates from
Theorem 4.1.

.. admonition:: Separate the implemented band from the severe-regime theorem
   :class: warning

   ModernDiD's adaptive band uses the bootstrap quantile plus the
   mild-regime selection penalty. It does not implement the severe-regime
   bias envelope above. The severe-regime coverage theorem therefore
   cannot be attached to the returned band without an additional argument.

Reading the package's bands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For successful data-driven selection, the implementation uses a
candidate-dimension bootstrap and adds
:math:`\max\{0,\log\log\widetilde J\}\theta^*`
to the critical value. Its finite bootstrap dimension set is a numerical
implementation choice rather than exactly
:math:`\widehat{\mathcal J}_-` in Procedure 2.

The default ``biters=99`` sets the number of multiplier draws used to
estimate critical values. Increasing that count reduces Monte Carlo error
without establishing asymptotic coverage or guaranteeing precise tail
quantiles at any fixed count. Passing ``seed`` makes those random draws
reproducible across runs of the same analysis.

With fixed ``j_x_segments``, the estimator uses the fixed-sieve
undersmoothing construction. With ``ucb_h=False`` or
``ucb_deriv=False``, it omits the corresponding bands.
The ``asy_se`` and ``deriv_asy_se`` result fields are standard errors
on the scale of the estimates. The band endpoints already apply their
reported critical values to those standard errors.

Changing the structural restrictions
-------------------------------------

Reducing the rapidly growing dimension of a multivariate tensor-product
sieve by restricting the structural function changes the model you estimate.
We distinguish that modeling decision from choosing a tuning setting
within an unrestricted model.

Tensor, additive, and restricted interactions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``basis="tensor"``, the basis is the product of marginal spline
bases. If coordinate :math:`j` has :math:`J_j` functions, the joint
dimension is :math:`\prod_jJ_j`. This is the construction used in the
main theoretical results above.

With ``basis="additive"``, the basis combines marginal functions and
an intercept. The structural restriction is

.. math::

   h_0(x)=c_0+\sum_{j=1}^d h_{j0}(x_j).

Component normalizations separate the intercept from the individual
functions. Section 6 describes centered marginal bases,

.. math::

   \widetilde\psi_{Jk}(x_j)
      =\psi_{Jk}(x_j)-\int_0^1\psi_{Jk}(v)\,dv.

Because additivity removes general interactions from the target function,
its estimation and component-band procedures use the additive-model
conditions developed in the paper's extension. The unrestricted
tensor-product theorem does not apply unchanged to that restricted model.

The ``basis="glp"`` construction retains main effects and selected
lower-order interactions of the marginal bases. Its dimension lies between
the additive and tensor constructions. The paper's main tensor-product
guarantees do not automatically cover this different approximation space.
Its use requires a structural approximation that those retained
interactions can support.

Partially linear structural functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If some regressors enter linearly, the theoretical model can take the form

.. math::

   h_0(x)=h_{10}(x_1)+x_2'\beta_0,
   \qquad
   \psi^J(x)=(\psi_1^J(x_1)',x_2')'.

The sieve TSLS regression then estimates the nonlinear function and the
linear coefficients together. For a band on the nonlinear component,
Section 6 replaces the evaluation vector with
:math:`(\psi_1^J(x_1)',0_{d_2}')'`.
This changes the target of the bootstrap contrast.
Because the high-level ``npiv`` API does not expose a separate partially
linear specification argument, this extension describes the method rather
than an automatic option of that call.

Nonparametric regression as a special case
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``w`` equals ``x``, the package makes the instrument and regressor
bases match. The instrument projection drops out,

.. math::

   \widehat c_J=
      (\boldsymbol\Psi_J'\boldsymbol\Psi_J)^{-}
      \boldsymbol\Psi_J'\mathbf Y,
   \qquad
   \mathbf M_J=
      (\boldsymbol\Psi_J'\boldsymbol\Psi_J)^{-}
      \boldsymbol\Psi_J'.

The paper's regression procedure replaces the estimated inverse
ill-posedness measure with
:math:`v_n=\max\{1,(0.1\log n)^4\}` as part of its regression-specific
selection and band construction.
ModernDiD also uses that cutoff safeguard when the arrays match.
Its generic numerical selection still applies the final dimension
truncation. The mild-regime interpretation follows from
:math:`\tau_J=1` rather than from stronger IV assumptions.

This regression special case supports the data-driven dose estimator
described in the :ref:`continuous treatment background <background-didcont>`.
For an IV analysis, the :ref:`nonparametric IV example <example_npiv>`
shows the fitted structural function, its derivative, and the regions
where the instrument-supported estimate is most uncertain.
