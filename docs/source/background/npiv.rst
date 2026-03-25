.. _background-npiv:

Nonparametric Instrumental Variables
=====================================

The ``npiv`` module implements nonparametric instrumental variables (NPIV)
estimation with data-driven tuning parameter selection and uniform confidence
bands, following the methodology of `Chen, Christensen, and Kankanala (2024)
<https://arxiv.org/abs/2107.11869>`_. This approach estimates a structural
function and its derivatives (such as elasticities) from a conditional moment
restriction using sieve two-stage least squares, with procedures that adapt to
unknown smoothness and instrument strength.

NPIV estimation arises in many empirical settings where a researcher wants to
estimate a flexible, nonparametric relationship but some regressors are
endogenous. Applications include consumer demand (Blundell, Chen, and
Kristensen, 2007), demand for differentiated products (Berry and Haile, 2014;
Compiani, 2022), international trade (Adao, Arkolakis, and Ganapati, 2020),
and Engel curve estimation.

The module has two main pieces. A data-driven choice of sieve dimension that
achieves minimax convergence rates in sup-norm, and uniform confidence bands
that are honest (correct coverage over a class of data-generating processes)
and adaptive (contract at the minimax rate).


The NPIV Model
--------------

The starting point is a conditional moment restriction that links the outcome, the
endogenous regressors, and the instruments without specifying a functional form for the
structural relationship. The structural function :math:`h_0` satisfies

.. math::

   \mathbb{E}[Y - h_0(X) \mid W] = 0 \quad \text{(almost surely)},

where :math:`Y` is a scalar outcome, :math:`X` is a vector of possibly
endogenous regressors, and :math:`W` is a vector of instrumental variables.
The conditional distribution of :math:`(X, Y)` given :math:`W` is otherwise
unspecified. This model nests nonparametric regression as the special case
:math:`W = X`.

Substituting a sieve approximation :math:`h_0(x) \approx (\psi^J(x))' c_J`
into the conditional moment restriction gives

.. math::

   Y = (\psi^J(X))' c_J + \text{bias}_J + u, \quad \mathbb{E}[u \mid W] = 0,

where :math:`u = Y - h_0(X)` is the structural error and
:math:`\text{bias}_J = h_0(X) - (\psi^J(X))' c_J` is the approximation bias.
When the bias term is small relative to :math:`u`, this looks like a linear
instrumental variables model with :math:`\psi^J(X)` as :math:`J` endogenous
variables and :math:`c_J` as the unknown parameter vector. Unlike standard
nonparametric regression, :math:`\mathbb{E}[u \mid X] \neq 0` in general, so
least squares applied directly to :math:`X` would be inconsistent for
:math:`h_0`.


Sieve TSLS Estimation
---------------------

Given the conditional moment restriction, :math:`h_0` is estimated by projecting onto a
finite-dimensional sieve space and applying two-stage least squares with basis functions of
the instruments. The function :math:`h_0` is approximated by a linear combination of :math:`J`
B-spline basis functions

.. math::

   h_0(x) \approx (\psi^J(x))' c_J,

where :math:`\psi^J(x) = (\psi_{J1}(x), \ldots, \psi_{JJ}(x))'` is a vector
of basis functions and :math:`c_J` is a coefficient vector. The coefficients
are estimated by two-stage least squares using :math:`K` B-spline basis
functions of :math:`W` as instruments

.. math::

   \hat{c}_J = (\boldsymbol{\Psi}_J' \mathbf{P}_K \boldsymbol{\Psi}_J)^{-}
   \boldsymbol{\Psi}_J' \mathbf{P}_K \mathbf{Y},

where :math:`\boldsymbol{\Psi}_J` and :math:`\mathbf{B}_K` are :math:`n
\times J` and :math:`n \times K` matrices of basis evaluations, the projection
onto the instrument space is

.. math::

   \mathbf{P}_K = \mathbf{B}_K (\mathbf{B}_K' \mathbf{B}_K)^{-}
   \mathbf{B}_K',

and :math:`(\cdot)^{-}` denotes the Moore-Penrose inverse. Defining the
:math:`J \times n` matrix

.. math::

   \mathbf{M}_J = (\boldsymbol{\Psi}_J' \mathbf{P}_{K(J)}
   \boldsymbol{\Psi}_J)^{-} \boldsymbol{\Psi}_J' \mathbf{P}_{K(J)},

the estimators of :math:`h_0` and its derivatives become

.. math::

   \hat{h}_J(x) = (\psi^J(x))' \mathbf{M}_J \mathbf{Y}, \quad
   \partial^a \hat{h}_J(x) = (\partial^a \psi^J(x))' \mathbf{M}_J \mathbf{Y}.

The matrix :math:`\mathbf{M}_J` recurs in the variance estimation and
bootstrap steps that follow.

B-spline Basis Choice
~~~~~~~~~~~~~~~~~~~~~

Of the many sieve bases available (polynomial splines, wavelets, Fourier
series, various polynomials), only B-splines and Cohen-Daubechies-Vial (CDV)
wavelets have been shown to achieve the optimal minimax sup-norm convergence
rates under a suitable choice of :math:`J`
(`Chen and Christensen, 2018 <https://doi.org/10.3982/ECTA12560>`_). Both
bases share a bounded Lebesgue constant, meaning the
:math:`L^\infty` norm of the :math:`L^2` projection onto the sieve space
remains bounded as :math:`J` grows. Bases without this property, such as
polynomials and Fourier series, cannot attain the minimax sup-norm rate and
therefore cannot yield rate-adaptive estimators or confidence bands.

B-splines are characterized by their order :math:`r` (equivalently, polynomial
degree :math:`r - 1`). The module uses a cubic B-spline (:math:`r = 4`) to
approximate :math:`h_0` and a quartic B-spline (:math:`r = 5`) for the
reduced-form relationship between the instruments and the endogenous
regressors, since the reduced form is smoother than :math:`h_0` itself.

Dyadic Grid and Instrument Linkage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The set of candidate sieve dimensions forms a dyadic grid

.. math::

   \mathcal{T} = \{J = (2^l + r - 1)^d : l \in \mathbb{N}_0\},

where :math:`l` is the resolution level. For scalar :math:`X` with cubic
B-splines, this gives :math:`\mathcal{T} = \{4, 5, 7, 11, 19, 35, \ldots\}`.
Using a dyadic grid ensures enough separation between consecutive candidates
that the bias and variance of estimators at different :math:`J` can be
accurately compared, improving numerical stability.

The instrument dimension :math:`K(J)` is linked to :math:`J` through the
resolution levels. Given the resolution level :math:`l` for the basis for
:math:`X`, the resolution level for the instrument basis is
:math:`l_w = \lceil (l + q) d / d_w \rceil` for some :math:`q \in \mathbb{N}_0`,
where :math:`d_w` is the dimension of :math:`W`. This defines a mapping
:math:`K(J)` satisfying :math:`\lim_{J \to \infty} K(J)/J = c \in [1, \infty)`.
Setting :math:`q` to the second- or third-smallest value for which
:math:`K(J) \geq J` holds for all :math:`J` is recommended. Larger values of
:math:`q` are inadvisable because the number of basis functions grows
exponentially in the resolution level.

The performance of the estimator is sensitive to the choice of :math:`J` and
not sensitive to :math:`K` as long as :math:`K \geq J`. If :math:`J` is too
small, the estimator has large bias. If :math:`J` is too large, the estimator
is noisy and confidence bands are uninformatively wide.


Ill-Posedness and Instrument Strength
-------------------------------------

What distinguishes NPIV from nonparametric regression is the *sieve measure of
ill-posedness*, which quantifies how difficult it is to invert the conditional
expectation operator and recover :math:`h_0`. Let :math:`T:
L_X^2 \to L_W^2` denote the operator :math:`Th(w) = \mathbb{E}[h(X) \mid W =
w]` and define

.. math::

   \tau_J = \sup_{h \in \Psi_J : \|h\|_{L_X^2} \neq 0}
   \frac{\|h\|_{L_X^2}}{\|Th\|_{L_W^2}}.

Since conditional expectations are (weakly) contractive, :math:`\tau_J \geq 1`.
The model is classified into two regimes.

- **Mildly ill-posed** with :math:`\tau_J \asymp J^{\varsigma/d}` for some
  :math:`\varsigma \geq 0`. This includes nonparametric regression as the
  special case :math:`\varsigma = 0`.
- **Severely ill-posed** with :math:`\tau_J \asymp \exp(C J^{\varsigma/d})`
  for some :math:`C, \varsigma > 0`.

The data-driven procedures adapt to both regimes without requiring the
researcher to know which regime applies.


Data-Driven Sieve Dimension Selection
-------------------------------------

The estimator's performance hinges on the sieve dimension :math:`J`. Too small and the
approximation bias dominates; too large and the estimate is noisy. This section describes
a Lepski-type procedure that selects :math:`J` from the data, adapting to the unknown
smoothness of :math:`h_0` and the degree of ill-posedness.

Why Cross Validation Fails with Endogeneity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The standard leave-one-out cross-validation criterion is

.. math::

   \text{CV}(J) = \frac{1}{n} \sum_{i=1}^n (Y_i - \hat{h}_{-i,J}(X_i))^2,

where :math:`\hat{h}_{-i,J}` is the estimator computed from a sub-sample that
excludes the :math:`i`-th observation. This can be expanded into three terms

.. math::

   \text{CV}(J) = \underbrace{\frac{1}{n} \sum_{i=1}^n (h_0(X_i) -
   \hat{h}_{-i,J}(X_i))^2}_{\text{MSE estimate}} +
   \underbrace{\frac{1}{n} \sum_{i=1}^n u_i^2}_{\text{independent of } J} +
   \underbrace{\frac{2}{n} \sum_{i=1}^n u_i (h_0(X_i) -
   \hat{h}_{-i,J}(X_i))}_{\text{cross term}}.

The first term estimates the mean-squared error, the second does not depend
on :math:`J`, and the third estimates
:math:`\mathbb{E}[u(h_0(X) - \hat{h}_J(X))]`. In nonparametric regression
where :math:`\mathbb{E}[u \mid X] = 0`, this cross term vanishes
asymptotically, making CV a valid criterion for choosing :math:`J` (Li, 1987).
In models with endogeneity where :math:`\mathbb{E}[u \mid X] \neq 0`, the
cross term depends on :math:`J` and may be non-negligible even asymptotically.
Cross validation then gives a biased estimate of the MSE, and a
cross-validated :math:`J` may not even yield a consistent estimator of
:math:`h_0`. Even in the exogenous case, CV balances bias and sampling
uncertainty in :math:`L^2` norm, which is not the right criterion for
estimation or adaptive UCBs in sup-norm.

Instead, the module uses a Lepski-type procedure that remains valid under
endogeneity.

Step 1: Maximum Feasible Dimension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first step determines how many basis functions the data can support before the estimator
becomes too noisy. This upper bound depends on the sample size and the empirical
ill-posedness. The search grid is determined by

.. math::

   \hat{J}_{\max} = \min\left\{J \in \mathcal{T} : J\sqrt{\log J}\,
   \hat{s}_J^{-1} \leq 10\sqrt{n} < J^+\sqrt{\log J^+}\,
   \hat{s}_{J^+}^{-1}\right\},

where :math:`\mathcal{T}` is a dyadic grid of candidate values and
:math:`\hat{s}_J` is the smallest singular value of

.. math::

   (\mathbf{B}_K'\mathbf{B}_K)^{-1/2}
   (\mathbf{B}_K'\boldsymbol{\Psi}_J)
   (\boldsymbol{\Psi}_J'\boldsymbol{\Psi}_J)^{-1/2},

the sample analog of the inverse ill-posedness measure. The candidate set is
then

.. math::

   \hat{\mathcal{J}} = \{J \in \mathcal{T} : 0.1(\log \hat{J}_{\max})^2
   \leq J \leq \hat{J}_{\max}\}.

Step 2: Lepski Selection
~~~~~~~~~~~~~~~~~~~~~~~~

For each pair :math:`(J, J_2)` with :math:`J_2 > J`, the procedure computes a
sup-:math:`t` statistic for the difference in estimates, studentized by the
estimated standard deviation of the difference

.. math::

   \sup_{x \in \mathcal{X}} \left|
   \frac{\hat{h}_J(x) - \hat{h}_{J_2}(x)}{\hat{\sigma}_{J,J_2}(x)}
   \right|.

The variance of the difference is estimated as

.. math::

   \hat{\sigma}_{J,J_2}^2(x) = \hat{\sigma}_J^2(x) + \hat{\sigma}_{J_2}^2(x)
   - 2\,\tilde{\sigma}_{J,J_2}(x),

where the individual variance terms are computed from the
heteroskedasticity-robust formula

.. math::

   \hat{\sigma}_J^2(x) = (\psi^J(x))' \mathbf{M}_J \widehat{\mathbf{U}}_{J,J}
   \mathbf{M}_J' \psi^J(x),

with :math:`\widehat{\mathbf{U}}_{J,J}` being a diagonal matrix whose
:math:`i`-th entry is :math:`\hat{u}_{i,J}^2` (the squared TSLS residuals),
and the cross term is
:math:`\tilde{\sigma}_{J,J_2}(x) = (\psi^J(x))' \mathbf{M}_J
\widehat{\mathbf{U}}_{J,J_2} \mathbf{M}_{J_2}' \psi^{J_2}(x)` with
:math:`\widehat{\mathbf{U}}_{J,J_2}` having diagonal entries
:math:`\hat{u}_{i,J} \hat{u}_{i,J_2}`.

The bootstrap significance level is set to

.. math::

   \hat{\alpha} = \min\left\{0.5,\;
   \left(\frac{\log \hat{J}_{\max}}{\hat{J}_{\max}}\right)^{1/2}\right\}.

A multiplier bootstrap then determines the critical value
:math:`\theta_{1-\hat{\alpha}}^*` as the :math:`(1-\hat{\alpha})` quantile of

.. math::

   \sup_{\{(x, J, J_2) \in \mathcal{X} \times \hat{\mathcal{J}} \times
   \hat{\mathcal{J}} : J_2 > J\}} \left|
   \frac{D_J^*(x) - D_{J_2}^*(x)}{\hat{\sigma}_{J,J_2}(x)}
   \right|,

where

.. math::

   D_J^*(x) = (\psi^J(x))' \mathbf{M}_J \hat{\mathbf{u}}_J^*

is a bootstrap analog of the estimation error. Here
:math:`\hat{\mathbf{u}}_J^* = (\hat{u}_{1,J}\varpi_1, \ldots,
\hat{u}_{n,J}\varpi_n)'` with :math:`(\varpi_i)_{i=1}^n` drawn i.i.d.
:math:`N(0,1)` independently of the data. The bootstrap weights
:math:`\varpi_i` are held fixed when computing the supremum over
:math:`(x, J, J_2)` for each draw. Taking 1000 independent draws is
sufficient in practice.

The data-driven dimension is

.. math::

   \tilde{J} = \min\{\hat{J}, \hat{J}_n\},

where :math:`\hat{J}` is the smallest dimension passing the Lepski test

.. math::

   \hat{J} = \min\left\{J \in \hat{\mathcal{J}} :
   \sup_{(x, J_2) \in \mathcal{X} \times \hat{\mathcal{J}} : J_2 > J}
   \left|\frac{\hat{h}_J(x) - \hat{h}_{J_2}(x)}
   {\hat{\sigma}_{J,J_2}(x)}\right| \leq 1.1\,\theta_{1-\hat{\alpha}}^*
   \right\},

and :math:`\hat{J}_n = \max\{J \in \hat{\mathcal{J}} : J < \hat{J}_{\max}\}`
is a conservative truncation.

Practical Implementation
~~~~~~~~~~~~~~~~~~~~~~~~

The supremums over :math:`x` in Steps 1 and 2 are computed as maxima over a
fine grid of evaluation points, since the functions involved are continuous in
:math:`x`. The constants 10 and 0.1 in Step 1 can in principle be replaced by
other values, as long as :math:`\hat{\mathcal{J}}` contains several candidate
dimensions to search over. The constant 1.1 in the Lepski test can be replaced
by any constant larger than 1. The value 1.1 performs well in simulations and
is consistent with other implementations of Lepski's method
(Chernozhukov, Chetverikov, and Kato, 2014).

.. tip::

   B-spline order :math:`r` should match the derivatives of interest. For the structural
   function and first derivatives, :math:`r \geq 3` and minimal smoothness
   :math:`\underline{p} \geq 1` suffice. For second derivatives and cross elasticities,
   :math:`r \geq 4` and :math:`\underline{p} \geq 2` are needed.

In the empirical application and the vast majority of simulation designs
(between 99.6% and 100% depending on the design and sample size), the
data-driven choice satisfies :math:`\tilde{J} = \hat{J}`, meaning the Lepski
selection rather than the conservative truncation determines the final
dimension.


Minimax Rate Adaptivity
-----------------------

The data-driven estimator :math:`\hat{h}_{\tilde{J}}` achieves the minimax
sup-norm convergence rate across both ill-posedness regimes. Let
:math:`\mathcal{H}^p` denote the Holder ball of smoothness :math:`p`.

In the mildly ill-posed regime, there exists a universal constant :math:`C` for
which

.. math::

   \sup_{p \in [\underline{p}, \bar{p}]} \sup_{h_0 \in \mathcal{H}^p}
   \mathbb{P}_{h_0}\left(\|\hat{h}_{\tilde{J}} - h_0\|_\infty > C
   \left(\frac{\log n}{n}\right)^{\frac{p}{2(p+\varsigma)+d}}\right) \to 0.

In the severely ill-posed regime,

.. math::

   \sup_{p \in [\underline{p}, \bar{p}]} \sup_{h_0 \in \mathcal{H}^p}
   \mathbb{P}_{h_0}\left(\|\hat{h}_{\tilde{J}} - h_0\|_\infty > C
   (\log n)^{-p/\varsigma}\right) \to 0.

The same data-driven choice :math:`\tilde{J}` also yields minimax rates for
derivative estimation. For a derivative of order :math:`|a|`, in the mildly
ill-posed regime the rate is

.. math::

   \left(\frac{\log n}{n}\right)^{\frac{p-|a|}{2(p+\varsigma)+d}},

and in the severely ill-posed regime it is

.. math::

   (\log n)^{-(p-|a|)/\varsigma}.


Uniform Confidence Bands
------------------------

Point estimates of :math:`h_0` are useful only if accompanied by a measure of uncertainty.
Pointwise confidence intervals (one at each evaluation point) understate uncertainty because
they do not account for the multiplicity of simultaneous statements. Uniform confidence bands
cover the entire function with prescribed probability, giving a more honest picture. Two
approaches are available, depending on whether the sieve dimension was chosen by the
data-driven procedure or fixed in advance.

Variance Estimation and the Multiplier Bootstrap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both confidence band constructions rely on the same variance estimation and
bootstrap machinery. The estimation error :math:`\hat{h}_J(x) - h_0(x)` is
approximated by

.. math::

   D_J(x) = (\psi^J(x))' \mathbf{M}_J \hat{\mathbf{u}}_J,

where :math:`\hat{\mathbf{u}}_J = (Y_1 - \hat{h}_J(X_1), \ldots,
Y_n - \hat{h}_J(X_n))'` is the :math:`n \times 1` residual vector. The
heteroskedasticity-robust variance of :math:`D_J(x)` is estimated by

.. math::

   \hat{\sigma}_J^2(x) = (\psi^J(x))' \mathbf{M}_J
   \widehat{\mathbf{U}}_{J,J} \mathbf{M}_J' \psi^J(x),

where :math:`\widehat{\mathbf{U}}_{J,J}` is a diagonal matrix with entries
:math:`\hat{u}_{i,J}^2`. The derivative counterparts are

.. math::

   D_J^a(x) = (\partial^a \psi^J(x))' \mathbf{M}_J \hat{\mathbf{u}}_J,
   \quad
   \hat{\sigma}_J^{a\,2}(x) = (\partial^a \psi^J(x))' \mathbf{M}_J
   \widehat{\mathbf{U}}_{J,J} \mathbf{M}_J' (\partial^a \psi^J(x)).

The multiplier bootstrap replicates these statistics by replacing
:math:`\hat{\mathbf{u}}_J` with
:math:`\hat{\mathbf{u}}_J^* = (\hat{u}_{1,J}\varpi_1, \ldots,
\hat{u}_{n,J}\varpi_n)'`, where :math:`(\varpi_i)_{i=1}^n` are drawn i.i.d.
:math:`N(0,1)` independently of the data. This yields

.. math::

   D_J^*(x) = (\psi^J(x))' \mathbf{M}_J \hat{\mathbf{u}}_J^*, \quad
   D_J^{a*}(x) = (\partial^a \psi^J(x))' \mathbf{M}_J \hat{\mathbf{u}}_J^*.

Drawing the weights many times (1000 draws is sufficient) and computing the
resulting sup-:math:`t` statistics gives the critical values for the
confidence bands.

Undersmoothing Approach
~~~~~~~~~~~~~~~~~~~~~~~

Given a fixed sieve dimension :math:`J`, the multiplier bootstrap constructs
uniform confidence bands

.. math::

   C_{n,J}(x) = \left[\hat{h}_J(x) \pm z_{1-\alpha,J}^* \hat{\sigma}_J(x)
   \right],

where :math:`z_{1-\alpha,J}^*` is the :math:`(1-\alpha)` quantile of
the bootstrap sup-:math:`t` statistic

.. math::

   \sup_{x \in \mathcal{X}} \left|\frac{D_J^*(x)}{\hat{\sigma}_J(x)}\right|.

For derivatives, the band is

.. math::

   C_{n,J}^a(x) = \left[\partial^a \hat{h}_J(x) \pm z_{1-\alpha,J}^{a*}
   \hat{\sigma}_J^a(x)\right],

where :math:`z_{1-\alpha,J}^{a*}` is the corresponding quantile using
:math:`D_J^{a*}(x) / \hat{\sigma}_J^a(x)`.

These bands have correct coverage provided :math:`J` exceeds the
oracle-optimal dimension :math:`J_0`, so that approximation bias is of smaller
order than sampling uncertainty. In practice, :math:`J_0` depends on the
unknown smoothness of :math:`h_0` and other unknown model features, so the
researcher must guess how large :math:`J` should be. Choosing :math:`J` too
conservatively produces wider bands than necessary.

Data-Driven Adaptive UCBs
~~~~~~~~~~~~~~~~~~~~~~~~~

The data-driven UCBs avoid the efficiency loss of undersmoothing by folding
information from the dimension selection step into the critical value. Let
:math:`\underline{p} > d/2` denote the minimal degree of smoothness
assumed for :math:`h_0` (for instance, :math:`\underline{p} = 1` when
:math:`X` is scalar and :math:`h_0` is assumed to be at least Lipschitz).
Define :math:`\hat{A} = \log\log\tilde{J}` and the candidate set

.. math::

   \hat{\mathcal{J}}_- = \begin{cases}
   \{J \in \hat{\mathcal{J}} : J < \hat{J}_n\}
   & \text{if } \tilde{J} = \hat{J}, \\
   \hat{\mathcal{J}} & \text{if } \tilde{J} = \hat{J}_n.
   \end{cases}

The critical value :math:`z_{1-\alpha}^*` is the :math:`(1-\alpha)` quantile
of

.. math::

   \sup_{(x,J) \in \mathcal{X} \times \hat{\mathcal{J}}_-}
   \left|\frac{D_J^*(x)}{\hat{\sigma}_J(x)}\right|,

which takes the supremum over both evaluation points and candidate dimensions,
so the resulting bands are robust to the particular dimension selected. The
:math:`100(1-\alpha)\%` UCB for :math:`h_0` is

.. math::

   C_n(x) = \left[\hat{h}_{\tilde{J}}(x) \pm \text{cv}^*(x)\,
   \hat{\sigma}_{\tilde{J}}(x)\right],

where the critical value function has two cases

.. math::

   \text{cv}^*(x) = \begin{cases}
   z_{1-\alpha}^* + \hat{A}\,\theta_{1-\hat{\alpha}}^*
   & \text{if } \tilde{J} = \hat{J}, \\
   z_{1-\alpha}^* + \hat{A}\,\max\!\left\{\theta_{1-\hat{\alpha}}^*,\;
   \tilde{J}^{-\underline{p}/d} / \hat{\sigma}_{\tilde{J}}(x)\right\}
   & \text{if } \tilde{J} = \hat{J}_n.
   \end{cases}

In the mildly ill-posed regime, which covers the vast majority of simulations
and the empirical application, only the first case is relevant. The second case
includes a bias correction for possible residual approximation bias when the
conservative truncation binds, as can happen in the severely ill-posed regime.

Derivative UCBs
~~~~~~~~~~~~~~~

UCBs for derivatives :math:`\partial^a h_0` (with :math:`0 < |a| <
\underline{p}`) follow the same recipe. The critical value
:math:`z_{1-\alpha}^{a*}` is the :math:`(1-\alpha)` quantile of

.. math::

   \sup_{(x,J) \in \mathcal{X} \times \hat{\mathcal{J}}_-}
   \left|\frac{D_J^{a*}(x)}{\hat{\sigma}_J^a(x)}\right|,

and the UCB is

.. math::

   C_n^a(x) = \left[\partial^a \hat{h}_{\tilde{J}}(x) \pm
   \text{cv}^{a*}(x)\,\hat{\sigma}_{\tilde{J}}^a(x)\right],

where

.. math::

   \text{cv}^{a*}(x) = \begin{cases}
   z_{1-\alpha}^{a*} + \hat{A}\,\theta_{1-\hat{\alpha}}^*
   & \text{if } \tilde{J} = \hat{J}, \\
   z_{1-\alpha}^{a*} + \hat{A}\,\max\!\left\{\theta_{1-\hat{\alpha}}^*,\;
   \tilde{J}^{(|a|-\underline{p})/d} / \hat{\sigma}_{\tilde{J}}^a(x)\right\}
   & \text{if } \tilde{J} = \hat{J}_n.
   \end{cases}

Honesty and Adaptivity
~~~~~~~~~~~~~~~~~~~~~~

In the mildly ill-posed regime, the data-driven UCBs satisfy two properties.

- **Honest** in that coverage is guaranteed uniformly over a generic class of
  data-generating processes

  .. math::

     \liminf_{n \to \infty} \inf_{h_0 \in \mathcal{G}}
     \mathbb{P}_{h_0}(h_0(x) \in C_n(x) \;\forall x \in \mathcal{X})
     \geq 1 - \alpha.

- **Adaptive** in that the band width contracts at (within a :math:`\log\log n`
  factor of) the minimax rate

  .. math::

     \sup_{x \in \mathcal{X}} |C_n(x)| = O_p\left((\log\log n)
     \left(\frac{\log n}{n}\right)^{\frac{p}{2(p+\varsigma)+d}}\right).

The same properties hold for derivative UCBs, with the band width contracting
at the derivative minimax rate

.. math::

   (\log\log n)\left(\frac{\log n}{n}\right)^{\frac{p-|a|}{2(p+\varsigma)+d}}.

In the severely ill-posed regime, the UCBs with the critical value
corresponding to :math:`\tilde{J} = \hat{J}_n` have valid (and in fact
conservative) coverage. In simulation studies calibrated to an empirically
relevant Engel curve design that is severely ill-posed, the UCBs maintain
correct coverage across all sample sizes despite the coverage guarantee being
formally established only for the mildly ill-posed case.

The data-driven bands are therefore asymptotically more efficient than
undersmoothed bands, which sacrifice estimation efficiency for coverage. In
simulation studies calibrated to an international trade application, the
data-driven bands are approximately 40% narrower than undersmoothed bands with
comparable coverage, and have substantially higher power for detecting
departures from parametric specifications.


Multivariate Basis Construction
-------------------------------

When :math:`X` is multivariate, the module supports three types of basis
construction from the marginal B-spline bases.

- **Tensor product** uses the full Kronecker product
  :math:`\psi^J(x) = \psi_1^{J_1}(x_1) \otimes \cdots \otimes
  \psi_d^{J_d}(x_d)`, yielding :math:`\prod_i J_i` basis functions. This
  provides the most flexible approximation but the dimension grows
  exponentially.

- **Additive** uses the concatenation of marginal bases,
  yielding :math:`\sum_i J_i` basis functions and restricting
  :math:`h_0` to an additive structure :math:`h_0(x) = \sum_i h_i(x_i)`.

- **Generalized linear product (GLP)** is a hierarchical construction that
  includes main effects and selected interactions, providing a compromise
  between the tensor and additive bases.

The theory and data-driven procedures apply to all three constructions.


Extensions
----------

The data-driven procedures carry over to structured models that mitigate the
curse of dimensionality when additional assumptions on :math:`h_0` are
warranted.

Additive Structural Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When :math:`h_0` is assumed to take the additive form

.. math::

   h_0(x) = c_0 + h_{10}(x_1) + \ldots + h_{d0}(x_d),

where :math:`c_0` is an intercept and the component functions :math:`h_{i0}`
are suitably normalized for identifiability, the curse of dimensionality can be
circumvented. Stone (1985) showed that imposing additivity in nonparametric
regression yields estimators that achieve the same optimal rate for general
:math:`d` as for :math:`d = 1`.

The additive basis concatenates centered marginal B-spline bases. For each
coordinate :math:`i`, the centered basis functions are

.. math::

   \tilde{\psi}_{Jj}(x_i) = \psi_{Jj}(x_i) - \int_0^1 \psi_{Jj}(v)\,dv,

and the full basis vector is

.. math::

   \psi^J(x) = (1, \tilde{\psi}_1^J(x_1)', \ldots,
   \tilde{\psi}_d^J(x_d)')'.

The data-driven choice of :math:`J` follows the
same procedure as before, just with this basis plugged in. UCBs for each
component :math:`h_{i0}` restrict the bootstrap sup-statistics to the
coordinates of interest, taking supremums only over the support
:math:`\mathcal{X}_i` of :math:`x_i`.

Partially Linear Structural Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In many applications, some regressors enter :math:`h_0` linearly while others
enter nonparametrically, giving the partially linear specification (Ai and
Chen, 2003)

.. math::

   h_0(x) = h_{10}(x_1) + x_2' \beta_0,

where :math:`x = (x_1', x_2')'`, :math:`h_{10}` is an unknown function of the
:math:`d_1`-dimensional subvector :math:`x_1`, and :math:`\beta_0` is an
unknown finite-dimensional parameter vector. When :math:`X` is exogenous, this
reduces to the partially linear regression model of Robinson (1988).

The basis vector takes the form
:math:`\psi^J(x) = (\psi_1^J(x_1)', x_2')'`, and the TSLS estimator jointly
estimates the sieve coefficients and the linear parameters. For dimension
selection and UCBs targeting :math:`h_{10}`, one substitutes
:math:`\psi_0^J(x_1) = (\psi_1^J(x_1)', 0_{d_2})'` in place of
:math:`\psi^J(x)` when computing contrasts and variance terms, so that the
:math:`t`-statistics depend only on :math:`x_1` and supremums are taken over
:math:`\mathcal{X}_1`.

Nonparametric Regression
~~~~~~~~~~~~~~~~~~~~~~~~

Setting :math:`W = X` reduces the NPIV model to nonparametric regression.
The instrument projection drops out and the TSLS estimator collapses to
ordinary least squares

.. math::

   \hat{h}_J(x) = (\psi^J(x))' \hat{c}_J, \quad
   \hat{c}_J = (\boldsymbol{\Psi}_J' \boldsymbol{\Psi}_J)^{-}
   \boldsymbol{\Psi}_J' \mathbf{Y},

and :math:`\mathbf{M}_J = (\boldsymbol{\Psi}_J' \boldsymbol{\Psi}_J)^{-}
\boldsymbol{\Psi}_J'`. The maximum feasible dimension in Step 1 simplifies to

.. math::

   \hat{J}_{\max} = \min\left\{J \in \mathcal{T} : J\sqrt{\log J}\,v_n
   \leq 10\sqrt{n} < J^+\sqrt{\log J^+}\,v_n\right\},

with :math:`v_n = \max\{1, (0.1 \log n)^4\}` replacing the ill-posedness
measure :math:`\hat{s}_J^{-1}`. The Lepski selection and bootstrap are
otherwise unchanged, and the data-driven UCBs simplify to

.. math::

   C_n(x) = \left[\hat{h}_{\tilde{J}}(x) \pm
   (z_{1-\alpha}^* + \hat{A}\,\theta_{1-\hat{\alpha}}^*)\,
   \hat{\sigma}_{\tilde{J}}(x)\right],

without the additional bias correction term present in the severely ill-posed
case, since nonparametric regression corresponds to :math:`\varsigma = 0`
(the mildly ill-posed regime with the strongest possible instruments). The
conservative truncation :math:`\hat{J}_n` is not needed, so
:math:`\tilde{J} = \hat{J}` directly.

The minimax rate-adaptivity and honesty-plus-adaptivity guarantees from the
general NPIV case still hold here, specialized to the nonparametric regression
rates.


.. note::

   For complete theoretical details including formal regularity conditions,
   proofs of minimax rate adaptivity, and extensions to partially linear and
   partially additive models, refer to `Chen, Christensen, and Kankanala (2024)
   <https://arxiv.org/abs/2107.11869>`_. The undersmoothing UCB approach
   builds on `Chen and Christensen (2018)
   <https://arxiv.org/abs/1508.03365>`_.
