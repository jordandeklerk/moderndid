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

The module provides two key contributions. The first is a data-driven choice of
sieve dimension that achieves minimax convergence rates in sup-norm. The second
is uniform confidence bands that are honest (correct coverage over a class of
data-generating processes) and adaptive (contract at the minimax rate).


The NPIV Model
--------------

The structural function :math:`h_0` is identified by the conditional moment
restriction

.. math::

   \mathbb{E}[Y - h_0(X) \mid W] = 0 \quad \text{(almost surely)},

where :math:`Y` is a scalar outcome, :math:`X` is a vector of possibly
endogenous regressors, and :math:`W` is a vector of instrumental variables.
The conditional distribution of :math:`(X, Y)` given :math:`W` is otherwise
unspecified. This model nests nonparametric regression as the special case
:math:`W = X`.


Sieve TSLS Estimation
---------------------

The function :math:`h_0` is approximated by a linear combination of :math:`J`
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

and :math:`(\cdot)^{-}` denotes the Moore-Penrose inverse.

Estimates of :math:`h_0` and its derivatives are

.. math::

   \hat{h}_J(x) = (\psi^J(x))' \hat{c}_J, \quad
   \partial^a \hat{h}_J(x) = (\partial^a \psi^J(x))' \hat{c}_J.

The performance of the estimator is sensitive to the choice of :math:`J` and
not sensitive to :math:`K` as long as :math:`K \geq J`. If :math:`J` is too
small, the estimator has large bias. If :math:`J` is too large, the estimator
is noisy and confidence bands are uninformatively wide.


Ill-Posedness and Instrument Strength
-------------------------------------

A key feature that distinguishes NPIV from nonparametric regression is the
*sieve measure of ill-posedness*, which quantifies the difficulty of inverting
the conditional expectation operator to recover :math:`h_0`. Let :math:`T:
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

Standard cross validation is not valid for choosing :math:`J` in models with
endogeneity because the cross-validation criterion includes a term
:math:`\mathbb{E}[u(h_0(X) - \hat{h}_J(X))]` that may be non-negligible when
:math:`\mathbb{E}[u \mid X] \neq 0`.

The module implements a two-step procedure for data-driven choice of sieve
dimension.

Step 1: Maximum Feasible Dimension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The search grid is determined by

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

A multiplier bootstrap determines the critical value
:math:`\theta_{1-\hat{\alpha}}^*` as the :math:`(1-\hat{\alpha})` quantile of

.. math::

   \sup_{\{(x, J, J_2) \in \mathcal{X} \times \hat{\mathcal{J}} \times
   \hat{\mathcal{J}} : J_2 > J\}} \left|
   \frac{D_J^*(x) - D_{J_2}^*(x)}{\hat{\sigma}_{J,J_2}(x)}
   \right|,

where

.. math::

   D_J^*(x) = (\psi^J(x))' \mathbf{M}_J \hat{\mathbf{u}}_J^*

is a bootstrap version of the estimation error using i.i.d. :math:`N(0,1)`
weights.

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

Undersmoothing Approach
~~~~~~~~~~~~~~~~~~~~~~~

Given a fixed sieve dimension :math:`J`, the multiplier bootstrap constructs
uniform confidence bands

.. math::

   C_{n,J}(x) = \left[\hat{h}_J(x) \pm z_{1-\alpha,J}^* \hat{\sigma}_J(x)
   \right],

where :math:`\hat{\sigma}_J^2(x)` is the heteroskedasticity-robust variance
estimate and :math:`z_{1-\alpha,J}^*` is the :math:`(1-\alpha)` quantile of
the bootstrap sup-:math:`t` statistic

.. math::

   \sup_{x \in \mathcal{X}} \left|\frac{D_J^*(x)}{\hat{\sigma}_J(x)}\right|.

This approach requires :math:`J` to be chosen larger than optimal for
estimation so that bias is negligible, resulting in wider bands.

Data-Driven Adaptive UCBs
~~~~~~~~~~~~~~~~~~~~~~~~~

The data-driven UCBs avoid the efficiency loss of undersmoothing. The critical
value combines the bootstrap quantile with a penalty from the dimension
selection step

.. math::

   C_n(x) = \left[\hat{h}_{\tilde{J}}(x) \pm \text{cv}^*(x)\,
   \hat{\sigma}_{\tilde{J}}(x)\right],

where

.. math::

   \text{cv}^*(x) = z_{1-\alpha}^* + (\log\log \tilde{J})\,
   \theta_{1-\hat{\alpha}}^*.

Here :math:`z_{1-\alpha}^*` is the :math:`(1-\alpha)` quantile of

.. math::

   \sup_{(x,J) \in \mathcal{X} \times \hat{\mathcal{J}}_-}
   \left|\frac{D_J^*(x)}{\hat{\sigma}_J(x)}\right|,

which takes the supremum over both evaluation points and candidate dimensions,
providing robustness to the choice of :math:`J`.

For derivative UCBs, the construction is analogous using derivative basis
functions and variance estimates.

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

- **Additive** uses the concatenation of marginal bases
  :math:`\psi^J(x) = (\psi_1^{J_1}(x_1)', \ldots, \psi_d^{J_d}(x_d)')'`,
  yielding :math:`\sum_i J_i` basis functions. This restricts
  :math:`h_0(x) = \sum_i h_i(x_i)` to an additive structure.

- **Generalized linear product (GLP)** is a hierarchical construction that
  includes main effects and selected interactions, providing a compromise
  between the tensor and additive bases.

The theory and data-driven procedures apply to all three constructions.


.. note::

   For complete theoretical details including formal regularity conditions,
   proofs of minimax rate adaptivity, and extensions to partially linear and
   partially additive models, refer to `Chen, Christensen, and Kankanala (2024)
   <https://arxiv.org/abs/2107.11869>`_. The undersmoothing UCB approach
   builds on `Chen and Christensen (2018)
   <https://arxiv.org/abs/1508.03365>`_.
