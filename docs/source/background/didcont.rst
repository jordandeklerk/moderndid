.. _background-didcont:

DiD with Continuous Treatments
==============================

The ``didcont`` module implements difference-in-differences estimation for settings where treatment intensity varies continuously across units, following the methodology of `Callaway, Goodman-Bacon, and Sant'Anna (2024) <https://arxiv.org/abs/2107.02637>`_.
This approach addresses the unique challenges that arise when treatment is not simply binary but operates with varying intensity or "dose" across units.

Continuous treatments arise naturally in many empirical settings. Pollution exposure dissipates across space, affecting locations near sources more severely than distant ones. Localities spend different amounts on public goods and services. Students choose how long to stay in school. Medicare subsidies vary with hospital patient composition. In all these cases, treatment intensity varies substantially, and researchers often care about both the overall effect of the policy and how effects vary with dose.

This module provides tools for identifying, estimating, and conducting inference on well-defined causal parameters in continuous DiD designs. A central insight is that with continuous treatments, there are fundamentally two types of causal parameters, level effects and causal responses, each requiring different identifying assumptions.

Setup and Notation
------------------

Consider a setup with two time periods, :math:`t = 1` (pre-treatment) and :math:`t = 2` (post-treatment). In the first period, no unit is treated. In the second period, units receive a treatment "dose" denoted :math:`D_i`, which can be continuous or multi-valued discrete. The support of :math:`D` is :math:`\mathcal{D} = \{0\} \cup \mathcal{D}_{+}`, where :math:`\mathcal{D}_{+}` contains all positive doses and zero represents untreated units.

.. admonition:: Assumption 1 (Random Sampling)

   The observed data consist of :math:`\{Y_{i,t=2}, Y_{i,t=1}, D_i\}_{i=1}^n`, which is independent and identically distributed.

.. admonition:: Assumption 2 (Continuous or Multi-Valued Discrete Treatment)

   In period :math:`t = 1`, no unit is treated, while in period :math:`t = 2`, the treatment dosage :math:`D` has support :math:`\mathcal{D} = \{0\} \cup \mathcal{D}_{+}` and is either

   (a) *Continuous*. :math:`\mathcal{D}_{+} = \mathcal{D}_{+}^c = [d_L, d_U]` with :math:`0 < d_L < d_U < \bar{d} < \infty`. The density :math:`f_{D|D>0}` satisfies :math:`a_f^{-1} < f_{D|D>0}(d) < a_f` for some positive constant :math:`a_f < \infty` and all :math:`d \in \mathcal{D}_{+}^c`, and :math:`\mathbb{E}[\Delta Y | D = d]` is continuously differentiable on :math:`\mathcal{D}_{+}^c`.

   (b) *Multi-valued discrete*. :math:`\mathcal{D}_{+} = \mathcal{D}_{+}^{mv} = \{d_1, d_2, \ldots, d_J\}` where :math:`0 < d_1 < d_2 < \cdots < d_J < \bar{d} < \infty`, and :math:`\mathbb{P}(D = d) > 0` for all :math:`d \in \mathcal{D}`.

   In both cases, we require a positive mass of untreated units, :math:`\mathbb{P}(D = 0) > 0`.

Potential Outcomes Framework
----------------------------

We adopt the potential outcomes framework where :math:`Y_{i,t}(d)` denotes the potential outcome for unit :math:`i` at time :math:`t` under dose :math:`d`. The observed outcome in each period satisfies

.. math::

   Y_{i,t=1} = Y_{i,t=1}(0), \quad Y_{i,t=2} = Y_{i,t=2}(D_i).

.. admonition:: Assumption 3 (No-Anticipation and Observed Outcomes)

   For all units and all :math:`d \in \mathcal{D}`,

   .. math::

      Y_{i,t=1} = Y_{i,t=1}(d) = Y_{i,t=1}(0), \quad Y_{i,t=2} = Y_{i,t=2}(D_i).

This assumption rules out anticipatory effects, ensuring that in the pre-treatment period, all units exhibit their untreated potential outcomes regardless of their future dose. In the post-treatment period, we observe the potential outcome corresponding to the actual dose received. Let :math:`\Delta Y = Y_{t=2} - Y_{t=1}` denote the change in outcomes from period 1 to period 2.

Parameters of Interest
----------------------

With continuous treatments, two fundamentally different types of causal effects can be defined. Understanding the distinction between these parameters is crucial for proper interpretation of continuous DiD results.

Level Treatment Effects
~~~~~~~~~~~~~~~~~~~~~~~

The **level treatment effect** of dose :math:`d` for a given unit is the difference between its potential outcome under dose :math:`d` and its untreated potential outcome

.. math::

   Y_{t=2}(d) - Y_{t=2}(0).

This extends the binary treatment effect concept to a "dose-response function." The **average treatment effect on the treated** at dose :math:`d` among units receiving dose :math:`d'` is

.. math::

   ATT(d | d') = \mathbb{E}[Y_{t=2}(d) - Y_{t=2}(0) | D = d'].

When :math:`d' = d`, this yields :math:`ATT(d | d)`, the average effect of dose :math:`d` compared to no treatment among units that actually received dose :math:`d`. This is the natural extension of the binary ATT to the continuous case.

The population-level **average treatment effect** is

.. math::

   ATE(d) = \mathbb{E}[Y_{t=2}(d) - Y_{t=2}(0)].

Note that :math:`ATT(d | d)` and :math:`ATE(d)` differ when there is selection into dose group :math:`d` on the basis of treatment effects. When units with larger treatment effects systematically choose higher doses, we have :math:`ATT(d | d) \neq ATE(d)`.

Causal Responses
~~~~~~~~~~~~~~~~

The **causal response** at dose :math:`d` measures the effect of a marginal change in the dose. For continuous treatments, the causal response is defined as the derivative of the potential outcome with respect to dose

.. math::

   Y'_{t=2}(d) = \lim_{h \to 0^+} \frac{Y_{t=2}(d + h) - Y_{t=2}(d)}{h}.

For discrete treatments, the causal response between adjacent doses :math:`d_j` and :math:`d_{j-1}` is

.. math::

   Y_{t=2}(d_j) - Y_{t=2}(d_{j-1}).

When treatment is binary, level treatment effects and causal responses coincide, but they do not under a continuous treatment. This distinction has important practical implications since even if all :math:`ATT(d | d)` parameters are large and positive, some causal response parameters could be zero or negative.

The **average causal response on the treated** (ACRT) for continuous treatments is

.. math::

   ACRT(d | d') = \left.\frac{\partial ATT(l | d')}{\partial l}\right|_{l=d} = \left.\frac{\partial \mathbb{E}[Y_{t=2}(l) | D = d']}{\partial l}\right|_{l=d}.

When :math:`d' = d`, this gives the average marginal effect of increasing the dose among units at that dose level. Equivalently, :math:`ACRT(d | d)` equals the derivative of the :math:`t = 2` average potential outcome for units that received dose :math:`d`, evaluated at :math:`d`.

The population-level **average causal response** is

.. math::

   ACR(d) = \frac{\partial ATE(d)}{\partial d} = \frac{\partial \mathbb{E}[Y_{t=2}(d)]}{\partial d}.

For discrete treatments, the analogous parameters are

.. math::

   ACRT(d_j | d_k) &= \mathbb{E}[Y_{t=2}(d_j) - Y_{t=2}(d_{j-1}) | D = d_k], \\
   ACR(d_j) &= \mathbb{E}[Y_{t=2}(d_j) - Y_{t=2}(d_{j-1})].

Summary Parameters
~~~~~~~~~~~~~~~~~~

In practice, researchers often want to aggregate these functional parameters into lower-dimensional summary measures. Natural aggregations use the dose distribution among treated units

.. math::

   ATT^o &= \mathbb{E}[ATT(D | D) | D > 0], \quad & ATE^o &= \mathbb{E}[ATE(D) | D > 0], \\
   ACRT^o &= \mathbb{E}[ACRT(D | D) | D > 0], \quad & ACR^o &= \mathbb{E}[ACR(D) | D > 0].

These provide "best" approximations in the sense of minimizing the mean squared distance between the summary parameter and the underlying functional parameters. The parameters :math:`ACRT^o` and :math:`ACR^o` are average derivative-type parameters, which have been extensively studied in the econometrics literature on efficient estimation.

Identification Assumptions
--------------------------

The identification of treatment effect parameters relies on assumptions that restrict how untreated potential outcomes evolve over time across dose groups.

Parallel Trends
~~~~~~~~~~~~~~~

The standard **parallel trends** assumption extends naturally from the binary case.

.. admonition:: Assumption 4 (Parallel Trends)

   For all :math:`d \in \mathcal{D}`,

   .. math::

      \mathbb{E}[Y_{t=2}(0) - Y_{t=1}(0) | D = d] = \mathbb{E}[Y_{t=2}(0) - Y_{t=1}(0) | D = 0].

This assumption states that the average evolution of untreated potential outcomes would be the same across all dose groups in the absence of treatment. Under parallel trends, the untreated group provides a valid counterfactual for the path of outcomes that treated units would have experienced without treatment.

Parallel trends is an assumption about untreated potential outcomes :math:`Y_t(0)` only. It says nothing about how treated potential outcomes :math:`Y_t(d)` for :math:`d > 0` evolve across dose groups.

Strong Parallel Trends
~~~~~~~~~~~~~~~~~~~~~~

A different assumption is required to identify causal response parameters and to make valid comparisons across dose groups.

.. admonition:: Assumption 5 (Strong Parallel Trends)

   For all :math:`d \in \mathcal{D}`,

   .. math::

      \mathbb{E}[Y_{t=2}(d) - Y_{t=1}(0)] = \mathbb{E}[Y_{t=2}(d) - Y_{t=1}(0) | D = d].

Under Assumption 3, the right-hand side of this equation is the observed average evolution of outcomes for dose group :math:`d`. Strong parallel trends says that the average evolution of outcomes for the entire population if all experienced dose :math:`d` (the left-hand side) equals the path of outcomes that dose group :math:`d` actually experienced.

An equivalent characterization under Assumption 4 is that strong parallel trends holds if and only if

.. math::

   ATT(d | d) = ATE(d) \quad \text{for all } d \in \mathcal{D}.

This means strong parallel trends rules out selection-on-gains into particular dose groups. While this condition does not impose full treatment effect homogeneity, it does ensure that observed outcome changes for each dose group reflect what would have happened to all other groups had they received that dose.

.. note::

   Conventional pre-tests for differential pre-trends cannot distinguish between Assumptions 4 and 5. Because only untreated potential outcomes are observed before treatment, pre-treatment periods cannot test the additional content of strong parallel trends, which necessarily involves treated potential outcomes :math:`Y_t(d)` for :math:`d > 0`.

Relationship Between Assumptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In general, Assumptions 4 and 5 are non-nested, though Assumption 5 will typically be stronger in most applications. To see this, consider that Assumption 4 restricts only the evolution of :math:`Y_t(0)` across dose groups, while Assumption 5 restricts the evolution of :math:`Y_t(d)` for each :math:`d \in \mathcal{D}`.

When maintained jointly with Assumption 4, Assumption 5 can be understood as a structural assumption that allows extrapolation of treatment effects, ensuring that the treatment effects of dose :math:`d` among dose group :math:`d` equal the treatment effects of dose :math:`d` for the entire population.

Identification Results
----------------------

Identification Under Parallel Trends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Under parallel trends (Assumption 4), the dose-specific average treatment effect on the treated is identified. Specifically, under Assumptions 1 to 4, :math:`ATT(d | d)` is identified for all :math:`d \in \mathcal{D}_{+}`, and it is given by

.. math::

   ATT(d | d) = \mathbb{E}[\Delta Y | D = d] - \mathbb{E}[\Delta Y | D = 0].

Furthermore, :math:`ATT^o = \mathbb{E}[\Delta Y | D > 0] - \mathbb{E}[\Delta Y | D = 0]`.

The identification argument proceeds as follows. By definition,

.. math::

   ATT(d | d) = \mathbb{E}[Y_{t=2}(d) - Y_{t=2}(0) | D = d].

Adding and subtracting :math:`\mathbb{E}[Y_{t=1}(0) | D = d]` and applying Assumption 4,

.. math::

   ATT(d | d) &= \mathbb{E}[Y_{t=2}(d) - Y_{t=1}(0) | D = d] - \mathbb{E}[Y_{t=2}(0) - Y_{t=1}(0) | D = d] \\
              &= \mathbb{E}[Y_{t=2}(d) - Y_{t=1}(0) | D = d] - \mathbb{E}[Y_{t=2}(0) - Y_{t=1}(0) | D = 0] \\
              &= \mathbb{E}[\Delta Y | D = d] - \mathbb{E}[\Delta Y | D = 0],

where the final equality uses the fact that :math:`Y_{t=2}(d)` and :math:`Y_{t=1}(0)` are observed for units with :math:`D = d`.

Non-Identification of Causal Responses Under Parallel Trends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A central result is that causal response parameters are **not identified** under parallel trends alone. Under Assumptions 1 to 4, the following decompositions reveal the source of the identification failure.

(a) For continuous treatments with :math:`d \in \mathcal{D}_{+}^c`,

.. math::

   \frac{\partial \mathbb{E}[\Delta Y | D = d]}{\partial d} &= \frac{\partial ATT(d | d)}{\partial d} \\
   &= ACRT(d | d) + \underbrace{\left.\frac{\partial ATT(d | l)}{\partial l}\right|_{l=d}}_{\text{selection bias}}.

(b) For any :math:`(h, l) \in \mathcal{D} \times \mathcal{D}` with :math:`h > l`,

.. math::

   \mathbb{E}[\Delta Y | D = h] - \mathbb{E}[\Delta Y | D = l] &= ATT(h | h) - ATT(l | l) \\
   &= \underbrace{\mathbb{E}[Y_{t=2}(h) - Y_{t=2}(l) | D = h]}_{\text{causal response}} \\
   &\quad + \underbrace{(ATT(l | h) - ATT(l | l))}_{\text{selection bias}}.

The proof for part (b) is instructive. Starting from the identification result above,

.. math::

   ATT(h | h) - ATT(l | l) = \mathbb{E}[Y_{t=2}(h) - Y_{t=2}(0) | D = h] - \mathbb{E}[Y_{t=2}(l) - Y_{t=2}(0) | D = l].

Adding and subtracting :math:`\mathbb{E}[Y_{t=2}(l) | D = h]`,

.. math::

   &= \mathbb{E}[Y_{t=2}(h) - Y_{t=2}(l) | D = h] \\
   &\quad + \mathbb{E}[Y_{t=2}(l) - Y_{t=2}(0) | D = h] - \mathbb{E}[Y_{t=2}(l) - Y_{t=2}(0) | D = l] \\
   &= \mathbb{E}[Y_{t=2}(h) - Y_{t=2}(l) | D = h] + (ATT(l | h) - ATT(l | l)).

The selection bias term :math:`ATT(l | h) - ATT(l | l)` captures the fact that different dose groups may experience different treatment effects at the same dose :math:`l`. Even if untreated potential outcomes evolve identically (parallel trends), comparing outcome paths between dose groups conflates causal responses with this selection-on-gains phenomenon.

For discrete treatments, taking :math:`h = d_j` and :math:`l = d_{j-1}` yields

.. math::

   \mathbb{E}[\Delta Y | D = d_j] - \mathbb{E}[\Delta Y | D = d_{j-1}] &= ACRT(d_j | d_j) \\
   &\quad + \underbrace{ATT(d_{j-1} | d_j) - ATT(d_{j-1} | d_{j-1})}_{\text{selection bias}}.

Identification Under Strong Parallel Trends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Under strong parallel trends (Assumption 5), both level effects and causal responses are identified without selection bias. The following results hold under Assumptions 1 to 3 and 5.

(a) For :math:`d \in \mathcal{D}_{+}`,

.. math::

   ATE(d) = \mathbb{E}[\Delta Y | D = d] - \mathbb{E}[\Delta Y | D = 0].

(b) When treatment is continuous, for :math:`d \in \mathcal{D}_{+}^c`,

.. math::

   ACR(d) = \frac{\partial \mathbb{E}[\Delta Y | D = d]}{\partial d} = \frac{\partial ATE(d)}{\partial d}.

(c) For any :math:`(h, l) \in \mathcal{D} \times \mathcal{D}`,

.. math::

   ATE(h) - ATE(l) = \mathbb{E}[Y_{t=2}(h) - Y_{t=2}(l)] = \mathbb{E}[\Delta Y | D = h] - \mathbb{E}[\Delta Y | D = l].

For part (a), the argument is similar to the identification under parallel trends but uses Assumption 5 instead

.. math::

   ATE(d) &= \mathbb{E}[Y_{t=2}(d) - Y_{t=2}(0)] \\
          &= \mathbb{E}[Y_{t=2}(d) - Y_{t=1}(0)] - \mathbb{E}[Y_{t=2}(0) - Y_{t=1}(0)] \\
          &= \mathbb{E}[Y_{t=2}(d) - Y_{t=1}(0) | D = d] - \mathbb{E}[Y_{t=2}(0) - Y_{t=1}(0) | D = 0] \\
          &= \mathbb{E}[\Delta Y | D = d] - \mathbb{E}[\Delta Y | D = 0],

where the third equality applies Assumption 5 to both terms.

Parts (b) and (c) follow because strong parallel trends ensures that lower-dose groups are valid counterfactuals for higher-dose groups. The selection bias term vanishes since :math:`ATT(l | h) = ATT(l | l) = ATE(l)` for all :math:`h, l`.

Under Assumptions 1 to 3 and 5, the summary parameters have the following identification results.

(a) :math:`ATE^o = \mathbb{E}[\Delta Y | D > 0] - \mathbb{E}[\Delta Y | D = 0]`.

(b) For continuous treatments,

.. math::

   ACR^o = \mathbb{E}\left[\left.\frac{\partial \mathbb{E}[\Delta Y | D = d]}{\partial d}\right|_{d=D} \,\middle|\, D > 0\right] = \int_{d_L}^{d_U} \left.\frac{\partial \mathbb{E}[\Delta Y | D = d]}{\partial d}\right|_{d=s} f_{D|D>0}(s) \, ds.

(c) For discrete treatments,

.. math::

   ACR^o = \sum_{j=1}^{J} \left(\mathbb{E}[\Delta Y | D = d_j] - \mathbb{E}[\Delta Y | D = d_{j-1}]\right) \mathbb{P}(D = d_j | D > 0).

The Case Without Untreated Units
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In some applications, all units receive some positive amount of treatment. Without untreated units, it is infeasible to directly recover :math:`ATT(d | d)` or :math:`ATE(d)`. However, a natural alternative is to compare dose group :math:`d` to dose group :math:`d_L` (the lowest dose).

Under parallel trends, when there are no untreated units,

.. math::

   \mathbb{E}[\Delta Y | D = d] - \mathbb{E}[\Delta Y | D = d_L] = ATT(d | d) - ATT(d_L | d_L).

This comparison is related to underlying causal parameters, but the right-hand side mixes together the average causal response of moving from :math:`d_L` to :math:`d` with selection bias.

Under strong parallel trends,

.. math::

   \mathbb{E}[\Delta Y | D = d] - \mathbb{E}[\Delta Y | D = d_L] = ATE(d) - ATE(d_L) = \mathbb{E}[Y_{t=2}(d) - Y_{t=2}(d_L)],

which has a clean causal interpretation without selection bias.

Estimation Methods
------------------

Given the identification results above, this section describes estimation procedures that target well-defined causal parameters.

Discrete Treatments
~~~~~~~~~~~~~~~~~~~

When the treatment is multi-valued discrete, estimation is straightforward. Regressing outcome changes on a saturated set of dose indicators with untreated units as the omitted category,

.. math::

   \Delta Y_i = \beta_0 + \sum_{j=1}^{J} \mathbf{1}\{D_i = d_j\} \beta_j + \varepsilon_i,

yields OLS coefficients :math:`\widehat{\beta} = (\widehat{\beta}_1, \ldots, \widehat{\beta}_J)'` that consistently estimate :math:`ATT(d_j | d_j)` under parallel trends. Under strong parallel trends, each :math:`\widehat{\beta}_j` estimates :math:`ATE(d_j)`, and :math:`\widehat{\beta}_j - \widehat{\beta}_{j-1}` estimates :math:`ACR(d_j)`.

Continuous Treatments - Sieve Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For continuous treatments, the module provides sieve-based estimation using B-spline basis functions. Consider regression specifications of the form

.. math::

   \Delta Y_i = \sum_{k=1}^{K} \psi_{Kk}(D_i) \beta_{Kk} + \varepsilon_i,

where :math:`\psi^K(d) = (\psi_{K1}(d), \ldots, \psi_{KK}(d))'` is a :math:`K`-dimensional vector of B-spline basis functions (including an intercept), :math:`\beta_K = (\beta_{K1}, \ldots, \beta_{KK})'` is a vector of unknown parameters, and :math:`\varepsilon_i` is an idiosyncratic error term.

The OLS estimator is

.. math::

   \widehat{\beta}_K = \mathbb{E}_n\left[\mathbf{1}\{D > 0\} \psi^K(D) \psi^K(D)'\right]^{-} \mathbb{E}_n\left[\mathbf{1}\{D > 0\} \psi^K(D) (\Delta Y - \mathbb{E}_n[\Delta Y | D = 0])\right],

where for a given matrix :math:`A`, :math:`A^{-}` denotes the Moore-Penrose inverse, and

.. math::

   \mathbb{E}_n[B | D > 0] = \frac{\sum_{i=1}^n \mathbf{1}\{D_i > 0\} B_i}{\sum_{i=1}^n \mathbf{1}\{D_i > 0\}}.

The estimators for the dose-response function and its derivative are

.. math::

   \widehat{ATE}_K(d) = (\psi^K(d))' \widehat{\beta}_K, \quad \widehat{ACR}_K(d) = (\partial \psi^K(d))' \widehat{\beta}_K,

where :math:`\partial \psi^K(d) = (d\psi_{K1}(d)/dd, \ldots, d\psi_{KK}(d)/dd)'` contains the derivatives of the basis functions.

The user controls the spline degree and number of interior knots, allowing flexible modeling of the dose-response relationship. With ``degree=3`` and ``num_knots=0`` (the default), this fits a global cubic polynomial.

Data-Driven Nonparametric Estimation (CCK)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For fully nonparametric estimation without arbitrary tuning parameter choices, the module implements the data-driven sieve estimator of `Chen, Christensen, and Kankanala (2024) <https://arxiv.org/abs/2107.11869>`_. This approach uses dyadic cubic B-splines with adaptive selection of the sieve dimension.

Let :math:`\mathcal{K} = \{(2^k + 3) : k \in \mathbb{N} \cup \{0\}\}` be the set of candidate sieve dimensions. The data-driven choice :math:`\widehat{K}` uses a Lepskii-type selection procedure. The key idea is to select the most parsimonious specification across all candidates, provided that the estimated :math:`ATE_K(d)` curves are not "statistically different" from each other.

**Algorithm (Data-Driven Sieve Dimension Selection)**

1. Compute the data-driven index set of sieve dimensions

   .. math::

      \widehat{\mathcal{K}} = \left\{K \in \mathcal{K} : 0.1(\log \widehat{K}_{\max})^2 \le K \le \widehat{K}_{\max}\right\},

   where :math:`\widehat{K}_{\max} = \min\{K \in \mathcal{K} : K\sqrt{\log K} v_n \le 10\sqrt{n} < K^+\sqrt{\log K^+} v_n\}` with :math:`v_n = \max\{1, (0.1 \log n)^4\}` and :math:`K^+ = \min\{k \in \mathcal{K} : k > K\}`.

2. For bootstrap draws :math:`\{\omega_i\}_{i=1}^n` (iid standard normal, independent of the data), compute the sup-t statistic

   .. math::

      \sup_{(d, K, K_2) \in \mathcal{D}_{+}^c \times \widehat{\mathcal{K}} \times \widehat{\mathcal{K}} : K_2 > K} \left|\mathbb{Z}_n^*(d, K, K_2)\right|,

   where :math:`\mathbb{Z}_n^*(d, K, K_2)` is a normalized bootstrap process comparing estimators at different sieve dimensions. Let :math:`\gamma_{1-\widehat{\alpha}}^*` denote the :math:`(1 - \widehat{\alpha})` quantile.

3. The data-driven choice is

   .. math::

      \widehat{K} = \inf\left\{K \in \widehat{\mathcal{K}} : \sup_{(d, K_2) \in \mathcal{D}_{+}^c \times \widehat{\mathcal{K}} : K_2 > K} \frac{\sqrt{n}|\widehat{ATE}_K(d) - \widehat{ATE}_{K_2}(d)|}{\widehat{\sigma}_{K,K_2}(d)} \le 1.1 \gamma_{1-\widehat{\alpha}}^*\right\}.

The intuition is that if increasing :math:`K` leads to a statistically different estimate of :math:`ATE_K(d)`, then it is "worth it" to increase the dimension. This is how the algorithm trades off bias and variance.

Convergence Rates and Confidence Bands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The data-driven estimators achieve the minimax rate for estimating :math:`ATE(d)` and :math:`ACR(d)` in sup-norm. Under appropriate regularity conditions, let :math:`\mathcal{H}^p` denote the Hölder ball of smoothness :math:`p` and let :math:`p \in [\underline{p}, \bar{p}]` with :math:`\bar{p} > \underline{p} > 0.5`. The following convergence results hold.

For level effects, there exists a universal constant :math:`C_1 > 0` for which

.. math::

   \sup_{p \in [\underline{p}, \bar{p}]} \sup_{ATE(\cdot) \in \mathcal{H}^p} \mathbb{P}_{ATE}\left(\sup_{d \in \mathcal{D}_{+}^c} |(\widehat{ATE}_{\widehat{K}} - ATE)(d)| > C_1 \left(\frac{\log n}{n}\right)^{\frac{p}{2p+1}}\right) \to 0.

For derivatives, when :math:`\underline{p} > 1`, there exists a universal constant :math:`C_1' > 0` for which

.. math::

   \sup_{p \in [\underline{p}, \bar{p}]} \sup_{ATE(\cdot) \in \mathcal{H}^p} \mathbb{P}_{ATE}\left(\sup_{d \in \mathcal{D}_{+}^c} |(\widehat{ACR}_{\widehat{K}} - ACR)(d)| > C_1' \left(\frac{\log n}{n}\right)^{\frac{p-1}{2p+1}}\right) \to 0.

The convergence rates :math:`(\log n / n)^{p/(2p+1)}` for level effects and :math:`(\log n / n)^{(p-1)/(2p+1)}` for derivatives are the minimax rates for estimating functions in Hölder balls under sup-norm loss. As expected, the derivative estimator converges more slowly.

**Uniform Confidence Bands**. The module provides data-driven uniform confidence bands (UCBs) that are both honest (asymptotically correct coverage) and adaptive (contract at the minimax rate). For :math:`ATE(d)`,

.. math::

   C_n(d) = \left[\widehat{ATE}_{\widehat{K}}(d) - (z_{1-\alpha}^* + \widehat{A}\gamma_{1-\widehat{\alpha}}^*) \frac{\widehat{\sigma}_{\widehat{K}}(d)}{\sqrt{n}}, \; \widehat{ATE}_{\widehat{K}}(d) + (z_{1-\alpha}^* + \widehat{A}\gamma_{1-\widehat{\alpha}}^*) \frac{\widehat{\sigma}_{\widehat{K}}(d)}{\sqrt{n}}\right],

where :math:`z_{1-\alpha}^*` is the :math:`(1-\alpha)` quantile of a bootstrap sup-t statistic and :math:`\widehat{A} = \log \log \widehat{K}` inflates critical values to account for potential bias.

Summary Parameter Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Binarized DiD for :math:`ATT^o`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The summary parameter :math:`ATT^o` is estimated by a simple regression

.. math::

   \Delta Y_i = \beta_0^{bin} + D_i^{>0} \beta^{bin} + \epsilon_i,

where :math:`D_i^{>0} = \mathbf{1}\{D_i > 0\}`. The OLS coefficient :math:`\widehat{\beta}^{bin}` consistently estimates :math:`ATT^o` under parallel trends (or :math:`ATE^o` under strong parallel trends).

Average Causal Response
^^^^^^^^^^^^^^^^^^^^^^^

The summary parameter :math:`ACR^o` is estimated using the plug-in principle

.. math::

   \widehat{ACR}^o = \mathbb{E}_n[\widehat{ACR}_{\widehat{K}}(D) | D > 0] = \frac{1}{n_{D>0}} \sum_{i : D_i > 0} \widehat{ACR}_{\widehat{K}}(D_i),

where :math:`n_{D>0} = \sum_{i=1}^n \mathbf{1}\{D_i > 0\}`.

Under appropriate regularity conditions, the estimator is asymptotically normal,

.. math::

   \sqrt{n_{D>0}} \frac{(\widehat{ACR}^o - ACR^o)}{\widehat{\sigma}_{ACR^o}} \xrightarrow{d} \mathcal{N}(0, 1),

where :math:`\widehat{\sigma}_{ACR^o}^2 \xrightarrow{p} V_{ACR}` with :math:`V_{ACR}` being the semiparametric efficiency bound

.. math::

   V_{ACR} = \text{Var}\left[ACR(D) - (\Delta Y - \mathbb{E}[\Delta Y | D, D > 0]) \frac{f'_{D|D>0}(D)}{f_{D|D>0}(D)} \,\middle|\, D > 0\right].

Extensions to Staggered Adoption
--------------------------------

The methodology extends to settings with multiple time periods and variation in treatment timing. Let :math:`G_i` denote the time period when unit :math:`i` first receives a positive dose, with :math:`G_i = \infty` for never-treated units. The potential outcomes are indexed by both timing and dose, :math:`Y_{i,t}(g, d)`.

The **group-time-dose average treatment effect** is

.. math::

   ATE(g, t, d) = \mathbb{E}[Y_t(g, d) - Y_t(0) | G = g],

which measures the average effect in period :math:`t` of becoming treated in period :math:`g` with dose :math:`d`, among units in timing group :math:`g`.

Under a multi-period version of strong parallel trends, this is identified as

.. math::

   ATE(g, t, d) = \mathbb{E}[Y_t - Y_{g-1} | G = g, D = d] - \mathbb{E}[Y_t - Y_{g-1} | G = \infty, D = 0].

The expression involves "long differences" in outcomes from period :math:`g - 1` (the last period before treatment) to :math:`t`. Not-yet-treated units can also be used as a comparison group.

Aggregation Strategies
~~~~~~~~~~~~~~~~~~~~~~

The high-dimensional :math:`ATE(g, t, d)` parameters can be aggregated in two main ways.

**Dose Aggregation**. Averaging across timing groups and time periods yields dose-response functions

.. math::

   ATE^{dose}(d), \quad ACR^{dose}(d),

which highlight heterogeneity across different dose levels. These are analogous to :math:`ATE(d)` and :math:`ACR(d)` in the two-period case.

**Event-Study Aggregation**. Averaging across doses while keeping event-time structure yields

.. math::

   ATT^{es}(e), \quad ACR^{es}(e),

where :math:`e = t - g` is the time since treatment. These highlight how treatment effects and causal responses evolve with length of exposure.

Pre-treatment event-study estimates (:math:`e < 0`) can be used to assess the plausibility of parallel trends assumptions. However, such tests cannot distinguish between standard parallel trends and strong parallel trends, since pre-treatment periods only involve untreated potential outcomes.

.. note::

   For complete theoretical details including formal assumptions, asymptotic properties, and efficiency results, refer to `Callaway, Goodman-Bacon, and Sant'Anna (2024) <https://arxiv.org/abs/2107.02637>`_. The nonparametric estimation procedures build on `Chen, Christensen, and Kankanala (2024) <https://arxiv.org/abs/2107.11869>`_.
