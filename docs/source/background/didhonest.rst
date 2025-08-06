.. _background-didhonest:

Honest DiD Sensitivity Analysis
===============================

The ``didhonest`` module provides tools for conducting sensitivity analysis in difference-in-differences (DiD) models based on the work of `Rambachan and Roth (2023) <https://asheshrambachan.github.io/assets/files/hpt-draft.pdf>`_. These methods allow researchers to assess how violations of the parallel trends assumption might affect their conclusions.
This approach addresses the shortcomings of traditional pre-trends tests, which can suffer from low power against meaningful violations of parallel trends and can introduce statistical distortions from pre-testing.

Model Setup and Causal Decomposition
------------------------------------

The functionality of this module is based on a vector of "event-study coefficients", :math:`\hat{\boldsymbol{\beta}} \in \mathbb{R}^{\underline{T}+\bar{T}}`, which can be partitioned into coefficients for pre-treatment and post-treatment periods, :math:`\hat{\boldsymbol{\beta}} = (\hat{\boldsymbol{\beta}}_{pre}', \hat{\boldsymbol{\beta}}_{post}')'`. These coefficients can be obtained from various DiD estimators, such as the simple difference-in-differences in a non-staggered design, or more advanced estimators for staggered treatment adoption settings (e.g., `Callaway and Sant'Anna (2020) <https://psantanna.com/files/Callaway_SantAnna_2020.pdf>`_ or `Sun and Abraham (2020) <https://arxiv.org/pdf/1804.05785>`_).

The true parameter vector, :math:`\boldsymbol{\beta}`, is assumed to have the following causal decomposition

.. math::

   \boldsymbol{\beta} = \begin{pmatrix} \boldsymbol{\tau}_{pre} \\ \boldsymbol{\tau}_{post} \end{pmatrix} + \begin{pmatrix} \boldsymbol{\delta}_{pre} \\ \boldsymbol{\delta}_{post} \end{pmatrix}.

The first term, :math:`\boldsymbol{\tau}`, represents the treatment effects of interest. A key assumption is that there is no anticipation of the treatment, so the pre-treatment causal effects are zero, :math:`\boldsymbol{\tau}_{pre} = \mathbf{0}`. The second term, :math:`\boldsymbol{\delta}`, represents the difference in trends between the treated and comparison groups that would have occurred in the absence of treatment. For instance, in a canonical DiD setup, :math:`\boldsymbol{\tau}_{post}` is the vector of period-specific average treatment effects on the treated (ATTs), and :math:`\boldsymbol{\delta}` is the difference in trends of untreated potential outcomes.

The conventional parallel trends assumption imposes the strong restriction that :math:`\boldsymbol{\delta}_{post} = \mathbf{0}`. This methods developed here relax that assumption.

Partial Identification and the Restriction Set
--------------------------------------------

The goal is to conduct inference on a scalar parameter of interest, typically a linear combination of post-treatment effects, :math:`\theta = \mathbf{\ell}' \boldsymbol{\tau}_{post}`. Without assuming :math:`\boldsymbol{\delta}_{post} = \mathbf{0}`, the parameter :math:`\theta` is only partially identified. Identification is achieved by assuming that the true trend violation, :math:`\boldsymbol{\delta}`, lies within a researcher-specified set :math:`\Delta`. The identified set for :math:`\theta` is the set of all values consistent with the data and the restriction :math:`\boldsymbol{\delta} \in \Delta`. This set is given by

.. math::

   \mathcal{S}(\boldsymbol{\beta}, \Delta) := \bigg\{\theta: \exists \boldsymbol{\delta} \in \Delta, \boldsymbol{\tau}_{post} \in \mathbb{R}^{\bar{T}} \text{ s.t. } \mathbf{\ell}' \boldsymbol{\tau}_{post} = \theta, \boldsymbol{\beta} = \boldsymbol{\delta} + \begin{pmatrix} \mathbf{0} \\ \boldsymbol{\tau}_{post} \end{pmatrix} \bigg\}.

When :math:`\Delta` is closed and convex, this identified set is a simple interval, :math:`[\theta^{lb}, \theta^{ub}]`.

The choice of :math:`\Delta` is critical and must be guided by economic context. The paper proposes several classes of restrictions that can be specified as polyhedra (sets defined by linear inequalities):

- **Relative Magnitudes (RM)**: This formalizes the idea that post-treatment violations are not substantially larger than pre-treatment violations. The set :math:`\Delta^{RM}(\bar{M})` bounds the change in the trend violation between any two consecutive post-treatment periods by a factor :math:`\bar{M}` of the maximum change in the pre-treatment period. This set is given by

  .. math::

     \Delta^{RM}(\bar{M}) = \bigg\{\boldsymbol{\delta}: |\delta_{t+1} - \delta_t| \le \bar{M} \cdot \max_{s<0} |\delta_{s+1} - \delta_s|, \forall t \ge 0 \bigg\}.

  A natural benchmark is :math:`\bar{M}=1`. If you believe the confounding factors causing deviations from parallel trends after treatment are similar in size to those before treatment, the relative magnitude bounds approach is suitable.

- **Smoothness Restrictions (SD)**: This formalizes the idea that the differential trend evolves smoothly over time, relaxing the assumption of a perfectly linear trend. The set :math:`\Delta^{SD}(M)` bounds the discrete second derivative of the trend by a constant :math:`M`. This set is given by

  .. math::

     \Delta^{SD}(M) = \bigg\{\boldsymbol{\delta}: |(\delta_{t+1} - \delta_t) - (\delta_t - \delta_{t-1})| \le M, \forall t \bigg\}.

  This is a relaxation of controlling for a linear time trend (which corresponds to :math:`M=0`). When you expect the differences in trends between the treatment and control group to evolve smoothly over time without sharp changes, the smoothness restriction approach is appropriate.

- **Combined and Shape Restrictions**: The framework is flexible and can accommodate other assumptions, such as combining the RM and SD restrictions (**SDRM**), or imposing **Sign and Monotonicity Restrictions** based on knowledge of the economic setting (e.g., if the bias is known to be positive or increasing). If you believe that trends change smoothly over time, but are unsure about the exact level of smoothness, you can combine the smoothness and relative magnitude bounds approaches to set reasonable limits on trend changes.

Inference Methods
-----------------

The module provides two primary methods for constructing uniformly valid confidence sets for :math:`\theta` under the chosen restriction :math:`\boldsymbol{\delta} \in \Delta`.

**Moment Inequality-Based Inference**

This general approach casts the inference problem as a test of a system of moment inequalities with linear nuisance parameters. For a polyhedral restriction :math:`\Delta = \{\boldsymbol{\delta}: A\boldsymbol{\delta} \le d\}`, the null hypothesis :math:`H_0: \theta = \bar{\theta}` is equivalent to the existence of a nuisance parameter vector :math:`\tilde{\boldsymbol{\tau}}` such that a set of linear moment inequalities holds. The methods developed here implement the conditional and hybrid ARP methods from `Andrews, Roth, and Pakes (2021) <https://arxiv.org/pdf/1909.10062>`_ to test this hypothesis.

The ARP framework considers linear conditional moment inequalities of the form

.. math::

   E_{P_{D|Z}}[Y_i(\beta) - X_i(\beta)\delta | Z_i] \le 0 \quad \text{ a.s.},

where :math:`\beta` is the parameter of interest, :math:`\delta \in \mathbb{R}^p` is a nuisance parameter, :math:`Z_i` is a subvector of the data :math:`D_i`, :math:`Y_i(\beta) = y(D_i, \beta) \in \mathbb{R}^k` and :math:`X_i(\beta) = x(Z_i, \beta) \in \mathbb{R}^{k \times p}` for known functions :math:`y(\cdot, \cdot)` and :math:`x(\cdot, \cdot)`. The key properties of this structure are that the nuisance parameter :math:`\delta` enters linearly and the Jacobian of the moments with respect to :math:`\delta`, namely :math:`-X_i(\beta)`, is non-random conditional on :math:`Z_i`. This implies that the variance of the moments conditional on :math:`Z_i` does not depend on :math:`\delta`.

To construct tests, ARP exploits a conditional asymptotic approximation. Under mild conditions, the scaled sample moments satisfy

.. math::

   Y_{n,0} - X_{n,0}\delta | \{Z_i\} \approx^d N \big(\mu_{n,0} - X_{n,0}\delta, \Sigma_0 \big)

where :math:`Y_{n,0} = \frac{1}{\sqrt{n}}\sum_i Y_i(\beta_0)`, :math:`X_{n,0} = \frac{1}{\sqrt{n}}\sum_i X_i(\beta_0)`, :math:`\mu_{n,0} = \frac{1}{\sqrt{n}}\sum_i E_{P_{D|Z}}[Y_i(\beta_0)|Z_i]`, and :math:`\Sigma_0 = E_P[\text{Var}_{P_{D|Z}}(Y_i(\beta_0)|Z_i)]`. Crucially, the variance :math:`\Sigma_0` does not depend on :math:`\delta`, which substantially simplifies inference.

The test statistic is the profiled max statistic

.. math::

   \hat{\eta}_{n,0} = \min_\delta \max_j \big\{e_j'(Y_{n,0} - X_{n,0}\delta)/\sigma_{0,j}\big\},

where :math:`e_j` is the :math:`j`-th standard basis vector and :math:`\sigma_{0,j} = \sqrt{e_j'\Sigma_0 e_j}`. This can be equivalently represented as the solution to the linear program

.. math::

   \hat{\eta}_{n,0} = \min_{\eta,\delta} \eta \quad \text{s.t} \quad Y_{n,0} - X_{n,0}\delta \le \eta \cdot \sigma_0,

where :math:`\sigma_0 = (\sigma_{0,1}, \ldots, \sigma_{0,k})'`. The dual representation is

.. math::

   \hat{\eta}_{n,0} = \max_\gamma \gamma' Y_{n,0} \quad \text{s.t} \quad \gamma \ge 0, \quad \gamma' X_{n,0} = 0, \quad \gamma' \sigma_0 = 1.

The maximum is obtained at one of the finite set of vertices :math:`V(X_{n,0}, \sigma_0)` of the feasible set.

ARP Testing Approaches
----------------------

ARP develops three testing approaches based on this structure. Each approach offers different trade-offs in terms of power and robustness.

Least Favorable (LF) Test
~~~~~~~~~~~~~~~~~~~~~~~~~

The least favorable (LF) test uses the critical value :math:`c_{\alpha,LF}` defined as the :math:`1-\alpha` quantile of

.. math::

   c_{\alpha,LF} = \max_{\gamma \in V(X_{n,0}, \sigma_0)} \gamma' \xi, \quad \text{for} \quad \xi \sim N(0, \Sigma_0).

This test has exact asymptotic size when all moments bind simultaneously in population, but can be conservative when some moments are far from binding.

Conditional Test
~~~~~~~~~~~~~~~~

The conditional test addresses the conservativeness of the LF test by conditioning on the identity of the optimal vertex

.. math::

   \hat{\gamma} = \text{argmax}_{\gamma \in V(X_{n,0}, \sigma_0)} \gamma' Y_{n,0}.

Under the null hypothesis, the test statistic follows a truncated normal distribution

.. math::

   \hat{\eta}_{n,0} | \{\hat{\gamma} = \gamma, S_{n,0,\gamma} = s\} \sim TN \big(\gamma' \mu_{n,0}, \gamma' \Sigma_0 \gamma, [\mathcal{V}_{n,0}^{lo}, \mathcal{V}_{n,0}^{up}] \big).

where :math:`S_{n,0,\gamma} = (I - \frac{\Sigma_0 \gamma \gamma'}{\gamma' \Sigma_0 \gamma})Y_{n,0}` and the truncation points are

.. math::

   \mathcal{V}_{n,0}^{lo} = \max_{\substack{\tilde{\gamma} \in V(X_{n,0}, \sigma_0): \\ \gamma' \Sigma_0 \gamma > \gamma' \Sigma_0 \tilde{\gamma}}} \frac{\gamma' \Sigma_0 \gamma \cdot \tilde{\gamma}' s}{\gamma' \Sigma_0 \gamma - \gamma' \Sigma_0 \tilde{\gamma}}, \quad
   \mathcal{V}_{n,0}^{up} = \min_{\substack{\tilde{\gamma} \in V(X_{n,0}, \sigma_0): \\ \gamma' \Sigma_0 \gamma < \gamma' \Sigma_0 \tilde{\gamma}}} \frac{\gamma' \Sigma_0 \gamma \cdot \tilde{\gamma}' s}{\gamma' \Sigma_0 \gamma - \gamma' \Sigma_0 \tilde{\gamma}}.

This test has the property of being insensitive to slack moments in the strong sense that as a subset of moments becomes arbitrarily slack, the conditional test converges to the test that drops these moments ex-ante.

Hybrid Test
~~~~~~~~~~~

The hybrid test combines the strengths of both approaches. For some :math:`0 < \kappa < \alpha`, it first performs a size :math:`\kappa` LF test. If this rejects, the hybrid test rejects. Otherwise, it performs a size :math:`\frac{\alpha-\kappa}{1-\kappa}` test that conditions on both :math:`\hat{\gamma} = \gamma` and the event that the LF test did not reject. The critical value uses a modified upper truncation point

.. math::

   \mathcal{V}_{n,0}^{up,H} = \min \big\{\mathcal{V}_{n,0}^{up}, c_{\kappa,LF} \big\}.

The recommended approach in ARP is to set :math:`\kappa = \alpha/10`.

This approach is computationally tractable even with many post-treatment periods and has strong theoretical guarantees. The resulting confidence sets are uniformly valid, consistent (having power approaching 1 against fixed alternatives outside the identified set), and have optimal local asymptotic power under a linear independence constraint qualification.

Fixed-Length Confidence Intervals
---------------------------------

This method constructs confidence intervals of the form :math:`(a + \mathbf{v}' \hat{\boldsymbol{\beta}}) \pm \chi`, where the half-length :math:`\chi` is fixed. The affine estimator :math:`a + \mathbf{v}' \hat{\boldsymbol{\beta}}` and the length :math:`\chi` are chosen to minimize the interval's length while maintaining valid coverage.

For certain choices of :math:`\Delta` that are convex and centro-symmetric, :math:`\Delta^{SD}(M)`, for instance, FLCIs have attractive finite-sample optimality properties and can offer substantial power gains. However, for many other plausible restrictions (e.g., relative magnitudes or those involving sign/monotonicity constraints), FLCIs can be inconsistent, meaning they may fail to shrink to the true identified set even with infinite data.

.. note::
   The recommended practice is to use the hybrid moment inequality approach for general forms of :math:`\Delta`, as it is broadly valid and has strong asymptotic properties. The FLCI approach should be preferred only in special cases (like for :math:`\Delta^{SD}`) where its conditions for optimality and consistency are met.
