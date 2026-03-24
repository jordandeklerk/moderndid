.. _background-didhonest:

Honest DiD Sensitivity Analysis
===============================

The ``didhonest`` module provides tools for conducting sensitivity analysis in
difference-in-differences (DiD) models based on the work of
`Rambachan and Roth (2023) <https://asheshrambachan.github.io/assets/files/hpt-draft.pdf>`_.
Rather than testing whether parallel trends holds, this approach asks how large violations
would need to be before the conclusions change.

.. warning::

   Traditional pre-trends tests have two problems beyond low power. First, conditioning on
   not rejecting a pre-test biases subsequent point estimates and confidence intervals. Under
   correct parallel trends, post-treatment estimates are biased toward zero and confidence
   intervals are too short. Under violated parallel trends, post-treatment estimates can be
   biased *away* from zero, inflating apparent significance. These distortions arise from
   selecting on a noisy signal and cannot be fixed by adjusting the pre-test significance
   level.

Model Setup and Causal Decomposition
------------------------------------

What we estimate in a typical event-study regression reflects two components. The first is
the causal treatment effect. The second is the difference in trends between treated and
comparison groups that would have existed without treatment. This decomposition makes explicit what parallel trends
buys you and what goes wrong when it fails.

The functionality of this module is based on a vector of "event-study coefficients"

.. math::

   \hat{\boldsymbol{\beta}} \in \mathbb{R}^{\underline{T}+\bar{T}}, \quad \hat{\boldsymbol{\beta}} = (\hat{\boldsymbol{\beta}}_{pre}', \hat{\boldsymbol{\beta}}_{post}')'.

These coefficients can be partitioned into coefficients for pre-treatment and post-treatment periods. They can be obtained
from various DiD estimators, such as the simple difference-in-differences in a non-staggered design, or more advanced
estimators for staggered treatment adoption settings (e.g., `Callaway and Sant'Anna (2020) <https://psantanna.com/files/
Callaway_SantAnna_2020.pdf>`_ or `Sun and Abraham (2020) <https://arxiv.org/pdf/1804.05785>`_).

The true parameter vector, :math:`\boldsymbol{\beta}`, is assumed to have the following causal decomposition

.. math::

   \boldsymbol{\beta} = \begin{pmatrix} \boldsymbol{\tau}_{pre} \\ \boldsymbol{\tau}_{post} \end{pmatrix} + \begin{pmatrix} \boldsymbol{\delta}_{pre} \\ \boldsymbol{\delta}_{post} \end{pmatrix}.

The first term, :math:`\boldsymbol{\tau}`, represents the treatment effects of interest. A key assumption is that there is no
anticipation of the treatment, so the pre-treatment causal effects are zero, :math:`\boldsymbol{\tau}_{pre} = \mathbf{0}`. The
second term, :math:`\boldsymbol{\delta}`, represents the difference in trends between the treated and comparison groups that
would have occurred in the absence of treatment. For instance, in a canonical DiD setup, :math:`\boldsymbol{\tau}_{post}` is
the vector of period-specific average treatment effects on the treated (ATTs), and :math:`\boldsymbol{\delta}` is the
difference in trends of untreated potential outcomes.

The conventional parallel trends assumption imposes the strong restriction that :math:`\boldsymbol{\delta}_{post} =
\mathbf{0}`. The methods developed here relax that assumption.

Partial Identification and the Restriction Set
----------------------------------------------

Relaxing the assumption :math:`\boldsymbol{\delta}_{post} = \mathbf{0}` means the treatment
effect is no longer point identified. Instead, for any restriction on how large
:math:`\boldsymbol{\delta}` can be, there is a set of treatment effect values consistent with
the data and the restriction. This *identified set* is the central object.

The goal is inference on a scalar parameter,
:math:`\theta = \mathbf{\ell}' \boldsymbol{\tau}_{post}`. Without assuming :math:`\boldsymbol{\delta}_{post} = \mathbf{0}`,
the parameter :math:`\theta` is only partially identified. Identification is achieved by assuming that the true trend
violation, :math:`\boldsymbol{\delta}`, lies within a researcher-specified set :math:`\Delta`. The identified set for
:math:`\theta` is the set of all values consistent with the data and the restriction :math:`\boldsymbol{\delta} \in \Delta`.
This set is given by

.. math::

   \mathcal{S}(\boldsymbol{\beta}, \Delta) := \bigg\{\theta: \exists \boldsymbol{\delta} \in \Delta, \boldsymbol{\tau}_{post} \in \mathbb{R}^{\bar{T}} \text{ s.t. }
        \mathbf{\ell}' \boldsymbol{\tau}_{post} = \theta,
        \boldsymbol{\beta} = \boldsymbol{\delta} + \begin{pmatrix} \mathbf{0} \\ \boldsymbol{\tau}_{post} \end{pmatrix} \bigg\}.

When :math:`\Delta` is closed and convex, this identified set is a simple interval,

.. math::

   \mathcal{S}(\boldsymbol{\beta}, \Delta) = [\theta^{lb}(\boldsymbol{\beta}, \Delta), \theta^{ub}(\boldsymbol{\beta}, \Delta)],

where the bounds are given by

.. math::

   \theta^{lb}(\boldsymbol{\beta}, \Delta) &= \mathbf{\ell}' \boldsymbol{\beta}_{post} - \max_{\boldsymbol{\delta}} \mathbf{\ell}' \boldsymbol{\delta}_{post} \quad \text{s.t.}
        \quad \boldsymbol{\delta} \in \Delta, \, \boldsymbol{\delta}_{pre} = \boldsymbol{\beta}_{pre} \\
   \theta^{ub}(\boldsymbol{\beta}, \Delta) &= \mathbf{\ell}' \boldsymbol{\beta}_{post} - \min_{\boldsymbol{\delta}} \mathbf{\ell}' \boldsymbol{\delta}_{post} \quad \text{s.t.}
        \quad \boldsymbol{\delta} \in \Delta, \, \boldsymbol{\delta}_{pre} = \boldsymbol{\beta}_{pre}.

This characterization follows from observing that the identified set can be equivalently written as

.. math::

   \mathcal{S}(\boldsymbol{\beta}, \Delta) = \{\theta: \exists \boldsymbol{\delta} \in \Delta \text{ s.t. }
        \boldsymbol{\delta}_{pre} = \boldsymbol{\beta}_{pre},
        \theta = \mathbf{\ell}' \boldsymbol{\beta}_{post} - \mathbf{\ell}' \boldsymbol{\delta}_{post}\}.

Restriction Classes
-------------------

The restriction set :math:`\Delta` encodes what the researcher believes about the nature of
possible confounding, and the choice should be guided by economic reasoning about the threats
to identification.

.. tip::

   If the main concerns are differential economic shocks that hit treated and control groups
   at different times, bounding post-treatment violations relative to pre-treatment violations
   (relative magnitudes) is natural. If the concerns are instead about smooth secular trends,
   bounding the curvature of the trend (smoothness) is more appropriate.

All restriction classes below can be written as polyhedra or finite unions of polyhedra.

.. admonition:: Relative Magnitudes Restriction (RM)

   This formalizes the idea that post-treatment violations are not substantially larger than pre-treatment violations. The set
   :math:`\Delta^{RM}(\bar{M})` bounds the change in the trend violation between any two consecutive post-treatment periods by
   a factor :math:`\bar{M}` of the maximum change in the pre-treatment period,

   .. math::

      \Delta^{RM}(\bar{M}) = \bigg\{\boldsymbol{\delta}: |\delta_{t+1} - \delta_t| \le \bar{M} \cdot \max_{s<0} |\delta_{s+1} - \delta_s|, \forall t \ge 0 \bigg\}.

A natural benchmark is :math:`\bar{M}=1`. If you believe the confounding factors causing deviations from parallel trends after
treatment are similar in size to those before treatment, the relative magnitude bounds approach is suitable.

.. admonition:: Smoothness Restriction (SD)

   This formalizes the idea that the differential trend evolves smoothly over time, relaxing the assumption of a perfectly
   linear trend. The set :math:`\Delta^{SD}(M)` bounds the discrete second derivative of the trend by a constant :math:`M`,

   .. math::

      \Delta^{SD}(M) = \bigg\{\boldsymbol{\delta}: |(\delta_{t+1} - \delta_t) - (\delta_t - \delta_{t-1})| \le M, \forall t \bigg\}.

Setting :math:`M=0` recovers the linear time trend assumption. Positive :math:`M` relaxes this
by allowing the slope of the trend difference to change by up to :math:`M` between consecutive
periods.

.. admonition:: Smoothness and Relative Magnitudes (SDRM)

   This combines the smoothness and relative magnitudes restrictions, bounding the maximum deviation from a linear trend in
   the post-treatment period by :math:`\bar{M}` times the equivalent maximum in the pre-treatment period,

   .. math::

      \Delta^{SDRM}(\bar{M}) = \bigg\{\boldsymbol{\delta}: |(\delta_{t+1} - \delta_t) - (\delta_t - \delta_{t-1})| \le \bar{M}
         \cdot \max_{s<0} |(\delta_{s+1} - \delta_s) - (\delta_s - \delta_{s-1})|,
         \forall t \ge 0 \bigg\}.

This allows the magnitude of possible non-linearity to explicitly depend on the observed pre-trends.

.. admonition:: Sign and Monotonicity Restrictions

   Additional restrictions can be imposed based on economic knowledge about the direction of confounding. For example, if the
   researcher believes the bias :math:`\boldsymbol{\delta}_{post}` is non-negative or monotonically increasing, these
   constraints can be added to :math:`\Delta` to tighten the identified set.

Combining smoothness with relative magnitudes is useful when the researcher expects smooth
trends but wants the bound on curvature to scale with the observed pre-treatment
non-linearity.

Polyhedral Representation
~~~~~~~~~~~~~~~~~~~~~~~~~

To accommodate a broad range of restrictions, the framework considers restriction sets that can be written as polyhedra or
finite unions of polyhedra.

A restriction class :math:`\Delta` is *polyhedral* if it takes the form

.. math::

   \Delta = \{\boldsymbol{\delta}: A\boldsymbol{\delta} \le d\}

for some known matrix :math:`A` and vector :math:`d`. All of the restriction classes described above can be written either as
polyhedral restrictions or finite unions of such restrictions. For instance, :math:`\Delta^{SD}(M)` can be written directly
as a polyhedron, while :math:`\Delta^{RM}(\bar{M})` can be written as the finite union of polyhedra, where each polyhedron
corresponds to a different location for the maximum pre-treatment violation.

When :math:`\Delta` is the finite union of sets, the identified set is the union of the identified sets for its subcomponents

.. math::

   \Delta = \bigcup_{k=1}^{K} \Delta_k \quad \Rightarrow \quad \mathcal{S}(\boldsymbol{\beta}, \Delta) = \bigcup_{k=1}^{K} \mathcal{S}(\boldsymbol{\beta}, \Delta_k).

This allows confidence sets to be constructed by taking the union of confidence sets for each component.

Inference Methods
-----------------

Standard confidence intervals assume point identification. With partial identification,
coverage must hold for every value in the identified set, not just a single point. The
module provides two methods for constructing uniformly valid confidence sets for
:math:`\theta` under the restriction :math:`\boldsymbol{\delta} \in \Delta`.

1. **Moment inequality-based inference** uses conditional and hybrid tests from the Andrews,
   Roth, and Pakes (ARP) framework. Uniformly consistent and can achieve optimal local
   asymptotic power under regularity conditions.

2. **Fixed-length confidence intervals (FLCIs)** have a pre-specified length that accounts
   for worst-case bias. Simpler but can be inconsistent for some restriction classes.

Moment Inequality-Based Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This general approach casts the inference problem as a test of a system of moment inequalities with linear nuisance
parameters. For a polyhedral restriction :math:`\Delta = \{\boldsymbol{\delta}: A\boldsymbol{\delta} \le d\}`, the null
hypothesis :math:`H_0: \theta = \bar{\theta}` is equivalent to the existence of a nuisance parameter vector :math:`
\tilde{\boldsymbol{\tau}}` such that a set of linear moment inequalities holds. The methods developed here implement the
conditional and hybrid ARP methods from `Andrews, Roth, and Pakes (2021) <https://arxiv.org/pdf/1909.10062>`_ to test this
hypothesis.

The ARP framework considers linear conditional moment inequalities of the form

.. math::

   E_{P_{D|Z}}[Y_i(\beta) - X_i(\beta)\delta | Z_i] \le 0 \quad \text{ a.s.},

where :math:`\beta` is the parameter of interest, :math:`\delta \in \mathbb{R}^p` is a nuisance parameter, :math:`Z_i` is a
subvector of the data :math:`D_i`, :math:`Y_i(\beta) = y(D_i, \beta) \in \mathbb{R}^k` and :math:`X_i(\beta) = x(Z_i, \beta)
\in \mathbb{R}^{k \times p}` for known functions :math:`y(\cdot, \cdot)` and :math:`x(\cdot, \cdot)`. The key properties of
this structure are that the nuisance parameter :math:`\delta` enters linearly and the Jacobian of the moments with respect to
:math:`\delta`, namely :math:`-X_i(\beta)`, is non-random conditional on :math:`Z_i`. This implies that the variance of the
moments conditional on :math:`Z_i` does not depend on :math:`\delta`.

To construct tests, ARP exploits a conditional asymptotic approximation. Under mild conditions, the scaled sample moments
satisfy

.. math::

   Y_{n,0} - X_{n,0}\delta | \{Z_i\} \approx^d N \big(\mu_{n,0} - X_{n,0}\delta, \Sigma_0 \big)

where

.. math::

   \begin{aligned}
   Y_{n,0} &= \frac{1}{\sqrt{n}}\sum_i Y_i(\beta_0), \quad & X_{n,0} &= \frac{1}{\sqrt{n}}\sum_i X_i(\beta_0), \\
   \mu_{n,0} &= \frac{1}{\sqrt{n}}\sum_i E_{P_{D|Z}}[Y_i(\beta_0)|Z_i], \quad & \Sigma_0 &= E_P[\text{Var}_{P_{D|Z}}(Y_i(\beta_0)|Z_i)].
   \end{aligned}

Because :math:`\Sigma_0` does not depend on :math:`\delta`, inference simplifies considerably.

The test statistic is the profiled max statistic

.. math::

   \hat{\eta}_{n,0} = \min_\delta \max_j \big\{e_j'(Y_{n,0} - X_{n,0}\delta)/\sigma_{0,j}\big\},

where :math:`e_j` is the :math:`j`-th standard basis vector and :math:`\sigma_{0,j} = \sqrt{e_j'\Sigma_0 e_j}`. This can be
equivalently represented as the solution to the linear program

.. math::

   \hat{\eta}_{n,0} = \min_{\eta,\delta} \eta \quad \text{s.t} \quad Y_{n,0} - X_{n,0}\delta \le \eta \cdot \sigma_0,

where :math:`\sigma_0 = (\sigma_{0,1}, \ldots, \sigma_{0,k})'`. The dual representation is

.. math::

   \hat{\eta}_{n,0} = \max_\gamma \gamma' Y_{n,0} \quad \text{s.t} \quad \gamma \ge 0, \quad \gamma' X_{n,0} = 0, \quad \gamma' \sigma_0 = 1.

The maximum is obtained at one of the finite set of vertices :math:`V(X_{n,0}, \sigma_0)` of the feasible set.

ARP Testing Approaches
^^^^^^^^^^^^^^^^^^^^^^

ARP develops three testing approaches based on this structure. Each approach offers different trade-offs in terms of power
and robustness.

**Least Favorable (LF) Test**

The least favorable (LF) test uses the critical value :math:`c_{\alpha,LF}` defined as the :math:`1-\alpha` quantile of

.. math::

   c_{\alpha,LF} = \max_{\gamma \in V(X_{n,0}, \sigma_0)} \gamma' \xi, \quad \text{for} \quad \xi \sim N(0, \Sigma_0).

This test has exact asymptotic size when all moments bind simultaneously in population, but can be conservative when some
moments are far from binding.

**Conditional Test**

The conditional test addresses the conservativeness of the LF test by conditioning on the identity of the optimal vertex

.. math::

   \hat{\gamma} = \text{argmax}_{\gamma \in V(X_{n,0}, \sigma_0)} \gamma' Y_{n,0}.

Under the null hypothesis, the test statistic follows a truncated normal distribution

.. math::

   \hat{\eta}_{n,0} | \{\hat{\gamma} = \gamma, S_{n,0,\gamma} = s\} \sim TN \big(\gamma' \mu_{n,0}, \gamma' \Sigma_0 \gamma, [\mathcal{V}_{n,0}^{lo}, \mathcal{V}_{n,0}^{up}] \big),

where

.. math::

   S_{n,0,\gamma} = \left(I - \frac{\Sigma_0 \gamma \gamma'}{\gamma' \Sigma_0 \gamma}\right)Y_{n,0}

and the truncation points are

.. math::

   \mathcal{V}_{n,0}^{lo} = \max_{\substack{\tilde{\gamma} \in V(X_{n,0}, \sigma_0): \\ \gamma' \Sigma_0 \gamma > \gamma' \Sigma_0 \tilde{\gamma}}} \frac{\gamma' \Sigma_0 \gamma \cdot \tilde{\gamma}' s}{\gamma' \Sigma_0 \gamma - \gamma' \Sigma_0 \tilde{\gamma}}, \quad
   \mathcal{V}_{n,0}^{up} = \min_{\substack{\tilde{\gamma} \in V(X_{n,0}, \sigma_0): \\ \gamma' \Sigma_0 \gamma < \gamma' \Sigma_0 \tilde{\gamma}}} \frac{\gamma' \Sigma_0 \gamma \cdot \tilde{\gamma}' s}{\gamma' \Sigma_0 \gamma - \gamma' \Sigma_0 \tilde{\gamma}}.

This test has the property of being insensitive to slack moments in the strong sense that as a subset of moments becomes
arbitrarily slack, the conditional test converges to the test that drops these moments ex-ante.

**Hybrid Test**

The hybrid test combines the strengths of both approaches. For some :math:`0 < \kappa < \alpha`, it first performs a size
:math:`\kappa` LF test. If this rejects, the hybrid test rejects. Otherwise, it performs a size :math:`\frac{\alpha-\kappa}
{1-\kappa}` test that conditions on both :math:`\hat{\gamma} = \gamma` and the event that the LF test did not reject. The
critical value uses a modified upper truncation point

.. math::

   \mathcal{V}_{n,0}^{up,H} = \min \big\{\mathcal{V}_{n,0}^{up}, c_{\kappa,LF} \big\}.

The recommended approach in ARP is to set :math:`\kappa = \alpha/10`.

This approach is computationally tractable even with many post-treatment periods.

**Uniform Consistency**

The conditional and hybrid tests are uniformly asymptotically consistent, meaning that power against fixed alternatives
outside the identified set converges uniformly to 1. Formally, for any :math:`x > 0`,

.. math::

   \lim_{n \to \infty} \inf_{P \in \mathcal{P}} \mathbb{P}_{P}\big(\theta_P^{ub} + x \notin \mathcal{C}_{\alpha,n}^{Hyb}\big) = 1,

where :math:`\theta_P^{ub} = \sup \mathcal{S}(\boldsymbol{\beta}_P, \Delta)` is the upper bound of the identified set. An
analogous result holds for the lower bound. The confidence sets shrink to the identified set as the sample grows.

**Optimal Local Asymptotic Power**

Under an additional regularity condition known as the Linear Independence Constraint Qualification (LICQ), the conditional
test achieves optimal local asymptotic power. LICQ ensures that the bounds of the identified set are differentiable with
respect to the moment means, avoiding challenges related to inference on non-differentiable parameters.

Recall that the upper bound of the identified set is given by the optimization problem

.. math::

   \theta^{ub}(\boldsymbol{\beta}, \Delta) = \mathbf{\ell}' \boldsymbol{\beta}_{post} - \min_{\boldsymbol{\delta}} \mathbf{\ell}' \boldsymbol{\delta}_{post}
   \quad \text{s.t.} \quad \boldsymbol{\delta} \in \Delta, \, \boldsymbol{\delta}_{pre} = \boldsymbol{\beta}_{pre}.

Let :math:`B^*` denote the set of binding constraints at the optimum, and let :math:`A_{(B^*, post)}` denote the submatrix of
:math:`A` corresponding to these binding constraints and the post-treatment components. LICQ holds in direction :math:`
\mathbf{\ell}` if there exists a solution :math:`\boldsymbol{\tau}_{post}^*` such that the gradient of the binding
constraints, :math:`-A_{(B^*, post)}`, has full row rank. Under LICQ, the local asymptotic power of the conditional test
converges to the power envelope for tests that control size in the finite-sample normal model.

Fixed-Length Confidence Intervals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An alternative approach to constructing confidence sets is to use fixed-length confidence intervals (FLCIs). Unlike the
moment inequality approach which adapts its length based on the data, FLCIs have a pre-specified length that accounts for
the worst-case bias from potential violations of parallel trends. The appeal of this approach is its simplicity and, under
certain conditions, near-optimal expected length.

This method constructs confidence intervals of the form

.. math::

   (a + \mathbf{v}' \hat{\boldsymbol{\beta}}) \pm \chi,

where the half-length :math:`\chi` is fixed. The affine estimator :math:`a + \mathbf{v}' \hat{\boldsymbol{\beta}}` and the
length :math:`\chi` are chosen to minimize the interval's length while maintaining valid coverage. The smallest valid half-
length is the :math:`1-\alpha` quantile of :math:`|\mathcal{N}(\bar{b}, \mathbf{v}'\Sigma_n \mathbf{v})|`, where :math:`\bar{b}`
is the affine estimator's worst-case bias,

.. math::

   \bar{b} = \sup_{\boldsymbol{\delta} \in \Delta} |a + \mathbf{v}'(\boldsymbol{\delta} + \boldsymbol{\tau}) - \theta| =
   \sup_{\boldsymbol{\delta} \in \Delta} |a + \mathbf{v}'\boldsymbol{\delta} - \mathbf{\ell}'\boldsymbol{\delta}_{post}|.

**Finite-Sample Near-Optimality**

For certain choices of :math:`\Delta`, the optimal FLCI has near-optimal expected length in the finite-sample normal model.
Two conditions are sufficient.

1. :math:`\Delta` is convex and centrosymmetric (i.e., :math:`\tilde{\boldsymbol{\delta}} \in \Delta` implies :math:`-
   \tilde{\boldsymbol{\delta}} \in \Delta`)
2. The true :math:`\boldsymbol{\delta} \in \Delta` is such that :math:`(\tilde{\boldsymbol{\delta}} - \boldsymbol{\delta})
   \in \Delta` for all :math:`\tilde{\boldsymbol{\delta}} \in \Delta`

The smoothness restriction :math:`\Delta^{SD}(M)` satisfies condition (1), but :math:`\Delta^{RM}(\bar{M})` satisfies
neither (it is non-convex and not centrosymmetric). When these conditions hold, the expected length of the shortest possible
confidence set that satisfies the coverage requirement is at most 28% shorter than the length of the optimal FLCI (when
:math:`\alpha = 0.05`).

**Inconsistency of FLCIs**

For many restriction classes of practical interest, FLCIs can be inconsistent. An FLCI is consistent if, as the sample size
grows, the probability that it contains any fixed point outside the identified set converges to zero. Formally, consistency
requires that for all :math:`\theta^{out}` outside the identified set,

.. math::

   \lim_{n \to \infty} \mathbb{P}_{\hat{\boldsymbol{\beta}}_n \sim \mathcal{N}(\boldsymbol{\delta} + \boldsymbol{\tau}, \Sigma_n)}\big(\theta^{out} \in \mathcal{C}_{\alpha,n}^{FLCI}\big) = 0.

A sufficient condition for consistency is that the length of the identified set, :math:`\lambda(\mathcal{S})`, is constant
for all :math:`\boldsymbol{\delta} \in \Delta`.

**Example.** For :math:`\Delta^{RM}(\bar{M})` with :math:`\bar{M} > 0`, all affine estimators have infinite worst-case bias,
since :math:`|\delta_1|` can be arbitrarily large when :math:`|\delta_{-1}|` is sufficiently large. Thus, the only valid
FLCI is the entire real line, which is clearly inconsistent.

For :math:`\Delta^{SD}(M)`, the identified set always has the same length (:math:`2M` in the
three-period case), so FLCIs are consistent. This is why FLCIs are well-suited for smoothness
restrictions but not for relative magnitudes. The conditional and hybrid confidence sets can
"adapt" their length based on :math:`\hat{\boldsymbol{\beta}}_{pre}`, which is what allows
them to remain consistent where FLCIs fail.

Sensitivity Analysis and Breakdown Values
------------------------------------------

Given a baseline restriction class (say :math:`\Delta^{RM}(\bar{M})` or
:math:`\Delta^{SD}(M)`), the recommended practice is to report confidence sets for a range of
values of the tuning parameter (:math:`\bar{M} \geq 0` or :math:`M \geq 0`). This produces a
sensitivity analysis plot showing how the conclusions change as the allowed violations grow.

A particularly useful summary is the *breakdown value*, the smallest :math:`\bar{M}` or
:math:`M` at which a hypothesis of interest (typically a null effect) can no longer be
rejected. If the breakdown value is large, the finding is robust to substantial violations
of parallel trends. If the breakdown value is small, the finding is fragile.

Interpreting the magnitude of the breakdown value requires domain knowledge. One approach is
to calibrate :math:`M` against a concrete confounder. For instance, if the outcome is adult
employment and the concern is confounding from differences in school quality, one can use
estimates of the employment effect of a one-standard-deviation change in teacher value-added
to translate :math:`M` into units of a specific confounder. This makes the sensitivity
analysis more informative than an abstract bound.

Conditional Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When pre-treatment covariates :math:`X` are available, the sensitivity analysis can be
sharpened. If violations of parallel trends and treatment effects both vary with :math:`X`,
then conducting the analysis conditional on :math:`X` produces tighter identified sets than
the unconditional version, because the conditional restrictions eliminate variation in
:math:`\boldsymbol{\delta}` that is explained by :math:`X`. The identified sets are then
averaged over the covariate distribution to recover marginal bounds.

Recommended Practice
~~~~~~~~~~~~~~~~~~~~

For general forms of :math:`\Delta`, the hybrid moment inequality approach should be
preferred because it is uniformly consistent and has strong power properties. FLCIs should be
used only for restriction classes like :math:`\Delta^{SD}(M)` where the consistency and
finite-sample near-optimality conditions are met. In either case, results should be reported
as sensitivity analysis plots over a range of :math:`\bar{M}` or :math:`M` values rather than
at a single point, and the breakdown value should be highlighted.

.. note::
   For the full theoretical details, including proofs and regularity conditions, refer to
   `Rambachan and Roth (2023) <https://asheshrambachan.github.io/assets/files/hpt-draft.pdf>`_.
