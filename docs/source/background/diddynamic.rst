.. _background-diddynamic:

Dynamic Covariate Balancing DiD
================================

The ``diddynamic`` module implements the dynamic covariate balancing (DCB) estimator of
`Viviano and Bradic (2026) <https://doi.org/10.1093/biomet/asag016>`_ for estimating treatment
effects in panel data where treatments change over time. The method accommodates settings
where treatment assignments depend on high-dimensional covariates, past outcomes, and past
treatments, and where outcomes and time-varying covariates may depend on the entire
trajectory of past treatments.

Many empirical questions involve treatment sequences rather than one-time interventions.
Countries transition to and from democracy, patients start and stop medications, workers
move in and out of training programmes. In each case the causal quantity of interest depends
on the *sequence* of treatments. For example, researchers studying the effect of democracy
on GDP growth (`Acemoglu et al., 2019 <https://doi.org/10.1086/700936>`_) want to compare
countries that were democratic for five consecutive years against those that were not,
averaging over earlier treatment assignments. Standard DiD methods designed for one-time
adoption cannot address this question.

Why Standard Approaches Are Problematic
---------------------------------------

Three common strategies for handling dynamic treatments each have important limitations.

**Standard DiD and event studies** assume staggered adoption, where once a unit receives
treatment it remains treated. When units can switch treatment status in response to previous
outcomes, parallel trends is violated
(`Ghanem et al., 2022 <https://doi.org/10.3982/ECTA19402>`_;
`Marx et al., 2022 <https://doi.org/10.1016/j.jeconom.2021.12.014>`_). The standard TWFE
regression compounds this problem by collapsing different treatment histories into a single
coefficient, and negative weighting becomes more severe because the pool of "control" units
changes in each period.

**Standard local projections** (`Jordà, 2005 <https://doi.org/10.1257/0002828053828518>`_)
model observed outcomes as a linear function of current and lagged treatments and covariates.
Because the model is specified on *observed* rather than *potential* outcomes, the estimated
treatment effect absorbs the distribution of future treatment assignments and therefore
depends on the propensity score. With dynamic selection into treatment, this conflation biases
the estimated coefficients. In the democracy application of Viviano and Bradic (2026), local
projections substantially underestimate the long-run effect compared to DCB.

**Inverse probability weighting** (IPW) estimators reweight observations by the probability of
the observed treatment sequence. For :math:`T` periods, the IPW weight for unit :math:`i` is
the product of :math:`T` conditional probabilities,

.. math::

   w_i = \prod_{t=1}^T \frac{1}{P(D_{i,t} = d_t \mid H_{i,t})},

which can be highly unstable for even moderately long histories. If any single-period
propensity score is close to zero, the product explodes. In the Acemoglu et al. data, the
estimated probability of being democratic for just two consecutive years already drops below
0.1 for some countries, making IPW weights for five-year histories extremely variable. The
DCB estimator avoids this problem by replacing propensity score estimation with a quadratic
program that directly constructs balancing weights with minimum variance.

Setup and Notation
------------------

Consider a panel with :math:`n` i.i.d. units observed over :math:`T` periods. For unit
:math:`i` in period :math:`t`, let :math:`X_{i,t}` denote time-varying covariates,
:math:`D_{i,t} \in \{0,1\}` the binary treatment, and :math:`Y_{i,t}` the outcome. Define
the history vector

.. math::

   H_{i,t} = \bigl[D_{i,1}, \ldots, D_{i,t-1},\;
              X_{i,1}, \ldots, X_{i,t},\;
              Y_{i,1}, \ldots, Y_{i,t-1}\bigr] \in \mathbb{R}^{p_t},

which collects all information from period 1 through :math:`t`, excluding the treatment
assigned in the current period. The dimension :math:`p_t` grows with :math:`t` since each
additional period contributes covariates, treatments, and outcomes to the history. Since
covariates and outcomes at period :math:`t` may themselves depend on earlier treatments,
we also need the *potential* history under a given treatment path :math:`d_{1:(t-1)}`:

.. math::

   H_{i,t}(d_{1:(t-1)}) = \bigl[d_{1:(t-1)},\;
   X_{i,1:t}(d_{1:(t-1)}),\;
   Y_{i,1:(t-1)}(d_{1:(t-1)})\bigr],

where :math:`X_{i,t}(d_{1:(t-1)})` and :math:`Y_{i,t-1}(d_{1:(t-1)})` are the covariates
and outcomes that would have been observed under treatment path :math:`d_{1:(t-1)}`.

Estimands of Interest
~~~~~~~~~~~~~~~~~~~~~

The primary estimand is the average treatment effect of two treatment histories
:math:`d_{1:T}` and :math:`d'_{1:T}`,

.. math::

   \text{ATE}(d_{1:T}, d'_{1:T}) = \mu_T(d_{1:T}) - \mu_T(d'_{1:T}),
   \quad
   \mu_T(d_{1:T}) = \mathbb{E}\bigl[Y_T(d_{1:T})\bigr],

where :math:`Y_T(d_{1:T})` is the potential outcome at period :math:`T` under the full
treatment history :math:`d_{1:T}`. This estimand captures the total effect of exposure to
history :math:`d_{1:T}` compared to :math:`d'_{1:T}`, including both direct effects on the
outcome and indirect effects mediated through intermediate covariates and outcomes.

Several specific estimands are of practical interest.

- :math:`\text{ATE}((1,1),(0,0))` is the total effect of being treated for two consecutive
  periods compared to no treatment, the most common target in applications with short panels.
- :math:`\text{ATE}((1,0),(0,0))` is the direct effect of treatment in the first period only,
  allowing the treatment to "wear off" in the second period. The difference between this
  and :math:`\text{ATE}((1,1),(0,0))` reveals how much of the total effect comes from
  sustained versus one-time exposure.
- For long panels with :math:`T` periods, averaging over earlier treatment assignments
  produces estimands of the form

  .. math::

     \mathbb{E}\bigl[Y_T(D_{1:(T-h)}, d_{(T-h+1):T})\bigr]
     - \mathbb{E}\bigl[Y_T(D_{1:(T-h)}, d'_{(T-h+1):T})\bigr],

  which compare the last :math:`h` periods of treatment while averaging over past
  assignments. This is the estimand targeted by the ``histories_length`` option.

Identifying Assumptions
-----------------------

Identification of :math:`\mu_T(d_{1:T})` rests on three assumptions that generalize the
standard unconfoundedness framework to the dynamic setting. We present these first for two
periods to build intuition, then state the general versions.

Two-Period Case
~~~~~~~~~~~~~~~

In the two-period case, we observe :math:`(X_{i,1}, D_{i,1}, Y_{i,1}, X_{i,2}, D_{i,2},
Y_{i,2})` for each unit and define :math:`H_{i,2} = [D_{i,1}, X_{i,1}, X_{i,2}, Y_{i,1}]`.
The potential outcome :math:`Y_{i,2}(d_1, d_2)` is the outcome that would be observed if
unit :math:`i` received treatment :math:`d_1` in period 1 and :math:`d_2` in period 2.

.. admonition:: Assumption 1 (No Anticipation)

   For :math:`d_1 \in \{0,1\}`, let :math:`Y_{i,1}(d_1, 1) = Y_{i,1}(d_1, 0)` and
   :math:`X_{i,2}(d_1, 1) = X_{i,2}(d_1, 0)`.

   Intermediate potential outcomes and covariates depend only on past but not future
   treatment assignments. The treatment status at :math:`t = 2` has no contemporaneous effect
   on covariates.

This allows anticipatory effects governed by expectations (individuals may choose treatments
based on expected future utilities) but not on future treatment realisations.

.. admonition:: Assumption 2 (Sequential Ignorability)

   For all :math:`(d_1, d_2) \in \{0,1\}^2`,

   (A) :math:`Y_{i,2}(d_1, d_2) \perp D_{i,2} \mid D_{i,1}, X_{i,1}, X_{i,2}, Y_{i,1}`,

   (B) :math:`(Y_{i,2}(d_1, d_2), H_{i,2}(d_1)) \perp D_{i,1} \mid X_{i,1}`.

Part (A) states that second-period treatment is unconfounded given the full first-period
history. Part (B) states that first-period treatment is unconfounded given baseline
covariates. Together they require that there are no unobserved confounders *after* controlling
for observable characteristics, but they allow treatment decisions to depend on all observed
past information, including past outcomes and treatments.

.. admonition:: Assumption 3 (Potential Local Projections)

   For some :math:`\beta_{d_1,d_2}^{(1)} \in \mathbb{R}^{p_1}` and
   :math:`\beta_{d_1,d_2}^{(2)} \in \mathbb{R}^{p_2}`,

   .. math::

      \mathbb{E}[Y_{i,2}(d_1, d_2) \mid X_{i,1} = x_1] &= x_1\,\beta_{d_1,d_2}^{(1)}, \\
      \mathbb{E}[Y_{i,2}(d_1, d_2) \mid X_{i,1}, X_{i,2}, Y_{i,1}, D_{i,1} = d_1]
      &= [d_1, X_{i,1}, X_{i,2}, Y_{i,1}]\,\beta_{d_1,d_2}^{(2)}.

Following in spirit the local projection framework of
`Jordà (2005) <https://doi.org/10.1257/0002828053828518>`_, Assumption 3 imposes linearity on
expected *potential* outcomes rather than observed outcomes. This is a crucial distinction.
A model on realized outcomes would impose restrictions on the distribution of treatment
assignments (the propensity score), whereas a potential outcome model does not. The model
allows coefficients to be heterogeneous across treatment histories :math:`(d_1, d_2)` and the
dimensions :math:`p_1, p_2` can grow with :math:`n`, accommodating high-dimensional
covariates and their transformations.

In high dimensions, Assumption 3 can be relaxed to approximate linearity up to an order
:math:`o(n^{-1/2})`, covering settings where many covariates and their transformations
approximate the conditional mean function well enough for valid inference.

General Case
~~~~~~~~~~~~

The two-period assumptions extend naturally to :math:`T` periods. The key change is that
sequential ignorability now conditions on the full history :math:`H_{i,t}` at each period
rather than just baseline covariates or a single lag.

.. admonition:: Assumption 1' (No Anticipation, General)

   For any :math:`d_{1:T} \in \{0,1\}^T` and :math:`t \leq T`, the potential history
   :math:`H_{i,t}(d_{1:T})` is constant in :math:`d_{t:T}`.

.. admonition:: Assumption 2' (Sequential Ignorability, General)

   For all :math:`d_{1:T} \in \{0,1\}^T` and each :math:`t \leq T`,

   .. math::

      \bigl(Y_{i,T}(d_{1:T}),\, H_{i,t+1}(d_{1:(t+1)}),\, \ldots,\,
      H_{i,T-1}(d_{1:(T-1)})\bigr) \perp D_{i,t} \mid H_{i,t}.

.. admonition:: Assumption 3' (Potential Local Projections, General)

   For some :math:`\beta_{d_{1:T}}^{(t)} \in \mathbb{R}^{p_t}`,

   .. math::

      \mathbb{E}\bigl[Y_{i,T}(d_{1:T}) \mid D_{i,1:(t-1)} = d_{1:(t-1)},\,
      X_{i,1:t},\, Y_{i,1:(t-1)}\bigr]
      = H_{i,t}(d_{1:(t-1)})\,\beta_{d_{1:T}}^{(t)}.

.. admonition:: Assumption 4 (Overlap)

   :math:`P(D_{i,t} = d_t \mid H_{i,t}) \in (\delta, 1 - \delta)` for some
   :math:`\delta \in (0,1)` and each :math:`t \in \{1, \ldots, T\}`.

Strict overlap ensures that there exist weights satisfying the dynamic balance constraints
introduced below, with the true inverse probability weights being one such set of weights.

Identification
~~~~~~~~~~~~~~

The identifying assumptions connect the potential outcome model to observable quantities,
enabling recursive estimation. In the two-period case, the identification result takes a
particularly transparent form.

**Lemma (Identification, Two Periods).** Under Assumptions 1--3, for any
:math:`(d_1, d_2) \in \{0,1\}^2`,

.. math::

   \mathbb{E}[Y_{i,2} \mid H_{i,2},\, D_{i,2} = d_2,\, D_{i,1} = d_1]
   &= \mathbb{E}[Y_{i,2}(d_1, d_2) \mid H_{i,2},\, D_{i,1} = d_1]
   = H_{i,2}(d_1)\,\beta_{d_1,d_2}^{(2)}, \\[6pt]
   \mathbb{E}\bigl[\mathbb{E}[Y_{i,2} \mid H_{i,2},\, D_{i,2} = d_2,\, D_{i,1} = d_1]
   \;\big|\; X_{i,1},\, D_{i,1} = d_1\bigr]
   &= \mathbb{E}[Y_{i,2}(d_1, d_2) \mid X_{i,1}]
   = X_{i,1}\,\beta_{d_1,d_2}^{(1)}.

The first line uses sequential ignorability (Assumption 2A) to replace the observed outcome
:math:`Y_{i,2}` with the potential outcome :math:`Y_{i,2}(d_1, d_2)`, and then applies the
potential local projection (Assumption 3) to write the conditional expectation as linear in
:math:`H_{i,2}(d_1)`. The second line iterates backward, using Assumption 2B and the
first-period projection to express the iterated conditional expectation as linear in
:math:`X_{i,1}`.

This two-step identification is the key insight connecting marginal structural models
(`Robins et al., 2000 <https://doi.org/10.1097/00001648-200009000-00011>`_) to local
projections in economics, and it motivates the recursive estimation strategy in which
coefficients are estimated backward through time.

**Identification, General.** For :math:`T` periods, under Assumptions 1'--3', the result
generalises. For each period :math:`t`,

.. math::

   \mathbb{E}[Y_{i,T} \mid H_{i,t}, D_{i,t} = d_t, D_{i,1:(t-1)} = d_{1:(t-1)}]
   = H_{i,t}(d_{1:(t-1)})\,\beta_{d_{1:T}}^{(t)},

and projecting backward,

.. math::

   \mathbb{E}\bigl[\mathbb{E}[Y_{i,T} \mid H_{i,t+1}, D_{i,1:t} = d_{1:t}]
   \;\big|\; H_{i,t}, D_{i,1:(t-1)} = d_{1:(t-1)}\bigr]
   = H_{i,t}(d_{1:(t-1)})\,\beta_{d_{1:T}}^{(t)}.

These recursive relationships show that the coefficients :math:`\beta_{d_{1:T}}^{(t)}` can
be estimated from observable data by regressing backward through time, and the potential
outcome :math:`\mu_T(d_{1:T})` can then be recovered by combining these estimates with
appropriately chosen balancing weights.

Estimation with Dynamic Balancing
---------------------------------

Estimation of :math:`\mu_T(d_{1:T})` proceeds in two stages, recursive coefficient
estimation followed by sequential balancing weight construction.

Recursive Coefficient Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Coefficients are estimated recursively backward through time using penalised regression.
Two model specifications are available, differing in how they handle treatment effect
heterogeneity.

**Fully interacted model.** Let :math:`\hat{\beta}_{d_{1:T}}^{(T)}` be the coefficient of
the regression of :math:`Y_{i,T}` onto :math:`H_{i,T}` for all units with
:math:`D_{i,1:T} = d_{1:T}` (dropping collinear columns). Then for each
:math:`t = T-1, \ldots, 1`, regress the fitted values
:math:`H_{i,t+1}\hat{\beta}_{d_{1:T}}^{(t+1)}` onto :math:`H_{i,t}` for units with
:math:`D_{i,1:t} = d_{1:t}` to obtain :math:`\hat{\beta}_{d_{1:T}}^{(t)}`.

**Linear model.** Let :math:`\tilde{\beta}^{(T)}` be the coefficient from regressing
:math:`Y_{i,T}` onto :math:`(H_{i,T}, D_{i,1:T})` for *all* units, without penalising the
treatment indicators :math:`D_{i,1:T}`. Create fitted values by plugging in the target
history :math:`d_T` for the treatment indicator. Then proceed recursively as above.

The linear model pools information across treatment paths, which improves precision with
long histories at the cost of imposing treatment effect homogeneity (additive and linear, as
in `Acemoglu et al. (2019) <https://doi.org/10.1086/700936>`_). The fully interacted model
allows arbitrary heterogeneity but requires the effective sample size to not shrink
exponentially in :math:`T`.

Both specifications use LASSO with cross-validated penalty, where treatment indicators are
left unpenalised to avoid shrinking the treatment effect toward zero. An alternative
``lasso_subsample`` strategy partitions the data into separate fitting and evaluation sets,
which can improve stability when the sample is small relative to the number of covariates.

Sequential Balancing Weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The key insight of DCB is that valid inference requires controlling the high-dimensional
bias through balancing conditions rather than propensity score estimation.

**Two-period estimator.** In the two-period case, the potential outcome estimator takes the
form of a sequential augmented estimator,

.. math::

   \hat{\mu}_2(d_1, d_2) =
   \hat{\gamma}_2(d_{1:2})^\top\bigl(Y_2 - H_2\hat{\beta}_{d_{1:2}}^{(2)}\bigr)
   + \hat{\gamma}_1(d_{1:2})^\top\bigl(H_2\hat{\beta}_{d_{1:2}}^{(2)} - X_1\hat{\beta}_{d_{1:2}}^{(1)}\bigr)
   + \bar{X}_1\hat{\beta}_{d_{1:2}}^{(1)},

where :math:`\bar{X}_1` is the sample mean. The last term,
:math:`\bar{X}_1\hat{\beta}_{d_{1:2}}^{(1)}`, would suffice as an estimator in low
dimensions, but it is not :math:`\sqrt{n}`-consistent when covariates are high-dimensional.
The first two terms correct for this by reweighting the period-specific residuals, with the
second-period term adjusting for the gap between the outcome and the period-2 prediction, and
the first-period term adjusting for the gap between the period-2 and period-1 predictions.

The estimation error decomposes as

.. math::

   \hat{\mu}_2 - \bar{X}_1\beta^{(1)} =
   \underbrace{\bigl(\hat{\gamma}_1^\top X_1 - \bar{X}_1\bigr)
   \bigl(\beta^{(1)} - \hat{\beta}^{(1)}\bigr)
   + \bigl(\hat{\gamma}_2^\top H_2 - \hat{\gamma}_1^\top H_2\bigr)
   \bigl(\beta^{(2)} - \hat{\beta}^{(2)}\bigr)}_{T_1\text{ (bias)}}
   + T_2 + T_3,

where :math:`T_2, T_3` are mean-zero under measurability conditions on the weights. The bias
:math:`T_1` is bounded by

.. math::

   |T_1| \leq
   \|\hat{\beta}^{(1)} - \beta^{(1)}\|_1\,
   \|\bar{X}_1 - \hat{\gamma}_1^\top X_1\|_\infty
   \;+\;
   \|\hat{\beta}^{(2)} - \beta^{(2)}\|_1\,
   \|\hat{\gamma}_2^\top H_2 - \hat{\gamma}_1^\top H_2\|_\infty.

This product-of-rates structure shows that to make the bias :math:`o(n^{-1/2})`, the weights
must satisfy *dynamic balance constraints*: the weighted covariates under
:math:`\hat{\gamma}_t` must be close (in :math:`\ell_\infty` norm) to the weighted covariates
under :math:`\hat{\gamma}_{t-1}`.

- The first term, :math:`\|\bar{X}_1 - \hat{\gamma}_1^\top X_1\|_\infty`, coincides with
  the static balancing condition of
  `Athey et al. (2018) <https://doi.org/10.1111/rssb.12268>`_.
- The second term,
  :math:`\|\hat{\gamma}_2^\top H_2 - \hat{\gamma}_1^\top H_2\|_\infty`, is novel and
  specific to the dynamic setting. It requires that histories in the second period are
  balanced once reweighted by the first-period weights.

**General estimator.** For :math:`T` periods, the estimator generalises to

.. math::

   \hat{\mu}_T(d_{1:T}) = \sum_{i=1}^n \Biggl\{
   \hat{\gamma}_{i,T}\,Y_{i,T}
   - \sum_{t=2}^T (\hat{\gamma}_{i,t} - \hat{\gamma}_{i,t-1})\,H_{i,t}\hat{\beta}_{d_{1:T}}^{(t)}
   - \Bigl(\hat{\gamma}_{i,1} - \frac{1}{n}\Bigr)\,X_{i,1}\hat{\beta}_{d_{1:T}}^{(1)}
   \Biggr\}.

**Error decomposition.** The estimation error of the general estimator decomposes into three
components. Define the residuals and prediction gaps as

.. math::

   \varepsilon_{i,T} = Y_{i,T} - H_{i,T}\beta_{d_{1:T}}^{(T)},
   \qquad
   \nu_{i,t} = H_{i,t+1}\beta_{d_{1:T}}^{(t+1)} - H_{i,t}\beta_{d_{1:T}}^{(t)}.

Then

.. math::

   \hat{\mu}_T(d_{1:T}) - \bar{X}_1\beta_{d_{1:T}}^{(1)} =
   \underbrace{\sum_{t=1}^T
   \bigl(\hat{\gamma}_t^\top H_t - \hat{\gamma}_{t-1}^\top H_t\bigr)
   \bigl(\beta_{d_{1:T}}^{(t)} - \hat{\beta}_{d_{1:T}}^{(t)}\bigr)}_{(I_1)\text{: bias}}
   + \underbrace{\hat{\gamma}_T^\top \varepsilon_T}_{(I_2)}
   + \underbrace{\sum_{t=2}^T \hat{\gamma}_{t-1}^\top \nu_{t-1}}_{(I_3)}.

The bias :math:`(I_1)` depends on the product of the coefficient estimation error and the
imbalance of the balancing weights, motivating the dynamic balance constraints. The remaining
terms :math:`(I_2)` and :math:`(I_3)` are mean-zero provided the weights satisfy two
measurability conditions: (i) :math:`\hat{\gamma}_t` is measurable with respect to
:math:`\sigma(H_{i,t}, D_{i,t})` but not :math:`Y_{i,T}`, and (ii) :math:`\hat{\gamma}_{i,t}`
is zero whenever :math:`D_{i,1:t} \neq d_{1:t}`. Both conditions are satisfied by the DCB
quadratic program by construction.

**Algorithm (DCB).** Initialise :math:`\hat{\gamma}_{i,0} = 1/n`. For each
:math:`t \in \{1, \ldots, T\}`, solve

.. math::

   \hat{\gamma}_t = \arg\min_{\gamma_t} \sum_{i=1}^n \gamma_{i,t}^2
   \quad\text{s.t.}\quad
   & \bigl\|\tfrac{1}{n}\sum_i (\hat{\gamma}_{i,t-1}\,H_{i,t}
     - \gamma_{i,t}\,H_{i,t})\bigr\|_\infty \leq K_{1,t}\,\delta_t(n,p_t), \\
   & \mathbf{1}^\top\gamma_t = 1,\;\; \gamma_t \geq 0,\;\;
     \|\gamma_t\|_\infty \leq C_{n,t}, \\
   & \gamma_{i,t} = 0 \;\text{if}\; D_{i,1:t} \neq d_{1:t}.

The weights minimise their :math:`\ell_2` norm (equivalently, maximise the effective sample
size) subject to dynamic balance, normalisation, and non-negativity constraints. Non-zero
weights are restricted to units whose observed treatment path matches the target history up
to period :math:`t`.

Tuning Parameter Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The quadratic program requires choosing three quantities. The balance tolerance and the upper
bound on individual weights are set following the theory as

.. math::

   \delta_t(n, p_t) = \frac{\log^{3/2}(p_t n)}{\sqrt{n}},
   \qquad
   C_{n,t} = \log(n) \cdot n^{-2/3}.

The implementation uses a data-driven grid search over :math:`K_{1,t}`, selecting the
smallest value for which the quadratic program admits a feasible solution. This approach
minimises the estimator's bias and, within the set of weighting estimators with the smallest
bias, selects the one with the smallest variance. When ``adaptive_balancing=True``, the
algorithm refines the constraints to weight more heavily the covariates with non-zero
estimated coefficients, further reducing the effective number of binding constraints.

The computational complexity scales polynomially in :math:`n` and :math:`p`, consisting of
a sequence of :math:`T` quadratic programs with linear constraints.

Comparison with IPW
~~~~~~~~~~~~~~~~~~~

The inverse probability weights

.. math::

   \hat{\gamma}_{i,t}^* =
   \hat{\gamma}_{i,t-1}\,\frac{\mathbf{1}\{D_{i,t}=d_t\}}{P(D_{i,t}=d_t \mid H_{i,t})}
   \bigg/
   \sum_i \hat{\gamma}_{i,t-1}\,\frac{\mathbf{1}\{D_{i,t}=d_t\}}{P(D_{i,t}=d_t \mid H_{i,t})}

are a feasible solution to the DCB quadratic program (as formalised in the Existence theorem
below). A useful diagnostic is :math:`1/(n\|\hat{\gamma}_t\|^2)`, which measures the effective
sample size at time :math:`t`. Larger values indicate more precise treatment effect
estimates. In the Acemoglu et al. application, the effective sample size for DCB at horizon
:math:`h = 3` is 62, compared to 10 for IPW at the same horizon.

DCB does not require consistent estimation or correct specification of the propensity
score. AIPW and IPW-MSM alternatives are also available as benchmarks but require
propensity score estimation.

Existence of Feasible Weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A key theoretical result is that the DCB quadratic program always admits a feasible
solution under the overlap assumption.

**Theorem (Existence).** Under Assumptions 1'--4, with

.. math::

   \delta_t(n, p_t) \geq c_{0,t}\,n^{-1/2}\log^{3/2}(p_t n)
   \quad\text{and}\quad
   C_{n,t} \geq \frac{\bar{c}}{n\delta^t}

for sufficiently large :math:`\bar{c}`, the following holds with probability approaching 1.
For each :math:`t \in \{1, \ldots, T\}`, there exists a feasible solution
:math:`\hat{\gamma}_t^*(\hat{\gamma}_{t-1})` of the form

.. math::

   \hat{\gamma}_{i,0}^* = \frac{1}{n}, \quad
   \hat{\gamma}_{i,t}^*(\hat{\gamma}_{t-1}) =
   \frac{\hat{\gamma}_{i,t-1}\,w_{i,t}^*}{\sum_j \hat{\gamma}_{j,t-1}\,w_{j,t}^*},
   \quad
   w_{i,t}^* = \frac{\mathbf{1}\{D_{i,t} = d_t\}}{P(D_{i,t} = d_t \mid H_{i,t})}.

The stabilised IPW weights, reweighted by the solution from the previous period, satisfy all
constraints in the quadratic program. Since DCB minimises the :math:`\ell_2` norm over this
constraint set, the DCB solution is guaranteed to have variance no larger than IPW.

**Corollary (Weight Stability).** The DCB weights satisfy the period-by-period bound

.. math::

   n\|\hat{\gamma}_t\|^2 \leq n\|\hat{\gamma}_t^*(\hat{\gamma}_{t-1})\|^2
   \quad\text{and}\quad
   n\|\hat{\gamma}_t\|^2 \leq n\,c_t\,\|\hat{\gamma}_{t-1}\|^2

for a finite constant :math:`c_t < \infty`. The first inequality shows that DCB weights have
smaller :math:`\ell_2` norm than IPW at every period. The second shows that the weights' norm
is controlled across periods by the norm in the previous period, preventing the kind of
explosive growth that plagues IPW weights in long panels.

Theoretical Properties and Inference
-------------------------------------

Convergence Rate
~~~~~~~~~~~~~~~~

Under Assumptions 1'--4 and the coefficient consistency condition

.. math::

   \max_t \|\hat{\beta}_{d_{1:T}}^{(t)} - \beta_{d_{1:T}}^{(t)}\|_1
   \cdot \delta_t(n, p_t) = o_p(n^{-1/2}),

the DCB estimator achieves a parametric :math:`n^{-1/2}` convergence rate even when the
number of covariates grows with :math:`n`, provided

.. math::

   \frac{\log\bigl(n \sum_t p_t\bigr)}{n^{1/4}} \to 0.

The coefficient consistency condition is
satisfied by standard high-dimensional estimators such as LASSO under sparsity and restricted
eigenvalue conditions. Two sufficient forms of this condition are:

(a) With sub-Gaussian covariates,

.. math::

   \max_t \|\hat{\beta}^{(t)} - \beta^{(t)}\|_1 = O_p(n^{-1/4}).

(b) With uniformly bounded covariates,

.. math::

   \max_t \|\hat{\beta}^{(t)} - \beta^{(t)}\|_1 = o_p(1/\log n).

Rate Advantage Over AIPW
~~~~~~~~~~~~~~~~~~~~~~~~~

The convergence rate conditions highlight a key advantage of DCB over augmented IPW. In
high-dimensional settings, AIPW requires *both*

.. math::

   \|\hat{e} - e\| = o_p(n^{-1/4})
   \quad\text{and}\quad
   \|\hat{\beta} - \beta\| = o_p(n^{-1/4}),

where :math:`e` denotes the propensity score. That is, AIPW demands consistent estimation of
both the propensity score and the outcome model at rates faster than :math:`n^{-1/4}` (the
product-of-rates condition for doubly robust estimators). DCB requires only the condition on
the outcome model coefficients, with *no condition on the propensity score*. This is
because the balancing weights inherit a product-of-rates structure of the form

.. math::

   \|\hat{\beta} - \beta\|_1 \cdot \delta(n, p),

where :math:`\delta(n, p)` is the balance tolerance controlled by the quadratic program
rather than a propensity score estimation error.

Propagation of Error Over Time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A natural concern with dynamic estimation is how error accumulates over multiple periods. The
total bias is bounded by

.. math::

   \sum_{t=1}^T
   \bigl\|\hat{\beta}_{d_{1:T}}^{(t)} - \beta_{d_{1:T}}^{(t)}\bigr\|_1
   \cdot
   \bigl\|\hat{\gamma}_t^\top H_t - \hat{\gamma}_{t-1}^\top H_t\bigr\|_\infty.

As the number of periods increases, it becomes harder to guarantee approximate balance,
reflected in the balance constraint constants growing at rate :math:`K_{1,t} = \log^{1/2}(t)`.
In practice, this means the effective sample size decreases with longer treatment histories,
which is why the ``histories_length`` option is recommended for long panels to diagnose this
trade-off between identification of long-run effects and statistical precision.

Variance Estimation
~~~~~~~~~~~~~~~~~~~

The analytical variance estimator is

.. math::

   \hat{V}_T(d_{1:T}) = \sum_{i=1}^n \Biggl\{
   n\hat{\gamma}_{i,T}^2\bigl(Y_{i,T} - H_{i,T}\hat{\beta}^{(T)}\bigr)^2
   + \sum_{t=1}^{T-1} n\hat{\gamma}_{i,t}^2
   \bigl(H_{i,t+1}\hat{\beta}^{(t+1)} - H_{i,t}\hat{\beta}^{(t)}\bigr)^2
   + \frac{1}{n}\bigl(\bar{X}_1\hat{\beta}^{(1)} - X_{i,1}\hat{\beta}^{(1)}\bigr)^2
   \Biggr\},

Each of the three terms captures uncertainty from a different source. The first accounts for
the final-period residual variance, weighted by :math:`\hat{\gamma}_{i,T}^2`. The second
accounts for the between-period prediction gaps, weighted by :math:`\hat{\gamma}_{i,t}^2`.
The third accounts for the baseline covariate variation. Under homoskedasticity the variance
is proportional to a weighted sum of :math:`\|\hat{\gamma}_t\|^2`, though homoskedasticity
is not required for the result.

The normalised estimator converges in distribution to a standard normal,

.. math::

   \frac{\sqrt{n}\bigl(\hat{\mu}_T(d_{1:T}) - \mu_T(d_{1:T})\bigr)}
   {\hat{V}_T(d_{1:T})^{1/2}} \;\xrightarrow{d}\; \mathcal{N}(0,1).

ATE Inference
~~~~~~~~~~~~~

Inference on the ATE for two histories :math:`d_{1:T}` and :math:`d'_{1:T}` with
:math:`d_1 \neq d'_1` follows as a direct corollary. The two potential outcome estimators
:math:`\hat{\mu}_T(d_{1:T})` and :math:`\hat{\mu}_T(d'_{1:T})` use disjoint sets of units
(those with :math:`D_{i,1} = d_1` versus :math:`D_{i,1} = d'_1`), so the ATE variance is
the sum of the individual variances,

.. math::

   \text{Var}\bigl(\widehat{\text{ATE}}\bigr) =
   \hat{V}_T(d_{1:T}) + \hat{V}_T(d'_{1:T}).

When conditioning on baseline covariates :math:`X_1`, the relevant variance for each
potential outcome subtracts the baseline variation term,

.. math::

   \hat{V}_T^{\text{cond}}(d_{1:T}) = \hat{V}_T(d_{1:T})
   - \frac{1}{n}\sum_{i=1}^n
   \bigl(\bar{X}_1\hat{\beta}^{(1)} - X_{i,1}\hat{\beta}^{(1)}\bigr)^2,

and the ATE variance is the sum of these conditional variances.

Critical Values
~~~~~~~~~~~~~~~

Two approaches to critical values are available.

**Gaussian quantiles** use the standard normal distribution. They are tighter but require
stronger regularity conditions on the balancing weights.

**Robust quantiles** use a chi-squared distribution to account for the estimation error of
the balancing weights and provide valid inference under weaker conditions. Specifically,
the robust approach constructs a test statistic :math:`T_n^2` that converges to a weighted
chi-squared distribution under the null, and uses the :math:`(1 - \alpha)` quantile of this
distribution for confidence intervals. Robust quantiles are recommended by default.

Clustered Standard Errors
~~~~~~~~~~~~~~~~~~~~~~~~~

When observations are correlated within clusters (for example, countries within the same
geographic region), the analytical variance estimator is replaced by a cluster-robust
sandwich estimator. The cluster-robust variance aggregates influence function contributions
within each cluster,

.. math::

   \hat{V}_T^{cl}(d_{1:T}) = \sum_{c=1}^C
   \Bigl(\sum_{i \in \mathcal{C}_c} \psi_i\Bigr)^2,

where :math:`\psi_i` is the influence function for unit :math:`i` and
:math:`\mathcal{C}_c` denotes the set of units in cluster :math:`c`. This produces valid
standard errors under arbitrary within-cluster dependence.

Practical Extensions
--------------------

Pooled Regression
~~~~~~~~~~~~~~~~~

By default, the estimator uses the outcome in the final period only. With the pooled option,
the regression model becomes

.. math::

   Y_{i,t}(d_{1:t}) = \beta_0 + \beta_1 d_t + \beta_2 Y_{i,t-1}(d_{1:(t-1)})
   + X_{i,t}(d_{1:(t-1)})\gamma + \tau_t + \varepsilon_{i,t},

where :math:`\tau_t` denotes time fixed effects. Pooling combines outcomes from all periods
into a single regression, increasing the effective sample size at the cost of assuming
that the treatment effect is stationary across periods. Standard errors are automatically
clustered at the unit level to account for within-unit serial correlation, unless a larger
clustering variable is specified.

Treatment History Length
~~~~~~~~~~~~~~~~~~~~~~~~

With long panels, selecting the full treatment history :math:`h = T` can thin the effective
sample because only units matching the entire treatment path contribute non-zero weights.
The ``histories_length`` option estimates effects for varying history lengths
:math:`h \in \{h_1, \ldots, h_K\}`, where each :math:`h` uses the last :math:`h` elements
of the treatment histories. The estimand becomes

.. math::

   \mathbb{E}\bigl[Y_{i,T}(D_{1:(T-h)}, d_{(T-h+1):T})\bigr]
   - \mathbb{E}\bigl[Y_{i,T}(D_{1:(T-h)}, d'_{(T-h+1):T})\bigr],

which compares the effect of the last :math:`h` periods of treatment while averaging over
prior assignments. Following the recommendation of the paper, reporting effects for multiple
values of :math:`h` (say :math:`h \in \{1, \ldots, 10\}` in a long panel) traces out how
the treatment effect evolves with exposure length. The standard errors at each horizon help
disentangle the trade-off between identification of long-run effects and statistical
precision.

Impulse Response
~~~~~~~~~~~~~~~~

Setting ``impulse_response=True`` changes the treatment sequences for each history length
:math:`h` to :math:`d_{1:h} = (1, 0, \ldots, 0)` versus :math:`d'_{1:h} = (0, \ldots, 0)`.
This measures the effect of a one-period treatment shock at varying horizons, analogous to
impulse response functions in time series analysis. The impulse response is useful for
studying how a transient treatment (such as a one-time policy intervention) propagates
through the system over time.

Heterogeneous Effects Across Periods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``final_periods`` option estimates treatment effects at multiple final time periods,
holding the treatment histories fixed. This reveals how the same treatment history produces
different outcomes at different points in time, which can arise from time-varying confounders,
cohort effects, or secular trends.

.. note::

   For the complete theoretical treatment, including formal proofs of the existence of
   feasible weights, the connection between DCB and marginal structural models, comparisons
   with standard local projections and DiD, and extensive numerical simulations, see the
   original paper by
   `Viviano and Bradic (2026) <https://doi.org/10.1093/biomet/asag016>`_.
