.. _background-diddynamic:

Dynamic Covariate Balancing DiD
================================

The ``diddynamic`` module implements the dynamic covariate balancing (DCB) estimator of
`Viviano and Bradic (2026) <https://doi.org/10.1093/biomet/asag016>`_ for panel data where
treatments change over time. It handles settings where treatment assignments depend on
high-dimensional covariates, past outcomes, and past treatments, and where outcomes and
time-varying covariates may depend on the entire trajectory of past treatments.

Think about a country deciding whether to adopt democratic institutions. That decision is
shaped by past economic performance, the political trajectories of neighbours, and dozens
of other observable factors. Once the country transitions, its GDP, trade flows, and
institutional quality all change, which in turn influence whether democracy persists in the
next period. This kind of feedback loop between treatments and outcomes over time is
precisely what makes dynamic treatment regimes so challenging to analyse. The standard DiD
toolkit, built for settings where treatment is a one-time permanent event, simply cannot
accommodate this back-and-forth.

Why standard approaches fall short
-----------------------------------

Before introducing DCB, it helps to understand what goes wrong with the usual tools.

**Standard DiD and event studies** assume staggered adoption, where once a unit receives
treatment it stays treated forever. When countries can switch in and out of democracy based
on past economic outcomes, the parallel trends assumption breaks down
(`Ghanem et al., 2022 <https://doi.org/10.3982/ECTA19402>`_;
`Marx et al., 2022 <https://doi.org/10.1016/j.jeconom.2021.12.014>`_). TWFE compounds the
problem by collapsing different treatment sequences into a single coefficient, and negative
weighting gets worse because the pool of "control" units shifts every period.

**Standard local projections** (`Jordà, 2005 <https://doi.org/10.1257/0002828053828518>`_)
regress observed outcomes on current and lagged treatments plus covariates. The trouble is
that this model is written in terms of *observed* rather than *potential* outcomes. With
dynamic treatment selection, the resulting coefficient conflates the causal effect with the
distribution of future treatment decisions, which means it depends on the propensity score.
In the empirical application of
`Viviano and Bradic (2026) <https://doi.org/10.1093/biomet/asag016>`_, local projections
substantially underestimate long-run treatment effects compared to DCB.

**Inverse probability weighting** (IPW) takes a different approach, reweighting each unit
by the probability of the treatment sequence it actually experienced. For :math:`T` periods,
the weight for unit :math:`i` is the product of :math:`T` conditional probabilities,

.. math::

   w_i = \prod_{t=1}^T \frac{1}{P(D_{i,t} = d_t \mid H_{i,t})},

and this product can blow up fast. If any single-period propensity score is close to zero,
the whole product explodes. In many empirical settings the estimated probability of following
a given treatment path for just two consecutive periods already drops below 0.1 for some
units, making IPW weights for longer histories wildly variable. DCB sidesteps this entirely
by
constructing balancing weights through a quadratic program that never estimates the
propensity score at all.

Setup and notation
------------------

We observe a panel of :math:`n` i.i.d. units over :math:`T` periods. For unit :math:`i` in
period :math:`t`, let :math:`X_{i,t}` denote time-varying covariates,
:math:`D_{i,t} \in \{0,1\}` the binary treatment, and :math:`Y_{i,t}` the outcome. All the
information available up to (but not including) the treatment decision at time :math:`t` is
collected in the history vector

.. math::

   H_{i,t} = \bigl[D_{i,1}, \ldots, D_{i,t-1},\;
              X_{i,1}, \ldots, X_{i,t},\;
              Y_{i,1}, \ldots, Y_{i,t-1}\bigr] \in \mathbb{R}^{p_t},

which grows with :math:`t` as each additional period contributes its own covariates,
treatments, and outcomes. Since those covariates and outcomes may themselves depend on
earlier treatments, we also define the *potential* history under treatment path
:math:`d_{1:(t-1)}`,

.. math::

   H_{i,t}(d_{1:(t-1)}) = \bigl[d_{1:(t-1)},\;
   X_{i,1:t}(d_{1:(t-1)}),\;
   Y_{i,1:(t-1)}(d_{1:(t-1)})\bigr],

capturing the covariates and outcomes that *would have been observed* had the unit followed
treatment path :math:`d_{1:(t-1)}`.

What we are estimating
~~~~~~~~~~~~~~~~~~~~~~

The target is the average treatment effect of two treatment histories :math:`d_{1:T}` and
:math:`d'_{1:T}`,

.. math::

   \text{ATE}(d_{1:T}, d'_{1:T}) = \mu_T(d_{1:T}) - \mu_T(d'_{1:T}),
   \quad
   \mu_T(d_{1:T}) = \mathbb{E}\bigl[Y_T(d_{1:T})\bigr],

where :math:`Y_T(d_{1:T})` is the potential outcome at the final period under the full
history :math:`d_{1:T}`. This captures the total effect, including both direct effects on
the outcome and indirect effects that propagate through intermediate covariates and
outcomes.

A few concrete examples make this more tangible.

- :math:`\text{ATE}((1,1),(0,0))` asks what happens when a unit is treated for two
  consecutive periods compared to untreated for two periods. This is the most common target
  in short-panel applications.
- :math:`\text{ATE}((1,0),(0,0))` isolates the direct effect of a single period of
  treatment that is then reversed. Comparing this to :math:`\text{ATE}((1,1),(0,0))` reveals
  how much of the total effect comes from sustained versus one-time exposure.
- With long panels, we often want to average over earlier treatment assignments and focus on
  the last :math:`h` periods. The resulting estimand,

  .. math::

     \mathbb{E}\bigl[Y_T(D_{1:(T-h)}, d_{(T-h+1):T})\bigr]
     - \mathbb{E}\bigl[Y_T(D_{1:(T-h)}, d'_{(T-h+1):T})\bigr],

  is what the ``histories_length`` option targets. Varying :math:`h` traces out how the
  treatment effect evolves with exposure length.

Identifying assumptions
-----------------------

Identification rests on three core assumptions that generalise the standard
unconfoundedness framework to the dynamic setting. We develop these first for two periods,
where the logic is easiest to follow, and then state the general versions.

The two-period case
~~~~~~~~~~~~~~~~~~~

With two periods, we observe :math:`(X_{i,1}, D_{i,1}, Y_{i,1}, X_{i,2}, D_{i,2},
Y_{i,2})` for each unit and define :math:`H_{i,2} = [D_{i,1}, X_{i,1}, X_{i,2}, Y_{i,1}]`.
The potential outcome :math:`Y_{i,2}(d_1, d_2)` represents the outcome a unit would achieve
if it received treatment :math:`d_1` in period 1 and :math:`d_2` in period 2.

.. admonition:: Assumption 1 (No Anticipation)

   For :math:`d_1 \in \{0,1\}`, let :math:`Y_{i,1}(d_1, 1) = Y_{i,1}(d_1, 0)` and
   :math:`X_{i,2}(d_1, 1) = X_{i,2}(d_1, 0)`.

   Intermediate outcomes and covariates depend only on past treatments, not on future ones.
   Treatment at :math:`t = 2` has no contemporaneous effect on covariates.

This is a standard restriction in the causal inference literature. It allows forward-looking
behaviour (a unit may choose treatment in anticipation of future benefits) but rules out
effects from treatment realisations that haven't happened yet.

.. admonition:: Assumption 2 (Sequential Ignorability)

   For all :math:`(d_1, d_2) \in \{0,1\}^2`,

   (A) :math:`Y_{i,2}(d_1, d_2) \perp D_{i,2} \mid D_{i,1}, X_{i,1}, X_{i,2}, Y_{i,1}`,

   (B) :math:`(Y_{i,2}(d_1, d_2), H_{i,2}(d_1)) \perp D_{i,1} \mid X_{i,1}`.

Part (A) says that, once we condition on everything observed through period 1, the
second-period treatment is as good as random. Part (B) says the same for the first-period
treatment conditional on baseline covariates. Together, these allow treatment decisions to
depend on all observed history (including past outcomes and treatments) as long as there are
no unobserved confounders *after* conditioning.

.. admonition:: Assumption 3 (Potential Local Projections)

   For some :math:`\beta_{d_1,d_2}^{(1)} \in \mathbb{R}^{p_1}` and
   :math:`\beta_{d_1,d_2}^{(2)} \in \mathbb{R}^{p_2}`,

   .. math::

      \mathbb{E}[Y_{i,2}(d_1, d_2) \mid X_{i,1} = x_1] &= x_1\,\beta_{d_1,d_2}^{(1)}, \\
      \mathbb{E}[Y_{i,2}(d_1, d_2) \mid X_{i,1}, X_{i,2}, Y_{i,1}, D_{i,1} = d_1]
      &= [d_1, X_{i,1}, X_{i,2}, Y_{i,1}]\,\beta_{d_1,d_2}^{(2)}.

This is where DCB departs from both standard local projections and from IPW. Following the
spirit of `Jordà (2005) <https://doi.org/10.1257/0002828053828518>`_, Assumption 3 imposes
linearity, but on expected *potential* outcomes rather than observed ones. This distinction
matters greatly. A model on realised outcomes would tie the estimated treatment effect to the
propensity score, whereas a potential outcome model does not. Coefficients can differ across
treatment histories :math:`(d_1, d_2)`, and the dimensions :math:`p_1, p_2` are allowed to
grow with :math:`n`, so the model can accommodate large numbers of covariates and their
transformations. In high dimensions, the linearity can be relaxed to an approximation
accurate to :math:`o(n^{-1/2})`, which is enough for valid inference.

General case
~~~~~~~~~~~~

Moving from two to :math:`T` periods is conceptually straightforward. The key change is that
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

Overlap ensures that the dynamic balance constraints introduced below are feasible.
Intuitively, every unit must have a positive probability of following the target treatment
path at each step. The true IPW weights turn out to be one feasible set of weights, but DCB
will find better ones.

From assumptions to estimable quantities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The assumptions above connect the potential outcome model to things we can actually measure.
In the two-period case, the identification result is especially clean.

**Lemma (Identification, Two Periods).** Under Assumptions 1--3, for any
:math:`(d_1, d_2) \in \{0,1\}^2`,

.. math::

   \mathbb{E}\bigl[Y_{i,2} \mid H_{i,2},\, D_{i,2} = d_2,\, D_{i,1} = d_1\bigr]
   &= \mathbb{E}\bigl[Y_{i,2}(d_1, d_2) \mid H_{i,2},\, D_{i,1} = d_1\bigr] \\
   &= H_{i,2}(d_1)\,\beta_{d_1,d_2}^{(2)},

and, iterating the conditional expectation,

.. math::

   \mathbb{E}\bigl[\mathbb{E}[Y_{i,2} \mid H_{i,2},\, D_{i,2} = d_2,\, D_{i,1} = d_1]
     \,\big|\, X_{i,1},\, D_{i,1} = d_1\bigr]
   &= \mathbb{E}\bigl[Y_{i,2}(d_1, d_2) \mid X_{i,1}\bigr] \\
   &= X_{i,1}\,\beta_{d_1,d_2}^{(1)}.

Read from top to bottom, the logic is recursive. The first line uses sequential ignorability
(Assumption 2A) to swap the observed outcome for the potential outcome, then applies the
linear model (Assumption 3) to express the conditional expectation in terms of
:math:`H_{i,2}(d_1)`. The second line iterates backward, using Assumption 2B and the
first-period projection to push the conditioning down to baseline covariates :math:`X_{i,1}`.

This two-step recursion is the key insight of the paper. It connects the marginal structural
models literature
(`Robins et al., 2000 <https://doi.org/10.1097/00001648-200009000-00011>`_) to local
projections in economics, and it motivates the backward estimation strategy at the heart of
DCB.

For :math:`T` periods, the same recursion applies at each step. For every :math:`t`,

.. math::

   \mathbb{E}[Y_{i,T} \mid H_{i,t}, D_{i,t} = d_t, D_{i,1:(t-1)} = d_{1:(t-1)}]
   = H_{i,t}(d_{1:(t-1)})\,\beta_{d_{1:T}}^{(t)},

and projecting backward,

.. math::

   \mathbb{E}\bigl[\mathbb{E}[Y_{i,T} \mid H_{i,t+1}, D_{i,1:t} = d_{1:t}]
   \;\big|\; H_{i,t}, D_{i,1:(t-1)} = d_{1:(t-1)}\bigr]
   = H_{i,t}(d_{1:(t-1)})\,\beta_{d_{1:T}}^{(t)}.

These relationships tell us how to estimate the coefficients from data (regress backward
through time) and how to recover the potential outcome :math:`\mu_T(d_{1:T})` by combining
those estimates with balancing weights.

Estimation
----------

With the identification result in hand, estimation proceeds in two stages.

Recursive coefficient estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The coefficients are estimated by working backward from the final period, with penalised
regression at each step. Two model specifications are available, trading off flexibility
against sample efficiency.

The **fully interacted model** starts by regressing :math:`Y_{i,T}` onto :math:`H_{i,T}`
using only units with :math:`D_{i,1:T} = d_{1:T}`. It then regresses the fitted values
:math:`H_{i,t+1}\hat{\beta}_{d_{1:T}}^{(t+1)}` onto :math:`H_{i,t}` for units with
:math:`D_{i,1:t} = d_{1:t}`, working backward to :math:`t = 1`. This allows completely
heterogeneous treatment effects but limits the sample at each step to units on the target
path.

The **linear model** instead regresses :math:`Y_{i,T}` onto :math:`(H_{i,T}, D_{i,1:T})`
using *all* units, leaving treatment indicators unpenalised. It then plugs in the target
history for the treatment indicators and proceeds backward. This pools information across
treatment paths, improving precision with long histories at the cost of assuming treatment effects are additive and linear.

Both specifications use LASSO with cross-validated penalty, keeping treatment indicators
unpenalised to avoid shrinking the treatment effect toward zero. An alternative
``lasso_subsample`` strategy partitions the data into separate fitting and evaluation sets,
which can improve stability when the sample is small relative to the number of covariates.

Sequential balancing weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is where DCB diverges most sharply from existing methods. Rather than estimating
propensity scores and inverting them (as IPW does), DCB finds weights that directly
*balance* the covariate distributions across treatment groups. The balancing is done
sequentially, one period at a time, through a series of quadratic programs.

To see why balancing matters, consider the two-period estimator,

.. math::

   \hat{\mu}_2(d_1, d_2) =
   \hat{\gamma}_2(d_{1:2})^\top\bigl(Y_2 - H_2\hat{\beta}_{d_{1:2}}^{(2)}\bigr)
   + \hat{\gamma}_1(d_{1:2})^\top\bigl(H_2\hat{\beta}_{d_{1:2}}^{(2)} - X_1\hat{\beta}_{d_{1:2}}^{(1)}\bigr)
   + \bar{X}_1\hat{\beta}_{d_{1:2}}^{(1)},

where :math:`\bar{X}_1` is the sample mean. In low dimensions, the last term alone would
give a consistent estimator. But with many covariates, the LASSO estimates
:math:`\hat{\beta}` have non-negligible bias, and the first two terms are the correction.
They reweight the period-specific residuals so that the high-dimensional estimation error
washes out.

When we decompose the estimation error, the key term is

.. math::

   |T_1| \leq
   \|\hat{\beta}^{(1)} - \beta^{(1)}\|_1\,
   \|\bar{X}_1 - \hat{\gamma}_1^\top X_1\|_\infty
   \;+\;
   \|\hat{\beta}^{(2)} - \beta^{(2)}\|_1\,
   \|\hat{\gamma}_2^\top H_2 - \hat{\gamma}_1^\top H_2\|_\infty.

This has a beautiful product-of-rates structure. Each factor is the product of a coefficient
estimation error and a *covariate imbalance* term. To make the bias vanish at the
:math:`o(n^{-1/2})` rate needed for valid inference, we need the weighted covariates under
:math:`\hat{\gamma}_t` to be close to those under :math:`\hat{\gamma}_{t-1}`.

- The first imbalance term,
  :math:`\|\bar{X}_1 - \hat{\gamma}_1^\top X_1\|_\infty`, is the same static balancing
  condition that appears in cross-sectional studies
  (`Athey et al., 2018 <https://doi.org/10.1111/rssb.12268>`_).
- The second,
  :math:`\|\hat{\gamma}_2^\top H_2 - \hat{\gamma}_1^\top H_2\|_\infty`, is new. It
  requires that the second-period histories be balanced after reweighting by the
  first-period weights. This is the *dynamic* balancing condition that gives the method
  its name.

For :math:`T` periods, the estimator generalises to

.. math::

   \hat{\mu}_T(d_{1:T}) = \sum_{i=1}^n \Biggl\{
   \hat{\gamma}_{i,T}\,Y_{i,T}
   - \sum_{t=2}^T (\hat{\gamma}_{i,t} - \hat{\gamma}_{i,t-1})\,H_{i,t}\hat{\beta}_{d_{1:T}}^{(t)}
   - \Bigl(\hat{\gamma}_{i,1} - \frac{1}{n}\Bigr)\,X_{i,1}\hat{\beta}_{d_{1:T}}^{(1)}
   \Biggr\},

and the estimation error decomposes into three terms. Define the residuals and prediction
gaps as

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

The bias :math:`(I_1)` is the product of coefficient errors and imbalances, exactly as in
the two-period case. The remaining terms :math:`(I_2)` and :math:`(I_3)` are mean-zero as
long as the weights satisfy two natural conditions: (i) :math:`\hat{\gamma}_t` depends
only on :math:`(H_{i,t}, D_{i,t})` and not on the outcome :math:`Y_{i,T}`, and
(ii) :math:`\hat{\gamma}_{i,t} = 0` whenever the unit's treatment path doesn't match the
target. Both are built into the quadratic program by construction.

The DCB algorithm
~~~~~~~~~~~~~~~~~~

With the motivation in place, the algorithm itself is simple. Initialise
:math:`\hat{\gamma}_{i,0} = 1/n` and for each :math:`t \in \{1, \ldots, T\}`, solve

.. math::

   \hat{\gamma}_t = \arg\min_{\gamma_t} \sum_{i=1}^n \gamma_{i,t}^2
   \quad\text{s.t.}\quad
   & \bigl\|\tfrac{1}{n}\sum_i (\hat{\gamma}_{i,t-1}\,H_{i,t}
     - \gamma_{i,t}\,H_{i,t})\bigr\|_\infty \leq K_{1,t}\,\delta_t(n,p_t), \\
   & \mathbf{1}^\top\gamma_t = 1,\;\; \gamma_t \geq 0,\;\;
     \|\gamma_t\|_\infty \leq C_{n,t}, \\
   & \gamma_{i,t} = 0 \;\text{if}\; D_{i,1:t} \neq d_{1:t}.

In words, at each period we find the weights with the smallest :math:`\ell_2` norm (which
maximises the effective sample size) subject to the dynamic balance constraint, a summing-to-
one normalisation, non-negativity, an upper bound on individual weights, and the requirement
that only units on the target treatment path receive positive weight.

The tuning parameters are set by the theory. The balance tolerance scales as

.. math::

   \delta_t(n, p_t) = \frac{\log^{3/2}(p_t n)}{\sqrt{n}},
   \qquad
   C_{n,t} = \log(n) \cdot n^{-2/3},

and a data-driven grid search selects the smallest :math:`K_{1,t}` for which the quadratic
program has a feasible solution. This minimises bias first, then variance. When
``adaptive_balancing=True``, the algorithm tightens the constraints on covariates with large
estimated coefficients, further improving balance where it matters most.

Computationally, this is a sequence of :math:`T` quadratic programs with linear constraints,
so the cost scales polynomially in :math:`n` and :math:`p`.

Why DCB beats IPW
~~~~~~~~~~~~~~~~~~

A natural question is why not just use IPW. The answer comes from a clean theoretical
result. The normalised IPW weights,

.. math::

   \hat{\gamma}_{i,t}^* =
   \hat{\gamma}_{i,t-1}\,\frac{\mathbf{1}\{D_{i,t}=d_t\}}{P(D_{i,t}=d_t \mid H_{i,t})}
   \bigg/
   \sum_i \hat{\gamma}_{i,t-1}\,\frac{\mathbf{1}\{D_{i,t}=d_t\}}{P(D_{i,t}=d_t \mid H_{i,t})}

are themselves a *feasible solution* to the DCB quadratic program. Since DCB minimises the
:math:`\ell_2` norm over a constraint set that includes IPW, the DCB weights are guaranteed
to have variance no larger than IPW. And in practice the improvement is often dramatic. The effective sample size diagnostic
:math:`1/(n\|\hat{\gamma}_t\|^2)` typically shows DCB retaining several times more effective
observations than IPW at the same horizon.

On top of lower variance, DCB has a major theoretical advantage. AIPW in high dimensions
requires consistent estimation of *both* the propensity score and the outcome model, each at
rate :math:`o_p(n^{-1/4})`. DCB requires only the outcome model condition, with no
assumption on the propensity score at all. The balancing weights absorb the role of the
propensity score through the quadratic program, and the product-of-rates structure

.. math::

   \|\hat{\beta} - \beta\|_1 \cdot \delta(n, p)

replaces the usual product of propensity score and outcome model errors with a product of
outcome model error and balance tolerance, which is controlled mechanically rather than
estimated.

**Weight stability across periods.** A further reassurance comes from the following result.
The DCB weights satisfy

.. math::

   n\|\hat{\gamma}_t\|^2 \leq n\|\hat{\gamma}_t^*(\hat{\gamma}_{t-1})\|^2
   \quad\text{and}\quad
   n\|\hat{\gamma}_t\|^2 \leq n\,c_t\,\|\hat{\gamma}_{t-1}\|^2

for a finite constant :math:`c_t`. The first bound says DCB beats IPW at every period. The
second says the weights' norm is controlled from one period to the next, preventing the
explosive growth that plagues IPW in long panels.

Inference
---------

Convergence and variance
~~~~~~~~~~~~~~~~~~~~~~~~~

Under the identifying assumptions and the coefficient consistency condition

.. math::

   \max_t \|\hat{\beta}_{d_{1:T}}^{(t)} - \beta_{d_{1:T}}^{(t)}\|_1
   \cdot \delta_t(n, p_t) = o_p(n^{-1/2}),

the DCB estimator converges at the parametric :math:`n^{-1/2}` rate, even with
high-dimensional covariates, as long as

.. math::

   \frac{\log\bigl(n \sum_t p_t\bigr)}{n^{1/4}} \to 0.

This condition is satisfied by LASSO under standard sparsity and restricted eigenvalue
assumptions, either with sub-Gaussian covariates
(:math:`\max_t \|\hat{\beta}^{(t)} - \beta^{(t)}\|_1 = O_p(n^{-1/4})`) or uniformly
bounded covariates
(:math:`\max_t \|\hat{\beta}^{(t)} - \beta^{(t)}\|_1 = o_p(1/\log n)`).

A natural worry with sequential estimation is that errors compound over time. They do, but
in a controlled way. The total bias is bounded by

.. math::

   \sum_{t=1}^T
   \bigl\|\hat{\beta}_{d_{1:T}}^{(t)} - \beta_{d_{1:T}}^{(t)}\bigr\|_1
   \cdot
   \bigl\|\hat{\gamma}_t^\top H_t - \hat{\gamma}_{t-1}^\top H_t\bigr\|_\infty,

with the balance constants growing at rate :math:`K_{1,t} = \log^{1/2}(t)`. In practice,
the effective sample size shrinks with longer histories, which is why reporting effects at
multiple history lengths is the recommended diagnostic.

The analytical variance estimator is

.. math::

   \hat{V}_T(d_{1:T}) = \sum_{i=1}^n \Biggl\{
   n\hat{\gamma}_{i,T}^2\bigl(Y_{i,T} - H_{i,T}\hat{\beta}^{(T)}\bigr)^2
   + \sum_{t=1}^{T-1} n\hat{\gamma}_{i,t}^2
   \bigl(H_{i,t+1}\hat{\beta}^{(t+1)} - H_{i,t}\hat{\beta}^{(t)}\bigr)^2
   + \frac{1}{n}\bigl(\bar{X}_1\hat{\beta}^{(1)} - X_{i,1}\hat{\beta}^{(1)}\bigr)^2
   \Biggr\},

with three terms reflecting three sources of uncertainty. The first captures the
final-period residual variance (weighted by the squared final-period weights), the second
captures the between-period prediction gaps (weighted by the squared intermediate weights),
and the third captures the baseline covariate variation. The normalised estimator is
asymptotically standard normal,

.. math::

   \frac{\sqrt{n}\bigl(\hat{\mu}_T(d_{1:T}) - \mu_T(d_{1:T})\bigr)}
   {\hat{V}_T(d_{1:T})^{1/2}} \;\xrightarrow{d}\; \mathcal{N}(0,1).

For the ATE comparing two histories :math:`d_{1:T}` and :math:`d'_{1:T}` with
:math:`d_1 \neq d'_1`, the two estimators use disjoint sets of units, so

.. math::

   \text{Var}\bigl(\widehat{\text{ATE}}\bigr) =
   \hat{V}_T(d_{1:T}) + \hat{V}_T(d'_{1:T}).

When conditioning on baseline covariates :math:`X_1`, the baseline variation term drops out,

.. math::

   \hat{V}_T^{\text{cond}}(d_{1:T}) = \hat{V}_T(d_{1:T})
   - \frac{1}{n}\sum_{i=1}^n
   \bigl(\bar{X}_1\hat{\beta}^{(1)} - X_{i,1}\hat{\beta}^{(1)}\bigr)^2,

and the ATE variance is the sum of these conditional variances.

Critical values and clustering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two flavours of critical values are available. **Gaussian quantiles** use the standard
normal and give tighter intervals when the balancing weights are well-behaved. **Robust
quantiles** use a chi-squared distribution that accounts for weight estimation error,
providing valid coverage under weaker conditions. Robust quantiles are the default and are
generally recommended.

When within-cluster correlation is a concern (countries in the same region, patients in the
same hospital), the analytical variance is replaced by a cluster-robust sandwich,

.. math::

   \hat{V}_T^{cl}(d_{1:T}) = \sum_{c=1}^C
   \Bigl(\sum_{i \in \mathcal{C}_c} \psi_i\Bigr)^2,

where :math:`\psi_i` is the influence function and :math:`\mathcal{C}_c` is the set of
units in cluster :math:`c`. This is valid under arbitrary within-cluster dependence.

Practical extensions
--------------------

Pooled regression
~~~~~~~~~~~~~~~~~

By default, the estimator targets the outcome in the final period. With ``pooled=True``,
the regression pools all periods into a single model with time fixed effects,

.. math::

   Y_{i,t}(d_{1:t}) = \beta_0 + \beta_1 d_t + \beta_2 Y_{i,t-1}(d_{1:(t-1)})
   + X_{i,t}(d_{1:(t-1)})\gamma + \tau_t + \varepsilon_{i,t},

which increases the effective sample size at the cost of assuming the treatment effect is
stable across periods. Standard errors are automatically clustered at the unit level to
account for serial correlation, unless a larger clustering variable is specified.

Treatment history length
~~~~~~~~~~~~~~~~~~~~~~~~

With long panels, using the full treatment history thins the sample because only units
on the exact target path get positive weight. The ``histories_length`` option lets you
estimate effects at multiple horizons :math:`h \in \{h_1, \ldots, h_K\}`, each using the
last :math:`h` elements of the treatment sequences. The resulting estimand,

.. math::

   \mathbb{E}\bigl[Y_{i,T}(D_{1:(T-h)}, d_{(T-h+1):T})\bigr]
   - \mathbb{E}\bigl[Y_{i,T}(D_{1:(T-h)}, d'_{(T-h+1):T})\bigr],

averages over prior assignments and isolates the effect of the last :math:`h` periods.
Reporting a range of :math:`h` values (say 1 through 10 in a long panel) traces out how the
effect builds with exposure and reveals the precision trade-off at each horizon.

Impulse response
~~~~~~~~~~~~~~~~

Setting ``impulse_response=True`` flips the treatment sequences for each :math:`h` to
:math:`d_{1:h} = (1, 0, \ldots, 0)` versus :math:`d'_{1:h} = (0, \ldots, 0)`. This
measures the effect of a one-period treatment shock at increasing horizons, much like an
impulse response function in time series. It is particularly useful for studying how a
transient policy intervention propagates through the system over time.

Heterogeneous effects across periods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``final_periods`` option estimates treatment effects at multiple final time points,
holding the treatment histories fixed. This reveals whether the same treatment sequence
produces different outcomes at different calendar times, which could reflect time-varying
confounders, cohort effects, or secular trends.

.. note::

   For formal proofs, the connection to marginal structural models, detailed comparisons
   with local projections and DiD, and extensive Monte Carlo evidence, see the full paper by
   `Viviano and Bradic (2026) <https://doi.org/10.1093/biomet/asag016>`_.
