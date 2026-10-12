.. _background-diddynamic:

Dynamic covariate balancing
===========================

Dynamic covariate balancing (DCB) compares average outcomes under two
specified sequences of treatment. Either sequence may switch treatment on and
off over time. The challenge is that outcomes and covariates affected by
earlier treatment can also influence later treatment decisions. In a study of
democracy and growth, for example, past economic conditions may affect whether
democracy is adopted and whether it persists. Accounting for that feedback is
part of estimating the effect of sustained democracy on later GDP.

The ``diddynamic`` module implements the estimator of `Viviano and Bradic (2026)
<https://doi.org/10.1093/biomet/asag016>`_. Its sequential ignorability assumption
requires the observed history to account for confounding at each treatment
decision. Under that assumption and models for potential-outcome means, the
estimator works backward through those histories and corrects outcome
projections with balancing weights. This approach permits covariates to respond
to earlier treatment without estimating treatment probabilities.

.. admonition:: Path effects under parallel trends
   :class: note

   The :ref:`intertemporal DiD framework <background-didinter>` instead
   identifies path effects from parallel trends for outcome changes.

We start with the effect of a full treatment history and derive the recursive
projections from the identifying restrictions. The error of the resulting
estimator then explains why the balance constraints must hold at every period.
The later inference results make precise when those corrections support a normal
approximation. The :ref:`worked example <example_dyn_balancing>` connects these
steps to the choices you make in :func:`~moderndid.diddynamic.dyn_balancing`.

What a treatment history changes
--------------------------------

A sustained intervention and a temporary intervention can produce different
outcomes even when both end in the same treatment state. We therefore define
the target using the entire assignment sequence. For a panel of :math:`n`
i.i.d. units observed over a fixed number :math:`T` of periods, write
:math:`D_{i,t}\in\{0,1\}` for treatment, :math:`X_{i,t}` for covariates, and
:math:`Y_{i,t}` for the outcome. The information observed before assigning
:math:`D_{i,t}` is

.. math::

   H_{i,t}
   = [D_{i,1:(t-1)}, X_{i,1:t}, Y_{i,1:(t-1)}]
   \in\mathbb R^{p_t},
   \qquad H_{i,1}=X_{i,1}.

Although current covariates precede the current treatment decision, they
and earlier outcomes may already reflect earlier treatment. Let
:math:`H_{i,t}(d_{1:(t-1)})` denote the history that would arise under the
specified earlier assignments. Both the observed and the potential history
vectors can include an intercept.

In the implementation, ``xformla`` and ``fixed_effects`` determine the
columns used in each period's projection and balance constraints. The
package does not automatically append every past outcome and covariate.
Supply the lagged variables needed to represent your conditioning history
as columns in the data. Although the default ``lasso_plain`` coefficient fit
appends treatment indicators, those indicators alone do not account for
outcome-dependent selection.

The final potential outcome :math:`Y_{i,T}(d_{1:T})` incorporates the effects
of the full path. Those effects include changes that the path transmits through
intermediate outcomes and covariates. The population targets are

.. math::

   \mu_T(d_{1:T}) = \mathbb E[Y_{i,T}(d_{1:T})],

.. math::

   \operatorname{ATE}(d_{1:T},d'_{1:T})
   = \mu_T(d_{1:T})-\mu_T(d'_{1:T}).

For example, :math:`\operatorname{ATE}((1,1),(0,0))` compares two treated
periods with two untreated periods. The contrast
:math:`\operatorname{ATE}((1,0),(0,0))` measures the final-period effect of
a temporary intervention. It includes any effect transmitted through the
first-period outcome or second-period covariates. Interpreting it as a direct
effect would require additional restrictions on those pathways.

In the API, ``ds1`` and ``ds2`` specify the two histories. The result fields
``mu1`` and ``mu2`` estimate their potential-outcome means and ``att``
stores their difference, even though this target averages over the population
rather than conditioning on membership in a treated group.

.. admonition:: Pass binary treatments and covariate columns
   :class: important

   :func:`~moderndid.diddynamic.dyn_balancing` currently supports binary
   treatments and requires at least one covariate supplied through
   ``xformla``, ``fixed_effects``, or both. Since formula terms must name
   existing columns, create transformations such as log GDP before passing
   their column names.

Identification in two periods
-----------------------------

Before constructing weights, we need to explain why observed outcomes reveal
the outcome under a different treatment history. With two periods, the
available data are :math:`(X_{i,1},D_{i,1},Y_{i,1},X_{i,2},D_{i,2},Y_{i,2})`.
Assumptions 3.1 through 3.3 in the paper connect these observations to
the potential outcomes and histories under each path.

.. admonition:: Assumption 3.1 No anticipation
   :class: assumption

   For every :math:`d_1\in\{0,1\}`,

   .. math::

      Y_{i,1}(d_1,1)=Y_{i,1}(d_1,0),
      \qquad
      X_{i,2}(d_1,1)=X_{i,2}(d_1,0).

Both the first-period outcome and the second-period covariates can respond
to first-period treatment. Since they precede second-period assignment,
its realization leaves both unchanged.

.. admonition:: Assumption 3.2 Sequential ignorability
   :class: assumption

   For every :math:`(d_1,d_2)\in\{0,1\}^2`,

   .. math::

      Y_{i,2}(d_1,d_2)\perp D_{i,2}
      \mid D_{i,1},X_{i,1},X_{i,2},Y_{i,1},

   .. math::

      (Y_{i,2}(d_1,d_2),H_{i,2}(d_1))\perp D_{i,1}
      \mid X_{i,1}.

These independence restrictions permit treatment to depend on observed past
outcomes. They require the covariates and earlier observations to account for
confounding at each decision. In particular, the second restriction concerns
both the final outcome and the intermediate history under first-period
treatment. Adjusting only for baseline covariates at every period would discard
information used by the first restriction.

.. admonition:: Assumption 3.3 Potential local projections
   :class: assumption

   For every :math:`(d_1,d_2)\in\{0,1\}^2`, there are vectors
   :math:`\beta_{d_1,d_2}^{(1)}` and
   :math:`\beta_{d_1,d_2}^{(2)}` such that

   .. math::

      \mathbb E[Y_{i,2}(d_1,d_2)\mid X_{i,1}]
      = X_{i,1}\beta_{d_1,d_2}^{(1)},

   .. math::

      \mathbb E[Y_{i,2}(d_1,d_2)\mid
      D_{i,1}=d_1,X_{i,1},X_{i,2},Y_{i,1}]
      = H_{i,2}(d_1)\beta_{d_1,d_2}^{(2)}.

Allowing coefficients to differ across treatment histories gives each
potential-outcome mean its own linear representation in the chosen
covariates. You can include transformations or interactions as columns to
enlarge that representation. A linear regression for the observed outcome
alone does not impose this model, because observed future treatment can vary
across units.

.. admonition:: Lemma 3.1 Two-period identification
   :class: theorem

   Under Assumptions 3.1 through 3.3, for a target history :math:`(d_1,d_2)`
   on the relevant support,

   .. math::

      \mathbb E[Y_{i,2}\mid H_{i,2},D_{i,1}=d_1,D_{i,2}=d_2]
      =H_{i,2}(d_1)\beta_{d_1,d_2}^{(2)},

   .. math::

      \mathbb E[H_{i,2}(d_1)\beta_{d_1,d_2}^{(2)}
      \mid X_{i,1},D_{i,1}=d_1]
      =X_{i,1}\beta_{d_1,d_2}^{(1)}.

Identification starts with the outcome projection among units following
the target path, integrates over the history generated by first-period
treatment, and averages the resulting baseline projection to identify
:math:`\mu_2(d_1,d_2)`. Positive treatment probabilities on the relevant
histories make these conditional comparisons available. The overlap condition
below supplies a stronger bound for the estimation results.

This backward integration explains why including intermediate covariates need
not remove the pathways you want to measure. DCB projects the final potential
outcome onto those covariates and subsequently averages over the intermediate
history that the target treatment would generate.

More than two treatment decisions
---------------------------------

Each additional decision creates another intermediate history to integrate
out. We keep the outcome period fixed at :math:`T` and work backward through
those histories. Assumption 4.1 in the paper collects the identifying
restrictions for this setting.

.. admonition:: Assumption 4.1 Sequential identification and projections
   :class: assumption

   For every :math:`d_{1:T}\in\{0,1\}^T` and :math:`t\leq T`, the following
   restrictions hold.

   The potential history :math:`H_{i,t}(d_{1:T})` is constant in
   :math:`d_{t:T}` and can therefore be written
   :math:`H_{i,t}(d_{1:(t-1)})`.

   The final potential outcome and subsequent potential histories satisfy

   .. math::

      \bigl(Y_{i,T}(d_{1:T}),
      H_{i,t+1}(d_{1:t}),\ldots,
      H_{i,T}(d_{1:(T-1)})\bigr)
      \perp D_{i,t}\mid H_{i,t}.

   The list of subsequent histories is empty when :math:`t=T`. For some
   :math:`\beta_d^{(t)}\in\mathbb R^{p_t}`,

   .. math::

      \mathbb E[Y_{i,T}(d_{1:T})\mid
      D_{i,1:(t-1)}=d_{1:(t-1)},X_{i,1:t},Y_{i,1:(t-1)}]
      = H_{i,t}(d_{1:(t-1)})\beta_d^{(t)}.

Ending the list at :math:`H_{i,T}(d_{1:(T-1)})` makes the independence
condition match Assumption 3.2 when :math:`T=2`. The arXiv version of the
paper ends it at :math:`H_{i,T-1}(d_{1:(T-1)})` instead. That shorter list is
empty when :math:`T=2` and leaves out the :math:`H_{i,2}(d_1)` that
Assumption 3.2 includes.

Write :math:`Q_T^d(H_{i,T})=H_{i,T}\beta_d^{(T)}` for the final-period
projection on units with the target earlier path. The backward recursion is

.. math::

   Q_t^d(H_{i,t})
   =\mathbb E[Q_{t+1}^d(H_{i,t+1})\mid
   H_{i,t},D_{i,1:t}=d_{1:t}]
   =H_{i,t}\beta_d^{(t)},\qquad t<T.

At the last period, the regression response is the observed final outcome.
At earlier periods, the response is the next projection evaluated for the
specified future path. Regressing the raw final outcome at every earlier period
would average over observed future treatment decisions and answer a different
question. This is the distinction between potential local projections and a
local projection of observed outcomes discussed in
`Section 3 of the paper <https://arxiv.org/html/2103.01280#S3>`_.

Fitting the projections
-----------------------

The recursive models must be estimated before the balancing correction can be
computed. With many covariates, regularization can make those estimates usable
but introduce bias. Starting from a projection at the final period, we
regress each period's fitted values on the history one period earlier until
the recursion reaches baseline.

``method="lasso_subsample"`` fits each projection on units whose observed
path agrees with the target through that period. Fitting each path separately
lets the coefficients differ across paths. It also leaves fewer observations at
later periods because fewer units follow the target path that long. The
default ``method="lasso_plain"`` pools units across paths and includes
treatment indicators as regressors. Evaluating those indicators at the target
history imposes a more restrictive additive specification.

.. admonition:: Refit without the additive specification
   :class: tip

   Refit with ``method="lasso_subsample"`` to see whether the additive
   treatment terms drive your estimate. The
   :ref:`dynamic covariate balancing example <example_dyn_balancing>` makes
   that comparison on the democracy data.

With ``regularization=True``, the coefficient stage uses cross-validated
LASSO with the number of folds set by ``nfolds``. The ``lasso_plain`` fit
leaves the most recent ``lags`` treatment indicators unpenalized. Since
``lags`` defaults to the length of ``ds1``, every treatment indicator stays
unpenalized by default. That fit scales penalized columns to unit standard
deviation and chooses the largest penalty within one standard error of the
minimum cross-validation error. With ``regularization=False``, every
projection uses ridge regression with a negligible penalty instead. These
fitting choices do not establish the population linearity or estimation-rate
conditions required by the paper.

Why the weights are sequential
------------------------------

Balancing only the baseline covariates leaves the second treatment decision
unadjusted. We need the second-period histories under the new weights to match
the histories already reweighted for first-period treatment. For two periods,
the corrected estimator is

.. math::

   \begin{aligned}
   &\widehat\mu_2(d_1,d_2)
   ={}\widehat\gamma_2^\top(Y_2-H_2\widehat\beta_d^{(2)})\\
   &\quad+\widehat\gamma_1^\top
      (H_2\widehat\beta_d^{(2)}-X_1\widehat\beta_d^{(1)})
   +\overline X_1\widehat\beta_d^{(1)}.
   \end{aligned}

The average fitted baseline outcome receives two corrections, one from
the final residual and one from the change between successive projections.
The contribution from coefficient estimation is bounded by

.. math::

   \begin{aligned}
   |I_1|\leq{}&
   \|\widehat\beta_d^{(1)}-\beta_d^{(1)}\|_1
   \|\overline X_1-\widehat\gamma_1^\top X_1\|_\infty\\
   &+\|\widehat\beta_d^{(2)}-\beta_d^{(2)}\|_1
   \|\widehat\gamma_2^\top H_2-
        \widehat\gamma_1^\top H_2\|_\infty.
   \end{aligned}

The product of the :math:`\ell_1` coefficient error across coordinates and
the :math:`\ell_\infty` error in the most imbalanced history coordinate can
be small even when neither component vanishes at the rate of the final
estimator. Controlling weighted history means at every period therefore
limits how much coefficient error reaches the final estimate.

For a general path, initialize :math:`\widehat\gamma_{i,0}=1/n` and solve
one quadratic program for each :math:`t=1,\ldots,T`.

.. math::

   \widehat\gamma_t\in\arg\min_{\gamma_t}\sum_{i=1}^n\gamma_{i,t}^2

.. math::

   \begin{gathered}
   \left\|\sum_i(\widehat\gamma_{i,t-1}-\gamma_{i,t})H_{i,t}
   \right\|_\infty\leq K_{1,t}\delta_t(n,p_t),\\
   \sum_i\gamma_{i,t}=1,\qquad
   0\leq\gamma_{i,t}\leq C_{n,t},\\
   \gamma_{i,t}=0\quad\text{if }D_{i,1:t}\ne d_{1:t}.
   \end{gathered}

Since the weights sum to one, the balance constraint compares weighted
means directly against the sample mean at baseline and against the history
mean under the preceding period's weights later on. Minimizing their squared
norm spreads weight among eligible units subject to these constraints.
For normalized nonnegative weights, the effective sample size is
:math:`1/\sum_i\widehat\gamma_{i,t}^2`.

The implementation uses the number :math:`n_t` of eligible path observations
in its tuning rules. It caps each weight at :math:`\log(n_t)n_t^{-2/3}` and,
for more than one history coordinate, sets each balance tolerance to a grid
constant times :math:`\sqrt{\log(p_t)/\sqrt{n_t}}`. The grid controlled by
``lb``, ``ub``, and ``grid_length`` searches those constants for feasible
tolerances and stops estimation if no weights satisfy the constraints.
Increasing ``ub`` permits weaker balance without creating observations
following a missing treatment path.

``adaptive_balancing=True`` gives tighter bounds to covariates selected using
the estimated projection coefficients. The fields ``gammas`` and
``imbalances`` let you examine the resulting concentration and balance.
A feasible solve establishes the numerical constraints for that sample
without verifying sequential ignorability or the asymptotic rate conditions.

The estimator and its error
---------------------------

Once the projections and weights are available, the general estimator combines
the same corrections across all periods. Suppressing the path argument on the
weights, it is

.. math::

   \begin{aligned}
   \widehat\mu_T(d)
   =\sum_{i=1}^n\biggl\{
   &\widehat\gamma_{i,T}Y_{i,T}\\
   &-\sum_{t=2}^T(\widehat\gamma_{i,t}-\widehat\gamma_{i,t-1})
       H_{i,t}\widehat\beta_d^{(t)}\\
   &-\left(\widehat\gamma_{i,1}-\frac1n\right)
       X_{i,1}\widehat\beta_d^{(1)}\biggr\}.
   \end{aligned}

Define the final residual and the one-step prediction gaps by

.. math::

   \varepsilon_{i,T}=Y_{i,T}-H_{i,T}\beta_d^{(T)},
   \qquad
   \nu_{i,t}=H_{i,t+1}\beta_d^{(t+1)}-H_{i,t}\beta_d^{(t)}.

Lemma 4.2 in the paper is an algebraic identity when final weights are zero
outside the target path. It separates the error around the sample baseline
projection as follows.

.. math::

   \widehat\mu_T(d)-\overline X_1\beta_d^{(1)}=I_1+I_2+I_3,

.. math::

   \begin{aligned}
   I_1&=\sum_{t=1}^T
       (\widehat\gamma_t^\top H_t-
        \widehat\gamma_{t-1}^\top H_t)
       (\beta_d^{(t)}-\widehat\beta_d^{(t)}),\\
   I_2&=\sum_i\widehat\gamma_{i,T}\varepsilon_{i,T},\\
   I_3&=\sum_{t=1}^{T-1}\sum_i\widehat\gamma_{i,t}\nu_{i,t}.
   \end{aligned}

The same product bound controls :math:`I_1` at every period and leaves
sampling variation in the remaining terms. Their mean-zero argument uses
weights that depend on information available at their own decision period
and are zero outside the corresponding target prefix. For two periods,
Theorem 4.1 requires :math:`\widehat\gamma_1` to be measurable with respect to
:math:`\sigma(X_1,D_1)` and :math:`\widehat\gamma_2` with respect to
:math:`\sigma(X_1,X_2,Y_1,D_1,D_2)`, in addition to Assumptions 3.1 through
3.3 and those support restrictions. Under these conditions,
:math:`\mathbb E[I_2\mid X_1,D_1,Y_1,X_2,D_2]=0` and
:math:`\mathbb E[I_3\mid X_1,D_1]=0`.

.. admonition:: Adaptive bounds can use later outcomes
   :class: warning

   Outcome-based adaptive bounds and tuning can make the fitted weights
   depend on later outcomes. The paper's measurability conditions concern the
   statistical construction of the weights, not just their support and
   balance constraints. Do not infer that the default adaptive fit satisfies
   every theorem condition because its quadratic programs are feasible.

When feasible weights exist
---------------------------

When few units follow the required early assignments, fewer histories are
available to balance at the next decision.
The paper uses overlap and tail restrictions to show that suitable tolerances
admit weights with probability approaching one. Since the implementation
searches only a finite grid of tolerances, feasibility in a particular sample
remains a numerical question.

.. admonition:: Assumption 5.1 Overlap and history tails
   :class: assumption

   For each target path and period, there is a common
   :math:`\epsilon\in(0,1/2)` such that

   .. math::

      \epsilon<\Pr(D_{i,t}=d_t\mid H_{i,t})<1-\epsilon.

   Each coordinate of :math:`X_{i,1}` is sub-Gaussian. For :math:`t\geq2`,
   each coordinate of :math:`H_{i,t}` is conditionally sub-Gaussian given
   :math:`H_{i,t-1}`.

Sub-Gaussian tails bound the probability of very large values in each history
coordinate. Overlap prevents any target decision from becoming arbitrarily
unlikely given the observed history and remains an assumption even though
DCB never estimates a propensity score.

.. admonition:: Theorem 5.1 and Corollary 1 Feasible weights
   :class: theorem

   Suppose Assumptions 4.1 and 5.1 hold, :math:`T` is fixed and finite, and
   the theoretical balance and cap sequences satisfy

   .. math::

      \delta_t(n,p_t)\geq c_{0,t}n^{-1/2}\log^{3/2}(p_tn),
      \qquad C_{n,t}\geq\frac{\overline c}{n\epsilon^t},

   for finite constants :math:`c_{0,t}` and sufficiently large finite
   :math:`\overline c>0`. For sufficiently large :math:`n`, the recursive
   candidate

   .. math::

      \gamma_{i,t}^{*}
      =\frac{\widehat\gamma_{i,t-1}
             \mathbf1\{D_{i,t}=d_t\}/\pi_t(H_{i,t})}
            {\sum_j\widehat\gamma_{j,t-1}
             \mathbf1\{D_{j,t}=d_t\}/\pi_t(H_{j,t})},
      \qquad \pi_t(H)=\Pr(D_{i,t}=d_t\mid H_{i,t}=H),

   is feasible at every period with probability approaching one.
   Under the same conditions, the minimizing feasible weights satisfy,
   on an event whose probability approaches one,

   .. math::

      n\|\widehat\gamma_t\|_2^2
      \leq n\|\gamma_t^*\|_2^2,
      \qquad
      n\|\widehat\gamma_t\|_2^2
      \leq n c_t\|\widehat\gamma_{t-1}\|_2^2,

   for finite constants :math:`c_t`.

By combining the current-period true propensity with the previous-period
DCB weights, the candidate provides a reference for bounding weight
concentration. Under homoskedastic projection errors, smaller weight norms
also reduce the corresponding variance components. Arbitrary
heteroskedasticity does not imply a universal variance ordering between DCB
and a separately estimated IPW procedure.

For comparison, ordinary path IPW has unnormalized weights

.. math::

   w_i(d)=\prod_{t=1}^T
   \frac{\mathbf1\{D_{i,t}=d_t\}}
        {\Pr(D_{i,t}=d_t\mid H_{i,t})}.

Small probabilities can compound across periods in these path weights.
Setting ``balancing="ipw"`` averages the final outcomes with normalized path
weights from logistic propensity scores. Setting ``balancing="aipw"`` puts
those weights in place of the balancing weights in the corrected estimator
above. The inverse probability weights then correct the outcome projections
of every period. Double robustness keeps that augmented estimator
consistent when either nuisance model is correct. Its normal approximation
still needs both nuisance estimates to converge fast enough. DCB instead
controls error through projection error and achieved imbalance.

Uncertainty and the population target
-------------------------------------

The linear models, balance bounds, and weight caps must work together for the
coefficient correction to be negligible at the inference scale. We state the
remaining conditions before presenting the normal limit. These are
Assumption 5.2 of the paper expressed in the residual and gap notation above.

.. admonition:: Assumption 5.2 Projection rates and outcome moments
   :class: assumption

   Uniformly over the fixed periods and target histories,

   .. math::

      \max_t\|\widehat\beta_d^{(t)}-\beta_d^{(t)}\|_1
      \delta_t(n,p_t)=o_p(n^{-1/2}),
      \qquad
      \delta_t(n,p_t)\geq c_{0,t}n^{-1/2}\log^{3/2}(p_tn).

   In addition, either
   :math:`\max_t\|\widehat\beta_d^{(t)}-\beta_d^{(t)}\|_1
   =O_p(n^{-1/4})`, or that maximum is :math:`o_p(1/\log n)` and all
   history coordinates are almost surely bounded by a finite common constant.

   There is a finite common :math:`C` such that

   .. math::

      \mathbb E[\varepsilon_{i,T}^4\mid H_{i,T},D_{i,T}]<C,
      \qquad
      \mathbb E[\nu_{i,t}^4\mid H_{i,t-1},D_{i,t-1}]<C
      \quad\text{almost surely}.

   At :math:`t=1`, the preceding conditioning set is empty.
   The final outcome :math:`Y_{i,T}` is sub-Gaussian. There is a common
   :math:`u_{\min}>0` such that the final residual variance conditional on
   :math:`(H_{i,T},D_{i,T})` and every one-step projection-gap variance
   conditional on its preceding history and treatment exceed
   :math:`u_{\min}` almost surely.

Although the paper prints the gap fourth-moment bound conditional on
:math:`(H_{i,t-1},D_{i,t-1})`, its normality proof conditions that gap on
:math:`(H_{i,t},D_{i,t})`. Imposing the same finite bound at this latter
history supplies the moment condition the proof uses.

The notation :math:`o_p(n^{-1/2})` means that multiplying the term by
:math:`\sqrt n` makes it converge to zero in probability. The first condition
therefore removes coefficient estimation from the leading sampling error.
Although a LASSO fit can satisfy it under appropriate sparsity and design
conditions, cross-validation alone does not check those conditions.

With :math:`\widehat\varepsilon_{i,T}` and :math:`\widehat\nu_{i,t}` computed
from the estimated coefficients, the population-mean variance scale is

.. math::

   \begin{aligned}
   \widehat V_T(d)={}&
      n\sum_i\widehat\gamma_{i,T}^2\widehat\varepsilon_{i,T}^2
   +n\sum_{t=1}^{T-1}\sum_i
         \widehat\gamma_{i,t}^2\widehat\nu_{i,t}^2\\
   &+\frac1n\sum_i
       \bigl((X_{i,1}-\overline X_1)\widehat\beta_d^{(1)}\bigr)^2.
   \end{aligned}

Population inference accounts for sampling different baseline covariates
as well as variation in the final residual and successive changes in
projected outcomes. The variance of the estimator is approximated by
:math:`\widehat V_T(d)/n`, rather than by :math:`\widehat V_T(d)` itself.

.. admonition:: Theorems 5.2 and 5.3 Rate and normality
   :class: theorem

   Under the conditions of Theorem 5.1 and Assumption 5.2, for fixed finite
   :math:`T`, suppose

   .. math::

      \frac{\log(n\sum_{t=1}^T p_t)}{n^{1/4}}\longrightarrow0

   as :math:`n,p_1,\ldots,p_T\longrightarrow\infty`.
   The estimator has the rate

   .. math::

      \widehat\mu_T(d)-\mu_T(d)=O_p(n^{-1/2}).

   For the normal limit, use weights measurable from information available
   at their own decision period and the shrinking cap
   :math:`C_{n,t}=\log(n)n^{-2/3}`. Impose the gap fourth-moment bound
   conditional on :math:`(H_{i,t},D_{i,t})` described above. Under these
   additional conditions, the studentized error satisfies

   .. math::

      \frac{\sqrt n(\widehat\mu_T(d)-\mu_T(d))}
           {\sqrt{\widehat V_T(d)}}
      \xrightarrow{d}\mathcal N(0,1).

The paper's statistical construction keeps the number of periods fixed
even as the dimension of the histories grows. Its assumptions and limits
do not establish a normal approximation for arbitrarily long paths or every
finite-sample tuning choice.

Population and conditional variances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You may instead treat the observed baseline covariates as fixed. The
corresponding mean target is :math:`\overline X_1\beta_d^{(1)}` and its
variance scale omits baseline sampling variation.

.. math::

   \widehat V_T^{\mathrm{cond}}(d)
   =\widehat V_T(d)-\frac1n\sum_i
      \bigl((X_{i,1}-\overline X_1)\widehat\beta_d^{(1)}\bigr)^2.

The package reports ``var_mu1`` and ``var_mu2`` as these conditional variance
estimates divided by :math:`n`. Its ``var_att`` is the conditional variance of
the difference between the two estimates, also divided by :math:`n`. Since a
unit carries weight under a path only while its observed treatments follow
that path, the two estimates weight the same units only before the period
:math:`s` in which the paths first differ. The conditional ATE variance
therefore subtracts twice their covariance over the periods before :math:`s`,

.. math::

   \begin{aligned}
   \widehat V_{\mathrm{ATE}}^{\mathrm{cond}}={}&
      \widehat V_T^{\mathrm{cond}}(d)
      +\widehat V_T^{\mathrm{cond}}(d')\\
   &-2n\sum_{t=1}^{s-1}\sum_i
      \widehat\gamma_{i,t}(d)\widehat\gamma_{i,t}(d')
      \widehat\nu_{i,t}(d)\widehat\nu_{i,t}(d').
   \end{aligned}

Because the covariance sum is empty when the first treatments differ,
``var_att`` without clustering reduces to ``var_mu1 + var_mu2``, the paper's
conditional ATE variance divided by :math:`n`. For the population ATE, the
shared baseline covariates also require the variance of their contrast.

.. math::

   \widehat V_{\mathrm{ATE}}^{\mathrm{pop}}
   =\widehat V_{\mathrm{ATE}}^{\mathrm{cond}}
   +\frac1n\sum_i\left[
      (X_{i,1}-\overline X_1)
      (\widehat\beta_d^{(1)}-\widehat\beta_{d'}^{(1)})
      \right]^2.

This is the covariance adjustment in Theorem C.1 of the paper. Its conditions
are Assumptions 4.1, 5.1, and 5.2, different first treatments
:math:`d_1\ne d'_1`, and
:math:`\log(np_T)/n^{1/4}\to0` as the sample and history dimensions grow.
Under these conditions and the weight conditions of the normal limit above,
the error of the estimated contrast, divided by
:math:`\sqrt{\widehat V_{\mathrm{ATE}}^{\mathrm{pop}}/n}`, has a standard
normal limit.

.. admonition:: Paths that share their first treatment
   :class: note

   The paper states no ATE variance for paths that share their first
   treatment. The package derives the covariance over the periods before
   :math:`s` by following the proof of Theorem C.1 under independent units.

Critical values and clustering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default interval uses a standard normal critical value at significance
level ``alpha``. ``robust_quantile=True`` instead uses the square root of a
chi-squared quantile with :math:`2T` degrees of freedom for the ATE interval
and takes the printed p-value from the same distribution. A ``maketables``
table of the fit reports the same interval and p-value. For the
potential-outcome intervals, :func:`~moderndid.to_df` and
:func:`~moderndid.plots.plot_dyn_balancing` use :math:`T` degrees of freedom
instead. Larger critical values widen the intervals without establishing the
assumptions behind the variance estimates.

For the single clustering column that ``clustervars`` accepts, the
implemented conditional variance of a path sums squared cluster contributions
separately for the final residual and for each projection gap.

.. math::

   \begin{aligned}
   \widehat v_T^{\mathrm{cluster}}(d)
   ={}&\sum_c\left(\sum_{i\in\mathcal C_c}
          \widehat\gamma_{i,T}\widehat\varepsilon_{i,T}\right)^2\\
   &+\sum_{t=1}^{T-1}\sum_c\left(\sum_{i\in\mathcal C_c}
          \widehat\gamma_{i,t}\widehat\nu_{i,t}\right)^2.
   \end{aligned}

The clustered ``var_att`` applies the same sums to the difference between
the two paths' weighted terms. It totals that difference within each cluster
and period before squaring. Units of one cluster that follow different paths
therefore enter the covariance in every period, even when the first
treatments differ. Because the calculation omits cross-period covariance
between cluster sums, the implemented formula does not provide a general
sandwich variance for arbitrary dependence within clusters. The
independent-unit theorems above do not establish validity for every
clustered or pooled design.

Choosing the window and intervention
------------------------------------

Longer histories change both the scientific question and the number of units
following the target path. We can shorten the intervention window, pool
calendar windows, or repeat the fit at different final periods. Each choice
changes the target or introduces an additional modeling restriction.

With ``histories_length`` set, each length :math:`h` uses the last
:math:`h` entries of the supplied paths and leaves earlier observed
assignments outside that intervention window. The intended contrast is

.. math::

   \begin{aligned}
   &\mathbb E[Y_{i,T}(D_{i,1:(T-h)},d_{(T-h+1):T})]\\
   &\quad-\mathbb E[Y_{i,T}(D_{i,1:(T-h)},d'_{(T-h+1):T})].
   \end{aligned}

Its identifying conditions must hold for the histories and covariates supplied
at the new window's baseline. ``impulse_response=True`` uses
:math:`(1,0,\ldots,0)` versus :math:`(0,\ldots,0)` at each requested length
to measure a temporary intervention's later total effect. ``final_periods``
repeats the comparison at specified calendar endpoints rather than fixing the
outcome period throughout.

With ``pooled=True``, every complete window of ``len(ds1)`` periods ending
at or after ``initial_period`` enters as a separate unit history. Pooling
shares a projection across these windows under a model such as

.. math::

   Y_{i,t}(d_{1:t})
   =\beta_0+\beta_1d_t+\beta_2Y_{i,t-1}(d_{1:(t-1)})
      +X_{i,t}(d_{1:(t-1)})\gamma+\tau_t+\varepsilon_{i,t}.

While the treatment and history coefficients are common across calendar
periods, passing the time column to ``fixed_effects`` adds the calendar
dummies :math:`\tau_t`. Since a unit can contribute multiple overlapping
windows, the package clusters on ``idname`` by default when pooling. That
default inherits the limitations of the clustered variance described above.
Compare the treatment paths, achieved balance, and weight concentration across
your chosen windows before interpreting changes in estimates as changes in
treatment effects. The
:ref:`dynamic covariate balancing example <example_dyn_balancing>` compares
longer histories and stacked earlier windows on the democracy data.
