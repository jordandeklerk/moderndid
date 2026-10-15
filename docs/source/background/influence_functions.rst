.. _background-influence-functions:

Influence functions
===================

Many of the background guides you'll find here work with influence functions. They compute standard
errors from them, combine effects with them, and perturb them to draw the multiplier bootstrap
behind simultaneous bands. However, none of those guides really stop to explain what an influence
function is. We recommend reading this page first if these ideas are new to you and skipping it if
they're not.

In a large sample, an estimator's error behaves like an average of contributions, one from each
unit. Starting from the simplest DiD comparison, the page shows how standard errors, averages of
effects, and bootstrap draws follow from those contributions and why the doubly robust estimators in
these guides take the form they do. The presentation relies heavily on the lecture note on
semiparametric estimators by
`Chen (2024) <https://faculty.washington.edu/yenchic/short_note/note_EIF.pdf>`_.

Each unit's contribution to the error
-------------------------------------

Suppose you observe :math:`n` independent units whose data :math:`W_1,\ldots,W_n` come from a common
distribution :math:`P`. The target :math:`\theta` you're after, such as an ATT or a vector of
group-time effects, is a feature of that distribution that an estimator :math:`\widehat\theta`
approximates from the sample. The estimator is asymptotically linear with influence function
:math:`\psi` when

.. math::

   \sqrt n(\widehat\theta-\theta)
   =\frac{1}{\sqrt n}\sum_{i=1}^n\psi(W_i)+o_p(1),
   \qquad
   \mathbb{E}[\psi(W)]=0,

and :math:`\mathbb{E}[\psi(W)\psi(W)']` is finite. For a vector target, :math:`\psi` is a vector of
the same length.

The :math:`\sqrt n` in front keeps an error of order :math:`n^{-1/2}` at a stable scale as the sample
grows. In large samples you can set aside the :math:`o_p(1)` term, a remainder that converges to
zero in probability. What remains says that the estimation error is the sample average of the
contributions :math:`\psi(W_1),\ldots,\psi(W_n)`, one from each unit. Everything a large-sample
analysis needs is therefore in their distribution.

The name comes from robust statistics, where `Hampel (1974)
<https://doi.org/10.1080/01621459.1974.10482962>`_ used the influence curve to measure how strongly a
single observation at :math:`w` pulls on an estimate. The same reading holds for the estimators on
these pages. If you drop unit :math:`i` from the sample and recompute, the estimate moves by about
:math:`-\psi(W_i)/n`. On the county data of the :ref:`minimum wage example <example_staggered_did>`,
refitting :func:`~moderndid.att_gt` without one county at a time gives changes that match this
prediction to within a few percent.

For the sample mean of :math:`Y_1,\ldots,Y_n`, the expansion holds exactly with
:math:`\psi(W)=Y-\mu` for :math:`\mu=\mathbb{E}[Y]` and leaves no remainder. Since dropping
:math:`Y_i` moves the mean by exactly :math:`(\bar Y-Y_i)/(n-1)`, you can check the leave-one-out
reading by hand.

Now take the simplest DiD comparison, a panel without covariates that records each unit's outcome
once before and once after treatment. Let :math:`D` indicate the treated group and :math:`\Delta Y`
the outcome change. Write :math:`q=\mathbb{P}(D=1)` for the treated share,
:math:`\mu_d=\mathbb{E}[\Delta Y\mid D=d]` for the mean change in group :math:`d`, and
:math:`\mathbb{E}_n[Z]=n^{-1}\sum_{i=1}^nZ_i` for a sample average. Under parallel trends, the
difference between the two groups' sample mean changes estimates the ATT
:math:`\theta=\mu_1-\mu_0`. If you subtract each group's population mean inside its own average, you
get the exact identity

.. math::

   \widehat\theta-\theta
   =\frac{\mathbb{E}_n[D(\Delta Y-\mu_1)]}{\mathbb{E}_n[D]}
   -\frac{\mathbb{E}_n[(1-D)(\Delta Y-\mu_0)]}{\mathbb{E}_n[1-D]}.

Each numerator averages mean-zero terms and is of order :math:`n^{-1/2}`. Since replacing the
denominators by their limits :math:`q` and :math:`1-q` changes the error only at a smaller order,
the influence function is

.. math::

   \psi(W)=\frac{D}{q}(\Delta Y-\mu_1)-\frac{1-D}{1-q}(\Delta Y-\mu_0).

You can read the two terms one group at a time. A treated unit contributes its deviation from the
treated mean change divided by the treated share. A comparison unit contributes minus its deviation
from the comparison mean change divided by the comparison share. Units from the smaller group
therefore carry more influence each. With :math:`\sigma_d^2` the variance of :math:`\Delta Y` in
group :math:`d`, the variance of the influence function divided by :math:`n` recovers a familiar
formula,

.. math::

   \frac{\mathbb{E}[\psi(W)^2]}{n}
   =\frac{\sigma_1^2}{nq}+\frac{\sigma_0^2}{n(1-q)}.

It's the usual two-sample variance of a difference in means between :math:`nq` treated units and
:math:`n(1-q)` comparison units.

One target can have many asymptotically linear estimators whose influence functions differ. An
outcome regression estimator, an inverse probability weighting estimator, and a doubly robust
estimator of the same ATT each have their own. We come back to which of them gives the smallest
variance in the :ref:`last section <background-influence-functions-efficiency>`.

.. admonition:: Notation across the guides
   :class: note

   The staggered and triple differences guides write influence functions as :math:`\psi` and
   those of first-step fits as :math:`\ell`. The doubly robust guide uses :math:`\phi` for its
   estimators and :math:`\eta` for the efficient influence function. Since :math:`\psi^K` denotes
   the spline basis in the continuous treatment guide, that guide also uses :math:`\phi`.

.. _background-influence-functions-inference:

From contributions to standard errors
-------------------------------------

Once you see the leading error as a sample average of independent mean-zero contributions, the
central limit theorem gives its large-sample distribution,

.. math::

   \sqrt n(\widehat\theta-\theta)\xrightarrow{d}N(0,\Sigma),
   \qquad
   \Sigma=\mathbb{E}[\psi(W)\psi(W)'].

This is the reasoning behind the familiar :math:`\sigma/\sqrt n` for a sample mean, applied to the
contributions instead of the outcomes. Since you don't know :math:`\psi` itself, the estimated
influence function :math:`\widehat\psi` plugs in estimates of the means, shares, and model limits
inside it. Its sample second moment then gives an estimate of :math:`\Sigma` and the standard error
of each component,

.. math::

   \widehat\Sigma=\frac1n\sum_{i=1}^n\widehat\psi(W_i)\widehat\psi(W_i)',
   \qquad
   \widehat{se}(\widehat\theta_j)=\sqrt{\widehat\Sigma_{jj}/n}
   =\frac1n\Bigl(\sum_{i=1}^n\widehat\psi_j(W_i)^2\Bigr)^{1/2}.

The results of :func:`~moderndid.att_gt` and :func:`~moderndid.aggte` already store
:math:`\widehat\psi` for you, in an ``influence_func`` array with one column per effect and one row
per unit, or per observation for repeated cross-sections. Without the bootstrap or clustering, the
standard errors they report are exactly this formula.

Covariances and averages
~~~~~~~~~~~~~~~~~~~~~~~~

When group-time effects share comparison units, their errors are correlated. The off-diagonal
entries of :math:`\widehat\Sigma` measure that correlation directly because each unit contributes to
several columns of :math:`\widehat\psi`. If you average the effects with weights :math:`a` fixed in
advance, the average :math:`a'\widehat\theta` has influence function :math:`a'\psi`. Its standard
error comes from the same formula without any new derivation.

The leave-one-out reading already treats :math:`\psi(W_i)` as a derivative, the rate at which the
estimate responds as unit :math:`i`'s weight changes. You can then read asymptotic linearity as a
first-order Taylor expansion from the true distribution to the sample's in which :math:`\psi` plays
the role of the derivative. Influence functions therefore follow the chain rule and product rule of
ordinary derivatives. For a smooth function :math:`f`, the estimate :math:`f(\widehat\theta)` has
influence function :math:`\nabla f(\theta)'\psi`. This is the delta method applied one unit at a
time. The product rule comes in when the weights themselves are estimated, such as the cohort
shares behind an event study. That average sums products of estimated weights and estimated
effects. Its influence function therefore adds a term for the estimated weights to
:math:`a'\psi`. The :ref:`staggered DiD guide <background-did-aggregation-weights>` works out that
term for the aggregations in :func:`~moderndid.aggte`.

.. admonition:: Standard errors for your own contrasts
   :class: tip

   For a comparison that :func:`~moderndid.aggte` doesn't report, multiply the ``influence_func``
   array by your fixed weights :math:`a` to get one contribution per row. Square the contributions,
   sum them, take the square root, and divide by :math:`n`, the number of rows in the array. With
   clustering, sum the contributions within each cluster before squaring and still divide by
   :math:`n`.

Clustered sampling
~~~~~~~~~~~~~~~~~~

Everything so far treats the units as independent, an assumption about how your data were sampled.
If counties in the same state share shocks, their contributions are correlated. A sum of squares
over units then misses that dependence. Summing the contributions within each cluster :math:`c` and
treating the cluster sums as independent gives the cluster-robust covariance of
:math:`\widehat\theta`,

.. math::

   \widehat\Sigma_{cl}
   =\frac{1}{n^2}\sum_c
      \Bigl(\sum_{i\in c}\widehat\psi(W_i)\Bigr)
      \Bigl(\sum_{i\in c}\widehat\psi(W_i)\Bigr)'.

With one unit per cluster, you're back to the unclustered estimate :math:`\widehat\Sigma/n`. Without
the bootstrap, :func:`~moderndid.att_gt` and :func:`~moderndid.aggte` compute their clustered
standard errors from this formula. What matters for the approximation is the number of independent
clusters rather than the number of units. The :ref:`staggered DiD guide <background-did-bootstrap>`
discusses when your design has enough clusters for it.

The multiplier bootstrap
~~~~~~~~~~~~~~~~~~~~~~~~

Having every unit's contribution on hand also makes the bootstrap cheap. Instead of refitting the
models, each draw multiplies every unit's contribution by an independent random weight :math:`V_i`
with mean zero and variance one,

.. math::

   \widehat\theta^{*}=\widehat\theta+\frac1n\sum_{i=1}^nV_i\widehat\psi(W_i).

Each draw gives every unit the weight :math:`1+V_i` in place of one. By the leave-one-out reading,
that change moves the estimate by about :math:`V_i\widehat\psi(W_i)/n` for each unit. With
multipliers of plus or minus one, as :func:`~moderndid.att_gt` draws them, each unit either counts
twice or drops out. A new sample shifts the estimate in much the same way by including some kinds of
units more often and others less often. Given the data, the perturbation has covariance
:math:`\widehat\Sigma/n`, the same as the analytic estimate. Reusing the same :math:`V_i` for every
effect of a unit preserves the covariance across effects. Drawing one multiplier per cluster instead
of per unit reproduces the clustered covariance. Since a simultaneous band's critical value depends
on the joint distribution of every effect it covers, that's where the draws matter most. The
:ref:`staggered DiD guide <background-did-bootstrap>` walks through the algorithm and shows how the
package turns the draws into standard errors and critical values.

.. _background-influence-functions-fitted:

Estimators built on fitted models
---------------------------------

So far, the influence functions have involved only means and shares. Most estimators in these
guides, though, average a score that depends on fitted nuisance models. A propensity score
:math:`p(X;\widehat\pi)` and an outcome regression :math:`m(X;\widehat\beta)` stand in for unknown
functions of the covariates. Any error in those fits passes into your estimate along with the
sampling error of the average itself.

To see what the fits add, stack the fitted parameters in :math:`\widehat\gamma` and write the
estimator as :math:`\widehat\theta=\mathbb{E}_n[h(W;\widehat\gamma)]` for a score :math:`h`. Under
the conditions the guides state, the fit converges to a population limit :math:`\gamma^*` whether
or not its model is correct. The score must identify the target at that limit,
:math:`\mathbb{E}[h(W;\gamma^*)]=\theta`. The guides also assume that the fit has an expansion of
its own,

.. math::

   \sqrt n(\widehat\gamma-\gamma^*)
   =\frac{1}{\sqrt n}\sum_{i=1}^n\ell_\gamma(W_i)+o_p(1).

With :math:`M=\mathbb{E}[\partial_\gamma h(W;\gamma^*)]` collecting the derivatives of the mean
score, moving the parameters from :math:`\gamma^*` to :math:`\widehat\gamma` changes the average
score by approximately :math:`M'(\widehat\gamma-\gamma^*)`. Substituting the expansion of
:math:`\widehat\gamma` into that approximation gives the influence function

.. math::

   \psi(W)=h(W;\gamma^*)-\theta+M'\ell_\gamma(W).

The first two terms are the contribution the estimate would have if :math:`\gamma^*` were known.
The last term, the estimation effect of the first step, carries the fitted parameters' own error
into the estimate. The derivative :math:`M` sets how much of that error gets through. Since both
errors come from the same units, the influence function adds them unit by unit instead of adding
their variances. The fitting term can therefore lower the variance as well as raise it. Estimated
normalizing constants, such as the mean of a weight, enter the same way and appear in the guides as
centering terms. In the first section's example, estimating the treated share :math:`q` centers
each unit's change at its group's mean and takes the group means out of the variance. So whether
:math:`M` is zero decides whether fitting the models matters for the standard error.

In the doubly robust scores these guides use, the propensity and outcome parts of :math:`M` vanish
for different reasons. When the propensity model is correct, its weights balance every function of
the covariates between treated and comparison units. The derivative with respect to the outcome
regression is then zero. When the outcome model is correct, the comparison units' residuals have
conditional mean zero. The derivative with respect to the propensity parameters is then zero. With
both models correct, fitting adds nothing to the first-order error. With only one model correct,
though, the term for fitting the correct model generally remains.

That's why the :ref:`staggered DiD guide <background-did-fitting>` and the :ref:`triple differences
guide <background-tripledid-inference>` keep both terms in their complete influence functions. When
you compare their formulas, you'll find :math:`M` defined with the opposite sign and the fitting
terms subtracted. The improved estimators in the :ref:`doubly robust guide
<background-drdid-fitting>` take a different route. Their fitting methods have first-order
conditions that set both derivatives to zero even when one model is wrong.

.. admonition:: Estimated nuisances change the standard error
   :class: warning

   A standard error computed from :math:`h(W;\widehat\gamma)-\widehat\theta` alone treats the
   fitted models and the estimated normalizing constants as known. The centering terms from those
   constants belong in the influence function even when both models are correct. When one model
   is wrong, the very case double robustness exists to cover, a fitting term can belong there too
   unless the fitting methods remove it.

.. _background-influence-functions-efficiency:

The efficient influence function
--------------------------------

We can now return to the question the first section left open. Since estimators of the same target
can have different influence functions, their asymptotic variances :math:`\mathbb{E}[\psi(W)^2]` can
differ too. Efficiency theory asks how small that variance can get among regular estimators. An
estimator is regular when its scaled error :math:`\sqrt n(\widehat\theta-\theta)` keeps the same
limiting distribution under data-generating processes that approach :math:`P` at the
:math:`n^{-1/2}` rate. The error is measured against each process's own target :math:`\theta`.
Changes of that size sit at the edge of what a sample of :math:`n` units can detect. Regularity
rules out tricks like Hodges' estimator, an estimator that reports exactly zero whenever a standard
estimate is small. It looks unbeatable when the true effect is zero but does badly when the effect
is small and nonzero.

A lower bound exists because a flexible model contains many simple parametric ones. Take any
one-parameter family of distributions that passes through :math:`P` and stays inside the model.
Within that family, the classical Cramér-Rao bound limits how precisely a regular estimator can
recover :math:`\theta`. An estimator that has to work across the whole model can do no better than
it could in the hardest of those families. The semiparametric efficiency bound is the largest
Cramér-Rao bound among these families. In the literature, the hardest family is called the least
favorable submodel.

Chen's note also gives a geometric picture of the influence functions that regular estimators can
have. Think of the tangent space of a model as the set of directions in which the data distribution
can move while staying inside the model. Each direction is a mean-zero function of the data called
the score of the move. It is positive where the move makes values of :math:`W` more likely and
negative where it makes them less likely. Two functions are orthogonal here when the mean of their
product is zero.

Since both must track the target the same way along every direction the model allows, the
influence functions of any two regular estimators differ only by a function orthogonal to the
tangent space. Projecting any of them onto the tangent space therefore gives the same function
:math:`\psi_{\mathrm{eff}}`, the efficient influence function. The part of :math:`\psi` left outside
is uncorrelated with every direction in which the model lets the distribution move. That part adds
variance without carrying any information about the target :math:`\theta`. Since the projection
and that part are orthogonal, the variance of a scalar estimator splits into two parts,

.. math::

   \mathbb{E}[\psi(W)^2]
   =\mathbb{E}[\psi_{\mathrm{eff}}(W)^2]
   +\mathbb{E}\bigl[(\psi(W)-\psi_{\mathrm{eff}}(W))^2\bigr]
   \geq\mathbb{E}[\psi_{\mathrm{eff}}(W)^2].

The lower bound is the semiparametric efficiency bound, the smallest asymptotic variance any regular
estimator can have. An estimator whose influence function equals :math:`\psi_{\mathrm{eff}}`
attains the bound. The least favorable submodel moves the distribution in the direction of
:math:`\psi_{\mathrm{eff}}` itself, where a shift that the data can barely detect moves the target
the most.

When the model places no restriction on the distribution of the observed data, the tangent space
contains every mean-zero function. Every regular estimator then shares the same influence function,
the efficient one. The difference in mean changes from the first section is such a case.
Estimators that rely on a parametric working model, such as the outcome regression and inverse
probability weighting estimators mentioned there, are regular only in the smaller model where that
working model holds. Their influence functions can therefore differ from the efficient one.

Differentiating the target
~~~~~~~~~~~~~~~~~~~~~~~~~~

In practice, you find the efficient influence function by differentiating the target. Write
:math:`\theta(P)` for the target as a function of the data distribution. Then move :math:`P` along a
smooth path :math:`P_t` inside the model whose score at :math:`t=0` is a mean-zero function
:math:`g`. In a model without restrictions, :math:`p_t(w)=p(w)\{1+tg(w)\}` with a bounded :math:`g`
is such a path. The influence function of every regular estimator represents the derivative of the
target along all such paths,

.. math::

   \frac{d}{dt}\theta(P_t)\Big|_{t=0}=\mathbb{E}[\psi(W)g(W)].

In words, a move that makes units with large values of :math:`\psi` more common raises the target.
The rate is the covariance between :math:`\psi` and the move's score. The efficient influence
function is the one representation that lies in the tangent space. In a model without restrictions
it is the only representation. Differentiating the target then gives you the efficient influence
function without first guessing an estimator. For intuition, let the path shift a little
probability toward a single point :math:`w`. The target then changes at the rate
:math:`\psi_{\mathrm{eff}}(w)` as the shift begins. That's Hampel's original definition of influence
and the derivative behind the leave-one-out reading in the first section. `Hines et al. (2022)
<https://doi.org/10.1080/00031305.2021.2021984>`_ use this point-mass calculation to derive
efficient influence functions by hand. In a restricted model, the derivative gives an influence
function that you still have to project onto the tangent space.

Missing outcomes and double robustness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We'll follow Chen's missing-data example because it ends at the doubly robust form these guides
keep using. Suppose an outcome :math:`Y` is observed only when :math:`R=1` but covariates :math:`X`
are always observed. The outcome is missing at random given the covariates, :math:`Y\perp R\mid X`.
With :math:`e(X)=\mathbb{P}(R=1\mid X)` bounded away from zero and
:math:`m(X)=\mathbb{E}[Y\mid X,R=1]`, the target :math:`\theta=\mathbb{E}[Y]` has two
representations,

.. math::

   \theta=\mathbb{E}\left[\frac{RY}{e(X)}\right]=\mathbb{E}[m(X)].

The first representation leads to inverse probability weighting and the second to regression
adjustment. Since both describe the same function of the observed-data distribution,
differentiating either one along a path gives the same derivative. Chen works through both
calculations and arrives at the same efficient influence function,

.. math::

   \psi_{\mathrm{eff}}(W)=\frac{R}{e(X)}\bigl(Y-m(X)\bigr)+m(X)-\theta.

You can read it as a regression prediction plus a correction. Every unit, observed or not, starts
from the regression's prediction :math:`m(X)`. Each observed unit then adds its residual
:math:`Y-m(X)` with weight :math:`1/e(X)` so that it also speaks for similar units whose outcomes
are missing.

A comparison with plain inverse probability weighting shows what this form gains. If you knew
:math:`e(X)`, the weighting estimator would have influence function

.. math::

   \psi_{\mathrm{IPW}}(W)=\frac{RY}{e(X)}-\theta
   =\psi_{\mathrm{eff}}(W)+\frac{m(X)\{R-e(X)\}}{e(X)}.

The extra term only records which units happened to respond. It is uncorrelated with
:math:`\psi_{\mathrm{eff}}` and adds :math:`\mathbb{E}[m(X)^2\{1-e(X)\}/e(X)]` to the variance.
When :math:`e(X)` is known, the model can't move the response probabilities. Its tangent space then
leaves out every function of the form :math:`c(X)\{R-e(X)\}`. Since the extra term has that form,
projecting :math:`\psi_{\mathrm{IPW}}` onto the tangent space removes it and leaves
:math:`\psi_{\mathrm{eff}}`. Fitting :math:`e(X)` instead adds an estimation effect that offsets
some or all of that term. That's why weighting with an estimated propensity score can be more
precise than weighting with the true one.

If you set the sample average of :math:`\psi_{\mathrm{eff}}` to zero with fitted :math:`\widehat e`
and :math:`\widehat m`, you get the doubly robust estimator,

.. math::

   \widehat\theta=\mathbb{E}_n\left[
      \frac{R}{\widehat e(X)}\bigl(Y-\widehat m(X)\bigr)+\widehat m(X)\right].

This estimator is also the regression plug-in :math:`\mathbb{E}_n[\widehat m(X)]` plus the average
of the weighted residuals. The plug-in alone carries the regression's error into the answer. The
residual average removes the first-order part of that error. `Fisher and Kennedy (2021)
<https://doi.org/10.1080/00031305.2020.1717620>`_ draw this correction as a step along a tangent
line from the fitted distribution back toward the true one. They compare the step with one step of
Newton's method. The same recipe of adding the average estimated influence function to a plug-in
estimate, known as the one-step estimator, works for other targets too.

To see where the robustness comes from, replace the true functions with working versions
:math:`\tilde e` and :math:`\tilde m`. The mean of the efficient score then misses the target by

.. math::

   \mathbb{E}\left[\frac{R}{\tilde e(X)}\bigl(Y-\tilde m(X)\bigr)+\tilde m(X)\right]-\theta
   =\mathbb{E}\left[\frac{\{e(X)-\tilde e(X)\}\{m(X)-\tilde m(X)\}}{\tilde e(X)}\right].

Because the gap is a product of the two functions' errors, it vanishes when either working function
is correct. The ATT estimators in the :ref:`doubly robust guide <background-drdid-combining>` stay
consistent for the same reason.

For the ATT, the missing quantity is the untreated outcome change of treated units, a change that
only comparison units reveal. Because the target averages over treated units rather than the whole
population, comparison units are weighted by the propensity odds :math:`p(X)/\{1-p(X)\}` rather
than by the inverse :math:`1/\{1-p(X)\}` of their probability of being a comparison unit.
Proposition 1 in the :ref:`doubly robust guide <background-drdid-efficiency>` gives the resulting
efficient influence function from `Sant'Anna and Zhao (2020)
<https://psantanna.com/files/SantAnna_Zhao_DRDID.pdf>`_.

Why fitted models can drop out
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Like the influence function of any regular estimator, the efficient influence function is
orthogonal to every direction in which the distribution can move without changing the target.
Moving a nuisance function while the target stays fixed is one of those directions. Much like a
smooth function near a point where its slope is zero, the mean of the efficient score then changes
only at second order when the nuisance functions plugged into it move away from their true values.
The product of errors above shows this for missing outcomes. The derivative :math:`M` from the
:ref:`previous section <background-influence-functions-fitted>` is therefore zero at the true
nuisance functions. Chen's Theorem 2 makes this precise in a parametric model. With nuisance
estimates that converge to the truth at the :math:`n^{-1/2}` rate, an estimator that solves an
influence function's estimating equation keeps that influence function.

That result explains why a doubly robust estimator built on the efficient influence function is
locally efficient. When both of its working models are correct, its influence function equals
:math:`\psi_{\mathrm{eff}}` and its variance attains the efficiency bound. A doubly robust estimator
built on another score, such as the first repeated cross-section score in the doubly robust guide,
can stay above the bound.

.. admonition:: Robustness and efficiency
   :class: note

   A doubly robust estimator stays consistent when one working model is wrong. Even one built on
   the efficient influence function reaches the efficiency bound only when both models are
   correct.

Even though machine learning fits converge more slowly than :math:`n^{-1/2}`, the same
orthogonality lets you use them as nuisance estimates. The second-order term stays negligible as
long as a product of errors like the one above shrinks faster than :math:`n^{-1/2}`. Two errors that
each shrink faster than :math:`n^{-1/4}` meet that requirement. That's a much lower bar than the
:math:`n^{-1/2}` rate of a correctly specified parametric model. A flexible model also partly fits
the noise of the units it was trained on. Since its residuals for those units come out too small,
the correction would remove too little of the plug-in's error. Cross-fitting, as in
`Chernozhukov et al. (2018) <https://doi.org/10.1111/ectj.12097>`_, predicts for each unit from
models trained on other units so that a unit's own noise never enters its own prediction.
:func:`~moderndid.didml` also cross-fits its machine learning nuisance models. Its doubly robust
score corrects their predictions with balancing weights in place of the propensity odds.

If you want to see these pieces work together, the :ref:`staggered DiD guide <background-did>`
assembles them for group-time effects, from the nuisance fits through aggregation to simultaneous
bands. For two-period designs, the :ref:`doubly robust DiD guide <background-drdid>` derives the
efficient influence functions and compares what panels and repeated cross-sections can achieve.
For the full theory, `Tsiatis (2006) <https://doi.org/10.1007/0-387-37345-4>`_ and chapter 25 of
`van der Vaart (1998) <https://doi.org/10.1017/CBO9780511802256>`_ are the standard references.
