.. _estimator-overview:

Choosing an estimator
=====================

Choosing an estimator starts with your treatment history and the effect you
want to learn about. A method that compares adoption cohorts needs different
information from one that compares changing treatment histories or doses.
We'll use those distinctions to find the method that matches your design
before turning to its function arguments.

Start by checking whether treatment starts once and stays in place, whether
units receive different doses, and whether you need to account for treatment
changes later on. Although the functions share familiar names for outcomes
and periods, their inputs, identifying assumptions, and inference options
differ.
For the inputs your method needs, :doc:`data` explains how to prepare them
before estimation. The :ref:`background guides <background>` state the
assumptions behind each method and explain what they identify.

Treatment that starts once
--------------------------

When a policy takes effect in different years across units and remains in
place afterward, an adoption cohort is the set of units first treated in the
same period. Several estimators keep each cohort's effects separate so that
an earlier cohort's changing response does not become another cohort's
untreated comparison. The choice between them depends on the model you want
for untreated outcomes and the flexibility you need in adjusting for
covariates.

Group-time effects with :func:`~moderndid.att_gt`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~moderndid.att_gt` estimates an average treatment effect for each
adoption cohort in each period. You identify the outcome, period, unit, and
first treatment period through ``yname``, ``tname``, ``idname``, and
``gname``. Treatment must be binary and absorbing, meaning it stays in place
once it begins. Both panel data and repeated cross-sections are supported
through the ``panel`` argument.

Never-treated units provide the default untreated comparisons for every
adoption cohort. You can also use future adopters with
``control_group="notyettreated"`` as long as they remain untreated and
outside any anticipation window. Either choice
requires parallel trends for the units it compares, possibly conditional
on covariates in ``xformla``. The default ``est_method="dr"`` combines
outcome regression and propensity score weighting; ``"reg"`` and ``"ipw"``
use each approach separately. After you've followed the
:doc:`first analysis <quickstart>`, the
:ref:`staggered DiD example <example_staggered_did>` examines these choices on
the minimum wage data.

:func:`~moderndid.aggte` averages the fitted effects by exposure length,
cohort, or calendar period, or into an overall average. These summaries
answer different questions about the effects you have already estimated.
The :doc:`results`
guide explains their weights and how to read their uncertainty.

Regression models with :func:`~moderndid.etwfe`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to specify the comparison through a regression,
:func:`~moderndid.etwfe` includes a separate treatment indicator for each
adoption cohort in each treated period. Its linear model therefore allows
effects to differ across cohorts and over time. After fitting the model,
:func:`~moderndid.emfx` computes the summary you want through ``type="simple"``,
``"event"``, ``"group"``, or ``"calendar"``.

This approach also supports ``family="poisson"``, ``"logit"``, and
``"probit"`` when the outcome calls for a nonlinear model. Those choices
change the scale on which you model untreated trends. To interpret their
estimates causally, their assumptions need to suit your application.
Comparison groups use ``cgroup="notyet"`` or
``"never"``; covariance estimates use ``vcov`` rather than the bootstrap
arguments of ``att_gt``. The :ref:`extended TWFE example <example_etwfe>`
compares the regression choices on the minimum wage data. For the assumptions
behind each model, :doc:`../background/etwfe` explains the linear and nonlinear
specifications.

Flexible covariate adjustment with :func:`~moderndid.didml`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~moderndid.didml` estimates group-time effects using machine learning
models for the covariate adjustments. To calculate those adjustments, it
applies models trained on other parts of the sample, a procedure called
cross-fitting. Alongside those group-time effects, the fitted result includes
predictions of conditional treatment effects for individual units so you can
examine heterogeneity. You can use :func:`~moderndid.aggte_didml`
to produce a dynamic event study from its fitted results. The
:ref:`machine learning API <api-didml>` describes the model choices, aggregation,
and functions for examining heterogeneity.

The current implementation uses balanced panel data and drops incomplete
units. Flexible covariate adjustment still requires the design's parallel
trends and overlap assumptions, including when you use conditional effects
to study heterogeneity.

.. admonition:: Machine learning inference is currently pointwise
   :class: important

   ``didml`` does not yet support clustered standard errors or bootstrap
   simultaneous bands, even if ``clustervars`` or ``cband`` is supplied.
   These limits matter if your policy is assigned to groups of units or your
   conclusions rely on coverage over a whole event study.

A single two-period comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your data contain one period before treatment and one after it,
:func:`~moderndid.drdid` estimates a single average treatment effect on the
treated without constructing a staggered-adoption summary. The same function
supports either panel data or repeated cross-sections and uses ``treatname``
to identify the treatment group instead of an adoption-year column.

The default improved doubly robust method uses propensity score tilting and
weighted outcome regression. :func:`~moderndid.ipwdid` and
:func:`~moderndid.ordid` provide the weighting and regression approaches
separately. The :ref:`two-period API <api-drdid>` describes these wrappers
and their estimators for array inputs; :doc:`../background/drdid` explains
what double robustness requires and how panel and cross-section inference
differ.

Doses and eligibility
---------------------

An adoption year alone may leave out information essential to the research
question. A policy can give treated units different amounts of exposure or
apply only to eligible units within an adopting group. We need to retain
that information when choosing the estimator and defining its comparison.

Continuous treatment with :func:`~moderndid.cont_did`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~moderndid.cont_did` handles units that adopt once and receive a dose
that stays fixed after adoption. To link each unit's treatment timing to its
dose, name the adoption-date column with ``gname`` and the dose column with
``dname``. The estimator fits effects over the dose and can average them into
an event study.

With ``aggregation="dose"``, the result contains curves for level effects
and their slopes. With ``aggregation="eventstudy"``,
``target_parameter="level"`` or ``"slope"`` chooses what is averaged at
each exposure length. A level effect compares outcomes with no treatment
among units that received a particular dose. Interpreting differences or
slopes across doses as causal responses requires additional assumptions
about how units select their doses. The :doc:`continuous treatment background
<../background/didcont>` explains this distinction before the
:ref:`continuous treatment example <example_cont_did>` applies it to
geological exposure to fracking.

The implementation currently requires a balanced panel and supports neither
covariates nor sampling weights nor clustered inference. The B-spline
method always bootstraps its standard errors regardless of ``boot``.
The data-driven ``dose_est_method="cck"`` option requires two periods and
a single treated cohort and cannot produce an event study. Check
:func:`~moderndid.cont_did` for these restrictions before adapting a binary
DiD specification to doses.

Triple differences with :func:`~moderndid.ddd`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~moderndid.ddd` uses an eligibility partition to separate units that can
receive treatment within groups that enable it. You record when a group's
policy starts in the column named by ``gname`` and distinguish eligible and
ineligible units with ``pname``. Its target is the average effect among
eligible units in the treated group.

The additional comparison can account for local trends shared by eligible
and ineligible units even when an ordinary DiD comparison would fail.
It requires parallel trends in the eligible-ineligible outcome gap across
treatment-enabling groups, conditional on the specified covariates.
The function handles two or multiple periods in panels and repeated
cross-sections. In multiple periods, :func:`~moderndid.agg_ddd` produces
summaries from the group-time results. The :ref:`triple differences example
<example_triple_did>` shows how the insurance application supplies this
partition under the identifying assumption developed in
:doc:`../background/tripledid`.

Treatment that changes over time
--------------------------------

Some treatments can switch off, increase, or decrease after their first change.
Since the subsequent history can affect the outcome, a single adoption year
no longer describes the exposure you want to study. Two approaches in moderndid
address different questions about these histories and rely on different
identifying assumptions.

Effects of changes with :func:`~moderndid.did_multiplegt`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~moderndid.did_multiplegt` estimates event-study effects after a
group's first change in treatment. Its panel treatment column ``dname`` can
be binary or take multiple nonnegative values that change over time.
Switchers are compared with groups that had the same first-period treatment
and have not changed yet. Validity relies on parallel trends and no
anticipation for these comparisons, including when lagged treatments affect
the outcome.

The arguments ``effects`` and ``placebo`` set the requested post-change and
pre-change horizons. With ``normalized=True``, the estimates are scaled by
the cumulative treatment change and average effects of current and lagged
treatment. By default, periods after a group's treatment has been both above
and below its first-period value are excluded because keeping them can
introduce negative weights. For a
continuous baseline treatment, the analytical variance lacks a proved
asymptotic normal approximation. The API recommends bootstrapping the estimates
for this continuous treatment specification.
The :ref:`intertemporal treatment example <example_inter_did>` follows bank
branching deregulation through the comparisons and normalization derived
in :doc:`../background/didinter`.

Comparing histories with :func:`~moderndid.diddynamic.dyn_balancing`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~moderndid.diddynamic.dyn_balancing` compares potential outcomes under two binary
treatment histories supplied as ``ds1`` and ``ds2``. It is intended for
settings where observed past outcomes and covariates help explain how units
select into treatment over time. The dynamic balancing method constructs
weights by solving a quadratic program; ``balancing="ipw"`` and ``"aipw"``
provide alternative estimators.

This approach relies on sequential conditional independence, overlap, and
restrictions on the outcome projections rather than a DiD parallel trends
assumption. The current implementation requires binary panel histories and at least one
covariate or fixed effect for the outcome projections and the balancing. The :ref:`dynamic covariate
balancing example <example_dyn_balancing>` compares democracy histories and
economic growth under the identifying assumptions developed in
:doc:`../background/diddynamic`.

Sensitivity and instrumental variables
---------------------------------------

After estimating a DiD event study, you may want to examine how much your
conclusions depend on parallel trends. If your design instead identifies a
structural relationship through an instrument, the package also provides an
estimator for that separate problem.

Relaxing parallel trends with :func:`~moderndid.honest_did`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~moderndid.honest_did` computes confidence sets under specified bounds
on violations of parallel trends. For a moderndid event study, estimate with
``att_gt(base_period="universal")`` and aggregate with
``aggte(type="dynamic")`` before passing the result to ``honest_did``.
The input needs influence functions and consecutive event times on either
side of its omitted reference period.

Under a smoothness restriction, you bound how much the differential trend
can change from one period to the next. Relative
magnitude restrictions compare possible later violations with the deviations
observed before treatment. You choose the restrictions and their magnitudes
rather than asking the data to establish parallel trends. The
:ref:`sensitivity analysis example <example_honest_did>` examines both choices
on Medicaid expansion data and shows how to supply external estimates.
For the precise restrictions on differential trends, the
:doc:`sensitivity background <../background/didhonest>` states their formal
definitions and derives the confidence sets.

Structural functions with :func:`~moderndid.npiv`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~moderndid.npiv` estimates a nonparametric structural function and its
derivatives when a regressor is endogenous and suitable instruments are
available. You supply the outcome, endogenous regressor, and instrument
through ``yname``, ``xname``, and ``wname``, or pass arrays directly.
The method approximates the function with B-splines and estimates their
coefficients by instrumental variables.

Leaving ``j_x_segments`` unset selects the sieve dimension from the data
and constructs adaptive uniform confidence bands. Supplying a fixed
dimension requires the approximation bias to be small enough for its bands
to be valid. The :ref:`nonparametric IV example <example_npiv>` fits an Engel
curve under the moment condition and band construction explained in
:doc:`../background/npiv`. This standalone IV estimator also provides the
nonparametric estimation method used by the CCK option in ``cont_did``.
