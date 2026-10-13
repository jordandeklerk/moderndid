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
differ. The :doc:`data guide <data>` explains how to prepare the inputs your
method needs. The :ref:`background guides <background>` state the
assumptions behind each method and explain what they identify.

Treatment that starts once
--------------------------

When a policy takes effect in different years across units and remains in
place afterward, an adoption cohort is the set of units first treated in the
same period. Several estimators keep each cohort's effects separate so that
an earlier cohort's changing response never enters another cohort's
untreated comparison. The choice between them depends on the model you want
for untreated outcomes and the flexibility you need in adjusting for
covariates.

Group-time effects with :func:`~moderndid.att_gt`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~moderndid.att_gt` estimates an average treatment effect for each
adoption cohort in each period. You identify the outcome, period, unit, and
first treatment period through ``yname``, ``tname``, ``idname``, and
``gname``. Both panel data and repeated cross-sections are supported
through the ``panel`` argument.

.. admonition:: Use ``att_gt`` for treatment that stays on
   :class: important

   Treatment must be binary and absorbing, in the sense that a unit stays
   treated in every period after it first receives treatment. The estimators
   under `Treatment that changes over time`_ handle treatment that switches
   off or changes in intensity.

Never-treated units provide the default untreated comparisons for every
adoption cohort. You can also use future adopters with
``control_group="notyettreated"`` as long as they remain untreated and
outside any anticipation window. Either choice
requires parallel trends for the units it compares, possibly conditional
on covariates in ``xformla``. The default ``est_method="dr"`` combines
outcome regression and propensity score weighting. With ``"reg"`` or
``"ipw"``, the estimator uses only the regression or only the weighting.
After you've followed the :doc:`first analysis <quickstart>`, the
:ref:`staggered DiD example <example_staggered_did>` examines these choices on
the minimum wage data. The :doc:`staggered DiD background <../background/did>`
states the assumptions behind each comparison group and derives the
regression, weighting, and doubly robust estimators.

:func:`~moderndid.aggte` averages the fitted effects by exposure length,
cohort, or calendar period, or into an overall average. These summaries
answer different questions about the effects you have already estimated.
The :doc:`results guide <results>` explains their weights and how to read
their uncertainty.

Regression models with :func:`~moderndid.etwfe`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to specify the comparison through a regression,
:func:`~moderndid.etwfe` includes a separate treatment indicator for each
adoption cohort in each treated period. Its linear model therefore allows
effects to differ across cohorts and over time. After fitting the model,
:func:`~moderndid.emfx` computes the summary you want through ``type="simple"``,
``"event"``, ``"group"``, or ``"calendar"``.

This approach also supports ``family="poisson"``, ``"logit"``, and
``"probit"`` when the outcome calls for a nonlinear model. You choose the
comparison group with ``cgroup="notyet"`` or ``"never"`` and the covariance
estimate with ``vcov`` rather than the bootstrap arguments of ``att_gt``. The
:ref:`extended TWFE example <example_etwfe>` compares the regression choices
on the minimum wage data. For the assumptions behind each model, the
:doc:`ETWFE background <../background/etwfe>` explains the linear and
nonlinear specifications.

.. admonition:: Nonlinear families change parallel trends
   :class: warning

   With ``family="poisson"``, untreated means must grow by common factors
   across cohorts rather than by common amounts. The ``"logit"`` family
   instead asks for a common change in the log-odds. Choose the family whose
   version of parallel trends you can defend for your outcome.

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
trends and overlap assumptions. The conditional effects you use to study
heterogeneity rely on them too.

Since the default bootstrap refits every cell in each of its 100 draws, a
``didml`` fit takes about 100 times as long as one with ``boot=False``. The
faster influence-function standard errors of ``boot=False`` treat the
cross-fitting split as fixed and can understate the sampling variability.
With ``cband=True``, the bootstrap draws also give a simultaneous band over
the group-time effects. The event study from :func:`~moderndid.aggte_didml`
takes its band from the same draws.

.. admonition:: Machine learning standard errors ignore clusters
   :class: warning

   ``didml`` ignores ``clustervars`` with a warning and treats units as
   independent. If your policy is assigned to groups of units, its standard
   errors and bands can come out too narrow.

A single two-period comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your data contain one period before treatment and one after it,
:func:`~moderndid.drdid` estimates a single average treatment effect on the
treated without constructing a staggered-adoption summary. The same function
supports either panel data or repeated cross-sections and uses ``treatname``
to identify the treatment group instead of an adoption-year column.

The default improved doubly robust estimator, ``est_method="imp"``, fits the
propensity score by inverse probability tilting and the comparison group's
outcome regressions by weighted least squares. With repeated cross-sections
it also regresses the treated group's outcome in each period.
:func:`~moderndid.ipwdid` and :func:`~moderndid.ordid` provide the weighting
and regression approaches separately. The :ref:`two-period API <api-drdid>`
describes these wrappers and their estimators for array inputs. The
:doc:`two-period background <../background/drdid>` explains what double
robustness requires and how panel and cross-section inference differ.

.. admonition:: Local efficiency under either design
   :class: note

   Like the traditional ``est_method="trad"``, the default reaches the
   semiparametric efficiency bound with panel data and repeated cross-sections
   alike whenever the propensity score and outcome models are correctly
   specified.

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
each exposure length. A level effect compares the outcomes of units that
received a particular dose with their outcomes under no treatment. The
:doc:`continuous treatment background <../background/didcont>` explains how
level effects differ from causal responses to the dose before the
:ref:`continuous treatment example <example_cont_did>` applies that
distinction to geological exposure to fracking.

.. admonition:: Dose slopes need stronger assumptions
   :class: warning

   Interpreting differences or slopes across doses as causal responses
   requires additional assumptions about how units select their doses.

The implementation currently requires a balanced panel and does not support
covariates, sampling weights, or clustered inference. The B-spline
method always computes its standard errors with a bootstrap.
The data-driven ``dose_est_method="cck"`` option cannot produce an event
study and requires two periods, a single treated cohort, and four or more
distinct treated doses. Check
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
The function handles two or multiple periods in panels and repeated
cross-sections. With multiple periods, :func:`~moderndid.agg_ddd` averages the
group-time results into event-study, cohort, calendar-time, or overall
summaries. The :ref:`triple differences example <example_triple_did>` shows
how the insurance application supplies this partition under the identifying
assumption developed in :doc:`../background/tripledid`.

.. admonition:: Argue parallel trends for the eligibility gap
   :class: important

   Triple differences require parallel trends in the eligible-ineligible
   outcome gap across treatment-enabling groups, conditional on the specified
   covariates.

Treatment that changes over time
--------------------------------

Some treatments can switch off, increase, or decrease after their first change.
Since the subsequent history can affect the outcome, a single adoption year
no longer describes the exposure you want to study. Two approaches in ModernDiD
address different questions about these histories and rely on different
identifying assumptions.

Effects of changes with :func:`~moderndid.did_multiplegt`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~moderndid.did_multiplegt` estimates event-study effects after a
group's first change in treatment. Its panel treatment column ``dname`` can
be binary or take multiple nonnegative values that change over time.
Switchers are compared with groups that had the same first-period treatment
and have not changed yet. These comparisons rely on parallel trends and no
anticipation and stay valid when lagged treatments affect the outcome.

The arguments ``effects`` and ``placebo`` set the requested post-change and
pre-change horizons. With ``normalized=True``, each estimate is divided by
the average cumulative change in treatment and becomes a weighted average of
the effects of current and lagged treatment. By default, periods after a
group's treatment has been both above and below its first-period value are
excluded because keeping them can introduce negative weights. The
:ref:`intertemporal treatment example <example_inter_did>` follows bank
branching deregulation through the comparisons and normalization derived
in :doc:`../background/didinter`.

.. admonition:: Bootstrap with a continuous baseline
   :class: tip

   When ``continuous`` adds a polynomial in a continuous baseline treatment,
   the analytical variance is not backed by proven asymptotic normality.
   :func:`~moderndid.did_multiplegt` warns you in that case and recommends
   ``boot=True`` for inference.

Comparing histories with :func:`~moderndid.diddynamic.dyn_balancing`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~moderndid.diddynamic.dyn_balancing` compares potential outcomes under
two binary treatment histories supplied as ``ds1`` and ``ds2``. It is intended for
settings where observed past outcomes and covariates help explain how units
select into treatment over time. Its default method constructs dynamic
balancing weights by solving a quadratic program. You can instead choose
inverse probability weighting with ``balancing="ipw"`` or its augmented form
with ``"aipw"``.

This approach relies on sequential conditional independence, overlap, and
restrictions on the outcome projections rather than a DiD parallel trends
assumption. The current implementation requires binary panel histories and
at least one covariate or fixed effect for the outcome projections and the
balancing. The :ref:`dynamic covariate balancing example
<example_dyn_balancing>` compares democracy histories and
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
on violations of parallel trends. The input needs influence functions and
consecutive event times on either side of its omitted reference period.

.. admonition:: Estimate with a universal base period
   :class: important

   For a ModernDiD event study, estimate with
   ``att_gt(base_period="universal")`` and aggregate with
   ``aggte(type="dynamic")`` before passing the result to ``honest_did``.
   An event study with any other base period raises an error.

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
