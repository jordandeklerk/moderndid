.. _estimator-overview:

Choosing an estimator
=====================

Choosing an estimator starts with your treatment history and the effect you
want to learn about. A method that compares adoption cohorts needs different
information from one that compares changing treatment histories or doses.
We'll use those distinctions to find the method that matches your design
before turning to its function arguments.

The design map below lists each method with its function and the pages that
cover it. Once you've picked a method, the :doc:`data guide <data>` explains
how to prepare the inputs it needs.

Design map
----------

.. tab-set::

   .. tab-item:: Treatment that starts once

      .. list-table::
         :header-rows: 1

         * - Method
           - Function
           - Example
           - Background
         * - :ref:`Staggered DiD <overview-att-gt>`
           - :func:`~moderndid.att_gt`
           - :ref:`Example <example_staggered_did>`
           - :doc:`Background <../background/did>`
         * - :ref:`Extended TWFE <overview-etwfe>`
           - :func:`~moderndid.etwfe`
           - :ref:`Example <example_etwfe>`
           - :doc:`Background <../background/etwfe>`
         * - :ref:`Machine learning DiD <overview-didml>`
           - :func:`~moderndid.didml`
           -
           -
         * - :ref:`Two-period DiD <overview-drdid>`
           - :func:`~moderndid.drdid`, :func:`~moderndid.ipwdid`,
             :func:`~moderndid.ordid`
           -
           - :doc:`Background <../background/drdid>`

   .. tab-item:: Doses and eligibility

      .. list-table::
         :header-rows: 1

         * - Method
           - Function
           - Example
           - Background
         * - :ref:`Continuous treatment <overview-cont-did>`
           - :func:`~moderndid.cont_did`
           - :ref:`Example <example_cont_did>`,
             :ref:`Fracking paper replication <example_cont_did_replication>`
           - :doc:`Background <../background/didcont>`
         * - :ref:`Triple differences <overview-ddd>`
           - :func:`~moderndid.ddd`
           - :ref:`Example <example_triple_did>`
           - :doc:`Background <../background/tripledid>`

   .. tab-item:: Treatment that changes over time

      .. list-table::
         :header-rows: 1

         * - Method
           - Function
           - Example
           - Background
         * - :ref:`Intertemporal treatment <overview-did-multiplegt>`
           - :func:`~moderndid.did_multiplegt`
           - :ref:`Example <example_inter_did>`
           - :doc:`Background <../background/didinter>`
         * - :ref:`Dynamic covariate balancing <overview-dyn-balancing>`
           - :func:`~moderndid.diddynamic.dyn_balancing`
           - :ref:`Example <example_dyn_balancing>`
           - :doc:`Background <../background/diddynamic>`

   .. tab-item:: Sensitivity and instrumental variables

      .. list-table::
         :header-rows: 1

         * - Method
           - Function
           - Example
           - Background
         * - :ref:`Sensitivity analysis <overview-honest-did>`
           - :func:`~moderndid.honest_did`
           - :ref:`Example <example_honest_did>`
           - :doc:`Background <../background/didhonest>`
         * - :ref:`Nonparametric IV <overview-npiv>`
           - :func:`~moderndid.npiv`
           - :ref:`Example <example_npiv>`
           - :doc:`Background <../background/npiv>`

Treatment that starts once
--------------------------

When a policy takes effect in different years across units and remains in
place afterward, an adoption cohort is the set of units first treated in the
same period. Several estimators keep each cohort's effects separate so that
an earlier cohort's changing response never enters another cohort's
untreated comparison. The choice between them depends on the model you want
for untreated outcomes and the flexibility you need in adjusting for
covariates.

.. _overview-att-gt:

Group-time effects with :func:`~moderndid.att_gt`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~moderndid.att_gt` estimates an average treatment effect for each
adoption cohort in each period. Both panel data and repeated cross-sections
are supported through the ``panel`` argument.

.. admonition:: Use ``att_gt`` for treatment that stays on
   :class: important

   Treatment must be binary and absorbing, in the sense that a unit stays
   treated in every period after it first receives treatment. The estimators
   under `Treatment that changes over time`_ handle treatment that switches
   off or changes in intensity.

Never-treated units provide the default untreated comparisons for every
adoption cohort. You can also use future adopters with
``control_group="notyettreated"`` as long as they remain untreated and
outside any anticipation window. Either choice requires parallel trends for
the units it compares, possibly conditional on covariates in ``xformla``. The
default ``est_method="dr"`` combines outcome regression and propensity score
weighting. With ``"reg"`` or ``"ipw"``, the estimator uses only the regression
or only the weighting. You can see these choices at work on the minimum wage
data in the :ref:`staggered DiD example <example_staggered_did>` and find the
assumptions behind each one in the
:doc:`staggered DiD background <../background/did>`.

The :doc:`results guide <results>` explains how :func:`~moderndid.aggte`
averages the fitted effects into event-study, cohort, calendar, or overall
summaries.

.. _overview-etwfe:

Regression models with :func:`~moderndid.etwfe`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to specify the comparison through a regression,
:func:`~moderndid.etwfe` includes a separate treatment indicator for each
adoption cohort in each treated period. Its linear model therefore allows
effects to differ across cohorts and over time. After fitting the model,
:func:`~moderndid.emfx` computes the summary you want through ``type="simple"``,
``"event"``, ``"group"``, or ``"calendar"``.

This approach also supports ``family="poisson"``, ``"logit"``, and ``"probit"``
when the outcome calls for a nonlinear model. The
:ref:`extended TWFE example <example_etwfe>` compares the regression choices
on the minimum wage data under the assumptions that the
:doc:`ETWFE background <../background/etwfe>` states for each linear and
nonlinear model.

.. admonition:: Nonlinear families change parallel trends
   :class: warning

   With ``family="poisson"``, untreated means must grow by common factors
   across cohorts rather than by common amounts. The ``"logit"`` family
   instead asks for a common change in the log-odds. Choose the family whose
   version of parallel trends you can defend for your outcome.

.. _overview-didml:

Flexible covariate adjustment with :func:`~moderndid.didml`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~moderndid.didml` estimates group-time effects using machine learning
models for the covariate adjustments. To calculate those adjustments, it
applies models trained on other parts of the sample, a procedure called
cross-fitting. Its result also predicts a conditional treatment effect for
each unit so you can examine heterogeneity. You can use
:func:`~moderndid.aggte_didml` to produce a dynamic event study from its
fitted results. The :ref:`machine learning API <api-didml>` describes the
model choices, aggregation, and functions for examining heterogeneity.

Flexible covariate adjustment still requires the design's parallel trends and
overlap assumptions. The conditional effects you use to study heterogeneity
rely on them too. The current implementation uses balanced panel data and
drops incomplete units.

.. admonition:: Machine learning standard errors ignore clusters
   :class: warning

   ``didml`` ignores ``clustervars`` with a warning and treats units as
   independent. If your policy is assigned to groups of units, its standard
   errors and bands can come out too narrow.

.. _overview-drdid:

A single two-period comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your data contain one period before treatment and one after it,
:func:`~moderndid.drdid` estimates a single average treatment effect on the
treated without constructing a staggered-adoption summary. The same function
supports either panel data or repeated cross-sections and uses ``treatname``
to identify the treatment group instead of an adoption-year column.

Under conditional parallel trends and overlap, the default doubly robust
estimate remains consistent if either the propensity score model or the
outcome model is correct. :func:`~moderndid.ipwdid` and :func:`~moderndid.ordid`
provide the weighting and regression approaches separately. The
:doc:`two-period background <../background/drdid>` derives these estimators
and explains how panel and cross-section inference differ.

Doses and eligibility
---------------------

An adoption year alone may leave out information essential to the research
question. A policy can give treated units different amounts of exposure or
apply only to eligible units within an adopting group. We need to retain
that information when choosing the estimator and defining its comparison.

.. _overview-cont-did:

Continuous treatment with :func:`~moderndid.cont_did`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~moderndid.cont_did` handles units that adopt once and receive a dose
that stays fixed after adoption. To link each unit's treatment timing to its
dose, name the adoption-date column with ``gname`` and the dose column with
``dname``. The estimator fits level effects and their slopes over the dose
and can average either one into an event study.

A level effect compares the outcomes of units that received a particular dose
with their outcomes under no treatment. The
:doc:`continuous treatment background <../background/didcont>` explains how
level effects differ from causal responses to the dose before the
:ref:`continuous treatment example <example_cont_did>` applies that
distinction to geological exposure to fracking.

.. admonition:: Dose slopes need stronger assumptions
   :class: warning

   Although each level effect needs only parallel trends, interpreting
   differences or slopes across doses as causal responses requires additional
   assumptions about how units select their doses.

Before adapting a binary DiD specification to doses, check that your design
fits the current restrictions of :func:`~moderndid.cont_did`. It requires a
balanced panel and does not support covariates, sampling weights, or
clustered inference. The B-spline method always computes its standard errors
with a bootstrap. The data-driven ``dose_est_method="cck"`` option cannot
produce an event study and requires two periods, a single treated cohort, and
four or more distinct treated doses.

.. _overview-ddd:

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
how weather insurance sold only to the tobacco farmers of selected counties
supplies this partition under the identifying assumption developed in
:doc:`../background/tripledid`.

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

.. _overview-did-multiplegt:

Effects of changes with :func:`~moderndid.did_multiplegt`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~moderndid.did_multiplegt` estimates event-study effects after a
group's first change in treatment. Its panel treatment column ``dname`` can
be binary or take multiple nonnegative values that change over time.
Switchers are compared with groups that had the same first-period treatment
and have not changed yet. These comparisons rely on parallel trends and no
anticipation and stay valid when lagged treatments affect the outcome.

With ``normalized=True``, each estimate is divided by the average cumulative
change in treatment and becomes a weighted average of the effects of current
and lagged treatment. The
:ref:`intertemporal treatment example <example_inter_did>` follows bank
branching deregulation through the comparisons and normalization derived
in :doc:`../background/didinter`.

.. _overview-dyn-balancing:

Comparing histories with :func:`~moderndid.diddynamic.dyn_balancing`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~moderndid.diddynamic.dyn_balancing` compares potential outcomes under
two binary treatment histories supplied as ``ds1`` and ``ds2``. It is intended for
settings where observed past outcomes and covariates help explain how units
select into treatment over time.

This approach relies on sequential conditional independence, overlap, and
restrictions on the outcome projections rather than a DiD parallel trends
assumption. The current implementation requires binary panel histories and
at least one covariate or fixed effect for the outcome projections and the
balancing. The :ref:`dynamic covariate balancing example
<example_dyn_balancing>` compares economic growth across democracy histories
under the identifying assumptions developed in
:doc:`../background/diddynamic`.

Sensitivity and instrumental variables
---------------------------------------

After estimating a DiD event study, you may want to examine how much your
conclusions depend on parallel trends. If your design instead identifies a
structural relationship through an instrument, the package also provides an
estimator for that separate problem.

.. _overview-honest-did:

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
rather than asking the data to establish parallel trends. Both restrictions
are defined formally in the
:doc:`sensitivity background <../background/didhonest>` and applied to
Medicaid expansion data in the
:ref:`sensitivity analysis example <example_honest_did>`.

If your event study comes from a regression or another estimator, the
:ref:`sensitivity analysis example <example_honest_did_external>` passes its
coefficients and covariance matrix to the functions behind
:func:`~moderndid.honest_did`.

.. _overview-npiv:

Structural functions with :func:`~moderndid.npiv`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~moderndid.npiv` estimates a nonparametric structural function and its
derivatives when a regressor is endogenous and suitable instruments are
available. You supply the outcome, endogenous regressor, and instrument
through ``yname``, ``xname``, and ``wname``, or pass arrays directly.
The method approximates the function with B-splines and estimates their
coefficients by instrumental variables. The
:ref:`nonparametric IV example <example_npiv>` fits an Engel curve under the
moment condition and band construction explained in
:doc:`../background/npiv`.
