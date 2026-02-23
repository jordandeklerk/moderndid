.. _estimator-overview:

==================
Estimator Overview
==================

ModernDiD provides several estimators for different research designs. All
estimators share a common API pattern, so once you learn one the others
follow naturally. This page provides an overview of each estimator, its key
arguments, and important caveats. For detailed usage with real data, see the
individual example pages.


Choosing the right estimator
-----------------------------

The choice of estimator depends on the structure of your treatment variable
and research question.

- Many applied settings involve a binary treatment that turns on permanently
  at staggered times across groups. :func:`~moderndid.att_gt` handles this
  staggered adoption case and is a good starting point for most analyses.
- When treatment intensity varies continuously across units,
  :func:`~moderndid.cont_did` extends the framework to recover
  dose-response functions showing how effects scale with dosage.
- When a policy enables treatment for a group but only a subset of units
  within that group is actually eligible, :func:`~moderndid.ddd` exploits
  this additional within-group variation. Parental leave affecting women but
  not men, or minimum wage affecting hourly but not salaried workers, are
  canonical examples.
- When treatment is not permanent and can switch on and off or change
  intensity over time, :func:`~moderndid.did_multiplegt` provides valid
  inference by comparing units whose treatment changed to those with the
  same baseline treatment that have not yet changed.
- After running any estimator, :func:`~moderndid.honest_did` assesses how
  large violations of the parallel trends assumption would need to be to
  overturn your conclusions.


Staggered Difference-in-Differences
-----------------------------------

The :func:`~moderndid.did.att_gt` function is the primary estimator for
staggered treatment adoption with binary, absorbing treatment. It estimates
group-time average treatment effects on the treated (ATT(g,t)) and
is the recommended starting point for most DiD analyses. This implements the
`Callaway and Sant'Anna (2021) <https://doi.org/10.1016/j.jeconom.2020.12.001>`_
framework.

.. code-block:: python

    result = did.att_gt(
        data=data,
        yname="outcome",
        tname="year",
        idname="unit_id",
        gname="first_treated",
        xformla="~ covariate",
        control_group="nevertreated",
        est_method="dr",
    )

The result includes group-time ATT estimates, analytical standard errors, a
variance-covariance matrix, and influence functions. A Wald pre-test for
parallel trends is computed automatically from pre-treatment periods.

Group-time estimates are typically aggregated into interpretable summary
parameters using :func:`~moderndid.aggte`:

.. code-block:: python

    # Event study (dynamic effects relative to treatment)
    event_study = did.aggte(result, type="dynamic")

    # Simple weighted average across all post-treatment (g,t) cells
    simple_agg = did.aggte(result, type="simple")

    # Group-level averages (one ATT per cohort)
    group_agg = did.aggte(result, type="group")

    # Calendar-time averages (one ATT per period)
    calendar_agg = did.aggte(result, type="calendar")

Clustered standard errors require ``boot=True``. When ``clustervars`` is
specified without the bootstrap, the reported standard errors do not account
for clustering. At most two clustering variables are supported.

When a Dask or Spark DataFrame is passed as ``data``, the estimator
automatically routes to a distributed implementation. See :doc:`distributed`
for configuration details.


Triple Difference-in-Differences
--------------------------------

The :func:`~moderndid.didtriple.ddd` function leverages an additional
dimension of variation such as eligibility status. The API follows
the same pattern as the other estimators.
This implements the
`Ortiz-Villavicencio and Sant'Anna (2025) <https://arxiv.org/abs/2505.09942>`_
framework.

.. code-block:: python

    result = did.ddd(
        data=data,
        yname="outcome",
        tname="year",
        idname="unit_id",
        gname="first_treated",
        pname="eligible",              # partition/eligibility variable
        xformla="~ covariate",
        control_group="nevertreated",
        est_method="dr",
    )

The triple DiD estimator adds ``pname`` to specify the partition variable
that identifies eligible units within treatment groups. All other core
arguments work the same as ``att_gt``.

The estimator automatically detects whether the data has two periods or
multiple periods, and whether the data is a balanced panel or repeated
cross-sections. For two-period data the ``control_group`` and
``base_period`` parameters are ignored since there is only one possible
comparison. Like ``att_gt``, passing a Dask or Spark DataFrame automatically
routes to a distributed implementation.


Difference-in-Differences with Continuous Treatments
----------------------------------------------------

The :func:`~moderndid.didcont.cont_did` function handles settings with
treatment intensity rather than binary treatment. This implements the
`Callaway, Goodman-Bacon, and Sant'Anna (2024) <https://arxiv.org/abs/2107.02637>`_
framework.

.. code-block:: python

    result = did.cont_did(
        data=data,
        yname="outcome",
        tname="year",
        idname="unit_id",
        gname="first_treated",
        dname="dose",
        control_group="notyettreated",
        anticipation=0,
        base_period="varying",
        alp=0.05,
        boot=True,
        biters=1000,
        clustervars=["unit_id"],
        # Method-specific options
        target_parameter="level",      # level or slope
        aggregation="dose",            # dose or eventstudy
        dose_est_method="parametric",  # parametric or cck
    )

All the inference options (``alp``, ``boot``, ``biters``, ``clustervars``,
``cband``) work the same way across estimators. The shared estimation
options (``control_group``, ``anticipation``, ``base_period``) also behave
identically.

.. important::

   The continuous treatment estimator does not yet support covariates (only
   ``xformla="~1"``), unbalanced panels, or discrete treatment values.
   Two-way clustering is not supported. The CCK estimation method
   (``dose_est_method="cck"``) requires exactly two groups and two time
   periods, and cannot be combined with event study aggregation.


Difference-in-Differences with Intertemporal Treatment Effects
--------------------------------------------------------------

The :func:`~moderndid.didinter.did_multiplegt` function handles settings with
non-binary, non-absorbing (time-varying) treatments where lagged treatments
may affect outcomes. This implements the
`de Chaisemartin and D'Haultfoeuille (2024) <https://doi.org/10.1162/rest_a_01414>`_
framework.

.. code-block:: python

    result = did.did_multiplegt(
        data=data,
        yname="outcome",
        tname="year",
        idname="unit_id",
        dname="treatment",            # treatment variable (can vary over time)
        effects=5,                    # number of post-treatment periods
        placebo=3,                    # number of placebo periods
        cluster="unit_id",
    )

Unlike ``att_gt`` which requires a ``gname`` (first treatment period), the
intertemporal estimator uses ``dname`` directly since treatment can change
multiple times. The estimator compares units whose treatment changes
("switchers") to units with the same baseline treatment that have not yet
switched. Setting ``effects=L`` produces estimates for each period of
exposure from 1 through L, and ``placebo=K`` produces K pre-treatment
placebo estimates for testing parallel trends.

.. code-block:: python

    result = did.did_multiplegt(
        data=data,
        yname="outcome",
        tname="year",
        idname="unit_id",
        dname="treatment",
        # Effect options
        effects=5,
        placebo=3,
        normalized=True,              # normalize by cumulative treatment change
        effects_equal=True,           # chi-squared test for equal effects
        # Inference options
        cluster="unit_id",
        ci_level=95.0,
        boot=True,
        biters=1000,
        # Control options
        controls=["covariate1", "covariate2"],
        trends_lin=True,              # unit-specific linear trends
    )

By default, units that experience both treatment increases and decreases
(bidirectional switchers) are dropped because they can violate the
no-sign-reversal property required for causal identification. Set
``keep_bidirectional_switchers=True`` to override this, but interpret
results with caution.

The result includes an average total effect (ATE) per unit of treatment
that accounts for both contemporaneous and lagged effects. The ATE is not
computed when ``trends_lin=True``.

.. important::

   When ``continuous > 0``, the variance estimators are not backed by
   proven asymptotic normality. Bootstrap inference (``boot=True``) is
   recommended.


Sensitivity Analysis for Parallel Trends Violations
---------------------------------------------------

The :mod:`~moderndid.didhonest` module assesses robustness to parallel
trends violations. It takes results from ``att_gt``, or external event
study results, and produces confidence intervals that remain valid under
specified degrees of parallel trends violation. This follows the
`Rambachan and Roth (2023) <https://doi.org/10.1093/restud/rdad018>`_
framework.

.. code-block:: python

    from moderndid.didhonest import honest_did

    # First estimate group-time effects
    result = did.att_gt(
        data=data,
        yname="outcome",
        tname="year",
        idname="unit_id",
        gname="first_treated",
    )

    # Aggregate into an event study (required)
    event_study = did.aggte(result, type="dynamic")

    # Then conduct sensitivity analysis
    sensitivity = honest_did(event_study, event_time=0, sensitivity_type="smoothness")

The input must be a dynamic event study aggregation (not group- or
calendar-level), and the event study must have influence functions computed.
Pre-treatment and post-treatment event times must be consecutive integers
with no gaps. The requested ``event_time`` must exist in the post-treatment
periods.


Next steps
----------

Each estimator has a dedicated example page that walks through a full
analysis with real or simulated data.

- :doc:`example_staggered_did` for staggered adoption with ``att_gt``
- :doc:`example_cont_did` for dose-response with ``cont_did``
- :doc:`example_triple_did` for triple differences with ``ddd``
- :doc:`example_inter_did` for time-varying treatments with ``did_multiplegt``
- :doc:`example_honest_did` for sensitivity analysis with ``honest_did``

For scaling any of these estimators to large datasets, see
:doc:`distributed`.
