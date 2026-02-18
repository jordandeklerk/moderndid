.. _estimator-overview:

==================
Estimator Overview
==================

ModernDiD provides additional estimators beyond :func:`~moderndid.did.att_gt`
for different research designs. All estimators share the same core API, so
once you learn one, the others follow the same pattern with
estimator-specific parameters. This page provides a brief overview of each
estimator and its key arguments. For detailed usage with real data, see the
individual example pages. The baseline staggered DiD estimator follows
`Callaway and Sant'Anna (2021) <https://doi.org/10.1016/j.jeconom.2020.12.001>`_.


Choosing the right estimator
-----------------------------

The choice depends on the structure of your treatment variable.

Most applied settings involve a binary treatment that turns on permanently
at staggered times across groups. :func:`~moderndid.att_gt` handles this
standard case and should be the starting point for most analyses.

When treatment intensity varies continuously across units,
:func:`~moderndid.cont_did` extends the framework to recover
dose-response functions showing how effects scale with dosage.

When a policy enables treatment for a group but only a subset of units
within that group is actually eligible, :func:`~moderndid.ddd` exploits
this additional within-group variation. Parental leave affecting women but
not men, or minimum wage affecting hourly but not salaried workers, are
canonical examples.

When treatment is not permanent and can switch on and off or change
intensity over time, :func:`~moderndid.did_multiplegt` provides valid
inference by comparing units whose treatment changed to those with the
same baseline treatment that have not yet changed.

After running any estimator, :func:`~moderndid.honest_did` assesses how
large violations of the parallel trends assumption would need to be to
overturn your conclusions.


Continuous Treatment DiD
------------------------

The :func:`~moderndid.didcont.cont_did` function handles settings with
treatment intensity rather than binary treatment. We can compare the function
signatures for the binary treatment case and the continuous treatment case.
This implements the
`Callaway, Goodman-Bacon, and Sant'Anna (2024) <https://arxiv.org/abs/2107.02637>`_
framework.

.. code-block:: python

    # Binary treatment (att_gt)
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

    # Continuous treatment (cont_did)
    result = did.cont_did(
        data=data,
        yname="outcome",
        tname="year",
        idname="unit_id",
        gname="first_treated",
        dname="dose",                  # additional dose variable
        xformula="~ covariate",
        control_group="notyettreated",
    )

The core arguments are identical. The continuous treatment estimator adds
``dname`` to specify the dose variable. It also provides method-specific
options for dose-response estimation.

.. code-block:: python

    result = did.cont_did(
        data=data,
        yname="outcome",
        tname="year",
        idname="unit_id",
        gname="first_treated",
        dname="dose",
        # Shared options
        xformula="~ covariate",
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

All the inference options (``alp``, ``boot``, ``biters``, ``clustervars``, ``cband``)
work the same way across estimators. The shared estimation options
(``control_group``, ``anticipation``, ``base_period``) also behave
identically.


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
        pname="eligible",              # additional: partition/eligibility
        xformla="~ covariate",
        control_group="nevertreated",
        est_method="dr",
    )

The triple DiD estimator adds ``pname`` to specify the partition variable
that identifies eligible units within treatment groups. All other core
arguments work the same as ``att_gt``.


Intertemporal DiD
-----------------

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
switched.

.. code-block:: python

    result = did.did_multiplegt(
        data=data,
        yname="outcome",
        tname="year",
        idname="unit_id",
        dname="treatment",
        # Effect options
        effects=5,                    # dynamic effects for 5 periods
        placebo=3,                    # 3 placebo periods
        normalized=True,              # normalize by cumulative treatment change
        # Inference options
        cluster="unit_id",
        ci_level=95.0,
        boot=True,                    # bootstrap inference
        biters=1000,
        # Control options
        controls=["covariate1", "covariate2"],
        trends_lin=True,              # unit-specific linear trends
    )


Sensitivity Analysis
--------------------

The :mod:`~moderndid.didhonest` module assesses robustness to parallel
trends violations. It takes results from ``att_gt``, or external event study results,
and produces confidence intervals that remain valid under specified degrees of
parallel trends violation. This follows the
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

    # Then conduct sensitivity analysis on the event study
    event_study = did.aggte(result, type="dynamic")
    sensitivity = honest_did(event_study, event_time=0, sensitivity_type="smoothness")

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
