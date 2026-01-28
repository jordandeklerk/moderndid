.. _quickstart:

====================
ModernDiD Quickstart
====================

ModernDiD estimates causal effects using difference-in-differences methods.
The :func:`~moderndid.did.att_gt` function computes group-time average treatment
effects for staggered adoption designs. Other estimators handle continuous
treatments (:func:`~moderndid.didcont.cont_did`), triple differences
(:func:`~moderndid.didtriple.ddd`), and sensitivity analysis
(:mod:`~moderndid.didhonest`). Most estimators support both panel data and
repeated cross-sections.

All estimators share a consistent API built around four core arguments that
describe your data structure. For an introduction to difference-in-differences
terminology and setup, see :ref:`Introduction to Difference-in-Differences <causal_inference>`.
For theoretical details on each estimator, see the :ref:`Background <background>` section.


Core Arguments
==============

Every estimator requires four variables that identify the panel structure.

``yname``
    The outcome variable you want to measure treatment effects on
    (such as employment, earnings, or test scores).

``tname``
    The time period variable that indexes when observations occur
    (such as year, quarter, or month).

``idname``
    The unit identifier that tracks the same unit across time periods
    (such as state, county, or individual ID).

``gname``
    The group variable indicating when each unit was first treated.
    Units that never receive treatment should have a value of 0.

Options like ``xformla`` for covariates, ``est_method`` for estimation
approach, and ``bstrap`` for bootstrap inference work similarly across
all estimators. Once you learn one, the others follow the same patterns.


Basic Usage
===========

We will focus on the :func:`~moderndid.did.att_gt` function, which estimates group-time average
treatment effects for staggered adoption designs. Other estimators follow the same pattern with
estimator-specific parameters in addition to the core arguments.

.. code-block:: python

    import moderndid as did

    result = did.att_gt(
        data=data,
        yname="outcome",
        tname="year",
        idname="unit_id",
        gname="first_treated",
    )

The :func:`~moderndid.did.aggte` function aggregates group-time effects
into interpretable summaries.

.. code-block:: python

    # Event study aggregation
    event_study = did.aggte(result, type="dynamic")

    # Overall average treatment effect
    overall = did.aggte(result, type="simple")

    # Effects by treatment cohort
    by_group = did.aggte(result, type="group")

    # Effects by calendar time
    by_time = did.aggte(result, type="calendar")


Estimation Options
==================

The ``att_gt`` function provides several options for customizing the
estimation procedure. The defaults reflect best practices, but you may
want to adjust them based on your data and research design.


Estimation Methods
------------------

The ``est_method`` parameter controls how group-time effects are
estimated. The default is ``"dr"`` for doubly robust, but you can also
use inverse probability weighting or outcome regression.

.. code-block:: python

    # Doubly robust (default) - combines IPW and regression
    result = did.att_gt(
        data=data,
        yname="outcome",
        tname="year",
        idname="unit_id",
        gname="first_treated",
        est_method="dr",
    )

    # Inverse probability weighting
    result = did.att_gt(..., est_method="ipw")

    # Outcome regression
    result = did.att_gt(..., est_method="reg")

Doubly robust estimation is recommended because it provides consistent
estimates if either the propensity score model or the outcome regression
model is correctly specified.


Covariates
----------

When treated and comparison groups differ on observable characteristics,
include covariates using the ``xformla`` parameter. This strengthens the
parallel trends assumption by conditioning on these differences.

.. code-block:: python

    result = did.att_gt(
        data=data,
        yname="outcome",
        tname="year",
        idname="unit_id",
        gname="first_treated",
        xformla="~ covariate1 + covariate2",
    )

The formula uses R-style syntax. Covariates should be time-invariant or
measured before treatment to avoid conditioning on post-treatment variables.


Control Group
-------------

By default, ``att_gt`` uses never-treated units as the comparison group.
You can instead use not-yet-treated units, which includes units that will
be treated in the future but have not yet been treated at time *t*.

.. code-block:: python

    result = did.att_gt(
        data=data,
        yname="outcome",
        tname="year",
        idname="unit_id",
        gname="first_treated",
        control_group="notyettreated",
    )

Using not-yet-treated units can increase precision when there are few
never-treated units, but requires assuming that treatment timing is
unrelated to potential outcomes.


Bootstrap Inference
-------------------

By default, ``att_gt`` uses the multiplier bootstrap to compute standard
errors and simultaneous confidence bands.

.. code-block:: python

    result = did.att_gt(
        data=data,
        yname="outcome",
        tname="year",
        idname="unit_id",
        gname="first_treated",
        bstrap=True,       # Use bootstrap (default)
        biters=1000,       # Number of bootstrap iterations (default)
        cband=True,        # Compute uniform confidence bands (default)
        alp=0.05,          # Significance level (default)
        random_state=42,   # For reproducibility
    )

Setting ``bstrap=False`` uses analytical standard errors instead of
bootstrap, which is faster but does not support clustering or uniform
confidence bands. Setting ``cband=False`` computes pointwise confidence
intervals instead of simultaneous bands.


Clustering
----------

Standard errors can be clustered at one or two levels using the
``clustervars`` parameter. Clustering requires using the bootstrap.

.. code-block:: python

    result = did.att_gt(
        data=data,
        yname="outcome",
        tname="year",
        idname="unit_id",
        gname="first_treated",
        clustervars=["unit_id"],
        bstrap=True,
    )

You can cluster on at most two variables, and one of them must be the
unit identifier (``idname``).


Anticipation
------------

If units can anticipate treatment before it actually occurs, account for
this using the ``anticipation`` parameter. This shifts the base period
backward.

.. code-block:: python

    result = did.att_gt(
        data=data,
        yname="outcome",
        tname="year",
        idname="unit_id",
        gname="first_treated",
        anticipation=1,  # Allow 1 period of anticipation
    )

With ``anticipation=1``, the base period for group *g* becomes period
*g-2* instead of *g-1*, treating period *g-1* as potentially affected
by anticipation of the upcoming treatment.


Base Period
-----------

The ``base_period`` parameter controls which pre-treatment period is used
as the comparison. The default ``"varying"`` uses the period immediately
before treatment (or before anticipation). Setting ``base_period="universal"``
uses the same base period for all groups.

.. code-block:: python

    # Varying base period (default)
    result = did.att_gt(..., base_period="varying")

    # Universal base period
    result = did.att_gt(..., base_period="universal")

A universal base period can be useful when you want all pre-treatment
estimates to be relative to the same reference point.


Panel Data Options
------------------

By default, ``att_gt`` expects a balanced panel where all units are
observed in all time periods. If your panel is unbalanced, set
``allow_unbalanced_panel=True``.

.. code-block:: python

    result = did.att_gt(
        data=data,
        yname="outcome",
        tname="year",
        idname="unit_id",
        gname="first_treated",
        allow_unbalanced_panel=True,
    )

For repeated cross-sections rather than panel data, set ``panel=False``
and omit the ``idname`` parameter.


Sampling Weights
----------------

If your data has sampling weights, specify them using ``weightsname``.

.. code-block:: python

    result = did.att_gt(
        data=data,
        yname="outcome",
        tname="year",
        idname="unit_id",
        gname="first_treated",
        weightsname="weight_column",
    )


Visualization
=============

ModernDiD provides built-in plotting functions using plotnine.

.. code-block:: python

    # Plot group-time effects
    did.plot_gt(result)

    # Plot event study
    did.plot_event_study(event_study)


Other Estimators
================

ModernDiD provides additional estimators for different research designs.
All estimators share the same core API, so once you learn ``att_gt``, or another estimator,
the others follow the same pattern with estimator-specific parameters.


Continuous Treatment DiD
------------------------

The :func:`~moderndid.didcont.cont_did` function handles settings with
treatment intensity rather than binary treatment. We can compare the function
signatures for the binary treatment case and the continuous treatment case.

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
        biters=1000,
        clustervars=["unit_id"],
        # Method-specific options
        target_parameter="level",      # level or slope
        aggregation="dose",            # dose or eventstudy
        dose_est_method="parametric",  # parametric or cck
    )

All the inference options (``alp``, ``biters``, ``clustervars``, ``cband``)
work the same way across estimators. The shared estimation options
(``control_group``, ``anticipation``, ``base_period``) also behave
identically.


Triple Difference-in-Differences
--------------------------------

The :func:`~moderndid.didtriple.ddd` function leverages an additional
dimension of variation such as eligibility status. The API follows
the same pattern as the other estimators.

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


Sensitivity Analysis
--------------------

The :mod:`~moderndid.didhonest` module assesses robustness to parallel
trends violations. It takes results from ``att_gt``, or external event study results,
and produces confidence intervals that remain valid under specified degrees of
parallel trends violation.

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

See the examples for detailed usage of each estimator.


.. toctree::
   :maxdepth: 1
   :hidden:

   self
   example_staggered_did
