.. _quickstart:

====================
ModernDiD Quickstart
====================

ModernDiD estimates causal effects using difference-in-differences methods.
The :func:`~moderndid.did.att_gt` function computes group-time average treatment
effects for staggered adoption designs. Other estimators handle continuous
treatments (:func:`~moderndid.didcont.cont_did`), triple differences
(:func:`~moderndid.didtriple.ddd`), intertemporal treatments
(:func:`~moderndid.didinter.did_multiplegt`), and sensitivity analysis
(:mod:`~moderndid.didhonest`). Most estimators support both panel data and
repeated cross-sections.

All estimators share a consistent API built around core parameters that
describe your data structure. For an introduction to difference-in-differences
terminology and setup, see :ref:`Introduction to Difference-in-Differences <causal_inference>`.
For theoretical details on each estimator, see the :ref:`Background <background>` section.


Dataframe Agnostic
------------------

ModernDiD accepts any DataFrame that implements the
`Arrow PyCapsule Interface <https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html>`_.
This includes polars, pandas, pyarrow Table, duckdb, cudf, and other Arrow-compatible DataFrames.
You can pass your data directly without manual conversion.

.. code-block:: python

    import moderndid as did

    # Works with pandas
    import pandas as pd
    df_pandas = pd.read_csv("data.csv")
    result = did.att_gt(data=df_pandas, ...)

    # Works with polars
    import polars as pl
    df_polars = pl.read_csv("data.csv")
    result = did.att_gt(data=df_polars, ...)

    # Works with pyarrow
    import pyarrow.parquet as pq
    table = pq.read_table("data.parquet")
    result = did.att_gt(data=table, ...)

    # Works with duckdb
    import duckdb
    df_duck = duckdb.query("SELECT * FROM 'data.parquet'").arrow()
    result = did.att_gt(data=df_duck, ...)

Conversion happens automatically via
`narwhals <https://narwhals-dev.github.io/narwhals/>`_, with all internal
operations using Polars for performance. This means you get the speed benefits
of Polars regardless of your input format.


Core Arguments
--------------

Every estimator requires three variables that identify the panel structure.

``yname``
    The outcome variable you want to measure treatment effects on
    (such as employment, earnings, or test scores).

``tname``
    The time period variable that indexes when observations occur
    (such as year, quarter, or month).

``idname``
    The unit identifier that tracks the same unit across time periods
    (such as state, county, or individual ID).

The treatment variable differs by estimator. Most estimators use ``gname``
to indicate when each unit was first treated (units never treated should
have a value of 0). The intertemporal estimator uses ``dname`` instead,
since treatment can change multiple times over the panel.

Options like ``xformla`` for covariates, ``est_method`` for estimation
approach, and ``boot`` for bootstrap inference work similarly across
all estimators. Once you learn one, the others follow the same patterns.


Basic Usage
-----------

Estimation in ModernDiD is typically a two-step process. First, compute group-time
effects, then aggregate them into the summary you need. We will focus on the :func:`~moderndid.did.att_gt` function
for now, but the same pattern generally applies to all estimators.

The :func:`~moderndid.did.att_gt` function
estimates a separate average treatment effect for every combination of
treatment cohort (group) and calendar period. A treatment cohort is the set
of units that were first treated at the same time. The result is a full
matrix of effects that captures how treatment impacts evolve for each cohort
over time.

.. code-block:: python

    import moderndid as did

    result = did.att_gt(
        data=data,
        yname="outcome",
        tname="year",
        idname="unit_id",
        gname="first_treated",
    )

The group-time matrix is typically too detailed to
interpret directly. The :func:`~moderndid.did.aggte` function collapses it
into one of four summary forms, each answering a different question.

.. code-block:: python

    # Event study: average effect at each time relative to treatment
    event_study = did.aggte(result, type="dynamic")

    # Simple: single weighted average across all groups and periods
    overall = did.aggte(result, type="simple")

    # Group: one average per treatment cohort
    by_group = did.aggte(result, type="group")

    # Calendar: one average per calendar period
    by_time = did.aggte(result, type="calendar")

The ``"dynamic"`` aggregation is the most common starting point because the
resulting event study plot lets you visually assess pre-trends and trace how
treatment effects build or fade over time.


Estimation Options
------------------

The ``att_gt`` function provides several options for customizing the
estimation procedure. The defaults reflect best practices, but you may
want to adjust them based on your data and research design.


Estimation Methods
^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^

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
^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^

By default, ``att_gt`` uses the multiplier bootstrap to compute standard
errors and simultaneous confidence bands.

.. code-block:: python

    result = did.att_gt(
        data=data,
        yname="outcome",
        tname="year",
        idname="unit_id",
        gname="first_treated",
        boot=True,         # Use bootstrap (default is False)
        biters=1000,       # Number of bootstrap iterations (default)
        cband=True,        # Compute uniform confidence bands (default)
        alp=0.05,          # Significance level (default)
        random_state=42,   # For reproducibility
    )

Setting ``boot=False`` (the default) uses analytical standard errors instead
of bootstrap, which is faster but does not support clustering or uniform
confidence bands. Setting ``cband=False`` computes pointwise confidence
intervals instead of simultaneous bands.


Clustering
^^^^^^^^^^

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
        boot=True,
    )

You can cluster on at most two variables, and one of them must be the
unit identifier (``idname``).


Anticipation
^^^^^^^^^^^^

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
^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^

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
-------------

ModernDiD provides built-in plotting functions that return
`plotnine <https://plotnine.org/>`_ ``ggplot`` objects.

.. code-block:: python

    # Plot group-time effects
    did.plot_gt(result)

    # Plot event study
    did.plot_event_study(event_study)

Because the return value is a standard ``ggplot`` object, you can customize
it with any plotnine layer using the ``+`` operator.

.. code-block:: python

    from plotnine import labs, theme, element_text

    plot = did.plot_event_study(event_study, ref_period=-1)
    plot = (
        plot
        + labs(
            title="Effect of Minimum Wage on Employment",
            x="Years Relative to Treatment",
            y="ATT (log points)",
        )
        + theme(plot_title=element_text(size=14, weight="bold"))
    )

ModernDiD also ships with ready-made themes for common use cases.

.. code-block:: python

    from moderndid.plots import theme_publication

    did.plot_event_study(event_study) + theme_publication()


Next steps
----------

This page covered the core workflow for staggered DiD with ``att_gt``.
From here you can explore several directions depending on your research
design.

- :ref:`Panel Data Utilities <panel-utilities>` shows how to diagnose and
  clean messy panel data before estimation.
- :ref:`Estimator Overview <estimator-overview>` surveys all available
  estimators, including continuous treatment, triple differences,
  intertemporal DiD, and sensitivity analysis.
- :ref:`Distributed Estimation <distributed>` explains how to scale
  estimation to datasets that do not fit on one machine using Dask.
- The :ref:`Examples <user-guide>` section walks through each estimator
  end-to-end with real and simulated data.
