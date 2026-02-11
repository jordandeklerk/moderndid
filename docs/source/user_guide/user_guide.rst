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

All estimators share a consistent API built around four core arguments that
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


Panel Data Utilities
--------------------

Every estimator in ModernDiD has a robust preprocessing pipeline that
automatically handles most panel irregularities. Rows with missing
values are dropped, unbalanced units are either kept or removed
depending on the ``allow_unbalanced_panel`` setting, treatment columns
are encoded, and weights are normalized, all before estimation begins.

Most of the time you can pass raw data straight to an estimator and it
will work fine. The :mod:`moderndid.panel` module is there for when you
want to understand what the pipeline is doing under the hood, or when
you want to make cleaning decisions yourself rather than relying on the
defaults.

Like the estimators, every panel utility function accepts any
Arrow-compatible DataFrame (pandas, polars, pyarrow, etc.), converts to
Polars internally for speed, and returns results in your original
format.


Diagnosing the Data
^^^^^^^^^^^^^^^^^^^^

:func:`~moderndid.panel.diagnose_panel` gives you a quick summary of
the panel's structure before you hand it to an estimator. Here we load
the `Favara and Imbs (2015) <https://doi.org/10.1257/aer.20121416>`_
banking-deregulation dataset, a county-level panel that, like many real
datasets, is not perfect.

.. code-block:: python

    import moderndid as did

    data = did.load_favara_imbs()
    diag = did.diagnose_panel(data,
                              idname="county",
                              tname="year",
                              treatname="inter_bra")
    print(diag)

.. code-block:: text

    ==========================================================================================
     Panel Diagnostics
    ==========================================================================================

    ┌───────────────────────────┬───────┐
    │ Metric                    │ Value │
    ├───────────────────────────┼───────┤
    │ Units                     │  1048 │
    │ Periods                   │    12 │
    │ Observations              │ 12538 │
    │ Balanced                  │    No │
    │ Duplicate unit-time pairs │     0 │
    │ Unbalanced units          │     5 │
    │ Gaps                      │    38 │
    │ Rows with missing values  │   524 │
    │ Single-period units       │     1 │
    │ Early-treated units       │     0 │
    │ Treatment time-varying    │   Yes │
    └───────────────────────────┴───────┘

    ------------------------------------------------------------------------------------------
     Suggestions
    ------------------------------------------------------------------------------------------
     Call fill_panel_gaps() to fill 38 missing unit-time pairs
     Call make_balanced_panel() to drop 5 units not observed in all periods
     524 rows contain missing values and will be dropped during preprocessing
     Call complete_data() or make_balanced_panel() to drop 1 units observed in only one period
     Treatment varies within units — verify this is expected or call get_group()
    ==========================================================================================

A balanced 1048 x 12 panel would have 12,576 observations, but we only
have 12,538. The report shows that 5 counties are not observed in every
year, creating 38 missing county-year pairs. It also flags 524 rows with
missing values that the preprocessing pipeline will silently drop, and
one county observed in only a single year. The ``inter_bra`` column
changes within counties over time. That is expected here because
interstate branching deregulation rolls out at different dates, but
exactly the kind of thing you want to catch early if your treatment is
supposed to be time-invariant.

You could pass this data directly to
:func:`~moderndid.didinter.did_multiplegt` and it would work. The
preprocessing pipeline would silently drop the 5 incomplete counties.
The value of running diagnostics first is that you see *what* gets
dropped and can decide whether that is acceptable for your analysis.


Fixing the Gaps
^^^^^^^^^^^^^^^

If you do want to handle the gaps yourself, the diagnostics suggest two
strategies.

:func:`~moderndid.panel.fill_panel_gaps` keeps every county and fills
the 38 missing county-year pairs with ``null`` rows. This preserves as
many units as possible, which is useful when you plan to impute the
missing values or pass the data to an estimator with
``allow_unbalanced_panel=True``.

.. code-block:: python

    filled = did.fill_panel_gaps(data, idname="county", tname="year")
    filled.shape

.. code-block:: text

    (12576, 7)

The panel is now a full 1048 x 12 rectangle.

:func:`~moderndid.panel.make_balanced_panel` takes the opposite
approach and drops the 5 incomplete counties entirely. You lose a few
units, but every remaining county is observed in all 12 years with no
nulls. This is what the preprocessing pipeline does by default when
``allow_unbalanced_panel=False``.

.. code-block:: python

    balanced = did.make_balanced_panel(data, idname="county", tname="year")
    balanced.shape

.. code-block:: text

    (12516, 7)

That gives 1043 counties x 12 years.

If your data had duplicate unit-time pairs, you would need to resolve
those before calling any estimator, since duplicates cause a hard
error in the preprocessing pipeline.
:func:`~moderndid.panel.deduplicate_panel` handles this by keeping the
last occurrence by default, or can average numeric columns with
``strategy="mean"``.


Building the Group-Timing Variable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most ModernDiD estimators take ``gname`` as an argument, a column indicating the first
period each unit was treated (0 for never-treated). Many datasets
instead store a raw binary treatment indicator that flips from 0 to 1
when treatment begins. :func:`~moderndid.panel.get_group` converts
between the two. It looks at when each unit's treatment first turns on
and writes that period into a new ``"G"`` column.

.. code-block:: python

    groups = did.get_group(data, idname="county", tname="year", treatname="inter_bra")
    groups["G"].unique().sort()

.. code-block:: text

    [0, 1995, 1996, 1997, 1998, 2000, 2001]

The output shows six distinct deregulation cohorts plus the
never-treated group (``0``). This ``"G"`` column can be passed directly
to ``gname`` in any estimator.

See :ref:`api-panel` for the full list of panel utility functions,
including :func:`~moderndid.panel.scan_gaps` for listing exactly which
unit-time pairs are missing and :func:`~moderndid.panel.are_varying`
for checking which columns change within units over time.


Basic Usage
-----------

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

ModernDiD provides built-in plotting functions using plotnine.

.. code-block:: python

    # Plot group-time effects
    did.plot_gt(result)

    # Plot event study
    did.plot_event_study(event_study)


Other Estimators
----------------

ModernDiD provides additional estimators for different research designs.
All estimators share the same core API, so once you learn ``att_gt``, or another estimator,
the others follow the same pattern with estimator-specific parameters.


Continuous Treatment DiD
^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


Intertemporal DiD
^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^

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
