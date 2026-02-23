.. _panel-utilities:

====================
Panel Data Utilities
====================

ModernDiD's :mod:`~moderndid.core.panel` module provides tools for inspecting and
cleaning panel data before estimation. Every estimator has a robust
preprocessing pipeline that automatically handles most panel irregularities,
so these utilities are optional. They are useful when you want to understand
what the pipeline is doing under the hood, or when you want to make cleaning
decisions yourself rather than relying on the defaults.

Like the estimators, every panel utility function accepts any
Arrow-compatible DataFrame, converts to
Polars internally for speed, and returns results in your original dataframe
format.


Diagnosing the Data
-------------------

:func:`~moderndid.core.panel.diagnose_panel` gives you a quick summary of
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
---------------

If you do want to handle the gaps yourself, the diagnostics suggest a couple of
strategies.

:func:`~moderndid.core.panel.fill_panel_gaps` keeps every county and fills
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

:func:`~moderndid.core.panel.make_balanced_panel` takes the opposite
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
:func:`~moderndid.core.panel.deduplicate_panel` handles this by keeping the
last occurrence by default, or can average numeric columns with
``strategy="mean"``.


Building the Group-Timing Variable
-----------------------------------

Most ModernDiD estimators take ``gname`` as an argument, a column indicating the first
period each unit was treated (0 for never-treated). Many datasets
instead store a raw binary treatment indicator that flips from 0 to 1
when treatment begins. :func:`~moderndid.core.panel.get_group` converts
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

Inspection Helpers
------------------

Several lightweight functions answer common questions about a panel
without running full diagnostics.

.. code-block:: python

    # Quick boolean checks
    did.is_balanced_panel(data, idname="county", tname="year")
    did.has_gaps(data, idname="county", tname="year")

    # Which columns change within units over time?
    did.are_varying(data, idname="county", cols=["inter_bra", "state"])
    # {"inter_bra": True, "state": False}

    # List the exact missing unit-time pairs
    gaps = did.scan_gaps(data, idname="county", tname="year")

:func:`~moderndid.core.panel.complete_data` keeps only units observed in at
least ``min_periods`` time periods, which is useful for dropping units
with too few observations.

.. code-block:: python

    # Keep units observed in at least 10 of 12 periods
    trimmed = did.complete_data(data, idname="county", tname="year", min_periods=10)

:func:`~moderndid.core.panel.deduplicate_panel` removes duplicate unit-time
pairs. The default keeps the last occurrence; ``strategy="mean"`` averages
numeric columns instead.

.. code-block:: python

    deduped = did.deduplicate_panel(data, idname="county", tname="year", strategy="last")


Reshaping and Transformations
-----------------------------

These functions convert between panel formats and compute common
transformations.

.. code-block:: python

    # Pivot long panel to wide (one column per period)
    wide = did.panel_to_wide(data, idname="county", tname="year")

    # Unpivot wide back to long
    long = did.wide_to_panel(wide, idname="county", stub_names=["outcome"], tname="year")

    # First-difference the outcome variable (adds a "dy" column)
    diffed = did.get_first_difference(data, idname="county", yname="outcome", tname="year")

For repeated cross-section data (no unit tracked over time),
:func:`~moderndid.core.panel.assign_rc_ids` adds a unique ``"rowid"`` column
that some estimators require.

.. code-block:: python

    rc_data = did.assign_rc_ids(data)


Next steps
----------

Once your data is clean, you are ready to estimate treatment effects.

- :ref:`Quickstart <quickstart>` walks through ``att_gt`` estimation,
  aggregation, and all available options.
- :ref:`Estimator Overview <estimator-overview>` surveys additional
  estimators for continuous treatments, triple differences, and more.
