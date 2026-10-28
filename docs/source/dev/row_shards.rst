.. _dev-row-shards:

Rows on shards
==============

The :doc:`panel engine <shards>` reads shards that hold each unit's outcome in
every period. That layout fits neither repeated cross-sections, where each unit
appears once, nor an unbalanced panel, where some units miss periods. The row
engine in ``row_kernels.py`` and ``row_driver.py`` therefore reads a
``RowShard`` with one entry for each observation. Most of the panel engine
carries over to rows, from its passes and sums over units to its keyed draws.
What changes is how a shard holds the data, whether cells can share a fit, and
how a row without an identifier gets its bootstrap key.

On Spark, a fit first lays its data out as a table with one row for each unit
of a panel or one row for each observation of repeated cross-sections. The
figure below sets the unit table of a balanced panel, the unit table of an
unbalanced panel, and the row table of repeated cross-sections beside the shard
that each one becomes.

.. container:: mdid-diagram

   .. raw:: html
      :file: diagrams/shard-layouts.svg

.. rst-class:: mdid-caption

The three table layouts and the shards that their readers make from a slice of
rows. Dashed columns and arrays appear only in fits that need them.

Because each row of an unbalanced panel's unit table holds a whole unit, its
shards keep each unit's rows together in period order. For the walkthrough,
we'll treat the county rows as repeated cross-sections, as if each year had
sampled new counties. After ``preprocess_did`` prepares them,
``row_shards_from_did_data`` deals them into three shards. Each shard should
hold about a third of the 2,500 county-years in flat arrays of one entry per
row. A period index on each row takes the place of the panel's period axis.

.. code-block:: python

   import moderndid as did
   import numpy as np

   from moderndid.did.shards.local import row_shards_from_did_data

   data = did.load_mpdta()

   # Every county-year as an observation of its own, with no identifier to pair the years.
   rc_columns = {
       "yname": "lemp",
       "tname": "year",
       "gname": "first.treat",
       "idname": None,
       "xformla": "~ lpop",
   }
   rc_prepared = did.preprocess_did(data, **rc_columns, panel=False)
   row_shards = row_shards_from_did_data(rc_prepared, 3, random_state=0)

   print("Rows per shard:", [len(shard.y) for shard in row_shards])
   print("Outcomes of the first shard:", row_shards[0].y.shape)
   print("Covariates of the first shard:", row_shards[0].covariates.shape)
   print("Period indices of its first rows:", row_shards[0].period[:6])

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         Rows per shard: [834, 833, 833]
         Outcomes of the first shard: (834,)
         Covariates of the first shard: (834, 1)
         Period indices of its first rows: [3 1 0 0 2 3]

Where the panel's shards held one entry for every unit and period, each array
here holds one entry for every row. Each period index points into the five
sorted years of the fit. In an unbalanced panel, the ``units`` array points
each row to its unit. Here it stays None because every county-year is a unit
by itself.

A fit for every cell
--------------------

The panel engine saved work by letting cells with the same units, covariates,
and weights share one fit, as the 12 county cells shared 8. Rows can't share
that way, because a cell of rows fits its models on the rows of both of its
periods rather than on units set by the earlier one. When ``config.panel`` is
False, ``plan_cells`` therefore gives every cell a fit of its own.

The rest of the fit runs as it does for a panel. ``fit_att_gt_on_row_shards``
makes the same four kinds of passes as ``fit_att_gt_on_shards`` with the
kernels of ``row_kernels.py``. It reuses the logit iterations, least squares
solves, and seed handling of ``driver.py`` and ends in the same
``_finish_shard_fit``, the step that runs the moments pass and builds the
result. The code below fits the county rows both in memory and on the three
shards and counts the cells, fits, and passes.

.. code-block:: python

   from moderndid.did.shards.local import LocalExecutor
   from moderndid.did.shards.plan import plan_cells
   from moderndid.did.shards.row_driver import fit_att_gt_on_row_shards

   # The in-memory fit of the same rows and the row engine's fit on the three shards.
   rc_result = did.att_gt(data, **rc_columns, panel=False, boot=False, cband=False)
   executor = LocalExecutor(row_shards)
   settings = {"bootstrap": False, "uniform_bands": False}
   rc_sharded = fit_att_gt_on_row_shards(
       executor, rc_prepared.config, rc_prepared.cohort_counts, None, settings
   )

   # The cells of the plan and the fits they use.
   plan = plan_cells(rc_prepared.config, rc_prepared.cohort_counts)

   print("Cells and fits:", len(plan.cells.fit), len(plan.fits.group))
   print("Passes over the shards:", executor.n_passes)
   print(f"Largest gap in the effects: {np.abs(rc_sharded.att_gt - rc_result.att_gt).max():.1e}")
   print(f"Largest gap in the standard errors: {np.abs(rc_sharded.se_gt - rc_result.se_gt).max():.1e}")

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         Cells and fits: 12 12
         Passes over the shards: 9
         Largest gap in the effects: 2.0e-16
         Largest gap in the standard errors: 8.3e-17

The two fits agree on every effect and standard error up to rounding. The 12
cells take 12 fits where the balanced panel's took 8. Like the panel fit, this
one makes nine passes because the logits of every cell iterate together, one
pass for each of six iterations.

The moments pass still sums one row :math:`z_i` for each unit as in the panel
engine. What rows change is how each unit's :math:`z_i` gets built from its
observations. Each row contributes the estimator's influence function to its
cell, scaled by the number of units in the data over the number of the cell's
rows as in memory. In repeated cross-sections, each observation is a unit of
its own that gets its own :math:`z_i` and puts its weight in its cohort's
column. In an unbalanced panel, ``row_moments_kernel`` first adds up the
contributions of a unit's rows and puts their mean weight in that column.

.. admonition:: Keep each unit's rows in one shard
   :class: important

   The moments pass adds up a unit's rows only within a shard. A unit split
   between two shards would therefore count as two units. The summary pass
   checks the unit count and stops such a fit with a RuntimeError.

Keys for rows without identifiers
---------------------------------

The :ref:`draws of the panel engine <dev-shard-draws>` compute each unit's
bootstrap sign from a key made from its identifier. An unbalanced panel keys
its units the same way through ``canonical_keys``. A clustered fit of either
layout keys each cluster instead. Repeated cross-sections are harder, because
the in-memory fit gives each of their rows its position in the data as its
identifier, even when ``idname`` names a unit. A Spark table has no stable row
order to copy those positions from.

Without an identifier or an order to go by, a row can only be told apart by its
values. ``row_keys`` therefore hashes each row's period index, cohort, outcome,
raw sampling weight, and covariates into a 64-bit key. The raw weight goes in
rather than the normalized one because the mean that normalizes the weights is
a Spark sum whose last bits can change with the partitioning.

Duplicate rows, such as the same answers given by two survey respondents, would
still share a key and every sign drawn from it. The in-memory bootstrap instead
draws a separate multiplier for each of them. The key therefore also hashes
each row's copy index, its position among the rows whose values all match its
own. To see the copy index at work, the code below repeats the first county
row and keys the rows with and without copy indices, on one shard and on seven.

.. code-block:: python

   import polars as pl

   from moderndid.did.shards.layout import row_keys

   # The county rows with the first one repeated, like a survey answer given twice.
   doubled = did.preprocess_did(pl.concat([data, data.head(1)]), **rc_columns, panel=False)
   whole = row_shards_from_did_data(doubled, 1, random_state=0)[0]
   seven = row_shards_from_did_data(doubled, 7, random_state=1)

   # The same values keyed with every copy index set to 0. The fit has no weights to key.
   no_copies = np.zeros(len(whole.y))
   unnumbered = row_keys(whole.period, whole.cohort, whole.y, None, whole.covariates, no_copies)
   split_keys = np.concatenate([shard.unit_keys for shard in seven])

   print("Rows:", len(whole.y))
   print("Distinct keys:", len(np.unique(whole.unit_keys)))
   print("Distinct keys without copy indices:", len(np.unique(unnumbered)))
   print("Same keys from seven shards:", np.array_equal(np.sort(split_keys), np.sort(whole.unit_keys)))

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         Rows: 2501
         Distinct keys: 2501
         Distinct keys without copy indices: 2500
         Same keys from seven shards: True

All 2,501 rows have distinct keys, the repeated county-year included.
Keyed without its copy index of 1, that row would take the original's key.
Seven shards give the same keys as one, since it never matters which of two
interchangeable copies gets which index.

Numbering the copies needs every copy of a row in the same place. On Spark,
``build_row_table`` gathers them with one shuffle, the step in which Spark
moves rows between partitions so that identical rows meet. Because a fit
without draws needs no row keys, the builder adds that shuffle only for an
unclustered fit that draws under ``boot=True`` or the default ``cband=True``.
If a fit would draw from rows without keys, ``fit_att_gt_on_row_shards`` stops
with a RuntimeError after its summary pass.

.. admonition:: Number copies over the hashed values
   :class: important

   The columns that number the copies, in ``_row_rows`` on Spark and in
   ``_copy_indices`` in memory, have to be exactly the values that
   ``row_keys`` hashes. Rows that differ only in an extra column would share a
   key. Rows that differ only in a missing one would count as copies of each
   other.

The guide to :doc:`fits on Spark <spark_fits>` follows a call to ``att_gt``
through the builders that lay out these rows and the tasks that read them.
