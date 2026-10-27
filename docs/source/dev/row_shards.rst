.. _dev-row-shards:

Rows on shards
==============

The :doc:`panel engine <shards>` reads shards that hold each unit's outcome in
every period. That layout fits neither repeated cross-sections, where each unit
appears once, nor an unbalanced panel, where some units miss periods. The row
engine in ``row_kernels.py`` and ``row_driver.py`` therefore reads a
``RowShard`` that holds the period index, cohort, outcome, weight, and
covariates of each observation. In an unbalanced panel, a shard also keeps the
rows of every unit together in period order and gives each row the position of
its unit in ``units``. We'll prepare the county rows as repeated
cross-sections with ``preprocess_did``, the one-call form of the preprocessing
builder, before ``row_shards_from_did_data`` splits them into three shards.

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

Each array holds one entry for every row where the panel's shards held one for
every unit and period. Each period index points into the five sorted years of
the fit. ``units`` stays None because every county-year is a unit by itself.

A fit for every cell
--------------------

Because a cell of rows fits its models on the rows of both of its periods, no
two cells can share a fit. ``plan_cells`` therefore gives every cell a fit of
its own when ``config.panel`` is False. ``fit_att_gt_on_row_shards`` makes the
same four kinds of passes as the panel driver with the kernels of
``row_kernels.py``. It shares the panel driver's logit iterations, least
squares solves, seed handling, and ``_finish_shard_fit``.

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

A row's contribution to a cell is the estimator's influence function scaled by
the number of units in the data over the number of the cell's rows, as in
memory. A row of repeated cross-sections forms a :math:`z_i` by itself and
puts its weight in its cohort's column. In an unbalanced panel,
``row_moments_kernel`` first adds up the contributions of a unit's rows and
puts their mean weight in that column. Since each shard sums the rows of its
own units, a unit split between two shards would count as two units. The
summary pass's check of the unit count then stops the fit with a RuntimeError.

Keys for rows without identifiers
---------------------------------

Like the :ref:`draws of the panel engine <dev-shard-draws>`, the bootstrap
draws of a row fit need a key for every unit. An unbalanced panel keys each
unit by its identifier through ``canonical_keys``, just as a balanced panel
does. A clustered fit of either layout keys each cluster instead. For repeated
cross-sections, though, the in-memory fit gives each row its position in the
data as its identifier, even when ``idname`` names a unit. A Spark table has no
stable row order to copy those positions from.

``row_keys`` therefore builds each row's 64-bit key from the row's period
index, cohort, outcome, raw sampling weight, and covariates. The raw weight
goes in rather than the normalized one because the mean that normalizes the
weights is a Spark sum whose last bits can change with the partitioning.
Duplicate rows, such as the same answers given by two survey respondents, would
still share a key and every sign drawn from it. The in-memory bootstrap instead
draws a separate multiplier for each of them. The key therefore also hashes
each row's copy index, its position among the rows whose values all match its
own. Repeating the first of the county rows shows what the copy index does.

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

On Spark, numbering the copies takes one shuffle that brings identical rows
together. ``build_row_table`` adds it only when an unclustered fit draws, with
``boot=True`` or the default ``cband=True``, since a fit without draws needs no
keys. If a fit would draw from rows without them, ``fit_att_gt_on_row_shards``
stops with a RuntimeError after its summary pass.

.. admonition:: Number copies over the hashed values
   :class: important

   The columns that number the copies, in ``_row_rows`` on Spark and in
   ``_copy_indices`` in memory, have to be exactly the values that
   ``row_keys`` hashes. Rows that differ only in an extra column would share a
   key. Rows that differ only in a missing one would count as copies of each
   other.

The guide to :doc:`fits on Spark <spark_fits>` follows a call to ``att_gt``
through the builders that lay out these rows and the tasks that read them.
