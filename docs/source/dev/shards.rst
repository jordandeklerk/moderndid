.. _dev-shards:

Fits on shards
==============

When a panel sits in a Spark DataFrame, the two things an in-memory fit leans
on can't come to the driver, the Python process that runs the user's code. The
rows of the panel may be too many for one machine. The influence function
matrix holds a row for every unit and may be too large as well. We made
``att_gt`` work on a cluster by rewriting it around one idea. Every quantity
the estimator needs is a sum over units or something the driver can compute
from such sums. When each block of units computes its share wherever it lives,
the driver only has to add the shares up.

The shard engine in ``moderndid/did/shards/`` puts that idea into code. It has
a panel engine for balanced panels and a :doc:`row engine <row_shards>` for
repeated cross-sections and unbalanced panels. We'll rebuild the county fit
from the :doc:`architecture guide <architecture>` out of three blocks of
counties to show how the panel engine works. Along the way you'll see how the
counties split into blocks, how statistics from each block add up to the
in-memory ones, and how a whole fit runs on such statistics, its bootstrap
draws included. The :ref:`Spark guide <spark>` walks through such fits from the
user's side. The figure below shows the first step on the county data.

.. container:: mdid-diagram

   .. raw:: html
      :file: diagrams/shard-sums.svg

.. rst-class:: mdid-caption

One pass over three shards of the county panel. Each shard runs the same
kernel on its own counties. The driver adds the partials that come back and
divides the totals into each year's mean.

The code below starts from the in-memory fit that every later step should
match. It also prepares that fit's sample with ``preprocess_did``, the one-call
form of the preprocessing builder.

.. code-block:: python

   import moderndid as did
   import numpy as np

   data = did.load_mpdta()
   columns = {
       "yname": "lemp",
       "tname": "year",
       "gname": "first.treat",
       "idname": "countyreal",
       "xformla": "~ lpop",
   }
   result = did.att_gt(data, **columns, boot=False, cband=False)
   prepared = did.preprocess_did(data, **columns)

Splitting the units into shards
-------------------------------

Before any statistic can be computed block by block, the data has to be cut
into blocks. A sum over units adds back up from its parts only when each unit
sits in exactly one block along with everything the estimator reads about it.
The engine therefore cuts between units and never between the periods of one
unit. Each block, called a shard, holds a set of whole units as NumPy arrays of
their outcomes, weights, and covariates in every period. However the units are
dealt out, a statistic that sums over units then adds up across the shards to
its value on all the data.

Keeping every period of a unit together also limits the panel engine to
balanced panels. A panel passed with ``allow_unbalanced_panel=True`` still goes
to this engine when all of its units turn out complete, since preprocessing
then leaves ``config.panel`` True in memory as well. We'll deal the 500
prepared counties into three shards with ``shards_from_did_data``, the function
that the engine's parity tests use to build theirs. Each shard should hold
about a third of the counties with all five years of each.

.. code-block:: python

   from moderndid.did.shards.local import shards_from_did_data

   shards = shards_from_did_data(prepared, 3, random_state=0)

   print("Units per shard:", [len(shard.unit_keys) for shard in shards])
   print("Outcomes of the first shard:", shards[0].y.shape)
   print("Covariates of the first shard:", shards[0].covariates.shape)

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         Units per shard: [167, 167, 166]
         Outcomes of the first shard: (167, 5)
         Covariates of the first shard: (167, 5, 1)

Each of the 500 counties went whole to one shard and kept its outcome in all
five years. The covariate array leaves out the intercept because the engine
adds a column of ones whenever it builds a design.

Statistics that add up across shards
------------------------------------

With the counties split, the simplest statistic to rebuild is the mean outcome
of each year. No shard can compute that mean alone because each holds only a
third of the counties. Each shard can still count its own units and total their
outcomes in every year. Added across the shards, those counts and totals become
the ones for all 500 counties. Dividing the totals by the count then gives each
year's mean.

The code below writes that computation as a kernel, the engine's name for a
function that runs on one shard. The kernel hands its shares back in a
``Partial``, a container of named statistics that combine across shards.
Running a kernel on every shard and combining the partials makes one pass of
the engine. ``LocalExecutor`` runs passes over shards held in memory, such as
these three. Every kernel also receives a state that carries what earlier
passes found, such as the current coefficients of a logit. This kernel needs
nothing from earlier passes and gets None. The means it gives should match the
ones computed from the prepared arrays of all 500 counties.

.. code-block:: python

   from moderndid.did.shards.container import Partial
   from moderndid.did.shards.local import LocalExecutor


   def outcome_sums(shard, state):
       # Each shard counts its units and sums their outcomes in every period.
       sums = {"units": len(shard.unit_keys), "outcomes": shard.y.sum(axis=0)}
       return Partial(sums=sums, maxima={}, factors={}, keyed={})


   # The executor runs the kernel on every shard and adds up the sums.
   totals = LocalExecutor(shards).run(outcome_sums, None)
   sharded_means = totals.sums["outcomes"] / totals.sums["units"]

   # The same means from the prepared arrays, which hold every unit at once.
   memory_means = np.array([outcomes.mean() for outcomes in prepared.outcomes_tensor])

   print("Mean outcome by year:", np.round(sharded_means, 4))
   print("Means agree:", np.allclose(sharded_means, memory_means))

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         Mean outcome by year: [5.7985 5.744  5.7535 5.7764 5.7903]
         Means agree: True

The means from the three shards agree with those of all 500 counties at once,
as they must when every county counts in exactly one shard. A kernel defined in
a session like this one works only through ``LocalExecutor``, though. The
section on :ref:`tasks of a Spark pass <dev-spark-tasks>` explains why a kernel
that runs on Spark lives in an importable module.

Beyond means, the doubly robust estimator fits an outcome regression and a
propensity score logit for each group-time effect. A regression could be summed
across shards too, since each shard can add up its own :math:`X'X` and
:math:`X'y` for the driver to solve as normal equations. On an ill-conditioned
design, though, their coefficients drift from the in-memory ones because those
sums square its condition number.

Each shard therefore factors its own rows of the design as :math:`QR` and
returns the pair :math:`(R, Q'B)` for its outcomes :math:`B` instead. Stacking
two pairs and factoring again gives the pair for all their rows in a
tall-skinny QR factorization. The merge works because :math:`R'R` equals
:math:`X'X` for the rows behind each :math:`R`. Stacked factors therefore carry
everything their rows would without ever forming :math:`X'X`. The first year's
regression of the outcome on log population below should come out the same
from three shards as from all the counties at once.

.. code-block:: python

   def first_year_factors(shard, state):
       # QR factors of the first year's regression of the outcome on log population.
       design = np.column_stack([np.ones(len(shard.unit_keys)), shard.covariates[:, 0]])
       q, r = np.linalg.qr(design)
       return Partial(sums={}, maxima={}, factors={"ols": (r, q.T @ shard.y[:, [0]])}, keyed={})


   # Stacking the three shards' factors and factoring again gives those of all the units.
   r, qtb = LocalExecutor(shards).run(first_year_factors, None).factors["ols"]
   sharded_coefficients = np.linalg.solve(r, qtb).ravel()

   # The same regression on all 500 counties at once.
   memory_coefficients = np.linalg.lstsq(
       prepared.covariates_tensor[0], prepared.outcomes_tensor[0], rcond=None
   )[0]

   print("Coefficients:", np.round(sharded_coefficients, 4))
   print("Coefficients agree:", np.allclose(sharded_coefficients, memory_coefficients))

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         Coefficients: [2.2067 1.0842]
         Coefficients agree: True

The three shards and the full data give the same intercept and slope for log
population. The driver solves every outcome regression of a fit from factors
merged this way. The propensity score logits take the same route, because
iteratively reweighted least squares turns each Newton step of a logit into a
weighted regression. Merged factors also decide which propensity score
covariates a fit keeps, since a covariate that adds nothing over the earlier
ones leaves the logit without a unique solution. The driver hands each fit's
merged :math:`R` to ``_independent_columns``, the rule that the in-memory
estimators apply to their own QR factor.

Partials carry two more kinds of statistics, maxima and keyed sums. Maxima,
such as each fit's largest propensity score, combine across shards by keeping
the larger of two values. Clustered standard errors need the other kind, sums
within each cluster rather than over all the units. Since a cluster's units can
sit in several shards, no single shard sees a whole cluster. A partial
therefore keeps one row of sums for each cluster that its shard touched, filed
under that cluster's key. The key is a 64-bit number computed from the
cluster's identifier, as :ref:`random draws on shards <dev-shard-draws>`
explains. Combining two partials adds the rows that share a key. Once every
shard's partial is in, each row holds the sum of its whole cluster.

.. admonition:: A split cluster becomes two clusters
   :class: warning

   A keyed sum is final only once every unit of its key has arrived. If
   ``finalize_keyed`` turns it into a Gram matrix before then, a cluster whose
   units sit in two partials counts as two clusters.

Passes of a fit
---------------

A whole fit can't happen in one pass because some statistics depend on others.
The doubly robust estimator's sums need the propensity scores, for example.
Those scores in turn need logit coefficients that settle only after several
Newton steps. A fit on shards therefore goes back and forth between the shards
and the driver. In each pass the shards compute partials from the current
state. The driver then turns the combined partial into the state of the next
pass, such as the logits' updated coefficients.

``fit_att_gt_on_shards`` runs that loop for the panel engine and should rebuild
the in-memory ``result`` from ``LocalExecutor`` and the county shards. The code
below compares the two fits and counts the passes that the shard fit took.

.. code-block:: python

   from moderndid.did.shards.driver import fit_att_gt_on_shards

   # The fit runs every pass on the shards through the executor.
   executor = LocalExecutor(shards)
   settings = {"bootstrap": False, "uniform_bands": False}
   sharded = fit_att_gt_on_shards(executor, prepared.config, prepared.cohort_counts, None, settings)

   print("Passes over the shards:", executor.n_passes)
   print(f"Largest gap in the effects: {np.abs(sharded.att_gt - result.att_gt).max():.1e}")
   print(f"Largest gap in the standard errors: {np.abs(sharded.se_gt - result.se_gt).max():.1e}")

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         Passes over the shards: 9
         Largest gap in the effects: 2.8e-17
         Largest gap in the standard errors: 1.4e-17

The effects and standard errors match the in-memory ``result`` up to the last
bits of rounding. Before its first pass, the fit planned which models those
effects would need.

Each group-time effect gets a cell of its own, made of a treated cohort, its
comparison cohorts, and two periods. ``plan_cells`` chooses them with the same
``cell_spec`` that the in-memory fit uses. Cells that agree on the treated
cohort, the comparison cohorts, and the earlier of their two periods fit their
models on the same units, covariates, and weights. ``plan_cells`` therefore
gives each such group one propensity score fit and one regression design.
Every pass then does that work once for the group rather than once for each
cell. On the county panel, the 12 cells share 8 fits.

.. admonition:: Key new options into the shared fits
   :class: important

   A new option that changes which units, covariates, or weights a cell uses
   belongs in the key that ``plan_cells`` groups cells by. Leaving it out
   fails no test unless a parity test covers the option.

The figure below follows the doubly robust fit's nine passes in order. After a
summary pass, six logit passes each ran ``irls_kernel`` for one Newton step of
the propensity score logits. A cell sums pass and a moments pass then finished
the fit.

.. container:: mdid-diagram

   .. raw:: html
      :file: diagrams/shard-passes.svg

.. rst-class:: mdid-caption

The passes of a fit on shards. In each pass, the driver sends one kernel to
every shard and solves what the next pass needs from the sums that come back.

The pass count depends on the estimator, since an outcome regression fit has no
logit to iterate and makes only three passes. Where there is a logit, the
passes repeat the in-memory fit iteration by iteration, because each iteration
whose scores separate the units adds a separation warning. The engine replays
a shared fit's warnings for every cell that uses it, both in the overlap screen
that skips a cell whose propensity score reaches 0.999 and at the cell's
estimator, just as the in-memory fit warns at both.

Counting passes matters because on Spark every pass takes a fixed time to
schedule on top of its work. Spark keeps a table in partitions, blocks of rows
that it processes in parallel, and runs a pass as one task on every partition.
Each task builds the factors and sums of its own units, work that grows with
its rows. What it sends back depends instead on the numbers of fits, cells,
coefficients, draws, and clusters. For a fit with :math:`k` coefficients, the
summary pass and each logit iteration return a :math:`k \times k` triangular
factor. Only the moments pass returns statistics that grow with the number of
effects and draws, as the next section shows.

.. admonition:: Add statistics to an existing pass
   :class: important

   Because every pass adds its fixed time, a new statistic that needs the final
   logit coefficients belongs in ``cell_sums_kernel`` or ``moments_kernel``
   rather than in a pass of its own.

After the moments pass, ``_finish_att_gt`` turns the effects and their
covariance into the result, just as it does for an in-memory fit.

Inference from sums over units
------------------------------

The standard errors bring back the problem from the introduction, because they
come from influence functions that hold a value for every unit and effect. A
variance never needs those values one unit at a time, though. The covariance
of two effects averages the products of their influence functions over the
units, a sum like any other on this page. The moments pass therefore stacks
each unit's influence functions into a row :math:`z_i` and returns the sums of
products in the display below. The row also holds the unit's first-period
sampling weight in its cohort's column and a constant. Those columns let an
aggregation add the influence function of the cohort probabilities it
estimates for its weights. Whenever the fit asks for the bootstrap or a
uniform band, the pass also returns a sum :math:`S_b` for each bootstrap draw
:math:`b` with the multipliers :math:`v_{bi}` that :ref:`the next section
<dev-shard-draws>` explains. A uniform band needs those draws even when
``boot=False`` because ``aggte`` inherits ``cband`` and draws for its band.
The figure below shows what stays on each shard and what reaches the driver.

.. container:: mdid-diagram

   .. raw:: html
      :file: diagrams/shard-moments.svg

.. rst-class:: mdid-caption

The moments pass of the county fit. Each shard keeps its unit rows and returns
only sums whose size doesn't depend on its number of units. The dashed block S
appears only in fits that draw bootstrap signs.

.. math::

   Z'Z = \sum_{i=1}^{n} z_i z_i', \qquad
   S_b = \sum_{i=1}^{n} v_{bi} z_i, \qquad
   \hat\sigma_c^2 = \frac{c'\,Z'Z\,c}{n^2}

Every aggregated influence function is a linear combination :math:`c'z_i` of
each unit's row. The variance of any aggregation therefore follows from
:math:`Z'Z` alone. A clustered fit uses the Gram matrix of the cluster sums in
its place. The result keeps these sums in an
:class:`~moderndid.InfluenceMoments` in place of the influence function
matrix. Its first block of :math:`Z'Z` should equal ``psi.T @ psi`` for the
in-memory influence function matrix ``psi``. The code below checks that block
and aggregates an event study from the sums alone.

.. code-block:: python

   moments = sharded.influence_moments
   k = len(result.att_gt)

   # The in-memory influence functions and the covariance they give.
   psi = result.influence_func
   n = result.n_units
   contribution_variance = psi.T @ psi / n

   # The influence functions fill the first k entries of each unit's row.
   print("Length of each unit's row:", moments.gram.shape[0])
   print("Covariance agrees:", np.allclose(moments.gram[:k, :k] / n, contribution_variance))

   # An event study aggregates from those sums alone.
   sharded_study = did.aggte(sharded, type="dynamic")
   memory_study = did.aggte(result, type="dynamic")
   print("Event study errors agree:", np.allclose(sharded_study.se_by_event, memory_study.se_by_event))

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         Length of each unit's row: 16
         Covariance agrees: True
         Event study errors agree: True

The 16 entries of each row are the 12 influence functions, the weights of the
three treated cohorts, and the constant. The event study agrees because both
kinds of fit aggregate through the same coefficient columns :math:`c`. In
``compute_aggte.py``, ``_UnitLevelEvaluator`` applies them to the unit rows of
an in-memory fit and ``_MomentEvaluator`` to the sums of a fit on shards.

.. admonition:: Change an aggregation through its coefficients
   :class: important

   A change to an aggregation belongs in its coefficient columns, never in code
   that reads ``influence_func``, ``G``, or ``weights_ind`` directly. A fit on
   shards leaves all three of those fields as None.

These sums make the moments pass return a fit's largest partials, about
:math:`8L(L + B)` bytes from each task for :math:`B` draws and the :math:`L`
entries of :math:`z_i`. Keyed sums add about :math:`8(L + 1)` bytes for each
cluster a task touched. The setup guide's :ref:`troubleshooting section
<spark-troubleshooting>` covers the limits that Spark puts on them.

.. _dev-shard-draws:

Random draws on shards
----------------------

Unlike every statistic so far, the multiplier bootstrap behind ``boot=True`` and
the simultaneous bands needs random numbers. In each of its ``biters`` draws,
every unit gets a Rademacher sign :math:`v_{bi}`, +1 or −1 with equal chance.
The draw multiplies each unit's row by its sign and adds up the products into
:math:`S_b`. In a clustered fit, all the units of a cluster share one sign.

The hard part is giving each unit the same sign whichever task holds it. In
memory, ``multiplier_bootstrap`` takes the signs from one random stream and
hands them out in the order of the units or clusters. A Spark table has no such
order because its rows fall into partitions that change with the machines and
the plan that Spark adapts while a job runs. A sign handed out
by position could then go to a different unit whenever the partitions change.

The engine instead computes each sign from the seed, the draw, and the
identity of its unit or cluster, three inputs that no partitioning can change.
Like a hash, the computation turns the same inputs into the same sign wherever
it runs. ``canonical_keys`` captures that identity as a 64-bit key that
depends only on the identifier's value.

.. admonition:: How identifiers become keys
   :class: note

   An integer and a float that hold the same whole number both map to the bits
   of that integer. Any other float maps to the bits of its floating-point
   value. A string maps to its 8-byte BLAKE2b digest because Python salts its
   built-in ``hash`` anew in every process.

Turning a key into signs takes a generator that can jump straight to any
unit's draws. An ordinary generator steps through its sequence one number
after another. A counter-based generator instead maps a key and a counter
straight to random bits. ``keyed_rademacher`` feeds the seed and each unit's
key to Philox4x32-10, the counter-based generator of `Salmon et al. (2011)
<https://doi.org/10.1145/2063384.2063405>`_. Its counter holds the unit's key
and the index of a block of 128 draws. Each draw in the block reads its sign
from one of the 128 bits that come out, +1 for a set bit and −1 for a clear
one. Any task can therefore compute the sign of any unit in any draw without
stepping through the draws before it.
`Chamandy et al. (2012)
<https://research.google/pubs/estimating-uncertainty-for-massive-data-streams/>`_
key the bootstrap weights of each user in a sharded data stream to the user's
identifier for the same reason.

The figure below traces one sign from its inputs. Its path through row values
and ``row_keys`` serves repeated cross-sections without clusters.
:ref:`Rows on shards <dev-row-shards>` explains how those rows get their keys.

.. container:: mdid-diagram

   .. raw:: html
      :file: diagrams/keyed-draws.svg

.. rst-class:: mdid-caption

The sign of county 8001 in draw 500 under seed 7. Detail A magnifies the word
of 32 draws that holds it.

The code below draws the signs of the 500 counties and checks that they ignore
the order of the counties, the type of their identifiers, and the number of
draws.

.. code-block:: python

   from moderndid.core.bootstrap import canonical_keys, keyed_rademacher

   # The signs of the 500 counties in 1,000 draws from seed 7.
   counties = prepared.time_invariant_data["countyreal"].to_numpy()
   signs = keyed_rademacher(canonical_keys(counties), 7, 1000)

   # The same seed with the counties reversed, stored as floats, or drawn 300 times.
   reversed_signs = keyed_rademacher(canonical_keys(counties[::-1]), 7, 1000)
   float_signs = keyed_rademacher(canonical_keys(counties.astype(float)), 7, 1000)
   first_draws = keyed_rademacher(canonical_keys(counties), 7, 300)

   print("Same signs in reverse order:", np.array_equal(signs[::-1], reversed_signs))
   print("Same signs from float identifiers:", np.array_equal(signs, float_signs))
   print("Same first 300 draws:", np.array_equal(signs[:, :300], first_draws))

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         Same signs in reverse order: True
         Same signs from float identifiers: True
         Same first 300 draws: True

Every partitioning and every rerun of a task therefore gives the same draws.
Because draw :math:`b` always comes from block :math:`\lfloor b / 128 \rfloor`,
an aggregation with a smaller ``biters`` sees the same signs as a fit run with
that ``biters``.

.. admonition:: Seeds by partition change the draws
   :class: warning

   A new random draw on shards takes its signs from ``keyed_rademacher`` on
   ``canonical_keys``. It never seeds them by partition or by task, as Spark's
   ``rand`` function does, or keys them by Python's ``hash`` of a string.

A seed should also give the same signs on every machine of a cluster, whatever
NumPy version each one runs. ``keyed_rademacher`` therefore computes Philox in
NumPy's integer arithmetic rather than through a NumPy ``Generator``.
``tests/core/test_bootstrap.py`` pins the generator to the known-answer vectors
of the Random123 library. A change to the signs fails tests in that file and
changes every bootstrap number that a Spark fit reports for a seed.

Every pass on this page reads shards whose units hold an outcome in every
period. The guide to :doc:`rows on shards <row_shards>` shows how the same
passes, sums, and keyed draws fit repeated cross-sections and unbalanced
panels.
