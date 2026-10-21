.. _architecture:

Architecture and API design
===========================

When you're working on an estimator, you need to know how an analysis moves
from the user's data to the result they see. An unexpected estimate can
come from the calculation itself or from choices about which rows survive
preprocessing, how units are aligned, and how uncertainty is calculated.
We'll follow a call to :func:`~moderndid.att_gt` to see where each part of
that work happens and where a change belongs. Other estimators can take
different paths through the package depending on which pieces their
comparisons need.

Following a public call
-----------------------

We'll use a small staggered-adoption fit to follow the interfaces that a
contributor encounters. It reuses the column names from the
:doc:`first analysis <../user_guide/quickstart>` so you can focus on how the
package prepares and stores the calculation. Analytical inference keeps the
standard errors tied to the variance calculation that a later section inspects.

.. code-block:: python

   import moderndid as did

   data = did.load_mpdta()
   columns = {
       "yname": "lemp",
       "tname": "year",
       "gname": "first.treat",
       "idname": "countyreal",
       "xformla": "~ lpop",
   }
   result = did.att_gt(data, **columns, boot=False, cband=False)

Inside that call, ``moderndid/did/att_gt.py`` checks the arguments,
constructs a ``DIDConfig``, and passes the data through
``PreprocessDataBuilder``. The resulting ``DIDData`` goes to
``compute_att_gt`` for the group-time comparisons. Back in ``att_gt``, the
influence functions supply inference before the function returns an
``MPResult``.

You can follow that path in the `did source directory
<https://github.com/jordandeklerk/moderndid/tree/main/moderndid/did>`_.
The main files divide the work according to what you need to investigate.

.. container:: architecture-table

   .. list-table::
      :widths: 35 65
      :header-rows: 1

      * - Location
        - What happens there
      * - ``core/preprocess/``
        - Checks, transformations, configuration, and containers prepare the
          sample for numerical work.
      * - ``did/compute_att_gt.py``
        - Each eligible group-time comparison produces an effect and its
          influence function.
      * - ``did/att_gt.py`` and ``did/mboot.py``
        - The public function assembles effects and inference into ``MPResult``.
          The bootstrap helper computes standard errors and critical values
          from the influence functions.
      * - ``did/compute_aggte.py``
        - Aggregation combines the effects and their influence functions into
          summaries for ``aggte`` to return.


Other methods reuse whichever pieces suit the comparisons they make.
The two-period
``drdid`` entry point uses the builder to prepare panel or repeated
cross-section arrays. ``did_multiplegt`` selects a pipeline that prepares
switch dates and treatment histories. Multi-period ``ddd`` applies shared
transformers directly in its own preparation function. The ``etwfe``
implementation cleans its sample and constructs its regression in
``etwfe/compute.py``.
Array-based estimators and sensitivity routines can begin with numerical
inputs instead of a panel container.

Preparing the data
------------------

Each effect is calculated on the sample that preprocessing leaves behind.
The preprocessing code keeps the input names, the
retained sample, and the numerical arrays together so each comparison can
use an already prepared data layout. If you're tracing a change in the
sample or the order of units, this is where to start.

From input names to a configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At the data boundary, ``to_polars`` in ``core/dataframe.py`` passes
a Polars DataFrame through directly and converts other accepted inputs
through Narwhals. An input with an ``__arrow_c_stream__`` interface converts
through that interface. An eager DataFrame without it, such as a pandas
DataFrame older than 2.2, converts only when Narwhals supports its library.
Being called a DataFrame is therefore not enough to make an arbitrary object
valid input.
Once the data reach this boundary, shared selection, filtering, joins, and
sorting use Polars. Numerical estimators then work on arrays that an adapter
prepares for their calculation or dependency.

The public function records its choices in a configuration dataclass from
``core/preprocess/config.py``. While ``DIDConfig`` and ``ContDIDConfig`` share
the base class ``BasePreprocessConfig``, the two-period, triple-difference,
intertemporal, and dynamic balancing configurations build directly on
``ConfigMixin``. A configuration describes the information its method needs
rather than providing one universal set of options.

.. admonition:: Skipping ``validate`` skips the checks
   :class: warning

   A dataclass annotation alone doesn't convert or check the value you pass.
   For a ``DIDConfig``, the builder's ``validate`` stage rejects a comparison
   group, base period, or estimation method outside its accepted values and
   lets ``transform`` check how the rows fit together. Without ``validate`` in
   the chain, a misspelled option or a duplicate unit-period observation
   therefore goes unchecked.

To inspect the prepared sample directly, we can pass the same column names
to ``DIDConfig`` and run the builder ourselves with the public estimator's
default balanced-panel setting. Since ``validate`` stores the comparison
group as a ``ControlGroup`` member, the configuration can take the same
plain string that ``att_gt`` accepts.

.. code-block:: python

   from moderndid.core.preprocess import DIDConfig, PreprocessDataBuilder

   config = DIDConfig(
       **columns,
       allow_unbalanced_panel=False,
       control_group="nevertreated",
   )

   prepared = (
       PreprocessDataBuilder()
       .with_data(data)
       .with_config(config)
       .validate()
       .transform()
       .build()
   )

Calling the builder directly is useful when you're investigating the
prepared sample. An analysis should still enter through the public function
so its argument checks, inference, and result construction also run.

Checks on the rows that remain
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The builder checks the specification before it checks the structure of
the rows that remain after missing-data handling. ``with_config`` selects
the validators and transformer pipeline for the configuration type. The
builder's ``validate`` stage checks the named columns, their types,
reserved names, the values of categorical options, and limits on arguments
such as the number of cluster variables. Structural checks run later,
immediately after ``MissingDataHandler`` in pipelines that contain it.
That means a duplicate unit-period observation or a cohort that changes
within a unit is assessed on the rows that survive missing-data handling.

For ``DIDConfig``, ``DataTransformerPipeline.get_did_pipeline`` selects the
following transformations and applies them in this order.

.. code-block:: text

   ColumnSelector
   MissingDataHandler
   WeightNormalizer
   TreatmentEncoder
   EarlyTreatmentFilter
   ControlGroupCreator
   PanelBalancer
   RepeatedCrossSectionHandler
   DataSorter

Because columns outside the specification are removed first, they don't
affect which rows the missing-data step retains. Weights on the retained
rows are checked and divided by their mean before cohort encoding and
filters establish which units can enter a comparison. After panel
balancing chooses the layout, the final sort puts units in the order
that array construction expects.

Because these decisions depend on the estimator, the intertemporal pipeline can
retain missing outcomes or treatment values for later handling instead of
dropping every incomplete row. Dynamic balancing also keeps missing
outcomes, treatment values, and covariates for later handling while removing
rows whose unit or period identifiers are missing. Its structural checks
run after that removal, before pooling and filtering establish the
histories available for estimation. Use the selected
pipeline in `transformers.py
<https://github.com/jordandeklerk/moderndid/blob/main/moderndid/core/preprocess/transformers.py>`_
as the source for an ordering change rather than assuming every method
runs the sequence above.

.. admonition:: Read the resolved configuration
   :class: tip

   Since preprocessing updates the configuration you passed to the builder,
   inspect ``prepared.config`` for the retained periods, cohorts, unit
   count, and data layout rather than inferring them from the original
   arguments or raw data.

Arrays for the numerical calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At the end of transformation, a configuration updater records the sample's
periods and counts. For ``DIDConfig``, the ``build`` step delegates array
construction to ``TensorFactorySelector`` in ``core/preprocess/tensors.py``. A
balanced panel gets one outcome array per period together with covariate and
weight arrays aligned to the same units. A layout check verifies that the
sorted period blocks list units in the same order before these arrays are
built. An unbalanced panel or repeated cross-section keeps the row-based
layout its calculation requires.

You can see the distinction between outcome rows and paired units by
inspecting the prepared container. The per-period arrays use a common
unit axis even though the retained data contain several rows per unit.

.. code-block:: python

   print("Paired units:", prepared.config.id_count)
   print("Periods:", prepared.config.time_periods.tolist())
   print("Retained rows:", prepared.data.height)
   print("First-period outcomes:", prepared.outcomes_tensor[0].shape)
   print("First-period covariates:", prepared.covariates_tensor[0].shape)
   print("First-period weights:", prepared.weights_tensor[0].shape)

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         Paired units: 500
         Periods: [2003, 2004, 2005, 2006, 2007]
         Retained rows: 2500
         First-period outcomes: (500,)
         First-period covariates: (500, 2)
         First-period weights: (500,)

Across five periods, the 500 units account for the 2,500 retained rows.
Each period's arrays hold one outcome per unit and two covariate columns for
the intercept and ``lpop``. While matching shapes make these arrays
compatible, only the layout check establishes that their positions refer to
the same units.

``DIDData`` retains the processed Polars data and arrays alongside
``time_invariant_data``, summary counts, weights, and configuration. The field
definitions in `models.py
<https://github.com/jordandeklerk/moderndid/blob/main/moderndid/core/preprocess/models.py>`_
help you choose what a numerical helper should accept for another design.

.. admonition:: Containers for other designs
   :class: note

   ``TwoPeriodDIDData`` provides paired ``y0`` and ``y1`` arrays for a panel or
   ``y`` and ``post`` for repeated cross-sections. For triple differences,
   ``DDDData`` also carries eligibility and subgroup membership for its
   two-period comparison.

.. _consistent-argument-naming:

Giving arguments a consistent meaning
-------------------------------------

As you move between estimators, familiar argument names should help you
recognize what belongs in a specification. Most estimators share a name for
the same idea, such as
``yname`` for the outcome column, ``tname`` for the period, ``idname`` for
the unit identifier, and ``weightsname`` for sampling weights.
``xformla`` describes the covariates through a formula rather than naming
a precomputed matrix.

Treatment arguments need more context because different designs distinguish
adoption dates from observed groups and doses. In ``att_gt`` and
``cont_did``, ``gname`` identifies the first treatment period.
``did_multiplegt`` takes no ``gname`` and uses ``idname`` to identify the
group observed over time. ``dname`` records a dose in continuous DiD and the
treatment path in the intertemporal estimator. In triple differences,
``pname`` records which units are eligible to receive treatment.
A lower-level two-period helper may instead accept arrays named ``y0``,
``y1``, and ``d`` because its caller has already selected the comparison.

Inference arguments keep one name for each idea as well. An estimator that
takes a significance level names it ``alpha`` and reads it as a probability
such as 0.05. The number of bootstrap draws is ``biters`` in every estimator
that bootstraps. Because each method quantifies uncertainty differently,
clustering may use ``clustervars``, ``cluster``, or the regression's ``vcov``
specification.

.. admonition:: Check the signature first
   :class: tip

   Before copying an option from another estimator or extending a function,
   check its signature and preserve the meanings established by that interface.

Computing effects and uncertainty
---------------------------------

With the sample prepared, ``compute_att_gt`` forms group-time tasks and
passes them to ``parallel_map``. Each task selects its treatment and
comparison units, determines its base period, and calls the appropriate
two-period estimator. All contributions use the same unit indexing in the
influence-function matrix so the public wrapper can calculate covariances
between effects from different comparisons.

The returned influence functions also carry information that an effect's
standard error alone cannot supply. ``aggte`` combines the group-time
influence functions and accounts for estimated aggregation weights where
its target requires them. The multiplier bootstrap perturbs those
contributions without refitting the nuisance models for every draw.
Since a weighted bootstrap can repeat estimation on a resampled or
reweighted sample, the inference path belongs to the method as well as
the data layout.

For ``att_gt``, the dense influence-function matrix :math:`\Psi` has one
row per unit and one column per returned group-time effect. We can recover
the unit-level analytical covariance and standard errors from the
contributions stored in the fit. Keeping both divisions by :math:`n` visible
helps distinguish
the contribution variance from the covariance of the estimates.

.. code-block:: python

   import numpy as np

   psi = result.influence_func
   n = result.n_units
   contribution_variance = psi.T @ psi / n
   covariance = contribution_variance / n
   analytical_se = np.sqrt(np.diag(covariance))

   print("Influence matrix:", psi.shape)
   print("Stored scale agrees:", np.allclose(contribution_variance, result.vcov_analytical))
   print("Standard errors agree:", np.allclose(analytical_se, result.se_gt))

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         Influence matrix: (500, 12)
         Stored scale agrees: True
         Standard errors agree: True

Each of the 12 columns corresponds to the effect labeled by ``result.groups``
and ``result.times`` at that position. The standard errors agree because this
fit uses analytical inference at the unit level. This calculation does not
account for correlation between counties in the same state. Bootstrap
inference can replace the reported standard errors and critical value without
changing the meaning of ``vcov_analytical``. Its standard errors need not
reproduce the analytical diagonal above.

.. admonition:: Preserve the influence-function scale
   :class: important

   Before passing influence functions to another helper, check the
   normalization and the meaning of each row. A matrix of individual
   contributions and an already scaled covariance matrix need different
   sample-size adjustments. A mismatch changes standard errors even if
   the point estimates remain the same.

Keeping results useful after estimation
---------------------------------------

After a fit, the estimates are only part of what you'll need for
aggregation or sensitivity analysis. ``MPResult`` holds the group and
period labels, estimates, standard errors, critical value, analytical
variance, and influence functions along with the metadata and unit-level
information needed by aggregation. A fit on a Spark DataFrame holds
``influence_moments`` in place of the influence functions and unit-level arrays.
A dose-response curve or a sensitivity confidence set has a different
collection of fields because it answers a different question.

Result containers commonly use ``NamedTuple`` classes in the estimator's
``container.py`` so downstream code can rely on explicit named fields.
Although fields cannot be reassigned, arrays and dictionaries inside a
result remain mutable. When you need working copies, ``_replace`` creates
another container and lets you copy the fields you intend to change.

.. code-block:: python

   working = result._replace(
       att_gt=result.att_gt.copy(),
       estimation_params=result.estimation_params.copy(),
   )

Here the effect array and outer metadata dictionary are independent copies.
If you transform the effects, their labels, influence functions, and
uncertainty also need to describe the transformed target.

.. admonition:: Unreplaced fields stay shared
   :class: warning

   Unreplaced fields and objects nested inside the metadata dictionary remain
   shared with the original result and need their own copies before an operation
   changes them. Keeping that boundary explicit prevents a transformation from
   altering the fit that a plot, aggregation, or saved result still uses.

Document each public field with a ``#:`` comment and a full numpydoc
``Attributes`` description. Sphinx uses the comment for the field summary
and the description for the container's reference page. A field's shape,
units, ordering, and normalization often explain more to a contributor
than its Python type alone.

The metadata helps later code interpret those fields according to the
choices made during estimation. For example,
``MPResult.estimation_params`` records the comparison group, inference
settings, and base period for downstream aggregation and display. It is
not a complete copy of the original call or a substitute for saving an
analysis specification. If you're writing code that reads another result
type, check which metadata it stores and require the information your
calculation needs.

Connecting a result to reports and plots
----------------------------------------

When you print a result, turn it into a table, or plot it, you shouldn't
need to know how its fields are stored. Formatting, conversion, plotting, and
table interfaces connect results to each of those uses. Each connection is
added where the result contains the statistical information that use needs.

Printed reports
^^^^^^^^^^^^^^^

An estimator's formatter takes a result and returns the string for its
printed report. Helpers from ``core/format.py`` supply titles, effect tables,
significance notes, and footers. The code below repeats the existing
registration from ``did/format.py`` so you can see how ``attach_format``
connects the formatter to its result class.

.. code-block:: python

   from moderndid.core.format import attach_format
   from moderndid.did.container import MPResult
   from moderndid.did.format import format_mp_result

   attach_format(MPResult, format_mp_result)

``attach_format`` makes both ``__repr__`` and ``__str__`` use that formatter.
Because ``did/__init__.py`` imports the formatting module, the registration
already runs when this estimator package is imported. When adding a formatter,
you need that import connection for its registration to run too. Its report
should read the stored inference settings and critical values rather than
construct a different interval for display.

.. _architecture-maketables:

Publication tables
^^^^^^^^^^^^^^^^^^

Results that support publication tables expose the ``__maketables_*__``
interface directly on their container. ``maketables.ETable`` reads those
attributes without asking the estimator to fit again. Although the
estimator's preprocessing can remain in Polars, the coefficient property
returns a pandas DataFrame because the table interface requires that
representation.

.. container:: architecture-table

   .. list-table::
      :widths: 42 58
      :header-rows: 1

      * - Interface member
        - Information exposed
      * - ``__maketables_coef_table__``
        - Coefficient names with ``b``, ``se``, ``t``, ``p``, and available
          confidence interval columns.
      * - ``__maketables_stat__(key)``
        - A model statistic, or ``None`` when that statistic is unavailable.
      * - ``__maketables_depvar__``
        - The outcome label for the table's columns.
      * - ``__maketables_fixef_string__``
        - A fixed-effects description, or ``None`` when it doesn't apply.
      * - ``__maketables_vcov_info__``
        - ``vcov_type`` and ``clustervar`` metadata for inference notes.
      * - ``__maketables_stat_labels__``
        - Optional display labels for statistic keys.
      * - ``__maketables_default_stat_keys__``
        - Optional statistic keys to show when the caller doesn't choose them.


You can inspect this interface on the same result without importing a table
renderer or repeating the fit. Beyond the effects in the coefficient table,
the other members supply the information needed for table notes.

.. code-block:: python

   coefficients = result.__maketables_coef_table__
   print("Coefficient columns:", ", ".join(coefficients.columns))
   print("Units:", result.__maketables_stat__("N"))
   print("Inference:", result.__maketables_vcov_info__)

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         Coefficient columns: b, se, t, p, ci95l, ci95u, ci90l, ci90u
         Units: 500
         Inference: {'vcov_type': 'analytical', 'clustervar': None}

For this fit, the ``ci95*`` and ``ci90*`` columns contain 95 and 90 percent
pointwise intervals. ``core/maketables.py`` supplies the coefficient-table
builders, effect names, and metadata labels used by these properties.
``build_coef_table_with_ci`` can use stored critical values for the fitted
interval columns. Its ``t`` and ``p`` columns still use a pointwise normal
approximation. For a fit with simultaneous bands, those columns therefore
answer different inferential questions.
The :doc:`publication tables guide <../user_guide/publication_tables>` shows
how to present them. The adapters on ``DRDIDResult`` in ``drdid/container.py``
provide another complete implementation when you're extending the interface.

Data conversion and plotting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use fitted effects in another calculation, :func:`~moderndid.to_df`
extracts Polars data through a converter in ``core/converters.py``.
Each row of the group-time result retains the cohort and calendar period
alongside the estimate and its uncertainty. You can inspect those labels
without constructing a plot or going through the publication-table interface.

.. code-block:: python

   estimates = did.to_df(result)
   print("Rows:", estimates.height)
   print("Columns:", ", ".join(estimates.columns))

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         Rows: 12
         Columns: group, time, att, se, ci_lower, ci_upper, treatment_status

``to_df`` selects a converter from ``_DISPATCH`` using the result class's
name. Each converter creates columns its result can support, such as cohort
and calendar period here or event time for a dynamic aggregation.

Plot functions in ``plots/plots.py`` select supported results through
type checks and call the appropriate converter. Adding an entry to
``to_df`` therefore doesn't automatically enable a plotting function.
An event-study plot also checks the aggregation type because a cohort
or calendar-time aggregation can have similar arrays with a different
meaning on the horizontal axis.

When a result stores its critical value, the converter uses it to form
interval bounds. Dynamic aggregations omit normalized reference rows
whose standard errors are undefined.

.. admonition:: Handle reference periods per result
   :class: important

   Preserve the result-specific handling of reference periods and missing
   estimates when extending a converter rather than applying one filter to
   every result type.

The plotting functions return plotnine objects that readers can customize,
as the :doc:`plotting guide <../user_guide/plotting>` demonstrates.

Sensitivity analysis
^^^^^^^^^^^^^^^^^^^^

To carry an event study into sensitivity analysis, the result has to
describe both the coefficients and their joint uncertainty.
``EventStudyProtocol`` in ``didhonest/honest_did.py`` names
``aggregation_type``, ``influence_func``, ``event_times``, ``att_by_event``,
and ``estimation_params``. The public ``honest_did`` wrapper checks those
attributes before passing a result to its event-study implementation.

Even with those attributes present, the result must describe the event
study that the sensitivity calculation expects. That means
``aggregation_type="dynamic"``, a two-dimensional influence
matrix, and a universal base period when that setting is recorded. An event
study without influence functions, such as an aggregation of a Spark fit,
supplies its ``vcov_analytical`` instead and records ``n_units`` in its
``estimation_params``.
The event times must be consecutive on each side of the reference period
and include at least one pre-treatment and one post-treatment coefficient.
Before connecting a new result, check that its coefficients and
uncertainty meet those requirements. The :doc:`sensitivity background <../background/didhonest>`
explains why the common reference period matters.

.. admonition:: Record the clusters
   :class: important

   A clustered event study needs ``clustervars`` in its ``estimation_params``
   along with a ``cluster`` entry that gives each unit's cluster.
   ``honest_did`` reads both to add up the influence function within each
   cluster. An event study without influence functions records ``n_clusters``
   in place of ``cluster``, since its ``vcov_analytical`` already sums within
   clusters.

Choosing how numerical work runs
--------------------------------

Once an estimator is correct, profiling can show whether its time goes to
sample preparation, nuisance fits, array operations, or repeated inference.
We'll look at the shared bootstrap helpers, parallel tasks, and GPU arrays
separately because they affect different parts of the calculation. The
:doc:`benchmarking` guide explains how to measure a change on a specified
workload.

Shared bootstrap helpers
^^^^^^^^^^^^^^^^^^^^^^^^

The bootstrap and variance helpers in ``core/bootstrap.py`` are plain NumPy
functions. On the CPU, ``multiplier_bootstrap`` runs the multiplier bootstrap
of ``att_gt``, ``aggte``, ``cont_did``, and ``ddd``. It draws the Rademacher
weights in blocks and multiplies each block by the influence function matrix.
NumPy passes each of those products to BLAS. A loop over the draws would read
the whole influence function matrix once per draw and run far slower than the
matrix product on large samples. Before writing a new calculation as a loop,
time it against an array version on a large sample.

Three more helpers in the module serve the inference of ``did_multiplegt``.
For its clustered variance, ``compute_cluster_sums`` adds up the influence
functions within each cluster. The cluster bootstrap gathers each draw's rows
with ``gather_bootstrap_indices`` and turns the draws into standard errors
with ``compute_column_std``.

.. admonition:: Change both branches together
   :class: important

   Where a helper in ``core/bootstrap.py`` has a CuPy branch, that branch runs
   the calculation on the GPU when the active backend is CuPy. A change to
   what the NumPy code computes needs the same change in that branch.

Group-time tasks on threads
^^^^^^^^^^^^^^^^^^^^^^^^^^^

When group-time comparisons can be fitted separately, ``parallel_map`` in
``core/parallel.py`` handles their execution. It also runs the bootstrap draws
of ``did_multiplegt`` and the fits that ``dyn_balancing`` makes for several
history lengths or final periods. The default ``n_jobs=1`` runs the tasks one
after another in the calling thread. A worker count above one, or
``n_jobs=-1`` for one worker per reported CPU, runs them on a
``ThreadPoolExecutor``.

The helper calls a function with each tuple of arguments and returns its
results in input order. A small function that fits no model shows the labels
and the backend each task reads.

.. code-block:: python

   from moderndid.core.parallel import parallel_map
   from moderndid.cupy.backend import get_backend, use_backend


   def identify_comparison(group, period):
       return group, period, get_backend().__name__


   tasks = [(2004, 2005), (2007, 2007)]
   with use_backend("numpy"):
       labels = parallel_map(identify_comparison, tasks, n_jobs=2)
   print(labels)

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         [(2004, 2005, 'numpy'), (2007, 2007, 'numpy')]

Both tasks inherit the selected backend because each runs in its own
``contextvars.copy_context`` snapshot. Keeping returned results in input
order lets an estimator attach each influence-function column to its
group-time label even when tasks finish in a different order. Threads share
data in one process and avoid serializing it for separate workers.
Numerical dependencies can release the GIL during their calculations so
several tasks can make progress in parallel.
Assess worker counts on the workload you're changing because additional
workers can compete with the threads used by BLAS.

Array operations on a GPU
^^^^^^^^^^^^^^^^^^^^^^^^^

``cupy/backend.py`` stores the active array backend in a ``ContextVar``.
``get_backend`` returns the array module that ``to_device`` uses when moving
supported arrays. In the other direction, ``to_numpy`` brings them back to CPU
storage. The ``use_backend`` context manager restores the previous setting
even when a calculation raises an exception. An ``att_gt`` call with a ``backend``
argument enters that context before preprocessing and estimation.

Selected regression and bootstrap helpers in ``cupy/`` and tensor creation in
``core/preprocess/tensors.py`` dispatch to the GPU by checking the backend.
Selecting CuPy doesn't move Polars preprocessing or every Python helper to the
GPU. The ``att_gt`` wrapper converts its influence functions to NumPy before
inference and result construction to make the public result usable by CPU
consumers.

CuPy availability and a working CUDA device are checked when the backend
is selected. The ``gpu`` extra is separate from ``all`` because it has
CUDA-specific dependencies.

.. admonition:: Check GPU support per estimator
   :class: tip

   Use the :doc:`GPU guide <../user_guide/gpu>` for supported interfaces and
   installation rather than treating a backend argument on one function as a
   package-wide guarantee.

.. _architecture-shards:

Fitting a panel held in shards
------------------------------

Because a panel held in a Spark DataFrame can have more units than one
machine's memory holds, a fit on it brings neither the rows nor the
influence function matrix to the driver. The engine in ``did/shards/``
therefore computes ``att_gt`` from statistics over shards, blocks of whole
units. The adapter in ``spark/`` runs those computations on Spark and is the
only part of moderndid that imports PySpark. ``att_gt`` recognizes a Spark
DataFrame by its class and only then loads ``spark_att_gt`` from
``spark/att_gt.py``. That check comes before any step that reads the data.
Since a PySpark 4.2 DataFrame exposes ``__arrow_c_stream__``, converting it for
``check_columns`` would collect the whole table to the driver. The
:ref:`Spark guide <spark>` walks through such a fit from the user's side.

Since a shard holds every period of each of its units, any statistic that sums
over units adds up across shards however the units were split. We'll split the
prepared county panel into three shards with ``shards_from_did_data`` from
``local.py`` to see what each shard holds.

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
five years. The covariate array leaves out the intercept, since the kernels add
it back.

Statistics from each shard
^^^^^^^^^^^^^^^^^^^^^^^^^^

A kernel in ``kernels.py`` is a NumPy function ``kernel(shard, state)`` that
returns one shard's statistics as a ``Partial``. A single call handles every
group-time cell and picks out each cell's units through a mask on their
cohorts. ``combine_partials`` in ``partials.py`` merges the partials of two
sets of units by four rules.

- Sums add up.
- Maxima take the elementwise maximum.
- QR factor pairs :math:`(R, Q'B)` of a weighted design and its right-hand
  sides :math:`B` stack and factor again, as in a tall-skinny QR
  factorization.
- Keyed sums, such as sums by cluster, add up the rows that share a key.

``finalize_keyed`` turns the keyed sums of a partial into their Gram matrix and
their number of keys.

.. admonition:: A split cluster becomes two clusters
   :class: warning

   A keyed sum is final only once every unit of its key has arrived. If
   ``finalize_keyed`` runs before then, a cluster whose units sit in two
   partials enters the Gram matrix and the number of keys as two clusters.

Local and Spark executors
^^^^^^^^^^^^^^^^^^^^^^^^^

An executor needs one method, ``run(kernel, state)``, to make a pass over the
shards. A pass applies the kernel to every shard and returns the combined
partial with its keyed sums finalized. ``LocalExecutor`` in ``local.py`` loops
over shards in memory and finalizes once at the end, since a cluster can span
several of its shards.

``SparkExecutor`` in ``spark/executor.py`` runs each pass through
``mapInArrow`` over the unit table that ``build_unit_table`` in
``spark/preprocess.py`` builds with one row for each unit. Each task returns
its statistics as a single row. A task can finalize its own keyed sums because
a clustered fit partitions the unit table by cluster. Each row also carries the
task's moderndid and Python versions for the driver to check. They travel as
data because an exception raised on an executor would reach the driver as a
``PythonException`` rather than as an error that says what to install.

When a pass has more than 16 tasks, a grouped stage combines the rows of the
:math:`P` tasks in about :math:`\sqrt{P}` groups. The driver then collects one
row per group instead of one per task. Because the executor counts the tasks
only once a pass has run, the first pass takes that route too.

Wherever the grouped data has ``applyInArrow``, as on PySpark 4, the stage
hands each group through it to the function that ``_combine_task`` returns.
Elsewhere, as on open-source PySpark 3.5, ``applyInPandas`` calls the same
function through ``_pandas_combine``. That wrapper rebuilds the group as a
pyarrow Table of the fixed schema and returns the combined rows as pandas.
While it combines a group of up to about :math:`2\sqrt{P}` rows, it holds three
copies of their payloads, two in Arrow and one in pandas. The Arrow copies are
the batch that PySpark keeps while the wrapper runs and the table that the
wrapper rebuilds.

Code that runs on Spark keeps to six rules that the tests through
``LocalExecutor`` mostly can't check.

- It calls only DataFrame APIs that Spark Connect supports and never
  ``sparkContext``, RDDs, or broadcast variables. The second stage groups its
  rows with ``groupBy`` since ``repartition(k)`` would need a partition count
  that costs an extra job under Spark Connect.
- It runs on PySpark 3.5 as well as on PySpark 4. Code that needs an API newer
  than 3.5 keeps a 3.5 path beside it and picks between the two by feature,
  never by version. A check of the version would miss the newer APIs that
  Databricks adds to its PySpark 3.5, such as ``toArrow`` from Databricks
  Runtime 15.3 on. ``SparkExecutor.run`` checks the grouped data for
  ``applyInArrow`` this way, as the ``to_arrow`` test fixture checks a
  DataFrame for ``toArrow``. Where PySpark 3.5 takes only a literal, such as
  the value that ``array_position`` looks up, the code passes no Column.
  ``_kept_rows`` in ``spark/preprocess.py`` therefore numbers the periods with
  a CASE WHEN chain. Library code reads the version only in the floor check of
  ``spark/att_gt.py``. CI runs the Spark tests on PySpark 3.5 and 4.2.
- It never sets a session configuration, since Databricks documents that
  serverless compute lets a session set only a few of them.
- Kernels are module-level functions in ``kernels.py`` with NamedTuple states,
  because ``SparkExecutor`` pickles a kernel and the class of its state by
  reference. Since ``LocalExecutor`` never pickles, only ``tests/spark/`` and a
  pickle round-trip test in ``tests/did/`` catch a kernel that can't travel.
- Clusters stay whole in the unit table, since the shuffle hashes on the
  cluster alone and the ``coalesce`` after it only merges whole partitions. A
  repartition by anything but the cluster, or a ``REBALANCE`` hint, would split
  clusters and miscount them without an error.
- The kernel's working memory stays bounded through slices of 4096 rows from
  each batch, blocks of 2048 units in the moments kernel, and blocks of at most
  1024 draws of signs. A clustered task's cluster sums still grow by one row
  for each of its clusters. Serverless compute on Databricks needs that bound,
  since Databricks documents that it sizes the batches itself and caps the
  memory of each Python UDF at 1 GB.

Passes of a fit
^^^^^^^^^^^^^^^

``fit_att_gt_on_shards`` in ``driver.py`` takes an executor and runs the passes
of a fit on it. It plans the cells with the same ``cell_spec`` that
``compute_att_gt`` uses and repeats the in-memory screens and warnings cell by
cell. Cells that agree on the treated cohort, the comparison cohorts, and the
earlier period use the same units, covariates, and weights. ``plan.py``
therefore keys one propensity score fit and one regression design on those
three. A new option that changes which units, covariates, or weights a cell
uses belongs in that key. ``summary_kernel`` makes the first pass and gathers
the counts, weight sums, and QR factors that the fits start from. Since normal
equations can drift from the in-memory coefficients, the driver solves the
outcome regressions from the QR factors of their comparison designs. The
triangular factor also reproduces the condition number and rank checks of the
full design.

The propensity score fits follow the iterations of the in-memory binomial fit.
They start at :math:`\mu = (D + 1/2)/2`, stop once the deviance changes by at
most 1e-8, and run 100 iterations at most. Each iteration takes one pass of
``irls_kernel`` for all the fits at once. Only the first iteration fits the
working response, at the start, from the QR factor that ``summary_kernel``
returns. Since a refit of the working response drifts from the in-memory
coefficients as the design's condition number grows, each pass of
``irls_kernel`` returns the QR factor of the Newton step instead. The driver
solves it and adds the step to the coefficients. The number of iterations
matters beyond the coefficients, because each iteration whose scores separate
the units adds a separation warning.

In memory, each cell fits its logit twice, once in the overlap screen and once
in its estimator. The engine therefore replays a fit's separation warnings once
for every cell that shares it and once more when the cell reaches its
estimator. It also iterates every fit to the end, even one whose cells will all
fail a later check, because the in-memory screen has already fitted it by then.
The overlap screen reads the largest propensity score of the estimator's own
fit rather than fitting a second logit.

Once the fits are final, one pass of ``cell_sums_kernel`` sums the parts of
every effect before ``moments_kernel`` makes the last pass. The result goes
through ``_finish_att_gt``, the same step that assembles an in-memory result.

Inference from sums over units
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The last pass computes the sums over units that an
:class:`~moderndid.InfluenceMoments` holds in place of the influence function
matrix. Besides its influence functions, each unit's row :math:`z_i` holds its
sampling weight in the column of its treated cohort and a constant. That weight
comes from the first period, as it does in an in-memory ``aggte``. From these
columns ``compute_aggte`` recovers the cohort shares and the influence of the
estimated aggregation weights. The pass returns
:math:`Z'Z = \sum_i z_i z_i'` and, when the fit asks for the bootstrap or a
uniform band, the sum :math:`\sum_i v_{bi} z_i` for each draw :math:`b` of the
multipliers. A clustered fit also returns the Gram matrix of the cluster sums
of :math:`z_i`.

Every aggregated influence function is a linear combination :math:`c'z_i` of
each unit's row. ``compute_aggte`` therefore reads the analytical variance of
an aggregation as :math:`c'Z'Zc / n^2`. In a clustered fit, the cluster Gram
matrix takes the place of :math:`Z'Z`. ``honest_did`` then works from the event
study's ``vcov_analytical`` in place of its influence function.

Each aggregation is a set of such coefficient columns over the kept cells, the
weight columns of their cohorts, and the constant. ``_UnitLevelEvaluator``
applies the columns to the unit rows of an in-memory fit. ``_MomentEvaluator``
applies the same columns to the Gram matrix and the stored draws of a fit on
shards. A change to an aggregation therefore belongs in its coefficients, never
in code that reads ``influence_func``, ``G``, or ``weights_ind`` directly,
since a fit on shards has none of them.

Random draws on shards
^^^^^^^^^^^^^^^^^^^^^^

The multiplier bootstrap gives each unit one random sign in every draw, or each
cluster one when the fit clusters. In memory, ``multiplier_bootstrap`` takes
those signs from one random stream in the order of the units or clusters. A
Spark table offers no such order for its rows to follow. How the rows fall into
partitions changes with the number of executors, adaptive query execution, and
the data source. When Spark reruns a task that failed or ran slowly, the new
attempt may also read its rows in another order. A cluster's units, which share
one sign, can sit in several shards as well. Seeding each partition or task, as
the Spark backend that version 0.3.0 removed did, would tie the draws to all of
these.

Each multiplier is instead a pure function of the seed, the draw, and the key
of its unit or cluster. Since no sign depends on where a unit sits or on what
else its task reads, every partitioning and every rerun of a task gives the
same draws. `Chamandy et al. (2012)
<https://research.google/pubs/estimating-uncertainty-for-massive-data-streams/>`_
seed the bootstrap weights of each user in a sharded data stream with a hash of
the user's identifier for the same reason.

.. admonition:: Seeds by partition change the draws
   :class: warning

   A new random draw on shards takes its signs from ``keyed_rademacher`` on
   ``canonical_keys``. It never seeds them by partition or by task, as Spark's
   ``rand`` function does, or keys them by Python's ``hash`` of a string.

Two functions in ``core/bootstrap.py`` compute the multipliers from the
identifiers. ``canonical_keys`` maps each identifier to a 64-bit key that
depends only on its value. Since integers and floats that hold integers share
their keys, an identifier stored either way draws the same signs. Any other
float takes the bits of its value as its key. A string takes its 8-byte BLAKE2b
digest, because Python salts its built-in ``hash`` of a string anew in every
process. ``keyed_rademacher`` then turns the seed and each key into signs with
Philox4x32-10, a counter-based generator from `Salmon et al. (2011)
<https://doi.org/10.1145/2063384.2063405>`_. The
docstrings in `bootstrap.py
<https://github.com/jordandeklerk/moderndid/blob/main/moderndid/core/bootstrap.py>`_
give the exact layout of its key, its counter, and the bits that become signs.

Rather than build a NumPy generator object for every unit, which would be slow,
``keyed_rademacher`` computes the ten rounds of Philox with NumPy's integer
arithmetic. The draws then stay the same even when the executors and the
driver run different versions of NumPy. NumPy doesn't promise such stability
for the numbers that a ``Generator`` returns. ``tests/core/test_bootstrap.py``
pins the generator to the known-answer vectors of the Random123 library.

Because draw :math:`b` of a key always comes from block
:math:`\lfloor b / 128 \rfloor`, the first :math:`b` draws of a fit don't
depend on ``biters``. An aggregation with a smaller ``biters`` therefore reads
the first rows of the stored sums and sees the draws of a fit with that
``biters``. The moments pass stores draws whenever the fit asks for the
bootstrap or a uniform band, since ``aggte`` inherits ``cband`` from the fit
and draws for its band even when ``boot=False``.

.. admonition:: Read the caller's bootstrap settings
   :class: important

   The driver reads ``boot`` and ``cband`` from ``estimation_params``, which
   hold the caller's settings, rather than from the configuration. Since
   preprocessing turns the configuration's ``cband`` off when only two periods
   remain, reading it there would lose the band of a two-period fit.

A non-negative integer ``random_state`` below :math:`2^{64}` is the seed itself.
Any other seed, None included, makes a NumPy ``Generator`` whose first 64-bit
draw becomes the seed. The result stores the seed it used in
``influence_moments.random_state``. Since the draws are fixed once the fit
returns, ``aggte`` on a fit on shards can bootstrap with at most the stored
number of draws, only the stored seed, and only the fit's own clustering. Its
result holds the covariance of its effects in ``vcov_analytical`` but no
influence functions. The price of keyed draws is that a Spark fit and an
in-memory fit with the same ``random_state`` draw different signs. Both sets of
signs give valid draws of the same bootstrap distribution. Their standard
errors differ only by the noise of a finite number of draws.

Agreement with the in-memory fit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We keep the in-memory ``att_gt`` as the reference implementation and check the
engine against it. Because the two paths share the cell plan and the finishing
step but none of the estimators' algebra, agreement between them checks the
engine's sums. ``tests/did/test_shards.py`` runs the engine through
``LocalExecutor`` on 1, 3, and 7 shards and compares its results with
in-memory ``att_gt`` and ``aggte`` across the estimators, comparison groups,
base periods, weights, clusters, and anticipation. The
:ref:`testing guide <testing-spark>` covers the end-to-end tests on Spark.

.. admonition:: Behaviors the engine doesn't copy
   :class: note

   Since every sum adds up in another order, a decision within rounding of its
   threshold, such as the overlap screen at 0.999, can go the other way. The
   engine copies neither the iteration count of an exactly collinear logit,
   which varies from run to run inside statsmodels, nor the warnings and errors
   of the Gaussian GLM in ``reg_did_panel``, nor the RuntimeWarnings that NumPy
   raises inside statsmodels. A Spark fit also lists the cohorts of its
   small-groups warning in ascending order and stops before its first job on
   weights or covariates that aren't numbers.

The kernels and the driver repeat what the in-memory ``att_gt`` computes.
``build_unit_table`` in ``spark/preprocess.py`` runs the in-memory ``validate``
stage itself, on an empty frame with the data's schema, and repeats only the
``transform`` stage in Spark SQL. The driver also hands a few in-memory checks
stand-in arrays built from the kernels' statistics, such as the treated and
comparison weight totals for ``_check_cell_weights`` and the largest and
smallest positive comparison weights for ``_check_extreme_weights``.

.. admonition:: Change the shard path with the in-memory one
   :class: important

   A change to the in-memory estimators or to the ``transform`` stage needs the
   same change on the shard path. A check that starts reading more of its input
   than its stand-in holds needs a new kernel statistic. Only the tests in
   ``tests/spark/`` check ``build_unit_table``, since
   ``tests/did/test_shards.py`` prepares its shards with the in-memory
   preprocessing.

.. _architecture-imports:

Keeping imports and dependencies scoped
---------------------------------------

For a user to reach your function or result type through ``moderndid``,
its public name has to point to the module that defines it.
The root ``__init__.py`` uses ``__getattr__`` to load most names from
``_lazy_imports`` or ``_optional_imports`` when they're requested.
The optional mapping pairs a module path with the extra needed for its
dependencies so an import failure can name an installation command.
Listing a name in ``__all__`` marks it as an intended export without
establishing its lazy import route.

Some function names match estimator subpackages and are imported eagerly
to make ``from moderndid import ...`` resolve to the function. Root
imports are therefore mostly lazy rather than a promise that importing
``moderndid`` loads no estimator code.

.. admonition:: Test imports without the extra
   :class: important

   Adding an optional dependency requires checking both the package import and
   access to the feature in an environment without that extra.

The :doc:`new_estimator` guide develops a worked estimator from its target
through the sample requirements, computation, and inference to the result
a user receives. You can follow those decisions before returning here to
connect the implementation to the package's interfaces.
