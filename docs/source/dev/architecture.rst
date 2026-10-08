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
contributor encounters. The column names below are the same ones used in the
:doc:`first analysis <../user_guide/quickstart>`; here we're interested in how
the package prepares and stores the calculation. Analytical inference keeps
the standard errors tied to the variance calculation we'll inspect later.

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
        - The public function assembles effects and inference into ``MPResult``;
          the bootstrap helper works with influence functions.
      * - ``did/compute_aggte.py``
        - Aggregation combines the effects and their influence functions into
          summaries for ``aggte`` to return.


Other methods reuse the pieces that suit the comparisons they need to make.
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

Before looking at how an effect is calculated, we need to follow what
happens to the data. The preprocessing code keeps the input names, the
retained sample, and the numerical arrays together so each comparison can
use an already prepared data layout. If you're tracing a change in the
sample or the order of units, this is where to start.

From input names to a configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At the data boundary, ``to_polars`` in ``core/dataframe.py`` passes
a Polars DataFrame through directly and converts other accepted inputs
through Narwhals using their ``__arrow_c_stream__`` interface. Being called a
DataFrame is therefore not enough to make an arbitrary object valid input.
Once the data reach this boundary, shared selection, filtering, joins, and
sorting use Polars. Numerical estimators work on arrays once an adapter has
prepared the representation required by their calculation or dependency.

The public function records its choices in a configuration dataclass from
``core/preprocess/config.py``. ``DIDConfig`` and ``ContDIDConfig`` inherit
from ``BasePreprocessConfig``; two-period DiD, triple differences,
intertemporal effects, and dynamic balancing instead have separate
dataclasses using ``ConfigMixin``. A configuration describes the information
its method needs rather than providing one universal set of options.

To inspect the prepared sample directly, we can pass the same column names
to ``DIDConfig`` and run the builder ourselves. The balanced-panel setting
matches the public estimator's default. We also convert the comparison-group
string to an enum explicitly, as the public wrapper does for categorical
choices such as ``ControlGroup`` and ``BasePeriod``. A dataclass annotation
alone doesn't perform that conversion or validate every supplied value.

.. code-block:: python

   from moderndid.core.preprocess import DIDConfig, PreprocessDataBuilder
   from moderndid.core.preprocess.constants import ControlGroup

   config = DIDConfig(
       **columns,
       allow_unbalanced_panel=False,
       control_group=ControlGroup("nevertreated"),
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
reserved names, and argument values. Structural checks run later,
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
periods and counts. For ``DIDConfig``, ``build`` delegates to
``TensorFactorySelector`` in ``core/preprocess/tensors.py``. A balanced
panel gets one outcome array per period together with covariate and weight
arrays aligned to the same units. A layout check verifies that the sorted
period blocks list units in the same order before these arrays are built.
An unbalanced panel or repeated cross-section keeps the row-based layout
its calculation requires.

You can see the distinction between outcome rows and paired units by
inspecting the container we just built. The per-period arrays use a common
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
Each period's outcome array contains one value per unit and its two
covariate columns contain the intercept and ``lpop``.
Matching shapes make these arrays compatible; the layout check establishes
that their positions actually refer to the same units.

``DIDData`` retains the processed Polars data and arrays alongside
``time_invariant_data``, summary counts, weights, and configuration.
``TwoPeriodDIDData`` instead provides paired ``y0`` and ``y1`` arrays for a
panel or ``y`` and ``post`` for repeated cross-sections. For triple
differences, ``DDDData`` also carries eligibility and subgroup membership
for its two-period comparison. The field definitions in `models.py
<https://github.com/jordandeklerk/moderndid/blob/main/moderndid/core/preprocess/models.py>`_
help you choose what a numerical helper should accept for another design.

.. _consistent-argument-naming:

Giving arguments a consistent meaning
-------------------------------------

As you move between estimators, familiar argument names should help you
recognize what belongs in a specification. We reuse names wherever they
have the same meaning, such as
``yname`` for the outcome column, ``tname`` for the period, ``idname`` for
the unit identifier, and ``weightsname`` for sampling weights.
``xformla`` describes the covariates through a formula rather than naming
a precomputed matrix.

Treatment arguments need more context because different designs distinguish
adoption dates from observed groups and doses. In ``att_gt`` and
``cont_did``, ``gname`` identifies the first treatment period; in
``did_multiplegt`` it identifies the group observed over time.
``dname`` records a dose in continuous DiD and the treatment path in the
intertemporal estimator. In triple differences, ``pname`` records which
units are eligible to receive treatment.
A lower-level two-period helper may instead accept arrays named ``y0``,
``y1``, and ``d`` because its caller has already selected the comparison.

Inference arguments also reflect differences in how each method quantifies
uncertainty. Where it is accepted,
``alp`` expresses a significance level; ``did_multiplegt`` instead uses
``ci_level`` for the confidence level in percent. Bootstrap counts are
``biters`` in some methods and ``n_boot`` in ``ddd``. Clustering may use
``clustervars``, ``cluster``, or the regression's ``vcov`` specification.
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
the unit-level analytical covariance and standard errors from the stored contributions
in our fit. Keeping both divisions by :math:`n` visible helps distinguish
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
inference can replace the reported
standard errors and critical value without changing the meaning of
``vcov_analytical``; it need not reproduce the analytical diagonal above.

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
information needed by aggregation.
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
Unreplaced fields and objects nested inside the dictionary remain shared
and need their own copies before an operation changes them. Keeping that
boundary explicit prevents a transformation from altering the fit that a
plot, aggregation, or saved result still uses. If you transform the effects,
their labels, influence functions, and uncertainty also need to describe the
transformed target.

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
need to know how its fields are stored. We connect results to each of
those uses through formatting, conversion, plotting, and table interfaces.
Each connection is added where the result contains the statistical
information that use needs.

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
attributes without asking the estimator to fit again. The coefficient
property returns a pandas DataFrame because that is the table interface's
required representation; the estimator's preprocessing can remain in
Polars.

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
renderer or repeating the fit. The coefficient table describes the effects;
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
interval columns; its ``t`` and ``p`` columns use a pointwise normal
approximation. For a fit with simultaneous bands, those columns therefore
answer different inferential questions.
The :doc:`publication tables guide <../user_guide/publication_tables>` shows
how to present them. The adapters on ``DRDIDResult`` in ``drdid/container.py``
provide another complete implementation when you're extending the interface.

Data conversion and plotting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use fitted effects in another calculation, :func:`~moderndid.to_df`
extracts Polars data through a converter in ``core/converters.py``.
Each row of our group-time result retains the cohort and calendar period
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
whose standard errors are undefined. Preserve the result-specific
handling of reference periods and missing estimates when extending a
converter rather than applying one filter to every result type. The
plotting functions return plotnine objects that readers can customize,
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
matrix, and a universal base period when that setting is recorded.
The event times must be consecutive on each side of the reference period
and include at least one pre-treatment and one post-treatment coefficient.
Before connecting a new result, check that its coefficients and
uncertainty meet those requirements. The :doc:`sensitivity background <../background/didhonest>`
explains why the common reference period matters.

Choosing how numerical work runs
--------------------------------

Once an estimator is correct, profiling can show whether its time goes to
sample preparation, nuisance fits, array operations, or repeated inference.
We'll look at compiled CPU kernels, parallel tasks, and GPU arrays
separately because they affect different parts of the calculation. The
:doc:`benchmarking` guide explains how to measure a change on a specified
workload.

Compiled CPU kernels
^^^^^^^^^^^^^^^^^^^^

``core/numba_utils.py`` defines NumPy implementations and conditionally
replaces selected kernels when Numba is installed. Those kernels handle
operations such as cluster sums, multiplier draws, and column standard
deviations. Numerical loops use arrays and scalars so compilation doesn't
have to handle data selection, result objects, or formulas.

Random draws for the CPU multiplier bootstrap are generated outside the
compiled loop and supplied to the kernel. This separates control of the
random draws from the kernel's numerical accumulation. Keep the fallback and compiled
paths numerically consistent when changing a shared kernel. Measuring
the workload helps establish whether compilation improves that calculation.

Group-time tasks on threads
^^^^^^^^^^^^^^^^^^^^^^^^^^^

When group-time comparisons can be fitted separately, ``parallel_map`` in
``core/parallel.py`` handles their execution.
With ``n_jobs=1``, it runs them sequentially; a positive worker count uses
a ``ThreadPoolExecutor`` and ``n_jobs=-1`` uses the reported CPU count.

The helper calls a function with each tuple of arguments and returns its
results in input order. We can see both the labels and the inherited backend
in a small task that doesn't need to fit a model.

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

Both workers inherit the selected backend through their own
``contextvars.copy_context`` snapshots. Keeping returned results in input
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
``get_backend`` returns the array module for ``to_device`` to use when
moving supported arrays; ``to_numpy`` brings them back to CPU storage. The
``use_backend`` context manager restores the previous setting even when
a calculation raises an exception. An ``att_gt`` call with a ``backend``
argument enters that context before preprocessing and estimation.

GPU dispatch happens where an implementation checks the backend, including
selected regression and bootstrap helpers in ``cupy/`` and tensor creation
in ``core/preprocess/tensors.py``. Selecting CuPy doesn't move Polars
preprocessing or every Python helper to the GPU. The ``att_gt`` wrapper
converts its influence functions to NumPy before inference and result
construction to make the public result usable by CPU consumers.

CuPy availability and a working CUDA device are checked when the backend
is selected. The ``gpu`` extra is separate from ``all`` because it has
CUDA-specific dependencies. Use the :doc:`GPU guide <../user_guide/gpu>`
for supported interfaces and installation rather than treating a backend
argument on one function as a package-wide guarantee.

Keeping imports and dependencies scoped
---------------------------------------

For a user to reach your function or result type through ``moderndid``,
its public name has to point to the module that defines it.
The root ``__init__.py`` uses ``__getattr__`` to load most names from
``_lazy_imports`` or ``_optional_imports`` when they're requested.
The optional mapping pairs a module path with the extra needed for its
dependencies so an import failure can name an installation command.
``__all__`` describes the intended export list; adding a name there alone
doesn't establish its lazy import route.

Some function names match estimator subpackages and are imported eagerly
to make ``from moderndid import ...`` resolve to the function. Root
imports are therefore mostly lazy rather than a promise that importing
``moderndid`` loads no estimator code. Adding an optional dependency
requires checking both the package import and access to the feature in
an environment without that extra.

The :doc:`new_estimator` guide develops a worked estimator from its target
through the sample requirements, computation, and inference to the result
a user receives. You can follow those decisions before returning here to
connect the implementation to the package's interfaces.
