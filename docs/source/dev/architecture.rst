.. _architecture:

Architecture and API design
===========================

When you change an estimator or add a new one, you need to know how a call
travels from the user's data to the result they read. An estimate that looks
wrong can come from the rows that preprocessing kept, the order it gave the
units, or the way the standard errors were formed, not only from the formula.
A new estimator that prepares its own data or forms its own standard errors
also repeats checks the package already makes. Its result may then lack what
the reports, plots, and sensitivity analysis expect to read.

We'll follow one call to :func:`~moderndid.att_gt` on the minimum wage data
from the :ref:`staggered DiD example <example_staggered_did>` as it prepares
the sample, estimates each effect, measures its uncertainty, and stores the
result. At each stage you'll run the shared code yourself and see which parts a
new estimator can reuse and where it needs an approach of its own. Tools built
on this path, such as plots, publication tables, and fits on a GPU or on Spark,
come up only briefly where they connect to it.

Following one fit through the package
-------------------------------------

The fit below estimates the effect of the minimum wage increases on teen
employment for each adoption cohort and year. We skip the bootstrap so that its
standard errors come straight from the variance calculation that a later
section rebuilds.

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

Inside that call, ``att_gt`` checks its arguments and records them in a
configuration, a ``DIDConfig`` that lists what this method needs to know. A
builder turns the data and the configuration into the prepared sample, a
``DIDData`` container that holds the rows that survive and the arrays the
calculation reads. ``compute_att_gt`` estimates each group-time effect from
that sample along with its influence function, the contribution each unit
makes to the estimate's error. Back in ``att_gt``, the influence functions give
the standard errors before the function returns an ``MPResult``. Most
estimators that start from a DataFrame follow the same path with a
configuration, a prepared container, and a result type of their own.

Preparing the sample
--------------------

Every effect is computed on the sample that preprocessing leaves behind. If
you're tracing a change in which rows enter the fit or in the order of the
units, this is where to start.

A configuration for each method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each estimator that starts from a DataFrame uses one of the configuration
dataclasses in ``core/preprocess/config.py``, since the methods need different
column names and options.

The builder works in Polars because ``with_data`` first converts the user's
data with ``to_polars``. ``with_config`` then reads the configuration's type to
choose the checks and steps that suit the method. To inspect the prepared
sample, we run the builder ourselves with ``allow_unbalanced_panel=False``, the
default of ``att_gt`` but not of ``DIDConfig``.

.. code-block:: python

   from moderndid.core.preprocess import DIDConfig, PreprocessDataBuilder

   config = DIDConfig(**columns, allow_unbalanced_panel=False)

   prepared = (
       PreprocessDataBuilder()
       .with_data(data)
       .with_config(config)
       .validate()
       .transform()
       .build()
   )

``validate`` runs every check that needs no rows, from the named columns and
their types to the values of options such as the comparison group. Checks on
how the rows fit together, such as a unit observed twice in one period, wait
until ``transform`` has dropped the rows with missing values.

.. admonition:: Skipping validate skips the checks
   :class: warning

   A dataclass stores whatever values you give it without checking them.
   Without ``validate`` in the chain, ``transform`` also leaves out its checks
   on how the rows fit together. A misspelled option or a duplicate unit-period
   row then goes through without an error.

``preprocess_did`` runs the same chain in one call, although it keeps
unbalanced panels by default.

Steps that decide which rows remain
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Which rows survive depends on the order of the steps that ``transform`` runs.
For a ``DIDConfig``, ``DataTransformerPipeline.get_did_pipeline`` in
``transformers.py`` lists those steps in the order they run.

.. code-block:: python

   from moderndid.core.preprocess import DataTransformerPipeline

   for step in DataTransformerPipeline.get_did_pipeline().transformers:
       print(type(step).__name__)

.. container:: cell_output

   .. container:: output stream

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

The order is part of the method because each step sees only the rows that
earlier steps kept. Selecting the specification's columns first, for example,
means that a missing value in a column the fit never reads can't cost it a
row. The final sort puts the rows in the order the arrays expect.

.. admonition:: Read the resolved configuration
   :class: tip

   ``transform`` updates the configuration you gave the builder once its steps
   finish. Read ``prepared.config`` for the periods, cohorts, unit count, and
   layout of the retained sample rather than working them out from the
   arguments or the raw data.

Other pipelines differ, such as those of ``did_multiplegt`` and
``dyn_balancing`` that keep rows with a missing outcome or treatment for the
estimator to handle later. Read the pipeline your configuration selects before
changing an order.

Arrays for the calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the rows are settled, ``build`` lays them out as the arrays the numerical
code reads. For a ``DIDConfig``, ``TensorFactorySelector`` in
``core/preprocess/tensors.py`` picks the layout that fits the data. A balanced
panel gets an outcome array, a covariate array, and a weight array for each
period over the same units. Repeated cross-sections and unbalanced panels keep
one entry per row instead.

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

The 500 counties account for the 2,500 retained rows across the five years.
Each period's arrays hold one outcome per county and two covariate columns, the
intercept and ``lpop``. ``DIDData`` keeps those arrays beside the retained
Polars data, the cohort and period counts, and the resolved configuration.

A new estimator can reuse this preparation in more than one way. If its data
look like an existing method's, it can extend that configuration, the way the
``DIDMLConfig`` of ``didml`` extends ``DIDConfig`` to share the DiD steps and
the ``DIDData`` container. A method with a new layout adds its own
configuration, checks, steps, and builder branch, as the two-period ``drdid``
does. Multi-period ``ddd`` instead calls a few shared steps from its own
preparation function. Functions that take arrays, such as ``reg_did_panel``,
skip preparation altogether.

Estimating effects and their influence functions
------------------------------------------------

Estimation turns the prepared sample into one effect for each cohort and
period. ``compute_att_gt`` hands the cohort and period of every effect to
``parallel_map``, a helper that runs independent tasks one after another or on
threads. Each task chooses its comparison group and base period before calling
a two-period estimator from ``drdid``, such as ``drdid_panel`` on a panel. Each
influence function has one entry for every unit in the sample and a zero for a
unit outside the comparison. That shared unit axis lets ``att_gt`` compute
covariances between effects from different comparisons.

Beyond the standard errors, ``aggte`` combines the influence functions into the
influence function of an event study or an overall effect and adds the
uncertainty of the weights it estimates. The multiplier bootstrap draws random
signs for the units' contributions instead of refitting every model for each
draw. The :ref:`influence function background <background-influence-functions>`
develops these ideas from the simplest DiD comparison.

The influence matrix :math:`\Psi` that ``att_gt`` stores in its result has one
row per unit and one column per effect. We'll rebuild the analytical covariance
and standard errors from it. Dividing ``psi.T @ psi`` by :math:`n` once gives
the asymptotic covariance that ``vcov_analytical`` stores. A second division
gives the covariance of the estimates themselves.

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

Each of the 12 columns belongs to the effect that ``result.groups`` and
``result.times`` label at the same position. The standard errors agree because
this fit uses analytical inference at the unit level. A bootstrap would replace
the reported standard errors and critical value without changing what
``vcov_analytical`` means.

.. admonition:: Check the influence function scale
   :class: important

   Before handing influence functions to another helper, check how they are
   normalized and what each row stands for. A matrix of unit contributions and
   an already scaled covariance need different divisions by the sample size.
   Mixing them up changes the standard errors without moving any estimate.

Keeping what later calculations need
------------------------------------

Later calculations never see the data, since ``aggte``, the plots, and the
sensitivity analysis read only the result. ``MPResult`` therefore holds the
labels, estimates, standard errors, critical value, analytical covariance, and
influence functions. It also keeps each unit's cohort in ``G`` and sampling
weight in ``weights_ind`` because ``aggte`` estimates its aggregation weights
from them.

Every result type is a ``NamedTuple`` in the ``container.py`` of its
estimator's package so that downstream code reads fields by name. The class
docstring describes each field under Attributes, the section that becomes the
container's reference page. A field's shape, units, ordering, and normalization
tell a contributor more than its Python type does.

.. admonition:: Arrays inside a result are shared
   :class: warning

   A ``NamedTuple`` stops code from reassigning a field but not from writing
   into the array the field holds. Code that adjusts a result builds a new one
   with ``_replace`` from copies of the arrays it changes, since plots and
   aggregations of the original fit read the same arrays.

``estimation_params`` records how the fit was made for the code that reads it
later. ``aggte`` takes the bootstrap settings and the clusters from it and
copies the rest into its own result, where ``honest_did`` finds the base
period and the anticipation periods. Because the dictionary isn't a complete
copy of the call, code that reads another estimator's results should check
which entries that estimator records.

Connecting a result to the tools that read it
---------------------------------------------

Users print a result, put it in a table, plot it, or test its sensitivity
without knowing how its fields are stored. Each of those tools reads the
result through a small interface that a new result type can implement.

Printed reports
^^^^^^^^^^^^^^^

An estimator's formatter takes a result and returns the text of its printed
report. Helpers in ``core/format.py`` supply the title, the effect table, the
significance note, and the footer. ``attach_format`` connects a formatter to
its result class by making both ``__repr__`` and ``__str__`` call it. The code
below repeats the registration at the end of ``did/format.py``.

.. code-block:: python

   from moderndid.core.format import attach_format
   from moderndid.did.container import MPResult
   from moderndid.did.format import format_mp_result

   attach_format(MPResult, format_mp_result)

The registration runs when the estimator's package is imported because
``did/__init__.py`` imports its ``format`` module. A new formatter needs the
same import in its package's ``__init__.py``. Its report should read the
inference settings and critical value stored in the result rather than build a
different interval for display.

Tables, plots, and sensitivity analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The other tools accept only the result types that have been connected to them.
:func:`~moderndid.to_df` turns a result into a Polars DataFrame
with one row per effect. It finds the converter for that result by class name
in ``_DISPATCH``, a dictionary in ``core/converters.py``. The plot functions
check the result's type themselves and call the converter they need.
Publication tables come from ``maketables``, the package behind the
:ref:`publication tables guide <publication_tables>`. It builds a table from
``__maketables_*__`` attributes on the result's class without fitting anything
again. :func:`~moderndid.honest_did` asks how an event study's conclusions hold
up when parallel trends fails by a bounded amount. It accepts any result with
the attributes that ``EventStudyProtocol`` lists, such as ``influence_func``
and ``event_times``. Since each tool has rules of its own, such as how a result
records its reference period or its clusters, read a tool's code before
connecting a new result to it.

Choosing how numerical work runs
--------------------------------

Once an estimator gives the right answer, a profile shows whether its time
goes to preparing the sample, fitting nuisance models, array operations, or
repeated inference. The :doc:`benchmarking` guide shows how to measure a change
on a fixed workload. Before writing new code for a slow part, look at the
helpers the estimators already share. ``multiplier_bootstrap`` in
``core/bootstrap.py`` runs the multiplier bootstrap of ``att_gt``, ``aggte``,
``cont_did``, and ``ddd``. It multiplies each block of random signs by the
influence matrix in one matrix product rather than looping over the draws. Work
that splits into independent tasks can run on threads through ``parallel_map``
when ``n_jobs`` asks for more than one worker, as the group-time effects of
``att_gt`` do.

The ``backend`` argument of ``att_gt``, ``cont_did``, and ``ddd`` runs parts of
a fit on a GPU through CuPy, an array library with NumPy's interface, as the
:doc:`GPU guide <../user_guide/gpu>` describes. A helper that supports the GPU
keeps a CuPy branch beside its NumPy code. A change to what the NumPy code
computes needs the same change in that branch.

When the data sit in a Spark DataFrame, ``att_gt`` fits them from sums that
each machine computes over the units it holds. Its result keeps those sums in
``influence_moments`` in place of the unit-level influence functions.
:doc:`Fits on shards <shards>`, :doc:`rows on shards <row_shards>`, and
:doc:`fits on Spark <spark_fits>` follow the county fit onto a cluster.

Designing the public interface
------------------------------

A new estimator also becomes part of the interface that users call. They
expect its arguments to mean what the same names mean in other estimators and
``import moderndid`` to work without its optional dependencies.

.. _consistent-argument-naming:

Arguments with one meaning
^^^^^^^^^^^^^^^^^^^^^^^^^^

Most estimators share a name for the same idea, such as ``yname`` for the
outcome column, ``tname`` for the period, ``idname`` for the unit identifier,
and ``weightsname`` for sampling weights. ``xformla`` describes the covariates
with a formula rather than naming a precomputed matrix.

Treatment arguments differ because designs record different things about
treatment. In ``att_gt`` and ``cont_did``, ``gname`` holds the period a unit is
first treated. ``dname`` holds the dose in ``cont_did`` and a treatment that
can change over time in ``did_multiplegt``. In the triple-difference estimator
``ddd``, ``pname`` marks the units eligible for treatment.

Inference arguments keep one name for each idea as well. A significance level
is always ``alpha``, read as a probability such as 0.05. The number of
bootstrap draws is ``biters`` in every estimator that bootstraps. Clustering
alone keeps different names, such as ``clustervars`` in ``att_gt`` and
``cluster`` in ``ddd``, because each method handles clusters in its own
inference.

.. _architecture-imports:

Imports that load on demand
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For users to reach a new function or result type as ``moderndid.<name>``, the
root package has to know which module defines it. Rather than importing every
estimator up front, ``moderndid/__init__.py`` finds a name on first use through
a module-level ``__getattr__`` that looks it up in two dictionaries.

.. code-block:: python

   print(did._lazy_imports["att_gt"])
   print(did._optional_imports["honest_did"])

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         moderndid.did.att_gt
         ('moderndid.didhonest', 'didhonest')

Every install can load ``att_gt`` through its entry in ``_lazy_imports``. The
entry for ``honest_did`` in ``_optional_imports`` also names the ``didhonest``
extra. When that import fails, the error tells the user which extra to install
and how. Listing a name in ``__all__`` declares it a public export but doesn't
give it a route.

.. admonition:: Test imports without the extra
   :class: important

   Adding an optional dependency means checking ``import moderndid`` and the
   new feature in an environment without that extra. The :doc:`testing guide
   <../contributing/testing>` describes the CI jobs that run the fast tests
   with no extras and with ``plots`` alone.

To build a method of your own, the :doc:`adding an estimator <new_estimator>`
guide develops one from its target effect through its sample, computation, and
inference to the result a user receives. Each of its decisions belongs to one
of the stages this page followed.
