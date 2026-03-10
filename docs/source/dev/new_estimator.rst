.. _new-estimator:

========================
Creating a New Estimator
========================

Adding an estimator to **ModernDiD** requires familiarity with the econometric
methodology you are implementing and the ability to validate your code against
reference software, typically an R package.

This guide covers every integration point in the package, from preprocessing
to plotting. It is not meant to be read top to bottom in one sitting. Treat
it as a reference you return to step by step as you build.

In practice, most contributors start with the statistical core. You might write the
core computations first and get correct results on a small test case.
This is completely fine. Once the math works, you can work through
the steps in order to wire it into the rest of the package. The first few
steps (config, preprocessing, result object, estimator function) are the
foundation. The later steps (aggregation, formatting, plotting, maketables,
distributed, tests) can be done in any order and some may not apply to your
estimator at all.

Sticking to the patterns here ensures your estimator integrates with the
plotting, aggregation, sensitivity analysis, and publication table tools
that users expect. Some estimators will need to deviate for reasons like
extra dependencies or non-standard aggregation, but keep the user-facing API
as close to the standard pattern as you can and document any differences
clearly.

Before starting, read the :doc:`architecture` guide for background on the
preprocessing pipeline, result objects, formatting, and distributed support.


.. _new-estimator-dispatch-table:

How estimators plug into the pipeline
--------------------------------------

Every estimator plugs into the preprocessing pipeline through four pieces.
A config, a validator set, a transformer pipeline, and a data model.

.. list-table::
   :widths: 30 25 25 20
   :header-rows: 1

   * - Config type
     - Validator set
     - Transformer pipeline
     - Data model
   * - ``DIDConfig``
     - ``"did"``
     - ``get_did_pipeline()``
     - ``DIDData``
   * - ``ContDIDConfig``
     - ``"cont_did"``
     - ``get_cont_did_pipeline()``
     - ``ContDIDData``
   * - ``TwoPeriodDIDConfig``
     - ``"two_period"``
     - ``get_two_period_pipeline()``
     - ``TwoPeriodDIDData``
   * - ``DDDConfig``
     - ``"ddd"``
     - ``get_ddd_pipeline()``
     - ``DDDData``
   * - ``DIDInterConfig``
     - ``"didinter"``
     - ``get_didinter_pipeline()``
     - ``DIDInterData``

Adding a new estimator means adding a new row to this table. The steps in
this guide walk through each column and what you need to do for each step to
ensure your estimator works with the rest of the library.


.. _new-estimator-module-layout:

Module Layout
-------------

Every estimator lives in its own subpackage under ``moderndid/``. Follow the
naming convention used by existing modules (``did``, ``drdid``, ``didcont``,
``didtriple``, ``didinter``, ``didhonest``). A typical layout looks like this.

.. code-block:: text

   moderndid/
   └── myestimator/
       ├── __init__.py          # Public re-exports + format import
       ├── estimator.py         # Main user-facing function
       ├── compute.py           # Core numerical computation
       ├── container.py         # Result NamedTuples
       ├── format.py            # attach_format bindings
       └── aggte.py             # Aggregation (if multi-period)

File names may vary. For example, the ``did`` module uses ``att_gt.py`` and
``compute_att_gt.py`` instead of the generic names above.

Larger estimators may add subdirectories (``estimators/``, ``bootstrap/``,
``propensity/``) as needed.


Step 1: Define the Configuration
--------------------------------

Every estimator is driven by a configuration dataclass that holds user-facing
parameters like column names, significance level, and bootstrap options, along
with computed fields that the preprocessing pipeline fills in.

The inheritance chain starts with ``ConfigMixin``, a small mixin that
provides serialization.

.. code-block:: python

   class ConfigMixin:
       """Mixin for config methods."""

       def to_dict(self) -> dict[str, Any]:
           return {
               k: v.value if isinstance(v, Enum) else v
               for k, v in self.__dict__.items()
           }

``BasePreprocessConfig`` inherits from ``ConfigMixin`` and provides the
standard fields that all estimators share.

.. code-block:: python

   @dataclass
   class BasePreprocessConfig(ConfigMixin):
       yname: str
       tname: str
       gname: str

       idname: str | None = None
       xformla: str = "~1"
       panel: bool = True
       allow_unbalanced_panel: bool = True
       weightsname: str | None = None
       alp: float = DEFAULT_ALPHA
       boot: bool = False
       cband: bool = False
       biters: int = DEFAULT_BOOTSTRAP_ITERATIONS
       clustervars: list[str] = field(default_factory=list)
       anticipation: int = DEFAULT_ANTICIPATION_PERIODS
       faster_mode: bool = False
       pl: bool = False
       cores: int = DEFAULT_CORES
       true_repeated_cross_sections: bool = False
       # ... plus computed fields populated during preprocessing

All subclasses inherit these fields. ``faster_mode`` and ``cores`` control
parallel computation, ``pl`` selects the Polars-native code path, and
``true_repeated_cross_sections`` tells the pipeline that the data is a genuine
repeated cross-section rather than a panel with missing units.

``DIDConfig``, the configuration used by the staggered adoption estimator
:func:`~moderndid.att_gt`, inherits from ``BasePreprocessConfig`` and adds
estimator-specific fields.

.. code-block:: python

   @dataclass
   class DIDConfig(BasePreprocessConfig):
       control_group: ControlGroup = ControlGroup.NEVER_TREATED
       est_method: EstimationMethod = EstimationMethod.DOUBLY_ROBUST
       base_period: BasePeriod = BasePeriod.VARYING

If your estimator can reuse ``DIDConfig`` as-is, you do not need a new config.
Otherwise, create your own subclass of ``BasePreprocessConfig``. Your
subclass only needs to add estimator-specific parameters. Everything in
``BasePreprocessConfig`` is inherited automatically, including ``to_dict()``
from ``ConfigMixin``.

If your data layout diverges significantly from the standard panel model (for
example, separate pre/post arrays, or a completely different set of computed
fields), you can inherit from ``ConfigMixin`` directly instead of
``BasePreprocessConfig``. ``TwoPeriodDIDConfig``, ``DDDConfig``, and
``DIDInterConfig`` follow this pattern.

.. code-block:: python

   # myestimator/config.py
   from dataclasses import dataclass
   from moderndid.core.preprocess.config import BasePreprocessConfig

   @dataclass
   class MyEstimatorConfig(BasePreprocessConfig):
       # Add estimator-specific parameters only.
       # Fields from BasePreprocessConfig (yname, tname, gname, etc.)
       # are inherited automatically.
       my_special_param: float = 1.0
       another_param: str = "default"

When your parameter has a fixed set of valid values, use the enums from
``moderndid.core.preprocess.constants``, or add new enums specific to your estimator.

.. code-block:: python

   from moderndid.core.preprocess.constants import (
       ControlGroup,       # NEVER_TREATED, NOT_YET_TREATED
       EstimationMethod,   # DOUBLY_ROBUST, IPW, REGRESSION
       BasePeriod,         # UNIVERSAL, VARYING
       BootstrapType,      # WEIGHTED, MULTIPLIER, EMPIRICAL
       DataFormat,         # PANEL, REPEATED_CROSS_SECTION, UNBALANCED_PANEL
   )

During preprocessing, the builder populates computed fields on your config
such as ``time_periods``, ``treated_groups``, ``time_periods_count``,
``treated_groups_count``, ``id_count``, and ``data_format``. Your estimation
code can read these directly from the config after the builder completes.


Step 2: Integrate with the Preprocessing Pipeline
--------------------------------------------------

The preprocessing pipeline takes raw user data and produces a clean,
ready-to-estimate data container. Most new estimators need at least one
custom piece, whether that is a new validation check, a different
transformation, or a different data layout.

.. tip::

   If your estimator reuses ``DIDConfig`` and ``DIDData`` without changes,
   you can skip this step and call the pipeline as-is.

The pipeline has four extension points that you may need to customize.

1. **Validators** check the raw data before anything else runs, confirming
   that columns exist, treatment is time-invariant, weights are non-negative,
   and so on.
2. **Transformers** clean and reshape the data in sequence by selecting
   columns, dropping nulls, normalizing weights, encoding treatment, and
   balancing panels.
3. **Config updater** writes computed fields like ``time_periods``,
   ``treated_groups``, ``id_count``, and ``data_format`` onto the config
   after the transformers finish.
4. **Data model** is the typed container that ``build()`` constructs from
   the cleaned DataFrame and config.

The rest of this section walks through each one.


The builder interface
^^^^^^^^^^^^^^^^^^^^^^

Every estimator calls the pipeline through the same builder.

.. code-block:: python

   dp = (
       PreprocessDataBuilder()
       .with_data(data)          # Convert any Arrow-compatible DataFrame to Polars
       .with_config(config)      # Select validators + transformers for this config type
       .validate()               # Run all validators; raise on errors, warn on warnings
       .transform()              # Run all transformers in sequence, then update config
       .build()                  # Construct the typed data container
   )

The builder enforces the call order shown above. ``.with_config()`` uses
``isinstance`` checks on the config object to select the right validator
set and transformer pipeline, as shown in the
:ref:`dispatch table <new-estimator-dispatch-table>` at the top of this
guide.


Validators
^^^^^^^^^^^

Validators run before the transformers and check that the raw data is
suitable for estimation. Each validator subclasses ``BaseValidator``, an
abstract base class with a single method.

.. code-block:: python

   # In moderndid/core/preprocess/base.py

   class BaseValidator(ABC):
       @abstractmethod
       def validate(self, data: DataFrame, config: BaseConfig):
           """Validate data."""

The ``validate`` method returns a ``ValidationResult`` containing lists of
errors and warnings. Errors stop the pipeline, while warnings are emitted
but allow execution to continue. Here is a custom validator that checks
whether the outcome variable has variation.

.. code-block:: python

   # In moderndid/core/preprocess/validators.py

   from .base import BaseValidator
   from .models import ValidationResult

   class MyCustomValidator(BaseValidator):
       """Check that the outcome variable has variation."""

       def validate(self, data, config) -> ValidationResult:
           df = to_polars(data)
           errors = []
           warnings = []

           if df[config.yname].n_unique() < 2:
               errors.append("Outcome variable has no variation.")

           return ValidationResult(
               is_valid=len(errors) == 0,
               errors=errors,
               warnings=warnings,
           )

The library already provides several validators you can reuse. For example,
``ColumnValidator`` confirms that required columns exist and are numeric.
``WeightValidator`` rejects negative weights. ``TreatmentValidator`` ensures
treatment is time-invariant per unit. ``PanelStructureValidator`` catches
duplicate rows and panel imbalance. ``ClusterValidator`` verifies that
clustering variables do not vary over time. See
`validators.py <https://github.com/jordandeklerk/moderndid/blob/main/moderndid/core/preprocess/validators.py>`_
for the full set.

To make your validators run, register them in
``CompositeValidator._get_default_validators()`` in the same file. The
method dispatches on a string key, so add a new branch for your config
type early in the chain and return a list of validator instances.

.. code-block:: python

   # In CompositeValidator._get_default_validators()

   @staticmethod
   def _get_default_validators(config_type="did"):
       if config_type == "two_period":
           return [
               PrePostColumnValidator(),
               PrePostArgumentValidator(),
               # ...
           ]

       if config_type == "my_estimator":                # Add your branch
           return [
               ColumnValidator(),
               WeightValidator(),
               PanelStructureValidator(),
               MyCustomValidator(),
           ]

       # Default validators for "did", "cont_did", etc.
       common_validators = [
           ArgumentValidator(),
           ColumnValidator(),
           WeightValidator(),
           TreatmentValidator(),
           PanelStructureValidator(),
           ClusterValidator(),
       ]
       if config_type == "cont_did":
           common_validators.append(DoseValidator())
       return common_validators


Transformers
^^^^^^^^^^^^^

Transformers clean and reshape the data in sequence. Each transformer
subclasses ``BaseTransformer``, an abstract base class with a single method.

.. code-block:: python

   # In moderndid/core/preprocess/base.py

   class BaseTransformer(ABC):
       @abstractmethod
       def transform(self, data: DataFrame, config: BaseConfig):
           """Transform data."""

The ``transform`` method receives a Polars DataFrame and config, and returns
a Polars DataFrame. The output of one transformer feeds into the next.

For reference, here is the transformer pipeline that the staggered adoption
estimator :func:`~moderndid.att_gt` uses.

.. code-block:: python

   # DataTransformerPipeline.get_did_pipeline()

   DataTransformerPipeline([
       ColumnSelector(),                # Keep only relevant columns
       MissingDataHandler(),            # Drop rows with nulls
       WeightNormalizer(),              # Create normalized weight column (mean=1)
       TreatmentEncoder(),              # Recode gname=0 as inf (never-treated)
       EarlyTreatmentFilter(),          # Drop units treated before first period + anticipation
       ControlGroupCreator(),           # Coerce last cohort as never-treated if needed
       PanelBalancer(),                 # Keep only units in all time periods
       RepeatedCrossSectionHandler(),   # Add row index as idname for cross-sections
       DataSorter(),                    # Sort by tname, gname, idname
   ])

Other estimators use different pipelines. The continuous treatment estimator
adds ``TimePeriodRecoder`` and ``DoseValidatorTransformer``. The DDD
estimator adds ``DDDSubgroupCreator`` and ``DDDPostIndicatorCreator``. The
intertemporal estimator adds ``SwitcherIdentifier`` and ``SwitcherFilter``.
Look at the existing ``get_*_pipeline()`` factory methods in
`transformers.py <https://github.com/jordandeklerk/moderndid/blob/main/moderndid/core/preprocess/transformers.py>`_ for the full picture.

To write a custom transformer, subclass ``BaseTransformer``.

.. code-block:: python

   # In moderndid/core/preprocess/transformers.py

   from .base import BaseTransformer

   class MyCustomTransformer(BaseTransformer):
       """Describe what this transformer does in one line."""

       def transform(self, data, config) -> pl.DataFrame:
           df = to_polars(data)

           # Your transformation logic here.
           # Always return a Polars DataFrame.
           return df

A few rules to keep in mind when writing transformers.

- Always call ``to_polars(data)`` at the top. This is a no-op if the input
  is already Polars, but it protects against type mismatches.
- You can assume earlier transformers have already run, so columns are
  selected, nulls are handled, and weights exist.
- You can mutate fields on the config directly because the config is a
  mutable dataclass.
- If your transformer removes observations, emit a ``warnings.warn()`` so
  the user knows data was dropped.

Then add your pipeline as a ``@staticmethod`` factory on
``DataTransformerPipeline``, following the same pattern as
``get_did_pipeline()`` and the other existing factories. Reuse existing
transformers where they apply and slot your custom ones in at the right
point in the sequence.

.. code-block:: python

   # In DataTransformerPipeline

   @staticmethod
   def get_my_estimator_pipeline() -> "DataTransformerPipeline":
       return DataTransformerPipeline([
           ColumnSelector(),
           MissingDataHandler(),
           WeightNormalizer(),
           MyCustomTransformer(),      # Your own
           PanelBalancer(),
           DataSorter(),
       ])


Config updater
^^^^^^^^^^^^^^^

After the transformer pipeline finishes, a ``ConfigUpdater`` writes computed
fields onto the config so your estimation code can read them. The default
``ConfigUpdater.update()`` sets ``time_periods``, ``treated_groups``,
``id_count``, and ``data_format`` by inspecting the transformed DataFrame.

If your estimator needs additional computed fields, write a custom updater.
``DIDInterConfigUpdater``, for instance, computes ``max_effects_available``
and ``max_placebo_available`` from the switcher timing variable, while
``DDDConfigUpdater`` computes ``n_units``.

.. code-block:: python

   # In moderndid/core/preprocess/transformers.py

   class MyEstimatorConfigUpdater:
       @staticmethod
       def update(data: pl.DataFrame, config) -> None:
           config.time_periods = data[config.tname].unique().sort().to_numpy()
           config.time_periods_count = len(config.time_periods)
           config.id_count = data[config.idname].n_unique()
           # Add your own computed fields here

Then add an ``isinstance`` branch for your config type inside
``DataTransformerPipeline.transform()``. The existing dispatch looks like
this.

.. code-block:: python

   # In DataTransformerPipeline.transform()

   def transform(self, data, config):
       df = to_polars(data)
       for transformer in self.transformers:
           df = transformer.transform(df, config)

       if isinstance(config, DDDConfig):
           DDDConfigUpdater.update(df, config)
       elif isinstance(config, DIDInterConfig):
           DIDInterConfigUpdater.update(df, config)
       elif isinstance(config, MyEstimatorConfig):    # Add your branch
           MyEstimatorConfigUpdater.update(df, config)
       else:
           ConfigUpdater.update(df, config)

       return df


Data model
^^^^^^^^^^^

The data model is a dataclass that holds the cleaned data ready for
estimation. It lives in `models.py <https://github.com/jordandeklerk/moderndid/blob/main/moderndid/core/preprocess/models.py>`_.

Most estimators inherit from ``PreprocessedData``, which provides the
shared fields that the builder helpers and downstream tools expect.

.. code-block:: python

   @dataclass
   class PreprocessedData:
       data: pl.DataFrame                 # Cleaned panel/cross-section data
       time_invariant_data: pl.DataFrame  # One row per unit
       weights: np.ndarray                # Normalized sampling weights
       cohort_counts: pl.DataFrame        # Units per treatment cohort
       period_counts: pl.DataFrame        # Observations per time period
       crosstable_counts: pl.DataFrame    # Cohort x period cross-tabulation
       config: BasePreprocessConfig       # Config with computed fields
       cluster: np.ndarray | None = None  # Cluster IDs

Your subclass adds estimator-specific fields. For example, ``DIDData``
adds ``outcomes_tensor``, ``covariates_matrix``, and ``covariates_tensor``
for the tensor-based computation in ``att_gt``. ``ContDIDData`` adds
``time_map`` and ``original_time_periods`` for mapping recoded time indices
back to original values.

.. code-block:: python

   @dataclass
   class MyEstimatorData(PreprocessedData):
       my_special_matrix: np.ndarray | None = None
       config: MyEstimatorConfig = field(default_factory=MyEstimatorConfig)

If your data layout diverges from the standard panel structure (for
example, separate pre/post arrays instead of a full panel), use a
standalone dataclass. ``TwoPeriodDIDData`` and ``DDDData`` follow this
pattern.

.. code-block:: python

   @dataclass
   class MyEstimatorData:
       y_pre: np.ndarray
       y_post: np.ndarray
       treatment: np.ndarray
       weights: np.ndarray
       n_units: int
       config: MyEstimatorConfig


.. _new-estimator-register-builder:

Register in the builder
^^^^^^^^^^^^^^^^^^^^^^^^

Once you have validators, transformers, a config updater, and a data model,
wire them into ``PreprocessDataBuilder`` in
`builders.py <https://github.com/jordandeklerk/moderndid/blob/main/moderndid/core/preprocess/builders.py>`_.
You need to touch three places in that file.

**1. config dispatch.** Add an ``isinstance`` check in ``with_config()`` so
the builder selects your validators and transformers when it sees your config
type. The existing method chains ``elif`` branches in order of specificity,
so insert yours above any parent class it inherits from.

.. code-block:: python

   # In PreprocessDataBuilder.with_config()

   def with_config(self, config):
       self._config = config

       if isinstance(config, TwoPeriodDIDConfig):
           self._validator = CompositeValidator(config_type="two_period")
           self._transformer = DataTransformerPipeline.get_two_period_pipeline()
       elif isinstance(config, MyEstimatorConfig):    # Add your branch
           self._validator = CompositeValidator(config_type="my_estimator")
           self._transformer = DataTransformerPipeline.get_my_estimator_pipeline()
       elif isinstance(config, DIDConfig):
           self._validator = CompositeValidator(config_type="did")
           self._transformer = DataTransformerPipeline.get_did_pipeline()
       # ... remaining branches ...

       return self

.. warning::

   Place your ``isinstance`` check above any parent class your config
   inherits from. More specific types must be checked first, otherwise
   ``isinstance`` will match the parent and silently select the wrong
   pipeline.

**2. build dispatch.** Add a branch in ``build()`` that constructs your data
model. The method uses a chain of ``if``/``isinstance`` checks, each
returning immediately.

.. code-block:: python

   # In PreprocessDataBuilder.build()

   def build(self):
       if self._data is None or self._config is None:
           raise ValueError("Must set data and config before building")

       if isinstance(self._config, TwoPeriodDIDConfig):
           return self._build_two_period_did_data()
       if isinstance(self._config, MyEstimatorConfig):    # Add your branch
           return self._build_my_estimator_data()
       if isinstance(self._config, DIDConfig):
           return self._build_did_data()
       # ... remaining branches ...
       raise ValueError(f"Unknown config type: {type(self._config)}")

**3. Private build method.** Implement the method that constructs your data
container from the cleaned DataFrame and config. The builder provides
helper methods you can reuse.

.. code-block:: python

   def _build_my_estimator_data(self) -> MyEstimatorData:
       time_invariant_data = self._create_time_invariant_data()
       summary_tables = self._create_summary_tables(time_invariant_data)
       cluster = self._extract_cluster_variable(time_invariant_data)
       weights = self._extract_weights(time_invariant_data)

       return MyEstimatorData(
           data=self._data,
           time_invariant_data=time_invariant_data,
           weights=weights,
           cohort_counts=summary_tables["cohort_counts"],
           period_counts=summary_tables["period_counts"],
           crosstable_counts=summary_tables["crosstable_counts"],
           cluster=cluster,
           config=self._config,
       )

The helpers ``_create_time_invariant_data()``,
``_create_summary_tables()``, ``_extract_cluster_variable()``, and
``_extract_weights()`` are already defined on the builder. If your data
model needs extra arrays or custom processing, add that logic in your
private build method.


Step 3: Define the Result Object
---------------------------------

Create a ``NamedTuple`` for your results in ``myestimator/container.py``.
Include the standard attributes that downstream tools like plotting,
aggregation, sensitivity analysis, and maketables expect.

Give the class a ``"Container for ..."`` docstring with a numpydoc
``Attributes`` section, and add ``#:`` doc comments before each field so
that the API docs render meaningful descriptions.

.. code-block:: python

   # myestimator/container.py
   from typing import NamedTuple
   import numpy as np

   class MyEstimatorResult(NamedTuple):
       """Container for my estimator results.

       Attributes
       ----------
       att : ndarray
           Point estimates.
       se : ndarray
           Standard errors.
       critical_value : float
           Critical value for confidence intervals.
       influence_func : ndarray or None
           Influence function matrix for inference propagation.
       estimation_params : dict
           Parameters used for estimation.
       """

       #: Point estimates.
       att: np.ndarray
       #: Standard errors.
       se: np.ndarray
       #: Critical value for confidence intervals.
       critical_value: float
       #: Influence function matrix.
       influence_func: np.ndarray | None
       #: Parameters used for estimation.
       estimation_params: dict = {}

Every result carries an ``estimation_params`` dictionary that downstream
tools (formatters, maketables, aggregation) read from. The standard keys
are listed below.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Key
     - Description
   * - ``yname``
     - Outcome variable name.
   * - ``control_group``
     - ``"nevertreated"`` or ``"notyettreated"``.
   * - ``anticipation_periods``
     - Number of anticipation periods.
   * - ``estimation_method``
     - ``"dr"``, ``"ipw"``, or ``"reg"``.
   * - ``bootstrap``
     - Whether the bootstrap was used (bool).
   * - ``uniform_bands``
     - Whether simultaneous confidence bands were computed (bool).
   * - ``base_period``
     - ``"universal"`` or ``"varying"``.
   * - ``panel``
     - Whether the data is panel (bool).
   * - ``clustervars``
     - List of clustering variable names.
   * - ``cluster``
     - Cluster ID array (or ``None``).
   * - ``biters``
     - Number of bootstrap iterations.
   * - ``random_state``
     - Random seed used for the bootstrap.
   * - ``n_units``
     - Number of unique cross-sectional units.
   * - ``n_obs``
     - Number of observations.
   * - ``alpha``
     - Significance level.

Some result classes also carry an optional ``call_info`` dictionary for
debugging metadata. It is not read by any downstream tools.

For multi-period estimators producing group-time effects, include the standard
fields that the aggregation and plotting systems expect.

.. code-block:: python

   class MyEstimatorMPResult(NamedTuple):
       #: Treatment cohort for each estimate.
       groups: np.ndarray
       #: Time period for each estimate.
       times: np.ndarray
       #: Group-time ATT estimates.
       att_gt: np.ndarray
       #: Analytical variance-covariance matrix.
       vcov_analytical: np.ndarray
       #: Standard errors for each ATT(g,t).
       se_gt: np.ndarray
       #: Critical value for confidence intervals.
       critical_value: float
       #: Influence function matrix (n_units x n_estimates).
       influence_func: np.ndarray
       #: Number of unique cross-sectional units.
       n_units: int | None = None
       #: Wald statistic for pre-testing common trends.
       wald_stat: float | None = None
       #: P-value of the Wald statistic.
       wald_pvalue: float | None = None
       #: Aggregate treatment effects object (populated by aggte).
       aggregate_effects: object | None = None
       #: Significance level.
       alpha: float = 0.05
       #: Estimation parameters dictionary.
       estimation_params: dict = {}
       #: Unit-level group assignments.
       G: np.ndarray | None = None
       #: Unit-level sampling weights.
       weights_ind: np.ndarray | None = None


Step 4: Implement the Estimator
-------------------------------

The main estimator function is the user-facing entry point. It accepts
user-friendly parameters, handles backend delegation, preprocesses data,
runs the estimation, and returns an immutable result object.

The function follows four phases, shown in the code example below.

1. **Delegation.** Check for a CuPy backend, Dask collection, or Spark
   DataFrame and dispatch accordingly.
2. **Setup.** Validate inputs, construct the config dataclass, and run the
   preprocessing builder.
3. **Estimation.** Call the core compute function, derive standard errors
   from the influence function matrix, and optionally run the bootstrap.
4. **Finalization.** For multi-period estimators, compute a pre-test Wald
   statistic from pre-treatment estimates. Return the result NamedTuple.

Existing estimators often define a factory function like ``mp()`` in
``did/container.py`` that wraps the ``NamedTuple`` constructor, sets
defaults, and converts ``None`` values to empty dicts. A factory keeps
the packaging step clean and avoids repeating default logic across call sites.

The code below is long but mostly boilerplate. Sections 1 through 3 handle
backend delegation and sections 8 and 9 handle variance estimation and
bootstrapping; both can be copied almost verbatim. Your custom logic goes in
section 5 (config fields), section 7 (core computation), and section 11
(which ``estimation_params`` keys to populate).

.. code-block:: python

   # myestimator/estimator.py
   import numpy as np
   import scipy.stats
   from moderndid.core.preprocess import PreprocessDataBuilder
   from moderndid.cupy.backend import to_numpy
   from .config import MyEstimatorConfig
   from .container import MyEstimatorResult

   def my_estimator(
       data,
       yname: str,
       tname: str,
       idname: str = None,
       gname: str = None,
       xformla: str = None,
       alp: float = 0.05,
       boot: bool = False,
       cband: bool = True,
       biters: int = 1000,
       clustervars=None,
       control_group: str = "nevertreated",
       anticipation: int = 0,
       n_jobs: int = 1,
       backend=None,
   ) -> MyEstimatorResult:
       # 1. Backend delegation
       if backend is not None:
           from moderndid.cupy.backend import use_backend
           with use_backend(backend):
               return my_estimator(
                   data=data, yname=yname, tname=tname, idname=idname,
                   gname=gname, xformla=xformla, alp=alp, boot=boot,
                   biters=biters, n_jobs=n_jobs, backend=None,
               )

       # 2. Dask delegation
       from moderndid.dask._utils import is_dask_collection
       if is_dask_collection(data):
           from moderndid.dask._my_estimator import dask_my_estimator
           return dask_my_estimator(...)

       # 3. Spark delegation
       from moderndid.spark._utils import is_spark_dataframe
       if is_spark_dataframe(data):
           from moderndid.spark._my_estimator import spark_my_estimator
           return spark_my_estimator(...)

       # 4. Input validation
       if gname is None:
           raise ValueError("gname is required.")
       if not 0 < alp < 1:
           raise ValueError(f"alp={alp} must be between 0 and 1.")

       # 5. Build configuration
       config = MyEstimatorConfig(
           yname=yname,
           tname=tname,
           idname=idname,
           gname=gname,
           xformla=xformla if xformla is not None else "~1",
           alp=alp,
           boot=boot,
           biters=biters,
       )

       # 6. Preprocess data
       dp = (
           PreprocessDataBuilder()
           .with_data(data)
           .with_config(config)
           .validate()
           .transform()
           .build()
       )

       # 7. Core computation
       results = _compute(dp, n_jobs=n_jobs)

       # 8. Variance estimation from influence functions
       n_units = dp.config.id_count
       influence_funcs = to_numpy(np.array(results.influence_functions))
       variance_matrix = influence_funcs.T @ influence_funcs / n_units
       standard_errors = np.sqrt(np.diag(variance_matrix) / n_units)
       standard_errors[standard_errors <= np.sqrt(np.finfo(float).eps) * 10] = np.nan

       # 9. Bootstrap (if requested)
       critical_value = scipy.stats.norm.ppf(1 - alp / 2)
       if boot:
           from .mboot import mboot
           bootstrap_results = mboot(
               inf_func=influence_funcs,
               n_units=n_units,
               biters=biters,
               alp=alp,
           )
           standard_errors = bootstrap_results["se"]
           critical_value = bootstrap_results["crit_val"]

       # 10. Pre-test (for multi-period estimators, compute Wald statistic
       #     from pre-treatment influence functions; see att_gt for the pattern)

       # 11. Package results
       return MyEstimatorResult(
           att=results.att_values,
           se=standard_errors,
           critical_value=critical_value,
           influence_func=influence_funcs,
           estimation_params={
               "yname": yname,
               "control_group": control_group,
               "anticipation_periods": anticipation,
               "estimation_method": "dr",
               "bootstrap": boot,
               "uniform_bands": cband,
               "clustervars": clustervars,
               "cluster": dp.cluster,
               "n_units": n_units,
               "n_obs": len(dp.data),
               "alpha": alp,
           },
       )

Separate the core numerical logic into a ``compute.py`` module. This keeps the
main function focused on orchestration and makes the computation testable in
isolation. For group-time loops, use ``parallel_map`` from
``moderndid.core.parallel``.

.. code-block:: python

   # myestimator/compute.py
   from moderndid.core.parallel import parallel_map

   def _compute(preprocessed_data, n_jobs=1):
       args_list = [
           (group_idx, time_idx, preprocessed_data)
           for group_idx in range(len(preprocessed_data.config.treated_groups))
           for time_idx in range(len(preprocessed_data.config.time_periods))
       ]
       results = parallel_map(_estimate_single_cell, args_list, n_jobs=n_jobs)
       # ... collect results into arrays ...


Step 5: Add Aggregation Support (Multi-Period Estimators)
---------------------------------------------------------

If your estimator produces group-time effects, implement an aggregation
function following the ``aggte`` pattern in ``did/aggte.py``.

Aggregation computes weighted averages of group-time ATTs. The key step is
propagating uncertainty. If ``IF`` is the influence function matrix
(``n_units x n_cells``) and ``w`` is the weight vector, then the influence
function for the aggregate is ``IF @ w``.

.. code-block:: python

   # myestimator/aggte.py
   import numpy as np
   import scipy.stats

   def aggte(result, type="dynamic", ...):
       """Aggregate group-time effects.

       Parameters
       ----------
       result : MyEstimatorMPResult
           Group-time result from the estimator.
       type : {'simple', 'dynamic', 'group', 'calendar'}
           Aggregation type.
       """
       att_gt = result.att_gt
       inf_func = result.influence_func  # (n_units, n_cells)
       n_units = result.n_units

       if type == "simple":
           # Weighted average across all post-treatment cells
           weights = _compute_simple_weights(result)
           overall_att = weights @ att_gt
           # Propagate uncertainty through the influence function
           agg_inf = inf_func @ weights
           overall_se = np.sqrt(np.mean(agg_inf**2) / n_units)
           return AggResult(overall_att=overall_att, overall_se=overall_se, ...)

       if type == "dynamic":
           # Group cells by event time, aggregate within each
           for e in event_times:
               cell_mask = _cells_at_event_time(result, e)
               weights_e = _compute_event_weights(result, cell_mask)
               att_e = weights_e @ att_gt[cell_mask]
               inf_e = inf_func[:, cell_mask] @ weights_e
               se_e = np.sqrt(np.mean(inf_e**2) / n_units)
               # ... store att_e, se_e, inf_e for this event time

The aggregated result object should include ``overall_att`` and ``overall_se``
for the summary ATT and its standard error, ``aggregation_type`` set to one of
``"simple"``, ``"dynamic"``, ``"group"``, or ``"calendar"``, and for non-simple
aggregations the per-event arrays ``event_times``, ``att_by_event``,
``se_by_event``, ``critical_values``, ``influence_func``, and
``influence_func_overall``.

The plotting converters and the sensitivity analysis tools
(:func:`~moderndid.honest_did`) depend on these fields.


Step 6: Add Formatted Output
-----------------------------

Create a ``format.py`` module that gives your result readable ``print()``
and ``repr()`` output. Define a format function and register it with
``attach_format``, which monkey-patches ``__repr__`` and ``__str__`` on your
result class. The formatting section of the :doc:`architecture` guide covers
the design in more detail.

A format function builds a list of lines using shared helpers from
``moderndid.core.format`` and joins them into a single string. Below is a
minimal example.

.. code-block:: python

   # myestimator/format.py
   import scipy.stats
   from moderndid.core.format import (
       attach_format,
       format_footer,
       format_section_header,
       format_significance_note,
       format_single_result_table,
       format_title,
   )

   from .container import MyEstimatorResult


   def format_my_result(result):
       lines = []
       alpha = result.estimation_params.get("alpha", 0.05)
       conf_level = int((1 - alpha) * 100)
       z_crit = scipy.stats.norm.ppf(1 - alpha / 2)
       lci = result.att - z_crit * result.se
       uci = result.att + z_crit * result.se

       lines.extend(format_title("My Estimator Results"))
       lines.extend(
           format_single_result_table(
               "ATT", result.att, result.se,
               conf_level, lci, uci,
           )
       )
       lines.extend(format_significance_note())

       lines.extend(format_section_header("Estimation Details"))
       est_method = result.estimation_params.get("estimation_method")
       if est_method:
           lines.append(f" Estimation Method: {est_method}")

       lines.extend(format_footer("Reference: Author (Year)"))
       return "\n".join(lines)


   attach_format(MyEstimatorResult, format_my_result)

.. important::

   Import your ``format`` module in your package's ``__init__.py`` so that
   ``attach_format`` runs at import time. Without this import, your result
   objects will not have readable ``print()`` output.

``moderndid.core.format`` provides the following shared helpers.

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Helper
     - Purpose
   * - ``format_title(title, subtitle=None)``
     - Title block with thick separators.
   * - ``format_section_header(label)``
     - Section header with thin separators.
   * - ``format_footer(reference=None)``
     - Closing separator with optional citation.
   * - ``format_significance_note(band=False)``
     - Significance legend line.
   * - ``format_value(val, fmt, na_str)``
     - Format a number, ``"NA"`` for NaN.
   * - ``format_conf_interval(lci, uci, fmt)``
     - ``[lower, upper]`` string.
   * - ``compute_significance(lci, uci)``
     - ``"*"`` if the CI excludes zero.
   * - ``format_p_value(p)``
     - ``<0.001`` for small values.
   * - ``format_kv_line(key, value, indent)``
     - Indented key-value pair.
   * - ``format_single_result_table(...)``
     - Single-row estimate table.
   * - ``format_group_time_table(...)``
     - Multi-row group-time ATT table.
   * - ``format_event_table(...)``
     - Event-study table.
   * - ``format_horizon_table(...)``
     - Horizon-indexed table.
   * - ``adjust_separators(lines)``
     - Widen separators to match content width.


Step 7: Add Plotting Support
-----------------------------

Add a converter function to
`moderndid/core/converters.py <https://github.com/jordandeklerk/moderndid/blob/main/moderndid/core/converters.py>`_
that transforms your result to a Polars DataFrame for plotting. Converters
must use the standard column names that the plot functions expect, and should
filter out rows where the standard error is NaN (these correspond to
reference periods that should not be plotted).

For event study results, the expected columns are ``event_time`` (event time
relative to treatment), ``att`` (point estimate), ``se`` (standard error),
``ci_lower`` and ``ci_upper`` (confidence interval bounds), and
``treatment_status`` (``"Pre"`` or ``"Post"``).

For group-time results, use ``group`` and ``time`` instead of ``event_time``.

The existing converters use ``TYPE_CHECKING`` to avoid circular imports, so
add your result type to the guarded import block at the top of the file.

.. code-block:: python

   # At the top of moderndid/core/converters.py

   from typing import TYPE_CHECKING

   if TYPE_CHECKING:
       from moderndid.myestimator.container import MyEstimatorResult

Then add your converter function in the same file.

.. code-block:: python

   def myresult_to_polars(result: MyEstimatorResult) -> pl.DataFrame:
       event_times = result.event_times
       att = result.att
       se = result.se
       crit_val = result.critical_value if result.critical_value is not None else 1.96

       ci_lower = att - crit_val * se
       ci_upper = att + crit_val * se
       treatment_status = np.array(["Pre" if e < 0 else "Post" for e in event_times])

       df = pl.DataFrame({
           "event_time": event_times,
           "att": att,
           "se": se,
           "ci_lower": ci_lower,
           "ci_upper": ci_upper,
           "treatment_status": treatment_status,
       })
       return df.filter(~pl.col("se").is_nan())

Then update the relevant plot function in `plots.py <https://github.com/jordandeklerk/moderndid/blob/main/moderndid/plots/plots.py>`_ to
dispatch to your converter. The plotting module provides several plot
functions (``plot_event_study``, ``plot_group_time``, ``plot_dose_response``,
``plot_honest_did``). Choose the one that matches your result type. If none
fit, add a new plot function following the same pattern.

Plot functions use ``isinstance()`` and ``hasattr()`` checks to route each
result type to its converter. The dispatch order matters because more
specific checks like ``hasattr`` and concrete subclasses must come before
general ones. Below is the actual pattern from ``plot_event_study``.

.. code-block:: python

   # In moderndid/plots/plots.py

   from moderndid.myestimator.container import MyEstimatorResult
   from moderndid.core.converters import myresult_to_polars

   def plot_event_study(result, ...):
       # hasattr checks first for duck-typed results (e.g., PTEResult)
       if hasattr(result, "event_study") and not isinstance(result, (AGGTEResult, DDDAggResult)):
           df = pteresult_to_polars(result)
       # Then concrete isinstance checks, most specific first
       elif isinstance(result, DDDAggResult):
           df = dddaggresult_to_polars(result)
       elif isinstance(result, MyEstimatorResult):       # Add your type here
           df = myresult_to_polars(result)
       elif isinstance(result, AGGTEResult):
           df = aggteresult_to_polars(result)
       else:
           raise TypeError(f"Unsupported result type: {type(result).__name__}")
       # ... build ggplot


Step 8: Export the Public API
-----------------------------

Add your estimator to the module's ``__init__.py``. The format module must
be imported here so that ``attach_format`` runs at import time. Existing
packages use two patterns: ``did/__init__.py`` imports the format functions
by name, while ``didcont/__init__.py`` imports the module as a private name.
Either works.

.. code-block:: python

   # myestimator/__init__.py
   from .estimator import my_estimator
   from .container import MyEstimatorResult

   # Either import the format functions by name...
   from .format import format_my_result

   # ...or import the module privately (both trigger attach_format)
   # from . import format as _format

   __all__ = ["my_estimator", "MyEstimatorResult", "format_my_result"]

Then register exports in the top-level `__init__.py <https://github.com/jordandeklerk/moderndid/blob/main/moderndid/__init__.py>`_.

.. important::

   The package uses a custom ``__getattr__`` for lazy imports. Do not add
   direct import statements at the top of ``moderndid/__init__.py``. Instead,
   update the lookup dictionaries described below.

1. Add each exported name to ``__all__``.
2. Add entries to ``_lazy_imports``, which maps a name to its module path.
   If your module has optional dependencies, use ``_optional_imports``
   instead, which maps a name to a ``(module_path, extra_name)`` tuple.
3. If your result class has a public alias, add it to ``_aliases``.
4. Add ``"myestimator"`` to ``_submodules`` so that
   ``import moderndid.myestimator`` works.

.. code-block:: python

   # In moderndid/__init__.py

   __all__ = [
       ...
       "MyEstimatorResult",
       "my_estimator",
       "format_my_result",
   ]

   # For modules with no extra dependencies:
   _lazy_imports = {
       ...
       "MyEstimatorResult": "moderndid.myestimator.container",
       "my_estimator": "moderndid.myestimator.estimator",
       "format_my_result": "moderndid.myestimator.format",
   }

   # Or for modules requiring extra dependencies:
   _optional_imports = {
       ...
       "my_estimator": ("moderndid.myestimator", "myestimator"),
   }

   _submodules = [..., "myestimator"]

The ``__getattr__`` function resolves names in priority order. If your
function name shadows its subpackage name (like ``drdid``), add an eager
import at the bottom of ``__init__.py`` so the function takes precedence.

.. code-block:: python

   # In moderndid/__init__.py

   def __getattr__(name):
       if name in _aliases:
           module_path, attr_name = _aliases[name]
           module = importlib.import_module(module_path)
           return getattr(module, attr_name)

       if name in _lazy_imports:
           module = importlib.import_module(_lazy_imports[name])
           return getattr(module, name)

       if name in _optional_imports:
           module_path, extra = _optional_imports[name]
           try:
               module = importlib.import_module(module_path)
               return getattr(module, name)
           except ImportError as e:
               raise ImportError(
                   f"'{name}' requires extra dependencies: "
                   f"uv pip install 'moderndid[{extra}]'"
               ) from e

       if name in _submodules:
           return importlib.import_module(f"moderndid.{name}")

       raise AttributeError(f"module 'moderndid' has no attribute {name!r}")


.. _new-estimator-maketables:

Step 9: Add Maketables Support
------------------------------

Result objects should implement the
`maketables <https://py-econometrics.github.io/maketables/>`_ plug-in
interface so they work with ``maketables.ETable`` and ``MTable`` out of
the box. The two packages do not need to know about each other; just define
the right properties and methods on your ``NamedTuple``.

Start by adding these imports to your result class in
``myestimator/container.py``.

.. code-block:: python

   from moderndid.core.maketables import (
       build_coef_table_with_ci,
       build_single_coef_table,
       control_group_label,
       est_method_label,
       make_group_time_names,
       make_effect_names,
       se_type_label,
   )
   from moderndid.core.result import extract_n_obs, extract_vcov_info

Then implement the following properties and methods on your ``NamedTuple``.

The ``__maketables_coef_table__`` property returns a pandas ``DataFrame``
with the coefficient table. The index holds coefficient names. The required
columns are ``b`` for the estimate, ``se`` for the standard error, ``t``
for the t-statistic, and ``p`` for the p-value. You can optionally include
``ci95l``, ``ci95u``, ``ci90l``, and ``ci90u`` for confidence interval
bounds.

For a single-ATT estimator, use ``build_single_coef_table`` which handles
the t-statistic, p-value, and CI computation for you. For multi-coefficient
results (group-time effects, event studies), use ``build_coef_table_with_ci``
with coefficient names from ``make_group_time_names`` or ``make_effect_names``.

.. code-block:: python

   @property
   def __maketables_coef_table__(self):
       # Single-coefficient result
       return build_single_coef_table("ATT", self.att, self.se)

   # Or for group-time results:
   @property
   def __maketables_coef_table__(self):
       names = make_group_time_names(self.groups, self.times)
       return build_coef_table_with_ci(
           names, self.att_gt, self.se_gt,
           critical_values=self.critical_value,
       )

``__maketables_stat__`` is a method (not a property) that takes a string
key and returns the corresponding model-level statistic. ``maketables``
calls it once for each stat it wants to display, and you should return
``None`` for keys you do not support. The standard keys used across existing
estimators are ``"N"`` for the observation count, ``"se_type"`` for
analytical or bootstrap, ``"control_group"``, and ``"estimation_method"``.

Use the label helpers from ``moderndid.core.maketables`` so that the raw
values stored in ``estimation_params`` get mapped to readable strings.
``se_type_label(True)`` returns ``"Bootstrap"``.
``control_group_label("nevertreated")`` returns ``"Never Treated"``.
``est_method_label("dr")`` returns ``"Doubly Robust"``.

.. code-block:: python

   def __maketables_stat__(self, key: str):
       if key == "N":
           return self.estimation_params.get("n_obs")
       if key == "se_type":
           return se_type_label(self.estimation_params.get("bootstrap", False))
       if key == "control_group":
           return control_group_label(
               self.estimation_params.get("control_group")
           )
       if key == "estimation_method":
           return est_method_label(
               self.estimation_params.get("estimation_method")
           )
       return None

The remaining five properties are short boilerplate. Implement all of them
on your result class.

.. code-block:: python

   @property
   def __maketables_depvar__(self) -> str:
       """Dependent variable name, used as the column header."""
       return self.estimation_params.get("yname", "")

   @property
   def __maketables_fixef_string__(self) -> str | None:
       """Fixed-effects formula. DiD estimators typically return None."""
       return None

   @property
   def __maketables_vcov_info__(self) -> dict:
       """Variance-covariance metadata (vcov_type, clustervar)."""
       return extract_vcov_info(self.estimation_params)

   @property
   def __maketables_stat_labels__(self) -> dict[str, str]:
       """Map stat keys to display labels. Unlisted keys are shown as-is."""
       return {
           "control_group": "Control Group",
           "estimation_method": "Estimation Method",
       }

   @property
   def __maketables_default_stat_keys__(self) -> list[str]:
       """Stats to show when the user does not pass model_stats."""
       return ["N", "se_type", "control_group"]

``extract_vcov_info`` reads the ``"bootstrap"`` and ``"cluster"`` keys from
``estimation_params``, falling back to ``"clustervars"`` when ``"cluster"``
is absent. ``moderndid.core.result`` also provides ``extract_n_obs``, which
tries ``"n_obs"`` first, then ``"n_units"``, and as a last resort uses the
first dimension of the influence function array.


Step 10: Add Distributed Support (optional)
-------------------------------------------

If your estimator performs independent group-time computations that can be
parallelized across a cluster, add Dask and/or Spark backends. The existing
distributed implementations for :func:`~moderndid.att_gt` and
:func:`~moderndid.ddd` provide a template. See :doc:`distributed_architecture`
for the reduction patterns and memory strategy.

The distributed backends follow a single design rule. **Never materialize the
full dataset on any single machine**. All computation happens on workers via
partition-level sufficient statistics. Only small summary matrices return to
the driver.

Add your Dask implementation in ``moderndid/dask/_my_estimator.py`` and your
Spark implementation in ``moderndid/spark/_my_estimator.py``. The main
estimator function delegates to these when it detects a Dask or Spark input
(see Step 4).


Step 11: Write Tests
--------------------

See :ref:`how to write tests <testing-how-to-write>` for detailed guidance on
testing conventions. Tests live in multiple directories depending on what
they cover.

**Estimator tests** (``tests/myestimator/``). Test basic functionality with
simple synthetic data where you know the correct answer. Cover edge cases
like no treated units, all units treated, or singular covariate matrices.
Use parameterization to test multiple estimation methods without duplicating
code. Mark slow tests with ``@pytest.mark.slow`` so they can be skipped
during rapid iteration.

**Preprocessing tests** (``tests/core/``). If you added custom validators,
transformers, or builder logic during Step 2, add tests here alongside the
existing preprocessing tests.

**R validation tests** (``tests/validation/``). When an R reference
implementation exists, validate your output against it. Each existing
estimator has a ``test_r_*.py`` file that writes an R script, runs it with
``subprocess.run(["R", "--vanilla", "--quiet"], ...)``, serializes results
to JSON via ``jsonlite``, and compares with ``np.testing.assert_allclose``.
Use ``atol=1e-6`` for point estimates and ``atol=1e-4`` for standard errors
and confidence intervals; bootstrap SE tolerances may need to be wider.
Guard these tests with an R availability check so the suite passes on
machines without R. See ``tests/validation/test_r_did.py`` for a complete
example.

**Plotting tests** (``tests/plotting/``). If you added a plot converter,
verify that the returned DataFrame has the expected columns and that
reference periods with NaN standard errors are filtered out.
