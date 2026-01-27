.. _architecture:

====================================
Architecture and API Design
====================================

This document describes the internal architecture of ModernDiD and provides
guidance for contributors who want to add new estimators or extend existing
functionality. Understanding these patterns will help you write code that
integrates seamlessly with the rest of the package.

Overview
========

ModernDiD is organized around a shared core and specialized estimator modules.
The ``core`` module provides infrastructure that all estimators rely on,
including a preprocessing pipeline for validating and transforming input data,
configuration dataclasses that define estimator parameters, and base utilities
for result objects. Each estimator module (``did``, ``drdid``, ``didcont``,
``didtriple``) contains the statistical implementation for its methodology,
following the patterns established in core. The ``didhonest`` module provides
sensitivity analysis that works with results from any estimator, and the
``plots`` module offers unified visualization across all result types.

This architecture means that adding a new estimator involves implementing the
statistical logic while reusing the preprocessing, result handling, and plotting
infrastructure. The sections below cover each component in detail.

The Preprocessing Pipeline
==========================

Most estimators in ModernDiD use a shared preprocessing pipeline built on the
builder pattern. The multi-period DiD, two-period doubly robust DiD, and
continuous treatment modules all use ``PreprocessDataBuilder``, which ensures
consistent data handling and prevents reimplementing validation logic.

Some estimators may require specialized preprocessing that does not fit cleanly
into the shared pipeline. When an estimator has genuinely unique data structure
requirements that are not reusable across methods, it is acceptable to keep
that preprocessing logic contained within the estimator module. The triple
differences module is an example of this, as it requires partition-specific
subgroup creation and collinearity checks that do not apply to other estimators.
If you find yourself in this situation, follow similar validation patterns to
the shared pipeline and document why the custom approach is necessary.

Configuration Classes
---------------------

Each estimator type has a configuration dataclass that inherits from
``BasePreprocessConfig``. These classes serve as both input validation
specifications and metadata containers. They define the expected parameters
and their types, and after preprocessing completes, computed values like
``time_periods`` and ``treated_groups`` are stored on the config for use
during estimation.

.. code-block:: python

   from dataclasses import dataclass
   from moderndid.core.preprocess.config import BasePreprocessConfig

   @dataclass
   class DIDConfig(BasePreprocessConfig):
       yname: str
       tname: str
       gname: str
       idname: str | None = None
       xformla: str = "~1"
       panel: bool = True
       allow_unbalanced_panel: bool = True
       weightsname: str | None = None
       control_group: ControlGroup = ControlGroup.NEVER_TREATED
       est_method: EstimationMethod = EstimationMethod.DOUBLY_ROBUST
       base_period: BasePeriod = BasePeriod.VARYING
       anticipation: int = 0
       alp: float = 0.05
       bstrap: bool = False
       biters: int = 1000
       clustervars: list[str] = field(default_factory=list)

Data Container Classes
----------------------

Preprocessed data is stored in dataclass containers that provide a clean
interface for estimators. These containers bundle all the data an estimator
needs, including the processed DataFrame, precomputed tensors for efficient
computation, and the configuration that produced them. Properties on the
container provide convenient access to common queries about the data structure.

.. code-block:: python

   @dataclass
   class DIDData:
       data: pl.DataFrame              # Processed data in long format
       time_invariant_data: pl.DataFrame  # Unit-level data
       weights: np.ndarray             # Observation weights
       outcomes_tensor: list[np.ndarray] | None  # For balanced panels
       covariates_matrix: np.ndarray | None
       covariates_tensor: list[np.ndarray] | None
       config: DIDConfig
       cluster: np.ndarray | None

       @property
       def is_panel(self) -> bool: ...

       @property
       def is_balanced_panel(self) -> bool: ...

       @property
       def has_covariates(self) -> bool: ...

The Builder Pattern
-------------------

The ``PreprocessDataBuilder`` class provides a fluent interface for the
preprocessing pipeline. The builder automatically selects the appropriate
validators and transformers based on the configuration type, which allows
the same builder to handle multi-period DiD, two-period DiD, continuous
treatment, and triple differences without the caller needing to know which
validators apply to which estimator type.

.. code-block:: python

   from moderndid.core.preprocess import PreprocessDataBuilder, DIDConfig

   config = DIDConfig(
       yname="outcome",
       tname="period",
       idname="unit_id",
       gname="treatment_group",
   )

   preprocessed = (
       PreprocessDataBuilder()
       .with_data(raw_dataframe)
       .with_config(config)
       .validate()      # Runs all validators
       .transform()     # Applies all transformations
       .build()         # Returns DIDData
   )

Constants and Enums
-------------------

Categorical parameters use string enums for type safety. Enums prevent typos
and enable IDE autocompletion while still accepting string values from users.
When a user passes ``est_method="dr"``, the enum automatically converts it
to the proper type.

.. code-block:: python

   from enum import Enum

   class ControlGroup(str, Enum):
       NEVER_TREATED = "nevertreated"
       NOT_YET_TREATED = "notyettreated"

   class EstimationMethod(str, Enum):
       DOUBLY_ROBUST = "dr"
       IPW = "ipw"
       REGRESSION = "reg"

   class BasePeriod(str, Enum):
       UNIVERSAL = "universal"
       VARYING = "varying"

Consistent Argument Naming
==========================

All estimators follow consistent naming conventions for parameters. This makes
the API predictable and reduces cognitive load for users. When you learn that
``yname`` specifies the outcome column in one estimator, you know it means the
same thing in every other estimator.

Column Name Parameters
----------------------

These parameters specify column names in the input data:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Description
   * - ``yname``
     - Outcome/dependent variable column
   * - ``tname``
     - Time period column
   * - ``idname``
     - Unit/entity identifier column (required for panel data)
   * - ``gname``
     - Treatment group column (first period treated, 0 for never-treated)
   * - ``dname``
     - Dose/treatment intensity column (continuous treatment only)
   * - ``pname``
     - Partition/eligibility indicator (triple differences only)
   * - ``treatname``
     - Binary treatment indicator (two-period models only)
   * - ``weightsname``
     - Sampling weights column

Estimation Parameters
---------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Description
   * - ``xformla``
     - Covariate formula in Wilkinson notation (e.g., ``"~ x1 + x2"``)
   * - ``est_method``
     - Estimation method: ``"dr"``, ``"ipw"``, or ``"reg"``
   * - ``control_group``
     - Control group: ``"nevertreated"`` or ``"notyettreated"``
   * - ``base_period``
     - Base period: ``"varying"`` or ``"universal"``
   * - ``anticipation``
     - Number of anticipation periods (default: 0)
   * - ``panel``
     - Whether data is panel (True) or repeated cross-section (False)
   * - ``allow_unbalanced_panel``
     - Whether to allow unbalanced panel data

Inference Parameters
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Description
   * - ``alp``
     - Significance level (default: 0.05)
   * - ``bstrap``
     - Whether to use bootstrap inference
   * - ``biters``
     - Number of bootstrap iterations (default: 1000)
   * - ``boot_type``
     - Bootstrap type: ``"weighted"`` or ``"multiplier"``
   * - ``cband``
     - Whether to compute simultaneous confidence bands
   * - ``clustervars``
     - Variables for clustered standard errors
   * - ``random_state``
     - Random seed for reproducibility

Result Object Design
====================

All estimators return immutable NamedTuple result objects. Immutability ensures
results cannot be accidentally modified during interactive analysis. The explicit
attribute definitions with type hints create a clear interface that documents
exactly what each estimator returns. Because NamedTuples are also tuples, results
can be unpacked for quick access when only a few values are needed.

Core Result Attributes
----------------------

All result objects include these core attributes:

.. code-block:: python

   from typing import NamedTuple

   class MPResult(NamedTuple):
       # Point estimates
       groups: np.ndarray          # Treatment groups
       times: np.ndarray           # Time periods
       att_gt: np.ndarray          # Group-time ATT estimates

       # Inference
       se_gt: np.ndarray           # Standard errors
       critical_value: float       # Critical value for CI
       vcov_analytical: np.ndarray | None  # Variance-covariance matrix

       # For downstream analysis
       influence_func: np.ndarray | None  # Influence functions

       # Metadata
       estimation_params: dict     # All estimation parameters
       n_units: int               # Number of units

Influence Functions
-------------------

Influence functions are a key architectural element that enable several important
capabilities. Analytical standard errors can be computed from the influence
function without resampling, which is faster and often more stable than
bootstrap methods. When bootstrap inference is needed, multiplier bootstrap
uses influence functions for efficient resampling without refitting the model.
Aggregation of group-time effects into summary measures uses influence functions
to propagate uncertainty correctly. The ``honest_did`` sensitivity analysis
function uses influence functions to construct robust confidence intervals
under violations of parallel trends.

All estimators should compute and return influence functions when possible.
This is not optional infrastructure but a core part of what makes the downstream
analysis tools work correctly.

Metadata Preservation
---------------------

Every result object includes an ``estimation_params`` dictionary containing
all parameters used in estimation. This enables reproducibility because users
can see exactly what was computed. Downstream analysis functions like aggregation
need to know the original setup, such as whether bootstrap was used and what
the significance level was. Having all relevant information in one place also
simplifies debugging when results are unexpected.

Creating a New Estimator
========================

This section walks through creating a new estimator that integrates with the
ModernDiD architecture. Following these steps ensures your estimator works
seamlessly with the plotting, aggregation, and sensitivity analysis infrastructure.

Step 1: Define the Configuration
--------------------------------

If your estimator needs parameters beyond what ``DIDConfig`` provides, create
a new configuration class. Inherit from ``BasePreprocessConfig`` to ensure
your config works with the preprocessing builder.

.. code-block:: python

   # myestimator/config.py
   from dataclasses import dataclass, field
   from moderndid.core.preprocess.config import BasePreprocessConfig

   @dataclass
   class MyEstimatorConfig(BasePreprocessConfig):
       yname: str
       tname: str
       idname: str
       gname: str
       # Add estimator-specific parameters
       my_special_param: float = 1.0
       another_param: str = "default"

Step 2: Define the Result Object
--------------------------------

Create a NamedTuple for your results. Include the standard attributes that
downstream tools expect: point estimates, standard errors, influence functions,
and the estimation parameters dictionary.

.. code-block:: python

   # myestimator/results.py
   from typing import NamedTuple
   import numpy as np

   class MyEstimatorResult(NamedTuple):
       att: np.ndarray
       se: np.ndarray
       influence_func: np.ndarray | None
       estimation_params: dict

Step 3: Implement the Estimator
-------------------------------

The main estimator function should accept user-friendly parameters like column
names as strings, build the configuration and preprocess data internally, perform
estimation, and return an immutable result object. Users should not need to
understand the preprocessing pipeline to use your estimator.

.. code-block:: python

   # myestimator/estimator.py
   import numpy as np
   from moderndid.core.preprocess import PreprocessDataBuilder
   from .config import MyEstimatorConfig
   from .results import MyEstimatorResult

   def my_estimator(
       data,
       yname: str,
       tname: str,
       idname: str,
       gname: str,
       xformla: str = "~1",
       my_special_param: float = 1.0,
       alp: float = 0.05,
   ) -> MyEstimatorResult:
       # 1. Build configuration
       config = MyEstimatorConfig(
           yname=yname,
           tname=tname,
           idname=idname,
           gname=gname,
           xformla=xformla,
           my_special_param=my_special_param,
           alp=alp,
       )

       # 2. Preprocess data
       preprocessed = (
           PreprocessDataBuilder()
           .with_data(data)
           .with_config(config)
           .validate()
           .transform()
           .build()
       )

       # 3. Perform estimation
       att, se, influence_func = _compute_estimates(preprocessed)

       # 4. Return result
       return MyEstimatorResult(
           att=att,
           se=se,
           influence_func=influence_func,
           estimation_params={
               "yname": yname,
               "tname": tname,
               "idname": idname,
               "gname": gname,
               "xformla": xformla,
               "my_special_param": my_special_param,
               "alp": alp,
           },
       )


   def _compute_estimates(data):
       # Implementation details...
       pass

Step 4: Add Plotting Support
----------------------------

Create a converter function that transforms your result to a DataFrame suitable
for plotting. The converter should compute confidence intervals from standard
errors and critical values, create any indicator columns needed for visualization,
and handle missing values gracefully.

.. code-block:: python

   # myestimator/converters.py
   import polars as pl
   from .results import MyEstimatorResult

   def myresult_to_polars(result: MyEstimatorResult) -> pl.DataFrame:
       return pl.DataFrame({
           "estimate": result.att,
           "se": result.se,
           "ci_lower": result.att - 1.96 * result.se,
           "ci_upper": result.att + 1.96 * result.se,
       })

Then add a plotting function or integrate with existing plot functions:

.. code-block:: python

   # myestimator/plots.py
   from plotnine import ggplot, aes, geom_point, geom_errorbar
   from .converters import myresult_to_polars

   def plot_my_estimator(result, show_ci=True):
       df = myresult_to_polars(result)
       # Build ggplot...

Step 5: Export the Public API
-----------------------------

Add your estimator to the module's ``__init__.py``:

.. code-block:: python

   # myestimator/__init__.py
   from .estimator import my_estimator
   from .results import MyEstimatorResult

   __all__ = ["my_estimator", "MyEstimatorResult"]

And to the main package ``__init__.py`` if it should be a top-level export:

.. code-block:: python

   # moderndid/__init__.py
   from moderndid.myestimator import my_estimator, MyEstimatorResult

Step 6: Write Tests
-------------------

See :doc:`testing` for detailed guidance on testing conventions. Test basic
functionality with simple synthetic data where you know the correct answer.
Test edge cases like no treated units, all units treated, or singular covariate
matrices. When an R implementation exists, validate against it. Use parameterization
to test multiple estimation methods without duplicating code. Mark slow tests
appropriately so they can be skipped during rapid development iteration.

Plotting Architecture
=====================

The plotting module provides a unified interface for visualizing results from
all estimators. This design means users learn one set of plotting functions
that work across the entire package.

Converter Functions
-------------------

Each result type has a converter that transforms it to a long-format DataFrame
suitable for plotting. Converters handle computing confidence intervals from
standard errors and critical values, creating indicator columns for pre/post
treatment periods, and dealing with missing values and edge cases.

.. code-block:: python

   # plots/converters.py

   def mpresult_to_polars(result: MPResult) -> pl.DataFrame:
       """Convert MPResult to plotting DataFrame."""
       ...

   def aggteresult_to_polars(result: AGGTEResult) -> pl.DataFrame:
       """Convert AGGTEResult to plotting DataFrame."""
       ...

Plot Functions
--------------

Plot functions accept result objects and delegate to the appropriate converter
based on the result type. This design allows adding support for new result
types by simply adding a new converter function without modifying the plotting
logic itself.

.. code-block:: python

   # plots/plots.py

   def plot_event_study(result, show_ci=True, ref_line=0, ...):
       # Detect result type and use appropriate converter
       if isinstance(result, AGGTEResult):
           df = aggteresult_to_polars(result)
       elif isinstance(result, PTEResult):
           df = pteresult_to_polars(result)
       # ... build ggplot

Integration with Sensitivity Analysis
=====================================

The ``honest_did`` function performs sensitivity analysis on event study
results. To make your estimator compatible with sensitivity analysis, ensure
your aggregated result object implements the ``EventStudyProtocol``. Any
result object with the required attributes can be passed to ``honest_did``,
which makes it straightforward to add sensitivity analysis support for new
estimators.

.. code-block:: python

   from typing import Protocol, runtime_checkable

   @runtime_checkable
   class EventStudyProtocol(Protocol):
       aggregation_type: str
       influence_func: np.ndarray | None
       event_times: np.ndarray | None
       att_by_event: np.ndarray | None
       estimation_params: dict

.. toctree::
   :hidden:
