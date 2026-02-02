.. _architecture:

====================================
Architecture and API Design
====================================

Understanding the internal architecture of ModernDiD will help you add new
estimators or extend existing functionality. The patterns described here
ensure your code integrates seamlessly with the rest of the package.

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
infrastructure.

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
       boot: bool = False
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
   * - ``boot``
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

Implementation Standards
========================

ModernDiD uses specific libraries for data handling and performance-critical
code. Following these standards ensures consistency and maintains the
performance characteristics users expect.

Polars for Data Handling
------------------------

All internal data manipulation uses Polars rather than pandas. Polars provides
better performance for the operations common in DiD estimation and has a more
consistent API. When users pass any Arrow-compatible DataFrame (polars, pandas,
pyarrow, duckdb, etc.), the preprocessing pipeline converts it to Polars at the
boundary via the `Arrow PyCapsule Interface <https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html>`_
using `narwhals <https://narwhals-dev.github.io/narwhals/>`_, and all subsequent
operations work with Polars DataFrames.

If you are adding new preprocessing logic or data transformations within an
estimator, use Polars operations. Avoid converting back to pandas for intermediate
steps. The ``moderndid.core.dataframe`` module provides ``to_polars()`` for
converting user input at the API boundary.

Common Polars patterns used throughout the codebase:

.. code-block:: python

   import polars as pl
   from moderndid.core.dataframe import to_polars

   # Convert data to Polars
   df = to_polars(data)

   # Filter rows
   treated = df.filter(pl.col("group") > 0)
   balanced_ids = counts.filter(pl.col("len") == n_periods)

   # Add or modify columns
   df = df.with_columns(pl.Series(name="_weights", values=weights))
   df = df.with_columns(pl.col("group").cast(pl.Float64))

   # Conditional expressions
   df = df.with_columns(
       pl.when(pl.col("group") == 0)
       .then(pl.lit(float("inf")))
       .otherwise(pl.col("group"))
       .alias("group")
   )

   # Group operations
   counts = df.group_by("unit_id").len()
   complete_ids = counts.filter(pl.col("len") == n_periods)["unit_id"].to_list()

   # Window operations for panel data
   df = df.with_columns(
       (pl.col("outcome") - pl.col("outcome").shift(1).over("unit_id")).alias("dy")
   )

Numba for Performance
---------------------

Computationally intensive operations should use Numba JIT compilation rather
than pure Python loops. Numba compiles Python functions to machine code,
providing performance comparable to C while keeping the code readable. This
is particularly important for bootstrap procedures where the same computation
runs thousands of times, and for any code that runs in tight loops over large
arrays.

Consider computing bootstrap standard errors for group-time effects. Each
bootstrap iteration requires resampling clusters and recomputing weighted
means. In pure Python, this nested loop over bootstrap iterations and
clusters is painfully slow.

.. code-block:: python

   import numpy as np

   # Pure Python - slow
   def bootstrap_means_python(data, weights, cluster_ids, n_boot):
       n_clusters = len(np.unique(cluster_ids))
       unique_clusters = np.unique(cluster_ids)
       results = np.zeros(n_boot)

       for b in range(n_boot):
           sampled = np.random.choice(unique_clusters, size=n_clusters, replace=True)
           total = 0.0
           weight_sum = 0.0
           for c in sampled:
               mask = cluster_ids == c
               total += np.sum(data[mask] * weights[mask])
               weight_sum += np.sum(weights[mask])
           results[b] = total / weight_sum

       return results

With 1000 bootstrap iterations and 500 clusters, this function spends most of
its time in Python's interpreter rather than doing actual computation. Numba
eliminates this overhead by compiling the function to machine code.

.. code-block:: python

   import numba as nb

   @nb.njit(cache=True, parallel=True)
   def bootstrap_means_numba(data, weights, cluster_ids, n_boot, seed):
       n_clusters = len(np.unique(cluster_ids))
       unique_clusters = np.unique(cluster_ids)
       results = np.zeros(n_boot)

       for b in nb.prange(n_boot):
           np.random.seed(seed + b)
           sampled = np.random.choice(unique_clusters, size=n_clusters, replace=True)
           total = 0.0
           weight_sum = 0.0
           for c in sampled:
               for i in range(len(cluster_ids)):
                   if cluster_ids[i] == c:
                       total += data[i] * weights[i]
                       weight_sum += weights[i]
           results[b] = total / weight_sum

       return results

The Numba version uses ``nb.prange`` instead of ``range`` for the outer loop,
enabling automatic parallelization across CPU cores. The ``cache=True`` argument
stores the compiled function on disk, avoiding recompilation on subsequent runs.
Speedups vary depending on the workload and data size, but can be substantial
for the nested loops common in bootstrap procedures.

The pattern used throughout ModernDiD defines a pure Python/NumPy fallback
first, then conditionally overrides it with a Numba-compiled version. This
ensures the code works even when Numba is not installed.

.. code-block:: python

   try:
       import numba as nb
       HAS_NUMBA = True
   except ImportError:
       HAS_NUMBA = False
       nb = None


   def _compute_impl(data, weights):
       # Pure NumPy fallback
       return np.sum(data * weights) / np.sum(weights)


   if HAS_NUMBA:

       @nb.njit(cache=True)
       def _compute_impl(data, weights):
           total = 0.0
           weight_sum = 0.0
           for i in range(len(data)):
               total += data[i] * weights[i]
               weight_sum += weights[i]
           return total / weight_sum

For element-wise operations on arrays, ``guvectorize`` provides a cleaner
interface than writing explicit loops. It defines a generalized ``ufunc`` that
NumPy can broadcast automatically. The signature specifies input and output
array shapes.

.. code-block:: python

   from numba import float64, guvectorize

   # Pure Python/NumPy fallback
   def _safe_divide_impl(x, y, out=None):
       if out is None:
           out = np.zeros_like(x, dtype=float)
       mask = np.abs(y) >= 1e-10
       np.divide(x, y, out=out, where=mask)
       out[~mask] = 0.0
       return out


   if HAS_NUMBA:

       @guvectorize(
           [(float64[:], float64[:], float64[:])],
           "(n),(n)->(n)",
           nopython=True,
           cache=True,
       )
       def _safe_divide_impl(x, y, result):
           for i in range(x.shape[0]):
               if np.abs(y[i]) < 1e-10:
                   result[i] = 0.0
               else:
                   result[i] = x[i] / y[i]

The signature ``"(n),(n)->(n)"`` means the function takes two arrays of the
same length and produces an output array of that length. Numba handles memory
allocation and broadcasting, so the compiled function works seamlessly with
NumPy's array operations.

When writing Numba functions, avoid Python objects and stick to NumPy arrays
and scalar types. Numba works best with simple numerical code. If you need
complex logic, keep it in pure Python and only JIT-compile the hot loops.

Creating a New Estimator
========================

A well-integrated estimator gives users a consistent experience across the
package. When your estimator follows the established patterns, users can apply
what they learned from other estimators without consulting documentation for
basic usage. It also means your estimator automatically works with the plotting,
aggregation, and sensitivity analysis tools that users expect.

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

Create a ``NamedTuple`` for your results. Include the standard attributes that
downstream tools expect, e.g., point estimates, standard errors, influence functions,
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
