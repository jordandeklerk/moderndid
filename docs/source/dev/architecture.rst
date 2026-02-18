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
configuration dataclasses that define estimator parameters, shared formatting
utilities for structured table output, and base utilities for result objects.
Each estimator module (``did``, ``drdid``, ``didcont``, ``didtriple``)
contains the statistical implementation for its methodology, following the
patterns established in core. Each module also has a ``format.py`` that
registers display formatting for its result objects. The ``didhonest`` module
provides sensitivity analysis that works with results from any estimator, and
the ``plots`` module offers unified visualization across all result types.

This architecture means that adding a new estimator involves implementing the
statistical logic while reusing the preprocessing, result handling, formatting,
and plotting infrastructure.

The Preprocessing Pipeline
==========================

All estimators in ModernDiD use a shared preprocessing pipeline built on the
builder pattern. The ``PreprocessDataBuilder`` provides a single entry point
for data validation and transformation, with module-specific behavior controlled
by configuration classes, validators, and transformers. Each module defines its
own config class (e.g., ``DIDConfig``, ``TwoPeriodDIDConfig``, ``DDDConfig``,
``DIDInterConfig``) and data container (e.g., ``DIDData``, ``DDDData``), but
the builder itself is shared. The builder selects the appropriate validators
and transformers based on the config type, so module-specific requirements like
partition-based subgroup creation or time-varying treatment handling are
expressed through the pipeline rather than bypassing it.

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
preprocessing pipeline. The configuration object passed to the builder
determines which validators and transformers run, so the caller only needs
to construct the right config and the builder handles the rest.

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

Thread-Based Parallelism
------------------------

Estimators that loop over group-time cells can use the ``parallel_map``
utility in ``moderndid.core.parallel`` to distribute work across threads.
Threads work well here because the per-cell computation is dominated by
NumPy, SciPy, and statsmodels C extensions that release the GIL. This
avoids the serialization overhead of multiprocessing while still achieving
concurrency.

.. code-block:: python

   from moderndid.core.parallel import parallel_map

   # Build a list of argument tuples, one per group-time cell
   args_list = [
       (group_idx, time_idx, data)
       for group_idx in range(n_groups)
       for time_idx in range(n_times)
   ]

   # Run sequentially or in parallel depending on n_jobs
   results = parallel_map(estimate_single_cell, args_list, n_jobs=n_jobs)

The ``n_jobs`` parameter follows scikit-learn conventions: ``1`` runs
sequentially, ``-1`` uses all available cores, and any value ``> 1`` uses
that many worker threads. Expose ``n_jobs`` as a parameter on your
estimator function with a default of ``1`` so that sequential execution
remains the default and parallelism is opt-in.

CuPy GPU Acceleration
---------------------

On machines with NVIDIA GPUs, ModernDiD can offload regression and propensity
score estimation to the GPU via `CuPy <https://cupy.dev/>`_. All GPU-related
code lives in the ``moderndid/cupy/`` module, which provides three files:

- ``backend.py`` — ``get_backend()``, ``set_backend()``, ``use_backend()``,
  ``to_device()``, ``to_numpy()`` for switching between NumPy and CuPy
  array libraries. The active backend is stored in a ``ContextVar``, so
  ``use_backend()`` scopes the override to a block and reverts
  automatically.
- ``regression.py`` — ``cupy_wls`` (weighted least squares) and
  ``cupy_logistic_irls`` (logistic regression via IRLS), implemented with
  the current backend's array operations.
- ``bootstrap.py`` — GPU-accelerated multiplier bootstrap and cluster
  aggregation helpers.

Estimators use a dispatch pattern that checks the active backend at runtime.
When the backend is CuPy, arrays are moved to the GPU with ``to_device()``
and regression is performed with ``cupy_wls`` or ``cupy_logistic_irls``.
Results are moved back to NumPy with ``to_numpy()`` before being returned.
When the backend is NumPy (the default), the standard statsmodels path runs
with no overhead.

.. code-block:: python

   from moderndid.cupy.backend import get_backend, to_numpy
   from moderndid.cupy.regression import cupy_wls

   xp = get_backend()
   if xp is not np:
       beta, fitted = cupy_wls(xp.asarray(y), xp.asarray(X), xp.asarray(w))
       params = to_numpy(beta)
   else:
       result = sm.WLS(y, X, weights=w).fit()
       params = result.params

Users enable GPU acceleration by installing the ``gpu`` extra
(``uv pip install moderndid[gpu]``) and either passing ``backend="cupy"``
to ``att_gt``/``ddd`` or calling ``set_backend("cupy")`` before running
an estimator. The ``gpu`` extra is not included in ``all`` because it
requires CUDA hardware.

The Formatting System
=====================

Every result object in ModernDiD has a formatted ``__repr__`` and ``__str__``
that produces structured table output when users call ``print()`` on a result.
This is implemented through a formatting layer in ``moderndid/core/format.py``
and per-module ``format.py`` files.

The ``attach_format`` Function
------------------------------

The core mechanism is ``attach_format``, which monkey-patches ``__repr__`` and
``__str__`` onto a result class so that printing it produces formatted output
instead of raw NamedTuple contents.

.. code-block:: python

   from moderndid.core.format import attach_format

   # At module level in myestimator/format.py
   attach_format(MyEstimatorResult, format_my_result)

After this call, ``print(result)`` will call ``format_my_result(result)``
instead of showing the default NamedTuple representation.

Shared Format Helpers
---------------------

The ``moderndid.core.format`` module provides helper functions that produce
consistent output across all estimators. Format functions should compose these
helpers rather than building strings from scratch.

``format_title(title, subtitle=None)``
   Produces the title block with thick ``=`` separators.

``format_section_header(label)``
   Produces a labeled section with thin ``-`` separators (used for Data Info,
   Estimation Details, Inference).

``format_footer(reference=None)``
   Produces a closing ``=`` separator with an optional citation line.

``format_significance_note(band=False)``
   Produces the significance code legend. Pass ``band=True`` when the table
   shows confidence bands rather than confidence intervals.

``format_single_result_table(label, att, se, conf_level, lci, uci, ...)``
   Builds a one-row summary table (e.g., for an overall ATT).

``format_event_table(col1_header, event_values, att, se, lower, upper, conf_level, band_type)``
   Builds a multi-row event study table.

``format_group_time_table(groups, times, att, se, lci, uci, conf_level, band_type)``
   Builds a group-time ATT table.

``format_horizon_table(horizons, estimates, std_errors, ci_lower, ci_upper, conf_level, ...)``
   Builds a horizon-indexed table for intertemporal effects.

``adjust_separators(lines)``
   Widens ``=`` and ``-`` separator lines to match the widest content line,
   useful when tables are wider than the default 78-character width.

All table helpers use `prettytable <https://github.com/prettytable/prettytable>`_ with the ``SINGLE_BORDER`` style internally.

Writing a Format Function
-------------------------

A format function takes a result object and returns a string. The typical
structure is title block, summary table, detail table, significance note,
metadata sections, and footer.

.. code-block:: python

   # myestimator/format.py
   from scipy import stats

   from moderndid.core.format import (
       attach_format,
       format_event_table,
       format_footer,
       format_section_header,
       format_significance_note,
       format_single_result_table,
       format_title,
   )

   from .results import MyEstimatorResult


   def format_my_result(result):
       lines = []

       lines.extend(format_title("My Estimator Results"))

       alpha = result.estimation_params.get("alp", 0.05)
       conf_level = int((1 - alpha) * 100)
       z_crit = stats.norm.ppf(1 - alpha / 2)

       lci = result.overall_att - z_crit * result.overall_se
       uci = result.overall_att + z_crit * result.overall_se

       lines.extend(
           format_single_result_table(
               "ATT", result.overall_att, result.overall_se,
               conf_level, lci, uci,
           )
       )

       lines.extend(format_significance_note())

       lines.extend(format_section_header("Data Info"))
       lines.append(f" Observations: {result.n_units}")

       lines.extend(format_footer("Reference: Author (Year)"))
       return "\n".join(lines)


   attach_format(MyEstimatorResult, format_my_result)

The ``attach_format`` call at module level means the format is registered as
soon as the module is imported. Each estimator module's ``__init__.py`` should
import the format module to ensure registration happens at import time.

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
       n_jobs: int = 1,
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

       # 3. Perform estimation (use parallel_map for group-time loops)
       att, se, influence_func = _compute_estimates(preprocessed, n_jobs=n_jobs)

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


   def _compute_estimates(data, n_jobs=1):
       from moderndid.core.parallel import parallel_map

       args_list = [
           (group_idx, time_idx, data)
           for group_idx in range(len(data.config.treated_groups))
           for time_idx in range(len(data.config.time_periods))
       ]
       results = parallel_map(_estimate_single_cell, args_list, n_jobs=n_jobs)
       # ... collect results into arrays ...

Step 4: Add Formatted Output
-----------------------------

Create a ``format.py`` module that gives your result readable ``print()``
output. Import the shared helpers from ``moderndid.core.format``, define a
format function, and register it with ``attach_format``. See
:ref:`The Formatting System <architecture>` above for the full pattern.

.. code-block:: python

   # myestimator/format.py
   from moderndid.core.format import (
       attach_format,
       format_footer,
       format_section_header,
       format_significance_note,
       format_single_result_table,
       format_title,
   )

   from .results import MyEstimatorResult


   def format_my_result(result):
       lines = []
       lines.extend(format_title("My Estimator Results"))
       # ... build sections using helpers ...
       lines.extend(format_footer("Reference: Author (Year)"))
       return "\n".join(lines)


   attach_format(MyEstimatorResult, format_my_result)

Step 5: Add Plotting Support
-----------------------------

Create a converter function in ``moderndid/plots/converters.py`` that
transforms your result to a Polars DataFrame for plotting. Converters must
use the standard column names that the plot functions expect, and should
filter out rows where the standard error is NaN (these correspond to
reference periods that should not be plotted).

For event study results, the expected columns are:

- ``event_time``: event time relative to treatment
- ``att``: point estimate
- ``se``: standard error
- ``ci_lower``: lower confidence interval bound
- ``ci_upper``: upper confidence interval bound
- ``treatment_status``: ``"Pre"`` or ``"Post"``

For group-time results, use ``group`` and ``time`` instead of ``event_time``.

.. code-block:: python

   # In moderndid/plots/converters.py

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

Then update the relevant plot function in ``moderndid/plots/plots.py`` to
dispatch to your converter. Plot functions use ``isinstance()`` checks to
route each result type to its converter:

.. code-block:: python

   # In moderndid/plots/plots.py

   from moderndid.myestimator.results import MyEstimatorResult
   from moderndid.plots.converters import myresult_to_polars

   def plot_event_study(result, ...):
       if isinstance(result, AGGTEResult):
           df = aggteresult_to_polars(result)
       elif isinstance(result, MyEstimatorResult):       # Add your type
           df = myresult_to_polars(result)
       # ... build ggplot

Step 6: Export the Public API
-----------------------------

Add your estimator to the module's ``__init__.py``. Import the format module
to ensure ``attach_format`` runs at import time.

.. code-block:: python

   # myestimator/__init__.py
   from .estimator import my_estimator
   from .format import format_my_result
   from .results import MyEstimatorResult

   __all__ = ["my_estimator", "MyEstimatorResult", "format_my_result"]

Then register exports in the top-level ``moderndid/__init__.py``. The package
uses a lazy-loading system with a custom ``__getattr__`` to defer imports
until first access, so you should not add direct import statements. Instead,
update the appropriate dictionaries:

1. Add each exported name to ``__all__``.

2. Add entries to ``_lazy_imports`` if the module has no extra dependencies,
   or to ``_optional_imports`` if it requires optional packages. The format
   for ``_lazy_imports`` maps each name to its module path. The format for
   ``_optional_imports`` maps each name to a ``(module_path, extra_name)``
   tuple, where ``extra_name`` is used in the installation hint
   (``uv pip install 'moderndid[extra_name]'``).

3. If your result class is re-exported under a different public name, add
   an entry to ``_aliases`` mapping the public name to
   ``(module_path, actual_class_name)``.

4. If you want ``import moderndid.myestimator`` to work, add
   ``"myestimator"`` to ``_submodules``.

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
       "MyEstimatorResult": "moderndid.myestimator.results",
       "my_estimator": "moderndid.myestimator.estimator",
       "format_my_result": "moderndid.myestimator.format",
   }

   # Or for modules requiring extra dependencies:
   _optional_imports = {
       ...
       "my_estimator": ("moderndid.myestimator", "myestimator"),
   }

   _submodules = [..., "myestimator"]

Step 7: Add Aggregation Support (Multi-Period Estimators)
---------------------------------------------------------

If your estimator produces group-time effects that should be aggregated into
event studies, group summaries, or an overall ATT, implement an aggregation
function following the ``aggte`` pattern. The aggregation function takes a
multi-period result object and returns an aggregated result, using influence
functions from the original estimation to propagate uncertainty correctly.

.. code-block:: python

   # myestimator/aggte.py

   def aggte(result, type="dynamic", ...):
       """Aggregate group-time effects.

       Parameters
       ----------
       result : MyMPResult
           Group-time result from the estimator.
       type : {'simple', 'dynamic', 'group', 'calendar'}
           Aggregation type.
       """
       # Use influence functions to compute aggregated ATT and SE
       ...

The aggregated result object should include ``overall_att``, ``overall_se``,
``aggregation_type``, and for non-simple aggregations: ``event_times``,
``att_by_event``, ``se_by_event``, ``critical_values``, and
``influence_func``. These fields are needed by the plotting converters and
the sensitivity analysis tools.

Step 8: Write Tests
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

Each result type has a converter in ``moderndid/plots/converters.py`` that
transforms it to a long-format Polars DataFrame suitable for plotting.
Converters handle computing confidence intervals from standard errors and
critical values, creating indicator columns for pre/post treatment periods,
and dealing with missing values and edge cases.

All converters follow the naming convention ``{resulttype}_to_polars`` and
produce DataFrames with standardized column names. Rows where the standard
error is NaN (reference periods) must be filtered out:

.. code-block:: python

   df = df.filter(~pl.col("se").is_nan())

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
based on the result type. Adding support for a new result type requires two
changes: writing a converter function in ``converters.py`` and adding an
``isinstance()`` branch in the relevant plot function in ``plots.py``.

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
