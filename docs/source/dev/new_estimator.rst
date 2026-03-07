.. _new-estimator:

========================
Creating a New Estimator
========================

Adding an estimator to **ModernDiD** requires familiarity with the underlying
econometric methodology, experience implementing statistical estimators in
numerical software, and a good degree of comfort with concepts common throughout
the field of econometrics and statistics. Contributors should
be able to validate their implementation against reference software (typically
an R package) before submitting.

A well-integrated estimator gives users a consistent experience across the
package. When your estimator follows the established patterns, users can apply
what they learned from other estimators without consulting documentation for
basic usage. It also means your estimator automatically works with the plotting,
aggregation, and sensitivity analysis tools that users expect.

This guide describes the general pattern that all existing estimators follow.
Stick to it as closely as possible so that your estimator integrates cleanly
with the rest of the package. That said, some estimators will need to deviate. For example,
an estimator may require additional dependencies, a different
data container, non-standard aggregation logic, or inference procedures that
don't fit the bootstrap/influence-function workflow. When deviations are
necessary, document them clearly and keep the user-facing API as close to the
standard pattern as you can.

Before starting, read the :doc:`architecture` guide to understand the
preprocessing pipeline, result object conventions, formatting system,
`maketables <https://py-econometrics.github.io/maketables/>`_ integration,
`plotnine <https://plotnine.org/>`_ plotting, and distributed support
(Dask and Spark) that your estimator will build on.

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
and the estimation parameters dictionary. Use a ``"Container for ..."`` docstring
with a numpydoc ``Attributes`` section, and add ``#:`` doc comments before each
field so that the API docs render meaningful descriptions.

.. code-block:: python

   # myestimator/results.py
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
       influence_func : ndarray or None
           Influence function matrix.
       estimation_params : dict
           Parameters used for estimation.
       """

       #: Point estimates.
       att: np.ndarray
       #: Standard errors.
       se: np.ndarray
       #: Influence function matrix.
       influence_func: np.ndarray | None
       #: Parameters used for estimation.
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
:ref:`The Formatting System <architecture>` for the full pattern.

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

For event study results, the expected columns are

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
route each result type to its converter.

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
update the appropriate dictionaries.

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
``aggregation_type``, and for non-simple aggregations, ``event_times``,
``att_by_event``, ``se_by_event``, ``critical_values``, and
``influence_func``. These fields are needed by the plotting converters and
the sensitivity analysis tools.

Step 8: Add Maketables Support
------------------------------

Implement the maketables plug-in interface on your result class so it works
with ``maketables.ETable`` and ``MTable`` out of the box. Use the shared
helpers in ``moderndid.core.maketables`` to build the coefficient table,
map labels, and report model-level statistics. See
:ref:`Adding Maketables Support to a New Estimator <architecture-maketables>`
for the full interface and a worked example.

Step 9: Add Distributed Support (Optional)
-------------------------------------------

If your estimator performs independent group-time computations that can be
parallelized across a cluster, add Dask and/or Spark backends. The existing
distributed implementations for :func:`~moderndid.att_gt` and
:func:`~moderndid.ddd` provide a template. See :doc:`distributed_architecture`
for the reduction patterns and memory strategy.

Step 10: Write Tests
--------------------

See :ref:`how to write tests <testing-how-to-write>` for detailed guidance on
testing conventions. Test basic
functionality with simple synthetic data where you know the correct answer.
Test edge cases like no treated units, all units treated, or singular covariate
matrices. When an R implementation exists, validate against it. Use parameterization
to test multiple estimation methods without duplicating code. Mark slow tests
appropriately so they can be skipped during rapid development iteration.
