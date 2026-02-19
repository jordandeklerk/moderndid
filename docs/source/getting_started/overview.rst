.. _overview:

******************
What is ModernDiD?
******************

Difference-in-differences (DiD) is one of the most widely used methods for
causal inference from observational data. The modern DiD literature has
produced many estimators, but implementations are scattered across separate
R and Stata packages with incompatible APIs and output formats.

ModernDiD brings them altogether into a single Python library with a consistent API,
shared result objects, and unified plotting. It runs locally, scales to distributed Dask
clusters for datasets that exceed single-machine memory, and accelerates computation on NVIDIA
GPUs via CuPy.

.. code-block:: python

    import moderndid as did

    data = did.load_mpdta()

    result = did.att_gt(
        data=data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
    )

    event_study = did.aggte(result, type="dynamic")
    did.plot_event_study(event_study)

Install with:

.. code-block:: console

    uv pip install moderndid


.. _whatis-design:

Key features
------------

- **Consistent API across all estimators.** The same parameter names
  (``yname``, ``tname``, ``idname``, ``gname``) work across all estimators.
  Result objects share common structures and plotting functions accept
  results from any module.
- **Correctness first.** Every estimator is validated against reference R
  packages. Test suites include numerical checks for point estimates, standard
  errors, and confidence intervals.
- **DataFrame agnostic.** ModernDiD accepts any
  `Arrow-compatible <https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html>`_
  DataFrame, including polars, pandas, pyarrow, duckdb, and more, with no manual
  conversion.
- **Scales to clusters.** For data that exceed single-machine memory, passing a
  `Dask <https://www.dask.org/>`_ DataFrame to ``att_gt`` or ``ddd`` activates
  the :doc:`distributed backend </user_guide/distributed>`. Computation runs
  on workers using partition-level sufficient statistics, and supports
  multi-node clusters (Databricks, YARN, Kubernetes).
- **Fast.** Internally it uses Polars for grouping and reshaping,
  `Numba <https://numba.pydata.org/>`_ JIT compilation for bootstrap paths,
  and thread-based parallelism via ``n_jobs``. On NVIDIA hardware, the
  :doc:`GPU backend </user_guide/gpu>` accelerates regression and propensity
  score estimation via `CuPy <https://cupy.dev/>`_.
- **Consistent visualization.** All modules provide plotting functions built on
  `plotnine <https://plotnine.org/>`_ for event studies, dose-response curves,
  and sensitivity plots, with full grammar-of-graphics customization.


.. _whatis-methods:

Estimators
----------

ModernDiD covers the main DiD designs used in applied work.

- **Staggered adoption** (:mod:`~moderndid.did`) —
  `Callaway and Sant'Anna (2021) <https://arxiv.org/abs/1803.09015>`_.
  Group-time average treatment effects with staggered adoption, aggregated
  into event-study, group, or calendar summaries.
- **Two-period doubly robust DiD** (``drdid``) —
  `Sant'Anna and Zhao (2020) <https://arxiv.org/abs/1812.01723>`_.
  Classic two-period, two-group designs using inverse probability weighting,
  outcome regression, or doubly robust combinations.
- **Continuous treatment** (:mod:`~moderndid.didcont`) —
  `Callaway, Goodman-Bacon, and Sant'Anna (2024) <https://arxiv.org/abs/2107.02637>`_.
  Dose-response functions for settings where treatment intensity varies.
- **Triple differences** (:mod:`~moderndid.didtriple`) —
  `Ortiz-Villavicencio and Sant'Anna (2025) <https://arxiv.org/abs/2505.09942>`_.
  Adds a third dimension (such as eligibility status) to relax parallel
  trends assumptions.
- **Intertemporal treatment** (:mod:`~moderndid.didinter`) —
  `de Chaisemartin and D'Haultfoeuille (2024) <https://doi.org/10.1162/rest_a_01414>`_.
  Non-binary, non-absorbing treatments where current outcomes may depend on
  treatment history.
- **Sensitivity analysis** (:mod:`~moderndid.didhonest`) —
  `Rambachan and Roth (2023) <https://asheshrambachan.github.io/assets/files/hpt-draft.pdf>`_.
  Robust inference when parallel trends may be violated.


Next steps
----------

- :doc:`installation` for detailed install options and optional extras.
- :ref:`Quickstart <quickstart>` to learn the API with working examples.
- :ref:`Introduction to DiD <causal_inference>` for background on the
  difference-in-differences framework.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   self
   installation
   causal_inference
