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
and Spark clusters for datasets that exceed single-machine memory, and accelerates computation
on NVIDIA GPUs via CuPy.

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

- **DataFrame agnostic.** Pass any
  `Arrow-compatible <https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html>`_
  DataFrame such as `polars <https://pola.rs/>`_,
  `pandas <https://pandas.pydata.org/>`_,
  `pyarrow <https://arrow.apache.org/docs/python/>`_,
  `duckdb <https://duckdb.org/>`_, and more powered by
  `narwhals <https://narwhals-dev.github.io/narwhals/>`_.
- **Distributed computing.** Scale DiD estimators to billions of observations
  across multi-node `Dask <https://www.dask.org/>`_ and
  `Spark <https://spark.apache.org/>`_ clusters with automatic dispatch.
  Simply pass a Dask or Spark DataFrame to supported estimators and the
  :doc:`distributed backend </user_guide/distributed>` activates transparently.
- **Fast computation.**
  `Polars <https://pola.rs/>`_ for internal data wrangling,
  `NumPy <https://numpy.org/>`_ vectorization,
  `Numba <https://numba.pydata.org/>`_ JIT compilation, and threaded parallel
  compute.
- **GPU acceleration.** Optional
  `CuPy <https://cupy.dev/>`_-accelerated regression and propensity score
  estimation across all doubly robust and IPW estimators on NVIDIA GPUs, with
  multi-GPU scaling in distributed environments. See the
  :doc:`GPU guide </user_guide/gpu>` for details.
- **Native plots.** Built-in visualizations powered by
  `plotnine <https://plotnine.org/>`_, returning standard ``ggplot`` objects
  you can customize with the full grammar of graphics.
- **Robust inference.** Analytical standard errors, bootstrap (weighted and
  multiplier), and simultaneous confidence bands.


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
