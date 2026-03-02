.. _overview:

******************
What is ModernDiD?
******************

Difference-in-differences (DiD) is one of the most widely used methods for
causal inference from observational data. The modern DiD literature has
produced many estimators, but implementations are scattered across separate
R and Stata packages with incompatible APIs and output formats.

ModernDiD brings them together into a single Python library with a
consistent API. Every estimator follows the same three-step workflow of
**estimate, aggregate, visualize**. Switching between designs means
changing one function call, not learning a new package.

.. code-block:: python

    import moderndid as did

    # Works with pandas, polars, pyarrow, duckdb, or any Arrow-compatible DataFrame
    data = did.load_mpdta()

    # Estimate group-time ATTs
    result = did.att_gt(
        data=data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
    )

    # Aggregate into an event study
    event_study = did.aggte(result, type="dynamic")

    # Visualize
    did.plot_event_study(event_study)

The same pattern applies across every estimator: :func:`~moderndid.cont_did`
for continuous treatments, :func:`~moderndid.ddd` for triple differences,
:func:`~moderndid.did_multiplegt` for intertemporal designs, and
:func:`~moderndid.honest_did` for sensitivity analysis. See the
:ref:`estimator overview <estimator-overview>` for guidance on choosing the
right one.

See :doc:`installation` for install options and optional extras.


.. _whatis-design:

Key features
------------

- **Consistent API.** Every estimator returns a typed result object with the
  same interface for summarizing, aggregating, and plotting. Learn the
  pattern once and apply it everywhere.
- **DataFrame agnostic.** Pass any
  `Arrow-compatible <https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html>`_
  DataFrame such as
  `polars <https://pola.rs/>`_,
  `pandas <https://pandas.pydata.org/>`_,
  `pyarrow <https://arrow.apache.org/docs/python/>`_,
  `duckdb <https://duckdb.org/>`_, and more, powered by
  `narwhals <https://narwhals-dev.github.io/narwhals/>`_.
- **Scales up.** Runs locally on a laptop, then transparently scales to
  multi-node `Dask <https://www.dask.org/>`_ and
  `Spark <https://spark.apache.org/>`_ clusters for datasets that exceed
  single-machine memory. Just pass a Dask or Spark DataFrame and the
  :doc:`distributed backend </user_guide/distributed>` activates
  automatically.
- **Fast computation.**
  `Polars <https://pola.rs/>`_ for internal data wrangling,
  `NumPy <https://numpy.org/>`_ vectorization,
  `Numba <https://numba.pydata.org/>`_ JIT compilation, and threaded
  parallel compute.
- **GPU acceleration.** Optional
  `CuPy <https://cupy.dev/>`_-accelerated regression and propensity score
  estimation on NVIDIA GPUs, with multi-GPU scaling in distributed
  environments. See the :doc:`GPU guide </user_guide/gpu>`.
- **Native plots.** Built-in
  `plotnine <https://plotnine.org/>`_ visualizations returning standard
  ``ggplot`` objects you can customize with the full grammar of graphics.
- **Robust inference.** Analytical standard errors, bootstrap (weighted and
  multiplier), and simultaneous confidence bands.


.. _whatis-methods:

Estimators
----------

ModernDiD covers the main DiD designs used in applied work. Each estimator
targets a different treatment structure. See the
:ref:`estimator overview <estimator-overview>` for detailed descriptions,
key arguments, and guidance on when to use each one.

.. list-table::
   :widths: 30 40 30
   :header-rows: 1

   * - Design
     - When to use
     - Reference
   * - **Staggered adoption** (:func:`~moderndid.att_gt`)
     - Binary treatment turns on permanently at different times
     - `Callaway & Sant'Anna (2021) <https://arxiv.org/abs/1803.09015>`_
   * - **Two-period DR DiD** (``drdid``)
     - Classic two-period, two-group setting
     - `Sant'Anna & Zhao (2020) <https://arxiv.org/abs/1812.01723>`_
   * - **Continuous treatment** (:func:`~moderndid.cont_did`)
     - Treatment intensity varies across units
     - `Callaway et al. (2024) <https://arxiv.org/abs/2107.02637>`_
   * - **Triple differences** (:func:`~moderndid.ddd`)
     - Within-group eligibility variation (e.g., eligible vs. ineligible)
     - `Ortiz-Villavicencio & Sant'Anna (2025) <https://arxiv.org/abs/2505.09942>`_
   * - **Intertemporal treatment** (:func:`~moderndid.did_multiplegt`)
     - Non-absorbing treatment that switches on/off over time
     - `de Chaisemartin & D'Haultfoeuille (2024) <https://doi.org/10.1162/rest_a_01414>`_
   * - **Sensitivity analysis** (:func:`~moderndid.honest_did`)
     - Assess robustness to parallel trends violations
     - `Rambachan & Roth (2023) <https://asheshrambachan.github.io/assets/files/hpt-draft.pdf>`_


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
