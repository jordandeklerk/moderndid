.. _overview:

******************
What is ModernDiD?
******************

Difference-in-differences (DiD) is one of the most widely used methods for
causal inference from observational data. The modern DiD literature has
produced many estimators, but implementations are scattered across separate
R and Stata packages with incompatible APIs and output formats.

**ModernDiD** brings them together into a single Python library with a
consistent API for applied researchers, economists, and data scientists. If
you already use R packages for DiD, **ModernDiD** gives you the same
estimators in Python with a unified interface. If you are new to DiD, the
:ref:`introduction <causal_inference>` covers the key ideas.

Every estimator follows the same three-step workflow of **estimate,
aggregate, visualize**. Switching between designs means changing one
function call, not learning a new package.

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

Switching between estimators means changing one function call. The
aggregation and plotting interface stays the same.


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


.. _whatis-methods:

Estimators
----------

**ModernDiD** covers the main DiD designs used in applied work. Each estimator
targets a different treatment structure. See the
:ref:`estimator overview <estimator-overview>` for detailed descriptions,
key arguments, and guidance on when to use each one.

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Design
     - Reference
   * - **Staggered adoption** (:func:`~moderndid.att_gt`)
     - `Callaway & Sant'Anna (2021) <https://arxiv.org/abs/1803.09015>`_
   * - **Two-period DR DiD** (:func:`~moderndid.drdid`)
     - `Sant'Anna & Zhao (2020) <https://arxiv.org/abs/1812.01723>`_
   * - **Continuous treatment** (:func:`~moderndid.cont_did`)
     - `Callaway et al. (2024) <https://arxiv.org/abs/2107.02637>`_
   * - **Triple differences** (:func:`~moderndid.ddd`)
     - `Ortiz-Villavicencio & Sant'Anna (2025) <https://arxiv.org/abs/2505.09942>`_
   * - **Intertemporal treatment** (:func:`~moderndid.did_multiplegt`)
     - `de Chaisemartin & D'Haultfoeuille (2024) <https://doi.org/10.1162/rest_a_01414>`_
   * - **Dynamic covariate balancing DiD** (:func:`~moderndid.dyn_balancing`)
     - `Viviano & Bradic (2026) <https://doi.org/10.1093/biomet/asag016>`_
   * - **Sensitivity analysis** (:func:`~moderndid.honest_did`)
     - `Rambachan & Roth (2023) <https://asheshrambachan.github.io/assets/files/hpt-draft.pdf>`_
   * - **Extended TWFE** (:func:`~moderndid.etwfe`)
     - `Wooldridge (2023) <https://doi.org/10.1093/ectj/utad016>`_ and `Wooldridge (2025) <https://doi.org/10.1007/s00181-025-02807-z>`_
   * - **Nonparametric IV** (:func:`~moderndid.npiv`)
     - `Chen, Christensen & Kankanala (2024) <https://arxiv.org/abs/2107.11869>`_


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
