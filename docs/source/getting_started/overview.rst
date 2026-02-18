.. _overview:

******************
What is ModernDiD?
******************

ModernDiD is a Python library for modern difference-in-differences (DiD)
methods for causal inference. DiD methods estimate causal effects by comparing how
outcomes change over time for treated units against how they change for
untreated units, attributing the divergence to the treatment itself.

Recent econometric advances have extended the DiD framework to handle staggered treatment
adoption, continuous treatment intensities, heterogeneous effects, intertemporal
dynamics, potential violations of the parallel trends assumption, and more. ModernDiD
brings these methods together under a single, unified API.


.. _whatis-unified:

Why a unified package?
----------------------

The modern DiD literature has produced many methodological advances, but
implementations remain fragmented across separate R and Stata packages, each
with its own API, data requirements, and output formats. Researchers must
learn multiple interfaces, convert data between formats, and reconcile
different output structures.

ModernDiD provides a single interface where the same parameter names
work across all estimators. Result objects share common structures and
plotting functions accept results from any module.


.. _whatis-methods:

Methods
-------

ModernDiD covers the main DiD designs used in applied work.

**Multi-period staggered adoption** (:mod:`~moderndid.did`) implements
`Callaway and Sant'Anna (2021) <https://arxiv.org/abs/1803.09015>`_
to estimate group-time average treatment effects with staggered adoption,
then aggregate them into event-study, group, or calendar summaries.

**Two-period doubly robust DiD** (``drdid``) provides
`Sant'Anna and Zhao (2020) <https://arxiv.org/abs/1812.01723>`_
estimators for classic two-period, two-group designs using inverse
probability weighting, outcome regression, or doubly robust combinations.

**Continuous-treatment DiD** (:mod:`~moderndid.didcont`) implements
`Callaway, Goodman-Bacon, and Sant'Anna (2024) <https://arxiv.org/abs/2107.02637>`_
for settings where treatment intensity varies, producing dose-response
functions that show how effects change with dose.

**Triple-difference DiD** (:mod:`~moderndid.didtriple`) implements
`Ortiz-Villavicencio and Sant'Anna (2025) <https://arxiv.org/abs/2505.09942>`_,
which adds a third dimension (such as eligibility status) to relax
parallel trends assumptions.

**Intertemporal DiD** (:mod:`~moderndid.didinter`) implements
`de Chaisemartin and D'Haultfoeuille (2024) <https://doi.org/10.1162/rest_a_01414>`_
for non-binary, non-absorbing treatments where current outcomes may depend
on treatment history.

**Sensitivity analysis** (:mod:`~moderndid.didhonest`) provides
`Rambachan and Roth (2023) <https://asheshrambachan.github.io/assets/files/hpt-draft.pdf>`_
for robust inference when parallel trends may be violated.


.. _whatis-design:

Design
------

ModernDiD is built around practical design principles.

**Correctness first.** Every estimator is validated against reference R
packages. Test suites include numerical checks for point estimates, standard
errors, and confidence intervals.

**Safe defaults.** Doubly robust estimation is the default to protect against
model misspecification. Never-treated units are the default control group to
avoid contamination from already treated units.

**Performance on a single machine.** ModernDiD accepts any
`Arrow-compatible <https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html>`_
DataFrame, including polars, pandas, pyarrow, and duckdb. Internally it uses
Polars for grouping and reshaping, and uses
`Numba <https://numba.pydata.org/>`_ JIT compilation for bootstrap-intensive
code paths. Estimators that loop over group-time cells support thread-based
parallelism via ``n_jobs``. On NVIDIA hardware, installing the ``gpu`` extra
(``uv pip install moderndid[gpu]``) enables
`CuPy <https://cupy.dev/>`_-accelerated regression and propensity score
estimation for doubly robust and IPW estimators.

**Distributed computing when needed.** For data that exceed single-machine memory,
passing a `Dask <https://www.dask.org/>`_ DataFrame to ``att_gt`` or ``ddd``
activates the distributed backend. Computation runs on workers using
partition-level sufficient statistics, and only small summary matrices are
returned to the driver. The backend supports multi-node clusters
(Databricks, YARN, Kubernetes) and matches local estimator results
numerically. See the :ref:`Distributed Computing guide <distributed>` for
architecture details.

**Transparent outputs.** Result objects include influence functions,
variance-covariance matrices, and estimation metadata. Warnings explain data
conditions that may affect inference.

**Consistent visualization.** All modules provide plotting functions built on
`plotnine <https://plotnine.org/>`_ for event studies, dose-response curves,
and sensitivity plots, with full grammar-of-graphics customization.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   self
   installation
   causal_inference
