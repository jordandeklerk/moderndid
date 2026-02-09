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

**Multi-period staggered DiD** (:mod:`~moderndid.did`) implements
`Callaway and Sant'Anna (2021) <https://arxiv.org/abs/1803.09015>`_
for estimating group-time average treatment effects with staggered adoption,
including aggregation to event studies, group effects, and calendar time effects.

**Doubly robust two-period DiD** (``drdid``) provides
`Sant'Anna and Zhao (2020) <https://arxiv.org/abs/1812.01723>`_
estimators for classic two-period, two-group settings with inverse probability
weighting, outcome regression, or doubly robust combinations.

**Continuous treatment DiD** (:mod:`~moderndid.didcont`) implements
`Callaway, Goodman-Bacon, and Sant'Anna (2024) <https://arxiv.org/abs/2107.02637>`_
for settings with continuous treatment intensity, producing dose-response
functions showing how effects vary with treatment dose.

**Triple difference-in-differences** (:mod:`~moderndid.didtriple`)
implements `Ortiz-Villavicencio and Sant'Anna (2025) <https://arxiv.org/abs/2505.09942>`_
which leverages a third dimension of variation (such as eligibility status)
to relax parallel trends assumptions.

**Intertemporal DiD** (:mod:`~moderndid.didinter`) implements
`de Chaisemartin and D'Haultfoeuille (2024) <https://doi.org/10.1162/rest_a_01414>`_
for non-binary, non-absorbing treatments where lagged treatments may affect outcomes.

**Sensitivity analysis** (:mod:`~moderndid.didhonest`) provides
`Rambachan and Roth (2023) <https://asheshrambachan.github.io/assets/files/hpt-draft.pdf>`_
tools for robust inference under violations of parallel trends.


.. _whatis-design:

Design
------

**Correctness.** Every estimator is validated against reference R packages.
Test suites include numerical comparisons ensuring ModernDiD produces the
same point estimates, standard errors, and confidence intervals.

**Sensible defaults.** Doubly robust estimation is default because it protects
against misspecification. Never-treated units serve as the default control
group to avoid contamination from already-treated units.

**Performance.** ModernDiD accepts any
`Arrow-compatible <https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html>`_
DataFrame including polars, pandas, pyarrow, and duckdb. Internal operations
use Polars for fast grouping and reshaping. Bootstrap procedures use
`Numba <https://numba.pydata.org/>`_ JIT compilation for near-C speeds.
Estimators that loop over group-time cells support optional thread-based
parallelism via the ``n_jobs`` parameter. On machines with NVIDIA GPUs,
installing the ``gpu`` extra (``uv pip install moderndid[gpu]``) enables
`CuPy <https://cupy.dev/>`_-accelerated regression and propensity score
estimation across all doubly robust and IPW estimators.

**Transparency.** Result objects include influence functions, variance-covariance
matrices, and estimation metadata. Warning messages explain when data issues
might affect results.

**Visualization.** All modules include plotting functions built on
`plotnine <https://plotnine.org/>`_ for event studies, dose-response curves,
and sensitivity plots with full customization through the grammar of graphics.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   self
   installation
   causal_inference
