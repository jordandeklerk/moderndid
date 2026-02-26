.. _acknowledgements:

################
Acknowledgements
################

Like many open-source software projects, ModernDiD builds on work and ideas
first developed in other packages. In this section, we want to acknowledge and
express our appreciation for the authors of these packages and their creativity
and hard work.

Software
========

Unless explicitly stated otherwise, all ModernDiD code is written
independently from scratch. The packages listed below have influenced
ModernDiD's API design or algorithmic choices, or are used for testing
ModernDiD as reference implementations, but no source code has been copied
except where explicitly stated (with license and permission details provided
inline).

did (R)
-------

The `did <https://github.com/bcallaway11/did>`_ R package by
`Brantly Callaway <https://bcallaway11.github.io/>`_ and
`Pedro H.C. Sant'Anna <https://pedrohcgs.github.io/>`_ is the foundation of
ModernDiD's multi-period staggered DiD estimator. The
`att_gt() <api/generated/multiperiod/moderndid.att_gt.html>`_ function in
ModernDiD implements the methodology in
`Callaway and Sant'Anna (2021) <https://doi.org/10.1016/j.jeconom.2020.12.001>`_
and follows the ``did`` package's API conventions for specifying group-time
average treatment effects, aggregation schemes, and inference.

More concretely, we have borrowed the following API conventions and ideas
directly from ``did``:

- **Core API** --
  `att_gt() <api/generated/multiperiod/moderndid.att_gt.html>`_ function and
  argument names (``yname``, ``tname``, ``idname``, ``gname``), control group
  options (``never_treated``, ``not_yet_treated``), and anticipation period
  handling
- **Aggregation** --
  `aggte() <api/generated/multiperiod/moderndid.aggte.html>`_ function with
  aggregation types (``group``, ``dynamic``, ``calendar``, ``simple``) directly
  mirrors the R package
- **Estimation methods** -- The doubly robust (``dr``), inverse probability
  weighting (``ipw``), and outcome regression (``reg``) estimation method
  options
- **Inference** -- Analytical standard error formulas and multiplier bootstrap
  for simultaneous confidence bands
- **Plotting** --
  `plot_gt() <api/generated/plotting/moderndid.plots.plot_gt.html>`_ and
  `plot_event_study() <api/generated/plotting/moderndid.plots.plot_event_study.html>`_
  follow the visual conventions of ``did``'s ``ggdid()``

You can learn more about ``did`` `on GitHub <https://github.com/bcallaway11/did>`_
or by reading the `associated paper <https://doi.org/10.1016/j.jeconom.2020.12.001>`_.

ModernDiD is benchmarked and validated against ``did`` via R scripts to ensure
numerical equivalence for coefficients, standard errors, and confidence
intervals.

DRDID (R)
---------

The `DRDID <https://github.com/pedrohcgs/DRDID>`_ R package by
`Pedro H.C. Sant'Anna <https://pedrohcgs.github.io/>`_ and
Jun Zhao is the foundation for
ModernDiD's two-period doubly robust estimators in the
`drdid <api/generated/drdid/moderndid.drdid.html>`_ module. The methodology
follows
`Sant'Anna and Zhao (2020) <https://doi.org/10.1016/j.jeconom.2020.06.003>`_,
which develops locally efficient doubly robust DiD estimators for both panel and
repeated cross-section settings.

ModernDiD implements all estimators from the ``DRDID`` package: the doubly
robust improved estimator
(`drdid_imp_rc() <api/generated/drdid/moderndid.drdid_imp_rc.html>`_),
traditional doubly robust
(`drdid_rc() <api/generated/drdid/moderndid.drdid_rc.html>`_), IPW estimators
(`ipw_did_rc() <api/generated/drdid/moderndid.ipw_did_rc.html>`_), and outcome
regression estimators
(`reg_did_rc() <api/generated/drdid/moderndid.reg_did_rc.html>`_), for both
panel data and repeated cross-sections.

You can learn more about ``DRDID`` `on GitHub <https://github.com/pedrohcgs/DRDID>`_
or by reading the `associated paper <https://doi.org/10.1016/j.jeconom.2020.06.003>`_.

ModernDiD is benchmarked and validated against ``DRDID`` via R scripts to
ensure numerical equivalence.

contdid (R)
-----------

The `contdid <https://github.com/bcallaway11/contdid>`_ R package by
`Brantly Callaway <https://bcallaway11.github.io/>`_,
`Andrew Goodman-Bacon <https://goodman-bacon.com/>`_, and
`Pedro H.C. Sant'Anna <https://pedrohcgs.github.io/>`_ is the basis for
ModernDiD's continuous treatment DiD estimator. The
`cont_did() <api/generated/didcont/moderndid.cont_did.html>`_ function
implements the methodology in
`Callaway, Goodman-Bacon, and Sant'Anna (2024) <https://arxiv.org/abs/2107.02637>`_,
which extends the DiD framework to settings where treatment intensity varies
continuously across units.

You can learn more about ``contdid``
`on GitHub <https://github.com/bcallaway11/contdid>`_ or by reading the
`associated paper <https://arxiv.org/abs/2107.02637>`_.

ModernDiD is benchmarked and validated against ``contdid`` via R scripts to
ensure numerical equivalence.

triplediff (R)
--------------

The `triplediff <https://github.com/marcelortizv/triplediff>`_ R package by
`Marcel Ortiz-Villavicencio <https://marcelortizv.github.io/>`_ and
`Pedro H.C. Sant'Anna <https://pedrohcgs.github.io/>`_ is the basis for
ModernDiD's triple difference-in-differences estimator. The
`ddd() <api/generated/didtriple/moderndid.ddd.html>`_ function implements the
methodology in
`Ortiz-Villavicencio and Sant'Anna (2025) <https://arxiv.org/abs/2505.09942>`_,
which develops doubly robust triple DiD estimators for staggered adoption
designs.

You can learn more about ``triplediff``
`on GitHub <https://github.com/marcelortizv/triplediff>`_ or by reading the
`associated paper <https://arxiv.org/abs/2505.09942>`_.

ModernDiD is benchmarked and validated against ``triplediff`` via R scripts to
ensure numerical equivalence.

did_multiplegt_dyn (R / Stata / Python)
---------------------------------------

The `did_multiplegt_dyn <https://github.com/Credible-Answers/did_multiplegt_dyn>`_
package (available in `R and Stata <https://github.com/Credible-Answers/did_multiplegt_dyn>`_
and `Python <https://github.com/Credible-Answers/py_did_multiplegt_dyn>`_) by
`Clément de Chaisemartin <https://www.sciencespo.fr/department-economics/directory/dechaisemartin-clement/>`_
and
`Xavier D'Haultfoeuille <https://faculty.crest.fr/xdhaultfoeuille/>`_ is the
basis for ModernDiD's intertemporal treatment effects estimator. The
`did_multiplegt() <api/generated/didinter/moderndid.did_multiplegt.html>`_
function implements the methodology in
`de Chaisemartin and D'Haultfoeuille (2024) <https://doi.org/10.1162/rest_a_01414>`_,
which estimates treatment effects in settings with potentially non-binary,
non-absorbing treatments.

You can learn more about ``did_multiplegt_dyn``
`on GitHub <https://github.com/Credible-Answers/did_multiplegt_dyn>`_ or by
reading the `associated paper <https://doi.org/10.1162/rest_a_01414>`_.

ModernDiD is validated against ``did_multiplegt_dyn`` via R scripts to ensure
numerical equivalence.

HonestDiD (R)
-------------

The `HonestDiD <https://github.com/asheshrambachan/HonestDiD>`_ R package by
`Ashesh Rambachan <https://asheshrambachan.github.io/>`_ and
`Jonathan Roth <https://jonathandroth.github.io/>`_ is the basis for
ModernDiD's sensitivity analysis tools. The
`honest_did() <api/generated/honestdid/moderndid.honest_did.html>`_ function
implements the methodology in
`Rambachan and Roth (2023) <https://doi.org/10.1093/restud/rdad018>`_, which
provides a more credible approach to evaluating the parallel trends assumption
by constructing robust confidence sets under violations of parallel trends.

You can learn more about ``HonestDiD``
`on GitHub <https://github.com/asheshrambachan/HonestDiD>`_ or by reading the
`associated paper <https://doi.org/10.1093/restud/rdad018>`_.

PyFixest (Python)
-----------------

`PyFixest <https://github.com/py-econometrics/pyfixest>`_ by
`Alexander Fischer <https://s3alfisc.github.io/blog/>`_ is a fast and user-friendly
fixed effects regression package for Python. ModernDiD uses ``pyfixest`` in its
test suite as a reference implementation for validating regression-based DiD
estimators. ``PyFixest``'s approach to building an ergonomic Python econometrics
library has also influenced ModernDiD's API design philosophy.

You can learn more about ``PyFixest``
`on GitHub <https://github.com/py-econometrics/pyfixest>`_ or via its
`documentation <https://py-econometrics.github.io/pyfixest/>`_.

Other software
==============

Here we list other foundational software without which a project like ModernDiD
would not be possible:

- `NumPy <https://numpy.org/>`_ -- Array computing
- `SciPy <https://scipy.org/>`_ -- Scientific computing
- `Polars <https://pola.rs/>`_ -- Fast DataFrames for internal data wrangling
- `Narwhals <https://narwhals-dev.github.io/narwhals/>`_ -- DataFrame-agnostic compatibility layer
- `PyArrow <https://arrow.apache.org/docs/python/>`_ -- Apache Arrow for Python
- `statsmodels <https://www.statsmodels.org/>`_ -- Statistical models
- `Numba <https://numba.pydata.org/>`_ -- JIT compilation for numerical code
- `CuPy <https://cupy.dev/>`_ -- GPU-accelerated array computing
- `Dask <https://www.dask.org/>`_ -- Distributed computing
- `PySpark <https://spark.apache.org/docs/latest/api/python/>`_ -- Distributed computing on Spark
- `plotnine <https://plotnine.org/>`_ -- Grammar of graphics plotting
- `CVXPY <https://www.cvxpy.org/>`_ -- Convex optimization
- `formulaic <https://matthewwardrop.github.io/formulaic/>`_ -- Formula parsing
- `PrettyTable <https://github.com/jazzband/prettytable>`_ -- Table formatting

Papers and algorithms
=====================

The following papers describe the core methodologies implemented in ModernDiD:

- Abadie, A. (2005). "Semiparametric Difference-in-Differences Estimators."
  *Review of Economic Studies*, 72(1), 1-19.
  `DOI:10.1111/0034-6527.00321 <https://doi.org/10.1111/0034-6527.00321>`_.
- Callaway, B., & Sant'Anna, P. H. C. (2021). "Difference-in-Differences with
  Multiple Time Periods." *Journal of Econometrics*, 225(2), 200-230.
  `DOI:10.1016/j.jeconom.2020.12.001 <https://doi.org/10.1016/j.jeconom.2020.12.001>`_.
- Callaway, B., Goodman-Bacon, A., & Sant'Anna, P. H. C. (2024).
  "Difference-in-Differences with a Continuous Treatment."
  `arXiv:2107.02637 <https://arxiv.org/abs/2107.02637>`_.
- Chen, X., Christensen, T. M., & Kankanala, S. (2024). "Adaptive Estimation
  and Uniform Confidence Bands for Nonparametric Structural Functions and
  Elasticities."
  `arXiv:2107.11869 <https://arxiv.org/abs/2107.11869>`_.
- de Chaisemartin, C., & D'Haultfoeuille, X. (2024).
  "Difference-in-Differences Estimators of Intertemporal Treatment Effects."
  *Review of Economics and Statistics*, 106(6), 1723-1736.
  `DOI:10.1162/rest_a_01414 <https://doi.org/10.1162/rest_a_01414>`_.
- Graham, B., Pinto, C., & Egel, D. (2012). "Inverse Probability Tilting for
  Moment Condition Models with Missing Data." *Review of Economic Studies*,
  79(3), 1053-1079.
  `DOI:10.1093/restud/rdr047 <https://doi.org/10.1093/restud/rdr047>`_.
- Ortiz-Villavicencio, M., & Sant'Anna, P. H. C. (2025). "Better Understanding
  Triple Differences Estimators."
  `arXiv:2505.09942 <https://arxiv.org/abs/2505.09942>`_.
- Rambachan, A., & Roth, J. (2023). "A More Credible Approach to Parallel
  Trends." *Review of Economic Studies*, 90(5), 2555-2591.
  `DOI:10.1093/restud/rdad018 <https://doi.org/10.1093/restud/rdad018>`_.
- Sant'Anna, P. H. C., & Zhao, J. (2020). "Doubly Robust
  Difference-in-Differences Estimators." *Journal of Econometrics*, 219(1),
  101-122.
  `DOI:10.1016/j.jeconom.2020.06.003 <https://doi.org/10.1016/j.jeconom.2020.06.003>`_.
