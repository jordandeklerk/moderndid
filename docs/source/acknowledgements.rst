.. _acknowledgements:

################
Acknowledgements
################

Like many open-source software projects, ModernDiD builds on work and ideas
first developed in other packages. We want to thank the authors of those
packages for their creativity and hard work.

Software
========

Unless explicitly stated otherwise, all ModernDiD code is written
independently from scratch. The packages listed below have influenced
ModernDiD's API design or algorithmic choices or serve as reference
implementations in its tests. No source code has been copied except where a
note says so and gives the license and permission details.

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

We borrowed the following API conventions and ideas directly from ``did``.

- The `att_gt() <api/generated/multiperiod/moderndid.att_gt.html>`_ function,
  its argument names (``yname``, ``tname``, ``idname``, ``gname``), its control
  group options (``nevertreated``, ``notyettreated``), and its handling of
  anticipation periods
- The `aggte() <api/generated/multiperiod/moderndid.aggte.html>`_ function and
  its aggregation types (``group``, ``dynamic``, ``calendar``, ``simple``)
- The doubly robust (``dr``), inverse probability weighting (``ipw``), and
  outcome regression (``reg``) estimation methods
- Analytical standard error formulas and the multiplier bootstrap for
  simultaneous confidence bands
- The visual conventions of ``did``'s ``ggdid()`` in
  `plot_gt() <api/generated/plotting/moderndid.plots.plot_gt.html>`_ and
  `plot_event_study() <api/generated/plotting/moderndid.plots.plot_event_study.html>`_

ModernDiD is benchmarked and validated against ``did`` through R scripts that
check its coefficients, standard errors, and confidence intervals for numerical
agreement.

DRDID (R)
---------

The `DRDID <https://github.com/pedrohcgs/DRDID>`__ R package by
`Pedro H.C. Sant'Anna <https://pedrohcgs.github.io/>`_ and
Jun Zhao is the foundation for ModernDiD's two-period doubly robust
estimators.
`Sant'Anna and Zhao (2020) <https://doi.org/10.1016/j.jeconom.2020.06.003>`_
develop the locally efficient doubly robust DiD estimators that
`drdid() <api/generated/drdid/moderndid.drdid.html>`__ computes under
``est_method="imp"`` or ``"trad"``.

ModernDiD implements every estimator in the ``DRDID`` package for both panel
data and repeated cross-sections. The repeated cross-section functions include
the locally efficient improved and traditional doubly robust estimators
(`drdid_imp_local_rc() <api/generated/drdid/moderndid.drdid_imp_local_rc.html>`_
and `drdid_rc() <api/generated/drdid/moderndid.drdid_rc.html>`_), their
counterparts that are not locally efficient
(`drdid_imp_rc() <api/generated/drdid/moderndid.drdid_imp_rc.html>`_ and
`drdid_trad_rc() <api/generated/drdid/moderndid.drdid_trad_rc.html>`_), the IPW
estimators (`ipw_did_rc() <api/generated/drdid/moderndid.ipw_did_rc.html>`_),
and the outcome regression estimators
(`reg_did_rc() <api/generated/drdid/moderndid.reg_did_rc.html>`_).

ModernDiD is benchmarked and validated against ``DRDID`` through R scripts
that check numerical agreement.

contdid (R)
-----------

The `contdid <https://github.com/bcallaway11/contdid>`_ R package by
`Brantly Callaway <https://bcallaway11.github.io/>`_,
`Andrew Goodman-Bacon <https://goodman-bacon.com/>`_, and
`Pedro H.C. Sant'Anna <https://pedrohcgs.github.io/>`_ is the basis for
ModernDiD's continuous treatment DiD estimator. The
`cont_did() <api/generated/didcont/moderndid.cont_did.html>`_ function
implements the methodology of
`Callaway, Goodman-Bacon, and Sant'Anna (2025) <https://psantanna.com/files/CGBS_v4.pdf>`_.
Their paper extends the DiD framework to settings where treatment intensity
varies continuously across units.

ModernDiD is benchmarked and validated against ``contdid`` through R scripts
that check numerical agreement.

npiv (R)
--------

The `npiv <https://github.com/JeffreyRacine/npiv>`_ R package by
`Jeffrey S. Racine <https://experts.mcmaster.ca/people/racinej>`_ and
`Timothy M. Christensen <https://tmchristensen.com/>`_ implements
nonparametric instrumental variables estimation and uniform confidence band
construction using sieve-based methods. ModernDiD's
`npiv() <api/generated/npiv/moderndid.npiv.html>`_ function implements the
methodology of `Chen and Christensen (2018) <https://doi.org/10.3982/QE722>`_ and
`Chen, Christensen, and Kankanala (2024) <https://arxiv.org/abs/2107.11869>`_.
The continuous DiD estimator
`cont_did() <api/generated/didcont/moderndid.cont_did.html>`_ uses it for its
data-driven dose-response estimates.

triplediff (R)
--------------

The `triplediff <https://github.com/marcelortizv/triplediff>`_ R package by
`Marcel Ortiz-Villavicencio <https://marcelortizv.github.io/>`_ and
`Pedro H.C. Sant'Anna <https://pedrohcgs.github.io/>`_ is the basis for
ModernDiD's triple difference-in-differences estimator. The
`ddd() <api/generated/didtriple/moderndid.ddd.html>`_ function implements the
methodology of
`Ortiz-Villavicencio and Sant'Anna (2025) <https://arxiv.org/abs/2505.09942>`_.
Their paper develops doubly robust triple DiD estimators for staggered adoption
designs.

ModernDiD is benchmarked and validated against ``triplediff`` through R scripts
that check numerical agreement.

DynBalancing (R)
----------------

The `DynBalancing <https://github.com/dviviano/DynBalancing>`_ R package by
`Davide Viviano <https://dviviano.github.io/>`_ and
`Jelena Bradic <https://www.jelenabradic.net/>`_ is the basis for
ModernDiD's dynamic covariate balancing DiD estimator. The
`dyn_balancing() <api/generated/diddynamic/moderndid.diddynamic.dyn_balancing.html>`_
function implements the methodology of
`Viviano and Bradic (2026) <https://doi.org/10.1093/biomet/asag016>`_.
Their paper estimates the effects of time-varying treatments in panel data
with sequential covariate balancing weights and potential local projections.

ModernDiD is benchmarked and validated against ``DynBalancing`` through R
scripts that check numerical agreement.

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
function implements the methodology of
`de Chaisemartin and D'Haultfoeuille (2024) <https://doi.org/10.1162/rest_a_01414>`_.
Their paper estimates treatment effects when treatments can be non-binary and
non-absorbing.

ModernDiD is validated against ``did_multiplegt_dyn`` through R scripts that
check numerical agreement.

etwfe (R)
---------

The `etwfe <https://github.com/grantmcdermott/etwfe>`_ R package by
`Grant McDermott <https://grantmcdermott.com/>`_ is the basis for
ModernDiD's extended two-way fixed effects estimator. The
`etwfe() <api/generated/etwfe/moderndid.etwfe.html>`_ and
`emfx() <api/generated/etwfe/moderndid.emfx.html>`_ functions implement the
methodology of
`Wooldridge (2025) <https://doi.org/10.1007/s00181-025-02807-z>`_. That paper
shows that a fully saturated TWFE regression with cohort-by-time interactions
recovers heterogeneous treatment effects without the negative weighting problem
of conventional TWFE.

We borrowed the following API conventions and ideas directly from ``etwfe``.

- The `etwfe() <api/generated/etwfe/moderndid.etwfe.html>`_ function with its
  ``cgroup`` control group options and its ``fe`` fixed effects specification
- The `emfx() <api/generated/etwfe/moderndid.emfx.html>`_ function with its
  aggregation types (``simple``, ``event``, ``group``, ``calendar``) and its
  ``post_only`` and ``window`` options
- Poisson, logit, and probit models through the ``family`` argument, as in
  `Wooldridge (2023) <https://doi.org/10.1093/ectj/utad016>`_
- Within-cohort demeaning of control variables through the Mundlak device
- The ``xvar`` argument for treatment effect heterogeneity by a covariate

ModernDiD is benchmarked and validated against ``etwfe`` through R scripts that
check the numerical agreement of coefficients, standard errors, and aggregated
treatment effects across all aggregation types, control group strategies,
fixed effects specifications, and nonlinear model families.

HonestDiD (R)
-------------

The `HonestDiD <https://github.com/asheshrambachan/HonestDiD>`_ R package by
`Ashesh Rambachan <https://asheshrambachan.github.io/>`_ and
`Jonathan Roth <https://jonathandroth.github.io/>`_ is the basis for
ModernDiD's sensitivity analysis tools. The
`honest_did() <api/generated/honestdid/moderndid.honest_did.html>`_ function
implements the methodology of
`Rambachan and Roth (2023) <https://doi.org/10.1093/restud/rdad018>`_. Their
paper constructs confidence sets that stay valid under bounded violations of
parallel trends.

PyFixest (Python)
-----------------

`PyFixest <https://github.com/py-econometrics/pyfixest>`_ by
`Alexander Fischer <https://s3alfisc.github.io/blog/>`_ is a fast and user-friendly
fixed effects regression package for Python. ModernDiD uses ``pyfixest`` as the
regression backend for the ETWFE estimator
(`etwfe() <api/generated/etwfe/moderndid.etwfe.html>`_), where it handles the
saturated interaction regression with high-dimensional fixed effects absorption.
``PyFixest``'s approach to building an ergonomic Python econometrics library has
also influenced ModernDiD's API design philosophy. You can learn more about how
``PyFixest`` works in its
`documentation <https://py-econometrics.github.io/pyfixest/>`_.

Other software
==============

A project like ModernDiD would not be possible without the following
foundational software.

- `NumPy <https://numpy.org/>`_ for array computing
- `SciPy <https://scipy.org/>`_ for scientific computing
- `Polars <https://pola.rs/>`_ for fast DataFrames in internal data handling
- `Narwhals <https://narwhals-dev.github.io/narwhals/>`_ for compatibility
  across DataFrame libraries
- `PyArrow <https://arrow.apache.org/docs/python/>`_ for Apache Arrow in Python
- `statsmodels <https://www.statsmodels.org/>`_ for statistical models
- `scikit-learn <https://scikit-learn.org/>`_ for machine learning, such as
  LASSO and ridge regression
- `CuPy <https://cupy.dev/>`_ for GPU array computing
- `plotnine <https://plotnine.org/>`_ for grammar of graphics plotting
- `CVXPY <https://www.cvxpy.org/>`_ for convex optimization
- `HiGHS <https://highs.dev/>`_ for linear programming
- `formulaic <https://matthewwardrop.github.io/formulaic/>`_ for formula parsing
- `PrettyTable <https://github.com/jazzband/prettytable>`_ for table formatting

Papers and algorithms
=====================

The papers below describe the core methodologies that ModernDiD implements.

- Abadie, A. (2005). "Semiparametric Difference-in-Differences Estimators."
  *Review of Economic Studies*, 72(1), 1-19.
  `DOI:10.1111/0034-6527.00321 <https://doi.org/10.1111/0034-6527.00321>`_.
- Callaway, B., & Sant'Anna, P. H. C. (2021). "Difference-in-Differences with
  Multiple Time Periods." *Journal of Econometrics*, 225(2), 200-230.
  `DOI:10.1016/j.jeconom.2020.12.001 <https://doi.org/10.1016/j.jeconom.2020.12.001>`_.
- Callaway, B., Goodman-Bacon, A., & Sant'Anna, P. H. C. (2025).
  "Difference-in-Differences with a Continuous Treatment."
  *American Economic Review*, forthcoming.
  `December 31, 2025 manuscript <https://psantanna.com/files/CGBS_v4.pdf>`_.
- Chen, X., & Christensen, T. M. (2018). "Optimal Sup-norm Rates and Uniform
  Inference on Nonlinear Functionals of Nonparametric IV." *Quantitative
  Economics*, 9(1), 39-84.
  `DOI:10.3982/QE722 <https://doi.org/10.3982/QE722>`_.
- Chen, X., Christensen, T. M., & Kankanala, S. (2024). "Adaptive Estimation
  and Uniform Confidence Bands for Nonparametric Structural Functions and
  Elasticities." *Review of Economic Studies*, 92(1), 162-196.
  `DOI:10.1093/restud/rdae025 <https://doi.org/10.1093/restud/rdae025>`_.
- de Chaisemartin, C., & D'Haultfoeuille, X. (2024).
  "Difference-in-Differences Estimators of Intertemporal Treatment Effects."
  *Review of Economics and Statistics*, 106(6), 1723-1736.
  `DOI:10.1162/rest_a_01414 <https://doi.org/10.1162/rest_a_01414>`_.
- Graham, B., Pinto, C., & Egel, D. (2012). "Inverse Probability Tilting for
  Moment Condition Models with Missing Data." *Review of Economic Studies*,
  79(3), 1053-1079.
  `DOI:10.1093/restud/rdr047 <https://doi.org/10.1093/restud/rdr047>`_.
- Hatamyar, J., Kreif, N., Rocha, R., & Huber, M. (2023). "Machine Learning for
  Staggered Difference-in-Differences and Dynamic Treatment Effect
  Heterogeneity." `arXiv:2310.11962 <https://arxiv.org/abs/2310.11962>`_.
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
- Viviano, D., & Bradic, J. (2026). "Dynamic Covariate Balancing: Estimating
  Treatment Effects over Time with Potential Local Projections." *Biometrika*,
  asag016.
  `DOI:10.1093/biomet/asag016 <https://doi.org/10.1093/biomet/asag016>`_.
- Wooldridge, J. M. (2023). "Simple Approaches to Nonlinear
  Difference-in-Differences with Panel Data." *The Econometrics Journal*,
  26(3), C31-C66.
  `DOI:10.1093/ectj/utad016 <https://doi.org/10.1093/ectj/utad016>`_.
- Wooldridge, J. M. (2025). "Two-Way Fixed Effects, the Two-Way Mundlak
  Regression, and Difference-in-Differences Estimators." *Empirical Economics*,
  69(5), 2545-2587.
  `DOI:10.1007/s00181-025-02807-z <https://doi.org/10.1007/s00181-025-02807-z>`_.
