.. module:: moderndid

.. _reference:

####################
moderndid reference
####################

:Release: |version|
:Date: |today|

This reference manual details functions, modules, and objects
included in moderndid, describing what they are and what they do.

.. _did-module:

DiD with Multiple Time Periods
==============================

The ``moderndid.did`` module implements difference-in-differences estimators for
multiple time periods and staggered treatment adoption following
`Callaway and Sant'Anna (2020) <https://psantanna.com/files/Callaway_SantAnna_2020.pdf>`_.
This module provides tools for estimating group-time average treatment effects and
aggregating them into summary treatment effect parameters.

.. toctree::
   :maxdepth: 3

   did

.. _drdid-module:

Doubly Robust DiD
=================

The ``moderndid.drdid`` module contains a comprehensive suite of doubly robust
difference-in-differences estimators based on `Sant'Anna and Zhao (2020)
<https://psantanna.com/files/SantAnna_Zhao_DRDID.pdf>`_. These estimators combine
outcome regression and propensity score methods to provide improved local efficiency
and robustness to model mis-specification for both panel data and repeated cross-sections.

.. toctree::
   :maxdepth: 3

   drdid

.. _didhonest-module:

Sensitivity Analysis and Honest DiD
====================================

The ``moderndid.didhonest`` module implements sensitivity analysis for difference-in-differences
following `Rambachan and Roth (2023) <https://asheshrambachan.github.io/assets/files/hpt-draft.pdf>`_.
It provides tools for conducting inference that is robust to violations of parallel trends
assumptions, including bounds under various restrictions on post-treatment bias.

.. toctree::
   :maxdepth: 3

   didhonest

.. _bootstrap-module:

Bootstrap Methods
=================

The ``moderndid.bootstrap`` module provides comprehensive bootstrap inference methods
for all DiD estimators. These include weighted bootstrap for panel and repeated
cross-section data, as well as multiplier bootstrap methods and semiparametric
influence functions.

.. toctree::
   :maxdepth: 3

   bootstrap

.. _propensity-module:

Propensity Score Methods
========================

The ``moderndid.propensity`` module implements propensity score estimation and
weighting methods used across various DiD estimators. This includes IPW (inverse
probability weighting), IPT (inverse probability tilting), and AIPW (augmented IPW) methods.

.. toctree::
   :maxdepth: 3

   propensity

----

.. warning::
   The following modules are currently under development and not yet available for use.
   They are included here to show the planned scope of the moderndid package.

.. _didcont-module:

Continuous Treatment DiD
========================

The ``moderndid.didcont`` module extends difference-in-differences methods to
settings with continuous treatment variables following `Callaway et al. (2024)
<https://arxiv.org/pdf/2107.02637>`_, allowing for dose-response analysis
in causal inference with panel data.

.. toctree::
   :maxdepth: 3

   didcont

.. _didinter-module:

Intertemporal DiD
=================

The ``moderndid.didinter`` module provides methods for difference-in-differences
analysis following `Chaisemartin & D'Haultfœuille (2024) <https://arxiv.org/pdf/2007.04267>`_,
focusing on intertemporal effects where treatment may be non-binary and non-absorbing.

.. toctree::
   :maxdepth: 3

   didinter

.. _didml-module:

Machine Learning DiD
===================

The ``moderndid.didml`` module integrates machine learning methods with
difference-in-differences estimation following `Hatamyar et al. (2023)
<https://arxiv.org/pdf/2310.11962>`_, providing tools for flexible model
specification and high-dimensional covariate adjustment.

.. toctree::
   :maxdepth: 3

   didml

.. _drdidweak-module:

Weak Doubly Robust DiD
======================

The ``moderndid.drdidweak`` module implements doubly robust estimators following
`Ma et al. (2023) <https://arxiv.org/pdf/2304.08974>`_ that maintain good properties
even under weak overlap conditions or limited common support between treatment and control groups.

.. toctree::
   :maxdepth: 3

   drdidweak

.. _didcomp-module:

Compositional DiD
=================

The ``moderndid.didcomp`` module addresses DiD setups with repeated cross-sectional
data and potential compositional changes across time periods following
`Sant'Anna & Xu (2025) <https://arxiv.org/pdf/2304.13925>`_.

.. toctree::
   :maxdepth: 3

   didcomp

.. _didlocal-module:

Local DiD Estimators
====================

The ``moderndid.didlocal`` module implements local projections DiD following
`Dube et al. (2025) <https://www.nber.org/system/files/working_papers/w31184/w31184.pdf>`_
to address possible biases arising from negative weighting.

.. toctree::
   :maxdepth: 3

   didlocal

.. _did2s-module:

Two-Stage DiD
=============

The ``moderndid.did2s`` module implements two-stage difference-in-differences estimators
following `Gardner (2021) <https://jrgcmu.github.io/2sdd_current.pdf>`_. This approach
addresses issues with two-way fixed effects regressions in settings with staggered
treatment timing.

.. toctree::
   :maxdepth: 3

   did2s

.. _didbacon-module:

Bacon Decomposition
===================

The ``moderndid.didbacon`` module implements the Bacon decomposition following
`Goodman-Bacon (2019) <https://cdn.vanderbilt.edu/vu-my/wp-content/uploads/sites/2318/2019/07/29170757/ddtiming_7_29_2019.pdf>`_.
This tool decomposes two-way fixed effects DiD estimates into a weighted average of all possible 2x2
DiD comparisons.

.. toctree::
   :maxdepth: 3

   didbacon

.. _functional-module:

Functional Form Tests
=====================

The ``moderndid.functional`` module implements specification tests for
difference-in-differences models following `Roth & Sant'Anna (2023)
<https://arxiv.org/pdf/2010.04814>`_, including tests for functional form
assumptions and pre-treatment trends.

.. toctree::
   :maxdepth: 3

   functional

.. _utilities-module:

Utilities
=========

The ``moderndid.utils`` module provides utility functions for panel data
manipulation, formula parsing, and common operations used across the package.

.. toctree::
   :maxdepth: 3

   utilities

Acknowledgements
================

The moderndid package implements various difference-in-differences methodologies from
the econometric literature. We acknowledge the original authors of these methods,
and the authors of the R packages that inspired this package.
