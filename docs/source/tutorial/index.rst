========
Tutorial
========

didpy is a collection of difference-in-differences (DiD) estimators and
convenience functions built on NumPy. It contains a number of
open-source Python packages for causal inference with DiD methods.

Before reading this tutorial, we recommend that you read
:doc:`../installing` (if you haven't done so already).

.. table::
   :class: contentstable

   +--------------------+-----------------------------------------------------+
   | Subpackage         | Description                                         |
   +====================+=====================================================+
   | ``drdid``          | Doubly robust DiD estimators                       |
   | ``ipw``            | Inverse propensity weighted DiD estimators         |
   | ``reg``            | Regression-based DiD estimators                     |
   | ``twfe``           | Two-way fixed effects DiD estimators               |
   | ``bootstrap``      | Bootstrap inference methods                         |
   | ``propensity``     | Propensity score estimation utilities              |
   +--------------------+-----------------------------------------------------+

Additional functionality is provided by the ``utils`` module for
panel data processing and validation.

To import didpy modules, we recommend using the pattern:

.. code-block:: python

   import didpy

Then functions can be accessed as ``didpy.drdid_imp_panel``,
``didpy.ipw_did_panel``, etc.

.. toctree::
   :hidden:

   getting_started
   basic_usage
   advanced_usage
