.. _development:

###########
Development
###########

To make sense of a change to an estimator, you need to know what happens between
the data a user supplies and the result they read. We'll follow that
calculation through the package so you can see where a correction or a new
method belongs. The architecture guide comes first because it explains
which parts you can reuse and where estimators need their own approach.

If you're preparing your first contribution, the
:ref:`Contributing guides <contributing-index>` cover the environment,
tests, and review process. You can return here to trace a calculation,
extend the package, or investigate a result that needs closer attention.

.. list-table::
   :class: section-index-table
   :header-rows: 1
   :widths: 35 65

   * - Guide
     - Description
   * - :doc:`architecture`
     - Follow data preparation, estimation, inference, and result presentation.
   * - :doc:`new_estimator`
     - Develop an estimator from its target through computation and inference.
   * - :doc:`debugging`
     - Locate a difference in the data, fitted effects, or uncertainty.
   * - :doc:`benchmarking`
     - Measure a defined workload and compare performance across revisions.

.. toctree::
   :maxdepth: 2
   :hidden:

   architecture
   new_estimator
   debugging
   benchmarking

.. raw:: html

   <p class="mdid-footer-logo"><img src="../_static/logo-wordmark.svg" alt="ModernDiD logo"></p>
