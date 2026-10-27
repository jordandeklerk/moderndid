.. _development:

###########
Development
###########

If you've found a confusing explanation, a result that needs checking, or an
estimator you'd like to add, the Contributing guides help you turn that idea
into a change we can review. We'll begin with a working checkout and follow the
change through testing and a pull request. The review and release guides explain
what happens once a contribution is ready to join the package.

You can work on a guide or investigate a particular estimator without learning
the whole library first. When you need to follow an estimator's implementation
or extend it, the guides on how the package works trace the calculation from the
data a user supplies to the result they read. The architecture guide comes first
because it explains which parts you can reuse and where estimators need their own
approach.

.. _contributing-index:

Contributing
============

.. list-table::
   :class: section-index-table
   :header-rows: 1
   :widths: 35 65

   * - Guide
     - Description
   * - :doc:`../contributing/guide`
     - Set up a checkout, run focused checks, and preview documentation.
   * - :doc:`../contributing/workflow`
     - Keep a change easy to review as you work on a branch and open a pull request.
   * - :doc:`../contributing/testing`
     - Choose tests that check behavior, numerical results, and meaningful edge cases.
   * - :doc:`../contributing/reviewing`
     - Review the method, implementation, and evidence behind a change.
   * - :doc:`../contributing/releasing`
     - Check package metadata and artifacts before tagging and publishing a release.

.. toctree::
   :hidden:
   :maxdepth: 1

   ../contributing/guide
   ../contributing/workflow
   ../contributing/testing
   ../contributing/reviewing
   ../contributing/releasing

How the package works
=====================

.. list-table::
   :class: section-index-table
   :header-rows: 1
   :widths: 35 65

   * - Guide
     - Description
   * - :doc:`architecture`
     - Follow data preparation, estimation, inference, and result presentation.
   * - :doc:`shards`
     - Rebuild a fit from sums over blocks of units, as a fit on Spark does.
   * - :doc:`row_shards`
     - Fit repeated cross-sections and unbalanced panels from shards of rows.
   * - :doc:`spark_fits`
     - Follow a fit on Spark from the table it builds to the tasks of each pass.
   * - :doc:`new_estimator`
     - Develop an estimator from its target through computation and inference.
   * - :doc:`debugging`
     - Locate a difference in the data, fitted effects, or uncertainty.
   * - :doc:`benchmarking`
     - Measure a defined workload and compare performance across revisions.

.. toctree::
   :hidden:
   :maxdepth: 1

   architecture
   shards
   row_shards
   spark_fits
   new_estimator
   debugging
   benchmarking

.. raw:: html

   <p class="mdid-footer-logo"><img src="../_static/logo-wordmark.svg" alt="ModernDiD logo"></p>
