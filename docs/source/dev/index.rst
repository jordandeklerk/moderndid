.. _development:

###########
Development
###########

Everything you need to contribute to **ModernDiD** and understand its internals.
For usage documentation see the :ref:`User Guide <user-guide>`.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Guide
     - Description
   * - :doc:`contributing`
     - Set up a development environment and make your first contribution.
   * - :doc:`architecture`
     - Understand the library's estimator, data, and result interfaces.
   * - :doc:`new_estimator`
     - Add an estimator that fits the shared API.
   * - :doc:`testing`
     - Write tests and validate implementations against reference packages.
   * - :doc:`workflow` and :doc:`reviewing`
     - Prepare changes and review pull requests.
   * - :doc:`debugging` and :doc:`benchmarking`
     - Diagnose issues and measure performance.
   * - :doc:`distributed_architecture`
     - Work with the Dask and Spark backends.
   * - :doc:`releasing`
     - Prepare and publish a library release.

.. toctree::
   :maxdepth: 2
   :hidden:

   contributing
   architecture
   new_estimator
   testing
   workflow
   reviewing
   debugging
   benchmarking
   distributed_architecture
   releasing

.. raw:: html

   <p class="mdid-footer-logo"><img src="../_static/logo-wordmark.svg" alt="ModernDiD logo"></p>
