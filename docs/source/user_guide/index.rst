.. _user-guide:
.. _overview:

==========
User Guide
==========

ModernDiD helps you estimate treatment effects using difference-in-differences
and related methods. A useful analysis starts by deciding which units can
supply an untreated comparison and which effect answers your research
question. This guide takes you through those decisions, from preparing the
data and fitting an estimator to explaining your results in figures and
publication tables.

If you're using the package for the first time, :doc:`getting_started` takes
you from installing it to understanding the untreated comparison and running
your first analysis on county employment data. From there, :doc:`fundamentals`
helps you adapt the analysis to your study by explaining what the estimator
needs from your data and how to interpret the effects it returns.

.. toctree::
   :hidden:
   :maxdepth: 2

   getting_started
   fundamentals
   scaling

.. _whatis-design:

Find the guide you need
-----------------------

We've grouped these pages by where you are in an analysis to help you read
them in order or return to the decision you're working on.

.. list-table::
   :class: section-index-table
   :header-rows: 1
   :widths: 30 70

   * - Section
     - What you'll work through
   * - :doc:`getting_started`
     - Install the package, understand a DiD comparison, and run your first analysis.
   * - :doc:`fundamentals`
     - Choose an estimator, prepare data, read results, and produce figures and tables.
   * - :doc:`scaling`
     - Choose CPU parallelism or GPU acceleration for the work your estimator does.

.. _whatis-methods:

Follow an analysis or a derivation
----------------------------------

The :doc:`Examples <../examples/index>` follow research questions through
complete analyses, including the specification choices and checks behind
their conclusions. After working through the guide's building blocks, you
can use those applications to see how the choices fit together for a
particular treatment design and dataset.

The :doc:`Background <../background/index>` pages develop the assumptions and
theoretical results behind each estimator. The :ref:`API Reference <api>`
documents each estimator's arguments and result fields for checking the details
of a call once you've chosen the design.


.. raw:: html

   <p class="mdid-footer-logo"><img src="../_static/logo-wordmark.svg" alt="ModernDiD logo"></p>
