=============================
pyDiD documentation
=============================

.. only:: not release

   .. warning::

      This documentation is for the latest development version of pyDiD.
      There is no stable release yet.

**Date**: |today| **Version**: |version|

**Useful links**:
`Installation <installing.html>`__ |
`Source Repository <https://github.com/jordandeklerk/pyDiD>`__ |
`Issue Tracker <https://github.com/jordandeklerk/pyDiD/issues>`__ |

**pyDiD** is a Python package implementing modern DiD estimators for panel and repeated cross-section data,
including staggered treatment timing, multiple time periods, doubly robust methods, continuous treatments,
two-stage DiD, local projection DiD, machine learning approaches, and diagnostic tools like the Bacon-Goodman
decomposition and sensitivity tests for functional form.

.. grid:: 2
    :gutter: 4

    .. grid-item-card::
        :text-align: center

        .. image:: _static/user_guide.svg
            :height: 100px

        User Guide
        ^^^^^^^^^^

        The user guide provides in-depth information on the key concepts of pyDiD
        with useful background information and explanation.

        +++

        .. button-ref:: user_guide
            :expand:
            :color: secondary
            :click-parent:

            To the user guide

    .. grid-item-card::
        :text-align: center

        .. image:: _static/api_reference.svg
            :height: 100px

        API Reference
        ^^^^^^^^^^^^^

        The reference guide contains a detailed description of the functions,
        modules, and objects included in pyDiD. The reference describes how the
        methods work and which parameters can be used.

        +++

        .. button-ref:: api/index
            :expand:
            :color: secondary
            :click-parent:

            To the reference guide

    .. grid-item-card::
        :text-align: center

        .. image:: _static/tutorial.svg
            :height: 100px

        Tutorial
        ^^^^^^^^

        Step-by-step tutorials and examples to get started with pyDiD,
        including practical applications and best practices.

        +++

        .. button-ref:: tutorial/index
            :expand:
            :color: secondary
            :click-parent:

            To the tutorial

    .. grid-item-card::
        :text-align: center

        .. image:: _static/developer.svg
            :height: 100px

        Developer Guide
        ^^^^^^^^^^^^^^^

        Want to contribute to pyDiD? The contributing guidelines will guide
        you through the process of improving pyDiD.

        +++

        .. button-ref:: dev/index
            :expand:
            :color: secondary
            :click-parent:

            To the development guide

.. toctree::
   :maxdepth: 1
   :hidden:

   installing
   user_guide
   api/index
   tutorial/index
   dev/index
   release/index
