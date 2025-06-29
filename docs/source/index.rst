.. _pydid_docs_mainpage:

###################
pyDiD documentation
###################

.. toctree::
   :maxdepth: 1
   :hidden:

   User Guide <user_guide>
   API reference <api/index>
   Tutorial <tutorial/index>
   Development <dev/index>
   release/index


**Version**: |version|

**Download documentation**:
`Historical versions of documentation <https://github.com/jordandeklerk/pyDiD/releases>`_

**Useful links**:
`Installation <installing.html>`_ |
`Source Repository <https://github.com/jordandeklerk/pyDiD>`_ |
`Issue Tracker <https://github.com/jordandeklerk/pyDiD/issues>`_ |

.. only:: not release

   .. warning::

      This documentation is for the latest development version of pyDiD.
      There is no stable release yet.

pyDiD is a comprehensive Python package for modern difference-in-differences (DiD)
estimation. It provides a unified framework for causal inference using DiD methods
with panel and repeated cross-section data, including advanced estimators for
staggered treatment timing, multiple time periods, doubly robust methods, continuous
treatments, two-stage DiD, local projection DiD, machine learning approaches, and
diagnostic tools like the Bacon-Goodman decomposition and sensitivity tests.



.. grid:: 1 1 2 2
    :gutter: 2 3 4 4

    .. grid-item-card::
        :img-top: _static/user_guide.svg
        :text-align: center

        User guide
        ^^^

        The user guide provides in-depth information on the
        key concepts of pyDiD with useful background information and explanation.

        +++

        .. button-ref:: user_guide
            :expand:
            :color: secondary
            :click-parent:

            To the user guide

    .. grid-item-card::
        :img-top: _static/api_reference.svg
        :text-align: center

        API reference
        ^^^

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
        :img-top: _static/tutorial.svg
        :text-align: center

        Tutorial
        ^^^

        The tutorial provides step-by-step guides and examples to get started with DiD estimation,
        including practical applications and best practices.

        +++

        .. button-ref:: tutorial/index
            :expand:
            :color: secondary
            :click-parent:

            To the tutorial

    .. grid-item-card::
        :img-top: _static/developer.svg
        :text-align: center

        Contributor's guide
        ^^^

        Want to add to the codebase? Can help add functionality or improve the
        documentation? The contributing guidelines will guide you through the
        process of improving pyDiD.

        +++

        .. button-ref:: dev/index
            :expand:
            :color: secondary
            :click-parent:

            To the contributor's guide
