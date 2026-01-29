.. currentmodule:: moderndid

=====================
ModernDiD Unreleased
=====================

:Date: |today|

This section documents changes that have been merged since the last release
and will be included in the next version.

Highlights
==========

- **Bug fixes**: Fixed group aggregation bug and issues in ``cont_did()`` and
  ``honestdid`` modules
- **R validation tests**: Added comprehensive validation tests against R packages
  to ensure numerical accuracy
- **Enhanced documentation**: Major documentation overhaul including new quickstart
  guide and improved API documentation
- **Better error handling**: Added propensity score trimming and collinearity checks
  with informative error messages

Bug Fixes
=========

- Fix group aggregation bug in ``aggte()`` function
- Fix bugs in ``cont_did()`` continuous treatment estimator
- Fix bugs in ``honestdid`` sensitivity analysis module
- Fix base period parameter handling

Improvements
============

Testing
-------

- Add comprehensive R validation tests for ``att_gt()`` and ``aggte()``
- Add R validation tests for DR-DiD estimators
- Add R validation tests for HonestDiD sensitivity analysis
- Simplify base period parameter and remove unused tests
- Improve test organization and coverage

Documentation
-------------

- Add new quickstart guide with practical examples
- Revise getting started overview
- Enhance main function docstrings and result output
- Add repeated cross-section examples for ``ddd()``
- Improve API documentation for HonestDiD module
- Add plotting code examples to documentation
- Extend development docs to include project architecture

Error Handling
--------------

- Add propensity score trimming functionality
- Add collinearity checks for covariates
- Improve error messages with more informative diagnostics
- Add ``random_state`` parameter for reproducibility in bootstrap

Refactoring
-----------

- Update random number generation for consistency
- Update parameter names in documentation
- Remove unused functions from ``cont_did`` module
- Remove redundant DGP code


Contributors
============

A total of 1 person contributed to these changes.

* Jordan Deklerk


Pull Requests Merged
====================

A total of 20 pull requests have been merged since v0.0.3, of which 19 are
feature or fix PRs (excluding automated dependency updates).

* `#138 <https://github.com/jordandeklerk/moderndid/pull/138>`__: Add implementation standards and intro to DiD section to docs
* `#137 <https://github.com/jordandeklerk/moderndid/pull/137>`__: Reorganize development docs
* `#136 <https://github.com/jordandeklerk/moderndid/pull/136>`__: Add more detail to the development docs
* `#135 <https://github.com/jordandeklerk/moderndid/pull/135>`__: Fix computations in ``didhonest`` and add more R validation
* `#134 <https://github.com/jordandeklerk/moderndid/pull/134>`__: Fix group aggregation in ``aggte()`` and add more R validation
* `#133 <https://github.com/jordandeklerk/moderndid/pull/133>`__: Add more R validation for ``drdid`` module
* `#132 <https://github.com/jordandeklerk/moderndid/pull/132>`__: Fix bugs in ``cont_did()`` and add more R validation
* `#130 <https://github.com/jordandeklerk/moderndid/pull/130>`__: Update getting started docs and test suite
* `#129 <https://github.com/jordandeklerk/moderndid/pull/129>`__: Add repeated cross-section examples for ``ddd()``
* `#128 <https://github.com/jordandeklerk/moderndid/pull/128>`__: Improve test coverage
* `#127 <https://github.com/jordandeklerk/moderndid/pull/127>`__: Update parameter names for ``drdid()``
* `#126 <https://github.com/jordandeklerk/moderndid/pull/126>`__: Add random state for ``cont_did()``
* `#125 <https://github.com/jordandeklerk/moderndid/pull/125>`__: Add propensity trimming and collinearity checks for DDD
* `#124 <https://github.com/jordandeklerk/moderndid/pull/124>`__: Clean up docs for ``HonestDiD``
* `#123 <https://github.com/jordandeklerk/moderndid/pull/123>`__: Add DDD plotting functions
* `#122 <https://github.com/jordandeklerk/moderndid/pull/122>`__: Add new DDD tests for coverage
* `#121 <https://github.com/jordandeklerk/moderndid/pull/121>`__: Rework main function docstrings
* `#120 <https://github.com/jordandeklerk/moderndid/pull/120>`__: Update docs for ``agg_ddd()`` function
* `#119 <https://github.com/jordandeklerk/moderndid/pull/119>`__: Update changelog for v0.0.3
