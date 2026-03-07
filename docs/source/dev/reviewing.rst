.. _reviewing:

======================================
Reviewing and Maintainer Guidelines
======================================

Reviewing open pull requests helps move the project forward. We encourage
people outside the project to get involved as well; it's a great way to get
familiar with the codebase.

For reviewers
=============

Anyone can review a pull request. You don't need to be a maintainer or an
expert in every part of the codebase. Reviews of documentation, CI configuration,
and general code quality are always welcome. Reviews of estimator implementations,
however, should come from contributors with experience in econometrics or
statistics, since correctness depends on understanding the underlying methodology.

Communication
-------------

Review is a collaborative process, not an adversarial one. The goal is to
improve the code together.

- Every PR, good or bad, is an act of generosity. Opening with a positive
  comment helps the author feel rewarded, and your subsequent remarks will
  be heard more clearly.
- Be specific. Instead of "this is confusing," explain what is confusing
  and suggest a concrete alternative.
- Distinguish requirements from suggestions. Mark non-blocking style
  preferences as "nit:" or "suggestion:" so the author knows what must be
  addressed and what is optional.
- If a contributor solved a tricky problem well or wrote clean code, say so.
  Positive feedback matters.
- Ask questions instead of making demands. "Could you explain why this uses
  a list comprehension instead of a generator?" invites discussion. "Change
  this to a generator" assumes you know better without context.
- Try to respond within a few business days. If you can't review promptly,
  leave a comment letting the author know when you'll get to it.

What to look for
-----------------

When reviewing a pull request, consider the following areas. Not every item
applies to every PR, so use judgment about what matters most for the change
at hand.

**Correctness**

- Does the code do what it claims to do?
- Are edge cases handled? Consider empty inputs, single observations,
  missing data, and boundary conditions common in panel data
  (single-period groups, all-treated cohorts, unbalanced panels).
- For numerical code, are there potential overflow, underflow, or
  division-by-zero issues? Are tolerances appropriate?

**Design and architecture**

- Does the change follow the patterns described in :doc:`architecture`?
  New estimators should use the preprocessing pipeline, return immutable
  ``NamedTuple`` results, and include influence functions.
- Is the public API consistent with existing estimators? Check parameter
  names against the :ref:`consistent argument naming <consistent-argument-naming>`
  conventions.
- Are new dependencies justified? **ModernDiD** keeps optional dependencies
  truly optional, and core functionality should work with only the base
  dependencies.

**Tests**

- Are there tests for the new or changed behavior? See
  :ref:`how to write tests <testing-how-to-write>` for conventions.
- Do tests cover both the happy path and meaningful edge cases?
- Are numerical tolerances appropriate for the type of computation?
  (See :ref:`numerical tolerances <testing-numerical-tolerances>`.)
- For new estimators, is there a validation test against the corresponding
  R package in `validation <https://github.com/jordandeklerk/moderndid/tree/main/tests/validation>`__?

**Performance**

- Could the change introduce performance regressions on large datasets?
- If the code adds loops over observations, should it use Numba or
  vectorized operations instead?
- For distributed code, does the change maintain the "never materialize
  full data on the driver" principle?

**Documentation**

- Do public functions have docstrings following the NumPy docstring
  standard?
- If the change affects user-facing behavior, is the user guide updated?
- Are commit messages and the PR description clear about what changed
  and why?

**Style**

- Does the code pass ``pixi run lint`` without new warnings?
- Are variable names descriptive? Avoid single-letter names except for
  conventional loop variables (``i``, ``j``) and well-known mathematical
  notation (``X``, ``y``, ``n``).

For maintainers
===============

Maintainers have merge access and carry additional responsibilities beyond
reviewing code.

Merge criteria
--------------

Before merging a pull request, verify the following.

1. CI must pass. All test jobs (core, Dask, Spark) must be green. Do not
   merge with failing checks unless there is a known flaky test that is
   unrelated to the PR, and document this in a comment.

2. At least one approving review. Every PR needs at least one review from
   someone who did not author the change.

3. No unresolved conversations. All review threads should be resolved
   before merging. If a suggestion was declined, the author should explain
   why.

4. Scope is appropriate. Large PRs that mix multiple concerns should be
   split if possible. A PR that adds a new estimator should not also
   refactor the plotting system.

Merge strategy
--------------

Use squash merge for most PRs. This keeps ``main`` history clean with
one commit per logical change. Ensure the squashed commit message follows
the :doc:`commit conventions <workflow>`.

Use a regular merge for large feature branches where the individual commits
tell a meaningful story (e.g., a multi-step estimator implementation
where each commit adds a distinct piece).

Handling stale PRs
------------------

If a PR has had no activity for 30 days, leave a friendly comment asking if
the author plans to continue. If there is no response after another 14 days,
close the PR with a comment explaining that it can be reopened when the
author is ready.

If the work is valuable and the author is unresponsive, it's acceptable
to open a new PR based on their branch, crediting the original author
in the commit message.

Backporting
-----------

**ModernDiD** does not currently maintain multiple release branches. All
development targets ``main``, and releases are cut from ``main`` via tags.
If a multi-branch strategy becomes necessary in the future, this section
will be updated with backporting procedures.

Triaging issues
---------------

When new issues come in, start by trying to reproduce the bug. Ask for a
minimal reproducible example if one is not provided. Use labels to categorize
issues (bug, enhancement, documentation, etc.). If an issue is well-scoped
and doesn't require deep knowledge of the codebase, label it as a good first
issue to help onboard new contributors.

Standard responses
------------------

Maintaining a set of canned responses saves time and keeps communication
consistent. Here are templates for common situations that you can adapt as
needed.

**Requesting a minimal example** (when a bug report lacks enough detail to
reproduce)::

   Thanks for the report. Could you provide a minimal reproducible example?
   Ideally this would include the imports, a small dataset (or a call to
   one of our data generators like `gen_did_scalable()`), and the exact
   function call that triggers the issue. That will help us diagnose it
   quickly.

**Redirecting a usage question** (when an issue is really a support
request)::

   This looks like a usage question rather than a bug. The issue tracker
   is best reserved for bugs and feature requests. You might find the
   answer in the user guide: https://moderndid.readthedocs.io/

**Acknowledging a good first contribution** (when a first-time contributor
opens a PR)::

   Welcome and thanks for your first PR! I'll review this in the next
   few days. In the meantime, please make sure CI passes. You can check
   the status at the bottom of this PR.

**Requesting that unrelated changes be split** (when a PR mixes concerns)::

   Thanks for working on this. The PR currently includes both the bug fix
   and some unrelated refactoring. Could you split these into separate PRs?
   That makes each one easier to review and keeps the git history clear.

**Closing a stale PR**::

   It looks like this PR has been inactive for a while. I'm going to close
   it for now, but feel free to reopen when you're ready to continue. If
   someone else wants to pick up the work, the branch is still available.
