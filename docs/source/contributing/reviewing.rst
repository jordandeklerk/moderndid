.. _reviewing:

########################
Reviewing a contribution
########################

As a reviewer, you're trying to understand what changes for the
user and whether the method and code support that behavior. You can help with
part of that work even if you don't know every estimator; reading an
explanation as a new user or reproducing an input error often reveals something
the author has missed. We'll follow the change through its calculation, tests,
and explanation so the discussion has evidence to work from.

Beginning with the purpose
==========================

Begin with the pull request's description so you can identify the problem, the
intended behavior after the change, and the checks used to support it before
reading the implementation. If that reasoning is missing, ask for it before
trying to infer the purpose from the diff.

Review the parts you can assess and point out any methodological question that
needs another reviewer. Estimator changes need someone to check the
identification conditions, estimand, and inference against the source paper; a
successful test run cannot establish that the method itself was interpreted
correctly.

When you write a comment, explain which input produces a problem, where an
argument in the paper is lost, or what a reader would misunderstand. Giving
that consequence helps the author assess the concern and respond to it.
Distinguish a correction needed for correctness from an optional wording or
implementation suggestion so the author can see what needs to change.

Following the calculation
=========================

To assess a numerical change, first check how units, periods, treatment groups,
missing values, and weights reach the calculation. Those choices matter
because a correct formula applied to a different sample can still produce the
wrong effect. The :ref:`architecture guide <architecture>` shows where input
handling and estimation meet.

From there, follow the estimate into its influence function or other inference
calculation, if the method uses one. Check normalization, clustering, and
aggregation against the specification the pull request claims to implement.
Since existing estimators use several different result structures and
estimation paths, assess the relevant method rather than requiring every
function to follow one universal template.

For a new public argument or result field, compare what it means with related
APIs so users can carry the same concept from one estimator to another. Check
imports of optional dependencies too, since users of other estimators should still
be able to work without that extra. Anyone who does need it should get an
installation message when the extra is unavailable.

Assessing the evidence
======================

A regression test should demonstrate the failure that motivated a fix. For a
new estimator, look for numerical checks of both estimates and inference,
including a small case you can understand independently of the implementation.
The :doc:`testing` page explains how to choose those checks and interpret their
tolerances.

Review the reported test commands as well as their status. A skipped reference
comparison needs a dependency or platform check before it can support a
numerical claim. Although a seeded bootstrap makes a run repeatable, checking
only one dataset may still miss errors in weighting or sample alignment.

Performance evidence should measure the operation being changed, including any
setup costs that contribute to its runtime. For example, replacing a loop over
group-time comparisons calls for a benchmark of that workload. The
:ref:`benchmarking guide <benchmarking>` explains the project's timing tools;
a speed claim should state the input and settings used to measure it.

Reading the user's explanation
==============================

If behavior changes, read the affected docstring and guide as someone who has
not followed the development discussion. The explanation should say what the
function computes, what data it needs, and which choices change the
interpretation of its results. Worked guides should show the outputs and plots
they ask the reader to interpret.

A rendered documentation review can reveal broken references, clipped math, or
stale output that source text alone won't show. Check important callouts in
both color schemes and confirm that code a reader copies has the imports and
inputs it needs. :doc:`guide` gives the scratch build and preview commands.

Deciding whether to merge
=========================

If you're merging the contribution, read the final description against the
implementation and the answers to any numerical or methodological concerns.
The relevant checks need to pass for that version of the change. A failure in
an unrelated environment still needs an explanation so it isn't silently
treated as evidence for this change.

Resolve review discussions by recording the decision and its reason. When a
suggestion is declined, the technical explanation should remain visible to
future readers. If the pull request combines work that needs different reviews,
discuss splitting it so each change can be assessed on its own evidence.

Choose the merge strategy available in the repository that best preserves a
readable account of the contribution. A squash can collect a single change's
development commits into one commit, or a merge can retain a useful sequence of
separate implementation steps. Check the resulting commit message rather than
assuming the pull request title explains every merged edit.

Helping with issues and unfinished work
=======================================

When an issue comes in, try to reproduce the report on a small input so you can
identify the affected estimator or documentation page. If the report doesn't
include the function call, relevant package versions, or expected behavior, ask
for those details before diagnosing the calculation. A usage question may also
reveal a gap in the guide even when the calculation is working as intended.

If a contribution has been inactive, ask whether the author plans to continue
and describe what remains to be checked. Any decision to close it should leave
enough context for the author or another contributor to resume the work. When
continuing someone else's branch, credit the original contribution so readers
can follow where the work began.

Once a reviewed change is merged, the :doc:`release guide <releasing>` explains
how maintainers check the distribution that will carry it to users.
