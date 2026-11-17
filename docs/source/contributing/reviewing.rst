.. _reviewing:

########################
Reviewing a contribution
########################

A review decides whether a pull request does what its description says.
Passing checks can't settle that on their own, because tests written alongside
an estimator can share its misreading of the paper. A test that skips also
counts toward a passing run. We'll read a change in the order its effects reach
a user, from the description through the calculation and the evidence to the
documentation. You can help without knowing every estimator, since reading a
guide as a new user or reproducing an input error often turns up something the
author missed.

Beginning with the purpose
==========================

Read the description before the diff, since it should name the problem, the
intended behavior, and the checks that support it. Ask for that reasoning when
it's missing rather than inferring it from the code.

Review what you can assess and flag any methodological question for another
reviewer. An estimator change needs someone to check its identification
conditions, estimand, and inference against the source paper. A comment helps
most when it gives the consequence of a problem, such as the input that
triggers it, the step of the paper's argument that the code drops, or what a
reader would misunderstand.

.. admonition:: Label required changes
   :class: tip

   Mark each comment as needed for correctness or as an optional suggestion
   about wording or implementation. The author can then see what has to change
   before the merge.

Following the calculation
=========================

For a numerical change, first check how units, periods, treatment groups,
missing values, and weights reach the calculation. A correct formula applied to
the wrong sample still gives the wrong effect. The :ref:`architecture guide
<architecture>` shows where input handling and estimation meet.

Then follow the estimate into its influence function or whatever else the
method uses for inference. Check normalization, clustering, and aggregation
against the specification that the pull request claims to implement. Judge
each estimator against its own method, because the estimators don't share one
result structure or one estimation path.

.. admonition:: Keep shared names consistent
   :class: important

   A new public argument or result field should mean what the same name means
   in other estimators. Most of them take ``alpha`` for the significance level
   and ``biters`` for the number of bootstrap draws, for example.

Code that needs an extra, a group of optional packages that only some users
install, has to leave the rest of the package importable without it. A user who
calls that code without the extra should get a message naming the install
command. Two jobs on each pull request check both by installing moderndid with
no extras or with the plots extra alone.

Assessing the evidence
======================

A regression test should fail before the fix and pass after it. For a new
estimator, look for checks of both the estimates and the inference on more than
one dataset, since a single dataset can hide an error in weighting or in sample
alignment. One check should be a small case you can verify without the
implementation. The :doc:`testing` guide explains how to choose those checks
and their tolerances.

The GPU tests show how a run can pass with nothing checked. They need CuPy, the
library through which moderndid runs on a GPU, and a CUDA device to run on. In
the command below, ``-k ddd`` picks out the GPU tests of
:func:`~moderndid.ddd`, the triple difference estimator. The ``-rs`` option
lists each skipped test with its reason in a report that ``-q`` keeps short.

.. code-block:: bash

   pixi run -e dev pytest tests/core/test_gpu.py -k ddd -rs -q

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         sssssss                                                                  [100%]
         =========================== short test summary info ============================
         SKIPPED [3] tests/core/test_gpu.py:172: needs CuPy and a CUDA GPU
         SKIPPED [2] tests/core/test_gpu.py:194: needs CuPy and a CUDA GPU
         SKIPPED [2] tests/core/test_gpu.py:220: needs CuPy and a CUDA GPU
         7 skipped, 58 deselected in 0.02s

On a machine without a GPU, the run exits without an error even though none of
the seven tests ran. Since pull requests don't run the GPU tests either, a
change to GPU code needs the run on a cloud GPU that the :ref:`GPU testing
guide <testing-gpu>` describes.

.. admonition:: A skipped comparison is not evidence
   :class: warning

   Before you accept a numerical claim, check the skip summary of the run
   behind it. The :ref:`validation tests <testing-validation>` skip whenever
   their reference implementation is missing.

A speed claim should state the input and settings behind it. Replacing a loop
over group-time cells, for example, calls for a benchmark of that workload. The
:ref:`benchmarking guide <benchmarking>` shows how to run one with the
project's timing tools.

Reading the user's explanation
==============================

If behavior changes, read the affected docstring and guide as someone who
hasn't followed the discussion. They should say what the function computes,
what data it needs, and which choices change how its results read. A rendered
build also reveals broken references, clipped math, and stale outputs that the
source hides. Check callout boxes in both color schemes and confirm that code a
reader might copy has the imports and inputs it needs. The :doc:`setup guide
<guide>` gives the commands for a build of your own.

Deciding whether to merge
=========================

Before merging, read the final description against the implementation and the
answers to any numerical or methodological concern. The relevant checks need
to pass on that final version of the change.

.. admonition:: Explain unrelated failures
   :class: important

   A failure in an unrelated job still needs an explanation before the merge.
   Without one, nobody reading the pull request later can tell whether this
   change caused it.

Resolve each discussion by recording the decision and its reason. A declined
suggestion should keep its technical explanation visible to future readers. If
the pull request combines work that needs different reviews, discuss splitting
it so each part can be judged on its own evidence.

Of the merge methods the repository allows, pick the one that leaves the most
readable history. A squash merge collects all of a change's development commits
into one commit. A merge commit keeps separate implementation steps that are
worth keeping apart. Either way, check the title and the resulting commit
message rather than assuming they describe every merged edit.

Helping with issues and unfinished work
=======================================

When an issue comes in, try to reproduce it on a small input to find the
estimator or page it affects. If the report leaves out the function call, the
package versions, or the expected behavior, ask for them before diagnosing the
calculation. A usage question can also reveal a gap in a guide when the
calculation works as intended.

For a contribution that has gone quiet, ask whether the author plans to
continue and describe what remains to be checked. A decision to close it should leave enough
context for someone to resume the work. When you continue someone else's
branch, keep their commits so the history shows where the work began.

Once a reviewed change is merged, the :doc:`release guide <releasing>` explains
how maintainers check the distribution that carries it to users.
