.. _workflow:

#########################
Working on a contribution
#########################

Once you've found a fix or a clearer explanation, someone else needs enough
context to check your work. We'll keep the problem, the new behavior, and its
tests together in a separate branch and a focused pull request. That gives the
reviewer a path through the change and lets you keep working without mixing it
with unrelated edits in your checkout.

Starting a branch
=================

If you've followed :doc:`guide`, ``origin`` points to your fork and
``upstream`` points to the main repository. Start a new contribution from the
latest upstream ``main`` after saving any work already in progress.

.. code-block:: bash

   git fetch upstream
   git switch -c fix-bootstrap-standard-errors upstream/main

Choose a branch name that describes the work, such as
``fix-bootstrap-standard-errors`` or ``clarify-continuous-dose-guide``. The
name helps you and a reviewer identify the branch; there is no hook that
enforces a particular naming pattern.

Keeping the change focused
==========================

As you work, inspect the diff for edits that don't help explain or fix the
problem. A formatting cleanup in another estimator is easier to review
separately because it needs different evidence from your bug fix. You can stage
selected parts of a file when several changes share your working directory.

.. code-block:: bash

   git diff
   git add -p
   git diff --staged

A commit message should name the behavior it changes so someone reading the
history doesn't need the original discussion. A subject such as ``BUG: preserve
county weights in an unbalanced panel`` explains more than ``fix another
issue``. Prefixes such as ``BUG``, ``ENH``, ``DOC``, ``TEST``, and ``MAINT``
can help identify the kind of work even though the hooks don't enforce them.
The generated changelog uses GitHub release content rather than parsing these
prefixes.

If the reason for the change isn't apparent from its subject, add it to the
commit body. For numerical code, that explanation may need to name the
estimand, the sample on which a failure occurs, or the distinction between an
estimate and its standard error.

Updating a branch during development
====================================

When upstream changes affect your work, fetch the new commits before choosing
how to incorporate them. If you're the only person working on the branch, a
rebase can replay your commits on the current ``main``.

.. code-block:: bash

   git fetch upstream
   git rebase upstream/main

Resolve any conflicts by checking the intended behavior on both sides; rerun
the affected tests after the rebase because a clean merge of text can still
change a calculation. If you've already shared the branch, coordinate before
rewriting its history. A merge is another way to incorporate upstream work
without changing existing commit identities.

.. admonition:: Keep review changes visible
   :class: tip

   Once review has begun, pushing additional commits usually makes feedback
   easier to follow. Discuss a rebase with collaborators before replacing
   the history they have already reviewed.

Opening the pull request
========================

After running the checks for your change, push the branch to your fork and open
a pull request against the main repository's ``main`` branch.

.. code-block:: bash

   git push -u origin fix-bootstrap-standard-errors

Use the description to give a reviewer the context you had when making the
change. For a bug fix, a small input that fails before the change and succeeds
afterward often explains the problem and its resolution more clearly than a
tour of the edited functions. If you're adding an estimator, explain which
design it supports and give the method's source and the checks used to verify
its estimates and inference.

Include the commands you ran and any limits to that evidence. For example,
check a changed guide through a rendered build as well as a unit test, and
confirm that a numerical validation ran rather than being skipped. Link the
relevant issue with ``Fixes #123`` when the pull request resolves it, or ``Refs
#123`` when the issue should remain open.

You can open a draft while a methodological question or implementation choice
is still being discussed. Describe the remaining work so a reviewer can focus
on that question without assuming the change is ready to merge.

Following the review
====================

The Actions checks on the pull request show which job and command failed. Use
that output to reproduce the failure locally rather than rerunning unrelated
tests. If feedback changes the specification or implementation, update the
tests and explanation that support the new behavior too.

Push follow-up changes to the same branch so they remain in the original pull
request. If you disagree with a suggestion, explain the technical reason in its
thread and give the reviewer evidence to assess it. :doc:`reviewing` describes
the method, code, and documentation checks that help us decide when a
contribution is ready to merge.
