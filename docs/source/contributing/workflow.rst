.. _workflow:

#########################
Working on a contribution
#########################

Once you have a fix or a clearer explanation, it reaches users only after
review. Reviewers see only your diff and what you write about it. A diff that
mixes in unrelated edits, or a description that leaves out how you checked the
result, makes them reconstruct your reasoning before they can judge it. We'll
follow one change from a new branch to a pull request and through its review.

Starting a branch
=================

Each contribution gets its own branch, a line of commits kept apart from
``main`` and from your other work. Name it after the work and start it from the
newest ``main`` of the repository that the :doc:`setup guide <guide>` named
``upstream``. Commit or set aside any work in progress first, since uncommitted
edits follow you onto the new branch.

.. code-block:: bash

   git fetch upstream
   git switch -c fix-unbalanced-county-weights upstream/main

Keeping the change focused
==========================

Keep each pull request to the edits that its problem needs. A formatting
cleanup in another estimator needs different evidence from your bug fix and is
easier to review on its own. The docstring and any guide that explain a change
to what users pass in, get back, or conclude from a result belong in the same
pull request. Read your diff before each commit and stage only the parts that
belong together.

.. code-block:: bash

   git diff
   git add -p
   git diff --staged

``git add -p`` asks about each changed block of a file before staging it.
``git diff --staged`` then shows what the next commit will hold.

Write each commit message for someone who reads the history without the
original discussion. A subject such as
``BUG: preserve county weights in an unbalanced panel`` says more than
``fix another issue``. The project's subjects open with a prefix for the kind
of work, such as ``BUG``, ``ENH``, ``DOC``, ``TEST``, or ``MAINT``. If the
subject doesn't say why the change is needed, give the reason in the commit
body. In numerical code, that reason may name the estimand, the sample on which
the old code failed, or whether an estimate or its standard error was wrong.

Updating a branch during development
====================================

When upstream changes affect your work, fetch them before deciding how to bring
them in. If you're the only person working on the branch, a rebase replays your
commits on top of the newest ``main``.

.. code-block:: bash

   git fetch upstream
   git rebase upstream/main

Resolve any conflicts by checking the intended behavior on both sides. Rerun
the affected tests afterward, because text that merges cleanly can still change
a calculation. A branch you've already pushed then needs
``git push --force-with-lease``, the force push that refuses to overwrite
commits on your fork that you haven't fetched. Merging ``upstream/main``
instead brings in the same work without rewriting any commit.

.. admonition:: Keep review changes visible
   :class: tip

   Once review has begun, new commits on top of the branch show reviewers what
   changed since their last read. Discuss a rebase with them before replacing
   history they've already reviewed.

Opening the pull request
========================

A pull request asks the main repository to merge your branch. Once the checks
for your change pass locally, push the branch to your fork and follow the link
that GitHub prints to open the pull request against ``main``.

.. code-block:: bash

   git push -u origin fix-unbalanced-county-weights

With ``-u``, later pushes from this branch go to your fork without naming it.
Give the pull request a title like a commit subject, prefix included, because
the changelog lists each merged pull request by its title.

Use the description to give a reviewer the context you had. For a bug fix, a
small input that fails before the change and succeeds after it explains more
than a tour of the edited functions. For a new estimator, name the design it
supports, the paper it implements, and the checks behind its estimates and
inference.

List the commands you ran and the limits of that evidence. A changed guide
needs a rendered build as well as a unit test. A numerical validation supports
a claim only if it ran rather than skipped. When the pull request resolves an
issue, ``Fixes #123`` in the description closes the issue at the merge.
``Refs #123`` links an issue without closing it at the merge.

.. admonition:: Use a draft for unsettled choices
   :class: tip

   Open a draft pull request while a methodological question or an
   implementation choice is still under discussion. Describe the remaining work
   so reviewers can focus on that question.

Following the review
====================

When one of the checks on the pull request fails, its log names the command
and the environment to rerun locally. The :doc:`testing` guide describes what
each check runs and on which platforms.

If feedback changes the specification or the implementation, update the tests
and the explanation that support the new behavior too. Push follow-up commits
to the same branch so they stay in the pull request. If you disagree with a
suggestion, give the technical reason and your evidence in its thread.
:doc:`reviewing` describes how reviewers read the method, the code, and the
documentation before deciding to merge.
