.. _workflow:

==============================
Git Workflow and Conventions
==============================

Following consistent git conventions keeps the commit history readable, makes
automated changelog generation possible, and helps reviewers understand your
changes at a glance.

Commit message format
=====================

Every commit message should begin with a **category prefix** followed by a
colon and a short description. The prefix indicates the type of change and is
used by tooling to generate changelogs and filter history.

Use the imperative mood in the subject line ("add feature", not "added feature"
or "adds feature"). Keep the subject under 72 characters. If more detail is
needed, add a blank line followed by a longer explanation in the body.

::

   ENH: add bootstrap confidence intervals for DDD estimator

   Implements the multiplier bootstrap for the triple-differences
   estimator, following the same pattern as the DiD module. Uses
   Mammen two-point weights and supports both pointwise and
   simultaneous confidence bands.

Standard prefixes
-----------------

The following prefixes are used throughout the project. Choose the one that
best describes the primary purpose of your commit.

::

   BUG:   Bug fix
   BEN:   Benchmark additions or modifications
   CI:    Changes to CI configuration or workflows
   DEV:   Development tool changes (pre-commit, pixi, tox config)
   DOC:   Documentation only (docstrings, RST files, README)
   ENH:   Enhancement to existing functionality
   FEAT:  New feature or module
   FIX:   Equivalent to BUG (either is acceptable)
   MAINT: Maintenance (dependency updates, cleanup, deprecation removal)
   REF:   Code refactoring with no behavior change
   REL:   Release-related changes (version bumps, release notes)
   TEST:  Test additions or modifications

When a commit spans multiple categories, use the prefix for the most
significant change. A refactoring that also fixes a bug should use ``BUG``.
A new feature that includes its tests should use ``FEAT``.

Messages should be understandable without looking at the code changes. A commit
message like ``MAINT: fixed another one`` is an example of what not to do; the
reader has to go look for context elsewhere.

Branch naming
=============

Create a new branch for each piece of work. It's usually a good idea to use a descriptive name that
reflects what the branch does, with words separated by hyphens::

   git checkout -b fix-bootstrap-standard-errors
   git checkout -b add-ddd-event-study-plots
   git checkout -b ref-consolidate-preprocessing

There is no strict naming convention beyond clarity. It's best to avoid generic names like
``patch`` or ``update``. The branch name appears in the merge commit, so
future readers should be able to tell what the branch was about.

Working with branches
=====================

Starting a new branch
---------------------

Always branch from an up-to-date ``main``::

   git fetch upstream
   git checkout -b my-feature upstream/main

This ensures your branch starts from the latest code and avoids unnecessary
merge conflicts.

Keeping your branch current
----------------------------

If ``main`` has moved forward while you're working, rebase your branch onto
the latest ``main``::

   git fetch upstream
   git rebase upstream/main

Rebasing replays your commits on top of the updated ``main``, resulting in a
clean, linear history. If you've already pushed your branch and need to force
push after rebasing::

   git push --force-with-lease origin my-feature

The ``--force-with-lease`` flag is safer than ``--force`` because it refuses
to overwrite commits that someone else may have pushed to your branch.

.. note::

   Rebasing on ``main`` is preferred over merging upstream back to your
   branch. Using ``git merge`` and ``git pull`` is discouraged when
   working on feature branches.

Making clean commits
--------------------

Each commit should represent a single logical change. Avoid commits that mix
unrelated changes (e.g., a bug fix and a formatting cleanup in the same
commit). If you have unstaged changes you want to split across commits, use::

   git add -p

This lets you stage individual hunks interactively.

If you have work-in-progress commits that you want to clean up before
submitting a PR, use interactive rebase to squash or reword them::

   git rebase -i upstream/main

Common operations in interactive rebase are

- ``squash`` combines multiple small commits into one meaningful commit
- ``reword`` fixes a commit message without changing the code
- ``drop`` removes a commit entirely (e.g., a debugging commit you forgot
  to remove)

Recovering from mistakes
------------------------

**Undo the last commit** (keep changes staged)::

   git reset --soft HEAD~1

**Undo the last commit** (keep changes unstaged)::

   git reset HEAD~1

**Discard all uncommitted changes** (use with care)::

   git checkout -- .

**Recover a deleted branch or lost commit**::

   git reflog

The reflog shows recent HEAD positions and is useful for finding commits
that are no longer referenced by any branch.

Pull request workflow
=====================

1. Push your branch to your fork::

      git push -u origin my-feature

2. Open a pull request against ``main`` on GitHub.

3. Write a clear title and description. The title should follow the same
   prefix convention as commits (e.g., "ENH: add bootstrap for DDD
   estimator"). The description should explain *what* changed and *why*,
   not just restate the diff. Reference any related issues with
   "Closes #123" or "Fixes #456".

4. CI runs automatically. All checks must pass before merging.

5. Address review feedback by pushing new commits to the same branch.
   Avoid force-pushing during review unless asked, as it makes it harder
   for reviewers to see incremental changes.

6. Once approved, the maintainer will merge your PR. We typically use
   squash merges for single-purpose PRs and regular merges for larger
   branches with meaningful individual commits.

Skipping CI on draft commits
-----------------------------

If you push a work-in-progress commit and don't want to consume CI resources,
add ``[skip ci]`` to the commit message. This skips all GitHub Actions jobs
for that push::

   git commit -m "DOC: wip draft of user guide [skip ci]"

Use this sparingly and only for genuine drafts. Remove the tag before
requesting review, since CI must pass before merging.

Linking issues
--------------

Use GitHub keywords in commit messages and PR descriptions to link to related
issues. When the PR is merged, referenced issues are closed automatically.

- ``Closes #123`` or ``Fixes #123`` closes the issue on merge
- ``See #123`` or ``Refs #123`` links without closing
