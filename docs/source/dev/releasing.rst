.. _releasing:

================
Release Process
================

Most of the release pipeline is automated through GitHub Actions, but several
manual steps are required to initiate a release.

Overview
========

ModernDiD follows `semantic versioning <https://semver.org/>`__. Given a
version ``MAJOR.MINOR.PATCH``, increment **MAJOR** for incompatible API
changes, **MINOR** for new backwards-compatible functionality, and **PATCH**
for backwards-compatible bug fixes.

The current version is defined dynamically in `__init__.py <https://github.com/jordandeklerk/moderndid/tree/main/moderndid/__init__.py>`__ and
referenced by the build system through ``pyproject.toml``.

Pre-release checklist
=====================

Before starting a release, verify the following.

1. All CI checks pass on ``main``. Check the
   `Actions tab <https://github.com/jordandeklerk/moderndid/actions>`__ to
   confirm that the latest commit on ``main`` is green across all test
   matrices (core, Dask, Spark).

2. The full test suite passes. The weekly scheduled run
   (``test-full.yml``) exercises slow tests that are skipped in normal CI.
   If the most recent weekly run has failures, investigate and fix them
   before releasing.

3. Documentation builds cleanly. Run ``pixi run docs`` locally and check
   for warnings or broken cross-references.

4. Review recent changes. Scan the commit log since the last release to
   understand what's included::

      git log $(git describe --tags --abbrev=0)..HEAD --oneline

Preparing the release
=====================

Step 1: Bump the version
-------------------------

Update the version string in ``moderndid/__init__.py``::

   __version__ = "0.2.0"

Commit the version bump::

   git add moderndid/__init__.py
   git commit -m "REL: bump version to 0.2.0"

Step 2: Update release notes
-----------------------------

Add a section to `release <https://github.com/jordandeklerk/moderndid/tree/main/docs/source/release>`__ for the new version if you maintain
per-version release note files. At minimum, verify that the ``CHANGELOG.md``
will be auto-generated correctly (the changelog is generated from GitHub
releases by the ``post-release.yml`` workflow).

Step 3: Push and create a PR
------------------------------

Push the release preparation branch and open a PR::

   git push origin release-0.2.0

The PR title should follow the commit convention::

   REL: prepare 0.2.0 release

Once CI passes and the PR is approved, merge to ``main``.

Step 4: Tag the release
-------------------------

After the release PR is merged, tag the release on ``main``::

   git checkout main
   git pull upstream main
   git tag -a v0.2.0 -m "Release v0.2.0"
   git push upstream v0.2.0

The tag **must** start with ``v`` (e.g., ``v0.2.0``) to trigger the
publish workflow.

Automated publishing
====================

Pushing a ``v*`` tag triggers the ``publish.yml`` GitHub Actions workflow,
which handles the rest automatically.

1. The ``build-package`` job checks out the tagged commit, builds both a
   wheel and a source distribution, and generates
   `build provenance attestations <https://docs.github.com/en/actions/security-guides/using-artifact-attestations-to-establish-provenance-for-builds>`__.

2. The ``publish`` job downloads the built artifacts and uploads them to
   `PyPI <https://pypi.org/p/moderndid>`__ using
   `Trusted Publishing <https://docs.pypi.org/trusted-publishers/>`__ (OIDC).
   No API tokens are stored in the repository. Authentication is handled
   entirely through GitHub's OIDC identity.

The publish job runs in the ``publish`` GitHub environment, which requires
maintainer approval before execution. This provides a manual gate to prevent
accidental releases.

Post-release
============

After the tag is pushed and the package is published, complete the following
steps.

1. Create a GitHub Release. Go to the repository's
   `Releases page <https://github.com/jordandeklerk/moderndid/releases>`__
   and create a new release from the tag. Use the "Generate release notes"
   button to auto-populate the description with PR titles since the last
   release.

2. Verify the changelog update. The ``post-release.yml`` workflow runs
   automatically when a release is published. It uses
   `changelog-from-release <https://github.com/rhysd/changelog-from-release>`__
   to regenerate ``CHANGELOG.md`` from all GitHub releases and opens a PR
   with the updated file.

3. Verify the PyPI page. Check that the new version appears at
   https://pypi.org/p/moderndid and that the README renders correctly.

4. Verify documentation. If documentation is hosted on Read the Docs or a
   similar service, confirm that the new version builds and is available.

Hotfix releases
===============

If a critical bug is found in a released version, create a branch from
``main`` with the fix, follow the normal PR process for review and merge,
bump only the patch version (e.g., ``0.2.0`` to ``0.2.1``), and follow the
standard tagging and release process above.

Since ModernDiD does not maintain separate release branches, hotfixes go
through ``main`` like any other change.

Troubleshooting
===============

If the publish job failed, check the Actions log. Common causes include the
tag not matching the ``v*`` pattern, the PyPI Trusted Publisher configuration
being misconfigured (check the ``publish`` environment settings in the
repository), or the version in ``moderndid/__init__.py`` not matching the tag
(PyPI rejects duplicate versions).

If the changelog PR wasn't created, verify that the ``post-release.yml``
workflow has the ``contents: write`` and ``pull-requests: write`` permissions.
Check the Actions tab for the "Post-release" workflow to confirm it was
triggered.
