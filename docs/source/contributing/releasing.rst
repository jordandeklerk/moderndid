.. _releasing:

###################
Preparing a release
###################

A release publishes the reviewed code on ``main`` to PyPI under a new version
number and tells users in its notes what changed for their analyses. Because
PyPI never accepts a second upload of a version, even after the first is
deleted, a mistake that reaches it takes another release to fix. The checks
that could stop a bad release therefore all come before the tag that publishes
it.

This guide is for maintainers who can push to the main repository. We'll check
the code on ``main``, prepare the version and notes in a pull request, and
build the distributions before the tag. The last section follows the publish
workflow and what comes after it.

Checking the release candidate
==============================

Before anything carries a new version number, make sure that the code on
``main`` is ready to ship. From a checkout whose ``upstream`` remote points at
the main repository, as in the :doc:`setup guide <guide>`, the command below
lists one line for each pull request merged since the last release.

.. code-block:: bash

   git fetch upstream
   git log --oneline --first-parent v0.4.1..upstream/main

The pull request checks already ran the fast suite on each of those changes.
Start the "Full test suite" workflow on ``main`` from the Actions tab to run
the slow tests on every supported Python version along with the example
notebooks. If the changes touch the CuPy backend, an estimator's GPU route, or
the ``gpu`` extra, start the "GPU tests" workflow on ``main`` too, since only
``main`` can use its Modal token. For a change to an estimator's calculation,
run that estimator's :ref:`reference validation <testing-validation>` on your
machine.

.. admonition:: Match the test run to the release
   :class: important

   A green run supports the release only if it ran on the code you'll tag and
   covered the checks that its changes need. When the latest scheduled run
   predates the last merge, start a new one.

After building the documentation into a scratch folder as the setup guide
shows, read every page that the changes touch. Some example pages load a
stored result from ``docs/source/prerun`` instead of running a slow estimation
during the build. Such a file keeps showing old numbers after the code changes.
Delete any whose call, estimator, or result container changed since it was
saved. A local docs build then writes a fresh file to commit.

Preparing the version and notes
===============================

A pull request of its own brings the version number and the release notes into
``main`` under a title such as ``REL: prepare release v0.4.2``. Pick the
number from what the release changes for users. A patch release such as 0.4.2
fixes bugs or makes code faster without breaking code written for the previous
version. A minor release such as 0.5.0 adds new functions or options and,
while the version starts with 0, may also rename or remove part of the API, as
0.4.0 did when ``agg_ddd`` and ``compute_aggte`` took ``na_rm`` in place of
``dropna``.

The version lives in two files that nothing keeps in step. Flit, the build
backend that ``pyproject.toml`` names, reads ``__version__`` from
``moderndid/_version.py`` into the metadata of the distributions.

.. code-block:: python

   __version__ = "0.4.2"

The other copy is the ``version`` key in the ``[workspace]`` table of
``pixi.toml``.

.. code-block:: toml

   [workspace]
   version = "0.4.2"

Changing ``_version.py`` also makes the ``make-notebooks`` pre-commit hook pin
the install cell of every runnable notebook in ``docs/notebooks/`` to the new
version. Commit the rewritten notebooks in the preparation pull request too.

.. admonition:: Tag soon after the merge
   :class: tip

   Once the preparation pull request merges, the notebooks that the latest
   docs open from ``main`` install a version that PyPI doesn't have yet. Tag
   the release soon after the merge to close that gap.

The notes go in a new page, ``docs/source/release/<version>-notes.rst``, that
you add to the table and the hidden toctree of
``docs/source/release/index.rst``. Like the latest notes, it opens with what
changed for users and whether their code keeps working. A patch release's
notes call it a patch release in their first sentence. Each item says how a
change affects an analysis and what a user has to do.

The list of merged pull requests at the end names only the ones that people
wrote. Updates from Renovate and pre-commit.ci, the bots that bump
dependencies and hooks, stay out of it along with those from the changelog
workflow. The sentence above the list gives the total number merged and how
many the list shows.

Checking the built distributions
================================

Users install the wheel and source distribution that the publish workflow
builds rather than your checkout. Before you tag, build both from the
preparation branch and install the wheel the way a user would.

.. code-block:: bash

   rm -rf dist
   pixi run -e build package
   ls dist

The ``package`` task builds both files with ``python -m build`` in an isolated
environment, as the publish workflow does. Because the build never removes
earlier files, clearing ``dist`` first leaves only the two new ones. Both file
names carry the version that Flit read from ``_version.py``.

.. code-block:: bash

   python -m venv /tmp/moderndid-wheel
   /tmp/moderndid-wheel/bin/python -m pip install dist/moderndid-0.4.2-py3-none-any.whl
   /tmp/moderndid-wheel/bin/python -c "import moderndid as did; print(did.__version__, did.load_mpdta().shape)"

The last command prints the version and the shape of the county data,
``(2500, 6)``. Loading the county data in the fresh environment also shows
that the wheel carries the package's data files. Install any extras that the
changed code needs into that environment and run that code there too. If
review changes the preparation branch, repeat the checks that the edits
affect.

Tagging the reviewed commit
===========================

Pushing a tag whose name starts with ``v`` publishes the release. The tag
therefore goes on the commit that merged the reviewed preparation pull request.
Replace 0.4.2 in the commands below with the version you built.

.. code-block:: bash

   git fetch upstream
   git log -1 --oneline upstream/main
   git tag -a v0.4.2 -m "Release v0.4.2" upstream/main

The log line should name the merge of your preparation pull request.

.. admonition:: A pushed tag starts the upload
   :class: warning

   Check the tag, its commit, and the version in the built files before you
   push, because the publish workflow uploads whatever the tag points at.

.. code-block:: bash

   git push upstream v0.4.2

Following publication
=====================

After the push, the publish workflow in ``.github/workflows/publish.yml``
builds the two distributions and uploads them to PyPI. It builds on every push
to ``main`` as well but uploads only for a tag. Through trusted publishing,
PyPI accepts the upload on a token that GitHub issues to the workflow run
rather than a stored password. The upload job runs in a GitHub environment
named ``publish``. If the repository settings give that environment a required
reviewer, the job waits for an approval in the Actions tab.

Once the upload succeeds, check that
`PyPI <https://pypi.org/project/moderndid/>`__ shows the new version and its
description. Then create a GitHub Release from
the tag and let GitHub generate its notes. The generated notes list each merged
pull request by its title and leave out the bots that ``.github/release.yml``
names. Rename any merged pull request whose title wouldn't explain its change
to a user before you generate them.

Publishing the GitHub Release starts the post-release workflow in
``.github/workflows/post-release.yml``. It rebuilds ``CHANGELOG.md`` at the
root of the repository from the GitHub Releases and opens a pull request with
the update for you to merge. Read the Docs then builds its stable version from
the new tag. Check that the stable docs show the new version and its page in
the :ref:`changelog <release-notes>`.
