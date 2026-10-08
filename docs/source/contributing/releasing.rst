.. _releasing:

###################
Preparing a release
###################

A release turns the reviewed source on ``main`` into the distribution users
install. Before the GitHub workflow builds and publishes that distribution,
maintainers need to choose the version, check the artifacts, and explain the
changes in the release notes. We'll follow those checks before the tag that
starts publication.

Checking the release candidate
==============================

Once you've chosen the commit you intend to release, review the changes since
the previous tag to see which estimator tests and reference comparisons it
needs. A successful run on ``main`` supports this release only if it covered
that same commit and the relevant checks, including slow tests affected by the
changes. You can trigger the weekly full-suite workflow manually when its
latest run doesn't cover the candidate.

Build and read the documentation using the scratch procedure in :doc:`guide`.
If an example uses a stored result, confirm that it was recomputed after any
change to the call, estimator, or result structure; a cached output is not
evidence that the candidate produces that result.

Review changes to the public API before choosing a version that reflects their
consequences for existing users. Under `semantic versioning
<https://semver.org/>`__, patch versions describe compatible bug fixes and
minor versions describe compatible additions. A release that changes accepted
inputs or the interpretation of an existing result needs an explicit
compatibility explanation, especially while the project is in its initial
development series.

Preparing the version and notes
===============================

The distribution version is defined by ``__version__`` in
``moderndid/_version.py``. ``pyproject.toml`` marks the version as dynamic;
Flit reads this file when building the package. Since the top-level attribute
reports installed distribution metadata, ``moderndid.__version__`` can continue
to show the previous installed version until you reinstall. The workspace
version in ``pixi.toml`` is separate from the version written into the
distribution.

For example, a patch release after ``0.2.0`` would update the version
assignment to ``0.2.1`` on a preparation branch.

.. code-block:: python

   __version__ = "0.2.1"

The release notes under ``docs/source/release/`` should help users understand
how changes to estimates, inference, input handling, or defaults affect their
analysis and what they need to do to obtain the corrected behavior. Include
the new page in that section's table and hidden toctree so readers can find it
alongside the other releases.

The documentation release notes and ``CHANGELOG.md`` have different sources.
The changelog workflow regenerates ``CHANGELOG.md`` from GitHub Releases after
publication; it doesn't infer the release explanation from a version assignment
or commit prefix. Prepare the GitHub Release description with the same care as
the documentation page.

Checking the built distributions
================================

The build environment's package task builds a wheel and source distribution in
``dist/``. Run it from the repository root after the preparation changes are
ready to review.

.. code-block:: bash

   pixi run -e build package

Check the metadata inside the artifacts rather than relying on a version
printed by an existing editable installation. The following inspection prints
the package name and version for each distribution currently in ``dist/`` so
old files are visible too.

.. code-block:: python

   from email.parser import Parser
   from pathlib import Path
   from tarfile import open as open_tar
   from zipfile import ZipFile

   distributions = Path("dist")

   for wheel in sorted(distributions.glob("*.whl")):
       with ZipFile(wheel) as archive:
           metadata_name = next(
               name for name in archive.namelist()
               if name.endswith(".dist-info/METADATA")
           )
           metadata = Parser().parsestr(archive.read(metadata_name).decode())
       print(wheel.name, metadata["Name"], metadata["Version"])

   for source in sorted(distributions.glob("*.tar.gz")):
       with open_tar(source) as archive:
           metadata_name = next(
               name for name in archive.getnames()
               if name.count("/") == 1 and name.endswith("/PKG-INFO")
           )
           metadata = Parser().parsestr(
               archive.extractfile(metadata_name).read().decode()
           )
       print(source.name, metadata["Name"], metadata["Version"])

Once the metadata agrees with the intended version, check that the wheel and
source distribution include any package data the release needs. Installing the
candidate wheel in a separate environment lets you check its public imports
and affected behavior using the artifact users will receive. An editable
checkout can miss files omitted from that distribution, even when its own
checks pass.

Submit the version and notes through a pull request so the release candidate is
reviewed before it is tagged. If review changes the candidate, repeat the
checks affected by those edits against the final commit.

Tagging the reviewed commit
===========================

After the preparation pull request has merged, bring your local ``main`` to the
upstream commit and confirm its identity. The version below is an example; the
tag must agree with the version you checked in the artifacts.

.. code-block:: bash

   git fetch upstream
   git switch main
   git merge --ff-only upstream/main
   git rev-parse HEAD
   git tag -a v0.2.1 -m "Release v0.2.1"

.. admonition:: Check before pushing the tag
   :class: warning

   Pushing a ``v*`` tag starts the publishing workflow for that commit.
   Confirm the tag, artifact metadata, and reviewed commit agree before
   taking this step.

A maintainer with access to the upstream repository can push the tag after
those checks are complete.

.. code-block:: bash

   git push upstream v0.2.1

Following publication
=====================

``.github/workflows/publish.yml`` builds distributions on pushes to ``main``
and on ``v*`` tag pushes. Its publish job runs only for a tag push and
downloads the artifacts from the build job. Authentication uses PyPI Trusted
Publishing through GitHub's identity token rather than a package-upload token
stored in the workflow.

Before a release, review the protection rules for the GitHub environment named
``publish`` and the PyPI publisher configuration used by the publish job. Any
approval requirements are configured in repository settings rather than
established by the environment name alone.

After publication succeeds, check that the expected version and package
description appear on `PyPI <https://pypi.org/project/moderndid/>`__. Create
the GitHub Release from the same tag and review its generated description
before publishing it. Generated pull request titles may need context before
they explain a change to a user.

``post-release.yml`` runs on GitHub Release publication events and also
supports a manual trigger. It regenerates ``CHANGELOG.md`` and opens a pull
request for the update. Check that pull request and the Read the Docs build so
the published package and its documentation describe the same release.

If a job fails, use its logs to distinguish a build error, an identity or
environment configuration error, and an upload rejected because the version
already exists. A GitHub tag does not by itself validate the metadata inside
the package. If a published release needs a correction, prepare a new version
through the same reviewed process rather than trying to replace its uploaded
files.
