.. _contributing:

########################
Setting up to contribute
########################

A checkout of moderndid gives you a place to trace an unexpected estimate,
improve an explanation, or try a method the package doesn't yet provide.
We'll set it up here and use the tests and documentation
build to check what your change does before presenting it for review.

For a bug report, include a small dataset or an existing data loader, the
function call, and the behavior you expected to see. You can open a report in
the `issue tracker <https://github.com/jordandeklerk/moderndid/issues>`__
before working on a fix. For a new estimator or a change to the public API,
describing the proposal there first gives us a place to discuss the method and
its scope.

Preparing your checkout
=======================

Fork the repository on GitHub, clone your fork, and add the main repository as
``upstream`` so you can keep your checkout current.

.. code-block:: bash

   git clone https://github.com/your-username/moderndid.git
   cd moderndid
   git remote add upstream https://github.com/jordandeklerk/moderndid.git

The project uses `Pixi <https://pixi.sh/>`__ to manage its development,
documentation, and validation environments. Their dependencies and tasks are
defined in ``pixi.toml`` and their resolved versions are recorded for each
supported platform in ``pixi.lock``. The Pixi environments support Linux and
macOS; the package itself also runs on Windows.

After `installing Pixi <https://pixi.sh/latest/installation/>`__, install the
development environment and create a branch for your contribution.

.. code-block:: bash

   pixi install -e dev
   git switch -c fix-bootstrap-standard-errors
   pixi run -e dev python -c "import moderndid; print(moderndid.__file__)"

The path printed above tells you which copy of the package Python is using.
It should point into this checkout because the installation is editable.
That means changes to the source become available without reinstalling the package. You
can keep using ``pixi run -e dev`` to select this environment for a command even
if another virtual environment is active in your terminal.

If you're developing on Windows or prefer a virtual environment, use Python
3.12 or newer and install the development extras directly.

.. code-block:: bash

   python -m venv .venv

Activate ``.venv`` so the installation commands below apply to your isolated
environment. On Linux and macOS, use ``source .venv/bin/activate``; in Windows
PowerShell, use ``.venv\Scripts\Activate.ps1``.

.. code-block:: bash

   python -m pip install --upgrade pip
   python -m pip install -e ".[all,test,dev]"

You can then run ``pytest`` and ``pre-commit`` directly in that environment.
The separate documentation dependencies are available through the ``doc`` extra
if your contribution needs a Sphinx build.

Finding the code for your change
================================

You can find each estimator under ``moderndid/`` and its tests and fixtures in
the corresponding folder under ``tests/``. Shared input handling in
``moderndid/core/`` determines the sample and specification those estimators
receive. Use the :ref:`architecture guide <architecture>` to understand how
those pieces connect or :ref:`adding an estimator <new-estimator>` to follow a
new method through the public API.

Before editing an estimator, reproduce the behavior on the smallest input that
still exposes it. A preprocessing error needs a different check from an
incorrect influence function or confidence interval. Keeping that distinction
clear helps you choose a test that would fail for the original problem rather
than merely exercise the new code.

Public functions use NumPy-style docstrings to explain their inputs and
results. Worked analyses belong in the guides under ``docs/source/`` so readers
can follow the choices that give those results their meaning. If your change
alters a user's inputs, results, or interpretation, update that explanation
alongside the code. The :doc:`testing` page covers regression tests and
numerical checks in more detail.

Checking your work locally
==========================

During development, run the file or test that covers the behavior you're
changing. For example, work on group-time effects can begin with a focused test
file rather than every estimator in the package.

.. code-block:: bash

   pixi run -e dev pytest tests/did/test_att_gt.py -m "not slow" -vv
   pixi run lint

The lint task runs the hooks in ``.pre-commit-config.yaml``. They check file
syntax and whitespace as well as Python linting and formatting with Ruff. Since
some hooks edit files, inspect the resulting diff and rerun the checks if they
report fixes. You can install the same hooks to run before each local commit.

.. code-block:: bash

   pixi run -e check pre-commit install

.. admonition:: Work on a branch
   :class: important

   Keep your contribution on its own branch because the hooks refuse
   commits to ``main`` during the local commit checks.

Previewing documentation
========================

Read your edited page after a build as well as in its source form. Since Sphinx
resolves cross-references and executes the example pages, the rendered
review can reveal stale calls or outputs that a text edit would miss. We'll
use a scratch directory to keep this build separate from the maintainer's
``docs/_build`` and ``docs/_doctree`` folders.

.. code-block:: bash

   docs_scratch=$(mktemp -d "${TMPDIR:-/tmp}/moderndid-docs.XXXXXX")
   pixi run -e docs env -u VIRTUAL_ENV sphinx-build \
       -b html --keep-going -W \
       -d "$docs_scratch/doctree" docs/source "$docs_scratch/html"
   pixi run -e docs python -m http.server 8765 \
       --directory "$docs_scratch/html" --bind 127.0.0.1

Open ``http://127.0.0.1:8765`` to read the page as a user would. Check the
links, code blocks, outputs, and figures in both color schemes; stop the server
with ``Ctrl+C`` when you're done. Reuse the same scratch directory for later
builds so unchanged examples can use their execution cache. Run one Sphinx
build at a time because concurrent builds can write to the same cache.

Managing dependencies
=====================

A new dependency needs a place in both the user installation and the
environment that tests it. ``pyproject.toml`` describes what users install and
``pixi.toml`` defines the environments used to develop and check the package.
If only one estimator needs the dependency, it usually belongs in that
estimator's optional extra and the matching Pixi feature. Consider whether
users of unrelated estimators would need it before adding it to the base
package.

Update both dependency declarations and regenerate ``pixi.lock`` when changing
an environment. For a documentation import, check the ``doc`` extra and the
``docs`` Pixi feature. Optional public names also need the appropriate import
registration described in :ref:`adding an estimator <new-estimator>` so a
missing dependency produces a useful installation message.

The minimum versions in ``pyproject.toml`` are installation constraints; the
lockfile and CI installs can select newer releases. Passing those checks alone
doesn't establish compatibility with every declared minimum. If you change a
version floor, test the affected behavior at that floor and explain the reason
in the pull request.

Once the change has a focused test and a clean rendered explanation,
:doc:`workflow` shows how to present it for review without losing the context
that helped you make it.
