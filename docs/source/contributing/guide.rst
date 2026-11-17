.. _contributing:

########################
Setting up to contribute
########################

To change moderndid, you need a checkout in which Python imports your edits and
the project's checks run as they will on your pull request. Without one, a test
can pass against your own versions of the dependencies yet fail against
everyone else's. A style problem or a broken example page would also first
surface in review.

We'll install the environments that the project locks to exact versions. You'll
then run one test file and the style checks before building the documentation
into a folder of your own.

.. admonition:: Propose larger changes first
   :class: tip

   Describe a new estimator or a change to the public API in the
   `issue tracker <https://github.com/jordandeklerk/moderndid/issues>`__ before
   writing code. The discussion there settles the method and its scope before
   review.

Getting a copy of the repository
================================

Your work lives in a fork, your own copy of moderndid on GitHub where you can
push branches. Fork the repository, clone your fork, and add the main
repository under the name ``upstream`` so you can fetch what others merge.

.. code-block:: bash

   git clone https://github.com/your-username/moderndid.git
   cd moderndid
   git remote add upstream https://github.com/jordandeklerk/moderndid.git

The :doc:`workflow` guide uses both names, ``origin`` for the fork you cloned
and ``upstream`` for the main repository.

Installing the development environment
======================================

`Pixi <https://pixi.sh/>`__ installs packages into named environments inside
your checkout and takes every version from ``pixi.lock``, a record of the exact
versions for each platform. You'll use the ``dev`` environment for the tests,
``docs`` for the documentation, and ``check`` for the style checks. The lock
file covers Linux and macOS but not Windows, where a virtual environment takes
Pixi's place as the next section shows.

After `installing Pixi <https://pixi.sh/latest/installation/>`__, install the
``dev`` environment and check which copy of moderndid it imports.
``pixi run -e <name>`` runs a command inside an environment and installs the
environment first if it's missing.

.. code-block:: bash

   pixi install -e dev
   pixi run -e dev python -c "import moderndid; print(moderndid.__file__)"

The path it prints should end in ``moderndid/__init__.py`` inside your clone.
The environment installs moderndid in editable mode, in which Python reads your
source files directly. Each edit takes effect the next time Python imports the
package.

Using a virtual environment instead
===================================

On Windows, or without Pixi, install the package in editable mode into a
virtual environment with Python 3.12 or newer.

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -e ".[all,test,dev]"

In Windows PowerShell, activate it with ``.venv\Scripts\Activate.ps1`` rather
than the ``source`` line. Each name in the brackets is an extra, a group of
optional packages that pip installs on request. Below, drop the
``pixi run -e <name>`` prefix from each command and run
``pre-commit run --all-files`` in place of ``pixi run lint``. To build the
documentation, install the ``doc`` extra too and start the build command at
``sphinx-build``.

Running the tests and style checks
==================================

While you work, run the tests that cover your change rather than the whole
suite. The tests for :func:`~moderndid.att_gt`, the group-time effects
estimator in ``moderndid/did/``, sit in the matching ``tests/did/`` folder.

.. code-block:: bash

   pixi run -e dev pytest tests/did/test_att_gt.py -m "not slow" -vv

The :doc:`testing` guide shows how to choose the tests a change needs and how
to write new ones. To find the code behind a behavior, start from the
:ref:`architecture guide <architecture>`.

The style checks are pre-commit hooks, small programs listed in
``.pre-commit-config.yaml`` that Git can also run before each commit. Besides
catching slips such as trailing whitespace or a leftover debugger call, they
run Ruff, a linter and formatter for Python. One more hook rewrites the
runnable notebooks that the next section describes.

The command ``pixi run lint`` runs every hook over every file. Here ``lint`` is
a task, a named command that ``pixi.toml`` defines.

.. code-block:: bash

   pixi run lint
   pixi run -e check pre-commit install

Hooks such as Ruff's formatter fix what they find and still report a failure.
Review their edits with ``git diff`` and run the checks again until they pass.
A clean run here also means the same hooks will pass in the style check on
your pull request. The second command makes Git run the hooks before each
commit.

.. admonition:: Run the checks on a branch
   :class: important

   The ``no-commit-to-branch`` hook refuses every commit to the ``main``
   branch. It also fails ``pixi run lint`` whenever your checkout is on
   ``main``. Start a branch as :doc:`workflow` shows before you run the style
   checks.

Previewing the documentation
============================

Whenever your change touches a docstring or a page, build the documentation
with Sphinx, the tool that turns ``docs/source/`` into the website. A build
shows whether every example page still runs its code and whether every link to
another page or section still finds its target.

The ``docs`` task in ``pixi.toml`` rebuilds every page into ``docs/_build`` on
each run. Instead, we'll build into a folder outside the checkout that later
builds can reuse.

.. code-block:: bash

   docs_scratch="${TMPDIR:-/tmp}/moderndid-docs"
   pixi run -e docs env -u VIRTUAL_ENV sphinx-build -b html \
       -d "$docs_scratch/doctree" docs/source "$docs_scratch/html"
   pixi run -e docs python -m http.server 8765 \
       --directory "$docs_scratch/html" --bind 127.0.0.1

Without ``env -u VIRTUAL_ENV``, IPython would put the packages of a virtual
environment active in your terminal ahead of the docs environment's when the
example pages run. The last command serves the site at
``http://127.0.0.1:8765`` until you stop it with Ctrl+C.

A build passes only when it reports no warning and no error. In the browser,
check the links, code, outputs, and figures of your pages in both color
schemes. Look closely at links to the API, since a role such as ``:func:`` that
finds no target renders as plain code without a warning.

A later build into the same folder rereads only the pages that changed and
reruns only the example pages whose cells changed. The rest reuse their outputs
from the execution cache in ``$docs_scratch/.jupyter_cache``. Run one build at a
time, because two builds can write to that cache at once.

.. admonition:: The cache misses library changes
   :class: warning

   The cache keys each example on its own cells and can't tell when library
   code that a page calls has changed. After such a change, delete
   ``$docs_scratch/.jupyter_cache`` before the next build. Otherwise the pages
   keep showing what the old code printed.

The ``make-notebooks`` hook writes the runnable copy of each example page that
the buttons at the top of the page open in Colab or molab. It keeps these
copies in ``docs/notebooks/`` and undoes any hand edit to them. A new example
page needs its name in ``EXAMPLES`` at the top of
``docs/notebooks/make_notebooks.py`` and a ``git add`` for its new notebook,
since the hook can't flag a file that Git doesn't track yet.

Adding a dependency
===================

A new dependency usually goes into two files, since it has to reach both the
users who install moderndid and the environments that test it.
``pyproject.toml`` declares the base package and the extras that pip installs
for users. In ``pixi.toml``, each estimator's extra has a feature of the same
name, a group of packages that the environments combine.

.. admonition:: Keep the base package light
   :class: tip

   If only one estimator needs a dependency, it usually belongs in that
   estimator's extra and the feature of the same name. Before adding it to the
   base package, consider whether users of unrelated estimators would need it
   too.

A package that only the documentation imports goes into the ``doc`` extra and
the ``docs`` feature. An optional public name also needs the import
registration that the :ref:`architecture guide <architecture-imports>`
describes. With it, a user who lacks the extra gets a message naming the
install command.

After any dependency change, regenerate ``pixi.lock`` with ``pixi lock`` and
``uv.lock`` with ``uv lock``, because the GPU test run installs from
``uv.lock``. Since the lock files and the installs on pull requests pick the
newest releases allowed, only polars gets tested at its lowest allowed version.
If you change a version floor, test the affected behavior at that floor and
give your reason in the pull request.

Once your change has a test and the checks run clean, :doc:`workflow` carries
it from a branch to a pull request.
