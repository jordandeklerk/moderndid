"""Add buttons that open a page's generated notebook in Colab or molab to the top of the page.

``docs/notebooks/make_notebooks.py`` writes the notebooks. Since the buttons point at the commit, branch, or tag
that Read the Docs built, each docs version opens its own notebooks. A local build points at main.
A page without a notebook, such as the quickstart, shows no buttons.
"""

import os
import posixpath
from pathlib import Path

from docutils import nodes

HOSTS = {"Colab": "https://colab.research.google.com/github", "molab": "https://molab.marimo.io/github"}
ICONS = {"Colab": "_static/colab.svg", "molab": "_static/molab.svg"}


def setup(app):
    """Register the buttons with Sphinx."""
    # Sphinx caches the doctrees it reads. Adding the links as it writes a page gives them this build's ref.
    app.connect("doctree-resolved", _add_buttons)
    metadata = {"parallel_read_safe": True, "parallel_write_safe": True}
    return metadata


def _add_buttons(app, doctree, docname):
    """Put the buttons right under the title of a page that has a generated notebook."""
    section, _, name = docname.rpartition("/")
    if section == "user_guide":
        path, services = f"{name}.ipynb", ("Colab", "molab")
    elif section == "r_comparisons":
        # Since molab can't run the second language the comparison notebooks call, they open in Colab alone.
        path, services = f"r_comparisons/{name}.ipynb", ("Colab",)
    else:
        return
    title = next(doctree.findall(nodes.title), None)
    if title is None or not (Path(app.srcdir).parent / "notebooks" / path).exists():
        return
    blob = f"jordandeklerk/moderndid/blob/{_git_ref()}/docs/notebooks/{path}"
    folder = posixpath.dirname(docname)
    # A space between the links keeps the search index from fusing their labels into one word.
    links = " ".join(
        f'<a href="{HOSTS[service]}/{blob}" target="_blank" rel="noopener">'
        f'<img src="{posixpath.relpath(ICONS[service], folder)}" alt="">Open in {service}</a>'
        for service in services
    )
    title.parent.insert(
        title.parent.index(title) + 1, nodes.raw("", f'<p class="mdid-notebook-buttons">{links}</p>', format="html")
    )


def _git_ref():
    """Return the commit, branch, or tag that Read the Docs built so each docs version opens its own notebooks."""
    if os.environ.get("READTHEDOCS_VERSION_TYPE") == "external":
        return os.environ["READTHEDOCS_GIT_COMMIT_HASH"]
    return os.environ.get("READTHEDOCS_GIT_IDENTIFIER", "main")
