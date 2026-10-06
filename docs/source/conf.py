"""moderndid sphinx configuration."""

import math
import os
import sys
from datetime import date
from importlib.metadata import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

# Executed pages load the stored results in prerun, and the notebook kernels inherit this path.
os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(Path(__file__).parent), os.environ.get("PYTHONPATH")]))

# -- Project information

_metadata = metadata("moderndid")

project = _metadata["Name"]
author = _metadata["Author-email"].split("<", 1)[0].strip()
copyright = f"{date.today().year}, {author}"

version = _metadata["Version"]
release = version


# -- General configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.napoleon",
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_design",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_immaterial",
    "last_updated",
    "semantic_highlighting",
]

templates_path = ["_templates"]

exclude_patterns = [
    "Thumbs.db",
    ".DS_Store",
    ".ipynb_checkpoints",
    "_static/**",
    "_templates/**",
    "_ext/**",
]

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "autolink"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

rst_prolog = r"""
.. |l_vec| replace:: :math:`\ell_{vec}`
"""


# -- Options for extensions

extlinks = {
    "issue": ("https://github.com/jordandeklerk/moderndid/issues/%s", "GH#%s"),
    "pull": ("https://github.com/jordandeklerk/moderndid/pull/%s", "PR#%s"),
}

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
}

autosummary_generate = True
autodoc_mock_imports = ["pyfixest"]
autodoc_typehints = "signature"
autodoc_default_options = {
    "members": False,
    "undoc-members": False,
    "show-inheritance": False,
}

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_ivar = True
napoleon_use_admonition_for_examples = True

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "polars": ("https://docs.pola.rs/api/python/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "plotnine": ("https://plotnine.org/", None),
}

# -- Options for HTML output

html_theme = "sphinx_immaterial"

html_logo = "_static/logo-wordmark.svg"
html_favicon = "_static/logo-mark.svg"

html_theme_options = {
    "font": {"text": "PT Sans", "code": "Fira Mono"},
    "repo_url": "https://github.com/jordandeklerk/moderndid",
    "repo_name": "ModernDiD",
    "icon": {"repo": "fontawesome/brands/git-alt"},
    "features": [
        "header.autohide",
        "navigation.instant",
        "navigation.tabs",
        "navigation.tabs.sticky",
        "navigation.path",
        "navigation.top",
        "navigation.footer",
        "navigation.tracking",
        "announce.dismiss",
        "search.highlight",
        "search.share",
        "toc.follow",
    ],
    "toc_title": "On this page",
    "globaltoc_collapse": False,
    "palette": [
        {
            "media": "(prefers-color-scheme)",
            "toggle": {"icon": "material/brightness-auto", "name": "Switch to light mode"},
        },
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "white",
            "accent": "blue",
            "toggle": {"icon": "material/weather-sunny", "name": "Switch to dark mode"},
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "black",
            "accent": "blue",
            "toggle": {"icon": "material/weather-night", "name": "Switch to system preference"},
        },
    ],
}

html_title = "ModernDiD"
html_static_path = ["_static"]

html_css_files = ["css/custom.css", "css/landing.css"]
html_js_files = [
    ("js/copybutton-shim.js", {"priority": 200}),
    "js/header-title-link.js",
    "js/toc-rail.js",
]
html_use_modindex = True
html_copy_source = False
html_domain_indices = False
html_file_suffix = ".html"

htmlhelp_basename = "moderndid"

sphinx_immaterial_custom_admonitions = [
    {"name": "example", "override": True, "icon": "material/code-braces", "color": (49, 91, 196)},
    {"name": "important", "override": True, "icon": "material/alert-decagram", "color": (124, 77, 255)},
    {"name": "assumption", "icon": "material/format-list-checks", "color": (201, 63, 117)},
    {"name": "theorem", "icon": "material/equal-box", "color": (0, 200, 83)},
]

myst_enable_extensions = ["linkify", "colon_fence", "dollarmath"]
myst_heading_anchors = 3

# Example pages run during the build, so an example that stops working fails it. The cache reruns a page
# only when its cells change.
nb_execution_mode = "cache"
nb_execution_raise_on_error = True
nb_execution_timeout = 600
nb_output_stderr = "remove"
# Draw each cell's prints in one box. The kernel sends them in chunks that vary with timing.
nb_merge_streams = True
# A hide-input cell folds its code behind one line.
nb_code_prompt_show = "Show code"
nb_code_prompt_hide = "Hide code"

plot_pre_code = """
import numpy as np
np.random.seed(123)
"""

plot_include_source = True
plot_formats = [("png", 96)]
plot_html_show_formats = False
plot_html_show_source_link = False

phi = (math.sqrt(5) + 1) / 2

font_size = 13 * 72 / 96.0  # 13 px

plot_rcparams = {
    "font.size": font_size,
    "axes.titlesize": font_size,
    "axes.labelsize": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "legend.fontsize": font_size,
    "figure.figsize": (3 * phi, 3),
    "figure.subplot.bottom": 0.2,
    "figure.subplot.left": 0.2,
    "figure.subplot.right": 0.9,
    "figure.subplot.top": 0.85,
    "figure.subplot.wspace": 0.4,
    "text.usetex": False,
}


def _landing_template(app, pagename, templatename, context, doctree):
    """Render the documentation home with the landing page template."""
    if pagename == app.config.root_doc:
        return "landing.html"
    return None


def _sidebar_sections(app, pagename, templatename, context, doctree):
    """Display User Guide and API sections as sidebar headings over their pages."""
    # The theme turns only the sidebar's second level into headings. Since the API index nests its
    # pages under its own section headings, each page's generated entries stay folded.
    guide_page = pagename.startswith("user_guide/") and not pagename.startswith("user_guide/example_")
    if guide_page or pagename.startswith("api/"):
        theme = context["config"]["theme"]
        features = [*theme["features"], "navigation.sections"]
        context["config"] = {**context["config"], "theme": {**theme, "features": features}}
        _link_parents(context["nav"])


def _link_parents(entries, parent=None):
    """Reconnect copied navigation entries to their parents on the current page."""
    for entry in entries:
        entry.parent = parent
        _link_parents(entry.children, entry)


def _open_examples_boxes(app, doctree):
    """Render API examples as open, collapsible example boxes."""
    from docutils import nodes

    for node in doctree.findall(nodes.admonition):
        if "example" in node["classes"] and node.get("collapsible") is None:
            node["collapsible"] = "open"


def _panel_diagnostics_signature(app, what, name, obj, options, signature, return_annotation):
    """Render the dataclass factory as valid syntax so Sphinx can identify its parameters."""
    if what == "class" and name == "moderndid.core.panel.PanelDiagnostics" and signature:
        # inspect renders a dataclass factory as <factory>. Sphinx then treats each
        # annotated parameter declaration as a single, unmatchable name.
        return signature.replace("= <factory>", "= field(default_factory=list)"), return_annotation
    return None


def setup(app):
    """Register the landing page, grouped guide and API navigation, and API example boxes."""
    app.connect("html-page-context", _landing_template)
    app.connect("html-page-context", _sidebar_sections, priority=600)
    app.connect("doctree-read", _open_examples_boxes)
    app.connect("autodoc-process-signature", _panel_diagnostics_signature)
