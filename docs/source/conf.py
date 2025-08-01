# pylint: disable=redefined-builtin,invalid-name
"""causaldid sphinx configuration."""

import math
import os
from importlib.metadata import metadata

# -- Project information

_metadata = metadata("causaldid")

project = _metadata["Name"]
author = _metadata["Author-email"].split("<", 1)[0].strip()
copyright = f"2025, {author}"

version = _metadata["Version"]
if os.environ.get("READTHEDOCS", False):
    rtd_version = os.environ.get("READTHEDOCS_VERSION", "")
    if "." not in rtd_version and rtd_version.lower() != "stable":
        version = "dev"
else:
    branch_name = os.environ.get("BUILD_SOURCEBRANCHNAME", "")
    if branch_name == "main":
        version = "dev"
release = version


# -- General configuration

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "numpydoc",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
]

templates_path = ["_templates"]

exclude_patterns = [
    "Thumbs.db",
    ".DS_Store",
    ".ipynb_checkpoints",
]

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "autolink"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# Ensure all our internal links work
nitpicky = True
nitpick_ignore = [
    # Common type annotation issues
    ("py:class", "ndarray"),
    ("py:class", "array_like"),
    ("py:class", "optional"),
    # Ignore custom result objects that aren't part of the public API
    ("py:obj", "MPResult"),
    ("py:obj", "AGGTEResult"),
    ("py:obj", "APRCIResult"),
    ("py:obj", "ARPNuisanceCIResult"),
    ("py:obj", "FLCIResult"),
    ("py:obj", "DeltaRMResult"),
    ("py:obj", "DeltaRMBResult"),
    ("py:obj", "DeltaRMMResult"),
    ("py:obj", "DeltaSDResult"),
    ("py:obj", "DeltaSDBResult"),
    ("py:obj", "DeltaSDMResult"),
    ("py:obj", "DeltaSDRMResult"),
    ("py:obj", "DeltaSDRMBResult"),
    ("py:obj", "DeltaSDRMMResult"),
    ("py:obj", "OriginalCSResult"),
    ("py:obj", "DRDIDResult"),
    ("py:obj", "DRDIDLocalRCResult"),
    ("py:obj", "DRDIDPanelResult"),
    ("py:obj", "DRDIDRCResult"),
    ("py:obj", "DRDIDTradRCResult"),
    ("py:obj", "HonestDiDResult"),
    ("py:obj", "IPWDIDPanelResult"),
    ("py:obj", "IPWDIDRCResult"),
    ("py:obj", "IPWDIDResult"),
    ("py:obj", "ORDIDResult"),
    ("py:obj", "RegDIDPanelResult"),
    ("py:obj", "RegDIDRCResult"),
    ("py:obj", "StdIPWDIDPanelResult"),
    ("py:obj", "StdIPWDIDRCResult"),
    ("py:obj", "TWFEDIDPanelResult"),
    ("py:obj", "TWFEDIDRCResult"),
    ("py:obj", "WOLSResult"),
    # Ignore miscellaneous words that are misinterpreted
    ("py:obj", "instance"),
    ("py:obj", "from"),
    ("py:obj", "parallel"),
    ("py:obj", "similar"),
    ("py:obj", "Additional"),
    ("py:obj", "parameters"),
    ("py:obj", "boot_drdid_rc"),
    ("py:obj", "wboot_drdid_rc_imp2"),
    ("py:obj", "wboot_aipw_rc"),
]

# -- Options for extensions

extlinks = {
    "issue": ("https://github.com/jordandeklerk/causaldid/issues/%s", "GH#%s"),
    "pull": ("https://github.com/jordandeklerk/causaldid/pull/%s", "PR#%s"),
}

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# File extensions
source_suffix = [".rst", ".md"]

autosummary_generate = True
autodoc_typehints = "none"
autodoc_default_options = {
    "members": False,
}

numpydoc_show_class_members = False
numpydoc_xref_param_type = True
numpydoc_xref_ignore = {"of", "or", "optional", "scalar", "default"}
singulars = ("int", "list", "dict", "float")
numpydoc_xref_aliases = {
    "ndarray": ":class:`numpy.ndarray`",
    "DataFrame": ":class:`pandas.DataFrame`",
    "Series": ":class:`pandas.Series`",
    "pd.DataFrame": ":class:`pandas.DataFrame`",
    "pd.Series": ":class:`pandas.Series`",
    "np.ndarray": ":class:`numpy.ndarray`",
    "np.random.Generator": ":class:`numpy.random.Generator`",
    "matplotlib.figure.Figure": ":class:`matplotlib.figure.Figure`",
    **{f"{singular}s": f":any:`{singular}s <{singular}>`" for singular in singulars},
}

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- Options for HTML output

html_theme = "pydata_sphinx_theme"

html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.ico"

html_sidebars = {"**": ["sidebar-nav-bs"]}

html_theme_options = {
    "header_links_before_dropdown": 6,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/jordandeklerk/causaldid",
            "icon": "fa-brands fa-github",
        },
    ],
    "logo": {
        "text": "causaldid",
        "image_light": "_static/logo.svg",
        "image_dark": "_static/logo.svg",
    },
    "collapse_navigation": True,
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["search-button", "theme-switcher", "navbar-icon-links"],
    "navbar_persistent": [],
    "secondary_sidebar_items": ["page-toc"],
    "show_version_warning_banner": False,
}

html_title = f"{project} v{version} Manual"
html_static_path = ["_static"]
html_last_updated_fmt = "%b %d, %Y"

html_css_files = [
    "custom.css",
]
html_context = {"default_mode": "light"}
html_use_modindex = True
html_copy_source = False
html_domain_indices = False
html_file_suffix = ".html"

htmlhelp_basename = "causaldid"

# -----------------------------------------------------------------------------
# Matplotlib plot_directive options
# -----------------------------------------------------------------------------

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
