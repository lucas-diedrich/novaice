# Configuration file for the Sphinx documentation builder.

# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "extensions"))
# Add src directory to path so Sphinx can import modules without installation
sys.path.insert(0, str(HERE.parent / "src"))


# -- Project information -----------------------------------------------------

# Try to get metadata from installed package, fallback to reading pyproject.toml
try:
    from importlib.metadata import PackageNotFoundError, metadata

    info = metadata("novaice")
    project = info["Name"]
    author = info["Author"]
    version = info["Version"]
    release = info["Version"]
    urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
    repository_url = urls["Source"]
except PackageNotFoundError:
    # Fallback: read from pyproject.toml if package not installed
    import tomllib

    pyproject_path = HERE.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    project = pyproject["project"]["name"]
    author = ", ".join(a["name"] for a in pyproject["project"]["authors"])
    version = pyproject["project"]["version"]
    release = version
    repository_url = pyproject["project"]["urls"]["Source"]

copyright = f"{datetime.now():%Y}, {author}."

bibtex_bibfiles = ["references.bib"]
templates_path = ["_templates"]
nitpicky = True  # Warn about broken links
needs_sphinx = "4.0"

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "lucas-diedrich",
    "github_repo": project,
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
# They can be extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "sphinx_tabs.tabs",
    "sphinx.ext.mathjax",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinxext.opengraph",
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
]

autosummary_generate = True
autodoc_member_order = "groupwise"
default_role = "literal"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
myst_heading_anchors = 6  # create anchors for h1-h6
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_url_schemes = ("http", "https", "mailto")
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True
typehints_defaults = "braces"

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

intersphinx_mapping = {
    # TODO: replace `3.13` with `3` once ReadTheDocs supports building with Python 3.14
    "python": ("https://docs.python.org/3.13", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_title = project

html_theme_options = {
    "repository_url": repository_url,
    "use_repository_button": True,
    "path_to_docs": "docs/",
    "navigation_with_keys": False,
}

pygments_style = "default"

nitpick_ignore = [
    # If building the documentation fails because of a missing link that is outside your control,
    # you can add an exception to this list.
    #     ("py:class", "igraph.Graph"),
]
