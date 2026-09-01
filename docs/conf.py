"""Sphinx configuration for the PolyPY-MPC documentation."""

from datetime import date

# -- Project information ------------------------------------------------------

project = "PolyPY-MPC"
author = "Fabio Faliero and contributors"
copyright = f"{date.today().year}, {author}"

try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("polypympc")
except Exception:  # package not installed (e.g. first bootstrap)
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Autodoc / Napoleon ----------------------------------------------------

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
# Heavy / compiled dependencies are mocked so the API can be documented
# without a full scientific stack (and without building pycvxset from source).
autodoc_mock_imports = [
    "casadi",
    "cvxpy",
    "pycvxset",
    "pycddlib",
    "mosek",
    "scs",
    "osqp",
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

# -- MyST ------------------------------------------------------------------

myst_enable_extensions = ["dollarmath", "colon_fence", "deflist"]
myst_heading_anchors = 3

# -- Intersphinx ---------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- sphinx-gallery ----------------------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["auto_examples"],
    "filename_pattern": r".*_example\.py",
    # The turtlebot4 case study needs recorded data / CLI args; it is shown via
    # literalinclude in the user guide instead of the gallery.
    "ignore_pattern": r"__init__\.py|[\\/]turtlebot4[\\/]",
    # Render the example sources without executing them. The examples need a
    # full solver stack (and some need MOSEK / recorded data), so executing
    # them in CI is deferred. Flip to "True" once they are CI-safe.
    "plot_gallery": "False",
    "download_all_examples": False,
    "remove_config_comments": True,
}

# -- HTML output ---------------------------------------------------------

html_theme = "furo"
html_title = f"PolyPY-MPC {version}"
html_static_path = ["_static"]
# Do not publish the raw page sources (_sources/) alongside the HTML.
html_copy_source = False
html_show_sourcelink = False
