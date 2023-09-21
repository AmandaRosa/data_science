# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
import data_science

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Data Science"
copyright = "2022, EDGE Analytics Team"
author = "EDGE Analytics Team"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = data_science.__version__[:3]
# The full version, including alpha/beta/rc tags.
release = data_science.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "myst_parser",
    "numpydoc",
]

nbsphinx_execute = "never"

# Generate the API documentation when building
autosummary_generate = True
numpydoc_show_class_members = False
add_module_names = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = "sphinx_book_theme"
html_logo = "_static/petro.png"
html_favicon = "_static/petro.png"
html_title = "Data Science"
html_sidebars = {}
html_theme_options = {
    "repository_url": "https://gitlab.com/edge-analytics/artificial-intelligence/data_science",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
