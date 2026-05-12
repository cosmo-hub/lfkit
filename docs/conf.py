# Configuration file for the Sphinx documentation builder.

from __future__ import annotations

import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")

# -----------------------------------------------------------------------------
# Ensure src/ layout imports work
# -----------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath("../src"))

warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module=r"colorspacious\.comparison",
)

# -----------------------------------------------------------------------------
# Project information
# -----------------------------------------------------------------------------
project = "LFKit"
author = "Nikolina Šarčević"
copyright = "2026, Nikolina Šarčević"

# Remove default "documentation" suffix in browser title
html_title = "LFKit Documentation"

# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_design",
    "sphinx_multiversion",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

#templates_path = ["_templates"]  # if uncomment this removes the sidebar logo
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -----------------------------------------------------------------------------
# Autodoc / autosummary
# -----------------------------------------------------------------------------
autosummary_generate = True
autodoc_typehints = "description"

napoleon_google_docstring = True
napoleon_numpy_docstring = False


# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------
html_baseurl = "https://cosmology-kit.github.io/lfkit/"

html_theme = "furo"
html_permalinks_icon = "<span>#</span>"

html_static_path = ["_static"]

html_theme_options = {
    "light_logo": "logos/lfkit_logo-icon.png",
    "dark_logo": "logos/lfkit_logo-icon.png",

    "light_css_variables": {
        "color-brand-primary": "#28A8C8",
        "color-brand-content": "#28A8C8",
        "color-link": "#28A8C8",
        "color-link--hover": "#FE5019",
        "color-link--visited": "#28A8C8",
    },
    "dark_css_variables": {
        "color-brand-primary": "#28A8C8",
        "color-brand-content": "#28A8C8",
        "color-link": "#28A8C8",
        "color-link--hover": "#FE5019",
        "color-link--visited": "#28A8C8",
    },
}

# (optional) favicon
html_favicon = "_static/logos/lfkit_logo-icon.png"

# Custom styling
html_css_files = [
    "custom.css",
]

# -----------------------------------------------------------------------------
# Matplotlib plot directive
# -----------------------------------------------------------------------------
plot_html_show_source_link = False

plot_formats = [("png", 300)]

plot_rcparams = {
    "figure.dpi": 150,
    "savefig.dpi": 150,
}

# -----------------------------------------------------------------------------
# Sphinx multiversion
# -----------------------------------------------------------------------------
smv_tag_whitelist = r"^v\d+\.\d+\.\d+$"
smv_branch_whitelist = r"^main$"
smv_remote_whitelist = r"^origin$"
smv_released_pattern = r"^refs/tags/v\d+\.\d+\.\d+$"
smv_outputdir_format = "{ref.name}"
smv_site_url = "https://cosmology-kit.github.io/lfkit/"

# -----------------------------------------------------------------------------
# Copybutton configuration
# -----------------------------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
copybutton_copy_empty_lines = False
