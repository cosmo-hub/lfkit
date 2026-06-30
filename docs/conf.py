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
    "source_repository": "https://github.com/cosmo-hub/lfkit/",
    "source_branch": "main",
    "source_directory": "docs/",

    "light_logo": "logos/lfkit_logo-icon.png",
    "dark_logo": "logos/lfkit_logo-icon.png",

    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/cosmo-hub/lfkit",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0"
                     viewBox="0 0 16 16">
                     style="color: #FE5019;">
                    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54
                    2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
                    0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49
                    -2.69-.94-.09-.23-.48-.94-.82-1.13
                    -.28-.15-.68-.52-.01-.53.63-.01 1.08.58
                    1.23.82.72 1.21 1.87.87 2.33.66
                    .07-.52.28-.87.51-1.07-1.78-.2-3.64-.89
                    -3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2
                    -.36-1.02.08-2.12 0 0 .67-.21 2.2.82
                    A7.65 7.65 0 0 1 8 3.87c.68 0 1.36.09
                    2 .26 1.53-1.04 2.2-.82 2.2-.82.44
                    1.1.16 1.92.08 2.12.51.56.82 1.27.82
                    2.15 0 3.07-1.87 3.75-3.65 3.95.29
                    .25.54.73.54 1.48 0 1.07-.01 1.93-.01
                    2.2 0 .21.15.46.55.38A8.013 8.013
                    0 0 0 16 8c0-4.42-3.58-8-8-8z">
                    </path>
                </svg>
            """,
            "class": "lfkit-github-icon",
        },
    ],

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
