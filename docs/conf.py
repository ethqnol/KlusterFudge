# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath(".."))  # Source code dir

project = "KlusterFudge"
copyright = "2025, ClusterFudge Team"
author = "ClusterFudge Team"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",  # Core API generation
    "sphinx.ext.napoleon",  # Support for Google/NumPy style docstrings
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "sphinx.ext.githubpages",  # Auto-generate .nojekyll
]

# Mock heavy dependencies that might cause import errors
autodoc_mock_imports = ["numba", "numpy", "pandas", "scikit-learn"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_title = "KlusterFudge v0.1.0"
html_static_path = ["_static"]
html_show_copyright = False
html_show_sphinx = False

# -- Furo Specific Options ---------------------------------------------------
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#58a6ff",
        "color-brand-content": "#58a6ff",
    },
    "dark_css_variables": {
        "color-brand-primary": "#58a6ff",
        "color-brand-content": "#58a6ff",
    },
}
