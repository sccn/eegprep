# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

# Add the source directory to the path so we can import eegprep
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# -- Project information -------------------------------------------------------
project = "eegprep"
copyright = "2024, eegprep contributors"
author = "eegprep contributors"

# Import version from eegprep package
try:
    import eegprep
    version = eegprep.__version__
    release = version
except ImportError as e:
    # Handle import errors gracefully during documentation build
    print(f"Warning: Could not import eegprep: {e}")
    version = "0.2.23"
    release = version

# -- General configuration -------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_gallery.gen_gallery",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_togglebutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_theme_options = {
    # Logo configuration
    "logo": {
        "text": "eegprep",
    },
    # Navigation structure
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],
    "navbar_persistent": [],
    "primary_sidebar_end": ["sidebar-ethical-ads"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "secondary_sidebar_items": ["page-toc"],
    "header_links_before_dropdown": 4,
    # Social and repository links
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/sccn/eegprep",
            "icon": "fab fa-github-square",
        },
    ],
    # Navigation display
    "show_nav_level": 2,
    "use_edit_page_button": False,
    # Search settings
    "search_bar_text": "Search documentation...",
    # Sidebar behavior
    "collapse_navigation": False,
}

html_context = {
    "github_user": "NeuroTechX",
    "github_repo": "eegprep",
    "github_version": "main",
    "doc_path": "docs/source",
}

# -- Options for autodoc -------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": False,
    "show-inheritance": True,
}

autodoc_typehints = "description"
autodoc_typehints_format = "short"

# -- Options for Napoleon (Google-style docstrings) ---------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for intersphinx ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "mne": ("https://mne.tools/stable", None),
}

# -- Options for sphinx-gallery ------------------------------------------------
# https://sphinx-gallery.github.io/stable/configuration.html

sphinx_gallery_conf = {
    # Directory where example scripts are located
    "examples_dirs": "examples",
    # Directory where gallery will be generated
    "gallery_dirs": "auto_examples",
    # Pattern for example filenames
    "filename_pattern": "/plot_",
    # Pattern for files to ignore
    "ignore_pattern": r"__init__\.py",
    # Whether to execute examples
    "plot_gallery": True,
    # Whether to download all examples
    "download_all_examples": False,
    # Abort build on example error
    "abort_on_example_error": False,
    # Image srcset configuration
    "image_srcset": [],
    # Default thumbnail file
    "default_thumb_file": None,
    # Show line numbers in code blocks
    "line_numbers": False,
    # Remove config comments from code blocks
    "remove_config_comments": False,
    # Expected failing examples
    "expected_failing_examples": set(),
    # Passing examples
    "passing_examples": [],
    # Stale examples
    "stale_examples": [],
    # Run stale examples
    "run_stale_examples": False,
    # Backreferences directory
    "backreferences_dir": None,
}

# -- Options for MyST parser ---------------------------------------------------
# https://myst-parser.readthedocs.io/en/latest/configuration.html

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

myst_heading_anchors = 2

# -- Options for sphinx_copybutton -----------------------------------------------
# https://sphinx-copybutton.readthedocs.io/

copybutton_exclude = ".linenos, .gp, .go"

# -- Options for sphinx_togglebutton -----------------------------------------------
# https://sphinx-togglebutton.readthedocs.io/

# No specific configuration needed for togglebutton

# -- Additional settings -------------------------------------------------------

# Suppress warnings for missing references
suppress_warnings = ["ref.python"]

# Source file suffix
source_suffix = {
    ".rst": None,
    ".md": "markdown",
}

# Master document
master_doc = "index"

# Language for content autogenerated by Sphinx
language = "en"

# Pygments style
pygments_style = "sphinx"

# HTML output options
html_use_smartquotes = True
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

# Additional CSS
html_css_files = [
    "custom.css",
]

# Additional JavaScript
html_js_files = []

# -- Analytics Configuration --------------------------------------------------
# Optional: Configure analytics (e.g., Google Analytics)
# Uncomment and configure if needed:
# html_js_files = [
#     ('https://www.googletagmanager.com/gtag/js?id=YOUR_GA_ID', {'async': 'async'}),
# ]

# -- Search Configuration --------------------------------------------------
# Enable full-text search
html_search_language = "en"
html_search_options = {
    "type": "default",
}

