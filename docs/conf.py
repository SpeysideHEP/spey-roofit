# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path("./ext").resolve()))

# -- Project information -----------------------------------------------------

project = "spey-roofit"
copyright = "2025-2026, Jack Y. Araz"
author = "Jack Y. Araz"
# read the version from the source instead of importing the package, so that the
# documentation can be configured without ``ROOT`` or ``spey`` being importable
release = re.search(
    r'__version__\s*=\s*"([^"]+)"',
    (Path(__file__).parent.parent / "src" / "spey_roofit" / "_version.py").read_text(),
).group(1)
version = ".".join(release.split(".")[:3])
language = "en"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "xref",
    "myst_nb",
]
nb_execution_mode = "off"

# external links, used through :xref:`key` in this documentation and in the docstrings
xref_links = {
    "1007.1727": ("[arXiv:1007.1727]", "https://doi.org/10.48550/arXiv.1007.1727"),
    "2404.06614": ("[arXiv:2404.06614]", "https://doi.org/10.48550/arXiv.2404.06614"),
    "1009.1003": ("[arXiv:1009.1003]", "https://doi.org/10.48550/arXiv.1009.1003"),
    "physics/0306116": (
        "[arXiv:physics/0306116]",
        "https://doi.org/10.48550/arXiv.physics/0306116",
    ),
    "2307.06996": ("[arXiv:2307.06996]", "https://doi.org/10.48550/arXiv.2307.06996"),
    "HEPData": ("HEPData", "https://www.hepdata.net"),
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
}
myst_enable_extensions = ["amsmath", "colon_fence", "deflist", "dollarmath"]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "none"
# the plug-in imports ``ROOT`` lazily, but a documentation build should not need
# ``ROOT`` or ``combine`` to be installed
autodoc_mock_imports = ["ROOT", "HiggsAnalysis"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "_generated"]

master_doc = "index"
man_pages = [(master_doc, "spey-roofit", "spey-roofit Documentation", [author], 1)]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_provider": "github",
    "repository_url": "https://github.com/SpeysideHEP/spey-roofit",
    "use_repository_button": True,
    "use_issues_button": False,
    "path_to_docs": "docs",
    "use_edit_page_button": True,
    "use_fullscreen_button": True,
    "show_navbar_depth": 1,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/SpeysideHEP/spey-roofit",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "arXiv",
            "url": "https://arxiv.org/abs/2307.06996",
            "icon": "https://img.shields.io/static/v1?style=plastic&label=arXiv&message=2307.06996&color=brightgreen",
            "type": "url",
        },
    ],
}

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_logo = "img/spey-roofit.png"
html_favicon = "img/spey-roofit.png"
logo_only = True

htmlhelp_basename = "spey-roofit-doc"

# sphinx-copybutton: do not copy the prompts of the interactive examples
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
copybutton_here_doc_delimiter = "EOF"

latex_documents = [
    (master_doc, "spey-roofit.tex", "spey-roofit Documentation", "Jack Y. Araz", "manual")
]
