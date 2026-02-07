"""Sphinx configuration for Read the Docs."""

from __future__ import annotations

from pathlib import Path
import importlib.metadata
import sys


DOCS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS_DIR.parent

# Ensure local package modules are importable by Sphinx/autodoc.
sys.path.insert(0, str(PROJECT_ROOT))


project = "grilly"
author = "Nicolas Cloutier"

try:
    release = importlib.metadata.version("grilly")
except importlib.metadata.PackageNotFoundError:
    release = "0.3.0"
version = release


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "notfound.extension",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = ".rst"
master_doc = "index"


html_theme = "sphinx_rtd_theme"
html_title = f"{project} {version}"
html_static_path = []


# Prevent RTD docs builds from failing when optional runtime deps are absent.
autodoc_mock_imports = [
    "vulkan",
    "torch",
    "transformers",
    "spacy",
    "sentence_transformers",
    "numba",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True


# Required by sphinx-notfound-page for RTD links.
notfound_urls_prefix = "/en/latest/"
