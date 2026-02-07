"""Sphinx configuration for Read the Docs."""

from __future__ import annotations

from pathlib import Path
import importlib.metadata
import sys
import types


DOCS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS_DIR.parent
PROJECT_PARENT = PROJECT_ROOT.parent
EXT_DIR = DOCS_DIR / "_ext"

# Ensure local package modules are importable by Sphinx/autodoc.
# This repository uses a flat package layout where `__init__.py` lives at
# project root, so we add both parent and root paths for portability.
sys.path.insert(0, str(PROJECT_PARENT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXT_DIR))

# If the package cannot be imported by name yet (common on RTD with flat
# layouts), register a lightweight package stub without executing __init__.py.
if "grilly" not in sys.modules:
    module = types.ModuleType("grilly")
    module.__file__ = str(PROJECT_ROOT / "__init__.py")
    module.__path__ = [str(PROJECT_ROOT)]
    sys.modules["grilly"] = module


project = "grilly"
author = "Nicolas Cloutier"

try:
    release = importlib.metadata.version("grilly")
except importlib.metadata.PackageNotFoundError:
    release = "0.3.0"
version = release


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "notfound.extension",
    "api_doc_enhancer",
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
    "numpy",
    "torch",
    "transformers",
    "spacy",
    "sentence_transformers",
    "numba",
    "vulkan",
    "pyvma",
    "scipy",
    "matplotlib",
    "blake3",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
    "special-members": "__init__,__call__",
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = True


# Required by sphinx-notfound-page for RTD links.
notfound_urls_prefix = "/en/latest/"
