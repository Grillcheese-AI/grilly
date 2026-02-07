"""Sphinx configuration for Read the Docs."""

from __future__ import annotations

from pathlib import Path
import importlib.metadata
import importlib.machinery
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
    module.__package__ = "grilly"
    module.__spec__ = importlib.machinery.ModuleSpec("grilly", loader=None, is_package=True)
    sys.modules["grilly"] = module


def _register_pkg_stub(name: str, path: Path) -> None:
    if name in sys.modules or not path.is_dir():
        return
    pkg = types.ModuleType(name)
    pkg.__file__ = str(path / "__init__.py")
    pkg.__path__ = [str(path)]
    pkg.__package__ = name
    pkg.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    sys.modules[name] = pkg


# Ensure namespace/package discovery works even when an installed wheel is
# incomplete or shadowed during docs builds.
_register_pkg_stub("grilly.datasets", PROJECT_ROOT / "datasets")


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
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "api/modules.rst",
    "api/grilly*.rst",
    "api/generated/grilly.rst",
    "api/generated/grilly.test_quick.rst",
    "api/generated/grilly.tests*.rst",
    "api/generated/grilly.tutorials*.rst",
]

source_suffix = ".rst"
master_doc = "index"


html_theme = "sphinx_rtd_theme"
html_title = f"{project} {version}"
html_static_path = []
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": True,
    "sticky_navigation": True,
    "titles_only": False,
}


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
    "inherited-members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autoclass_content = "both"
autodoc_typehints = "description"
autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = True

napoleon_google_docstring = True
napoleon_numpy_docstring = True


# Required by sphinx-notfound-page for RTD links.
notfound_urls_prefix = "/en/latest/"


def _sanitize_known_docstrings(app, what, name, obj, options, lines):
    """Normalize a few legacy docstrings that still trip docutils parsing."""
    if name.endswith("LSTMCell.forward"):
        lines[:] = ["Compute one LSTM step and return ``(h_new, c_new)``."]
    elif name.endswith("nn.module.Module") or name.endswith("grilly.nn.Module"):
        lines[:] = ["Base class for Grilly neural network modules."]


def setup(app):
    app.connect("autodoc-process-docstring", _sanitize_known_docstrings, priority=900)
