"""
Path Setup for Grilly Tutorials
================================

Import this module at the top of tutorials to ensure grilly is importable
regardless of where the tutorial is run from.

Usage:
    import _path_setup  # noqa: F401  (must be first import!)
    from grilly import Compute
"""

import sys
from pathlib import Path


def setup_grilly_path():
    """Add grilly package to Python path if not already importable."""
    # Try importing grilly first
    try:
        import grilly
        return  # Already importable
    except ImportError:
        pass

    # Find the grilly package directory
    # This file is at: grilly/tutorials/_path_setup.py
    # We need to add: grilly/ (parent of grilly package) to path

    tutorials_dir = Path(__file__).resolve().parent
    grilly_pkg_dir = tutorials_dir.parent  # grilly/grilly/
    root_dir = grilly_pkg_dir.parent  # grilly/

    # Add root to path (contains grilly package)
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    # Verify it works
    try:
        import grilly
    except ImportError as e:
        print(f"Warning: Could not import grilly even after path setup: {e}")
        print(f"  Tutorials dir: {tutorials_dir}")
        print(f"  Root dir: {root_dir}")
        print(f"  sys.path: {sys.path[:3]}...")


# Run setup on import
setup_grilly_path()
