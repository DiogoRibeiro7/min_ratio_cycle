import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "Min Ratio Cycle"
author = "Diogo Ribeiro"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# Mock optional dependencies that are not installed in the documentation
# build environment.
autodoc_mock_imports = ["matplotlib", "networkx", "numpy", "psutil"]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "alabaster"
html_static_path = ["_static"]
