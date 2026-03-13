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

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["brand.css"]

html_theme_options = {
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/DiogoRibeiro7/min_ratio_cycle/",
    "source_branch": "develop",
    "source_directory": "docs/source/",
    "light_css_variables": {
        "color-brand-primary": "#0f766e",
        "color-brand-content": "#0b7285",
        "color-foreground-primary": "#102a43",
        "color-foreground-secondary": "#334e68",
        "color-background-primary": "#f8fbff",
        "color-background-secondary": "#eef6ff",
        "color-sidebar-background": "#e8f4ff",
        "color-sidebar-item-background--hover": "#d8ebff",
        "color-sidebar-link-text--top-level": "#0b3c5d",
        "color-link": "#0b7285",
        "color-link--hover": "#0f766e",
        "color-admonition-background": "#effaff",
    },
    "dark_css_variables": {
        "color-brand-primary": "#5eead4",
        "color-brand-content": "#67e8f9",
        "color-foreground-primary": "#dbeafe",
        "color-foreground-secondary": "#bfdbfe",
        "color-background-primary": "#0b1220",
        "color-background-secondary": "#111b2e",
        "color-sidebar-background": "#0e172a",
        "color-sidebar-item-background--hover": "#13213a",
        "color-link": "#67e8f9",
        "color-link--hover": "#5eead4",
        "color-admonition-background": "#0f2238",
    },
}
