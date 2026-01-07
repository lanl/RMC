# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Options for locating modules
#
import sys, os
confpath = os.path.dirname(__file__)
sys.path.append(confpath)
rootpath = os.path.realpath(os.path.join(confpath, "..", ".."))
sys.path.append(rootpath)


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'rmc'
copyright = '2025-2026, dev'
author = 'developers'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinxcontrib.bibtex',
]

templates_path = ['_templates']
exclude_patterns = []

bibtex_bibfiles = ['references.bib']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']


