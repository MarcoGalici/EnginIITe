# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.append(os.path.abspath(".."))


project = 'EnginIITe'
copyright = '2024, Marco Galici, Matteo Troncia, Eliana Carolina Ormeno Mejia, Orlando Mauricio Valarezo Rivera, Jose Pablo Chaves Avila'
author = 'Marco Galici, Matteo Troncia, Eliana Carolina Ormeno Mejia, Orlando Mauricio Valarezo Rivera, Jose Pablo Chaves Avila'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.intersphinx', 'sphinx.ext.mathjax', 'sphinx.ext.autodoc', 'sphinx.ext.autosummary']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
