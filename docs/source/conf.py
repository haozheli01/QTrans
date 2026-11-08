# -*- coding: utf-8 -*-
# Configuration file for the QTrans documentation builder.

import sys
import os
import sphinx_rtd_theme
import sphinxcontrib

# -- Project information -----------------------------------------------------
project = 'QTrans'
copyright = '2026, Yang\'s Group'
author = 'Shixu Liu, Haozhe Li'
release = '0.1'

# -- General configuration ---------------------------------------------------
sys.path.insert(0, os.path.abspath('./_ext'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinxcontrib.bibtex',
    'sphinx.ext.mathjax',
    'sphinx_design',
    'param_card',
]
source_suffix = '.rst'
templates_path = ['_templates']
exclude_patterns = []
master_doc = 'index'
bibtex_bibfiles = ['refs.bib']


# -- Options for HTML output -------------------------------------------------

html_css_files = [
    'custom.css',
    'style.css'
]
html_static_path = ['_static']
html_theme = "sphinx_rtd_theme"
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_logo = "./_static/logo.ico"
html_favicon = "./_static/html_mini.ico"

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'QTransdoc'


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'QTrans.tex', 'QTrans Documentation',
     author.split(','), 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'QTrans Documentation', 'QTrans Documentation',
     author.split(','), 1)
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'QTrans Documentation', 'QTrans Documentation',
     author.split(','), 'QTrans', 'QTrans for transport properties',
     'Programming'),
]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']

