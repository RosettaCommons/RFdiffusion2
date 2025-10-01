# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'RFdiffusion2'
copyright = '2025, Institute for Protein Design, University of Washington'
author = 'Woody Ahern, Jason Yim, Doug Tischer, Saman Salike, Seth M. Woodbury, Donghyo Kim, Indrek Kalvet, Yakov Kipnis, Brian Coventry, Han Raut Altae-Tran, Magnus Bauer, Regina Barzilay, Tommi S. Jaakkola, Rohith Krishna, David Baker'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_mdinclude',
    #'myst_parser', # to use markdown instead of ReST
    'sphinx_copybutton',
]

#myst_enable_extensions = ["colon_fence"] # see https://mystmd.org/guide/syntax-overview for more information

suppress_warnings = [
    'app.add_node',
    'app.add_directive',
    'app.add_role',
    'app.add_generic_role',
    'app.add_source_parser',
    'config.cache',
    'download.not_readable',
    'epub.unknown_project_files',
    'epub.duplicated_toc_entry',
    'i18n.inconsistent_references',
    'index',
    'image.not_readable',
    'ref.term',
    'ref.ref',
    'ref.numref',
    'ref.keyword',
    'ref.option',
    'ref.citation',
    'ref.footnote',
    'ref.doc',
    'ref.python',
    'misc.highlighting_failure',
    'toc.circular',
    'toc.excluded',
    'toc.not_readable',
    'toc.secnum',
    # autodoc,
    # autodoc.import_object,
    # autosummary,
    # intersphinx.external,
]

napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']


html_theme_options = {
    "sidebar_hide_name":False,
    "top_of_page_buttons": ["edit"],
    ""
    #"announcement": "<em>THIS DOCUMENTATION IS CURRENTLY UNDER CONSTRUCTION</em>",
    "light_css_variables": {
        "color-brand-primary": "#F68A33", # Rosetta Teal
        "color-brand-content": "#37939B", # Rosetta Orange
        "color-admonition-background": "#FB35D6", # Rosetta light orange
        "font-stack": "Open Sans, sans-serif",
        "font-stack--headings": "Open Sans, sans-serif",
        "color-background-hover": "#DCE8E8ff",
        "color-announcement-background" : "#F68A33dd",
        "color-announcement-text": "#070707",
        "color-brand-visited": "#37939B",
        },
    "dark_css_variables": {
        "color-brand-primary": "#37939B", # Rosetta teal
        "color-brand-content": "#F68A33", # Rosetta orange
        "color-admonition-background": "#FB35D6", # Rosetta light orange
        "font-stack": "Open Sans, sans-serif",
        "font-stack--headings": "Open Sans, sans-serif",
        "color-brand-visited": "#37939B",
        }
    }
