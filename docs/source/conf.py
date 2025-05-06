# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('./'))
from helper import *
currpath = os.path.dirname(__file__)
import bnlearn

########################################################################################
# -- Download rst file -----------------------------------------------------
download_file('https://erdogant.github.io/docs/rst/sponsor.rst', "sponsor.rst")
download_file('https://erdogant.github.io/docs/rst/add_carbon.add', "add_carbon.add")
download_file('https://erdogant.github.io/docs/rst/add_top.add', "add_top.add")
download_file('https://erdogant.github.io/docs/rst/add_bottom.add', "add_bottom.add")
########################################################################################
add_includes_to_rst_files(top=False, bottom=True)
########################################################################################
# Import PDF from directory in rst files
# embed_in_rst(currpath, 'pdf', '.pdf', "Additional Information", 'Additional_Information.rst')
########################################################################################
# Import notebooks in HTML format
# convert_ipynb_to_html(currpath, 'notebooks', ext='.ipynb')
# embed_in_rst(currpath, 'notebooks', '.html', "Notebook", 'notebook.rst')
########################################################################################


# -- Project information -----------------------------------------------------

project = 'bnlearn'
copyright = '2022, Erdogan Taskesen'
author = 'Erdogan Taskesen'

# The master toctree document.
master_doc = 'index'

# The full version, including alpha/beta/rc tags
release = 'bnlearn'

# -- General configuration ---------------------------------------------------

# Mock importing packages to avoid complicated dependencies that needs to be installed
# autodoc_mock_imports = ['pgmpy', 'networkx', 'sklearn', 'statsmodels']

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
	"sphinx.ext.intersphinx",
	"sphinx.ext.autosectionlabel",
    # "rst2pdf.pdfbuilder",
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
# pdf_documents = [('index', u'rst2pdf', u'Sample rst2pdf doc', u'Erdogan Taskesen'),]

autodoc_mock_imports = ['torch']

#extensions = [
#    "sphinx.ext.autodoc",
#    "sphinx.ext.coverage",
#    "sphinx.ext.mathjax",
#    "sphinx.ext.autosectionlabel",
#    "sphinx.ext.napoleon",
#]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build"]


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
#html_theme = 'default'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = ['css/custom.css',]

# A list of files that should not be packed into the epub file
epub_exclude_files = ['search.html']

# html_sidebars = { '**': ['globaltoc.html', 'relations.html', 'carbon_ads.html', 'sourcelink.html', 'searchbox.html'] }
