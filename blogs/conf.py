# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import shutil
import jinja2
import os

from rocm_docs import ROCmDocs

# Environement to process Jinja templates.
jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader("."))

# Jinja templates to render out.
templates = [

]

latex_engine = "xelatex"
latex_elements = {
    "fontpkg": r"""
\usepackage{tgtermes}
\usepackage{tgheros}
\renewcommand\ttdefault{txtt}
"""
}

# configurations for PDF output by Read the Docs
project = "ROCm Blogs"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved."
version = "6.0.0"
release = "6.0.0"
setting_all_article_info = False
all_article_info_os = ["linux", "windows"]
all_article_info_author = ""

exclude_patterns = ['temp']

external_toc_path = "./sphinx/_toc.yml"

docs_core = ROCmDocs("ROCm Blogs")
docs_core.setup()

external_projects_current_project = "rocm"

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)
html_theme_options = {
    "link_main_doc": False
}
