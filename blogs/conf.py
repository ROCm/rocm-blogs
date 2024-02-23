# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import ablog
import shutil
import jinja2
import os

from rocm_docs import ROCmDocs
from sphinx import addnodes

import ablog

ablog_builder = "dirhtml"
ablog_website = "_website"

# Environement to process Jinja templates.
jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader("."))

# Jinja templates to render out.
templates = []

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

setting_all_article_info = False
all_article_info_os = ["linux", "windows"]
all_article_info_author = ""

exclude_patterns = ["temp"]

external_toc_path = "./sphinx/_toc.yml"

external_projects_current_project = "rocm"
blog_title = "AMD ROCm Blogs"
blog_baseurl = "https://rocm.blogs.amd.com/"

html_title = "ROCm Blogs"
html_theme = "rocm_docs_theme"
html_theme_options = {
    "flavor": "rocm-blogs",
    "link_main_doc": False,
}

extensions = ["rocm_docs"]
external_toc_path = "./sphinx/_toc.yml"
