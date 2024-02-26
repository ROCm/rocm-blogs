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

extensions = ["rocm_docs", "ablog", "sphinx.ext.intersphinx"]
external_toc_path = "./sphinx/_toc.yml"

templates_path = [ablog.get_html_templates_path()]

html_sidebars = {
    "**": [
        "postcard.html",
        "recentposts.html",
        "tagcloud.html",        
        "categories.html",
        "archives.html",
    ]
}

blog_authors = {
    'Justin Chang': ('Justin Chang', 'http://rocm.blogs.amd.com/authors/justin-chang.html'),
    'Rene Van Oostrum': ('Rene Van Oostrum',
               'https://rocm.blogs.amd.com/authors/rene-van-oostrum.html'),
}
blog_feed_length = 10
blog_feed_archives = True
blog_feed_fulltext = True
blog_feed_templates = {
    "atom": {
        "content": "{{ title }}{% for tag in post.tags %}" " #{{ tag.name|trim()|replace(' ', '') }}" "{% endfor %}",
    },
    "social": {
        "content": "{{ title }}{% for tag in post.tags %}" " #{{ tag.name|trim()|replace(' ', '') }}" "{% endfor %}",
    },
}

nitpicky = True
nitpick_ignore = []
for line in open("nitpick-exceptions"):
    if line.strip() == "" or line.startswith("#"):
        continue
    dtype, target = line.split(None, 1)
    target = target.strip()
    nitpick_ignore.append((dtype, target))


def parse_event(env, sig, signode):
    event_sig_re = re.compile(r"([a-zA-Z-]+)\s*\((.*)\)")
    m = event_sig_re.match(sig)
    if not m:
        signode += addnodes.desc_name(sig, sig)
        return sig
    name, args = m.groups()
    signode += addnodes.desc_name(name, name)
    plist = addnodes.desc_parameterlist()
    for arg in args.split(","):
        arg = arg.strip()
        plist += addnodes.desc_parameter(arg, arg)
    signode += plist
    return name


def setup(app):
    from sphinx.ext.autodoc import cut_lines
    from sphinx.util.docfields import GroupedField

    app.connect("autodoc-process-docstring", cut_lines(4, what=["module"]))
    app.add_object_type(
        "confval",
        "confval",
        objname="configuration value",
        indextemplate="pair: %s; configuration value",
    )
    fdesc = GroupedField("parameter", label="Parameters", names=["param"], can_collapse=True)
    app.add_object_type("event", "event", "pair: %s; event", parse_event, doc_field_types=[fdesc])
