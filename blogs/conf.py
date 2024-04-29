# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import shutil

import ablog
import jinja2
from rocm_docs import ROCmDocs
from sphinx.ext.autodoc import cut_lines
from sphinx.util.docfields import GroupedField

from sphinx import addnodes

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
}

extensions = ["rocm_docs", "ablog", "sphinx.ext.intersphinx"]
external_toc_path = "./sphinx/_toc.yml"

templates_path = [ablog.get_html_templates_path(), "."]

html_sidebars = {
    "**": [
        "postcard.html",
        "./templates/recentposts.html",
        "tagcloud.html",
        "categories.html",
        "archives.html",
    ]
}

blog_authors = {
    "alessandro fanfarillo": (
        "alessandro fanfarillo",
        "http://rocm.blogs.amd.com/authors/alessandro-fanfarillo.html",
    ),
    "anton smirnov": (
        "anton smirnov",
        "http://rocm.blogs.amd.com/authors/anton-smirnov.html",
    ),
    "asitav mishra": (
        "asitav mishra",
        "http://rocm.blogs.amd.com/authors/asitav-mishra.html",
    ),
    "clint greene": (
        "clint greene",
        "http://rocm.blogs.amd.com/authors/clint-greene.html",
    ),
    "damon mcdougall": (
        "damon mcdougall",
        "http://rocm.blogs.amd.com/authors/damon-mcdougall.html",
    ),
    "david doscher": (
        "david doscher",
        "http://rocm.blogs.amd.com/authors/david-doscher.html",
    ),
    "douglas jia": (
        "douglas jia",
        "http://rocm.blogs.amd.com/authors/douglas-jia.html",
    ),
    "eliot li": (
        "eliot li",
        "http://rocm.blogs.amd.com/authors/eliot-li.html",
    ),
    "fabricio flores": (
        "fabricio flores",
        "http://rocm.blogs.amd.com/authors/fabricio-flores.html",
    ),
    "gina sitaraman": (
        "gina sitaraman",
        "http://rocm.blogs.amd.com/authors/gina-sitaraman.html",
    ),
    "justin chang": (
        "justin chang",
        "http://rocm.blogs.amd.com/authors/justin-chang.html",
    ),
    "mahdieh ghazimirsaeed": (
        "mahdieh ghazimirsaeed",
        "http://rocm.blogs.amd.com/authors/mahdieh-ghazimirsaeed.html",
    ),
    "maria ruiz varela": (
        "maria ruiz varela",
        "http://rocm.blogs.amd.com/authors/maria-ruiz-varela.html",
    ),
    "nicholas curtis": (
        "nicholas curtis",
        "http://rocm.blogs.amd.com/authors/nicholas-curtis.html",
    ),
    "nicholas malaya": (
        "nicholas malaya",
        "http://rocm.blogs.amd.com/authors/nicholas-malaya.html",
    ),
    "noah wolfe": (
        "noah wolfe",
        "http://rocm.blogs.amd.com/authors/noah-wolfe.html",
    ),
    "noel chalmers": (
        "noel chalmers",
        "http://rocm.blogs.amd.com/authors/noel-chalmers.html",
    ),
    "ossian oreilly": (
        "ossian oreilly",
        "http://rocm.blogs.amd.com/authors/ossian-oreilly.html",
    ),
    "paul mullowney": (
        "paul mullowney",
        "http://rocm.blogs.amd.com/authors/paul-mullowney.html",
    ),
    "phillip dang": (
        "phillip dang",
        "http://rocm.blogs.amd.com/authors/phillip-dang.html",
    ),
    "rajat arora": (
        "rajat arora",
        "http://rocm.blogs.amd.com/authors/rajat-arora.html",
    ),
    "rene van oostrum": (
        "rene van oostrum",
        "http://rocm.blogs.amd.com/authors/rene-van-oostrum.html",
    ),
    "sean miller": (
        "sean miller",
        "http://rocm.blogs.amd.com/authors/sean-miller.html",
    ),
    "sean song": (
        "sean song",
        "http://rocm.blogs.amd.com/authors/sean-song.html",
    ),
    "seung rok jung": (
        "seung rok jung",
        "http://rocm.blogs.amd.com/authors/seung-rok-jung.html",
    ),
    "suyash tandon": (
        "suyash tandon",
        "http://rocm.blogs.amd.com/authors/suyash-tandon.html",
    ),
    "thomas gibson": (
        "thomas gibson",
        "http://rocm.blogs.amd.com/authors/thomas-gibson.html",
    ),
    "vara lakshmi bayanagari": (
        "vara lakshmi bayanagari",
        "http://rocm.blogs.amd.com/authors/vara-lakshmi-bayanagari.html",
    ),
    "vicky tsang": (
        "vicky tsang",
        "http://rocm.blogs.amd.com/authors/vicky-tsang.html",
    ),
    "yao fehlis": (
        "yao fehlis",
        "http://rocm.blogs.amd.com/authors/yao-fehlis.html",
    ),
}
blog_feed_archives = True
blog_feed_fulltext = True
blog_feed_templates = {
    "atom": {
        "content": "{{ title }}{% for tag in post.tags %}"
        " #{{ tag.name|trim()|replace(' ', '') }}"
        "{% endfor %}",
    },
    "social": {
        "content": "{{ title }}{% for tag in post.tags %}"
        " #{{ tag.name|trim()|replace(' ', '') }}"
        "{% endfor %}",
    },
}
blog_feed_length = 10

html_static_path = ['_static']

html_css_files = [
    'css/custom.css',
]

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
    app.connect("autodoc-process-docstring", cut_lines(4, what=["module"]))
    app.add_object_type(
        "confval",
        "confval",
        objname="configuration value",
        indextemplate="pair: %s; configuration value",
    )
    fdesc = GroupedField(
        "parameter", label="Parameters", names=["param"], can_collapse=True
    )
    app.add_object_type(
        "event", "event", "pair: %s; event", parse_event, doc_field_types=[fdesc]
    )
