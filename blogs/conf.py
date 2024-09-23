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

html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "rocm.blogs.amd.com")
html_context = {}
if os.environ.get("READTHEDOCS", "") == "True":
    html_context["READTHEDOCS"] = True

html_title = "ROCm Blogs"
html_theme = "rocm_docs_theme"
html_theme_options = {
    "flavor": "rocm-blogs"
}

extensions = ["rocm_docs", "ablog", "sphinx.ext.intersphinx", 'hoverxref.extension', 'myst_parser']
external_toc_path = "./sphinx/_toc.yml"

hoverxref_api_host = "/_"

templates_path = ["."]

html_sidebars = {
    "**": [
        "search-field.html",
        "postcard.html",
        "recentposts.html",
        "tagcloud.html",
        "categories.html",
        "archives.html",
    ]
}

blog_authors = {
"Alessandro Fanfarillo": (
        "Alessandro Fanfarillo",
        "http://rocm.blogs.amd.com/authors/alessandro-fanfarillo.html",
),
"Alex Voicu": (
        "Alex Voicu",
        "http://rocm.blogs.amd.com/authors/alex-voicu.html",
),
"Anton Smirnov": (
        "Anton Smirnov",
        "http://rocm.blogs.amd.com/authors/anton-smirnov.html",
),
"Asitav Mishra": (
        "Asitav Mishra",
        "http://rocm.blogs.amd.com/authors/asitav-mishra.html",
),
"Bob Robey": (
        "Bob Robey",
        "http://rocm.blogs.amd.com/authors/bob-robey.html",
),
"Clint Greene": (
        "Clint Greene",
        "http://rocm.blogs.amd.com/authors/clint-greene.html",
),
"Damon McDougall": (
        "Damon McDougall",
        "http://rocm.blogs.amd.com/authors/damon-mcdougall.html",
),
"David Doscher": (
        "David Doscher",
        "http://rocm.blogs.amd.com/authors/david-doscher.html",
),
"Douglas Jia": (
        "Douglas Jia",
        "http://rocm.blogs.amd.com/authors/douglas-jia.html",
),
"Eliot Li": (
        "Eliot Li",
        "http://rocm.blogs.amd.com/authors/eliot-li.html",
),
"Fabricio Flores": (
        "Fabricio Flores",
        "http://rocm.blogs.amd.com/authors/fabricio-flores.html",
),
"George Markomanolis": (
        "George Markomanolis",
        "http://rocm.blogs.amd.com/authors/george-markomanolis.html",
),
"Gina Sitaraman": (
        "Gina Sitaraman",
        "http://rocm.blogs.amd.com/authors/gina-sitaraman.html",
),
"Justin Chang": (
        "Justin Chang",
        "http://rocm.blogs.amd.com/authors/justin-chang.html",
),
"Mahdieh Ghazimirsaeed": (
        "Mahdieh Ghazimirsaeed",
        "http://rocm.blogs.amd.com/authors/mahdieh-ghazimirsaeed.html",
),
"Matt Elliott": (
        "Matt Elliott",
        "http://rocm.blogs.amd.com/authors/matt-elliott.html",
),
"Maria Ruiz Varela": (
        "Maria Ruiz Varela",
        "http://rocm.blogs.amd.com/authors/maria-ruiz-varela.html",
),
"Nicholas Curtis": (
        "Nicholas Curtis",
        "https://rocm.blogs.amd.com/authors/nicholas-curtis.html",
),
"Nicholas Malaya": (
        "Nicholas Malaya",
        "http://rocm.blogs.amd.com/authors/nicholas-malaya.html",
),
"Noah Wolfe": (
        "Noah Wolfe",
        "http://rocm.blogs.amd.com/authors/noah-wolfe.html",
),
"Noel Chalmers": (
        "Noel Chalmers",
        "http://rocm.blogs.amd.com/authors/noel-chalmers.html",
),
"Ossian O'Reilly": (
        "Ossian O'Reilly",
        "http://rocm.blogs.amd.com/authors/ossian-oreilly.html",
),
"Paul Mullowney": (
        "Paul Mullowney",
        "http://rocm.blogs.amd.com/authors/paul-mullowney.html",
),
"Phillip Dang": (
        "Phillip Dang",
        "http://rocm.blogs.amd.com/authors/phillip-dang.html",
),
"Rajat Arora": (
        "Rajat Arora",
        "http://rocm.blogs.amd.com/authors/rajat-arora.html",
),
"Rene Van Oostrum": (
        "Rene Van Oostrum",
        "http://rocm.blogs.amd.com/authors/rene-van-oostrum.html",
),
"Sean Miller": (
        "Sean Miller",
        "http://rocm.blogs.amd.com/authors/sean-miller.html",
),
"Sean Song": (
        "Sean Song",
        "http://rocm.blogs.amd.com/authors/sean-song.html",
),
"Seungrok Jung": (
        "Seungrok Jung",
        "http://rocm.blogs.amd.com/authors/seung-rok-jung.html",
),
"Suyash Tandon": (
        "Suyash Tandon",
        "http://rocm.blogs.amd.com/authors/suyash-tandon.html",
),
"Thomas Gibson": (
        "Thomas Gibson",
        "http://rocm.blogs.amd.com/authors/thomas-gibson.html",
),
"Vara Lakshmi Bayanagari": (
        "Vara Lakshmi Bayanagari",
        "http://rocm.blogs.amd.com/authors/vara-lakshmi-bayanagari.html",
),
"Vicky Tsang": (
        "Vicky Tsang",
        "http://rocm.blogs.amd.com/authors/vicky-tsang.html",
),
"Yao Fehlis": (
        "Yao Fehlis",
        "http://rocm.blogs.amd.com/authors/yao-fehlis.html",
),
"Arseny Moskvichev": (
        "Arseny Moskvichev",
        "http://rocm.blogs.amd.com/authors/arseny-moskvichev.html",
),
"Adeem Jassani": (
        "Adeem Jassani",
        "http://rocm.blogs.amd.com/authors/adeem-jassani.html",
),
"Lei Shao": (
        "Lei Shao",
        "http://rocm.blogs.amd.com/authors/lei-shao.html",
),
"Luise Chen": (
        "Luise Chen",
        "http://rocm.blogs.amd.com/authors/luise-chen.html",
)
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
    'css/custom.css', 'css/asciinema-player.css'
]

html_js_files = [
    'js/asciinema-player.min.js',
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
