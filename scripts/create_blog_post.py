"""Generate a new blog post directory tree from issue-form metadata.

Invoked by .github/workflows/auto-create-blog-post-pr.yml after the issue
form is parsed. All inputs come in as positional argv strings.
"""

import os
import re
import sys
from datetime import datetime


# 15 positional args are required (indices 0..14). The original `< 14`
# check let a 14-arg invocation through and crash on `args[14]`.
REQUIRED_ARG_COUNT = 15


# Map common Unicode punctuation to ASCII equivalents so titles that use
# non-breaking hyphens (U+2011), em-dashes, curly quotes, etc. survive the
# round-trip through YAML, Sphinx/MyST, and AMD.com consistently instead of
# rendering as missing glyphs or being silently stripped.
UNICODE_PUNCT_MAP = {
    "\u2010": "-",    # hyphen
    "\u2011": "-",    # non-breaking hyphen   <-- the "On‑Prem" case
    "\u2012": "-",    # figure dash
    "\u2013": "-",    # en dash
    "\u2014": "-",    # em dash
    "\u2015": "-",    # horizontal bar
    "\u2018": "'",    # left single curly quote
    "\u2019": "'",    # right single curly quote
    "\u201A": "'",    # single low-9 quote
    "\u201C": '"',    # left double curly quote
    "\u201D": '"',    # right double curly quote
    "\u201E": '"',    # double low-9 quote
    "\u2026": "...",  # horizontal ellipsis
    "\u00A0": " ",    # non-breaking space
}


def normalize_unicode_punct(text: str) -> str:
    """Replace common Unicode punctuation with ASCII equivalents."""
    for uni, ascii_eq in UNICODE_PUNCT_MAP.items():
        text = text.replace(uni, ascii_eq)
    return text


def yaml_dq_escape(text: str) -> str:
    """Escape a string for safe embedding inside a YAML double-quoted scalar.

    We collapse any internal newlines / tabs to single spaces because the
    template uses single-line scalars throughout.
    """
    text = text.replace("\\", "\\\\").replace('"', '\\"')
    text = re.sub(r"[\r\n\t]+", " ", text)
    return text


def safe_field(text: str) -> str:
    """Normalize Unicode punctuation, trim whitespace, then YAML-escape."""
    return yaml_dq_escape(normalize_unicode_punct(text).strip())


def truncate_string(input_string: str) -> str:
    """Filesystem-safe slug derived from a free-form string."""
    cleaned = normalize_unicode_punct(input_string)
    # Strip characters that are unsafe in directory names. The original list
    # missed `<>:"\\'` and let apostrophes through.
    cleaned = re.sub(r"[!@#$%^&*?/|<>:\"\\']", "", cleaned)
    cleaned = re.sub(r"\s+", "-", cleaned)
    return cleaned.lower()


def gather_args():
    args = sys.argv[1:]
    if len(args) < REQUIRED_ARG_COUNT:
        print(
            f"Not enough arguments provided. "
            f"Expected {REQUIRED_ARG_COUNT}, got {len(args)}."
        )
        sys.exit(1)
    return args


def create_blog_post_from_args():
    args = gather_args()
    raw_blog_title              = args[0]
    blog_file_path              = args[1]
    raw_authors                 = args[2]
    raw_tags                    = args[3]
    raw_category                = args[4]
    raw_audience                = args[5]
    raw_kvp                     = args[6]
    raw_keywords                = args[7]
    raw_amd_technical_blog_type = args[8]
    raw_amd_applications        = args[9]
    raw_description             = args[10]
    raw_hardware_amd_deployment = args[11]
    raw_software_amd_deployment = args[12]
    raw_amd_category_topic      = args[13]
    raw_market_vertical         = args[14]

    # Display title: Unicode normalized (so "On‑Prem" becomes "On-Prem"), but
    # NOT YAML-escaped — used in the markdown H1.
    blog_title_md = normalize_unicode_punct(raw_blog_title).strip()
    # YAML form of the same title — escaped for embedding inside double quotes.
    blog_title_yaml = yaml_dq_escape(blog_title_md)

    today_dt = datetime.today()
    today = today_dt.strftime("%d %b %Y")
    year = today_dt.strftime("%Y")

    images_template = """
Upload your blog thumbnail here. Please delete this file after uploading image.

delete me before publishing blog
"""

    # Every variable string field is now double-quoted in the template, and
    # every value passed in is run through safe_field(). This makes the
    # frontmatter robust to apostrophes, ampersands, colons, quotes, leading
    # special characters, etc.
    blog_template = """---
blogpost: true
blog_title: "{blog_title_yaml}"
date: "{today}"
author: "{blog_authors}"
thumbnail: ''
tags: "{blog_tags}"
category: "{blog_category}"
target_audience: "{blog_audience}"
key_value_propositions: "{blog_key_value_proposition}"
language: English
myst:
    html_meta:
        "author": "{blog_authors}"
        "description lang=en": "{blog_description}"
        "keywords": "{blog_keywords}"
        "vertical": "{blog_market_vertical}"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "{blog_amd_technical_blog_type}"
        "amd_blog_hardware_platforms": "{blog_hardware_amd_deployment}"
        "amd_blog_development_tools": "{blog_software_amd_deployment}"
        "amd_blog_applications": "{blog_amd_applications}"
        "amd_blog_topic_categories": "{blog_amd_category_topic}"
        "amd_blog_authors": "{blog_authors}"
---

<!---
Copyright (c) {year} Advanced Micro Devices, Inc. (AMD)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
--->

# {blog_title_md}

ROCm Blogs follow a consistent magazine article approach where there is no explicit introduction per se,
but rather each blog starts with a brief, wide-scoped introductory text, without a section title,
before moving into the blog's first section.
The introductory text should include a concise description of your blog: briefly describe for the
reader how they will benefit from the blog, detailing its main deliverables. Please use an active-voice,
call-to-action approach.

## Body

This is where you unleash your creativity. Please follow these general guidelines:

- use actionable, hands-on, conversational approach, guiding your reader through the blog and its content, maintaining engagement. Use active voice, call-to-action (CTA) text (e.g. "Interested in learning more?", "Run this function by using", "Try implementing this yourself")

- keep your writing structured, engaging, and actionable. Divide the blog's content into logical sections.

- Make sure you provide the required background and prerequisites for your blog. Outline any foundational knowledge and tools the reader will likely require.

- When describing a process use step-by-step guide, employ numbered steps or subheadings to guide the reader through the process.

- Integrate examples and use cases: provide real-world applications and scenarios. Reflect on common pitfalls and possible troubleshooting approaches, addressing potential mistakes and solutions.

Leeway into figures, equations, etc.

## Sample markdown

This section covers some markdown techniques commonly used in a blogs.

This is a table.

|      | SPX (MI300X) | CPX (MI300X) |
| ---- | :----------: | :----------: |
| NPS1 |      ✔       |      ✔       |
| NPS4 |              |      ✔       |

Below is a code snippet from the console. You can also use bash, C++, python and other languages.

```console
echo "c 226:128 rwm" > /sys/fs/cgroup/devices/devices.deny #Deny access to device 226:128 in docker (renderD128)

echo "c 226:128 rwm" > /sys/fs/cgroup/devices/devices.allow #Allow access to device 226:128 in docker (renderD128)
```

```{note}
This is how to add a note. See the [myst markdown admonition guide](https://mystmd.org/guide/admonitions) for more details.
```

## Summary

ROCm Blogs follow a consistent magazine-article approach where each blog ends with a "Summary" section.
Please provide a brief summary of your blog, reiterating the main takeaways and deliverables, as well
as what the reader learned from it.

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. 
However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.
THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
AMD, the AMD Arrow logo, [insert all other AMD trademarks used in the material here per AMD Trademarks] and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. [Insert any third party trademark attribution here per AMD's Third Party Trademark List.]
© [Insert year written*] Advanced Micro Devices, Inc. All rights reserved
"""

    blog_template = blog_template.format(
        blog_title_md=blog_title_md,
        blog_title_yaml=blog_title_yaml,
        today=today,
        year=year,
        blog_authors=safe_field(raw_authors),
        blog_tags=safe_field(raw_tags),
        blog_category=safe_field(raw_category),
        blog_audience=safe_field(raw_audience),
        blog_key_value_proposition=safe_field(raw_kvp),
        blog_keywords=safe_field(raw_keywords),
        blog_amd_technical_blog_type=safe_field(raw_amd_technical_blog_type),
        blog_amd_applications=safe_field(raw_amd_applications),
        blog_description=safe_field(raw_description),
        blog_hardware_amd_deployment=safe_field(raw_hardware_amd_deployment),
        blog_software_amd_deployment=safe_field(raw_software_amd_deployment),
        blog_amd_category_topic=safe_field(raw_amd_category_topic),
        blog_market_vertical=safe_field(raw_market_vertical),
        note="{note}",  # leave the MyST {note} admonition literal
    )

    blog_file_path = truncate_string(blog_file_path[:30])
    dir_blog_name = "-".join(blog_file_path.split("-")[:3])
    dir_category_name = truncate_string(raw_category)

    if dir_category_name == "applications-models":
        dir_category_name = "artificial-intelligence"
    elif dir_category_name == "software-tools-optimizations":
        dir_category_name = "software-tools-optimization"

    os.makedirs(f"blogs/{dir_category_name}/{dir_blog_name}", exist_ok=True)
    os.makedirs(f"blogs/{dir_category_name}/{dir_blog_name}/images", exist_ok=True)

    with open(f"blogs/{dir_category_name}/{dir_blog_name}/README.md", "w") as f:
        f.write(blog_template)

    with open(f"blogs/{dir_category_name}/{dir_blog_name}/images/README.md", "w") as f:
        f.write(images_template)


if __name__ == "__main__":
    create_blog_post_from_args()
