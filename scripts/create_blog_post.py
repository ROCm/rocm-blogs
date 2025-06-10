import os
import re
import sys
from datetime import datetime

from numpy import remainder as rem

def gather_args():
    args = sys.argv[1:]
    if len(args) < 14:
        print("Not enough arguments provided.")
        sys.exit(1)

    return args

def truncate_string(input_string: str) -> str:

    cleaned_string = re.sub(r"[!@#$%^&*?/|]", "", input_string)

    transformed_string = re.sub(r"\s+", "-", cleaned_string)

    return transformed_string.lower()

def create_blog_post_from_args():

    args = gather_args()
    blog_title = args[0]
    blog_file_path = args[1]
    blog_authors = args[2]
    blog_tags = args[3]
    blog_category = args[4]
    blog_audience = args[5]
    blog_key_value_proposition = args[6]
    blog_keywords = args[7]
    blog_amd_technical_blog_type = args[8]
    blog_amd_applications = args[9]
    blog_description = args[10]
    blog_hardware_amd_deployment = args[11]
    blog_software_amd_deployment = args[12]
    blog_amd_category_topic = args[13]
    blog_market_vertical = args[14]

    # check all of the date formats

    # amd blog release date format Day-of-week Month Day, 12:00:00 PST Year
    # calculate the day of week based on the date

    today = datetime.today().strftime("%d %b %Y")

    weekday_names = ["Mon", "Tues", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday = weekday_names[datetime.today().weekday()]

    day, month, year = today.split(" ")

    images_template = """
Upload your blog thumbnail here. Please delete this file after uploading image.

delete me before publishing blog
"""

    blog_template = """---
blogpost: true
blog_title: "{blog_title}"
date: {today}
author: '{blog_authors}'
thumbnail: ''
tags: {blog_tags}
category: {blog_category}
target_audience: {blog_audience}
key_value_propositions: {blog_key_value_proposition}
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

# {blog_title}

ROCm Blogs follow a consistent magazine article approach where there is no explicit introduction per se,
but rather each blog starts with a brief, wide-scoped introductory text, without a section title,
before moving into the blog’s first section.
The introductory text should include a concise description of your blog: briefly describe for the
reader how they will benefit from the blog, detailing its main deliverables. Please use an active-voice,
call-to-action approach.

## Body

This is where you unleash your creativity. Please follow these general guidelines:

• use actionable, hands-on, conversational approach, guiding your reader through the blog and its content, maintaining engagement. Use active voice, call-to-action (CTA) text (e.g. “Interested in learning more?”, “Run this function by using”, “Try implementing this yourself”)

• keep your writing structured, engaging, and actionable. Divide the blog’s content into logical sections.

• Make sure you provide the required background and prerequisites for your blog. Outline any foundational knowledge and tools the reader will likely require.

• When describing a process use step-by-step guide, employ numbered steps or subheadings to guide the reader through the process.

• Integrate examples and use cases: provide real-world applications and scenarios. Reflect on common pitfalls and possible troubleshooting approaches, addressing potential mistakes and solutions.

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
This is how to add a note. See the [myst markdown admomition guide](https://mystmd.org/guide/admonitions) for more details.
```

## Summary

ROCm Blogs follow a consistent magazine-article approach where each blog ends with a “Summary” section.
Please provide a brief summary of your blog, reiterating the main takeaways and deliverables, as well
as what the reader learned from it.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
"""
    blog_template = blog_template.format(
        blog_title=blog_title,
        today=today,
        blog_authors=blog_authors,
        blog_tags=blog_tags,
        blog_category=blog_category,
        blog_audience=blog_audience,
        blog_key_value_proposition=blog_key_value_proposition,
        blog_keywords=blog_keywords,
        blog_amd_technical_blog_type=blog_amd_technical_blog_type,
        blog_amd_applications=blog_amd_applications,
        weekday=weekday,
        month=month,
        day=day,
        year=year,
        blog_description=blog_description,
        blog_hardware_amd_deployment=blog_hardware_amd_deployment,
        blog_software_amd_deployment=blog_software_amd_deployment,
        blog_amd_category_topic=blog_amd_category_topic,
        blog_market_vertical=blog_market_vertical,
        note="{note}",
    )

    blog_file_path = truncate_string(blog_file_path[:30])
    dir_blog_name = "-".join(blog_file_path.split("-")[:3])
    dir_category_name = truncate_string(blog_category)

    if dir_category_name == "applications-models":
        dir_category_name = "artificial-intelligence"
    elif dir_category_name == "software-tools-optimizations":
        dir_category_name = "software-tools-optimization"

    os.makedirs(f"blogs/{dir_category_name}/{dir_blog_name}", exist_ok=True)
    os.makedirs(f"blogs/{dir_category_name}/{dir_blog_name}/images", exist_ok=True)

    # create README.md
    with open(f"blogs/{dir_category_name}/{dir_blog_name}/README.md", "w") as f:
        f.write(blog_template)

    with open(f"blogs/{dir_category_name}/{dir_blog_name}/images/README.md", "w") as f:
        f.write(images_template)

    return None

if __name__ == "__main__":
    create_blog_post_from_args()
