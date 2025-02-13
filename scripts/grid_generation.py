# Index.md Generator
# Updated 2025 January 27
# Version 1.4.1


from blog import grab_authors, grab_href, grab_image
from datetime import datetime

INDEX_CSS = "./scripts/css/index.css"
INDEX_HTML = "./scripts/html/index.html"


def generate_blog_grid(
    blogs, output_file="latest_blogs.md", max_blogs=8, max_category=4
):

    index_template = """
---
title: ROCm Blogs
myst:
  html_meta:
    "description lang=en": "AMD ROCm™ software blogs"
    "keywords": "AMD GPU, MI300, MI250, ROCm, blog"
    "property=og:locale": "en_US"
---
<style>
{CSS}
</style>
{HTML}
"""

    with open(INDEX_CSS, "r") as f:

        css = f.read()
    with open(INDEX_HTML, "r") as f:

        html = f.read()
    index_template = index_template.format(CSS=css, HTML=html)

    # remove the first new line

    index_template = index_template[1:]

    grid_items = []
    application_grid_items = []
    software_grid_items = []
    eco_grid_items = []

    for index, blog in enumerate(blogs):

        title = blog.blog_title if hasattr(blog, "blog_title") else "No Title"

        date = blog.date.strftime("%B %d, %Y") if blog.date else "No Date"

        # look at myst description

        if hasattr(blog, "myst"):

            print(blog.myst.get("html_meta").get("description lang=en"))
            description = (
                blog.myst.get("html_meta").get("description lang=en")
                if blog.myst.get("html_meta").get("description lang=en")
                else "No Description"
            )

            if len(description) > 150:
                description = description[:150] + "..."
            else:
                # add invisible characters to ensure the card is the same size

                description = description + "..." + " " * (150 - len(description))
        # Get authors from the blog (assuming it's a comma-separated string)

        authors_list = getattr(blog, "author", "").split(",")

        href = grab_href(blog)

        image = (
            grab_image(blog, href)
            if hasattr(blog, "thumbnail")
            else "./images/generic.jpg"
        )

        # Join author links with commas

        if authors_list:

            authors_html = grab_authors(authors_list)
        if authors_html:

            authors_html = f"by {authors_html}"
        # Create grid item card with authors

        grid_item = f"""
:::{{grid-item-card}}
:padding: 1
:img-top: {image}
:class-img-top: small-sd-card-img-top
:class-body: small-sd-card
:class: small-sd-card
+++
<a href="{href}" class="small-card-header-link">
    <h2 class="card-header">{title}</h2>
</a>
<p class="paragraph">{description}</p>
<div class="date">{date} {authors_html}</div>
:::
"""

        if index < max_blogs:

            grid_items.append(grid_item)
        elif (
            blog.category == "Applications & models"
            and len(application_grid_items) < max_category
        ):

            application_grid_items.append(grid_item)
        elif (
            blog.category == "Software tools & optimizations"
            and len(software_grid_items) < max_category
        ):

            software_grid_items.append(grid_item)
        elif (
            blog.category == "Ecosystems and Partners"
            and len(eco_grid_items) < max_category
        ):

            eco_grid_items.append(grid_item)
    print(f"{software_grid_items}")

    grid_content = "".join(grid_items)
    application_grid_content = "".join(application_grid_items)
    software_grid_content = "".join(software_grid_items)
    eco_grid_content = "".join(eco_grid_items)

    # Write the grid content to the Markdown file

    with open(output_file, "w", encoding="utf-8") as f:

        f.write(grid_content)
    print(f"Grid content successfully written to {output_file}")

    index_template = index_template.replace("{grid_items}", grid_content)

    index_template = index_template.replace("{eco_grid_items}", eco_grid_content)

    index_template = index_template.replace(
        "{application_grid_items}", application_grid_content
    )

    index_template = index_template.replace(
        "{software_grid_items}", software_grid_content
    )

    index_template = index_template.replace(
        "{datetime}", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    # dangerous

    # write new index.md

    with open("blogs/index.md", "w", encoding="utf-8") as f:

        f.write(index_template)
    return index_template
