import re
from datetime import datetime

from blog import grab_authors, grab_image
from giscus import giscus
from quick_share import quickshare

IMAGE_CSS_PATH = "./scripts/css/image_blog.css"
IMAGE_HTML_PATH = "./scripts/html/image_blog.html"
BLOG_CSS_PATH = "./scripts/css/blog.css"
AUTHOR_HTML_PATH = "./scripts/html/author_attribution.html"


def calculate_read_time(words: int) -> int:

    return round(words / 245)


def truncate_string(input_string: str) -> str:
    # remove special characters
    cleaned_string = re.sub(r"[!@#$%^&*?/|]", "", input_string)
    # remove spaces
    transformed_string = re.sub(r"\s+", "-", cleaned_string)

    return transformed_string.lower()


def blog_generation(blogs, minimum_date="September 1, 2024"):

    blogs_to_process = []

    for blog in blogs:

        readme_file = blog.file_path

        with open(readme_file, "r", encoding="utf-8") as file:

            lines = file.readlines()

        giscus_comment = giscus()

        lines.append(giscus_comment)

        with open(readme_file, "w", encoding="utf-8") as file:

            # add date class style

            file.writelines(lines)

        if blog.date < datetime.strptime(minimum_date, "%B %d, %Y"):

            print(f"Skipping {blog.file_path}: Date is before {minimum_date}.")

            continue
        else:

            blogs_to_process.append(blog)
    for blog in blogs_to_process:

        authors_list = getattr(blog, "author", "").split(",")
        date = blog.date.strftime("%B %d, %Y") if blog.date else "No Date"
        language = blog.language if hasattr(blog, "language") else "en"
        category = blog.category if hasattr(blog, "category") else "blog"
        tags = blog.tags if hasattr(blog, "tags") else ""

        tags = tags.split(",")
        tag_links = []
        for tag in tags:
            tag = tag.strip()
            truncated_tag = truncate_string(tag)
            tag_links.append(truncated_tag)

        for tag, tag_link in zip(tags, tag_links):
            print(f"Tag: {tag}, Tag Link: {tag_link}")
            ntag = tag.strip()
            tag_html = f'<a href="https://rocm.blogs.amd.com/blog/tag/{tag_link}.html">{ntag}</a>'
            tags[tags.index(tag)] = tag_html
        tags = ", ".join(tags)
        category_link = truncate_string(category)
        category = category.strip()
        category = f'<a href="https://rocm.blogs.amd.com/blog/category/{category_link}.html">{category}</a>'
        blog_read_time = (
            str(calculate_read_time(blog.word_count))
            if hasattr(blog, "word_count")
            else "No Read Time"
        )

        if authors_list:

            authors_html = grab_authors(authors_list)

            if authors_html:

                authors_html = authors_html.replace("././", "../../").replace(
                    ".md", ".html"
                )

                authors_string = f"{authors_html}"

            # grab blog link

            readme_file = blog.file_path

            with open(readme_file, "r", encoding="utf-8") as file:

                lines = file.readlines()
            title, line_number = None, None

            for i, line in enumerate(lines):

                # only check for # , do not check if there are more than one #
                # in the line

                if line.startswith("#") and line.count("#") == 1:

                    title = line

                    line_number = i

                    break
            if title:

                # insert the author attribution after the title
                # title
                #
                # author attribution
                #
                # content

                blog_image = None

                # quickshare

                quickshare_button = quickshare(blog)

                with open(IMAGE_CSS_PATH, "r") as f:

                    image_css = f.read()
                with open(IMAGE_HTML_PATH, "r") as f:

                    image_html = f.read()
                with open(BLOG_CSS_PATH, "r") as f:

                    blog_css = f.read()
                with open(AUTHOR_HTML_PATH, "r") as f:

                    authors_html = f.read()
                authors_html = (
                    authors_html.replace("{authors_string}", authors_string)
                    .replace("{date}", date)
                    .replace("{language}", language)
                    .replace("{category}", category)
                    .replace("{tags}", tags)
                    .replace("{read_time}", blog_read_time)
                    .replace(
                        "{word_count}",
                        (
                            str(blog.word_count)
                            if hasattr(blog, "word_count")
                            else "No Word Count"
                        ),
                    )
                )
                blog_template = f"""
<style>
{blog_css}
</style>
"""
                image_template = f"""
<style>
{image_css}
</style>
{image_html}
"""
                print(blog.image_paths)

                if blog.image_paths:

                    blog_image = "../../_static/" + blog.image_paths[0]

                image_template = image_template.replace("{IMAGE}", blog_image).replace(
                    "{TITLE}", blog.blog_title
                )

                lines.insert(line_number + 1, f"\n{blog_template}\n")

                lines.insert(line_number + 2, f"\n{image_template}\n")

                lines.insert(line_number + 3, f"\n{authors_html}\n")

                lines.insert(line_number + 4, f"\n{quickshare_button}\n")

                with open(readme_file, "w", encoding="utf-8") as file:

                    # add date class style

                    file.writelines(lines)
                print(f"Author attribution inserted in '{readme_file}'.")
        else:

            print("No authors found in metadata.")
