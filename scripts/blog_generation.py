from blog import grab_authors, grab_image
from datetime import datetime
from giscus import giscus
from quick_share import quickshare

IMAGE_CSS_PATH = "./scripts/css/image_blog.css"
IMAGE_HTML_PATH = "./scripts/html/image_blog.html"
BLOG_CSS_PATH = "./scripts/css/blog.css"


def author_attribution(blogs, minimum_date="September 1, 2024"):

    blogs_to_process = []

    for blog in blogs:

        if blog.date < datetime.strptime(minimum_date, "%B %d, %Y"):

            print(f"Skipping {blog.file_path}: Date is before {minimum_date}.")

            continue
        else:

            blogs_to_process.append(blog)
    for blog in blogs_to_process:

        authors_list = getattr(blog, "author", "").split(",")
        date = blog.date.strftime("%B %d, %Y") if blog.date else "No Date"

        if authors_list:

            authors_html = grab_authors(authors_list)

            if authors_html:

                authors_html = authors_html.replace("././", "../../").replace(
                    ".md", ".html"
                )

                authors_string = f"{date}, by {authors_html}"

                authors_html = f'<div class="author_string">{authors_string}</div>'
            # grab blog link

            readme_file = blog.file_path

            with open(readme_file, "r", encoding="utf-8") as file:

                lines = file.readlines()
            title, line_number = None, None

            for i, line in enumerate(lines):

                # only check for # , do not check if there are more than one # in the line

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

                    blog_image = "../../_" + blog.image_paths[0][2:]

                image_template = image_template.replace("{IMAGE}", blog_image).replace(
                    "{TITLE}", blog.blog_title
                )

                lines.insert(line_number + 1, f"\n{blog_template}\n")

                lines.insert(line_number + 2, f"\n{image_template}\n")

                lines.insert(line_number + 3, f"\n{authors_html}\n")

                # add the image to html

                lines.insert(line_number + 4, f"\n{quickshare_button}\n")

                giscus_comment = giscus()

                # insert at the end of the file

                lines.append(giscus_comment)

                with open(readme_file, "w", encoding="utf-8") as file:

                    # add date class style

                    file.writelines(lines)
                print(f"Author attribution inserted in '{readme_file}'.")
        else:

            print("No authors found in metadata.")
