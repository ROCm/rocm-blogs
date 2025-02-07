from blog import grab_href

CSS_PATH = "./scripts/css/social-bar.css"
HTML_PATH = "./scripts/html/social-bar.html"


def quickshare(blog):

    with open(CSS_PATH, "r") as f:

        css = f.read()
    with open(HTML_PATH, "r") as f:

        html = f.read()
    social_bar = """
<style>
{CSS}
</style>
{HTML}
"""

    social_bar = social_bar.format(CSS=css, HTML=html)

    url = f"http://rocm.blogs.amd.com{grab_href(blog)[1:]}"
    title = blog.blog_title if hasattr(blog, "blog_title") else "No Title"

    if hasattr(blog, "myst"):
        description = blog.myst.get("html_meta", {}).get(
            "description lang=en", "No Description"
        )
    else:
        description = "No Description"
    title = f"{title} | ROCm Blogs"

    social_bar = (
        social_bar.replace("{URL}", url)
        .replace("{TITLE}", title)
        .replace("{TEXT}", description)
    )

    return social_bar
