# Index.md Generator
# Updated 2025 January 27
# Version 1.4.1


import os
import re
import yaml
import time
import shutil
from datetime import datetime
from PIL import Image, ImageOps


class Blog:

    def __init__(self, file_path, metadata):

        self.file_path = file_path
        self.metadata = metadata

        # Dynamically set attributes based on metadata

        for key, value in metadata.items():

            setattr(self, key, value)
        # Ensure the 'date' field exists

        if "date" in metadata:

            self.date = self.parse_date(metadata["date"])
        else:

            self.date = None

    def normalize_date_string(self, date_str):

        # do not remove

        date_str = date_str.replace("Sept", "Sep")

        return date_str

    def parse_date(self, date_str):

        # Normalize the date string

        date_str = self.normalize_date_string(date_str)

        # Define possible date formats, including string-based months

        date_formats = [
            "%d-%m-%Y",  # e.g. 8-08-2024
            "%d/%m/%Y",  # e.g. 8/08/2024
            "%d-%B-%Y",  # e.g. 8-August-2024
            "%d-%b-%Y",  # e.g. 8-Aug-2024
            "%d %B %Y",  # e.g. 8 August 2024
            "%d %b %Y",  # e.g. 8 Aug 2024
            "%d %B, %Y",  # e.g. 8 August, 2024
            "%d %b, %Y",  # e.g. 8 Aug, 2024
        ]

        for fmt in date_formats:

            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        print(f"Invalid date format in {self.file_path}: {date_str}")

        return None

    def __repr__(self):

        return f"Blog(file_path='{self.file_path}', metadata={self.__dict__})"


def find_readme_files(root_dir):

    readme_files = []

    for dirpath, dirnames, filenames in os.walk(root_dir):

        for filename in filenames:

            if filename.lower() == "readme.md":  # Case-insensitive matching

                full_path = os.path.join(dirpath, filename)

                readme_files.append(full_path)
    return readme_files


def extract_metadata(file_path):

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    # Regular expression to match YAML front matter

    yaml_pattern = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

    match = yaml_pattern.match(content)

    if match:

        yaml_content = match.group(1)

        try:

            metadata = yaml.safe_load(yaml_content)

            return metadata
        except yaml.YAMLError as e:

            print(f"Error parsing YAML in {file_path}: {e}")

            return None
    else:

        print(f"No metadata found in {file_path}.")

        return None


def create_blog_objects(readme_files):

    blog_objects = []

    for file_path in readme_files:

        metadata = extract_metadata(file_path)

        if metadata:

            blog = Blog(file_path, metadata)

            blog_objects.append(blog)
        else:

            print(f"Skipping {file_path}: No valid metadata found.")
    return blog_objects


def quicksort_desc(arr, low, high):
    if low < high:
        pivot_index = partition(arr, low, high)
        quicksort_desc(arr, low, pivot_index - 1)
        quicksort_desc(arr, pivot_index + 1, high)


def partition(arr, low, high):
    pivot = arr[high]["date_epoch"]
    x = low - 1
    for y in range(low, high):
        if arr[y]["date_epoch"] > pivot:
            x += 1
            arr[x], arr[y] = arr[y], arr[x]
    arr[x + 1], arr[high] = arr[high], arr[x + 1]
    return x + 1


def sort_blogs_by_date(blogs):

    blogs_with_date = []
    for blog in blogs:
        if hasattr(blog, "date") and blog.date is not None:

            blog_date_record = {
                "original_blog": blog,
                "date_epoch": int(blog.date.timestamp()),
                "date_str": blog.date.strftime("%Y-%m-%d"),
            }
            blogs_with_date.append(blog_date_record)
    quicksort_desc(blogs_with_date, 0, len(blogs_with_date) - 1)

    sorted_blogs = []
    for entry in blogs_with_date:
        # Perform extra complication by re-validating the date

        if hasattr(entry["original_blog"], "date"):
            sorted_blogs.append(entry["original_blog"])
    return sorted_blogs


def grab_authors(authors_list):

    author_pages_dir = (
        "./blogs/authors"  # Directory where author markdown files are stored
    )

    author_links = []

    for author in authors_list:

        # Clean author name and format it correctly for the file system

        author_name = author.strip().replace(" ", "-").lower()

        # Path to the author's markdown file in the 'authors' directory

        author_file = os.path.join(author_pages_dir, f"{author_name}.md")

        print(f"Checking for author file: {author_file}")  # Debug print

        if os.path.exists(author_file):

            # If the author file exists, create a clickable link to the author's page

            author_page = author_file.replace(
                ".md", ".html"
            )  # Convert .md to .html for the link

            print(author_page)

            author_page = author_page.replace("blogs", ".")

            author_links.append(f'<a href="{author_page}">{author.strip()}</a>')
        else:

            # If no author page exists, display the author's name as plain text

            print(f"Author file {author_file} does not exist.")

            author_links.append(author.strip())
    return ", ".join(author_links) if author_links else ""


def optimize_image(image):

    os.chdir("blogs")
    try:
        with Image.open(image) as img:

            print(img.format, img.size, img.mode)

            before_size = os.path.getsize(image)

            # scaling_factor = 0.3

            original_width, original_height = img.size

            max_width, max_height = (720, 480)
            scaling_factor = min(max_width / original_width, max_height / original_height)

            new_width = int(original_width * scaling_factor)
            new_height = int(original_height * scaling_factor)

            img = img.resize((new_width, new_height), resample=Image.LANCZOS)

            img.save(image, optimize=True, quality=80)

            after_size = os.path.getsize(image)

            print(
                f"Before optimization: {before_size} - After optimization: {after_size} - Total reduction of {((before_size-after_size)/before_size)*100} percent"
            )

            with open("optimize.txt", "a") as f:

                f.write(
                    f"Before optimization: {before_size} - After optimization: {after_size} - Total reduction of {((before_size-after_size)/before_size)*100} percent\n on {image}\n"
                )
    except Exception as error:

        print(f"Error optimizing image {image}: {error}")
    os.chdir("..")


def grab_image(blog, href):
    # Generate an image or use default

    image = blog.thumbnail if hasattr(blog, "thumbnail") else "./images/generic.jpg"

    # check if image path is in the correct format

    if not image.startswith("./images/"):

        image = "./images/" + image
    # remove README.html

    image_href = "./blogs" + ((href[1:].replace("/README.html", image[1:])))
    image_href = image_href.replace("\\", "/")

    # check if image is in images directory (blogs/images)

    temp_image = image.replace("//", "/").replace("./", "blogs/")

    print("\n-------------------------------------------------------------------\n")
    # print image size

    print(f"Link: {href}")

    if not os.path.exists(temp_image):

        print(f"Image {image} does not exist.")

        image = "./images/generic.jpg"

        if os.path.exists(image_href):

            print(f"Image {image_href} exists.")
            image = image_href.replace("./blogs", ".")

            print("The current working directory is: ", os.getcwd())

            optimize_image(image)
        else:

            print(f"Image {image_href} does not exist.")
            image = "./images/generic.jpg"
    # check if images are in the relative blog directory

    elif os.path.exists(href.replace(".html", ".md").replace("blogs", ".")):

        print(href.replace(".html", ".md").replace("blogs", "."))
    else:

        print(f"Image {image} exists.")

        print("The current working directory is: ", os.getcwd())

        optimize_image(image)
    return image


def grab_href(blog):
    href = blog.file_path.replace(".md", ".html")
    href = href.replace("blogs", ".")

    # swap href \ to / for windows

    href = href.replace("\\", "/")

    return href


def quickshare(blog):
    html = """
<style>

.icon-bar.fixed,
.icon-bar.horizontal {
    display: none;
}

@media screen and (min-width: 1520px) {
    .icon-bar.fixed {
        display: flex;
        position: fixed;
        top: 50%;
        right: 0;
        transform: translateY(-50%);
        flex-direction: column;
        z-index: 1000;
    }
    .icon-bar.fixed a {
        display: block;
        text-align: center;
        padding: 16px;
        font-size: 20px;
        color: white;
        transition: background-color 0.3s, color 0.3s;
        background-size: 25px 25px;
    }
    .icon-bar.fixed a svg {
        width: 25px;
        height: 25px;
    }
    .icon-bar.fixed a:hover {
        background-color: #000;
    }
}

@media screen and (max-width: 1520px) {
    .icon-bar.horizontal {
        display: flex;
        flex-direction: row;
        justify-content: flex-start;
        margin: 20px 0;
    }
    .icon-bar.horizontal a {
        width: 30px;
        height: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        color: white;
        margin-left: 8px;
        transition: background-color 0.3s, color 0.3s;
        background-size: 30px 30px;
        padding: 4px;
    }
    .icon-bar.horizontal a svg {
        width: 20px;
        height: 20px;
    }
    .icon-bar.horizontal a:first-child {
        margin-left: 0;
    }
    .icon-bar.horizontal a:hover {
        background-color: #000;
    }
}

.facebook {
    background: #3B5998;  
}
.twitter {
    background: #55ACEE;
}
.reddit {
    background: #dd4b39;
}
.linkedin {
    background: #007bb5;
}
.google {
    background: #bb0000;
}
</style>

<div class="icon-bar fixed">
    <a href="https://www.linkedin.com/shareArticle?mini=true&amp;url={URL}&amp;title={TITLE}" class="linkedin">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-linkedin" viewBox="0 0 16 16">
        <path d="M0 1.146C0 .513.526 0 1.175 0h13.65C15.474 0 16 .513 16 1.146v13.708c0 .633-.526 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854zm4.943 12.248V6.169H2.542v7.225zm-1.2-8.212c.837 0 1.358-.554 1.358-1.248-.015-.709-.52-1.248-1.342-1.248S2.4 3.226 2.4 3.934c0 .694.521 1.248 1.327 1.248zm4.908 8.212V9.359c0-.216.016-.432.08-.586.173-.431.568-.878 1.232-.878.869 0 1.216.662 1.216 1.634v3.865h2.401V9.25c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193v.025h-.016l.016-.025V6.169h-2.4c.03.678 0 7.225 0 7.225z"/>
        </svg>
    </a>
    <a href="https://twitter.com/intent/tweet?url={URL}&amp;text={TEXT}" class="twitter">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-twitter-x" viewBox="0 0 16 16">
        <path d="M12.6.75h2.454l-5.36 6.142L16 15.25h-4.937l-3.867-5.07-4.425 5.07H.316l5.733-6.57L0 .75h5.063l3.495 4.633L12.601.75Zm-.86 13.028h1.36L4.323 2.145H2.865z"/>
        </svg>
    </a>
    <a href="https://www.reddit.com/submit?url={URL}&amp;title={TITLE}" class="reddit">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-reddit" viewBox="0 0 16 16">
        <path d="M6.167 8a.83.83 0 0 0-.83.83c0 .459.372.84.83.831a.831.831 0 0 0 0-1.661m1.843 3.647c.315 0 1.403-.038 1.976-.611a.23.23 0 0 0 0-.306.213.213 0 0 0-.306 0c-.353.363-1.126.487-1.67.487-.545 0-1.308-.124-1.671-.487a.213.213 0 0 0-.306 0 .213.213 0 0 0 0 .306c.564.563 1.652.61 1.977.61zm.992-2.807c0 .458.373.83.831.83s.83-.381.83-.83a.831.831 0 0 0-1.66 0z"/>
        <path d="M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0m-3.828-1.165c-.315 0-.602.124-.812.325-.801-.573-1.9-.945-3.121-.993l.534-2.501 1.738.372a.83.83 0 1 0 .83-.869.83.83 0 0 0-.744.468l-1.938-.41a.2.2 0 0 0-.153.028.2.2 0 0 0-.086.134l-.592 2.788c-1.24.038-2.358.41-3.17.992-.21-.2-.496-.324-.81-.324a1.163 1.163 0 0 0-.478 2.224q-.03.17-.029.353c0 1.795 2.091 3.256 4.669 3.256s4.668-1.451 4.668-3.256c0-.114-.01-.238-.029-.353.401-.181.688-.592.688-1.069 0-.65-.525-1.165-1.165-1.165"/>
        </svg>
    </a>
    <a href="https://www.facebook.com/sharer/sharer.php?u={URL}&quote={TEXT}" class="facebook">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-facebook" viewBox="0 0 16 16">
        <path d="M16 8.049c0-4.446-3.582-8.05-8-8.05C3.58 0-.002 3.603-.002 8.05c0 4.017 2.926 7.347 6.75 7.951v-5.625h-2.03V8.05H6.75V6.275c0-2.017 1.195-3.131 3.022-3.131.876 0 1.791.157 1.791.157v1.98h-1.009c-.993 0-1.303.621-1.303 1.258v1.51h2.218l-.354 2.326H9.25V16c3.824-.604 6.75-3.934 6.75-7.951"/>
        </svg>
    </a>
    <a href="mailto:?subject={URL}&body={TEXT}" class="google">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-envelope-arrow-up" viewBox="0 0 16 16">
        <path d="M0 4a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v4.5a.5.5 0 0 1-1 0V5.383l-7 4.2-1.326-.795-5.64 3.47A1 1 0 0 0 2 13h5.5a.5.5 0 0 1 0 1H2a2 2 0 0 1-2-1.99zm1 7.105 4.708-2.897L1 5.383zM1 4v.217l7 4.2 7-4.2V4a1 1 0 0 0-1-1H2a1 1 0 0 0-1 1"/>
        <path d="M12.5 16a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7m.354-5.354 1.25 1.25a.5.5 0 0 1-.708.708L13 12.207V14a.5.5 0 0 1-1 0v-1.717l-.28.305a.5.5 0 0 1-.737-.676l1.149-1.25a.5.5 0 0 1 .722-.016"/>
        </svg>
    </a>
</div>

<div class="icon-bar horizontal">
    <a href="https://www.linkedin.com/shareArticle?mini=true&amp;url={URL}&amp;title={TITLE}" class="linkedin">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-linkedin" viewBox="0 0 16 16">
        <path d="M0 1.146C0 .513.526 0 1.175 0h13.65C15.474 0 16 .513 16 1.146v13.708c0 .633-.526 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854zm4.943 12.248V6.169H2.542v7.225zm-1.2-8.212c.837 0 1.358-.554 1.358-1.248-.015-.709-.52-1.248-1.342-1.248S2.4 3.226 2.4 3.934c0 .694.521 1.248 1.327 1.248zm4.908 8.212V9.359c0-.216.016-.432.08-.586.173-.431.568-.878 1.232-.878.869 0 1.216.662 1.216 1.634v3.865h2.401V9.25c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193v.025h-.016l.016-.025V6.169h-2.4c.03.678 0 7.225 0 7.225z"/>
        </svg>
    </a>
    <a href="https://twitter.com/intent/tweet?url={URL}&amp;text={TEXT}" class="twitter">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-twitter-x" viewBox="0 0 16 16">
        <path d="M12.6.75h2.454l-5.36 6.142L16 15.25h-4.937l-3.867-5.07-4.425 5.07H.316l5.733-6.57L0 .75h5.063l3.495 4.633L12.601.75Zm-.86 13.028h1.36L4.323 2.145H2.865z"/>
        </svg>
    </a>
    <a href="https://www.reddit.com/submit?url={URL}&amp;title={TITLE}" class="reddit">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-reddit" viewBox="0 0 16 16">
        <path d="M6.167 8a.83.83 0 0 0-.83.83c0 .459.372.84.83.831a.831.831 0 0 0 0-1.661m1.843 3.647c.315 0 1.403-.038 1.976-.611a.23.23 0 0 0 0-.306.213.213 0 0 0-.306 0c-.353.363-1.126.487-1.67.487-.545 0-1.308-.124-1.671-.487a.213.213 0 0 0-.306 0 .213.213 0 0 0 0 .306c.564.563 1.652.61 1.977.61zm.992-2.807c0 .458.373.83.831.83s.83-.381.83-.83a.831.831 0 0 0-1.66 0z"/>
        <path d="M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0m-3.828-1.165c-.315 0-.602.124-.812.325-.801-.573-1.9-.945-3.121-.993l.534-2.501 1.738.372a.83.83 0 1 0 .83-.869.83.83 0 0 0-.744.468l-1.938-.41a.2.2 0 0 0-.153.028.2.2 0 0 0-.086.134l-.592 2.788c-1.24.038-2.358.41-3.17.992-.21-.2-.496-.324-.81-.324a1.163 1.163 0 0 0-.478 2.224q-.03.17-.029.353c0 1.795 2.091 3.256 4.669 3.256s4.668-1.451 4.668-3.256c0-.114-.01-.238-.029-.353.401-.181.688-.592.688-1.069 0-.65-.525-1.165-1.165-1.165"/>
        </svg>
    </a>
    <a href="https://www.facebook.com/sharer/sharer.php?u={URL}&quote={TEXT}" class="facebook">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-facebook" viewBox="0 0 16 16">
        <path d="M16 8.049c0-4.446-3.582-8.05-8-8.05C3.58 0-.002 3.603-.002 8.05c0 4.017 2.926 7.347 6.75 7.951v-5.625h-2.03V8.05H6.75V6.275c0-2.017 1.195-3.131 3.022-3.131.876 0 1.791.157 1.791.157v1.98h-1.009c-.993 0-1.303.621-1.303 1.258v1.51h2.218l-.354 2.326H9.25V16c3.824-.604 6.75-3.934 6.75-7.951"/>
        </svg>
    </a>
    <a href="mailto:?subject={URL}&body={TEXT}" class="google">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-envelope-arrow-up" viewBox="0 0 16 16">
        <path d="M0 4a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v4.5a.5.5 0 0 1-1 0V5.383l-7 4.2-1.326-.795-5.64 3.47A1 1 0 0 0 2 13h5.5a.5.5 0 0 1 0 1H2a2 2 0 0 1-2-1.99zm1 7.105 4.708-2.897L1 5.383zM1 4v.217l7 4.2 7-4.2V4a1 1 0 0 0-1-1H2a1 1 0 0 0-1 1"/>
        <path d="M12.5 16a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7m.354-5.354 1.25 1.25a.5.5 0 0 1-.708.708L13 12.207V14a.5.5 0 0 1-1 0v-1.717l-.28.305a.5.5 0 0 1-.737-.676l1.149-1.25a.5.5 0 0 1 .722-.016"/>
        </svg>
    </a>
</div>
"""
    url = f"http://rocm.blogs.amd.com{grab_href(blog)[1:]}"
    title = blog.blog_title if hasattr(blog, "blog_title") else "No Title"

    if hasattr(blog, "myst"):
        description = blog.myst.get("html_meta", {}).get(
            "description lang=en", "No Description"
        )
    else:
        description = "No Description"
    title = f"{title} | ROCm Blogs"

    html = (
        html.replace("{URL}", url)
        .replace("{TITLE}", title)
        .replace("{TEXT}", description)
    )

    return html
    

def generate_blog_grid(
    blogs, output_file="latest_blogs.md", max_blogs=9, max_category=3
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

<!--
Updated 2024 October 10
Generated {datetime}
-->

<h1><a href="blog/atom.xml"><i class="fa fa-rss fa-rotate-270"></i></a> AMD ROCm™ Blogs</h1>

<script>
  const buttonWrapper = document.getElementById('buttonWrapper');

  const observer = new MutationObserver((mutationsList) => {
    for (const mutation of mutationsList) {
      if (mutation.type === 'attributes' && mutation.attributeName === 'data-mode') {
        console.log(`Data mode changed to: ${newMode}`);
        if (newMode === 'light') {
          buttonWrapper.style.setProperty('--original-background', 'white');
          buttonWrapper.style.setProperty('--hover-background-colour', 'white');
        } else {
          buttonWrapper.style.setProperty('--original-background', 'black');
          buttonWrapper.style.setProperty('--hover-background-colour', 'black');
        }
      }
    }
  });
</script>

<style>
  .bd-main .bd-content .bd-article-container {
    max-width: 100%;
  }
  .bd-sidebar-secondary {
    display: none;
  }
  .sd-card-large.sd-card {}
  #buttonWrapper:hover {
    border-color: hsla(231, 99%, 66%, 1);
    transform: scale(1.05);
    background-color: var(--hover-background-colour);
  }
  .small-sd-card-large.sd-card {}
  #buttonWrapper:hover {
    border-color: hsla(231, 99%, 66%, 1);
    transform: scale(1.05);
    background-color: var(--hover-background-colour);
  }
  #buttonWrapper {
    border-color: #A9A9A9;
    background-color: var(--original-background)
    text-align: center;
    font-weight: bold;
    font-size: 12px;
    border-radius: 1px;
    transition: transform 0.2s, border-color 0.2s;
  }
  h2 {
    margin: 0;
    font-size: 1.5em;
  }
  .container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px;
    box-sizing: border-box;
    width: 100%;
  }
  .read-more-btn {
    font-size: 20px;
    padding: 10px;
    font-weight: bold;
    cursor: pointer;
    display: inline-block;
    align-items: center;
    text-decoration: none;
    overflow: hidden;
    gap: 7px;
    display: block;
    text-align: left;
    margin-left: 0;
    margin-top: 10px;
  }
  .read-more-btn-small {
    font-size: 15px;
    padding: 10px;
    font-weight: bold;
    cursor: pointer;
    display: inline-block;
    align-items: center;
    text-decoration: none;
    overflow: hidden;
    gap: 7px;
    display: block;
    text-align: left;
    margin-left: 0;
    margin-top: 10px;
  }
  .arrows {
    font-size: 20px;
    display: inline-block;
    font-weight: bold;
    transition: transform 0.3s ease, color 0.3s ease, font-size 0.3s ease;
  }
  .read-more-btn:hover .arrows {
    transform: translateX(8px);
  }
  .arrows-small {
    font-size: 15px;
    display: inline-block;
    font-weight: bold;
    transition: transform 0.3s ease, color 0.3s ease, font-size 0.3s ease;
  }
  .read-more-btn-small:hover .arrows-small {
    transform: translateX(10px);
  }
  .date {
    font-size: 13px;
    font-weight: 300;
    line-height: 22.5px;
    text-transform: none;
    margin-bottom: 10px;
  }
  .paragraph {
    font-size: 16px;
    line-height: 24px;
    margin-bottom: 10px;
  }
  .large-sd-card-img-top.sd-card-img-top {
    width: 100%;
    height: 21vw;
    object-fit: cover;
  }
  .small-sd-card-img-top.sd-card-img-top {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
  .large-sd-card.sd-card-body {
    width: 100%;
    height: 15%;
  }
  .small-sd-card {
    width: 45px;
    height: 0;
    display: none;
  }
  .bd-content .sd-card .sd-card-footer {
    border-top: none;
  }
  .card-header {
    font-size: 16px;
    font-family: 'Arial', sans-serif;
    font-weight: bold;
    line-height: 1.4;
    margin-bottom: 10px;
  }
  .paragraph {
    font-size: 12px;
    font-family: 'Arial', sans-serif;
    line-height: 1.4;
    margin-bottom: 10px;
  }
</style>

<div class="container">
  <h2>Recent Posts</h2>
  <a href="blog.html">
    <button id="buttonWrapper">
      See All >>
    </button>
  </a>
</div>

::::{grid} 1 2 2 3
:margin 2
{grid_items}
::::

<div class="container">
  <h2>Ecosystems and partners</h2>
  <a href="blog/category/ecosystems-and-partners.html">
    <button id="buttonWrapper">
      See All >>
    </button>
  </a>
</div>

::::{grid} 1 2 2 3
:margin 2
{eco_grid_items}
::::

<div class="container">
  <h2>Applications & models</h2>
  <a href="blog/category/applications-models.html">
    <button id="buttonWrapper">
      See All >>
    </button>
  </a>
</div>

::::{grid} 1 2 2 3
:margin 2
{application_grid_items}
::::

<div class="container">
  <h2>Software tools & optimizations</h2>
  <a href="blog/category/software-tools-optimizations.html">
    <button id="buttonWrapper">
      See All >>
    </button>
  </a>
</div>

::::{grid} 1 2 2 3
:margin 2
{software_grid_items}
::::

<h2> Stay informed</h2>
<ul>
  <li><a href="blog/atom.xml"> Subscribe to our <i class="fa fa-rss fa-rotate-270"></i> RSS feed</a></li>
  <li><a href="https://github.com/ROCm/rocm-blogs"> Watch our GitHub repo </a></li>
</ul>

"""

    # remove the first new line

    index_template = index_template[1:]

    grid_items = []
    application_grid_items = []
    software_grid_items = []
    eco_grid_items = []

    holder = []

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

        image = grab_image(blog, href)

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


def author_attribution(blogs, minimum_date="September 1, 2024"):

    blogs_to_process = []

    for blog in blogs:

        if blog.date < datetime.strptime(minimum_date, "%B %d, %Y"):

            print(f"Skipping {blog.file_path}: Date is before {minimum_date}.")

            continue
        else:

            blogs_to_process.append(blog)
    print(f"Processing {len(blogs_to_process)} blogs...")

    print(f"Current working directory as of author attribution: {os.getcwd()}")

    print("\n-------------------------------------------------------------------\n")

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

                print(authors_html)

                # replace with just blogs/{author_name}.html

                # make it work with markdown and html
                # authors_html = f'<span style="font-size:0.7em;">{authors_string}</span>'

                print(f"Authors: {authors_string}")

                authors_html = f'<div class="date">{authors_string}</div>'
            print(f"Current working directory as of write file: {os.getcwd()}")

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

                quickshare_button = quickshare(blog)

                lines.insert(
                    line_number + 1,
                    """
<style>
.date {
  font-size: 13px;
  font-weight: 300;
  line-height: 22.5px;
  text-transform: none;
  margin-bottom: 10px;
}
</style>\n""",
                )

                lines.insert(line_number + 2, f"\n{authors_html}\n")

                lines.insert(line_number + 3, f"\n{quickshare_button}")

                with open(readme_file, "w", encoding="utf-8") as file:

                    # add date class style

                    file.writelines(lines)
                print(f"Author attribution inserted in '{readme_file}'.")
        else:

            print("No authors found in metadata.")


def main():

    root_directory = "blogs"  # Specify the root directory

    start_time = time.time()

    print(os.getcwd())

    # change cwd to parent directory

    os.chdir("..")

    if not os.path.exists(root_directory):

        print(f"The directory '{root_directory}' does not exist.")

        return
    print(
        f"Searching for 'readme.md' files in '{root_directory}' and subdirectories..."
    )

    readme_files = find_readme_files(root_directory)

    if not readme_files:

        print("No 'readme.md' files found.")

        return
    print(f"Found {len(readme_files)} 'readme.md' file(s).")

    blogs = create_blog_objects(readme_files)

    # Sort blogs by date

    sorted_blogs = sort_blogs_by_date(blogs)

    for blog in sorted_blogs:

        if hasattr(blog, "author"):

            print(blog.author)
    # Generate the grid for the top 15 latest blogs

    generate_blog_grid(sorted_blogs)

    author_attribution(sorted_blogs)

    with open("time.txt", "w") as f:

        f.write(f"Time taken: {time.time() - start_time} seconds")
    print(f"Time taken: {time.time() - start_time} seconds")

    # change back working directory

    os.chdir("blogs")


if __name__ == "__main__":
    main()
