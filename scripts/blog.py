import io
import os
import re
import yaml

from PIL import Image
from datetime import datetime


class Blog:

    def __init__(self, file_path, metadata, image=None):

        self.file_path = file_path
        self.metadata = metadata
        self.image = image
        self.image_paths = []

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

    def load_image_to_memory(self, image_path, format="PNG"):

        try:

            with Image.open(image_path) as img:

                buffer = io.BytesIO()

                img.save(buffer, format=format)

                buffer.seek(0)

                self.image = buffer.getvalue()

                print(f"Image loaded into memory; size: {len(self.image)} bytes.")
        except Exception as e:

            print(f"Error loading image to memory: {e}")

    def save_image(self, output_path):

        if self.image is None:

            print("No image data available in memory to save.")

            return
        try:

            with open(output_path, "wb") as file:

                file.write(self.image)

                print(f"Image saved to disk at: {output_path}")
        except Exception as error:

            print(f"Error saving image to disk: {error}")

    def save_image_path(self, image_path):

        self.image_paths.append(image_path)

        print(f"Image path saved: {image_path}")

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


@staticmethod
def find_readme_files(root_dir):

    readme_files = []

    for dirpath, dirnames, filenames in os.walk(root_dir):

        for filename in filenames:

            if filename.lower() == "readme.md":  # Case-insensitive matching

                full_path = os.path.join(dirpath, filename)

                readme_files.append(full_path)
    return readme_files


@staticmethod
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


@staticmethod
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


@staticmethod
def quicksort_desc(arr, low, high):
    if low < high:
        pivot_index = partition(arr, low, high)
        quicksort_desc(arr, low, pivot_index - 1)
        quicksort_desc(arr, pivot_index + 1, high)

    return arr


@staticmethod
def partition(arr, low, high):
    pivot = arr[high]["date_epoch"]
    x = low - 1
    for y in range(low, high):
        if arr[y]["date_epoch"] > pivot:
            x += 1
            arr[x], arr[y] = arr[y], arr[x]
    arr[x + 1], arr[high] = arr[high], arr[x + 1]
    return x + 1


@staticmethod
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

            original_width, original_height = img.size

            max_width, max_height = (1280, 420)
            scaling_factor = min(
                max_width / original_width, max_height / original_height
            )

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


def grab_href(blog):
    href = blog.file_path.replace(".md", ".html")
    href = href.replace("blogs", ".")

    # swap href \ to / for windows

    href = href.replace("\\", "/")

    return href


def grab_image(blog, href):
    
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

    if os.path.exists(href.replace(".html", ".md").replace("blogs", ".")):

        print(href.replace(".html", ".md").replace("blogs", "."))
    elif not os.path.exists(temp_image):

        print(f"Image1 {image} does not exist.")

        blog.save_image_path(image)

        image = "./images/generic.jpg"

        if os.path.exists(image_href):

            print(f"Image2 {image_href} exists.")

            blog.save_image_path(image_href)

            image = image_href.replace("./blogs", ".")

            blog.save_image_path(image)

            print("The current working directory is: ", os.getcwd())

            optimize_image(image)
        else:

            print(f"Image3 {image_href} does not exist.")
            image = "./images/generic.jpg"
    # check if images are in the relative blog directory

    else:

        print(f"Image4 {image} exists.")

        blog.save_image_path(image)

        print("The current working directory is: ", os.getcwd())
    optimize_image(image)

    print("-------------------------------------------------------------------\n")
    print(image)

    return image
