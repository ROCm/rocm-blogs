import os
import time

from blog import *
from blog_generation import blog_generation
from grid_generation import generate_blog_grid
from quick_share import quickshare


def main():
    root_directory = "blogs"  # Specify the root directory

    start_time = time.time()

    # do ls

    print("Current working directory: ", os.getcwd())

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

    blog_generation(sorted_blogs)

    with open("time.txt", "w") as f:

        f.write(f"Time taken: {time.time() - start_time} seconds")
    print(f"Time taken: {time.time() - start_time} seconds")

    # change back working directory

    os.chdir("blogs")


if __name__ == "__main__":
    main()
