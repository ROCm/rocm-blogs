import os
import pathlib
import re
from concurrent.futures import ThreadPoolExecutor
import markdown

def parse_myst_metadata(myst_list):
    myst_dict = {}

    pattern = re.compile(r'^"([^"]+)":\s*(.*)$')

    for item in myst_list:
        item = item.strip()
        if not item:
            continue
        match = pattern.match(item)
        if match:
            key, value = match.groups()
            value = value.strip()
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1].strip()
            myst_dict[key] = value
        else:
            if item.endswith(":"):
                key = item.rstrip(":").strip().strip('"')
                myst_dict[key] = ""
    return myst_dict


def check_metadata(files: list[str]) -> list[str]:

    total_missing = []
    thumbnail_names = {}  # Track thumbnail names across all files

    for file in files:
        print(f"Checking metadata for {file}")

        blog_skip = 0
    
        # Do not check authors or contributors
        if (
            "authors" in pathlib.Path(file).parts
            or "contributor-bios.md" in pathlib.Path(file).parts
        ):
            print(
                f"Skipping metadata check for {file} in authors subfolder or contributor-bios.md."
            )
            blog_skip = 1
    
        metadata_fields = {
            "blog_title",
            "thumbnail",
            "date",
            "author",
            "tags",
            "category",
            "language",
            "target_audience",
            "key_value_propositions",
        }
        amd_metadata_fields = {
            "amd_category",
            "amd_asset_type",
            "amd_blog_topic_categories",
            "amd_technical_blog_type",
            "amd_blog_hardware_platforms",
            "amd_blog_development_tools",
            "amd_blog_applications",
        }
    
        try:
            data = pathlib.Path(file).read_text(encoding="utf-8")
            md = markdown.Markdown(extensions=["meta"])
            md.convert(data)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            blog_skip = 1
            continue

        # check only blogs (NOT REDUNDANT)
        if "blogpost" not in md.Meta or md.Meta["blogpost"][0].lower() != "true":
            print(f"Skipping metadata check for {file} because 'blogpost' is not true")
            blog_skip = 1

        if blog_skip == 0:

            print("Blog has passed all previous checks, now checking for metadata")
        
            missing = []
        
            for field in metadata_fields:
                if field not in md.Meta:
                    if (
                        "ecosystems-and-partners" in pathlib.Path(file).parts
                        and field == "author"
                    ):
                        print("Author exempt from ecosystems and partners")
                        continue
                    missing.append(field)
        
            myst_content = {}
            if "myst" in md.Meta:
                myst_content = parse_myst_metadata(md.Meta["myst"])
        
            for field in amd_metadata_fields:
                if field not in myst_content:
                    missing.append(field)
        
            missing_text = ", ".join(missing)

            if len(missing) > 0:
                result = f"{file} is missing metadata field(s): {missing_text}. Please check guide-to-blogs-metadata.md"
                total_missing.append(result)

            # Check thumbnail metadata
            if "thumbnail" in md.Meta:
                thumbnail_value = md.Meta["thumbnail"][0].strip().strip('"').strip("'")
                # Skip if thumbnail is empty or too short
                if thumbnail_value and len(thumbnail_value) > 1:
                    # Extract just the filename from the path
                    thumbnail_filename = os.path.basename(thumbnail_value)
                    thumbnail_filename_lower = thumbnail_filename.lower()
                    thumbnail_name_without_ext = os.path.splitext(thumbnail_filename_lower)[0]

                    # Check if thumbnail contains directory structure (like 'images/' or 'image/')
                    if "/" in thumbnail_value or "\\" in thumbnail_value:
                        result = (
                            f"{file} has a thumbnail with directory path: '{thumbnail_value}'. "
                            f"Thumbnails should be filename only, not include paths like 'images/'. "
                            f"Place the image in the blog's images/ folder or central blogs/images/ folder."
                        )
                        total_missing.append(result)

                    # Check for obviously generic patterns
                    generic_patterns = [
                        r'^thumbnail$',           # Just "thumbnail"
                        r'^\d+$',                 # Just a number like "1"
                        r'^image\d*$',            # "image" or "image1"
                        r'^thumb\d*$',            # "thumb" or "thumb1"
                        r'^img\d*$',              # "img" or "img1"
                        r'^pic\d*$',              # "pic" or "pic1"
                        r'^[a-z]$',               # Single letter
                        r'^[a-z][a-z0-9]*$',      # Single letter followed by numbers
                        r'^[a-z][a-z0-9]*\d*$',   # Single letter followed by numbers and then optional numbers
                        r'^[a-z][a-z0-9]*\d*\d*$',   # Single letter followed by numbers and then optional numbers
                        r'^[a-z][a-z0-9]*\d*\d*\d*$',   # Single letter followed by numbers and then optional numbers
                        r'^[a-z]{,10}$', # Less than 10 characters
                    ]
                    is_generic_pattern = any(re.match(p, thumbnail_name_without_ext) for p in generic_patterns)

                    if is_generic_pattern:
                        result = (
                            f"{file} has a non-descriptive thumbnail name: '{thumbnail_value}'. "
                            f"Please use a name that relates to your blog content. "
                            f"For example, if your blog is 'Introducing AMD EVLM: Efficient Vision-Language Models', "
                            f"your thumbnail should be something like 'amd-evlm-blog.jpg'."
                        )
                        total_missing.append(result)

                    # Check for duplicate thumbnail names across blogs
                    if thumbnail_filename_lower in thumbnail_names:
                        result = (
                            f"{file} uses the same thumbnail name as {thumbnail_names[thumbnail_filename_lower]}: '{thumbnail_filename}'. "
                            f"Each blog should have a unique thumbnail name. "
                            f"Please rename to something that relates to your blog content."
                        )
                        total_missing.append(result)
                    else:
                        thumbnail_names[thumbnail_filename_lower] = file

    return total_missing


def main():
     # get all the markdown files from given bash command
    files = os.popen("git ls-files").read().split("\n")
    files = [file for file in files if file.endswith(".md")]

    print(f"Checking {len(files)} files")
    print("files: " + str(files))

    error_flag = 0
    errors = check_metadata(files)

    if errors:
        error_flag = 1
        for error in errors:
            print(error)

    if error_flag:
        exit(1)


def test():
    root_dir = os.getcwd()
    root = pathlib.Path(root_dir)
    candidates = list(root.rglob("README.md"))

    print(f"Function being run in: {root_dir}")

    def process_path(path: pathlib.Path) -> str | None:
        if path.is_file():
            return str(path.resolve())
        return None

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_path, candidates))
    readme_files = [result for result in results if result is not None]

    if not readme_files:
        raise FileNotFoundError("No 'README.md' files found.")

    print(f"Found {len(readme_files)} 'README.md' file(s).")

    error_flag = 0
    errors = check_metadata(readme_files)

    if errors:
        error_flag = 1
        for error in errors:
            print(error)

    if error_flag:
        exit(1)


if __name__ == "__main__":
    test()
