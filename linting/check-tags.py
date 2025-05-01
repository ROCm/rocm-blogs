# check-tags.py
# read .md file and compare it to the approved tags in the taglist.csv
# file and categories.csv file
from concurrent.futures import ThreadPoolExecutor
import csv
import difflib
import os
import pathlib
import re
import markdown
import logging
import re
import yaml

# import_approved_tags() -> list
# Import the approved tags from the taglist.csv file.
def import_approved_tags() -> list:

    print(pathlib.Path.cwd())

    tags_path = "linting/csv/taglist.csv"

    approved_tags = []

    # read the approved tags from the taglist.csv file
    with open(tags_path, "r") as f:

        approved_tags = csv.DictReader(f)
        approved_tags = [row["tags"] for row in approved_tags]
        approved_tags = approved_tags[0]

    return approved_tags

def import_approved_categories() -> list:

    category_path = "linting/csv/category.csv"

    approved_categories = []

    with open(category_path, "r") as f:

        approved_categories = csv.DictReader(f)

        for row in approved_categories:
            approved_categories = row["categories"]

    return approved_categories

def import_amd_tags() -> dict:
    amd_tags_path = [
        "amd_blog_applications.csv",
        "amd_blog_development_tools.csv",
        "amd_blog_hardware_platforms.csv",
        "amd_technical_blog_type.csv",
        "amd_blog_topic_categories.csv",
    ]

    amd_tags = dict()

    for path in amd_tags_path:
        full_path = f"./linting/csv/{path}"
        tag_key = path[: len(path) - 4]
        
        with open(full_path, "r") as f:
            approved_tags = csv.DictReader(f)
            
            amd_tags[tag_key] = []
            
            for row in approved_tags:
                # The value is currently a string representation of a list
                tag_value = row[tag_key]

                if tag_value.startswith('[') and tag_value.endswith(']'):
                    tag_list = [item.strip().strip("'\"") for item in re.split("(?<!Design)(?<!Tools)(?<!Features)(?<!Virtex),", tag_value[1:-1])]
                    amd_tags[tag_key].extend(tag_list)
                else:
                    if tag_value and tag_value not in amd_tags[tag_key]:
                        amd_tags[tag_key].append(tag_value)

    return amd_tags

def extract_metadata(blog_path) -> dict:
        """Extract metadata from the blog files."""

        logs_dir = pathlib.Path("logs")
        logs_dir.mkdir(exist_ok=True)

        log_filepath = logs_dir / f"check-tags-extract-metadata.log"
        log_file_handle = open(log_filepath, "w", encoding="utf-8")

        yaml_pattern = re.compile(r"^---\n(.*?)\n---", re.DOTALL)

        if not blog_path:
            log_file_handle.write(
                "No blog paths provided. Please check the configuration."
            )
            return {}

        file_path = blog_path
        log_file_handle.write(f"Processing file: {file_path}\n")
        log_file_handle.write("Extracting metadata...\n")

        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        match = yaml_pattern.match(content)

        if match:
            yaml_content = match.group(1)

            try:
                metadata = yaml.safe_load(yaml_content)

                log_file_handle.write(f"Extracted metadata: {metadata}\n")

                return metadata
            except yaml.YAMLError as error:
                log_file_handle.write(f"Error parsing YAML: {error}\n")
                
                return {}
        else:
            log_file_handle.write("No YAML front matter found.\n")
            
            return {}

# check_tags(file: str) -> None
# Grab the tags from the markdown file and compare them to the approved
# tags in the taglist.csv file.
def check_tags(files: list[str], approved_tags: list, approved_categories: list) -> None:

    logs_dir = pathlib.Path("logs")
    logs_dir.mkdir(exist_ok=True)

    log_filepath = logs_dir / f"check-tags.log"
    log_file_handle = open(log_filepath, "w", encoding="utf-8")

    error = 0

    for file in files:

        log_file_handle.write(f"Processing file: {file}\n")
        md = extract_metadata(file)

        log_file_handle.write(f"Extracted metadata: {md}\n")

        if not md:
            log_file_handle.write(f"No metadata found in {file}.\n")
        else:

            if "tags" in md:
                md_tags = md["tags"].split(", ")

                log_file_handle.write(f"Tags: {md_tags}\n")

                # go through the tags in the markdown file and compare them to the
                # approved tags
                for tag in md_tags:

                    # not in approved tags
                    if tag not in approved_tags:
                        print(
                            f"🔴 {file} has an unapproved tag: {tag}. Please ensure the tag matches the allowed taglist file. If needed, please raise a separate PR to update the taglist file."
                        )
                        log_file_handle.write(
                            f"🔴 {file} has an unapproved tag: {tag}. Please ensure the tag matches the allowed taglist file. If needed, please raise a separate PR to update the taglist file.\n"
                        )
                        error = 1
                    else:
                        log_file_handle.write(
                            f"🟢 {file} has an approved tag: {tag}\n"
                        )

            log_file_handle.write(f"Checking {file} for categories\n")
            log_file_handle.write(f"Categories: {md}\n")

            if "category" in md:
                md_category = md["category"].split(", ")

                for category in md_category:

                    # print(f"Checking {file} for category: {category}")

                    if category not in approved_categories:
                        print(
                            f"🔴 {file} has an unapproved category: {category}. Please ensure the category matches the allowed categories. If needed, please raise a separate PR to update the category file."
                        )
                        log_file_handle.write(
                            f"🔴 {file} has an unapproved category: {category}. Please ensure the category matches the allowed categories. If needed, please raise a separate PR to update the category file.\n"
                        )
                        error = 1
                    else:
                        log_file_handle.write(
                            f"🟢 {file} has an approved category: {category}\n"
                        )

            amd_tags = import_amd_tags()

            log_file_handle.write(f"AMD Tags: {amd_tags}\n")

            possible_tags = [
                "amd_blog_applications",
                "amd_blog_development_tools",
                "amd_blog_hardware_platforms",
                "amd_technical_blog_type",
                "amd_blog_topic_categories",
            ]

            log_file_handle.write(f"🔵 Checking {file} for AMD Blog Tags\n")

            if "myst" not in md:
                print(
                    f"🔴 {file} does not have the myst tag. Please ensure the myst tag is present in the markdown file."
                )
                log_file_handle.write(
                    f"🔴 {file} does not have the myst tag. Please ensure the myst tag is present in the markdown file.\n"
                )
                error = 1
                continue

            for tag in possible_tags:
                
                log_file_handle.write("-" * 20 + "\n")
                log_file_handle.write(f"{tag}\n")
                log_file_handle.write("-" * 20 + "\n")

                if tag in md["myst"]["html_meta"]:

                    log_file_handle.write(f"🔵 Found tag: {tag} in {file}, checking for correct values.\n")
                    md_tag = re.split("(?<!Design)(?<!Tools)(?<!Features)(?<!Virtex), ", md["myst"]["html_meta"][tag])

                    log_file_handle.write(f"🟡 {tag} for {file}: {md_tag}\n")

                    if len(md_tag) < 1:
                        print(
                            f"🔴 {file} has an empty {md_tag} AMD Blog Tag field."
                        )
                        log_file_handle.write(
                            f"🔴 {file} has an empty {md_tag} AMD Blog Tag field.\n"
                        )
                        error = 1

                    for entry in md_tag:

                        if len(entry) < 1:
                            print(
                                f"🔴 {file} has an empty {entry} field."
                            )
                            log_file_handle.write(
                                f"🔴 {file} has an empty {entry} field.\n"
                            )
                            error = 1

                        elif entry not in amd_tags[tag]:
                            print(f"Entry to check: '{entry}' (type: {type(entry)})")
                            print(f"Available tags: {amd_tags[tag]}")

                            closest_match = difflib.get_close_matches(entry, amd_tags[tag], n=1, cutoff=0.4)
                            print(f"Closest matches: {closest_match}")

                            closest_match_str = " ".join(closest_match)

                            match_found = f"Did you mean ({closest_match_str}?)" if closest_match else "No matches found."
                            
                            print(f"Entry (detailed): {repr(entry)}")
                            for t in amd_tags[tag]:
                                if len(t) == len(entry) or abs(len(t) - len(entry)) <= 2:
                                    print(f"Potential match (detailed): {repr(t)}")
                            
                            print(f"🔴 {file} has an unapproved {tag}: {entry}. {match_found}")
                            log_file_handle.write(
                                f"🔴 {file} has an unapproved {tag}: {entry}. {match_found}\n"
                            )
                            log_file_handle.write(f"🔴 {file} has an unapproved {tag}: {entry}.\n")
                            error = 1
                else:
                    log_file_handle.write(
                        f"🟡 {file} does not have the {tag}. Please ensure {tag} is present in the markdown file.\n"
                    )
            
            log_file_handle.write("=" * 20 + "\n")

    return error

def main():
    approved_tags = import_approved_tags()
    approved_categories = import_approved_categories()

    # get all the markdown files from given bash command
    files = os.popen("git ls-files").read().split("\n")
    files = [file for file in files if file.endswith(".md")]

    print(f"Checking {len(files)} files")
    print("files: " + str(files))

    # go through all the markdown files and check the tags
    error = 0

    if check_tags(files, approved_tags, approved_categories) == 1:
        error = 1

    exit(error)

def test():
    root_dir = os.getcwd()
    root = pathlib.Path(root_dir).parent
    candidates = list(root.rglob("README.md"))

    print(root)

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
    approved_tags = import_approved_tags()
    approved_categories = import_approved_categories()
    if check_tags(readme_files, approved_tags, approved_categories) == 1:
        error_flag = 1

    if error_flag:
        exit(1)


if __name__ == "__main__":
    test()
