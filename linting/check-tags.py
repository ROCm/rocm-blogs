# check-tags.py
# read .md file and compare it to the approved tags in the taglist.csv
# file and categories.csv file
#
# Hard/soft fail behavior:
#   - When the env var CHANGED_FILES_PATH points at a file containing the list
#     of .md paths changed in the current PR (or push), errors found in those
#     files cause a hard failure (exit 1).
#   - Errors in any other (legacy) file are reported as ⚠️ warnings only and
#     do not fail the build. This prevents pre-existing issues from blocking
#     unrelated PRs.
#   - When CHANGED_FILES_PATH is unset (e.g. local `test()` runs), every error
#     is treated as blocking — same as before.
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


def _norm_path(p) -> str:
    """Normalize a path for comparison against the PR diff list.

    `git ls-files` and `git diff --name-only` both return repo-relative paths
    with forward slashes, so this is mostly defensive.
    """
    s = str(p).replace("\\", "/")
    if s.startswith("./"):
        s = s[2:]
    return s


def _load_changed_files():
    """Load the set of files in scope for hard-fail from CHANGED_FILES_PATH.

    Returns None when no list is configured, which means every error is
    blocking (legacy behavior — used by `test()` and direct local runs).
    """
    path = os.environ.get("CHANGED_FILES_PATH")
    if not path:
        print("ℹ️  CHANGED_FILES_PATH not set — every error will be blocking.")
        return None

    p = pathlib.Path(path)
    if not p.exists():
        print(f"⚠️  CHANGED_FILES_PATH={path} does not exist; treating all files as blocking.")
        return None

    with open(p, "r", encoding="utf-8") as f:
        entries = {_norm_path(line.strip()) for line in f if line.strip()}

    print(f"Loaded {len(entries)} changed .md file(s) in scope for hard-fail:")
    for e in sorted(entries):
        print(f"  - {e}")

    return entries


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

# check_tags(files, approved_tags, approved_categories, changed_files=None) -> int
# Grab the tags from the markdown files and compare them to the approved
# tags / categories. Errors in files listed in `changed_files` are blocking
# (return 1). Errors in any other file are reported as warnings only.
# If `changed_files` is None, every error is blocking (legacy behavior).
def check_tags(files: list[str], approved_tags: list, approved_categories: list, changed_files=None) -> int:

    logs_dir = pathlib.Path("logs")
    logs_dir.mkdir(exist_ok=True)

    log_filepath = logs_dir / f"check-tags.log"
    log_file_handle = open(log_filepath, "w", encoding="utf-8")

    hard_error = 0
    soft_warnings = 0

    def is_blocking(file_path: str) -> bool:
        if changed_files is None:
            return True
        return _norm_path(file_path) in changed_files

    def report(file_path: str, message: str) -> None:
        """Print + log an error, and decide whether it counts toward hard-fail."""
        nonlocal hard_error, soft_warnings
        if is_blocking(file_path):
            line = f"🔴 {message}"
            hard_error = 1
        else:
            line = f"⚠️  [soft-fail, file not in PR diff] {message}"
            soft_warnings += 1
        print(line)
        log_file_handle.write(line + "\n")

    for file in files:

        log_file_handle.write(f"Processing file: {file}\n")
        md = extract_metadata(file)

        log_file_handle.write(f"Extracted metadata: {md}\n")

        if not md:
            log_file_handle.write(f"No metadata found in {file}.\n")
        else:

            if "tags" in md and md["tags"]:
                md_tags = md["tags"].split(", ")

                log_file_handle.write(f"Tags: {md_tags}\n")

                # go through the tags in the markdown file and compare them to the
                # approved tags
                for tag in md_tags:

                    # not in approved tags
                    if tag not in approved_tags:
                        report(
                            file,
                            f"{file} has an unapproved tag: {tag}. Please ensure the tag matches the allowed taglist file. If needed, please raise a separate PR to update the taglist file.",
                        )
                    else:
                        log_file_handle.write(
                            f"🟢 {file} has an approved tag: {tag}\n"
                        )

            log_file_handle.write(f"Checking {file} for categories\n")
            log_file_handle.write(f"Categories: {md}\n")

            if "category" in md and md["category"]:
                md_category = md["category"].split(", ")

                for category in md_category:

                    if category not in approved_categories:
                        report(
                            file,
                            f"{file} has an unapproved category: {category}. Please ensure the category matches the allowed categories. If needed, please raise a separate PR to update the category file.",
                        )
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

            if not md.get("myst") or not md["myst"].get("html_meta"):
                report(
                    file,
                    f"{file} does not have the myst tag. Please ensure the myst tag is present in the markdown file.",
                )
                continue

            for tag in possible_tags:

                log_file_handle.write("-" * 20 + "\n")
                log_file_handle.write(f"{tag}\n")
                log_file_handle.write("-" * 20 + "\n")

                if tag in md["myst"]["html_meta"]:

                    log_file_handle.write(f"🔵 Found tag: {tag} in {file}, checking for correct values.\n")
                    tag_value = md["myst"]["html_meta"][tag]

                    if not tag_value:
                        report(file, f"{file} has an empty {tag} AMD Blog Tag field.")
                        continue

                    # Special split patterns for specific multi-part values
                    special_patterns = {
                        "Design, Simulation & Modeling": "Design, Simulation & Modeling",
                        "Virtex, Kintex & Artix FPGAs": "Virtex, Kintex & Artix FPGAs",
                        "Tools, Features, and Optimizations": "Tools, Features, and Optimizations"
                    }

                    md_tag = []
                    remaining = tag_value

                    # First, extract any special patterns
                    for pattern in special_patterns.values():
                        if pattern in remaining:
                            md_tag.append(pattern)
                            remaining = remaining.replace(pattern, "###PLACEHOLDER###")

                    # Then split the remaining parts
                    if remaining:
                        parts = [item.strip() for item in remaining.split(", ")]
                        # ignore ###PLACEHOLDER###
                        for part in parts:
                            if part != "###PLACEHOLDER###" and part:
                                md_tag.append(part)

                    log_file_handle.write(f"🟡 {tag} for {file}: {md_tag}\n")

                    if len(md_tag) < 1:
                        report(file, f"{file} has an empty {md_tag} AMD Blog Tag field.")

                    for entry in md_tag:

                        if len(entry) < 1:
                            report(file, f"{file} has an empty {entry} field.")

                        elif entry not in amd_tags[tag]:
                            print(f"Entry to check: '{entry}' (type: {type(entry)})")
                            print(f"Available tags: {amd_tags[tag]}")

                            closest_match = difflib.get_close_matches(entry, amd_tags[tag], n=1, cutoff=0.4)
                            print(f"Closest matches: {closest_match}")

                            if closest_match:
                                suggested = closest_match[0]
                                # Check if the suggested value is valid
                                if suggested in amd_tags[tag]:
                                    match_found = f"Please use '{suggested}' instead."
                                else:
                                    match_found = f"Did you mean '{suggested}'?"
                            else:
                                match_found = "No matches found."

                            print(f"Entry (detailed): {repr(entry)}")
                            for t in amd_tags[tag]:
                                if len(t) == len(entry) or abs(len(t) - len(entry)) <= 2:
                                    print(f"Potential match (detailed): {repr(t)}")

                            report(
                                file,
                                f"{file} has an unapproved {tag}: '{entry}'. {match_found}",
                            )
                else:
                    log_file_handle.write(
                        f"🟡 {file} does not have the {tag}. Please ensure {tag} is present in the markdown file.\n"
                    )

            log_file_handle.write("=" * 20 + "\n")

    # Summary line so the build log makes the situation obvious.
    summary = (
        f"\n=== check-tags summary ===\n"
        f"Hard (blocking) errors: {'YES' if hard_error else 'none'}\n"
        f"Soft warnings on legacy files: {soft_warnings}\n"
    )
    print(summary)
    log_file_handle.write(summary)

    return hard_error

def main():
    approved_tags = import_approved_tags()
    approved_categories = import_approved_categories()
    changed_files = _load_changed_files()

    # get all the markdown files from given bash command
    files = os.popen("git ls-files").read().split("\n")
    files = [file for file in files if file.endswith(".md")]

    print(f"Checking {len(files)} files")
    print("files: " + str(files))

    # go through all the markdown files and check the tags
    error = check_tags(files, approved_tags, approved_categories, changed_files=changed_files)

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
    # Local test() runs intentionally pass changed_files=None so everything is
    # treated as blocking — same as before the soft-fail change.
    if check_tags(readme_files, approved_tags, approved_categories) == 1:
        error_flag = 1

    if error_flag:
        exit(1)


if __name__ == "__main__":
    main()
