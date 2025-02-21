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


def check_metadata(file: str) -> int:
    print(f"Checking metadata for {file}")

    # Do not check authors or contributors
    if (
        "authors" in pathlib.Path(file).parts
        or "contributor-bios.md" in pathlib.Path(file).parts
    ):
        print(
            f"Skipping metadata check for {file} in authors subfolder or contributor-bios.md."
        )
        return 0

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
        "amd_blog_type",
        "amd_technical_blog_type",
        "amd_developer_type",
        "amd_deployment",
        "amd_product_type",
        "amd_developer_tool",
        "amd_applications",
        "amd_industries",
        "amd_blog_releasedate",
    }

    try:
        data = pathlib.Path(file).read_text(encoding="utf-8")
        md = markdown.Markdown(extensions=["meta"])
        md.convert(data)
    except Exception as e:
        print(f"Error reading {file}: {e}")
        return 1

    # check only blogs (NOT REDUNDANT)
    if "blogpost" not in md.Meta or md.Meta["blogpost"][0].lower() != "true":
        print(f"Skipping metadata check for {file} because 'blogpost' is not true")
        return 0

    missing = []
    error = 0

    for field in metadata_fields:
        if field not in md.Meta:
            if (
                "ecosystems-and-partners" in pathlib.Path(file).parts
                and field == "author"
            ):
                print("Author exempt from ecosystems and partners")
                continue
            missing.append(field)
            error = 1

    myst_content = {}
    if "myst" in md.Meta:
        myst_content = parse_myst_metadata(md.Meta["myst"])

    for field in amd_metadata_fields:
        if field not in myst_content:
            missing.append(field)
            error = 1

    missing_text = ", ".join(missing)

    if error != 0:
        print(
            f"{file} is missing metadata field(s): {missing_text} (error flag: {error}). Please check guide-to-blogs-metadata.md"
        )
    return error


def main():
    file = input()

    error_flag = 0
    if check_metadata(file) == 1:
        error_flag = 1

    if error_flag:
        exit(1)


def test():
    root_dir = os.getcwd()
    root = pathlib.Path(root_dir)
    candidates = list(root.rglob("README.md"))

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
    for file in readme_files:
        if check_metadata(file) == 1:
            error_flag = 1

    if error_flag:
        exit(1)


if __name__ == "__main__":
    main()
