# metadata-check.py
# read .md file and make sure there is correct metadata
import pathlib
import markdown
import csv
import os


# check_metadata(file: str) -> None
def check_metadata(file: str) -> None:

    # Check if the file is in the authors subfolder or contributor-bios.md
    if (
        "authors" in pathlib.Path(file).parts
        or "contributor-bios.md" in pathlib.Path(file).parts
    ):
        print(
            f"Skipping metadata check for {file} in authors subfolder or contributor-bios.md."
        )
        return

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

    # read the markdown file
    try:
        data = pathlib.Path(file).read_text(encoding="utf-8")
        md = markdown.Markdown(extensions=["meta"])
        md.convert(data)
    except:
        return 1

    # error flag, 0 = no error, 1 = error
    # you want it to print out all the errors, so you shouldnt exit on the first one
    missing = []
    error = 0

    for field in metadata_fields:
        if field not in md.Meta:
            missing.append(field)
            if (
                "ecosystems-and-partners" in pathlib.Path(file).parts
                and missing == "author"
            ):
                print("Author exempt from ecosystems and partners")
                pass
            else:
                error = 1

    missing_text = " ".join(missing)
    print(
        f"{file} is missing a metadata field: {missing_text} with error {error}, please take a look at guide-to-blogs-metadata.md"
    )
    exit(error)


def main():

    file = input()
    check_metadata(file)


main()
