# metadata-check.py
# read .md file and make sure there is correct metadata
import pathlib
import markdown
import csv
import os

# check_metadata(file: str) -> None
def check_metadata(file: str) -> None:

    metadata_fields = {'blog_title', 'thumbnail', 'date', 'author', 'tags', 'category', 'language'};

    # read the markdown file
    try:
        data = pathlib.Path(file).read_text(encoding='utf-8')
        md = markdown.Markdown(extensions=['meta'])
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
        error = 1

    missing_text = ' '.join(missing)
    print(f'{file} is missing a metadata field: {missing_text} with error {error}')
    exit(error)

def main():

    file = input()
    check_metadata(file)

main()
