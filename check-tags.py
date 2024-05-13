# check-tags.py
# read .md file and compare it to the approved tags in the taglist.csv file
import pathlib
import markdown
import csv
import os

# import_approved_tags() -> list
# Import the approved tags from the taglist.csv file.
def import_approved_tags() -> list:

    tags_path = '.taglist.csv'
    approved_tags = []

    # read the approved tags from the taglist.csv file
    with open(tags_path, 'r') as f:

        approved_tags = csv.DictReader(f)
        approved_tags = [row['tags'] for row in approved_tags]
        approved_tags = approved_tags[0]

    return approved_tags

# check_tags(file: str) -> None
# Grab the tags from the markdown file and compare them to the approved tags in the taglist.csv file.
def check_tags(file: str, approved_tags: list) -> None:

    # read the markdown file
    data = pathlib.Path(file).read_text(encoding='utf-8')
    md = markdown.Markdown(extensions=['meta'])
    md.convert(data)

    # error flag, 0 = no error, 1 = error
    # you want it to print out all the errors, so you shouldnt exit on the first one
    error = 0
    if ('tags' in md.Meta):
        md_tags = md.Meta['tags'][0].split(', ')

        # go through the tags in the markdown file and compare them to the approved tags
        for tag in md_tags:

            # not in approved tags
            if tag not in approved_tags:
                print(f'{file} has an unapproved tag: {tag}. Please ensure the tag matches the allowed taglist file. If needed, please raise a separate PR to update the taglist file.')
                error = 1

    return error
    
def main():
    approved_tags = import_approved_tags()

    # get all the markdown files from given bash command
    files = os.popen('git ls-files').read().split('\n')
    files = [file for file in files if file.endswith('.md')]

    print (f'Checking {len(files)} files')
    print ("files: " + str(files))

    # go through all the markdown files and check the tags
    error = 0

    for file in files:

        if check_tags(file, approved_tags) == 1:

            error = 1
            
    exit(error)

if __name__ == '__main__':
    main()
