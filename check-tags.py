# check-tags.py
# read .md file and compare it to the approved tags in the taglist.csv file and categories.csv file
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

def import_approved_categories() -> list:

    category_path = '.category.csv'
    approved_categories = []

    with open(category_path, 'r') as f:

        approved_categories = csv.DictReader(f)

        for row in approved_categories:
            approved_categories = row['categories']

    return approved_categories

# check_tags(file: str) -> None
# Grab the tags from the markdown file and compare them to the approved tags in the taglist.csv file.
def check_tags(file: str, approved_tags: list, approved_categories: list) -> None:

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

    if ('category' in md.Meta):
        md_category = md.Meta['category'][0].split(', ')

        for category in md_category:

            print(f'Checking {file} for category: {category}')

            if category not in approved_categories:
                print(f'{file} has an unapproved category: {category}. Please ensure the category matches the allowed categories. If needed, please raise a separate PR to update the category file.')
                error = 1

    return error
    
def main():
    approved_tags = import_approved_tags()
    approved_categories = import_approved_categories()


    # get all the markdown files from given bash command
    files = os.popen('git ls-files').read().split('\n')
    files = [file for file in files if file.endswith('.md')]

    print (f'Checking {len(files)} files')
    print ("files: " + str(files))

    # go through all the markdown files and check the tags
    error = 0

    for file in files:

        if check_tags(file, approved_tags, approved_categories) == 1:

            error = 1
            
    exit(error)

if __name__ == '__main__':
    main()
