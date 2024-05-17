# check-tags.py
# read .md file and check if it has myst
import pathlib
import markdown
import csv
import os

# check_myst(file: str) -> None
# Check if .md file has myst in it's metadata
def check_myst(file: str) -> None:

    # read the markdown file
    data = pathlib.Path(file).read_text(encoding='utf-8')
    md = markdown.Markdown(extensions=['meta'])
    md.convert(data)

    # error flag, 0 = no error, 1 = error
    # you want it to print out all the errors, so you shouldnt exit on the first one
    if ('tags' in md.Meta):
        if ('html_meta' in md.Meta):
            return 0
        print(f'{file} has no MyST HTML metadata, please add that in, use https://myst-parser.readthedocs.io/en/v0.15.1/syntax/syntax.html#setting-html-metadata.')
        return 1
    
def main():
    
    # get all the markdown files from given bash command
    files = os.popen('git ls-files').read().split('\n')
    files = [file for file in files if file.endswith('.md')]

    print (f'Checking {len(files)} files')
    print ("files: " + str(files))

    # go through all the markdown files
    error = 0
    for file in files:

        if check_myst(file) == 1:
            error = 1
            
    exit(error)

if __name__ == '__main__':
    main()
