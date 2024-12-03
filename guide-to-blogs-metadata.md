---
blogpost: false
blog_title: "Guide to ROCm Blogs Metadata"
date: December 3, 2024
author: Danny Guan
thumbnail: ''
tags: 
category: 
language: English
myst:
html_meta:
"author": "Danny Guan"
"description lang=en": "Guide to ROCm Blogs Metadata"
"keywords": ""
"property=og:locale": "en_US"
---

# Guide to Blogs Metadata

## Metadata Structure

Ensure that you have all of the metadata fields as listed below. Correct metadata is required for internal linting and scraping tools, proper metadata is also key for internal and external search optimizations. Please note that the description field under html_meta, "description lang=en": "blog snippet text/abstract" will need to be changed with the description provided by the author, usually a brief synopsis of the blog (150 characters max), anything additional characters after the 150 mark will be cut.

```markdown
---
blogpost: true
blog_title: "Title of blog"
date: Date of blog
author: Author(s) of blog
thumbnail: 'Image name'
tags: LLM, PyTorch, AI/ML, Fine-Tuning
category: Applications & models
language: English
myst:
html_meta:
"author": "Author(s) of blog"
"description lang=en": "blog snippet text/abstract 150 Characters MAX"
"keywords": "Tag(s) associated with blog content"
"property=og:locale": "en_US"
---
```

## Adding Images to Blogs

When adding images to your blog, you need to ensure that you are placing it in the correct area, this area will be looked at by automated processes. If you are planning on adding an image to your blog file, ensure there is an images folder, if there isn't an images folder within your blog directory, you must add one. Image name is the name of the image including the image type. Do not add any paths, just add the image name, the automated tool will take care of everything. **We recommend authors place their images in their respective blog directory.**

Images can be placed in TWO places, these places are the following:

1. Within the directory of your/the blog
  Directory Structure: YOUR_BLOG_NAME/images/IMAGE_NAME
2. Within the centralized blog images folder
  blogs/images
