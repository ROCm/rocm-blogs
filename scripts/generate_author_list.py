import os
import re

authors_dir = "blogs/authors"
output_file = ".authorlist.txt"
author_entries = []

for filename in os.listdir(authors_dir):
    if filename.endswith(".md"):
        path = os.path.join(authors_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

            # Try extracting from <meta name="description" content="...">
            meta_match = re.search(r'<meta\s+name="description"\s+content="([^"]+)"', content, re.IGNORECASE)
            if meta_match:
                name = meta_match.group(1).strip()
            else:
                # Fallback: Try extracting from "# Author Name"
                heading_match = re.search(r"^#\s+(.+)", content, re.MULTILINE)
                if heading_match:
                    name = heading_match.group(1).strip()
                else:
                    name = None

            if name:
                parts = name.split()
                first = parts[0]
                last = parts[-1] if len(parts) > 1 else ''
                author_entries.append((first, last))

# Write unique, sorted names to output file
with open(output_file, "w", encoding="utf-8") as f:
    for first, last in sorted(set(author_entries)):
        f.write(first + "\n")
        f.write(last + "\n")
        
print(f"Wrote {len(set(author_entries))} authors to {output_file}")
