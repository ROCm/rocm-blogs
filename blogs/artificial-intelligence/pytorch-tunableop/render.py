#!/usr/bin/env python3

import jinja2
import ast
import inspect
import subprocess


def include_source(name, func=None):
    with open(name, "r") as f:
        source = f.read()

    if func:
        tree = ast.parse(source)

        # Search for FunctionDef nodes with the specified function name
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func:
                # Return the source code for the found function
                # Retrieve any decorator lines first
                decorator_lines = ""
                for decorator in node.decorator_list:
                    decorator_lines += f"@{ast.get_source_segment(source, decorator)}\n"

                # Retrieve function body next
                source = decorator_lines + ast.get_source_segment(source, node)
                found = True
                break

        if not found:
            raise ValueError(f"Unable to find function: {func}")

    source = source.strip()
    return source


def main():
    src = "README.mdt"
    dest = "README.md"

    env = jinja2.Environment(loader=jinja2.FileSystemLoader("."))
    env.globals["include_source"] = include_source

    template = env.get_template(src)

    output = template.render()

    with open(dest, "w") as f:
        f.write(output)

    subprocess.run(
        [
            "markdownlint-cli2",
            "--fix",
            "--config",
            "../../../.markdownlint.yaml",
            "README.md",
        ]
    )

    print(f"Rendered {src} -> {dest}")


if __name__ == "__main__":
    main()
