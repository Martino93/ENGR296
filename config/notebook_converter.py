import sys
import nbformat
from nbconvert import MarkdownExporter


def convert_notebook_to_markdown(notebook_path, output_path):
    # Load the notebook file
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Create a Markdown exporter
    md_exporter = MarkdownExporter()

    # Convert the notebook to Markdown
    (body, resources) = md_exporter.from_notebook_node(nb)

    # Write the Markdown output to a file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"Notebook has been converted to Markdown and saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_notebook.py notebook.ipynb output.md")
    else:
        notebook_file = sys.argv[1]
        output_file = sys.argv[2]
        convert_notebook_to_markdown(notebook_file, output_file)


# python config/notebook_converter.py Models/cs235_phase_1.ipynb Models/cs235_phase_1.md
