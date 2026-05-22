"""Execute the notebook end-to-end and embed static PNG outputs for Plotly figures.

This script:
- Inserts a small startup cell to force Plotly to use the `png` renderer via
  `kaleido`.
- Executes the notebook with nbconvert's ExecutePreprocessor.
- Writes an executed notebook `risk_analysis_visuals.executed.ipynb` and an HTML
  export `risk_analysis_visuals.executed.html`.

Usage:
    python scripts/execute_notebook.py
"""
import nbformat
from nbformat import NotebookNode
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter
import sys
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "risk_analysis_visuals.ipynb"
OUT_NB = NOTEBOOK.with_name("risk_analysis_visuals.executed.ipynb")
OUT_HTML = NOTEBOOK.with_name("risk_analysis_visuals.executed.html")


def make_startup_cell() -> NotebookNode:
    source = (
        "import plotly.io as pio\n"
        "pio.renderers.default = \"png\"\n"
        "try:\n"
        "    pio.kaleido.scope.default_format = \"png\"\n"
        "except Exception:\n"
        "    pass\n"
    )
    return nbformat.v4.new_code_cell(source)


def main():
    if not NOTEBOOK.exists():
        print(f"Notebook not found: {NOTEBOOK}")
        sys.exit(2)

    nb = nbformat.read(str(NOTEBOOK), as_version=4)

    # Prepend startup cell to force PNG renderer
    nb.cells.insert(0, make_startup_cell())

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    try:
        ep.preprocess(nb, {'metadata': {'path': str(NOTEBOOK.parent)}})
    except Exception as e:
        print("Execution failed:", e)
        # still write partial output
        nbformat.write(nb, str(OUT_NB))
        sys.exit(1)

    nbformat.write(nb, str(OUT_NB))
    print("Wrote executed notebook:", OUT_NB)

    # Export to HTML
    html_exporter = HTMLExporter()
    body, _ = html_exporter.from_notebook_node(nb)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(body)
    print("Wrote HTML export:", OUT_HTML)


if __name__ == "__main__":
    main()
