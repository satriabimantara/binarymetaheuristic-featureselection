"""Execute every notebook in order, in place (outputs are kept in the .ipynb).

python scripts/run_notebooks.py [pattern]      e.g.  python scripts/run_notebooks.py 0[2-9]_
Raw results are cached in results/raw, so a re-run only recomputes what is missing.
"""
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NB = Path(__file__).resolve().parents[1] / "notebooks"
pattern = re.compile(sys.argv[1]) if len(sys.argv) > 1 else None
for path in sorted(NB.glob("*.ipynb")):
    if pattern and not pattern.search(path.name):
        continue
    t = time.time()
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(NB)}}).execute()
    nbformat.write(nb, path)
    print(f"{path.name}: {time.time() - t:.0f}s", flush=True)
