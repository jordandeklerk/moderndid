"""Run the generated example notebooks top to bottom, as Jupyter notebooks and as marimo apps."""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOKS = Path(__file__).resolve().parent


def main():
    """Run every example notebook, or the named subset, and stop at the first failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="Notebook names such as example_npiv; defaults to every example.")
    parser.add_argument("--timeout", type=int, default=1200, help="Maximum seconds per cell (default: 1200).")
    args = parser.parse_args()
    paths = [NOTEBOOKS / f"{name}.ipynb" for name in args.names] or sorted(NOTEBOOKS.glob("example_*.ipynb"))
    # Putting this interpreter first on PATH makes a notebook's !pip install packages into this environment.
    os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"
    for path in paths:
        print(f"Running {path.name} with Jupyter", flush=True)
        notebook = nbformat.read(path, as_version=4)
        # Starting the kernel in the notebook's folder imports the installed package as a reader's runtime would.
        client = NotebookClient(
            notebook,
            timeout=args.timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(path.parent)}},
        )
        client.create_kernel_manager().kernel_spec.argv = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        client.execute()
        print(f"Running {path.name} with marimo", flush=True)
        with tempfile.TemporaryDirectory() as folder:
            app = Path(folder) / f"{path.stem}.py"
            subprocess.run([sys.executable, "-m", "marimo", "convert", str(path), "-o", str(app)], check=True)
            subprocess.run([sys.executable, str(app)], check=True, cwd=folder)


if __name__ == "__main__":
    main()
