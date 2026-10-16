"""Execute the documented Python/R comparisons and save their actual outputs."""

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

EXAMPLES = (
    "r_staggered",
    "r_two_period",
    "r_triple",
    "r_continuous",
    "r_intertemporal",
    "r_honest",
    "r_etwfe",
    "r_npiv",
)
ROOT = Path(__file__).resolve().parents[3]


def main():
    """Run all examples, or the named subset, with the current Python interpreter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("examples", nargs="*", choices=EXAMPLES, help="Example names; defaults to all of them.")
    parser.add_argument("--timeout", type=int, default=1200, help="Maximum seconds per cell (default: 1200).")
    args = parser.parse_args()

    for name in args.examples or EXAMPLES:
        path = ROOT / "docs" / "source" / "r_comparisons" / f"{name}.ipynb"
        print(f"Executing {path.relative_to(ROOT)}", flush=True)
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=args.timeout,
            kernel_name="python3",
            coalesce_streams=True,
            resources={"metadata": {"path": str(ROOT)}},
        )
        # Use this environment even when a user's default python3 kernel points elsewhere.
        manager = client.create_kernel_manager()
        manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client.execute()
        notebook.metadata["mystnb"] = {"execution_mode": "off"}
        nbformat.validate(notebook)
        # Preserve the previous successful outputs if any cell raises an exception.
        nbformat.write(notebook, path)
        print(f"Saved {name}", flush=True)


if __name__ == "__main__":
    main()
