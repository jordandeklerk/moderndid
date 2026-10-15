"""Run the tests marked gpu on a Modal GPU."""

import subprocess
from pathlib import Path

import modal

# Since Modal imports this file again as /root/gpu_tests.py inside the container, parents[2] would raise there.
ROOT = Path(__file__).resolve().parent.parent.parent
REMOTE_ROOT = "/root/moderndid"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(ROOT), extras=["diddynamic", "didcont", "didhonest", "didml", "etwfe", "plots", "gpu", "test"])
    .add_local_dir(ROOT / "moderndid", f"{REMOTE_ROOT}/moderndid", ignore=["**/__pycache__"])
    .add_local_dir(ROOT / "tests", f"{REMOTE_ROOT}/tests", ignore=["**/__pycache__"])
    .add_local_file(ROOT / "pyproject.toml", f"{REMOTE_ROOT}/pyproject.toml")
)

app = modal.App("moderndid-gpu-tests", image=image)


@app.function(gpu="T4", timeout=30 * 60)
def gpu_tests():
    """Run the tests marked gpu and fail when CuPy cannot reach the GPU."""
    subprocess.run(["nvidia-smi"], check=True)
    # Since the tests skip when CuPy finds no GPU, a broken CUDA setup would otherwise pass with every test skipped.
    subprocess.run(["python", "-c", "import cupy; assert cupy.is_available()"], check=True)
    # The image does not mount the benchmarks package that the benchmark tests import.
    subprocess.run(
        ["python", "-m", "pytest", "-m", "gpu", "--ignore=tests/benchmarks", "-ra", "-p", "no:cacheprovider"],
        cwd=REMOTE_ROOT,
        check=True,
    )
