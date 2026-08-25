"""Shared Modal image/app definition for the qr vs qr-slim comparison.

The DESC source tree is mounted read-only and put on PYTHONPATH rather than
pip-installed, so a code edit needs no image rebuild. Dependencies come from
the repo's own requirements.txt; jax is then upgraded in place to the CUDA
build at the same version the local env uses.
"""

import os
import pathlib

import modal

REPO = pathlib.Path(__file__).parent.parent
JAX_VERSION = "0.9.2"

app = modal.App("desc-qr-slim")

_base = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install_from_requirements(str(REPO / "requirements.txt"))
)

gpu_image = (
    _base.pip_install(f"jax[cuda12]=={JAX_VERSION}")
    .env({"PYTHONPATH": "/root/DESC", "JAX_ENABLE_X64": "1"})
    .add_local_dir(str(REPO / "desc"), "/root/DESC/desc")
)

cpu_image = _base.env(
    {"PYTHONPATH": "/root/DESC", "JAX_PLATFORMS": "cpu"}
).add_local_dir(str(REPO / "desc"), "/root/DESC/desc")

# Results survive between runs here.
results = modal.Volume.from_name("desc-qr-slim-results", create_if_missing=True)
RESULTS_DIR = "/results"

# HARD CONCURRENCY CAP. The workspace GPU limit is 10. Every benchmark function
# uses single-use containers (one container per measurement, for cold-process
# timing and peak-memory isolation), so a `.map()` over N configs will try to
# start N containers at once unless the function itself caps them. Without this
# a 90-point sweep means 90 simultaneous A100s. Keep well under the limit so
# other work in the workspace is not starved.
# L=20 with jac_chunk=1000 peaked at 33.8 GB and L=12 at 2.3 GB, so the 40GB
# card is sufficient for everything up to L=20. Override with GPU="A100-80GB"
# only for the large-n kernel sweeps (n>=21007), which do need the headroom.
# Per-run override: DESC_BENCH_GPU=A100-80GB modal run -m modal_bench.solve ...
# The decorator is evaluated at import time in the LOCAL process, so setting the
# env var on the modal command line is enough.
GPU = os.environ.get("DESC_BENCH_GPU", "A100-40GB")

MAX_GPU_CONTAINERS = 4
MAX_CPU_CONTAINERS = 8

# init_gpu / build_R / gram_probe live in _bench_core, which imports no modal and
# can therefore also run as a standalone subprocess inside a container.
