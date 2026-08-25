"""Kernel-level sweep: one alpha factorization, three routes, on an A100.

The unit timed is exactly the body of the LM alpha loop -- the thing that differs
between ``tr_method="qr"`` (master) and ``"qr-fixed"``:
factorize ``[R; sqrt(alpha)*I]`` and produce ``(Rtil, Q^T z)``.

Two points of method, both learned the hard way:

* ONE CONTAINER PER ``n``, with one SUBPROCESS per (method, block) inside it.
  Each measurement still gets a clean jit cache and a clean peak-memory
  high-water mark (neither has a reset API), but every method at a given ``n``
  is timed on the same physical card. Modal hands out both A100-SXM4-80GB and
  A100 80GB PCIe, which differ in clocks and bandwidth -- one container per
  measurement would let ``qr`` land on one variant and ``qr-fixed`` on the other
  and report the difference as a speedup.
* Timing and peak memory depend only on SHAPES, so a synthetic ``R`` suffices
  for those. Accuracy depends on values, so the Gram residual here is
  indicative and is confirmed against solve-captured ``R`` separately.
"""

import json

from . import ledger
from .common import MAX_GPU_CONTAINERS, RESULTS_DIR, app, gpu_image, results

# Reduced sizes n = LinearConstraintProjection._dim_x_reduced, measured for
# precise_QA / W7-X / HELIOTRON at L=M=N in {12,16,20,25}. See shapes.json.
SIZES = {
    3434: "precise_QA/W7-X L12",
    5009: "HELIOTRON L12",
    7602: "precise_QA/W7-X L16",
    11166: "HELIOTRON L16",
    14242: "precise_QA/W7-X L20",
    21007: "HELIOTRON L20",
    26896: "precise_QA/W7-X L25",
    38830: "HELIOTRON L25",
}
BLOCKS = [128, 256, 512, 1024, 2048]


@app.function(
    image=gpu_image,
    gpu="A100-80GB",
    timeout=7200,
    single_use_containers=True,
    max_containers=MAX_GPU_CONTAINERS,  # workspace GPU limit is 10; stay well under
    volumes={RESULTS_DIR: results},
    retries=0,
)
def sweep_n(spec: dict):
    """All (method, block) points at one n, each in its own subprocess."""
    import subprocess
    import sys
    import time

    n, alpha, cond, reps = spec["n"], spec["alpha"], spec["cond"], spec["reps"]
    blocks = spec.get("blocks", BLOCKS)

    gpu_name = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()

    methods = spec.get("methods", ["qr", "qr-fixed"])
    points = [dict(method="qr", n=n, block=None)] if "qr" in methods else []
    for b in blocks:
        if b > n:
            continue
        for m in [x for x in methods if x != "qr"]:
            points.append(dict(method=m, n=n, block=b))

    rows = []
    for p in points:
        cfg = dict(
            p,
            alpha=alpha,
            cond=cond,
            reps=reps,
            seed=spec.get("seed", 0),
            mem_fraction=spec.get("mem_fraction", "0.95"),
            deterministic=spec.get("deterministic", False),
        )
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-m", "modal_bench._bench_core", json.dumps(cfg)],
            capture_output=True,
            text=True,
            cwd="/root",
        )
        rec = None
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT "):
                rec = json.loads(line[len("RESULT ") :])
        if rec is None:
            rec = dict(
                cfg,
                ok=False,
                error=f"subprocess exit {proc.returncode}",
                traceback=(proc.stderr or proc.stdout)[-800:],
            )
        rec["gpu_name"] = gpu_name
        rec["subprocess_s"] = time.perf_counter() - t0
        rows.append(rec)
    return dict(n=n, gpu_name=gpu_name, rows=rows)


def _fmt(r):
    if r.get("ok"):
        return (
            f"n={r['n']:6d} {r['method']:10s} b={str(r['block']):>5s} "
            f"{r['time_s']*1e3:9.1f} ms  peak={r['peak_GB']:6.2f} GB  "
            f"gram={r['gram_rel']:.2e}  spread={r['time_spread']*100:4.1f}%  "
            f"limit={r['limit_GB']:.0f}GB"
        )
    return (
        f"n={r['n']:6d} {r['method']:10s} b={str(r['block']):>5s}  "
        f"FAILED  {str(r.get('error'))[:90]}"
    )


@app.local_entrypoint()
def main(
    sizes: str = "3434,14242",
    alpha: float = 2.2e-14,
    cond: float = 1e10,
    reps: int = 5,
    blocks: str = "",
    methods: str = "",
    deterministic: bool = False,
    out: str = "modal_bench/kernel_a100_80.json",
):
    """Sweep one alpha factorization over sizes, methods and block widths."""
    ns = [int(x) for x in sizes.split(",")]
    bl = [int(x) for x in blocks.split(",")] if blocks else BLOCKS
    ms = methods.split(",") if methods else None
    specs = [
        dict(
            n=n,
            alpha=alpha,
            cond=cond,
            reps=reps if n <= 15000 else 3,
            blocks=bl,
            deterministic=deterministic,
            **({"methods": ms} if ms else {}),
        )
        for n in ns
    ]

    ledger.open_section(
        "Kernel sweep: qr vs qr-fixed (A100-80GB)",
        dict(
            sizes=ns,
            blocks=bl,
            alpha=alpha,
            cond=cond,
            reps=reps,
            mem_fraction=0.95,
            deterministic=deterministic,
            methods=methods or "all",
            note="one container per n; one subprocess per measurement",
        ),
    )
    ledger.table_header(
        "     n  method       block     time_ms    peak_GB    gram_rel  spread  gpu"
    )

    all_rows = []
    for res in sweep_n.map(specs, order_outputs=True, return_exceptions=True):
        if isinstance(res, Exception):
            msg = f"container failure: {type(res).__name__}: {str(res)[:200]}"
            print("  " + msg, flush=True)
            ledger.note("  " + msg)
            continue
        print(f"--- n={res['n']} on {res['gpu_name']} ---", flush=True)
        for r in res["rows"]:
            print(_fmt(r), flush=True)
            all_rows.append(r)
            if r.get("ok"):
                line = (
                    f"{r['n']:6d}  {r['method']:10s} {str(r['block']):>5s}  "
                    f"{r['time_s']*1e3:10.1f}  {r['peak_GB']:9.2f}  "
                    f"{r['gram_rel']:10.2e}  {r['time_spread']*100:5.1f}%  "
                    f"{r['gpu_name']}"
                )
            else:
                line = (
                    f"{r['n']:6d}  {r['method']:10s} {str(r['block']):>5s}  "
                    f"FAILED {str(r.get('error'))[:70]}"
                )
            ledger.row(r, line)
        with open(out, "w") as fh:
            json.dump(all_rows, fh, indent=2)
    ledger.table_end()
    print(f"wrote {out} and modal_bench/ledger.md", flush=True)
