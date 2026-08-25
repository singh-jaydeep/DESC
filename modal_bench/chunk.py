"""What jac_chunk_size="auto" resolves to, and what actually fits at L=20.

``objective_funs.py:521``:

    estimated_memory_usage = 2.4e-7 * dim_f * dim_x + 1        # GB
    max_chunk_size = round((avail_mem / estimated - 0.22) / 0.85 * dim_x)

Two things to check against the card rather than against arithmetic:

1. what ``avail_mem`` DESC actually reports on an A100-80GB, and hence what
   ``auto`` picks at each resolution;
2. what peak a Jacobian evaluation really costs at a chosen chunk size -- the
   heuristic's estimate was 79.5 GB at L=16 where the measured peak was 54.4 GB,
   so it is not accurate enough to size the run by.

Measures the peak of ONE jac evaluation, which is all that is needed: the L=16
solve showed the Jacobian sets the high-water mark and the alpha loop never
touches it.
"""

import json

from . import ledger
from .common import GPU, MAX_GPU_CONTAINERS, app, gpu_image


@app.function(
    image=gpu_image,
    gpu=GPU,
    timeout=7200,
    memory=32768,
    single_use_containers=True,
    max_containers=MAX_GPU_CONTAINERS,
    retries=0,
)
def probe(spec: dict):
    """Time and measure one Jacobian evaluation at a given chunk size."""
    from ._bench_core import init_gpu

    jax = init_gpu()
    import time

    import numpy as np

    from desc import config as desc_config
    from desc.examples import get
    from desc.objectives import (
        ForceBalance,
        ObjectiveFunction,
        get_fixed_boundary_constraints,
    )
    from desc.optimize._constraint_wrappers import LinearConstraintProjection

    name, L, chunk = spec["name"], spec["L"], spec["chunk"]
    dev = jax.local_devices()[0]
    GB = 1024**3
    out = dict(spec, avail_mem_GB=desc_config.get("avail_mem"))
    try:
        eq = get(name)
        eq.change_resolution(L=L, M=L, N=L, L_grid=2 * L, M_grid=2 * L, N_grid=2 * L)
        obj = ObjectiveFunction(ForceBalance(eq), jac_chunk_size=chunk)
        cons = get_fixed_boundary_constraints(eq)
        lcp = LinearConstraintProjection(obj, ObjectiveFunction(cons))
        lcp.build(verbose=0)

        out["dim_f"] = int(obj.dim_f)
        out["dim_x_full"] = int(obj.dim_x)
        out["n_reduced"] = int(lcp._dim_x_reduced)
        out["chunk_resolved"] = obj._jac_chunk_size
        est = 2.4e-7 * obj.dim_f * obj.dim_x + 1
        out["heuristic_estimate_GB"] = est

        x = lcp.x(eq)
        t0 = time.perf_counter()
        J = lcp.jac_scaled_error(x)
        jax.block_until_ready(J)
        out["jac_s"] = time.perf_counter() - t0
        st = dev.memory_stats() or {}
        out["jac_peak_GB"] = st.get("peak_bytes_in_use", 0) / GB
        out["limit_GB"] = st.get("bytes_limit", 0) / GB
        out["J_GB"] = float(np.prod(J.shape)) * 8 / GB
        out["ok"] = True
    except Exception as e:
        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    return out


@app.local_entrypoint()
def main(cases: str = "", out: str = "modal_bench/chunk.json"):
    """Sweep jac_chunk_size and report what fits."""
    default = [
        ["precise_QA", 16, "auto"],  # reproduce what the solve run did
        ["precise_QA", 20, "auto"],  # what does auto pick when it must chunk?
        ["precise_QA", 20, 2000],
        ["precise_QA", 20, 1000],
        ["precise_QA", 20, 500],
    ]
    todo = json.loads(cases) if cases else default
    specs = [dict(name=c[0], L=c[1], chunk=c[2]) for c in todo]

    ledger.open_section(
        "jac_chunk_size: what auto picks, and what actually fits",
        dict(
            note="one jac evaluation per point; the L=16 solve showed jac sets "
            "the peak and the alpha loop never raises it"
        ),
    )
    ledger.table_header(
        "case          L  chunk_in  chunk_used   dim_f  n_red   est_GB  "
        "jac_peak_GB  jac_s"
    )
    rows = []
    for r in probe.map(specs, order_outputs=True, return_exceptions=True):
        if isinstance(r, Exception):
            msg = f"container failure: {type(r).__name__}: {str(r)[:160]}"
            print("  " + msg, flush=True)
            ledger.note("  " + msg)
            continue
        rows.append(r)
        if r.get("ok"):
            line = (
                f"{r['name']:<12}{r['L']:>3}  {str(r['chunk']):>8}  "
                f"{str(r['chunk_resolved']):>10}  {r['dim_f']:>6}  "
                f"{r['n_reduced']:>5}  {r['heuristic_estimate_GB']:>7.1f}  "
                f"{r['jac_peak_GB']:>11.2f}  {r['jac_s']:>6.1f}"
            )
        else:
            line = (
                f"{r['name']:<12}{r['L']:>3}  {str(r['chunk']):>8}  "
                f"FAILED {str(r.get('error'))[:80]}"
            )
        print(line, flush=True)
        ledger.row(r, line)
        with open(out, "w") as fh:
            json.dump(rows, fh, indent=2)
    ledger.table_end()
    print(f"wrote {out}", flush=True)
