"""Compare the equilibria that different tr_methods actually converged to.

Two questions, and the second is what makes the first interpretable:

1. Is either solution PROBLEMATIC in its own right -- force-balance residual,
   flux surfaces that self-intersect (sign change in sqrt(g)), a wrong iota or
   volume?
2. Is the qr vs qr-fixed difference LARGER than the difference between two
   methods DESC already ships? ``svd`` is the reference: if qr-vs-qr-fixed sits
   inside qr-vs-svd, then qr-fixed is no more "different" than two accepted
   methods already are from each other, and the comparison has a yardstick
   rather than a bare number.

Loads the equilibria saved by ``solve.py`` from the results volume, so no solve
is re-run.
"""

import json

from . import ledger
from ._bench_core import init_gpu
from .common import GPU, MAX_GPU_CONTAINERS, RESULTS_DIR, app, gpu_image, results


@app.function(
    image=gpu_image,
    gpu=GPU,
    timeout=7200,
    memory=32768,
    single_use_containers=True,
    max_containers=MAX_GPU_CONTAINERS,
    volumes={RESULTS_DIR: results},
    retries=0,
)
def diagnose(files: list):
    """Physics diagnostics for each saved equilibrium, plus pairwise deltas."""
    jax = init_gpu()  # noqa: F841
    import numpy as np

    from desc.equilibrium import Equilibrium
    from desc.grid import LinearGrid, QuadratureGrid

    out = {"per_eq": {}, "pairs": {}}
    eqs = {}
    for f in files:
        try:
            eq = Equilibrium.load(f"{RESULTS_DIR}/{f}")
            eqs[f] = eq
            grid = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
            lg = LinearGrid(L=20, M=0, N=0, NFP=eq.NFP, axis=False)

            d = eq.compute(
                ["|F|", "<|F|>_vol", "sqrt(g)", "V", "R0/a", "W_B"], grid=grid
            )
            di = eq.compute(["iota"], grid=lg)
            sg = np.asarray(d["sqrt(g)"])
            Fv = float(np.asarray(d["<|F|>_vol"]))
            rec = dict(
                force_vol=Fv,
                force_max=float(np.abs(np.asarray(d["|F|"])).max()),
                volume=float(np.asarray(d["V"])),
                aspect=float(np.asarray(d["R0/a"])),
                W_B=float(np.asarray(d["W_B"])),
                # a sign change in sqrt(g) means the flux surfaces have folded
                sqrtg_min=float(sg.min()),
                sqrtg_max=float(sg.max()),
                sqrtg_sign_change=bool(sg.min() * sg.max() <= 0),
                iota_axis=float(np.asarray(di["iota"])[0]),
                iota_edge=float(np.asarray(di["iota"])[-1]),
            )
            out["per_eq"][f] = rec
        except Exception as e:
            import traceback

            out["per_eq"][f] = dict(
                error=f"{type(e).__name__}: {str(e)[:200]}",
                tb=traceback.format_exc()[-500:],
            )

    names = [f for f in files if "error" not in out["per_eq"].get(f, {})]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ea, eb = eqs[a], eqs[b]
            try:
                dR = np.asarray(ea.R_lmn) - np.asarray(eb.R_lmn)
                dZ = np.asarray(ea.Z_lmn) - np.asarray(eb.Z_lmn)
                nR = np.linalg.norm(np.asarray(ea.R_lmn))
                nZ = np.linalg.norm(np.asarray(ea.Z_lmn))
                ra, rb = out["per_eq"][a], out["per_eq"][b]
                rel = lambda k: (  # noqa: E731
                    abs(ra[k] - rb[k]) / max(abs(ra[k]), 1e-300)
                )
                out["pairs"][f"{a} || {b}"] = dict(
                    dR_rel=float(np.linalg.norm(dR) / nR),
                    dZ_rel=float(np.linalg.norm(dZ) / nZ),
                    dV_rel=rel("volume"),
                    dW_B_rel=rel("W_B"),
                    daspect_rel=rel("aspect"),
                    diota_axis=abs(ra["iota_axis"] - rb["iota_axis"]),
                    diota_edge=abs(ra["iota_edge"] - rb["iota_edge"]),
                    force_ratio=ra["force_vol"] / max(rb["force_vol"], 1e-300),
                )
            except Exception as e:
                out["pairs"][f"{a} || {b}"] = dict(
                    error=f"{type(e).__name__}: {str(e)[:150]}"
                )
    return out


@app.function(
    image=gpu_image,
    gpu=GPU,
    timeout=3600,
    memory=32768,
    single_use_containers=True,
    max_containers=MAX_GPU_CONTAINERS,
    volumes={RESULTS_DIR: results},
    retries=0,
)
def reference(name: str = "precise_QA", L: int = 12, pert: float = 0.01):
    """Control: check the SHIPPED equilibrium against this same diagnostic.

    If the unperturbed example already reports a sqrt(g) sign change, the
    diagnostic is wrong. If it is clean and the solved equilibria are not, the
    solves really are producing folded flux surfaces.

    The perturbed arm uses ``eq.perturb`` on the boundary. It previously used
    ``eq.R_lmn *= (1 + pert*randn)`` on the INTERIOR, which is one of the broken
    methods catalogued in PERTURBATION.md -- and note that that arm reported the
    state as healthy, because ``eq.compute`` reads the interior coefficients and
    so cannot see a boundary/interior inconsistency. Read PERTURBATION.md before
    changing this.
    """
    jax = init_gpu()  # noqa: F841
    import numpy as np

    from desc.examples import get
    from desc.grid import LinearGrid, QuadratureGrid
    from desc.objectives import (
        ForceBalance,
        ObjectiveFunction,
        get_fixed_boundary_constraints,
    )

    out = {}
    for tag, do_pert in (("shipped", False), ("perturbed_start", True)):
        eq = get(name)
        eq.change_resolution(L=L, M=L, N=L, L_grid=2 * L, M_grid=2 * L, N_grid=2 * L)
        if do_pert:  # see PERTURBATION.md -- eq.perturb, never a raw coefficient hit
            rng = np.random.default_rng(0)
            Rb, Zb = np.asarray(eq.Rb_lmn), np.asarray(eq.Zb_lmn)
            eq = eq.perturb(
                deltas={
                    "Rb_lmn": pert * rng.standard_normal(Rb.size) * np.abs(Rb),
                    "Zb_lmn": pert * rng.standard_normal(Zb.size) * np.abs(Zb),
                },
                objective=ObjectiveFunction(ForceBalance(eq), jac_chunk_size=1000),
                constraints=get_fixed_boundary_constraints(eq),
                order=2,
                verbose=0,
                copy=True,
            )
        grid = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
        lg = LinearGrid(L=20, M=0, N=0, NFP=eq.NFP, axis=False)
        d = eq.compute(["|F|", "<|F|>_vol", "sqrt(g)", "V", "R0/a"], grid=grid)
        di = eq.compute(["iota"], grid=lg)
        sg = np.asarray(d["sqrt(g)"])
        rho = np.asarray(lg.nodes[:, 0])
        io = np.asarray(di["iota"])
        order = np.argsort(rho)  # LinearGrid node order is not sorted by rho
        out[tag] = dict(
            force_vol=float(np.asarray(d["<|F|>_vol"])),
            force_max=float(np.abs(np.asarray(d["|F|"])).max()),
            volume=float(np.asarray(d["V"])),
            aspect=float(np.asarray(d["R0/a"])),
            sqrtg_min=float(sg.min()),
            sqrtg_max=float(sg.max()),
            sqrtg_sign_change=bool(sg.min() * sg.max() <= 0),
            frac_sqrtg_positive=float((sg > 0).mean()),
            iota_axis=float(io[order][0]),
            iota_edge=float(io[order][-1]),
        )
    return out


@app.local_entrypoint()
def main(files: str = "", out: str = "modal_bench/eqdiag.json"):
    """Diagnose saved equilibria and print pairwise differences."""
    fl = [x.strip() for x in files.split(",") if x.strip()]
    if not fl:
        print("pass --files with comma-separated eq_*.h5 names on the volume")
        return
    res = diagnose.remote(fl)

    ledger.open_section("Equilibrium diagnostics", dict(files=fl))
    print("\n--- per equilibrium ---")
    ledger.note("```")
    for f, r in res["per_eq"].items():
        if "error" in r:
            line = f"{f}: ERROR {r['error']}"
        else:
            line = (
                f"{f}\n    <|F|>_vol={r['force_vol']:.6e}  "
                f"max|F|={r['force_max']:.4e}  V={r['volume']:.6f}  "
                f"R0/a={r['aspect']:.6f}  W_B={r['W_B']:.8e}\n"
                f"    sqrt(g) in [{r['sqrtg_min']:.4e}, {r['sqrtg_max']:.4e}]"
                f"  sign_change={r['sqrtg_sign_change']}  "
                f"iota {r['iota_axis']:.6f} -> {r['iota_edge']:.6f}"
            )
        print(line, flush=True)
        ledger.note(line)
    print("\n--- pairwise ---")
    ledger.note("")
    for k, r in res["pairs"].items():
        if "error" in r:
            line = f"{k}: ERROR {r['error']}"
        else:
            line = (
                f"{k}\n    dR={r['dR_rel']:.3e} dZ={r['dZ_rel']:.3e}  "
                f"dV={r['dV_rel']:.3e} dW_B={r['dW_B_rel']:.3e} "
                f"daspect={r['daspect_rel']:.3e}\n"
                f"    diota axis={r['diota_axis']:.3e} edge={r['diota_edge']:.3e}"
                f"  force ratio={r['force_ratio']:.3f}"
            )
        print(line, flush=True)
        ledger.note(line)
    ledger.note("```")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {out}", flush=True)


@app.local_entrypoint()
def control(name: str = "precise_QA", res: int = 12, pert: float = 0.05):
    """Run the reference control and print it."""
    import json as _json

    r = reference.remote(name, res, pert)
    for tag, d in r.items():
        print(f"\n{tag}:")
        for k, v in d.items():
            print(f"    {k:<22} {v}")
    with open("modal_bench/eqdiag_control.json", "w") as fh:
        _json.dump(r, fh, indent=2)
