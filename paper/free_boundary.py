"""Vacuum free-boundary solve with fixed coils, then physics and coil scores.

    python paper/free_boundary.py EQ.h5 COILS.h5 OUT [--shells 4,8,12] [--repair] [--LMN 14]

Stock DESC: ``VacuumBoundaryError(field_fixed=True)`` with ForceBalance, FixCurrent,
FixPressure and FixPsi as proximal constraints, ``proximal-lsq-exact``, the boundary
modes released in shells (|m|, |n| <= k free). ``--repair`` re-labels a creased
boundary poloidal angle and re-solves the interior before every shell
(``paper/lib/bdry_repair``); every repair is logged (repair log rule, PORT_PLAN 9.3).
Writes OUT/eq_fb.h5, OUT/score.json and OUT/poincare.png. Ported from
coil_sandbox/from_axis/ss/fb_rescore.py (js/qa-poster).
"""

import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("eq")
ap.add_argument("coils")
ap.add_argument("out")
ap.add_argument("--shells", default="4,8,12", help="k per shell, or 'none'")
ap.add_argument("--maxiter", type=int, default=60)
ap.add_argument("--no-poincare", action="store_true")
ap.add_argument("--repair", action="store_true", help="repair before every shell")
ap.add_argument("--chunk", type=int, default=200, help="ForceBalance jac_chunk_size")
ap.add_argument("--helicity", default="1,0")
ap.add_argument("--bounds", default=None, help="bounds file for the coil check")
ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
ap.add_argument("--LMN", type=int, default=None, help="raise to basis LMN first")
ap.add_argument("--solve-tol", type=float, default=None, help="inner solve tolerances")
ap.add_argument("--solve-maxiter", type=int, default=None)
a = ap.parse_args()

from desc import set_device  # noqa: E402

set_device(a.device)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import numpy as np  # noqa: E402

from desc.coils import coil_source_grid  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.io import load  # noqa: E402
from desc.objectives import (  # noqa: E402
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixPressure,
    FixPsi,
    ForceBalance,
    ObjectiveFunction,
    QuasisymmetryBoozer,
    VacuumBoundaryError,
)
from desc.plotting import plot_surfaces, poincare_plot  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
import dense_check as D  # noqa: E402
from bdry_repair import repair  # noqa: E402

os.makedirs(a.out, exist_ok=True)
t0 = time.time()
eq, cs = load(a.eq), load(a.coils)
src = coil_source_grid(cs, n=64)
hel = tuple(int(v) for v in a.helicity.split(","))


def bn_rms(eq):
    """Area-weighted rms of B.n / <|B|> of the coils on the boundary."""
    g = LinearGrid(M=2 * eq.M_grid, N=2 * eq.N_grid, NFP=eq.NFP, sym=False)
    d = eq.compute(["x", "n_rho", "|e_theta x e_zeta|"], grid=g, basis="xyz")
    B = np.asarray(
        cs.compute_magnetic_field(np.asarray(d["x"]), basis="xyz", source_grid=src)
    )
    bn = np.sum(B * np.asarray(d["n_rho"]), 1)
    w = np.asarray(g.weights) * np.asarray(d["|e_theta x e_zeta|"])
    Bm = np.sum(w * np.linalg.norm(B, axis=1)) / np.sum(w)
    return float(np.sqrt(np.sum(w * bn**2) / np.sum(w)) / Bm)


def physics(eq):
    """A, Boozer QS on four surfaces, iota, B.n rms, fraction of negative sqrt(g)."""
    d = eq.compute(["R0", "a", "R0/a", "<|B|>_vol"])
    Bv = float(d["<|B|>_vol"])
    qs = []
    for r in (0.25, 0.5, 0.75, 1.0):
        grid = LinearGrid(
            rho=np.array([r]), M=2 * eq.M_grid, N=2 * eq.N_grid, NFP=eq.NFP, sym=False
        )
        o = QuasisymmetryBoozer(eq, helicity=hel, normalize=False, grid=grid)
        o.build(verbose=0)
        qs.append(float(np.linalg.norm(o.compute_unscaled(eq.params_dict))) / Bv)
    ig = LinearGrid(
        rho=np.linspace(0, 1, 21), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym
    )
    iota = ig.compress(eq.compute("iota", grid=ig)["iota"])
    fold = []
    for rho in (0.8, 1.0):
        g = LinearGrid(rho=np.array([rho]), M=64, N=64, NFP=eq.NFP, sym=False)
        s = np.asarray(eq.compute("sqrt(g)", grid=g)["sqrt(g)"])
        fold.append(float((s < 0).mean()))
    return dict(
        A=float(d["R0/a"]),
        a=float(d["a"]),
        R0=float(d["R0"]),
        qs_booz=qs,
        iota_min=float(iota.min()),
        iota_max=float(iota.max()),
        iota_axis=float(iota[0]),
        iota_edge=float(iota[-1]),
        bn_rms=bn_rms(eq),
        neg_jac_frac_rho08_1=fold,
    )


def repair_line(tag, info):
    """One-line repair summary (PORT_PLAN 9.3)."""
    if info is None or info.get("action") == "none":
        return f"REPAIR {tag}: not needed"
    if "failed" in info:
        return f"REPAIR {tag}: FAILED {info['failed']}"
    mv = info["surface_moved_mm"]
    return (
        f"REPAIR {tag}: fired; surface moved rms {mv['rms']:.3f} / p99 {mv['p99']:.2f}"
        f" / max {mv['max']:.2f} mm"
        + ("  STILL UNHEALTHY" if info.get("still_unhealthy") else "")
    )


if a.LMN is not None and (eq.L, eq.M, eq.N) != (a.LMN,) * 3:
    n = a.LMN
    eq.change_resolution(L=n, M=n, N=n, L_grid=2 * n, M_grid=2 * n, N_grid=2 * n)
    eq.solve(
        objective=ObjectiveFunction(ForceBalance(eq, jac_chunk_size=a.chunk)),
        verbose=1,
        maxiter=300,
        ftol=1e-12,
        xtol=1e-14,
        gtol=1e-12,
    )
    print(f"[{time.time() - t0:.0f}s] raised to basis {n}, re-solved", flush=True)
log = {"eq": a.eq, "coils": a.coils, "argv": sys.argv, "start": physics(eq)}
log["shells"], log["repairs"] = [], []
print("START", json.dumps(log["start"]), flush=True)
done_f = os.path.join(a.out, "shells_done.json")
eq_f = os.path.join(a.out, "eq_fb.h5")
done = (
    json.load(open(done_f)) if os.path.exists(done_f) and os.path.exists(eq_f) else []
)
if done:
    eq = load(eq_f)
    log["shells"] = done
    print(f"RESUME: shells {[d['k'] for d in done]} done", flush=True)
shells = [] if a.shells == "none" else [int(v) for v in a.shells.split(",")]
for k in shells:
    if any(d["k"] == k for d in done):
        continue
    rinfo = None
    if a.repair:  # before the shell, so the shell removes the refit's B.n cost
        try:
            eq, rinfo = repair(eq, chunk=a.chunk)
        except Exception as e:  # noqa: BLE001  (keep the eq if the repair fails)
            rinfo = dict(failed=repr(e)[:300])
        log["repairs"].append(dict(at=f"before shell {k}", **rinfo))
        print(repair_line(f"before shell {k}", rinfo), flush=True)
    cons = (
        ForceBalance(eq=eq, jac_chunk_size=a.chunk),
        FixCurrent(eq=eq),
        FixPressure(eq=eq),
        FixPsi(eq=eq),
    )
    obj = ObjectiveFunction(
        VacuumBoundaryError(eq=eq, field=cs, field_fixed=True, field_grid=src)
    )
    R, Z = eq.surface.R_basis.modes, eq.surface.Z_basis.modes
    Rm = R[np.max(np.abs(R), 1) > k, :]
    Zm = Z[np.max(np.abs(Z), 1) > k, :]
    bc = tuple(
        c
        for c in (
            FixBoundaryR(eq=eq, modes=Rm) if len(Rm) else None,
            FixBoundaryZ(eq=eq, modes=Zm) if len(Zm) else None,
        )
        if c is not None
    )
    solve = {"verbose": 0}
    if a.solve_tol is not None:
        solve.update(ftol=a.solve_tol, xtol=a.solve_tol, gtol=a.solve_tol)
    if a.solve_maxiter is not None:
        solve["maxiter"] = a.solve_maxiter
    try:
        eq, res = eq.optimize(
            obj,
            cons + bc,
            optimizer="proximal-lsq-exact",
            verbose=1,
            maxiter=a.maxiter,
            ftol=1e-8,
            xtol=1e-10,
            gtol=1e-8,
            copy=True,
            options={"solve_options": solve},
        )
    except Exception as e:  # noqa: BLE001  (e.g. out of memory: keep the last shell)
        log["shells"].append(dict(k=k, failed=repr(e)[:300]))
        print(f"shell k={k} FAILED ({type(e).__name__}); keeping the last", flush=True)
        break
    b = bn_rms(eq)
    msg = str(res.get("message", ""))
    log["shells"].append(dict(k=k, bn_rms=b, nit=int(res.get("nit", -1)), message=msg))
    print(f"[{time.time() - t0:.0f}s] shell k={k}: B.n rms {b:.3e} ({msg})", flush=True)
    eq.save(os.path.join(a.out, "tmp_eq_fb.h5"))
    os.replace(os.path.join(a.out, "tmp_eq_fb.h5"), eq_f)
    json.dump([d for d in log["shells"] if "failed" not in d], open(done_f, "w"))
log["free_boundary"] = physics(eq)
log["repairs_fired"] = sum(r.get("action") == "repaired" for r in log["repairs"])
print("FREE BOUNDARY", json.dumps(log["free_boundary"]), flush=True)
if a.bounds:
    P = D.load_bounds(a.bounds, eq, len(D.leaf_coils(cs)))
    m = D.evaluate(cs, eq, vacuum=True)
    log["coil_check"] = {k: m[k] for k in ("L", "kappa", "dcc", "dpc", "linked")}
    log["coil_verdict"] = D.verdict(m, P)
    print("COILS", D.report(m, P), D.verdict(m, P), flush=True)
json.dump(log, open(os.path.join(a.out, "score.json"), "w"), indent=1, default=str)
if a.no_poincare:
    raise SystemExit(0)
fig, ax = plot_surfaces(eq)
gt = LinearGrid(
    rho=np.linspace(0.1, 1.1, 11), theta=np.array([0.0]), zeta=np.array([0.0]), NFP=1
)
dt = eq.compute(["R", "Z"], grid=gt)
fig, ax = poincare_plot(
    cs,
    np.asarray(dt["R"]),
    np.asarray(dt["Z"]),
    ntransit=150,
    NFP=eq.NFP,
    ax=ax,
    grid=src,
    size=1.5,
)
fig.savefig(os.path.join(a.out, "poincare.png"), dpi=130, bbox_inches="tight")
print(f"[{time.time() - t0:.0f}s] wrote {a.out}", flush=True)
