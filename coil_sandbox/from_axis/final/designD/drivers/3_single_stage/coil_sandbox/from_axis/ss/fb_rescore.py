"""Rule-2 score of a single-stage state: vacuum FREE-BOUNDARY solve with the state's coils (no other objective),
starting from the state's equilibrium, then Boozer QS / iota / B.n / Jacobian folds / coil check / Poincare.

    python fb_rescore.py DIR/eq_kK.h5 DIR/coils_kK.h5 OUTDIR [--shells 4,6,8] [--maxiter 60]
"""
import os, sys, json, time, argparse
sys.path.insert(0, "../../sandbox")
import boot; boot.setup(default="gpu")
import numpy as np, matplotlib; matplotlib.use("Agg")
from desc.io import load
from desc.grid import LinearGrid
from desc.objectives import (VacuumBoundaryError, ObjectiveFunction, ForceBalance, FixCurrent, FixPressure, FixPsi,
                             FixBoundaryR, FixBoundaryZ, QuasisymmetryBoozer)
from desc.plotting import poincare_plot, plot_surfaces
import common as S
from repair_lib import repair

ap = argparse.ArgumentParser()
ap.add_argument("eq"); ap.add_argument("coils"); ap.add_argument("out")
ap.add_argument("--shells", default="4,8,12")
ap.add_argument("--maxiter", type=int, default=60)
ap.add_argument("--no-poincare", action="store_true")
ap.add_argument("--repair", action="store_true", help="BEFORE every shell: equal-arclength boundary angle + interior re-solved by continuation (repair_lib.py). Not after: the refit itself costs 0.2-0.4%% of B.n (ss12_li030_q15), so the last word must be a free-boundary shell; check the result with tip.py")
ap.add_argument("--chunk", type=int, default=200, help="ForceBalance jac_chunk_size (raise on a large-memory GPU)")
ap.add_argument("--helicity", default="1,0")
ap.add_argument("--bounds", default="../../bounds/fa_qa2_a0167.json")
ap.add_argument("--device", default="gpu")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
t0 = time.time()
eq, cs = load(a.eq), load(a.coils)
src = S.source_grid(cs, gl_m=64)[0]
hel = tuple(int(v) for v in a.helicity.split(","))


def bn_rms(eq):
    g = LinearGrid(M=2 * eq.M_grid, N=2 * eq.N_grid, NFP=eq.NFP, sym=False)
    d = eq.compute(["x", "n_rho", "|e_theta x e_zeta|"], grid=g, basis="xyz")
    B = np.asarray(cs.compute_magnetic_field(np.asarray(d["x"]), basis="xyz", source_grid=src))
    bn = np.sum(B * np.asarray(d["n_rho"]), 1)
    w = np.asarray(g.weights) * np.asarray(d["|e_theta x e_zeta|"])
    Bm = np.sum(w * np.linalg.norm(B, axis=1)) / np.sum(w)
    return float(np.sqrt(np.sum(w * bn**2) / np.sum(w)) / Bm)


def physics(eq):
    d = eq.compute(["R0", "a", "R0/a", "<|B|>_vol"])
    Bv = float(d["<|B|>_vol"])
    qs = []
    for r in (0.25, 0.5, 0.75, 1.0):
        o = QuasisymmetryBoozer(eq, helicity=hel, normalize=False, grid=LinearGrid(
            rho=np.array([r]), M=2 * eq.M_grid, N=2 * eq.N_grid, NFP=eq.NFP, sym=False))
        o.build(verbose=0)
        qs.append(float(np.linalg.norm(o.compute_unscaled(eq.params_dict))) / Bv)
    ig = LinearGrid(rho=np.linspace(0, 1, 21), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
    iota = ig.compress(eq.compute("iota", grid=ig)["iota"])
    fold = []
    for rho in (0.8, 1.0):
        g = LinearGrid(rho=np.array([rho]), M=64, N=64, NFP=eq.NFP, sym=False)
        s = np.asarray(eq.compute("sqrt(g)", grid=g)["sqrt(g)"])
        fold.append(float((s < 0).mean()))
    return dict(A=float(d["R0/a"]), a=float(d["a"]), R0=float(d["R0"]), qs_booz=qs, iota_min=float(iota.min()),
                iota_max=float(iota.max()), iota_axis=float(iota[0]), iota_edge=float(iota[-1]), bn_rms=bn_rms(eq),
                neg_jac_frac_rho08_1=fold)


log = {"eq": a.eq, "coils": a.coils, "start": physics(eq), "shells": []}
print("START (single-stage state)", json.dumps(log["start"]), flush=True)
done_f = os.path.join(a.out, "shells_done.json")  # resume: shells already finished in this OUT are skipped
done = json.load(open(done_f)) if os.path.exists(done_f) and os.path.exists(os.path.join(a.out, "eq_fb.h5")) else []
if done:
    eq = load(os.path.join(a.out, "eq_fb.h5"))
    log["shells"] = done
    print(f"RESUME: shells {[d['k'] for d in done]} already done, continuing from eq_fb.h5", flush=True)
for k in ([] if a.shells == "none" else [int(v) for v in a.shells.split(",")]):  # "none": score as is
    if any(d["k"] == k for d in done):
        continue
    rinfo = None
    if a.repair:  # BEFORE the shell: the refit moves the surface ~0.5-1.5 mm (B.n 0.2-0.4% rms), which this shell removes
        try:
            eq, rinfo = repair(eq, chunk=a.chunk)
        except Exception as e:  # keep the equilibrium if the repair itself fails
            rinfo = dict(failed=repr(e)[:300])
            print(f"REPAIR FAILED before shell k={k}: {e!r}", flush=True)
    cons = (ForceBalance(eq=eq, jac_chunk_size=a.chunk), FixCurrent(eq=eq), FixPressure(eq=eq), FixPsi(eq=eq))
    obj = ObjectiveFunction(VacuumBoundaryError(eq=eq, field=cs, field_fixed=True, field_grid=src))
    Rm = eq.surface.R_basis.modes[np.max(np.abs(eq.surface.R_basis.modes), 1) > k, :]
    Zm = eq.surface.Z_basis.modes[np.max(np.abs(eq.surface.Z_basis.modes), 1) > k, :]
    bc = tuple(c for c in (FixBoundaryR(eq=eq, modes=Rm) if len(Rm) else None,
                           FixBoundaryZ(eq=eq, modes=Zm) if len(Zm) else None) if c is not None)
    try:
        eq, res = eq.optimize(obj, cons + bc, optimizer="proximal-lsq-exact", verbose=1, maxiter=a.maxiter,
                              ftol=1e-8, xtol=1e-10, gtol=1e-8, copy=True)
    except Exception as e:  # e.g. GPU out of memory at the full-boundary shell: score the last completed shell
        log["shells"].append(dict(k=k, failed=repr(e)[:300]))
        print(f"[{time.time() - t0:.0f}s] shell k={k} FAILED ({type(e).__name__}); scoring the previous shell", flush=True)
        break
    b = bn_rms(eq)
    log["shells"].append(dict(k=k, bn_rms=b, nit=int(res.get("nit", -1)), message=str(res.get("message", "")), repair=rinfo))
    print(f"[{time.time() - t0:.0f}s] shell k={k}: B.n rms {b:.3e} ({res.get('message', '')})", flush=True)
    eq.save(os.path.join(a.out, "tmp_eq_fb.h5"))
    os.replace(os.path.join(a.out, "tmp_eq_fb.h5"), os.path.join(a.out, "eq_fb.h5"))
    json.dump([d for d in log["shells"] if "failed" not in d], open(done_f, "w"))
log["free_boundary"] = physics(eq)
print("FREE BOUNDARY", json.dumps(log["free_boundary"]), flush=True)
P = S.paper_bounds(eq, S.n_unique(cs), None, a.bounds)
P = {k: P[k] for k in S.UPPER + S.LOWER}
m = S.evaluate(cs, eq, vacuum=True)
log["coil_check"] = {k: m[k] for k in ("L", "kappa", "d_cc", "d_pc", "linked")}
log["coil_verdict"] = S.verdict(m, P)
print("COILS", S.report(m, P), S.verdict(m, P), flush=True)
json.dump(log, open(os.path.join(a.out, "score.json"), "w"), indent=1)
if a.no_poincare:
    raise SystemExit(0)
fig, ax = plot_surfaces(eq)
gt = LinearGrid(rho=np.linspace(0.1, 1.1, 11), theta=np.array([0.0]), zeta=np.array([0.0]), NFP=1)
dt = eq.compute(["R", "Z"], grid=gt)
fig, ax = poincare_plot(cs, np.asarray(dt["R"]), np.asarray(dt["Z"]), ntransit=150, NFP=eq.NFP, ax=ax, grid=src,
                        size=1.5)
fig.savefig(os.path.join(a.out, "poincare.png"), dpi=130, bbox_inches="tight")
print(f"[{time.time() - t0:.0f}s] wrote {a.out}", flush=True)
