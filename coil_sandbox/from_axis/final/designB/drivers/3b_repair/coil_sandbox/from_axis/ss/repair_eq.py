"""Repair one equilibrium from the command line and report what it cost (thin wrapper over the bdry_repair package).

Prints, before and after: the boundary's angle health and the coils' B.n on it, the edge Jacobian, QS (Boozer, / <B>),
iota and force error; and how far the surface moved.

    python repair_eq.py EQ.h5 COILS.h5 OUT.h5 [--rule chordal|curvature] [--max-steps 8] [--force]      (GPU)

--max-steps 0 is the old repair (linear fit at fixed equal-arclength labels). Without --force a healthy boundary is
left alone (OUT is then a copy of EQ).
"""
import sys, argparse, json
sys.path.insert(0, "../../sandbox")
import boot; boot.setup(default="gpu")
import numpy as np
from desc.io import load
from desc.grid import LinearGrid
from desc.objectives import QuasisymmetryBoozer
import common as S
from repair_lib import repair

ap = argparse.ArgumentParser()
ap.add_argument("eq"); ap.add_argument("coils"); ap.add_argument("out")
ap.add_argument("--rule", default="chordal"); ap.add_argument("--max-steps", type=int, default=8)
ap.add_argument("--force", action="store_true"); ap.add_argument("--chunk", type=int, default=200)
ap.add_argument("--device", default="gpu")
a = ap.parse_args()
eq, cs = load(a.eq), load(a.coils)
src = S.source_grid(cs, gl_m=64)[0]


def surf_report(obj, tag):
    g = LinearGrid(rho=np.array([1.0]), M=64, N=64, NFP=eq.NFP, sym=False)
    d = obj.compute(["x", "n_rho", "|e_theta x e_zeta|", "curvature_k1_rho", "curvature_k2_rho"], grid=g, basis="xyz")
    B = np.asarray(cs.compute_magnetic_field(np.asarray(d["x"]), basis="xyz", source_grid=src))
    ar = np.asarray(d["|e_theta x e_zeta|"]); w = np.asarray(g.weights) * ar; w /= w.sum()
    Bavg = np.sum(w * np.linalg.norm(B, axis=1))
    bn = np.abs(np.sum(B * np.asarray(d["n_rho"]), 1)) / Bavg
    k = np.maximum(np.abs(np.asarray(d["curvature_k1_rho"])), np.abs(np.asarray(d["curvature_k2_rho"])))
    r = dict(area_min_over_median=float(ar.min() / np.median(ar)), kappa_max=float(k.max()), bn_mean=float(np.sum(w * bn)),
             bn_rms=float(np.sqrt(np.sum(w * bn**2))), bn_max=float(bn.max()), bn_99=float(np.sort(bn)[int(0.99 * bn.size)]))
    print(tag, json.dumps(r), flush=True)
    return r


def eq_report(e, tag):
    out = []
    for rho in (0.5, 0.8, 0.9, 0.95, 1.0):
        g = LinearGrid(rho=np.array([rho]), M=64, N=64, NFP=e.NFP, sym=False)
        s = np.asarray(e.compute("sqrt(g)", grid=g)["sqrt(g)"])
        out.append(f"{rho}:{s.min() / np.median(s):+.3f}")
    Bv = float(e.compute("<|B|>_vol")["<|B|>_vol"]); qs = []
    for rho in (0.25, 0.5, 0.75, 1.0):
        o = QuasisymmetryBoozer(e, helicity=(1, 0), normalize=False, grid=LinearGrid(
            rho=np.array([rho]), M=2 * e.M_grid, N=2 * e.N_grid, NFP=e.NFP, sym=False)); o.build(verbose=0)
        qs.append(float(np.linalg.norm(o.compute_unscaled(e.params_dict))) / Bv)
    ig = LinearGrid(rho=np.linspace(0, 1, 11), M=e.M_grid, N=e.N_grid, NFP=e.NFP, sym=e.sym)
    io = ig.compress(e.compute("iota", grid=ig)["iota"])
    f = e.compute("<|F|>_vol")["<|F|>_vol"]
    print(tag, "sqrt(g) min/median", " ".join(out), "| QS", ["%.2e" % v for v in qs], f"| iota {io[0]:.4f} -> {io[-1]:.4f} | <|F|>_vol {float(f):.3e}", flush=True)


surf_report(eq, "boundary before:")
eq_report(eq, "equilibrium before:")

new, info = repair(eq, chunk=a.chunk, verbose=1, force=a.force, rule=a.rule, max_steps=a.max_steps)
if new is not eq:
    surf_report(new, "boundary after: ")
    eq_report(new, "equilibrium after: ")
new.save(a.out)
json.dump(info, open(a.out.replace(".h5", "") + "_repair.json", "w"), indent=1)
print("wrote", a.out)
