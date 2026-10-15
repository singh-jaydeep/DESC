"""Repair a free-boundary equilibrium whose BOUNDARY parametrization has creased (|e_theta x e_zeta| -> 0 at a point:
a kink of a few mm with a 10-20% B.n spike that the coils' field lines do not have; see tip.py / tip_trace.py).

  1. re-parametrize the same boundary by equal poloidal arclength in each zeta plane (origin: the old theta = 0 curve,
     which keeps stellarator symmetry) and refit it as a FourierRZToroidalSurface;
  2. solve the interior again from scratch on that boundary by DESC's continuation (solve_continuation_automatic),
     instead of inheriting flux surfaces from a chain of perturbations.

    python repair_eq.py EQ.h5 COILS.h5 OUT.h5 [--MN 12] [--L 12] [--no-solve]      (GPU)
"""
import sys, argparse, json
sys.path.insert(0, "../../sandbox")
import boot; boot.setup(default="gpu")
import numpy as np
from desc.io import load
from desc.grid import LinearGrid
from desc.geometry import FourierRZToroidalSurface
from desc.equilibrium import Equilibrium
from desc.continuation import solve_continuation_automatic
from desc.objectives import QuasisymmetryBoozer
import common as S

ap = argparse.ArgumentParser()
ap.add_argument("eq"); ap.add_argument("coils"); ap.add_argument("out")
ap.add_argument("--MN", type=int, default=None); ap.add_argument("--L", type=int, default=None)
ap.add_argument("--nt", type=int, default=96); ap.add_argument("--nz", type=int, default=96)
ap.add_argument("--no-solve", action="store_true"); ap.add_argument("--chunk", type=int, default=200)
ap.add_argument("--device", default="gpu")
a = ap.parse_args()
eq, cs = load(a.eq), load(a.coils)
src = S.source_grid(cs, gl_m=64)[0]
MN, L = a.MN or eq.M, a.L or eq.L


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

# ---- 1. equal-arclength poloidal angle in every zeta plane
zeta = np.linspace(0, 2 * np.pi / eq.NFP, a.nz, endpoint=False)
tf = np.linspace(0, 2 * np.pi, 4001)
tn = np.linspace(0, 2 * np.pi, a.nt, endpoint=False)
pts, th, ze = [], [], []
for z in zeta:
    d = eq.compute(["R", "Z"], grid=LinearGrid(rho=np.array([1.0]), theta=tf, zeta=np.array([z]), NFP=eq.NFP))
    R, Z = np.asarray(d["R"]), np.asarray(d["Z"])
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(R), np.diff(Z)))])
    s = 2 * np.pi * s / s[-1]
    pts.append(np.column_stack([np.interp(tn, s, R), np.full(a.nt, z), np.interp(tn, s, Z)]))
    th.append(tn); ze.append(np.full(a.nt, z))
pts, th, ze = np.concatenate(pts), np.concatenate(th), np.concatenate(ze)
surf = FourierRZToroidalSurface.from_values(pts, th, M=MN, N=MN, NFP=eq.NFP, sym=eq.sym)
gchk = LinearGrid(rho=np.array([1.0]), theta=tn, zeta=zeta, NFP=eq.NFP)
dc = surf.compute(["R", "Z"], grid=gchk)
nodes = gchk.nodes
# distance of each fitted node to the ORIGINAL curve in its plane (parametrization independent)
err = []
for z in zeta[:: max(1, a.nz // 16)]:
    d = eq.compute(["R", "Z"], grid=LinearGrid(rho=np.array([1.0]), theta=tf, zeta=np.array([z]), NFP=eq.NFP))
    m = np.isclose(nodes[:, 2], z)
    dd = np.hypot(np.asarray(dc["R"])[m][:, None] - np.asarray(d["R"])[None], np.asarray(dc["Z"])[m][:, None] - np.asarray(d["Z"])[None])
    err.append(dd.min(1))
err = np.concatenate(err)
print(f"refit M = N = {MN}: distance of the new surface from the old one: mean {err.mean() * 1e3:.3f} mm, max {err.max() * 1e3:.3f} mm", flush=True)
surf_report(surf, "boundary after: ")

# ---- 2. interior from scratch by continuation
new = Equilibrium(Psi=float(eq.Psi), NFP=eq.NFP, sym=eq.sym, L=L, M=MN, N=MN, L_grid=2 * L, M_grid=2 * MN, N_grid=2 * MN,
                  surface=surf, pressure=eq.pressure, current=eq.current, iota=eq.iota)
if a.no_solve:
    new.save(a.out); raise SystemExit(0)
fam = solve_continuation_automatic(new, verbose=1, jac_chunk_size=a.chunk)
new = fam[-1]
eq_report(new, "equilibrium after: ")
surf_report(new, "boundary of the solved equilibrium:")
new.save(a.out)
print("wrote", a.out)
