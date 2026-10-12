"""Near-axis result -> DESC: pyQSC boundary at minor radius r -> fixed-boundary vacuum solve, and the folded
coils -> PiecewisePlanarArcCoil (B = 2) CoilSets. Scores with the sandbox's own dense check (B.n etc.) and QS.

python to_desc.py RUN --r 0.167 [--LMN 8] [--out DIR]       (run from anywhere; uses coil_sandbox's DESC)
"""
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
SB = "/home/jay/Documents/desc_dev/DESC/coil_sandbox"
sys.path.insert(0, os.path.join(SB, "sandbox"))
import boot; boot.setup(default="gpu")
sys.path.insert(0, os.path.join(HERE, "pylib"))  # pyQSC 0.1.3, vendored
sys.path.insert(0, HERE)
import numpy as np
import jax.numpy as jnp
from qsc import Qsc
from desc.equilibrium import Equilibrium
from desc.coils import CoilSet, MixedCoilSet, PiecewisePlanarArcCoil
from desc.grid import LinearGrid
import common as S
from folded import Layout, coil_arcs, full_coilset, biot_savart

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--r", type=float, required=True)
ap.add_argument("--LMN", type=int, default=8)
ap.add_argument("--nosolve", action="store_true")
ap.add_argument("--out", default=None)
ap.add_argument("--device", default="gpu")
a = ap.parse_args()
out = a.out or a.run + f"_r{a.r}"
os.makedirs(out, exist_ok=True)

d = np.load(a.run + ".npz")
L = Layout(int(d["nfp"]), int(d["nc"]), int(d["K"]), int(d["Kax"]), str(d["mode"]))
rc, zs, eta, C = L.split(jnp.asarray(d["p"])[:L.n])
q = Qsc(rc=np.asarray(rc), zs=np.asarray(zs), nfp=L.nfp, etabar=float(eta), nphi=101)
print(f"pyQSC: iota {q.iota:.4f} helicity {q.helicity} max elong {q.max_elongation:.2f} "
      f"L_gradB min {q.min_L_grad_B:.3f}", flush=True)

# ---- coils (A): mine are mu0 I / 4 pi
arcs = []
for i in range(L.nc):
    (xu, _, _), (xl, _, _), I = coil_arcs(C[i], L)
    pts = np.concatenate([np.asarray(xu)[:-1], np.asarray(xl)[:-1]])
    arcs.append(PiecewisePlanarArcCoil.from_values(float(I) / 1e-7, pts, B=2, M=L.K, basis="xyz", fit_method="chord"))
cs = MixedCoilSet(*[CoilSet(c, NFP=L.nfp, sym=True) for c in arcs])
# check the conversion reproduces my field
X, Ia = full_coilset(C, L)
pt = np.array([float(rc.sum()), 0.03, 0.02])
B_mine = np.asarray(biot_savart(jnp.asarray(pt), X, Ia))
B_desc = np.asarray(cs.compute_magnetic_field(pt[None], basis="xyz", source_grid=LinearGrid(N=400)))[0]
print(f"coil conversion: |dB|/|B| = {np.linalg.norm(B_desc - B_mine) / np.linalg.norm(B_mine):.2e}  "
      f"hinge |z| max {max(float(np.abs(np.asarray(c.hinges).reshape(-1, 3)[:, 2]).max()) for c in arcs):.1e}", flush=True)
cs.save(os.path.join(out, "coils.h5"))

# ---- equilibrium
eq = Equilibrium.from_near_axis(q, r=a.r, L=a.LMN, M=a.LMN, N=a.LMN)
eq.save(os.path.join(out, "eq_na.h5"))
res = {}


def score(eq, tag):
    from desc.objectives import QuasisymmetryBoozer
    B = float(eq.compute("<|B|>_vol")["<|B|>_vol"])
    hels = [(1, 0)] if q.helicity == 0 else [(1, L.nfp), (1, -L.nfp)]
    best = None
    for hel in hels:
        qs = []
        for rr in (0.25, 0.5, 0.75, 1.0):
            o = QuasisymmetryBoozer(eq, helicity=hel, normalize=False, grid=LinearGrid(
                rho=np.array([rr]), M=2 * eq.M_grid, N=2 * eq.N_grid, NFP=eq.NFP, sym=False))
            o.build(verbose=0)
            qs.append(float(np.linalg.norm(o.compute_unscaled(eq.params_dict))) / B)
        if best is None or qs[1] < best[1][1]:
            best = (hel, qs)
    ig = LinearGrid(rho=np.linspace(0, 1, 11), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
    iota = ig.compress(eq.compute("iota", grid=ig)["iota"])
    m = S.evaluate(cs, eq, vacuum=True)
    dd = eq.compute(["R0", "a", "R0/a"])
    r = {"helicity": best[0], "qs_booz_rho.25,.5,.75,1": best[1], "iota_min_max": [float(iota.min()), float(iota.max())],
         "A": float(dd["R0/a"]), "a": float(dd["a"]), "R0": float(dd["R0"]), "bn": m["bn"], "bn_max": m["bn_max"],
         "d_pc": m["d_pc"], "d_cc": m["d_cc"], "L": m["L"], "kappa": m["kappa"], "linked": m["linked"]}
    print(tag, json.dumps(r), flush=True)
    return r


res["near_axis_unsolved"] = score(eq, "unsolved")
if not a.nosolve:
    eq.solve(verbose=1, maxiter=200, ftol=1e-10, xtol=1e-12, gtol=1e-10)
    eq.save(os.path.join(out, "eq_solved.h5"))
    res["fixed_boundary_solved"] = score(eq, "solved")
json.dump(res, open(os.path.join(out, "score.json"), "w"), indent=1)
