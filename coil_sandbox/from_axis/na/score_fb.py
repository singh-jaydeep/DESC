"""Phase 2 scorer: the coils' OWN flux surface, by a vacuum free-boundary DESC solve (no NAE term).

    python score_fb.py runs2/NAME [--LMN 10] [--a A] [--kmax K] [--out DIR]     (from na/, GPU)

1. Coils: exact conversion to DESC PiecewisePlanarArcCoil (B = 2) CoilSets (convert.py), saved as coils.h5.
2. Initial guess only: pyQSC (vendored) at r = a -> Equilibrium.from_near_axis -> fixed-boundary vacuum solve.
3. Free boundary (DESC's vacuum recipe, docs/notebooks/tutorials/free_boundary_equilibrium.ipynb):
   objective VacuumBoundaryError(eq, coils, field_fixed=True); constraints ForceBalance, FixCurrent (0),
   FixPressure (0), FixPsi; boundary modes released in shells |m|,|n| <= k = 2, 4, ...; proximal-lsq-exact.
   Psi = pi a^2 B0 (from the NAE) selects which coil-field surface becomes the boundary.
4. Score: Boozer QS (|B_nonsym| / <|B|>, rho .25/.5/.75/1, as single_stage.py Physics), iota profile, A, a, R0;
   B.n / <B> of the coils on the final boundary (GL 64 per arc: 24 is too coarse for these arcs);
   axis and iota vs the NAE prediction (diagnostic); common.evaluate (d_pc against the real boundary, etc.);
   a Poincare plot of the COILS' field with the free-boundary surfaces overlaid (fake-surface check).
"""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "sandbox"))
import boot; boot.setup(default="gpu")
sys.path.insert(0, os.path.join(HERE, "pylib"))
sys.path.insert(0, HERE)
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qsc import Qsc
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid
from desc.objectives import (VacuumBoundaryError, ObjectiveFunction, ForceBalance, FixCurrent, FixPressure,
                             FixPsi, FixBoundaryR, FixBoundaryZ, QuasisymmetryBoozer)
from desc.plotting import poincare_plot, plot_surfaces
import common as S
from folded import Layout
from convert import to_desc_coils

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--LMN", type=int, default=10)
ap.add_argument("--a", type=float, default=None, help="minor radius (default: the run's --a)")
ap.add_argument("--gl", type=int, default=64, help="Gauss-Legendre nodes per arc for the coil field")
ap.add_argument("--kstep", type=int, default=2)
ap.add_argument("--maxiter", type=int, default=100, help="per boundary shell")
ap.add_argument("--ntransit", type=int, default=150)
ap.add_argument("--out", default=None)
ap.add_argument("--device", default="gpu")
ap.add_argument("--resume", default=None, help="eq_fb.h5 to continue the free boundary from")
ap.add_argument("--shells", default=None, help="comma list of k (default kstep, 2 kstep, ..., LMN)")
a = ap.parse_args()

z = np.load(a.run + ".npz")
J = json.load(open(a.run + ".json"))
args = J["args"]
r = a.a or args["a"]
out = a.out or a.run + f"_fb_L{a.LMN}"
os.makedirs(out, exist_ok=True)
log = {"run": a.run, "a_target": r, "LMN": a.LMN, "gl": a.gl}
t0 = time.time()


def stamp(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# ---- coils (exact)
L = Layout(int(z["nfp"]), int(z["nc"]), int(z["K"]), int(z["Kax"]), str(z["mode"]))
rc, zs, eta, C = L.split(jnp.asarray(z["p"])[:L.n])
cs, arcs = to_desc_coils(C, L)
cs.save(os.path.join(out, "coils.h5"))
src = S.source_grid(cs, gl_m=a.gl)[0]

# ---- initial guess from the NAE (only an initial guess)
q = Qsc(rc=np.asarray(rc), zs=np.asarray(zs), nfp=L.nfp, etabar=float(eta), nphi=101)
log["nae"] = dict(iota=float(q.iota), helicity=float(q.helicity), max_elongation=float(q.max_elongation),
                  axis_rc=np.asarray(rc).tolist(), axis_zs=np.asarray(zs).tolist(), etabar=float(eta))
stamp(f"NAE: iota {q.iota:.4f} helicity {q.helicity} max elong {q.max_elongation:.2f}")
eq = Equilibrium.from_near_axis(q, r=r, L=a.LMN, M=a.LMN, N=a.LMN)
eq.solve(verbose=0, maxiter=200, ftol=1e-8, xtol=1e-10, gtol=1e-8)
eq.save(os.path.join(out, "eq_nae_fixed.h5"))
stamp("NAE fixed-boundary initial guess solved")

# orientation check: the equilibrium field and the coil field must point the same way
gb = LinearGrid(rho=np.array([1.0]), M=8, N=8, NFP=eq.NFP)
db = eq.compute(["B", "x"], grid=gb, basis="xyz")
Bc = np.asarray(cs.compute_magnetic_field(np.asarray(db["x"]), basis="xyz", source_grid=src))
align = float(np.mean(np.sum(np.asarray(db["B"]) * Bc, 1) / (np.linalg.norm(db["B"], axis=1) * np.linalg.norm(Bc, axis=1))))
stamp(f"cos(B_eq, B_coils) on the initial boundary: mean {align:.4f}")
log["initial_alignment"] = align
if align < 0:
    raise SystemExit("equilibrium and coil fields point opposite ways: fix the Psi sign convention first")


def bn_rms(eq):
    g = LinearGrid(M=2 * eq.M_grid, N=2 * eq.N_grid, NFP=eq.NFP, sym=False)
    d = eq.compute(["x", "n_rho", "|B|", "|e_theta x e_zeta|"], grid=g, basis="xyz")
    B = np.asarray(cs.compute_magnetic_field(np.asarray(d["x"]), basis="xyz", source_grid=src))
    bn = np.sum(B * np.asarray(d["n_rho"]), 1)
    w = np.asarray(g.weights) * np.asarray(d["|e_theta x e_zeta|"])
    Bm = np.sum(w * np.linalg.norm(B, axis=1)) / np.sum(w)
    return float(np.sqrt(np.sum(w * bn**2) / np.sum(w)) / Bm), float(np.max(np.abs(bn)) / Bm)


def physics(eq):
    d = eq.compute(["R0", "a", "R0/a", "<|B|>_vol"])
    Bv = float(d["<|B|>_vol"])
    hels = [(1, 0)] if q.helicity == 0 else [(1, L.nfp), (1, -L.nfp)]
    best = None
    for hel in hels:
        v = []
        for rr in (0.25, 0.5, 0.75, 1.0):
            o = QuasisymmetryBoozer(eq, helicity=hel, normalize=False, grid=LinearGrid(
                rho=np.array([rr]), M=2 * eq.M_grid, N=2 * eq.N_grid, NFP=eq.NFP, sym=False))
            o.build(verbose=0)
            v.append(float(np.linalg.norm(o.compute_unscaled(eq.params_dict))) / Bv)
        if best is None or v[1] < best[1][1]:
            best = (hel, v)
    ig = LinearGrid(rho=np.linspace(0, 1, 11), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
    iota = ig.compress(eq.compute("iota", grid=ig)["iota"])
    bn, bnmax = bn_rms(eq)
    return {"R0": float(d["R0"]), "a": float(d["a"]), "A": float(d["R0/a"]), "helicity": best[0],
            "qs_booz_rho.25,.5,.75,1": best[1], "iota_axis": float(iota[0]), "iota_edge": float(iota[-1]),
            "iota_min": float(iota.min()), "iota_max": float(iota.max()), "bn_rms": bn, "bn_max": bnmax,
            "axis_R0n": np.round(np.asarray(eq.axis.R_n), 5).tolist(),
            "axis_Z0n": np.round(np.asarray(eq.axis.Z_n), 5).tolist()}


log["nae_fixed_boundary"] = physics(eq)
stamp("NAE fixed boundary: " + json.dumps(log["nae_fixed_boundary"]))

# ---- free boundary
if a.resume:
    from desc.io import load
    eq = load(a.resume)
    stamp(f"resumed from {a.resume}")
shells = [int(v) for v in a.shells.split(",")] if a.shells else list(range(a.kstep, a.LMN + 1, a.kstep))
if shells[-1] != a.LMN and not a.shells:
    shells.append(a.LMN)
log["shells"] = []
for k in shells:
    # optimize(copy=True) returns a new eq: build the objective and constraints for THIS eq every shell
    constraints = (ForceBalance(eq=eq), FixCurrent(eq=eq), FixPressure(eq=eq), FixPsi(eq=eq))
    objective = ObjectiveFunction(VacuumBoundaryError(eq=eq, field=cs, field_fixed=True, field_grid=src))
    Rm = eq.surface.R_basis.modes[np.max(np.abs(eq.surface.R_basis.modes), 1) > k, :]
    Zm = eq.surface.Z_basis.modes[np.max(np.abs(eq.surface.Z_basis.modes), 1) > k, :]
    bc = tuple(c for c in (FixBoundaryR(eq=eq, modes=Rm) if len(Rm) else None,
                           FixBoundaryZ(eq=eq, modes=Zm) if len(Zm) else None) if c is not None)
    eq, res = eq.optimize(objective, constraints + bc, optimizer="proximal-lsq-exact", verbose=1,
                          maxiter=a.maxiter, ftol=1e-8, xtol=1e-10, gtol=1e-8, copy=True)
    bn, bnmax = bn_rms(eq)
    log["shells"].append(dict(k=k, bn_rms=bn, bn_max=bnmax, nit=int(res.get("nit", -1)),
                              message=str(res.get("message", ""))))
    stamp(f"free boundary shell k={k}: B.n rms {bn:.3e} max {bnmax:.3e} ({res.get('message', '')})")
    eq.save(os.path.join(out, f"eq_fb_k{k}.h5"))
    eq.save(os.path.join(out, "eq_fb.h5"))

log["free_boundary"] = physics(eq)
stamp("FREE BOUNDARY: " + json.dumps(log["free_boundary"]))
m = S.evaluate(cs, eq, vacuum=True)
log["coil_check"] = {k: m[k] for k in ("L", "kappa", "d_cc", "d_pc", "linked")}
stamp("coil check vs the free-boundary surface: " + json.dumps(log["coil_check"]))

# ---- Poincare: the coils' own field lines, with the free-boundary surfaces overlaid
try:
    fig, ax = plot_surfaces(eq)
    gt = LinearGrid(rho=np.linspace(0.1, 1.1, 11), theta=np.array([0.0]), zeta=np.array([0.0]), NFP=1)
    dt = eq.compute(["R", "Z"], grid=gt)
    fig, ax = poincare_plot(cs, np.asarray(dt["R"]), np.asarray(dt["Z"]), ntransit=a.ntransit, NFP=eq.NFP,
                            ax=ax, grid=src, size=1.5)
    fig.savefig(os.path.join(out, "poincare.png"), dpi=130, bbox_inches="tight")
    stamp("Poincare plot written")
except Exception as e:  # keep the score even if tracing fails
    log["poincare_error"] = repr(e)
    stamp(f"Poincare failed: {e!r}")

log["time_s"] = round(time.time() - t0)
json.dump(log, open(os.path.join(out, "score.json"), "w"), indent=1)
stamp(f"wrote {out}/score.json")
