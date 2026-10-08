"""Audit an optimized equilibrium: what changed, and did it game the objective?

    python 922_stage1opt_results/analyze_result.py runs/ctrl_final/eq_lo.h5

eps_B is a RATIO -- sqrt(out-of-plane defect) / in-plane extent -- so it can be lowered
either honestly (flatter curve) or by inflating the denominator. And the DP segmentation and
plane normals are FROZEN at build, so the reported value can drift from the true eps_B.
Both are checked explicitly below, along with every constraint and the dropped QS term.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, "sandbox")
import boot  # noqa: E402

boot.setup(default="cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, HERE)
import jax.numpy as jnp  # noqa: E402
from clamshell import best_breaks, eps_from_frozen, freeze  # noqa: E402
from contour_curve import ContourClamshell, _curves_from_params  # noqa: E402
from desc.examples import get  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.io import load  # noqa: E402
from desc.objectives import (  # noqa: E402
    AspectRatio, ForceBalance, GenericObjective, QuasisymmetryTwoTerm)

import argparse as _ap
_p = _ap.ArgumentParser()
_p.add_argument("path", nargs="?", default="922_stage1opt_results/runs/ctrl_final/eq_lo.h5")
_p.add_argument("--bound-force", type=float, default=1e-4)
_p.add_argument("--bound-aspect", type=float, default=1e-2)
_p.add_argument("--bound-iota", type=float, default=1e-3)
_a = _p.parse_args()
path = _a.path
B_FORCE, B_ASPECT, B_IOTA = _a.bound_force, _a.bound_aspect, _a.bound_iota
eq0 = get("precise_QA")
eq1 = load(path)
NFP = eq0.NFP
CKW = dict(K=8, M=32, N=64, n=360, Mt=16, Nt=16)
print(f"baseline: precise_QA        optimized: {path}\n")

# ---------------------------------------------------------------- 1. constraints
print("=== 1. CONSTRAINT AUDIT (did it stay inside the bounds it was given?) ===")
gi = LinearGrid(rho=np.array([0.0, .25, .5, .75, 1.0]), M=eq0.M, N=eq0.N, NFP=NFP)
gq = LinearGrid(M=eq0.M, N=eq0.N, NFP=NFP, rho=np.linspace(0.1, 1.0, 5), sym=True)


def ev(obj, eq):
    obj.build(verbose=0)
    return np.asarray(obj.compute_scaled_error(*obj.xs(eq)))


f0, f1 = ev(ForceBalance(eq=eq0), eq0), ev(ForceBalance(eq=eq1), eq1)
print(f"  force balance |.|max (normalized)  {np.abs(f0).max():.3e} -> {np.abs(f1).max():.3e}"
      f"   bound {B_FORCE:.0e}   {'OK' if np.abs(f1).max() <= B_FORCE else '** VIOLATED **'}"
      f"   [{np.abs(f1).max()/np.abs(f0).max():.1f}x baseline]")
a0 = float(eq0.compute("R0/a")["R0/a"])
a1 = float(eq1.compute("R0/a")["R0/a"])
print(f"  aspect ratio                       {a0:.5f} -> {a1:.5f}"
      f"   bound 6+-{B_ASPECT:g}   {'OK' if abs(a1-6) <= B_ASPECT else '** VIOLATED **'}")
i0 = np.asarray(eq0.compute("iota", grid=gi)["iota"])
i1 = np.asarray(eq1.compute("iota", grid=gi)["iota"])
di = np.abs(i1 - i0).max()
print(f"  iota max |change|                  {di:.3e}   bound {B_IOTA:g}   "
      f"{'OK' if di <= B_IOTA else '** VIOLATED **'}")
q0, q1 = ev(QuasisymmetryTwoTerm(eq=eq0, helicity=(1, 0), grid=gq), eq0), \
         ev(QuasisymmetryTwoTerm(eq=eq1, helicity=(1, 0), grid=gq), eq1)
r0, r1 = np.sqrt(np.mean(q0**2)), np.sqrt(np.mean(q1**2))
print(f"  QS two-term rms  (** DROPPED **)   {r0:.4e} -> {r1:.4e}   "
      f"factor {r1/r0:.2f}x   <- the price of dropping it")

# ---------------------------------------------------------------- 2. objective
print("\n=== 2. OBJECTIVE: is the reported eps_1 improvement REAL? ===")
oE = ContourClamshell(eq0, quantity="eps1", **CKW)
oE.build(verbose=0)
E0 = np.asarray(oE.compute(eq0.params_dict))
# frozen-at-baseline read of the NEW equilibrium (what the optimizer was minimising)
E1_frozen = np.asarray(oE.compute(eq1.params_dict))
# honest read: re-freeze the DP on the new geometry
oE1 = ContourClamshell(eq1, quantity="eps1", **CKW)
oE1.build(verbose=0)
E1_fresh = np.asarray(oE1.compute(eq1.params_dict))
print(f"  eps_1 baseline                mean {E0.mean():.5f}")
print(f"  eps_1 final, FROZEN DP        mean {E1_frozen.mean():.5f}  ({(E1_frozen.mean()/E0.mean()-1)*100:+.2f}%)"
      f"   <- what the optimizer saw")
print(f"  eps_1 final, FRESH DP         mean {E1_fresh.mean():.5f}  ({(E1_fresh.mean()/E0.mean()-1)*100:+.2f}%)"
      f"   <- the truth")
gap = (E1_frozen - E1_fresh) / E1_fresh
print(f"  frozen-vs-fresh gap           max {gap.max()*100:+.2f}%  "
      f"({'frozen is an upper bound, as Danskin requires' if gap.min() >= -1e-9 else '** FROZEN BELOW FRESH -- IMPOSSIBLE, BUG **'})")

# ---------------------------------------------------------------- 3. gaming
print("\n=== 3. GAMING CHECK: numerator vs denominator of the ratio ===")
for tag, eq in (("baseline", eq0), ("final", eq1)):
    o = ContourClamshell(eq, quantity="eps1", **CKW)
    o.build(verbose=0)
    xs, ws = o._chain(eq.params_dict, o._constants)
    defect, extent, length = [], [], []
    for k in range(CKW["K"]):
        X, W = np.asarray(xs[k]), np.asarray(ws[k])
        fz = freeze(X, W, best_breaks(X, W, 1, stride=4)[1])
        Wa = W.sum()
        D = X - (W[:, None] * X).sum(0) / Wa
        defect.append(np.sqrt(np.sum(W * (D @ fz["normals"][0]) ** 2) / Wa))
        M = (D * W[:, None]).T @ D / Wa
        evs = np.linalg.eigvalsh(M)
        extent.append(np.sqrt(np.sqrt(max(evs[1], 0) * max(evs[2], 0))))
        length.append(float(np.sum(W) * 2 * np.pi / len(W)))
    print(f"  {tag:9} out-of-plane defect {np.mean(defect):.5f} m | in-plane extent "
          f"{np.mean(extent):.5f} m | contour length {np.mean(length):.4f} m")
    if tag == "baseline":
        d0, x0, l0 = np.mean(defect), np.mean(extent), np.mean(length)
    else:
        print(f"  -> defect {(np.mean(defect)/d0-1)*100:+.2f}%,  extent "
              f"{(np.mean(extent)/x0-1)*100:+.2f}%,  length {(np.mean(length)/l0-1)*100:+.2f}%")
        print("     (eps_1 falls HONESTLY only if the defect shrinks faster than the extent grows)")

# ---------------------------------------------------------------- 4. proxy validity
print("\n=== 4. IS THE PROXY STILL VALID on the new equilibrium? ===")
for tag, eq in (("baseline", eq0), ("final", eq1)):
    o = ContourClamshell(eq, quantity="eps1", **CKW)
    o.build(verbose=0)
    d = o.diagnostics()
    print(f"  {tag:9} monotonic={d['monotonic']}  I/G={d['I']/d['G']:+.2e}  "
          f"newton_res={d['worst_newton_residual']:.1e}  gap_seg={d['worst_frozen_gap_seg']:.4f}  "
          f"G={d['G']:.4e}")

# ---------------------------------------------------------------- 5. what moved
print("\n=== 5. WHAT CHANGED IN THE BOUNDARY ===")
mR, mZ = eq0.surface.R_basis.modes, eq0.surface.Z_basis.modes
dR = np.asarray(eq1.surface.R_lmn) - np.asarray(eq0.surface.R_lmn)
dZ = np.asarray(eq1.surface.Z_lmn) - np.asarray(eq0.surface.Z_lmn)
a = float(eq0.compute("a")["a"])
print(f"  minor radius a {a:.5f} -> {float(eq1.compute('a')['a']):.5f} m")
print(f"  |dR|max {np.abs(dR).max():.3e} m ({np.abs(dR).max()/a*100:.2f}% of a), "
      f"|dZ|max {np.abs(dZ).max():.3e} m ({np.abs(dZ).max()/a*100:.2f}% of a)")
idx = np.argsort(-np.abs(np.concatenate([dR, dZ])))[:6]
allm = np.vstack([mR, mZ])
alld = np.concatenate([dR, dZ])
tags = ["R"] * len(dR) + ["Z"] * len(dZ)
for j in idx:
    print(f"    {tags[j]}(m={int(allm[j,1]):+d}, n={int(allm[j,2]):+d})  {alld[j]:+.3e} m")
v0 = float(eq0.compute("V")["V"]); v1 = float(eq1.compute("V")["V"])
print(f"  volume {v0:.5f} -> {v1:.5f} m^3 ({(v1/v0-1)*100:+.2f}%)")

# ---------------------------------------------------------------- 6. C
print("\n=== 6. DID C MOVE? (independence of the two metrics) ===")
oC = ContourClamshell(eq0, quantity="C", **CKW)
oC.build(verbose=0)
C0 = np.asarray(oC.compute(eq0.params_dict))
oC1 = ContourClamshell(eq1, quantity="C", **CKW)
oC1.build(verbose=0)
C1 = np.asarray(oC1.compute(eq1.params_dict))
print(f"  C mean {C0.mean():.4f} -> {C1.mean():.4f}  ({(C1.mean()/C0.mean()-1)*100:+.2f}%)")
de = E1_fresh.mean() / E0.mean() - 1
dc = C1.mean() / C0.mean() - 1
if abs(dc) < 0.2 * abs(de):
    verdict = "INDEPENDENT levers (C barely responds)"
elif dc * de > 0:
    verdict = f"they TRACK (same sign, |dC/de1| = {abs(dc/de):.2f})"
else:
    verdict = f"they are in TENSION (opposite sign, |dC/de1| = {abs(dc/de):.2f})"
print(f"  eps_1 moved {de*100:+.2f}%, C moved {dc*100:+.2f}%  -> {verdict}")
