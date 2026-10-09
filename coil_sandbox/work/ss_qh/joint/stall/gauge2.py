"""Two tests of the theta-gauge story (DIAGNOSIS.md), on one equilibrium.

measure EQ: drift EQ by a tight re-solve (as gauge.py), then compare the original and the drifted state's
    QS cost with the stock (computational dtheta dzeta) measure and with the area measure |e_theta x e_zeta|,
    and iota, each at 1x, 2x, 3x grid resolution. A gauge-invariant integral changes only by quadrature
    error, which falls with resolution; a non-invariant one does not.

pin EQ EPS P: determinism of the re-solve with ForceBalance (+ LambdaCondensation(EPS, P) if EPS > 0).
    Solve once tightly from the loaded state (state A, the one-time move to the pinned gauge), then twice
    more from A + small random interior noise (R, Z, lambda), and compare the QS cost and iota of the
    three end states. Without the pin the end states differ along the gauge; with it they should agree.

    python work/ss_qh/joint/stall/gauge2.py measure EQ.h5
    python work/ss_qh/joint/stall/gauge2.py pin EQ.h5 EPS P [TOL MAXITER]
"""

import os
import sys

SB = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
sys.path.insert(0, os.path.join(SB, "sandbox"))
import boot  # noqa: E402

boot.setup(default="gpu")

import numpy as np  # noqa: E402

import common as S  # noqa: E402
import desc.objectives as O  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from gauge_fix import LambdaCondensation  # noqa: E402

RHO = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
HEL = (1, 4)


def qs_costs(eq, k):
    g = LinearGrid(rho=RHO, M=k * eq.M_grid, N=k * eq.N_grid, NFP=eq.NFP, sym=False)
    d = eq.compute(["f_C", "|B|", "|e_theta x e_zeta|"], grid=g, helicity=HEL)
    f2 = np.asarray(d["f_C"] / d["|B|"] ** 3) ** 2
    A = np.asarray(d["|e_theta x e_zeta|"])
    w = np.asarray(g.weights)
    return float(np.sum(f2 * w) / np.sum(w)), float(np.sum(f2 * A * w) / np.sum(A * w))


def iota(eq, k):
    g = LinearGrid(rho=np.linspace(0.05, 1.0, 20), M=k * eq.M_grid, N=k * eq.N_grid, NFP=eq.NFP, sym=eq.sym)
    return np.asarray(g.compress(eq.compute("iota", grid=g)["iota"]))


def force_cost(eq):
    fb = O.ObjectiveFunction(O.ForceBalance(eq))
    fb.build(verbose=0)
    return 0.5 * float(np.sum(np.asarray(fb.compute_scaled_error(fb.x(eq))) ** 2))


REF = os.environ.get("PIN_REF") == "1"
ANCHOR = {}


def solve(eq, eps, p, tol, maxiter):
    terms = [O.ForceBalance(eq)]
    if eps > 0:
        lc = LambdaCondensation(eq, weight=eps, power=p, ref=REF)
        terms.append(lc)
    obj = O.ObjectiveFunction(tuple(terms))
    if eps > 0 and REF:  # anchor at the FIRST solve's start (the loaded state), for every later solve too
        obj.build(verbose=0)
        if "L0" not in ANCHOR:
            ANCHOR["L0"] = lc.constants["L0"]
        lc._constants["L0"] = ANCHOR["L0"]
    _, res = eq.solve(objective=obj, constraints=O.get_fixed_boundary_constraints(eq), ftol=tol, xtol=tol,
                      gtol=tol, maxiter=maxiter, verbose=0)
    return int(res["nit"]), str(res["message"])[:40]


def measure(path):
    eq0, _ = S.load_equilibrium(path)
    eq1 = eq0.copy()
    nit, msg = solve(eq1, 0.0, 0.0, 1e-13, 100)
    print(f"drift solve: {nit} its, {msg}; force {force_cost(eq0):.6e} -> {force_cost(eq1):.6e}")
    for k in (1, 2, 3):
        s0, a0 = qs_costs(eq0, k)
        s1, a1 = qs_costs(eq1, k)
        i0, i1 = iota(eq0, k), iota(eq1, k)
        print(f"grid {k}x: QS stock {s0:.8e} rel change {abs(s1 - s0) / s0:.3e} | QS area {a0:.8e} rel change "
              f"{abs(a1 - a0) / a0:.3e} | iota max change {np.abs(i1 - i0).max():.3e}", flush=True)


def pin(path, eps, p, tol, maxiter):
    eq0, _ = S.load_equilibrium(path)
    eqA = eq0.copy()
    nit, msg = solve(eqA, eps, p, tol, maxiter)
    fA = force_cost(eqA)
    dL = np.linalg.norm(np.asarray(eqA.L_lmn - eq0.L_lmn))
    dRZ = np.linalg.norm(np.asarray(eqA.R_lmn - eq0.R_lmn)) + np.linalg.norm(np.asarray(eqA.Z_lmn - eq0.Z_lmn))
    sA, aA = qs_costs(eqA, 1)
    iA = iota(eqA, 1)
    print(f"eps {eps:g} p {p:g}: solve A {nit} its {msg!r}; force {force_cost(eq0):.6e} -> {fA:.6e}; "
          f"|dL| {dL:.3e} |dRZ| {dRZ:.3e}; |L| {np.linalg.norm(np.asarray(eqA.L_lmn)):.4f} "
          f"(orig {np.linalg.norm(np.asarray(eq0.L_lmn)):.4f}); QS stock {sA:.8e}", flush=True)
    rng = np.random.default_rng(0)
    for trial in range(2):
        eqB = eqA.copy()
        for nm in ("R_lmn", "Z_lmn", "L_lmn"):
            x = np.asarray(getattr(eqB, nm))
            setattr(eqB, nm, x + 1e-6 * rng.standard_normal(x.shape))
        nit, msg = solve(eqB, eps, p, tol, maxiter)
        sB, aB = qs_costs(eqB, 1)
        iB = iota(eqB, 1)
        print(f"  trial {trial}: {nit} its {msg!r}; force {force_cost(eqB):.6e}; QS stock rel diff vs A "
              f"{abs(sB - sA) / sA:.3e}; area {abs(aB - aA) / aA:.3e}; iota max diff {np.abs(iB - iA).max():.3e}; "
              f"|dL| vs A {np.linalg.norm(np.asarray(eqB.L_lmn - eqA.L_lmn)):.3e}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "measure":
        measure(sys.argv[2])
    else:
        tol = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-10
        maxiter = int(sys.argv[6]) if len(sys.argv) > 6 else 200
        pin(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), tol, maxiter)
