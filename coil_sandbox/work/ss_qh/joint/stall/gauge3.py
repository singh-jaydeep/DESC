"""Gauge pins: determinism and physical bias (DIAGNOSIS.md, follow-up).

KIND: none | tang (TangentialPin) | lambda (anchored LambdaCondensation, p = 0).

det EQ KIND EPS: solve tightly from EQ (the anchor), then twice more from that state + 1e-6 random interior noise;
    spread of QS cost and iota between the end states (smaller = more deterministic).

bias EQ BDRY_EQ KIND EPS: give EQ the boundary of BDRY_EQ and solve with the pin anchored at EQ's state (as the
    proximal re-solve does after a boundary step). Then re-solve that result WITHOUT any pin, and compare both
    with a plain (unpinned) solve of the same problem: a physically neutral pin changes neither QS, iota nor the
    force cost; a biased one leaves a non-equilibrium that the unpinned re-solve moves.

    python work/ss_qh/joint/stall/gauge3.py det EQ.h5 tang 1e-2
    python work/ss_qh/joint/stall/gauge3.py bias EQ.h5 BDRY.h5 tang 1e-2
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
from gauge_fix import LambdaCondensation, TangentialPin  # noqa: E402

RHO = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
TOL = float(os.environ.get("TOL", "1e-10"))
MAXITER = int(os.environ.get("MAXITER", "300"))
REANCHOR = os.environ.get("REANCHOR") == "1"


def qs_cost(eq):
    g = LinearGrid(rho=RHO, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
    d = eq.compute(["f_C", "|B|"], grid=g, helicity=(1, 4))
    return float(np.mean(np.asarray(d["f_C"] / d["|B|"] ** 3) ** 2))


def qs_booz(eq):
    out = []
    for r in (0.25, 0.5, 1.0):
        o = O.QuasisymmetryBoozer(eq, helicity=(1, 4), normalize=False,
                                  grid=LinearGrid(rho=np.array([r]), M=2 * eq.M_grid, N=2 * eq.N_grid, NFP=eq.NFP,
                                                  sym=False))
        o.build(verbose=0)
        out.append(float(np.linalg.norm(o.compute_unscaled(eq.params_dict))) / float(eq.compute("<|B|>_vol")["<|B|>_vol"]))
    return out


def iota(eq):
    g = LinearGrid(rho=np.linspace(0.05, 1.0, 20), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
    return np.asarray(g.compress(eq.compute("iota", grid=g)["iota"]))


def force_cost(eq):
    fb = O.ObjectiveFunction(O.ForceBalance(eq))
    fb.build(verbose=0)
    return 0.5 * float(np.sum(np.asarray(fb.compute_scaled_error(fb.x(eq))) ** 2))


def pin_term(eq, kind, eps):
    if kind == "tang":
        return TangentialPin(eq, weight=eps)
    if kind == "lambda":
        return LambdaCondensation(eq, weight=eps, power=0.0, ref=True)
    return None


INNER = os.environ.get("INNER", "gn")  # gn: lsq-exact (DESC default); exact/secant: lsq-composite hessian


def solve(eq, pin=None):
    terms = [O.ForceBalance(eq)] + ([pin] if pin is not None else [])
    obj = O.ObjectiveFunction(tuple(terms))
    obj.build(verbose=0)
    kw = {} if INNER == "gn" else dict(optimizer="lsq-composite", options={"hessian": INNER})
    _, res = eq.solve(objective=obj, constraints=O.get_fixed_boundary_constraints(eq), ftol=TOL, xtol=TOL, gtol=TOL,
                      maxiter=MAXITER, verbose=0, **kw)
    return int(res["nit"]), str(res["message"])[:30]


def report(label, eq, ref=None):
    q, i, fc, b = qs_cost(eq), iota(eq), force_cost(eq), qs_booz(eq)
    s = (f"{label:<28} force {fc:.4e}  QS {q:.6e}  Booz {b[0]:.3e} {b[1]:.3e} {b[2]:.3e}  iota [{i.min():.5f}, "
         f"{i.max():.5f}]")
    if ref is not None:
        s += (f"  | vs plain: QS rel {abs(q - ref[0]) / ref[0]:.2e}  iota max {np.abs(i - ref[1]).max():.2e}  "
              f"force rel {(fc - ref[2]) / ref[2]:+.2e}")
    print(s, flush=True)
    return q, i, fc


def det(path, kind, eps):
    eq0, _ = S.load_equilibrium(path)
    eqA = eq0.copy()
    pin = pin_term(eqA, kind, eps)  # anchored at the loaded state
    print(f"A: {solve(eqA, pin)}")
    qA, iA, _ = report(f"{kind} {eps:g} A", eqA)
    rng = np.random.default_rng(0)
    for t in range(2):
        eqB = eqA.copy()
        for nm in ("R_lmn", "Z_lmn", "L_lmn"):
            x = np.asarray(getattr(eqB, nm))
            setattr(eqB, nm, x + 1e-6 * rng.standard_normal(x.shape))
        pinB = pin_term(eqB, kind, eps)
        if pinB is not None:  # same anchor as A
            pinB.build(verbose=0)
            for k_ in ("R0", "Z0", "L0", "tR", "tZ"):
                if k_ in pin.constants:
                    pinB._constants[k_] = pin.constants[k_]
        nit = solve(eqB, pinB)
        qB, iB, _ = report(f"  trial {t} {nit}", eqB)
        print(f"     spread vs A: QS rel {abs(qB - qA) / qA:.2e}  iota max {np.abs(iB - iA).max():.2e}")


def bias(path, bpath, kind, eps, nsteps=10):
    """Move EQ's boundary to BDRY_EQ's in NSTEPS perturb (order 2) + solve increments, as proximal steps do,
    once unpinned (the reference) and once with the pin anchored at the START state throughout."""
    eq0, _ = S.load_equilibrium(path)
    eb, _ = S.load_equilibrium(bpath)
    dR = (np.asarray(eb.Rb_lmn) - np.asarray(eq0.Rb_lmn)) / nsteps
    dZ = (np.asarray(eb.Zb_lmn) - np.asarray(eq0.Zb_lmn)) / nsteps
    report("anchor (start eq)", eq0)
    anchor = pin_term(eq0.copy(), kind, eps)
    if anchor is not None:
        anchor.build(verbose=0)

    def walk(use_pin):
        e = eq0.copy()
        pin = None
        if use_pin:
            pin = pin_term(e, kind, eps)
            pin.build(verbose=0)
            for k_ in ("R0", "Z0", "L0", "tR", "tZ"):
                if k_ in anchor.constants:
                    pin._constants[k_] = anchor.constants[k_]
        terms = [O.ForceBalance(e)] + ([pin] if pin is not None else [])
        obj = O.ObjectiveFunction(tuple(terms))
        obj.build(verbose=0)
        its = []
        for _ in range(nsteps):
            if pin is not None and REANCHOR:  # anchor follows the accepted state
                fresh = pin_term(e.copy(), kind, eps)
                fresh.build(verbose=0)
                for k_ in ("R0", "Z0", "L0", "tR", "tZ"):
                    if k_ in fresh.constants:
                        pin._constants[k_] = fresh.constants[k_]
            e.perturb(deltas={"Rb_lmn": dR, "Zb_lmn": dZ}, objective=obj,
                      constraints=O.get_fixed_boundary_constraints(e), order=2, verbose=0)
            _, res = e.solve(objective=obj, constraints=O.get_fixed_boundary_constraints(e), ftol=TOL, xtol=TOL,
                             gtol=TOL, maxiter=MAXITER, verbose=0)
            its.append(int(res["nit"]))
        print(f"  walk {'pinned' if use_pin else 'plain'}: solve its per step {its}", flush=True)
        return e

    eP = walk(False)
    ref = report("plain (unpinned)", eP)
    eK = walk(True)
    report(f"{kind} {eps:g} pinned", eK, ref)
    print("unpinned re-solve:", solve(eK))
    report(f"{kind} {eps:g} -> unpinned", eK, ref)
    print("plain re-solved again (reference spread):", solve(eP))
    report("plain -> re-solved", eP, ref)


if __name__ == "__main__":
    if sys.argv[1] == "det":
        det(sys.argv[2], sys.argv[3], float(sys.argv[4]))
    else:
        bias(sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]))
