"""Single stage (vacuum equilibrium + coils), weighted least squares with continuation.

    python paper/single_stage.py --eq EQ.h5 --start COILS.h5 --bounds B.json --out DIR \
        --fix-hinge-z 0 [--fold-max 60 --fold-sides-from STAGE2/result.json] \
        --A-min 6.8 --A-max 7.6 --iota-min 0.35 --qs-rho 0.25,0.5,0.75,1.0 \
        --tau-bn 5e-4 --tau-qs-rel 0.15 --pc-gap 0.002 --kmin 1 --kmax 6 --repair-steps

``proximal-lsq-composite`` on things = [eq, coils]: ForceBalance, Psi, pressure,
current, the boundary modes outside |m|, |n| <= k (and R_00), the coil current sum and
the hinge joint plane are proximal constraints; everything else is one weighted least-
squares objective whose bounded rows are kept exact as hinges in the model. Each
term's weight is 1 / (tau x its scale): a term whose rows all equal their tolerance
tau costs 1/2.

  physics   VacuumBoundaryBn (B.n on the moving boundary, normalization frozen at the
            start); QuasisymmetryBoozer on each --qs-rho surface (tau_qs = tau-qs-rel x
            the start's rms); AspectRatio in [A-min, A-max]; |iota| >= iota-min on 20
            surfaces; volume within +-vol-band of the start.
  coils     CoilLength, CoilCurvature at the bounds; CoilSetDistanceRows and
            PlasmaCoilSetDistanceRows (the plasma rows follow the moving boundary);
            optional ArcFold, CoilCurrentBand.

The coil bounds are weighted, not hard: the single stage is a physics refinement, and
the stage-2 polish (paper/stage2.py on the free-boundary equilibrium) restores hard
coil feasibility. Every step writes eq_kK.h5 and coils_kK.h5 and appends physics, the
dense coil check and the repair record to result.json; a rerun of the same command
continues after the last finished k. Ported from coil_sandbox/single_stage.py
(js/qa-poster), cut to the terms designs B and F used.
"""

import argparse
import glob
import json
import os
import re
import sys
import time


def floats(s):
    """Comma-separated floats."""
    return [float(v) for v in s.split(",")]


ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
g = ap.add_argument_group("problem")
g.add_argument("--eq", required=True, help="solved vacuum equilibrium (.h5)")
g.add_argument("--start", required=True, help="coilset (.h5)")
g.add_argument("--bounds", required=True)
g.add_argument("--out", required=True)
g.add_argument("--a", type=float, default=None, help="minor radius for the bounds")
g.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
g.add_argument("--fix-hinge-z", type=float, default=None, help="joint plane z (m)")
g.add_argument("--fold-max", type=float, default=None, help="fold cap (deg)")
g.add_argument("--fold-sides-from", default=None, help="stage-2 result.json")
g.add_argument("--current-band", type=float, default=None)
g.add_argument("--repair-steps", action="store_true", help="repair before every k")
g.add_argument("--dry", action="store_true", help="build, print the terms, exit")
g = ap.add_argument_group("physics targets")
g.add_argument("--helicity", default="1,0")
g.add_argument("--qs-rho", default="0.25,0.5,0.75,1.0")
g.add_argument("--A-min", type=float, required=True)
g.add_argument("--A-max", type=float, required=True)
g.add_argument("--iota-min", type=float, required=True)
g.add_argument("--vol-band", type=float, default=0.02, help="0: no volume term")
g = ap.add_argument_group("tolerances")
g.add_argument("--tau-bn", type=float, default=5e-4, help="rms of B.n / <|B|>")
g.add_argument("--tau-qs-rel", type=float, default=0.15, help="x the start's QS rms")
g.add_argument("--tau-A", type=float, default=0.01)
g.add_argument("--tau-V", type=float, default=0.005)
g.add_argument("--tau-iota", type=float, default=0.01)
g.add_argument("--tau-L", type=float, default=0.01)
g.add_argument("--tau-kappa", type=float, default=0.01)
g.add_argument("--tau-rows", type=float, default=0.01)
g.add_argument("--tau-shape", type=float, default=0.02, help="fold, current band")
g = ap.add_argument_group("grids and rows")
g.add_argument("--eval-mn", default="25,25", help="B.n grid M,N")
g.add_argument("--src-gl", type=int, default=64, help="Biot-Savart GL nodes per arc")
g.add_argument("--coil-grid-n", type=int, default=150, help="curvature LinearGrid N")
g.add_argument("--rows-select", type=float, default=1.6, help="x the bound")
g.add_argument("--cc-rows", type=int, default=40000)
g.add_argument("--pc-rows", type=int, default=32000)
g.add_argument("--pc-gap", type=float, default=None, help="plasma-row gap (m)")
g.add_argument("--plasma-mn", default="40,40")
g = ap.add_argument_group("continuation and solver")
g.add_argument("--kmin", type=int, default=1)
g.add_argument("--kmax", type=int, default=6)
g.add_argument("--maxiter", type=int, default=50, help="per step")
g.add_argument("--hessian", default="secant", choices=["gn", "secant"])
g.add_argument("--solve-tol", type=float, default=1e-6, help="inner eq solve tol")
g.add_argument("--force-chunk", type=int, default=400, help="ForceBalance jac chunk")
a = ap.parse_args()

from desc import set_device  # noqa: E402

set_device(a.device)

import numpy as np  # noqa: E402

import desc.objectives as O  # noqa: E402
from desc.coils import coil_quad_grid, coil_source_grid  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.io import load  # noqa: E402
from desc.optimize import Optimizer  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
import dense_check as D  # noqa: E402

os.makedirs(a.out, exist_ok=True)
hel = tuple(int(v) for v in a.helicity.split(","))
rhos = floats(a.qs_rho)


def boozer(eq, weight=1.0):
    """One QuasisymmetryBoozer per --qs-rho surface."""
    return [
        O.QuasisymmetryBoozer(
            eq,
            helicity=hel,
            normalize=False,
            weight=weight,
            name=f"QS Boozer rho {r:g}",
            grid=LinearGrid(
                rho=np.array([r]),
                M=2 * eq.M_grid,
                N=2 * eq.N_grid,
                NFP=eq.NFP,
                sym=False,
            ),
        )
        for r in rhos
    ]


def qs_rms(eq):
    """Root mean square over the --qs-rho surfaces of the Boozer QS residual norm."""
    v = []
    for o in boozer(eq):
        o.build(verbose=0)
        v.append(float(np.linalg.norm(o.compute_unscaled(eq.params_dict))))
    return float(np.sqrt(np.mean(np.square(v))))


def iota_grid(eq):
    """20 surfaces for the iota floor."""
    return LinearGrid(
        rho=np.linspace(0.05, 1.0, 20), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym
    )


def terms(eq, cs, A):
    """Every objective term, weighted by its tolerance (see the module docstring)."""
    eM, eN = (int(v) for v in a.eval_mn.split(","))
    pM, pN = (int(v) for v in a.plasma_mn.split(","))
    gbn = LinearGrid(rho=np.array([1.0]), M=eM, N=eN, NFP=eq.NFP, sym=eq.sym)
    s_bn = np.sqrt(np.sum(np.array(gbn.weights)))
    gi = iota_grid(eq)
    s_iota = np.sqrt(
        np.sum(np.array(gi.compress(gi.spacing[:, 0], surface_label="rho")))
    )
    cg = LinearGrid(N=a.coil_grid_n)
    s_kappa = np.sqrt(np.sum(np.array(cg.spacing[:, 2]) ** 2))
    imin = a.iota_min
    ib = (imin, np.inf) if A["iota_sign"] >= 0 else (-np.inf, -imin)
    t = [
        O.VacuumBoundaryBn(
            eq,
            cs,
            eval_grid=gbn,
            field_grid=coil_source_grid(cs, n=a.src_gl),
            norm=A["bn_norm"],
            weight=1 / (a.tau_bn * s_bn),
        ),
        *boozer(eq, weight=1 / (A["tau_qs"] * np.sqrt(len(rhos)))),
        O.AspectRatio(
            eq,
            bounds=(a.A_min, a.A_max),
            normalize=False,
            weight=1 / (a.tau_A * 0.5 * (a.A_min + a.A_max)),
        ),
        O.RotationalTransform(
            eq,
            bounds=ib,
            normalize=False,
            grid=gi,
            weight=1 / (a.tau_iota * imin * s_iota),
        ),
    ]
    if a.vol_band > 0:
        V0, f = A["V0"], a.vol_band
        t.append(
            O.Volume(
                eq,
                bounds=(V0 * (1 - f), V0 * (1 + f)),
                normalize=False,
                weight=1 / (a.tau_V * V0),
            )
        )
    if P["L"] is not None:
        t.append(
            O.CoilLength(
                cs,
                bounds=(0, P["L"]),
                normalize=False,
                grid=coil_quad_grid(cs),
                weight=1 / (a.tau_L * P["L"]),
            )
        )
    t.append(
        O.CoilCurvature(
            cs,
            bounds=(0, P["kappa"]),
            normalize=False,
            grid=cg,
            weight=1 / (a.tau_kappa * P["kappa"] * s_kappa),
        )
    )
    t += [
        O.CoilSetDistanceRows(
            cs,
            select_distance=a.rows_select * P["dcc"],
            bounds=(P["dcc"], np.inf),
            normalize=False,
            max_active_rows=a.cc_rows,
            weight=1 / (a.tau_rows * P["dcc"]),
        ),
        O.PlasmaCoilSetDistanceRows(
            eq,
            cs,
            select_distance=a.rows_select * P["dpc"],
            bounds=(P["dpc"], np.inf),
            normalize=False,
            max_active_rows=a.pc_rows,
            plasma_grid=LinearGrid(M=pM, N=pN, NFP=eq.NFP, sym=eq.sym),
            gap=a.pc_gap,
            weight=1 / (a.tau_rows * P["dpc"]),
        ),
    ]
    ts = a.tau_shape
    if a.fold_max is not None:
        fr = np.radians(a.fold_max)  # d cos = sin(f) df
        t.append(
            O.ArcFold(
                cs,
                bounds=(float(np.cos(fr)), np.inf),
                sides=A["sides"],
                weight=1 / (ts * np.sin(fr) * fr),
            )
        )
    if a.current_band is not None:
        f = a.current_band
        t.append(O.CoilCurrentBand(cs, bounds=(1 - f, 1 + f), weight=1 / (ts * f)))
    return t


def proximal_constraints(eq, cs, k):
    """Proximal constraints: force balance, |m|, |n| <= k boundary modes free."""
    R, Z = eq.surface.R_basis.modes, eq.surface.Z_basis.modes
    Rfix = np.vstack(([0, 0, 0], R[np.max(np.abs(R), axis=1) > k]))
    Zfix = Z[np.max(np.abs(Z), axis=1) > k]
    base = [
        c
        for c in O.get_fixed_boundary_constraints(eq)
        if not isinstance(c, (O.FixBoundaryR, O.FixBoundaryZ))
    ]
    cons = [
        O.ForceBalance(eq, jac_chunk_size=a.force_chunk),
        O.FixBoundaryR(eq, modes=Rfix),
        *base,
        O.FixSumCoilCurrent(cs),
    ]
    if len(Zfix):
        cons.append(O.FixBoundaryZ(eq, modes=Zfix))
    if a.fix_hinge_z is not None:
        cons.append(O.FixHingeHeight(cs, height=a.fix_hinge_z))
    n_free = (len(R) - len(Rfix)) + (len(Z) - len(Zfix))
    return tuple(cons), n_free


def physics(eq):
    """A, R0, a, V, iota range and Boozer QS / <|B|>_vol on the --qs-rho surfaces."""
    d = eq.compute(["R0", "a", "R0/a", "V", "<|B|>_vol"])
    gi = LinearGrid(
        rho=np.linspace(0, 1, 21), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym
    )
    iota = gi.compress(eq.compute("iota", grid=gi)["iota"])
    B = float(d["<|B|>_vol"])
    qs = []
    for o in boozer(eq):
        o.build(verbose=0)
        qs.append(float(np.linalg.norm(o.compute_unscaled(eq.params_dict))) / B)
    return dict(
        A=float(d["R0/a"]),
        R0=float(d["R0"]),
        a=float(d["a"]),
        V=float(d["V"]),
        iota_min=float(np.min(iota)),
        iota_max=float(np.max(iota)),
        qs_booz=dict(zip([f"{r:g}" for r in rhos], qs)),
    )


def coil_check(cs, eq):
    """Dense check against the current boundary (the B.n cache is keyed by eq)."""
    D._BN.clear()
    m = D.evaluate(cs, eq, vacuum=True)
    line = (
        f"bn {m['bn']:.4e} (max {m['bn_max']:.3e})  {D.report(m, P)}  "
        f"linked {m['linked']} -> {D.verdict(m, P)}"
    )
    return m, line


def do_repair(eq, tag):
    """Repair the boundary if it needs it; log it (PORT_PLAN 9.3)."""
    from bdry_repair import repair

    try:
        new, info = repair(eq, chunk=a.force_chunk)
    except Exception as e:  # noqa: BLE001  (keep the eq if the repair fails)
        print(f"REPAIR {tag}: FAILED {e!r}", flush=True)
        return eq, dict(at=tag, failed=repr(e)[:300])
    if info["action"] == "none":
        print(f"REPAIR {tag}: not needed", flush=True)
    else:
        mv = info["surface_moved_mm"]
        print(
            f"REPAIR {tag}: fired; surface moved rms {mv['rms']:.3f} / p99 "
            f"{mv['p99']:.2f} / max {mv['max']:.2f} mm",
            flush=True,
        )
    return new, dict(at=tag, **info)


# ------------------------------------------------------------------- start
rec_f, anchors_f = os.path.join(a.out, "result.json"), os.path.join(
    a.out, "anchors.json"
)
done = sorted(
    int(re.search(r"eq_k(\d+)\.h5$", f).group(1))
    for f in glob.glob(os.path.join(a.out, "eq_k*.h5"))
)
rec = json.load(open(rec_f)) if done and os.path.exists(rec_f) else None
if done:
    if done[-1] >= a.kmax:
        raise SystemExit(f"k={done[-1]} already done in {a.out}")
    a.eq = os.path.join(a.out, f"eq_k{done[-1]}.h5")
    a.start = os.path.join(a.out, f"coils_k{done[-1]}.h5")
    a.kmin = done[-1] + 1
    print(f"RESUME after k={done[-1]}", flush=True)
eq, cs = load(a.eq), load(a.start)
dd = eq.compute(["p", "current"])
if np.max(np.abs(dd["p"])) > 1e-8 or np.max(np.abs(dd["current"])) > 1e-8:
    raise SystemExit("vacuum equilibria only: p or current is nonzero")
coils = D.leaf_coils(cs)
P = D.load_bounds(a.bounds, eq, len(coils), a.a)
if a.fix_hinge_z is not None and not done:
    for c in coils:  # snap the start onto the joint plane
        h = np.array(c.hinges).reshape(-1, 3)
        h[:, 2] = a.fix_hinge_z
        c.hinges = h.reshape(-1)

if os.path.exists(anchors_f):  # frozen at the original start, kept across resumes
    A = json.load(open(anchors_f))
else:
    A = dict(V0=float(eq.compute("V")["V"]), tau_qs=a.tau_qs_rel * qs_rms(eq))
    A["iota_sign"] = float(
        np.sign(np.mean(np.asarray(eq.compute("iota", grid=iota_grid(eq))["iota"])))
    )
    A["bn_norm"] = None
    A["sides"] = None
    if a.fold_sides_from:
        A["sides"] = json.load(open(a.fold_sides_from))["info"]["fold_sides"]
    t0 = terms(eq, cs, A)[0]
    t0.build(verbose=0)
    A["bn_norm"] = float(t0.normalization)
    json.dump(A, open(anchors_f, "w"))
print(f"BOUNDS {P}\nANCHORS {A}", flush=True)
p0 = physics(eq)
m0, line0 = coil_check(cs, eq)
print(f"START physics {p0}\nSTART coils   {line0}", flush=True)
if rec is None:
    rec = dict(settings=vars(a), bounds=P, anchors=A, start=dict(physics=p0, coils=m0))
    rec["steps"], rec["repairs"] = [], []
if a.dry:
    obj = O.ObjectiveFunction(tuple(terms(eq, cs, A)))
    obj.build(verbose=0)
    for t in obj.objectives:
        err = np.asarray(t.compute_scaled_error(*[th.params_dict for th in t.things]))
        print(f"  {t.name[:34]:34s} rows {t.dim_f:7d} cost {0.5 * err @ err:.4g}")
    raise SystemExit(0)

if a.repair_steps and not done:
    eq, info = do_repair(eq, "start")
    rec["repairs"].append(info)
for k in range(a.kmin, a.kmax + 1):
    cons, n_free = proximal_constraints(eq, cs, k)
    obj = O.ObjectiveFunction(tuple(terms(eq, cs, A)))
    print(f"\n===== k={k}: {n_free} boundary modes free =====", flush=True)
    solve = dict(verbose=0, ftol=a.solve_tol, xtol=a.solve_tol, gtol=a.solve_tol)
    opts = dict(
        perturb_options={"order": 2, "verbose": 0},
        solve_options=solve,
        hessian=a.hessian,
    )
    t0 = time.time()
    (eq, cs), res = Optimizer("proximal-lsq-composite").optimize(
        [eq, cs],
        objective=obj,
        constraints=cons,
        maxiter=a.maxiter,
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-8,
        verbose=3,
        copy=False,
        options=opts,
    )
    wall = time.time() - t0
    eq.save(os.path.join(a.out, f"raw_eq_k{k}.h5"))  # before any repair
    p = physics(eq)
    m, line = coil_check(cs, eq)
    print(f"k={k} physics {p}\nk={k} coils   {line}", flush=True)
    print(f"k={k} solver  {res['message']}  nit {int(res['nit'])}  {wall:.0f} s")
    rinfo = None
    if a.repair_steps:
        eq, rinfo = do_repair(eq, f"k={k}")
        rec["repairs"].append(rinfo)
    eq.save(os.path.join(a.out, f"eq_k{k}.h5"))
    cs.save(os.path.join(a.out, f"coils_k{k}.h5"))
    fold = None
    if a.fold_max is not None:
        from desc.objectives._piecewise_planar import fold_angles

        fold = fold_angles(cs, sides=A["sides"]).tolist()
    rec["steps"].append(
        dict(
            k=k,
            n_free=n_free,
            wall_s=wall,
            nit=int(res["nit"]),
            message=str(res["message"]),
            physics=p,
            coils=m,
            fold_deg=fold,
            repair=rinfo,
        )
    )
    rec["repairs_fired"] = sum(r.get("action") == "repaired" for r in rec["repairs"])
    tmp = rec_f + ".tmp"
    json.dump(rec, open(tmp, "w"), indent=1, default=str)
    os.replace(tmp, rec_f)
print(f"\nwrote {a.out}: eq_k*.h5, coils_k*.h5, result.json")
