"""Stage 2 with the slack-free hinge augmented Lagrangian (`lsq-auglag-composite`, main repo).

    python run_al.py --eq EQ --start COILS.h5 --bounds B.json --out DIR [--vacuum] [--maxiter 800]

The problem is `sandbox/al_problem.build` (the ../devtools/coil_auglag recipe: node coil-coil
rows, plasma distance field, per-node |kappa|, kappa_MS, length; QuadraticFlux objective).
Per outer iteration: one log line per constraint group, the brute-force feasibility check of
../devtools/coil_auglag/check.py, and DIR/outer.json + DIR/x_outerK.npy. At the end, the
sandbox dense check (`common.evaluate`) against the bounds, DIR/result.h5 and result.json.
--maxiter 0: build everything at the start, report row counts, the field build and memory, stop.
--fix-hinge-z Z (arc coils): one demountable joint plane. Every hinge is snapped to z = Z and its z is
held there by FixParameters (linear, eliminated exactly; x and y stay free). Stellarator symmetry
maps z to -z, so only Z = 0 gives a single plane. START/RESULT also get the exact vertical-removal
check of sandbox/demount.py (diagnostic only, not a constraint).
--horizontal-hinges (arc coils): one horizontal joint plane PER COIL. Each coil's hinges are snapped to their
mean height and held level by sandbox/demount.HingeLevel (z_b - z_0 = 0; linear, eliminated exactly); each coil's height
stays free. Same diagnostics as --fix-hinge-z. Exclusive with it.
--link-mu MU: initial AL penalty MU on the signed per-pair coil-coil rows only (default: the solver's 10, like every
row). They are inactive unless a pair links, so a large MU costs nothing until then and makes the merit function
reject a step that links a pair (a crossing within one step, which no distance row can see).
"""

import sys

sys.path.insert(0, "sandbox")
import boot  # noqa: E402

boot.setup(default="gpu")

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import al_problem  # noqa: E402
import common as S  # noqa: E402

sys.path.insert(0, os.path.join(boot.MAIN, "devtools", "coil_auglag"))
from check import feasibility  # noqa: E402  (devtools brute-force check)

from desc.io import load  # noqa: E402
from desc.optimize import Optimizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--eq", required=True)
ap.add_argument("--start", required=True)
ap.add_argument("--bounds", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--vacuum", action="store_true")
ap.add_argument("--a", type=float, default=None)
ap.add_argument("--maxiter", type=int, default=800)
ap.add_argument("--gtol", type=float, default=1e-6)
ap.add_argument("--ctol", type=float, default=1e-6)
ap.add_argument("--field-n", type=int, default=50, help="QuadraticFlux field grid LinearGrid(N)")
ap.add_argument("--curv-n", type=int, default=300)
ap.add_argument("--row-n", type=int, default=256)
ap.add_argument("--cc-k", type=int, default=10000)
ap.add_argument("--cc-gap", choices=["bounds", "coils"], default="bounds",
                help="coil-coil node gap: from the bounds (row-n nodes) or from the coils at build")
ap.add_argument("--options", default="{}", help="JSON passed to the solver")
ap.add_argument("--link-mu", type=float, default=None,
                help="initial AL penalty on the signed per-pair coil-coil rows (linking guard); e.g. 1e8")
ap.add_argument("--fix-hinge-z", type=float, default=None,
                help="arc coils: snap every hinge to this height and hold its z there (one joint plane at 0)")
ap.add_argument("--horizontal-hinges", action="store_true",
                help="arc coils: each coil's hinges level (own height, free); sandbox/demount.HingeLevel")
ap.add_argument("--device", default=None)  # read by boot
a = ap.parse_args()
if a.horizontal_hinges and a.fix_hinge_z is not None:
    raise SystemExit("--horizontal-hinges and --fix-hinge-z are exclusive")
os.makedirs(a.out, exist_ok=True)
S.memwatch()

eq, kind = S.load_equilibrium(a.eq)
S.check_vacuum_choice(eq, kind, a.vacuum)
cs0 = load(a.start)
nc = S.n_unique(cs0)
shadow = None
if a.fix_hinge_z is not None or a.horizontal_hinges:
    import demount as D

    if not S.is_arc(cs0):
        raise SystemExit("--fix-hinge-z/--horizontal-hinges need arc coils")
    moved = 0.0
    for c in S.iter_unique(cs0):
        p = c.params_dict
        # FixParameters on hinges[:, 2] is the lab-frame height only if the leaf has no rigid transform
        if not (np.allclose(np.asarray(p["rotmat"]).reshape(3, 3), np.eye(3)) and np.allclose(p["shift"], 0)):
            raise SystemExit("--fix-hinge-z: a leaf coil has a rotmat/shift, so hinges[:, 2] is not its height")
        h0 = np.asarray(c.hinges)
        h = h0.reshape(-1, 3).copy()
        zt = a.fix_hinge_z if a.fix_hinge_z is not None else h[:, 2].mean()
        moved = max(moved, float(np.abs(h[:, 2] - zt).max()))
        h[:, 2] = zt
        c.hinges = h.reshape(h0.shape)
    what = (f"every hinge snapped to z = {a.fix_hinge_z} m and its z held there" if a.fix_hinge_z is not None
            else "each coil's hinges snapped to their mean height and held level (heights free)")
    print(f"JOINT PLANE: {what} (largest move {moved:.2e} m; the START line is after the snap)", flush=True)
    shadow = D.PlasmaShadow(eq)


def joint(cs):
    """Largest |hinge z - Z| (or per-coil hinge-height spread) and the vertical-removal verdict (sandbox/demount.py)."""
    if shadow is None:
        return {}, ""
    zs = [np.asarray(c.hinges).reshape(-1, 3)[:, 2] for c in S.iter_unique(cs)]
    if a.fix_hinge_z is not None:
        dz, lab = max(float(np.abs(z - a.fix_hinge_z).max()) for z in zs), f"max |hinge z - {a.fix_hinge_z}|"
    else:
        dz, lab = max(float(np.ptp(z)) for z in zs), f"max hinge-height spread (coil z {np.round([z.mean() for z in zs], 3)})"
    rep = D.demount_report(cs, eq, shadow=shadow)
    cl = min(min(r["hinge_clearance"]) for r in rep)
    bf = max(x["blocked_frac"] for r in rep for x in r["arcs"])
    ok = all(r["ok"] for r in rep)
    return (dict(max_hinge_dz=dz, min_hinge_clearance=cl, worst_arc_blocked=bf, removable=ok, report=rep),
            f"  joint: {lab} {dz:.1e} m, min hinge clearance {cl:+.3f} m, worst arc "
            f"blocked {100 * bf:.2f}% -> {'REMOVABLE' if ok else 'NOT REMOVABLE'}")

P = S.paper_bounds(eq, nc, a.a, a.bounds)
print(f"eq {a.eq} NFP {eq.NFP} sym {eq.sym}; start {a.start}: {nc} unique coils ({type(next(S.iter_unique(cs0))).__name__}), "
      f"{len(cs0)} top-level; {'VACUUM' if a.vacuum else 'finite beta'}\nBOUNDS {S.fmt_bounds(P)}", flush=True)

t0 = time.time()
obj, cons, info = al_problem.build(eq, cs0, P, vacuum=a.vacuum, field_N=a.field_n, curv_N=a.curv_n,
                                   row_N=a.row_n, cc_K=a.cc_k, cc_gap=a.cc_gap)
if a.fix_hinge_z is not None:
    from desc.objectives import FixParameters

    zi = np.arange(int(next(S.iter_unique(cs0)).B)) * 3 + 2  # the z slot of each hinge

    def z_mask(t):
        return {k: (zi if k == "hinges" else False) for k in t} if isinstance(t, dict) else [z_mask(q) for q in t]

    # no target: FixParameters holds the values in cs0, which the snap above set to --fix-hinge-z
    cons = (*cons, FixParameters(cs0, params=z_mask(cs0.params_dict), name="hinge joint plane"))
    info["joint_plane"] = f"hinge z fixed at {a.fix_hinge_z} ({len(zi)} per coil)"
if a.horizontal_hinges:
    cons = (*cons, D.HingeLevel(cs0))
    info["joint_plane"] = "hinges level per coil (HingeLevel), heights free"
obj.build(verbose=0)
nonlinear = [c for c in cons if not c.linear]
for c in nonlinear:
    if not c.built:
        t1 = time.time()
        c.build(verbose=0)
        print(f"  built {c.name} in {time.time() - t1:.1f} s", flush=True)
cc = next(c for c in nonlinear if type(c).__name__ == "CoilSetDistanceRows")
pc = next(c for c in nonlinear if type(c).__name__ == "PlasmaCoilDistanceField")
params0 = [{k: np.asarray(v) for k, v in q.items()} for q in obj.unpack_state(obj.x(cs0), False)[0]]
print(f"BUILD {time.time() - t0:.1f} s; cc gap {float(np.max(cc._constants['gap'])) * 1e3:.3f} mm on "
      f"{cc._n_nodes} nodes/coil, pc node margin {pc.field_info['node_margin'] * 1e3:.3f} mm on {info['n_nodes']} "
      f"nodes/coil; cc active rows at start "
      f"{int(cc.active_row_count(params0))} of {a.cc_k}; field {getattr(pc, 'field_info', {})}", flush=True)
print(f"INFO {info}", flush=True)
for c in nonlinear:
    print(f"  {c.name:32s} rows {getattr(c, 'num_row_ids', c.dim_f)}", flush=True)
check_cs = cs0.copy()
tree = S.surface_tree(eq)
m0 = S.evaluate(cs0, eq, tree, vacuum=a.vacuum)
j0, jl0 = joint(cs0)
print(f"START  bn {m0['bn']:.4e}  {S.report(m0, P)}  linked {m0['linked']}  -> {S.verdict(m0, P)}{jl0}", flush=True)
true0 = feasibility(check_cs, cc, pc, params0)
print(f"START  true: cc {true0['cc']:.5f} pc {true0['pc']:.5f} max L {max(true0['length']):.4f} "
      f"max |k| {max(true0['curvature']):.3f} linked {true0['linked']} gap {true0['gap']}", flush=True)
if a.maxiter == 0:
    print("maxiter 0: build check only", flush=True)
    raise SystemExit(0)

sizes = [getattr(c, "num_row_ids", c.dim_f) for c in nonlinear]
offsets = np.concatenate([[0], np.cumsum(sizes)])
outer_log = []


def report(h):
    x = np.asarray(h["x"])
    k = h["outer"]
    np.save(os.path.join(a.out, f"x_outer{k:02d}.npy"), x)
    params = [{q: np.asarray(v) for q, v in p.items()} for p in obj.unpack_state(x, False)[0]]
    true = feasibility(check_cs, cc, pc, params)
    ids, y, mu, viol = (np.asarray(h[q]) for q in ("ids", "y", "mu", "violation"))
    grp = np.where(ids >= 0, np.searchsorted(offsets, ids, side="right") - 1, -1)
    groups = {}
    for j, c in enumerate(nonlinear):
        m = grp == j
        if np.any(m):
            groups[c.name] = dict(rows=int(m.sum()), mu=[float(np.min(mu[m])), float(np.median(mu[m])),
                                                         float(np.max(mu[m]))],
                                  max_y=float(np.abs(y[m]).max()), nonzero_y=int(np.sum(y[m] != 0)),
                                  max_violation=float(viol[m].max()))
    outer_log.append(dict(outer=k, inner_iterations=h["inner_iterations"], optimality=h["optimality"],
                          gtolk=h["gtolk"], inner_converged=bool(h["inner_converged"]),
                          constr_violation=h["constr_violation"], ctolk=h["ctolk"], groups=groups,
                          true=dict(true, linked=[list(map(int, q)) for q in true["linked"]]),
                          t=time.time() - t_run))
    json.dump(outer_log, open(os.path.join(a.out, "outer.json"), "w"), indent=1, default=float)
    print(f"[outer {k}] {h['inner_iterations']} inner its, opt {h['optimality']:.2e} (gtolk {h['gtolk']:.1e}, "
          f"{'met' if h['inner_converged'] else 'NOT met'}), violation {h['constr_violation']:.2e} "
          f"(ctolk {h['ctolk']:.1e}), {time.time() - t_run:.0f} s", flush=True)
    for name, g in groups.items():
        print(f"   {name[:30]:30s} rows {g['rows']:6d} mu {g['mu'][0]:.1e}/{g['mu'][1]:.1e}/{g['mu'][2]:.1e} "
              f"max|y| {g['max_y']:.2e} (nonzero {g['nonzero_y']}) max viol {g['max_violation']:.2e}", flush=True)
    print(f"   true: cc {true['cc']:.5f} m, pc {true['pc']:.5f} m, max L {max(true['length']):.4f}, "
          f"max |k| {max(true['curvature']):.3f}, linked {true['linked']}, gap {true['gap']}", flush=True)


def save_x(x):
    np.save(os.path.join(a.out, "x_last.npy"), np.asarray(x))
    return False


options = {"outer_callback": report, "callback": save_x, **json.loads(a.options)}
if a.link_mu is not None:
    j = nonlinear.index(cc)  # constraint row id = group offset + local id; signed rows follow the candidates
    link_ids = offsets[j] + cc._num_candidates + np.arange(len(cc._pairs))
    options["initial_penalty_rows"] = {"ids": link_ids, "mu": a.link_mu}
    print(f"LINK GUARD: initial mu {a.link_mu:.1e} on the {len(link_ids)} signed per-pair rows "
          f"(ids {link_ids[0]}..{link_ids[-1]}); every other row starts at the solver default", flush=True)
t_run = time.time()
(cs1,), res = Optimizer("lsq-auglag-composite").optimize(
    cs0, objective=obj, constraints=cons, maxiter=a.maxiter, verbose=1, ftol=0, xtol=0, gtol=a.gtol,
    ctol=a.ctol, options=options, copy=True)
wall = time.time() - t_run
cs1.save(os.path.join(a.out, "result.h5"))
m1 = S.evaluate(cs1, eq, tree, vacuum=a.vacuum)
active = [k for k, v in S.margins(m1, P).items() if abs(v - 1) <= 0.01]
print(f"SOLVER {res['message']} | {len(res['outer_history'])} outer / {res['nit']} inner its, opt "
      f"{res['optimality']:.2e}, violation {res['constr_violation']:.2e}, {wall:.0f} s", flush=True)
print(f"START  bn {m0['bn']:.4e}  {S.report(m0, P)}  -> {S.verdict(m0, P)}")
j1, jl1 = joint(cs1)
print(f"RESULT bn {m1['bn']:.4e}  {S.report(m1, P)}  linked {m1['linked']}  -> {S.verdict(m1, P)}  active {active}{jl1}",
      flush=True)
json.dump(dict(settings=vars(a), bounds=P, info=info, solver=dict(message=str(res["message"]), nit=int(res["nit"]),
               n_outer=len(res["outer_history"]), optimality=float(res["optimality"]),
               constr_violation=float(res["constr_violation"]), wall_s=wall),
               start=m0, result=m1, margins=S.margins(m1, P), active=active, joint_start=j0, joint_result=j1),
          open(os.path.join(a.out, "result.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print(f"wrote {a.out}/result.h5, result.json, outer.json", flush=True)
