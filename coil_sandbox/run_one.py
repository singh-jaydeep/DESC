"""One coil optimization for any equilibrium, with the precise_QH sweep's recipe.

    # stage 2: minimize the normal field error under the coil constraints
    python run_one.py --eq precise_QA --start work/qa/start.h5 --out work/qa/run1

    # feasibility only: move a start inside the bounds first (stops once the dense check passes)
    python run_one.py --eq precise_QA --start work/qa/start.h5 --mode feasibility --margin 0.25 \\
        --out work/qa/feas
    python run_one.py --eq precise_QA --start work/qa/feas/result.h5 --out work/qa/run1

  objective    stage2:      QuadraticFlux on LinearGrid(M, N, NFP, sym), targeting (B_coil + B_plasma).n = 0;
                            B_plasma is the plasma's own field by virtual casing (finite beta,
                            net current), computed once at build; --vacuum drops it
               feasibility: FixParameters(coilset) at --anchor-weight (default 1e-3, a weak anchor to
                            the start; ~1 keeps the result close to the start in parameter space)
  constraints  CoilLength (each coil <= length-mult x L), CoilCurvature (max |kappa|),
               CoilMeanSquaredCurvature, convexity (--convex-form signed: in-plane signed
               curvature >= --convex-kmin at every node, per plane (whole planar coil, or each
               arc of an arc coil), smooth [sandbox/convexity.py]; total: the paper's int |kappa| ds <= 2 pi + excess, kinked
               wherever a node's curvature changes sign), CoilSetDistancePenalty (all pairs, two-pass, signed),
               PlasmaCoilSetDistancePenalty (two-pass); a bound set to null in the bounds
               file drops its constraint (default bounds: no convexity)
  linear       stage2: FixSumCoilCurrent (each current free); feasibility: FixCoilCurrent
  solver       lsq-auglag, second-order constraint term, trust-radius cap 0.5,
               max_inner_iter 100, ctol 1e-4

Bounds: the reactor set of Gil et al. (Table III) scaled by a / 1.7, for the number of
independent coils in the start (`sandbox/common.py: paper_bounds`), or `--bounds file.json`.
Every verdict is against those bounds; the run is told a backed-off set (--dcc-backoff,
--dpc-backoff, and for arcs the curvature backoffs and the segment sagitta), printed at the start.

Writes to --out: result.h5, result.json (settings, bounds, start/result metrics, history, solver
info), result.png, and ckpt_itNNNN.h5 checkpoints.

Memory: wrap it in a cgroup cap where one is available (WSL2 has no OOM killer):
    systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 python run_one.py ...
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox"))
import boot  # noqa: E402

DEVICE = boot.setup(default="gpu")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import common as S  # noqa: E402
from convexity import CoilSignedCurvature  # noqa: E402
from corners import CoilCornerAngle  # noqa: E402
import demount as D  # noqa: E402
import desc.objectives as O  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.io import load  # noqa: E402
from desc.optimize import Optimizer  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = ap.add_argument_group("problem")
    g.add_argument("--eq", required=True, help="desc.examples name, DESC .h5, wout_*.nc, or input file")
    g.add_argument("--start", required=True, help="coilset .h5")
    g.add_argument("--out", required=True, help="output directory")
    g.add_argument("--mode", choices=["stage2", "feasibility"], default="stage2")
    g.add_argument("--length-mult", type=float, default=1.0, help="length bound = this x L")
    g.add_argument("--margin", type=float, default=0.0,
                   help="aim this fraction INSIDE every bound (feasibility starts use 0.25)")
    g.add_argument("--fix-hinges", action="store_true", help="arc coils: hold the hinge points fixed")
    g.add_argument("--anchor-weight", type=float, default=1e-3,
                   help="feasibility mode: weight of the pull toward the start (FixParameters)")
    g.add_argument("--margin-skip", default="",
                   help="comma-separated bounds the margin does NOT apply to (L, kappa, kms, dcc, dpc)")
    g.add_argument("--a", type=float, default=None, help="minor radius for the bounds (default: DESC's)")
    g.add_argument("--bounds", default=None, help="bounds JSON (sandbox/common.py: paper_bounds)")
    g.add_argument("--device", default=None, help="gpu or cpu (default gpu)")
    g.add_argument("--vacuum", action="store_true",
                   help="ignore the plasma's own field (only right for vacuum equilibria)")
    g = ap.add_argument_group("iterations and output")
    g.add_argument("--maxiter", type=int, default=800)
    g.add_argument("--check-every", type=int, default=50, help="dense check of the iterate every N iterations")
    g.add_argument("--ckpt-every", type=int, default=50, help="checkpoint every N iterations (0: never)")
    g.add_argument("--diag-log", default=None, help="step-level solver log (JSONL) file name in --out")
    g = ap.add_argument_group("enforced bounds")
    g.add_argument("--dcc-backoff", type=float, default=0.025)
    g.add_argument("--dpc-backoff", type=float, default=0.02)
    g.add_argument("--kappa-lb", choices=["zero", "none"], default="zero",
                   help="CoilCurvature lower bound: 0 (default) or -inf. |kappa| >= 0 always, but with 0 the "
                        "slack is floored where |kappa| has its kink (inflections, common on planar coils)")
    g.add_argument("--no-arc-backoffs", action="store_true", help="skip the arc curvature and sagitta backoffs")
    g.add_argument("--convex-form", choices=["signed", "total"], default="signed",
                   help="how convexity is enforced (the verdict always uses the total-curvature excess)")
    g.add_argument("--convex-kmin", type=float, default=0.0,
                   help="signed form: lower bound on the in-plane signed curvature [1/m]")
    g.add_argument("--convex-backoff", type=float, default=0.5,
                   help="total form: enforce this fraction of the allowed total-curvature excess")
    g.add_argument("--demount-margin", type=float, default=None,
                   help="metres every hinge must sit OUTSIDE the plasma's vertical shadow, so its "
                        "two pieces can be lifted clear (sandbox/demount.py); off by default")
    g.add_argument("--demount-backoff", type=float, default=0.02,
                   help="extra metres enforced on --demount-margin; verdicts use the bare margin")
    g.add_argument("--demount-arcs", action="store_true",
                   help="also penalise arc NODES that cannot leave along their arc's direction "
                        "(the hinge condition is necessary but not sufficient)")
    g.add_argument("--demount-delta", type=float, default=0.5,
                   help="--demount-arcs: radial scale of the penalty (m)")
    g.add_argument("--demount-eps", type=float, default=0.25,
                   help="--demount-arcs: vertical softening of the penalty (m)")
    g.add_argument("--level-hinges", action="store_true",
                   help="hold every coil's hinges at a COMMON HEIGHT, chosen by the optimizer: "
                        "horizontal joint chords without a shared joint plane. Linear, so it is "
                        "factorized out exactly and costs the solver nothing. Weaker than "
                        "--fix-hinge-z, which pins the height too")
    g.add_argument("--fix-hinge-z", type=float, default=None,
                   help="snap every hinge to this height and hold it there: a common demountable "
                        "joint plane. Stellarator symmetry makes 0 (the midplane) the only height "
                        "that gives ONE plane for the whole machine")
    g.add_argument("--corner-max", type=float, default=None,
                   help="arc coils: bound the tangent TURN ANGLE at every hinge by this many degrees "
                        "(per hinge, not a max over hinges; see sandbox/corners.py). Unset = no corner "
                        "constraint, which is the historical behaviour -- corners are otherwise only "
                        "reported, and reach 117 deg in the 16-coil arcB2 optima")
    g.add_argument("--planarity", choices=["auto", "on", "off"], default="auto",
                   help="spline coils: constrain each C0 segment's knots to be coplanar "
                        "(auto: on whenever the coils are splines with break points)")
    g.add_argument("--planarity-tol", type=float, default=0.0,
                   help="0: exact (target 0); >0: allow this out-of-plane distance [m] as a bound")
    g.add_argument("--speed-floor", type=float, default=0.0,
                   help="constrain |x_s| / mean|x_s| >= this at every node (0 disables). For spline "
                        "coils this is better conditioned than a finer curvature grid: collocation "
                        "error enters |x_s| LINEARLY but kappa cubed. See sandbox/speed.py")
    g.add_argument("--arclength-weight", type=float, default=0.0,
                   help="add CoilArclengthVariance to the OBJECTIVE at this weight (0 disables). "
                        "Splines have a non-unique parameterization, and the optimizer exploits it: "
                        "kappa = |x' x x''|/|x'|^3 blows up wherever |x'| collapses, so a cusp BETWEEN "
                        "constraint nodes is free field error. MEASURED on a cold spline run: kappa "
                        "0.90 on the 301-node grid vs 3.45 on a fine one, the peak sitting exactly at "
                        "|x_s| = 0.116-0.167 of its mean. This penalises the cause; it lives in the "
                        "objective so it stays out of the second-order CONSTRAINT Hessian")
    g.add_argument("--torsion-weight", type=float, default=0.0,
                   help="add CoilTorsion to the OBJECTIVE at this weight (a soft planarity pull that "
                        "lets the AL explore off the planar manifold; 0 disables it)")
    g = ap.add_argument_group("grids")
    g.add_argument("--coil-grid-n", type=int, default=150, help="kappa_MS and both distances")
    g.add_argument("--kappa-grid-n", type=int, default=None,
                   help="separate, finer grid for CURVATURE and convexity (default: --coil-grid-n). "
                        "Spline coils need it: kappa = |x' x x''|/|x'|^3 spikes wherever the "
                        "parameterization slows, and a spike BETWEEN constraint nodes is free to the "
                        "optimizer. MEASURED on a cold spline run: kappa 0.90 on the 301-node grid, "
                        "3.45 on a fine one, with the peak exactly at |x_s| = 0.167 of its mean")
    g.add_argument("--eval-m", type=int, default=25)
    g.add_argument("--eval-n", type=int, default=25)
    g.add_argument("--plasma-m", type=int, default=64, help="plasma grid for d_pc")
    g.add_argument("--plasma-n", type=int, default=64)
    g.add_argument("--src-gl", type=int, default=24, help="GL nodes per arc (arcs)")
    g.add_argument("--src-n", type=int, default=400, help="LinearGrid(N) source grid (smooth coils)")
    g.add_argument("--vc-m", type=int, default=None, help="virtual-casing source grid M (default 2 x eq.M_grid)")
    g.add_argument("--vc-n", type=int, default=None, help="virtual-casing source grid N (default 2 x eq.N_grid)")
    g.add_argument("--vc-chunk", type=int, default=64, help="chunk size of the virtual-casing integral")
    g = ap.add_argument_group("constraint evaluation and memory")
    g.add_argument("--dcc-k", type=int, default=20000, help="two-pass capacity, coil-coil (0: dense)")
    g.add_argument("--dpc-k", type=int, default=20000, help="two-pass capacity per coil, plasma-coil")
    g.add_argument("--no-signed", action="store_true", help="drop the topology guard on the coil-coil rows")
    g.add_argument("--dcc-neighbors", type=int, default=None, help="coil-coil neighbours per coil (default all)")
    g.add_argument("--dcc-chunk", type=int, default=4)
    g.add_argument("--dpc-chunk", type=int, default=1)
    g.add_argument("--qf-chunk", type=int, default=25)
    g.add_argument("--coil-jac-chunk", type=int, default=None,
                   help="jac_chunk_size for the per-coil constraints (length, curvature, kappa_MS, "
                        "convexity). Their Jacobians are dim_f x dim_x with dim_f = coils x grid "
                        "nodes, so a fine --kappa-grid-n multiplies them; chunk to fit VRAM")
    g = ap.add_argument_group("solver")
    g.add_argument("--no-second-order", action="store_true")
    g.add_argument("--so-chunk", type=int, default=16)
    g.add_argument("--so-rtol", type=float, default=0.05)
    g.add_argument("--max-tr", type=float, default=0.5, help="trust-radius cap (inf to remove)")
    g.add_argument("--max-inner-iter", type=int, default=100)
    g.add_argument("--ctol", type=float, default=1e-4)
    return ap.parse_args()


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    S.memwatch()
    eq, kind = S.load_equilibrium(a.eq)
    S.check_vacuum_choice(eq, kind, a.vacuum)
    cs0 = load(a.start)
    nc = S.n_unique(cs0)
    arc = S.is_arc(cs0)
    spline = S.is_spline(cs0)
    pw = S.is_piecewise(cs0)  # has C0 corners: arc hinges, or spline breaks
    tree = S.surface_tree(eq)
    if a.fix_hinge_z is not None:
        if not arc:
            raise SystemExit("--fix-hinge-z needs arc coils")
        moved = 0.0
        for c in S.iter_unique(cs0):
            h0 = np.asarray(c.hinges)
            h = h0.reshape(-1, 3).copy()
            moved = max(moved, float(np.abs(h[:, 2] - a.fix_hinge_z).max()))
            h[:, 2] = a.fix_hinge_z
            c.hinges = h.reshape(h0.shape)
        print(f"JOINT PLANE: every hinge snapped to z = {a.fix_hinge_z} m and held there "
              f"(largest move {moved:.4f} m; the START line below is AFTER the snap)", flush=True)

    B = S.paper_bounds(eq, nc, a.a, a.bounds)
    P = {k: B[k] for k in S.UPPER + S.LOWER}  # the judged set
    if P["L"] is not None:
        P["L"] *= a.length_mult
    T = S.scaled_set(P, a.margin)  # the target set
    for k in filter(None, (v.strip() for v in a.margin_skip.split(","))):
        if k not in T:
            raise SystemExit(f"--margin-skip: unknown bound {k!r} (use {', '.join(S.UPPER + S.LOWER)})")
        T[k] = P[k]
    if P["convex"] is not None:
        T["convex"] = P["convex"]  # a margin on an excess that is 0 when convex means nothing
    ENF = dict(T, dcc=T["dcc"] * (1 + a.dcc_backoff), dpc=T["dpc"] * (1 + a.dpc_backoff))
    if ENF["convex"] is not None:
        ENF["convex"] *= a.convex_backoff
    if pw and not a.no_arc_backoffs:
        ENF["kappa"] *= 0.995
        if ENF["kms"] is not None:
            ENF["kms"] *= 0.99
        L_coil = P["L"] if P["L"] is not None else (P["Ltot"] / nc if P["Ltot"] is not None else None)
        if L_coil is not None:  # segment sagitta on the enforcement grid
            ENF["dcc"] += 1.4 * P["kappa"] * (L_coil / a.coil_grid_n) ** 2 / 8

    src, src_kind = S.source_grid(cs0, a.src_gl, a.src_n)
    lg, lg_kind = S.quad_grid(cs0, a.src_gl, a.src_n)  # length: composite ACROSS the C0 corners
    cg = LinearGrid(N=a.coil_grid_n)
    kg = LinearGrid(N=a.kappa_grid_n) if a.kappa_grid_n else cg  # curvature and convexity
    exp = S.expected_coils(cs0, eq)
    print(f"eq {a.eq} ({kind}) NFP {eq.NFP} sym {eq.sym}; start {a.start}: {nc} unique coils, arc={arc}, "
          f"expect {exp} coils; source grid {src_kind} ({src.num_nodes} nodes/coil), length grid "
          f"{lg_kind} ({lg.num_nodes}), constraint grid {cg.num_nodes}, curvature grid {kg.num_nodes}; "
          f"mode {a.mode}; "
          f"{'VACUUM (plasma field ignored)' if a.vacuum else 'plasma field included (virtual casing)'}\n"
          f"SETTINGS {vars(a)}\nBOUNDS (judged, {B['scale']}, a={B['a']:.5f}) {S.fmt_bounds(P)}\n"
          + (f"TARGET (margin {a.margin}) {S.fmt_bounds(T)}\n" if a.margin else "")
          + f"ENFORCED {S.fmt_bounds(ENF)}"
          + (f"  (convexity: {a.convex_form}" + (f", kappa_s >= {a.convex_kmin})" if a.convex_form == "signed"
                                                  else ")") if ENF["convex"] is not None else ""), flush=True)

    cons = []
    CJ = {"jac_chunk_size": a.coil_jac_chunk} if a.coil_jac_chunk else {}
    if ENF["L"] is not None:
        cons.append(O.CoilLength(cs0, bounds=(0.0, ENF["L"]), normalize_target=True, grid=lg, **CJ))
    if ENF["Ltot"] is not None:  # budget over the INDEPENDENT coils (one half period), not per coil
        cons.append(O.CoilLength(cs0, bounds=(0.0, ENF["Ltot"]), loss_function="sum", normalize_target=True,
                                 grid=lg, name="total coil length", **CJ))
    if ENF["kappa"] is not None:
        cons.append(O.CoilCurvature(cs0, bounds=(0.0 if a.kappa_lb == "zero" else -np.inf, ENF["kappa"]),
                                    normalize_target=True, grid=kg, **CJ))
    if ENF["kms"] is not None:
        cons.append(O.CoilMeanSquaredCurvature(cs0, bounds=(0.0, ENF["kms"]), normalize_target=True, grid=cg,
                                              **CJ))
    if ENF["convex"] is not None and a.convex_form == "total":
        cons.append(O.CoilIntegratedCurvature(cs0, bounds=(0.0, 2 * np.pi + ENF["convex"]), normalize_target=True,
                                              grid=kg))
    elif ENF["convex"] is not None:
        from desc.coils import (
            FourierPlanarCoil, PiecewisePlanarArcCoil, PolarPlanarArcCoil, SplineXYZCoil)

        if not all(isinstance(c, (FourierPlanarCoil, PiecewisePlanarArcCoil, PolarPlanarArcCoil,
                                  SplineXYZCoil))
                   for c in S.iter_unique(cs0)):
            raise SystemExit("--convex-form signed needs planar, planar-arc or spline coils; "
                             "use --convex-form total")
        cons.append(CoilSignedCurvature(cs0, bounds=(a.convex_kmin, np.inf), normalize_target=True, grid=kg,
                                        **CJ))
    if a.corner_max is not None:
        # theta <= corner_max at EVERY hinge <=> cos(theta) >= cos(corner_max). Per hinge,
        # not a max over hinges (see sandbox/corners.py). Arc coils only.
        from desc.coils import PiecewisePlanarArcCoil, PolarPlanarArcCoil

        if not all(isinstance(c, (PiecewisePlanarArcCoil, PolarPlanarArcCoil))
                   for c in S.iter_unique(cs0)):
            raise SystemExit("--corner-max needs arc coils")
        cons.append(CoilCornerAngle(cs0, bounds=(float(np.cos(np.radians(a.corner_max))), np.inf),
                                    **CJ))
    shadow, dirs = None, None
    if a.demount_margin is not None or a.demount_arcs:
        if not arc:
            raise SystemExit("--demount-margin / --demount-arcs need arc coils")
        shadow = D.PlasmaShadow(eq)
        dirs = D.arc_directions(cs0)
    if a.demount_margin is not None:
        cons.append(D.HingeShadowClearance(cs0, eq, shadow=shadow,
                                           bounds=(a.demount_margin + a.demount_backoff, np.inf),
                                           **CJ))
    if a.demount_arcs:
        cons.append(D.CoilDemountPenalty(cs0, eq, shadow=shadow, delta=a.demount_delta,
                                         eps=a.demount_eps, directions=dirs, target=0.0,
                                         grid=cg, **CJ))
    cons += [
        O.CoilSetDistancePenalty(cs0, d_min=ENF["dcc"], grid=cg, dist_chunk_size=1, distance_method="segment",
                                 num_neighbors=a.dcc_neighbors, max_active_pairs=(a.dcc_k or None),
                                 signed=(not a.no_signed) and bool(a.dcc_k), deriv_mode="rev",
                                 jac_chunk_size=a.dcc_chunk),
        O.PlasmaCoilSetDistancePenalty(eq, cs0, d_min=ENF["dpc"],
                                       plasma_grid=LinearGrid(M=a.plasma_m, N=a.plasma_n, NFP=eq.NFP, sym=eq.sym),
                                       coil_grid=cg, eq_fixed=True, dist_chunk_size=2, max_active_pairs=a.dpc_k,
                                       distance_method="segment", deriv_mode="rev", jac_chunk_size=a.dpc_chunk),
    ]
    want_planar = a.planarity == "on" or (a.planarity == "auto" and spline and pw)
    if want_planar:
        from planarity import CoilSetPlanarity

        if not spline:
            raise SystemExit("--planarity needs spline coils (planarity is structural for the arc classes)")
        kw = {"target": 0.0} if a.planarity_tol <= 0 else {"bounds": (-a.planarity_tol, a.planarity_tol)}
        cons.append(CoilSetPlanarity(cs0, **kw))
        how = "exact (target 0)" if a.planarity_tol <= 0 else f"|d| <= {a.planarity_tol} m"
        print(f"PLANARITY enforced on the knots: {how}", flush=True)
    if a.speed_floor > 0:
        from speed import CoilSpeedFloor

        cons.append(CoilSpeedFloor(cs0, alpha=a.speed_floor, grid=cg, **CJ))
        print(f"SPEED FLOOR |x_s| / mean >= {a.speed_floor} at every node", flush=True)
    if a.fix_hinges:
        if not arc:
            raise SystemExit("--fix-hinges needs arc coils")

        def hinge_mask(tree):
            if isinstance(tree, dict):
                return {k: (k == "hinges") for k in tree}
            return [hinge_mask(t) for t in tree]

        cons.append(O.FixParameters(cs0, params=hinge_mask(cs0.params_dict), name="fixed hinges"))
    if a.level_hinges:
        if not arc:
            raise SystemExit("--level-hinges needs arc coils")
        if a.fix_hinge_z is not None:
            raise SystemExit("--level-hinges and --fix-hinge-z are redundant: --fix-hinge-z "
                             "already levels the hinges AND pins the height")
        cons.append(D.HingeLevel(cs0))
    if a.fix_hinge_z is not None:
        zi = np.arange(int(next(S.iter_unique(cs0)).B)) * 3 + 2  # the z slot of each hinge

        def z_mask(tree):
            if isinstance(tree, dict):
                return {k: (zi if k == "hinges" else False) for k in tree}
            return [z_mask(t) for t in tree]

        # no target: FixParameters defaults to the values already in the coilset, which the
        # snap above set to --fix-hinge-z
        cons.append(O.FixParameters(cs0, params=z_mask(cs0.params_dict), name="hinge joint plane"))
    if a.mode == "stage2":
        vc_src = S.vc_source_grid(eq, a.vc_m, a.vc_n)
        obj_terms = (O.QuadraticFlux(eq, field=cs0, eval_grid=LinearGrid(M=a.eval_m, N=a.eval_n, NFP=eq.NFP,
                                                                           sym=eq.sym),
                                     field_grid=src, source_grid=vc_src, vacuum=a.vacuum, bs_chunk_size=10,
                                     B_plasma_chunk_size=a.vc_chunk, jac_chunk_size=a.qf_chunk),)
        if a.arclength_weight > 0:
            obj_terms += (O.CoilArclengthVariance(cs0, target=0, weight=a.arclength_weight, grid=cg),)
            print(f"ARCLENGTH-VARIANCE penalty in the objective at weight {a.arclength_weight}", flush=True)
        if a.torsion_weight > 0:
            obj_terms += (O.CoilTorsion(cs0, target=0, weight=a.torsion_weight, grid=kg),)
            print(f"TORSION penalty in the objective at weight {a.torsion_weight}", flush=True)
        cons.append(O.FixSumCoilCurrent(cs0))
    else:
        obj_terms = (O.FixParameters(cs0, weight=a.anchor_weight),)
        cons.append(O.FixCoilCurrent(cs0))
    objective = O.ObjectiveFunction(obj_terms)

    def demount_line(c):
        """One line on removability, always judged by exact ray casting against the boundary."""
        if shadow is None:
            return ""
        rep = D.demount_report(c, eq, shadow=shadow, directions=dirs,
                               margin=(a.demount_margin or 0.0))
        cl = min(min(r["hinge_clearance"]) for r in rep)
        bf = max(x["blocked_frac"] for r in rep for x in r["arcs"])
        return (f"  demount: min hinge clearance {cl:+.3f} m, worst arc blocked {100 * bf:.2f}%"
                f"  -> {'REMOVABLE' if all(r['ok'] for r in rep) else 'NOT REMOVABLE'}")

    m0 = S.evaluate(cs0, eq, tree, vacuum=a.vacuum)
    print(f"START  bn {m0['bn']:.4e}  {S.report(m0, P)}  linked {m0['linked']}  coils {m0['n_coils']}  "
          f"-> {S.verdict(m0, P)}" + demount_line(cs0), flush=True)
    if m0["n_coils"] != exp:
        print(f"WARNING: {m0['n_coils']} coils, expected {exp}", flush=True)
    if m0["linked"]:
        print("WARNING: the start is LINKED; the signed coil-coil rows will try to undo it", flush=True)

    hist, state = [], {"n": 0}

    def at(x):
        ck = cs0.copy()
        ck.params_dict = objective.unpack_state(x, False)[0]
        return ck

    def track(x, *_):
        state["n"] += 1
        stop = False
        if state["n"] % a.check_every == 0:
            try:
                ck = at(x)
                m = S.evaluate(ck, eq, tree, vacuum=a.vacuum)
                hist.append(dict(iter=state["n"], bn=m["bn"], **S.margins(m, P), linked=m["linked"],
                                 verdict=S.verdict(m, P)))
                print(f"  [it {state['n']:>4}] bn {m['bn']:.4e}  {S.report(m, P)}  -> {S.verdict(m, P)}"
                      + (f"  target {S.verdict(m, T)}" if a.margin else "")
                      + demount_line(ck), flush=True)
                if a.mode == "feasibility" and not S.failed(m, T):
                    print("  feasible against the target set: stopping", flush=True)
                    stop = True
            except Exception as e:
                print(f"  [it {state['n']}] check failed {type(e).__name__}: {e}", flush=True)
        if a.ckpt_every and state["n"] % a.ckpt_every == 0:
            try:
                at(x).save(os.path.join(a.out, f"ckpt_it{state['n']:04d}.h5"))
            except Exception as e:
                print(f"  checkpoint failed {type(e).__name__}: {e}", flush=True)
        return stop

    opts = dict(tr_method="svd", max_trust_radius=a.max_tr, max_inner_iter=a.max_inner_iter, max_inner_stalls=50,
                track_outer=True, y_update_gate="residual", callback=track)
    if not a.no_second_order:
        opts.update(second_order="constraints", second_order_chunk=a.so_chunk, second_order_rtol=a.so_rtol)
    if a.diag_log:
        live = [c for c in cons if c.__class__.__name__ not in ("FixSumCoilCurrent", "FixCoilCurrent", "FixParameters")]
        for c in live:
            if not c.built:
                c.build(verbose=0)
        opts.update(diag_log=os.path.join(a.out, a.diag_log), diag_blocks=[(c.name, int(c.dim_f)) for c in live])

    t0 = time.time()
    (cs1,), res = Optimizer("lsq-auglag").optimize(cs0, objective=objective, constraints=tuple(cons),
                                                   maxiter=a.maxiter, verbose=3, ftol=1e-6, xtol=1e-6, gtol=1e-8,
                                                   ctol=a.ctol, copy=True, options=opts)
    wall = time.time() - t0
    m1 = S.evaluate(cs1, eq, tree, vacuum=a.vacuum)
    active = [k for k, v in S.margins(m1, P).items() if abs(v - 1) <= 0.01]
    info = dict(wall_s=wall, message=str(res["message"]), nit=int(res["nit"]), n_outer=int(res.get("n_outer", -1)),
                constr_violation=float(res["constr_violation"]),
                y_absmax_final=float(np.max(np.abs(np.asarray(res["y"])))))
    print(f"\nSOLVER {info}")
    print(f"START  bn {m0['bn']:.4e}  {S.report(m0, P)}  -> {S.verdict(m0, P)}")
    print(f"RESULT bn {m1['bn']:.4e}  {S.report(m1, P)}  linked {m1['linked']}  -> {S.verdict(m1, P)}  "
          f"active {active}" + (f"  target {S.verdict(m1, T)}" if a.margin else "")
          + demount_line(cs1), flush=True)

    cs1.save(os.path.join(a.out, "result.h5"))
    json.dump(dict(settings=vars(a), eq_kind=kind, device=DEVICE, vacuum=a.vacuum, length_multiplier=a.length_mult, bounds=P,
                   target=T, enforced=ENF, solver=info, start=m0, result=m1, margins=S.margins(m1, P),
                   failed=S.failed(m1, P), active=active, history=hist,
                   demount=(D.demount_report(cs1, eq, shadow=shadow, directions=dirs,
                                             margin=(a.demount_margin or 0.0))
                            if shadow is not None else None)),
              open(os.path.join(a.out, "result.json"), "w"), indent=1, default=float)

    if hist:
        it = [h["iter"] for h in hist]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        axes[0].plot(it, [h["bn"] for h in hist], color="#2a78d6", lw=2)
        axes[0].axhline(m0["bn"], color="#898781", lw=1.5)
        axes[0].set_yscale("log")
        axes[0].set_title("⟨|B·n|⟩/⟨|B|⟩ (grey: start)", loc="left")
        for ax, keys, title in ((axes[1], ("kappa", "kMS", "L", "convex"), "upper bounds (bad above 1)"),
                                (axes[2], ("d_cc", "d_pc"), "clearances (bad below 1)")):
            for k, col in zip(keys, ("#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6")):
                if k in hist[0]:
                    ax.plot(it, [h[k] for h in hist], color=col, lw=2, label=k)
            ax.axhline(1.0, color="#d03b3b", lw=1)
            ax.set_title(title, loc="left")
            ax.legend(frameon=False)
        for ax in axes:
            ax.set_xlabel("iteration")
        fig.suptitle(f"{a.mode}: {os.path.basename(a.start)} on {a.eq}, bn {m0['bn']:.3e} -> {m1['bn']:.3e}, "
                     f"{S.verdict(m1, P)}", x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "result.png"), dpi=110)
        plt.close(fig)
    print(f"wrote {a.out}/result.h5, result.json" + (", result.png" if hist else ""))


if __name__ == "__main__":
    main()
