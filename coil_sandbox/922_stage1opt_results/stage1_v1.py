"""Stage-1 clamshell optimization, v1: proxy contours built inside the objective.

Unlike v0 (`stage1_clamshell.py`) the curves are contours of the current potential, so they
are exact streamlines and equal-current by construction -- there is no curve DOF, no pin and
no streamline constraint, and the optimizer cannot cheat the metric by moving a curve off
its streamline.

Configuration follows REVIEW_contour_curve.md section E.

    python 922_stage1opt_results/stage1_v1.py --arm lo --maxiter 400
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, "sandbox")
import boot  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--arm", choices=["lo", "mid", "hi"], default="lo")
p.add_argument("--quantity", choices=["C", "eps1"], default="C",
               help="what to optimize. eps1 = the planar-compatibility control")
p.add_argument("--target-frac", type=float, default=0.85,
               help="target = frac * baseline, per contour. An ACHIEVABLE target: "
                    "target=0 asks for a 100%% reduction, so the residual and its gradient "
                    "stay O(1) forever and the objective never stops pulling.")
p.add_argument("--no-qs", action="store_true", help="drop the quasisymmetry constraint")
p.add_argument("--eq-res", type=int, default=None,
               help="raise the equilibrium to L=M=N=this and re-solve BEFORE optimizing. "
                    "Measured: the eps_1 gain erodes -41.5%% (L=8) -> -37.0%% (L=10) -> "
                    "-33.4%% (L=12), so an L=8 optimizer spends part of its effort in space "
                    "the resolution cannot represent. Re-measuring afterwards catches that "
                    "but does not avoid it.")
p.add_argument("--from", dest="from_eq", default=None,
               help="start from a saved equilibrium. Also RE-FREEZES the DP segmentation and "
                    "plane normals on the new geometry, which a long run needs: the "
                    "frozen-vs-fresh eps_1 gap grew 0.34%% -> 1.44%% over 300 iterations.")
# Constraints that mean "stay acceptable" are INEQUALITIES with bounds, weight 1 -- not
# equalities with a target. `ForceBalance` defaults to target=0, i.e. EXACT force balance,
# which precise_QA does not achieve (2.07e-5 max normalized) and no equilibrium ever will.
# That made the constraint unsatisfiable, so its violation could never decrease and the AL
# had no choice but to ramp the penalty: measured 10 -> 2.41e6 in ten iterations, after
# which the trust radius froze at 2.2e-4 and the objective could not move (runs/ctrl_eps1).
# With bounds, anything inside the tolerance is genuinely feasible and contributes zero.
p.add_argument("--bound-force", type=float, default=1e-4,
               help="acceptable |force balance| in NORMALIZED units (start: 2.07e-5 max)")
p.add_argument("--bound-aspect", type=float, default=1e-2)
p.add_argument("--bound-iota", type=float, default=1e-3)
p.add_argument("--bound-eps1", type=float, default=1e-2, help="fraction of baseline eps_1")
p.add_argument("--K", type=int, default=16, help="contours per field period (REVIEW C1)")
p.add_argument("--M", type=int, default=32, help="potential grid theta (REVIEW C2)")
p.add_argument("--N", type=int, default=64, help="potential grid zeta")
p.add_argument("--nodes", type=int, default=480, help="nodes per contour (REVIEW C3)")
p.add_argument("--mt", type=int, default=16, help="Phi~ mode truncation")
p.add_argument("--maxiter", type=int, default=400)
p.add_argument("--mmax", type=int, default=3, help="free boundary modes |m|,|n| <= mmax")
p.add_argument("--chunk", type=int, default=None)
p.add_argument("--out", default="922_stage1opt_results/runs/v1")
p.add_argument("--device", default="gpu")
p.add_argument("--method", default="proximal-lsq-auglag",
               help="proximal-lsq-auglag eliminates the 1145 INTERIOR DOF by re-solving the "
                    "equilibrium each step. Without it the optimizer reduces the objective by "
                    "moving the interior and breaking force balance rather than reshaping the "
                    "boundary -- measured: eps_1 -14.9%% with |dR| only 0.004%% of a, force "
                    "balance 20x outside its bound and QS 76x worse (runs/ctrl_final).")
# The proximal re-solve's own convergence, NOT an AL constraint. Measured: with DESC's
# defaults, force balance ended at 2.33e-4 normalized against a 3.34e-6 baseline -- the
# re-solve was stopping 70x short, so every result carried a degraded equilibrium.
p.add_argument("--solve-ftol", type=float, default=1e-6)
p.add_argument("--solve-xtol", type=float, default=1e-8)
p.add_argument("--solve-gtol", type=float, default=1e-10)
p.add_argument("--solve-maxiter", type=int, default=50)
# eps_1 uses a LIVE best-fit plane by default. The frozen normal made eps_1 an upper bound
# that prices RIGID TILT as non-planarity (measured: 2 deg -> 1.08x, 4 deg -> 1.28x,
# 8 deg -> 1.89x, against a whole-ladder signal of 0.60x), so a planar basin at a different
# plane orientation was unreachable. B = 1 has no segmentation to freeze and lambda_0 is
# simple by a factor 60-130, so nothing needed freezing there in the first place.
# With a method that has no AL (e.g. proximal-lsq), the "stay acceptable" constraints move
# INTO the ObjectiveFunction as bounded terms: zero residual and zero gradient inside the
# bound, quadratic outside -- the bounds semantics comes from the objective, not from the AL.
# They must be SCALED, though. The clamshell term is weighted 1/|base-target| so its residual
# starts at exactly 1.0 per contour; a bounded iota term at weight 1 would contribute 0.05
# when a full bound-width out, i.e. ~400x weaker, and iota would drift freely -- which is the
# axisymmetry slide the constraint exists to block. --soft-weight W makes a violation of one
# full bound-width contribute a residual of W.
# The proximal re-solve owns force balance, but between projections the boundary can walk
# into shapes this spectral resolution cannot represent: the proximal-lsq-exact run ended at
# |force|max 1.028e-2 normalized, 498x the 2.065e-5 baseline and 100x the AL runs' bound.
# A ForceBalance term in the OBJECTIVE (in addition to the constraint proximal solves) keeps
# it on track and penalises unresolvable boundaries. 0 disables it.
# PROPORTIONAL asks every contour for the SAME FRACTIONAL reduction (target_k = f * base_k),
# which explicitly orders the already-planar curves to flatten too -- and they are the cheapest
# to move, so they dominate the descent. Measured 2026-09-23 on the -32.9% run: the easiest
# quartile fell 56.0% while the HARDEST fell 37.1%, and the foliation's spread WIDENED from
# 1.66x to 3.43x. A planar coilset is limited by its WORST coil, so that is backwards.
#
# WORST puts a single one-sided bound (0, f * max(base)) on every contour: curves already below
# it contribute zero residual AND zero gradient, so all the effort goes into the offenders.
# Deliberately NOT loss_function="max" (which the vendored DESC does support): a hard max kinks
# whenever the argmax switches curves, which is exactly the failure CLAUDE.md records for hard
# distance minimums ("stalled runs with trust radii of 1e-4 to 1e-6"). The one-sided bound is
# the C1 analogue, the same form the coil distance penalties settled on.
p.add_argument("--target-mode", choices=["proportional", "worst"], default="proportional",
               help="proportional: target_k = frac * base_k per contour (original). "
                    "worst: single bound (0, frac * max(base)), penalising only the offenders")
# Floor on min L_grad(B): Kappel PPCF 66 (2024) 025018 gets R^2 = 0.944 for min(L_gradB)/a vs
# min coil-surface distance, so a falling min means coils must come CLOSER. Measured: eps_1
# optimization drives it down ~24%. dim_f is per node, so bounds=(floor, inf) is one-sided per
# node -- only the offending patch is penalised. The min converges smoothly in the grid
# (3.1788 -> 3.1623 over M=N=8..64 on baseline), so M=N=24 costs 0.7% and is absorbed by the
# backoff; judge the verdict on a finer grid.
p.add_argument("--lgradb-floor", type=float, default=0.0,
               help="floor on min L_grad(B) as a FRACTION of the ORIGINAL equilibrium's own min "
                    "(1.0 = never get worse than the start). 0 disables it")
p.add_argument("--lgradb-weight", type=float, default=0.0)
p.add_argument("--lgradb-grid", type=int, default=24, help="M=N of the enforcement grid")
# Default 1.0, NOT a backoff. Measured: at 1.01 the floor sits 1% above the baseline's own
# min, so 14/2401 nodes violate AT THE START and the term costs 4.0 -- as much as the whole
# eps_1 objective -- before the run begins. The grid error it was correcting is 0.7%. Raise it
# only if the judged (fine-grid) min undershoots the enforced one enough to matter.
p.add_argument("--lgradb-backoff", type=float, default=1.0,
               help="enforce the floor this much high, to absorb the enforcement grid's error "
                    "(1.0 = none; the M=N=24 grid overstates the true min by ~0.7%%)")
p.add_argument("--lgradb-chunk", type=int, default=None)
p.add_argument("--fb-weight", type=float, default=0.0,
               help="weight of a soft ForceBalance term in the objective (0 = off)")
# ForceBalance has dim_f = 5346 against the clamshell term's 8, so it dominates the Jacobian
# memory. Chunk it PER SUB-OBJECTIVE, never on the ObjectiveFunction: `jac_chunk_size` there
# requires deriv_mode="batched", and with several objectives DESC picks "blocked" and raises
# (CLAUDE.md; NOTES 9c item 4, hit twice).
p.add_argument("--fb-chunk", type=int, default=None,
               help="jac_chunk_size for the soft ForceBalance term (try 500 if it OOMs)")
# The outer optimizer's own tolerances. NOT passed before, so DESC's defaults applied and
# proximal-lsq-exact stopped at iteration 50 on ftol=1e-2 with optimality still 0.232.
# Every FRESH lsq run stalls in the first ~10 iterations with steps at 1e-7 while a RESTART
# of the same problem moves freely: measured initial trust radius 2.174e+00 fresh vs 5.418e+02
# on restart (250x), and the restart beat 300 fresh iterations in 12. DESC's default
# initial_trust_radius="scipy" scales from ||x0||, which is evidently far too small here.
# --tr-ratio multiplies it, which fixes the cause instead of chaining restarts around it.
p.add_argument("--tr-ratio", type=float, default=None,
               help="initial_trust_ratio: scales the initial trust radius (try 100-500)")
p.add_argument("--ftol", type=float, default=None)
p.add_argument("--xtol", type=float, default=None)
p.add_argument("--gtol", type=float, default=None)
p.add_argument("--soft-weight", type=float, default=10.0,
               help="residual per bound-width of violation for soft (non-AL) constraints")
p.add_argument("--frozen-eps1", action="store_true",
               help="restore the pre-2026-09-23 frozen-normal eps_1 (upper bound, not "
                    "rotation-invariant). For reproducing the original ladder only.")
p.add_argument("--second-order", action="store_true",
               help="AL second-order constraint model. OFF by default: it OOM'd at 4.42 GiB "
                    "here, and CLAUDE.md's rationale for it is the degenerate COIL DISTANCE "
                    "penalties, none of which are in this constraint set.")
args = p.parse_args()
boot.setup(default=args.device)

import numpy as np  # noqa: E402

sys.path.insert(0, HERE)
from contour_curve import ContourClamshell  # noqa: E402
from desc.examples import get  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.objectives import (  # noqa: E402
    AspectRatio,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixPressure,
    FixPsi,
    ForceBalance,
    GenericObjective,
    ObjectiveFunction,
    QuasisymmetryTwoTerm,
)
from desc.optimize import Optimizer  # noqa: E402
from desc.utils import errorif  # noqa: E402

os.makedirs(args.out, exist_ok=True)
eq = get("precise_QA")
if args.from_eq:
    from desc.io import load as _load
    eq = _load(args.from_eq)
    print(f"restarting from {args.from_eq} (DP re-frozen on this geometry)", flush=True)
if args.eq_res and args.eq_res != eq.L:
    import time as _t
    from desc.objectives import ForceBalance as _FB, ObjectiveFunction as _OF
    _t0 = _t.time()
    eq.change_resolution(L=args.eq_res, M=args.eq_res, N=args.eq_res,
                         L_grid=2 * args.eq_res, M_grid=2 * args.eq_res,
                         N_grid=2 * args.eq_res)
    _fo = _OF(_FB(eq=eq), deriv_mode="batched", jac_chunk_size=args.chunk or 100)
    eq.solve(objective=_fo, maxiter=100, verbose=0)
    print(f"raised to L=M=N={args.eq_res} and re-solved in {_t.time()-_t0:.0f}s", flush=True)
NFP = eq.NFP
print(f"equilibrium L{eq.L} M{eq.M} N{eq.N} NFP{NFP}  dim_x={eq.dim_x}", flush=True)

CKW = dict(K=args.K, M=args.M, N=args.N, n=args.nodes, Mt=args.mt, Nt=args.mt,
           frozen_normal=args.frozen_eps1)
print(f"eps_1 normal: {'FROZEN at build (legacy)' if args.frozen_eps1 else 'LIVE (exact, rotation-invariant)'}",
      flush=True)

# ---- baselines, read back through the objectives so targets always match dim_f ----
oC = ContourClamshell(eq, quantity="C", **CKW)
oC.build(verbose=1)
C0 = np.asarray(oC.compute(eq.params_dict))
oE = ContourClamshell(eq, quantity="eps1", **CKW)
oE.build(verbose=0)
E0 = np.asarray(oE.compute(eq.params_dict))
d = oC.diagnostics()
print(f"diagnostics: monotonic={d['monotonic']}  dPhi/dzeta {d['dPhi_dzeta_min']:.3e}.."
      f"{d['dPhi_dzeta_max']:.3e}  I/G={d['I']/d['G']:+.2e}  "
      f"newton_res={d['worst_newton_residual']:.2e}  gap_seg={d['worst_frozen_gap_seg']:.4f}",
      flush=True)
print(f"baseline C    mean {C0.mean():.4f}  range {C0.min():.4f}-{C0.max():.4f}", flush=True)
print(f"baseline eps1 mean {E0.mean():.4f}  range {E0.min():.4f}-{E0.max():.4f}", flush=True)
np.save(os.path.join(args.out, f"C0_{args.arm}.npy"), C0)
np.save(os.path.join(args.out, f"BASE0_{args.arm}.npy"), E0 if args.quantity == "eps1" else C0)

BASE = C0 if args.quantity == "C" else E0
if args.from_eq:
    import numpy as _np
    _b = os.path.join(os.path.dirname(args.from_eq), f"BASE0_{args.arm}.npy")
    if os.path.exists(_b):
        BASE0 = _np.load(_b)
        print(f"leg baseline {BASE.mean():.5f}; ORIGINAL baseline {BASE0.mean():.5f} "
              f"(cumulative {(BASE.mean()/BASE0.mean()-1)*100:+.2f}%)", flush=True)
f = args.target_frac
if args.target_mode == "worst":
    errorif(args.arm != "lo", ValueError, "--target-mode worst only makes sense for --arm lo")
    T = f * float(BASE.max())
    OBJ_TARGET, OBJ_BOUNDS = None, (np.zeros_like(BASE), np.full_like(BASE, T))
    # weight so the WORST contour's residual starts at exactly 1.0
    w_obj = 1.0 / max(float(BASE.max()) - T, 1e-12)
    r0 = w_obj * np.maximum(0.0, BASE - T)
    print(f"optimizing {args.quantity} WORST-OFFENDER: single bound (0, {T:.5f}) = "
          f"{f:.2f} x max(baseline {BASE.max():.5f}); {int((BASE > T).sum())}/{BASE.size} "
          f"contours are above it", flush=True)
    print(f"objective residual at start: max {r0.max():.3f} (the worst contour, 1.0 by "
          f"construction), rms {float(np.sqrt(np.mean(r0**2))):.3f}", flush=True)
else:
    OBJ_TARGET = {"lo": f * BASE, "hi": (2.0 - f) * BASE, "mid": BASE.copy()}[args.arm]
    OBJ_BOUNDS = None
    print(f"optimizing {args.quantity}: target = {f if args.arm=='lo' else 2-f:.2f} x baseline "
          f"(mean {BASE.mean():.4f} -> {OBJ_TARGET.mean():.4f})", flush=True)
    # objective weight = 1 / (how far we are asking it to move), so the residual is exactly
    # 1.0 at the start and 0.0 on target -- O(1) by construction, per the user's point
    w_obj = 1.0 / np.maximum(np.abs(BASE - OBJ_TARGET), 1e-12)
    print(f"objective residual at start = "
          f"{float(np.sqrt(np.mean((w_obj*(BASE-OBJ_TARGET))**2))):.3f} (O(1) by construction)",
          flush=True)
objective_terms = [
    ContourClamshell(eq, quantity=args.quantity, target=OBJ_TARGET, bounds=OBJ_BOUNDS,
                     weight=w_obj, name=f"clamshell {args.quantity}",
                     jac_chunk_size=args.chunk, **CKW)
]
AL = "auglag" in args.method
SW = args.soft_weight

# ---- constraints ----
gq = LinearGrid(M=eq.M, N=eq.N, NFP=NFP, rho=np.linspace(0.1, 1.0, 5), sym=True)
_qs = QuasisymmetryTwoTerm(eq=eq, helicity=(1, 0), grid=gq)
_qs.build(verbose=0)
qs_rms = float(np.sqrt(np.mean(np.asarray(_qs.compute_scaled_error(*_qs.xs(eq))) ** 2)))
# NOTE: do NOT shrink this to M=0, N=0 to save memory. GenericObjective already compresses
# to dim_f = 5 (one per surface), so it buys nothing -- and iota needs a real theta/zeta
# grid for the surface average: M=0,N=0 returns -3.84, -1.78, ... instead of ~0.42.
gi = LinearGrid(rho=np.array([0.0, 0.25, 0.5, 0.75, 1.0]), M=eq.M, N=eq.N, NFP=NFP)
_io = GenericObjective("iota", thing=eq, grid=gi)
_io.build(verbose=0)
iota0 = np.asarray(_io.compute(*_io.xs(eq)))
# Tie the iota bound to the ORIGINAL equilibrium across restarts. Recomputing it on the
# restarted geometry RE-CENTRES the bound and hands the optimizer a fresh +-bound_iota of
# room on EVERY leg. Measured 2026-09-23: a 600-iteration continuation took the entire new
# +-0.05 within 12 iterations (iota 0.4122..0.4277 -> 0.4587..0.4823), a cumulative drift of
# +0.06 from baseline -- larger than the bound ever nominally allowed. iota is the
# anti-trivialization constraint (SPEC 3b: a vacuum stellarator with iota != 0 and no net
# current MUST be 3D), so relaxing it is the cheapest possible way to flatten the contours.
# The same flaw affects the leg1 -> leg2 ladder in NOTES 9k, whose recorded 8.01e-2 drift
# against a +-0.05 bound is this effect and was logged as a violation without a cause.
_iref = (os.path.join(os.path.dirname(args.from_eq), f"iota0_{args.arm}.npy")
         if args.from_eq else None)
if _iref and os.path.exists(_iref):
    _i_new = iota0
    iota0 = np.load(_iref)
    print(f"iota reference: ORIGINAL equilibrium ({_iref}); this leg starts at "
          f"{_i_new.min():.5f}..{_i_new.max():.5f}, bounds stay {iota0.min():.5f}..{iota0.max():.5f}"
          f" +- {args.bound_iota}", flush=True)
np.save(os.path.join(args.out, f"iota0_{args.arm}.npy"), iota0)
print(f"baseline QS rms {qs_rms:.4e}   iota {iota0.min():.5f}..{iota0.max():.5f}", flush=True)

mR = eq.surface.R_basis.modes
mZ = eq.surface.Z_basis.modes
freeR = (np.abs(mR[:, 1]) <= args.mmax) & (np.abs(mR[:, 2]) <= args.mmax) & \
        ~((mR[:, 1] == 0) & (mR[:, 2] == 0))
freeZ = (np.abs(mZ[:, 1]) <= args.mmax) & (np.abs(mZ[:, 2]) <= args.mmax)
print(f"free boundary DOF: R {freeR.sum()} + Z {freeZ.sum()} = {freeR.sum()+freeZ.sum()}",
      flush=True)

constraints = [
    # Under proximal, ForceBalance must be an EQUALITY (bounds=None): the wrapper SOLVES
    # it at every step rather than penalising it, and `_parse_nonlinear_constraints` only
    # routes bounds-free equilibrium constraints to the proximal path. The infeasibility
    # that broke the single-stage runs does not arise, because the equilibrium solver --
    # not the AL -- drives force balance to its own convergence floor each step.
    (ForceBalance(eq=eq) if args.method.startswith("prox")
     else ForceBalance(eq=eq, bounds=(-args.bound_force, args.bound_force), weight=1.0)),
    FixBoundaryR(eq=eq, modes=mR[~freeR]),
    FixBoundaryZ(eq=eq, modes=mZ[~freeZ]),
    FixPressure(eq=eq), FixCurrent(eq=eq), FixPsi(eq=eq),
]
# "stay acceptable" terms. Under AL they are genuine constraints at weight 1 (the multipliers
# do the scaling). Without AL they become objective terms and must carry the scale themselves.
soft = [
    AspectRatio(eq=eq, bounds=(6.0 - args.bound_aspect, 6.0 + args.bound_aspect),
                weight=1.0 if AL else SW / args.bound_aspect),
    GenericObjective("iota", thing=eq, grid=gi,
                     weight=1.0 if AL else SW / args.bound_iota,
                     bounds=(iota0 - args.bound_iota, iota0 + args.bound_iota)),
]
if not args.no_qs:
    soft.append(
        QuasisymmetryTwoTerm(eq=eq, helicity=(1, 0), grid=gq, bounds=(-qs_rms, qs_rms),
                             weight=1.0 if AL else SW / qs_rms))
if args.quantity == "C":
    # eps_1 held at baseline (SPEC [ADV-3]) so the arms differ only in clamshell-ness.
    # Meaningless when eps_1 IS the objective.
    soft.append(ContourClamshell(
        eq, quantity="eps1", name="eps1 hold",
        weight=1.0 if AL else SW / (args.bound_eps1 * E0),
        bounds=((1 - args.bound_eps1) * E0, (1 + args.bound_eps1) * E0), **CKW))

if AL:
    constraints += soft
    print(f"constraint handling: AUGMENTED LAGRANGIAN ({len(soft)} nonlinear constraints)",
          flush=True)
else:
    objective_terms += soft
    print(f"constraint handling: SOFT objective terms ({len(soft)} bounded terms, "
          f"residual {SW:g} per bound-width of violation); constraints = force balance "
          f"+ linear fixes only", flush=True)
if args.lgradb_floor > 0:
    gL = LinearGrid(rho=np.array([1.0]), M=args.lgradb_grid, N=args.lgradb_grid, NFP=NFP,
                    sym=False)
    _lg = GenericObjective("L_grad(B)", thing=eq, grid=gL)
    _lg.build(verbose=0)
    L_now = np.asarray(_lg.compute(*_lg.xs(eq)))
    # Tie the floor to the ORIGINAL equilibrium across restarts, for the same reason iota is
    # tied: recomputing it on a restarted (already degraded) geometry would ratchet the floor
    # DOWN every leg and the constraint would never bind.
    _lref = (os.path.join(os.path.dirname(args.from_eq), f"lgradb0_{args.arm}.npy")
             if args.from_eq else None)
    L_ref = np.load(_lref) if (_lref and os.path.exists(_lref)) else L_now
    np.save(os.path.join(args.out, f"lgradb0_{args.arm}.npy"), L_ref)
    floor = args.lgradb_floor * float(L_ref.min()) * args.lgradb_backoff
    a_now = float(eq.compute("a")["a"])
    objective_terms.append(GenericObjective(
        "L_grad(B)", thing=eq, grid=gL, bounds=(floor, np.inf), weight=args.lgradb_weight,
        name="L_gradB floor", jac_chunk_size=args.lgradb_chunk))
    print(f"L_gradB floor {floor:.5f} m = {floor/a_now:.4f} a  "
          f"({args.lgradb_floor:g} x the ORIGINAL min {L_ref.min():.5f} x backoff "
          f"{args.lgradb_backoff:g}); this equilibrium starts at min {L_now.min():.5f} m = "
          f"{L_now.min()/a_now:.4f} a, {int((L_now < floor).sum())}/{L_now.size} nodes below "
          f"it; weight {args.lgradb_weight:g}", flush=True)

if args.fb_weight > 0:
    objective_terms.append(ForceBalance(eq=eq, weight=args.fb_weight, name="force (soft)",
                                       jac_chunk_size=args.fb_chunk))
    print(f"soft ForceBalance in the objective at weight {args.fb_weight:g}"
          f"{'' if args.fb_chunk is None else f', jac_chunk_size={args.fb_chunk}'}", flush=True)
objective = ObjectiveFunction(objective_terms)

opt = Optimizer(args.method)
t0 = time.time()
opts = {"second_order": "constraints"} if args.second_order else {}
if args.method.startswith("prox"):
    opts["solve_options"] = dict(ftol=args.solve_ftol, xtol=args.solve_xtol,
                                 gtol=args.solve_gtol, maxiter=args.solve_maxiter,
                                 verbose=0)
    print(f"proximal re-solve: ftol={args.solve_ftol:g} xtol={args.solve_xtol:g} "
          f"gtol={args.solve_gtol:g} maxiter={args.solve_maxiter}", flush=True)
for _k in ("ftol", "xtol", "gtol"):
    if getattr(args, _k) is not None:
        opts[_k] = getattr(args, _k)
        print(f"outer {_k} = {getattr(args, _k):g}", flush=True)
if args.tr_ratio is not None:
    opts["initial_trust_ratio"] = args.tr_ratio
    print(f"initial_trust_ratio = {args.tr_ratio:g}", flush=True)
(eq_new,), result = opt.optimize((eq,), objective, constraints, maxiter=args.maxiter,
                                 verbose=3, options=opts, copy=True)
dt = time.time() - t0
print(f"\nwall clock {dt:.1f} s for {args.maxiter} iterations -> {dt/max(args.maxiter,1):.2f} s/it",
      flush=True)
eq_new.save(os.path.join(args.out, f"eq_{args.arm}.h5"))

for q, base in (("C", C0), ("eps1", E0)):
    oo = ContourClamshell(eq_new, quantity=q, **CKW)
    oo.build(verbose=0)
    v = np.asarray(oo.compute(eq_new.params_dict))
    print(f"{q:5}: mean {base.mean():.4f} -> {v.mean():.4f}  "
          f"({(v.mean()-base.mean())/base.mean()*100:+.2f}%)", flush=True)
    if q == args.quantity:
        C1 = v
np.save(os.path.join(args.out, f"C1_{args.arm}.npy"), C1)
print("saved", os.path.join(args.out, f"eq_{args.arm}.h5"), flush=True)
