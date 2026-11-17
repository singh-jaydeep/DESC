"""Stage-2 coil optimization for the piecewise-planar coil paper.

    python paper/stage2.py --eq EQ --start COILS.h5 --bounds BOUNDS.json --out DIR \
        [--vacuum] [--fold-max 60] [--fix-hinge-z 0] [--current-band 0.5] ...

The equilibrium is FIXED; the coils (FourierPlanar, FourierXYZ, or arc coils) are
optimized. Bounds come from BOUNDS.json (paper/lib/dense_check.load_bounds: absolute, or
Gil's reactor set scaled by a / 1.7). Solver ``lsq-auglag-composite``: slack-free
augmented Lagrangian, every constraint below exact at convergence (AL settings in
desc/optimize/aug_lagrangian_composite.py; --gtol, --ctol, --maxiter = total inner
iterations).

Objective: QuadraticFlux (area-weighted B.n on the boundary, normalized by B R0 a),
weight 1, on LinearGrid(--eval-mn) on rho = 1; Biot-Savart source composite
Gauss-Legendre 24 nodes per arc for arc coils, LinearGrid(--field-n) otherwise.
--vacuum: coil field only (else the plasma field by virtual casing is added).

Constraints (hard):
* sum of the coil currents fixed; gauges: FourierPlanar largest normal component pinned;
  FourierXYZ phase pinned and relative speed within 1 -+ --speed-tol on
  LinearGrid(N=100);
* coil length <= L (composite GL 24 per arc; LinearGrid(N=400) for smooth coils);
* |curvature| <= kappa on --curv-n nodes per arc (--curv-grid arc) or LinearGrid(N=
  --curv-n) (--curv-grid linear);
* coil-coil distance >= d_cc: ``CoilSetDistanceRows``, unsigned top-k node-pair rows
  within --cc-sel x d_cc in --cc-k slots, each row a node distance minus a gap so it is
  a lower bound; --cc-gap coils: node grid chosen at build so the gap is <= 2% of d_cc;
  --cc-gap bounds: 2 --row-n + 1 nodes per coil, gap from ``node_gap_estimate``;
* plasma-coil distance >= d_pc, boundary fixed: --pc-mode min (default) upstream
  ``PlasmaCoilSetMinDistance`` (hard minimum, one row per coil) on plasma
  LinearGrid(--pc-plasma-mn, default 32 x 32, full surface) and 2 --pc-row-n + 1 coil
  nodes; --pc-mode rows ``PlasmaCoilSetDistanceRows`` (smooth top-k rows within --pc-sel
  x d_pc, --pc-k slots, default 128 x 128 half surface), for problems where the minimum
  stalls;
* --current-band f: each current within (1 -+ f) x the average at build;
* arc coils: --fold-max (``ArcFold``, cos(fold) >= cos(fold_max), arc orientation from
  the start or --fold-sides-from); --turn-max (``HingeTurningAngle``, cos(turn) >=
  cos(turn_max) at every hinge); --fix-hinge-z (``FixHingeHeight``: every hinge at that
  z; the start is snapped onto it) or --horizontal-hinges (``HingeLevel``: each coil's
  hinges at one free height); --hinge-zmax-rel (``HingeMidHeightRelative``: each coil's
  mean hinge height within +- that many m of the axis height at its phi);
  --hinge-tilt-max (``HingeTilt``: |sin(chord elevation)| <= sin(tilt_max));
  --polar-rmin (``PolarArcRadius``, polar arcs);
* --bn-max c: ``PointwiseBn``, |B.n| / |B| <= c at every node of LinearGrid(
  --bn-max-grid) on the boundary (vacuum only).
Experimental (paper/lib/experimental.py, off): --exp-pair-rows, --link-mu (link guard).

Every run writes to DIR: cmd.txt, commit.txt, run.log, outer.json (one record per outer
iteration: per-group penalties, multipliers, violations, and a dense check), x_last.npy,
result.h5 and result.json (settings, bounds, dense check at start and end).

Device: ``--device gpu|cpu`` (default gpu). On an 8 GB GPU with a pointwise B.n cap set
``XLA_PYTHON_CLIENT_MEM_FRACTION=0.92`` (the default 0.75 runs out of memory).
"""

import argparse
import json
import os
import subprocess
import sys
import time

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--eq", required=True, help="DESC .h5 or desc.examples name")
ap.add_argument("--start", required=True, help="start coilset (.h5)")
ap.add_argument("--bounds", required=True, help="bounds file (json)")
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
ap.add_argument("--vacuum", action="store_true", help="no plasma field (vacuum eq)")
ap.add_argument("--a", type=float, default=None, help="minor radius for the bounds")
ap.add_argument("--maxiter", type=int, default=800)
ap.add_argument("--gtol", type=float, default=1e-6)
ap.add_argument("--ctol", type=float, default=1e-6)
ap.add_argument("--options", default="{}", help="JSON options for the solver")
# grids and row selection
ap.add_argument("--field-n", type=int, default=50, help="smooth coils: Biot-Savart N")
ap.add_argument("--eval-mn", default="25,25", help="QuadraticFlux eval grid M,N")
ap.add_argument(
    "--curv-grid",
    choices=["linear", "arc"],
    default="arc",
    help="curvature nodes: LinearGrid(N=curv-n) (as the poster runs) or "
    "coil_curvature_grid with curv-n nodes per arc",
)
ap.add_argument(
    "--curv-n", type=int, default=1000, help="curvature nodes per arc (or N)"
)
ap.add_argument("--row-n", type=int, default=256, help="cc rows: ~2 N + 1 nodes/coil")
ap.add_argument("--cc-k", type=int, default=10000, help="cc row slots")
ap.add_argument("--cc-sel", type=float, default=2.5, help="cc select distance / d_cc")
ap.add_argument(
    "--cc-gap",
    choices=["bounds", "coils"],
    default="bounds",
    help="cc gap from the bounds (node_gap_estimate) or the coils at build",
)
ap.add_argument("--speed-ratio", type=float, default=2.0, help="for the bounds gap")
ap.add_argument(
    "--speed-tol",
    type=float,
    default=0.5,
    help="FourierXYZ: node speed within (1 -+ this) x the mean (gap ratio 1 + this)",
)
ap.add_argument(
    "--pc-mode",
    choices=["min", "rows"],
    default="min",
    help="plasma-coil distance: min = upstream PlasmaCoilSetMinDistance (hard minimum, "
    "one row per coil); rows = PlasmaCoilSetDistanceRows (smooth top-k rows), for "
    "problems where the hard minimum stalls",
)
ap.add_argument(
    "--pc-row-n", type=int, default=256, help="pc coil nodes: ~2 N + 1 per coil"
)
ap.add_argument(
    "--pc-plasma-mn",
    default=None,
    help="pc plasma grid M,N (default 32,32 for min, 128,128 for rows; the rows' gap "
    "grows as (cell size)^2 x surface curvature: design F 47 mm at 32,32, 5 mm at "
    "128,128)",
)
ap.add_argument(
    "--pc-plasma-sym",
    choices=["half", "full"],
    default=None,
    help="plasma grid over the stellarator-symmetric half (symmetric coilsets only) "
    "or the full surface (default: full for min, half for rows)",
)
ap.add_argument("--pc-k", type=int, default=20000, help="pc row slots")
ap.add_argument(
    "--pc-sel", type=float, default=1.2, help="pc select distance / d_pc bound"
)
ap.add_argument("--jac-chunk", type=int, default=16, help="jac_chunk_size")
ap.add_argument(
    "--exp-pair-rows",
    nargs="?",
    const="unsigned",
    choices=["unsigned", "signed"],
    default=None,
    help="EXPERIMENTAL: add the branch's per-pair closest-node rows "
    "(paper/lib/experimental.py); 'signed' negates linked pairs (the link guard)",
)
ap.add_argument(
    "--link-mu",
    type=float,
    default=None,
    help="EXPERIMENTAL link guard: initial AL penalty on the signed per-pair rows, "
    "e.g. 1e8 (needs --exp-pair-rows signed)",
)
# piecewise-planar constraints
ap.add_argument("--fold-max", type=float, default=None, help="fold cap (deg)")
ap.add_argument(
    "--turn-max",
    type=float,
    default=None,
    help="cap on the turning angle at every hinge (deg); off by default",
)
ap.add_argument(
    "--fold-sides-from",
    default=None,
    help="result.json of an earlier run: reuse its arc orientation for the fold",
)
ap.add_argument("--fix-hinge-z", type=float, default=None, help="joint plane z (m)")
ap.add_argument(
    "--horizontal-hinges", action="store_true", help="each coil's hinges level"
)
ap.add_argument(
    "--hinge-zmax-rel", type=float, default=None, help="|hinge mid z - axis z| (m)"
)
ap.add_argument("--hinge-tilt-max", type=float, default=None, help="chord tilt (deg)")
ap.add_argument("--polar-rmin", type=float, default=None, help="polar radius floor")
ap.add_argument("--current-band", type=float, default=None, help="I / I_avg in 1 -+ f")
ap.add_argument("--bn-max", type=float, default=None, help="cap on |B.n|/|B|")
ap.add_argument("--bn-max-grid", default="32,64", help="B.n cap grid M,N")
a = ap.parse_args()

if a.horizontal_hinges and a.fix_hinge_z is not None:
    raise SystemExit("--horizontal-hinges and --fix-hinge-z are exclusive")
if a.fix_hinge_z is not None and (a.hinge_zmax_rel or a.hinge_tilt_max):
    raise SystemExit("--fix-hinge-z fixes the hinges; drop --hinge-zmax-rel/-tilt-max")
if a.bn_max is not None and not a.vacuum:
    raise SystemExit("--bn-max is vacuum only")

os.makedirs(a.out, exist_ok=True)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


sys.stdout = _Tee(sys.__stdout__, open(os.path.join(a.out, "run.log"), "a"))
open(os.path.join(a.out, "cmd.txt"), "w").write(" ".join(sys.argv) + "\n")

from desc import set_device  # noqa: E402

set_device(a.device)

import numpy as np  # noqa: E402

import desc  # noqa: E402
from desc.coils import (  # noqa: E402
    FourierPlanarCoil,
    FourierXYZCoil,
    PolarPlanarArcCoil,
    coil_curvature_grid,
    coil_quad_grid,
    coil_source_grid,
)
from desc.grid import LinearGrid  # noqa: E402
from desc.io import load  # noqa: E402
from desc.objectives import (  # noqa: E402
    ArcFold,
    CoilArclengthResidual,
    CoilCurrentBand,
    CoilCurvature,
    CoilLength,
    CoilSetDistanceRows,
    FixHingeHeight,
    FixParameters,
    FixSumCoilCurrent,
    HingeLevel,
    HingeMidHeightRelative,
    HingeTilt,
    HingeTurningAngle,
    ObjectiveFunction,
    PlasmaCoilSetDistanceRows,
    PlasmaCoilSetMinDistance,
    PointwiseBn,
    PolarArcRadius,
    QuadraticFlux,
    node_gap_estimate,
)
from desc.objectives._coils import _coil_node_grid  # noqa: E402
from desc.objectives._piecewise_planar import (  # noqa: E402
    fold_angles,
    side_signs,
    turning_angles,
)
from desc.optimize import Optimizer  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
import dense_check as D  # noqa: E402

root = os.path.dirname(os.path.dirname(os.path.abspath(desc.__file__)))
commit = subprocess.run(
    ["git", "-C", root, "describe", "--always", "--dirty"],
    capture_output=True,
    text=True,
).stdout.strip()
open(os.path.join(a.out, "commit.txt"), "w").write(commit + "\n")
print(f"DESC {desc.__file__} at {commit}; device {a.device}", flush=True)

# ------------------------------------------------------------------ inputs
eq = D.load_equilibrium(a.eq)
cs0 = load(a.start)
coils = D.leaf_coils(cs0)
P = D.load_bounds(a.bounds, eq, len(coils), a.a)
arc = hasattr(coils[0], "B")
polar = isinstance(coils[0], PolarPlanarArcCoil)
kind = type(coils[0]).__name__
xyz = isinstance(coils[0], FourierXYZCoil)
if not arc and not xyz and not isinstance(coils[0], FourierPlanarCoil):
    raise SystemExit(f"arc, FourierPlanar or FourierXYZ coils only, got {kind}")
if (
    a.fold_max or a.turn_max or a.fix_hinge_z is not None or a.horizontal_hinges
) and not arc:
    raise SystemExit("fold and hinge options need arc coils")
a.pc_plasma_mn = a.pc_plasma_mn or ("32,32" if a.pc_mode == "min" else "128,128")
a.pc_plasma_sym = a.pc_plasma_sym or ("full" if a.pc_mode == "min" else "half")
if a.pc_plasma_sym == "half" and not getattr(cs0, "sym", False):
    raise SystemExit(
        "the half plasma grid needs a symmetric coilset: --pc-plasma-sym full"
    )
if a.polar_rmin is not None and not polar:
    raise SystemExit("--polar-rmin needs polar arc coils")
print(
    f"eq {a.eq} NFP {eq.NFP}; start {a.start}: {len(coils)} {kind}; "
    f"{'vacuum' if a.vacuum else 'finite beta'}\nBOUNDS {P}",
    flush=True,
)

if a.fix_hinge_z is not None or a.horizontal_hinges:
    moved = 0.0
    for c in coils:  # snap the start onto the joint plane(s)
        h = np.array(c.hinges).reshape(-1, 3)
        zt = a.fix_hinge_z if a.fix_hinge_z is not None else h[:, 2].mean()
        moved = max(moved, float(np.abs(h[:, 2] - zt).max()))
        h[:, 2] = zt
        c.hinges = h.reshape(-1)
    print(f"JOINT PLANE: hinges snapped (largest move {moved:.2e} m)", flush=True)

sides = None
if arc:
    B0 = int(coils[0].B)
    if a.fold_sides_from:
        info0 = json.load(open(a.fold_sides_from))["info"]
        sides = np.asarray(info0["fold_sides"], dtype=float)
    else:
        sides = np.array([side_signs(c.params_dict, B0, polar) for c in coils])

# --------------------------------------------------------------- problem
jcs = a.jac_chunk
eM, eN = (int(v) for v in a.eval_mn.split(","))
field_grid = coil_source_grid(cs0, N=a.field_n)
obj = ObjectiveFunction(
    QuadraticFlux(
        eq,
        field=cs0,
        eval_grid=LinearGrid(M=eM, N=eN, NFP=eq.NFP, sym=eq.sym),
        field_grid=field_grid,
        vacuum=a.vacuum,
        bs_chunk_size=10,
        jac_chunk_size=jcs,
    ),
    deriv_mode="blocked",
)
cons = [FixSumCoilCurrent(cs0)]
info = dict(kind=kind, field_grid=field_grid.num_nodes)
if isinstance(coils[0], FourierPlanarCoil):  # scaling the normal is an exact gauge
    pins = [{"normal": np.array([np.argmax(np.abs(c.normal))])} for c in coils]
    cons.append(FixParameters(cs0, pins))
    info["gauge"] = "largest normal component pinned"
if xyz:  # s -> s + c is a gauge: pin the n = +-1 coefficient most aligned with it

    def phase_tangent(p, modes):
        """d/dc of the X, Y, Z coefficients under s -> s + c."""
        t = {}
        for key in ("X_n", "Y_n", "Z_n"):
            v = np.asarray(p[key])
            out = np.zeros_like(v)
            for i, n in enumerate(modes):
                if n:
                    j = int(np.where(modes == -n)[0][0])
                    out[j] += (-abs(n) if n > 0 else abs(n)) * v[i]
            t[key] = out
        return t

    modes = coils[0].X_basis.modes[:, 2]
    pins = []
    for c in coils:
        t = phase_tangent(c.params_dict, modes)
        k, i = max(
            ((k, i) for k in t for i in np.where(np.abs(modes) == 1)[0]),
            key=lambda ki: abs(t[ki[0]][ki[1]]),
        )
        pins.append({k: np.array([i])})
    cons.append(FixParameters(cs0, pins))
    st_ = a.speed_tol
    cons.append(
        CoilArclengthResidual(
            cs0, relative=True, bounds=(-st_, st_), grid=LinearGrid(N=100)
        )
    )
    a.speed_ratio = 1 + st_
    info["gauge"] = f"phase pinned, relative speed within +-{st_}"
if P["L"] is not None:
    cons.append(
        CoilLength(
            cs0, bounds=(0, P["L"]), grid=coil_quad_grid(cs0), jac_chunk_size=jcs
        )
    )
if P["kappa"] is not None:
    cg = (
        coil_curvature_grid(cs0, n=a.curv_n)
        if a.curv_grid == "arc"
        else LinearGrid(N=a.curv_n)
    )
    cons.append(CoilCurvature(cs0, bounds=(0, P["kappa"]), grid=cg, jac_chunk_size=jcs))
    info["curvature_nodes"] = cg.num_nodes

# distances
arc_nodes = 2 * a.row_n + 1
row_grid = _coil_node_grid(cs0, arc_nodes) if arc else LinearGrid(N=a.row_n)
n_nodes = row_grid.num_nodes
gap = node_gap_estimate(n_nodes, P["L"], P["kappa"], P["dcc"], a.speed_ratio)
cc = CoilSetDistanceRows(
    cs0,
    select_distance=a.cc_sel * P["dcc"],
    bounds=(P["dcc"], np.inf),
    grid=row_grid if a.cc_gap == "bounds" else None,
    gap=gap if a.cc_gap == "bounds" else None,
    max_active_rows=a.cc_k,
    jac_chunk_size=jcs,
)
pM, pN = (int(v) for v in a.pc_plasma_mn.split(","))
pc_grid = _coil_node_grid(cs0, 2 * a.pc_row_n + 1) if arc else LinearGrid(N=a.pc_row_n)
pc_plasma = LinearGrid(M=pM, N=pN, NFP=eq.NFP, sym=a.pc_plasma_sym == "half")
if a.pc_mode == "min":
    pc = PlasmaCoilSetMinDistance(
        eq,
        cs0,
        bounds=(P["dpc"], np.inf),
        plasma_grid=pc_plasma,
        coil_grid=pc_grid,
        eq_fixed=True,
        jac_chunk_size=jcs,
    )
else:
    pc = PlasmaCoilSetDistanceRows(
        eq,
        cs0,
        select_distance=a.pc_sel * P["dpc"],
        bounds=(P["dpc"], np.inf),
        plasma_grid=pc_plasma,
        coil_grid=pc_grid,
        eq_fixed=True,
        max_active_rows=a.pc_k,
        jac_chunk_size=jcs,
    )
cons += [cc, pc]
if a.exp_pair_rows:
    from experimental import PairClosestRows

    cc.build(verbose=0)
    cons.append(PairClosestRows(cc, signed=a.exp_pair_rows == "signed"))
    info["experimental"] = f"per-pair closest-node rows ({a.exp_pair_rows})"

# piecewise-planar constraints
if a.fix_hinge_z is not None:
    cons.append(FixHingeHeight(cs0, height=a.fix_hinge_z))
    info["joint_plane"] = f"every hinge at z = {a.fix_hinge_z}"
if a.horizontal_hinges:
    cons.append(HingeLevel(cs0))
    info["joint_plane"] = "hinges level per coil, heights free"
if a.hinge_zmax_rel is not None:
    cons.append(HingeMidHeightRelative(cs0, eq, a.hinge_zmax_rel))
if a.hinge_tilt_max is not None:
    st = float(np.sin(np.radians(a.hinge_tilt_max)))
    cons.append(HingeTilt(cs0, bounds=(-st, st)))
if a.polar_rmin is not None:
    cons.append(PolarArcRadius(cs0, bounds=(a.polar_rmin, np.inf)))
if a.fold_max is not None:
    cmin = float(np.cos(np.radians(a.fold_max)))
    cons.append(ArcFold(cs0, bounds=(cmin, np.inf), sides=sides))
if a.turn_max is not None:
    cmin = float(np.cos(np.radians(a.turn_max)))
    cons.append(HingeTurningAngle(cs0, bounds=(cmin, np.inf)))
if arc:
    info["fold_sides"] = sides.tolist()
if a.current_band is not None:
    f = a.current_band
    cons.append(CoilCurrentBand(cs0, bounds=(1 - f, 1 + f)))
if a.bn_max is not None:
    bM, bN = (int(v) for v in a.bn_max_grid.split(","))
    cons.append(
        PointwiseBn(
            eq,
            cs0,
            LinearGrid(M=bM, N=bN, NFP=eq.NFP, sym=eq.sym),
            field_grid=field_grid,
            bounds=(-a.bn_max, a.bn_max),
            jac_chunk_size=jcs,
        )
    )
cons = tuple(cons)

t0 = time.time()
obj.build(verbose=0)
nonlinear = [c for c in cons if not c.linear]
for c in nonlinear:
    c.build(verbose=0)
params0 = cs0.params_dict
if a.pc_mode == "rows":
    pc_line = (
        f"pc rows: gap {1e3 * float(np.max(pc._constants['gap'])):.3f} mm on "
        f"{pc._n_nodes} nodes/coil x {pc._n_points} plasma points, active "
        f"{pc.active_row_count(params0)} of {a.pc_k}"
    )
else:
    pc_line = (
        f"pc hard min: {pc_grid.num_nodes} nodes/coil x {pc_plasma.num_nodes} "
        f"plasma nodes ({a.pc_plasma_sym}), {pc.dim_f} rows"
    )
print(
    f"BUILD {time.time() - t0:.1f} s; "
    f"cc gap {1e3 * float(np.max(cc._constants['gap'])):.3f}"
    f" mm on {cc._n_nodes} nodes/coil, active {cc.active_row_count(params0)} of "
    f"{a.cc_k}; {pc_line}",
    flush=True,
)
for c in nonlinear:
    print(f"  {c.name:32s} rows {getattr(c, 'num_row_ids', c.dim_f)}", flush=True)
if a.pc_mode == "rows" and float(np.max(pc._constants["gap"])) > 0.05 * P["dpc"]:
    print(
        "WARNING: the plasma-coil gap exceeds 5% of d_pc, so the constraint is that "
        "much conservative; refine --pc-plasma-mn (the gap goes as cell size^2 x "
        "surface curvature)",
        flush=True,
    )
info.update(
    cc_gap=float(np.max(cc._constants["gap"])),
    pc_mode=a.pc_mode,
    pc_plasma=f"{a.pc_plasma_mn} {a.pc_plasma_sym}",
    pc_gap=float(np.max(pc._constants["gap"])) if a.pc_mode == "rows" else 0.0,
    cc_nodes=cc._n_nodes,
    pc_nodes=pc_grid.num_nodes,
)
print(f"INFO {info}", flush=True)

tree = D.surface_tree(eq)


def shape(cs_or_params):
    """Per-coil folds, currents over their mean, as a dict for the logs."""
    cur = np.array([float(np.ravel(c.current)[0]) for c in D.leaf_coils(cs_or_params)])
    out = dict(I_over_avg=np.round(cur / cur.mean(), 4).tolist())
    if arc:
        fold = fold_angles(cs_or_params, sides=sides)
        out["fold_deg"] = np.round(fold, 2).tolist()
        out["turn_deg"] = np.round(turning_angles(cs_or_params), 2).tolist()
    return out


m0 = D.evaluate(cs0, eq, tree, vacuum=a.vacuum)
print(
    f"START  bn {m0['bn']:.4e} max {m0['bn_max']:.3e}  {D.report(m0, P)}  "
    f"linked {m0['linked']} -> {D.verdict(m0, P)}  {shape(cs0)}",
    flush=True,
)
if a.maxiter == 0:
    print("maxiter 0: build check only", flush=True)
    raise SystemExit(0)

# ----------------------------------------------------------------- solve
sizes = [getattr(c, "num_row_ids", c.dim_f) for c in nonlinear]
offsets = np.concatenate([[0], np.cumsum(sizes)])
outer_log = []
check_cs = cs0.copy()
t_run = time.time()


def report(h):
    """Log one outer iteration: groups, a dense check, outer.json."""
    k = h["outer"]
    check_cs.params_dict = obj.unpack_state(np.asarray(h["x"]), False)[0]
    ids, y, mu, viol = (np.asarray(h[q]) for q in ("ids", "y", "mu", "violation"))
    grp = np.where(ids >= 0, np.searchsorted(offsets, ids, side="right") - 1, -1)
    groups = {}
    for j, c in enumerate(nonlinear):
        m = grp == j
        if np.any(m):
            groups[c.name] = dict(
                rows=int(m.sum()),
                mu=[
                    float(np.min(mu[m])),
                    float(np.median(mu[m])),
                    float(np.max(mu[m])),
                ],
                max_y=float(np.abs(y[m]).max()),
                nonzero_y=int(np.sum(y[m] != 0)),
                max_violation=float(viol[m].max()),
            )
    m = D.evaluate(check_cs, eq, tree, vacuum=a.vacuum)
    rec = dict(
        outer=k,
        inner_iterations=h["inner_iterations"],
        optimality=h["optimality"],
        gtolk=h["gtolk"],
        inner_converged=bool(h["inner_converged"]),
        constr_violation=h["constr_violation"],
        ctolk=h["ctolk"],
        message=str(h.get("message", "")),
        groups=groups,
        dense=m,
        shape=shape(check_cs),
        t=time.time() - t_run,
    )
    outer_log.append(rec)
    json.dump(
        outer_log,
        open(os.path.join(a.out, "outer.json"), "w"),
        indent=1,
        default=float,
    )
    met = "met" if h["inner_converged"] else "NOT met"
    print(
        f"[outer {k}] {h['inner_iterations']} inner its, opt {h['optimality']:.2e} "
        f"(gtolk {h['gtolk']:.1e}, {met}), violation {h['constr_violation']:.2e} "
        f"(ctolk {h['ctolk']:.1e}), {time.time() - t_run:.0f} s",
        flush=True,
    )
    for name, g in groups.items():
        print(
            f"   {name[:30]:30s} rows {g['rows']:6d} mu {g['mu'][0]:.1e}/"
            f"{g['mu'][1]:.1e}/{g['mu'][2]:.1e} max|y| {g['max_y']:.2e} "
            f"(nonzero {g['nonzero_y']}) max viol {g['max_violation']:.2e}",
            flush=True,
        )
    print(
        f"   dense: bn {m['bn']:.4e}  {D.report(m, P)}  linked {m['linked']}  "
        f"{rec['shape']}",
        flush=True,
    )


def save_x(x):
    """Keep the latest accepted state, for a restart."""
    np.save(os.path.join(a.out, "x_last.npy"), np.asarray(x))
    return False


options = {"outer_callback": report, "callback": save_x, **json.loads(a.options)}
if a.link_mu is not None:
    if a.exp_pair_rows != "signed":
        raise SystemExit("--link-mu needs --exp-pair-rows signed")
    j = next(
        i for i, c in enumerate(nonlinear) if type(c).__name__ == "PairClosestRows"
    )
    link_ids = offsets[j] + np.arange(nonlinear[j].dim_f)
    options["initial_penalty_rows"] = {"ids": link_ids, "mu": a.link_mu}
    info["link_guard"] = f"initial mu {a.link_mu:g} on {link_ids.size} signed pair rows"
    print(f"LINK GUARD: {info['link_guard']}", flush=True)
(cs1,), res = Optimizer("lsq-auglag-composite").optimize(
    cs0,
    objective=obj,
    constraints=cons,
    maxiter=a.maxiter,
    verbose=1,
    ftol=0,
    xtol=0,
    gtol=a.gtol,
    ctol=a.ctol,
    options=options,
    copy=True,
)
wall = time.time() - t_run
cs1.save(os.path.join(a.out, "result.h5"))
m1 = D.evaluate(cs1, eq, tree, vacuum=a.vacuum)
active = [k for k, v in D.margins(m1, P).items() if abs(v - 1) <= 0.01]
print(
    f"SOLVER {res['message']} | {len(res['outer_history'])} outer / {res['nit']} "
    f"inner its, opt {res['optimality']:.2e}, violation "
    f"{res['constr_violation']:.2e}, {wall:.0f} s",
    flush=True,
)
print(f"START  bn {m0['bn']:.4e}  {D.report(m0, P)}  -> {D.verdict(m0, P)}")
print(
    f"RESULT bn {m1['bn']:.4e} max {m1['bn_max']:.3e}  {D.report(m1, P)}  linked "
    f"{m1['linked']} -> {D.verdict(m1, P)}  active {active}  {shape(cs1)}",
    flush=True,
)
solver = dict(
    message=str(res["message"]),
    success=bool(res["success"]),
    nit=int(res["nit"]),
    n_outer=len(res["outer_history"]),
    optimality=float(res["optimality"]),
    constr_violation=float(res["constr_violation"]),
    wall_s=wall,
)
json.dump(
    dict(
        settings=vars(a),
        commit=commit,
        bounds=P,
        info=info,
        solver=solver,
        start=m0,
        result=m1,
        margins=D.margins(m1, P),
        active=active,
        shape_start=shape(cs0),
        shape_result=shape(cs1),
    ),
    open(os.path.join(a.out, "result.json"), "w"),
    indent=1,
    default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o),
)
print(f"wrote {a.out}/result.h5, result.json, outer.json", flush=True)
