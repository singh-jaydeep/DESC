"""Single-stage (equilibrium + coils) optimization of a VACUUM equilibrium, by weighted least squares.

    # look at every term at the start (build, print the table, exit): do this before any run
    python single_stage.py --eq precise_QA --start work/ss_qa/start_arcB2.h5 --out work/ss_qa/dry --dry

    # boundary continuation k = 1..4 (modes with |m|, |n| <= k free), 50 iterations per step
    python single_stage.py --eq precise_QA --start work/ss_qa/start_arcB2.h5 --out work/ss_qa/run1 \\
        --kmax 4 --maxiter 50

Method: `proximal-lsq-exact` on things = [eq, coils]. ForceBalance, Psi, pressure, current and
the boundary modes outside the current step's |m|, |n| <= k (plus R_00, which sets the scale)
are the proximal constraints. Everything else is ONE weighted least-squares objective; there
is no augmented Lagrangian. A term with bounds contributes only its excursion outside them.

  physics   --field-term bn (default): VacuumBoundaryBn (sandbox/bnormal.py), rows sqrt(g) B_coil.n
                                  on rho = 1 -- quadratic flux on the moving boundary; the coil
                                  current SUM is held fixed (B.n cannot see the current scale)
            --field-term vbe: VacuumBoundaryError, which adds rows sqrt(g) (B_in^2 - B_coil^2)
                                  (~2 dB_t/B, the tangential mismatch); currents free
            QuasisymmetryTwoTerm  helicity --helicity, on the surfaces --qs-rho
            AspectRatio           bounds (--A-min, --A-max)
            RotationalTransform   bounds (--iota-min, inf) on every surface of its grid
  coils     CoilLength, CoilCurvature, CoilMeanSquaredCurvature   bounds (0, enforced value)
            CoilSetDistancePenalty       all pairs, two-pass, signed (the topology guard)
            PlasmaCoilSetMinDistance     hard minimum (kinked), eq NOT fixed; bounds (d_pc, inf)

`QuadraticFlux` cannot be used as the objective: it holds the equilibrium fixed (DESC refuses it
when eq is optimized). It is still the verdict: every step ends with the dense check
(common.evaluate: <|B.n|>/<|B|>, geometry against the unbacked-off bounds) on the CURRENT boundary.

Joint plane: --fix-hinge-z Z (arc coils) snaps every hinge to z = Z and holds hinges[:, 2] there by
FixParameters (linear, exact), as run_al.py does; only Z = 0 gives one plane under stellarator symmetry.

Bounds: --bounds (default bounds/precise_qa_sweep910.json, absolute, so they do not move with a or R0).
Enforced values carry run_one.py's backoffs (d_cc x1.025, d_pc x1.02; arcs kappa x0.995,
kappa_MS x0.99, and the segment sagitta on d_cc). With weights instead of multipliers the
violations will not settle where the backoffs assume; judge by the dense check.

Writes to --out: eq_kK.h5 and coils_kK.h5 after each step, result.json (settings, bounds, the
start table, and per-step physics + coil metrics).

Memory: run under the cgroup cap, one job at a time:
    systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 python single_stage.py ...
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox"))
import boot  # noqa: E402

DEVICE = boot.setup(default="gpu")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import common as S  # noqa: E402
from bnormal import VacuumBoundaryBn  # noqa: E402
from qs_scale_invariant import QuasisymmetryTwoTermScaleInvariant  # noqa: E402
import desc.objectives as O  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.io import load  # noqa: E402
from desc.optimize import Optimizer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def floats(s):
    return [float(v) for v in s.split(",")]


def parse():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = ap.add_argument_group("problem")
    g.add_argument("--eq", required=True, help="desc.examples name or DESC .h5 (a SOLVED vacuum equilibrium)")
    g.add_argument("--start", required=True, help="coilset .h5")
    g.add_argument("--out", required=True, help="output directory")
    g.add_argument("--bounds", default=os.path.join(HERE, "bounds", "precise_qa_sweep910.json"))
    g.add_argument("--a", type=float, default=None, help="minor radius for reactor-scaled bounds (not used "
                   "by absolute bounds files)")
    g.add_argument("--device", default=None, help="gpu or cpu (default gpu)")
    g.add_argument("--length-mult", type=float, default=1.0, help="length bound = this x the file's L (run_one.py's)")
    g.add_argument("--fix-coils", action="store_true",
                   help="freeze the coils: optimize the equilibrium ALONE against the same physics terms, B.n "
                        "of the fixed coils and d_pc (a diagnostic: does the eq side alone converge?)")
    g.add_argument("--fix-hinge-z", type=float, default=None,
                   help="arc coils: snap every hinge to this height and hold its z there (one joint plane at 0)")
    g.add_argument("--horizontal-hinges", action="store_true",
                   help="arc coils: each coil's hinges level at its own (free) height; sandbox/demount.HingeLevel")
    g.add_argument("--dry", action="store_true", help="build, print the term table at the start, and exit")
    g.add_argument("--field-term", choices=["bn", "vbe"], default="bn",
                   help="bn: B.n rows only (sandbox/bnormal.py; quadratic flux on the moving boundary; the "
                        "current sum is then held fixed). vbe: DESC's VacuumBoundaryError, B.n AND B^2 rows "
                        "(currents free; the B^2 rows set their scale)")
    g.add_argument("--fix-current-sum", action="store_true",
                   help="hold the SUM of the coil currents (stage 2's FixSumCoilCurrent); always on with "
                        "--field-term bn")
    g = ap.add_argument_group("physics targets")
    g.add_argument("--helicity", default="1,0", help="QS helicity M,N (QA 1,0; QH 1,NFP)")
    g.add_argument("--qs-rho", default="0.2,0.4,0.6,0.8,1.0", help="surfaces of the QS two-term residual")
    g.add_argument("--A-min", type=float, default=5.5)
    g.add_argument("--A-max", type=float, default=6.5)
    g.add_argument("--vol-band", type=float, default=0.02,
                   help="plasma volume held within +-this fraction of the START volume (0: no volume term). "
                        "B.n alone favours SHRINKING the plasma (the coil ripple decays inward): MEASURED "
                        "tau_QS x10, k=2 without it: a -5%%, V -10%%")
    g.add_argument("--V0", type=float, default=None,
                   help="centre of the volume band [m^3] (default: the start's volume). Pass the ORIGINAL value "
                        "when continuing from a later step, with --tau-qs and --bn-norm, so nothing re-anchors")
    g.add_argument("--bn-norm", type=float, default=None,
                   help="frozen B.n normalization (default: computed at the start); pass it when continuing")
    g.add_argument("--qs-form", choices=["scale-invariant", "stock"], default="scale-invariant",
                   help="scale-invariant: f_C/|B|^3 pointwise (DESC PR #2304, sandbox/qs_scale_invariant.py); "
                        "stock: f_C / B_scale^3, whose value moves with |B| (~B^3), i.e. with plasma size at "
                        "fixed Psi")
    g.add_argument("--iota-min", type=float, default=0.40, help="lower bound on iota, every surface")
    g = ap.add_argument_group("continuation and solver")
    g.add_argument("--kmin", type=int, default=1, help="first step: boundary modes |m|, |n| <= kmin free")
    g.add_argument("--kmax", type=int, default=4, help="last step")
    g.add_argument("--maxiter", type=int, default=50, help="iterations per step")
    g.add_argument("--ftol", type=float, default=1e-6)
    g.add_argument("--xtol", type=float, default=1e-6)
    g.add_argument("--gtol", type=float, default=1e-8)
    g.add_argument("--max-tr", type=float, default=None, help="max_trust_radius (default: solver's)")
    g.add_argument("--perturb-order", type=int, default=2)
    g.add_argument("--solve-tol", type=float, default=None,
                   help="ftol = xtol = gtol of the proximal equilibrium re-solve (default: the solver's). MEASURED "
                        "at the default: ~6e-4 of noise on a cost of 1.16 (precise_QA, k=1), which stalls the "
                        "outer solver once the available decrease is smaller; 1e-13 cut it ~20x at 8x the time")
    g = ap.add_argument_group("tolerances: a term whose residual equals its tolerance everywhere costs 1/2")
    g.add_argument("--tau-bn", type=float, default=5e-3,
                   help="rms of the boundary error rows, ~B.n/B and ~2 dB/B (VacuumBoundaryError normalized "
                        "by B R0 a; sqrt(g) ~ R0 a)")
    g.add_argument("--tau-qs", type=float, default=None, help="rms of the normalized QS two-term residual")
    g.add_argument("--tau-qs-rel", type=float, default=2.0,
                   help="if --tau-qs is unset: this multiple of the START's QS rms")
    g.add_argument("--tau-A", type=float, default=0.01, help="fractional excursion of A outside its bounds")
    g.add_argument("--tau-V", type=float, default=0.005,
                   help="fractional excursion of the volume outside its band")
    g.add_argument("--tau-iota", type=float, default=0.01, help="rms over rho of the excursion / iota_min")
    g.add_argument("--tau-L", type=float, default=0.01, help="fractional excursion, per coil")
    g.add_argument("--tau-kappa", type=float, default=0.01,
                   help="fractional excursion, rms over each coil's nodes, summed over coils")
    g.add_argument("--tau-kms", type=float, default=0.01, help="fractional excursion, per coil")
    g.add_argument("--tau-dpc", type=float, default=0.01, help="fractional shortfall, per coil row")
    g.add_argument("--tau-dcc", type=float, default=1e-6,
                   help="coil-coil penalty divided by the node pairs (N^2): the pair-mean of "
                        "max(0, 1 - d/d_min)^2 per coil pair; 1e-6 ~ 1%% penetration over 1%% of the pairs")
    g = ap.add_argument_group("enforced bounds")
    g.add_argument("--dcc-backoff", type=float, default=0.025)
    g.add_argument("--dpc-backoff", type=float, default=0.02)
    g.add_argument("--no-arc-backoffs", action="store_true")
    g = ap.add_argument_group("grids and memory")
    g.add_argument("--eval-m", type=int, default=25, help="VacuumBoundaryError grid")
    g.add_argument("--eval-n", type=int, default=25)
    g.add_argument("--plasma-m", type=int, default=32, help="plasma grid for d_pc (sym, half the surface)")
    g.add_argument("--plasma-n", type=int, default=32)
    g.add_argument("--coil-grid-n", type=int, default=150, help="kappa, kappa_MS and both distances")
    g.add_argument("--src-gl", type=int, default=24, help="Biot-Savart GL nodes per arc")
    g.add_argument("--src-n", type=int, default=400, help="Biot-Savart LinearGrid(N) for smooth coils")
    g.add_argument("--bs-chunk", type=int, default=None, help="Biot-Savart chunk (eval points)")
    g.add_argument("--jac-chunk", type=int, default=None, help="jac_chunk_size of the heavy terms")
    g.add_argument("--dcc-k", type=int, default=20000, help="two-pass capacity, coil-coil")
    return ap.parse_args()


def enforced(P, a, piecewise):
    """The bound values the optimizer is given (run_one.py's backoffs)."""
    E = dict(P, dcc=P["dcc"] * (1 + a.dcc_backoff), dpc=P["dpc"] * (1 + a.dpc_backoff))
    if piecewise and not a.no_arc_backoffs:
        E["kappa"] *= 0.995
        if E["kms"] is not None:
            E["kms"] *= 0.99
        L_coil = P["L"] if P["L"] is not None else None
        if L_coil is not None:  # segment sagitta on the enforcement grid
            E["dcc"] += 1.4 * P["kappa"] * (L_coil / a.coil_grid_n) ** 2 / 8
    return E


def qs_grid(eq, a):
    return LinearGrid(rho=np.array(floats(a.qs_rho)), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)


def build_terms(eq, cs, a, E):
    """All objective terms (physics, then coils), each weighted by its tolerance.

    Rule: weight = 1 / (tau * s), with s chosen so that a residual equal to tau in every row
    costs 1/2. DESC scales a residual as f / normalization * quad_weights * weight, so for a
    term whose rows are a quadrature, s = sqrt(sum quad_weights^2) and the tolerance is on the
    quadrature-weighted rms. The quad weights each term will use (DESC's rules) are computed
    here from fresh grid copies, BEFORE build (some builds scale grid.weights in place):
      VacuumBoundaryError   sqrt(w_grid), tiled over its 2 blocks  -> s^2 = 2 sum w_grid
      QuasisymmetryTwoTerm  w_grid sqrt(N)                         -> s^2 = N sum w_grid^2
      RotationalTransform   sqrt(d rho) per surface                -> s^2 = sum d rho
      CoilCurvature         d theta per node (NOT sqrt: the 1/N grid dependence); s^2 =
                            sum_nodes d theta^2 of ONE coil, so the cost is sum over coils of
                            the node-mean -- grid independent
      per-coil scalars      1 (length, kappa_MS, d_pc min): each coil's fractional excursion
      CoilSetDistancePenalty  1, but its value sums ~N^2 segment pairs: s = N^2 (pair mean)
    Coil and shape terms are un-normalized and divided by their bound, so their rows are
    FRACTIONAL excursions."""
    src, _ = S.source_grid(cs, a.src_gl, a.src_n)
    lg, _ = S.quad_grid(cs, a.src_gl, a.src_n)
    cg = LinearGrid(N=a.coil_grid_n)
    hel = tuple(int(v) for v in a.helicity.split(","))
    JC = {"jac_chunk_size": a.jac_chunk} if a.jac_chunk else {}

    def grid_bn():
        return LinearGrid(rho=np.array([1.0]), M=a.eval_m, N=a.eval_n, NFP=eq.NFP, sym=eq.sym)

    def grid_iota():
        return LinearGrid(rho=np.linspace(0.05, 1.0, 20), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)

    g = grid_bn()
    s_bn = np.sqrt((1 if a.field_term == "bn" else 2) * np.sum(np.array(g.weights)))
    g = qs_grid(eq, a)
    s_qs = np.sqrt(g.num_nodes * np.sum(np.array(g.weights) ** 2))
    g = grid_iota()
    s_iota = np.sqrt(np.sum(np.array(g.compress(g.spacing[:, 0], surface_label="rho"))))
    s_kappa = np.sqrt(np.sum(np.array(cg.spacing[:, 2]) ** 2))
    s_dcc = float(cg.num_nodes) ** 2
    A_scale = 0.5 * (a.A_min + a.A_max)
    S_ = dict(bn=s_bn, qs=s_qs, iota=s_iota, kappa=s_kappa, dcc=s_dcc)

    FK = {}
    if a.field_term == "bn":
        Field = VacuumBoundaryBn
        FK = {"norm": getattr(a, "bn_norm", None)}  # frozen at the start's value (set in main)
    else:
        Field = O.VacuumBoundaryError
    QS = QuasisymmetryTwoTermScaleInvariant if a.qs_form == "scale-invariant" else O.QuasisymmetryTwoTerm
    terms = [
        Field(eq, cs, eval_grid=grid_bn(), field_grid=src, bs_chunk_size=a.bs_chunk, field_fixed=a.fix_coils,
              weight=1 / (a.tau_bn * s_bn), **FK, **JC),
        QS(eq, helicity=hel, grid=qs_grid(eq, a), weight=1 / (a.tau_qs * s_qs), **JC),
        O.AspectRatio(eq, bounds=(a.A_min, a.A_max), normalize=False, weight=1 / (a.tau_A * A_scale)),
        O.RotationalTransform(eq, bounds=(a.iota_min, np.inf), normalize=False, grid=grid_iota(),
                              weight=1 / (a.tau_iota * a.iota_min * s_iota)),
    ]
    if a.vol_band > 0:  # band around the START volume (a.V0, set once in main)
        terms.append(O.Volume(eq, bounds=(a.V0 * (1 - a.vol_band), a.V0 * (1 + a.vol_band)), normalize=False,
                              weight=1 / (a.tau_V * a.V0)))
    dpc = O.PlasmaCoilSetMinDistance(eq, cs, bounds=(E["dpc"], np.inf), normalize=False,
                                     plasma_grid=LinearGrid(M=a.plasma_m, N=a.plasma_n, NFP=eq.NFP, sym=eq.sym),
                                     coil_grid=cg, eq_fixed=False, coils_fixed=a.fix_coils,
                                     weight=1 / (a.tau_dpc * E["dpc"]))
    if a.fix_coils:  # the coil-only terms are constants: leave them out
        return terms + [dpc], S_
    if E["L"] is not None:
        terms.append(O.CoilLength(cs, bounds=(0.0, E["L"]), normalize=False, grid=lg,
                                  weight=1 / (a.tau_L * E["L"])))
    terms.append(O.CoilCurvature(cs, bounds=(0.0, E["kappa"]), normalize=False, grid=cg,
                                 weight=1 / (a.tau_kappa * E["kappa"] * s_kappa)))
    if E["kms"] is not None:
        terms.append(O.CoilMeanSquaredCurvature(cs, bounds=(0.0, E["kms"]), normalize=False, grid=cg,
                                                weight=1 / (a.tau_kms * E["kms"])))
    terms += [
        O.CoilSetDistancePenalty(cs, d_min=E["dcc"], grid=cg, dist_chunk_size=1, distance_method="segment",
                                 max_active_pairs=(a.dcc_k or None), signed=bool(a.dcc_k), deriv_mode="rev",
                                 normalize=False, weight=1 / (a.tau_dcc * s_dcc)),
        dpc,
    ]
    return terms, S_


def start_qs_rms(eq, a):
    """The start's QS two-term rms (normalized, quadrature-weighted), to set tau_qs relative to it."""
    g = qs_grid(eq, a)
    s = np.sqrt(g.num_nodes * np.sum(np.array(g.weights) ** 2))
    QS = QuasisymmetryTwoTermScaleInvariant if a.qs_form == "scale-invariant" else O.QuasisymmetryTwoTerm
    o = QS(eq, helicity=tuple(int(v) for v in a.helicity.split(",")), grid=qs_grid(eq, a))
    o.build(verbose=0)
    return float(np.linalg.norm(np.asarray(o.compute_scaled_error(eq.params_dict)))) / s


def hinge_z_constraint(cs, z):
    """FixParameters on every hinge's z (the values already in cs, snapped to z by main)."""
    zi = np.arange(int(next(S.iter_unique(cs)).B)) * 3 + 2

    def z_mask(t):
        return {q: (zi if q == "hinges" else False) for q in t} if isinstance(t, dict) else [z_mask(u) for u in t]

    return O.FixParameters(cs, params=z_mask(cs.params_dict), name="hinge joint plane")


def max_hinge_dz(cs, z):
    """Largest |hinge z - z|; with z None, the largest per-coil hinge-height spread."""
    if z is None:
        return max(float(np.ptp(np.asarray(c.hinges).reshape(-1, 3)[:, 2])) for c in S.iter_unique(cs))
    return max(float(np.abs(np.asarray(c.hinges).reshape(-1, 3)[:, 2] - z).max()) for c in S.iter_unique(cs))


def proximal_constraints(eq, k, fix_sum, cs, hinge_z=None, level=False):
    """ForceBalance plus the fixed-boundary set, with only |m|, |n| <= k boundary modes free
    (R_00 always fixed: it sets the machine size)."""
    R = eq.surface.R_basis.modes
    Z = eq.surface.Z_basis.modes
    Rfix = np.vstack(([0, 0, 0], R[np.max(np.abs(R), axis=1) > k]))
    Zfix = Z[np.max(np.abs(Z), axis=1) > k]
    base = [c for c in O.get_fixed_boundary_constraints(eq)
            if not isinstance(c, (O.FixBoundaryR, O.FixBoundaryZ))]
    cons = [O.ForceBalance(eq), O.FixBoundaryR(eq, modes=Rfix), *base]
    if len(Zfix):
        cons.append(O.FixBoundaryZ(eq, modes=Zfix))
    if fix_sum:
        cons.append(O.FixSumCoilCurrent(cs))
    if hinge_z is not None:
        cons.append(hinge_z_constraint(cs, hinge_z))
    if level:
        import demount

        cons.append(demount.HingeLevel(cs))
    n_free = (len(R) - len(Rfix)) + (len(Z) - len(Zfix))
    return tuple(cons), n_free


def term_table(terms, title):
    """value, bounds and weighted share of the cost for every term, at the things' current state."""
    rows, tot = [], 0.0
    for t in terms:
        p = [th.params_dict for th in t.things]
        raw = np.atleast_1d(np.asarray(t.compute_unscaled(*p)))
        err = np.atleast_1d(np.asarray(t.compute_scaled_error(*p)))
        c = 0.5 * float(np.sum(err**2))
        tot += c
        if t.bounds is not None:
            lo, hi = (float(np.ravel(b)[0]) for b in t.bounds)
            tgt = f"[{lo:.4g}, {hi:.4g}]"
        else:
            tgt = f"= {float(np.ravel(t.target)[0]):.4g}"
        norm = getattr(t, "_normalization", None)
        norm = "-" if norm is None or not t._normalize else f"{float(np.ravel(norm)[0]):.4g}"
        qw = np.asarray(t.constants.get("quad_weights", 1.0), float)
        s_built = float(np.sqrt(np.sum(np.broadcast_to(qw, (int(t.dim_f),)) ** 2)))
        rows.append(dict(name=t.name, dim_f=int(t.dim_f), weight=float(np.ravel(t.weight)[0]), norm=norm,
                         target=tgt, raw_min=float(raw.min()), raw_max=float(raw.max()), s_built=s_built,
                         cost=c, err_over_tol=float(np.sqrt(2 * c))))
    print(f"\n{title}\n{'term':<32}{'dim_f':>6}{'weight':>10}{'norm':>9}{'s(qw)':>9}  {'target/bounds':<20}"
          f"{'raw min':>11}{'raw max':>11}{'cost':>11}{'err/tol':>9}{'share':>7}")
    for r in rows:
        r["share"] = r["cost"] / tot if tot > 0 else 0.0
        print(f"{r['name'][:31]:<32}{r['dim_f']:>6}{r['weight']:>10.3g}{r['norm']:>9}{r['s_built']:>9.4g}  "
              f"{r['target'][:19]:<20}{r['raw_min']:>11.4g}{r['raw_max']:>11.4g}{r['cost']:>11.4g}"
              f"{r['err_over_tol']:>9.3g}{100 * r['share']:>6.1f}%")
    print(f"{'total cost 0.5 sum f^2':<32}{tot:>98.4g}", flush=True)
    return rows


class Physics:
    """Physics diagnostics of the current equilibrium: A, iota range, and the Boozer QS error
    (symmetry-breaking |B|_mn over <|B|>_vol, L2 over modes) on a few surfaces. The Boozer
    objectives are built once and re-evaluated on eq.params_dict, which follows eq in place."""

    RHO = (0.25, 0.5, 0.75, 1.0)

    def __init__(self, eq, helicity):
        hel = tuple(int(v) for v in helicity.split(","))
        self.eq = eq
        self.booz = []
        for r in self.RHO:
            o = O.QuasisymmetryBoozer(eq, helicity=hel, normalize=False,
                                      grid=LinearGrid(rho=np.array([r]), M=2 * eq.M_grid, N=2 * eq.N_grid,
                                                      NFP=eq.NFP, sym=False))
            o.build(verbose=0)
            self.booz.append(o)
        self.igrid = LinearGrid(rho=np.linspace(0.0, 1.0, 21), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)

    def __call__(self):
        eq = self.eq
        d = eq.compute(["R0", "a", "R0/a", "V", "<|B|>_vol"])
        iota = self.igrid.compress(eq.compute("iota", grid=self.igrid)["iota"])
        B = float(d["<|B|>_vol"])
        qs = [float(np.linalg.norm(o.compute_unscaled(eq.params_dict))) / B for o in self.booz]
        return dict(R0=float(d["R0"]), a=float(d["a"]), A=float(d["R0/a"]), V=float(d["V"]), B_vol=B,
                    iota_min=float(np.min(iota)), iota_max=float(np.max(iota)),
                    qs_booz={f"{r:g}": v for r, v in zip(self.RHO, qs)})


def fmt_phys(p):
    q = "  ".join(f"rho {r} {v:.2e}" for r, v in p["qs_booz"].items())
    return (f"A {p['A']:.3f}  R0 {p['R0']:.4f}  a {p['a']:.4f}  iota [{p['iota_min']:.4f}, {p['iota_max']:.4f}]  "
            f"QS(Boozer, |B_nonsym|/<B>) {q}")


def coil_check(cs, eq, P):
    """Dense check against the CURRENT boundary. common.evaluate caches boundary data by id(eq),
    and eq changes in place here, so the cache is cleared first."""
    S._BN_CACHE.clear()
    m = S.evaluate(cs, eq, S.surface_tree(eq), vacuum=True)
    return m, f"bn {m['bn']:.4e} (max {m['bn_max']:.3e})  {S.report(m, P)}  linked {m['linked']}  -> {S.verdict(m, P)}"


def main():
    a = parse()
    if a.fix_coils:
        a.fix_current_sum = False
        print("--fix-coils: coils frozen; only the equilibrium is optimized", flush=True)
    elif a.field_term == "bn" and not a.fix_current_sum:
        a.fix_current_sum = True
        print("--field-term bn is blind to the current scale: holding the coil current SUM fixed", flush=True)
    os.makedirs(a.out, exist_ok=True)
    S.memwatch(every=120)
    eq, kind = S.load_equilibrium(a.eq)
    dd = eq.compute(["p", "current"])
    if np.max(np.abs(dd["p"])) > 1e-8 or np.max(np.abs(dd["current"])) > 1e-8:
        raise SystemExit("this driver is for VACUUM equilibria (VacuumBoundaryError); p or current is nonzero")
    if kind.startswith("input"):
        raise SystemExit("solve the equilibrium first: an input-file equilibrium is boundary-only")
    cs = load(a.start)
    nc = S.n_unique(cs)
    if a.fix_hinge_z is not None and a.horizontal_hinges:
        raise SystemExit("--fix-hinge-z and --horizontal-hinges are exclusive")
    if a.fix_hinge_z is not None or a.horizontal_hinges:
        if not S.is_arc(cs) or a.fix_coils:
            raise SystemExit("--fix-hinge-z/--horizontal-hinges need arc coils that are optimized")
        moved = 0.0
        for c in S.iter_unique(cs):
            p = c.params_dict
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
        print(f"JOINT PLANE: {what} (largest move {moved:.2e} m)", flush=True)
    B = S.paper_bounds(eq, nc, a.a, a.bounds)
    P = {k: B[k] for k in S.UPPER + S.LOWER}
    if P["L"] is not None:
        P["L"] *= a.length_mult
    E = enforced(P, a, S.is_piecewise(cs))
    print(f"eq {a.eq} ({kind}) NFP {eq.NFP} sym {eq.sym} L,M,N {eq.L},{eq.M},{eq.N} dim_x {eq.dim_x}; "
          f"start {a.start}: {nc} unique coils ({type(next(S.iter_unique(cs))).__name__}), "
          f"{S.expected_coils(cs, eq)} total\nSETTINGS {vars(a)}\n"
          f"BOUNDS (judged, {B['scale']}) {S.fmt_bounds(P)}\nENFORCED {S.fmt_bounds(E)}", flush=True)

    phys = Physics(eq, a.helicity)
    p0 = phys()
    m0, line0 = coil_check(cs, eq, P)
    print(f"START physics  {fmt_phys(p0)}\nSTART coils    {line0}", flush=True)

    if a.V0 is None:
        a.V0 = float(eq.compute("V")["V"])
    qs0 = start_qs_rms(eq, a)
    if a.tau_qs is None:
        a.tau_qs = a.tau_qs_rel * qs0
    print(f"QS two-term rms at the start {qs0:.4g} -> tau_qs {a.tau_qs:.4g}", flush=True)
    terms, S_ = build_terms(eq, cs, a, E)
    if a.field_term == "bn" and a.bn_norm is None:  # freeze the B.n scale at the start's value
        terms[0].build(verbose=0)
        a.bn_norm = float(terms[0].normalization)
        terms, S_ = build_terms(eq, cs, a, E)
    if a.field_term == "bn":
        print(f"B.n normalization frozen at the start: {a.bn_norm:.5g}; volume band around V0 = {a.V0:.5g}",
              flush=True)
    print("quadrature scales s (expected; compare the s(qw) column): "
          + "  ".join(f"{k} {v:.4g}" for k, v in S_.items()), flush=True)
    objective = O.ObjectiveFunction(tuple(terms))
    objective.build(verbose=0)
    table0 = term_table(terms, "TERMS AT THE START")
    rec = dict(settings=vars(a), qs_rms_start=qs0, device=DEVICE, bounds=P, enforced=E, start=dict(physics=p0, coils=m0, terms=table0),
               steps=[])
    json.dump(rec, open(os.path.join(a.out, "result.json"), "w"), indent=1, default=float)
    if a.dry:
        print(f"\n--dry: wrote {a.out}/result.json; no optimization run")
        return

    for k in range(a.kmin, a.kmax + 1):
        cons, n_free = proximal_constraints(eq, k, a.fix_current_sum, cs, a.fix_hinge_z, a.horizontal_hinges)
        objective = O.ObjectiveFunction(tuple(build_terms(eq, cs, a, E)[0]))
        print(f"\n===== step k={k}: {n_free} boundary modes free, maxiter {a.maxiter} =====", flush=True)
        solve = {"verbose": 0}
        if a.solve_tol is not None:
            solve.update(ftol=a.solve_tol, xtol=a.solve_tol, gtol=a.solve_tol)
        opts = {"perturb_options": {"order": a.perturb_order, "verbose": 0}, "solve_options": solve}
        if a.max_tr is not None:
            opts["max_trust_radius"] = a.max_tr
        t0 = time.time()
        things = [eq] if a.fix_coils else [eq, cs]
        out, res = Optimizer("proximal-lsq-exact").optimize(
            things, objective=objective, constraints=cons, maxiter=a.maxiter, ftol=a.ftol, xtol=a.xtol,
            gtol=a.gtol, verbose=3, copy=False, options=opts)
        wall = time.time() - t0
        eq = out[0]
        if not a.fix_coils:
            cs = out[1]
        phys.eq = eq
        p = phys()
        m, line = coil_check(cs, eq, P)
        table = term_table(objective.objectives, f"TERMS AFTER k={k}")
        print(f"k={k} physics  {fmt_phys(p)}\nk={k} coils    {line}\nk={k} solver   {res['message']}  "
              f"nit {int(res['nit'])}  wall {wall / 60:.1f} min", flush=True)
        if a.fix_hinge_z is not None:
            print(f"k={k} joint    max |hinge z - {a.fix_hinge_z}| {max_hinge_dz(cs, a.fix_hinge_z):.1e} m", flush=True)
        if a.horizontal_hinges:
            zc = [round(float(np.mean(np.asarray(c.hinges).reshape(-1, 3)[:, 2])), 3) for c in S.iter_unique(cs)]
            print(f"k={k} joint    max hinge-height spread {max_hinge_dz(cs, None):.1e} m, coil z {zc}", flush=True)
        eq.save(os.path.join(a.out, f"eq_k{k}.h5"))
        cs.save(os.path.join(a.out, f"coils_k{k}.h5"))
        rec["steps"].append(dict(k=k, n_free=n_free, wall_s=wall, nit=int(res["nit"]), message=str(res["message"]),
                                 physics=p, coils=m, terms=table))
        json.dump(rec, open(os.path.join(a.out, "result.json"), "w"), indent=1, default=float)
    print(f"\nwrote {a.out}: eq_k*.h5, coils_k*.h5, result.json")


if __name__ == "__main__":
    main()
