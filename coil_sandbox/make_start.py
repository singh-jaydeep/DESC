"""Make a first coilset for an equilibrium, and check it.

    python make_start.py --eq precise_QA --out work/qa/start_planar.h5
    python make_start.py --eq my_eq.h5 --nc 4 --r-over-a 3.3 --rep arc --B 5 --out work/mine/start_arcB5.h5
    python make_start.py --eq input.my_config --rep xyz --N 6 --out work/mine/start_xyzN6.h5
    python make_start.py --eq my_eq.h5 --from-coils work/mine/planar.h5 --rep arc --B 2 --M 5 --phase 0.25 \
        --out work/mine/start_arcB2.h5          # arcs fitted to an existing (e.g. optimized) coilset

Circles centred on the magnetic axis, `--nc` per half field period, radius `--r-over-a` x a,
with currents matched to the equilibrium (DESC's `initialize_modular_coils`), then converted:

  --rep planar   FourierPlanarCoil, raised to --N modes (the circles are N = 0)
  --rep xyz      FourierXYZCoil with --N modes
  --rep arc      PiecewisePlanarArcCoil with --B arcs and --M modes per arc, fitted to the circle
                 with fit_method="chord" (the default "parameter" fit is a known silent bug)

With --from-coils the unique coils of that coilset are converted instead of circles (--nc and
--r-over-a are then ignored). For arcs, --phase (a fraction of a turn) moves the first hinge
along each coil's own parameter; the fit's max deviation from the source coil is printed.

It prints the dense check against the bounds (see `sandbox/common.py: paper_bounds`) and writes
the coilset plus `<out>.json`. A start does not need to be feasible, but it must NOT be linked,
and it helps if nothing is far outside the bounds: `run_one.py --mode feasibility` moves a start
inside the bounds before a stage-2 run. Circles are not always a good start: on precise_QH the
circle start was linked for the r/a that suited precise_QA, so check the `linked` count and try
another --r-over-a.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox"))
import boot  # noqa: E402

boot.setup(default="cpu")

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

import common as S  # noqa: E402
from desc.coils import (  # noqa: E402
    CoilSet, MixedCoilSet, PiecewisePlanarArcCoil, PolarPlanarArcCoil, initialize_modular_coils)
from desc.grid import Grid, LinearGrid  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eq", required=True, help="desc.examples name, DESC .h5, wout_*.nc, or input file")
    ap.add_argument("--out", required=True, help="output .h5")
    ap.add_argument("--nc", type=int, default=4, help="independent coils per half field period")
    ap.add_argument("--r-over-a", type=float, default=3.3, help="circle radius / minor radius")
    ap.add_argument("--rep", choices=["planar", "xyz", "arc", "polararc", "spline"], default="planar",
                    help="arc: PiecewisePlanarArcCoil (each arc a graph OVER its chord); polararc: "
                         "PolarPlanarArcCoil (each arc a polar graph about its chord midpoint); "
                         "spline: SplineXYZCoil with C0 breaks on the source coil's hinges")
    ap.add_argument("--knots", type=int, default=32, help="spline: knots per coil (a multiple of B)")
    ap.add_argument("--method", default="cubic2",
                    help="spline: cubic2 (C2, continuous curvature) or cubic (C1, jagged curvature)")
    ap.add_argument("--N", type=int, default=7, help="Fourier modes (planar, xyz)")
    ap.add_argument("--B", type=int, default=5, help="arcs per coil (arc)")
    ap.add_argument("--M", type=int, default=3, help="modes per arc (arc)")
    ap.add_argument("--a", type=float, default=None, help="minor radius for the bounds (default: DESC's)")
    ap.add_argument("--bounds", default=None, help="bounds JSON (see sandbox/common.py: paper_bounds)")
    ap.add_argument("--device", default=None, help="gpu or cpu (default cpu)")
    ap.add_argument("--vacuum", action="store_true", help="ignore the plasma's own field in the B.n check")
    ap.add_argument("--from-coils", default=None, help="convert this coilset's unique coils instead of circles")
    ap.add_argument("--phase", default="0", help="arcs: first hinge at this fraction of a turn; one value, "
                    "or a comma-separated value per unique coil")
    ap.add_argument("--fit-samples", type=int, default=720, help="arcs: samples per coil for the fit")
    ap.add_argument("--cut-angle", default=None,
                    help="polar arcs, B=2, from --from-coils: hinges where the line through the coil "
                         "centroid at this many degrees from vertical crosses the coil; one value, or a "
                         "comma-separated value per unique coil")
    ap.add_argument("--hinge-s", default=None,
                    help="arcs from --from-coils: the hinge PARAMETER values explicitly, as "
                         "'s0,s1,...;s0,s1,...' -- one semicolon-separated group per unique coil, B values "
                         "each. Unlike --cut-angle the chord need not pass through the centroid, so this is "
                         "the way to use hinges found by a search (e.g. planarizability.py's DP breakpoints)")
    ap.add_argument("--chord-angle", type=float, default=None,
                    help="arcs, B=2, from --from-coils: put the hinges at the coil's extreme points along the "
                         "in-plane direction this many degrees from vertical (0 = vertical chord, 90 = "
                         "inboard-outboard); overrides --hinges")
    ap.add_argument("--hinges", choices=["phase", "axis"], default="phase",
                    help="arcs from --from-coils: 'phase' splits at equal PARAMETER intervals from --phase; "
                         "'axis' puts hinges at the coil's long-axis ends (B=2), or (B>2) at equal arclength "
                         "starting --phase of the coil length after a long-axis end; either way the coil is "
                         "resampled so its own parameterization does not matter")
    a = ap.parse_args()

    eq, kind = S.load_equilibrium(a.eq)
    S.check_vacuum_choice(eq, kind, a.vacuum)
    print(f"equilibrium {a.eq} ({kind}): NFP {eq.NFP} sym {eq.sym} M {eq.M} N {eq.N}", flush=True)
    b = S.paper_bounds(eq, a.nc, a.a, a.bounds)
    print(f"bounds ({b['scale']}, a = {b['a']:.5f} m, f = {b['f']:.5f}): {S.fmt_bounds(b)}", flush=True)

    if a.from_coils:
        from desc.io import load

        circles = load(a.from_coils)
        a.nc = S.n_unique(circles)
        b = S.paper_bounds(eq, a.nc, a.a, a.bounds)
        src_label = f"fit to {a.from_coils}"
        print(f"  ({a.nc} unique coils in {a.from_coils}; bounds for them: {S.fmt_bounds(b)})", flush=True)
    else:
        circles = initialize_modular_coils(eq, num_coils=a.nc, r_over_a=a.r_over_a)
        src_label = f"r/a {a.r_over_a}"
    dev = None
    if a.rep == "planar":
        if a.from_coils:
            raise SystemExit("--from-coils with --rep planar: not supported (planar coils are already planar)")
        cs = circles.copy()
        for c in cs:
            c.change_resolution(N=a.N)
        label = f"planar N={a.N}"
    elif a.rep == "xyz":
        cs = circles.to_FourierXYZ(N=a.N)
        label = f"FourierXYZ N={a.N}"
    elif a.rep == "spline":
        # Knots uniform in the SOURCE coil's parameter, so a knot lands on every hinge of an
        # arc coil (its hinges sit at s = 2 pi i / B exactly) and the C0 corners carry over.
        from desc.coils import SplineXYZCoil

        uniq = list(S.iter_unique(circles))
        # With --from-coils the breaks inherit the source coil's hinges (an arc coil puts them at
        # s = 2 pi i / B exactly). Cold from circles there are no hinges, so place --B of them
        # evenly in parameter; on a circle that is a C0 point with a zero corner, and the optimizer
        # opens it up if that pays.
        B = int(getattr(uniq[0], "B", a.B))
        if a.knots % B:
            raise SystemExit(f"--knots {a.knots} must be a multiple of the source B={B}")
        k = np.linspace(0, 2 * np.pi, a.knots, endpoint=False)
        coils, dev = [], []
        for c in uniq:
            pts = np.asarray(c.compute("x", grid=Grid(np.stack([0 * k, 0 * k, k], 1), sort=False,
                                                      jitable=False), basis="xyz")["x"])
            sp = SplineXYZCoil.from_values(float(c.current), pts, knots=k, method=a.method, basis="xyz",
                                           break_indices=np.arange(B) * (a.knots // B))
            coils.append(sp)
            sd = np.linspace(0, 2 * np.pi, 4001, endpoint=False)
            gg = Grid(np.stack([0 * sd, 0 * sd, sd], 1), sort=False, jitable=False)
            dev.append(float(np.linalg.norm(
                np.asarray(sp.compute("x", grid=gg, basis="xyz")["x"])
                - np.asarray(c.compute("x", grid=gg, basis="xyz")["x"]), axis=1).max()))
        cs = MixedCoilSet(*[CoilSet(c, NFP=eq.NFP, sym=eq.sym) for c in coils])
        label = (f"SplineXYZ {a.knots} knots, {a.method}, {B} C0 breaks "
                 + ("on the source hinges" if a.from_coils else "evenly spaced (cold start)"))
    else:
        coils, dev = [], []
        n = a.fit_samples
        uniq = list(S.iter_unique(circles))
        phases = [float(v) for v in str(a.phase).split(",")]
        phases = phases * len(uniq) if len(phases) == 1 else phases
        if len(phases) != len(uniq):
            raise SystemExit(f"--phase: {len(phases)} values for {len(uniq)} coils")
        hs = [None] * len(uniq)
        if a.hinge_s is not None:
            hs = [[float(v) for v in g.split(",")] for g in str(a.hinge_s).split(";")]
            if len(hs) != len(uniq):
                raise SystemExit(f"--hinge-s: {len(hs)} coil groups for {len(uniq)} unique coils")
            for g in hs:
                if len(g) != a.B:
                    raise SystemExit(f"--hinge-s: {len(g)} hinges in a group, expected B={a.B}")
        cuts = [None] * len(uniq)
        if a.cut_angle is not None:
            cuts = [float(v) for v in str(a.cut_angle).split(",")]
            cuts = cuts * len(uniq) if len(cuts) == 1 else cuts
            if len(cuts) != len(uniq):
                raise SystemExit(f"--cut-angle: {len(cuts)} values for {len(uniq)} coils")
        for c, ph, cut, hg in zip(uniq, phases, cuts, hs):
            if hg is not None:
                pts = S.resample_between(c, sorted(np.mod(hg, 2 * np.pi)), n_per=max(n // a.B, 60))
            elif cut is not None:
                if a.B != 2 or a.rep != "polararc":
                    raise SystemExit("--cut-angle is for --rep polararc --B 2")
                pts = S.resample_between(c, S.centroid_line_hinges(c, cut), n_per=max(n // 2, 60))
            elif a.chord_angle is not None:
                if a.B != 2:
                    raise SystemExit("--chord-angle is for B=2")
                pts = S.resample_between(c, S.direction_hinges(c, a.chord_angle), n_per=max(n // 2, 60))
            elif a.hinges == "axis":
                h = S.long_axis_hinges(c)
                if a.B > 2:  # first hinge at a long-axis end shifted by --phase (arclength fraction), rest equal arclength
                    h = S.arclength_hinges(c, a.B, h[0], offset=ph)
                pts = S.resample_between(c, h, n_per=max(n // len(h), 60))
            else:
                s = np.mod(2 * np.pi * (ph + np.arange(n) / n), 2 * np.pi)
                src = c.compute("x", grid=Grid(np.stack([0 * s, 0 * s, s], 1), sort=False, jitable=False),
                                basis="xyz")
                pts = np.asarray(src["x"])
            if a.rep == "polararc":
                arc = PolarPlanarArcCoil.from_values(float(c.current), pts, B=a.B, M=a.M, basis="xyz")
            else:
                arc = PiecewisePlanarArcCoil.from_values(float(c.current), pts, B=a.B, M=a.M, basis="xyz",
                                                         fit_method="chord")
            coils.append(arc)
            fit = np.asarray(arc.compute("x", grid=LinearGrid(N=400), basis="xyz")["x"])
            dense = np.asarray(c.compute("x", grid=LinearGrid(N=2000), basis="xyz")["x"])
            dev.append(float(cKDTree(dense).query(fit)[0].max()))
        cs = MixedCoilSet(*[CoilSet(c, NFP=eq.NFP, sym=eq.sym) for c in coils])
        label = f"{'polar ' if a.rep == 'polararc' else ''}arcs B={a.B} M={a.M} " + (
            f"hinges at given parameters {a.hinge_s}" if a.hinge_s is not None else
            f"cut line at {a.cut_angle} deg from vertical" if a.cut_angle is not None else
            f"chord at {a.chord_angle} deg from vertical" if a.chord_angle is not None
                                            else "hinges on the long axis" if a.hinges == "axis" else f"phase {a.phase}")

    m = S.evaluate(cs, eq, vacuum=a.vacuum)
    exp = S.expected_coils(cs, eq)
    print(f"{label}, {src_label}: {m['n_coils']} coils (expected {exp}), {S.n_unique(cs)} unique")
    if dev is not None:
        print("  fit max deviation from the source coil: " + ", ".join(f"{d * 1e3:.1f} mm" for d in dev))
    print(f"  L {m['L']:.4f} m  kappa {m['kappa']:.3f}  kappa_MS {m['kms']:.3f}  d_cc {m['d_cc'] * 1e3:.1f} mm  "
          f"d_pc {m['d_pc'] * 1e3:.1f} mm  convexity excess {m['convex']:.2e}  corner {m['corner']:.0f} deg  "
          f"linked {m['linked']}")
    print(f"  B.n ({'vacuum' if a.vacuum else 'coils + plasma'}): <|B.n|>/<|B|> {m['bn']:.3e}  "
          f"<|B.n|/|B|> {m['bn_local']:.3e}  max|B.n| {m['bn_max_T']:.3f} T")
    print(f"  value / bound: {S.report(m, b)}  -> {S.verdict(m, b)}")
    if m["linked"]:
        print("  WARNING: linked pairs " + ", ".join(f"{p[0]}-{p[1]}" for p in m["linked_pairs"])
              + ". Do not start a run from this; try another --r-over-a.")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cs.save(a.out)
    json.dump(dict(eq=a.eq, eq_kind=kind, vacuum=a.vacuum, nc=a.nc, r_over_a=a.r_over_a, rep=a.rep, N=a.N, B=a.B, M=a.M,
                   knots=a.knots, method=a.method,
                   from_coils=a.from_coils, phase=a.phase, hinges=a.hinges, chord_angle=a.chord_angle,
                   cut_angle=a.cut_angle,
                   fit_deviation=dev,
                   bounds=b, metrics=m, margins=S.margins(m, b), verdict=S.verdict(m, b)),
              open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1, default=float)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
