"""Experiment definitions for the Hurwitz-comparison benchmark.

Each function returns a list of config dicts. Run one experiment:

    python -m bench.cases --exp E0 --device cpu
    python -m bench.cases --exp E1 --device gpu --outdir runs/E1

`--dry-run` prints the configs (and the paper-unit bounds each implies) without
solving, which is the cheap way to check a sweep before it costs GPU time.

Experiments are ordered so that each one's premise is established by the
previous. E0 and E1 are diagnostics on OUR formulation -- run them before any
sweep, because if E1 shows the outer loop is not advancing then every number
from a sweep is a report on a half-solved subproblem, not on the method.
"""

import argparse
import json
import os

from bench.bench import DEFAULTS, configure_device, rescale_to_eq


def _base(**kw):
    cfg = dict(DEFAULTS)
    cfg.update(kw)
    return cfg


# ---------------------------------------------------------------------------
# E0 -- current-degeneracy control  (CHEAP, run first, CPU is fine)
# ---------------------------------------------------------------------------
def E0_current_degeneracy():
    """Does leaving currents free let the solver buy cost with current shrink.

    `QuadraticFlux(vacuum=True)` is homogeneous of degree 2 in the currents, so
    scaling all currents by s scales the cost by s^2 while leaving the reported
    metric mean|B.n|/mean|B| unchanged. The notebook's `FixCoilCurrent` on one
    index leaves nc-1 currents free; its saved runs drifted to 0.63x the initial
    total. This isolates the effect: identical problem, three current treatments.

    Read: compare `normalized_BdotN` against `solver.cost` and `current_drift`.
    If "one" reaches a much lower cost than "sum" at the same or worse metric,
    the cost reduction was bought with current, not geometry -- and cost history
    is not a progress signal for this problem.
    """
    return [
        _base(tag=f"E0_{mode}", current_mode=mode, nc=5, maxiter=150)
        for mode in ("sum", "one", "all")
    ]


# ---------------------------------------------------------------------------
# E1 -- auglag convergence diagnosis  (CHEAP-ish, run before any sweep)
# ---------------------------------------------------------------------------
def E1_auglag_convergence():
    """Why has the solver never reported convergence.

    Three candidate causes, distinguished by what the run records:

      (a) `gtol` is below the problem's noise floor. Then `n_outer` grows
          normally, `constr_violation` sits at/below `ctol`, and the new
          "precision" status should fire. Fix: report at achievable precision.
      (b) The outer loop is starved -- `gtolk` unreachable, so `n_outer` stays
          ~1-3 over hundreds of iterations and `y` never converges. This is the
          failure `inner_reduction` was written to fix; the sweep over it says
          whether the default of 10 is right for this problem.
      (c) `ctol` is below the achievable violation, so `mu` ratchets forever.
          Shows up as `constr_violation` plateauing above `ctol` with `mu` large.

    `inner_reduction` 3 / 10 / 30 crossed with `ctol` 1e-3 / 1e-4 / 1e-5 tells
    (b) from (c): if only `ctol` matters it is the constraint tolerance; if only
    `inner_reduction` matters it is the inner/outer balance.
    """
    out = []
    for ir in (3.0, 10.0, 30.0):
        for ctol in (1e-3, 1e-4, 1e-5):
            out.append(
                _base(
                    tag=f"E1_ir{ir:g}_ctol{ctol:g}",
                    nc=5,
                    maxiter=400,
                    ctol=ctol,
                    options={"inner_reduction": ir, "track_residual": True},
                )
            )
    return out


# ---------------------------------------------------------------------------
# E2 -- can we reach the paper's front at all?  (the headline question)
# ---------------------------------------------------------------------------
def E2_matched_setup():
    """Paper's exact setup, at their constraints. The reproduce-or-not test.

    5 coils per half period, order-16 FourierXYZ, fixed total current, their
    Table 1 bounds. `r_over_a` is swept because they initialize from circles on
    a torus of radius R1 drawn from U(0.35, 0.75) at R0 = 1 -- i.e.
    r_over_a ~ 2.0 to 4.4 in DESC's terms -- and initialization is the single
    biggest lever on which basin you land in. Their front is the result of 4363
    cold starts; a handful of starts from us is not going to match its best
    point, so the question is whether we land ON the front's cloud, not at its
    tip.

    Success criterion: at least one case passing `paper_filter` with
    normalized_BdotN <= 4e-3 (their filter bound), and ideally within the
    front's interquartile range, 7.0e-5 to 4.5e-4.
    """
    return [
        _base(
            tag=f"E2_roa{roa:g}",
            nc=5,
            order=16,
            r_over_a=roa,
            maxiter=500,
            ctol=1e-4,
            options={"inner_reduction": 10.0, "track_residual": True},
        )
        for roa in (2.0, 2.5, 3.0, 3.5, 4.0)
    ]


def E2b_coil_count():
    """Is our nc=3/4 work actually harder than their nc=5 (almost certainly).

    Fewer coils is strictly less freedom at fixed field error, so this
    establishes the penalty we pay for nc=3 and nc=4 and lets us report our
    existing results against the right baseline instead of against their nc=5
    numbers. Run at the paper's constraints so only coil count varies.
    """
    return [
        _base(
            tag=f"E2b_nc{nc}",
            nc=nc,
            order=16,
            r_over_a=3.0,
            maxiter=500,
            options={"inner_reduction": 10.0, "track_residual": True},
        )
        for nc in (3, 4, 5, 6)
    ]


# ---------------------------------------------------------------------------
# E3 -- auglag vs penalty, the actual claim being made
# ---------------------------------------------------------------------------
def E3_auglag_vs_penalty():
    """Does auglag-with-bounds beat weighted penalty without weight tuning.

    This is the claim the whole approach rests on, and the paper is the ideal
    foil: they got their front by RANDOM SAMPLING of penalty weights -- 4363
    cold starts, hyperparameters drawn from hand-tuned ranges, then filtered.
    That is precisely the cost auglag is supposed to remove.

    So the comparison worth making is not "our best point vs their best point"
    but COST TO REACH THE FRONT: how many runs, and how much weight tuning, to
    land a filter-passing coilset. Auglag gets 5 starts with no weight tuning;
    the penalty baseline gets the same 5 starts at three weight sets spanning an
    order of magnitude, standing in for a small random search.

    Read: fraction of runs passing `paper_filter`, and the spread of
    normalized_BdotN within each method. Auglag should win on run-to-run
    variance even where it ties on best point.
    """
    out = []
    for roa in (2.5, 3.0, 3.5):
        out.append(
            _base(
                tag=f"E3_auglag_roa{roa:g}",
                method="lsq-auglag",
                nc=5,
                r_over_a=roa,
                maxiter=500,
                options={"inner_reduction": 10.0, "track_residual": True},
            )
        )
    # penalty baseline: weights spanning 10x either side of the notebook's set
    wsets = {
        "lo": {
            "quadratic_flux": 40,
            "coil_coil": 100,
            "plasma_coil": 1,
            "curvature": 120,
            "length": 2,
            "linking": 1,
        },
        "mid": {
            "quadratic_flux": 400,
            "coil_coil": 1000,
            "plasma_coil": 10,
            "curvature": 1200,
            "length": 20,
            "linking": 10,
        },
        "hi": {
            "quadratic_flux": 4000,
            "coil_coil": 10000,
            "plasma_coil": 100,
            "curvature": 12000,
            "length": 200,
            "linking": 100,
        },
    }
    for roa in (2.5, 3.0, 3.5):
        for name, w in wsets.items():
            out.append(
                _base(
                    tag=f"E3_penalty_{name}_roa{roa:g}",
                    method="lsq-exact",
                    nc=5,
                    r_over_a=roa,
                    maxiter=500,
                    weights=w,
                )
            )
    return out


# ---------------------------------------------------------------------------
# E4 -- trace a front by moving BOUNDS (auglag's structural advantage)
# ---------------------------------------------------------------------------
def E4_bound_front():
    """Trace a front by scaling the constraint bounds, not by tuning weights.

    Under auglag a bound is a physical quantity with units, so a front can be
    traced by MOVING THE BOUND -- each point is a well-posed problem with a
    stated engineering limit, and every solution is feasible by construction.
    Under a penalty method the analogous sweep is over weights, where no single
    run is guaranteed to respect any limit and feasibility is only recovered by
    filtering afterwards (which is what discarded most of their 4363 runs).

    d_cs is the axis to sweep: on their published front it is the strongest
    single predictor of max force (R^2 = 0.926, Spearman +0.977, force ~
    64823*d_cs - 4488 N/m). So sweeping d_cs traces the force axis of their
    Figure 3 WITHOUT a force model -- which is what makes a comparison possible
    at all given we have no self-force objective. Their front spans
    d_cs = 0.178 to 0.271 m, so the scale factors cover it.

    Read: our (normalized_BdotN, d_cs) curve overlaid on their front. If ours
    lies on or below theirs in field error at matched d_cs, we are competitive
    on the trade-off they published, force model or not.
    """
    out = []
    # their front spans d_cs 0.178-0.271 m; paper bound is 0.166 m
    for s in (1.05, 1.20, 1.35, 1.50, 1.63):
        out.append(
            _base(
                tag=f"E4_dcs{s:g}",
                nc=5,
                r_over_a=3.0,
                maxiter=500,
                bound_scale={"d_cs": s},
                options={"inner_reduction": 10.0, "track_residual": True},
            )
        )
    # length is the other axis worth one pass: their cap is tight (5 m = 1.59
    # L0), and the notebook ran at 6*L0, ~3.8x looser.
    for s in (0.8, 1.0, 1.5, 2.0):
        out.append(
            _base(
                tag=f"E4_len{s:g}",
                nc=5,
                r_over_a=3.0,
                maxiter=500,
                bound_scale={"length": s},
                options={"inner_reduction": 10.0, "track_residual": True},
            )
        )
    return out


# ---------------------------------------------------------------------------
# E5 -- the C0 classes, once the above is trustworthy
# ---------------------------------------------------------------------------
def E5_representations():
    """Compare representations at the paper's constraints.

    The actual research interest (per the branch) is the piecewise-planar-arc
    C0 classes; the paper is only the yardstick that says our stage-2 setup is
    sound. This runs the smooth reps first as controls. `PiecewisePlanarArcCoil`
    /`PolarPlanarArcCoil` are NOT included yet -- `initialize_modular_coils`
    has no `to_*` conversion for them, so they need the projection path from the
    `desc-coil-init` workflow. Add them here once that produces a valid initial
    coilset.

    Read: field error at matched constraints across reps. The expected result is
    that C0 reps pay a field-error penalty for buildability; this quantifies it
    against a published smooth-coil front rather than against our own runs only.
    """
    return [
        _base(
            tag=f"E5_{rep}_N{n}",
            nc=5,
            rep=rep,
            order=n,
            r_over_a=3.0,
            maxiter=500,
            options={"inner_reduction": 10.0, "track_residual": True},
        )
        for rep, n in (
            ("FourierXYZ", 16),
            ("FourierXYZ", 10),
            ("FourierPlanar", 10),
            ("FourierPlanar", 5),
        )
    ]


EXPERIMENTS = {
    "E0": E0_current_degeneracy,
    "E1": E1_auglag_convergence,
    "E2": E2_matched_setup,
    "E2b": E2b_coil_count,
    "E3": E3_auglag_vs_penalty,
    "E4": E4_bound_front,
    "E5": E5_representations,
}


def main():
    """CLI entry point."""
    p = argparse.ArgumentParser(description="run a benchmark experiment")
    p.add_argument("--exp", required=True, choices=sorted(EXPERIMENTS))
    p.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--outdir", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--only", default=None, help="substring filter on tag, to run a single case"
    )
    args = p.parse_args()

    cfgs = EXPERIMENTS[args.exp]()
    if args.only:
        cfgs = [c for c in cfgs if args.only in c["tag"]]
    outdir = args.outdir or os.path.join("runs", args.exp)

    if args.dry_run:
        lim = rescale_to_eq(1.03067)  # precise_QA
        print(f"{args.exp}: {len(cfgs)} cases -> {outdir}")
        print(
            f"paper bounds at DESC R0=1.03067: "
            f"L<={lim['length_max']:.4f} m  k<={lim['kappa_max']:.4f} 1/m  "
            f"d_cc>={lim['d_cc_min']:.4f} m  d_cs>={lim['d_cs_min']:.4f} m"
        )
        for c in cfgs:
            bs = c.get("bound_scale", {})
            extra = f" bound_scale={bs}" if bs else ""
            print(
                f"  {c['tag']:28s} nc={c['nc']} rep={c['rep']} N={c['order']} "
                f"roa={c['r_over_a']} {c['method']} cur={c['current_mode']} "
                f"maxiter={c['maxiter']} ctol={c['ctol']:g}"
                f" opts={c.get('options', {})}{extra}"
            )
        return

    configure_device(args.device)
    if args.device == "gpu":
        from desc import set_device

        set_device("gpu")
    from bench.bench import run_case, summarize

    recs = []
    for i, c in enumerate(cfgs, 1):
        print(f"\n########## [{i}/{len(cfgs)}] {c['tag']} ##########", flush=True)
        try:
            rec = run_case(c, outdir=outdir)
            recs.append(rec)
            print(summarize(rec), flush=True)
        except Exception as e:  # keep the sweep alive; record the failure
            print(f"!! {c['tag']} FAILED: {type(e).__name__}: {e}", flush=True)
            with open(os.path.join(outdir, f"{c['tag']}.FAILED.json"), "w") as fh:
                json.dump(
                    {
                        "tag": c["tag"],
                        "error": f"{type(e).__name__}: {e}",
                        "config": {k: v for k, v in c.items()},
                    },
                    fh,
                    indent=1,
                )
    print(f"\n{len(recs)}/{len(cfgs)} cases completed -> {outdir}")


if __name__ == "__main__":
    main()
