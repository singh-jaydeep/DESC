"""The precise_QA error-vs-length sweep: the final config set, as DATA.

    python tools/sweep_configs.py                 # the grid, with status and cost
    python tools/sweep_configs.py --json OUT      # machine-readable config, one row per run
    python tools/sweep_configs.py --commands      # shell commands, in dependency order
    python tools/sweep_configs.py --commands --smoke   # one cheap point per representation

Replaces `sweep_length.py`, which is stale in four ways (HANDOFF_sweep_configs sec 3.4):
its multiplier list predates the current length axis, it applied `dcc_backoff=True`
unconditionally (now wrong for the smooth representations), and it emitted neither
`--ctol` nor `--dcc-signed`.

WHAT A POINT ON THIS FRONT MEANS. Relaxing `--length-max` does not solve the same
problem better; it solves a DIFFERENT problem. Each point is "the achievable field error
at this length budget", and the front is the trade between coil length and field error.
It is NOT an optimizer comparison and must not be reported as one.

THE ONE TRAP THAT HAS BITTEN TWICE. `--curv-ms-max` is the ENABLE SWITCH for the
mean-square curvature constraint, not merely its bound. Omit it and `curvature_ms` is
silently absent from `terms=[...]` -- that is how `al_planar_N14` came to violate it by
10.9% and be void. Every command this module emits carries it, for every representation.

REPORTING TRAPS, both live:

1. `verify` adjudicates feasibility against the UNBACKED-OFF paper bounds always, so
   every run with `--length-max` above 5.153350 is recorded `feasible: NO / length`.
   Correct and intended; it makes the generated table's `feasible` column meaningless
   for this sweep. Adjudicate the front against `length_max_override`.

2. `--length-max` re-anchors nothing else. The other four bounds stay at their paper
   values, so as length is relaxed the ACTIVE SET changes hands. Which constraint binds
   at each point is a RESULT of this sweep, not an aside; record it per run.
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------------------
# Bounds. Hurwitz Table 1 rescaled to precise_QA, R0 = 1.030670 (SWEEP_BLUEPRINT sec 4).
# kappa_ms rescales as /f^2, NOT /f -- it is (1/L)*int kappa^2 dl with no square root,
# so it is 1/m^2 and a single power of f left the bound 3.07% too loose. 5.821456 is the
# WRONG value and still appears in `diag_ymask.py --curv-ms-max` help text.
# --------------------------------------------------------------------------------------
L_PAPER = 5.153350
KAPPA_BOUND = 11.642912
KAPPA_MS_BOUND = 5.648225
DCC_BOUND = 0.08554561095366726
DPC_BOUND = 0.171091

# Enforcement grid for CoilSetMinDistance; the sagitta backoff below is O(1/N^2) in it.
DIST_GRID_N = 150

# --------------------------------------------------------------------------------------
# The length axis. Five multipliers, chosen 2026-09-10.
#
# FLOOR. Every representation cold-starts at the exact circle's circumference,
# 2*pi*3.3*0.1718 = 3.5617 m = 0.691 x the paper bound, so a multiplier much below 0.8
# leaves the start sitting on its own bound and the point is not a real optimization.
# 0.8 keeps 15.7% headroom (0.75, the old floor, kept 8.5%).
#
# CEILING. Measured saturation is 8.7777 m (`len_arcB5_L10m` was given a 10 m budget and
# stopped at 8.7777, L_max margin 0.878). x1.6 = 8.2454 m is 6.1% BELOW that, so this
# axis stops short of the knee deliberately: every point on it is one where length still
# binds. The demonstration that the axis saturates rests on `len_arcB5_L10m`, which is
# off-grid, arc B=5 only, and stays in the cross-check table for exactly that reason.
# --------------------------------------------------------------------------------------
MULTIPLIERS = [0.8, 1.0, 1.2, 1.4, 1.6]

# --------------------------------------------------------------------------------------
# Representations. Seven, chosen 2026-09-10: the arc facet ladder B=2/3/5/7 spanning
# 14-49 DOF/coil, planar N=7 as the 1-facet anchor (8o measured it equal to N=14 cold
# and
# cheaper), and FourierXYZ at two resolutions as the unfaceted control.
#
# `minutes` is per run at maxiter=800. MEASURED entries come from SWEEP_RESULTS sec 4;
# EXTRAPOLATED ones scale those by DOF/coil and are calibrated by `--smoke`.
# --------------------------------------------------------------------------------------
# `arcB2` carries `--arc-bulge 0`. MEASURED 2026-09-10 (CONVENTIONS 8q): the fitted B=2
# start exceeds kappa_MS by 40.4% and no r/a or M rescues it, because `from_values` fits
# the transverse deviation as a graph over the CHORD and at B=2 that chord is a DIAMETER
# whose target is a semicircle -- a sqrt profile with vertical endpoint tangents. The
# truncated sine fit ripples, and kappa is a second derivative, so more modes make it
# worse. `--arc-bulge 0` keeps the fitted fundamental and drops the ripple modes.
REPS = [
    # key        flags                                        DOF  min  basis   smooth
    ("planarN7", ["--rep", "planar", "--fp-N", "7"], 21, 70, "extrap", True),
    (
        "arcB2",
        ["--rep", "piecewise", "-B", "2", "-M", "3", "--arc-bulge", "0"],
        14,
        34,
        "extrap",
        False,
    ),
    ("arcB3", ["--rep", "piecewise", "-B", "3", "-M", "3"], 21, 40, "measured", False),
    ("arcB5", ["--rep", "piecewise", "-B", "5", "-M", "3"], 35, 51, "measured", False),
    ("arcB7", ["--rep", "piecewise", "-B", "7", "-M", "3"], 49, 62, "extrap", False),
    ("xyzN4", ["--rep", "xyz", "--xyz-N", "4"], 27, 67, "measured", True),
    ("xyzN6", ["--rep", "xyz", "--xyz-N", "6"], 39, 85, "extrap", True),
]

# The cold start of each representation, as `_paper_start` kwargs. This is the single
# source of truth for the START, used by `dump_initial_starts.py`
# as well as by the commands below; `_check_specs()` asserts it agrees with `flags`.
#
# arcB2 is the one deviation from uniform staging, and it is forced (CONVENTIONS 8q):
# its fitted start exceeds kappa_MS by 40.4% and the single-mode fix costs plasma
# clearance, which only r/a buys back with any margin. MEASURED with `--arc-bulge 0`:
# r/a 3.50 -> d_pc 0.982 (infeasible), 3.55 -> 1.026, 3.60 -> 1.069, 3.65 -> 1.113,
# against d_cc 1.504 / 1.442 / 1.380 / 1.318. 3.60 is the roughly equal-margin point.
# CARRY THIS CAVEAT: B=2 sits at a different r/a from the other six, so the facet
# ladder is not staged uniformly and B=2-vs-B=3 mixes two changes.
START_SPEC = {
    "planarN7": dict(rep="planar", fp_N=7, r_over_a=3.3),
    "arcB2": dict(rep="piecewise", B=2, M=3, arc_bulge=0.0, r_over_a=3.60),
    "arcB3": dict(rep="piecewise", B=3, M=3, r_over_a=3.3),
    "arcB5": dict(rep="piecewise", B=5, M=3, r_over_a=3.3),
    "arcB7": dict(rep="piecewise", B=7, M=3, r_over_a=3.3),
    "xyzN4": dict(rep="xyz", xyz_N=4, r_over_a=3.3),
    "xyzN6": dict(rep="xyz", xyz_N=6, r_over_a=3.3),
}


def _check_specs():
    """START_SPEC and REPS must describe the same start, or the pins mean nothing."""
    for key, flags, *_ in REPS:
        spec = START_SPEC[key]
        f = dict(zip(flags[::2], flags[1::2]))
        assert f["--rep"] == spec["rep"], key
        for flag, k in (
            ("-B", "B"),
            ("-M", "M"),
            ("--fp-N", "fp_N"),
            ("--xyz-N", "xyz_N"),
            ("--arc-bulge", "arc_bulge"),
        ):
            if flag in f:
                assert float(f[flag]) == float(spec[k]), f"{key}: {flag}"
            else:
                assert k not in spec, f"{key}: {k} in START_SPEC but not in flags"


_check_specs()


# --------------------------------------------------------------------------------------
# The recipe. The sec 8i four-flag core plus --ctol, unchanged since paper_al_v13_chord.
# --------------------------------------------------------------------------------------
R_OVER_A_DEFAULT = 3.3  # equal-margin point of the nc=4 feasible window [3.151, 3.401]

# Device. GPU is the default: the sweep runs on a box where only one process fits in
# host RAM at a time, so the ~2x the campaign measured from GPU is worth having even
# though this workload is not GPU-bound. Override with `--device cpu` on this module;
# it rewrites every emitted command.
#
# THREE THINGS TO KNOW BEFORE SWITCHING TO GPU:
#  1. The `systemd-run MemoryMax` cap in the runner bounds HOST RAM, not VRAM. Cap the
#     device separately: XLA_PYTHON_CLIENT_PREALLOCATE=false, or
#     XLA_PYTHON_CLIENT_MEM_FRACTION=<f>. Without it JAX grabs most of the card up
# front.
#  2. `--dcc-signed` computes a linking number inside an AD trace, and the memory there
# is
#     the (ncoil, ncol, nnode, nnode, 3) broadcast -- O(N^2). At 801 nodes that is 3.9
# GB
#     and OOMs; the guard uses a coarse grid for exactly this reason. On an 8 GB card
# the
#     warm-start rungs are the ones that will fail first, not the cold anchors.
#  3. Results will not be bit-identical to CPU. That is fine for a sweep run entirely on
#     one device, and NOT fine for mixing devices within a continuation chain.
DEVICE = "gpu"

RECIPE = [
    "--convention",
    "paper",
    "--device",
    DEVICE,
    "--nc",
    "4",
    "--drop",
    "linking_number",  # 8i: unreadable on C0 arcs. NOT subsumed by d_cc
    "--distance-method",
    "segment",  # 8i: "point" manufactures a barrier
    "--max-trust-radius",
    "0.5",  # 8i: caps one oversized step. Does NOT guard topology
    "--fit-method",
    "chord",  # 4.1.1: "parameter" is a silent fit bug
    "--ctol",
    "1e-4",  # 8o: 1e-6 ratchets mu to 1.9e+05 off the noise floor
    "--max-inner-iter",
    "100",  # DO NOT TUNE -- HANDOFF_sweep_configs sec 9
    "--maxiter",
    "800",  # 500 leaves every relaxed-length run still descending
    "--check-every",
    "50",
]

MEM_CAP = "12G"  # WSL2 has no OOM killer; a cgroup cap is the only backstop


def dcc_backoff(length_bound, n=DIST_GRID_N):
    """Sagitta-derived d_cc backoff in METRES. SWEEP_BLUEPRINT sec 5.1.

        1.4 * kappa_bound * (L_bound / N)^2 / 8      (one sagitta, x1.4 safety)

    The segment-segment minimum runs on a polyline that cuts INSIDE the true curve, so a
    closest approach falling between nodes on a bulging stretch reports more clearance
    than exists. The sagitta of a chord subtending arclength s on a curve of curvature
    kappa is ~kappa*s^2/8, which bounds that over-report per coil.

    It is computable A PRIORI from the bounds the design must satisfy anyway -- it needs
    only `kappa_bound`, `L_bound` and `N`, never the answer. VALIDATED ONCE, at x1.5:
    predicted 4.52%, measured requirement 4.56% -- agreement to 0.04 percentage points.

    It is a BOUND, attained only when the closest approach coincides with a
    high-curvature stretch, and over-predicts by up to 24x when it does not. Sized here
    on the length BUDGET, which is conservative whenever the design does not spend it
    (at the 10 m point the budget was 10 m, the design used 8.78 m, and d_cc finished
    6.1% clear). To recover that, re-run `dcc_backoff` on the length ACTUALLY REACHED
    and
    iterate once; conservatism costs performance, never correctness.
    """
    return 1.4 * KAPPA_BOUND * (length_bound / n) ** 2 / 8


# Fixed backoffs on the two curvature bounds, ARC family only. Unlike d_cc these have no
# per-point formula: they were measured at the sweep's worst case (x1.5) and consume
# 0.16% and 0.19% of the 0.5% and 1.0% applied, so they are conservative at every milder
# point on this axis. SWEEP_BLUEPRINT sec 5.3.1.
KAPPA_BACKOFF = 0.005
KAPPA_MS_BACKOFF = 0.010


def enforced_bounds(rep_smooth, length_bound):
    """The bounds a run is TOLD, as opposed to the paper bounds it is JUDGED against.

    SCOPE, and this is the whole of it: the sagitta rule is calibrated on the C0 ARC
    family, where the curve is piecewise straight and the segment approximation really
    is that bad (3-11%). MEASURED on five converged FourierPlanar coilsets, the
    enforcement-grid error is 0.010% / 0.082% / 0.109% / 0.116% / 0.231% -- about 25x
    smaller than the 2.81% the rule prescribes at the paper length. So the smooth
    representations enforce the PAPER bounds directly and report compliance to ~0.2%.
    The user accepted this explicitly for the exploratory stage; a different
    representation must re-measure before inheriting it (SWEEP_BLUEPRINT sec 10.4).

    Returns the three overridable bounds. `curv_ms_max` is never None -- it is the
    ENABLE SWITCH for the constraint, so omitting it drops `curvature_ms` from the
    problem silently.
    """
    if rep_smooth:
        return dict(dcc_min=None, curv_max=None, curv_ms_max=KAPPA_MS_BOUND)
    return dict(
        dcc_min=DCC_BOUND + dcc_backoff(length_bound),
        curv_max=KAPPA_BOUND * (1 - KAPPA_BACKOFF),
        curv_ms_max=KAPPA_MS_BOUND * (1 - KAPPA_MS_BACKOFF),
    )


# --------------------------------------------------------------------------------------
# Chain topology, expressed as DATA rather than implied by execution order.
#
# CONVENTIONS 8p settled the start strategy: cold multistart does NOT replace
# continuation. Three perturbation classes failed to reach the warm start's basin (best
# valid cold point 3.3929e-02 against 3.1021e-02, 9.4% short), and the aimed run drove
# d_cc to 0.5975 of its bound, stayed UNLINKED, and won nothing -- so it is the
# topological CROSSING, not the clearance violation, that relocates the basin. A cold
# start cannot buy it. Continuation chains with `--dcc-signed` it is.
#
# TWO RUNGS MAXIMUM (8o): mu runs 1.0e+03 / 1.8e+04 / 1.9e+05 on successive rungs
# against
# 14.19 for a cold run, and by the third the AL clamps rather than optimizes. Five
# length
# points therefore need TWO cold anchors, not one -- x1.6 is four rungs from x1.0.
#
# The second anchor is placed at x1.4 rather than x1.6 on purpose. If the top of the
# axis
# were reported from a cold run while every other point is warm, the front would appear
# to flatten at x1.6 for a reason that is pure start strategy (9-21%) rather than
# geometry -- and saturation is exactly what this axis exists to measure. Anchoring cold
# at x1.4 and continuing to x1.6 keeps every REPORTED point warm-or-anchor, and the cold
# x1.4 doubles as the cold/warm control the campaign has only ever had at one length on
# one representation (8m's 10.5%, at x1.25 on arc B=5).
#
# Reported points are 0/1/1/2/1 rungs from a cold start. `role="control"` rows are run
# but are NOT sweep points.
# --------------------------------------------------------------------------------------
CHAIN = [
    # mult  start ("cold", or the multiplier to continue FROM)  role
    (1.0, "cold", "grid"),  # anchor
    (0.8, 1.0, "grid"),  # rung 1, tight side
    (1.2, 1.0, "grid"),  # rung 1
    (1.4, 1.2, "grid"),  # rung 2 -- the cap
    (1.4, "cold", "control"),  # second anchor + the cold/warm control at x1.4
    (1.6, "cold@1.4", "grid"),  # rung 1 from the cold x1.4 anchor
]


def tag(rep, mult, cold=False):
    """Run tag. `_c` marks a cold run at a multiplier that also has a warm one."""
    return f"sw_{rep}_L{('%g' % mult).replace('.', 'p')}" + ("_c" if cold else "")


def points():
    """Every run in the sweep, fully resolved, in dependency order.

    One row per RUN. `depends_on` is the tag whose h5 this run warm-starts from, or None
    for a cold start; the runner needs no other ordering information.
    """
    out = []
    for key, flags, dof, minutes, basis, smooth in REPS:
        for mult, start, role in CHAIN:
            cold = start == "cold"
            L = mult * L_PAPER
            b = enforced_bounds(smooth, L)
            if cold:
                dep = None
            elif start == "cold@1.4":
                dep = tag(key, 1.4, cold=True)
            else:
                dep = tag(key, start)
            t = tag(key, mult, cold=cold and role == "control")
            out.append(
                dict(
                    id=t,
                    rep=key,
                    rep_flags=flags,
                    dof=dof,
                    smooth=smooth,
                    r_over_a=START_SPEC[key]["r_over_a"],
                    mult=mult,
                    length_max=L,
                    role=role,
                    depends_on=dep,
                    rung=0 if cold else None,
                    minutes=minutes,
                    minutes_basis=basis,
                    enforced=b,
                    out_dir=f"runs/{t}",
                )
            )
    # rung depth, for the two-rung cap assertion
    by = {p["id"]: p for p in out}
    for p in out:
        d, n = p["depends_on"], 0
        while d is not None:
            n += 1
            d = by[d]["depends_on"]
        p["rung"] = n
        assert n <= 2, f"{p['id']} is {n} rungs from a cold start; the cap is 2"
    return out


def argv(p):
    """The full argument vector for one run."""
    c = list(RECIPE) + ["--r-over-a", f"{p['r_over_a']:g}"] + list(p["rep_flags"])
    if p.get("smoke_iters"):
        # override, not append -- argparse takes the LAST occurrence, but a reader of
        # the command should not have to know that
        for flag, val in (
            ("--maxiter", str(p["smoke_iters"])),
            ("--check-every", "10"),
        ):
            c[c.index(flag) + 1] = val
    c += ["--length-max", f"{p['length_max']:.6f}"]
    e = p["enforced"]
    # ALWAYS emitted: --curv-ms-max is the enable switch, not just a bound.
    c += ["--curv-ms-max", f"{e['curv_ms_max']:.6f}"]
    if e["curv_max"] is not None:
        c += ["--curv-max", f"{e['curv_max']:.6f}"]
    if e["dcc_min"] is not None:
        c += ["--dcc-min", f"{e['dcc_min']:.8f}"]
    if p["depends_on"] is not None:
        # REQUIRED on any warm start (8o): unguarded continuation linked 2/2, guarded
        # 3/3 unlinked. It does not prevent the crossing, it makes it RECOVERABLE.
        c += [
            "--dcc-signed",
            "--start-from",
            f"runs/{p['depends_on']}/{p['depends_on']}.h5",
        ]
    c += ["--out", p["out_dir"], "--tag", p["id"]]
    return c


def command(p):
    """One run's shell command, memory-capped and logged."""
    pre = [
        "systemd-run",
        "--user",
        "--scope",
        "-q",
        "-p",
        f"MemoryMax={MEM_CAP}",
        "-p",
        "MemorySwapMax=0",
        "python",
        "-u",
        "tools/diag_ymask.py",
    ]
    return " ".join(pre + argv(p)) + f" > runs/{p['id']}.log 2>&1"


def adjudicate_command(p):
    """The post-solve adjudication. NEVER run concurrently with a solve.

    Two runs died this session to `verify` running alongside one on this 15 GB machine
    (HANDOFF_sweep_configs sec 5.4). It is a separate serial step, not a background
    task.
    """
    return (
        f"python -u tools/adjudicate.py {p['out_dir']} "
        f"--length-max {p['length_max']:.6f} "
        f">> runs/{p['id']}.log 2>&1"
    )


def runlist_md():
    """The run list as markdown: every run, its start, and its length bound.

    Generated so it cannot drift from the commands. Donors always precede their
    dependants, so the `start` column can be read top to bottom.
    """
    L = [
        "# Sweep run list",
        "",
        f"{len(points())} runs — {len(REPS)} representations x {len(MULTIPLIERS)} "
        f"length bounds, plus one cold control per representation at x1.4.",
        "",
        f"Length bounds are relative to the paper value **{L_PAPER:.6f} m**. "
        "`start` is either a cold build or the run whose solved coilset is "
        "transplanted onto it.",
        "",
        "| run | representation | DOF/coil | start | xL | L (m) |",
        "|---|---|---|---|---|---|",
    ]
    for p in points():
        start = "cold" if p["depends_on"] is None else f"← `{p['depends_on']}`"
        L.append(
            f"| `{p['id']}` | {p['rep']} | {p['dof']} | {start} | "
            f"{p['mult']:g} | {p['length_max']:.4f} |"
        )
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--commands", action="store_true", help="emit shell commands")
    ap.add_argument(
        "--json",
        metavar="OUT",
        nargs="?",
        const="-",
        help="emit the machine-readable config (default stdout)",
    )
    ap.add_argument(
        "--runlist",
        metavar="OUT",
        nargs="?",
        const="-",
        help="emit the run list as markdown (default stdout)",
    )
    ap.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default=DEVICE,
        help="device for every emitted command (default %(default)s). "
        "See the DEVICE note above before choosing gpu.",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="one cheap point per representation, plumbing only",
    )
    ap.add_argument(
        "--smoke-iters",
        type=int,
        default=30,
        dest="smoke_iters",
        help="maxiter for --smoke (default 30). Enough to exercise the "
        "start, the constraint build and the record path; nowhere "
        "near enough to converge, and not meant to.",
    )
    a = ap.parse_args()
    if a.device != RECIPE[RECIPE.index("--device") + 1]:
        RECIPE[RECIPE.index("--device") + 1] = a.device

    if a.runlist:
        md = runlist_md()
        if a.runlist == "-":
            print(md)
        else:
            open(a.runlist, "w").write(md)
            print(f"wrote {a.runlist}")
        return

    pts = points()
    if a.smoke:
        # The x1.0 cold anchor of each representation at a short maxiter. This is the
        # plumbing check the handoff scopes IN: it exercises every representation's
        # start, both backoff regimes, the enable switch and the record path, without
        # running the sweep. Tagged separately so it can never collide with a real run.
        pts = [dict(p) for p in pts if p["mult"] == 1.0 and p["depends_on"] is None]
        for p in pts:
            p["id"] = "smoke_" + p["id"]
            p["out_dir"] = f"runs/{p['id']}"
            p["minutes"] = max(1, round(p["minutes"] * a.smoke_iters / 800))
            p["smoke_iters"] = a.smoke_iters

    if a.json:
        blob = dict(
            generator="tools/sweep_configs.py",
            equilibrium="precise_QA",
            nc=4,
            r_over_a=3.3,
            paper_bounds=dict(
                length_max=L_PAPER,
                kappa_max=KAPPA_BOUND,
                kappa_ms_max=KAPPA_MS_BOUND,
                d_cc_min=DCC_BOUND,
                d_pc_min=DPC_BOUND,
            ),
            recipe=RECIPE,
            multipliers=MULTIPLIERS,
            runs=[dict(p, argv=argv(p)) for p in pts],
        )
        s = json.dumps(blob, indent=1)
        if a.json == "-":
            print(s)
        else:
            open(a.json, "w").write(s + "\n")
            print(f"wrote {a.json}: {len(pts)} runs")
        return

    if a.commands:
        for p in pts:
            print(f"# {p['id']}  ({p['role']}, rung {p['rung']}, ~{p['minutes']} min)")
            print(command(p))
            print(adjudicate_command(p))
            print()
        return

    hdr = (
        f"{'run':22s} {'rep':10s} {'xL':>5} {'L (m)':>8} {'DOF':>4} {'rung':>4} "
        f"{'role':8s} {'d_cc enf':>10} {'start':22s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for p in pts:
        e = p["enforced"]
        d = f"{e['dcc_min']:.6f}" if e["dcc_min"] else "paper"
        print(
            f"{p['id']:22s} {p['rep']:10s} {p['mult']:5g} {p['length_max']:8.4f} "
            f"{p['dof']:4d} {p['rung']:4d} {p['role']:8s} {d:>10} "
            f"{p['depends_on'] or '(cold)':22s}"
        )

    grid = [p for p in pts if p["role"] == "grid"]
    tot = sum(p["minutes"] for p in pts)
    print(
        f"\n{len(pts)} runs ({len(grid)} sweep points + {len(pts)-len(grid)} controls)"
        f" over {len(REPS)} representations x {len(MULTIPLIERS)} lengths"
    )
    print(f"{tot} min = {tot/60:.1f} h sequential")
    per = {}
    for p in pts:
        per[p["rep"]] = per.get(p["rep"], 0) + p["minutes"]
    print(
        f"one worker per representation ({len(REPS)}-wide): "
        f"{max(per.values())} min = {max(per.values())/60:.1f} h wall"
    )
    print(
        "\nNOTE 7-wide does NOT fit this machine: MemoryMax=12G per run against 15 GB "
        "total.\n     Locally the sweep is sequential; the width needs the CPU-container "
        "target\n     of SWEEP_BLUEPRINT sec 8.3."
    )


if __name__ == "__main__":
    main()
