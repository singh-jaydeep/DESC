"""Save the COLD START coilset of every sweep configuration, and adjudicate each one.

    python tools/dump_initial_starts.py "initial starts"    # write
    python tools/dump_initial_starts.py "initial starts" --check   # verify

Writes, into the target directory:

* `start_<rep>.h5` -- one per representation, the exact coilset the sweep's cold anchors
  begin from. Loadable with `desc.io.load`.
* `README.md` -- the feasibility table below, so the numbers travel with the files.
* `starts.json` -- the same, machine-readable, with each start's SHA-256.

`--check` rebuilds every start and compares it to the recorded `starts.json` instead of
writing, exiting non-zero on any difference. That closes CONVENTIONS sec 9's
longest-open
item, though not the way it is worded there. It reads "the start coilsets are not in
git,
... they are regenerable from (nc, r_over_a, B, M), which is the real fix -- but the
generator must be committed and pinned." THE GENERATOR IS ALREADY COMMITTED: under
`--convention paper` the start is BUILT, never loaded (`diag_ymask._paper_start` calls
`initialize_modular_coils` on `desc.examples.get("precise_QA")`), and the h5-loading
path
is the legacy `--convention bundle|c0` one the sweep never touches. What was missing is
the other half -- nothing would DETECT a silent change. `initialize_modular_coils` and
`from_values` are live DESC code on a branch that has already changed the curve classes
twice (`6691e5a14` NaN gradient at kappa=0, `31e5b7f35` frozen planar-arc frame), and a
change there moves every point on the front with nothing in the record to show it.

WHAT THESE ARE. A sweep point's start does NOT depend on its length bound -- `--length-max`
moves a constraint, not the geometry -- so the 42 runs draw on only SEVEN distinct cold
starts, one per representation. Both cold anchors of a chain (x1.0 and x1.4) begin from
the identical coilset; every other point is warm from a solved neighbour.

EVERY START HERE MUST BE FEASIBLE against the paper bounds, and the script exits
non-zero
if one is not. That is not housekeeping. A cold start outside an enforced bound is the
`planar_N14_cert` configuration: the AL's fresh penalty phase begins with no slack
anywhere, and MEASURED, that run crossed into a linked topology between iterations 15
and
20 while every local signal said the geometry was improving (CONVENTIONS 8o). Two points
in this campaign were banked as improvements that turned out to BE the linking. So a
start
that violates a bound is not a slow start, it is a trap.

Length is exempt from the check in one direction only: a start is far SHORTER than the
bound (the circle is x0.69), which is fine and expected.
"""

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bundle  # noqa: E402,F401  -- MUST precede any `import desc`

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from sweep_configs import REPS, START_SPEC  # noqa: E402


def build(key):
    """The cold start of one representation, via the same entry point the sweep uses."""
    from diag_ymask import _paper_start

    from desc.examples import get

    spec = dict(START_SPEC[key])
    eq = get("precise_QA")
    cs = _paper_start(
        eq,
        rep=spec["rep"],
        nc=4,
        B=spec.get("B", 5),
        M=spec.get("M", 3),
        r_over_a=spec["r_over_a"],
        fp_N=spec.get("fp_N", 14),
        fit_method="chord",
        xyz_N=spec.get("xyz_N", 6),
        arc_bulge=spec.get("arc_bulge"),
    )
    return eq, cs, spec


def digest(coilset):
    """SHA-256 of the parameter vector -- the pin, so a silent change is detectable."""
    import numpy as np
    from bundle_probe import _iter_unique

    h = hashlib.sha256()
    for c in _iter_unique(coilset):
        for k in sorted(c.params_dict):
            v = np.ascontiguousarray(np.asarray(c.params_dict[k], dtype=np.float64))
            h.update(k.encode())
            h.update(v.tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("outdir", nargs="?", default="initial starts")
    ap.add_argument("--verify-n", type=int, default=400, dest="verify_n")
    ap.add_argument(
        "--check",
        action="store_true",
        help="rebuild and compare against the recorded starts.json instead "
        "of writing; non-zero exit on any difference",
    )
    a = ap.parse_args()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    manifest = os.path.join(a.outdir, "starts.json")
    if a.check:
        if not os.path.exists(manifest):
            raise SystemExit(f"no manifest at {manifest}; run without --check first")
        pinned = {r["rep"]: r for r in json.load(open(manifest))["starts"]}
    else:
        os.makedirs(a.outdir, exist_ok=True)

    from verify import hard_geometry, paper_bounds

    rows, bad = [], []
    dof = {r[0]: r[2] for r in REPS}
    lim = None
    for key, *_ in REPS:
        eq, cs, spec = build(key)
        if lim is None:
            lim = paper_bounds(eq)
        m = hard_geometry(cs, eq, n=a.verify_n)
        q = dict(
            length=float(m["max_length"]),
            kappa=float(m["max_max_kappa"]),
            kappa_MS=float(m["max_kappa_MS"]),
            d_cc=float(m["coil_coil_distance"]),
            d_pc=float(m["coil_surface_distance"]),
        )
        margins = dict(
            length=q["length"] / lim["length_max"],
            kappa=q["kappa"] / lim["kappa_max"],
            kappa_MS=q["kappa_MS"] / lim["kappa_ms_max"],
            d_cc=q["d_cc"] / lim["d_cc_min"],
            d_pc=q["d_pc"] / lim["d_pc_min"],
        )
        failed = [k for k in ("length", "kappa", "kappa_MS") if margins[k] > 1.0]
        failed += [k for k in ("d_cc", "d_pc") if margins[k] < 1.0]
        # a linked START would be pathological, but it costs nothing to check
        n_linked = int(m["n_linked_pairs"])
        if n_linked:
            failed.append("linked")
        path = os.path.join(a.outdir, f"start_{key}.h5")
        sha = digest(cs)
        if not a.check:
            cs.save(path)
        rows.append(
            dict(
                rep=key,
                dof=dof[key],
                spec=spec,
                file=os.path.basename(path),
                sha256=sha,
                quantities=q,
                margins=margins,
                n_linked_pairs=n_linked,
                feasible=not failed,
                failed=failed,
            )
        )
        print(
            f"{key:10s} r/a={spec['r_over_a']:.2f} L={q['length']:.4f} "
            f"kappa={q['kappa']:6.4f} kMS={q['kappa_MS']:6.4f}({margins['kappa_MS']:.3f}) "
            f"d_cc={margins['d_cc']:.3f} d_pc={margins['d_pc']:.3f}  "
            f"{'FEASIBLE' if not failed else 'INFEASIBLE: ' + ','.join(failed)}",
            flush=True,
        )
        if failed:
            bad.append(key)
        if a.check:
            old = pinned.get(key)
            if old is None:
                bad.append(f"{key}: not in the manifest")
            elif old["sha256"] != sha:
                bad.append(f"{key}: SHA CHANGED {old['sha256'][:16]} -> {sha[:16]}")
                for qk in q:
                    if abs(old["quantities"][qk] - q[qk]) > 1e-10 * max(
                        abs(q[qk]), 1e-30
                    ):
                        bad.append(
                            f"    {qk}: {old['quantities'][qk]:.9g} -> {q[qk]:.9g}"
                        )

    if a.check:
        if bad:
            print(
                "\nSTARTS DO NOT MATCH THE MANIFEST -- the sweep's starts have moved."
            )
            print(
                "Every point on the front is affected; do not merge results across it."
            )
            for line in bad:
                print("  " + str(line))
            raise SystemExit(1)
        print(f"\nall {len(rows)} starts match {manifest}")
        return

    json.dump(
        dict(
            equilibrium="precise_QA",
            nc=4,
            fit_method="chord",
            verify_grid_N=a.verify_n,
            paper_bounds=lim,
            starts=rows,
        ),
        open(manifest, "w"),
        indent=1,
        default=float,
    )

    L = [
        "# Initial cold starts for the 2026-09-10 sweep",
        "",
        "One coilset per representation -- the exact geometry the sweep's cold anchors",
        "begin from. Generated by `python tools/dump_initial_starts.py`; the grid itself is",
        "`c0/sweep_configs.py` and the board is `c0/SWEEP_RESULTS.md`.",
        "",
        "A point's start does not depend on its length bound (`--length-max` moves a",
        "constraint, not the geometry), so the 42 runs draw on only these seven starts.",
        "Both cold anchors of a chain (x1.0 and x1.4) begin from the identical coilset.",
        "",
        "Load one with:",
        "",
        "```python",
        "from desc.io import load",
        'cs = load("initial starts/start_arcB5.h5")',
        "```",
        "",
        "## Feasibility against the paper bounds",
        "",
        "Adjudicated at `VERIFY_N=400`. Margins are value/bound for the upper bounds",
        "(length, kappa, kappa_MS) and value/bound for the lower ones (d_cc, d_pc), so",
        "**every entry below 1.000 on an upper bound and above 1.000 on a lower bound is",
        "satisfied**. All seven are feasible; the script fails if one is not.",
        "",
        "| rep | DOF/coil | r/a | length (m) | kappa | kappa_MS | d_cc | d_pc | verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        mg, q = r["margins"], r["quantities"]
        L.append(
            f"| `{r['rep']}` | {r['dof']} | {r['spec']['r_over_a']:.2f} | "
            f"{q['length']:.4f} ({mg['length']:.3f}) | {q['kappa']:.4f} ({mg['kappa']:.3f}) | "
            f"{q['kappa_MS']:.4f} ({mg['kappa_MS']:.3f}) | {q['d_cc']:.5f} ({mg['d_cc']:.3f}) | "
            f"{q['d_pc']:.5f} ({mg['d_pc']:.3f}) | "
            f"{'OK' if r['feasible'] else '**' + ','.join(r['failed']) + '**'} |"
        )
    L += [
        "",
        f"Paper bounds: length <= {lim['length_max']:.6f} m, "
        f"kappa <= {lim['kappa_max']:.6f} 1/m, "
        f"kappa_MS <= {lim['kappa_ms_max']:.6f} 1/m^2, "
        f"d_cc >= {lim['d_cc_min']:.6f} m, d_pc >= {lim['d_pc_min']:.6f} m.",
        "",
        "**Why feasibility of the START matters.** A cold start outside an enforced",
        "bound is the `planar_N14_cert` configuration: the AL's fresh penalty phase",
        "begins with no slack anywhere, and that run crossed into a linked topology",
        "between iterations 15 and 20 while every local signal said the geometry was",
        "improving (CONVENTIONS 8o). Two points in this campaign were banked as",
        "improvements that turned out to BE the linking. A start that violates a bound",
        "is not a slow start, it is a trap.",
        "",
        "Length runs far *under* its bound at every start (the circle is ~0.69x), which",
        "is expected -- length is the swept axis and the optimizer grows into it.",
        "",
        "## Two things to look at",
        "",
        "**`arcB2` is staged differently, and this is the one caveat in the set.** It",
        "sits at r/a=3.60 where the other six sit at 3.30, and it carries",
        "`--arc-bulge 0`. Both are forced (CONVENTIONS 8q): `from_values` fits each",
        "arc as a transverse graph over its CHORD, and at B=2 that chord is a DIAMETER",
        "whose target profile is a semicircle -- a sqrt with vertical endpoint tangents",
        "that a truncated sine series cannot hold. The fit ripples, and since kappa is a",
        "second derivative the ripple dominates it: the fitted B=2 start exceeds",
        "kappa_MS by 40.4%, and MORE modes make it worse (7.93 at M=3, 27.14 at M=10).",
        "`--arc-bulge 0` keeps the fitted fundamental and drops the ripple modes, which",
        "more than halves kappa_MS -- but the ripple was also holding the plasma off, so",
        "d_pc then needs r/a to buy it back. **Consequence for the sweep: B=2-vs-B=3",
        "mixes two changes, so the bottom rung of the facet ladder is not a clean",
        "comparison with the rest of it.**",
        "",
        "**The four arc starts are not the same shape as the three smooth ones.**",
        "`planarN7`, `xyzN4` and `xyzN6` are all built from the identical circular",
        "modular coilset and differ only in parameterization, so their geometry rows are",
        "identical. The arcs are fits to that same circle and differ from it by the",
        "fit's own error, which is what the B=2 story above is about.",
        "",
    ]
    open(os.path.join(a.outdir, "README.md"), "w").write("\n".join(L))

    print(f"\nwrote {len(rows)} starts + README.md + starts.json to {a.outdir!r}")
    if bad:
        raise SystemExit(f"INFEASIBLE STARTS: {', '.join(bad)} -- fix before running")


if __name__ == "__main__":
    main()
