"""Health check for a run, from its log. Cheap enough to call every few minutes.

    python tools/monitor.py runs/sw_arcB5_L1.log
    python tools/monitor.py runs/sw_arcB5_L1.log --json     # machine-readable

Prints progress, the `bn` trend, and a STALL VERDICT, plus any of the known pathologies
below that it can see. It reads only the log, so it costs nothing and cannot disturb the
solve -- which matters, because two runs in this campaign died to a `verify` running
concurrently with a solve.

THE STALL VERDICT IS ADVISORY, AND DELIBERATELY CONSERVATIVE.

`--maxiter 500` is **too small** on this problem: both x1.25 and both x1.5 runs hit that
ceiling still descending, and the x1.5 certification run was still improving ~1% per 50
iterations at 800. So "no visible progress" before ~500 iterations usually means the run
is in its penalty phase, not that it is stuck. This tool will not call a stall before
`--min-iter` (default 500) whatever the trend looks like.

It also cannot distinguish a stall from ARRIVAL. Measured at one endpoint: four
constraints inside 0.5% of their bounds at once, `|Pg|/|g| = 0.993` -- a perfectly good
descent direction -- but `t_max = 9.17e-06`, so a near-active row blocks the step almost
immediately. That reads exactly like a stall and is actually the answer. Both cases are
reported as `stalled`; deciding between them needs the margins, which means adjudicating.

WHAT TO DO WITH A STALL. Do NOT kill the process: a run stopped by hand writes
checkpoints but no final `h5` and no record, which breaks the chain that warm-starts from
it -- seven runs were lost that way on 2026-09-09. Use `tools/promote_checkpoint.py`,
which adjudicates every checkpoint and promotes the last cleanly feasible one.
"""

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bundle  # noqa: E402,F401  -- MUST precede any `import desc`

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

CKPT = re.compile(r"\[ckpt it=\s*(\d+)\]\s*bn=([0-9.eE+-]+)\s*max=([0-9.eE+-]+)")


def parse(path):
    """Checkpoints, the last AL table row, the device line, and any error text."""
    txt = open(path, errors="replace").read()
    ck = [(int(a), float(b), float(c)) for a, b, c in CKPT.findall(txt)]
    # AL iteration table: fixed 15-char columns, a data row starts with an integer.
    # order: iteration nfev cost d_cost |step| optimality violation mu max|y| gtolk
    last = None
    for line in txt.splitlines():
        f = line.split()
        if len(f) >= 10 and f[0].isdigit():
            try:
                last = dict(iteration=int(f[0]), nfev=int(f[1]), cost=float(f[2]),
                            optimality=float(f[5]), violation=float(f[6]),
                            mu=float(f[7]), max_y=float(f[8]), gtolk=float(f[9]))
            except ValueError:
                pass
    dev = None
    m = re.search(r"^DEVICE requested=(\S+) obtained=(\S+)", txt, re.M)
    if m:
        dev = dict(requested=m.group(1), obtained=m.group(2),
                   mismatch=m.group(1) != m.group(2))
    err = None
    for marker in ("Traceback (most recent call last)", "RESOURCE_EXHAUSTED",
                   "XlaRuntimeError", "MemoryError", "Killed"):
        if marker in txt:
            err = marker
            break
    # The driver's last act is `wrote <outdir>/<tag>.json`; adjudication then appends
    # its own block. Either means the solve is over -- do not key off the AL's
    # termination sentence, which varies with why it stopped.
    done = ("ADJUDICATION" in txt) or bool(re.search(r"^wrote .*\.json$", txt, re.M))
    return ck, last, dev, err, done, txt


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log")
    ap.add_argument("--min-iter", type=int, default=500,
                    help="never call a stall before this iteration (default 500)")
    ap.add_argument("--window", type=int, default=3,
                    help="checkpoints of no progress required (default 3 = 150 iters)")
    ap.add_argument("--rel", type=float, default=0.005,
                    help="relative bn improvement below which a window counts as flat")
    ap.add_argument("--json", action="store_true", dest="as_json")
    a = ap.parse_args()

    if not os.path.exists(a.log):
        raise SystemExit(f"no log at {a.log}")
    ck, last, dev, err, done, txt = parse(a.log)
    age = time.time() - os.path.getmtime(a.log)

    verdict, why = "running", []
    if err:
        verdict, why = "ERROR", [f"log contains {err!r}"]
    elif done:
        verdict, why = "finished", []
    elif age > 900:
        verdict, why = "SILENT", [f"log untouched for {age/60:.0f} min"]

    it = ck[-1][0] if ck else (last["iteration"] if last else 0)
    trend = None
    if len(ck) >= a.window + 1:
        lo, hi = ck[-(a.window + 1)][1], ck[-1][1]
        trend = (lo - hi) / max(abs(lo), 1e-300)
        if verdict == "running" and it >= a.min_iter and trend < a.rel:
            verdict = "stalled"
            why.append(f"bn improved {100*trend:.2f}% over the last "
                       f"{a.window} checkpoints (~{a.window*50} iters), "
                       f"below the {100*a.rel:.1f}% threshold")
    if verdict == "running" and it < a.min_iter and trend is not None and trend < a.rel:
        why.append(f"flat, but only at iteration {it} -- below --min-iter {a.min_iter}, "
                   f"where runs are routinely still in the penalty phase")

    # known pathologies, from the campaign record
    flags = []
    if last:
        if last["mu"] > 1e3:
            flags.append(f"penalty mu = {last['mu']:.2e}. With --ctol 1e-4 it should "
                         f"settle near 15-30; a ratchet to 1e3+ collapses the trust "
                         f"region and means ctol is under the noise floor.")
        if last["gtolk"] < 1e-10:
            flags.append(f"gtolk = {last['gtolk']:.2e} -- the subproblem tolerance has "
                         f"run away below anything achievable; the outer loop is dead.")
        if last["violation"] > 1e-2 and it > 100:
            flags.append(f"constraint violation {last['violation']:.2e} at iteration "
                         f"{it} -- still far outside; check it is not heading for a "
                         f"crossing (adjudicate a checkpoint).")

    if dev and dev["mismatch"]:
        flags.append(f"device: requested {dev['requested']}, got {dev['obtained']}. "
                     f"Roughly 2x slower, and not bit-comparable with runs in the same "
                     f"chain that used the other device.")
    rec = dict(log=a.log, device=dev, iterations=it, checkpoints=len(ck), verdict=verdict,
               reasons=why, flags=flags, log_age_s=round(age, 1),
               bn=ck[-1][1] if ck else None, bn_max=ck[-1][2] if ck else None,
               bn_trend_rel=trend, last_row=last)
    if a.as_json:
        print(json.dumps(rec, indent=1))
        return

    print(f"{os.path.basename(a.log)}   {verdict.upper()}")
    print(f"  iteration {it}   {len(ck)} checkpoints   log touched "
          f"{age/60:.1f} min ago")
    if dev:
        mark = "  <-- MISMATCH" if dev["mismatch"] else ""
        print(f"  device requested={dev['requested']} obtained={dev['obtained']}{mark}")
    if ck:
        print(f"  bn = {ck[-1][1]:.4e}   max|B.n|/|B| = {ck[-1][2]:.3e}")
        for i, b, m in ck[-4:]:
            print(f"      it={i:>4}  bn={b:.4e}")
        if trend is not None:
            print(f"  bn improved {100*trend:.2f}% over the last {a.window} checkpoints")
    if last:
        print(f"  violation {last['violation']:.2e}   mu {last['mu']:.3g}   "
              f"optimality {last['optimality']:.2e}   gtolk {last['gtolk']:.2e}")
    for w in why:
        print(f"  · {w}")
    for f in flags:
        print(f"  ! {f}")
    if verdict == "stalled":
        print("\n  A stall here may be ARRIVAL (a multi-constraint corner), not failure.")
        print("  Do NOT kill the process -- that loses the final h5 and the record, and")
        print("  breaks any chain warm-starting from this run. Instead:")
        print(f"      python tools/promote_checkpoint.py {os.path.dirname(a.log)}/"
              f"{os.path.basename(a.log)[:-4]}")


if __name__ == "__main__":
    main()
