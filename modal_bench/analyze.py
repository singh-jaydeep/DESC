"""Turn the kernel-sweep ledger into the comparison table.

Reads ``ledger.jsonl`` (append-only, every measurement ever taken) and reduces it
to per-``n`` comparisons: each structured route at its BEST block width against
master's dense ``qr``, plus the block-width sensitivity and the peak-memory
ratio. Run with no arguments after any sweep.

Speed comparisons are only made within a single ``n``, because all points at one
``n`` ran in the same container and therefore on the same physical card; Modal
serves both A100-SXM4-80GB and A100 80GB PCIe, so cross-``n`` time ratios are not
necessarily hardware-matched. The GPU model is printed per row so that is visible.
"""

import collections
import json
import pathlib
import sys

JSONL = pathlib.Path(__file__).parent / "ledger.jsonl"


def load(path=JSONL):
    """All ledger records, successes and failures alike."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(path=JSONL):
    """Print the per-n comparison from the ledger."""
    every = load(path)
    rows = [r for r in every if r.get("ok")]
    if not rows:
        print("no successful measurements in the ledger yet")
        return
    by_n = collections.defaultdict(list)
    for r in rows:
        by_n[r["n"]].append(r)

    for n in sorted(by_n):
        rs = by_n[n]
        gpus = {r.get("gpu_name", "?") for r in rs}
        dense = [r for r in rs if r["method"] == "qr"]
        print(f"\n=== n = {n} === {' | '.join(sorted(gpus))}")
        if len(gpus) > 1:
            print(
                "  WARNING: mixed GPU models at this n; time ratios are not "
                "hardware-matched"
            )

        base = min((r["time_s"] for r in dense), default=None)
        base_mem = min((r["peak_GB"] for r in dense), default=None)
        if base:
            print(
                f"  qr (dense, master): {base*1e3:9.1f} ms   "
                f"peak {base_mem:6.2f} GB   gram {dense[0]['gram_rel']:.2e}"
            )

        for meth in ("qr-fixed",):
            cand = [r for r in rs if r["method"] == meth]
            if not cand:
                continue
            best = min(cand, key=lambda r: r["time_s"])
            worst = max(cand, key=lambda r: r["time_s"])
            spd = f"{base/best['time_s']:5.2f}x" if base else "   n/a"
            memr = f"{best['peak_GB']/base_mem:5.2f}x" if base_mem else "  n/a"
            print(
                f"  {meth:10s} best b={best['block']:<5d} "
                f"{best['time_s']*1e3:9.1f} ms  vs qr {spd}   "
                f"peak {best['peak_GB']:6.2f} GB ({memr} of qr)  "
                f"gram {best['gram_rel']:.2e}"
            )
            penalty = (worst["time_s"] / best["time_s"] - 1) * 100
            print(
                f"             block sensitivity: worst b={worst['block']} is "
                f"{penalty:.0f}% slower than best  "
                f"(run-to-run spread {best['time_spread']*100:.1f}%)"
            )

    # Failures are results, not noise: an OOM is exactly the memory ceiling this
    # sweep is meant to locate.
    fails = [r for r in every if not r.get("ok")]
    if fails:
        print("\n=== failures ===")
        for r in fails:
            err = str(r.get("error", ""))
            kind = "OOM" if "MEMORY" in err.upper() or "OOM" in err.upper() else "ERR"
            print(
                f"  [{kind}] n={r['n']:6d} {r['method']:10s} "
                f"b={r.get('block')}: {err[:120]}"
            )
    else:
        print("\nno failures recorded")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else JSONL)
