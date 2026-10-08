"""Bounded-step statistics from lsq-auglag's diag.jsonl, per run, over an iteration range:
how often the trust-region step leaves the slack box, how far it gets (p_stride), which
candidate is taken, which constraint blocks truncate it, and the inner optimality.
Usage: python step_stats.py RUN_DIR [RUN_DIR ...] [--its LO:HI]"""
import collections, json, re, sys
import numpy as np

args = [a for a in sys.argv[1:] if not a.startswith("--its")]
lo, hi = 1, 10**9
for a in sys.argv[1:]:
    if a.startswith("--its"):
        lo, hi = (int(v) for v in a.split("=", 1)[1].split(":"))
SEL = {-1: "full", 0: "truncated", 1: "reflected", 2: "steepest"}
row = re.compile(r"^\s+(\d+)\s+\d+\s+\S+\s+\S+\s+\S+\s+(\S+)\s")

for d in args:
    R = [json.loads(l) for l in open(f"{d}/diag.jsonl")]
    tr = [t for t in R if t["kind"] == "trial" and lo <= t["iteration"] <= hi]
    ac = [t for t in R if t["kind"] == "accept" and lo <= t["iteration"] <= hi]
    opt = [float(m.group(2)) for m in map(row.match, open(f"{d}/run.log")) if m and lo <= int(m.group(1)) <= hi]
    trunc = [t for t in tr if not t["in_bounds_full"]]
    ps = np.array([t["p_stride"] for t in trunc]) if trunc else np.array([np.nan])
    sel = collections.Counter(SEL[t["selected"]] for t in tr)
    blk = collections.Counter(b for t in trunc for b in {l.split(":")[1].split("[")[0] for l in t["hits_idx"]})
    print(f"== {d}  its {max(lo, min(t['iteration'] for t in tr))}-{max(t['iteration'] for t in tr)}: "
          f"{len(tr)} trials, {sum(t['accepted'] for t in tr)} accepted")
    print(f"   left the slack box: {len(trunc)}/{len(tr)} ({len(trunc) / len(tr):.0%})")
    print(f"   p_stride (5/25/50/75%): {np.percentile(ps, [5, 25, 50, 75]).round(4)}")
    print(f"   step taken: {dict(sel)}")
    print(f"   truncating blocks (trials): {dict(blk)}")
    print(f"   step/TR median {np.median([t['step_h_over_tr'] for t in tr]):.3f}, "
          f"model ratio median {np.median([t['Lreduction_ratio'] for t in tr]):.3f}, "
          f"TR median {np.median([t['trust_radius'] for t in tr]):.3g}")
    print(f"   optimality median/min {np.median(opt):.2e}/{min(opt):.2e}, "
          f"sign flips median {np.median([a['n_sign_flips'] for a in ac]):.0f}")
