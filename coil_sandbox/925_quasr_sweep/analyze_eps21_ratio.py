"""(a) Is eps_2 <= eps_1 always, as the nesting requires? (b) Where is the gap LARGE?

eps_2 minimises over 2-segmentations, which contains the degenerate split that reproduces the
single-plane answer, so eps_2 <= eps_1 must hold identically. Both use the same whole-curve
(lambda_1 lambda_2)^(1/4) denominator, and `freeze` recomputes n0 as the true minimum eigenvector,
so the denominators agree exactly and the ratio is a clean comparison of out-of-plane RMS. A
violation would therefore mean `best_breaks`'s `stride` search missed the near-degenerate split.

Generic B^-2 decay predicts eps_2/eps_1 = 0.25, so ratio << 0.25 is a real clamshell and
ratio >> 0.25 means two planes barely help.

For the coils (cheap, no solve) it also recomputes the DP breakpoints, to tell a GENUINE
two-plane structure (two comparable arcs) from the DP buying a low eps_2 with a sliver segment --
the same degeneracy that gave the arcB2 projection kappa = 358 in AUDIT.md section 3b.
"""
import argparse
import json
import sys

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("jsonl")
p.add_argument("--cache", default="925_quasr_sweep/cache")
p.add_argument("--top", type=int, default=8)
p.add_argument("--breaks", action="store_true", help="recompute coil DP breakpoints (needs DESC)")
a = p.parse_args()

rows = [json.loads(ln) for ln in open(a.jsonl) if ln.strip()]
ok = [r for r in rows if r.get("status") == "ok"]

# ---- (a) the nesting inequality -------------------------------------------
print("=== (a) is eps_2 <= eps_1? ===\n")
for fam, k1, k2 in (("proxy contours", "eps1", "eps2"), ("QUASR coils", "coil_eps1", "coil_eps2")):
    recs = [(r["ID"], i, e1, e2) for r in ok
            for i, (e1, e2) in enumerate(zip(r[k1], r[k2]))]
    viol = [t for t in recs if t[3] > t[2]]
    rat = np.array([t[3] / t[2] for t in recs])
    print(f"-- {fam}: {len(recs)} curves --")
    print(f"   violations (eps_2 > eps_1): {len(viol)}")
    for ID, i, e1, e2 in viol[:10]:
        print(f"      ID {ID} curve {i}: eps_1 {e1:.6f} < eps_2 {e2:.6f}  "
              f"(excess {(e2/e1-1)*100:.3f}%)")
    q = np.percentile(rat, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    print("   eps_2/eps_1 percentiles  min {:.4f}  p1 {:.4f}  p5 {:.4f}  p25 {:.4f}  "
          "med {:.4f}  p75 {:.4f}  p95 {:.4f}  p99 {:.4f}  max {:.4f}".format(*q))
    print(f"   fraction below the generic 0.25: {int((rat < 0.25).sum())}/{len(rat)}"
          f"   below 0.20: {int((rat < 0.20).sum())}   below 0.15: {int((rat < 0.15).sum())}")
    print()

# ---- (b) the extremes -----------------------------------------------------
print("=== (b) where is the gap large? ===\n")
meta = {r["ID"]: r for r in ok}
for fam, k1, k2 in (("proxy contours", "eps1", "eps2"), ("QUASR coils", "coil_eps1", "coil_eps2")):
    recs = sorted([(r["ID"], i, e1, e2, e2 / e1) for r in ok
                   for i, (e1, e2) in enumerate(zip(r[k1], r[k2]))], key=lambda t: t[4])
    print(f"-- {fam}: BEST {a.top} two-plane gains (smallest eps_2/eps_1) --")
    print(f"   {'ID':>8} {'#':>3} {'eps_1':>9} {'eps_2':>9} {'ratio':>7} {'1/ratio':>8}  device")
    for ID, i, e1, e2, rr in recs[:a.top]:
        m = meta[ID]
        print(f"   {ID:>8} {i:>3} {e1:>9.5f} {e2:>9.5f} {rr:>7.4f} {1/rr:>7.2f}x  "
              f"nfp{m['nfp']}{'QH' if m['helicity'] else 'QA'} A{m['aspect_ratio']:.1f} "
              f"elong {m['mean_elongation']:.2f}")
    print(f"   -- WORST {a.top} (largest eps_2/eps_1: two planes barely help) --")
    for ID, i, e1, e2, rr in recs[-a.top:][::-1]:
        m = meta[ID]
        print(f"   {ID:>8} {i:>3} {e1:>9.5f} {e2:>9.5f} {rr:>7.4f} {1/rr:>7.2f}x  "
              f"nfp{m['nfp']}{'QH' if m['helicity'] else 'QA'} A{m['aspect_ratio']:.1f} "
              f"elong {m['mean_elongation']:.2f}")
    print()

# ---- is a big gain a real clamshell, or a sliver? -------------------------
if not a.breaks:
    print("(pass --breaks to recompute the coil DP segmentations for the extremes)")
    sys.exit()

sys.path.insert(0, "sandbox")
sys.path.insert(0, "922_stage1opt_results")
import boot  # noqa: E402

boot.setup(default="cpu")
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
import clamshell as CL  # noqa: E402
import quasr as Q  # noqa: E402
from contour_curve import _eps1_live  # noqa: E402
from desc.grid import Grid  # noqa: E402

print("\n=== is a large gain a GENUINE clamshell, or a sliver segment? (coils) ===")
print("   segment balance = shorter arc / total, by ARCLENGTH. 0.5 = two equal arcs,")
print("   -> 0 = the DP bought a low eps_2 with a degenerate sliver (useless as an arc coil).\n")
recs = sorted([(r["ID"], i, e1, e2, e2 / e1) for r in ok
               for i, (e1, e2) in enumerate(zip(r["coil_eps1"], r["coil_eps2"]))],
              key=lambda t: t[4])
look = recs[:a.top] + recs[-a.top:]
print(f"   {'ID':>8} {'#':>3} {'ratio':>7} {'balance':>8} {'seg arclen fractions':>24}")
for ID, i, e1, e2, rr in look:
    base = [c for c in Q.coilset(ID, a.cache, full=False)]
    c = base[i]
    n = 480
    s = 2 * np.pi * np.arange(n) / n
    d = c.compute(["x", "x_s"], grid=Grid(np.stack([0 * s, 0 * s, s], 1), sort=False,
                                          jitable=False), basis="xyz")
    x = np.asarray(d["x"])
    w = np.linalg.norm(np.asarray(d["x_s"]), axis=1)
    brk = CL.best_breaks(x, w, 2, stride=8)[1]
    # arclength of each segment between consecutive breakpoints, cyclically
    cw = np.concatenate([[0.0], np.cumsum(w)])
    tot = cw[-1]
    b = sorted(int(v) for v in brk)
    segs = []
    for j in range(len(b)):
        lo, hi = b[j], b[(j + 1) % len(b)]
        L = (cw[hi] - cw[lo]) if hi > lo else (tot - cw[lo] + cw[hi])
        segs.append(L / tot)
    bal = min(segs)
    print(f"   {ID:>8} {i:>3} {rr:>7.4f} {bal:>8.3f}   "
          f"{'  '.join(f'{v:.3f}' for v in segs)}")
