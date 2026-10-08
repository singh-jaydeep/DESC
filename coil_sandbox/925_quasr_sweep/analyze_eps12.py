"""How do eps_1 and eps_2 relate WITHIN the proxy family, and WITHIN the coil family?

    python 925_quasr_sweep/analyze_eps12.py 925_quasr_sweep/eps_A10.jsonl

eps_2 is normalised by the same whole-curve in-plane extent as eps_1, so eps_2/eps_1 is a pure
ratio of out-of-plane RMS (best 2-plane split vs best single plane) and C = 4 eps_2/eps_1 measures
it against the generic B^-2 decay: C < 1 is a real two-plane preference, C = 1 generic, C > 1
worse than generic.

Pools per-contour and per-coil values as well as per-device means, because the means hide the
within-device spread that a real coil set has to live with.
"""
import argparse
import json

import numpy as np
from scipy import stats

p = argparse.ArgumentParser()
p.add_argument("jsonl")
a = p.parse_args()

rows = [json.loads(ln) for ln in open(a.jsonl) if ln.strip()]
ok = [r for r in rows if r.get("status") == "ok"]
print(f"{len(ok)} usable devices\n")


def report(name, e1, e2, unit):
    """eps_1 vs eps_2 for one family."""
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    C = 4 * e2 / e1
    print(f"-- {name}  (n = {len(e1)} {unit}) --")
    print(f"   eps_1      {e1.min():.5f} .. {e1.max():.5f}   median {np.median(e1):.5f}   "
          f"spread {e1.max()/e1.min():.2f}x")
    print(f"   eps_2      {e2.min():.5f} .. {e2.max():.5f}   median {np.median(e2):.5f}   "
          f"spread {e2.max()/e2.min():.2f}x")
    print(f"   eps_2/eps_1 {(e2/e1).min():.4f} .. {(e2/e1).max():.4f}  median "
          f"{np.median(e2/e1):.4f}   (generic B^-2 would be 0.2500)")
    print(f"   C          {C.min():.3f} .. {C.max():.3f}   median {np.median(C):.3f}   "
          f"{int((C < 1).sum())}/{len(C)} below 1 (real 2-plane preference)")
    sp, pe = stats.spearmanr(e1, e2), stats.pearsonr(e1, e2)
    print(f"   eps_1 vs eps_2:  Spearman {sp.statistic:+.3f} (p {sp.pvalue:.2g})   "
          f"Pearson {pe.statistic:+.3f} (p {pe.pvalue:.2g})")
    sl, ic, rv, pv, se = stats.linregress(np.log(e1), np.log(e2))
    print(f"   log-log slope: eps_2 ~ eps_1^{sl:.3f} +- {se:.3f}   (R^2 {rv**2:.3f})"
          f"   {'-> proportional' if abs(sl-1) < 2*se else '-> NOT proportional'}")
    spc = stats.spearmanr(e1, C)
    print(f"   eps_1 vs C:      Spearman {spc.statistic:+.3f} (p {spc.pvalue:.2g})"
          f"   {'<-- flatter curves split better' if spc.statistic > 0 else '<-- flatter curves split WORSE' if spc.pvalue < 0.05 else ''}")
    print()
    return C


# ---- per-device means ------------------------------------------------------
Cp_dev = report("PROXY contours, per-device means",
                [r["eps1_mean"] for r in ok], [r["eps2_mean"] for r in ok], "devices")
Cc_dev = report("QUASR COILS, per-device means",
                [r["coil_eps1_mean"] for r in ok], [r["coil_eps2_mean"] for r in ok], "devices")

# ---- pooled individual curves ---------------------------------------------
pe1 = [v for r in ok for v in r["eps1"]]
pe2 = [v for r in ok for v in r["eps2"]]
ce1 = [v for r in ok for v in r["coil_eps1"]]
ce2 = [v for r in ok for v in r["coil_eps2"]]
report("PROXY contours, pooled individual contours", pe1, pe2, "contours")
report("QUASR COILS, pooled individual coils", ce1, ce2, "coils")

# ---- do the two families agree on C, device by device? --------------------
print("-- do proxy C and coil C agree, device by device? --")
sp = stats.spearmanr(Cp_dev, Cc_dev)
pe = stats.pearsonr(Cp_dev, Cc_dev)
print(f"   Spearman {sp.statistic:+.3f} (p {sp.pvalue:.3f})   Pearson {pe.statistic:+.3f} "
      f"(p {pe.pvalue:.3f})   n={len(Cp_dev)}")
print(f"   proxy C  median {np.median(Cp_dev):.3f}   coil C median {np.median(Cc_dev):.3f}"
      f"   ratio of medians {np.median(Cc_dev)/np.median(Cp_dev):.2f}x")

# ---- within a device, do the curves rank the same way? -------------------
print("\n-- WITHIN a device, do eps_1 and eps_2 rank the curves the same way? --")
for label, k1, k2 in (("proxy contours", "eps1", "eps2"),
                      ("QUASR coils", "coil_eps1", "coil_eps2")):
    rs = []
    for r in ok:
        x, y = np.asarray(r[k1]), np.asarray(r[k2])
        u = len(np.unique(np.round(x, 12)))
        if u < 3:                      # stellarator symmetry makes contours degenerate in pairs
            continue
        rs.append(stats.spearmanr(x, y).statistic)
    rs = np.asarray([v for v in rs if np.isfinite(v)])
    if len(rs):
        print(f"   {label:16s} mean within-device Spearman {rs.mean():+.3f}   "
              f"median {np.median(rs):+.3f}   range {rs.min():+.3f}..{rs.max():+.3f}   "
              f"({int((rs > 0).sum())}/{len(rs)} positive)")
    else:
        print(f"   {label:16s} too few distinct curves per device")

# ---- spread within a device ----------------------------------------------
print("\n-- within-device spread (max/min across the curves of one device) --")
for label, k1, k2 in (("proxy", "eps1", "eps2"), ("coils", "coil_eps1", "coil_eps2")):
    s1 = np.array([max(r[k1]) / min(r[k1]) for r in ok])
    s2 = np.array([max(r[k2]) / min(r[k2]) for r in ok])
    print(f"   {label:6s} eps_1 spread median {np.median(s1):.2f}x (max {s1.max():.2f}x)   "
          f"eps_2 spread median {np.median(s2):.2f}x (max {s2.max():.2f}x)")
