"""Tabulate and correlate what `record_eps.py` recorded. Pure numpy/scipy, no DESC, no GPU.

    python 925_quasr_sweep/analyze_eps.py 925_quasr_sweep/eps_A10.jsonl [--iota-tol 0.01]

Three questions, in order:
  1. Who is OUT of the modular-proxy class, and does exclusion correlate with shape? If it does,
     every conclusion drawn from the survivors is conditioned on it.
  2. Does the proxy eps_1 / eps_2 rank the SAME WAY as eps_1 / eps_2 of QUASR's own coils?
     Rank, not value -- section 5 of AUDIT.md already showed the offset is not a constant.
  3. Is the offset explained by aspect ratio or coil standoff?

`--iota-tol` gates question 2 on |d iota| between our fixed-boundary vacuum solve and QUASR's own
edge iota. Where that is large, our equilibrium and their device are not the same configuration,
so comparing our proxy against their coils compares two different things. Filter on the ABSOLUTE
difference: the relative error is meaningless for the QA devices whose iota target was 0.1.
"""
import argparse
import json

import numpy as np
from scipy import stats

p = argparse.ArgumentParser()
p.add_argument("jsonl")
p.add_argument("--iota-tol", type=float, default=0.01,
               help="|d iota| above this = our solve did not reproduce their device")
a = p.parse_args()

rows = [json.loads(ln) for ln in open(a.jsonl) if ln.strip()]
ok = [r for r in rows if r.get("status") == "ok"]
bad = [r for r in rows if r.get("status") != "ok"]
print(f"{len(rows)} devices recorded: {len(ok)} ok, {len(bad)} excluded "
      f"({100*len(bad)/max(len(rows),1):.0f}%)\n")


def col(rs, k):
    return np.array([r[k] for r in rs], dtype=float)


# ---- 1. who is excluded -----------------------------------------------------
if bad:
    kinds = {}
    for r in bad:
        key = r["status"].split(":")[0] + (" / non-monotone" if "monotone" in r["status"] else "")
        kinds[key] = kinds.get(key, 0) + 1
    print("-- excluded, by reason --")
    for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print(f"   {v:3d}  {k}")
    have = [r for r in bad if "aspect_ratio" in r]
    if have and ok:
        print("\n-- does exclusion correlate with shape? (excluded vs kept, median) --")
        for k in ("aspect_ratio", "nfp", "helicity", "nc_per_hp", "mean_iota",
                  "mean_elongation", "max_elongation", "minor_radius", "qs_error"):
            b, g = np.median(col(have, k)), np.median(col(ok, k))
            try:
                pv = stats.mannwhitneyu(col(have, k), col(ok, k)).pvalue
            except ValueError:
                pv = float("nan")
            flag = "  <-- differs" if pv < 0.05 else ""
            print(f"   {k:18s} excluded {b:9.3f}   kept {g:9.3f}   Mann-Whitney p {pv:.3f}{flag}")
    print()

if not ok:
    raise SystemExit("nothing usable recorded")

# ---- the table --------------------------------------------------------------
ok.sort(key=lambda r: r["eps1_mean"])
print("-- recorded devices, sorted by proxy eps_1 --")
print(f"{'ID':>8} {'dev':>9} {'A':>5} {'d_pc/a':>7} {'K':>3} | {'eps1':>8} {'eps2':>8} {'C':>6} | "
      f"{'coil_e1':>8} {'coil_e2':>8} {'eta':>7} | {'ratio':>6} {'|diota|':>8} {'qs':>6}")
for r in ok:
    tag = f"nfp{r['nfp']}{'QH' if r['helicity'] else 'QA'}nc{r['nc_per_hp']}"
    di = r.get("iota_err_abs")
    if di is None:
        di = abs(abs(r["iota_edge"]) - abs(r["iota_edge_quasr"]))
    mark = "  *" if di > a.iota_tol else ""
    print(f"{r['ID']:>8} {tag:>9} {r['aspect_ratio']:>5.2f} {r['d_pc_over_a']:>7.2f} {r['K']:>3d} | "
          f"{r['eps1_mean']:>8.5f} {r['eps2_mean']:>8.5f} {r['C_mean']:>6.3f} | "
          f"{r['coil_eps1_mean']:>8.5f} {r['coil_eps2_mean']:>8.5f} {r['coil_eta_mean']:>7.4f} | "
          f"{r['coil_eps1_mean']/r['eps1_mean']:>6.2f} {di:>8.4f} {r['qs_error']:>6.2f}{mark}")
print("   * = |d iota| above the tolerance: our solve did not reproduce QUASR's device")

# ---- 2. does the proxy rank the coils? -------------------------------------
def diota(r):
    v = r.get("iota_err_abs")
    return v if v is not None else abs(abs(r["iota_edge"]) - abs(r["iota_edge_quasr"]))


good = [r for r in ok if diota(r) <= a.iota_tol]
print(f"\n-- rank agreement (n={len(ok)} all, n={len(good)} with |d iota| <= {a.iota_tol}) --")
for label, rs in (("all recorded", ok), (f"|d iota| <= {a.iota_tol}", good)):
    if len(rs) < 4:
        print(f"   {label:22s} n={len(rs)}, too few")
        continue
    for pk, ck, nm in (("eps1_mean", "coil_eps1_mean", "eps_1"),
                       ("eps2_mean", "coil_eps2_mean", "eps_2"),
                       ("C_mean", "coil_eps2_mean", "C vs coil eps_2")):
        x, y = col(rs, pk), col(rs, ck)
        sp = stats.spearmanr(x, y)
        pe = stats.pearsonr(x, y)
        print(f"   {label:22s} {nm:16s} Spearman {sp.statistic:+.3f} (p {sp.pvalue:.3f})   "
              f"Pearson {pe.statistic:+.3f} (p {pe.pvalue:.3f})   n={len(rs)}")

# ---- 3. what explains the offset? ------------------------------------------
print("\n-- what the proxy-to-coil eps_1 ratio tracks --")
ratio = col(ok, "coil_eps1_mean") / col(ok, "eps1_mean")
print(f"   ratio: {ratio.min():.2f} .. {ratio.max():.2f}  (median {np.median(ratio):.2f}, "
      f"spread {ratio.max()/ratio.min():.2f}x)")
for k in ("aspect_ratio", "d_pc_over_a", "d_cc_over_a", "nfp", "nc_per_hp", "mean_iota",
          "mean_elongation", "minor_radius"):
    v = col(ok, k)
    if np.ptp(v) == 0:
        continue
    sp = stats.spearmanr(v, ratio)
    print(f"   vs {k:18s} Spearman {sp.statistic:+.3f} (p {sp.pvalue:.3f})")

# ---- the eta identity ------------------------------------------------------
e1 = col(ok, "coil_eps1_mean")
pred = e1 / (e1 + 2)
got = col(ok, "coil_eta_mean")
rel = np.abs(got / pred - 1) * 100
print(f"\n-- eta_SVD = eps_1/(eps_1+2) on the real coils: max deviation {rel.max():.2f}%, "
      f"median {np.median(rel):.2f}% (n={len(ok)}) --")

# ---- trust ----------------------------------------------------------------
print("\n-- solve / objective health --")
for k, fmt in (("force_rel", "{:.2e}"), ("newton_res", "{:.1e}"), ("modes_kept", "{:.4f}"),
               ("t_total", "{:.0f}s")):
    v = col(ok, k)
    print(f"   {k:12s} min {fmt.format(v.min()):>10}  median {fmt.format(np.median(v)):>10}  "
          f"max {fmt.format(v.max()):>10}")
print(f"   monotonic    {sum(bool(r['monotonic']) for r in ok)}/{len(ok)}")
di = np.array([diota(r) for r in ok])
print(f"   |d iota|     min {di.min():.4f}  median {np.median(di):.4f}  max {di.max():.4f}   "
      f"({int((di<=a.iota_tol).sum())}/{len(ok)} within {a.iota_tol})")
