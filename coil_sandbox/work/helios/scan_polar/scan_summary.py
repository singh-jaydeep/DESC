"""Summary of the polar arcB2 hinge-placement scan on helios_repro.

    python work/helios/scan_polar/scan_summary.py

Writes work/helios/scan_polar/summary.json and summary.png, and prints a table.
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, "sandbox")
import boot  # noqa: E402

boot.setup(default="cpu")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

import common as S  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.integrals.singularities import compute_B_plasma  # noqa: E402
from desc.io import load  # noqa: E402

PAPER = 0.0023
CUTS = [0, 30, 60, 90, 120, 150]
REFS = {"planar N=7": "work/helios/planarN7_final_it300.h5",
        "transverse arcB2 M=5": "work/helios/arcB2_M5_stage2_w1/result.h5",
        "arcB3 M=5": "work/helios/arcB3_M5_final_it200.h5"}

eq, _ = S.load_equilibrium("sample_equilibria/helios_repro.h5")
b = S.paper_bounds(eq, 3, None, "bounds/helios_kruger2026.json")
Baxis = float(np.mean(np.asarray(eq.compute("|B|", grid=LinearGrid(rho=np.array([0.0]), N=64, NFP=eq.NFP))["|B|"])))
g = LinearGrid(rho=np.array([1.0]), theta=96, zeta=96, NFP=eq.NFP, sym=False)
d = eq.compute(["R", "phi", "Z", "n_rho", "|e_theta x e_zeta|"], grid=g)
X = np.stack([d["R"], d["phi"], d["Z"]], 1)
nrm = np.asarray(d["n_rho"])
w = np.asarray(d["|e_theta x e_zeta|"])
Bp = np.asarray(compute_B_plasma(eq, g, S.vc_source_grid(eq), chunk_size=64))


def last_bn(log):
    if not os.path.exists(log):
        return None
    m = re.findall(r"^RESULT bn ([0-9.e+-]+)", open(log).read(), flags=re.M)
    return float(m[-1]) if m else None


def paper_metrics(cs):
    B = np.asarray(cs.compute_magnetic_field(X, basis="rpz", source_grid=S.source_grid(cs)[0])) + Bp
    Bn = np.abs(np.sum(B * nrm, 1))
    mean_T = float(np.sum(Bn * w) / np.sum(w))
    return dict(mean_absBn_T=mean_T, pct_Baxis=100 * mean_T / Baxis, max_absBn_T=float(Bn.max()))


def geometry(cs):
    rows = []
    for c in S.iter_unique(cs):
        row = dict(current_MA=round(float(c.current) / 1e6, 2), L=round(S.curve_metrics(c)["L"], 3))
        if hasattr(c, "hinges"):
            H = np.asarray(c.hinges).reshape(-1, 3)
            row["hinges_RZ"] = [[round(float(np.hypot(h[0], h[1])), 2), round(float(h[2]), 2)] for h in H]
            xx = np.asarray(c.compute("x", grid=LinearGrid(zeta=np.linspace(0, 2 * np.pi, 1200, endpoint=False), NFP=1),
                                      basis="xyz")["x"])
            n0, n1 = (np.linalg.svd(q - q.mean(0))[2][-1] for q in (xx[:601], np.vstack([xx[600:], xx[:1]])))
            row["fold_deg"] = round(float(np.degrees(np.arccos(min(1.0, abs(n0 @ n1))))), 1)
            row["corner_deg"] = round(S.curve_metrics(c)["corner"], 1)
            row["arc_ref_margin"] = round(float(np.max(np.asarray(c.arc_ref_margin))), 3)
        rows.append(row)
    return rows


def pts(cs):
    return [np.asarray(c.compute("x", grid=LinearGrid(N=500), basis="xyz")["x"]) for c in S.iter_unique(cs)]


def coilset_distance(pa, pb):
    return max(max(cKDTree(a).query(bb)[0].max(), cKDTree(bb).query(a)[0].max()) for a, bb in zip(pa, pb))


out = dict(B_axis_T=Baxis, paper_pct=100 * PAPER, cuts={}, refs={})
finals = {}
for cut in CUTS:
    D = f"work/helios/scan_polar/cut{cut}"
    final = f"{D}/legC/result.h5" if os.path.exists(f"{D}/legC/result.h5") else f"{D}/legB/result.h5"
    if not os.path.exists(final):
        continue
    cs = load(final)
    m = S.evaluate(cs, eq)
    finals[cut] = pts(cs)
    out["cuts"][cut] = dict(
        final=final,
        bn_fit=float(json.load(open(f"{D}/fit.json"))["metrics"]["bn"]),
        bn_repair=last_bn(f"{D}/repair.log"), bn_held=last_bn(f"{D}/legA.log"),
        bn_free=last_bn(f"{D}/legB.log"), bn_continued=last_bn(f"{D}/legC.log"),
        bn=m["bn"], verdict=S.verdict(m, b), margins={k: round(v, 4) for k, v in S.margins(m, b).items()},
        worst_L_excess_m=max(p["L"] for p in m["per_coil"]) - b["L"],
        **paper_metrics(cs), coils=geometry(cs))
for name, p in REFS.items():
    cs = load(p)
    m = S.evaluate(cs, eq)
    finals[name] = pts(cs)
    out["refs"][name] = dict(file=p, bn=m["bn"], verdict=S.verdict(m, b), **paper_metrics(cs), coils=geometry(cs))

keys = list(finals)
out["distance_m"] = {f"{a} | {c}": round(coilset_distance(finals[a], finals[c]), 2)
                     for i, a in enumerate(keys) for c in keys[i + 1:]
                     if not (a in REFS and c in REFS)}
json.dump(out, open("work/helios/scan_polar/summary.json", "w"), indent=1, default=float)

p0 = out["refs"]["planar N=7"]["pct_Baxis"]
print(f"B_axis {Baxis:.3f} T; paper (planar + shaping) {100 * PAPER:.2f}%")
print(f"{'start':>22} | {'fit':>8} {'repair':>8} {'held':>8} {'free':>8} {'final bn':>9} | {'% B_axis':>8} {'max T':>6} "
      f"{'gap lin/log':>12} | verdict")
rows = [(f"polar cut {c}", r) for c, r in out["cuts"].items()] + [(n, r) for n, r in out["refs"].items()]
for name, r in sorted(rows, key=lambda t: t[1]["bn"]):
    pct = r["pct_Baxis"]
    gap = f"{100 * (p0 - pct) / (p0 - 100 * PAPER):.0f}%/{100 * np.log(p0 / pct) / np.log(p0 / (100 * PAPER)):.0f}%"
    st = " ".join(f"{r.get(k):8.4f}" if r.get(k) is not None else f"{'-':>8}"
                  for k in ("bn_fit", "bn_repair", "bn_held", "bn_free"))
    print(f"{name:>22} | {st} {r['bn']:9.4e} | {pct:8.3f} {r['max_absBn_T']:6.3f} {gap:>12} | {r['verdict']}")
print("\nper-coil geometry (hinges (R,Z) m, fold deg, corner deg, arc_ref margin, current MA):")
for name, r in sorted(rows, key=lambda t: t[1]["bn"]):
    if "hinges_RZ" in r["coils"][0]:
        print(f"  {name}: " + " | ".join(f"H {c['hinges_RZ']} fold {c['fold_deg']} corner {c['corner_deg']} "
                                         f"m {c['arc_ref_margin']} I {c['current_MA']}" for c in r["coils"]))
print("\ncoilset distances (m, max over coils):")
for k, v in sorted(out["distance_m"].items(), key=lambda t: t[1]):
    print(f"  {v:5.2f}  {k}")

# figure: bn by stage per cut
fig, ax = plt.subplots(figsize=(10, 4.8))
stages = [("bn_fit", "fit"), ("bn_repair", "repair"), ("bn_held", "hinges held"), ("bn", "final")]
xs = np.arange(len(out["cuts"]))
for k, (key, lab) in enumerate(stages):
    ax.bar(xs + (k - 1.5) * 0.2, [r.get(key) or np.nan for r in out["cuts"].values()], width=0.2, label=lab,
           color=["#c4c3be", "#898781", "#2a78d6", "#eb6834"][k])
for name, col in (("transverse arcB2 M=5", "#1baf7a"), ("arcB3 M=5", "#8a5cd6")):
    ax.axhline(out["refs"][name]["bn"], color=col, lw=1.5, ls="--", label=name)
ax.set_xticks(xs, [f"cut {c}°" for c in out["cuts"]])
ax.set_ylabel("⟨|B·n|⟩/⟨|B|⟩ (coils + plasma)")
ax.set_title("polar arcB2 hinge-placement scan on helios_repro", loc="left")
ax.legend(frameon=False, fontsize=8, ncol=3)
fig.tight_layout()
fig.savefig("work/helios/scan_polar/summary.png", dpi=110)
print("wrote work/helios/scan_polar/summary.json, summary.png")
