"""Side-by-side table and overlay plots for designs B and D and the literature references, all produced by the SAME
script and settings (../designX/make_figures.py; the references with `--ref NAME`).

    cd final/designD   (either design folder: the script is identical)
    for r in precise_QA reactor_QA NCSX ARIES-CS; do python make_figures.py --ref $r --out ../compare/ref_$r.json; done
    cd ../compare && python compare.py          (CPU, seconds; reads the json files only)

Writes COMPARE.md, references.json (read by make_figures.py's transport figure) and overlay_*.png.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DES = {n: json.load(open(os.path.join(HERE, "..", f"design{n}", "metrics.json"))) for n in ("B", "D")}
REFS = {}
for r in ("precise_QA", "reactor_QA", "NCSX", "ARIES-CS"):
    f = os.path.join(HERE, f"ref_{r}.json")
    if os.path.exists(f):
        REFS[r] = json.load(open(f))
json.dump(REFS, open(os.path.join(HERE, "references.json"), "w"), indent=1)
ALL = {"B": DES["B"], "D": DES["D"], **REFS}


def prof(m, key, rhos):
    p = m["equilibrium"]["profiles"]
    return [float(np.interp(r, p["rho"], p[key])) for r in rhos]


def j(v, fmt):
    return " / ".join(fmt.format(x) for x in v)


L = []
w = L.append
w("# Designs B and D against the literature (same script, same settings)\n")
w("Equilibria: B and D are vacuum free-boundary states of their coils at basis 14; the references are DESC's example "
  "equilibria (fixed boundary, no coils; finite beta where the example has it). Every number below comes from "
  "`make_figures.py` (QS in the literature Boozer convention; transport at the `hi` bounce-integral resolution).\n")
cols = list(ALL)
w("| | " + " | ".join(f"**{c}**" if c in DES else c for c in cols) + " |")
w("|---|" + "---|" * len(cols))
rows = [
    ("aspect ratio", lambda m: f"{m['equilibrium']['A']:.2f}"),
    ("NFP", lambda m: f"{m['equilibrium']['NFP']}"),
    ("iota, rho 0 / .5 / 1", lambda m: j([abs(v) for v in prof(m, "iota", [0, 0.5, 1])], "{:.3f}")),
    ("L2 breaking modes %, rho .25/.5/.75/1", lambda m: j([100 * v for v in m["equilibrium"]["qs"]["l2"]], "{:.3f}")),
    ("largest breaking mode %, rho .25/.5/.75/1",
     lambda m: j([100 * v for v in m["equilibrium"]["qs"]["max"]], "{:.3f}")),
    ("two-term rms(f_C/B^3), rho .25/.5/.75/1", lambda m: j(m["equilibrium"]["qs"]["two_term"], "{:.1e}")),
    ("eps_eff, rho .25/.5/.75", lambda m: j(m.get("transport", {}).get("hi", {}).get("effective ripple", []), "{:.1e}")),
    ("Gamma_c (Nemov), rho .25/.5/.75", lambda m: j(m.get("transport", {}).get("hi", {}).get("Gamma_c", []), "{:.1e}")),
    ("Gamma_c (Velasco), rho .25/.5/.75",
     lambda m: j(m.get("transport", {}).get("hi", {}).get("Gamma_c Velasco", []), "{:.1e}")),
    ("magnetic well (DESC, >0 = well), rho .5 / 1", lambda m: j(prof(m, "magnetic_well", [0.5, 1]), "{:+.1e}")),
    ("mirror ratio, rho .5 / 1", lambda m: j(prof(m, "mirror_ratio", [0.5, 1]), "{:.3f}")),
    ("max elongation (boundary sections)", lambda m: f"{max(m['equilibrium']['elongation']['second_moment']):.2f}"),
    ("coils' B.n rms / max", lambda m: f"{m['coils']['bn_rms']:.1e} / {m['coils']['bn_max']:.1e}" if "coils" in m
     else "(no coils)"),
    ("coils", lambda m: f"{m['coils']['n_unique']}/hp, {m['coils']['verdict']}" if "coils" in m else ""),
    ("traced iota vs equilibrium, axis offset", lambda m: f"{m['trace']['max_abs_iota_trace_minus_eq']:.0e}, "
     f"{m['trace']['axis_offset_mm']:.2f} mm" if "trace" in m else ""),
]
for lab, f in rows:
    cells = []
    for c in cols:
        try:
            cells.append(f(ALL[c]))
        except (KeyError, TypeError, ValueError, IndexError):
            cells.append("")
    w(f"| {lab} | " + " | ".join(cells) + " |")
w("\nFigures: `overlay_iota.png`, `overlay_qs.png`, `overlay_transport.png`.\n")
open(os.path.join(HERE, "COMPARE.md"), "w").write("\n".join(L))

# overlays
sty = {"B": dict(color="C0", lw=2.2), "D": dict(color="C3", lw=2.2)}
for i, r in enumerate(REFS):
    sty[r] = dict(color=f"C{i + 4}", lw=1.0, ls="--")
fig, ax = plt.subplots(figsize=(7, 4.6))
for n, m in ALL.items():
    p = m["equilibrium"]["profiles"]
    ax.plot(p["rho"], np.abs(p["iota"]), label=n, **sty[n])
ax.set_xlabel("rho")
ax.set_ylabel("|iota|")
ax.legend(fontsize=8)
ax.set_title("rotational transform")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "overlay_iota.png"), dpi=150)
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
for n, m in ALL.items():
    q = m["equilibrium"]["qs"]
    axs[0].semilogy(q["rho"], 100 * np.array(q["l2"]), marker="o", ms=3, label=n, **sty[n])
    axs[1].semilogy(q["rho"], 100 * np.array(q["max"]), marker="o", ms=3, label=n, **sty[n])
axs[0].set_title("L2 of the symmetry-breaking Boozer modes (% of B_00)")
axs[1].set_title("largest symmetry-breaking Boozer mode (% of B_00)")
for ax in axs:
    ax.set_xlabel("rho")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "overlay_qs.png"), dpi=150)
fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
for ax, qn in zip(axs, ("effective ripple", "Gamma_c", "Gamma_c Velasco")):
    for n, m in ALL.items():
        t = m.get("transport", {})
        v = t.get("hi", {}).get(qn)
        if isinstance(v, list):
            ax.semilogy(t["rho"], v, marker="o", ms=3, label=n, **sty[n])
    ax.set_title(qn.replace("effective ripple", "eps_eff"))
    ax.set_xlabel("rho")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "overlay_transport.png"), dpi=150)
print("wrote COMPARE.md, references.json, overlay_*.png")
