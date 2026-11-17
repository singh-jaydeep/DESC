"""Faces-sweep table: one CSV row per run and the per-class summary (stdlib only).

    python paper/faces/table.py [RUNS_DIR] [--tol 0.01]   (default paper/runs/faces)

Reads RUNS_DIR/<cfg>_s<seed>/<class>/{score.json,result.json}; writes
RUNS_DIR/faces.csv and prints a markdown table per config and class: rows, converged,
feasible strict / within --tol, min / median / max of bn_mean, and the class median
and best over the modular (smooth-coil) ones. converged = gtol reached (the AL's
success flag).
From coil_sandbox/from_axis/rr02/{pareto,summarize}.py (js/qa-poster).
"""

import csv
import glob
import json
import os
import statistics as st
import sys

args = [v for v in sys.argv[1:] if not v.startswith("--")]
runs = args[0] if args else os.path.join("paper", "runs", "faces")
tol = float(sys.argv[sys.argv.index("--tol") + 1]) if "--tol" in sys.argv else 0.01
rows = []
for f in sorted(glob.glob(os.path.join(runs, "*_s*", "*", "score.json"))):
    d = os.path.dirname(f)
    cls, seedd = os.path.basename(d), os.path.basename(os.path.dirname(d))
    cfg, seed = seedd.split("_s")
    sc = json.load(open(f))
    rj = os.path.join(d, "result.json")
    sv = json.load(open(rj))["solver"] if os.path.exists(rj) else {}
    rows.append(
        dict(
            config=f"precise_{cfg}",
            seed=seed,
            coil_class=cls.rstrip("0123456789"),
            n_faces=(
                "1" if cls == "planar1" else (cls[3:] if cls.startswith("arc") else "")
            ),
            converged=str(bool(sv.get("success"))).lower(),
            feasible=str(sc["feasible"]).lower(),
            bn_mean=sc["bn_mean"],
            bn_rms=sc["bn_rms"],
            bn_max=sc["bn_max"],
            L_over_bound=sc["L_over_bound"],
            kappa_over_bound=sc["kappa_over_bound"],
            dcc_over_bound=sc["dcc_over_bound"],
            dpc_over_bound=sc["dpc_over_bound"],
            linked=sc["linked"],
            fold_deg_max=sc.get("fold_deg_max", ""),
            corner_deg=sc.get("corner_deg", ""),
            dof_per_coil=sc["dof_per_coil"],
            outer_iterations=sv.get("n_outer", ""),
            inner_iterations=sv.get("nit", ""),
            wall_s=sv.get("wall_s", ""),
        )
    )
if not rows:
    raise SystemExit(f"no score.json under {runs}")
with open(os.path.join(runs, "faces.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)


def order(x):
    """Sort key: planar first, then arcs by faces, modular last."""
    c = x["coil_class"]
    return 0 if c == "planar" else 99 if c == "modular" else int(x["n_faces"])


def within(x):
    """Feasible within tol of every bound and unlinked."""
    return (
        x["L_over_bound"] <= 1 + tol
        and x["kappa_over_bound"] <= 1 + tol
        and x["dcc_over_bound"] >= 1 - tol
        and x["dpc_over_bound"] >= 1 - tol
        and int(x["linked"]) == 0
    )


print(
    f"| config | class | faces | rows | converged | feasible / within {100 * tol:g}% "
    "| bn_mean min / median / max | median / modular median | best / best modular |"
)
print("|---|---|---|---|---|---|---|---|---|")
for cfg in sorted({x["config"] for x in rows}):
    g = [x for x in rows if x["config"] == cfg]
    mod = [x["bn_mean"] for x in g if x["coil_class"] == "modular"]
    mref = st.median(mod) if mod else float("nan")
    mbest = min(mod) if mod else float("nan")
    for k in sorted({order(x) for x in g}):
        c = [x for x in g if order(x) == k]
        b = sorted(x["bn_mean"] for x in c)
        print(
            f"| {cfg} | {c[0]['coil_class']} | {c[0]['n_faces'] or '-'} | "
            f"{len(c)} | "
            f"{sum(x['converged'] == 'true' for x in c)} | "
            f"{sum(x['feasible'] == 'true' for x in c)} / "
            f"{sum(within(x) for x in c)} | "
            f"{b[0]:.3e} / {st.median(b):.3e} / {b[-1]:.3e} | "
            f"{st.median(b) / mref:.3f} | {b[0] / mbest:.3f} |"
        )
print(f"\n{len(rows)} rows; wrote {os.path.join(runs, 'faces.csv')}")
