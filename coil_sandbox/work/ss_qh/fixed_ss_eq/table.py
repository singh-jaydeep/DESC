"""Numbers table for the fixed single-stage QH boundary: dense check of every coilset (vacuum),
margins against the unbacked-off precise_qh_gil bounds. Writes table.md and table.json here."""
import sys; sys.path.insert(0, "sandbox")   # run from the sandbox root
import boot; boot.setup(default="cpu")
import json, os
import common as S
from desc.io import load

HERE = "work/ss_qh/fixed_ss_eq"
EQ = "work/ss_qh/arcB2/ss_k4/eq_k4.h5"
BOUNDS = "bounds/precise_qh_gil.json"
SETS = [  # label, path  (stage 2 with lsq-auglag-composite, 2026-10-07; old-solver table in table_oldsolver.md)
    ("arcB2 M5, single-stage start", "work/ss_qh/arcB2/ss_k4/coils_k4.h5"),
    ("arcB2 M5, stage 2", f"{HERE}/al_arcB2/result.h5"),
    ("planar N7, stage 2 (cold start)", f"{HERE}/al_planarN7/result.h5"),
    ("FourierXYZ N6, stage 2", f"{HERE}/al_xyzN6/result.h5"),
]
ACTIVE_TOL = 5e-3  # a margin within 0.5% of its bound counts as active

eq, _ = S.load_equilibrium(EQ)
tree = S.surface_tree(eq)
rows = []
for label, path in SETS:
    if not os.path.exists(path):
        print(f"skip {label}: {path} missing")
        continue
    cs = load(path)
    m = S.evaluate(cs, eq, tree=tree, vacuum=True)
    g = S.margins(m, S.paper_bounds(eq, S.n_unique(cs), None, BOUNDS))
    active = [k for k, v in g.items() if abs(v - 1) < ACTIVE_TOL]
    bad = [k for k, v in g.items() if (v < 1 - 1e-4 if k in ("d_cc", "d_pc") else v > 1 + 1e-4)]
    rows.append(dict(label=label, path=path, bn=m["bn"], bn_max=m["bn_max"], margins=g, linked=m["linked"],
                     active=active, over=bad))
    print(label, f"bn {m['bn']:.3e}", S.report(m, S.paper_bounds(eq, S.n_unique(cs), None, BOUNDS)), active, bad)

ref = next((r["bn"] for r in rows if r["label"].startswith("FourierXYZ")), None)
hdr = "| coilset | ⟨\\|B·n\\|⟩/⟨B⟩ | max | ÷ XYZ | L | κ | κ_MS | d_cc | d_pc | linked | active | over by >1e-4 |\n"
hdr += "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
body = ""
for r in rows:
    g = r["margins"]
    ratio = f"{r['bn'] / ref:.2f}" if ref else "–"
    body += (f"| {r['label']} | {r['bn']:.3e} | {r['bn_max']:.2e} | {ratio} | {g['L']:.3f} | {g['kappa']:.3f} | "
             f"{g['kMS']:.3f} | {g['d_cc']:.3f} | {g['d_pc']:.3f} | {r['linked']} | {', '.join(r['active']) or '–'} | "
             f"{', '.join(r["over"]) or 'none'} |\n")
open(f"{HERE}/table.md", "w").write(hdr + body)
json.dump(rows, open(f"{HERE}/table.json", "w"), indent=1)
print(hdr + body)
