"""Boundary x coils matrix for precise_QH (2026-10-08): reads every run_al.py result.json (dense check, margins vs
the unbacked-off bounds, solver outcome) and the single-stage log lines; writes matrix.md and matrix.json here.
Run from the sandbox root on CPU (only the boundary volumes are computed)."""
import sys; sys.path.insert(0, "sandbox")   # run from the sandbox root
import boot; boot.setup(default="cpu")
import json, os, re
import common as S
from desc.io import load

M = "work/ss_qh/matrix"
REPS = [("planarN7", "planar N7"), ("arcB2", "arcB2 M5"), ("xyzN6", "FourierXYZ N6")]
ROWS = [  # key, label, folder with the three run_al results, equilibrium, single-stage log (None for B0)
    ("B0", "B0: precise_QH", f"{M}/base", "precise_QH", None),
    ("B_planar", "B_planar (ss planar)", f"{M}/on_planar", f"{M}/ss/planar_k4b/eq_k4.h5", f"{M}/ss/planar_k4b/run.log"),
    ("B_arcB2", "B_arcB2 (ss arcB2)", f"{M}/on_arcB2", f"{M}/ss/arcB2_k4b/eq_k4.h5", f"{M}/ss/arcB2_k4b/run.log"),
    ("B_xyz", "B_xyz (ss XYZ)", f"{M}/on_xyz", f"{M}/ss/xyz_k4b/eq_k4.h5", f"{M}/ss/xyz_k4b/run.log"),
    ("ref_ss_k4", "ref: old arcB2 ss_k4", "work/ss_qh/fixed_ss_eq", "work/ss_qh/arcB2/ss_k4/eq_k4.h5",
     "work/ss_qh/arcB2/ss_k4/run.log"),
]
REF_DIRS = {"planarN7": "al_planarN7", "arcB2": "al_arcB2", "xyzN6": "al_xyzN6"}
DIAG = {"B_planar": "planarN7", "B_arcB2": "arcB2", "B_xyz": "xyzN6"}


def cell(folder, rep, key):
    d = f"{folder}/{REF_DIRS[rep] if key == 'ref_ss_k4' else rep}"
    p = f"{d}/result.json"
    if not os.path.exists(p):
        return dict(missing=True, dir=d, note=open(f"{d}/NOTE.txt").read() if os.path.exists(f"{d}/NOTE.txt") else "")
    j = json.load(open(p))
    s, r = j["solver"], j["result"]
    return dict(dir=d, bn=r["bn"], bn_max=r["bn_max"], margins=j["margins"], active=j["active"], linked=r["linked"],
                message=s["message"], outer=s.get("n_outer"), inner=s.get("nit"), opt=s.get("optimality"),
                viol=s.get("constr_violation"), wall_s=s.get("wall_s"), start_bn=j["start"]["bn"])


def physics(log, first):
    """START physics (first=True) or the last k=4 physics and coils lines of a single-stage log."""
    txt = open(log).read()
    if first:
        ph = re.findall(r"^START physics .*$", txt, re.M)[0]
        co = None
    else:
        ph = re.findall(r"^k=4 physics .*$", txt, re.M)[-1]
        co = re.findall(r"^k=4 coils .*$", txt, re.M)[-1]
    f = lambda pat, s: float(re.search(pat, s).group(1))
    out = dict(A=f(r"A ([\d.]+)", ph), R0=f(r"R0 ([\d.]+)", ph), a=f(r" a ([\d.]+)", ph),
               iota=[float(x) for x in re.search(r"iota \[([\d.]+), ([\d.]+)\]", ph).groups()],
               qs={rho: float(v) for rho, v in re.findall(r"rho ([\d.]+) ([\d.e+-]+)", ph)})
    if co:
        out["ss_coils_bn"] = f(r"bn ([\d.e+-]+)", co)
        out["ss_coils_line"] = co
    return out


rows = []
V0 = None
for key, label, folder, eqp, log in ROWS:
    eq, _ = S.load_equilibrium(eqp)
    V = float(eq.compute("V")["V"])
    V0 = V0 or V
    ph = physics(f"{M}/ss/planar/run.log", True) if key == "B0" else physics(log, False)
    cells = {rep: cell(folder, rep, key) for rep, _ in REPS}
    rows.append(dict(key=key, label=label, eq=eqp, V=V, V_over_V0=V / V0, physics=ph, cells=cells))
    print(label, {r: (c.get("bn"), c.get("message")) for r, c in cells.items()})

b0 = {rep: rows[0]["cells"][rep].get("bn") for rep, _ in REPS}
for row in rows:
    xyz = row["cells"]["xyzN6"].get("bn")
    for rep, _ in REPS:
        c = row["cells"][rep]
        if c.get("missing"):
            continue
        c["over_row_xyz"] = c["bn"] / xyz if xyz else None
        c["over_col_B0"] = c["bn"] / b0[rep] if b0[rep] else None

e = lambda x: "–" if x is None else f"{x:.3e}"
md = "# precise_QH boundary × coils matrix\n\n## ⟨|B·n|⟩/⟨B⟩ after run_al.py refinement (max in parentheses); ÷B0 = same column on precise_QH\n\n"
md += "| boundary ↓ / coils → | " + " | ".join(n for _, n in REPS) + " |\n|---|" + "---|" * len(REPS) + "\n"
for row in rows:
    out = []
    for rep, _ in REPS:
        c = row["cells"][rep]
        if c.get("missing"):
            out.append("missing")
            continue
        star = " **(diag)**" if DIAG.get(row["key"]) == rep else ""
        out.append(f"{c['bn']:.3e} ({c['bn_max']:.1e}), ÷B0 {c['over_col_B0']:.2f}{star}")
    md += f"| {row['label']} | " + " | ".join(out) + " |\n"

md += "\n## Every cell: margins (≤1 for L, κ, κ_MS; ≥1 for d_cc, d_pc), solver\n\n"
md += "| boundary | coils | B·n | ÷ row XYZ | ÷ col B0 | L | κ | κ_MS | d_cc | d_pc | linked | active | solver | outer/inner | opt | viol | wall s |\n"
md += "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
for row in rows:
    for rep, name in REPS:
        c = row["cells"][rep]
        if c.get("missing"):
            md += f"| {row['label']} | {name} | missing | | | | | | | | | | {c['note'][:80]} | | | | |\n"
            continue
        g = c["margins"]
        md += (f"| {row['label']} | {name} | {c['bn']:.3e} | {c['over_row_xyz']:.2f} | {c['over_col_B0']:.2f} | "
               f"{g['L']:.4f} | {g['kappa']:.4f} | {g['kMS']:.3f} | {g['d_cc']:.3f} | {g['d_pc']:.3f} | {c['linked']} | "
               f"{', '.join(c['active'])} | {c['message'].replace('|', '/')} | {c['outer']}/{c['inner']} | "
               f"{e(c['opt'])} | {e(c['viol'])} | {c['wall_s']:.0f} |\n")

md += "\n## Boundaries (Boozer QS = |B_nonsym|/⟨B⟩; single-stage coils = the k=4 coils before refinement)\n\n"
md += "| boundary | QS ρ=0.25 | 0.5 | 0.75 | 1 | A | R0 | a | iota | V/V0 | single-stage coils B·n |\n"
md += "|---|---|---|---|---|---|---|---|---|---|---|\n"
for row in rows:
    p = row["physics"]
    q = [p["qs"].get(k) for k in ("0.25", "0.5", "0.75", "1")]
    md += (f"| {row['label']} | " + " | ".join(e(x) for x in q) + f" | {p['A']:.3f} | {p['R0']:.4f} | {p['a']:.4f} | "
           f"[{p['iota'][0]:.4f}, {p['iota'][1]:.4f}] | {row['V_over_V0']:.4f} | {e(p.get('ss_coils_bn'))} |\n")

open(f"{M}/matrix.md", "w").write(md)
json.dump(rows, open(f"{M}/matrix.json", "w"), indent=1, default=float)
print(md)
