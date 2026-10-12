"""Normalized B.n of the coils on a surface: mean |B.n|/<|B|> (area-weighted), rms, max, percentiles, where the max is.
    python bn_report.py NAME=EQ.h5@COILS.h5 ..."""
import sys, json
sys.path.insert(0, "../../sandbox")
import boot; boot.setup(default="gpu")
import numpy as np
from desc.io import load
from desc.grid import LinearGrid
import common as S
out = {}
for spec in sys.argv[1:]:
    name, rest = spec.split("=", 1); ep, cp = rest.split("@")
    eq, cs = load(ep), load(cp)
    src = S.source_grid(cs, gl_m=64)[0]
    g = LinearGrid(rho=np.array([1.0]), M=96, N=96, NFP=eq.NFP, sym=False)
    d = eq.compute(["x", "n_rho", "|e_theta x e_zeta|", "R", "Z", "sqrt(g)"], grid=g, basis="xyz")
    B = np.asarray(cs.compute_magnetic_field(np.asarray(d["x"]), basis="xyz", source_grid=src))
    Bm = np.linalg.norm(B, axis=1)
    bn = np.sum(B * np.asarray(d["n_rho"]), 1)
    w = np.asarray(g.weights) * np.asarray(d["|e_theta x e_zeta|"]); w = w / w.sum()
    Bavg = np.sum(w * Bm)
    a = np.abs(bn)
    j = np.argmax(a)
    order = np.argsort(a); cw = np.cumsum(w[order])
    pct = {p: float(a[order][np.searchsorted(cw, p / 100)] / Bavg) for p in (50, 90, 99, 99.9)}
    local = a / Bm
    r = dict(mean=float(np.sum(w * a) / Bavg), rms=float(np.sqrt(np.sum(w * bn**2)) / Bavg), max=float(a.max() / Bavg),
             max_local=float(local.max()), pct=pct, area_frac_above_1pct=float(np.sum(w[a / Bavg > 0.01])),
             area_frac_above_0p5pct=float(np.sum(w[a / Bavg > 0.005])),
             max_at=dict(theta=float(g.nodes[j, 1]), zeta=float(g.nodes[j, 2]), R=float(np.hypot(*np.asarray(d["x"])[j, :2])),
                         Z=float(np.asarray(d["x"])[j, 2]), sqrtg_sign=float(np.sign(np.asarray(d["sqrt(g)"])[j]))),
             neg_jac_frac=float((np.asarray(d["sqrt(g)"]) < 0).mean()),
             max_excluding_folds=float(a[np.asarray(d["sqrt(g)"]) > 0.2 * np.median(np.asarray(d["sqrt(g)"]))].max() / Bavg))
    out[name] = r
    print(name, json.dumps(r), flush=True)
json.dump(out, open("qa2best/bn_report.json", "w"), indent=1)
