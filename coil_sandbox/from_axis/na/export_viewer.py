"""Build the offline 3D viewer of near-axis results: every coil, the hinges, the expansion axis, the NAE surface at
r = a, and each run's numbers.

    python export_viewer.py ../viewer.html "runs2/s1_*.json" runs2/qh_z0_nc4.json      (from na/, CPU, ~1 min)
"""
import os, sys, glob, json
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, HERE)
import numpy as np
import jax.numpy as jnp
from nae import NearAxis, elongation
from folded import Layout, coil_arcs, full_coilset


def r3(a):
    return np.round(np.asarray(a, dtype=float), 4).tolist()


def one(jpath):
    d = json.load(open(jpath))
    z = np.load(jpath[:-5] + ".npz")
    ar = d["args"]
    L = Layout(int(z["nfp"]), int(z["nc"]), int(z["K"]), int(z["Kax"]), str(z["mode"]))
    rc, zs, eta, C = L.split(jnp.asarray(z["p"]))
    X, I = full_coilset(C, L)
    X = np.asarray(X)                                   # (2 nfp nc, 2 NPT + 1, 3); first nc are the unique coils
    hinges = []
    for i in range(L.nc):
        (xu, _, _), _, _ = coil_arcs(C[i], L)
        hinges += [np.asarray(xu[0]), np.asarray(xu[-1])]
    # expansion axis and NAE surface (r = a) over the full torus
    na = NearAxis(L.nfp, 41)
    q = na.solve(rc, zs, eta, float(z["hel"]), iota0=ar["iota"])
    th = np.linspace(0, 2 * np.pi, 25)
    X1 = np.asarray(q["X1c"])[:, None] * np.cos(th)
    Y1 = np.asarray(q["Y1s"])[:, None] * np.sin(th) + np.asarray(q["Y1c"])[:, None] * np.cos(th)
    P = np.asarray(q["x"])[:, None] + ar["a"] * (X1[..., None] * np.asarray(q["n"])[:, None]
                                                 + Y1[..., None] * np.asarray(q["b"])[:, None])  # (41, 25, 3)
    ax = np.asarray(q["x"])
    surf, axis = [], []
    for j in range(L.nfp):
        t = 2 * np.pi * j / L.nfp
        R = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1.0]])
        surf.append(P @ R.T)
        axis.append(ax @ R.T)
    surf = np.concatenate(surf + [surf[0][:1]])          # close toroidally
    axis = np.concatenate(axis + [axis[0][:1]])
    fi = d["final"]
    name = os.path.basename(jpath)[:-5]
    info = dict(name=name, nfp=L.nfp, nc=L.nc, mode=L.mode, n_coils=int(X.shape[0]), hel=fi["hel"],
                iota_target=ar["iota"], iota=fi["iota"], cost=fi["cost"], dB=fi["max_dB"], dG=fi["max_dG_rel"],
                elong=fi["max_elong"], absZ=fi["max_abs_Zaxis"], R1=fi["axis_rc"][1], Z1=fi["axis_zs"][1],
                eta=fi["eta"], a=ar["a"], dpc=fi["min_dpc"], dpc_bound=ar["dpc"], dcc=fi["min_dcc"],
                dcc_bound=ar["dcc"], kappa=fi["max_kappa"], kappa_bound=ar["kmax"], Lmax=max(fi["Lcoil"]),
                L_bound=ar["Lmax"], I=fi["I_over_uniform"], optimality=fi.get("optimality"), nit=fi.get("nit"),
                seed=ar.get("seed"))
    return dict(info=info, coils=r3(X[:, ::2]), hinges=r3(np.array(hinges)), axis=r3(axis), surf=r3(surf))


def main():
    out = sys.argv[1]
    files = []
    for pat in sys.argv[2:]:
        files += sorted(glob.glob(pat))
    data = []
    for f in files:
        print("  ", f, flush=True)
        data.append(one(f))
    html = open(os.path.join(HERE, "viewer_template.html")).read().replace("__DATA__", json.dumps(data, separators=(",", ":")))
    open(out, "w").write(html)
    print(f"wrote {out} ({len(html) / 1024:.0f} kB, {len(data)} configs)")


if __name__ == "__main__":
    main()
