import sys; sys.path.insert(0, "../../sandbox")
import boot; boot.setup(default="cpu")
import numpy as np, desc.examples
from desc.io import load
from desc.grid import LinearGrid as LG
for spec in sys.argv[1:]:
    name, path = spec.split("=", 1)
    eq = load(path) if path.endswith(".h5") else desc.examples.get(path)
    if hasattr(eq, "__len__"): eq = eq[-1]
    d0 = eq.compute(["a", "R0", "R0/a"]); A_ = float(d0["a"])
    zin, zout, cut, zc, el = [], [], [], [], []
    for ph in np.linspace(0, np.pi / eq.NFP, 17):
        g = LG(theta=np.linspace(0, 2 * np.pi, 721), zeta=np.array([ph]), NFP=1)
        s = eq.surface.compute(["R", "Z"], grid=g); R, Z = np.asarray(s["R"]), np.asarray(s["Z"])
        zin.append(Z[R.argmin()]); zout.append(Z[R.argmax()]); zc.append((Z.max() + Z.min()) / 2)
        sg = np.sign(Z); k = np.where(sg != np.roll(sg, -1))[0]
        Rc = [R[i] - Z[i] * (R[(i + 1) % len(R)] - R[i]) / (Z[(i + 1) % len(R)] - Z[i]) for i in k]
        cut.append((max(Rc) - min(Rc)) / (R.max() - R.min()) if len(Rc) >= 2 else 0.0)
        el.append((Z.max() - Z.min()) / (R.max() - R.min()))
    e = np.maximum(np.abs(zin), np.abs(zout)) / A_
    ga = LG(rho=np.array([0.0]), theta=np.array([0.0]), zeta=np.linspace(0, 2 * np.pi / eq.NFP, 64), NFP=eq.NFP)
    ax = eq.compute(["R", "Z"], grid=ga)
    io = LG(rho=np.linspace(0, 1, 11), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
    iota = io.compress(eq.compute("iota", grid=io)["iota"])
    print(f"{name:12s} NFP {eq.NFP} A {float(d0['R0/a']):.2f} iota {abs(iota[0]):.3f}->{abs(iota[-1]):.3f} | axis max|Z|/a {np.abs(ax['Z']).max() / A_:.2f}  axis dR/a {(ax['R'].max() - ax['R'].min()) / A_:.2f} | "
          f"cut_min {min(cut):.2f} med {np.median(cut):.2f} | ext_med {np.median(e):.2f} max {e.max():.2f} | zc_max {np.abs(zc).max() / A_:.2f} | height/width med {np.median(el):.2f} max {max(el):.2f}", flush=True)
