"""How far each cross-section's R extremes sit from z = 0 (a transverse arc cannot overhang its z = 0 chord)."""
import sys; sys.path.insert(0, "sandbox")
import boot; boot.setup(default="cpu")
import numpy as np
import common as S
from desc.grid import LinearGrid as LG
for name in sys.argv[1:]:
    eq, _ = S.load_equilibrium(name)
    a = float(eq.compute("a")["a"])
    zin, zout, cut = [], [], []
    for ph in np.linspace(0, np.pi / eq.NFP, 17):
        g = LG(theta=np.linspace(0, 2 * np.pi, 721), zeta=np.array([ph]), NFP=1)
        d = eq.surface.compute(["R", "Z"], grid=g); R, Z = np.asarray(d["R"]), np.asarray(d["Z"])
        zin.append(Z[R.argmin()]); zout.append(Z[R.argmax()])
        # where z = 0 cuts the section, as a fraction of its R extent (1 = through the widest chord; 0 = misses)
        s = np.sign(Z); k = np.where(s != np.roll(s, -1))[0]
        Rc = [R[j] - Z[j] * (R[(j + 1) % len(R)] - R[j]) / (Z[(j + 1) % len(R)] - Z[j]) for j in k]
        cut.append((max(Rc) - min(Rc)) / (R.max() - R.min()) if len(Rc) >= 2 else 0.0)
    e = np.maximum(np.abs(zin), np.abs(zout)) / a
    print(f"{name:40s} |z| of R-extremes / a: median {np.median(e):.2f} max {e.max():.2f} | z=0 chord / R extent: min {min(cut):.2f} median {np.median(cut):.2f}", flush=True)
