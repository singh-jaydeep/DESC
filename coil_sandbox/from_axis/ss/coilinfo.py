import sys
sys.path.insert(0, "../../sandbox")
import boot; boot.setup(default="cpu")
import numpy as np
from desc.io import load
import common as S
for f in sys.argv[1:]:
    cs = load(f); I = []
    for c in S.iter_unique(cs):
        h = np.asarray(c.hinges).reshape(-1, 3); t = np.degrees(np.asarray(c.tilts))
        I.append(float(c.current))
        print(f, "hinges R,phi,z", [(round(float(np.hypot(x, y)), 4), round(float(np.arctan2(y, x)), 4), round(float(z), 6)) for x, y, z in h],
              "tilts deg", t.round(1).tolist(), "fold deg", round(float(t[1] - t[0] - 180), 1), "I/mean", None)
    I = np.array(I); print(" currents A", I.round(0).tolist(), "I/mean", (I / I.mean()).round(3).tolist())
eq = load("qa2best/ss7_b14_fb/eq_fb.h5")
print("Psi", float(eq.Psi), "L,M,N", eq.L, eq.M, eq.N, "NFP", eq.NFP)
