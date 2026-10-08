"""Raise the mode count M of a polar-arc coilset WITHOUT changing its geometry.

Each arc is r(phi) = |chord|/2 + sum_m a_m sin(m pi phi), m = 1..M, so appending
zero-amplitude higher modes leaves the curve pointwise identical while adding DOF.
Verified here by a dense pointwise comparison before the file is written.

    python work/helios/arc16/raise_M.py IN.h5 OUT.h5 NEW_M
"""
import sys, os
os.chdir("/home/singh/Documents/DESC2/coil_sandbox"); sys.path.insert(0, "sandbox")
os.environ["SANDBOX_DEVICE"] = "cpu"
import boot; boot.setup(default="cpu")
import numpy as np
import common as S
from desc.coils import CoilSet, MixedCoilSet, PolarPlanarArcCoil
from desc.grid import LinearGrid
from desc.io import load

src, out, newM = sys.argv[1], sys.argv[2], int(sys.argv[3])
cs = load(src)
eq = load("sample_equilibria/helios_repro.h5")
uniq = list(S.iter_unique(cs))
B = int(uniq[0].B); M = int(uniq[0].M)
if newM <= M:
    raise SystemExit(f"new M ({newM}) must exceed current M ({M})")
print(f"{src}: B={B} M={M} -> M={newM}")

coils, devs = [], []
for c in uniq:
    a = np.asarray(c.shape).reshape(B, M)
    a2 = np.zeros((B, newM)); a2[:, :M] = a          # pad higher modes with zero
    new = PolarPlanarArcCoil(current=float(c.current), hinges=np.asarray(c.hinges),
                             tilts=np.asarray(c.tilts), shape=a2.ravel(), B=B, M=newM)
    # The arc FRAME depends on arc_ref (a parameter, not a constructor argument). Without
    # carrying it over the arcs are placed against a freshly-chosen reference and the curve
    # moves by metres, so copy it -- and the rigid-body shift/rotmat -- explicitly.
    new.arc_ref = np.asarray(c.arc_ref)
    new.shift = np.asarray(c.shift)
    new.rotmat = np.asarray(c.rotmat)
    g = LinearGrid(N=2000)
    p0 = np.asarray(c.compute("x", grid=g, basis="xyz")["x"])
    p1 = np.asarray(new.compute("x", grid=g, basis="xyz")["x"])
    devs.append(float(np.linalg.norm(p1 - p0, axis=1).max()))
    coils.append(new)
print("  max pointwise deviation per coil: " + ", ".join(f"{d*1e6:.3f} um" for d in devs))
if max(devs) > 1e-6:
    raise SystemExit("geometry changed by more than 1 um -- not writing")

new_cs = MixedCoilSet(*[CoilSet(c, NFP=eq.NFP, sym=eq.sym) for c in coils])
m0 = S.evaluate(cs, eq, vacuum=False); m1 = S.evaluate(new_cs, eq, vacuum=False)
print(f"  bn {m0['bn']:.6e} -> {m1['bn']:.6e}   (identical geometry, so identical field)")
print(f"  free params per unique coil: {B*(M+2)+3*B if False else uniq[0].dim_x} -> {coils[0].dim_x}")
new_cs.save(out)
print(f"wrote {out}")
