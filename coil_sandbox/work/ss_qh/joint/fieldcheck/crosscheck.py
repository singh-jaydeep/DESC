"""Do the traced coil-field lines lie on the FREE-BOUNDARY equilibrium's surfaces better than on the target's?
For each case and line (rho0 <= 1): rho spread (max - min) in target and in free-boundary coordinates, and iota
(trace vs target vs free boundary at the line's mean free-boundary rho).  Usage: crosscheck.py CASE EQ_TARGET"""
import sys

sys.path.insert(0, "/home/singh/Documents/DESC2/coil_sandbox")
sys.path.insert(0, "/home/singh/Documents/DESC2/coil_sandbox/sandbox")
import boot  # noqa: E402

boot.setup("gpu")
import numpy as np  # noqa: E402
import desc.io  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
import common as S  # noqa: E402

F = "work/ss_qh/joint/fieldcheck"
case, eqt_f = sys.argv[1:3]
eqt = S.load_equilibrium(eqt_f)[0]
eqf = desc.io.load(f"{F}/{case}/eq_freeb.h5")
d = np.load(f"{F}/{case}/trace.npz")
r, z, phis, labels = d["r"], d["z"], d["phis"], list(d["labels"])
print(f"== {case}   (spread = max - min of the punctures' rho; iota from the unwrapped theta)")
print(" rho0 | spread target  spread freeb | iota trace  target   freeb@rho")
for j, lab in enumerate(labels):
    if lab > 1.0 + 1e-9:
        continue
    sub = np.arange(0, len(phis), 2)[:: max(1, len(phis) // 800)]
    x = np.column_stack([r[sub, j], phis[sub], z[sub, j]])
    out = []
    for e in (eqt, eqf):
        fc = np.asarray(e.map_coordinates(x, inbasis=("R", "phi", "Z"), outbasis=("rho", "theta", "zeta"),
                                          maxiter=60, tol=1e-9))
        out.append(fc)
    sp = [float(np.ptp(fc[:, 0])) for fc in out]
    thu = np.unwrap(out[1][:, 1])
    it_tr = (thu[-1] - thu[0]) / (phis[sub][-1] - phis[sub][0])
    rm = [float(np.mean(fc[:, 0])) for fc in out]
    it = [float(e.compute("iota", grid=LinearGrid(rho=np.array([min(rv, 1.0)]), NFP=e.NFP))["iota"][0])
          for e, rv in zip((eqt, eqf), rm)]
    print(f" {lab:4.2f} | {sp[0]:.4f}        {sp[1]:.4f}      | {it_tr:.4f}     {it[0]:.4f}   {it[1]:.4f}@{rm[1]:.3f}")
