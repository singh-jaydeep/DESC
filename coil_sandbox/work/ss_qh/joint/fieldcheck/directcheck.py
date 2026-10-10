"""Direct check of the traced-surface QS numbers: evaluate the actual coil |B| (Biot-Savart, currents rescaled as in
fieldcheck.py) at the traced-surface equilibrium's points, and fit Boozer modes in that equilibrium's (theta_B, zeta_B).
If the equilibrium represents the coil field, both fits give the same symmetry-breaking spectrum.
Normalization here is the surface B00 (modes.py uses <|B|>_vol, so values differ by ~1%).
Usage: directcheck.py CASE COILS EQ_TARGET"""
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
case, coils_f, eqt_f = sys.argv[1:4]
eqt = S.load_equilibrium(eqt_f)
eqt = eqt[0] if isinstance(eqt, tuple) else eqt
eqs = desc.io.load(f"{F}/{case}/eq_traced_L12.h5")
cs = desc.io.load(coils_f)
src, _ = S.source_grid(cs, 24, 400)
NFP = eqs.NFP
log = open(f"{F}/{case}/directcheck.txt", "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")


# current scale, as fieldcheck.py (mean |B| on the target boundary)
g = LinearGrid(rho=np.array([1.0]), M=24, N=24, NFP=NFP, sym=False)
d = eqt.compute(["R", "phi", "Z", "|B|"], grid=g)
Bc = np.asarray(cs.compute_magnetic_field(np.column_stack([d["R"], d["phi"], d["Z"]]), basis="rpz", source_grid=src))
scale = float(np.mean(d["|B|"]) / np.mean(np.linalg.norm(Bc, axis=1)))
for c in S.iter_unique(cs):
    c.current = c.current * scale

M, N = 12, 12
mm, nn = np.meshgrid(np.arange(0, M + 1), np.arange(-N, N + 1), indexing="ij")
modes = np.column_stack([mm.ravel(), nn.ravel()])
modes = modes[~((modes[:, 0] == 0) & (modes[:, 1] < 0))]  # cos(m tB - n NFP zB), stellarator symmetric
symm = modes[:, 1] == modes[:, 0]
symm2 = modes[:, 1] == -modes[:, 0]
say(f"=== {case}: coil |B| on the traced-surface equilibrium's surfaces, fit in its Boozer angles (M=N={M})")
say(" rho | breaking L2: eq |B|  coil |B| | ripple |n| 6-9: eq  coil | max | |B|coil-|B|eq |/B00")
for rho in (0.25, 0.5, 0.75, 0.9, 1.0):
    g = LinearGrid(rho=np.array([rho]), M=40, N=40, NFP=NFP, sym=False)
    d = eqs.compute(["R", "phi", "Z", "|B|", "theta_B", "zeta_B"], grid=g, M_booz=16, N_booz=16)
    x = np.column_stack([d["R"], d["phi"], d["Z"]])
    Bcoil = np.linalg.norm(np.asarray(cs.compute_magnetic_field(x, basis="rpz", source_grid=src)), axis=1)
    A = np.cos(np.outer(d["theta_B"], modes[:, 0]) - np.outer(NFP * d["zeta_B"], modes[:, 1]))
    out = []
    for B in (np.asarray(d["|B|"]), Bcoil):
        c = np.linalg.lstsq(A, B, rcond=None)[0]
        b00 = c[(modes[:, 0] == 0) & (modes[:, 1] == 0)][0]
        c = c / b00
        keep = symm if np.sum(np.abs(c[symm])) > np.sum(np.abs(c[symm2])) else symm2
        brk = ~keep
        rip = brk & (np.abs(modes[:, 1]) >= 6) & (np.abs(modes[:, 1]) <= 9)
        out.append((np.sqrt(np.sum(c[brk] ** 2)), np.sqrt(np.sum(c[rip] ** 2)), b00))
    dmax = np.max(np.abs(Bcoil - d["|B|"])) / out[0][2]
    say(f" {rho:.2f} | {out[0][0]:.2e}  {out[1][0]:.2e} | {out[0][1]:.2e}  {out[1][1]:.2e} | {dmax:.2e}")
say("done")
