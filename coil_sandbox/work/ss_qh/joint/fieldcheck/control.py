"""Method calibration: put the TARGET's own rho0 surface through the traced-surface pipeline (sampled at the same kind
of scattered (theta, phi) points, fit M=N=FIT, deform the target onto it in 4 re-solved steps, raise to RES, solve).
QS / iota should come back as the target's at rho x rho0; whatever does not is method error.
Usage: control.py CASE EQ_TARGET RHO0 FIT RES [stage2]"""
import sys
import time

sys.path.insert(0, "/home/singh/Documents/DESC2/coil_sandbox")
sys.path.insert(0, "/home/singh/Documents/DESC2/coil_sandbox/sandbox")
import boot  # noqa: E402

boot.setup("gpu")
import numpy as np  # noqa: E402
from desc.geometry import FourierRZToroidalSurface  # noqa: E402
from desc.grid import Grid, QuadratureGrid  # noqa: E402
import common as S  # noqa: E402
import single_stage as SS  # noqa: E402

F = "work/ss_qh/joint/fieldcheck"
case, eq_f = sys.argv[1:3]
rho0 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.95
FIT = int(sys.argv[4]) if len(sys.argv) > 4 else 10
RES = int(sys.argv[5]) if len(sys.argv) > 5 else 12
eq = S.load_equilibrium(eq_f)[0]
NFP = eq.NFP
out = open(f"{F}/{case}/control.txt", "a")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out.write(s + "\n")
    out.flush()


L10 = f"{F}/{case}/control_L{FIT}.h5"
t = time.time()
if len(sys.argv) <= 6 or sys.argv[6] != "stage2":
    # stage 1 (own process): fit + deform at RES = FIT, save. Stage 2 raises to RES in a fresh process: in one process
    # the res-12 solve ran out of GPU memory (the res-10 solves' compiled functions were still resident).
    rng = np.random.default_rng(0)
    n = 9600
    th, ze = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
    d = eq.compute(["R", "Z"], grid=Grid(np.column_stack([np.full(n, rho0), th, ze]), sort=False, jitable=True))
    surf = FourierRZToroidalSurface.from_values(np.column_stack([d["R"], ze, d["Z"]]), th, M=FIT, N=FIT, NFP=NFP,
                                                sym=True, check_orientation=False)
    eqs = eq.copy()
    eqs.change_resolution(L=FIT, M=FIT, N=FIT, L_grid=2 * FIT, M_grid=2 * FIT, N_grid=2 * FIT)
    dR, dZ = surf.R_lmn - eqs.Rb_lmn, surf.Z_lmn - eqs.Zb_lmn
    eqs.solve(ftol=1e-10, xtol=1e-12, gtol=1e-10, maxiter=100, verbose=0)
    for _ in range(4):
        eqs.Rb_lmn = eqs.Rb_lmn + dR * 0.25
        eqs.Zb_lmn = eqs.Zb_lmn + dZ * 0.25
        eqs.solve(ftol=1e-10, xtol=1e-12, gtol=1e-10, maxiter=200, verbose=0)
    eqs.save(L10)
    say(f"control stage 1 saved {L10} ({(time.time() - t) / 60:.1f} min)")
    sys.exit(0)
import desc.io  # noqa: E402

eqs = desc.io.load(L10)
eqs.change_resolution(L=RES, M=RES, N=RES, L_grid=2 * RES, M_grid=2 * RES, N_grid=2 * RES)
eqs.solve(ftol=1e-12, xtol=1e-14, gtol=1e-12, maxiter=500, verbose=0)
eqs.save(f"{F}/{case}/control_L{RES}.h5")
qg = QuadratureGrid(L=eqs.L_grid, M=eqs.M_grid, N=eqs.N_grid, NFP=NFP)
f = eqs.compute(["|F|", "|grad(|B|^2)|/2mu0", "sqrt(g)"], grid=qg)
w = np.abs(f["sqrt(g)"]) * qg.weights
fe = float(np.sum(f["|F|"] * w) / np.sum(f["|grad(|B|^2)|/2mu0"] * w))
ps = SS.Physics(eqs, "1,4")()
SS.Physics.RHO = tuple(rho0 * q for q in (0.25, 0.5, 0.75, 1.0))
pt = SS.Physics(eq, "1,4")()
say(f"=== CONTROL {case}: target's own rho {rho0} surface, fit M=N={FIT}, eq L=M=N={RES}, "
    f"{(time.time() - t) / 60:.1f} min, force {fe:.2e}")
say(f"target at rho x {rho0:g}", SS.fmt_phys(pt))
say("control (pipeline)   ", SS.fmt_phys(ps))
