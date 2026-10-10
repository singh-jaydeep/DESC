"""Converge the traced-surface equilibrium further and re-validate.
Usage: polish.py CASE EQ_TARGET IN.h5 OUT.h5 [RES] [MAXITER]
  RES: raise L=M=N to RES (grids 2x) before solving (boundary keeps its fitted modes).
Reports relative force error, Boozer QS / iota, and the validation table (inner traced lines' rho spread in this
equilibrium's coordinates; iota trace vs this equilibrium)."""
import json
import sys
import time

sys.path.insert(0, "/home/singh/Documents/DESC2/coil_sandbox")
sys.path.insert(0, "/home/singh/Documents/DESC2/coil_sandbox/sandbox")
import boot  # noqa: E402

boot.setup("gpu")
import numpy as np  # noqa: E402
import desc.io  # noqa: E402
from desc.grid import LinearGrid, QuadratureGrid  # noqa: E402
import common as S  # noqa: E402
import single_stage as SS  # noqa: E402

F = "work/ss_qh/joint/fieldcheck"
case, eqt_f, inp, outp = sys.argv[1:5]
RES = int(sys.argv[5]) if len(sys.argv) > 5 else 0
MAXIT = int(sys.argv[6]) if len(sys.argv) > 6 else 500
eqt = S.load_equilibrium(eqt_f)[0]
eqs = desc.io.load(inp)
logf = open(f"{F}/{case}/polish.txt", "a")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


def force(e):
    qg = QuadratureGrid(L=e.L_grid, M=e.M_grid, N=e.N_grid, NFP=e.NFP)
    f = e.compute(["|F|", "|grad(|B|^2)|/2mu0", "sqrt(g)"], grid=qg)
    w = np.abs(f["sqrt(g)"]) * qg.weights
    return float(np.sum(f["|F|"] * w) / np.sum(f["|grad(|B|^2)|/2mu0"] * w)), bool(np.all(f["sqrt(g)"] > 0))


say(f"=== {case}: polish {inp} -> {outp}  RES {RES or 'same'}  maxiter {MAXIT}")
fe0 = force(eqs)
say(f"start: L,M,N {eqs.L},{eqs.M},{eqs.N}  force {fe0[0]:.2e}  sqrt(g)>0 {fe0[1]}")
if RES:
    eqs.change_resolution(L=RES, M=RES, N=RES, L_grid=2 * RES, M_grid=2 * RES, N_grid=2 * RES)
t = time.time()
eqs.solve(ftol=1e-12, xtol=1e-14, gtol=1e-12, maxiter=MAXIT, verbose=1)
fe = force(eqs)
say(f"after solve ({(time.time() - t) / 60:.1f} min): L,M,N {eqs.L},{eqs.M},{eqs.N}  force {fe[0]:.2e}  "
    f"sqrt(g)>0 {fe[1]}")
eqs.save(outp)
ps = SS.Physics(eqs, "1,4")()
say("coil field (trc)", SS.fmt_phys(ps))
d = np.load(f"{F}/{case}/trace.npz")
tr = json.load(open(f"{F}/{case}/result.json")).get("trace", [])
say(" rho0 | spread @rho | iota trace  this-eq")
rows = []
for j, lab in enumerate(list(d["labels"])):
    if lab > 0.9 + 1e-9:
        continue
    sub = np.arange(0, len(d["phis"]), 2)[:: max(1, len(d["phis"]) // 800)]
    x = np.column_stack([d["r"][sub, j], d["phis"][sub], d["z"][sub, j]])
    fc = np.asarray(eqs.map_coordinates(x, inbasis=("R", "phi", "Z"), outbasis=("rho", "theta", "zeta"),
                                        maxiter=60, tol=1e-9))
    ok = np.isfinite(fc[:, 0])
    rr = fc[ok, 0]
    rr = rr[np.abs(rr - np.median(rr)) < 0.1]  # drop map_coordinates failures
    it_s = float(eqs.compute("iota", grid=LinearGrid(rho=np.array([np.mean(rr)]), NFP=eqs.NFP))["iota"][0])
    row = next((q for q in tr if abs(q["rho0"] - lab) < 1e-9), {})
    rows.append(dict(rho0=float(lab), spread=float(np.ptp(rr)), rho=float(np.mean(rr)), iota_trace=row.get("iota_trace"),
                     iota_eq=it_s))
    say(f" {lab:4.2f} | {np.ptp(rr):.4f} @{np.mean(rr):.3f} | {row.get('iota_trace', float('nan')):.4f}  {it_s:.4f}")
json.dump(dict(case=case, file=outp, force=fe[0], physics=ps, validation=rows),
          open(outp.replace(".h5", ".json"), "w"), indent=1, default=float)
