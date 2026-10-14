"""Fixed-boundary vacuum re-solve of an equilibrium's boundary at several spectral resolutions (boundary unchanged,
p = 0, I = 0, Psi fixed): does the interior (iota, vacuum residual mu0|J|a/|B|) converge with the basis?
Also re-solves at the input's own basis with tight tolerances, to separate convergence from resolution.

Usage (from ss/): python resladder.py EQ.h5 OUTDIR [--lmn 12,14,16] [--maxiter 300] [--chunk 200]
Writes OUTDIR/eq_bK.h5 and OUTDIR/resladder.json."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, "../..")
sys.path.insert(0, "../../sandbox")
import boot  # noqa: E402

boot.setup("gpu")
import numpy as np  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.objectives import ForceBalance, ObjectiveFunction  # noqa: E402
import common as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("eq")
ap.add_argument("out")
ap.add_argument("--lmn", default="12,14,16")
ap.add_argument("--maxiter", type=int, default=300)
ap.add_argument("--chunk", type=int, default=200)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
eq0 = S.load_equilibrium(a.eq)
eq0 = eq0[0] if isinstance(eq0, tuple) else eq0
RHO = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])


def diag(eq):
    ig = LinearGrid(rho=RHO, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
    iota = np.asarray(ig.compress(eq.compute("iota", grid=ig)["iota"]))
    av = float(eq.compute("a")["a"])
    jr = []
    for r in RHO[1:]:
        g = LinearGrid(rho=np.array([r]), M=2 * eq.M_grid, N=2 * eq.N_grid, NFP=eq.NFP, sym=False)
        d = eq.compute(["|J|", "|B|"], grid=g)
        jr.append(float(np.sqrt(np.mean((4e-7 * np.pi * np.asarray(d["|J|"]) * av / np.asarray(d["|B|"])) ** 2))))
    da = eq.compute(["R", "Z"], grid=LinearGrid(rho=np.array([0.0]), theta=np.array([0.0]), zeta=np.array([0.0]),
                                                NFP=eq.NFP))
    return dict(iota=iota.tolist(), J_res=jr, axis_R=float(da["R"][0]))


def show(tag, d, extra=""):
    print(f"{tag:>10}: iota rho {'/'.join(f'{r:g}' for r in RHO)} = {' '.join(f'{v:.4f}' for v in d['iota'])} | "
          f"mu0|J|a/|B| rho .1-1 = {' '.join(f'{v:.1e}' for v in d['J_res'])} | axis R {d['axis_R']:.5f} {extra}",
          flush=True)


res = {"eq": a.eq, "input": dict(L=eq0.L, M=eq0.M, N=eq0.N, **diag(eq0)), "solves": []}
show("input", res["input"], f"(L,M,N {eq0.L},{eq0.M},{eq0.N})")
for K in [int(v) for v in a.lmn.split(",")]:
    eq = eq0.copy()
    eq.change_resolution(L=K, M=K, N=K, L_grid=2 * K, M_grid=2 * K, N_grid=2 * K)
    t = time.time()
    try:
        _, info = eq.solve(objective=ObjectiveFunction(ForceBalance(eq, jac_chunk_size=a.chunk)), verbose=1,
                           maxiter=a.maxiter, ftol=1e-12, xtol=1e-14, gtol=1e-12, copy=False)
        msg, nit = str(info.get("message", "")), int(info.get("nit", -1))
    except Exception as e:  # noqa: BLE001  e.g. out of memory at the top of the ladder
        print(f"basis {K} FAILED: {e!r}"[:400], flush=True)
        res["solves"].append(dict(K=K, failed=repr(e)[:300]))
        break
    eq.save(os.path.join(a.out, f"eq_b{K}.h5"))
    d = dict(K=K, minutes=(time.time() - t) / 60, message=msg, nit=nit, **diag(eq))
    res["solves"].append(d)
    show(f"basis {K}", d, f"({d['minutes']:.1f} min, nit {nit}, {msg})")
    json.dump(res, open(os.path.join(a.out, "resladder.json"), "w"), indent=1)
