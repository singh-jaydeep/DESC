"""QS metrics as they appear in the literature, per flux surface, for our state and DESC reference QA equilibria.

    python qs_metrics.py OUT.json NAME=PATH_or_example[:M,N] ...     (helicity M,N default 1,0)

Per surface (rho = 0.25, 0.5, 0.75, 1; s = rho^2):
  booz_max   max |B_mn| over symmetry-breaking Boozer modes / B_00   (Landreman & Paul 2022 style spectra plots)
  booz_l2    sqrt(sum B_mn^2 over breaking modes) / B_00             (what this project has been quoting, / <B>)
  two_term   sqrt(< (f_C / B^3)^2 >)   f_C = (M iota - N)(B x grad psi).grad B - (M G + N I) B.grad B
             (the integrand of Landreman & Paul's f_QS; f_QS = sum over surfaces of the square)
  eps_eff    effective ripple eps_eff (Nemov), DESC "effective ripple" via Bounce2D
Boozer resolution M_booz = N_booz = 2 x the equilibrium's M, N; grids 2 x M_grid, N_grid.
"""
import sys, json
sys.path.insert(0, "../../sandbox")
import boot; boot.setup(default="gpu")
import numpy as np
import desc.examples
from desc.io import load
from desc.grid import LinearGrid

RHO = np.array([0.25, 0.5, 0.75, 1.0])
out = {}
for spec in sys.argv[2:]:
    name, rest = spec.split("=", 1)
    path, hel = (rest.rsplit(":", 1) + ["1,0"])[:2] if ":" in rest else (rest, "1,0")
    M, N = (int(v) for v in hel.split(","))
    eq = load(path) if path.endswith(".h5") else desc.examples.get(path)
    if hasattr(eq, "__len__"):
        eq = eq[-1]
    d0 = eq.compute(["R0/a", "<|B|>_vol", "a", "R0"])
    r = dict(A=float(d0["R0/a"]), B_vol=float(d0["<|B|>_vol"]), NFP=int(eq.NFP), helicity=[M, N], rho=RHO.tolist(),
             booz_max=[], booz_l2=[], two_term=[], iota=[])
    for rho in RHO:
        g = LinearGrid(rho=np.array([rho]), M=2 * eq.M_grid, N=2 * eq.N_grid, NFP=eq.NFP, sym=False)
        d = eq.compute(["|B|_mn_B", "B modes", "f_C", "|B|", "sqrt(g)", "iota"], grid=g, M_booz=2 * eq.M,
                       N_booz=2 * eq.N, helicity=(M, N))
        bmn = np.asarray(d["|B|_mn_B"]).ravel()
        modes = np.asarray(d["B modes"])
        m, n = modes[:, 1], modes[:, 2]
        b00 = float(bmn[(m == 0) & (n == 0)][0])
        # symmetric modes: m N_hel NFP... DESC convention: B = B(M theta - N zeta) -> keep modes with n M == m N (n in
        # units of NFP for N = NFP multiples); QA (M, N) = (1, 0): n == 0
        sym = (n * M * eq.NFP == m * N) if N != 0 else (n == 0)
        br = bmn[~sym]
        r["booz_max"].append(float(np.abs(br).max() / abs(b00)))
        r["booz_l2"].append(float(np.linalg.norm(br) / abs(b00)))
        w = np.abs(np.asarray(d["sqrt(g)"])) * np.asarray(g.weights)
        f = np.asarray(d["f_C"]) / np.asarray(d["|B|"]) ** 3
        r["two_term"].append(float(np.sqrt(np.sum(w * f**2) / np.sum(w))))
        r["two_term_median_abs"] = r.get("two_term_median_abs", []) + [float(np.median(np.abs(f)))]
        r["iota"].append(float(np.asarray(d["iota"])[0]))
    try:
        from desc.integrals import Bounce2D
        rr = RHO[:3]  # eps_eff needs field lines on closed surfaces; skip the folded edge
        ge = LinearGrid(rho=rr, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
        ang = Bounce2D.angle(eq, X=32, Y=32, rho=rr)
        de = eq.compute("effective ripple", grid=ge, angle=ang, Y_B=128, num_transit=20, num_well=20 * 30,
                        num_quad=32, num_pitch=45)
        r["eps_eff_rho"] = rr.tolist()
        r["eps_eff"] = np.asarray(ge.compress(de["effective ripple"])).tolist()
    except Exception as e:
        r["eps_eff_error"] = repr(e)[:300]
    out[name] = r
    print(name, json.dumps(r), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
