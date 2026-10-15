"""Does a vacuum equilibrium's INTERIOR field equal its coils' field? (VacuumBoundaryError only sees the boundary.)

For each rho: rms and max of |B_coil - B_eq| / |B_eq| over a (theta, zeta) grid, and the rms angle between them;
plus the vacuum residual mu0 |J| a / |B| (zero for an exact vacuum solution). Coil currents are scaled uniformly so that
<|B_coil|> = <|B_eq|> on the boundary (the scale is invisible to field lines and to B.n).

Usage (from ss/): python interior_check.py COILS.h5 EQ1.h5[=LABEL] [EQ2.h5[=LABEL] ...] [--json OUT.json]"""
import json
import sys

sys.path.insert(0, "../..")
sys.path.insert(0, "../../sandbox")
import boot  # noqa: E402

boot.setup("gpu")
import numpy as np  # noqa: E402
import desc.io  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
import common as S  # noqa: E402

args = sys.argv[1:]
out = None
if "--json" in args:
    i = args.index("--json")
    out = args[i + 1]
    del args[i:i + 2]
coils_f, specs = args[0], args[1:]
cs = desc.io.load(coils_f)
src, _ = S.source_grid(cs, 64, 400)
RHO = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
res = {"coils": coils_f, "eqs": {}}
for spec in specs:
    f, lab = (spec.split("=", 1) + [spec])[:2]
    eq = S.load_equilibrium(f)
    eq = eq[0] if isinstance(eq, tuple) else eq
    a = float(eq.compute("a")["a"])
    rows = []
    scale = None
    for rho in RHO[::-1]:  # boundary first: it sets the current scale
        g = LinearGrid(rho=np.array([rho]), M=2 * eq.M_grid, N=2 * eq.N_grid, NFP=eq.NFP, sym=False)
        d = eq.compute(["x", "B", "|B|", "|J|"], grid=g, basis="xyz")
        Bc = np.asarray(cs.compute_magnetic_field(np.asarray(d["x"]), basis="xyz", source_grid=src))
        Be = np.asarray(d["B"])
        if scale is None:
            scale = float(np.mean(np.asarray(d["|B|"])) / np.mean(np.linalg.norm(Bc, axis=1)))
        Bc = Bc * scale
        nb = np.linalg.norm(Be, axis=1)
        err = np.linalg.norm(Bc - Be, axis=1) / nb
        cosang = np.clip(np.sum(Bc * Be, 1) / (np.linalg.norm(Bc, axis=1) * nb), -1, 1)
        jres = 4e-7 * np.pi * np.asarray(d["|J|"]) * a / nb
        rows.append(dict(rho=rho, dB_rms=float(np.sqrt(np.mean(err**2))), dB_max=float(err.max()),
                         angle_rms_deg=float(np.degrees(np.sqrt(np.mean(np.arccos(cosang) ** 2)))),
                         J_res_rms=float(np.sqrt(np.mean(jres**2)))))
    rows = rows[::-1]
    res["eqs"][lab] = dict(file=f, scale=scale, L=eq.L, M=eq.M, N=eq.N, rows=rows)
    print(f"--- {lab} ({f}) L,M,N {eq.L},{eq.M},{eq.N}  current scale {scale:.5f}")
    print(f"{'rho':>5} {'|dB|/|B| rms':>13} {'max':>9} {'angle rms':>10} {'mu0|J|a/|B| rms':>16}")
    for r in rows:
        print(f"{r['rho']:5.2f} {r['dB_rms']:13.3e} {r['dB_max']:9.2e} {r['angle_rms_deg']:9.3f}d {r['J_res_rms']:16.3e}",
              flush=True)
if out:
    json.dump(res, open(out, "w"), indent=1)
