"""Is the re-solve noise the equilibrium's discretization error? Spread vs L = M = N.

For each resolution: the k5 equilibrium raised to L = M = N = RES (grids 2 RES), solved tightly; then two
re-solves from that state + 1e-6 random interior noise (relative to each coefficient vector's rms), and the
spread of QS (stock cost), Boozer QS and iota between them. If the noise is truncation error it falls fast with RES.

    python work/ss_qh/joint/stall/resolution.py EQ.h5 8,10,12
"""

import os
import sys

SB = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
sys.path.insert(0, os.path.join(SB, "sandbox"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import boot  # noqa: E402

boot.setup(default="gpu")

import numpy as np  # noqa: E402

import common as S  # noqa: E402
from gauge3 import force_cost, iota, qs_booz, qs_cost, solve  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402


def qs_area(eq):
    g = LinearGrid(rho=np.array([0.2, 0.4, 0.6, 0.8, 1.0]), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
    d = eq.compute(["f_C", "|B|", "|e_theta x e_zeta|"], grid=g, helicity=(1, 4))
    f2, A, w = (np.asarray(d["f_C"] / d["|B|"] ** 3) ** 2, np.asarray(d["|e_theta x e_zeta|"]),
                np.asarray(g.weights))
    return float(np.sum(f2 * A * w) / np.sum(A * w))

path = sys.argv[1]
res_list = sys.argv[2].split(",")  # "10" = basis 10, grids 2x; "8g3" = basis 8, grids 3x
eq0, _ = S.load_equilibrium(path)
for tag in res_list:
    res, gm = (int(tag.split("g")[0]), int(tag.split("g")[1])) if "g" in tag else (int(tag), 2)
    eq = eq0.copy()
    if res != eq.M or gm != 2:
        eq.change_resolution(L=res, M=res, N=res, L_grid=gm * res, M_grid=gm * res, N_grid=gm * res)
    nit = solve(eq)
    q, b, i, f, qa = qs_cost(eq), qs_booz(eq), iota(eq), force_cost(eq), qs_area(eq)
    print(f"RES {tag} (L_grid {eq.L_grid} M_grid {eq.M_grid} N_grid {eq.N_grid}): solve {nit}; force {f:.4e}; QS {q:.6e}; Booz {b[0]:.4e} {b[1]:.4e} {b[2]:.4e}; "
          f"iota [{i.min():.5f}, {i.max():.5f}]", flush=True)
    rng = np.random.default_rng(0)
    for t in range(2):
        e = eq.copy()
        for nm in ("R_lmn", "Z_lmn", "L_lmn"):
            x = np.asarray(getattr(e, nm))
            setattr(e, nm, x + 1e-6 * rng.standard_normal(x.shape))
        nit = solve(e)
        q2, b2, i2, qa2 = qs_cost(e), qs_booz(e), iota(e), qs_area(e)
        print(f"   trial {t} {nit}: QS rel {abs(q2 - q) / q:.2e}; QS area rel {abs(qa2 - qa) / qa:.2e}; Booz rel {abs(b2[0] - b[0]) / b[0]:.2e} "
              f"{abs(b2[2] - b[2]) / b[2]:.2e}; iota max {np.abs(i2 - i).max():.2e}", flush=True)
