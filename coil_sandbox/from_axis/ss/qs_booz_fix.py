"""Boozer metrics in the literature convention: amplitudes of cos(m theta_B - n zeta_B) modes (DESC's product basis
converted by ptolemy_linear_transform, as QuasisymmetryBoozer does). Rewrites booz_max / booz_l2 in qs_metrics.json."""
import sys, json
sys.path.insert(0, "../../sandbox")
import boot; boot.setup(default="gpu")
import numpy as np
import desc.examples
from desc.io import load
from desc.grid import LinearGrid
from desc.objectives import QuasisymmetryBoozer
J = sys.argv[1]; out = json.load(open(J))
for spec in sys.argv[2:]:
    name, path = spec.split("=", 1)
    eq = load(path) if path.endswith(".h5") else desc.examples.get(path)
    if hasattr(eq, "__len__"): eq = eq[-1]
    r = out[name]; r["booz_max"], r["booz_l2"] = [], []
    for rho in r["rho"]:
        g = LinearGrid(rho=np.array([rho]), M=2 * eq.M_grid, N=2 * eq.N_grid, NFP=eq.NFP, sym=False)
        o = QuasisymmetryBoozer(eq, helicity=tuple(r["helicity"]), normalize=False, grid=g); o.build(verbose=0)
        f = np.asarray(o.compute_unscaled(eq.params_dict))
        d = eq.compute(["|B|_mn_B", "B modes"], grid=g, M_booz=2 * eq.M, N_booz=2 * eq.N)
        md = np.asarray(d["B modes"]); b00 = abs(float(np.asarray(d["|B|_mn_B"]).ravel()[(md[:, 1] == 0) & (md[:, 2] == 0)][0]))
        r["booz_max"].append(float(np.abs(f).max() / b00)); r["booz_l2"].append(float(np.linalg.norm(f) / b00))
    r["booz_convention"] = "cos(m theta_B - n zeta_B) amplitudes / B_00"
    print(name, "max", ["%.2e" % v for v in r["booz_max"]], "l2", ["%.2e" % v for v in r["booz_l2"]], flush=True)
    json.dump(out, open(J, "w"), indent=1)
