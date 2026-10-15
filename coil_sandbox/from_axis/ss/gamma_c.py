"""Energetic-particle confinement proxy Gamma_c (Nemov et al. 2008, DESC "Gamma_c"; and Velasco et al. 2021,
DESC "Gamma_c Velasco") at two bounce-integral resolutions, with eps_eff alongside: a convergence check.

    python gamma_c.py OUT.json NAME=PATH_or_example ...     (rho 0.25 / 0.5 / 0.75, as eps_conv.py)
"""
import sys, json, time
sys.path.insert(0, "../../sandbox")
import boot; boot.setup(default="gpu")
import numpy as np
import desc.examples
from desc.io import load
from desc.grid import LinearGrid
from desc.integrals import Bounce2D
rr = np.array([0.25, 0.5, 0.75]); out = {}
NAMES = ["Gamma_c", "Gamma_c Velasco", "effective ripple"]
RES = dict(lo=dict(X=32, Y=32, Y_B=128, nt=20, npitch=45), hi=dict(X=48, Y=64, Y_B=256, nt=40, npitch=90))
for spec in sys.argv[2:]:
    name, path = spec.split("=", 1)
    eq = load(path) if path.endswith(".h5") else desc.examples.get(path)
    if hasattr(eq, "__len__"): eq = eq[-1]
    out[name] = dict(path=path, rho=rr.tolist())
    for tag, p in RES.items():
        out[name][tag] = {}
        ge = LinearGrid(rho=rr, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
        ang = Bounce2D.angle(eq, X=p["X"], Y=p["Y"], rho=rr)
        for q in NAMES:  # one at a time: a failure (e.g. memory) in one leaves the others
            t = time.time()
            try:
                d = eq.compute(q, grid=ge, angle=ang, Y_B=p["Y_B"], num_transit=p["nt"], num_well=p["nt"] * 30,
                               num_quad=32, num_pitch=p["npitch"])
                out[name][tag][q] = np.asarray(ge.compress(d[q])).tolist()
            except Exception as e:  # noqa: BLE001
                out[name][tag][q] = repr(e)[:200]
            v = out[name][tag][q]
            print(f"{name:12s} {tag} {q:16s} " + (" / ".join(f"{x:.3e}" for x in v) if isinstance(v, list) else v)
                  + f"   ({time.time() - t:.0f} s)", flush=True)
            json.dump(out, open(sys.argv[1], "w"), indent=1)
