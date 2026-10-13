"""eps_eff (Nemov, DESC "effective ripple" via Bounce2D) at two resolutions, rho 0.25/0.5/0.75: a convergence check.

    python eps_conv.py OUT.json NAME=PATH ...
"""
import sys, json
sys.path.insert(0, "../../sandbox")
import boot; boot.setup(default="gpu")
import numpy as np
import desc.examples
from desc.io import load
from desc.grid import LinearGrid
from desc.integrals import Bounce2D
rr = np.array([0.25, 0.5, 0.75]); out = {}
RES = dict(lo=dict(X=32, Y=32, Y_B=128, nt=20, npitch=45), hi=dict(X=48, Y=64, Y_B=256, nt=40, npitch=90))
for spec in sys.argv[2:]:
    name, path = spec.split("=", 1)
    eq = load(path) if path.endswith(".h5") else desc.examples.get(path)
    if hasattr(eq, "__len__"): eq = eq[-1]
    out[name] = {}
    for tag, p in RES.items():
        try:
            ge = LinearGrid(rho=rr, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
            ang = Bounce2D.angle(eq, X=p["X"], Y=p["Y"], rho=rr)
            de = eq.compute("effective ripple", grid=ge, angle=ang, Y_B=p["Y_B"], num_transit=p["nt"],
                            num_well=p["nt"] * 30, num_quad=32, num_pitch=p["npitch"])
            out[name][tag] = np.asarray(ge.compress(de["effective ripple"])).tolist()
        except Exception as e:  # noqa: BLE001
            out[name][tag] = repr(e)[:200]
        print(name, tag, out[name][tag], flush=True)
        json.dump(out, open(sys.argv[1], "w"), indent=1)
