"""Effective ripple eps_eff (DESC EffectiveRipple, 1/nu regime) vs rho: target, method control, coil field (traced-surface
equilibrium, res 12). Same rho labels in each equilibrium (the traced one is ~0.95 of the target's minor radius).
Usage: ripple.py CASE EQ_TARGET"""
import os
import sys

sys.path.insert(0, "/home/singh/Documents/DESC2/coil_sandbox")
sys.path.insert(0, "/home/singh/Documents/DESC2/coil_sandbox/sandbox")
import boot  # noqa: E402

boot.setup("gpu")
import time  # noqa: E402
import numpy as np  # noqa: E402
import desc.io  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.objectives import EffectiveRipple  # noqa: E402
import common as S  # noqa: E402

F = "work/ss_qh/joint/fieldcheck"
case, eqt_f = sys.argv[1:3]
eqt = S.load_equilibrium(eqt_f)
eqt = eqt[0] if isinstance(eqt, tuple) else eqt
eqs = {"target": eqt, "coil field": desc.io.load(f"{F}/{case}/eq_traced_L12.h5")}
if os.path.exists(f"{F}/{case}/control_L12.h5"):
    eqs["control"] = desc.io.load(f"{F}/{case}/control_L12.h5")
rho = np.array([0.25, 0.5, 0.75, 0.9])
XY, YB, NP, TR = (int(os.environ.get(k, d)) for k, d in (("RX", 32), ("RYB", 64), ("RNP", 51), ("RTR", 20)))
suffix = os.environ.get("RSUF", "")
log = open(f"{F}/{case}/ripple{suffix}.txt", "w")
log.write(f"=== {case}: eps_eff at rho {rho.tolist()} (X=Y={XY}, Y_B={YB}, {TR} field-period transits, {NP} pitches)\n")
for k, e in eqs.items():
    t0 = time.time()
    g = LinearGrid(rho=rho, M=e.M_grid, N=e.N_grid, NFP=e.NFP, sym=False)
    o = EffectiveRipple(e, grid=g, X=XY, Y=XY, Y_B=YB, field_period_transits=TR, num_pitch=NP, num_quad=32)
    o.build(verbose=0)
    v = np.asarray(o.compute(e.params_dict))
    s = f"{k:11s} " + "  ".join(f"{r:.2f}: {x:.3e}" for r, x in zip(rho, v)) + f"   ({time.time() - t0:.0f} s)"
    print(s, flush=True)
    log.write(s + "\n")
    log.flush()
