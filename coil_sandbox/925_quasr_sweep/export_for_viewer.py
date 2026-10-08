"""Export one QUASR device as the three files `view.py` wants: equilibrium, coilset, bounds.

    python 925_quasr_sweep/export_for_viewer.py --id 1328095

The equilibrium must be SOLVED -- `view.py --streamlines` builds the current-potential contours
from K_vc, which needs the interior field, and on an unsolved boundary the contour Newton diverges.

The coilset is written in `make_start.py`'s structure, MixedCoilSet(CoilSet(coil, NFP, sym)), so
symmetry is applied by DESC and `common.iter_unique` sees only the nc_per_hp independent coils --
which is what `view.py`'s coil-count check and per-coil metrics expect.

The bounds are QUASR's OWN engineering constraints for this device, absolute, so the viewer's
margins are judged against what their optimizer actually enforced rather than against the
reactor-scaled defaults (which, at a ~ 0.125 m, are a different problem entirely).
"""
import argparse
import json
import os
import sys

p = argparse.ArgumentParser()
p.add_argument("--id", type=int, default=1328095)
p.add_argument("--cache", default="925_quasr_sweep/cache")
p.add_argument("--out", default="925_quasr_sweep/viewers")
p.add_argument("--res", type=int, default=10)
p.add_argument("--device", default="gpu")
a = p.parse_args()

sys.path.insert(0, "sandbox")
import boot  # noqa: E402

boot.setup(default=a.device)
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402

import common as S  # noqa: E402
import quasr as Q  # noqa: E402
from desc.coils import CoilSet, MixedCoilSet  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.objectives import ForceBalance, ObjectiveFunction  # noqa: E402

os.makedirs(a.out, exist_ok=True)
rec = Q.record(a.id, a.cache)
tag = f"{a.id}"
eq_path = os.path.join(a.out, f"eq_{tag}.h5")
cs_path = os.path.join(a.out, f"coils_{tag}.h5")
bd_path = os.path.join(a.out, f"bounds_{tag}.json")

print(f"QUASR {a.id}: nfp {rec['nfp']} {'QH' if rec['helicity'] else 'QA'}  A {rec['aspect_ratio']:.2f}"
      f"  a {rec['minor_radius']:.4f}  nc/hp {rec['nc_per_hp']}", flush=True)

# ---- equilibrium ----------------------------------------------------------
eq, _ = Q.equilibrium(a.id, a.cache)
eq.change_resolution(L=a.res, M=a.res, N=a.res,
                     L_grid=2 * a.res, M_grid=2 * a.res, N_grid=2 * a.res)
print(f"  solving at L=M=N={a.res} ...", flush=True)
eq.solve(objective=ObjectiveFunction(ForceBalance(eq=eq), deriv_mode="batched",
                                     jac_chunk_size=100),
         maxiter=150, verbose=0, ftol=1e-10, xtol=1e-12, gtol=1e-12)
lg = LinearGrid(rho=np.array([1.0]), M=2 * a.res, N=2 * a.res, NFP=eq.NFP)
iot = float(np.asarray(lg.compress(eq.compute("iota", grid=lg)["iota"]))[0])
want = float(rec["iota_profile"][-1])
print(f"  iota_edge {iot:.5f}  QUASR {want:.5f}   |d iota| {abs(abs(iot) - abs(want)):.4f}")
eq.save(eq_path)
print(f"  -> {eq_path}")

# ---- coils ----------------------------------------------------------------
base = [c for c in Q.coilset(a.id, a.cache, full=False)]
cs = MixedCoilSet(*[CoilSet(c, NFP=rec["nfp"], sym=True) for c in base])
cs.save(cs_path)
print(f"  -> {cs_path}   ({len(base)} independent coils, "
      f"{S.expected_coils(cs, eq)} physical)")

# ---- bounds: QUASR's own constraints, absolute ----------------------------
# AS BUILT, not guessed. Every bound is this device's OWN achieved value from the index row,
# so a margin of exactly 1.000 in the viewer means our reconstruction reproduced QUASR's coils
# and anything else is OUR error. Guessing their thresholds does not work: the paper quotes
# kappa <= 5 and d >= 0.1, but this device reached max_kappa 10.83 (so curvature was not the
# active constraint) and sits at d_pc 0.15, not 0.1. Only length (pinned at the threshold for
# 100% of the database), msc and d_cc are on round numbers here.
bounds = {
    "_note": f"QUASR {a.id} AS BUILT: every bound is the device's own achieved value, so the "
             f"viewer's margins read 1.000 exactly when the reconstruction is faithful. "
             f"Absolute -- these devices live at R0 ~ 1 m with a = {rec['minor_radius']:.4f} m, "
             f"so reactor-scaled defaults would judge a different problem entirely.",
    "_note2": f"Pinned at a round threshold for this device: L_total "
              f"({rec['total_coil_length_threshold']:.0f}/{2 * rec['nfp']} = "
              f"{rec['total_coil_length_threshold'] / (2 * rec['nfp']):.3f}), "
              f"kappa_MS ({rec['max_msc']:.4f}), d_cc ({rec['min_coil2coil_dist']:.4f}). "
              f"NOT pinned: kappa ({rec['max_kappa']:.4f}) and d_pc "
              f"({rec['min_coil2surface_dist']:.4f}) -- recorded as-built for reference.",
    "_note3": "L is null: QUASR constrains the SUM over a half period, not each coil. "
              "convex_excess / planar_max null: FourierXYZ coils carry neither.",
    "scale": "absolute",
    "L": None,
    "L_total": rec["total_coil_length_threshold"] / (2 * rec["nfp"]),
    "kappa": float(rec["max_kappa"]),
    "kappa_MS": float(rec["max_msc"]),
    "d_cc": float(rec["min_coil2coil_dist"]),
    "d_pc": float(rec["min_coil2surface_dist"]),
    "convex_excess": None,
    "planar_max": None,
}
with open(bd_path, "w") as f:
    json.dump(bounds, f, indent=2)
print(f"  -> {bd_path}   L_total {bounds['L_total']:.3f}  kappa {bounds['kappa']:.3f}  "
      f"kappa_MS {bounds['kappa_MS']:.3f}  d_cc {bounds['d_cc']:.4f}  d_pc {bounds['d_pc']:.4f}")

print(f"\nnow:\n  python view.py {a.out}/view_{tag}.html --eq {eq_path} "
      f"--h5 quasr={cs_path} --streamlines {2 * rec['nc_per_hp']} --bounds {bd_path} "
      f"--vacuum --device cpu")
