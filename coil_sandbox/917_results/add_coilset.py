"""Append coilsets to viewer_data.json WITHOUT recomputing the ones already there.

build_data.py rebuilds everything, which costs a fresh virtual-casing pass plus a
Biot-Savart evaluation per coilset -- too much to run beside a GPU sweep on a 15 GB box.
This adds one or more coilsets incrementally and CACHES the plasma field (the expensive
part) to 917_results/_bplasma_cache.npz, so every later addition is just one Biot-Savart.

    python 917_results/add_coilset.py "KEY|LABEL|NOTE[|PATH]" ...

KEY is the name used in viewer_data.json. PATH defaults to 917_results/<KEY>.h5; give it
explicitly to pull a result straight out of work/. An existing KEY is REPLACED in place.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, "sandbox")
import boot  # noqa: E402

boot.setup(default="cpu")

import json  # noqa: E402

import numpy as np  # noqa: E402

import common as S  # noqa: E402
from desc.grid import Grid, LinearGrid  # noqa: E402
from desc.io import load  # noqa: E402
from desc.objectives._coils import _independent_coil_indices  # noqa: E402

JSON = "917_results/viewer_data.json"
CACHE = "917_results/_bplasma_cache.npz"
NC = 240

with open(JSON) as f:
    out = json.load(f)
NTHETA, NZETA = int(out["ntheta"]), int(out["nzeta"])

eq, _ = S.load_equilibrium("917_results/helios_repro.h5")

if os.path.exists(CACHE):
    z = np.load(CACHE)
    rpz, normal, Bp = z["rpz"], z["normal"], z["Bp"]
    print(f"plasma field from cache ({CACHE})", flush=True)
else:
    from desc.integrals.singularities import compute_B_plasma

    src = S.vc_source_grid(eq)
    NZP = max(NZETA // eq.NFP, src.num_zeta)
    NTH = max(NTHETA, src.num_theta)
    grid = LinearGrid(rho=np.array([1.0]), theta=NTH, zeta=NZP, NFP=eq.NFP, sym=False)
    d = eq.compute(["R", "phi", "Z", "n_rho"], grid=grid)
    R1, phi1, Z1 = (np.asarray(d[k]) for k in ("R", "phi", "Z"))
    n1 = np.asarray(d["n_rho"])
    print(f"virtual casing on one field period ({NZP} x {NTH}) ...", flush=True)
    Bp1 = np.asarray(compute_B_plasma(eq, grid, src, chunk_size=64))
    R = np.tile(R1, eq.NFP)
    Z = np.tile(Z1, eq.NFP)
    phi = np.concatenate([phi1 + 2 * np.pi * k / eq.NFP for k in range(eq.NFP)])
    normal = np.tile(n1, (eq.NFP, 1))
    Bp = np.tile(Bp1, (eq.NFP, 1))
    rpz = np.stack([R, phi, Z], axis=1)
    np.savez_compressed(CACHE, rpz=rpz, normal=normal, Bp=Bp)
    print(f"cached -> {CACHE}", flush=True)

for spec in sys.argv[1:]:
    # "|" separated, NOT ":" -- labels and notes contain colons.
    parts = spec.split("|")
    if len(parts) < 3:
        raise SystemExit(f"expected KEY|LABEL|NOTE[|PATH], got {spec!r}")
    key, label, note = parts[0], parts[1], parts[2]
    path = parts[3] if len(parts) > 3 else f"917_results/{key}.h5"
    cs = load(path)
    B = np.asarray(cs.compute_magnetic_field(rpz, basis="rpz",
                                             source_grid=S.source_grid(cs)[0])) + Bp
    Bn = np.sum(B * normal, axis=1)
    mB = float(np.mean(np.linalg.norm(B, axis=1)))
    s = np.linspace(0, 2 * np.pi, NC, endpoint=False)
    pos = np.asarray(cs._compute_position(
        grid=Grid(np.stack([0 * s, 0 * s, s], 1), sort=False, jitable=False), basis="xyz"))
    pos = np.concatenate([pos, pos[:, :1]], axis=1)
    uniq = [int(i) for i in _independent_coil_indices(cs)]
    nper = pos.shape[0] // len(uniq)
    m = S.evaluate(cs, eq, vacuum=False)
    rec = dict(key=key, label=label, note=note,
               bn_mean=float(np.mean(np.abs(Bn)) / mB), bn_max=float(np.max(np.abs(Bn)) / mB),
               bn_mean_T=float(np.mean(np.abs(Bn))), bn_max_T=float(np.max(np.abs(Bn))),
               n_coils=int(pos.shape[0]), n_unique=len(uniq),
               owner=[i // nper for i in range(pos.shape[0])],
               L_max=m["L"], kappa_max=m["kappa"], d_cc=m["d_cc"], d_pc=m["d_pc"],
               currents=[float(c.current) / 1e6 for c in S.iter_unique(cs)],
               bn=np.round(Bn / mB, 5).tolist(),
               coils=[np.round(p, 3).ravel().tolist() for p in pos])
    hit = [i for i, r in enumerate(out["coilsets"]) if r["key"] == key]
    if hit:
        out["coilsets"][hit[0]] = rec
        print(f"  REPLACED {label:<30} bn {rec['bn_mean']:.4e}", flush=True)
    else:
        out["coilsets"].append(rec)
        print(f"  added    {label:<30} bn {rec['bn_mean']:.4e}", flush=True)

with open(JSON, "w") as f:
    json.dump(out, f, separators=(",", ":"))
with open("917_results/data.js", "w") as f:
    f.write("window.HELIOS_DATA=" + json.dumps(out, separators=(",", ":")) + ";")
print(f"\nwrote {JSON} and 917_results/data.js ({len(out['coilsets'])} coilsets, "
      f"{os.path.getsize('917_results/data.js')/2**20:.2f} MiB)")
