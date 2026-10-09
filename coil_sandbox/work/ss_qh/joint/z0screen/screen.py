"""Screen QUASR boundaries for the z = 0 joint plane (arcB2, every hinge on z = 0), boundary only, CPU.

    python work/ss_qh/joint/z0screen/screen.py --helicity 1 --nfp 4 --A 8 --qs-max -3.5 --n 400 --out OUT.jsonl

Per device, over 17 toroidal angles in a half period (the cold/extremes.py metrics):
  cut_min / cut_med : where z = 0 cuts the cross-section, as a fraction of its R extent (1 = through the widest chord,
                      0 = misses it) -- a transverse arc is a graph over its z = 0 chord, so low values force D-shapes;
  ext_med / ext_max : |z| of the cross-section's R extremes / a;
  zc_max            : max |z-centre| / a (the vertical excursion; precise_QH 1.16, WISTELL-A 0.58).
Reference values: precise_QH cut_min 0.00 / med 0.27, ext_med 1.62; WISTELL-A 0.34 / 0.56, 1.08.
Resumable (skips IDs already in OUT). Fetches each namelist once into the 925 cache.
"""
import sys; sys.path.insert(0, "sandbox")
import boot; boot.setup(default="cpu")
import argparse, json, os
import numpy as np
import quasr as Q
from desc.grid import LinearGrid as LG

ap = argparse.ArgumentParser()
ap.add_argument("--helicity", type=int, default=1)
ap.add_argument("--nfp", type=int, default=4)
ap.add_argument("--A", type=float, default=8.0)
ap.add_argument("--qs-max", type=float, default=-3.5)
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--out", required=True)
a = ap.parse_args()
CACHE = "925_quasr_sweep/cache"
d = Q.index(CACHE)
m = (d["helicity"] == a.helicity) & (np.abs(d["aspect_ratio"] - a.A) < 0.05) & (d["qs_error"] < a.qs_max)
if a.nfp > 0:
    m &= d["nfp"] == a.nfp
idx = np.where(m)[0]
idx = idx[np.argsort(d["qs_error"][idx])][: a.n]
done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["ID"] for l in open(a.out)}
print(f"{m.sum()} devices match; screening {len(idx)} (best QS first), {len(done)} already done", flush=True)
for j in idx:
    ID = int(d["ID"][j])
    if ID in done:
        continue
    try:
        eq, rec = Q.equilibrium(ID, CACHE)
        A_ = float(rec["minor_radius"])
        zin, zout, cut, zc = [], [], [], []
        for ph in np.linspace(0, np.pi / eq.NFP, 17):
            g = LG(theta=np.linspace(0, 2 * np.pi, 721), zeta=np.array([ph]), NFP=1)
            s = eq.surface.compute(["R", "Z"], grid=g); R, Z = np.asarray(s["R"]), np.asarray(s["Z"])
            zin.append(Z[R.argmin()]); zout.append(Z[R.argmax()]); zc.append((Z.max() + Z.min()) / 2)
            sg = np.sign(Z); k = np.where(sg != np.roll(sg, -1))[0]
            Rc = [R[i] - Z[i] * (R[(i + 1) % len(R)] - R[i]) / (Z[(i + 1) % len(R)] - Z[i]) for i in k]
            cut.append((max(Rc) - min(Rc)) / (R.max() - R.min()) if len(Rc) >= 2 else 0.0)
        e = np.maximum(np.abs(zin), np.abs(zout)) / A_
        row = dict(ID=ID, nfp=int(rec["nfp"]), helicity=int(rec["helicity"]), A=float(rec["aspect_ratio"]), a=A_,
                   iota=float(rec["mean_iota"]), qs_error=float(rec["qs_error"]), nc_per_hp=int(rec["nc_per_hp"]),
                   max_elongation=float(rec["max_elongation"]), cut_min=float(min(cut)), cut_med=float(np.median(cut)),
                   ext_med=float(np.median(e)), ext_max=float(e.max()), zc_max=float(np.abs(zc).max() / A_))
    except Exception as ex:  # a bad namelist must not stop the screen
        row = dict(ID=ID, error=repr(ex)[:200])
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    if "cut_min" in row:
        print(f"{ID}: iota {row['iota']:.2f} qs {row['qs_error']:.2f} cut_min {row['cut_min']:.2f} med {row['cut_med']:.2f} "
              f"ext_med {row['ext_med']:.2f} zc_max {row['zc_max']:.2f}", flush=True)
