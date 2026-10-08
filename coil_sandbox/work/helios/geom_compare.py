"""Geometric comparison of finished polar arcB2 results: what do the good ones do differently?

    python work/helios/geom_compare.py            # all finals in work/helios/scan_polar

Per unique coil it reports position, hinge placement, arc-plane orientation, where the curvature
peaks, and where the coil comes closest to the plasma and to its neighbours. Locations are given as
a poloidal angle about the plasma centre (0 deg outboard midplane, 90 top, 180 inboard, 270 bottom).
Writes work/helios/geom_compare.{json,png}.
"""
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, "sandbox")
import boot  # noqa: E402

boot.setup(default="cpu")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

import common as S  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.io import load  # noqa: E402

P = "work/helios/scan_polar/"
N = 600
G = LinearGrid(N=N)

WIN = ["30", "60", "c0x30_c1x90_c2x30"]          # bn <= 1.08e-2, coil 2 in layout P
LOSE = ["90", "c0x30_c1x90_c2x0", "0", "c0x0_c1x150_c2x120", "150", "120", "c0x0_c1x120_c2x60"]


def final(lab):
    for leg in ("legC", "legB"):
        f = f"{P}cut{lab}/{leg}/result.h5"
        if os.path.exists(f):
            return f
    return None


def pol(x, R0, Z0):
    """poloidal angle (deg) about the plasma centre, 0 = outboard midplane, 90 = top"""
    return float(np.degrees(np.arctan2(x[2] - Z0, np.hypot(x[0], x[1]) - R0)) % 360)


def main():
    eq, _ = S.load_equilibrium("sample_equilibria/helios_repro.h5")
    tree = S.surface_tree(eq)
    R0 = float(eq.Rb_lmn[np.argmax(np.abs(eq.Rb_lmn))]) if False else 7.96
    Z0 = 0.0
    out = {}
    for lab in WIN + LOSE:
        f = final(lab)
        if f is None:
            continue
        cs = load(f)
        bn = json.load(open(os.path.splitext(f)[0] + ".json"))["result"]["bn"]
        pos = np.asarray(cs._compute_position(grid=LinearGrid(N=200)))   # all 12 coils
        trees = [cKDTree(p) for p in pos]
        coils = []
        for i, c in enumerate(S.iter_unique(cs)):
            d = c.compute(["x", "|curvature|"], grid=G, basis="xyz")
            x = np.asarray(d["x"])
            k = np.abs(np.asarray(d["|curvature|"]))
            R, Z = np.hypot(x[:, 0], x[:, 1]), x[:, 2]
            phi = np.degrees(np.arctan2(x[:, 1], x[:, 0]))
            dpc, _ = tree.query(x)
            j_pc = int(np.argmin(dpc))
            j_k = int(np.argmax(k))
            # nearest other coil among the 12 (this unique coil is copy 4*i)
            me = 4 * i
            best = min(((float(trees[j].query(pos[me])[0].min()), j) for j in range(len(pos)) if j != me))
            dcc, j_other = best
            j_cc = int(np.argmin(trees[j_other].query(pos[me])[0]))
            H = np.asarray(c.hinges).reshape(-1, 3)
            # arc plane normals (vector area of each half) and the angle to the toroidal direction
            normals = []
            for b in range(2):
                seg = x[: N // 2 + 1] if b == 0 else np.vstack([x[N // 2:], x[:1]])
                n = np.sum(np.cross(seg[:-1] - seg.mean(0), seg[1:] - seg.mean(0)), axis=0)
                normals.append(n / np.linalg.norm(n))
            cen = x.mean(0)
            tor = np.array([-cen[1], cen[0], 0.0]); tor /= np.linalg.norm(tor)
            coils.append(dict(
                i=i, current_MA=float(c.current) / 1e6,
                centroid_R=float(np.hypot(cen[0], cen[1])), centroid_Z=float(cen[2]),
                centroid_phi=float(np.degrees(np.arctan2(cen[1], cen[0]))),
                phi_span=float(phi.max() - phi.min()), R_span=[float(R.min()), float(R.max())],
                Z_span=[float(Z.min()), float(Z.max())],
                hinges=[[float(np.hypot(p[0], p[1])), float(p[2]), pol(p, R0, Z0)] for p in H],
                fold_deg=float(np.degrees(np.arccos(np.clip(abs(normals[0] @ normals[1]), -1, 1)))),
                tilt_deg=[float(np.degrees(np.arccos(np.clip(abs(n @ tor), -1, 1)))) for n in normals],
                kappa_max=float(k.max()), kappa_max_pol=pol(x[j_k], R0, Z0),
                kappa_max_RZ=[float(R[j_k]), float(Z[j_k])],
                kappa_mean=float(k.mean()),
                d_pc=float(dpc.min()), d_pc_pol=pol(x[j_pc], R0, Z0), d_pc_RZ=[float(R[j_pc]), float(Z[j_pc])],
                d_cc=float(dcc), d_cc_with=int(j_other), d_cc_pol=pol(pos[me][j_cc], R0, Z0),
                d_cc_RZ=[float(np.hypot(*pos[me][j_cc][:2])), float(pos[me][j_cc][2])],
            ))
        out[lab] = dict(bn=bn, win=lab in WIN, coils=coils)
    json.dump(out, open("work/helios/geom_compare.json", "w"), indent=1)

    # picture: R-Z projection per coil, winners solid, losers dashed; markers at kappa peak and d_pc point
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    th = np.linspace(0, 2 * np.pi, 200)
    for i in range(3):
        for row, labs, ttl in ((0, WIN, "winners"), (1, LOSE, "losers")):
            ax = axes[row][i]
            for lab in labs:
                if lab not in out:
                    continue
                cs = load(final(lab))
                c = list(S.iter_unique(cs))[i]
                x = np.asarray(c.compute("x", grid=G, basis="xyz")["x"])
                R, Z = np.hypot(x[:, 0], x[:, 1]), x[:, 2]
                d = out[lab]["coils"][i]
                ax.plot(R, Z, lw=1.4, label=f"{lab} ({out[lab]['bn']*100:.2f}%)")
                ax.plot(*d["kappa_max_RZ"], "o", ms=5, color=ax.lines[-1].get_color())
                ax.plot([h[0] for h in d["hinges"]], [h[1] for h in d["hinges"]], "s", ms=4,
                        color=ax.lines[-1].get_color(), mfc="none")
            ax.set_title(f"coil {i}, {ttl} (o = curvature peak, square = hinge)", fontsize=9, loc="left")
            ax.set_xlabel("R (m)"); ax.set_ylabel("Z (m)"); ax.set_aspect("equal")
            ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.suptitle("polar arcB2 finals on helios_repro: R-Z projection of each unique coil", fontsize=11)
    fig.tight_layout()
    fig.savefig("work/helios/geom_compare.png", dpi=120)
    print("wrote work/helios/geom_compare.{json,png}")

    hdr = f"{'run':20s} {'bn':>8s}  coil  {'I':>5s} {'phi':>6s} {'Rspan':>11s} {'Zspan':>12s} {'fold':>5s} " \
          f"{'tilt':>11s} {'kmax':>5s}@{'pol':>4s} {'d_pc':>5s}@{'pol':>4s} {'d_cc':>5s}@{'pol':>4s}"
    print(hdr)
    for lab, r in out.items():
        for d in r["coils"]:
            print(f"{lab:20s} {r['bn']:.2e}  c{d['i']}    {d['current_MA']:5.1f} {d['centroid_phi']:6.1f} "
                  f"{d['R_span'][0]:5.1f}-{d['R_span'][1]:5.1f} {d['Z_span'][0]:6.1f}-{d['Z_span'][1]:5.1f} "
                  f"{d['fold_deg']:5.0f} {d['tilt_deg'][0]:5.0f},{d['tilt_deg'][1]:5.0f} "
                  f"{d['kappa_max']:5.2f}@{d['kappa_max_pol']:4.0f} {d['d_pc']:5.2f}@{d['d_pc_pol']:4.0f} "
                  f"{d['d_cc']:5.2f}@{d['d_cc_pol']:4.0f}")


if __name__ == "__main__":
    main()
