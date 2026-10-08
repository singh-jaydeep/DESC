"""A transverse arcB2 start with every hinge at z = 0 (one joint plane for the whole machine).

    python work/ss_qh/joint/make_z0_start.py --eq precise_QH --vacuum --bounds B.json --currents C.h5 \
        --clear 1.3 --M 5 --out work/ss_qh/joint/starts/z0_D.h5          (sandbox root, CPU)

Fitting arcs to existing coils with hinges at their z = 0 crossings does not work on precise_QH:
every baseline coil overhangs its z = 0 chord by 5-42 cm, and a transverse arc is a graph over its
chord. So the start is built: coil i in the vertical plane phi_i = (i + 1/2) pi / (NFP nc), a
lens of two arches h sin(pi t)^power over one horizontal chord at z = 0 (t in [0, 1] along the chord).
The chord spans the cross-section's R range plus `clear` x d_pc each side; each arch height is the
smallest (1 mm scan) that keeps the in-plane distance to the cross-section >= gap_frac x clear x d_pc, separately
above and below. power 1 is the arcs' first transverse mode (exact fit, curvature 0 at the hinges, but 1.2-1.4 m deep
bowls); power 0.5 is a half ellipse (short, but kappa 2-2.5x the bound at the hinges); in between trades the two. Currents
are copied from --currents (same sum as the free arcB2 baseline). Stellarator symmetry maps
z = 0 to itself, so z = 0 is the only height that gives ONE joint plane.
"""
import sys; sys.path.insert(0, "sandbox")
import boot; boot.setup(default="cpu")
import argparse, json, os
import numpy as np
import common as S
from desc.coils import CoilSet, MixedCoilSet, PiecewisePlanarArcCoil
from desc.grid import LinearGrid
from desc.io import load
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--eq", default="precise_QH")
ap.add_argument("--vacuum", action="store_true")
ap.add_argument("--bounds", required=True)
ap.add_argument("--currents", required=True, help="coilset whose unique-coil currents are copied")
ap.add_argument("--nc", type=int, default=4)
ap.add_argument("--clear", type=float, default=1.3, help="in-plane clearance, in units of the d_pc bound")
ap.add_argument("--gap-frac", type=float, default=0.75,
                help="required in-plane gap of each arch, as a fraction of the chord margin clear x d_pc "
                     "(must be < 1: an arch cannot clear by more than its hinges stand off)")
ap.add_argument("--power", type=float, default=1.0,
                help="arch shape h sin(pi t)^power: 1 = pure sine (exact in the arc basis, kappa 0 at the hinges, "
                     "but deep bowls), 0.5 = half ellipse (short, but a vertical tangent at the hinges)")
ap.add_argument("--M", type=int, default=5)
ap.add_argument("--n", type=int, default=720, help="samples per coil for the fit")
ap.add_argument("--out", required=True)
a = ap.parse_args()

eq, kind = S.load_equilibrium(a.eq)
P = S.paper_bounds(eq, a.nc, None, a.bounds)
cur = [float(c.current) for c in S.iter_unique(load(a.currents))]
c_in = a.clear * P["dpc"]
coils, info = [], []
for i in range(a.nc):
    ph = np.pi / eq.NFP * (i + 0.5) / a.nc
    g = LinearGrid(theta=np.linspace(0, 2 * np.pi, 1441), zeta=np.array([ph]), NFP=1)
    d = eq.surface.compute(["R", "Z"], grid=g)
    R, Z = np.asarray(d["R"]), np.asarray(d["Z"])
    Rc = (R.max() + R.min()) / 2
    A = (R.max() - R.min()) / 2 + c_in
    tt = np.linspace(0, 1, 2001)
    sec = np.stack([R, Z], 1)
    tree = cKDTree(sec)

    def arch(h):  # in-plane (R, z) polyline of an arch of signed height h, inboard to outboard
        return np.stack([Rc - A + 2 * A * tt, h * np.sin(np.pi * tt) ** a.power], 1)

    def height(sign):  # smallest |h| (1 mm scan) with in-plane gap >= gap_frac x c_in
        for h in np.arange(0.0, 2.0, 1e-3):
            if tree.query(arch(sign * h))[0].min() >= a.gap_frac * c_in:
                return h
        raise SystemExit(f"coil {i}: no arch height up to 2 m clears the cross-section")

    Bu, Bl = height(+1), height(-1)
    n2 = a.n // 2
    t = np.arange(n2) / n2
    Rs = np.concatenate([Rc - A + 2 * A * t, Rc + A - 2 * A * t])          # upper inboard->outboard, lower back
    Zs = np.concatenate([Bu * np.sin(np.pi * t) ** a.power, -Bl * np.sin(np.pi * t) ** a.power])
    pts = np.stack([Rs * np.cos(ph), Rs * np.sin(ph), Zs], 1)
    arc = PiecewisePlanarArcCoil.from_values(cur[i], pts, B=2, M=a.M, basis="xyz", fit_method="chord")
    H = np.asarray(arc.hinges).reshape(-1, 3)
    fit = np.asarray(arc.compute("x", grid=LinearGrid(N=400), basis="xyz")["x"])
    dev = float(cKDTree(pts).query(fit)[0].max())
    # in-plane clearance actually achieved (dense polygon vs the cross-section)
    dense = np.stack([np.hypot(fit[:, 0], fit[:, 1]), fit[:, 2]], 1)
    gap = float(cKDTree(dense).query(np.stack([R, Z], 1))[0].min())
    info.append(dict(phi_deg=float(np.degrees(ph)), Rc=Rc, A=A, Bu=Bu, Bl=Bl, hinge_z=H[:, 2].tolist(),
                     fit_dev_mm=dev * 1e3, inplane_gap_m=gap, current=cur[i]))
    print(f"coil {i}: phi {np.degrees(ph):5.1f}  chord R [{Rc - A:.3f}, {Rc + A:.3f}]  up {Bu:.3f} m  down {Bl:.3f} m  "
          f"hinge z {np.abs(H[:, 2]).max():.1e}  fit dev {dev * 1e3:.2f} mm  in-plane gap {gap:.3f} m", flush=True)
    coils.append(arc)

cs = MixedCoilSet(*[CoilSet(c, NFP=eq.NFP, sym=eq.sym) for c in coils])
m = S.evaluate(cs, eq, vacuum=a.vacuum)
print(f"  L {m['L']:.4f} m  kappa {m['kappa']:.3f}  kappa_MS {m['kms']:.3f}  d_cc {m['d_cc'] * 1e3:.1f} mm  "
      f"d_pc {m['d_pc'] * 1e3:.1f} mm  linked {m['linked']}")
print(f"  B.n: <|B.n|>/<|B|> {m['bn']:.3e}   value / bound: {S.report(m, P)}  -> {S.verdict(m, P)}")
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
cs.save(a.out)
json.dump(dict(args=vars(a), coils=info, bounds=P, metrics=m, margins=S.margins(m, P), verdict=S.verdict(m, P)),
          open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1, default=float)
print(f"wrote {a.out}")
