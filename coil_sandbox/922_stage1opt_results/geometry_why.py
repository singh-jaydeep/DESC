"""WHY did eps_1 fall? A geometric decomposition of the contour's toroidal wander.

Along a contour of the current potential, dzeta/dtheta = K^zeta/K^theta. Writing
K = n x B / mu0 in the surface basis gives

    mu0 K^theta = -B_zeta / sqrt(g),   mu0 K^zeta = +B_theta / sqrt(g)
    =>  dzeta/dtheta = -B_theta / B_zeta        (COVARIANT components)

and a zeta = const curve on the surface is EXACTLY planar (every point shares phi = zeta).
So all of eps_1 comes from the contour's zeta excursion, which is driven entirely by the
variation of B_theta/B_zeta around the poloidal circuit.

In BOOZER coordinates B_theta = I(psi) and B_zeta = G(psi) are flux functions, so
dzeta_B/dtheta_B = -I/G, which is zero for a vacuum field: the contours are exactly
zeta_Boozer = const and exactly planar. The excursion we measure is therefore the
deviation of the DESC toroidal angle from the Boozer one.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, "sandbox")
import boot  # noqa: E402

boot.setup(default="cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, HERE)
import jax.numpy as jnp  # noqa: E402
from contour_curve import ContourClamshell, _curves_from_params, contour_zeta, \
    make_Phi_derivs, potential_from_grid, truncate  # noqa: E402
from desc.examples import get  # noqa: E402
from desc.grid import Grid, LinearGrid  # noqa: E402
from desc.io import load  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else "922_stage1opt_results/runs/prox_eps1_300/eq_lo.h5"
eq0, eq1 = get("precise_QA"), load(path)
K, M, N, n = 8, 32, 64, 360
CKW = dict(K=K, M=M, N=N, n=n, Mt=16, Nt=16)


def contours_and_zeta(eq):
    o = ContourClamshell(eq, quantity="eps1", **CKW)
    o.build(verbose=0)
    xs, ws = o._chain(eq.params_dict, o._constants)
    e1 = np.asarray(o.compute(eq.params_dict))
    # recover zeta(theta) for each contour
    c = o._constants
    data = None
    from desc.compute.utils import _compute as compute_fun
    from contour_curve import DATA_KEYS
    data = compute_fun("desc.equilibrium.equilibrium.Equilibrium", DATA_KEYS,
                       params=eq.params_dict, transforms=c["transforms"],
                       profiles=c["profiles"])
    perm = c["perm"]
    Pt, Pz, G, I, Pmn, mm, nn = potential_from_grid(
        data["K_vc"][perm], data["e_theta"][perm], data["e_zeta"][perm],
        data["|e_theta x e_zeta|"][perm], M, N)
    cf, mf, nf = truncate(Pmn, c["trunc"], M, N)
    th = c["theta"]
    zs = []
    for k in range(K):
        level = k * G / (K * eq.NFP)
        zs.append(np.asarray(contour_zeta(cf, mf, nf, G, th, level, newton=6)))
    return e1, np.array(zs), np.asarray(th), float(G)


print("=== 1. zeta EXCURSION of each contour (peak-to-peak, radians) ===")
res = {}
for tag, eq in (("baseline", eq0), ("final", eq1)):
    e1, zs, th, G = contours_and_zeta(eq)
    exc = np.array([np.ptp(np.unwrap(z)) for z in zs])
    res[tag] = (e1, exc, zs, th)
    print(f"  {tag:9} eps_1 mean {e1.mean():.5f} | zeta excursion mean {exc.mean():.5f} rad "
          f"({np.degrees(exc.mean()):.3f} deg), range {exc.min():.4f}-{exc.max():.4f}")
e0, x0, _, _ = res["baseline"]
e1f, x1, _, _ = res["final"]
print(f"  -> eps_1 {(e1f.mean()/e0.mean()-1)*100:+.2f}%,  zeta excursion "
      f"{(x1.mean()/x0.mean()-1)*100:+.2f}%")
print(f"  per-contour correlation eps_1 vs zeta excursion: baseline "
      f"{np.corrcoef(e0,x0)[0,1]:+.3f}, final {np.corrcoef(e1f,x1)[0,1]:+.3f}")
print(f"  ratio eps_1/excursion: baseline {np.mean(e0/x0):.5f}, final {np.mean(e1f/x1):.5f}"
      f"   (constant => excursion explains eps_1 entirely)")

print("\n=== 2. THE DRIVER: -B_theta/B_zeta on the rho=1 surface ===")
for tag, eq in (("baseline", eq0), ("final", eq1)):
    g = LinearGrid(rho=np.array([1.0]), theta=2*np.pi*np.arange(M)/M,
                   zeta=2*np.pi*np.arange(N)/N, NFP=1, sym=False)
    d = eq.compute(["B_theta", "B_zeta"], grid=g)
    r = -np.asarray(d["B_theta"]) / np.asarray(d["B_zeta"])
    print(f"  {tag:9} -B_theta/B_zeta: mean {r.mean():+.5f}  std {r.std():.5f}  "
          f"ptp {np.ptp(r):.5f}")
    if tag == "baseline":
        s0 = r.std()
    else:
        print(f"  -> std of the pitch ratio changed {(r.std()/s0-1)*100:+.2f}% "
              f"(this is what drives the excursion)")

print("\n=== 3. BOUNDARY CROSS-SECTIONS: what moved, by poloidal angle ===")
nth = 180
for zfrac, label in ((0.0, "zeta=0"), (0.25, "zeta=pi/2NFP"), (0.5, "zeta=pi/NFP")):
    ze = zfrac * 2 * np.pi / eq0.NFP
    th = np.linspace(0, 2*np.pi, nth, endpoint=False)
    gg = Grid(np.stack([np.ones(nth), th, ze*np.ones(nth)], 1), sort=False, jitable=False)
    d0 = eq0.surface.compute(["R", "Z"], grid=gg)
    d1 = eq1.surface.compute(["R", "Z"], grid=gg)
    dR = np.asarray(d1["R"]) - np.asarray(d0["R"])
    dZ = np.asarray(d1["Z"]) - np.asarray(d0["Z"])
    disp = np.hypot(dR, dZ)
    j = int(np.argmax(disp))
    print(f"  {label:14} max |displacement| {disp.max()*1000:.3f} mm at theta="
          f"{np.degrees(th[j]):6.1f} deg  (R={float(np.asarray(d0['R'])[j]):.4f}, "
          f"Z={float(np.asarray(d0['Z'])[j]):+.4f})  mean {disp.mean()*1000:.3f} mm")

print("\n=== 4. ELONGATION / SHAPING, by toroidal angle ===")
for tag, eq in (("baseline", eq0), ("final", eq1)):
    els = []
    for zfrac in np.linspace(0, 1, 5, endpoint=False):
        ze = zfrac * 2 * np.pi / eq.NFP
        th = np.linspace(0, 2*np.pi, 256, endpoint=False)
        gg = Grid(np.stack([np.ones(256), th, ze*np.ones(256)], 1), sort=False, jitable=False)
        d = eq.surface.compute(["R", "Z"], grid=gg)
        R, Z = np.asarray(d["R"]), np.asarray(d["Z"])
        els.append((Z.max()-Z.min()) / (R.max()-R.min()))
    print(f"  {tag:9} vertical/horizontal extent by zeta: "
          f"{' '.join(f'{v:.3f}' for v in els)}")
