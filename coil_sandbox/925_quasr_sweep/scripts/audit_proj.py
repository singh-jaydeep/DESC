"""Audit: what do the planar and arcB2 PROJECTIONS actually measure, and how noisy are they?

Runs on QUASR's own optimized FourierXYZ coils, which gives a real unrestricted baseline.

  (A) eps_1 / eps_2 of the REAL coils, next to eps_1 / eps_2 of the proxy contours of the
      same boundary. This is a direct test of the proxy claim, on data the program has
      never used.
  (B) The planar projection: how much field error does flattening each coil onto its own
      best-fit plane cost, and what does it do to the clearances?
  (C) The arcB2 projection: bn over a sweep of hinge placements (centroid cut line, equal
      parameter, and the DP-optimal 2-plane split), so the spread is measured rather than
      assumed.
"""
import sys, os, warnings, time
sys.path.insert(0, "sandbox")
sys.path.insert(0, "922_stage1opt_results")
import boot; boot.setup(default="gpu")
warnings.filterwarnings("ignore")
import numpy as np

import quasr as Q
import common as S
import clamshell as CL
from contour_curve import _eps1_live
from desc.coils import CoilSet, MixedCoilSet, PolarPlanarArcCoil
from desc.grid import LinearGrid, Grid

CACHE = "/tmp/claude-1000/-home-singh-Documents-DESC2-coil-sandbox/91f880f0-34c8-4d34-ba49-8b36d79ca617/scratchpad/quasr_cache"
ID = int(sys.argv[1]) if len(sys.argv) > 1 else 1630198
NPLANAR = int(sys.argv[2]) if len(sys.argv) > 2 else 8
ARC_M = 5

eq, rec = Q.equilibrium(ID, CACHE)
NFP, SYM = rec["nfp"], True
base = [c for c in Q.coilset(ID, CACHE, full=False)]
print(f"=== QUASR {ID}: nfp {NFP} {'QH' if rec['helicity'] else 'QA'}, "
      f"{len(base)} unique coils, a {rec['minor_radius']:.4f}, qs 1e{rec['qs_error']:.2f} ===", flush=True)


def wrap(coils):
    """make_start.py's structure: symmetry applied by DESC, so iter_unique sees the uniques."""
    return MixedCoilSet(*[CoilSet(c, NFP=NFP, sym=SYM) for c in coils])


tree = S.surface_tree(eq)


def ev(coils, tag):
    cs = wrap(coils)
    m = S.evaluate(cs, eq, tree=tree, vacuum=True)
    print(f"  {tag:34s} bn {m['bn']:.4e}  max {m['bn_max']:.3e}  "
          f"d_cc {m['d_cc']:.4f}  d_pc {m['d_pc']:.4f}  kappa {m['kappa']:7.3f}  "
          f"L/hp {m['L_total']:7.3f}  linked {m['linked']}", flush=True)
    return m


def curve_pts(c, n=400):
    s = 2 * np.pi * np.arange(n) / n
    g = Grid(np.stack([0 * s, 0 * s, s], 1), sort=False, jitable=False)
    d = c.compute(["x", "x_s"], grid=g, basis="xyz")
    x = np.asarray(d["x"])
    w = np.linalg.norm(np.asarray(d["x_s"]), axis=1)  # arclength weight
    return x, w


def eta_svd(x, w=None):
    """Pavone et al.'s non-planarity score sigma_min / sum(sigma), 0 = planar."""
    D = x - (x.mean(0) if w is None else (w[:, None] * x).sum(0) / w.sum())
    if w is not None:
        D = D * np.sqrt(w)[:, None]
    sv = np.linalg.svd(D, compute_uv=False)
    return sv[-1] / sv.sum()


# ---- (A) eps of the real coils -------------------------------------------
print("\n-- (A) eps_1 / eps_2 of QUASR's REAL coils (arclength-weighted) --", flush=True)
e1c, e2c, etac, etau = [], [], [], []
for i, c in enumerate(base):
    x, w = curve_pts(c, 480)
    e1 = float(_eps1_live(x, w))
    e2, _ = CL.eps_B(x, w, 2, stride=8)
    e1c.append(e1); e2c.append(e2)
    etac.append(eta_svd(x, w)); etau.append(eta_svd(x))
    print(f"   coil {i}: eps_1 {e1:.5f}  eps_2 {e2:.5f}  C {4*e2/e1:.3f}  "
          f"eta_SVD {etac[-1]:.5f} (unweighted {etau[-1]:.5f})")
print(f"   MEAN  eps_1 {np.mean(e1c):.5f}  eps_2 {np.mean(e2c):.5f}  eta_SVD {np.mean(etac):.5f}")

# ---- baseline -------------------------------------------------------------
print("\n-- (B) baseline and planar projection --", flush=True)
m0 = ev(base, "QUASR FourierXYZ N=16 (baseline)")

planar = [c.to_FourierPlanar(N=NPLANAR, grid=LinearGrid(N=200), basis="xyz") for c in base]
mp = ev(planar, f"to_FourierPlanar N={NPLANAR}")
dev = []
for c, p in zip(base, planar):
    d0 = np.asarray(c.compute("x", grid=LinearGrid(N=800), basis="xyz")["x"])
    d1 = np.asarray(p.compute("x", grid=LinearGrid(N=800), basis="xyz")["x"])
    from scipy.spatial import cKDTree
    dev.append(float(cKDTree(d0).query(d1)[0].max()))
print(f"   planar fit deviation: max {max(dev):.5f} m = {max(dev)/rec['minor_radius']:.3f} a")
print(f"   bn ratio planar/xyz = {mp['bn']/m0['bn']:.2f}x")

# ---- (C) arcB2 hinge sensitivity -----------------------------------------
print(f"\n-- (C) polar arcs B=2 M={ARC_M}: hinge placement sweep --", flush=True)


def arcs_from(hinge_fn, n=400):
    out = []
    for c in base:
        h = hinge_fn(c)
        pts = S.resample_between(c, sorted(np.mod(h, 2 * np.pi)), n_per=max(n // 2, 60))
        out.append(PolarPlanarArcCoil.from_values(float(c.current), pts, B=2, M=ARC_M, basis="xyz"))
    return out


res = {}
for cut in range(0, 180, 20):
    try:
        a = arcs_from(lambda c, cut=cut: S.centroid_line_hinges(c, cut))
        res[f"cut {cut:3d} deg"] = ev(a, f"cut-angle {cut} deg")
    except Exception as e:
        print(f"  cut {cut}: FAILED {type(e).__name__}: {str(e)[:120]}")

for ph in (0.0, 0.125, 0.25, 0.375):
    try:
        a = arcs_from(lambda c, ph=ph: np.array([2 * np.pi * ph, 2 * np.pi * (ph + 0.5)]))
        res[f"phase {ph:.3f}"] = ev(a, f"equal parameter, phase {ph}")
    except Exception as e:
        print(f"  phase {ph}: FAILED {type(e).__name__}: {str(e)[:120]}")

try:
    a = arcs_from(lambda c: S.long_axis_hinges(c))
    res["long axis"] = ev(a, "long-axis hinges")
except Exception as e:
    print(f"  long axis: FAILED {type(e).__name__}: {str(e)[:120]}")


def dp_hinges(c, n=480, stride=8):
    x, w = curve_pts(c, n)
    brk = CL.best_breaks(x, w, 2, stride=stride)[1]
    return np.array([2 * np.pi * b / n for b in brk])


try:
    a = arcs_from(dp_hinges)
    res["DP-optimal eps_2"] = ev(a, "DP-optimal (min out-of-plane RMS)")
except Exception as e:
    print(f"  DP: FAILED {type(e).__name__}: {str(e)[:200]}")

if res:
    bn = {k: v["bn"] for k, v in res.items()}
    lo, hi = min(bn, key=bn.get), max(bn, key=bn.get)
    print(f"\n   arcB2 bn over hinge placements: best {bn[lo]:.4e} ({lo})  "
          f"worst {bn[hi]:.4e} ({hi})  ratio {bn[hi]/bn[lo]:.2f}x")
    print(f"   vs xyz baseline {m0['bn']:.4e}: best {bn[lo]/m0['bn']:.1f}x, "
          f"worst {bn[hi]/m0['bn']:.1f}x")
    print(f"   vs planar projection {mp['bn']:.4e}")
