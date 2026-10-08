"""Show, for one device, WHY eps_2 << eps_1: the out-of-plane residual before and after the split.

    python 925_quasr_sweep/show_clamshell.py --id 1328095 --out 925_quasr_sweep/figures/

Top row: the proxy contour of the equilibrium with the largest eps_1/eps_2. Bottom row: the
device's own optimized coil with the largest gain. Left panels view each curve EDGE-ON to its
best-fit plane, so the out-of-plane excursion is the visible shape. Right panels plot the signed
out-of-plane distance against arclength -- exactly the quantity eps_1 and eps_2 are the RMS of --
for the single plane and for the two-segment split.

Needs a vacuum solve for the proxy contours (~170 s on GPU), so it takes one device at a time.
"""
import argparse
import os
import sys

p = argparse.ArgumentParser()
p.add_argument("--id", type=int, default=1328095)
p.add_argument("--cache", default="925_quasr_sweep/cache")
p.add_argument("--out", default="925_quasr_sweep/figures")
p.add_argument("--res", type=int, default=10)
p.add_argument("--nodes", type=int, default=480)
p.add_argument("--stride", type=int, default=8)
p.add_argument("--device", default="gpu")
a = p.parse_args()

sys.path.insert(0, "sandbox")
sys.path.insert(0, "922_stage1opt_results")
import boot  # noqa: E402

boot.setup(default=a.device)
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import clamshell as CL  # noqa: E402
import quasr as Q  # noqa: E402
from contour_curve import ContourClamshell, _eps1_live  # noqa: E402
from desc.grid import Grid  # noqa: E402
from desc.objectives import ForceBalance, ObjectiveFunction  # noqa: E402

# dataviz reference palette, categorical slots 1 and 2, in fixed order
C1, C2 = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8a85"
SURF, GRID = "#fcfcfb", "#e6e5e0"


def frame_and_residual(x, w, breaks=None):
    """Signed out-of-plane distance about the best-fit plane(s), plus an edge-on frame.

    With breaks=None this is the single-plane residual whose weighted RMS is eps_1's numerator.
    With breaks it is the per-segment residual whose weighted RMS is eps_2's numerator. Both are
    divided by the SAME whole-curve (lambda_1 lambda_2)^(1/4), so the RMS values are eps_1/eps_2.
    """
    W = w.sum()
    cen = (w[:, None] * x).sum(0) / W
    D = x - cen
    M = (D * w[:, None]).T @ D / W
    lam, vec = np.linalg.eigh(M)
    n0, e2, e1 = vec[:, 0], vec[:, 1], vec[:, 2]      # n0 = out-of-plane, e1 = longest in-plane
    tr = np.trace(M)
    l1l2 = 0.5 * (tr**2 - np.trace(M @ M)) - lam[0] * (tr - lam[0])
    denom = l1l2 ** 0.25
    if breaks is None:
        res = D @ n0
    else:
        res = np.zeros(len(x))
        b = sorted(int(v) for v in breaks)
        for j in range(len(b)):
            lo, hi = b[j], b[(j + 1) % len(b)]
            idx = np.arange(lo, hi) % len(x) if hi > lo else \
                np.concatenate([np.arange(lo, len(x)), np.arange(0, hi)])
            ww = w[idx]
            c = (ww[:, None] * x[idx]).sum(0) / ww.sum()
            Dd = x[idx] - c
            Ms = (Dd * ww[:, None]).T @ Dd / ww.sum()
            nk = np.linalg.eigh(Ms)[1][:, 0]
            res[idx] = Dd @ nk
    rms = np.sqrt((w * res**2).sum() / W) / denom
    return dict(res=res / denom, rms=rms, along=(D @ e1) / denom, out=(D @ n0) / denom,
                cen=cen, n0=n0, e1=e1, e2=e2)


def seg_index(n, breaks):
    b = sorted(int(v) for v in breaks)
    lab = np.zeros(n, dtype=int)
    for j in range(len(b)):
        lo, hi = b[j], b[(j + 1) % len(b)]
        idx = np.arange(lo, hi) if hi > lo else \
            np.concatenate([np.arange(lo, n), np.arange(0, hi)])
        lab[idx % n] = j
    return lab


def analyse(x, w):
    e1 = float(_eps1_live(x, w))
    brk = CL.best_breaks(x, w, 2, a.stride)[1]
    f1 = frame_and_residual(x, w)
    f2 = frame_and_residual(x, w, brk)
    return dict(eps1=e1, eps2=f2["rms"], eps1_check=f1["rms"], breaks=brk, f1=f1, f2=f2)


CACHE_NPZ = os.path.join(a.out, f"curves_{a.id}.npz")

# ---- the equilibrium's proxy contours -------------------------------------
rec = Q.record(a.id, a.cache)
K = 2 * rec["nc_per_hp"]
print(f"device {a.id}: nfp{rec['nfp']} {'QH' if rec['helicity'] else 'QA'} "
      f"A={rec['aspect_ratio']:.2f} a={rec['minor_radius']:.4f} nc/hp={rec['nc_per_hp']} K={K}",
      flush=True)
if os.path.exists(CACHE_NPZ):
    z = np.load(CACHE_NPZ)
    xs, ws = z["xs"], z["ws"]
    coil_x, coil_w = z["coil_x"], z["coil_w"]
    print(f"  reusing cached curves from {CACHE_NPZ}", flush=True)
else:
    eq, _ = Q.equilibrium(a.id, a.cache)
    eq.change_resolution(L=a.res, M=a.res, N=a.res,
                         L_grid=2 * a.res, M_grid=2 * a.res, N_grid=2 * a.res)
    print("  solving...", flush=True)
    eq.solve(objective=ObjectiveFunction(ForceBalance(eq=eq), deriv_mode="batched",
                                         jac_chunk_size=100),
             maxiter=150, verbose=0, ftol=1e-10, xtol=1e-12, gtol=1e-12)
    o = ContourClamshell(eq, quantity="eps1", K=K, M=64, N=128, n=a.nodes, Mt=16, Nt=16,
                         stride=a.stride)
    o.build(verbose=0)
    (xs, ws), _ = o._chain(eq.params_dict, o._constants, diagnostics=True)
    xs, ws = np.asarray(xs), np.asarray(ws)
    cx, cw = [], []
    for c in Q.coilset(a.id, a.cache, full=False):
        s_ = 2 * np.pi * np.arange(a.nodes) / a.nodes
        d = c.compute(["x", "x_s"], grid=Grid(np.stack([0 * s_, 0 * s_, s_], 1), sort=False,
                                              jitable=False), basis="xyz")
        cx.append(np.asarray(d["x"]))
        cw.append(np.linalg.norm(np.asarray(d["x_s"]), axis=1))
    coil_x, coil_w = np.asarray(cx), np.asarray(cw)
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(CACHE_NPZ, xs=xs, ws=ws, coil_x=coil_x, coil_w=coil_w)
    print(f"  cached curves -> {CACHE_NPZ}", flush=True)

best_c, best_gain = None, -1
for k in range(K):
    r = analyse(np.asarray(xs[k]), np.asarray(ws[k]))
    g = r["eps1"] / r["eps2"]
    print(f"  contour {k}: eps_1 {r['eps1']:.5f}  eps_2 {r['eps2']:.5f}  gain {g:.2f}x")
    if g > best_gain:
        best_gain, best_c, best_k = g, r, k

# ---- the device's own coils ------------------------------------------------
best_o, best_ogain = None, -1
for i, (x, w) in enumerate(zip(coil_x, coil_w)):
    r = analyse(np.asarray(x), np.asarray(w))
    g = r["eps1"] / r["eps2"]
    print(f"  coil {i}: eps_1 {r['eps1']:.5f}  eps_2 {r['eps2']:.5f}  gain {g:.2f}x")
    if g > best_ogain:
        best_ogain, best_o, best_oi = g, r, i

# ---- the figure -----------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), facecolor=SURF,
                         gridspec_kw=dict(width_ratios=[1.0, 1.25], hspace=0.62, wspace=0.30))
for ax in axes.flat:
    ax.set_facecolor(SURF)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3)

rows = [("proxy contour %d of the equilibrium" % best_k, best_c, best_gain),
        ("QUASR's own coil %d" % best_oi, best_o, best_ogain)]

for row, (title, r, gain) in enumerate(rows):
    n = len(r["f1"]["res"])
    lab = seg_index(n, r["breaks"])
    sfrac = np.arange(n) / n

    # --- left: edge-on view ------------------------------------------------
    ax = axes[row, 0]
    for j, col in ((0, C1), (1, C2)):
        m = lab == j
        al, ou = r["f1"]["along"].copy(), r["f1"]["out"].copy()
        al[~m] = np.nan
        ou[~m] = np.nan
        ax.plot(al, ou, color=col, lw=2.0, solid_capstyle="round",
                label=f"arc {j + 1}", zorder=3)
    ax.axhline(0, color=MUTED, lw=1.0, zorder=1)
    for b in r["breaks"]:
        ax.plot(r["f1"]["along"][int(b)], r["f1"]["out"][int(b)], "o", ms=8,
                mfc="white", mec=INK, mew=1.8, zorder=5)
    # Equal aspect makes a near-planar contour a featureless line (its out-of-plane
    # excursion is ~1% of its in-plane size), so the vertical axis is magnified and the
    # factor is stated -- never silently distorted.
    xr = np.ptp(r["f1"]["along"])
    yr = max(np.ptp(r["f1"]["out"]), 1e-12)
    mag = xr / yr
    ax.set_xlabel("along the curve's long in-plane axis", color=INK2, fontsize=9.5)
    ax.set_ylabel(f"out of the single best-fit plane\n(vertical scale x{mag:.0f})",
                  color=INK2, fontsize=9.5)
    ax.set_title(f"{title} — edge-on", color=INK, fontsize=10.5, loc="left", pad=26)
    ax.set_ylim(-0.62 * yr, 0.62 * yr)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, ncol=2,
              loc="lower left", bbox_to_anchor=(0, 1.005), borderaxespad=0)

    # --- right: the residual eps is the RMS of ----------------------------
    ax = axes[row, 1]
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(0, color=MUTED, lw=1.0, zorder=1)
    ax.plot(sfrac, r["f1"]["res"], color=C1, lw=2.0, zorder=3,
            label=f"one plane   RMS = eps_1 = {r['eps1']:.4f}")
    ax.plot(sfrac, r["f2"]["res"], color=C2, lw=2.0, zorder=4,
            label=f"two planes  RMS = eps_2 = {r['eps2']:.4f}")
    for b in r["breaks"]:
        ax.axvline(int(b) / n, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.set_xlabel("fraction of the way round the curve", color=INK2, fontsize=9.5)
    ax.set_ylabel("signed distance from the plane\n(in units of in-plane size)",
                  color=INK2, fontsize=9.5)
    ax.set_title(f"gain {gain:.2f}x   (dashed = the two hinges)", color=INK,
                 fontsize=10.5, loc="left", pad=26)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, ncol=2,
              loc="lower left", bbox_to_anchor=(0, 1.005), borderaxespad=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(*(np.array([-1.12, 1.12]) * np.abs(r["f1"]["res"]).max()))

fig.suptitle(f"QUASR {a.id}  (nfp {rec['nfp']} {'QH' if rec['helicity'] else 'QA'}, "
             f"A = {rec['aspect_ratio']:.1f}):  where eps_1 >> eps_2",
             color=INK, fontsize=13, x=0.012, ha="left", y=0.985)
fig.text(0.012, 0.945, "eps_2 splits the curve at the two hinges and gives each arc its own "
                       "plane; the residual it is the RMS of collapses where one plane could not "
                       "follow the bend.", color=INK2, fontsize=9.5, ha="left")
os.makedirs(a.out, exist_ok=True)
dest = os.path.join(a.out, f"clamshell_{a.id}.png")
fig.savefig(dest, dpi=150, bbox_inches="tight", facecolor=SURF)
print(f"\nwrote {dest}")
print(f"  proxy contour {best_k}: eps_1 {best_c['eps1']:.5f} -> eps_2 {best_c['eps2']:.5f}"
      f"  gain {best_gain:.2f}x   (eps_1 recomputed from the residual: "
      f"{best_c['eps1_check']:.5f})")
print(f"  coil {best_oi}:          eps_1 {best_o['eps1']:.5f} -> eps_2 {best_o['eps2']:.5f}"
      f"  gain {best_ogain:.2f}x   (eps_1 recomputed: {best_o['eps1_check']:.5f})")
