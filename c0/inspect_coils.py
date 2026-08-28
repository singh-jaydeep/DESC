"""Visual + numerical sanity check for a coilset before you trust it.

The trap this exists to catch: constraint objectives are evaluated on a grid, and an
optimizer that rides the constraint boundary will happily put violations BETWEEN the
samples. A curvature constraint on LinearGrid(N=12) (25 nodes) reported max 4.66 1/m
on a coilset whose true max was 146 1/m -- a 7 mm bend radius, invisible to the solve.

Usage:
    python c0/inspect_coils.py f1.h5 [f2.h5 ...] [--bound 4.657] [--out fig.png]
"""

import sys

import matplotlib

matplotlib.use("Agg")  # noqa: E402  -- must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from desc.grid import LinearGrid  # noqa: E402
from desc.io import load  # noqa: E402

N_FINE = 400  # 801 nodes/coil -- resolve what the constraint grid cannot


def _flat(d, out=None):
    """CoilSet.compute returns nested lists; flatten to one dict per physical coil."""
    out = [] if out is None else out
    if isinstance(d, dict):
        out.append(d)
    else:
        for x in d:
            _flat(x, out)
    return out


def inspect(path, bound, ax3d, axk, color):
    """Plot one coilset and report its curvature stats at N_FINE nodes."""
    cs = load(path)
    g = LinearGrid(N=N_FINE)
    data = _flat(cs.compute(["x", "curvature"], grid=g, basis="xyz"))
    zeta = np.asarray(g.nodes[:, 2])
    kall = []
    for i, d in enumerate(data):
        x = np.asarray(d["x"])
        k = np.abs(np.asarray(d["curvature"]))
        kall.append(k)
        ax3d.plot(x[:, 0], x[:, 1], x[:, 2], lw=0.8, color=color, alpha=0.55)
        hot = k > bound
        if hot.any():  # mark every point the constraint grid may have missed
            ax3d.scatter(
                x[hot, 0], x[hot, 1], x[hot, 2], s=6, color="red", depthshade=False
            )
        if i == 0:
            axk.semilogy(zeta, k, lw=0.9, color=color, label=path.split("/")[-1])
        else:
            axk.semilogy(zeta, k, lw=0.6, color=color, alpha=0.5)
    k = np.concatenate(kall)
    p50, p99, mx = np.percentile(k, [50, 99, 100])
    print(
        f"{path.split('/')[-1]:<34} median={p50:7.2f}  p99={p99:9.2f}  max={mx:9.2f}"
        f"  min bend R={100 / mx:7.1f} cm  frac>bound={(k > bound).mean():6.1%}"
    )
    return mx


if __name__ == "__main__":
    argv = sys.argv[1:]
    args, skip = [], False
    for i, a in enumerate(argv):  # skip flags AND their values
        if skip:
            skip = False
            continue
        if a.startswith("--"):
            skip = a in ("--bound", "--out")
            continue
        args.append(a)
    bound = 4.657
    out = "c0/coil_inspection.png"
    for i, a in enumerate(sys.argv):
        if a == "--bound":
            bound = float(sys.argv[i + 1])
        if a == "--out":
            out = sys.argv[i + 1]

    n = len(args)
    fig = plt.figure(figsize=(5.2 * n, 9))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    axk = fig.add_subplot(2, 1, 2)
    print(
        f"curvature at {2 * N_FINE + 1} nodes/coil, bound = {bound:.3f} 1/m "
        f"({100 / bound:.1f} cm bend radius)\n"
    )
    for j, path in enumerate(args):
        ax3d = fig.add_subplot(2, n, j + 1, projection="3d")
        inspect(path, bound, ax3d, axk, colors[j])
        ax3d.set_title(path.split("/")[-1], fontsize=9)
        ax3d.set_box_aspect((1, 1, 0.6))
    axk.axhline(bound, color="k", ls="--", lw=1, label=f"bound {bound:.2f} 1/m")
    axk.set_xlabel("zeta")
    axk.set_ylabel("|curvature| (1/m)")
    axk.legend(fontsize=7)
    axk.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"\nred points = above bound; wrote {out}")
