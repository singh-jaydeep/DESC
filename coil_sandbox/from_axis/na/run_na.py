"""Giuliani et al. (JCP 2022) near-axis single stage with two-face folded coils.

python run_na.py --nfp 2 --nc 4 --iota 0.42 --mode z0 --out qa_z0
"""
import argparse, json, time, sys
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from nae import NearAxis, helicity, elongation
from folded import Layout, coil_arcs, full_coilset, B_and_gradB, arc_curvature, coil_length

ap = argparse.ArgumentParser()
ap.add_argument("--nfp", type=int, default=2)
ap.add_argument("--nc", type=int, default=4)
ap.add_argument("--K", type=int, default=5, help="sine modes per arc")
ap.add_argument("--Kax", type=int, default=5)
ap.add_argument("--mode", default="z0", choices=["z0", "free"])
ap.add_argument("--iota", type=float, default=0.42)
ap.add_argument("--axis0", default="1,0.1;0,0.1", help="rc list;zs list (zs[0] ignored)")
ap.add_argument("--eta0", type=float, default=0.6)
ap.add_argument("--rcoil", type=float, default=0.42, help="initial coil half-chord")
ap.add_argument("--dpa", type=float, default=0.30, help="min coil-axis distance")
ap.add_argument("--dcc", type=float, default=0.06)
ap.add_argument("--Lmax", type=float, default=2.93)
ap.add_argument("--kmax", type=float, default=13.7)
ap.add_argument("--wpen", type=float, default=10.0, help="weight of the bound penalties")
ap.add_argument("--welong", type=float, default=0.0, help="penalty on max elongation above --emax")
ap.add_argument("--emax", type=float, default=4.0)
ap.add_argument("--imax", type=float, default=0.0, help="if > 0: each current in [0, imax x uniform current]")
ap.add_argument("--nphi", type=int, default=41)
ap.add_argument("--maxfev", type=int, default=300)
ap.add_argument("--init", default=None, help="start from a saved npz")
ap.add_argument("--out", required=True)
a = ap.parse_args()

L = Layout(a.nfp, a.nc, a.K, a.Kax, a.mode)
na = NearAxis(a.nfp, a.nphi)

# ---------------- initial guess ----------------
if a.init:
    d = np.load(a.init + ".npz")
    p0 = jnp.asarray(d["p"])
    hel = float(d["hel"])
    assert d["mode"] == a.mode or a.mode == "free"
    if d["mode"] != a.mode:  # z0 -> free: insert z = 0 for both hinges
        L0 = Layout(a.nfp, a.nc, a.K, a.Kax, "z0")
        rc, zs, eta, C0 = L0.split(p0)
        C = jnp.concatenate([C0[:, 0:2], jnp.zeros((a.nc, 1)), C0[:, 2:4], jnp.zeros((a.nc, 1)), C0[:, 4:]], 1)
        p0 = L.join(rc, zs, eta, C)
else:
    rcl, zsl = [np.array([float(v) for v in s.split(",")]) for s in a.axis0.split(";")]
    rc = np.zeros(a.Kax + 1); rc[: len(rcl)] = rcl
    zs = np.zeros(a.Kax + 1); zs[: len(zsl)] = zsl
    g = na.geometry(jnp.asarray(rc), jnp.asarray(zs))
    hel = helicity(g["n"], na.phi, a.nfp)
    rows = []
    for i in range(a.nc):
        ph = np.pi / a.nfp * (i + 0.5) / a.nc
        k = np.arange(a.Kax + 1) * a.nfp
        Rax = rc @ np.cos(k * ph)
        Zax = zs @ np.sin(k * ph)
        arch = np.zeros(a.K); arch[0] = 0.5   # height = half the chord (a sine "circle")
        # coil centred on the axis point: z0 mode can only centre radially; tilt shifts the arch
        if a.mode == "z0":
            hinge = [Rax - a.rcoil, ph, Rax + a.rcoil, ph]
        else:
            hinge = [Rax - a.rcoil, ph, Zax, Rax + a.rcoil, ph, Zax]
        I0 = 0.5 / (2 * a.nfp * a.nc)
        rows.append(hinge + [0.0, 0.0] + list(arch) + list(arch) + [I0])
    C = np.array(rows)
    p0 = L.join(jnp.asarray(rc), jnp.asarray(zs), a.eta0, jnp.asarray(C))
    # fix the current sign so B is along +phi on the axis
    X, I = full_coilset(L.split(p0)[3], L)
    B, _ = B_and_gradB(g["x"][0], X, I)
    if float(jnp.dot(B, g["t"][0])) < 0:
        C[:, -1] *= -1
        p0 = L.join(jnp.asarray(rc), jnp.asarray(zs), a.eta0, jnp.asarray(C))

print(f"nfp {a.nfp} nc {a.nc} mode {a.mode} helicity {hel} nparams {L.n}", flush=True)

phi_full = jnp.asarray(np.linspace(0, 2 * np.pi, 8 * a.nphi, endpoint=False))


def axis_full(rc, zs):
    k = jnp.arange(len(rc)) * a.nfp
    R = jnp.cos(k[None] * phi_full[:, None]) @ rc
    Z = jnp.sin(k[None] * phi_full[:, None]) @ zs
    return jnp.stack([R * jnp.cos(phi_full), R * jnp.sin(phi_full), Z], 1)


def pieces(p):
    rc, zs, eta, C = L.split(p)
    q = na.solve(rc, zs, eta, hel, iota0=a.iota)
    X, I = full_coilset(C, L)
    BG = jax.vmap(lambda r: B_and_gradB(r, X, I))(q["x"])
    return rc, zs, eta, C, q, X, I, BG


def residuals(p):
    rc, zs, eta, C, q, X, I, (Bc, Gc) = pieces(p)
    w = jnp.sqrt(q["dl"] * 2 * jnp.pi / a.nfp / a.nphi)  # sqrt(dl) quadrature over one period
    rB = ((Bc - q["B"]) * w[:, None]).ravel()
    rG = ((Gc - q["G"]) * w[:, None, None]).ravel()
    r_iota = jnp.atleast_1d((q["iota"] - a.iota) / a.iota)
    Lax = jnp.sum(q["dl"]) * 2 * jnp.pi / a.nphi  # full axis length
    r_len_ax = jnp.atleast_1d((Lax - 2 * jnp.pi) / (2 * jnp.pi))
    # bound penalties
    pen = []
    ax = axis_full(rc, zs)
    nb = X.shape[0]
    for i in range(a.nc):
        (xu, x1u, x2u), (xl, x1l, x2l), _ = coil_arcs(C[i], L)
        pts = jnp.concatenate([xu, xl[1:-1]])
        # coil-axis
        d = jnp.min(jnp.linalg.norm(pts[:, None] - ax[None], axis=-1), 1)
        pen.append(jnp.maximum(0, a.dpa - d) / a.dpa)
        # coil-coil (against every other coil of the full set; coil i itself is index i in block 0)
        others = jnp.concatenate([X[:i], X[i + 1:]])[:, :-1].reshape(-1, 3)
        dcc = jnp.min(jnp.linalg.norm(pts[:, None] - others[None], axis=-1), 1)
        pen.append(jnp.maximum(0, a.dcc - dcc) / a.dcc)
        # curvature (both arcs)
        kap = jnp.concatenate([arc_curvature(x1u, x2u), arc_curvature(x1l, x2l)])
        pen.append(jnp.maximum(0, kap - a.kmax) / a.kmax / jnp.sqrt(len(kap) / 8.0))
        pen.append(jnp.atleast_1d(jnp.maximum(0, coil_length(C[i], L) - a.Lmax) / a.Lmax))
    if a.imax > 0:
        Iref = 0.5 / (2 * a.nfp * a.nc)
        x = C[:, -1] / Iref
        pen.append(jnp.maximum(0, x - a.imax))
        pen.append(jnp.maximum(0, -x))
    r_pen = jnp.sqrt(a.wpen) * jnp.concatenate(pen)
    out = [rB, rG, r_iota, r_len_ax, r_pen]
    if a.welong > 0:
        e = elongation(q["X1c"], q["Y1s"], q["Y1c"])
        out.append(jnp.sqrt(a.welong) * jnp.maximum(0, e - a.emax) / a.emax / jnp.sqrt(a.nphi))
    return jnp.concatenate(out)


res_j = jax.jit(residuals)
jac_j = jax.jit(jax.jacfwd(residuals))


def report(p, tag):
    rc, zs, eta, C, q, X, I, (Bc, Gc) = pieces(p)
    w = q["dl"] * 2 * jnp.pi / a.nfp / a.nphi
    fB = float(0.5 * jnp.sum(jnp.sum((Bc - q["B"]) ** 2, 1) * w))
    fG = float(0.5 * jnp.sum(jnp.sum((Gc - q["G"]) ** 2, (1, 2)) * w))
    relB = float(jnp.max(jnp.linalg.norm(Bc - q["B"], axis=1)))
    relG = float(jnp.max(jnp.linalg.norm(Gc - q["G"], axis=(1, 2))) / jnp.max(jnp.linalg.norm(q["G"], axis=(1, 2))))
    e = np.asarray(elongation(q["X1c"], q["Y1s"], q["Y1c"]))
    r = np.asarray(res_j(p))
    Ls = [float(coil_length(C[i], L)) for i in range(a.nc)]
    gq = na.geometry(rc, zs)
    hnow = helicity(gq["n"], na.phi, a.nfp)
    info = dict(tag=tag, cost=float(0.5 * r @ r), fB=fB, fG=fG, max_dB=relB, max_dG_rel=relG,
                iota=float(q["iota"]), eta=float(eta), hel=hnow, max_elong=float(e.max()),
                axis_rc=np.asarray(rc).round(5).tolist(), axis_zs=np.asarray(zs).round(5).tolist(),
                Lcoil=np.round(Ls, 3).tolist(), I=np.asarray(C[:, -1]).round(5).tolist(),
                pen=float(np.sum(r[-(len(r) - 9 * a.nphi * 0):][-1:] ** 2)))
    print(json.dumps(info), flush=True)
    return info


info0 = report(p0, "init")
t0 = time.time()
sol = least_squares(lambda p: np.asarray(res_j(jnp.asarray(p))), np.asarray(p0),
                    jac=lambda p: np.asarray(jac_j(jnp.asarray(p))), method="trf", x_scale="jac",
                    max_nfev=a.maxfev, ftol=1e-12, xtol=1e-12, gtol=1e-12, verbose=1)
print(f"time {time.time() - t0:.0f}s status {sol.status} nfev {sol.nfev}", flush=True)
p = jnp.asarray(sol.x)
info = report(p, "final")
np.savez(a.out + ".npz", p=np.asarray(p), hel=hel, mode=a.mode, nfp=a.nfp, nc=a.nc, K=a.K, Kax=a.Kax,
         args=json.dumps(vars(a)))
json.dump(dict(init=info0, final=info, args=vars(a)), open(a.out + ".json", "w"), indent=1)
