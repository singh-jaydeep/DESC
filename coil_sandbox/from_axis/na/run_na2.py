"""Giuliani et al. (JCP 2022) near-axis single stage with two-face folded coils, solved with the branch's
composite least squares (desc.optimize.least_squares_composite.lsq_composite: hinges kept exact in the model).

Rows (pre-hinge values s, each scaled by its weight):
  targets : (B_coils - B_QS) and (gradB_coils - gradB_QS) on the axis (sqrt-dl quadrature), (iota - iota0)/iota0,
            axis length - 2 pi
  bounds  : coil node -> NAE surface at r = a     >= d_pc          (one row per coil node)
            coil node -> any other coil          >= d_cc
            arc curvature                        <= kappa_max     (per node)
            coil length                          <= L_max
            current                              in [0, imax x uniform]
            NAE elongation                       <= e_max         (per axis node), optional
Nearest-point distances: the nearest node is chosen without derivatives, the distance to it is differentiated
(the derivative of a min at its argmin).

python run_na2.py --nfp 4 --nc 5 --iota 1.25 --a 0.124 --dpc 0.11 --dcc 0.04 --seed 0 --out runs2/qh_z0_nc5_s0
(run from na/; CPU)
"""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "sandbox"))
import boot; boot.setup(default="cpu")
sys.path.insert(0, HERE)
import numpy as np
import jax
import jax.numpy as jnp
from desc.optimize.least_squares_composite import lsq_composite

from nae import NearAxis, helicity, elongation
from folded import Layout, coil_arcs, full_coilset, B_and_gradB, arc_curvature, coil_length, NPT

ap = argparse.ArgumentParser()
ap.add_argument("--nfp", type=int, default=4)
ap.add_argument("--nc", type=int, default=4)
ap.add_argument("--K", type=int, default=5)
ap.add_argument("--Kax", type=int, default=5)
ap.add_argument("--mode", default="z0", choices=["z0", "free"])
ap.add_argument("--iota", type=float, default=1.25)
ap.add_argument("--axis0", default="1,0.17;0,-0.13")
ap.add_argument("--eta0", type=float, default=1.6)
ap.add_argument("--rcoil", type=float, default=0.42, help="initial coil half-chord")
ap.add_argument("--a", type=float, default=0.124, help="minor radius of the NAE surface used for clearance")
ap.add_argument("--dpc", type=float, default=0.11, help="min coil - NAE surface (r = a) distance")
ap.add_argument("--dcc", type=float, default=0.0585)
ap.add_argument("--Lmax", type=float, default=2.93)
ap.add_argument("--kmax", type=float, default=13.7)
ap.add_argument("--imax", type=float, default=2.0, help="each current in [0, imax x uniform]; <= 0 disables")
ap.add_argument("--emax", type=float, default=0.0, help="max NAE elongation; <= 0 disables")
ap.add_argument("--wlen", type=float, default=0.0, help="regularizer: weight of sum_i (L_i / L_max)^2 (target 0)")
ap.add_argument("--wbend", type=float, default=0.0, help="regularizer: weight of sum_i int (kappa / kappa_max)^2 ds / L_i")
ap.add_argument("--wpen", type=float, default=100.0, help="weight^2 of the (normalized) bound rows")
ap.add_argument("--nphi", type=int, default=41)
ap.add_argument("--ntheta", type=int, default=48)
ap.add_argument("--maxiter", type=int, default=600)
ap.add_argument("--hessian", default="secant", choices=["gn", "secant"])
ap.add_argument("--seed", type=int, default=-1, help="-1: no randomization")
ap.add_argument("--init", default=None, help="start from a saved npz (same layout)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

L = Layout(a.nfp, a.nc, a.K, a.Kax, a.mode)
na = NearAxis(a.nfp, a.nphi)
Iref = 0.5 / (2 * a.nfp * a.nc)  # uniform current giving B0 ~ 1 at R ~ 1 (mu0/4pi units)
rng = np.random.default_rng(a.seed if a.seed >= 0 else 0)
rand = a.seed >= 0

# ---------------- initial guess ----------------
if a.init:
    d = np.load(a.init + ".npz")
    p0 = jnp.asarray(d["p"])
    hel = float(d["hel"])
else:
    rcl, zsl = [np.array([float(v) for v in s.split(",")]) for s in a.axis0.split(";")]
    rc = np.zeros(a.Kax + 1); rc[: len(rcl)] = rcl
    zs = np.zeros(a.Kax + 1); zs[: len(zsl)] = zsl
    rc_base, zs_base, eta0 = rc.copy(), zs.copy(), a.eta0
    for attempt in range(200):  # randomized starts: redraw until the NAE elongation is within the cap
        rc, zs, eta0 = rc_base.copy(), zs_base.copy(), a.eta0
        if rand:
            rc[1:] *= rng.uniform(0.5, 1.3)
            zs[1:] *= rng.uniform(0.5, 1.3)
            eta0 *= rng.uniform(0.7, 1.3)
        g = na.geometry(jnp.asarray(rc), jnp.asarray(zs))
        hel = helicity(g["n"], na.phi, a.nfp)
        q0 = na.solve(jnp.asarray(rc), jnp.asarray(zs), eta0, hel, iota0=a.iota)
        e0 = float(elongation(q0["X1c"], q0["Y1s"], q0["Y1c"]).max())
        if not rand or a.emax <= 0 or e0 <= a.emax:
            break
    else:
        raise SystemExit(f"no randomized start within elongation {a.emax}")
    # coil sizing from the NAE surface at r = a (one period, fine theta)
    th = np.linspace(0, 2 * np.pi, 181, endpoint=False)
    X1 = np.asarray(q0["X1c"])[:, None] * np.cos(th)
    Y1 = np.asarray(q0["Y1s"])[:, None] * np.sin(th) + np.asarray(q0["Y1c"])[:, None] * np.cos(th)
    P = (np.asarray(q0["x"])[:, None] + a.a * (X1[..., None] * np.asarray(q0["n"])[:, None]
                                               + Y1[..., None] * np.asarray(q0["b"])[:, None])).reshape(-1, 3)
    Pr, Pp, Pz = np.hypot(P[:, 0], P[:, 1]), np.arctan2(P[:, 1], P[:, 0]), P[:, 2]
    dph = 2 * np.pi / a.nfp / a.nphi
    m = 1.6 * a.dpc
    rows = []
    for i in range(a.nc):
        ph = np.pi / a.nfp * (i + 0.5) / a.nc
        sel = np.abs(np.angle(np.exp(1j * (Pp - ph)))) < 2.5 * dph
        R_, Z_ = Pr[sel], Pz[sel]
        Ra, Rb = R_.min() - 1.5 * m, R_.max() + 1.5 * m
        if rand:
            Ra -= rng.uniform(0, 0.3) * m; Rb += rng.uniform(0, 0.3) * m
        t_ = (R_ - Ra) / (Rb - Ra)
        sn = np.sin(np.pi * t_)
        hu = max(0.05, np.max((Z_ + m) / sn))      # upper sine arch clearing every point by m
        hl = max(0.05, np.max((-Z_ + m) / sn))     # lower
        au = np.zeros(a.K); al = np.zeros(a.K)
        au[0], al[0] = hu / (Rb - Ra), hl / (Rb - Ra)
        tu = tl = 0.0
        if rand:
            au[1:] = rng.normal(0, 0.02, a.K - 1); al[1:] = rng.normal(0, 0.02, a.K - 1)
            tu, tl = rng.normal(0, 0.15, 2)
        if a.mode == "z0":
            hinge = [Ra, ph, Rb, ph]
        else:
            hinge = [Ra, ph, 0.0, Rb, ph, 0.0]
        rows.append(hinge + [tu, tl] + list(au) + list(al) + [Iref])
    C = np.array(rows)
    p0 = L.join(jnp.asarray(rc), jnp.asarray(zs), eta0, jnp.asarray(C))
    X, I = full_coilset(L.split(p0)[3], L)
    B, _ = B_and_gradB(g["x"][0], X, I)
    if float(jnp.dot(B, g["t"][0])) < 0:  # orientation: currents positive with B along +phi
        # reverse each coil: swap the hinges; each arc keeps its side, so its tilt flips sign (the chord
        # reversed) and w(t) -> w(1 - t), i.e. a_k -> (-1)^(k+1) a_k
        nh, K = L.nh, a.K
        Hs = C[:, :nh].copy()
        C[:, : nh // 2], C[:, nh // 2: nh] = Hs[:, nh // 2:], Hs[:, : nh // 2]
        C[:, nh: nh + 2] *= -1
        sgn = (-1.0) ** (np.arange(1, K + 1) + 1)
        C[:, nh + 2: nh + 2 + K] *= sgn
        C[:, nh + 2 + K: nh + 2 + 2 * K] *= sgn
        p0 = L.join(jnp.asarray(rc), jnp.asarray(zs), eta0, jnp.asarray(C))
        X, I = full_coilset(L.split(p0)[3], L)
        B, _ = B_and_gradB(g["x"][0], X, I)
    assert float(jnp.dot(B, g["t"][0])) > 0, "could not orient the coils"

print(f"nfp {a.nfp} nc {a.nc} mode {a.mode} helicity {hel} nparams {L.n} seed {a.seed}", flush=True)

theta = jnp.asarray(np.linspace(0, 2 * np.pi, a.ntheta, endpoint=False))
rot = [jnp.asarray(np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1.0]]))
       for t in (-2 * np.pi / a.nfp, 0.0, 2 * np.pi / a.nfp)]


def nae_surface(q):
    """Points of the first-order NAE surface at r = a, periods -1, 0, 1 (covers every independent coil)."""
    X1 = q["X1c"][:, None] * jnp.cos(theta)[None]
    Y1 = q["Y1s"][:, None] * jnp.sin(theta)[None] + q["Y1c"][:, None] * jnp.cos(theta)[None]
    P = q["x"][:, None] + a.a * (X1[..., None] * q["n"][:, None] + Y1[..., None] * q["b"][:, None])
    P = P.reshape(-1, 3)
    return jnp.concatenate([P @ R.T for R in rot])


def nearest(p, S, k=8, tau=0.002):
    """Smooth distance from each p to the point set S: soft-min over its k nearest nodes (chosen without
    derivatives), -tau log sum exp(-d / tau). A plain argmin is kinked wherever the nearest node changes, which
    stalls the trust region on active clearance bounds. Underestimates the distance by at most tau log k (safe)."""
    sp, sS = jax.lax.stop_gradient(p), jax.lax.stop_gradient(S)
    idx = jnp.argsort(jnp.sum((sp[:, None] - sS[None]) ** 2, -1), 1)[:, :k]
    d = jnp.linalg.norm(p[:, None] - S[idx], axis=-1)
    return -tau * jax.scipy.special.logsumexp(-d / tau, axis=1)


def pieces(p):
    rc, zs, eta, C = L.split(p)
    q = na.solve(rc, zs, eta, hel, iota0=a.iota)
    X, I = full_coilset(C, L)
    BG = jax.lax.map(lambda r: B_and_gradB(r, X, I), q["x"])  # sequential: vmap made the Jacobian 5 GB at 32 coils
    return rc, zs, eta, C, q, X, I, BG


# row layout: (name, size, kind, lo, hi, scale)  scale normalizes s before the weight
def rows_spec():
    nn = 2 * NPT  # nodes per coil (both arcs, hinges once)
    spec = [("B", a.nphi * 3, "t"), ("G", a.nphi * 9, "t"), ("iota", 1, "t"), ("Lax", 1, "t"),
            ("dpc", a.nc * nn, "b"), ("dcc", a.nc * nn, "b"), ("kappa", a.nc * 2 * (NPT + 1), "b"),
            ("Lcoil", a.nc, "b"), ("I", a.nc, "b")]
    if a.emax > 0:
        spec.append(("elong", a.nphi, "b"))
    if a.wlen > 0:
        spec.append(("reg_len", a.nc, "t"))
    if a.wbend > 0:
        spec.append(("reg_bend", a.nc * 2 * (NPT + 1), "t"))
    return spec


SPEC = rows_spec()
wb = np.sqrt(a.wpen)
BOUNDS = dict(dpc=(a.dpc, np.inf, wb / a.dpc), dcc=(a.dcc, np.inf, wb / a.dcc), kappa=(-np.inf, a.kmax, wb / a.kmax),
              Lcoil=(-np.inf, a.Lmax, wb / a.Lmax), I=(0.0, a.imax * Iref if a.imax > 0 else np.inf, wb / Iref),
              elong=(-np.inf, a.emax, wb / max(a.emax, 1)))
lo, hi, tgt, isb, wrow = [], [], [], [], []
for name, n, kind in SPEC:
    if kind == "t":
        lo += [-np.inf] * n; hi += [np.inf] * n; tgt += [0.0] * n; isb += [False] * n; wrow += [1.0] * n
    else:
        l_, h_, w_ = BOUNDS[name]
        lo += [l_] * n; hi += [h_] * n; tgt += [0.0] * n; isb += [True] * n; wrow += [w_] * n
wrow = np.array(wrow)
lo_s, hi_s, tgt_s = np.array(lo) * wrow, np.array(hi) * wrow, np.array(tgt)
isb = np.array(isb)


def fun_raw(p):
    rc, zs, eta, C, q, X, I, (Bc, Gc) = pieces(p)
    w = jnp.sqrt(q["dl"] * 2 * jnp.pi / a.nfp / a.nphi)
    out = [((Bc - q["B"]) * w[:, None]).ravel(), ((Gc - q["G"]) * w[:, None, None]).ravel(),
           jnp.atleast_1d((q["iota"] - a.iota) / a.iota),
           jnp.atleast_1d((jnp.sum(q["dl"]) * 2 * jnp.pi / a.nphi - 2 * jnp.pi) / (2 * jnp.pi))]
    S = nae_surface(q)
    dpc, dcc, kap, Ls, bend = [], [], [], [], []
    for i in range(a.nc):
        (xu, x1u, x2u), (xl, x1l, x2l), _ = coil_arcs(C[i], L)
        pts = jnp.concatenate([xu, xl[1:-1]])
        dpc.append(nearest(pts, S))
        others = jnp.concatenate([X[:i], X[i + 1:]])[:, :-1].reshape(-1, 3)
        dcc.append(nearest(pts, others))
        kap.append(jnp.concatenate([arc_curvature(x1u, x2u), arc_curvature(x1l, x2l)]))
        Ls.append(coil_length(C[i], L))
        if a.wbend > 0:  # (kappa / kappa_max) sqrt(ds / L): squares sum to int (kappa/kappa_max)^2 ds / L
            ds = jnp.concatenate([jnp.linalg.norm(x1u, axis=-1), jnp.linalg.norm(x1l, axis=-1)]) / NPT
            bend.append(kap[-1] / a.kmax * jnp.sqrt(ds / Ls[-1]))
    out += [jnp.concatenate(dpc), jnp.concatenate(dcc), jnp.concatenate(kap), jnp.stack(Ls), C[:, -1]]
    if a.emax > 0:
        out.append(elongation(q["X1c"], q["Y1s"], q["Y1c"]))
    if a.wlen > 0:
        out.append(jnp.sqrt(a.wlen) * jnp.stack(Ls) / a.Lmax)
    if a.wbend > 0:
        out.append(jnp.sqrt(a.wbend) * jnp.concatenate(bend))
    return jnp.concatenate(out)


fun = jax.jit(lambda p: fun_raw(p) * wrow)
jac = jax.jit(jax.jacfwd(lambda p: fun_raw(p) * wrow))


def reg_slices():
    out, k = [], 0
    for name, n, kind in SPEC:
        if name.startswith("reg_"):
            out.append(slice(k, k + n))
        k += n
    return out


def blocks(s):
    out, k = {}, 0
    for name, n, kind in SPEC:
        out[name] = np.asarray(s[k:k + n]); k += n
    return out


def report(p, tag, extra=None):
    rc, zs, eta, C, q, X, I, (Bc, Gc) = pieces(p)
    sraw = blocks(np.asarray(fun_raw(p)))
    r = np.where(isb, np.maximum(0, np.asarray(fun(p)) - hi_s) - np.maximum(0, lo_s - np.asarray(fun(p))),
                 np.asarray(fun(p)) - tgt_s)
    gq = na.geometry(rc, zs)
    e = np.asarray(elongation(q["X1c"], q["Y1s"], q["Y1c"]))
    info = dict(tag=tag, cost=float(0.5 * r @ r), cost_targets=float(0.5 * np.sum(r[~isb] ** 2)),
                cost_bounds=float(0.5 * np.sum(r[isb] ** 2)),
                max_dB=float(jnp.max(jnp.linalg.norm(Bc - q["B"], axis=1))),
                max_dG_rel=float(jnp.max(jnp.linalg.norm(Gc - q["G"], axis=(1, 2)))
                                 / jnp.max(jnp.linalg.norm(q["G"], axis=(1, 2)))),
                iota=float(q["iota"]), eta=float(eta), hel=helicity(gq["n"], na.phi, a.nfp),
                max_elong=float(e.max()), axis_rc=np.asarray(rc).round(5).tolist(),
                axis_zs=np.asarray(zs).round(5).tolist(),
                max_abs_Zaxis=float(np.abs(np.asarray(gq["Z"])).max()),
                min_dpc=float(sraw["dpc"].min()), min_dcc=float(sraw["dcc"].min()),
                max_kappa=float(sraw["kappa"].max()), Lcoil=sraw["Lcoil"].round(3).tolist(),
                I_over_uniform=(sraw["I"] / Iref).round(3).tolist(),
                n_active_bounds=int(np.sum(isb & (r != 0))),
                L_total=float(np.sum(sraw["Lcoil"])),
                coil_R_max=float(np.max(np.hypot(np.asarray(X)[:a.nc, :, 0], np.asarray(X)[:a.nc, :, 1]))),
                coil_absz_max=float(np.max(np.abs(np.asarray(X)[:a.nc, :, 2]))),
                mean_kappa2=float(np.mean((sraw["kappa"] / a.kmax) ** 2)),
                cost_reg=float(0.5 * sum(np.sum(np.asarray(fun(p))[sl] ** 2) for sl in reg_slices())))
    info["cost_field"] = info["cost_targets"] - info["cost_reg"]
    if extra:
        info.update(extra)
    print(json.dumps(info), flush=True)
    return info


info0 = report(p0, "init")
t0 = time.time()
res = lsq_composite(lambda x: fun(x), jnp.asarray(p0), lambda x: jac(x), lo_s, hi_s, tgt_s, isb,
                    x_scale="jac", ftol=1e-10, xtol=1e-12, gtol=1e-8, verbose=2, maxiter=a.maxiter,
                    options=dict(hessian=a.hessian))
p = jnp.asarray(res.x)
info = report(p, "final", dict(time_s=round(time.time() - t0), status=str(res.message), nit=int(res.nit)
                               if hasattr(res, "nit") else None, optimality=float(res.optimality)
                               if hasattr(res, "optimality") else None))
np.savez(a.out + ".npz", p=np.asarray(p), hel=hel, mode=a.mode, nfp=a.nfp, nc=a.nc, K=a.K, Kax=a.Kax,
         args=json.dumps(vars(a)))
json.dump(dict(init=info0, final=info, args=vars(a)), open(a.out + ".json", "w"), indent=1)
