"""Regenerate this design's metrics.json, METRICS.md and figures/ from data/ (the same file in every design folder).

    python make_figures.py [--stages eq,coils,trace,transport,figures,md] [--device gpu|cpu]
    python make_figures.py --ref precise_QA --out ../compare/ref_precise_QA.json     (equilibrium metrics only)

Stages (each one updates its own key of metrics.json, so they can be run separately):
  eq         geometry, iota and shear, magnetic well, mirror ratio, elongation, Jacobian, QS (Boozer, literature
             convention) and the two-term measure, eps_eff (low resolution)                     ~3 min (GPU)
  coils      the dense coil check (lib/common.evaluate) against data/bounds.json, hinges, faces, currents,
             B.n and coil-plasma distance on the boundary                                       ~2 min
  trace      field-line tracing of the coil field: magnetic axis, rotation number per line, punctures mapped into
             the equilibrium (are its surfaces the coils' surfaces?), resonance flags.  TRACE=dense (default) is
             80 lines x 2000 field periods (~5-8 min); TRACE=quick is 30 x 500 (~1 min)
  transport  eps_eff, Gamma_c (Nemov) and Gamma_c (Velasco) at two bounce-integral resolutions   ~15 min
             (sets XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 itself: cuFFT plan memory)
  figures    figures/*.png from data/ and the cached work/ files
  md         METRICS.md from metrics.json
Default: every stage. GPU jobs on the laptop go under a memory cap (REPRODUCE.md).

DESC: the repository four levels up (final/designX -> DESC root), or $DESC_REPO; commit in REPRODUCE.md.
lib/ holds frozen copies of coil_sandbox/sandbox/{common,convexity,planarity}.py (the dense coil check).
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--stages", default="eq,coils,trace,transport,figures,md")
ap.add_argument("--device", default="gpu")
ap.add_argument("--ref", default=None, help="a DESC example name: equilibrium metrics only, written to --out")
ap.add_argument("--out", default=None)
A = ap.parse_args()
STAGES = ["eq"] if A.ref else A.stages.split(",")
if "transport" in STAGES or A.ref:
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
if A.device == "gpu":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
REPO = os.environ.get("DESC_REPO") or os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, "lib"))
sys.path.insert(0, REPO)
from desc import set_device  # noqa: E402

set_device(A.device)
import jax  # noqa: E402
import desc  # noqa: E402

print(f"DESC {os.path.dirname(os.path.realpath(desc.__file__))}", flush=True)
print(f"DEVICE requested={A.device} obtained={jax.devices()[0].platform} ({jax.devices()[0].device_kind})", flush=True)
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import desc.examples  # noqa: E402
import desc.io  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.vmec_utils import ptolemy_linear_transform  # noqa: E402

RHO4 = [0.25, 0.5, 0.75, 1.0]
RHO3 = [0.25, 0.5, 0.75]
DATA, WORK, FIG = (os.path.join(HERE, d) for d in ("data", "work", "figures"))
for d in (WORK, FIG):
    os.makedirs(d, exist_ok=True)
CFG = json.load(open(os.path.join(HERE, "design.json"))) if not A.ref else {"helicity": [1, 0]}
HEL = tuple(CFG["helicity"])
MJ = A.out or os.path.join(HERE, "metrics.json")
M = json.load(open(MJ)) if os.path.exists(MJ) and not A.ref else {}


def save():
    json.dump(M, open(MJ, "w"), indent=1)


def load_eq():
    if A.ref:
        eq = desc.examples.get(A.ref)
        return eq[-1] if hasattr(eq, "__len__") else eq
    eq = desc.io.load(os.path.join(DATA, "eq.h5"))
    return eq[-1] if hasattr(eq, "__len__") else eq


def load_coils():
    return desc.io.load(os.path.join(DATA, "coils.h5"))


def src_grid(cs):
    import common as S

    return S.source_grid(cs, 64, 400)[0]


def surf_grid(eq, rho, mult=2, sym=False):
    return LinearGrid(rho=np.atleast_1d(rho), M=mult * eq.M_grid, N=mult * eq.N_grid, NFP=eq.NFP, sym=sym)


# ---------------------------------------------------------------------------------------------------------------------
def boozer_breaking(eq, rho):
    """Symmetry-breaking Boozer amplitudes of |B| in the literature convention (cos/sin(m theta_B - n zeta_B), as
    QuasisymmetryBoozer), divided by B_00; with the mode labels (cos/sin, m, n)."""
    g = surf_grid(eq, rho)
    d = eq.compute(["|B|_mn_B", "B modes"], grid=g, M_booz=2 * eq.M, N_booz=2 * eq.N)
    modes = np.asarray(d["B modes"])
    bmn = np.asarray(d["|B|_mn_B"]).ravel()
    mat, vm, idx = ptolemy_linear_transform(modes, helicity=HEL, NFP=eq.NFP)
    b = mat @ bmn
    b00 = abs(float(bmn[(modes[:, 1] == 0) & (modes[:, 2] == 0)][0]))
    return b[idx] / b00, vm[idx]


def stage_eq():
    t = time.time()
    eq = load_eq()
    d0 = eq.compute(["R0", "a", "R0/a", "V", "<|B|>_vol", "<|B|>_axis"])
    out = dict(NFP=int(eq.NFP), LMN=[int(eq.L), int(eq.M), int(eq.N)], R0=float(d0["R0"]), a=float(d0["a"]),
               A=float(d0["R0/a"]), volume=float(d0["V"]), B_vol=float(d0["<|B|>_vol"]),
               B_axis=float(d0["<|B|>_axis"]), helicity=list(HEL))
    # profiles on a radial grid
    rr = np.linspace(0.0, 1.0, 21)
    rr[0] = 1e-3
    ig = LinearGrid(rho=rr, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
    dp = eq.compute(["iota", "iota_r", "magnetic well", "mirror ratio", "V_r(r)", "V_rr(r)", "psi_r", "psi_rr"],
                    grid=ig)
    c = ig.compress
    iota, iota_r = np.asarray(c(dp["iota"])), np.asarray(c(dp["iota_r"]))
    psi_r, psi_rr = np.asarray(c(dp["psi_r"])), np.asarray(c(dp["psi_rr"]))
    Vr, Vrr = np.asarray(c(dp["V_r(r)"])), np.asarray(c(dp["V_rr(r)"]))
    Vpp = (Vrr * psi_r - Vr * psi_rr) / psi_r**3  # d2V/dpsi2, psi = toroidal flux / 2 pi (DESC)
    out["profiles"] = dict(rho=rr.tolist(), iota=iota.tolist(), shear_hat=(-rr * iota_r / iota).tolist(),
                           magnetic_well=np.asarray(c(dp["magnetic well"])).tolist(),
                           mirror_ratio=np.asarray(c(dp["mirror ratio"])).tolist(), Vpp_psi=Vpp.tolist())
    out["iota_axis"], out["iota_edge"] = float(iota[0]), float(iota[-1])
    # elongation of the cross-sections (DESC a_major/a_minor on each zeta plane of the boundary)
    zz = np.linspace(0, np.pi / eq.NFP, 9)
    el = []
    for z in zz:
        gz = LinearGrid(rho=np.array([1.0]), theta=np.linspace(0, 2 * np.pi, 361), zeta=np.array([z]), NFP=eq.NFP)
        dz = eq.compute(["R", "Z"], grid=gz)
        x, y = np.asarray(dz["R"])[:-1], np.asarray(dz["Z"])[:-1]
        # ellipse with the same area second moments as the cross-section (polygon, Green's theorem)
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        cr = x * y1 - x1 * y
        ar = cr.sum() / 2
        cx, cy = ((x + x1) * cr).sum() / (6 * ar), ((y + y1) * cr).sum() / (6 * ar)
        ixx = ((x**2 + x * x1 + x1**2) * cr).sum() / (12 * ar) - cx**2
        iyy = ((y**2 + y * y1 + y1**2) * cr).sum() / (12 * ar) - cy**2
        ixy = ((x * y1 + 2 * x * y + 2 * x1 * y1 + x1 * y) * cr).sum() / (24 * ar) - cx * cy
        ev = np.sort(np.linalg.eigvalsh(np.array([[ixx, ixy], [ixy, iyy]])))
        el.append(float(np.sqrt(ev[1] / ev[0])))
    ga = LinearGrid(rho=np.array([1.0]), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
    out["elongation"] = dict(zeta=zz.tolist(), second_moment=el,
                             desc_a_major_over_a_minor=float(np.max(eq.compute("a_major/a_minor", grid=ga)
                                                                    ["a_major/a_minor"])))
    # Jacobian: no folds anywhere
    jac = {}
    for r in (0.5, 0.8, 0.9, 1.0):
        s = np.asarray(eq.compute("sqrt(g)", grid=surf_grid(eq, r))["sqrt(g)"])
        jac[f"{r:g}"] = dict(min_over_median=float(s.min() / np.median(s)), negative_fraction=float((s < 0).mean()))
    out["jacobian"] = jac
    # QS: Boozer (literature convention) and the two-term measure
    qs = dict(rho=RHO4, l2=[], max=[], two_term=[], top_modes=[])
    for r in RHO4:
        br, lab = boozer_breaking(eq, r)
        qs["l2"].append(float(np.linalg.norm(br)))
        qs["max"].append(float(np.abs(br).max()))
        top = np.argsort(-np.abs(br))[:5]
        qs["top_modes"].append([[int(lab[i, 1]), int(lab[i, 2]), "cos" if lab[i, 0] > 0 else "sin", float(br[i])]  # n in units of NFP
                                for i in top])
        g = surf_grid(eq, r)
        d = eq.compute(["f_C", "|B|", "sqrt(g)"], grid=g, helicity=HEL)
        w = np.abs(np.asarray(d["sqrt(g)"])) * np.asarray(g.weights)
        f = np.asarray(d["f_C"]) / np.asarray(d["|B|"]) ** 3
        qs["two_term"].append(float(np.sqrt(np.sum(w * f**2) / np.sum(w))))
    qs["convention"] = "cos/sin(m theta_B - n zeta_B) amplitudes / B_00, M_booz = 2M, N_booz = 2N (QuasisymmetryBoozer)"
    # breaking spectrum vs rho for the figure
    if not A.ref:
        rs = np.linspace(0.05, 1.0, 20)
        spec = [boozer_breaking(eq, r) for r in rs]
        np.savez(os.path.join(WORK, "booz_spectrum.npz"), rho=rs, amps=np.array([s[0] for s in spec]),
                 modes=spec[0][1])
    out["qs"] = qs
    out["minutes"] = (time.time() - t) / 60
    M["equilibrium"] = out
    save()
    print(f"[eq] A {out['A']:.3f} iota {out['iota_axis']:.4f} -> {out['iota_edge']:.4f}  QS L2 % "
          + " / ".join(f"{100 * v:.3f}" for v in qs["l2"]) + "  max % " + " / ".join(f"{100 * v:.3f}" for v in qs["max"])
          + f"  ({out['minutes']:.1f} min)", flush=True)
    if A.ref:
        M["transport"] = transport(eq, res=("hi",))
        save()


# ---------------------------------------------------------------------------------------------------------------------
def hinge_geometry(c):
    """Hinge points (lab frame, from the coil's break parameters), the hinge line's tilt, the faces' sizes and the
    fold angle between the two faces' planes."""
    import common as S

    brk = np.sort(np.asarray(S.coil_breaks(c)) % (2 * np.pi))
    from desc.grid import Grid

    nodes = np.stack([0 * brk, 0 * brk, brk], 1)
    H = np.asarray(c.compute("x", grid=Grid(nodes, sort=False, jitable=False), basis="xyz")["x"])
    faces = []
    normals = []
    for i in range(len(brk)):
        s0, s1 = brk[i], brk[(i + 1) % len(brk)] + (2 * np.pi if i == len(brk) - 1 else 0)
        s = np.linspace(s0, s1, 400) % (2 * np.pi)
        P = np.asarray(c.compute("x", grid=Grid(np.stack([0 * s, 0 * s, s], 1), sort=False, jitable=False),
                                 basis="xyz")["x"])
        Pc = P - P.mean(0)
        n = np.linalg.svd(Pc)[2][2]
        normals.append(n)
        u = (H[1] - H[0]) / np.linalg.norm(H[1] - H[0]) if len(H) > 1 else np.zeros(3)
        dv = P - H[0]
        height = np.linalg.norm(dv - np.outer(dv @ u, u), axis=1).max()  # face extent away from the hinge line
        faces.append(dict(length=float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))), height=float(height),
                          planarity_rms=float(np.sqrt(np.mean((Pc @ n) ** 2)))))
    fold = float(np.degrees(np.arccos(np.clip(abs(normals[0] @ normals[1]), -1, 1)))) if len(normals) == 2 else None
    dz = H[1:, 2] - H[0, 2]
    tilt = np.degrees(np.arctan2(dz, np.hypot(H[1:, 0] - H[0, 0], H[1:, 1] - H[0, 1])))
    return dict(hinges_xyz=H.tolist(), hinge_z=H[:, 2].tolist(), hinge_R=np.hypot(H[:, 0], H[:, 1]).tolist(),
                tilt_deg=tilt.tolist(), hinge_line_length=float(np.linalg.norm(H[1] - H[0])) if len(H) > 1 else None,
                faces=faces, fold_angle_deg=fold)


def boundary_maps(eq, cs, src, nt=96, nz=128):
    """B.n / <|B|> and the distance to the nearest coil on the boundary, one field period."""
    from scipy.spatial import cKDTree

    g = LinearGrid(rho=np.array([1.0]), theta=np.linspace(0, 2 * np.pi, nt, endpoint=False),
                   zeta=np.linspace(0, 2 * np.pi / eq.NFP, nz, endpoint=False), NFP=eq.NFP)
    d = eq.compute(["x", "n_rho", "|e_theta x e_zeta|"], grid=g, basis="xyz")
    X = np.asarray(d["x"])
    B = np.asarray(cs.compute_magnetic_field(X, basis="xyz", source_grid=src))
    w = np.asarray(d["|e_theta x e_zeta|"])
    Bm = np.sum(w * np.linalg.norm(B, axis=1)) / np.sum(w)
    bn = np.sum(B * np.asarray(d["n_rho"]), 1) / Bm
    pos = np.asarray(cs._compute_position(grid=LinearGrid(N=2000))).reshape(-1, 3)
    dist = cKDTree(pos).query(X)[0]
    sh = (nz, nt)  # LinearGrid orders theta fastest
    return dict(theta=g.nodes[:nt, 1], zeta=g.nodes[::nt, 2], bn=bn.reshape(sh), dpc=dist.reshape(sh),
                bn_rms=float(np.sqrt(np.sum(w * bn**2) / np.sum(w))), bn_max=float(np.abs(bn).max()))


def stage_coils():
    import common as S

    t = time.time()
    eq, cs = load_eq(), load_coils()
    src = src_grid(cs)
    bounds = json.load(open(os.path.join(DATA, "bounds.json")))
    P = S.paper_bounds(eq, S.n_unique(cs), None, os.path.join(DATA, "bounds.json"))
    P = {k: P[k] for k in S.UPPER + S.LOWER}
    m = S.evaluate(cs, eq, vacuum=True)
    per = []
    for i, (c, pm) in enumerate(zip(S.iter_unique(cs), m["per_coil"])):
        hg = hinge_geometry(c)
        per.append(dict(coil=i + 1, current_A=float(c.current), length=pm["L"], kappa_max=pm["kappa"],
                        min_bend_radius=1.0 / pm["kappa"], corner_deg=pm["corner"], **hg))
    bm = boundary_maps(eq, cs, src)
    np.savez(os.path.join(WORK, "boundary_maps.npz"), **{k: v for k, v in bm.items() if not np.isscalar(v)})
    # B.n at the free-boundary solver's own resolution and weighting (fb_rescore.bn_rms)
    g = LinearGrid(M=2 * eq.M_grid, N=2 * eq.N_grid, NFP=eq.NFP, sym=False)
    d = eq.compute(["x", "n_rho", "|e_theta x e_zeta|"], grid=g, basis="xyz")
    B = np.asarray(cs.compute_magnetic_field(np.asarray(d["x"]), basis="xyz", source_grid=src))
    w = np.asarray(g.weights) * np.asarray(d["|e_theta x e_zeta|"])
    Bm = np.sum(w * np.linalg.norm(B, axis=1)) / np.sum(w)
    bnv = np.sum(B * np.asarray(d["n_rho"]), 1) / Bm
    M["coils"] = dict(
        n_unique=len(per), n_total=int(m["n_coils"]), symmetry=f"NFP {eq.NFP}, stellarator symmetric",
        bounds=bounds, margins=S.margins(m, P), verdict=S.verdict(m, P),
        L_max=m["L"], L_total_unique=m["L_total"], kappa_max=m["kappa"], d_cc=m["d_cc"], d_pc=m["d_pc"],
        linked=m["linked"], current_sum_unique_A=float(sum(p["current_A"] for p in per)),
        hinge_abs_z_max=float(max(max(abs(z) for z in p["hinge_z"]) for p in per)),
        hinge_tilt_abs_max_deg=float(max(max(abs(v) for v in p["tilt_deg"]) for p in per)),
        bn_rms=float(np.sqrt(np.sum(w * bnv**2) / np.sum(w))), bn_max=float(np.abs(bnv).max()),
        dpc_on_boundary_min=float(bm["dpc"].min()), dpc_on_boundary_median=float(np.median(bm["dpc"])),
        per_coil=per, minutes=(time.time() - t) / 60)
    save()
    c = M["coils"]
    print(f"[coils] {c['verdict']}  " + "  ".join(f"{k} {v:.4f}" for k, v in c["margins"].items())
          + f"  B.n rms {c['bn_rms']:.2e} max {c['bn_max']:.2e}  hinge |z| max {c['hinge_abs_z_max']:.4f}  "
          f"tilt max {c['hinge_tilt_abs_max_deg']:.2f} deg  ({c['minutes']:.1f} min)", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
def stage_trace():
    """Rotation number of the coil field (as ss/iota_trace.py; Birkhoff average about the traced axis)."""
    from desc.magnetic_fields import field_line_integrate

    t0 = time.time()
    eq, cs = load_eq(), load_coils()
    src = src_grid(cs)
    dense = os.environ.get("TRACE", "dense") == "dense"
    NL, NPER, SPP = (80, 2000, 16) if dense else (30, 500, 16)
    NFP = eq.NFP
    TP = 2 * np.pi / NFP

    def trace(R, Z, phis):
        r, z = field_line_integrate(np.atleast_1d(R), np.atleast_1d(Z), phis, cs, source_grid=src, rtol=1e-10,
                                    atol=1e-10, bounds_R=(0.3, 1.4), bounds_Z=(-0.6, 0.6),
                                    max_steps=int(4000 * (phis[-1] - phis[0]) / TP) + 1000, chunk_size=np.size(R))
        return np.asarray(r), np.asarray(z)

    da = eq.compute(["R", "Z"], grid=LinearGrid(rho=np.array([0.0]), theta=np.array([0.0]), zeta=np.array([0.0]),
                                                NFP=NFP))
    x = np.array([float(da["R"][0]), float(da["Z"][0])])
    axis_eq = x.copy()
    h = 1e-5
    for _ in range(8):  # axis = fixed point of the one-period map
        r, z = trace([x[0], x[0] + h, x[0]], [x[1], x[1], x[1] + h], np.array([0.0, TP]))
        Pm = np.column_stack([r[-1], z[-1]])
        F = Pm[0] - x
        J = np.column_stack([(Pm[1] - Pm[0]) / h, (Pm[2] - Pm[0]) / h]) - np.eye(2)
        x = x - np.linalg.solve(J, F)
        if np.linalg.norm(F) < 1e-11:
            break
    ra, za = trace(x[0], x[1], np.arange(SPP + 1) * TP / SPP)
    ra, za = ra[:-1, 0], za[:-1, 0]
    Rb = float(eq.compute("R", grid=LinearGrid(rho=np.array([1.0]), theta=np.array([0.0]), zeta=np.array([0.0]),
                                               NFP=NFP))["R"][0])
    s = np.linspace(0.03, 1.08, NL)
    R0 = x[0] + s * (Rb - x[0])
    phis = np.arange(NPER * SPP + 1) * TP / SPP
    r, z = trace(R0, np.full_like(R0, x[1]), phis)
    np.savez(os.path.join(WORK, "trace.npz"), r=r[:: 1, :].astype(np.float32), z=z.astype(np.float32), phis=phis,
             s=s, axis=np.column_stack([ra, za]), SPP=SPP)
    k = np.arange(len(phis)) % SPP

    def birkhoff(dth):
        tt = (np.arange(len(dth)) + 0.5) / len(dth)
        w = np.exp(-1.0 / (tt * (1 - tt)))
        return float(np.sum(w * dth) / np.sum(w))

    rho_g = np.linspace(0, 1, 201)
    gi = LinearGrid(rho=rho_g, M=eq.M_grid, N=eq.N_grid, NFP=NFP)
    iota_prof = np.asarray(gi.compress(eq.compute("iota", grid=gi)["iota"]))
    lines = []
    for j in range(NL):
        ok = np.isfinite(r[:, j]) & np.isfinite(z[:, j])
        n_ok = len(ok) if ok.all() else int(np.argmin(ok))
        row = dict(s=float(s[j]), confined=bool(n_ok == len(ok)))
        if n_ok > 20 * SPP:
            th = np.unwrap(np.arctan2(z[:n_ok, j] - za[k[:n_ok]], r[:n_ok, j] - ra[k[:n_ok]]))
            dth = np.diff(th)
            row["iota"] = birkhoff(dth) / (TP / SPP)
            row["iota_err"] = abs(birkhoff(dth[: len(dth) // 2]) - birkhoff(dth[len(dth) // 2:])) / (TP / SPP)
        idx = np.arange(0, n_ok, SPP)[:: max(1, (n_ok // SPP) // 250)]
        try:
            fc = np.asarray(eq.map_coordinates(np.column_stack([r[idx, j], 0 * idx, z[idx, j]]),
                                               inbasis=("R", "phi", "Z"), outbasis=("rho", "theta", "zeta"),
                                               maxiter=60, tol=1e-10))
            rh = fc[:, 0]
            if np.all(np.isfinite(rh)) and np.all(rh < 1 - 1e-9):
                row.update(rho_mean=float(rh.mean()), rho_std=float(rh.std()),
                           iota_eq=float(np.interp(rh.mean(), rho_g, iota_prof)))
        except Exception as e:  # noqa: BLE001
            row["map_error"] = repr(e)[:120]
        lines.append(row)
    inside = [L for L in lines if "iota_eq" in L and "iota" in L]
    diff = [abs(abs(L["iota"]) - abs(L["iota_eq"])) for L in inside]
    flagged = [L for L in lines if L.get("iota_err", 0) > 1e-4]
    io = [abs(L["iota"]) for L in lines if "iota" in L and "iota_eq" in L]
    # rationals iota = p/q (reduced) in the traced range whose resonant harmonic is allowed by the field periods:
    # (m, n) = k (q, p) with n a multiple of NFP, i.e. k = NFP / gcd(p, NFP); island count m = k q <= 24 kept
    rats = []
    for q in range(1, 25):
        for p in range(1, q):
            k = NFP // np.gcd(p, NFP)
            if np.gcd(p, q) == 1 and k * q <= 24 and min(io) <= p / q <= max(io):
                rats.append((p, q, p / q))
    M["trace"] = dict(
        settings=dict(lines=NL, field_periods=NPER, samples_per_period=SPP, start="Z = 0, phi = 0, s = 0.03-1.08 of "
                      "the way from the traced axis to the outboard boundary"),
        axis_traced_RZ=x.tolist(), axis_eq_RZ=axis_eq.tolist(), axis_offset_mm=float(1e3 * np.hypot(*(x - axis_eq))),
        all_confined=all(L["confined"] for L in lines), iota_axis_traced=io[0] if io else None,
        iota_edge_traced=io[-1] if io else None, max_abs_iota_trace_minus_eq=float(max(diff)) if diff else None,
        max_rho_std=float(max(L["rho_std"] for L in inside)) if inside else None,
        rationals_inside=[f"{p}/{q}" for p, q, _ in sorted(rats, key=lambda v: -v[2])],
        rationals_island_count={f"{p}/{q}": int(NFP // np.gcd(p, NFP) * q) for p, q, _ in rats},
        flagged_lines=[dict(s=L["s"], iota=L.get("iota"), iota_err=L.get("iota_err"), rho_mean=L.get("rho_mean"),
                            rho_std=L.get("rho_std")) for L in flagged],
        lines=lines, minutes=(time.time() - t0) / 60)
    save()
    T = M["trace"]
    print(f"[trace] axis offset {T['axis_offset_mm']:.3f} mm, iota {T['iota_axis_traced']:.4f} -> "
          f"{T['iota_edge_traced']:.4f}, |iota_trace - iota_eq| <= {T['max_abs_iota_trace_minus_eq']:.1e}, "
          f"rationals {T['rationals_inside']}, flagged {len(flagged)} ({T['minutes']:.1f} min)", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
TRES = dict(lo=dict(X=32, Y=32, Y_B=128, nt=20, npitch=45), hi=dict(X=48, Y=64, Y_B=256, nt=40, npitch=90))


def transport(eq, which=("effective ripple", "Gamma_c", "Gamma_c Velasco"), res=("lo", "hi")):
    from desc.integrals import Bounce2D

    rr = np.array(RHO3)
    out = dict(rho=RHO3)
    for tag in res:
        p = TRES[tag]
        out[tag] = dict(settings=p)
        ge = LinearGrid(rho=rr, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
        ang = Bounce2D.angle(eq, X=p["X"], Y=p["Y"], rho=rr)
        for q in which:
            t = time.time()
            try:
                d = eq.compute(q, grid=ge, angle=ang, Y_B=p["Y_B"], num_transit=p["nt"], num_well=p["nt"] * 30,
                               num_quad=32, num_pitch=p["npitch"])
                out[tag][q] = np.asarray(ge.compress(d[q])).tolist()
            except Exception as e:  # noqa: BLE001
                out[tag][q] = repr(e)[:200]
            v = out[tag][q]
            print(f"[transport] {tag} {q:16s} " + (" / ".join(f"{x:.3e}" for x in v) if isinstance(v, list) else v)
                  + f" ({time.time() - t:.0f} s)", flush=True)
    return out


def stage_transport():
    t = time.time()
    M["transport"] = transport(load_eq())
    M["transport"]["minutes"] = (time.time() - t) / 60
    save()


# ---------------------------------------------------------------------------------------------------------------------
def stage_figures():
    from desc.plotting import plot_boozer_surface

    eq, cs = load_eq(), load_coils()
    E = M.get("equilibrium", {})
    name = CFG["name"]
    NFP = eq.NFP
    # 1. Poincare of the coil field over the equilibrium's surfaces, three planes
    tf = os.path.join(WORK, "trace.npz")
    if os.path.exists(tf):
        T = np.load(tf)
        SPP = int(T["SPP"])
        planes = [0, SPP // 4, SPP // 2]
        fig, axs = plt.subplots(1, 3, figsize=(17, 6.2))
        for ax, kp in zip(axs, planes):
            phi = kp * 2 * np.pi / NFP / SPP
            for rho in np.linspace(0.1, 1.0, 10):
                gg = LinearGrid(rho=np.array([rho]), theta=np.linspace(0, 2 * np.pi, 361), zeta=np.array([phi]),
                                NFP=NFP)
                dd = eq.compute(["R", "Z"], grid=gg)
                ax.plot(dd["R"], dd["Z"], "-", color="red", lw=1.4 if rho == 1 else 0.6,
                        zorder=3)
            r, z = T["r"][kp::SPP], T["z"][kp::SPP]
            for j in range(0, r.shape[1], max(1, r.shape[1] // 26)):  # ~26 of the traced lines
                ax.plot(r[:, j], z[:, j], ".", ms=0.5, color=plt.cm.viridis(j / r.shape[1]), rasterized=True)
            ax.plot(T["axis"][kp, 0], T["axis"][kp, 1], "k+", ms=10)
            ax.set_aspect("equal")
            ax.set_title(f"phi = {phi:.3f} ({kp}/{SPP} of a field period)")
            ax.set_xlabel("R (m)")
        axs[0].set_ylabel("Z (m)")
        fig.suptitle(f"{name}: field lines of the coils (dots) over the free-boundary equilibrium's surfaces "
                     f"(red; thick = boundary)")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "poincare.png"), dpi=150)
        plt.close(fig)
        # iota: equilibrium vs traced
        L = M.get("trace", {}).get("lines", [])
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
        pr = E["profiles"]
        axs[0].plot(pr["rho"], np.abs(pr["iota"]), "C0-", label="equilibrium")
        pts = [(l_["rho_mean"], abs(l_["iota"]), l_["rho_std"]) for l_ in L if "rho_mean" in l_ and "iota" in l_]
        if pts:
            a_, b_, c_ = np.array(pts).T
            axs[0].errorbar(a_, b_, xerr=c_, fmt="o", ms=3, mfc="none", color="k",
                            label="coil field, traced (at its rho in the equilibrium)")
        lo, hi = axs[0].get_ylim()
        for rat in M.get("trace", {}).get("rationals_inside", []):
            p, q = (int(v) for v in rat.split("/"))
            axs[0].axhline(p / q, color="C1", lw=0.6, ls=":")
            axs[0].text(1.0, p / q, f" {rat}", color="C1", fontsize=7, va="center")
        axs[0].set_xlabel("rho")
        axs[0].set_ylabel("iota")
        axs[0].legend(fontsize=8)
        axs[0].set_title("rotational transform")
        ok = [l_ for l_ in L if "iota" in l_]
        axs[1].semilogy([l_["s"] for l_ in ok], [max(l_["iota_err"], 1e-12) for l_ in ok], "k.-", ms=3)
        axs[1].axhline(1e-4, color="C3", lw=0.6, ls="--")
        axs[1].set_xlabel("start position (fraction of axis -> outboard boundary)")
        axs[1].set_ylabel("half-vs-half rotation number difference")
        axs[1].set_title("convergence of each traced line (spikes = island / resonance)")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "iota.png"), dpi=150)
        plt.close(fig)
    # 2. |B| in Boozer coordinates
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.6))
    for ax, r in zip(axs, RHO4):
        plot_boozer_surface(eq, rho=r, ax=ax, M_booz=2 * eq.M, N_booz=2 * eq.N, ncontours=40)
        ax.set_title(f"|B| on rho = {r}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "boozer_B.png"), dpi=130)
    plt.close(fig)
    # 3. breaking modes vs rho
    sp = np.load(os.path.join(WORK, "booz_spectrum.npz"))
    amps, modes, rs = sp["amps"], sp["modes"], sp["rho"]
    top = np.argsort(-np.abs(amps).max(0))[:8]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i in top:
        ax.plot(rs, 100 * np.abs(amps[:, i]), "-", label=f"{'cos' if modes[i, 0] > 0 else 'sin'} m={modes[i, 1]} "
                f"n={modes[i, 2]} (x NFP)")
    ax.plot(rs, 100 * np.linalg.norm(amps, axis=1), "k-", lw=2, label="L2 of all breaking modes")
    ax.set_xlabel("rho")
    ax.set_ylabel("% of B_00")
    ax.set_title(f"{name}: symmetry-breaking Boozer modes of |B|")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "qs_modes.png"), dpi=150)
    plt.close(fig)
    # 4. eps_eff and Gamma_c vs rho, with the references (compare/refs.json if present)
    tr = M.get("transport")
    if tr:
        refs_f = os.path.join(HERE, "..", "compare", "references.json")
        refs = json.load(open(refs_f)) if os.path.exists(refs_f) else {}
        fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
        for ax, q in zip(axs, ("effective ripple", "Gamma_c", "Gamma_c Velasco")):
            ax.semilogy(tr["rho"], tr["hi"][q], "ko-", lw=2, label=name)
            for i, (rn, rv) in enumerate(refs.items()):
                v = rv.get("transport", {}).get("hi", {}).get(q)
                if isinstance(v, list):
                    ax.semilogy(rv["transport"]["rho"], v, "s--", ms=4, color=f"C{i}", label=rn)
            ax.set_xlabel("rho")
            ax.set_title(q.replace("effective ripple", "eps_eff"))
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "transport.png"), dpi=150)
        plt.close(fig)
    # 5. B.n and coil distance on the boundary
    bf = os.path.join(WORK, "boundary_maps.npz")
    if os.path.exists(bf):
        bm = np.load(bf)
        fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
        for ax, key, lab, cmap in ((axs[0], "bn", "B.n / <|B|>", "RdBu_r"), (axs[1], "dpc", "distance to the nearest "
                                                                                  "coil (m)", "viridis")):
            v = bm[key]
            kw = dict(vmin=-np.abs(v).max(), vmax=np.abs(v).max()) if key == "bn" else {}
            pc = ax.pcolormesh(bm["zeta"], bm["theta"], v.T, cmap=cmap, shading="auto", **kw)
            fig.colorbar(pc, ax=ax, label=lab)
            ax.set_xlabel("zeta (one field period)")
            ax.set_ylabel("theta")
        axs[0].set_title(f"coils' normal field on the boundary (rms {M['coils']['bn_rms']:.1e})")
        axs[1].set_title(f"coil-plasma distance (min {bm['dpc'].min():.4f} m; bound "
                         f"{M['coils']['bounds']['d_pc']:.4f})")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "boundary_maps.png"), dpi=150)
        plt.close(fig)
    # 6. cross-sections, half a field period
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, phi in enumerate(np.linspace(0, np.pi / NFP, 5)):
        gg = LinearGrid(rho=np.array([1.0]), theta=np.linspace(0, 2 * np.pi, 361), zeta=np.array([phi]), NFP=NFP)
        dd = eq.compute(["R", "Z"], grid=gg)
        ax.plot(dd["R"], dd["Z"], "-", color=f"C{i}", label=f"phi = {phi:.2f}")
    ax.axhline(0, color="0.7", lw=0.5)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(f"{name}: boundary cross-sections over half a field period")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "cross_sections.png"), dpi=150)
    plt.close(fig)
    # 7. coils and plasma: top view (all coils) and 3D (the unique coils of one half period), hinges marked
    fig = plt.figure(figsize=(16, 7.5))
    ax = fig.add_subplot(1, 2, 1)
    gs = LinearGrid(rho=np.array([1.0]), theta=np.linspace(0, 2 * np.pi, 48), zeta=np.linspace(0, 2 * np.pi, 160),
                    NFP=1)
    X = np.asarray(eq.compute("x", grid=gs, basis="xyz")["x"]).reshape(160, 48, 3)
    pos = np.asarray(cs._compute_position(grid=LinearGrid(N=400)))
    nu = M.get("coils", {}).get("n_unique", 1)
    Rx = np.hypot(X[..., 0], X[..., 1])
    for f_ in (np.argmin, np.argmax):
        k_ = f_(Rx, axis=1)
        ax.plot(X[np.arange(160), k_, 0], X[np.arange(160), k_, 1], "-", color="C1", lw=2)
    for j_, p_ in enumerate(pos):
        ax.plot(p_[:, 0], p_[:, 1], color=f"C{j_ % nu}", lw=0.9)
    for pc in M.get("coils", {}).get("per_coil", []):
        H = np.array(pc["hinges_xyz"])
        ax.plot(H[:, 0], H[:, 1], "k.", ms=7)
    ax.set_aspect("equal")
    ax.set_title("top view: all coils (colour = unique coil), plasma's inner/outer edge (orange), unique hinges")
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    hp = slice(0, 160 // (2 * NFP) + 2)
    ax.plot_surface(X[hp, :, 0], X[hp, :, 1], X[hp, :, 2], color="C1", alpha=0.5, lw=0)
    for j_ in range(nu):
        ax.plot(pos[j_][:, 0], pos[j_][:, 1], pos[j_][:, 2], color=f"C{j_}", lw=1.5)
    for pc in M.get("coils", {}).get("per_coil", []):
        H = np.array(pc["hinges_xyz"])
        ax.plot(H[:, 0], H[:, 1], H[:, 2], "k.", ms=7)
    lim = np.abs(np.concatenate([pos[:nu].reshape(-1, 3)])).max()
    ax.set_xlim(-0.2, lim)
    ax.set_ylim(-0.2, lim)
    ax.set_zlim(-lim / 2, lim / 2)
    ax.set_box_aspect((1, 1, 0.5))
    ax.view_init(25, -50)
    ax.set_title("the unique coils of one half period over that part of the plasma")
    fig.suptitle(f"{name}: coils")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "coils.png"), dpi=140)
    plt.close(fig)
    print(f"[figures] wrote {sorted(os.listdir(FIG))}", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
def f3(v, fmt="{:.3f}", scale=1.0):
    return " / ".join(fmt.format(scale * x) if isinstance(x, (int, float)) else str(x) for x in v)


def stage_md():
    E, C, T, X = (M.get(k, {}) for k in ("equilibrium", "coils", "trace", "transport"))
    pr = E["profiles"]
    rr = np.array(pr["rho"])

    def at(key, rhos=RHO4):
        return [float(np.interp(r, rr, pr[key])) for r in rhos]

    L = []
    w = L.append
    w(f"# {CFG['name']}: metrics\n")
    w(f"Generated by `make_figures.py` from `data/` (do not edit by hand; numbers also in `metrics.json`). "
      f"{CFG.get('one_line', '')}\n")
    w("## Equilibrium (vacuum, free boundary with these coils)\n")
    w("| quantity | value |\n|---|---|")
    w(f"| NFP, helicity (M, N) | {E['NFP']}, {tuple(E['helicity'])} |")
    w(f"| resolution L, M, N | {E['LMN']} |")
    w(f"| R0, a, aspect ratio | {E['R0']:.4f} m, {E['a']:.4f} m, {E['A']:.3f} |")
    w(f"| volume, <\\|B\\|>_vol, <\\|B\\|>_axis | {E['volume']:.4f} m^3, {E['B_vol']:.4f} T, {E['B_axis']:.4f} T |")
    w(f"| iota at rho 0 / .25 / .5 / .75 / 1 | {f3(at('iota', [0] + RHO4), '{:.4f}')} |")
    w(f"| magnetic shear -(rho/iota) diota/drho at rho .25 / .5 / .75 / 1 | {f3(at('shear_hat'), '{:+.3f}')} |")
    w(f"| magnetic well (DESC, > 0 = well) at rho .25 / .5 / .75 / 1 | {f3(at('magnetic_well'), '{:+.2e}')} |")
    w(f"| V''(psi) at rho .25 / .5 / .75 / 1 (< 0 = well) | {f3(at('Vpp_psi'), '{:+.3e}')} |")
    w(f"| mirror ratio (Bmax-Bmin)/(Bmax+Bmin) at rho .25 / .5 / .75 / 1 | {f3(at('mirror_ratio'), '{:.4f}')} |")
    el = E["elongation"]
    w(f"| elongation of the boundary cross-sections (second-moment ellipse), phi 0 .. pi/NFP | "
      f"{f3(el['second_moment'], '{:.2f}')} (max {max(el['second_moment']):.2f}); DESC a_major/a_minor "
      f"{el['desc_a_major_over_a_minor']:.2f} |")
    jm = min(v["min_over_median"] for v in E["jacobian"].values())
    w(f"| Jacobian sqrt(g) min / median at rho .5 / .8 / .9 / 1 | "
      f"{f3([v['min_over_median'] for v in E['jacobian'].values()], '{:+.3f}')} (folds: "
      f"{'none' if jm > 0 else 'YES'}) |\n")
    q = E["qs"]
    w("### Quasisymmetry (literature convention: Boozer cos/sin(m theta_B - n zeta_B) amplitudes / B_00)\n")
    w("| rho | 0.25 | 0.5 | 0.75 | 1 |\n|---|---|---|---|---|")
    w("| L2 of the breaking modes (%) | " + " | ".join(f"{100 * v:.3f}" for v in q["l2"]) + " |")
    w("| largest breaking mode (%) | " + " | ".join(f"{100 * v:.3f}" for v in q["max"]) + " |")
    w("| largest mode (m, n), n in units of NFP | " + " | ".join(f"({t[0][0]}, {t[0][1]})" for t in q["top_modes"]) + " |")
    w("| two-term rms(f_C / B^3) | " + " | ".join(f"{v:.2e}" for v in q["two_term"]) + " |\n")
    if X:
        w("### Transport proxies (Bounce2D; `lo` and `hi` resolution as a convergence check)\n")
        w("| rho | " + " | ".join(f"{r}" for r in X["rho"]) + " |\n|---|---|---|---|")
        for qn, lab in (("effective ripple", "eps_eff"), ("Gamma_c", "Gamma_c (Nemov)"),
                        ("Gamma_c Velasco", "Gamma_c (Velasco)")):
            for tag in ("hi", "lo"):
                v = X.get(tag, {}).get(qn)
                if isinstance(v, list):
                    w(f"| {lab}, {tag} | " + " | ".join(f"{x:.2e}" for x in v) + " |")
        w("\nTreat Gamma_c as +-20% (the lo -> hi change reaches 20-30% for the smallest values).\n")
    if T:
        w("## Field-line tracing of the coil field (independent of the equilibrium)\n")
        s = T["settings"]
        w(f"{s['lines']} lines x {s['field_periods']} field periods ({s['start']}); iota = Birkhoff-weighted rotation "
          f"number about the traced axis.\n")
        w("| quantity | value |\n|---|---|")
        w(f"| traced iota, axis -> outermost line inside the boundary | {T['iota_axis_traced']:.4f} -> "
          f"{T['iota_edge_traced']:.4f} |")
        w(f"| max \\|iota_traced - iota_eq\\| (at each line's rho) | {T['max_abs_iota_trace_minus_eq']:.1e} |")
        w(f"| traced axis vs equilibrium axis | {T['axis_offset_mm']:.3f} mm |")
        w(f"| largest rho spread of a traced line in the equilibrium | {T['max_rho_std']:.1e} |")
        w(f"| all lines out to 1.08x the boundary confined | {T['all_confined']} |")
        ric = T.get("rationals_island_count", {})
        w("| rationals iota = p/q inside whose resonance the field periods allow (islands in the chain; <= 24 listed) | "
          + (", ".join(f"{r} ({ric.get(r, '?')})" for r in T["rationals_inside"]) or "none") + " |")
        fl = "; ".join(f"s {f['s']:.3f} (rho {f['rho_mean'] or float('nan'):.3f}, iota {abs(f['iota']):.5f}, "
                       f"err {f['iota_err']:.0e})" for f in T["flagged_lines"] if f.get("iota") is not None)
        w(f"| lines with half-vs-half error > 1e-4 (island candidates) | {fl or 'none'} |\n")
        if CFG.get("islands"):
            w(f"Island widths (estimated by hand from the dense trace, `final_work/NOTES.md`): {CFG['islands']}\n")
    if C:
        w("## Coils\n")
        b = C["bounds"]
        w(f"Bounds (`data/bounds.json`): L <= {b['L']:.4f} m, kappa <= {b['kappa']:.4f} /m, d_cc >= {b['d_cc']:.4f} m,"
          f" d_pc >= {b['d_pc']:.4f} m. {CFG.get('coil_class', '')}\n")
        w("| quantity | value |\n|---|---|")
        w(f"| coils | {C['n_unique']} unique, {C['n_total']} total ({C['symmetry']}) |")
        w(f"| verdict (dense check vs the bounds) | **{C['verdict']}** |")
        w("| margins (value / bound) | " + ", ".join(f"{k} {v:.4f}" for k, v in C["margins"].items()) + " |")
        w(f"| d_cc, d_pc (m) | {C['d_cc']:.4f}, {C['d_pc']:.4f} |")
        w(f"| B.n / <\\|B\\|> on the boundary, rms / max | {C['bn_rms']:.2e} / {C['bn_max']:.2e} |")
        w(f"| max \\|hinge z\\|, max hinge-line tilt | {C['hinge_abs_z_max']:.4f} m, {C['hinge_tilt_abs_max_deg']:.2f} "
          f"deg |")
        w(f"| sum of the unique coils' currents | {C['current_sum_unique_A']:.4e} A (as stored: R0 = "
          f"{E['R0']:.3f} m, <\\|B\\|> = {E['B_vol']:.3f} T) |\n")
        w("| coil | current (A) | length (m) | kappa max (1/m) | min bend radius (m) | hinge z (m) | hinge R (m) | "
          "tilt (deg) | fold angle (deg) | face lengths (m) | face heights from the hinge line (m) |")
        w("|---|---|---|---|---|---|---|---|---|---|---|")
        for p in C["per_coil"]:
            w(f"| {p['coil']} | {p['current_A']:.4e} | {p['length']:.5f} | {p['kappa_max']:.3f} | "
              f"{p['min_bend_radius']:.4f} | {f3(p['hinge_z'], '{:+.4f}')} | {f3(p['hinge_R'], '{:.4f}')} | "
              f"{f3(p['tilt_deg'], '{:+.2f}')} | {p['fold_angle_deg']:.1f} | "
              f"{f3([f['length'] for f in p['faces']], '{:.4f}')} | {f3([f['height'] for f in p['faces']], '{:.4f}')}"
              " |")
        w("")
    w("## Not yet computed (handoff step 3)\n")
    w("Coil forces and torques (incl. across the hinge), currents scaled to a reference machine, clearance with a "
      "finite winding pack, finite beta.\n")
    open(os.path.join(HERE, "METRICS.md"), "w").write("\n".join(L))
    print("[md] wrote METRICS.md", flush=True)


for st in STAGES:
    t = time.time()
    dict(eq=stage_eq, coils=stage_coils, trace=stage_trace, transport=stage_transport, figures=stage_figures,
         md=stage_md)[st]()
    print(f"=== stage {st} done in {(time.time() - t) / 60:.1f} min", flush=True)
