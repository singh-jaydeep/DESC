"""Quality of a coil set's ACTUAL vacuum field against the equilibrium it was designed for.

  trace : Poincare map of the coil field (field_line_integrate) on phi = 0 and phi = pi/NFP, started on the target's
          outboard theta = 0 line at rho = 0.1 .. 1.0 (+ 1.03, 1.06 outside). Each puncture is mapped into the target's
          flux coordinates: per line rho mean/std (surface offset / thickness; std >> 0 = islands or chaos), lost lines,
          and iota from the unwrapped poloidal angle vs the target's iota.
  freeb : vacuum free-boundary solve with the coils as the external field (VacuumBoundaryError, field fixed; Psi,
          p = 0, I = 0 fixed; every boundary mode free), started from the target. Then Boozer QS / iota / A of the
          coil-consistent plasma vs the target, and the boundary offset between them.

The coil current scale is invisible to B.n (the drivers hold the current sum fixed), so currents are first scaled
uniformly so that <|B_coil|> = <|B_eq|> on the target boundary; this does not move field lines.

Usage: fieldcheck.py NAME COILS.h5 EQ(.h5|example) OUTDIR [trace] [freeb]"""
import json
import os
import sys
import time

sys.path.insert(0, "/home/singh/Documents/DESC2/coil_sandbox")
sys.path.insert(0, "/home/singh/Documents/DESC2/coil_sandbox/sandbox")
import boot  # noqa: E402

boot.setup("gpu")
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import desc.io  # noqa: E402
import desc.objectives as O  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.magnetic_fields import field_line_integrate  # noqa: E402
import common as S  # noqa: E402
import single_stage as SS  # noqa: E402

name, coils_f, eq_f, out = sys.argv[1:5]
todo = set(sys.argv[5:]) or {"trace", "freeb"}
os.makedirs(out, exist_ok=True)
eq = S.load_equilibrium(eq_f)
if isinstance(eq, tuple):
    eq = eq[0]
cs = desc.io.load(coils_f)
src, _ = S.source_grid(cs, 24, 400)
res = {"name": name, "coils": coils_f, "eq": eq_f}
log = open(f"{out}/summary.txt", "a")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")
    log.flush()


# ---- current scale -------------------------------------------------------------------------------------------------
g = LinearGrid(rho=np.array([1.0]), M=24, N=24, NFP=eq.NFP, sym=False)
d = eq.compute(["R", "phi", "Z", "|B|", "n_rho"], grid=g)
x = np.column_stack([d["R"], d["phi"], d["Z"]])
Bc = np.asarray(cs.compute_magnetic_field(x, basis="rpz", source_grid=src))
scale = float(np.mean(d["|B|"]) / np.mean(np.linalg.norm(Bc, axis=1)))
for c in S.iter_unique(cs):
    c.current = c.current * scale
Bc = Bc * scale
bn = np.abs(np.sum(Bc * d["n_rho"], axis=1)) / np.mean(np.linalg.norm(Bc, axis=1))
dB = np.abs(np.linalg.norm(Bc, axis=1) - d["|B|"]) / d["|B|"]
say(f"=== {name}\ncoils {coils_f}\neq {eq_f} (L,M,N {eq.L},{eq.M},{eq.N})")
say(f"current scale applied {scale:.5f}; on the target boundary: <|B.n|>/<|B|> {bn.mean():.3e} (max {bn.max():.3e}), "
    f"| |B_coil| - |B_eq| | / |B_eq| mean {dB.mean():.3e} max {dB.max():.3e}")
res.update(scale=scale, bn_mean=float(bn.mean()), bn_max=float(bn.max()), dB_mean=float(dB.mean()), dB_max=float(dB.max()))
phys0 = SS.Physics(eq, "1,4")()
say("target physics  ", SS.fmt_phys(phys0))
res["target"] = phys0


# ---- trace ---------------------------------------------------------------------------------------------------------
def trace():
    NP = int(os.environ.get("NPER", 400))  # field periods traced
    rho = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
    # one grid per rho: LinearGrid sorts and de-duplicates rho, which shifted the starts in the first version
    pts = []
    for rv in rho:
        dd = eq.compute(["R", "Z"], grid=LinearGrid(rho=np.array([rv]), theta=np.array([0.0]), zeta=np.array([0.0]),
                                                      NFP=eq.NFP))
        pts.append((float(dd["R"][0]), float(dd["Z"][0])))
    R, Z = np.array(pts).T
    dR, dZ = (R[-1] - R[-3]) / 0.1, (Z[-1] - Z[-3]) / 0.1  # d(R,Z)/drho over rho 0.9 -> 1.0, extrapolated outside
    R = np.concatenate([R, [R[-1] + 0.03 * dR, R[-1] + 0.06 * dR]])
    Z = np.concatenate([Z, [Z[-1] + 0.03 * dZ, Z[-1] + 0.06 * dZ]])
    say("starts (R, Z): " + " ".join(f"{lab:g}:({r_:.4f},{z_:.4f})" for lab, r_, z_ in zip(list(rho) + [1.03, 1.06], R, Z)))
    labels = list(rho) + [1.03, 1.06]
    phis = np.arange(2 * NP + 1) * np.pi / eq.NFP
    t = time.time()
    r, z = field_line_integrate(R, Z, phis, cs, source_grid=src, rtol=1e-9, atol=1e-9, bounds_R=(0.5, 1.5),
                                bounds_Z=(-0.5, 0.5), max_steps=int(4000 * NP), chunk_size=len(R))
    r, z = np.asarray(r), np.asarray(z)  # (nphi, nlines)
    say(f"traced {len(R)} lines x {NP} field periods in {(time.time() - t) / 60:.1f} min")
    np.savez(f"{out}/trace.npz", r=r, z=z, phis=phis, labels=labels)
    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for plane, ax in enumerate(axs):
        sl = slice(plane, None, 2)
        ph = phis[sl][0]
        for j in range(r.shape[1]):
            ax.plot(r[sl, j], z[sl, j], ".", ms=0.6)
        gs = LinearGrid(rho=np.array([0.2, 0.4, 0.6, 0.8, 1.0]), theta=200, zeta=np.array([ph]), NFP=eq.NFP)
        ds = eq.compute(["R", "Z"], grid=gs)
        Rs, Zs = np.asarray(ds["R"]).reshape(5, -1, order="F"), np.asarray(ds["Z"]).reshape(5, -1, order="F")
        rr = gs.nodes[:, 0]
        for k, rv in enumerate([0.2, 0.4, 0.6, 0.8, 1.0]):
            m = np.isclose(rr, rv)
            ax.plot(np.append(ds["R"][m], ds["R"][m][0]), np.append(ds["Z"][m], ds["Z"][m][0]), "k-", lw=0.6)
        ax.set_aspect("equal")
        ax.set_title(f"{name}: phi = {ph:.3f} (black: target rho 0.2..1)")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
    fig.tight_layout()
    fig.savefig(f"{out}/poincare.png", dpi=150)
    # map every phi = 0 puncture into the target's flux coordinates
    iota_t = eq.compute("iota", grid=LinearGrid(rho=np.array(labels[:11]), NFP=eq.NFP))["iota"]
    for j, lab in enumerate(labels):
        ok = np.isfinite(r[:, j]) & np.isfinite(z[:, j])
        n_ok = int(ok.sum())
        if n_ok < len(phis):
            lost_at = float(phis[ok][-1] / (2 * np.pi / eq.NFP)) if n_ok else 0.0
        else:
            lost_at = None
        sub = np.where(ok)[0][:: max(1, n_ok // 400)]  # <= ~400 points per line for the coordinate map
        coords = np.column_stack([r[sub, j], phis[sub], z[sub, j]])
        try:
            fc = np.asarray(eq.map_coordinates(coords, inbasis=("R", "phi", "Z"), outbasis=("rho", "theta", "zeta"),
                                               maxiter=60, tol=1e-9))
        except Exception as e:  # noqa: BLE001
            say(f"  map_coordinates failed on line {lab}: {e}")
            fc = np.full((len(sub), 3), np.nan)
        rh, th = fc[:, 0], fc[:, 1]
        good = np.isfinite(rh) & (rh <= 1.0 + 1e-6)
        row = dict(rho0=float(lab), lost_after_fp=lost_at, n_inside=int(good.sum()), n=len(sub))
        if good.sum() > 10:
            row.update(rho_mean=float(np.mean(rh[good])), rho_std=float(np.std(rh[good])),
                       rho_min=float(np.min(rh[good])), rho_max=float(np.max(rh[good])))
            if good.all():
                thu = np.unwrap(th)
                dz = phis[sub][-1] - phis[sub][0]
                row["iota_trace"] = float((thu[-1] - thu[0]) / dz) if dz > 0 else None
                if lab <= 1.0:
                    row["iota_target"] = float(iota_t[labels.index(lab)])
        rows.append(row)
        f = lambda k, fmt: (fmt % row[k]) if row.get(k) is not None else "-"  # noqa: E731
        say(f"  line rho0 {lab:5.2f}: rho mean {f('rho_mean', '%.4f')} std {f('rho_std', '%.2e')} "
            f"[{f('rho_min', '%.4f')}, {f('rho_max', '%.4f')}]  iota trace {f('iota_trace', '%.4f')} target "
            f"{f('iota_target', '%.4f')}  inside {row['n_inside']}/{row['n']}  lost after "
            f"{'-' if lost_at is None else f'{lost_at:.0f} fp'}")
    res["trace"] = rows


# ---- free boundary -------------------------------------------------------------------------------------------------
def freeb():
    eqf = eq.copy()
    cons = (O.ForceBalance(eqf, jac_chunk_size=int(os.environ.get("EQCHUNK", 0)) or None), O.FixPressure(eqf),
            O.FixCurrent(eqf), O.FixPsi(eqf))
    obj = O.ObjectiveFunction(O.VacuumBoundaryError(eqf, field=cs, field_fixed=True, field_grid=src,
                                                    eval_grid=LinearGrid(rho=np.array([1.0]), M=2 * eqf.M,
                                                                         N=2 * eqf.N, NFP=eqf.NFP, sym=eqf.sym)))
    t = time.time()
    eqf, info = eqf.optimize(obj, cons, optimizer="proximal-lsq-exact", maxiter=int(os.environ.get("FB_ITERS", 60)),
                             ftol=1e-8, xtol=1e-10, gtol=1e-10, verbose=2,
                             options={"solve_options": {"ftol": 1e-10, "xtol": 1e-12, "gtol": 1e-12, "maxiter": 100}},
                             copy=False)
    say(f"free boundary: {info['message']}  nit {info['nit']}  {(time.time() - t) / 60:.1f} min  "
        f"cost {float(info['fun'] @ info['fun']) / 2:.3e}")
    eqf.save(f"{out}/eq_freeb.h5")
    p = SS.Physics(eqf, "1,4")()
    say("free-bdry physics", SS.fmt_phys(p))
    # boundary offset: distance from each free-boundary point to the target surface (dense nearest point)
    gb = LinearGrid(M=64, N=64, NFP=eq.NFP, sym=False, rho=np.array([1.0]))
    xt = eq.compute(["X", "Y", "Z"], grid=gb)
    xf = eqf.compute(["X", "Y", "Z"], grid=gb)
    from scipy.spatial import cKDTree

    tree = cKDTree(np.column_stack([xt["X"], xt["Y"], xt["Z"]]))
    dist, _ = tree.query(np.column_stack([xf["X"], xf["Y"], xf["Z"]]))
    a = phys0["a"]
    say(f"boundary offset free-bdry vs target: mean {dist.mean() * 1e3:.2f} mm, max {dist.max() * 1e3:.2f} mm "
        f"(= {dist.mean() / a:.3f} a, {dist.max() / a:.3f} a); <|B|> {p['B_vol']:.4f} vs {phys0['B_vol']:.4f}")
    res["freeb"] = dict(physics=p, offset_mean=float(dist.mean()), offset_max=float(dist.max()), nit=int(info["nit"]),
                        message=str(info["message"]))
    # Boozer |B| on rho = 0.5 and 1 for both
    from desc.plotting import plot_boozer_surface

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    for i, (e, tag) in enumerate([(eq, "target"), (eqf, "coil field (free bdry)")]):
        for k, rv in enumerate([0.5, 1.0]):
            plot_boozer_surface(e, rho=rv, ax=axs[i, k], fieldlines=4)
            axs[i, k].set_title(f"{tag}, rho {rv}")
    fig.tight_layout()
    fig.savefig(f"{out}/boozer.png", dpi=130)


if "trace" in todo:
    trace()
if "freeb" in todo:
    freeb()
json.dump(res, open(f"{out}/result.json", "w"), indent=1, default=float)
say("done")
