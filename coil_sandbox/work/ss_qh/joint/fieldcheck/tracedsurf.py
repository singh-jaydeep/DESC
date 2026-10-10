"""QS of a coil set's ACTUAL field, without a free-boundary solve.

In vacuum (curl B = 0, no plasma current) the field inside a closed magnetic surface is fixed by that surface (up to
an overall scale, which no normalized metric sees). So: trace one coil-field line from the target's outboard rho0
(default 1.0) at K planes per field period, fit a FourierRZ surface to the punctures (poloidal angle = geometric angle
about the target's axis), solve a fixed-boundary vacuum equilibrium on it, and report Boozer QS / iota.
Validation: the inner traced lines of trace.npz must lie on its flux surfaces (rho spread), and the traced iota must
match its iota.   Usage: tracedsurf.py CASE COILS.h5 EQ_TARGET [RHO0] [RES]"""
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
from desc.equilibrium import Equilibrium  # noqa: E402
from desc.geometry import FourierRZToroidalSurface  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.magnetic_fields import field_line_integrate  # noqa: E402
import common as S  # noqa: E402
import single_stage as SS  # noqa: E402

F = "work/ss_qh/joint/fieldcheck"
case, coils_f, eq_f = sys.argv[1:4]
rho0 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.95
RES = int(sys.argv[5]) if len(sys.argv) > 5 else 10
out = f"{F}/{case}"
eq = S.load_equilibrium(eq_f)[0]
cs = desc.io.load(coils_f)
src, _ = S.source_grid(cs, 24, 400)
NFP = eq.NFP
log = open(f"{out}/tracedsurf.txt", "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")
    log.flush()


# 1. trace one line, K planes per field period
K, NP = 48, int(os.environ.get("NPER", 400))
d0 = eq.compute(["R", "Z"], grid=LinearGrid(rho=np.array([rho0]), theta=np.array([0.0]), zeta=np.array([0.0]), NFP=NFP))
phis = np.arange(K * NP + 1) * (2 * np.pi / NFP) / K
t = time.time()
r, z = field_line_integrate(np.array([float(d0["R"][0])]), np.array([float(d0["Z"][0])]), phis, cs, source_grid=src,
                            rtol=1e-10, atol=1e-10, bounds_R=(0.5, 1.5), bounds_Z=(-0.5, 0.5), max_steps=int(4000 * NP))
r, z = np.asarray(r)[:, 0], np.asarray(z)[:, 0]
say(f"=== {case}: traced rho0 {rho0} for {NP} field periods x {K} planes in {(time.time() - t) / 60:.1f} min; "
    f"lost: {int((~np.isfinite(r)).sum())} points")
ok = np.isfinite(r)
r, z, ph = r[ok], z[ok], phis[ok]

# 2. poloidal parametrization: the TARGET's theta at each puncture (smooth on the traced surface, which stays within
#    ~0.02 in rho of a target surface; the geometric angle about the axis failed: sections are not star-shaped)
sub = np.arange(0, len(r), max(1, len(r) // 8000))
r, z, ph = r[sub], z[sub], ph[sub]
fc = np.asarray(eq.map_coordinates(np.column_stack([r, ph, z]), inbasis=("R", "phi", "Z"),
                                   outbasis=("rho", "theta", "zeta"), maxiter=80, tol=1e-10))
good = np.all(np.isfinite(fc), axis=1) & (fc[:, 0] < 0.999)
# map_coordinates occasionally fails to converge and lands far off (rho 0.11 for a rho ~0.95 line): drop outliers
med = np.median(fc[good, 0])
good &= np.abs(fc[:, 0] - med) < 0.05
say(f"mapped {good.sum()}/{len(fc)} punctures into target coordinates; their rho in [{fc[good, 0].min():.4f}, "
    f"{fc[good, 0].max():.4f}]")
r, z, ph, theta = r[good], z[good], ph[good], np.mod(fc[good, 1], 2 * np.pi)
coords = np.column_stack([r, np.mod(ph, 2 * np.pi), z])
from desc.grid import Grid  # noqa: E402

keep = np.ones(len(r), bool)
for it in range(4):  # robust refit: a few punctures map to a wrong theta with a plausible rho
    surf = FourierRZToroidalSurface.from_values(coords[keep], theta[keep], M=RES, N=RES, NFP=NFP, sym=True,
                                                check_orientation=False)
    g = Grid(np.column_stack([np.ones_like(theta), theta, np.mod(ph, 2 * np.pi)]), sort=False, jitable=True)
    fr = surf.compute(["R", "Z"], grid=g)
    res_all = np.hypot(np.asarray(fr["R"]) - r, np.asarray(fr["Z"]) - z)
    say(f"  fit pass {it}: {keep.sum()} points, residual median {np.median(res_all[keep]) * 1e3:.3f} mm, "
        f"max {res_all[keep].max() * 1e3:.3f} mm")
    new_keep = res_all < max(1e-3, 6 * np.median(res_all[keep]))
    if (new_keep == keep).all():
        break
    keep = new_keep
best = res_all[keep]
say(f"  kept {keep.sum()}/{len(keep)} punctures")
a_t = SS.Physics(eq, "1,4")()["a"]
say(f"surface fit M=N={RES}: point-to-fit residual mean {best.mean() * 1e3:.3f} mm, max {best.max() * 1e3:.3f} mm "
    f"(max {best.max() / a_t:.4f} a)")

# 3. fixed-boundary vacuum equilibrium on the traced surface
# start from the target (raised to RES) and deform its boundary onto the traced one: DESC's default initial guess on
# the strongly shaped QH boundary gave a non-nested solution (relative force error 0.87) in the first version
eqs = eq.copy()
eqs.change_resolution(L=RES, M=RES, N=RES, L_grid=2 * RES, M_grid=2 * RES, N_grid=2 * RES)
assert np.array_equal(eqs.surface.R_basis.modes, surf.R_basis.modes)
assert np.array_equal(eqs.surface.Z_basis.modes, surf.Z_basis.modes)
dR, dZ = surf.R_lmn - eqs.Rb_lmn, surf.Z_lmn - eqs.Zb_lmn
say(f"boundary change target(rho=1) -> traced: max |dR_mn| {np.abs(dR).max() * 1e3:.2f} mm, "
    f"max |dZ_mn| {np.abs(dZ).max() * 1e3:.2f} mm")
t = time.time()
eqs.solve(ftol=1e-10, xtol=1e-12, gtol=1e-10, maxiter=100, verbose=0)  # target at RES first
for _ in range(4):  # deform in 4 equal steps, re-solving each time
    # move the boundary a quarter of the way and re-solve (perturb() builds a Jacobian that ran out of GPU memory)
    eqs.Rb_lmn = eqs.Rb_lmn + dR * 0.25
    eqs.Zb_lmn = eqs.Zb_lmn + dZ * 0.25
    eqs.solve(ftol=1e-10, xtol=1e-12, gtol=1e-10, maxiter=200, verbose=0)
assert np.allclose(eqs.Rb_lmn, surf.R_lmn, atol=1e-9) and np.allclose(eqs.Zb_lmn, surf.Z_lmn, atol=1e-9)
from desc.grid import QuadratureGrid  # noqa: E402

qg = QuadratureGrid(L=eqs.L_grid, M=eqs.M_grid, N=eqs.N_grid, NFP=NFP)
fq = eqs.compute(["|F|", "|grad(|B|^2)|/2mu0", "sqrt(g)"], grid=qg)
wq = np.abs(fq["sqrt(g)"]) * qg.weights
say(f"traced-surface equilibrium: relative force error {np.sum(fq['|F|'] * wq) / np.sum(fq['|grad(|B|^2)|/2mu0'] * wq):.2e}, "
    f"sqrt(g) > 0 everywhere: {bool(np.all(fq['sqrt(g)'] > 0))}")
say(f"traced-surface equilibrium solved (L=M=N={RES}) in {(time.time() - t) / 60:.1f} min")
eqs.save(f"{out}/eq_traced.h5")
ps = SS.Physics(eqs, "1,4")()
SS.Physics.RHO = tuple(rho0 * q for q in (0.25, 0.5, 0.75, 1.0))  # target at the matching radii
pt = SS.Physics(eq, "1,4")()
SS.Physics.RHO = (0.25, 0.5, 0.75, 1.0)
say(f"target at rho x {rho0:g}", SS.fmt_phys(pt))
say("coil field (trc)", SS.fmt_phys(ps))

# 4. validation: inner traced lines in the traced equilibrium's coordinates
d = np.load(f"{out}/trace.npz")
tr = json.load(open(f"{out}/result.json")).get("trace", [])
say(" rho0 | rho spread: target  traced-eq | iota: trace   target   traced-eq")
rows = []
for j, lab in enumerate(list(d["labels"])):
    if lab > 0.9 * rho0 / 0.95 + 1e-9:
        continue
    sub = np.arange(0, len(d["phis"]), 2)[:: max(1, len(d["phis"]) // 800)]
    x = np.column_stack([d["r"][sub, j], d["phis"][sub], d["z"][sub, j]])
    sp = []
    for e in (eq, eqs):
        fc = np.asarray(e.map_coordinates(x, inbasis=("R", "phi", "Z"), outbasis=("rho", "theta", "zeta"),
                                          maxiter=60, tol=1e-9))
        sp.append((float(np.ptp(fc[:, 0])), float(np.mean(fc[:, 0]))))
    it_s = float(eqs.compute("iota", grid=LinearGrid(rho=np.array([sp[1][1]]), NFP=NFP))["iota"][0])
    row = next((q for q in tr if abs(q["rho0"] - lab) < 1e-9), {})
    rows.append(dict(rho0=float(lab), spread_target=sp[0][0], spread_traced=sp[1][0], rho_traced=sp[1][1],
                     iota_trace=row.get("iota_trace"), iota_target=row.get("iota_target"), iota_traced_eq=it_s))
    say(f" {lab:4.2f} |  {sp[0][0]:.4f}  {sp[1][0]:.4f}@{sp[1][1]:.3f} | {row.get('iota_trace', float('nan')):.4f}  "
        f"{row.get('iota_target', float('nan')):.4f}  {it_s:.4f}")
json.dump(dict(case=case, rho0=rho0, res=RES, fit_mean=float(best.mean()), fit_max=float(best.max()), target=pt,
               traced=ps, validation=rows), open(f"{out}/tracedsurf.json", "w"), indent=1, default=float)

# 5. Boozer |B|: target vs coil field
from desc.plotting import plot_boozer_surface  # noqa: E402

fig, axs = plt.subplots(2, 2, figsize=(12, 9))
for i, (e, tag) in enumerate([(eq, "target"), (eqs, "coil field (traced surface)")]):
    for k, rv in enumerate([0.5, 1.0]):
        plot_boozer_surface(e, rho=rv, ax=axs[i, k], fieldlines=4)
        axs[i, k].set_title(f"{case} {tag}, rho {rv}")
fig.tight_layout()
fig.savefig(f"{out}/boozer_traced.png", dpi=120)
say("done")
