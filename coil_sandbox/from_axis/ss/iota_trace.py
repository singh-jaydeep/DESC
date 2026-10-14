"""Rotational transform of a coil set's vacuum field from field-line tracing, independent of any equilibrium,
and the equilibria it is compared with mapped onto the same traced lines.

  1. magnetic axis of the coil field: fixed point of the one-field-period map on phi = 0 (Newton, FD Jacobian),
     started from the first equilibrium's axis.
  2. lines started on Z = 0, phi = 0 between the traced axis and 1.08x the first equilibrium's outboard boundary;
     each traced NPER field periods with SPP samples per period.
  3. iota = rotation number of the geometric angle about the traced axis, Birkhoff-weighted (converges much faster
     than end-to-end unwrapping on an invariant circle; the two halves of the trace are compared as an error check).
  4. every equilibrium given: the phi = 0 punctures of each line mapped to its rho; rho mean/std (std >> 0 = the
     traced surface is not one of that equilibrium's surfaces), and iota_eq(rho mean) vs the traced iota.

The coil current scale does not move field lines, so it is not touched.

Usage (from ss/): python iota_trace.py NAME COILS.h5 OUTDIR EQ1.h5[=LABEL] [EQ2.h5[=LABEL] ...]
env: NLINES (30), NPER (500), SPP (16)"""
import json
import os
import sys
import time

sys.path.insert(0, "../..")
sys.path.insert(0, "../../sandbox")
import boot  # noqa: E402

boot.setup("gpu")
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import desc.io  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.magnetic_fields import field_line_integrate  # noqa: E402
import common as S  # noqa: E402

name, coils_f, out = sys.argv[1:4]
eq_specs = [(s.split("=", 1) + [os.path.basename(os.path.dirname(s)) + "/" + os.path.basename(s)])[:2]
            for s in sys.argv[4:]]
NLINES, NPER, SPP = int(os.environ.get("NLINES", 30)), int(os.environ.get("NPER", 500)), int(os.environ.get("SPP", 16))
os.makedirs(out, exist_ok=True)
cs = desc.io.load(coils_f)
src, _ = S.source_grid(cs, 64, 400)
eqs = []
for f, lab in eq_specs:
    e = S.load_equilibrium(f)
    eqs.append((e[0] if isinstance(e, tuple) else e, lab, f))
eq0 = eqs[0][0]
NFP = eq0.NFP
TP = 2 * np.pi / NFP
log = open(f"{out}/summary.txt", "a")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")
    log.flush()


def trace(R, Z, phis, max_per=4000):
    r, z = field_line_integrate(np.atleast_1d(R), np.atleast_1d(Z), phis, cs, source_grid=src, rtol=1e-10,
                                atol=1e-10, bounds_R=(0.3, 1.4), bounds_Z=(-0.6, 0.6),
                                max_steps=int(max_per * (phis[-1] - phis[0]) / TP) + 1000,
                                chunk_size=np.size(R))
    return np.asarray(r), np.asarray(z)


say(f"=== {name}  coils {coils_f}  NLINES {NLINES} NPER {NPER} SPP {SPP}")
# ---- 1. axis --------------------------------------------------------------------------------------------------------
da = eq0.compute(["R", "Z"], grid=LinearGrid(rho=np.array([0.0]), theta=np.array([0.0]), zeta=np.array([0.0]),
                                            NFP=NFP))
x = np.array([float(da["R"][0]), float(da["Z"][0])])
h = 1e-5
for it in range(8):
    r, z = trace([x[0], x[0] + h, x[0]], [x[1], x[1], x[1] + h], np.array([0.0, TP]))
    P = np.column_stack([r[-1], z[-1]])  # images of base, +dR, +dZ
    F = P[0] - x
    J = np.column_stack([(P[1] - P[0]) / h, (P[2] - P[0]) / h]) - np.eye(2)
    dx = -np.linalg.solve(J, F)
    x = x + dx
    say(f"axis Newton {it}: R {x[0]:.9f} Z {x[1]:.2e} |F| {np.linalg.norm(F):.2e}")
    if np.linalg.norm(F) < 1e-11:
        break
axis_eq = (float(da["R"][0]), float(da["Z"][0]))
say(f"traced axis (R, Z) = ({x[0]:.6f}, {x[1]:.2e}); first equilibrium's axis ({axis_eq[0]:.6f}, {axis_eq[1]:.2e}); "
    f"offset {np.hypot(x[0] - axis_eq[0], x[1] - axis_eq[1]) * 1e3:.3f} mm")
phis1 = np.arange(SPP + 1) * TP / SPP
ra, za = trace(x[0], x[1], phis1)
ra, za = ra[:-1, 0], za[:-1, 0]  # axis at phi = k TP / SPP, periodic in k mod SPP

# ---- 2. lines -------------------------------------------------------------------------------------------------------
db = eq0.compute(["R"], grid=LinearGrid(rho=np.array([1.0]), theta=np.array([0.0]), zeta=np.array([0.0]), NFP=NFP))
Rb = float(db["R"][0])
s = np.linspace(0.03, 1.08, NLINES)
R0 = x[0] + s * (Rb - x[0])
Z0 = np.full_like(R0, x[1])
phis = np.arange(NPER * SPP + 1) * TP / SPP
t = time.time()
r, z = trace(R0, Z0, phis)
say(f"traced {NLINES} lines x {NPER} field periods in {(time.time() - t) / 60:.1f} min")
np.savez(f"{out}/trace.npz", r=r, z=z, phis=phis, s=s, axis=np.column_stack([ra, za]))


# ---- 3. rotation number ---------------------------------------------------------------------------------------------
def birkhoff(dth):
    n = len(dth)
    tt = (np.arange(n) + 0.5) / n
    w = np.exp(-1.0 / (tt * (1 - tt)))
    return float(np.sum(w * dth) / np.sum(w))


k = np.arange(len(phis)) % SPP
rows = []
for j in range(NLINES):
    ok = np.isfinite(r[:, j]) & np.isfinite(z[:, j])
    n_ok = int(np.argmin(ok)) if not ok.all() else len(ok)
    row = dict(s=float(s[j]), R0=float(R0[j]), lost_after_fp=None if n_ok == len(ok) else n_ok / SPP)
    if n_ok > 20 * SPP:
        th = np.unwrap(np.arctan2(z[:n_ok, j] - za[k[:n_ok]], r[:n_ok, j] - ra[k[:n_ok]]))
        dth = np.diff(th)
        dphi = TP / SPP
        row["iota"] = birkhoff(dth) / dphi
        half = len(dth) // 2
        row["iota_err"] = abs(birkhoff(dth[:half]) - birkhoff(dth[half:])) / dphi
        row["iota_unwrap"] = float((th[-1] - th[0]) / (phis[n_ok - 1] - phis[0]))
    rows.append(row)

# ---- 4. map onto each equilibrium -----------------------------------------------------------------------------------
cmp = {}
for eq, lab, f in eqs:
    cmp[lab] = []
    rho_g = np.linspace(0, 1, 201)
    gi = LinearGrid(rho=rho_g, M=eq.M_grid, N=eq.N_grid, NFP=NFP)
    iota_prof = np.asarray(gi.compress(eq.compute("iota", grid=gi)["iota"]))
    for j, row in enumerate(rows):
        n_ok = len(phis) if row["lost_after_fp"] is None else int(row["lost_after_fp"] * SPP)
        idx = np.arange(0, n_ok, SPP)[:: max(1, (n_ok // SPP) // 250)]  # <= ~250 phi = 0 (mod TP) punctures
        coords = np.column_stack([r[idx, j], np.zeros(len(idx)), z[idx, j]])
        try:
            fc = np.asarray(eq.map_coordinates(coords, inbasis=("R", "phi", "Z"), outbasis=("rho", "theta", "zeta"),
                                               maxiter=60, tol=1e-10))
            rh = fc[:, 0]
        except Exception as e:  # noqa: BLE001
            say(f"  [{lab}] map_coordinates failed on line {j}: {e}")
            rh = np.full(len(idx), np.nan)
        good = np.isfinite(rh) & (rh < 1.0 - 1e-9)  # map_coordinates clamps points outside the boundary to rho = 1
        c = dict(n=len(idx), n_inside=int(good.sum()))
        if good.all() and len(idx) > 10:
            c.update(rho_mean=float(rh.mean()), rho_std=float(rh.std()), rho_min=float(rh.min()),
                     rho_max=float(rh.max()), iota_eq=float(np.interp(rh.mean(), rho_g, iota_prof)))
        elif np.isfinite(rh).sum() > 10:
            c.update(rho_mean=float(np.nanmean(rh)), rho_std=float(np.nanstd(rh)))
        cmp[lab].append(c)
    say(f"--- {lab} ({f}): iota_eq axis {iota_prof[0]:.4f}  rho .25/.5/.75/1 "
        + " / ".join(f"{np.interp(v, rho_g, iota_prof):.4f}" for v in (0.25, 0.5, 0.75, 1.0)))
    cmp[lab + "#profile"] = dict(rho=rho_g.tolist(), iota=iota_prof.tolist())

say(f"{'s':>5} {'iota_trace':>10} {'+-':>8} {'unwrap':>8}  " + "  ".join(
    f"{'[' + lab[:22] + '] rho  std  iota_eq  d':>40}" for _, lab, _ in eqs))
for j, row in enumerate(rows):
    line = (f"{row['s']:5.3f} {row.get('iota', np.nan):10.5f} {row.get('iota_err', np.nan):8.1e} "
            f"{row.get('iota_unwrap', np.nan):8.5f}" + ("" if row["lost_after_fp"] is None
                                                     else f" LOST@{row['lost_after_fp']:.0f}fp"))
    for _, lab, _ in eqs:
        c = cmp[lab][j]
        if "iota_eq" in c:
            line += (f"   {c['rho_mean']:6.4f} {c['rho_std']:7.1e} {c['iota_eq']:7.4f} "
                     f"{abs(row.get('iota', np.nan)) - abs(c['iota_eq']):+8.4f}")
        else:
            line += f"   outside ({c['n_inside']}/{c['n']} inside)".ljust(42)
    say(line)
json.dump(dict(name=name, coils=coils_f, axis=x.tolist(), axis_eq=axis_eq, NPER=NPER, SPP=SPP, lines=rows,
               eqs={lab: f for _, lab, f in eqs}, compare=cmp), open(f"{out}/iota_trace.json", "w"), indent=1)

# ---- plot -----------------------------------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
for ci, (eq, lab, _) in enumerate(eqs):
    col = f"C{ci}"
    p = cmp[lab + "#profile"]
    axs[0].plot(p["rho"], np.abs(p["iota"]), "-", color=col, label=f"equilibrium: {lab}")
    pts = [(c["rho_mean"], abs(row["iota"]), c["rho_std"]) for c, row in zip(cmp[lab], rows)
           if "iota_eq" in c and "iota" in row]
    if pts:
        pr, pi, ps = np.array(pts).T
        axs[0].errorbar(pr, pi, xerr=ps, fmt="o", ms=3, color=col, mfc="none",
                        label=f"traced, placed at its rho in {lab}")
        axs[1].semilogy(pr, np.maximum(ps, 1e-6), "o-", ms=3, color=col, label=lab)
axs[0].set_xlabel("rho")
axs[0].set_ylabel("|iota|")
axs[0].set_title(f"{name}: traced coil-field iota vs equilibria")
axs[0].legend(fontsize=7)
axs[1].set_xlabel("rho (mean of the line's punctures)")
axs[1].set_ylabel("std of rho along the traced line")
axs[1].set_title("does each traced surface coincide with an equilibrium surface?")
axs[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(f"{out}/iota_trace.png", dpi=140)
say("done")
