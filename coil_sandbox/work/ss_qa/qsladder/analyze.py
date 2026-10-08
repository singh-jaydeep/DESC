"""Diagnostics for the tau_QS ladder: how does the boundary contort as QS is loosened?

    python work/ss_qa/qsladder/analyze.py [--out work/ss_qa/qsladder]

Cases are found automatically: the start (stage 2 M=5 on precise_QA), the tau_QS x10 k=4 result,
and every qsladder/qsNN/ run that has eq_k4.h5 + coils_k4.h5. Writes into --out:
  summary.json, SUMMARY.md             metrics per case, binding constraints
  xsections.png                        boundary cross-sections, every case over precise_QA
  maps.png                             per case: signed displacement from precise_QA, |B.n|/|B|, |f_C|/|B|^3
  spectrum.png                         |change| of each boundary mode (m, n) from precise_QA
  view.html                            3D viewer, every case on its own boundary
"""

import argparse
import glob
import json
import os
import subprocess
import sys

ROOT = "/home/singh/Documents/DESC2/coil_sandbox"
sys.path.insert(0, os.path.join(ROOT, "sandbox"))
import boot  # noqa: E402

boot.setup(default="cpu")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import common as S  # noqa: E402
import desc.objectives as O  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.io import load  # noqa: E402

W = os.path.join(ROOT, "work/ss_qa")
V0, IOTA_MIN, A_RANGE, VBAND = 0.60032, 0.40, (5.5, 6.5), 0.02
NT, NZ = 64, 64


def cases():
    """(label, eq, coils, tau multiplier, iota floor)."""
    out = [("start: stage 2 M5", "precise_QA", f"{W}/stage2_L0p8_M5/result.h5", 0, 0.40),
           ("tau_QS x10", f"{W}/ss_v2_qs10_k4_M5/eq_k4.h5", f"{W}/ss_v2_qs10_k4_M5/coils_k4.h5", 10, 0.40)]
    for d in sorted(glob.glob(f"{W}/qsladder/qs*/"), key=lambda p: int(os.path.basename(p.rstrip("/"))[2:])):
        m = int(os.path.basename(d.rstrip("/"))[2:])
        if os.path.exists(d + "eq_k4.h5") and os.path.exists(d + "coils_k4.h5"):
            out.append((f"tau_QS x{m}", d + "eq_k4.h5", d + "coils_k4.h5", m, 0.40))
    # iota-floor relaxations at tau_QS x1000: qsladder/iotaNN/ = floor 0.NN
    for d in sorted(glob.glob(f"{W}/qsladder/iota*/"), key=lambda p: -int(os.path.basename(p.rstrip("/"))[4:])):
        fl = int(os.path.basename(d.rstrip("/"))[4:]) / 100
        if os.path.exists(d + "eq_k4.h5") and os.path.exists(d + "coils_k4.h5"):
            out.append((f"x1000, iota>={fl:.2f}", d + "eq_k4.h5", d + "coils_k4.h5", 1000, fl))
    return out


def boozer_qs(eq):
    B = float(eq.compute("<|B|>_vol")["<|B|>_vol"])
    out = []
    for r in (0.25, 0.5, 0.75, 1.0):
        o = O.QuasisymmetryBoozer(eq, helicity=(1, 0), normalize=False,
                                  grid=LinearGrid(rho=np.array([r]), M=2 * eq.M_grid, N=2 * eq.N_grid, NFP=eq.NFP,
                                                  sym=False))
        o.build(verbose=0)
        out.append(float(np.linalg.norm(o.compute_unscaled(eq.params_dict))) / B)
    return out


def surface(eq, cs):
    """(theta, zeta) maps over one field period: xyz, outward normal (xyz), |B.n|/|B|, |f_C|/|B|^3."""
    th = np.linspace(0, 2 * np.pi, NT, endpoint=False)
    ze = np.linspace(0, 2 * np.pi / eq.NFP, NZ, endpoint=False)
    g = LinearGrid(rho=np.array([1.0]), theta=th, zeta=ze, NFP=1, sym=False)
    # basis="xyz" converts EVERY vector output (n_rho included), so positions come from a separate call
    d = eq.compute(["R", "phi", "Z", "n_rho", "|B|", "f_C"], grid=g, helicity=(1, 0))
    d["x"] = eq.compute("x", grid=g, basis="xyz")["x"]
    rpz = np.stack([d["R"], d["phi"], d["Z"]], 1)
    Bc = np.asarray(cs.compute_magnetic_field(rpz, basis="rpz", source_grid=S.source_grid(cs)[0]))
    n = np.asarray(d["n_rho"])  # rpz components
    bn = np.abs(np.sum(Bc * n, 1)) / np.linalg.norm(Bc, axis=1)
    qs = np.abs(np.asarray(d["f_C"])) / np.asarray(d["|B|"]) ** 3
    ph = np.asarray(d["phi"])
    nxyz = np.stack([n[:, 0] * np.cos(ph) - n[:, 1] * np.sin(ph), n[:, 0] * np.sin(ph) + n[:, 1] * np.cos(ph), n[:, 2]], 1)
    nd = np.asarray(g.nodes)
    it = np.rint(nd[:, 1] / (2 * np.pi) * NT).astype(int) % NT
    iz = np.rint(nd[:, 2] / (2 * np.pi / eq.NFP) * NZ).astype(int) % NZ

    def grid_map(v):
        M = np.zeros((NZ, NT) + np.shape(v)[1:])
        M[iz, it] = v
        return M

    return dict(x=grid_map(np.asarray(d["x"])), n=grid_map(nxyz), bn=grid_map(bn), qs=grid_map(qs))


def signed_disp(surf, ref_eq):
    """Distance from each boundary point to the precise_QA boundary (full torus, dense), signed by
    the REFERENCE outward normal at the nearest point: > 0 means this boundary is outside precise_QA."""
    th = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    ze = np.linspace(0, 2 * np.pi, 1024, endpoint=False)
    g = LinearGrid(rho=np.array([1.0]), theta=th, zeta=ze, NFP=1, sym=False)
    d = ref_eq.compute(["n_rho", "phi"], grid=g)  # rpz; rotated to xyz below
    X = np.asarray(ref_eq.compute("x", grid=g, basis="xyz")["x"])
    n, ph = np.asarray(d["n_rho"]), np.asarray(d["phi"])
    N = np.stack([n[:, 0] * np.cos(ph) - n[:, 1] * np.sin(ph), n[:, 0] * np.sin(ph) + n[:, 1] * np.cos(ph), n[:, 2]], 1)
    dist, j = cKDTree(X).query(surf["x"].reshape(-1, 3))
    sgn = np.sign(np.sum((surf["x"].reshape(-1, 3) - X[j]) * N[j], 1))
    return (dist * sgn).reshape(NZ, NT)


def overlap(A, B, rng, nnull=300):
    top = lambda M: M >= np.quantile(M, 0.9)  # noqa: E731
    ov = float(np.mean(top(A)[top(B)]))
    null = [np.mean(top(A)[top(np.roll(np.roll(B, rng.integers(4, NZ - 4), 0), rng.integers(4, NT - 4), 1))])
            for _ in range(nnull)]
    return ov, float(np.mean(np.array(null) >= ov)), float(spearmanr(A.ravel(), B.ravel()).correlation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{W}/qsladder")
    ap.add_argument("--no-viewer", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ref, _ = S.load_equilibrium("precise_QA")
    Bnd = S.paper_bounds(ref, 4, None, os.path.join(ROOT, "bounds/precise_qa_sweep910_L0p8.json"))
    P = {k: Bnd[k] for k in S.UPPER + S.LOWER}
    rng = np.random.default_rng(0)
    rows, surfs, eqs = [], {}, {}
    for label, eqp, csp, mult, iota_floor in cases():
        print(f"== {label}", flush=True)
        eq = S.load_equilibrium(eqp)[0]
        cs = load(csp)
        S._BN_CACHE.clear()
        m = S.evaluate(cs, eq, S.surface_tree(eq), vacuum=True)
        gi = LinearGrid(rho=np.linspace(0, 1, 41), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
        iota = gi.compress(eq.compute("iota", grid=gi)["iota"])
        dd = eq.compute(["R0/a", "V", "a", "R0"])
        s = surface(eq, cs)
        s["disp"] = signed_disp(s, ref)
        surfs[label], eqs[label] = s, eq
        ov_bn, p_bn, r_bn = overlap(np.abs(s["disp"]), s["bn"], rng)
        ov_qs, p_qs, r_qs = overlap(np.abs(s["disp"]), s["qs"], rng)
        ov_bq, p_bq, r_bq = overlap(s["bn"], s["qs"], rng)
        mg = S.margins(m, P)
        A, vr = float(dd["R0/a"]), float(dd["V"]) / V0
        binding = [k for k, v in mg.items() if abs(v - 1) < 0.005]
        if iota.min() < iota_floor * 1.002:
            binding.append("iota_floor")
        if vr < 1 - VBAND + 0.002 or vr > 1 + VBAND - 0.002:
            binding.append("volume_band")
        if A < A_RANGE[0] + 0.01 or A > A_RANGE[1] - 0.01:
            binding.append("aspect")
        rows.append(dict(
            case=label, tau_mult=mult, iota_floor=iota_floor, bn=m["bn"], bn_max=m["bn_max"], qs_booz=boozer_qs(eq), A=A, a=float(dd["a"]),
            R0=float(dd["R0"]), iota_min=float(iota.min()), iota_max=float(iota.max()), V_ratio=vr,
            margins=mg, linked=m["linked"], binding=binding,
            disp_rms_mm=1e3 * float(np.sqrt(np.mean(s["disp"] ** 2))), disp_max_mm=1e3 * float(np.max(np.abs(s["disp"]))),
            overlap_disp_bn=(ov_bn, p_bn, r_bn), overlap_disp_qs=(ov_qs, p_qs, r_qs), overlap_bn_qs=(ov_bq, p_bq, r_bq)))
        print(f"   bn {m['bn']:.4e} max {m['bn_max']:.3e}  A {A:.3f}  iota_min {iota.min():.4f}  V/V0 {vr:.4f}  "
              f"disp rms {rows[-1]['disp_rms_mm']:.2f} mm max {rows[-1]['disp_max_mm']:.2f} mm  binding {binding}",
              flush=True)
    json.dump(rows, open(f"{a.out}/summary.json", "w"), indent=1, default=float)

    labels = [r["case"] for r in rows]
    cols = ["#898781", "#2a78d6", "#1baf7a", "#eb6834", "#8a5cd6", "#d03b3b", "#b58b00"]
    # --- cross-sections
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8))
    thc = np.linspace(0, 2 * np.pi, 240)
    for ax, ph in zip(axes, np.radians([0, 30, 60, 90])):
        for k, lab in enumerate(["precise_QA"] + labels[1:]):
            eq = ref if lab == "precise_QA" else eqs[lab]
            g = LinearGrid(rho=np.array([1.0]), theta=thc, zeta=np.array([ph]), NFP=eq.NFP)
            x = eq.compute(["R", "Z"], grid=g)
            ax.plot(x["R"], x["Z"], color=cols[k % len(cols)], lw=2.2 if k == 0 else 1.4, label=lab)
        ax.set_aspect("equal"); ax.set_title(f"phi = {np.degrees(ph):.0f} deg", loc="left"); ax.set_xlabel("R (m)")
    axes[0].set_ylabel("Z (m)"); axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{a.out}/xsections.png", dpi=105); plt.close(fig)
    # --- maps
    fig, ax = plt.subplots(len(labels), 3, figsize=(15, 3.3 * len(labels)), squeeze=False)
    ext = [0, 2 * np.pi, 0, 180]
    vmax_d = max(np.max(np.abs(surfs[l]["disp"])) for l in labels) * 1e3
    for r_, lab in enumerate(labels):
        s = surfs[lab]
        im = ax[r_, 0].imshow(1e3 * s["disp"], origin="lower", aspect="auto", extent=ext, cmap="RdBu_r",
                              vmin=-vmax_d, vmax=vmax_d)
        fig.colorbar(im, ax=ax[r_, 0], label="mm (+ outside precise_QA)")
        im = ax[r_, 1].imshow(s["bn"], origin="lower", aspect="auto", extent=ext, cmap="Blues")
        fig.colorbar(im, ax=ax[r_, 1])
        im = ax[r_, 2].imshow(s["qs"], origin="lower", aspect="auto", extent=ext, cmap="Blues")
        fig.colorbar(im, ax=ax[r_, 2])
        for c_, t in enumerate(("boundary displacement from precise_QA", "|B.n|/|B| (coils)", "|f_C|/|B|^3 (QS, edge)")):
            ax[r_, c_].set_title(f"{lab}: {t}", fontsize=9, loc="left")
            ax[r_, c_].set_xlabel("theta"); ax[r_, c_].set_ylabel("zeta (deg)")
    fig.tight_layout(); fig.savefig(f"{a.out}/maps.png", dpi=90); plt.close(fig)
    # --- spectrum of the boundary change
    Rb0, Zb0 = np.array(ref.Rb_lmn), np.array(ref.Zb_lmn)
    Rm, Zm = ref.surface.R_basis.modes, ref.surface.Z_basis.modes
    Mx, Nx = int(np.max(np.abs(Rm[:, 1]))), int(np.max(np.abs(Rm[:, 2])))
    fig, ax = plt.subplots(1, len(labels) - 1, figsize=(4.2 * (len(labels) - 1), 3.8), squeeze=False)
    vmax = 0
    grids = []
    for lab in labels[1:]:
        eq = eqs[lab]
        G = np.zeros((2 * Mx + 1, 2 * Nx + 1))
        for (l, m, n), dv in zip(Rm, np.array(eq.Rb_lmn) - Rb0):
            G[m + Mx, n + Nx] += dv ** 2
        for (l, m, n), dv in zip(Zm, np.array(eq.Zb_lmn) - Zb0):
            G[m + Mx, n + Nx] += dv ** 2
        grids.append(1e3 * np.sqrt(G)); vmax = max(vmax, grids[-1].max())
    for k, (lab, G) in enumerate(zip(labels[1:], grids)):
        im = ax[0, k].imshow(G, origin="lower", cmap="Blues", vmin=0, vmax=vmax,
                             extent=[-Nx - .5, Nx + .5, -Mx - .5, Mx + .5])
        ax[0, k].set_title(f"{lab}: |dR_mn|,|dZ_mn| (mm)", fontsize=9, loc="left")
        ax[0, k].set_xlabel("n"); ax[0, k].set_ylabel("m (sign = cos/sin family)")
        fig.colorbar(im, ax=ax[0, k])
    fig.tight_layout(); fig.savefig(f"{a.out}/spectrum.png", dpi=100); plt.close(fig)
    # --- SUMMARY.md
    L = ["# tau_QS ladder: precise_QA arcB2 single stage (k=4, M=5, V +-2%, L x0.8; iota >= 0.40 unless the row says otherwise)", "",
         "| case | <B.n>/<B> | max | QS Boozer rho .25/.5/.75/1 | A | iota min | V/V0 | disp rms / max (mm) | binding |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        q = " / ".join(f"{v:.1e}" for v in r["qs_booz"])
        L.append(f"| {r['case']} | {r['bn']:.3e} | {r['bn_max']:.2e} | {q} | {r['A']:.3f} | {r['iota_min']:.4f} | "
                 f"{r['V_ratio']:.4f} | {r['disp_rms_mm']:.1f} / {r['disp_max_mm']:.1f} | {', '.join(r['binding']) or '-'} |")
    L += ["", "Co-location of the top-10% regions (chance = 0.10; p from random (theta, zeta) shifts; Spearman over the surface):", "",
          "| case | |disp| vs |B.n| | |disp| vs QS | |B.n| vs QS |", "|---|---|---|---|"]
    for r in rows:
        f = lambda t: "n/a" if not np.isfinite(t[2]) else f"{t[0]:.2f} (p {t[1]:.3f}, rho {t[2]:+.2f})"  # noqa: E731
        L.append(f"| {r['case']} | {f(r['overlap_disp_bn'])} | {f(r['overlap_disp_qs'])} | {f(r['overlap_bn_qs'])} |")
    L += ["", "Coil margins (value / bound; L, kappa, kMS <= 1, d_cc, d_pc >= 1):", ""]
    for r in rows:
        L.append(f"- {r['case']}: " + ", ".join(f"{k} {v:.3f}" for k, v in r["margins"].items()))
    L += ["", "Figures: xsections.png, maps.png, spectrum.png; 3D: view.html (each case on its own boundary)."]
    open(f"{a.out}/SUMMARY.md", "w").write("\n".join(L) + "\n")
    print(f"wrote {a.out}/SUMMARY.md", flush=True)
    if not a.no_viewer:
        cmd = [sys.executable, os.path.join(ROOT, "view.py"), f"{a.out}/view.html", "--eq", "precise_QA", "--vacuum",
               "--bounds", os.path.join(ROOT, "bounds/precise_qa_sweep910.json"), "--length-mult", "0.8",
               "--title", "precise_QA arcB2 single stage: tau_QS ladder (each case on its own boundary)"]
        for label, eqp, csp, _, _ in cases():
            nm = label.replace("=", "").replace(">", "")
            cmd += ["--h5", f"{nm}={csp}" + ("" if eqp == "precise_QA" else f"@{eqp}")]
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
