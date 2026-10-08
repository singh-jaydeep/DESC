"""Record eps_1 / eps_2 for QUASR devices. No optimization, no screening -- just measurement.

    python 925_quasr_sweep/record_eps.py --ids 1630198,197197 --out 925_quasr_sweep/eps.jsonl
    python 925_quasr_sweep/record_eps.py --nfp 2 --helicity 0 --n 20 --seed 0 --out ...

One JSON object per device per line, appended. Already-recorded IDs are skipped, so the file is
resumable and a crash costs one device. Devices that fall outside the modular-proxy class are
recorded with `status` set and their reason, never silently dropped.

Per device: a fixed-boundary vacuum solve (eps_1 needs K_vc, so it needs the interior field --
on an unsolved boundary the contour Newton diverges and the objective raises), then eps_1 and
eps_2 on K = 2 * nc_per_hp contours, plus the same two metrics on QUASR's OWN optimized coils
for contrast. The coil values need no solve and cost ~2 s, and the proxy-to-coil ratio is NOT a
constant across devices (4.9 and 19.7 on the first two measured), so record both.
"""
import argparse
import json
import os
import sys
import time

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--out", required=True, help="jsonl file, appended")
p.add_argument("--cache", default="925_quasr_sweep/cache", help="where QUASR downloads live")
p.add_argument("--ids", default=None, help="explicit comma-separated QUASR IDs")
p.add_argument("--n", type=int, default=None, help="sample this many from the filtered index")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--nfp", type=int, default=None)
p.add_argument("--helicity", type=int, default=None, help="0 = QA, 1 = QH")
p.add_argument("--nc-per-hp", type=int, default=None)
p.add_argument("--aspect", default=None, help="single value or LO,HI (aspect ratio is DISCRETE "
                                              "in QUASR: 24, 20, 12, 10, 8, 6.67, 6, 5, 4, 3.33, 2.86)")
p.add_argument("--qs-max", type=float, default=-3.0, help="keep qs_error (log10) below this")
p.add_argument("--res", type=int, default=10, help="solve at L=M=N=this. 10 gives eps_1 to ~0.5%% "
                                                   "in 125 s; 12 costs 597 s for 0.46%% more")
p.add_argument("--K", type=int, default=None, help="contours per field period (default 2 * nc_per_hp)")
p.add_argument("--nodes", type=int, default=240)
p.add_argument("--stride", type=int, default=8, help="eps_2 DP rotation-offset stride")
p.add_argument("--mt", type=int, default=16, help="Phi~ mode truncation, |m| and |n| <= this")
p.add_argument("--grid-M", type=int, default=64)
p.add_argument("--grid-N", type=int, default=128)
p.add_argument("--no-coils", action="store_true", help="skip the real-coil eps (saves ~2 s)")
p.add_argument("--maxiter", type=int, default=150)
p.add_argument("--device", default="gpu")
args = p.parse_args()

sys.path.insert(0, "sandbox")
sys.path.insert(0, "922_stage1opt_results")
import boot  # noqa: E402

boot.setup(default=args.device)

import warnings  # noqa: E402

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402

import clamshell as CL  # noqa: E402
import quasr as Q  # noqa: E402
from contour_curve import ContourClamshell, _eps1_live  # noqa: E402
from desc.grid import Grid, LinearGrid  # noqa: E402
from desc.objectives import ForceBalance, ObjectiveFunction  # noqa: E402

INDEX_KEYS = ["nfp", "helicity", "nc_per_hp", "aspect_ratio", "minor_radius", "mean_iota",
              "qs_error", "max_kappa", "max_msc", "min_coil2coil_dist", "min_coil2surface_dist",
              "coil_length_per_hp", "total_coil_length", "mean_elongation", "max_elongation",
              "volume", "Nsurfaces", "message"]


def select():
    """The device IDs to record."""
    if args.ids:
        return [int(v) for v in args.ids.split(",")]
    idx = Q.index(args.cache)
    keep = idx["qs_error"] < args.qs_max
    for key, val in (("nfp", args.nfp), ("helicity", args.helicity), ("nc_per_hp", args.nc_per_hp)):
        if val is not None:
            keep &= idx[key] == val
    if args.aspect:
        v = [float(x) for x in args.aspect.split(",")]
        keep &= (idx["aspect_ratio"] >= v[0]) & (idx["aspect_ratio"] <= (v[1] if len(v) > 1 else v[0]))
    ids = idx["ID"][keep]
    print(f"{len(ids)} devices match", flush=True)
    if args.n and args.n < len(ids):
        ids = np.random.default_rng(args.seed).choice(ids, args.n, replace=False)
    return sorted(int(i) for i in ids)


def curve_pts(c, n=480):
    s = 2 * np.pi * np.arange(n) / n
    g = Grid(np.stack([0 * s, 0 * s, s], 1), sort=False, jitable=False)
    d = c.compute(["x", "x_s"], grid=g, basis="xyz")
    return np.asarray(d["x"]), np.linalg.norm(np.asarray(d["x_s"]), axis=1)


def eta_svd(x, w):
    """The literature's non-planarity score. Equals eps_1/(eps_1+2) when the in-plane
    eigenvalues match, which they do to 0.3% on QUASR's coils -- recorded to check that."""
    D = (x - (w[:, None] * x).sum(0) / w.sum()) * np.sqrt(w)[:, None]
    sv = np.linalg.svd(D, compute_uv=False)
    return float(sv[-1] / sv.sum())


def measure(ID):
    rec = Q.record(ID, args.cache)
    row = dict(ID=ID, **{k: rec[k] for k in INDEX_KEYS})
    K = args.K or 2 * rec["nc_per_hp"]
    # coil standoff in minor radii. The proxy contours live on rho = 1 and know nothing about
    # standoff, and it is the variable QUASR spread widest (1.2 .. 10 a), so record it up front:
    # the proxy-to-coil eps_1 offset tracked it across the first four devices measured.
    row.update(K=K, res=args.res, nodes=args.nodes,
               d_pc_over_a=rec["min_coil2surface_dist"] / rec["minor_radius"],
               d_cc_over_a=rec["min_coil2coil_dist"] / rec["minor_radius"])

    # ---- real coils first: no solve needed, so it survives a solve failure ----
    if not args.no_coils:
        t0 = time.time()
        base = [c for c in Q.coilset(ID, args.cache, full=False)]
        c1, c2, ce = [], [], []
        for c in base:
            x, w = curve_pts(c)
            c1.append(float(_eps1_live(x, w)))
            c2.append(float(CL.eps_B(x, w, 2, stride=args.stride)[0]))
            ce.append(eta_svd(x, w))
        row.update(coil_eps1=c1, coil_eps2=c2, coil_eta_svd=ce,
                   coil_eps1_mean=float(np.mean(c1)), coil_eps2_mean=float(np.mean(c2)),
                   coil_eta_mean=float(np.mean(ce)), t_coils=time.time() - t0)

    # ---- the solve ----
    t0 = time.time()
    eq, _ = Q.equilibrium(ID, args.cache)
    eq.change_resolution(L=args.res, M=args.res, N=args.res,
                         L_grid=2 * args.res, M_grid=2 * args.res, N_grid=2 * args.res)
    eq.solve(objective=ObjectiveFunction(ForceBalance(eq=eq), deriv_mode="batched",
                                         jac_chunk_size=100),
             maxiter=args.maxiter, verbose=0, ftol=1e-10, xtol=1e-12, gtol=1e-12)
    row["t_solve"] = time.time() - t0

    # iota at the edge against QUASR's own value: an end-to-end check that our
    # fixed-boundary vacuum solve reproduces their coil-field device
    lg = LinearGrid(rho=np.array([1.0]), M=2 * args.res, N=2 * args.res, NFP=eq.NFP)
    iot = float(np.asarray(lg.compress(eq.compute("iota", grid=lg)["iota"]))[0])
    want = float(rec["iota_profile"][-1])
    # Record BOTH. The relative error is misleading for the QA devices whose iota target was
    # 0.1: on one A=5 device 35.7% relative was 0.024 absolute, while 0.11% on an iota=1.2
    # device was 0.0013. Filter on |d iota|, not on the percentage.
    row.update(iota_edge=iot, iota_edge_quasr=want,
               iota_err_abs=abs(abs(iot) - abs(want)),
               iota_err_pct=abs(abs(iot / want) - 1) * 100 if want else None)
    fg = LinearGrid(L=16, M=2 * args.res, N=2 * args.res, NFP=eq.NFP)
    fd = eq.compute(["|F|", "|grad(|B|^2)|/2mu0", "sqrt(g)"], grid=fg)
    fw = np.abs(np.asarray(fd["sqrt(g)"]))
    row["force_rel"] = float(np.sum(np.asarray(fd["|F|"]) * fw)
                             / np.sum(np.asarray(fd["|grad(|B|^2)|/2mu0"]) * fw))

    # ---- eps on the proxy contours ----
    t0 = time.time()
    o = ContourClamshell(eq, quantity="eps1", K=K, M=args.grid_M, N=args.grid_N,
                         n=args.nodes, Mt=args.mt, Nt=args.mt, stride=args.stride)
    o.build(verbose=0)              # raises if Phi_mod is non-monotone or Newton fails
    e1 = np.asarray(o.compute(eq.params_dict))
    (xs, ws), dg = o._chain(eq.params_dict, o._constants, diagnostics=True)
    e2 = np.array([CL.eps_B(np.asarray(xs[k]), np.asarray(ws[k]), 2, stride=args.stride)[0]
                   for k in range(K)])
    row.update(eps1=[float(v) for v in e1], eps2=[float(v) for v in e2],
               eps1_mean=float(e1.mean()), eps1_min=float(e1.min()), eps1_max=float(e1.max()),
               eps2_mean=float(e2.mean()), C_mean=float(np.mean(4 * e2 / e1)),
               monotonic=bool(dg["monotonic"]), newton_res=float(max(dg["newton_residual"])),
               modes_kept=float(dg["mode_truncation_kept_fraction"]),
               IoverG=float(dg["I"] / dg["G"]), t_eps=time.time() - t0, status="ok")
    return row


done = set()
if os.path.exists(args.out):
    with open(args.out) as f:
        for line in f:
            if line.strip():
                done.add(json.loads(line)["ID"])
    print(f"{len(done)} already in {args.out}", flush=True)

ids = [i for i in select() if i not in done]
print(f"recording {len(ids)} devices at L=M=N={args.res} -> {args.out}", flush=True)
os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

for j, ID in enumerate(ids, 1):
    t0 = time.time()
    try:
        row = measure(ID)
    except Exception as e:                                        # noqa: BLE001
        row = dict(ID=ID, status=f"{type(e).__name__}: {str(e)[:300]}")
        try:
            row.update({k: Q.record(ID, args.cache)[k] for k in INDEX_KEYS})
        except Exception:                                         # noqa: BLE001, S110
            pass
    row["t_total"] = time.time() - t0
    with open(args.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    if row["status"] == "ok":
        print(f"[{j}/{len(ids)}] {ID} nfp{row['nfp']}{'QH' if row['helicity'] else 'QA'} "
              f"nc{row['nc_per_hp']} A{row['aspect_ratio']:.1f} | eps1 {row['eps1_mean']:.5f} "
              f"eps2 {row['eps2_mean']:.5f} C {row['C_mean']:.3f} | coil eps1 "
              f"{row.get('coil_eps1_mean', float('nan')):.5f} | iota_err "
              f"{row['iota_err_pct']:.2f}% | {row['t_total']:.0f}s", flush=True)
    else:
        print(f"[{j}/{len(ids)}] {ID} SKIPPED -- {row['status']}", flush=True)
