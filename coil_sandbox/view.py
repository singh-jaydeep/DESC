"""Self-contained 3D HTML viewer for coilsets on an equilibrium (opens offline, no dependencies).

    python view.py work/qa/view.html --eq precise_QA --h5 start=work/qa/start.h5 --h5 run=work/qa/run1/result.h5
    python view.py work/mine/starts.html --eq my_eq.h5 --dir work/mine        # every *.h5 in a directory

For each coilset: all physical coils (count checked against coils x NFP x symmetry), the plasma
boundary, the closest coil-coil approach (sampled), each unique coil's tightest-curvature point,
linked pairs, and the dense-check numbers against the bounds (`sandbox/common.py`). For a run's
result.h5, the length bound comes from the result.json next to it.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "sandbox"))
import boot  # noqa: E402

boot.setup(default="cpu")

import argparse  # noqa: E402
import glob  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

import common as S  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.io import load  # noqa: E402
from desc.objectives._coils import _independent_coil_indices  # noqa: E402

NPTS = 160


def collect(name, path, eq, tree, a):
    cs = load(path)
    pos = np.asarray(cs._compute_position(grid=LinearGrid(N=NPTS), basis="xyz"))
    best = (0, 1, pos[0][0], pos[1][0], float("inf"))
    for i, j in itertools.combinations(range(pos.shape[0]), 2):
        d = np.linalg.norm(pos[i][:, None, :] - pos[j][None, :, :], axis=-1)
        p, q = divmod(int(np.argmin(d)), d.shape[1])
        if d[p, q] < best[4]:
            best = (i, j, pos[i][p], pos[j][q], float(d[p, q]))
    rows = [int(k) for k in _independent_coil_indices(cs)]
    peaks = []
    for k, c in enumerate(S.iter_unique(cs)):
        kap = np.abs(np.asarray(c.compute("|curvature|", grid=LinearGrid(N=NPTS))["|curvature|"]))
        peaks.append(dict(row=rows[k] if k < len(rows) else k, idx=int(np.argmax(kap)), kappa=round(float(kap.max()), 4)))
    rec = S.result_json_next_to(path) if os.path.basename(path) == "result.h5" else None
    if a.length_mult is not None:  # one multiplier for every coilset (e.g. single-stage files have no result.json)
        lmult = a.length_mult
    else:
        lmult = float(rec["length_multiplier"]) if rec else 1.0
    B = S.paper_bounds(eq, S.n_unique(cs), a.a, a.bounds)
    if B["L"] is not None:
        B["L"] *= lmult
    m = S.evaluate(cs, eq, tree, vacuum=a.vacuum)
    g = S.margins(m, B)
    return dict(
        name=name, file=os.path.relpath(path), n_coils=int(pos.shape[0]), expected=S.expected_coils(cs, eq),
        unique_rows=rows, length_multiplier=lmult,
        coils=[[[round(float(v), 5) for v in p] for p in c] for c in pos],
        closest=dict(i=best[0], j=best[1], a=[round(float(v), 5) for v in best[2]],
                     b=[round(float(v), 5) for v in best[3]], d=round(best[4], 6)),
        curv_peaks=peaks,
        linked=S.linked_pairs(pos),
        quantities=dict(length=round(m["L"], 6), kappa=round(m["kappa"], 6), kappa_MS=round(m["kms"], 6),
                        convex=float(m["convex"]), d_cc=round(m["d_cc"], 6), d_pc=round(m["d_pc"], 6)),
        margins={name: (round(g[k], 4) if k in g else None) for name, k in
                 (("length", "L"), ("kappa", "kappa"), ("kappa_MS", "kMS"), ("convex", "convex"),
                  ("d_cc", "d_cc"), ("d_pc", "d_pc"))},
        extra=dict(bn=m["bn"], bn_local=m["bn_local"], bn_max_T=m["bn_max_T"], vacuum=m["vacuum"],
                   corner=m["corner"]),
    )


def streamlines(eq, K, nodes=120):
    """K current-potential contours per field period, replicated over the full torus.

    These are the proxy modular coils: streamlines of K = n x B / mu0, cut at equal increments
    of the stream function Phi. Equal-current cutting crowds them where |K| is large, so the
    spacing carries information -- it is not a drawing choice.

    The foliation is field-period symmetric, so one period is computed and rotated NFP times
    rather than solving the whole torus. Each curve is coloured by its OWN eps_1 (the
    arclength-weighted RMS distance from its best-fit plane, over the curve's in-plane
    extent), so "which streamlines are non-planar, and where do they sit" is readable directly.
    """
    sys.path.insert(0, os.path.join(HERE, "922_stage1opt_results"))
    import contour_curve as CC
    import jax.numpy as jnp
    from desc.compute import get_profiles, get_transforms

    M, N, Mt = 32, 64, 16
    NFP = int(eq.NFP)
    th = 2 * np.pi * np.arange(M) / M
    ze = 2 * np.pi * np.arange(N) / N
    gg = LinearGrid(rho=np.array([1.0]), theta=th, zeta=ze, NFP=1, sym=False)
    nd = np.asarray(gg.nodes)
    nt = np.rint(nd[:, 1] / (2 * np.pi) * M).astype(int) % M
    nz = np.rint(nd[:, 2] / (2 * np.pi) * N).astype(int) % N
    const = dict(transforms=get_transforms(CC.DATA_KEYS, obj=eq, grid=gg),
                 profiles=get_profiles(CC.DATA_KEYS, obj=eq, grid=gg),
                 perm=jnp.asarray(np.lexsort((nz, nt))),
                 Rm=jnp.asarray(eq.surface.R_basis.modes),
                 Zm=jnp.asarray(eq.surface.Z_basis.modes),
                 theta=jnp.asarray(2 * np.pi * np.arange(nodes) / nodes),
                 trunc=CC.truncation_index(M, N, Mt, Mt))
    (xs, ws), diag = CC._curves_from_params(eq.params_dict, const, M, N, NFP, K, 6, 0.0,
                                            diagnostics=True)
    if not diag["monotonic"]:
        print("  (streamlines skipped: Phi_mod is not monotone in zeta)", flush=True)
        return None
    eps = np.asarray(CC._eps1_live_all(xs, ws))
    xs = np.asarray(xs)
    curves, e1 = [], []
    for p in range(NFP):                       # replicate the field period by rotation
        c, sn = np.cos(2 * np.pi * p / NFP), np.sin(2 * np.pi * p / NFP)
        R = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
        for k in range(K):
            curves.append([[round(float(v), 4) for v in q] for q in xs[k] @ R.T])
            e1.append(round(float(eps[k]), 5))
    print(f"  streamlines: {len(curves)} contours ({K}/period x NFP {NFP}), "
          f"eps_1 {eps.min():.4f}..{eps.max():.4f}, Newton residual "
          f"{max(diag['newton_residual']):.1e}", flush=True)
    return {"curves": curves, "eps1": e1,
            "lo": round(float(eps.min()), 5), "hi": round(float(eps.max()), 5)}


def surface_field(eq, grid, ref_path=None):
    """L_grad(B)/a on the plasma wireframe grid, and optionally its ratio to a reference.

    The grid is (zeta=64) x (theta=32) to match PLASMA's reshape, so scalar[i][j] lines up
    with PLASMA[i][j] exactly. The ratio is taken at matched (theta, zeta), which is the
    natural correspondence between two boundaries but NOT a point-to-point one -- the two
    surfaces are different shapes. Stated in the viewer.
    """
    out = {"abs": None, "ratio": None}
    try:
        aa = float(eq.compute("a")["a"])
        L = np.asarray(eq.compute("L_grad(B)", grid=grid)["L_grad(B)"]).reshape(64, 32) / aa
    except Exception as e:                                    # not every equilibrium has it
        print(f"  (no surface field: {e})", flush=True)
        return out
    out["abs"] = [[round(float(v), 4) for v in row] for row in L]
    out["abs_lo"], out["abs_hi"] = round(float(L.min()), 4), round(float(L.max()), 4)
    print(f"  L_gradB/a on the surface: {L.min():.3f} .. {L.max():.3f} (median {np.median(L):.3f})",
          flush=True)
    if ref_path:
        ref, _ = S.load_equilibrium(ref_path)
        ra = float(ref.compute("a")["a"])
        gr = LinearGrid(rho=np.array([1.0]), theta=32, zeta=64, NFP=1, sym=False)
        R = np.asarray(ref.compute("L_grad(B)", grid=gr)["L_grad(B)"]).reshape(64, 32) / ra
        ratio = L / R
        out["ratio"] = [[round(float(v), 4) for v in row] for row in ratio]
        out["ratio_lo"], out["ratio_hi"] = round(float(ratio.min()), 4), round(float(ratio.max()), 4)
        out["ref"] = os.path.basename(ref_path)
        print(f"  ratio to {os.path.basename(ref_path)}: {ratio.min():.3f} .. {ratio.max():.3f} "
              f"(median {np.median(ratio):.3f}); {100*np.mean(ratio < 1):.0f}% of the surface got WORSE",
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--eq", required=True)
    ap.add_argument("--h5", action="append", default=[], metavar="NAME=PATH[@EQ]",
                    help="a coilset; @EQ gives it its OWN equilibrium (checked and drawn against that "
                         "boundary, e.g. a single-stage result), otherwise --eq")
    ap.add_argument("--dir", default=None, help="show every *.h5 in this directory (checkpoints excluded)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--a", type=float, default=None)
    ap.add_argument("--bounds", default=None)
    ap.add_argument("--length-mult", type=float, default=None,
                    help="length bound multiplier for EVERY coilset (default: each result.json's, else 1). "
                         "Use with an UNSCALED bounds file, or the multiplier is applied twice")
    ap.add_argument("--device", default=None)
    ap.add_argument("--vacuum", action="store_true", help="ignore the plasma's own field in the B.n check")
    # Surface scalar field painted on the plasma wireframe. L_grad(B)/a is Kappel's
    # coil-proximity predictor (PPCF 66 (2024) 025018, R^2 = 0.944 for min(L_gradB)/a vs
    # min(d_cs)/a -- normalized by `a`, NOT R0, which only gets 0.644). SMALLER = coils must
    # come closer. --field-ref adds a ratio-to-reference mode, which is what answers "is the
    # change local or global": it shows WHERE the field moved, not just that it did.
    # The proxy coils of NOTES section 3: contours of the current potential Phi on rho=1, at
    # EQUAL increments of Phi. Equal-current cutting is what makes them crowd where |K| is
    # large, so the bunching in this view is physical, not a sampling choice.
    ap.add_argument("--streamlines", type=int, default=0, metavar="K",
                    help="draw K current-potential contours per field period on the boundary")
    ap.add_argument("--field-ref", default=None,
                    help="reference equilibrium; adds an L_gradB ratio field (this / ref) at "
                         "matched (theta, zeta)")
    a = ap.parse_args()

    items = []
    if a.dir:
        for f in sorted(glob.glob(os.path.join(a.dir, "*.h5"))):
            if "ckpt" not in os.path.basename(f):
                items.append((os.path.splitext(os.path.basename(f))[0], f))
    for spec in a.h5:
        if "=" not in spec:
            raise SystemExit(f"--h5 wants NAME=PATH, got {spec!r}")
        items.append(tuple(spec.split("=", 1)))
    if not items:
        raise SystemExit("nothing to show: pass --h5 or --dir")

    eq, kind = S.load_equilibrium(a.eq)
    S.check_vacuum_choice(eq, kind, a.vacuum)
    tree = S.surface_tree(eq)
    data = []
    g = LinearGrid(rho=np.array([1.0]), theta=32, zeta=64, NFP=1, sym=False)
    for name, path in items:
        own = None
        if "@" in path:
            path, eq_path = path.split("@", 1)
            own, _ = S.load_equilibrium(eq_path)
        print(f"  {name} ... ", end="", flush=True)
        S._BN_CACHE.clear()  # keyed by id(eq): never reuse another boundary's data
        d = collect(name, path, own or eq, S.surface_tree(own) if own is not None else tree, a)
        if own is not None:
            d["plasma"] = [[[round(float(v), 5) for v in p] for p in row] for row in
                           np.asarray(own.compute("x", grid=g, basis="xyz")["x"]).reshape(64, 32, 3)]
            d["eq"] = os.path.relpath(eq_path)
        print(f"{d['n_coils']} coils (expected {d['expected']}), {len(d['linked'])} linked", flush=True)
        data.append(d)
    plasma = np.asarray(eq.compute("x", grid=g, basis="xyz")["x"]).reshape(64, 32, 3)
    field = surface_field(eq, g, a.field_ref)
    stream = streamlines(eq, a.streamlines) if a.streamlines else None
    html = open(os.path.join(HERE, "sandbox", "viewer_template.html")).read()
    html = (html.replace("__TITLE__", a.title or f"Coils on {os.path.basename(a.eq)}")
            .replace("__DATA__", json.dumps(data, separators=(",", ":")))
            .replace("__PLASMA__", json.dumps([[[round(float(v), 5) for v in p] for p in row] for row in plasma],
                                              separators=(",", ":")))
            .replace("__FIELD__", json.dumps(field, separators=(",", ":")))
            .replace("__STREAM__", json.dumps(stream, separators=(",", ":"))))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w").write(html)
    print(f"wrote {a.out} ({len(html) / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
