"""Stage-2 problem for the slack-free hinge AL (`lsq-auglag-composite`, main repo).

The recipe of ../devtools/coil_auglag (AL10 coil-coil node rows, AL11 plasma distance field;
see GUIDE.md) on any equilibrium and bounds file: QuadraticFlux is the objective, every
engineering limit a constraint at its bound. No backoffs: the node gap and the field margin
make the distance rows lower bounds on the true distances.

Per representation (GUIDE section 3):
* FourierPlanarCoil: pin the largest `normal` component per coil (scaling it is an exact gauge).
* FourierXYZCoil: relative per-node speed rows within +-speed_tol as constraints, and per coil the
  n=+-1 coefficient most aligned with the phase s -> s + c pinned (QA4 in auglag_plan.md); the
  node gap is built with speed ratio 1 + speed_tol.
* Arc coils (C0): no gauge (arc_ref, rotmat, shift are fixed by default). Biot-Savart and length
  on composite Gauss-Legendre per arc (sandbox source_grid / quad_grid); distance rows on a grid
  with a node on every corner.
"""

import os

import numpy as np

import common as S


def phase_tangent(p, modes):
    """d/dc of FourierXYZ coefficients under s -> s + c (cos for n>0, sin for n<0).

    Copy of ../devtools/coil_auglag/common.py:phase_tangent (that module imports its own
    problem builders at load time, and the sandbox already has a module named `common`).
    """
    t = {}
    for key in ("X_n", "Y_n", "Z_n"):
        a = np.asarray(p[key])
        out = np.zeros_like(a)
        for i, n in enumerate(modes):
            if n == 0:
                continue
            j = int(np.where(modes == -n)[0][0])
            out[j] += (-abs(n) if n > 0 else abs(n)) * a[i]
        t[key] = out
    return t


def build(eq, coilset, bounds, vacuum=True, field_N=50, eval_M=25, eval_N=25, curv_N=300,
          kms_N=150, len_N=50, row_N=256, cc_sel=0.1, cc_K=10000, field_spacing=0.01,
          speed_ratio=2.0, speed_tol=0.5, cc_gap="bounds", jcs=16):
    """(objective, constraints, info) for `lsq-auglag-composite`.

    bounds: S.paper_bounds(...) dict (L, kappa, kms, dcc, dpc); a None bound is left out.
    row_N: about 2 row_N + 1 distance-row nodes per coil (both distance objectives).
    speed_ratio: largest/mean node spacing for the gap (planar, arcs); FourierXYZ uses 1 + speed_tol.
    cc_gap: "bounds" (node_gap_estimate from the bounds, on row_N) or "coils" (CoilSetDistanceRows picks the
        node grid and gap from the coils at build, gap <= 2% of d_cc; the plasma field's node margin likewise;
        for warm starts whose node spacing is uneven, e.g. arcs).
    """
    from desc.coils import FourierPlanarCoil, FourierXYZCoil
    from desc.grid import LinearGrid
    from desc.objectives import (
        CoilArclengthResidual,
        CoilCurvature,
        CoilLength,
        CoilMeanSquaredCurvature,
        CoilSetDistanceRows,
        FixParameters,
        FixSumCoilCurrent,
        ObjectiveFunction,
        PlasmaCoilDistanceField,
        QuadraticFlux,
    )
    from desc.objectives._coils import _coil_node_grid, node_gap_estimate

    coils = list(coilset)  # the coilset's own coil list (one entry per independent coil)
    kinds = {type(c).__name__ for c in S.iter_unique(coilset)}
    if len(kinds) != 1:
        raise SystemExit(f"mixed coil types {kinds}")
    kind = kinds.pop()
    piecewise = S.is_piecewise(coilset)
    xyz = kind == FourierXYZCoil.__name__
    if xyz:
        speed_ratio = 1 + speed_tol

    n_nodes = 2 * row_N + 1
    row_grid = _coil_node_grid(coilset, n_nodes) if piecewise else LinearGrid(N=row_N)
    n_nodes = row_grid.num_nodes
    if piecewise:
        field_grid, field_kind = S.source_grid(coilset)
        len_grid, _ = S.quad_grid(coilset)
    else:
        field_grid, field_kind = LinearGrid(N=field_N), f"uniform N={field_N}"
        len_grid = LinearGrid(N=len_N)
    gap = node_gap_estimate(n_nodes, bounds["L"], bounds["kappa"], bounds["dcc"], speed_ratio)
    h = speed_ratio * bounds["L"] / n_nodes  # largest node spacing, as in the gap estimate
    margin = 1.5 * h**2 / 8 * (bounds["kappa"] + 1 / bounds["dpc"])
    obj = ObjectiveFunction(
        QuadraticFlux(eq, field=coilset, eval_grid=LinearGrid(M=eval_M, N=eval_N, NFP=eq.NFP, sym=eq.sym),
                      field_grid=field_grid, vacuum=vacuum, bs_chunk_size=10, jac_chunk_size=jcs),
        deriv_mode="blocked",
    )
    cons = [FixSumCoilCurrent(coilset)]
    gauge = "none"
    if kind == FourierPlanarCoil.__name__:  # planar normal-scale gauge
        cons.append(FixParameters(coilset, [{"normal": np.array([np.argmax(np.abs(np.asarray(c.normal)))])}
                                            for c in coils]))
        gauge = "largest normal component pinned"
    elif xyz:  # phase s -> s + c: pin the n=+-1 coefficient most aligned with it

        modes = coils[0].X_basis.modes[:, 2]
        pins = []
        for p in coilset.params_dict:
            t = phase_tangent(p, modes)
            k, i = max(((k, i) for k in t for i in np.where(np.abs(modes) == 1)[0]),
                       key=lambda ki: abs(t[ki[0]][ki[1]]))
            pins.append({k: np.array([i])})
        cons.append(FixParameters(coilset, pins))
        cons.append(CoilArclengthResidual(coilset, relative=True, bounds=(-speed_tol, speed_tol),
                                          grid=LinearGrid(N=100)))
        gauge = f"phase pinned ({[list(p)[0] for p in pins]}), relative speed within +-{speed_tol}"
    if bounds["L"] is not None:
        cons.append(CoilLength(coilset, bounds=(0, bounds["L"]), grid=len_grid, jac_chunk_size=jcs))
    if bounds["kappa"] is not None:
        cons.append(CoilCurvature(coilset, bounds=(-np.inf, bounds["kappa"]), grid=LinearGrid(N=curv_N),
                                  jac_chunk_size=jcs))
    if bounds["kms"] is not None:
        cons.append(CoilMeanSquaredCurvature(coilset, bounds=(0, bounds["kms"]), grid=LinearGrid(N=kms_N),
                                             jac_chunk_size=jcs))
    cc_obj = CoilSetDistanceRows(coilset, select_distance=cc_sel, bounds=(bounds["dcc"], np.inf),
                                 grid=row_grid if cc_gap == "bounds" else None, max_active_rows=cc_K, signed=True,
                                 distance="node", gap=gap if cc_gap == "bounds" else None, jac_chunk_size=jcs)
    cons.append(cc_obj)
    cons.append(PlasmaCoilDistanceField(eq, coilset, bounds=(bounds["dpc"], np.inf), coil_grid=row_grid,
                                        spacing=field_spacing, node_margin=margin if cc_gap == "bounds" else None,
                                        jac_chunk_size=jcs))
    info = dict(kind=kind, gauge=gauge, field_grid=f"{field_kind} ({field_grid.num_nodes} nodes)",
                n_nodes=n_nodes, speed_ratio=speed_ratio, cc_gap_mode=cc_gap,
                cc_gap=gap if cc_gap == "bounds" else "from the coils at build (see BUILD line)",
                pc_node_margin=margin if cc_gap == "bounds" else "from the coils at build (see field_info)")
    return obj, tuple(cons), info
