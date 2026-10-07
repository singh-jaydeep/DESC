"""precise_QA stage-2 problem from the stage-two tutorial notebook (3 FourierXYZ coils)."""

import numpy as np

import desc.examples
from desc.coils import initialize_modular_coils
from desc.grid import LinearGrid
from desc.objectives import (
    CoilArclengthResidual,
    CoilArclengthVariance,
    CoilCurvature,
    CoilLength,
    CoilSetDistanceRows,
    CoilSetMinDistance,
    FixCoilCurrent,
    FixParameters,
    ObjectiveFunction,
    PlasmaCoilDistanceField,
    PlasmaCoilSetMinDistance,
    QuadraticFlux,
    node_gap_estimate,
)
from desc.objectives.normalization import compute_scaling_factors

JCS = 16  # jac_chunk_size for every objective


def build(
    coilset=None,
    arclen="vector",
    arclen_w=5,
    cc="rows",
    cc_alpha=500.0,
    cc_bound=0.1,
    cc_w=100,
    pair_N=30,
    curv="bound",
    curv_target=1.5,
    curv_w=30,
    N_coil=50,
):
    """Build (eq, coilset, objective, constraints).

    cc: "hard" (min), "soft" (softmin) or "rows" (CoilSetDistanceRows). arclen:
    "scalar" (one variance per coil), "vector" (per-node residuals) or "none".
    Bounds are in normalized units (normalize_target=False), as in the notebook.
    """
    eq = desc.examples.get("precise_QA")
    c0 = initialize_modular_coils(eq, num_coils=3, r_over_a=3.0).to_FourierXYZ()
    if coilset is None:
        coilset = c0
    coil_grid = LinearGrid(N=N_coil)
    plasma_grid = LinearGrid(M=25, N=25, NFP=eq.NFP, sym=eq.sym)
    if cc == "rows":
        a = np.mean([compute_scaling_factors(c)["a"] for c in c0])
        cc_obj = CoilSetDistanceRows(
            coilset,
            select_distance=2 * cc_bound * a,
            bounds=(cc_bound, np.inf),
            normalize_target=False,
            grid=LinearGrid(N=pair_N),
            weight=cc_w,
            jac_chunk_size=JCS,
            distance="curve",
        )
    else:
        soft = dict(use_softmin=True, softmin_alpha=cc_alpha) if cc == "soft" else {}
        cc_obj = CoilSetMinDistance(
            coilset,
            bounds=(cc_bound, np.inf),
            normalize_target=False,
            grid=coil_grid,
            weight=cc_w,
            dist_chunk_size=2,
            jac_chunk_size=JCS,
            **soft,
        )
    curv_kw = {"bounds": (-1, 2)} if curv == "bound" else {"target": curv_target}
    objs = [
        QuadraticFlux(
            eq,
            field=coilset,
            eval_grid=plasma_grid,
            field_grid=coil_grid,
            vacuum=True,
            weight=200,
            bs_chunk_size=10,
            jac_chunk_size=JCS,
        ),
        cc_obj,
        PlasmaCoilSetMinDistance(
            eq,
            coilset,
            bounds=(0.25, np.inf),
            normalize_target=False,
            plasma_grid=plasma_grid,
            coil_grid=coil_grid,
            eq_fixed=True,
            weight=10,
            dist_chunk_size=2,
            jac_chunk_size=JCS,
        ),
        CoilCurvature(
            coilset,
            normalize_target=False,
            grid=coil_grid,
            weight=curv_w,
            jac_chunk_size=JCS,
            **curv_kw,
        ),
        CoilLength(
            coilset,
            bounds=(0, 2 * np.pi * (c0[0].compute("length")["length"])),
            normalize_target=True,
            grid=coil_grid,
            weight=20,
            jac_chunk_size=JCS,
        ),
    ]
    if arclen != "none":
        cls = CoilArclengthResidual if arclen == "vector" else CoilArclengthVariance
        objs.append(cls(coilset, weight=arclen_w, grid=coil_grid))
    obj = ObjectiveFunction(tuple(objs), deriv_mode="blocked")
    mask = [False] * len(coilset)
    mask[0] = True
    cons = (FixCoilCurrent(coilset, indices=mask),)
    return eq, coilset, obj, cons


def build_al(
    coilset=None,
    pair_N=256,
    curv_N=200,
    arclen_w=0.025,
    len_factor=1.5,
    speed_ratio=1.2,
    cc_sel=None,
    cc_K=10000,
    spacing=0.01,
):
    """The notebook problem for lsq-auglag-composite, bounds converted to meters.

    Objective: QuadraticFlux plus arclength residuals (weight arclen_w relative to it,
    the F1 ratio). Constraints: the notebook's coil-coil (0.1 a_coil), plasma-coil
    (0.25 a_plasma) and |curvature| (2 / a_coil) bounds, length <= len_factor * L0;
    node rows (gap from the bounds with speed_ratio) and the plasma distance field,
    both on pair_N nodes; curvature on its own grid. Gauges: FixCoilCurrent on coil
    0, and per coil the n=+-1 coefficient most aligned with the phase s -> s + c.
    """
    from common import phase_tangent

    eq = desc.examples.get("precise_QA")
    c0 = initialize_modular_coils(eq, num_coils=3, r_over_a=3.0).to_FourierXYZ()
    coilset = c0 if coilset is None else coilset
    a_c = np.mean([compute_scaling_factors(c)["a"] for c in c0])
    a_p = compute_scaling_factors(eq)["a"]
    cc_bound, pc_bound, kmax = 0.1 * a_c, 0.25 * a_p, 2 / a_c
    lmax = len_factor * float(c0[0].compute("length")["length"])
    coil_grid = LinearGrid(N=50)
    plasma_grid = LinearGrid(M=25, N=25, NFP=eq.NFP, sym=eq.sym)
    pair_grid = LinearGrid(N=pair_N)
    n = pair_grid.num_nodes
    h = speed_ratio * lmax / n
    obj = ObjectiveFunction(
        (
            QuadraticFlux(
                eq,
                field=coilset,
                eval_grid=plasma_grid,
                field_grid=coil_grid,
                vacuum=True,
                bs_chunk_size=10,
                jac_chunk_size=JCS,
            ),
            CoilArclengthResidual(coilset, weight=arclen_w, grid=coil_grid),
        ),
        deriv_mode="blocked",
    )
    modes = c0[0].X_basis.modes[:, 2]
    pins = []
    for p in coilset.params_dict:
        t = phase_tangent(p, modes)
        k, i = max(
            ((k, i) for k in t for i in np.where(np.abs(modes) == 1)[0]),
            key=lambda ki: abs(t[ki[0]][ki[1]]),
        )
        pins.append({k: np.array([i])})
    mask = [False] * len(coilset)
    mask[0] = True
    cons = (
        FixCoilCurrent(coilset, indices=mask),
        FixParameters(coilset, pins),
        CoilSetDistanceRows(
            coilset,
            select_distance=cc_sel or 2 * cc_bound,
            bounds=(cc_bound, np.inf),
            grid=pair_grid,
            gap=node_gap_estimate(n, lmax, kmax, cc_bound, speed_ratio),
            max_active_rows=cc_K,
            jac_chunk_size=JCS,
        ),
        PlasmaCoilDistanceField(
            eq,
            coilset,
            bounds=(pc_bound, np.inf),
            coil_grid=pair_grid,
            spacing=spacing,
            node_margin=1.5 * h**2 / 8 * (kmax + 1 / pc_bound),
            jac_chunk_size=JCS,
        ),
        CoilCurvature(
            coilset,
            bounds=(-np.inf, kmax),
            grid=LinearGrid(N=curv_N),
            jac_chunk_size=JCS,
        ),
        CoilLength(coilset, bounds=(0, lmax), grid=coil_grid, jac_chunk_size=JCS),
    )
    return eq, coilset, obj, cons
