"""precise_QH stage-2 problem: 4 unique FourierPlanarCoil, N=7, bounds in meters."""

import numpy as np

import desc.examples
from desc.coils import initialize_modular_coils
from desc.grid import LinearGrid
from desc.objectives import (
    CoilCurvature,
    CoilLength,
    CoilSetDistanceRows,
    CoilSetLinkingNumber,
    CoilSetMinDistance,
    FixParameters,
    FixSumCoilCurrent,
    ObjectiveFunction,
    PlasmaCoilSetDistanceRows,
    PlasmaCoilSetMinDistance,
    QuadraticFlux,
)

JCS = 16  # jac_chunk_size for every objective


def build(
    coilset=None,
    r_over_a=2.5,
    N=7,
    N_coil=50,
    pair_N=128,
    cc="rows",
    pc="rows",
    qf_w=200,
    cc_w=300,
    pc_w=100,
    curv_w=100,
    len_w=200,
    cc_bound=0.058,
    pc_bound=0.11,
    kmax=13.7,
    lmax=2.92,
    cc_sel=0.116,
    cc_K=3000,
    pc_sel=0.16,
    pc_K=6000,
    signed=True,
    fixnorm=True,
    link_w=0.0,
    link_N=40,
):
    """Build (eq, coilset, objective, constraints).

    cc, pc: "rows" (smooth per-row distances) or "hard" (DESC's minimum).
    fixnorm pins the largest normal component of each coil: scaling a planar coil's
    normal is an exact gauge. link_w > 0 adds the built-in CoilSetLinkingNumber.
    """
    eq = desc.examples.get("precise_QH")
    if coilset is None:
        coilset = initialize_modular_coils(eq, num_coils=4, r_over_a=r_over_a)
        for c in coilset:
            c.change_resolution(N=N)
    coil_grid = LinearGrid(N=N_coil)
    plasma_grid = LinearGrid(M=25, N=25, NFP=eq.NFP, sym=eq.sym)
    if cc == "rows":
        cc_obj = CoilSetDistanceRows(
            coilset,
            select_distance=cc_sel,
            bounds=(cc_bound, np.inf),
            grid=LinearGrid(N=pair_N),
            max_active_rows=cc_K,
            signed=signed,
            weight=cc_w,
            jac_chunk_size=JCS,
        )
    else:
        cc_obj = CoilSetMinDistance(
            coilset,
            bounds=(cc_bound, np.inf),
            grid=coil_grid,
            weight=cc_w,
            dist_chunk_size=2,
            jac_chunk_size=JCS,
        )
    if pc == "rows":
        pc_obj = PlasmaCoilSetDistanceRows(
            eq,
            coilset,
            select_distance=pc_sel,
            bounds=(pc_bound, np.inf),
            plasma_grid=plasma_grid,
            coil_grid=LinearGrid(N=pair_N),
            max_active_rows=pc_K,
            weight=pc_w,
            jac_chunk_size=JCS,
        )
    else:
        pc_obj = PlasmaCoilSetMinDistance(
            eq,
            coilset,
            bounds=(pc_bound, np.inf),
            plasma_grid=plasma_grid,
            coil_grid=coil_grid,
            eq_fixed=True,
            weight=pc_w,
            dist_chunk_size=2,
            jac_chunk_size=JCS,
        )
    objs = [
        QuadraticFlux(
            eq,
            field=coilset,
            eval_grid=plasma_grid,
            field_grid=coil_grid,
            vacuum=True,
            weight=qf_w,
            bs_chunk_size=10,
            jac_chunk_size=JCS,
        ),
        cc_obj,
        pc_obj,
        CoilCurvature(
            coilset,
            bounds=(-np.inf, kmax),
            grid=coil_grid,
            weight=curv_w,
            jac_chunk_size=JCS,
        ),
        CoilLength(
            coilset,
            bounds=(0, lmax),
            grid=coil_grid,
            weight=len_w,
            jac_chunk_size=JCS,
        ),
    ]
    if link_w > 0:  # all 32x32 pairs per evaluation; keep the jac chunk small
        objs.append(
            CoilSetLinkingNumber(
                coilset, grid=LinearGrid(N=link_N), weight=link_w, jac_chunk_size=1
            )
        )
    obj = ObjectiveFunction(tuple(objs), deriv_mode="blocked")
    cons = (FixSumCoilCurrent(coilset),)
    if fixnorm:
        fix = [{"normal": np.array([np.argmax(np.abs(c.normal))])} for c in coilset]
        cons = cons + (FixParameters(coilset, fix),)
    return eq, coilset, obj, cons
