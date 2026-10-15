"""Exact conversion of the folded (near-axis prototype) coils to DESC PiecewisePlanarArcCoil (B = 2) CoilSets.

DESC's arc (desc/compute/_curve.py): x(t) = H_i + t c_i + w_i(t) perp_i, w_i = sum_m a_im sin(m pi t) [metres],
perp = perp0 cos(phi) + (e_par x perp0) sin(phi), perp0 = the frozen reference (+z for a non-vertical chord)
made perpendicular to the chord. folded.py uses the same frame with w scaled by |c|, and its lower arc runs the
chord backwards with u_l = -cos(alpha_l) v + sin(alpha_l) h, which is DESC's perp at phi = alpha_l + pi
(e_par -> -e_par flips h). So: hinges [Ha, Hb], tilts [alpha_u, alpha_l + pi], shape [|c| a_u, |c| a_l].
No fit, no approximation. Currents: folded's mu0/4pi units -> amps (I / 1e-7).
"""
import numpy as np
import jax.numpy as jnp

from desc.coils import CoilSet, MixedCoilSet, PiecewisePlanarArcCoil


def _cyl(R, ph, z):
    return np.array([R * np.cos(ph), R * np.sin(ph), z])


def to_desc_coils(C, L):
    """C: (nc, per) folded coil parameters, L: folded.Layout. Returns (MixedCoilSet, list of unique arcs)."""
    C = np.asarray(C)
    K, nh = L.K, L.nh
    arcs = []
    for row in C:
        if L.mode == "z0":
            Ha, Hb = _cyl(row[0], row[1], 0.0), _cyl(row[2], row[3], 0.0)
        else:
            Ha, Hb = _cyl(row[0], row[1], row[2]), _cyl(row[3], row[4], row[5])
        al_u, al_l = row[nh], row[nh + 1]
        au, alw = row[nh + 2: nh + 2 + K], row[nh + 2 + K: nh + 2 + 2 * K]
        Lc = np.linalg.norm(Hb - Ha)
        arcs.append(PiecewisePlanarArcCoil(current=float(row[nh + 2 + 2 * K]) / 1e-7, hinges=np.stack([Ha, Hb]),
                                           tilts=np.array([al_u, al_l + np.pi]),
                                           shape=np.stack([Lc * au, Lc * alw])))
    return MixedCoilSet(*[CoilSet(a, NFP=L.nfp, sym=True) for a in arcs]), arcs
