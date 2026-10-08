"""T0: pipeline zero-checks for the Phi-contour proxy.

T0a  Analytic pure-TF field on an axisymmetric shaped surface. Then K = n x B / mu0 is
     purely poloidal, Phi = G zeta / 2pi EXACTLY, Phi~ = 0, and every contour is a planar
     curve in a zeta = const plane -> eps_1 = 0. This exercises the metric, the secular
     extraction, the Poisson solve, the contouring and the curve fit against a closed form.
T0b  planarizability.py on a coilset that is planar by construction (planar_N7) -> eps_1
     at round-off. Checks the metric itself, independently of the proxy.
T0c  A real axisymmetric equilibrium. eps_1 is NOT expected to be 0 here: the sheet must
     carry the plasma's toroidal current while real coils do not, which tilts the contours
     out of plane at O(I/G). Measuring that is the point -- it is the proxy's known bias.

    python 922_stage1opt_results/t0_checks.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, "sandbox")
import boot  # noqa: E402

boot.setup(default="cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "917_results"))
import phi_contours as P  # noqa: E402
import planarizability as PL  # noqa: E402
from desc.examples import get  # noqa: E402
from desc.geometry import FourierRZToroidalSurface  # noqa: E402
from desc.grid import Grid, LinearGrid  # noqa: E402
from desc.io import load  # noqa: E402

MU_0 = 4e-7 * np.pi
OK, BAD = "[ ok ]", "[FAIL]"


# eps_B is sqrt(lambda_min), so its numerical floor is sqrt(machine eps) ~ 1.5e-8 even
# for an exactly planar curve. Never set a tolerance below ~1e-7.
EPS_FLOOR = 1e-6


def eps_spectrum(xyz, bmax=3):
    """eps_B straight from sampled points, with planarizability.py's own normalisation."""
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(
        np.vstack([xyz, xyz[:1]]), axis=0), axis=1))])
    s = np.linspace(0, d[-1], PL.NS, endpoint=False)
    Q = np.stack([np.interp(s, d, np.vstack([xyz, xyz[:1]])[:, j]) for j in range(3)], 1)
    C = PL.cost_table(Q)
    ev = np.linalg.eigvalsh(((Q - Q.mean(0)).T @ (Q - Q.mean(0))) / len(Q))
    scale = np.sqrt(np.sqrt(max(ev[1], 1e-30) * max(ev[2], 1e-30)))
    return [float(np.sqrt(max(PL.best_split(C, B)[0], 0) / len(Q)) / scale)
            for B in range(1, bmax + 1)]


# ---------------------------------------------------------------------------
# T0a: analytic pure-TF field on an axisymmetric surface
# ---------------------------------------------------------------------------
class _AxisymTF:
    """Stands in for an Equilibrium: an axisymmetric surface carrying B = B0 R0 / R phi-hat."""

    def __init__(self, R0=7.96, a=1.77, kappa=1.7, delta=0.3, B0=6.0):
        self.R0, self.B0 = R0, B0
        self.surface = FourierRZToroidalSurface(
            R_lmn=[R0, a, -a * delta / 2], modes_R=[[0, 0], [1, 0], [2, 0]],
            Z_lmn=[-kappa * a], modes_Z=[[-1, 0]], NFP=1, sym=True)
        self.NFP = 1
        self.name = "axisym-TF"

    def compute(self, keys, grid=None):
        d = self.surface.compute(["e_theta", "e_zeta", "n_rho", "|e_theta x e_zeta|", "R"],
                                 grid=grid, basis="rpz")
        out = {k: np.asarray(d[k]) for k in ("e_theta", "e_zeta", "|e_theta x e_zeta|")}
        R = np.asarray(d["R"])
        B = np.zeros((len(R), 3))
        B[:, 1] = self.B0 * self.R0 / R                      # rpz: (R, phi, Z)
        n = np.asarray(d["n_rho"])
        out["K_vc"] = np.cross(n, B) / MU_0
        return out


def t0a():
    print("\n--- T0a: analytic pure-TF field, axisymmetric shaped surface ---")
    tf = _AxisymTF()
    pot = P.current_potential(tf, M=64, N=128)
    s = pot.summary()
    G_exact = 2 * np.pi * tf.R0 * tf.B0 / MU_0
    ok = True

    relG = abs(abs(s["G"]) - G_exact) / G_exact
    ok &= relG < 1e-10
    print(f"{OK if relG < 1e-10 else BAD} G = {s['G']:.6e} A, exact |G| = {G_exact:.6e}  rel {relG:.2e}")
    ok &= abs(s["I"]) / G_exact < 1e-12
    print(f"{OK if abs(s['I'])/G_exact < 1e-12 else BAD} I = {s['I']:.3e} A  (exact 0)   I/G = {s['I']/s['G']:.2e}")
    ok &= s["residual"] < 1e-8
    print(f"{OK if s['residual'] < 1e-8 else BAD} stream-function residual {s['residual']:.2e}")
    rms = s["Phi_tilde_rms"] / abs(s["G"])
    ok &= rms < 1e-12
    print(f"{OK if rms < 1e-12 else BAD} |Phi~|_rms / |G| = {rms:.2e}   (exact 0)")
    print(f"       monotonic in zeta: {s['monotonic']}, |K| ratio {s['absK_ratio']:.3f}")

    cons = P.modular_contours(pot, 8, n_theta=256)
    dz = max(float(np.ptp(z)) for _, z in cons)
    ok &= dz < 1e-10
    print(f"{OK if dz < 1e-10 else BAD} contours lie in zeta = const planes: max spread {dz:.2e} rad")

    curves = [P.to_surface_curve(t, z, surface=tf.surface, N_fourier=16) for t, z in cons]
    g = Grid(np.stack([np.zeros(256), np.zeros(256), np.linspace(0, 2 * np.pi, 256, endpoint=False)], 1),
             sort=False, jitable=False)
    eps = [eps_spectrum(np.asarray(c.compute("x", grid=g, basis="xyz")["x"]))[0] for c in curves]
    ok &= max(eps) < EPS_FLOOR
    print(f"{OK if max(eps) < EPS_FLOOR else BAD} eps_1 of the 8 proxy coils: max {max(eps):.2e}"
          f"  (exact 0; floor ~1.5e-8)")
    return ok


# ---------------------------------------------------------------------------
def t0b():
    print("\n--- T0b: planarizability zero on a planar-by-construction coilset ---")
    cs = load("917_results/planar_N7.h5")
    eps = [PL.spectrum(c, bmax=1)[0][1][0] for c in cs]
    good = max(eps) < EPS_FLOOR
    print(f"{OK if good else BAD} eps_1 of planar_N7: max {max(eps):.2e}  (floor ~1.5e-8)")
    return good


# ---------------------------------------------------------------------------
def t0c():
    print("\n--- T0c: real axisymmetric equilibrium (the O(I/G) proxy bias) ---")
    eq = get("DSHAPE")
    pot = P.current_potential(eq, M=64, N=64)
    s = pot.summary()
    print(f"       G = {s['G']:.4e} A, I = {s['I']:.4e} A, I/G = {s['I_over_G']:.4e}")
    print(f"       residual {s['residual']:.2e}, monotonic {s['monotonic']}, "
          f"|K| ratio {s['absK_ratio']:.3f}")
    if not s["monotonic"]:
        print(f"{BAD} not monotone -> outside the modular-proxy class")
        return False
    cons = P.modular_contours(pot, 8, n_theta=256)
    curves = [P.to_surface_curve(t, z, eq=eq, N_fourier=24) for t, z in cons]
    g = Grid(np.stack([np.zeros(256), np.zeros(256), np.linspace(0, 2 * np.pi, 256, endpoint=False)], 1),
             sort=False, jitable=False)
    eps = np.array([eps_spectrum(np.asarray(c.compute("x", grid=g, basis="xyz")["x"]))[0]
                    for c in curves])
    print(f"       eps_1 of the proxy coils: {eps.min():.3e} .. {eps.max():.3e}")
    print(f"       |I/G| = {abs(s['I_over_G']):.3e}   ratio eps_1(max)/|I/G| = "
          f"{eps.max()/abs(s['I_over_G']):.2f}")
    good = eps.max() < 10 * abs(s["I_over_G"])
    print(f"{OK if good else BAD} eps_1 is O(I/G) as predicted (bias, not a bug)")
    return good


if __name__ == "__main__":
    res = {"T0a": t0a(), "T0b": t0b(), "T0c": t0c()}
    print("\n" + "=" * 60)
    for k, v in res.items():
        print(f"{OK if v else BAD} {k}")
    sys.exit(0 if all(res.values()) else 1)
