"""Equilibrium-side coil-complexity predictors, baseline vs optimized.

The question (user, 2026-09-23): eps_1 may CORRELATE with good planar coils without being a
useful thing to OPTIMIZE -- driving it could produce equilibria that are harder for every coil
representation. Two independent literature predictors say whether that is happening:

  * L_grad(B) = sqrt(2)|B| / ||grad B||_F on the LCFS. Kappel/Landreman/Malhotra PPCF 66 (2024)
    025018: R^2 = 0.944 for min(L_gradB)/a vs min(coil-surface distance)/a over 3027 QUASR
    configurations. NOTE the normalization by `a`, not R0 (0.644 if by R0). SMALLER = coils
    must come closer = harder.
  * pdrot = |grad_S alpha|, the rate at which the principal-curvature frame rotates across the
    surface. Pavone & Warmer arXiv:2604.26763, 7500 QI boundaries with SIMSOPT coil runs:
    Spearman rho = 0.936 against coil non-planarity, univariate R^2 = 0.700 -- the best single
    predictor they found, and specifically of NON-PLANARITY, which L_gradB does not predict
    (|rho| < 0.2 against every coil-complexity metric). LARGER = coils must be less planar.

alpha is a DIRECTOR angle (principal directions are unoriented lines), so it is only defined
mod pi and a naive atan2 has branch cuts. Work with the traceless part of the shape operator
in an orthonormal tangent frame: (c, s) with 2*alpha = atan2(s, c), and differentiate through
    d alpha = (c ds - s dc) / (2 (c^2 + s^2))
which is branch-free everywhere except umbilics (c = s = 0), reported separately.

Second derivatives come from FFT differentiation of R, Z on a uniform periodic (theta, zeta)
grid -- exact for band-limited data, and independent of DESC compute-key naming.
"""
import sys
sys.path.insert(0, "sandbox")
import boot; boot.setup(default="cpu")
import numpy as np                                                    # noqa: E402
from desc.io import load                                              # noqa: E402
from desc.examples import get                                         # noqa: E402
from desc.grid import LinearGrid                                      # noqa: E402

M, N = 128, 128          # uniform theta x zeta grid over ONE field period in zeta


def _d(f, axis, k):
    """Spectral derivative of a periodic array along `axis` with wavenumbers k."""
    F = np.fft.fft(f, axis=axis)
    shape = [1] * f.ndim
    shape[axis] = -1
    return np.real(np.fft.ifft(F * (1j * k).reshape(shape), axis=axis))


def surface_quantities(eq):
    NFP = int(eq.NFP)
    th = 2 * np.pi * np.arange(M) / M
    ze = 2 * np.pi * np.arange(N) / (N * NFP)          # one field period
    g = LinearGrid(rho=np.array([1.0]), theta=th, zeta=ze, NFP=1, sym=False)
    nodes = np.asarray(g.nodes)
    it = np.rint(nodes[:, 1] / (2 * np.pi) * M).astype(int) % M
    iz = np.rint(nodes[:, 2] * NFP / (2 * np.pi) * N).astype(int) % N
    perm = np.lexsort((iz, it))
    d = eq.compute(["R", "Z", "L_grad(B)", "|B|", "|e_theta x e_zeta|"], grid=g)
    R = np.asarray(d["R"])[perm].reshape(M, N)
    Z = np.asarray(d["Z"])[perm].reshape(M, N)
    Lg = np.asarray(d["L_grad(B)"])[perm].reshape(M, N)
    dA = np.asarray(d["|e_theta x e_zeta|"])[perm].reshape(M, N)
    zeta = ze[None, :] * np.ones((M, 1))

    kt = np.fft.fftfreq(M, 1 / M)
    kz = np.fft.fftfreq(N, 1 / N) * NFP                 # d/dzeta, not d/d(NFP zeta)
    Rt, Rz = _d(R, 0, kt), _d(R, 1, kz)
    Zt, Zz = _d(Z, 0, kt), _d(Z, 1, kz)
    Rtt, Rtz, Rzz = _d(Rt, 0, kt), _d(Rt, 1, kz), _d(Rz, 1, kz)
    Ztt, Ztz, Zzz = _d(Zt, 0, kt), _d(Zt, 1, kz), _d(Zz, 1, kz)
    c, s = np.cos(zeta), np.sin(zeta)

    def vec(a, b, cc):    # (a*cos - b*sin, a*sin + b*cos, cc)
        return np.stack([a * c - b * s, a * s + b * c, cc], axis=-1)

    x_t = vec(Rt, np.zeros_like(R), Zt)
    x_z = vec(Rz, R, Zz)
    x_tt = vec(Rtt, np.zeros_like(R), Ztt)
    x_tz = vec(Rtz, Rt, Ztz)
    x_zz = vec(Rzz - R, 2 * Rz, Zzz)

    E = np.sum(x_t * x_t, -1); F = np.sum(x_t * x_z, -1); G = np.sum(x_z * x_z, -1)
    nrm = np.cross(x_t, x_z); nrm = nrm / np.linalg.norm(nrm, axis=-1, keepdims=True)
    LL = np.sum(nrm * x_tt, -1); MM = np.sum(nrm * x_tz, -1); NN = np.sum(nrm * x_zz, -1)

    # shape operator in the orthonormal tangent frame u1 = x_t/sqrt(E), u2 = Gram-Schmidt
    W = G - F**2 / E
    S11 = LL / E
    S12 = (MM - (F / E) * LL) / (np.sqrt(E) * np.sqrt(W))
    S22 = (NN - 2 * (F / E) * MM + (F / E) ** 2 * LL) / W
    cc, ss = 0.5 * (S11 - S22), S12                    # director: 2 alpha = atan2(ss, cc)
    mag2 = cc**2 + ss**2
    cct, ccz = _d(cc, 0, kt), _d(cc, 1, kz)
    sst, ssz = _d(ss, 0, kt), _d(ss, 1, kz)
    a_t = 0.5 * (cc * sst - ss * cct) / mag2
    a_z = 0.5 * (cc * ssz - ss * ccz) / mag2
    # |grad_S alpha|^2 = g^{ij} a_i a_j, inverse first fundamental form
    det = E * G - F**2
    pdrot = np.sqrt((G * a_t**2 - 2 * F * a_t * a_z + E * a_z**2) / det)
    k1 = 0.5 * (S11 + S22) + np.sqrt(mag2)
    k2 = 0.5 * (S11 + S22) - np.sqrt(mag2)
    return dict(Lg=Lg, pdrot=pdrot, dA=dA, umbilic=np.sqrt(mag2),
                a=float(eq.compute("a")["a"]), k1=k1, k2=k2)


def report(tag, eq):
    q = surface_quantities(eq)
    a, dA, Lg, pd = q["a"], q["dA"], q["Lg"], q["pdrot"]
    w = dA / dA.sum()
    print(f"\n=== {tag} ===  a = {a:.5f} m   A = {float(eq.compute('R0/a')['R0/a']):.4f}")
    print(f"  L_gradB/a   min {Lg.min()/a:8.4f}   5th {np.percentile(Lg,5)/a:8.4f}   "
          f"median {np.percentile(Lg,50)/a:8.4f}   area-mean {np.sum(Lg*w)/a:8.4f}")
    print(f"  pdrot [1/m] max {pd.max():8.4f}   95th {np.percentile(pd,95):8.4f}   "
          f"median {np.percentile(pd,50):8.4f}   area-mean {np.sum(pd*w):8.4f}")
    print(f"  pdrot * a   max {pd.max()*a:8.4f}   95th {np.percentile(pd,95)*a:8.4f}   "
          f"median {np.percentile(pd,50)*a:8.4f}   area-mean {np.sum(pd*w)*a:8.4f}")
    print(f"  smallest |shape-operator director| (umbilic proximity) {q['umbilic'].min():.3e}")
    return dict(a=a, Lg_min=Lg.min()/a, Lg_5=np.percentile(Lg,5)/a,
                Lg_med=np.percentile(Lg,50)/a, Lg_mean=np.sum(Lg*w)/a,
                pd_max=pd.max()*a, pd_95=np.percentile(pd,95)*a,
                pd_med=np.percentile(pd,50)*a, pd_mean=np.sum(pd*w)*a)


if __name__ == "__main__":
    cases = [("baseline precise_QA", get("precise_QA"))]
    for p in sys.argv[1:]:
        cases.append((p.split("/")[-2] if "/" in p else p, load(p)))
    res = [(t, report(t, e)) for t, e in cases]
    b = res[0][1]
    print("\n" + "=" * 92)
    print("RELATIVE TO BASELINE  (L_gradB/a SMALLER = harder; pdrot*a LARGER = less planar)")
    print(f"{'run':22s} {'Lg min':>9s} {'Lg 5th':>9s} {'Lg med':>9s} {'Lg mean':>9s} "
          f"{'pd max':>9s} {'pd 95th':>9s} {'pd med':>9s} {'pd mean':>9s}")
    for t, r in res[1:]:
        print(f"{t:22s} " + " ".join(
            f"{(r[k]/b[k]-1)*100:+8.2f}%" for k in
            ("Lg_min","Lg_5","Lg_med","Lg_mean","pd_max","pd_95","pd_med","pd_mean")))
