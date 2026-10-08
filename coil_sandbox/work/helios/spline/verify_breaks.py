"""Validate `_splinexyz_helper` after the break-evaluation fix.

  1. closure and C0 continuity at every break
  2. x_s, x_ss, x_sss against finite differences of x (inside spans, away from breaks)
  3. the knots are interpolated exactly
  4. planarity: coplanar knots -> planar curve, per segment
  5. the no-break (periodic) path is untouched
  6. jit + AD through the whole thing (grad of length w.r.t. the knot values)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sandbox"))
import boot  # noqa: E402

boot.setup(default="cpu")

import numpy as np  # noqa: E402

import common as S  # noqa: E402
from desc.backend import jnp  # noqa: E402
from desc.coils import SplineXYZCoil  # noqa: E402
from desc.grid import Grid, LinearGrid  # noqa: E402
from desc.io import load  # noqa: E402


def sg(s):
    return Grid(np.stack([0 * s, 0 * s, np.mod(s, 2 * np.pi)], 1), sort=False, jitable=False)


def ev(c, s, key="x"):
    return np.asarray(c.compute(key, grid=sg(s), basis="xyz")[key])


def build(coil, n, method, breaks):
    k = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return SplineXYZCoil.from_values(float(coil.current), ev(coil, k), knots=k, method=method,
                                     basis="xyz",
                                     break_indices=None if breaks is None else np.array(breaks)), k


def main():
    arc = list(S.iter_unique(load("work/helios/polarB2_M5_cut30_final.h5")))[0]
    n, B = 64, 2
    ok = True

    for method in ("cubic", "cubic2"):
        spl, k = build(arc, n, method, [0, n // 2])
        brk = np.arange(B) * 2 * np.pi / B

        # 1. closure + C0 at the breaks. The curve MOVES across a finite probe offset
        # (|x_s| ~ L/2pi ~ 5.7 m/rad), so a continuous curve's apparent jump must scale
        # linearly with the offset; a real discontinuity would not shrink at all.
        jumps = [max(np.linalg.norm(ev(spl, np.array([b - e]))[0] - ev(spl, np.array([b + e]))[0])
                     for b in brk) for e in (1e-7, 1e-9)]
        clo = np.linalg.norm(ev(spl, np.array([0.0]))[0] - ev(spl, np.array([2 * np.pi - 1e-12]))[0])
        speed = np.linalg.norm(ev(spl, brk + 1e-7, "x_s"), axis=1).max()
        print(f"[{method}] C0: gap across +-1e-7 {jumps[0]:.2e} m (drift at |x_s|={speed:.2f} would "
              f"give {2e-7 * speed:.2e}), across +-1e-9 {jumps[1]:.2e} m   closure {clo:.2e} m")
        ok &= jumps[0] < 3 * 2e-7 * speed and jumps[1] < 3 * 2e-9 * speed and clo < 1e-9

        # 2. each derivative against a central difference of the previous one, at points
        # well inside a span (x_sss is only piecewise constant, so this is the right
        # comparison: differencing x_ss is exact within a span).
        s = np.array([0.37, 1.1, 2.0, 2.6, 3.7, 4.4, 5.2, 6.0])
        h = 1e-4
        # A central difference of x carries truncation h^2 |x_sss| / 6; differencing x_s
        # or x_ss is EXACT inside a span (x_ss is piecewise linear, x_sss constant).
        trunc = h**2 * np.abs(ev(spl, s, "x_sss")).max() / 6
        for lo_key, key, tol in (("x", "x_s", None), ("x_s", "x_ss", 1e-7),
                                 ("x_ss", "x_sss", 1e-6)):
            an = ev(spl, s, key)
            fd = (ev(spl, s + h, lo_key) - ev(spl, s - h, lo_key)) / (2 * h)
            rel = np.abs(an - fd).max() / max(np.abs(an).max(), 1e-12)
            t = tol if tol is not None else 5 * trunc / np.abs(an).max()
            print(f"[{method}] {key:5s} vs d/ds of {lo_key:4s}: max rel {rel:.2e} (tol {t:.2e}"
                  + ("" if tol is not None else ", = 5x the h^2|x_sss|/6 truncation") + ")")
            ok &= rel < t

        # 3. the knots are interpolated exactly
        P = ev(arc, k)
        print(f"[{method}] knot interpolation error {np.abs(ev(spl, k) - P).max():.2e} m")
        ok &= np.abs(ev(spl, k) - P).max() < 1e-10

        # 4. planarity per segment
        sd = np.linspace(0, 2 * np.pi, 2001, endpoint=False)
        x = ev(spl, sd)
        seg = np.minimum((sd * B / (2 * np.pi)).astype(int), B - 1)
        oop = max(np.abs((x[seg == i] - x[seg == i].mean(0))
                         @ np.linalg.svd(x[seg == i] - x[seg == i].mean(0))[2][-1]).max()
                  for i in range(B))
        print(f"[{method}] max out-of-plane per segment {oop:.2e} m")
        ok &= oop < 1e-10

    # 5. no-break path: identical to a direct periodic interp1d
    from interpax import interp1d

    spl, k = build(arc, n, "cubic2", None)
    sd = np.linspace(0, 2 * np.pi, 501, endpoint=False)
    ref = np.asarray(interp1d(sd, k, ev(arc, k), method="cubic2", period=2 * np.pi))
    print(f"\nno-break path vs direct interp1d: {np.abs(ev(spl, sd) - ref).max():.2e} m")
    ok &= np.abs(ev(spl, sd) - ref).max() < 1e-12

    # 6. jit + AD through a broken spline
    import jax

    spl, k = build(arc, n, "cubic2", [0, n // 2])
    g = LinearGrid(N=200)

    def length(X):
        p = dict(spl.params_dict)
        p["X"] = X
        d = spl.compute(["x_s", "ds"], grid=g, params=p)
        return jnp.sum(jnp.linalg.norm(d["x_s"], axis=-1) * d["ds"])

    L0 = float(length(spl.X))
    gr = np.asarray(jax.jit(jax.grad(length))(jnp.asarray(spl.X)))
    dh = 1e-6
    i = int(np.argmax(np.abs(gr)))
    Xp = np.asarray(spl.X).copy()
    Xp[i] += dh
    fd = (float(length(jnp.asarray(Xp))) - L0) / dh
    print(f"jit+grad: L {L0:.6f} m, dL/dX[{i}] analytic {gr[i]:.6e} vs FD {fd:.6e}, "
          f"rel {abs(gr[i] - fd) / max(abs(fd), 1e-12):.2e}")
    ok &= abs(gr[i] - fd) / max(abs(fd), 1e-12) < 1e-5 and np.all(np.isfinite(gr))

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
