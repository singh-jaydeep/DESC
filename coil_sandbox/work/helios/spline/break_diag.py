"""Is the spline error at a C0 break a property of the break, or of which hinge it is?

Converts coil 0 three ways at the same resolution:
  A  knots start at hinge 0  -> breaks at s = 0 (wrap) and s = pi (interior)
  B  knots start at hinge 1  -> the two hinges swap roles
  C  no break_indices        -> periodic spline, corners rounded

If the error follows the INTERIOR break in both A and B, it is the `break_indices`
implementation (`_splinexyz_helper`'s mask-and-fill), not the coil.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sandbox"))
import boot  # noqa: E402

boot.setup(default="cpu")

import numpy as np  # noqa: E402

import common as S  # noqa: E402
from desc.coils import SplineXYZCoil  # noqa: E402
from desc.grid import Grid  # noqa: E402
from desc.io import load  # noqa: E402


def sgrid(s):
    return Grid(np.stack([0 * s, 0 * s, np.mod(s, 2 * np.pi)], 1), sort=False, jitable=False)


def xyz(c, s):
    return np.asarray(c.compute("x", grid=sgrid(s), basis="xyz")["x"])


def run(coil, n, method, shift, breaks):
    """Spline through n knots of `coil` starting at parameter `shift`; error vs the coil."""
    knots = np.linspace(0, 2 * np.pi, n, endpoint=False)
    spl = SplineXYZCoil.from_values(float(coil.current), xyz(coil, knots + shift), knots=knots,
                                    method=method, basis="xyz",
                                    break_indices=(np.array(breaks) if breaks is not None else None))
    s = np.linspace(0, 2 * np.pi, 4001, endpoint=False)
    dev = np.linalg.norm(xyz(spl, s) - xyz(coil, s + shift), axis=1)
    # error in a 0.3-rad window either side of each hinge, in the SHIFTED parameter
    win = {}
    for name, b in (("s=0", 0.0), ("s=pi", np.pi)):
        m = np.abs(np.mod(s - b + np.pi, 2 * np.pi) - np.pi) < 0.3
        win[name] = float(dev[m].max()) * 1e3
    far = np.ones_like(s, dtype=bool)
    for b in (0.0, np.pi):
        far &= np.abs(np.mod(s - b + np.pi, 2 * np.pi) - np.pi) >= 0.3
    return win, float(dev[far].max()) * 1e3


def main():
    cs = load("work/helios/polarB2_M5_cut30_final.h5")
    c0 = list(S.iter_unique(cs))[0]
    n, method = 64, "cubic2"
    print(f"coil 0, {n} knots, {method}.  max |spline - coil| in mm\n")
    print(f"{'case':44s} {'near s=0':>10s} {'near s=pi':>10s} {'elsewhere':>10s}")
    for label, shift, breaks in (
        ("A  start at hinge 0, breaks [0, n/2]", 0.0, [0, n // 2]),
        ("B  start at hinge 1, breaks [0, n/2]", np.pi, [0, n // 2]),
        ("C  start at hinge 0, no breaks", 0.0, None),
        ("D  start at hinge 0, break [0] only", 0.0, [0]),
    ):
        win, far = run(c0, n, method, shift, breaks)
        print(f"{label:44s} {win['s=0']:10.3f} {win['s=pi']:10.3f} {far:10.3f}")

    print("\nresolution scaling of the interior break (case A):")
    for n in (32, 64, 128, 256, 512):
        win, far = run(c0, n, method, 0.0, [0, n // 2])
        print(f"  n={n:4d}  s=0 {win['s=0']:9.4f}   s=pi {win['s=pi']:9.4f}   elsewhere {far:9.4f}")


if __name__ == "__main__":
    main()
