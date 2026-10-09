"""A QUASR device's boundary as a DESC .h5 for fixed-equilibrium stage 2 (unsolved: vacuum B.n needs only the boundary).

    python work/ss_qh/joint/devices/prep_quasr.py ID OUT.h5 [A_TARGET]      (sandbox root, CPU)

With A_TARGET (a minor radius in m) the boundary is scaled by c = A_TARGET / a (lengths x c, Psi x c^2; exact for a
vacuum boundary), so absolute bounds made for that minor radius apply.

PHIEDGE = pi a^2 (QUASR's B0 ~ 1 T, sandbox/quasr.py). NOT valid for single stage until the interior is solved.
"""
import sys; sys.path.insert(0, "sandbox")
import boot; boot.setup(default="cpu")
import quasr as Q

ID, out = int(sys.argv[1]), sys.argv[2]
eq, rec = Q.equilibrium(ID, "925_quasr_sweep/cache")
print(f"QUASR {ID}: nfp {rec['nfp']} helicity {rec['helicity']} A {rec['aspect_ratio']:.3f} a {rec['minor_radius']:.4f} "
      f"iota {rec['mean_iota']:.3f} qs_error {rec['qs_error']:.3f} nc/hp {rec['nc_per_hp']}; boundary check {Q.check_boundary(ID, '925_quasr_sweep/cache', eq, rec)}")
if len(sys.argv) > 3:
    c = float(sys.argv[3]) / float(eq.compute("a")["a"])
    eq.Rb_lmn = eq.Rb_lmn * c
    eq.Zb_lmn = eq.Zb_lmn * c
    eq.Psi = eq.Psi * c**2
    eq.set_initial_guess()
    print(f"scaled by c = {c:.5f}: R0 {float(eq.compute('R0')['R0']):.4f} a {float(eq.compute('a')['a']):.4f}")
eq.save(out)
print(f"wrote {out}")
