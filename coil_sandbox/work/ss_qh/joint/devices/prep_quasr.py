"""A QUASR device's boundary as a DESC .h5 for fixed-equilibrium stage 2 (unsolved: vacuum B.n needs only the boundary).

    python work/ss_qh/joint/devices/prep_quasr.py ID OUT.h5      (sandbox root, CPU)

PHIEDGE = pi a^2 (QUASR's B0 ~ 1 T, sandbox/quasr.py). NOT valid for single stage until the interior is solved.
"""
import sys; sys.path.insert(0, "sandbox")
import boot; boot.setup(default="cpu")
import quasr as Q

ID, out = int(sys.argv[1]), sys.argv[2]
eq, rec = Q.equilibrium(ID, "925_quasr_sweep/cache")
print(f"QUASR {ID}: nfp {rec['nfp']} helicity {rec['helicity']} A {rec['aspect_ratio']:.3f} a {rec['minor_radius']:.4f} "
      f"iota {rec['mean_iota']:.3f} qs_error {rec['qs_error']:.3f} nc/hp {rec['nc_per_hp']}; boundary check {Q.check_boundary(ID, '925_quasr_sweep/cache', eq, rec)}")
eq.save(out)
print(f"wrote {out}")
