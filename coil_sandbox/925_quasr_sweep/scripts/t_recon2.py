import sys; sys.path.insert(0, "sandbox")
import boot; boot.setup(default="cpu")
import warnings; warnings.filterwarnings("ignore")
import quasr as Q
CACHE = "/tmp/claude-1000/-home-singh-Documents-DESC2-coil-sandbox/91f880f0-34c8-4d34-ba49-8b36d79ca617/scratchpad/quasr_cache"
import numpy as np
idx = Q.index(CACHE)
# pick a few devices spanning nfp / helicity / nc_per_hp, with decent qs_error
def pick(nfp, hel, nc):
    s = (idx["nfp"]==nfp)&(idx["helicity"]==hel)&(idx["nc_per_hp"]==nc)&(idx["qs_error"]<-3.0)
    if not s.any(): return None
    j = np.argsort(idx["qs_error"][s])[0]
    return int(idx["ID"][s][j])
tests = [1630198] + [i for i in (pick(2,0,4), pick(3,1,3), pick(5,1,2), pick(1,0,2)) if i]
for ID in tests:
    eq, rec = Q.equilibrium(ID, CACHE)
    b = Q.check_boundary(ID, CACHE, eq, rec)
    tab, m = Q.check_reconstruction(ID, CACHE, None, eq, rec)
    print(f"\nID {ID}  nfp {rec['nfp']} {'QH' if rec['helicity'] else 'QA'}  nc/hp {rec['nc_per_hp']}  "
          f"A {rec['aspect_ratio']:.2f}  a {rec['minor_radius']:.4f}  iota {rec['mean_iota']}  qs 1e{rec['qs_error']:.2f}")
    print(f"   boundary: a {b['a']['rel']:.1e}  A {b['A']['rel']:.1e}  V {b['V']['rel']:.1e}")
    for k in ("n_coils","total_coil_length","max_kappa","max_msc","min_coil2coil_dist","min_coil2surface_dist"):
        v=tab[k]; print(f"   {k:22s} {v['got']:11.6f} vs {v['want']:11.6f}   rel {v['rel']:.2e}")
    print(f"   bn {m['bn']:.3e}  bn_max {m['bn_max']:.3e}  linked {m['linked']}")
