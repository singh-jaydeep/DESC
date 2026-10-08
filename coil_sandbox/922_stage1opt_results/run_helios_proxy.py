import os, sys
os.chdir("/home/singh/Documents/DESC2/coil_sandbox")
sys.path.insert(0, "sandbox"); import boot; boot.setup(default="cpu")
sys.path.insert(0, "922_stage1opt_results"); sys.path.insert(0, "917_results")
import numpy as np, phi_contours as P, planarizability as PL
import common as S
from desc.grid import Grid
from t0_checks import eps_spectrum

eq, _ = S.load_equilibrium("917_results/helios_repro.h5")
pot = P.current_potential(eq, M=128, N=256)
s = pot.summary()
print("Helios current potential on rho=1")
for k in ["G","I","I_over_G","residual","monotonic","absK_min","absK_max","absK_ratio"]:
    v=s[k]; print(f"  {k:12} {v}" if isinstance(v,bool) else f"  {k:12} {v:.6e}")
print(f"  dPhi/dzeta range {s['dPhi_dzeta_min']:.4e} .. {s['dPhi_dzeta_max']:.4e}  (must not straddle 0)")
print(f"  |Phi~|_rms/|G| {s['Phi_tilde_rms']/abs(s['G']):.4e}")
g = Grid(np.stack([np.zeros(400), np.zeros(400), np.linspace(0,2*np.pi,400,endpoint=False)],1), sort=False, jitable=False)
for nc in (12, 16):
    cons = P.modular_contours(pot, nc, n_theta=512)
    cur = [P.to_surface_curve(t, z, eq=eq, N_fourier=32) for t, z in cons]
    E = np.array([eps_spectrum(np.asarray(c.compute("x", grid=g, basis="xyz")["x"]), bmax=3) for c in cur])
    L = [float(np.sum(np.linalg.norm(np.diff(np.vstack([x:=np.asarray(c.compute("x",grid=g,basis="xyz")["x"]), x[:1]]),axis=0),axis=1))) for c in cur]
    print(f"\nN_c = {nc}:  current/coil {pot.G/nc/1e6:.3f} MA,  length {min(L):.2f}-{max(L):.2f} m")
    print(f"  eps_1 {E[:,0].min():.4f}-{E[:,0].max():.4f}   eps_2 {E[:,1].min():.4f}-{E[:,1].max():.4f}   eps_3 {E[:,2].min():.4f}-{E[:,2].max():.4f}")
    m = E.mean(0)
    print(f"  mean: eps_1 {m[0]:.4f}  eps_2 {m[1]:.4f}  eps_3 {m[2]:.4f}"
          f"   ln(e1/e2)={np.log(m[0]/m[1]):.3f}  ln(e2/e3)={np.log(m[1]/m[2]):.3f}"
          f"   ratio {np.log(m[0]/m[1])/np.log(m[1]/m[2]):.1f}")
