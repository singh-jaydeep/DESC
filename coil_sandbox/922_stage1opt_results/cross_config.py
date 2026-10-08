import os, sys
os.chdir("/home/singh/Documents/DESC2/coil_sandbox")
sys.path.insert(0,"sandbox"); import boot; boot.setup(default="cpu")
sys.path.insert(0,"922_stage1opt_results"); sys.path.insert(0,"917_results")
import numpy as np, phi_contours as P, common as S
from desc.examples import get
from desc.grid import Grid, LinearGrid
from t0_checks import eps_spectrum

g = Grid(np.stack([np.zeros(400), np.zeros(400), np.linspace(0,2*np.pi,400,endpoint=False)],1), sort=False, jitable=False)
rows=[]
for name in ["precise_QA","precise_QH","HSX","W7-X","ESTELL","NCSX","ARIES-CS"]:
    try:
        eq = get(name)
    except Exception as e:
        print(f"{name}: load failed {e}"); continue
    gr = LinearGrid(rho=np.array([1.0]), M=32, N=32, NFP=eq.NFP, sym=False)
    p = np.asarray(eq.compute("p", grid=gr)["p"])
    beta = float(np.asarray(eq.compute("<beta>_vol")["<beta>_vol"]))
    pot = P.current_potential(eq, M=96, N=192)
    s = pot.summary()
    if not s["monotonic"]:
        print(f"{name:10} NOT MONOTONE -> outside the modular-proxy class; I/G {s['I_over_G']:.2e}")
        continue
    cons = P.modular_contours(pot, 4*eq.NFP, n_theta=384)
    cur = [P.to_surface_curve(t,z,eq=eq,N_fourier=32) for t,z in cons]
    E = np.array([eps_spectrum(np.asarray(c.compute("x",grid=g,basis="xyz")["x"]),bmax=3) for c in cur]).mean(0)
    null2, null3 = E[0]/4, E[0]/9
    print(f"{name:10} NFP{eq.NFP} <beta>{beta:8.2e} p_edge{float(p.max()):9.2e} "
          f"I/G {s['I_over_G']:+.2e} res {s['residual']:.1e} |K|rat {s['absK_ratio']:.2f} | "
          f"eps1 {E[0]:.4f} eps2 {E[1]:.4f}({null2:.4f}) eps3 {E[2]:.4f}({null3:.4f}) "
          f"ratio {np.log(E[0]/E[1])/np.log(E[1]/E[2]):.2f}")
