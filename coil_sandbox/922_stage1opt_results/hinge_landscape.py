import os, sys
os.chdir("/home/singh/Documents/DESC2/coil_sandbox")
sys.path.insert(0,"sandbox"); import boot; boot.setup(default="cpu")
sys.path.insert(0,"922_stage1opt_results")
import numpy as np, phi_contours as P
from clamshell import _cost_table
from desc.examples import get
from desc.grid import Grid

def curve_pts(c, n=240):
    g = Grid(np.stack([np.zeros(n), np.zeros(n), np.linspace(0,2*np.pi,n,endpoint=False)],1), sort=False, jitable=False)
    X = np.asarray(c.compute("x", grid=g, basis="xyz")["x"])
    d = np.roll(X,-1,0)-np.roll(X,1,0)
    return X, np.linalg.norm(d,axis=1)/2

for name in ["precise_QA","precise_QH"]:
    eq = get(name)
    pot = P.current_potential(eq, M=96, N=192)
    cons = P.modular_contours(pot, 4*eq.NFP, n_theta=384)
    for ci in (0,):
        th, ze = cons[ci]
        c = P.to_surface_curve(th, ze, eq=eq, N_fourier=32)
        X, w = curve_pts(c, 240)
        n = len(X)
        C = _cost_table(X, w)
        W = w.sum()
        # whole-curve normalisation
        mu = (w[:,None]*X).sum(0)/W; D = X-mu
        ev = np.linalg.eigvalsh((D*w[:,None]).T@D/W)
        scale = np.sqrt(np.sqrt(max(ev[1],1e-30)*max(ev[2],1e-30)))
        # eps_2 landscape over all breakpoint pairs (i, j)
        i = np.arange(n)[:,None]; j = np.arange(n)[None,:]
        L1 = (j-i-1) % n; L2 = (i-j-1) % n
        tot = C[i, L1] + C[j, L2]
        E2 = np.sqrt(np.clip(tot,0,None)/W)/scale
        np.fill_diagonal(E2, np.inf)
        e2 = E2.min(); k = np.unravel_index(np.argmin(E2), E2.shape)
        e1 = np.sqrt(C[0, n-1]/W)/scale
        # how broad is the basin?
        fr = [float((E2 <= e2*(1+t)).sum())/ (n*n-n) for t in (0.02,0.05,0.10,0.25)]
        # count well-separated local minima within 5%
        good = np.argwhere(E2 <= e2*1.05)
        seps, seen = 0, []
        for (a,b) in good[np.argsort(E2[E2<=e2*1.05])]:
            if all(min(abs(a-p)%n, n-abs(a-p)%n) + min(abs(b-q)%n, n-abs(b-q)%n) > n//8 for p,q in seen):
                seen.append((a,b)); seps += 1
        print(f"{name} coil{ci}: eps_1 {e1:.4f}  eps_2 {e2:.4f}  C={4*e2/e1:.3f}   best hinges at s/2pi = "
              f"{k[0]/n:.3f}, {k[1]/n:.3f}  (arc split {abs(k[1]-k[0])/n:.2f}/{1-abs(k[1]-k[0])/n:.2f})")
        print(f"    fraction of ALL hinge pairs within 2/5/10/25% of the best: "
              f"{fr[0]*100:.1f}% {fr[1]*100:.1f}% {fr[2]*100:.1f}% {fr[3]*100:.1f}%")
        print(f"    well-separated local minima within 5% of best: {seps}")
        print(f"    eps_2 at the WORST hinge pair: {E2[np.isfinite(E2)].max():.4f}  "
              f"({E2[np.isfinite(E2)].max()/e2:.1f}x the best)")
