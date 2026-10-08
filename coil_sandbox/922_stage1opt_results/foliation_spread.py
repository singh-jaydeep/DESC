import os, sys
os.chdir("/home/singh/Documents/DESC2/coil_sandbox")
sys.path.insert(0,"sandbox"); import boot; boot.setup(default="cpu")
sys.path.insert(0,"922_stage1opt_results")
import numpy as np, phi_contours as P
from clamshell import _cost_table
from desc.examples import get
from desc.grid import Grid

def pts(c, n=200):
    g = Grid(np.stack([np.zeros(n), np.zeros(n), np.linspace(0,2*np.pi,n,endpoint=False)],1), sort=False, jitable=False)
    X = np.asarray(c.compute("x", grid=g, basis="xyz")["x"])
    return X, np.linalg.norm(np.roll(X,-1,0)-np.roll(X,1,0),axis=1)/2

def eps12(X,w):
    n=len(X); C=_cost_table(X,w); W=w.sum()
    mu=(w[:,None]*X).sum(0)/W; D=X-mu
    ev=np.linalg.eigvalsh((D*w[:,None]).T@D/W)
    sc=np.sqrt(np.sqrt(max(ev[1],1e-30)*max(ev[2],1e-30)))
    e1=np.sqrt(C[0,n-1]/W)/sc
    i=np.arange(n)[:,None]; j=np.arange(n)[None,:]
    tot=C[i,(j-i-1)%n]+C[j,(i-j-1)%n]; np.fill_diagonal(tot,np.inf)
    e2=np.sqrt(max(tot.min(),0)/W)/sc
    return e1,e2

for name in ["precise_QA","precise_QH"]:
    eq=get(name)
    pot=P.current_potential(eq,M=96,N=192)
    NC=24                                  # 24 equal-dPhi contours over the full torus
    cons=P.modular_contours(pot,NC,n_theta=320)
    Kg=pot.absK                            # |K| on the (theta,zeta) grid
    th_g=pot.theta; ze_g=pot.zeta
    Cs=[]; Kbar=[]
    for (th,ze) in cons:
        c=P.to_surface_curve(th,ze,eq=eq,N_fourier=28)
        X,w=pts(c,200); e1,e2=eps12(X,w); Cs.append(4*e2/e1)
        ii=np.clip(np.round(th/(2*np.pi)*len(th_g)).astype(int)%len(th_g),0,len(th_g)-1)
        jj=np.clip(np.round(ze/(2*np.pi)*len(ze_g)).astype(int)%len(ze_g),0,len(ze_g)-1)
        Kbar.append(Kg[ii,jj].mean())
    Cs=np.array(Cs); Kbar=np.array(Kbar)
    r=np.corrcoef(Cs,Kbar)[0,1]
    # equal-current (uniform in Phi) mean == plain mean over these contours
    print(f"{name}: C over {NC} equal-current contours: {Cs.min():.3f}..{Cs.max():.3f} "
          f"(spread {Cs.max()/Cs.min():.2f}x), mean {Cs.mean():.4f}, median {np.median(Cs):.4f}")
    print(f"    |K| along contour: {Kbar.min():.3e}..{Kbar.max():.3e} ({Kbar.max()/Kbar.min():.3f}x)")
    print(f"    corr(C, |K|) across the foliation = {r:+.3f}")
    wK=Kbar/Kbar.sum()
    print(f"    equal-current mean {Cs.mean():.4f}  vs  |K|-weighted mean {float((Cs*wK).sum()):.4f}"
          f"   -> weighting changes the aggregate by {abs((Cs*wK).sum()-Cs.mean())/Cs.mean()*100:.2f}%")
