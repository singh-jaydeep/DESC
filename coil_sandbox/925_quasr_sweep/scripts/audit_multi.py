"""Is the proxy->real eps_1 offset STABLE across devices? If it is, the proxy ranks; if not, it doesn't.

Per device: vacuum solve -> eps_1/eps_2 of the proxy contours; eps_1/eps_2/eta_SVD of QUASR's
REAL coils; and the planar-projection bn ratio, for contrast with a real optimization.
"""
import sys, time, warnings
sys.path.insert(0, "sandbox"); sys.path.insert(0, "922_stage1opt_results")
import boot; boot.setup(default="gpu")
warnings.filterwarnings("ignore")
import numpy as np, quasr as Q, common as S, clamshell as CL
from contour_curve import ContourClamshell, _eps1_live
from desc.objectives import ForceBalance, ObjectiveFunction
from desc.coils import CoilSet, MixedCoilSet
from desc.grid import LinearGrid, Grid
CACHE = "/tmp/claude-1000/-home-singh-Documents-DESC2-coil-sandbox/91f880f0-34c8-4d34-ba49-8b36d79ca617/scratchpad/quasr_cache"
RES = 10

def curve_pts(c, n=480):
    s = 2*np.pi*np.arange(n)/n
    d = c.compute(["x","x_s"], grid=Grid(np.stack([0*s,0*s,s],1), sort=False, jitable=False), basis="xyz")
    return np.asarray(d["x"]), np.linalg.norm(np.asarray(d["x_s"]), axis=1)

def eta_svd(x, w):
    D = (x - (w[:,None]*x).sum(0)/w.sum()) * np.sqrt(w)[:,None]
    sv = np.linalg.svd(D, compute_uv=False)
    return sv[-1]/sv.sum()

rows = []
for ID in [1630198, 197197, 1265411, 1120872]:
    t0 = time.time()
    eq, rec = Q.equilibrium(ID, CACHE)
    K = 2*rec["nc_per_hp"]
    eq.change_resolution(L=RES, M=RES, N=RES, L_grid=2*RES, M_grid=2*RES, N_grid=2*RES)
    eq.solve(objective=ObjectiveFunction(ForceBalance(eq=eq), deriv_mode="batched", jac_chunk_size=100),
             maxiter=150, verbose=0, ftol=1e-10, xtol=1e-12, gtol=1e-12)
    lg = LinearGrid(rho=np.array([1.0]), M=2*RES, N=2*RES, NFP=eq.NFP)
    iot = float(np.asarray(lg.compress(eq.compute("iota", grid=lg)["iota"]))[0])
    # proxy
    o = ContourClamshell(eq, quantity="eps1", K=K, M=64, N=128, n=240, Mt=16, Nt=16, stride=8)
    o.build(verbose=0)
    p1 = np.asarray(o.compute(eq.params_dict))
    (xs, ws), dg = o._chain(eq.params_dict, o._constants, diagnostics=True)
    p2 = np.array([CL.eps_B(np.asarray(xs[k]), np.asarray(ws[k]), 2, stride=8)[0] for k in range(K)])
    # real coils
    base = [c for c in Q.coilset(ID, CACHE, full=False)]
    r1, r2, et = [], [], []
    for c in base:
        x, w = curve_pts(c)
        r1.append(float(_eps1_live(x, w))); r2.append(CL.eps_B(x, w, 2, stride=8)[0]); et.append(eta_svd(x, w))
    # planar projection
    tree = S.surface_tree(eq)
    wrap = lambda cl: MixedCoilSet(*[CoilSet(c, NFP=rec["nfp"], sym=True) for c in cl])
    m0 = S.evaluate(wrap(base), eq, tree=tree, vacuum=True)
    mp = S.evaluate(wrap([c.to_FourierPlanar(N=8, grid=LinearGrid(N=200), basis="xyz") for c in base]),
                    eq, tree=tree, vacuum=True)
    r = dict(ID=ID, nfp=rec["nfp"], hel=rec["helicity"], nc=rec["nc_per_hp"], A=rec["aspect_ratio"],
             qs=rec["qs_error"], iota_err=abs(abs(iot/rec["iota_profile"][-1])-1)*100,
             mono=dg["monotonic"], newton=max(dg["newton_residual"]),
             p1=p1.mean(), p2=p2.mean(), r1=np.mean(r1), r2=np.mean(r2), eta=np.mean(et),
             bn0=m0["bn"], bnp=mp["bn"], t=time.time()-t0)
    rows.append(r)
    print(f"ID {ID} nfp{r['nfp']}{'QH' if r['hel'] else 'QA'} nc{r['nc']} A{r['A']:.1f} qs1e{r['qs']:.2f} "
          f"| iota_err {r['iota_err']:.2f}% mono {r['mono']} newton {r['newton']:.0e} | "
          f"proxy eps1 {r['p1']:.5f} eps2 {r['p2']:.5f} | real eps1 {r['r1']:.5f} eps2 {r['r2']:.5f} "
          f"eta {r['eta']:.5f} | ratio r1/p1 {r['r1']/r['p1']:.2f} | bn xyz {r['bn0']:.2e} "
          f"planar-proj {r['bnp']:.2e} ({r['bnp']/r['bn0']:.0f}x) | {r['t']:.0f}s", flush=True)

print("\n=== summary ===")
print(f"{'ID':>8} {'dev':>10} {'proxy_eps1':>11} {'real_eps1':>10} {'ratio':>7} {'eta/eps1':>9} "
      f"{'eta_pred':>9} {'planar/xyz':>11}")
for r in rows:
    pred = r["r1"]/(r["r1"]+2)
    print(f"{r['ID']:>8} nfp{r['nfp']}{'QH' if r['hel'] else 'QA'}{'':>3} {r['p1']:>11.5f} {r['r1']:>10.5f} "
          f"{r['r1']/r['p1']:>7.2f} {r['eta']/r['r1']:>9.4f} {pred/r['r1']:>9.4f} {r['bnp']/r['bn0']:>10.0f}x")
rr = np.array([r["r1"]/r["p1"] for r in rows])
print(f"\nproxy->real eps_1 ratio: {rr.min():.2f} .. {rr.max():.2f}  (mean {rr.mean():.2f}, "
      f"spread {rr.max()/rr.min():.2f}x)")
p1 = np.array([r["p1"] for r in rows]); r1 = np.array([r["r1"] for r in rows])
print(f"proxy eps_1 across devices: {p1.min():.5f} .. {p1.max():.5f} ({p1.max()/p1.min():.2f}x)")
print(f"real  eps_1 across devices: {r1.min():.5f} .. {r1.max():.5f} ({r1.max()/r1.min():.2f}x)")
if len(rows) > 2:
    print(f"Spearman(proxy, real) = {__import__('scipy.stats', fromlist=['x']).spearmanr(p1, r1).statistic:+.3f}"
          f"   Pearson = {np.corrcoef(p1, r1)[0,1]:+.3f}")
