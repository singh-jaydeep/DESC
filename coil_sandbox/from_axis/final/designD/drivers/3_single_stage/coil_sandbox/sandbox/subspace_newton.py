"""Subspace-Newton solve of a nonzero-residual least-squares problem (the DESC fixed-boundary equilibrium).

Why (work/ss_qh/joint/stall/DIAGNOSIS.md): the discrete equilibrium is the minimizer of 1/2 |F|^2 with a
NONZERO residual floor. Along its soft directions (mostly theta relabelings, singular values ~1e-7 of the max)
the dropped term S = sum_i F_i grad^2 F_i dominates J^T J, so Gauss-Newton does not converge there: it stops
wherever gtol is met, and gauge-sensitive outputs (QS, iota) depend on the start and the tolerance. The exact
Hessian (lsq-composite, hessian="exact") reaches a unique minimizer but costs dim_x Hessian-vector products per
iteration. S only matters where J^T J is small, so this solver adds it only on the soft subspace:

    B = J^T J + V (V^T S V) V^T,   V = eigenvectors of J^T J with the k smallest eigenvalues,

with V^T S V from k Hessian-vector products (S v = H v - J^T J v, H v forward-over-reverse). B is
indefinite in the subspace (S has negative curvature there), so the step is a trust-region step in B's
eigenbasis (More-Sorensen; a fixed Levenberg-Marquardt shift of |min eig| crawls along negative curvature).
The decrease is measured 'paired' (1/2 sum (f - f_new)(f + f_new)): it is below the rounding of the cost.

Variables are those of a LinearConstraintProjection (the fixed-boundary set eliminated), as in eq.solve.
"""

import time

import jax
import numpy as np

from desc.backend import jnp


def subspace_newton(lcp, x0, k=128, maxiter=200, gtol=1e-14, ftol=1e-15, xtol=1e-14, tr0=1e-4, hvp_chunk=16,
                    verbose=1):
    """Minimize 1/2 |lcp.compute_scaled_error(x)|^2 from x0. Returns (x, info)."""
    fun = lcp.compute_scaled_error
    jac = lcp.jac_scaled_error

    def cost(x):
        f = fun(x)
        return 0.5 * jnp.sum(f**2)

    grad = jax.grad(cost)

    @jax.jit
    def hvp_block(x, Vb):
        return jax.vmap(lambda v: jax.jvp(grad, (x,), (v,))[1])(Vb)

    x = jnp.asarray(x0)
    f = np.asarray(fun(x))
    c = 0.5 * float(f @ f)
    radius = tr0
    hist = []
    t0 = time.time()

    def tr_step(w, P, g, radius):
        """Minimize g.d + 1/2 d.B.d over |d| <= radius in B's eigenbasis (More-Sorensen, no hard case)."""
        Pg = P.T @ g
        if w.min() > 0:
            d = -P @ (Pg / w)
            if np.linalg.norm(d) <= radius:
                return d, 0.0
        lo = max(0.0, -w.min()) * (1 + 1e-12) + 1e-300
        hi = lo + np.linalg.norm(g) / radius + abs(w).max()
        for _ in range(200):  # |d(shift)| decreasing in shift: bisection in log space
            mid = np.sqrt(lo * hi) if lo > 0 else 0.5 * hi
            nd = np.linalg.norm(Pg / (w + mid))
            if nd > radius:
                lo = mid
            else:
                hi = mid
            if hi / max(lo, 1e-300) < 1 + 1e-10:
                break
        return -P @ (Pg / (w + hi)), hi

    for it in range(maxiter):
        J = np.asarray(jac(x))
        g = J.T @ f
        gnorm = np.abs(g).max()
        if gnorm < gtol:
            msg = "gtol"
            break
        A = J.T @ J
        lam, Q = np.linalg.eigh(A)  # ascending
        kk = min(k, len(lam))
        V = Q[:, :kk]
        HV = np.concatenate([np.asarray(hvp_block(x, jnp.asarray(V[:, i:i + hvp_chunk].T))).T
                             for i in range(0, kk, hvp_chunk)], axis=1)  # n x k
        SV = HV - A @ V
        Ssub = 0.5 * (V.T @ SV + SV.T @ V)
        B = A + V @ Ssub @ V.T
        B = 0.5 * (B + B.T)
        w, P = np.linalg.eigh(B)
        accepted = False
        for _ in range(40):
            d, shift = tr_step(w, P, g, radius)
            pred = -(g @ d + 0.5 * d @ (B @ d))
            fn = np.asarray(fun(x + d))
            # paired decrease: c - c_new without the cancellation (it is below the rounding of c here)
            act = 0.5 * float(np.sum((f - fn) * (f + fn)))
            rho = act / pred if pred > 0 else -1.0
            if rho < 0.25:
                radius = 0.25 * np.linalg.norm(d)
            elif rho > 0.75 and np.linalg.norm(d) > 0.99 * radius:
                radius = 2.0 * radius
            if rho > 1e-4 and act > 0:
                accepted = True
                break
        if not accepted:  # at the precision floor this is convergence, not failure
            msg = "converged (precision floor)" if gnorm < 1e-6 else "no acceptable step"
            break
        x = x + d
        f = fn
        c = 0.5 * float(f @ f)
        hist.append(dict(it=it, cost=c, gmax=float(gnorm), step=float(np.linalg.norm(d)), rho=float(rho),
                         radius=float(radius), shift=float(shift), wmin=float(w.min()), lam_min=float(lam[0]),
                         lam_k=float(lam[kk - 1]), ssub_min=float(np.linalg.eigvalsh(Ssub).min())))
        if verbose:
            h = hist[-1]
            print(f"  it {it:3d} cost {c:.10e} |g|inf {gnorm:.2e} step {h['step']:.2e} rho {rho:+.3f} "
                  f"radius {radius:.1e} shift {shift:.1e} | B_min {h['wmin']:+.2e} lam_min {h['lam_min']:.2e} "
                  f"lam_k {h['lam_k']:.2e} S_sub_min {h['ssub_min']:+.2e}", flush=True)
        # converged only on an INTERIOR Newton step: after rejections the radius (and so the step and the
        # decrease) can be tiny without the iteration being anywhere near the minimizer
        if (shift == 0.0 and act < ftol * max(c, 1e-300)
                and np.linalg.norm(d) < xtol * (xtol + np.linalg.norm(x))):
            msg = "ftol+xtol"
            break
    else:
        msg = "maxiter"
    return x, dict(msg=msg, nit=len(hist), wall=time.time() - t0, hist=hist)


def install_proximal_hook(k=128, gtol=1e-11, ftol=1e-13, xtol=1e-12, maxiter=400, verbose=0):
    """Route ProximalProjection's inner equilibrium solve to subspace_newton.

    ProximalProjection re-solves with eq.solve(objective=<its LinearConstraintProjection named
    "Eq Update LinearConstraintProjection">), right after eq.perturb has updated that projection's
    boundary targets (desc/perturbations.py). Only that call is replaced; every other eq.solve is untouched,
    and so is the proximal Jacobian (its zero-residual implicit derivative is accurate at the true minimizer:
    work/ss_qh/joint/stall/DIAGNOSIS.md). The first call (the proximal build) moves a GN-converged state to the
    true least-squares minimizer (~190 its, ~1 min at L=M=N=8); later calls are warm (~20-30 its).
    """
    from scipy.optimize import OptimizeResult

    from desc.equilibrium import Equilibrium
    from desc.optimize._constraint_wrappers import LinearConstraintProjection

    if getattr(Equilibrium.solve, "_subspace_newton", False):
        return
    orig = Equilibrium.solve
    stats = {"calls": 0, "its": 0, "wall": 0.0}

    def solve(self, objective="force", constraints=None, *args, **kwargs):
        if not (isinstance(objective, LinearConstraintProjection)
                and objective.name == "Eq Update LinearConstraintProjection"):
            return orig(self, objective, constraints, *args, **kwargs)
        x, info = subspace_newton(objective, objective.x(self), k=k, maxiter=maxiter, gtol=gtol, ftol=ftol,
                                  xtol=xtol, verbose=verbose)
        self.params_dict = self.unpack_params(np.asarray(objective.recover(x)))
        stats["calls"] += 1
        stats["its"] += info["nit"]
        stats["wall"] += info["wall"]
        if verbose or stats["calls"] <= 3 or stats["calls"] % 10 == 0:
            print(f"    [subspace-newton] call {stats['calls']}: {info['nit']} its, {info['wall']:.1f} s ({info['msg']}); "
                  f"totals {stats['its']} its, {stats['wall']:.0f} s", flush=True)
        return self, OptimizeResult(x=x, nit=info["nit"], nfev=info["nit"], message=info["msg"], success=True,
                                    cost=info["hist"][-1]["cost"] if info["hist"] else np.nan)

    solve._subspace_newton = True
    solve.stats = stats
    Equilibrium.solve = solve
    return stats
