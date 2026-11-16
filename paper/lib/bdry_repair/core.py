"""repair(eq): decide whether the boundary needs it, re-parametrize with as little surface motion as the resolution
allows, rebuild the interior, verify.

Why a repair exists at all: B.n cannot see a re-parametrization of the boundary, so a free-boundary solve drifts along
that freedom until the poloidal angle stalls at a point (|e_theta x e_zeta| -> 0: a mm-scale kink with a 10-20% B.n
spike and a negative Jacobian at the edge that the coils' field lines do not have; ss/tip.py, ss/tip_trace.py).
"""
from .diagnostics import boundary_health, displacement, jac_min
from .interior import INTERIOR
from .refit import reparametrize

# A boundary is left alone when all three hold. Clean boundaries in this project sit at area 0.55-0.65 and curvature
# 40-50 /m; creased ones at area < 0.02 and curvature > 1e4 /m, with sqrt(g) < 0 at the edge.
HEALTHY = dict(area_min=0.30, kappa_max=300.0, jac_min=0.05)


def needs_repair(eq, healthy=HEALTHY):
    """(bool, dict(area, kappa, jac)) for the eq's boundary and edge coordinates."""
    area, kappa = boundary_health(eq, eq.NFP)
    jac = jac_min(eq)
    bad = area < healthy["area_min"] or kappa > healthy["kappa_max"] or min(jac) < healthy["jac_min"]
    return bad, dict(area=area, kappa=kappa, jac=jac)


def repair(eq, chunk=200, verbose=0, say=print, force=False, rule="chordal", max_steps=8, interior="continuation",
           healthy=HEALTHY, **guards):
    """Returns (new_eq, info). new_eq is eq itself (untouched) when the boundary is already healthy and not force.

    rule / max_steps / guards: see refit.reparametrize (max_steps=0 is the old fixed equal-arclength refit).
    interior: a key of interior.INTERIOR or a callable f(eq, surface, chunk, verbose).
    If the re-solved edge still folds, the refit is redone with fewer label steps (down to the fixed-label fit).
    """
    bad, h0 = needs_repair(eq, healthy)
    if not (bad or force):
        say(f"REPAIR: not needed (area element min/median {h0['area']:.3f}, max curvature {h0['kappa']:.0f} /m, sqrt(g) "
            f"min/median at rho .8/.95/1 {['%+.2f' % v for v in h0['jac']]})")
        return eq, dict(action="none", before=h0)
    solve = INTERIOR[interior] if isinstance(interior, str) else interior
    steps = max_steps
    while True:
        surf, hist = reparametrize(eq, rule=rule, max_steps=steps, say=say if verbose else None,
                                   **guards)
        kept = [h for h in hist if h["kept"]]
        new = solve(eq, surf, chunk=chunk, verbose=verbose)
        bad1, h1 = needs_repair(new, healthy)
        if not bad1 or steps == 0:
            break
        say(f"REPAIR: edge still unhealthy after {kept[-1]['step']} label steps ({h1}); retrying with fewer")
        steps = max(0, kept[-1]["step"] - 1) if kept[-1]["step"] else 0
    mv = displacement(eq, new, eq.NFP)
    info = dict(action="repaired", rule=rule, label_steps=kept[-1]["step"], before=h0, after=h1, still_unhealthy=bool(bad1),
                surface_moved_mm=mv, surface_moved_max_mm=mv["max"], fixed_label_fit_moved_mm=
                {k: hist[0][k] for k in ("rms", "p99", "max")}, refit_history=hist,
                area_min_over_median=[h0["area"], h1["area"]], kappa_max=[h0["kappa"], h1["kappa"]],
                jac_min_over_median_rho_0p8_0p95_1=[h0["jac"], h1["jac"]])
    say(f"REPAIR ({rule}, {kept[-1]['step']} label steps): surface moved rms {mv['rms']:.3f} / p99 {mv['p99']:.2f} / max "
        f"{mv['max']:.2f} mm (fixed-label refit: {hist[0]['rms']:.3f} / {hist[0]['p99']:.2f} / {hist[0]['max']:.2f}); area "
        f"element min/median {h0['area']:.3f} -> {h1['area']:.3f}; max curvature {h0['kappa']:.0f} -> {h1['kappa']:.0f} /m; "
        f"sqrt(g) min/median at rho .8/.95/1 {['%+.2f' % v for v in h0['jac']]} -> {['%+.2f' % v for v in h1['jac']]}"
        + ("  STILL UNHEALTHY" if bad1 else ""))
    return new, info
