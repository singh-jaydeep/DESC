"""Build a self-contained HTML viewer for one or more coilsets.

    # every start in a directory
    python tools/make_coil_viewer.py "initial starts/viewer.html" \
        --dir "initial starts"

    # named coilsets, e.g. a solved run against its start
    python c0/make_coil_viewer.py /tmp/cmp.html \
        --h5 start=c0/sweep_910/initial\\ starts/start_arcB5.h5 \
        --h5 solved=c0/runs/sw_arcB5_L1/sw_arcB5_L1.h5

THIS IS THE STANDARD for visualising coilsets in this campaign. Use it rather than
`plotly.write_html`, which embeds ~8 MB of library per file (`c0/coils_viewer.html` is
8.4 MB for three coilsets). Output here is one dependency-free file of a few hundred kB
that opens offline from `file://`.

WHAT IT IS FOR. Not decoration -- it is a geometric check on things the scalar metrics
cannot show, and this campaign has shipped two coilsets whose numbers looked like
improvements and whose geometry was wrong. Specifically:

* **All 16 coils are drawn, never 4.** The two representations nest DIFFERENTLY on disk --
  arcs as a `MixedCoilSet` of symmetric `CoilSet`s, xyz as one symmetric `CoilSet` -- and
  code that walks members and expands each gets 16 for arcs and **4 for xyz, silently**.
  This reads positions from the TOP level and asserts the count, so a viewer that
  renders
  four coils is itself the bug report.
* **The closest approach is drawn**, as a line between the two nearest points found on the
  sampled polylines, so `d_cc` is something you can see rather than a number to trust.
* **The tightest-curvature point is marked** on each coil.
* **Linked pairs are reported per coilset**, from the pairwise matrix.

The drawn closest approach is a SAMPLED-POINT estimate, for locating the contact. The
adjudicated `d_cc` in the stats panel is the segment-to-segment value at `VERIFY_N=400`,
which is the number that counts.
"""

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bundle  # noqa: E402,F401  -- MUST precede any `import desc`

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

NPTS = 160  # samples per coil
PLASMA_NTOR = 64
PLASMA_NPOL = 32
LINK_GRID_N = 100


def coil_positions(coilset, n=NPTS):
    """Every coil's sampled position, read from the TOP level. Shape (ncoil, n, 3)."""
    import numpy as np

    from desc.grid import LinearGrid

    pos = np.asarray(coilset._compute_position(grid=LinearGrid(N=n), basis="xyz"))
    if pos.ndim != 3 or pos.shape[-1] != 3:
        raise SystemExit(f"unexpected position shape {pos.shape}")
    return pos


def closest_pair(pos):
    """(i, j, pi, pj, d) for the nearest approach between two DIFFERENT coils."""
    import numpy as np

    best = (0, 1, pos[0][0], pos[1][0], float("inf"))
    nc = pos.shape[0]
    for i in range(nc):
        for j in range(i + 1, nc):
            d = np.linalg.norm(pos[i][:, None, :] - pos[j][None, :, :], axis=-1)
            k = int(np.argmin(d))
            a, b = divmod(k, d.shape[1])
            if d[a, b] < best[4]:
                best = (i, j, pos[i][a], pos[j][b], float(d[a, b]))
    return best


def curvature_peaks(coilset, n=NPTS):
    """Index of max |curvature| on each coil, and the value. Unique coils only."""
    import numpy as np
    from bundle_probe import _iter_unique

    from desc.grid import LinearGrid

    out = []
    g = LinearGrid(N=n)
    for c in _iter_unique(coilset):
        k = np.abs(np.asarray(c.compute("|curvature|", grid=g)["|curvature|"]))
        out.append((int(np.argmax(k)), float(np.max(k))))
    return out


def linked_pairs(coilset, n=LINK_GRID_N):
    """The pairwise linking matrix, reduced to the list of linked pairs (1-based)."""
    import numpy as np

    from desc.grid import LinearGrid

    lk = np.asarray(coilset._compute_linking_number(grid=LinearGrid(N=n)))
    out = []
    for i in range(lk.shape[0]):
        for j in range(i + 1, lk.shape[0]):
            if abs(float(lk[i, j])) > 0.5:
                out.append([i + 1, j + 1, round(float(lk[i, j]), 3)])
    return out


def plasma_wireframe(eq, ntor=PLASMA_NTOR, npol=PLASMA_NPOL):
    """The boundary surface as a (ntor, npol, 3) grid of points."""
    import numpy as np

    from desc.grid import LinearGrid

    g = LinearGrid(rho=np.array([1.0]), theta=npol, zeta=ntor, NFP=1, sym=False)
    x = np.asarray(eq.compute("x", grid=g, basis="xyz")["x"])
    # LinearGrid orders fastest in rho, then theta, then zeta
    return x.reshape(ntor, npol, 3)


def collect(name, path, eq, stats_by_rep):
    """Everything the viewer needs for one coilset."""
    from verify import hard_geometry, paper_bounds

    from desc.io import load

    cs = load(path)
    pos = coil_positions(cs)
    i, j, pi, pj, d = closest_pair(pos)
    peaks = curvature_peaks(cs)
    lim = paper_bounds(eq)
    m = hard_geometry(cs, eq, n=400)

    q = dict(
        length=float(m["max_length"]),
        kappa=float(m["max_max_kappa"]),
        kappa_MS=float(m["max_kappa_MS"]),
        d_cc=float(m["coil_coil_distance"]),
        d_pc=float(m["coil_surface_distance"]),
    )
    margins = dict(
        length=q["length"] / lim["length_max"],
        kappa=q["kappa"] / lim["kappa_max"],
        kappa_MS=q["kappa_MS"] / lim["kappa_ms_max"],
        d_cc=q["d_cc"] / lim["d_cc_min"],
        d_pc=q["d_pc"] / lim["d_pc_min"],
    )
    return dict(
        name=name,
        file=os.path.basename(path),
        n_coils=int(pos.shape[0]),
        n_unique=len(peaks),
        coils=[[[round(float(v), 5) for v in p] for p in c] for c in pos],
        closest=dict(
            i=i,
            j=j,
            a=[round(float(v), 5) for v in pi],
            b=[round(float(v), 5) for v in pj],
            d=round(d, 6),
        ),
        curv_peaks=[
            dict(coil=k, idx=idx, kappa=round(val, 4))
            for k, (idx, val) in enumerate(peaks)
        ],
        linked=linked_pairs(cs),
        quantities={k: round(v, 6) for k, v in q.items()},
        margins={k: round(v, 4) for k, v in margins.items()},
        meta=stats_by_rep.get(name, {}),
    )


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
:root{--bg:#f7f7f5;--fg:#1c1c1a;--mut:#6b6b66;--line:#dcdcd6;--pan:#fff;--acc:#2d6ca8}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
  --bg:#16171a;--fg:#e8e8e4;--mut:#9a9a93;--line:#2e3036;--pan:#1e2025;--acc:#6fb3ec}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:13px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
#wrap{display:flex;height:100vh;min-height:520px}
#side{width:300px;flex:0 0 300px;background:var(--pan);border-right:1px solid
#var(--line);
  overflow-y:auto;padding:14px}
#view{flex:1;position:relative;min-width:0}
canvas{display:block;width:100%;height:100%;cursor:grab}
canvas.drag{cursor:grabbing}
h1{font-size:15px;margin:0 0 2px}
.sub{color:var(--mut);font-size:11px;margin-bottom:14px}
h2{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);
  margin:16px 0 6px;font-weight:600}
button.sel{display:block;width:100%;text-align:left;padding:6px 8px;margin-bottom:3px;
  border:1px solid var(--line);background:transparent;color:var(--fg);border-radius:5px;
  cursor:pointer;font:inherit;font-size:12px}
button.sel:hover{border-color:var(--acc)}
button.sel[aria-pressed=true]{background:var(--acc);border-color:var(--acc);color:#fff}
table{width:100%;border-collapse:collapse;font-size:12px}
td{padding:2px 0;vertical-align:baseline}
td.k{color:var(--mut)}td.v{text-align:right;font-variant-numeric:tabular-nums}
.ok{color:#2e7d32}.bad{color:#c62828;font-weight:600}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]) .ok{color:#7fc784}
  :root:not([data-theme=light]) .bad{color:#f28b82}}
label.ck{display:block;margin:3px 0;font-size:12px;cursor:pointer}
.hint{color:var(--mut);font-size:11px;margin-top:12px}
#hud{position:absolute;top:10px;left:12px;font-size:11px;color:var(--mut);
  font-variant-numeric:tabular-nums;pointer-events:none}
#warn{margin:10px 0;padding:7px 9px;border-radius:5px;font-size:11.5px;
  background:#c6282814;border:1px solid #c6282855;color:#c62828}
@media (max-width:760px){#wrap{flex-direction:column}
  #side{width:100%;flex:0 0 auto;max-height:44vh;border-right:0;border-bottom:1px solid
  #var(--line)}
  #view{min-height:340px}}
</style></head><body>
<div id="wrap">
  <div id="side">
    <h1>__TITLE__</h1>
    <div class="sub" id="subtitle"></div>
    <h2>Coilset</h2><div id="picker"></div>
    <div id="warnbox"></div>
    <h2>Geometry <span style="text-transform:none;font-weight:400">(vs paper
    bounds)</span></h2>
    <table id="stats"></table>
    <h2>Show</h2>
    <label class="ck"><input type="checkbox" id="cPlasma" checked> plasma
    surface</label>
    <label class="ck"><input type="checkbox" id="cClosest" checked> closest
    approach</label>
    <label class="ck"><input type="checkbox" id="cCurv" checked> tightest-curvature
    point</label>
    <label class="ck"><input type="checkbox" id="cUnique"> unique coils only</label>
    <label class="ck"><input type="checkbox" id="cAxes" checked> axes</label>
    <h2>View</h2><div id="views"></div>
    <div class="hint">drag to rotate · scroll to zoom · shift-drag to pan</div>
  </div>
  <div id="view"><canvas id="cv"></canvas><div id="hud"></div></div>
</div>
<script>
const DATA = __DATA__;
const PLASMA = __PLASMA__;
const BOUNDS = __BOUNDS__;
const LBL = {length:"length",kappa:"κ max",kappa_MS:"κ MS",d_cc:"d_cc",d_pc:"d_pc"};
const UNIT = {length:"m",kappa:"1/m",kappa_MS:"1/m²",d_cc:"m",d_pc:"m"};
const LOWER = {d_cc:1,d_pc:1};   // these are lower bounds: margin must be >= 1

let cur = 0, yaw = 0.6, pitch = 0.5, zoom = 1, panx = 0, pany = 0;
const cv = document.getElementById("cv"), ctx = cv.getContext("2d");

function css(v){return
getComputedStyle(document.documentElement).getPropertyValue(v).trim()}
function rot(p){
  const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
  const x=p[0]*cy - p[1]*sy, y=p[0]*sy + p[1]*cy, z=p[2];
  return [x, y*cp - z*sp, y*sp + z*cp];   // [screen x, depth, screen y]
}
let scale=1, cx=0, cyv=0;
function proj(p){const r=rot(p);return [cx+(r[0]*scale+panx), cyv-(r[2]*scale+pany),
r[1]]}

function extent(){
  let m=0; for(const c of DATA[cur].coils) for(const p of c)
    m=Math.max(m,Math.abs(p[0]),Math.abs(p[1]),Math.abs(p[2]));
  return m||1;
}
function resize(){
  const r=cv.parentElement.getBoundingClientRect(), d=window.devicePixelRatio||1;
  cv.width=r.width*d; cv.height=r.height*d; ctx.setTransform(d,0,0,d,0,0);
  cx=r.width/2; cyv=r.height/2;
  scale=0.42*Math.min(r.width,r.height)/extent()*zoom;
  draw();
}
function seg(list,a,b,color,w){
  const A=proj(a),B=proj(b);
  list.push({z:(A[2]+B[2])/2,x1:A[0],y1:A[1],x2:B[0],y2:B[1],c:color,w:w});
}
function hue(i,n){return `hsl(${Math.round(360*i/n)},68%,50%)`}

function draw(){
  const d=DATA[cur]; if(!d) return;
  ctx.clearRect(0,0,cv.width,cv.height);
  const items=[];
  if(document.getElementById("cPlasma").checked && PLASMA.length){
    const col=css("--line"), nt=PLASMA.length, np=PLASMA[0].length;
    for(let i=0;i<nt;i+=2){for(let j=0;j<np;j++)
      seg(items,PLASMA[i][j],PLASMA[i][(j+1)%np],col,1);}
    for(let j=0;j<np;j+=4){for(let i=0;i<nt;i++)
      seg(items,PLASMA[i][j],PLASMA[(i+1)%nt][j],col,1);}
  }
  const only=document.getElementById("cUnique").checked;
  const n=only?d.n_unique:d.coils.length;
  for(let k=0;k<n;k++){
    const c=d.coils[k], col=hue(k,d.coils.length);
    for(let t=0;t<c.length;t++) seg(items,c[t],c[(t+1)%c.length],col,1.9);
  }
  items.sort((a,b)=>b.z-a.z);
  for(const s of items){
    ctx.strokeStyle=s.c; ctx.lineWidth=s.w; ctx.beginPath();
    ctx.moveTo(s.x1,s.y1); ctx.lineTo(s.x2,s.y2); ctx.stroke();
  }
  if(document.getElementById("cCurv").checked){
    ctx.fillStyle=css("--fg");
    d.curv_peaks.forEach(p=>{
      if(only && p.coil>=d.n_unique) return;
      const q=proj(d.coils[p.coil][p.idx]);
      ctx.beginPath(); ctx.arc(q[0],q[1],3,0,7); ctx.fill();
    });
  }
  if(document.getElementById("cClosest").checked){
    const A=proj(d.closest.a), B=proj(d.closest.b);
    ctx.strokeStyle="#e53935"; ctx.lineWidth=2.5; ctx.setLineDash([5,3]);
    ctx.beginPath(); ctx.moveTo(A[0],A[1]); ctx.lineTo(B[0],B[1]); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle="#e53935";
    [A,B].forEach(q=>{ctx.beginPath();ctx.arc(q[0],q[1],3.5,0,7);ctx.fill()});
  }
  if(document.getElementById("cAxes").checked){
    const L=extent()*0.55, ax=[[[0,0,0],[L,0,0],"x"],[[0,0,0],[0,L,0],"y"],
                               [[0,0,0],[0,0,L],"z"]];
    ctx.font="11px ui-monospace,monospace";
    for(const [a,b,nm] of ax){
      const A=proj(a),B=proj(b);
      ctx.strokeStyle=css("--mut"); ctx.lineWidth=1; ctx.beginPath();
      ctx.moveTo(A[0],A[1]); ctx.lineTo(B[0],B[1]); ctx.stroke();
      ctx.fillStyle=css("--mut"); ctx.fillText(nm,B[0]+3,B[1]+3);
    }
  }
  document.getElementById("hud").textContent =
    `${d.n_coils} coils drawn${only?" ("+d.n_unique+" shown)":""} · `+
    `closest pair ${d.closest.i+1}–${d.closest.j+1} @ ${(d.closest.d*1000).toFixed(1)}
    mm`;
}

function panel(){
  const d=DATA[cur];
  document.getElementById("subtitle").textContent=d.file;
  const w=document.getElementById("warnbox"); w.innerHTML="";
  const msgs=[];
  if(d.n_coils!==16) msgs.push(`<b>${d.n_coils} coils, expected 16.</b> The coilset was
  `+
    `walked wrongly, or the nesting differs from what is assumed.`);
  if(d.linked.length) msgs.push(`<b>${d.linked.length} LINKED PAIR`+
    `${d.linked.length>1?"S":""}:</b> `+d.linked.map(p=>p[0]+"–"+p[1]).join(", ")+
    `. This coilset is void; its flux is meaningless.`);
  if(msgs.length) w.innerHTML=`<div id="warn">${msgs.join("<br><br>")}</div>`;
  const rows=Object.keys(LBL).map(k=>{
    const v=d.quantities[k], mg=d.margins[k];
    const good = LOWER[k] ? mg>=1 : mg<=1;
    return `<tr><td class="k">${LBL[k]}</td><td class="v">${v} ${UNIT[k]}</td>`+
      `<td class="v ${good?"ok":"bad"}">${mg.toFixed(3)}</td></tr>`;
  }).join("");
  document.getElementById("stats").innerHTML=rows+
    `<tr><td class="k">linked pairs</td><td class="v"></td>`+
    `<td class="v ${d.linked.length?"bad":"ok"}">${d.linked.length}</td></tr>`;
  draw();
}

const pick=document.getElementById("picker");
DATA.forEach((d,i)=>{
  const b=document.createElement("button");
  b.className="sel"; b.textContent=d.name; b.setAttribute("aria-pressed",i===0);
  b.onclick=()=>{cur=i;[...pick.children].forEach((c,k)=>
    c.setAttribute("aria-pressed",k===i)); resize(); panel();};
  pick.appendChild(b);
});
const VIEWS={"3D":[0.6,0.5],"top (xy)":[0,Math.PI/2],"front (xz)":[0,0],"side
(yz)":[Math.PI/2,0]};
const vb=document.getElementById("views");
Object.entries(VIEWS).forEach(([nm,[y,p]],i)=>{
  const b=document.createElement("button");
  b.className="sel"; b.textContent=nm; b.setAttribute("aria-pressed",i===0);
  b.onclick=()=>{yaw=y;pitch=p;panx=pany=0;[...vb.children].forEach((c,k)=>
    c.setAttribute("aria-pressed",k===i)); draw();};
  vb.appendChild(b);
});
["cPlasma","cClosest","cCurv","cUnique","cAxes"].forEach(id=>
  document.getElementById(id).onchange=draw);

let drag=null;
cv.addEventListener("pointerdown",e=>{drag={x:e.clientX,y:e.clientY,s:e.shiftKey};
  cv.classList.add("drag"); cv.setPointerCapture(e.pointerId)});
cv.addEventListener("pointermove",e=>{
  if(!drag) return;
  const dx=e.clientX-drag.x, dy=e.clientY-drag.y;
  if(drag.s){panx+=dx; pany-=dy;} else {yaw+=dx*0.008; pitch+=dy*0.008;}
  drag.x=e.clientX; drag.y=e.clientY; draw();
});
["pointerup","pointercancel"].forEach(t=>cv.addEventListener(t,e=>{
  drag=null; cv.classList.remove("drag")}));
cv.addEventListener("wheel",e=>{e.preventDefault();
  zoom*=Math.exp(-e.deltaY*0.0012); resize()},{passive:false});
window.addEventListener("resize",resize);
matchMedia("(prefers-color-scheme:dark)").addEventListener("change",draw);
resize(); panel();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument(
        "--dir", default=None, help="directory of start_*.h5 to include, in sweep order"
    )
    ap.add_argument(
        "--h5",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="a named coilset; repeatable",
    )
    ap.add_argument("--title", default="Coil viewer")
    a = ap.parse_args()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    items = []
    if a.dir:
        try:
            from sweep_configs import REPS

            order = [r[0] for r in REPS]
        except Exception:
            order = []
        found = {
            os.path.basename(f)[len("start_") : -3]: f
            for f in glob.glob(os.path.join(a.dir, "start_*.h5"))
        }
        for k in order + sorted(set(found) - set(order)):
            if k in found:
                items.append((k, found[k]))
    for spec in a.h5:
        if "=" not in spec:
            raise SystemExit(f"--h5 wants NAME=PATH, got {spec!r}")
        n, p = spec.split("=", 1)
        items.append((n, p))
    if not items:
        raise SystemExit("nothing to show: pass --dir or --h5")

    # per-start metadata, when the manifest is alongside
    meta = {}
    mpath = os.path.join(a.dir or "", "starts.json")
    if os.path.exists(mpath):
        for r in json.load(open(mpath))["starts"]:
            meta[r["rep"]] = dict(dof=r["dof"], r_over_a=r["spec"]["r_over_a"])

    from verify import paper_bounds

    from desc.examples import get

    eq = get("precise_QA")
    data = []
    for name, path in items:
        print(f"  {name} ... ", end="", flush=True)
        d = collect(name, path, eq, meta)
        print(f"{d['n_coils']} coils, {len(d['linked'])} linked", flush=True)
        data.append(d)

    plasma = [
        [[round(float(v), 5) for v in p] for p in row] for row in plasma_wireframe(eq)
    ]
    html = (
        HTML.replace("__TITLE__", a.title)
        .replace("__DATA__", json.dumps(data, separators=(",", ":")))
        .replace("__PLASMA__", json.dumps(plasma, separators=(",", ":")))
        .replace(
            "__BOUNDS__", json.dumps({k: float(v) for k, v in paper_bounds(eq).items()})
        )
    )
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w").write(html)
    print(
        f"\nwrote {a.out}  ({len(html)/1024:.0f} kB, {len(data)} coilsets, "
        f"no external dependencies)"
    )


if __name__ == "__main__":
    main()
