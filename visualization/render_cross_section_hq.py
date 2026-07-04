"""
================================================================================
 render_cross_section_hq.py  —  Publication-quality cross-section (model-driven)
================================================================================

An illustration-grade version of the cross-section diagram. Every proportion is
still derived from the VirtualCell state (fidelity preserved), but the rendering
adds scientific-illustration polish:

  * radial/linear gradients for volume (cell body, nucleus)
  * soft drop shadows and inner glow
  * membrane texture, ECM fiber weave, actin bundle highlights
  * glowing nuclear-enriched YAP on stiff, dispersed dim YAP on soft
  * clean leader-line labels

Output is a self-contained SVG (crisp, editable, publication-ready).

Model -> illustration mapping (unchanged from the schematic version):
  cell height/spread <- nuclear_drive   |  actin bundles <- nuclear_drive
  nucleus size/flat  <- area, drive     |  lamina thickness <- laminAC
  YAP localization   <- yap_nc          |  focal adhesions <- drive
  ECM crosslinking   <- stiffness

Usage:
    python visualization/render_cross_section_hq.py --save cell_hq.svg   # pair
    python visualization/render_cross_section_hq.py --E 23 --save stiff_hq.svg

Author: Daniel Pérez-Calixto (INMEGEN / UNAM)
================================================================================
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from virtual_cell import VirtualCell

W, H = 760, 600
GROUND = 430


def _hexmix(c1, c2, x):
    a = tuple(int(c1[i:i+2], 16) for i in (1, 3, 5))
    b = tuple(int(c2[i:i+2], 16) for i in (1, 3, 5))
    m = tuple(int(a[i]*(1-x) + b[i]*x) for i in range(3))
    return f"#{m[0]:02x}{m[1]:02x}{m[2]:02x}"


def _defs(drive, func, uid):
    """SVG <defs>: gradients, filters, textures — id-suffixed to allow 2 panels."""
    cyto_hi = _hexmix("#ffd9c8", "#eaf1ff", func)
    cyto_lo = _hexmix("#e08a68", "#b9c8ea", func)
    nuc_hi = _hexmix("#b79af0", "#c3a9e8", func)
    nuc_lo = _hexmix("#6c3fb8", "#7e5cc0", func)
    ecm_c = _hexmix("#7f93c4", "#b9c6e2", 1 - drive)
    return f'''<defs>
  <radialGradient id="cyto{uid}" cx="50%" cy="75%" r="75%">
    <stop offset="0%" stop-color="{cyto_hi}" stop-opacity="0.95"/>
    <stop offset="70%" stop-color="{cyto_hi}" stop-opacity="0.55"/>
    <stop offset="100%" stop-color="{cyto_lo}" stop-opacity="0.65"/>
  </radialGradient>
  <radialGradient id="nuc{uid}" cx="42%" cy="38%" r="70%">
    <stop offset="0%" stop-color="{nuc_hi}"/>
    <stop offset="100%" stop-color="{nuc_lo}"/>
  </radialGradient>
  <linearGradient id="sub{uid}" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="{_hexmix('#eef2fb','#9aa0ac',drive)}"/>
    <stop offset="100%" stop-color="{_hexmix('#dde5f5','#6f7480',drive)}"/>
  </linearGradient>
  <radialGradient id="yapg{uid}" cx="50%" cy="50%" r="50%">
    <stop offset="0%" stop-color="#7ea8ff"/>
    <stop offset="60%" stop-color="#3a52c8"/>
    <stop offset="100%" stop-color="#26307f"/>
  </radialGradient>
  <filter id="soft{uid}" x="-30%" y="-30%" width="160%" height="160%">
    <feGaussianBlur in="SourceAlpha" stdDeviation="4"/>
    <feOffset dx="0" dy="4" result="off"/>
    <feComponentTransfer><feFuncA type="linear" slope="0.3"/></feComponentTransfer>
    <feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <filter id="glow{uid}" x="-60%" y="-60%" width="220%" height="220%">
    <feGaussianBlur stdDeviation="3.5" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <filter id="rough{uid}"><feTurbulence type="fractalNoise" baseFrequency="0.9"
    numOctaves="2" result="n"/><feDisplacementMap in="SourceGraphic" in2="n"
    scale="3"/></filter>
</defs>'''


def render_svg(state, title="", label=True, uid="A"):
    drive = float(np.clip(state.nuclear_drive / 55.0, 0, 1))
    func = float(np.clip(state.function_index, 0, 1))
    cx = W / 2
    rng = np.random.default_rng(2)

    half_w = 155 + drive * 175
    height = 265 - drive * 165
    top = GROUND - height
    x0, x1 = cx - half_w, cx + half_w
    actin_col = _hexmix("#e8635a", "#ef7a72", 0.3)
    lamina_col = _hexmix("#4a2f8a", "#6b52a8", func)

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'font-family="Helvetica, Arial, sans-serif">']
    s.append(_defs(drive, func, uid))
    s.append(f'<rect width="{W}" height="{H}" fill="white"/>')

    # ---- substrate block ----
    s.append(f'<rect x="0" y="{GROUND}" width="{W}" height="{H-GROUND}" '
             f'fill="url(#sub{uid})"/>')
    # ECM weave (denser & straighter on stiff, looser & wavier on soft)
    n_mesh = int(7 + drive * 16)
    ecm_c = _hexmix("#6b7fb0", "#aab8da", 1 - drive)
    for i in range(n_mesh):
        y = GROUND + 14 + i * (H - GROUND - 20) / n_mesh
        amp = 5 + (1 - drive) * 9
        pts = " ".join(f"{x},{y + amp*np.sin(x/26 + i*0.7):.0f}"
                       for x in range(-10, W+20, 16))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{ecm_c}" '
                 f'stroke-width="{1.2 + drive*1.6:.1f}" opacity="0.75" '
                 f'stroke-linecap="round"/>')
    step = int(70 - drive*38)
    for x in range(-20, W, max(step, 18)):
        s.append(f'<line x1="{x}" y1="{GROUND+12}" x2="{x+step*0.5:.0f}" '
                 f'y2="{H-12}" stroke="{ecm_c}" stroke-width="{0.9+drive:.1f}" '
                 f'opacity="0.5"/>')

    # ---- cell body with volume gradient + soft shadow ----
    path = (f"M {x0},{GROUND} C {x0},{top+height*0.12} "
            f"{cx-half_w*0.5},{top} {cx},{top} "
            f"C {cx+half_w*0.5},{top} {x1},{top+height*0.12} {x1},{GROUND} Z")
    s.append(f'<path d="{path}" fill="url(#cyto{uid})" '
             f'stroke="{_hexmix("#c97b5c","#9fb0d8",func)}" stroke-width="2.5" '
             f'filter="url(#soft{uid})"/>')
    # membrane highlight (inner rim)
    s.append(f'<path d="{path}" fill="none" stroke="white" stroke-width="1.4" '
             f'opacity="0.5"/>')

    # ---- F-actin bundles (highlighted tubes) ----
    n_fib = int(4 + drive * 12)
    for i in range(n_fib):
        fx = cx - half_w*0.82 + i*(1.64*half_w/max(n_fib-1, 1))
        wdt = 1.4 + drive*5
        yap_apex = top + height*0.22 + rng.uniform(-8, 8)
        d = (f'M {fx:.0f},{GROUND-5} Q {cx:.0f},{yap_apex:.0f} '
             f'{W-fx:.0f},{GROUND-5}')
        s.append(f'<path d="{d}" fill="none" stroke="{actin_col}" '
                 f'stroke-width="{wdt:.1f}" opacity="{0.45+0.4*drive:.2f}" '
                 f'stroke-linecap="round"/>')
        s.append(f'<path d="{d}" fill="none" stroke="#ffd0cc" '
                 f'stroke-width="{max(wdt-2,0.6):.1f}" opacity="0.5"/>')

    # ---- nucleus with gradient + lamina + inner chromatin texture ----
    rn = 58 + (state.nuclear_area-100)*0.55
    rx = rn*(1+0.25*drive); ry = rn*(1-0.35*drive)
    ny = GROUND - max(height*0.42, ry+22)
    s.append(f'<ellipse cx="{cx}" cy="{ny:.0f}" rx="{rx:.0f}" ry="{ry:.0f}" '
             f'fill="url(#nuc{uid})" filter="url(#soft{uid})"/>')
    # chromatin mesh texture
    for _ in range(int(14+drive*8)):
        a = rng.uniform(0, 2*np.pi); rr = rng.uniform(0.2, 0.85)
        px, py = cx+np.cos(a)*rx*rr, ny+np.sin(a)*ry*rr
        a2 = rng.uniform(0, 2*np.pi); rr2 = rng.uniform(0.2, 0.85)
        px2, py2 = cx+np.cos(a2)*rx*rr2, ny+np.sin(a2)*ry*rr2
        s.append(f'<line x1="{px:.0f}" y1="{py:.0f}" x2="{px2:.0f}" '
                 f'y2="{py2:.0f}" stroke="#ffffff" stroke-width="0.7" '
                 f'opacity="0.18"/>')
    lam_w = 2.5 + (state.laminAC-1)*4
    s.append(f'<ellipse cx="{cx}" cy="{ny:.0f}" rx="{rx:.0f}" ry="{ry:.0f}" '
             f'fill="none" stroke="{lamina_col}" stroke-width="{lam_w:.1f}"/>')
    s.append(f'<ellipse cx="{cx}" cy="{ny:.0f}" rx="{rx-3:.0f}" ry="{ry-3:.0f}" '
             f'fill="none" stroke="white" stroke-width="0.8" opacity="0.4"/>')

    # ---- YAP: glowing & nuclear on stiff, dim & cytoplasmic on soft ----
    nuc_frac = np.clip((state.yap_nc-1)/4.0, 0.1, 0.95)
    for _ in range(13):
        inside = rng.random() < nuc_frac
        if inside:
            a = rng.uniform(0, 2*np.pi); r = rng.uniform(0, 0.65)
            yx, yy = cx+np.cos(a)*rx*r, ny+np.sin(a)*ry*r
            fil = f' filter="url(#glow{uid})"'; op = 1.0
        else:
            yx = cx+rng.uniform(-half_w*0.75, half_w*0.75)
            yy = rng.uniform(top+25, GROUND-25)
            if abs(yx-cx) > half_w*0.88:
                continue
            fil = ''; op = 0.55
        s.append(f'<g{fil} opacity="{op}">'
                 f'<ellipse cx="{yx:.0f}" cy="{yy:.0f}" rx="12" ry="8.5" '
                 f'fill="url(#yapg{uid})" stroke="white" stroke-width="0.8"/>'
                 f'<text x="{yx:.0f}" y="{yy+3:.0f}" font-size="8" fill="white" '
                 f'text-anchor="middle" font-weight="bold">YAP</text></g>')

    # ---- focal adhesions with integrin legs ----
    n_fa = int(3 + drive*7)
    for i in range(n_fa):
        fx = cx - half_w*0.85 + i*(1.7*half_w/max(n_fa-1, 1))
        fw = 7 + drive*10
        s.append(f'<ellipse cx="{fx:.0f}" cy="{GROUND}" rx="{fw:.0f}" ry="4.5" '
                 f'fill="{_hexmix("#8a5fb0","#c0407a",drive)}" '
                 f'filter="url(#soft{uid})"/>')
        for dx in (-4, 0, 4):
            s.append(f'<line x1="{fx:.0f}" y1="{GROUND}" x2="{fx+dx:.0f}" '
                     f'y2="{GROUND+13}" stroke="#5a7bbf" stroke-width="2.2" '
                     f'stroke-linecap="round"/>')

    # ---- labels ----
    sub_label = "Soft substrate" if drive < 0.5 else "Stiff substrate"
    ecm_label = ("Loosely crosslinked ECM" if drive < 0.5
                 else "Highly crosslinked ECM")
    s.append(f'<text x="{cx}" y="{H-16}" font-size="15" text-anchor="middle" '
             f'fill="#333" font-weight="500">{sub_label}</text>')
    if label:
        lab = [
            ("Cytoskeleton (F-actin)", cx-half_w*0.45, top+height*0.24, 60, top+34, "start"),
            ("Nucleus", cx+rx*0.4, ny-ry*0.4, W-60, ny-36, "end"),
            ("Nuclear lamina", cx+rx*0.9, ny, W-60, ny+6, "end"),
            ("YAP", cx-rx*0.25, ny+ry*0.2, 60, ny+30, "start"),
            ("Focal adhesions", cx-half_w*0.55, GROUND, 60, GROUND-6, "start"),
            (ecm_label, cx+70, GROUND+46, W-60, GROUND+56, "end"),
        ]
        for txt, x0l, y0l, x1l, y1l, anch in lab:
            s.append(f'<line x1="{x0l:.0f}" y1="{y0l:.0f}" x2="{x1l:.0f}" '
                     f'y2="{y1l:.0f}" stroke="#444" stroke-width="0.8"/>')
            s.append(f'<circle cx="{x0l:.0f}" cy="{y0l:.0f}" r="2" fill="#444"/>')
            dx = 6 if anch == "start" else -6
            s.append(f'<text x="{x1l+dx:.0f}" y="{y1l+4:.0f}" font-size="12.5" '
                     f'fill="#222" text-anchor="{anch}">{txt}</text>')

    if title:
        s.append(f'<text x="{cx}" y="26" font-size="15" text-anchor="middle" '
                 f'font-weight="bold" fill="#222">{title}</text>')
    s.append('</svg>')
    return "\n".join(s)


def render_pair(save="cell_cross_section_hq.svg", phenotype="hepatocyte"):
    cell = VirtualCell(phenotype)
    soft = render_svg(cell.simulate(1.0, t=120),
                      title="Soft (1 kPa) — rounded, YAP cytoplasmic", uid="S")
    stiff = render_svg(cell.simulate(23.0, t=120),
                       title="Stiff (23 kPa) — spread, YAP nuclear", uid="T")

    def inner(x): return x.split(">", 1)[1].rsplit("</svg>", 1)[0]
    combined = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {2*W} {H}" '
                f'font-family="Helvetica, Arial, sans-serif">'
                f'<g>{inner(soft)}</g>'
                f'<g transform="translate({W},0)">{inner(stiff)}</g></svg>')
    Path(save).write_text(combined)
    print(f"Saved {save}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=float, default=None)
    ap.add_argument("--t", type=float, default=120.0)
    ap.add_argument("--phenotype", default="hepatocyte")
    ap.add_argument("--save", default="cell_cross_section_hq.svg")
    args = ap.parse_args()
    if args.E is None:
        render_pair(save=args.save, phenotype=args.phenotype)
    else:
        cell = VirtualCell(args.phenotype)
        st = cell.simulate(args.E, t=args.t)
        Path(args.save).write_text(render_svg(st, title=f"E = {args.E:.0f} kPa"))
        print(f"Saved {args.save}")
