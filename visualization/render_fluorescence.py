"""
================================================================================
 render_fluorescence.py  —  Immunofluorescence-style virtual-cell rendering
================================================================================

Generates top-down, microscopy-style images of the virtual cell that look like
real immunofluorescence (F-actin / DAPI / YAP), driven entirely by the model
state. This is the visualization that matches experimental IF panels and is
suitable for a figure alongside real microscopy.

Channels (as in the experiment):
  * F-actin (grayscale or red)  — stress-fiber bundles; sparse & radial on soft,
    dense & aligned on stiff (reproducing the classic soft->stiff progression)
  * DAPI (blue)                 — nucleus; size from nuclear_area
  * YAP (green)                 — nuclear-enriched when active (high yap_nc)

Model -> image mapping:
  cell spread & shape      <- nuclear_drive (round on soft, spread/polarized on stiff)
  number of stress fibers  <- nuclear_drive (actomyosin)
  fiber alignment          <- nuclear_drive (isotropic->aligned)
  nucleus size             <- nuclear_area
  YAP nuclear fraction     <- yap_nc

Usage:
    python visualization/render_fluorescence.py --save panel.png          # progression
    python visualization/render_fluorescence.py --E 23 --t 120 --save one.png

Dependencies: numpy, matplotlib, scipy.
Author: Daniel Pérez-Calixto (INMEGEN / UNAM)
================================================================================
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

from virtual_cell import VirtualCell


# ---------------------------------------------------------------------------
# Cell outline: round on soft, spread & polarized (angular) on stiff
# ---------------------------------------------------------------------------
def _cell_mask(drive, size=512, seed=0):
    """Boolean mask of the cell footprint on a size×size grid."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    cx = cy = size / 2
    # base radius grows with stiffness (spreading)
    base = size * (0.16 + 0.20 * drive)
    ang = np.arctan2(yy - cy, xx - cx)
    # soft: nearly circular; stiff: polygonal/polarized with protrusions
    n_lobes = 3 + int(round(drive * 5))
    wobble = (0.10 + 0.30 * drive) * np.sin(n_lobes * ang + rng.uniform(0, 6))
    wobble += (0.05 + 0.18 * drive) * np.sin((n_lobes + 2) * ang + rng.uniform(0, 6))
    # stiff cells are elongated (polarized)
    elong = 1 + 0.5 * drive * np.cos(ang - 0.5) ** 2
    r = base * (1 + wobble) * elong
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return dist < r, (cx, cy), base


# ---------------------------------------------------------------------------
# F-actin channel: stress fibers
# ---------------------------------------------------------------------------
def _actin_channel(state, size=512, seed=0):
    drive = np.clip(state.nuclear_drive / 55.0, 0, 1)
    mask, (cx, cy), base = _cell_mask(drive, size, seed)
    img = np.zeros((size, size))
    rng = np.random.default_rng(seed + 1)

    # cortical actin (rim) — dominant on soft substrate
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    from scipy.ndimage import binary_erosion
    rim = mask & ~binary_erosion(mask, iterations=max(int(base * 0.10), 2))
    img[rim] += (1.0 - 0.6 * drive)      # strong rim on soft, weaker on stiff

    # stress fibers — few/radial on soft, many/aligned on stiff
    n_fib = int(8 + drive * 110)
    dom_angle = rng.uniform(0, np.pi)    # dominant orientation (polarization)
    fiber_layer = np.zeros((size, size))
    for _ in range(n_fib):
        if drive > 0.4:
            th = dom_angle + rng.normal(0, 0.30 * (1 - drive) + 0.04)  # aligned
        else:
            th = rng.uniform(0, np.pi)   # isotropic
        # fiber as a thin line segment through the cell, clipped to the mask
        off = rng.uniform(-0.92, 0.92) * base
        L = base * rng.uniform(1.0, 1.9)
        t = np.linspace(-L, L, 400)
        px = cx - np.sin(th) * off + np.cos(th) * t
        py = cy + np.cos(th) * off + np.sin(th) * t
        px = np.clip(px, 0, size - 1).astype(int)
        py = np.clip(py, 0, size - 1).astype(int)
        inside = mask[py, px]
        bright = rng.uniform(0.5, 1.0) * (0.4 + 0.6 * drive)
        fiber_layer[py[inside], px[inside]] += bright
    # thin, crisp fibers: light blur then combine
    fiber_layer = gaussian_filter(fiber_layer, sigma=0.7)
    img += fiber_layer

    img = gaussian_filter(img, sigma=0.6)
    img *= mask
    # gamma for microscopy-like contrast
    if img.max() > 0:
        img = (img / img.max()) ** 0.8
    return img, mask, (cx, cy), base


# ---------------------------------------------------------------------------
# Nucleus (DAPI) and YAP channels
# ---------------------------------------------------------------------------
def _nucleus_yap(state, center, base, size=512, seed=0):
    cx, cy = center
    drive = np.clip(state.nuclear_drive / 55.0, 0, 1)
    yy, xx = np.mgrid[0:size, 0:size]
    # nucleus radius from area; slightly elongated on stiff
    rn = size * (0.055 + (state.nuclear_area - 100) * 0.00055)
    a = rn * (1 + 0.25 * drive); b = rn * (1 - 0.10 * drive)
    th = 0.5
    xr = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
    yr = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
    nuc = np.exp(-((xr / a) ** 2 + (yr / b) ** 2) * 2.0)
    dapi = gaussian_filter(nuc, 2)
    dapi /= dapi.max()

    # YAP: nuclear-enriched fraction from yap_nc
    nuc_frac = np.clip((state.yap_nc - 1) / 4.0, 0.1, 0.95)
    mask_cell, _, _ = _cell_mask(drive, size, seed)
    cyto = gaussian_filter(mask_cell.astype(float), 6)
    cyto /= cyto.max() + 1e-9
    yap = nuc_frac * dapi + (1 - nuc_frac) * 0.5 * cyto
    yap = gaussian_filter(yap, 2)
    yap /= yap.max() + 1e-9
    return dapi, yap


# ---------------------------------------------------------------------------
# Compose an RGB immunofluorescence image
# ---------------------------------------------------------------------------
def render_if(state, size=512, seed=0, mode="merge"):
    """Return an RGB image (H,W,3) in [0,1]. mode: 'merge' | 'actin' | 'yap'."""
    actin, mask, center, base = _actin_channel(state, size, seed)
    dapi, yap = _nucleus_yap(state, center, base, size, seed)

    rgb = np.zeros((size, size, 3))
    if mode == "actin":
        rgb[..., 0] = rgb[..., 1] = rgb[..., 2] = actin      # grayscale F-actin
    elif mode == "yap":
        rgb[..., 1] = yap                                    # green YAP
        rgb[..., 2] = 0.9 * dapi                             # blue nucleus
    else:  # merge: actin(red) + YAP(green) + DAPI(blue)
        rgb[..., 0] = 0.9 * actin
        rgb[..., 1] = 0.95 * yap
        rgb[..., 2] = 1.0 * dapi
    return np.clip(rgb, 0, 1)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def render_single(E, t, phenotype="hepatocyte", save=None, mode="merge"):
    cell = VirtualCell(phenotype)
    s = cell.simulate(E, t=t)
    img = render_if(s, mode=mode)
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="black")
    ax.imshow(img); ax.axis("off")
    ax.set_title(f"E = {E:.0f} kPa   {s.fibrosis_stage}", color="white", fontsize=11)
    # scale bar
    ax.plot([380, 460], [480, 480], "w-", lw=3)
    if save:
        fig.savefig(save, dpi=150, facecolor="black", bbox_inches="tight")
        print(f"Saved {save}")
    plt.close(fig)
    return s


def render_progression(save="if_progression.png", phenotype="hepatocyte",
                        Es=(1, 5, 13, 23), t=120):
    """A stiffness progression panel (soft -> stiff), F-actin + merge rows,
    mirroring the experimental IF layout."""
    cell = VirtualCell(phenotype)
    n = len(Es)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.2), facecolor="black")
    for j, E in enumerate(Es):
        s = cell.simulate(E, t=t)
        # row 0: F-actin (grayscale)
        axes[0, j].imshow(render_if(s, mode="actin"))
        axes[0, j].set_title(f"{E:.0f} kPa · {s.fibrosis_stage}",
                             color="white", fontsize=11)
        # row 1: merge (actin/YAP/DAPI)
        axes[1, j].imshow(render_if(s, mode="merge"))
        for ax in (axes[0, j], axes[1, j]):
            ax.axis("off")
            ax.plot([380, 460], [485, 485], "w-", lw=2.5)
    axes[0, 0].text(-0.08, 0.5, "F-actin", color="white", rotation=90,
                    va="center", ha="right", transform=axes[0, 0].transAxes,
                    fontsize=12)
    axes[1, 0].text(-0.08, 0.5, "actin / YAP / DAPI", color="white", rotation=90,
                    va="center", ha="right", transform=axes[1, 0].transAxes,
                    fontsize=11)
    fig.suptitle("Virtual cell — model-generated immunofluorescence (soft → stiff)",
                 color="white", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0.02, 0, 1, 0.96])
    fig.savefig(save, dpi=150, facecolor="black", bbox_inches="tight")
    print(f"Saved {save}")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="IF-style virtual-cell renderer")
    ap.add_argument("--E", type=float, default=None)
    ap.add_argument("--t", type=float, default=120.0)
    ap.add_argument("--phenotype", default="hepatocyte")
    ap.add_argument("--mode", default="merge", choices=["merge", "actin", "yap"])
    ap.add_argument("--save", default="if_progression.png")
    args = ap.parse_args()
    if args.E is None:
        render_progression(save=args.save, phenotype=args.phenotype)
    else:
        render_single(args.E, args.t, args.phenotype, save=args.save, mode=args.mode)
