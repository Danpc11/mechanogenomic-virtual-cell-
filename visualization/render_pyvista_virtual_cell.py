"""
================================================================================
 render_pyvista_virtual_cell.py  —  3D scientific visualization (PyVista+Trame)
================================================================================

Renders the mechanogenomic virtual cell in 3D from the model's own state
outputs (not a cartoon), and serves an interactive Trame app with sliders for
substrate stiffness and time. Every geometric feature is driven by a model
variable:

    nuclear ellipsoid size      <- nuclear_area A(t)
    nuclear flattening (z-axis) <- nuclear_drive sigma
    envelope thickness/opacity  <- lamin A/C
    intranuclear YAP spheres    <- YAP N/C ratio
    basal adhesion points       <- effective clutches n_c
    traction cones at adhesions <- traction T(E)
    color warmth                <- fibrosis stage / drive

Usage:
    # Static image (headless, works anywhere):
    python visualization/render_pyvista_virtual_cell.py --save E23_t120.png --E 23 --t 120

    # Interactive Trame app (needs a display / browser; run locally):
    python visualization/render_pyvista_virtual_cell.py --serve

Dependencies: pyvista, trame (pip install pyvista trame trame-vtk trame-vuetify).
For static export, pyvista with an off-screen backend is enough.
================================================================================
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pyvista as pv

from virtual_cell import VirtualCell


# ---------------------------------------------------------------------------
# Build the 3D scene from a model state
# ---------------------------------------------------------------------------
def build_scene(state, plotter=None):
    """Assemble a PyVista scene from a CellState. Returns the plotter."""
    p = plotter or pv.Plotter(off_screen=True, window_size=(700, 600))
    p.set_background("white")

    warm = np.clip(state.nuclear_drive / 55.0, 0, 1)
    edge = np.array([0.06, 0.43, 0.34]) * (1 - warm) + np.array([0.60, 0.24, 0.12]) * warm
    fill = np.array([0.62, 0.88, 0.80]) * (1 - warm) + np.array([0.94, 0.60, 0.48]) * warm

    # --- substrate plane (color/height ~ stiffness) ---
    sub = pv.Plane(center=(0, 0, -1.6), direction=(0, 0, 1), i_size=6, j_size=6)
    p.add_mesh(sub, color=fill * 0.6 + 0.4, opacity=0.5, show_edges=False)

    # --- nucleus: ellipsoid, size from area, flattening from drive ---
    r = 0.35 + (state.nuclear_area - 100) * 0.006      # in-plane radius from area
    r = float(np.clip(r, 0.4, 1.6))
    zscale = float(np.clip(1 - warm * 0.5, 0.4, 1.0))  # flattening from drive
    nucleus = pv.ParametricEllipsoid(r, r, r * zscale)
    lam_op = float(np.clip(0.35 + (state.laminAC - 1) * 0.25, 0.3, 0.8))
    p.add_mesh(nucleus, color=fill, opacity=0.55, smooth_shading=True)
    # envelope (lamina) — a slightly larger shell, thickness ~ lamin
    lam_w = float(np.clip(1 + (state.laminAC - 1) * 3, 1, 5))
    shell = pv.ParametricEllipsoid(r * 1.04, r * 1.04, r * zscale * 1.04)
    p.add_mesh(shell, color=edge, opacity=0.25, style="wireframe",
               line_width=lam_w)

    # --- YAP spheres inside the nucleus (count ~ YAP N/C) ---
    n_yap = int(np.clip(round(state.yap_nc * 3), 3, 30))
    rng = np.random.default_rng(0)
    pts = rng.normal(0, r * 0.35, size=(n_yap, 3))
    pts[:, 2] *= zscale
    yap_cloud = pv.PolyData(pts)
    p.add_mesh(yap_cloud.glyph(scale=False, geom=pv.Sphere(radius=0.05)),
               color=edge, opacity=0.9)

    # --- adhesions + traction cones on the basal plane (count ~ n_c) ---
    n_adh = int(np.clip(round(3 + warm * 9), 4, 14))
    ang = np.linspace(0, 2 * np.pi, n_adh, endpoint=False)
    for a in ang:
        x, y = np.cos(a) * r * 1.5, np.sin(a) * r * 1.5
        cone = pv.Cone(center=(x, y, -1.3), direction=(0, 0, 1),
                       height=0.3 + warm * 0.4, radius=0.08)
        p.add_mesh(cone, color=edge, opacity=0.8)

    # --- annotation ---
    label = (f"E = {state.E_kPa:.0f} kPa   t = {state.t_h:.0f} h   "
             f"{state.fibrosis_stage}\n"
             f"area {state.nuclear_area:.0f} um2   drive {state.nuclear_drive:.0f}   "
             f"YAP {state.yap_nc:.1f}   tau {state.tau_h:.0f} h")
    p.add_text(label, position="upper_left", font_size=10, color="black")
    p.camera_position = "iso"
    return p


# ---------------------------------------------------------------------------
# Static render
# ---------------------------------------------------------------------------
def render_static(E, t, phenotype="hepatocyte", save=None):
    cell = VirtualCell(phenotype)
    state = cell.simulate(E, t=t)
    p = build_scene(state)
    if save:
        p.screenshot(save)
        print(f"Saved {save}")
    else:
        p.show()
    return state


# ---------------------------------------------------------------------------
# Interactive Trame app
# ---------------------------------------------------------------------------
def serve(phenotype="hepatocyte"):
    """Launch an interactive Trame app with stiffness/time sliders."""
    from trame.app import get_server
    from trame.ui.vuetify import SinglePageLayout
    from trame.widgets import vuetify, vtk as vtk_widgets

    pv.OFF_SCREEN = False
    cell = VirtualCell(phenotype)
    plotter = pv.Plotter()
    state = cell.simulate(23, t=120)
    build_scene(state, plotter)

    server = get_server(client_type="vue2")
    ctrl = server.controller
    state_srv = server.state
    state_srv.E = 23
    state_srv.t = 120

    def update(**kwargs):
        plotter.clear()
        s = cell.simulate(float(state_srv.E), t=float(state_srv.t))
        build_scene(s, plotter)
        ctrl.view_update()

    state_srv.change("E")(update)
    state_srv.change("t")(update)

    with SinglePageLayout(server) as layout:
        layout.title.set_text("Mechanogenomic Virtual Cell")
        with layout.content:
            with vuetify.VContainer(fluid=True):
                with vuetify.VRow():
                    vuetify.VSlider(v_model=("E", 23), min=0.5, max=23, step=0.5,
                                    label="stiffness E (kPa)")
                    vuetify.VSlider(v_model=("t", 120), min=2, max=120, step=2,
                                    label="time t (h)")
                view = vtk_widgets.VtkRemoteView(plotter.ren_win)
                ctrl.view_update = view.update
    server.start()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="3D virtual-cell renderer")
    ap.add_argument("--E", type=float, default=23.0, help="stiffness (kPa)")
    ap.add_argument("--t", type=float, default=120.0, help="time (h)")
    ap.add_argument("--phenotype", default="hepatocyte")
    ap.add_argument("--save", default=None, help="PNG output path (static)")
    ap.add_argument("--serve", action="store_true", help="launch Trame app")
    args = ap.parse_args()
    if args.serve:
        serve(args.phenotype)
    else:
        render_static(args.E, args.t, args.phenotype, save=args.save)
