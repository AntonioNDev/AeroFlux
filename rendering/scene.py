from math import ceil

import matplotlib

matplotlib.use("TkAgg")

import numpy as np
from matplotlib import cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from tkinter import ttk

FPS = 30
DURATION = 4.0
HEAD = 10
LIGHT = np.array([0.4, -0.5, 0.8]) / np.linalg.norm([0.4, -0.5, 0.8])

FIELDS = {
    "pressure": ("Pressure", "Pa", "viridis"),
    "cp": ("Pressure coefficient", "", "coolwarm"),
    "wall_temperature": ("Wall temperature", "K", "inferno"),
}
VIEWS = {"iso": (24, -58), "side": (0, -90), "top": (90, -90), "front": (0, 0)}


class SceneView(ttk.Frame):
    def __init__(self, parent, colors: dict, on_play_state=None):
        super().__init__(parent)
        self.c = colors
        self.on_play_state = on_play_state or (lambda playing: None)
        self.mesh = None
        self.result = None
        self.field = "pressure"
        self.loop = False
        self._zoom = 1.45
        self._box = np.ones(3)
        self._poly = None
        self._timer = None
        self._bg = None
        self._frame = 0
        self._animated = []

        self.fig = Figure(facecolor=colors["bg"])
        self.ax = self.fig.add_axes([0.0, 0.0, 0.88, 1.0], projection="3d")
        self.cax = self.fig.add_axes([0.9, 0.22, 0.014, 0.56])
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().configure(bg=colors["bg"], highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("draw_event", self._on_draw)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

        self.hud = self.fig.text(0.02, 0.96, "", color=colors["muted"], fontsize=9, va="top")
        self.hint = self.fig.text(0.45, 0.5, "", color=colors["muted"], fontsize=12, ha="center")
        self._style_axes()
        self.clear()

    def _style_axes(self):
        self.ax.set_facecolor(self.c["bg"])
        self.ax.set_axis_off()
        self.cax.set_facecolor(self.c["bg"])
        self.cax.tick_params(colors=self.c["muted"], labelsize=8, length=2)
        for spine in self.cax.spines.values():
            spine.set_visible(False)

    def clear(self):
        self._stop_timer()
        self._drop_artists()
        self.mesh = None
        self.result = None
        self.cax.set_visible(False)
        self.hud.set_text("")
        self.hint.set_text("Import a model to begin")
        self.canvas.draw_idle()

    def clear_result(self):
        self._stop_timer()
        self.result = None
        if self.mesh is not None:
            self.show_model(self.mesh, keep_view=True)

    def show_model(self, mesh, keep_view=False):
        self._stop_timer()
        self.result = None
        self.mesh = mesh
        self._drop_artists()
        self.hint.set_text("")
        self.cax.set_visible(False)
        self.hud.set_text(mesh.name)
        self._set_bounds(mesh.bounds)
        self._poly = self._make_poly(mesh)
        self._paint(None)
        if not keep_view:
            self.view("iso")
        self.canvas.draw_idle()
        self.on_play_state(False)

    def show_result(self, result):
        self._stop_timer()
        self._drop_artists()
        self.result = result
        self.mesh = result.model
        self.hint.set_text("")

        lines = result.streamlines
        low = np.minimum(result.model.bounds[0], lines.reshape(-1, 3).min(axis=0))
        high = np.maximum(result.model.bounds[1], lines.reshape(-1, 3).max(axis=0))
        self._set_bounds(np.array([low, high]))

        self._poly = self._make_poly(result.model)
        self._paint(self.field)

        first = [line[:2] for line in lines]
        self.tail = Line3DCollection(first, colors=self.c["flow"], linewidths=0.8, alpha=0.35)
        self.head = Line3DCollection(first, colors=self.c["flow_hot"], linewidths=1.6, alpha=0.95)
        for art in (self.tail, self.head):
            self.ax.add_collection3d(art)
        self._animated = [self.tail, self.head]
        self.replay()

    def set_field(self, key):
        self.field = key
        if self.result is not None:
            self._paint(key)
            self._redraw()

    def set_loop(self, enabled: bool):
        self.loop = enabled

    def view(self, name):
        elev, azim = VIEWS[name]
        self.ax.view_init(elev=elev, azim=azim)
        self._redraw()

    def snapshot(self, path):
        self.fig.savefig(path, dpi=160, facecolor=self.fig.get_facecolor())

    def toggle(self):
        if self.result is None:
            return
        if self._timer is not None:
            self._stop_timer()
            self._set_animated(False)
            self.on_play_state(False)
            self._redraw()
        elif self._frame >= self.result.streamlines.shape[1] - 1:
            self.replay()
        else:
            self._set_animated(True)
            self._start_timer()

    def replay(self):
        if self.result is None:
            return
        self._stop_timer()
        steps = self.result.streamlines.shape[1]
        self._stride = max(1, ceil(steps / (FPS * DURATION)))
        self._frame = 0
        self._apply_frame()
        self._set_animated(True)
        self.canvas.draw()
        self._start_timer()

    def _make_poly(self, mesh):
        poly = Poly3DCollection(mesh.vertices[mesh.faces], linewidths=0)
        self.ax.add_collection3d(poly)
        return poly

    def _paint(self, field):
        mesh = self.mesh
        shade = 0.45 + 0.55 * np.abs(mesh.face_normals @ LIGHT)
        if field is None or self.result is None:
            base = np.array([0.55, 0.64, 0.78])
            colors = np.clip(base[None, :] * shade[:, None], 0, 1)
            colors = np.column_stack([colors, np.ones(len(colors))])
            self.cax.set_visible(False)
        else:
            title, unit, cmap_name = FIELDS[field]
            values = getattr(self.result, field)
            lo, hi = float(values.min()), float(values.max())
            if hi - lo < 1e-9:
                hi = lo + 1.0
            norm = Normalize(lo, hi)
            cmap = matplotlib.colormaps[cmap_name]
            colors = cmap(norm(values))
            colors[:, :3] *= (0.6 + 0.4 * shade)[:, None]
            self.cax.clear()
            self._style_axes()
            self.cax.set_visible(True)
            bar = self.fig.colorbar(cm.ScalarMappable(norm, cmap), cax=self.cax)
            bar.outline.set_visible(False)
            bar.ax.tick_params(colors=self.c["muted"], labelsize=8, length=2)
            self.hud.set_text(f"{self.mesh.name}  ·  {title}" + (f" ({unit})" if unit else ""))
        self._poly.set_facecolor(colors)
        self._poly.set_edgecolor("none")

    def _set_bounds(self, bounds):
        centre = bounds.mean(axis=0)
        half = (bounds[1] - bounds[0]).max() / 2.0 * 1.05
        for setter, c in zip((self.ax.set_xlim, self.ax.set_ylim, self.ax.set_zlim), centre):
            setter(c - half, c + half)
        self._apply_zoom()

    def _apply_zoom(self):
        self.ax.set_box_aspect((1, 1, 1), zoom=self._zoom)

    def _drop_artists(self):
        for art in list(self.ax.collections):
            art.remove()
        self._animated = []
        self._poly = None
        self._bg = None

    def _redraw(self):
        self._bg = None
        self.canvas.draw_idle()

    def _on_scroll(self, event):
        self._zoom = float(np.clip(self._zoom * (1.1 if event.step > 0 else 1 / 1.1), 0.6, 5.0))
        self._apply_zoom()
        self._redraw()

    def _set_animated(self, flag):
        for art in self._animated:
            art.set_animated(flag)

    def _start_timer(self):
        self.on_play_state(True)
        self._timer = self.after(int(1000 / FPS), self._tick)

    def _stop_timer(self):
        if self._timer is not None:
            self.after_cancel(self._timer)
            self._timer = None

    def _apply_frame(self):
        lines = self.result.streamlines
        k = self._frame + 1
        self.tail.set_segments([line[:k] for line in lines])
        self.head.set_segments([line[max(0, k - HEAD):k] for line in lines])

    def _draw_animated(self):
        for art in self._animated:
            art.do_3d_projection()
            self.ax.draw_artist(art)

    def _on_draw(self, _event):
        if self._animated and self._animated[0].get_animated():
            self._bg = self.canvas.copy_from_bbox(self.fig.bbox)
            self._draw_animated()

    def _tick(self):
        self._timer = None
        if self.result is None:
            return
        last = self.result.streamlines.shape[1] - 1
        self._frame = min(self._frame + self._stride, last)
        self._apply_frame()
        if self._bg is None:
            self.canvas.draw()
        else:
            self.canvas.restore_region(self._bg)
            self._draw_animated()
            self.canvas.blit(self.fig.bbox)

        if self._frame < last:
            self._timer = self.after(int(1000 / FPS), self._tick)
        elif self.loop:
            self._frame = 0
            self._timer = self.after(400, self._tick)
        else:
            self._set_animated(False)
            self._redraw()
            self.on_play_state(False)
