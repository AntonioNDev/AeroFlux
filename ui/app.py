import os
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from formulas.mesh import EXTENSIONS, MeshModel
from formulas.pipeline import Solver, TestSettings, run_test
from rendering.export import OUTPUT_DIR, export_file, write_run
from rendering.matlab_view import MatlabView
from rendering.scene import FIELDS, SceneView

from . import theme
from .theme import COLORS as C
from .widgets import Collapsible, Segmented, SliderRow, Spec, StatCard
from .worker import JobPool

PREVIEW_FACES = 1500

STRENGTH = (
    Spec("speed", "Airspeed", "m/s", 10, 2000, 120, 5, "{:,.0f}"),
    Spec("altitude", "Altitude", "m", 0, 30000, 0, 100, "{:,.0f}"),
    Spec("angle_of_attack", "Angle of attack", "°", -30, 30, 0, 0.5, "{:.1f}"),
    Spec("sideslip", "Sideslip", "°", -30, 30, 0, 0.5, "{:.1f}"),
    Spec("turbulence", "Turbulence", "×", 0, 2, 1, 0.05, "{:.2f}"),
    Spec("thermal_weight", "Heat load", "×", 0.2, 3, 1, 0.05, "{:.2f}"),
)
QUALITY = (
    Spec("mesh_detail", "Mesh detail", "faces", 200, 5000, 1200, 100, "{:,.0f}"),
    Spec("line_count", "Streamline grid", "per side", 3, 14, 7, 1, "{:.0f}"),
)
PRESETS = {
    "Default": {},
    "Takeoff": {"speed": 70, "angle_of_attack": 8},
    "Cruise": {"speed": 230, "altitude": 10000, "angle_of_attack": 2},
    "Supersonic": {"speed": 680, "altitude": 15000, "angle_of_attack": 3, "thermal_weight": 1.3},
    "Stress test": {"speed": 1200, "altitude": 5000, "angle_of_attack": 12,
                    "turbulence": 1.6, "thermal_weight": 1.8},
}
STATS = (
    ("mach", "Mach", "", 1, "{:.2f}"),
    ("elapsed_s", "Solve time", "s", 1, "{:.2f}"),
    ("drag_n", "Drag", "N", 1, "{:,.0f}"),
    ("lift_n", "Lift", "N", 1, "{:,.0f}"),
    ("cd", "Drag coeff.", "", 1, "{:.3f}"),
    ("cl", "Lift coeff.", "", 1, "{:.3f}"),
    ("max_pressure_pa", "Peak pressure", "kPa", 1e-3, "{:,.1f}"),
    ("max_temp_k", "Peak wall temp", "K", 1, "{:,.0f}"),
)


class App(tk.Tk):
    def __init__(self, solver: Solver, advisor=None):
        super().__init__()
        self.solver = solver
        self.advisor = advisor
        self.matlab = MatlabView()
        self.pool = JobPool(workers=2)
        self.model = None
        self.result = None
        self.out_folder = None
        self._active = {}
        self._polling = False

        self.title("Wind Tunnel")
        self.geometry("1360x820")
        self.minsize(1240, 740)
        self.fonts = theme.apply(self)

        self.field = tk.StringVar(value="pressure")
        self.renderer = tk.StringVar(value="matplotlib")
        self.loop = tk.BooleanVar(value=False)
        self.length = tk.StringVar(value="4.0")
        self.preset = tk.StringVar(value="Preset")
        self.status = tk.StringVar(value="Import a model to begin")
        self.play_text = tk.StringVar(value="Play")

        self._build_menu()
        self._build_layout()
        self._bind_keys()
        self._refresh()
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _build_menu(self):
        bar = tk.Menu(self)
        files = tk.Menu(bar, tearoff=False)
        mod = "Command" if self.tk.call("tk", "windowingsystem") == "aqua" else "Control"
        files.add_command(label="Import model…", accelerator=f"{mod}+O", command=self.import_model)
        files.add_command(label="Load demo shape", command=lambda: self._load(None))
        files.add_separator()
        files.add_command(label="Save data…", accelerator=f"{mod}+S", command=self.save_data)
        files.add_command(label="Open output folder", command=self.open_output)
        files.add_separator()
        files.add_command(label="Exit", command=self._close)
        bar.add_cascade(label="File", menu=files)
        self.config(menu=bar)

    def _bind_keys(self):
        mod = "Command" if self.tk.call("tk", "windowingsystem") == "aqua" else "Control"
        self.bind(f"<{mod}-o>", lambda e: self.import_model())
        self.bind(f"<{mod}-s>", lambda e: self.save_data())
        self.bind(f"<{mod}-r>", lambda e: self.restart())
        self.bind("<F5>", lambda e: self.start())
        self.bind("<space>", lambda e: self._space(e))

    def _space(self, event):
        if not isinstance(event.widget, (ttk.Entry, ttk.Combobox)):
            self.scene.toggle()

    def _build_layout(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_sidebar().grid(row=0, column=0, sticky="ns")
        self._build_stage().grid(row=0, column=1, sticky="nsew")
        self._build_results().grid(row=0, column=2, sticky="ns")
        self._build_statusbar().grid(row=1, column=0, columnspan=3, sticky="ew")

    def _build_sidebar(self):
        side = ttk.Frame(self, width=330)
        side.grid_propagate(False)
        side.pack_propagate(False)
        inner = ttk.Frame(side)
        inner.pack(fill="both", expand=True, padx=18, pady=16)

        ttk.Label(inner, text="Wind Tunnel", style="Title.TLabel").pack(anchor="w")
        ttk.Label(inner, text="Aero and thermal test bench", style="Muted.TLabel").pack(anchor="w", pady=(0, 12))

        card = tk.Frame(inner, bg=C["card"], highlightthickness=1, highlightbackground=C["line"])
        card.pack(fill="x")
        self.model_name = ttk.Label(card, text="No model", style="Card.TLabel", font=self.fonts["bold"])
        self.model_name.pack(anchor="w", padx=12, pady=(10, 0))
        self.model_info = ttk.Label(card, text="Import an stl, obj, ply or off file", style="CardMuted.TLabel")
        self.model_info.pack(anchor="w", padx=12)
        size_row = tk.Frame(card, bg=C["card"])
        size_row.pack(fill="x", padx=12, pady=(8, 0))
        ttk.Label(size_row, text="Length", style="Card.TLabel").pack(side="left")
        ttk.Label(size_row, text="m", style="CardMuted.TLabel").pack(side="right")
        length = ttk.Entry(size_row, textvariable=self.length, width=8, justify="right")
        length.pack(side="right", padx=4)
        length.bind("<Return>", lambda e: self._update_model_card())
        length.bind("<FocusOut>", lambda e: self._update_model_card())
        btns = tk.Frame(card, bg=C["card"])
        btns.pack(fill="x", padx=12, pady=10)
        self.import_btn = ttk.Button(btns, text="Import model", style="Card.TButton", command=self.import_model)
        self.import_btn.pack(side="left", fill="x", expand=True)
        ttk.Button(btns, text="Demo", style="Card.TButton", command=lambda: self._load(None)).pack(side="left", padx=(6, 0))

        row = ttk.Frame(inner)
        row.pack(fill="x", pady=(14, 4))
        ttk.Label(row, text="TEST STRENGTH", style="Section.TLabel").pack(side="left")
        combo = ttk.Combobox(row, textvariable=self.preset, values=list(PRESETS), state="readonly", width=12)
        combo.pack(side="right")
        combo.bind("<<ComboboxSelected>>", self._apply_preset)

        self.rows = {}
        for spec in STRENGTH:
            self._add_slider(inner, spec)

        self.quality = Collapsible(inner, "Quality")
        self.quality.pack(fill="x", pady=(8, 0))
        for spec in QUALITY:
            self._add_slider(self.quality.body, spec)

        actions = ttk.Frame(inner)
        actions.pack(side="bottom", fill="x")
        self.start_btn = ttk.Button(actions, text="Start test", style="Accent.TButton", command=self.start)
        self.start_btn.pack(fill="x")
        second = ttk.Frame(actions)
        second.pack(fill="x", pady=(8, 0))
        self.restart_btn = ttk.Button(second, text="Restart", command=self.restart)
        self.restart_btn.pack(side="left", fill="x", expand=True)
        self.save_btn = ttk.Button(second, text="Save data", command=self.save_data)
        self.save_btn.pack(side="left", fill="x", expand=True, padx=(8, 0))
        return side

    def _add_slider(self, parent, spec):
        row = SliderRow(parent, spec, on_change=lambda: self.preset.set("Custom"))
        row.pack(fill="x", pady=5)
        self.rows[spec.key] = row

    def _build_stage(self):
        stage = ttk.Frame(self, style="Stage.TFrame")
        stage.rowconfigure(1, weight=1)
        stage.columnconfigure(0, weight=1)

        top = ttk.Frame(stage, style="Stage.TFrame")
        top.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 0))
        fields = [(k, v[0]) for k, v in FIELDS.items()]
        Segmented(top, fields, self.field, command=lambda: self.scene.set_field(self.field.get())).pack(side="left")
        self.renderer_seg = Segmented(top, [("matplotlib", "Matplotlib"), ("matlab", "MATLAB")],
                                      self.renderer, command=self._renderer_changed)
        self.renderer_seg.pack(side="right")
        if not MatlabView.installed():
            self.renderer_seg.enable("matlab", False)

        self.scene = SceneView(stage, C, on_play_state=lambda p: self.play_text.set("Pause" if p else "Play"))
        self.scene.grid(row=1, column=0, sticky="nsew", padx=14, pady=10)

        bottom = ttk.Frame(stage, style="Stage.TFrame")
        bottom.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 12))
        for name in ("iso", "side", "top", "front"):
            ttk.Button(bottom, text=name.capitalize(), style="Quiet.TButton",
                       command=lambda n=name: self.scene.view(n)).pack(side="left", padx=(0, 4))
        self.replay_btn = ttk.Button(bottom, text="Replay", command=lambda: self.scene.replay())
        self.replay_btn.pack(side="right")
        self.play_btn = ttk.Button(bottom, textvariable=self.play_text, width=7, command=lambda: self.scene.toggle())
        self.play_btn.pack(side="right", padx=6)
        ttk.Checkbutton(bottom, text="Loop", variable=self.loop,
                        command=lambda: self.scene.set_loop(self.loop.get())).pack(side="right", padx=8)
        return stage

    def _build_results(self):
        panel = ttk.Frame(self, width=280)
        panel.grid_propagate(False)
        panel.pack_propagate(False)
        inner = ttk.Frame(panel)
        inner.pack(fill="both", expand=True, padx=16, pady=16)
        ttk.Label(inner, text="RESULTS", style="Section.TLabel").pack(anchor="w", pady=(0, 6))

        grid = ttk.Frame(inner)
        grid.pack(fill="x")
        grid.columnconfigure((0, 1), weight=1, uniform="stat")
        self.cards = {}
        for i, (key, label, unit, _scale, _fmt) in enumerate(STATS):
            card = StatCard(grid, label, unit)
            card.grid(row=i // 2, column=i % 2, sticky="ew", padx=3, pady=3)
            self.cards[key] = card

        ttk.Label(inner, text="ADVISOR", style="Section.TLabel").pack(anchor="w", pady=(16, 6))
        self.advice = tk.Text(inner, height=8, wrap="word", bd=0, highlightthickness=1, padx=10, pady=8,
                              bg=C["card"], fg=C["text"], highlightbackground=C["line"],
                              insertbackground=C["text"], font=self.fonts["small"])
        self.advice.pack(fill="both", expand=True)
        self._set_advice([])
        self.folder_btn = ttk.Button(inner, text="Open output folder", command=self.open_output)
        self.folder_btn.pack(fill="x", pady=(10, 0))
        return panel

    def _build_statusbar(self):
        bar = tk.Frame(self, bg=C["bg"])
        self.progress = ttk.Progressbar(bar, maximum=100)
        self.progress.pack(fill="x")
        row = tk.Frame(bar, bg=C["bg"])
        row.pack(fill="x", padx=14, pady=5)
        ttk.Label(row, textvariable=self.status, style="Stage.TLabel").pack(side="left")
        if self.solver.placeholder:
            ttk.Label(row, text="Placeholder solver: numbers are not real", style="Stage.TLabel",
                      foreground=C["warn"]).pack(side="right")
        return bar

    # actions

    def import_model(self):
        if "run" in self._active:
            return
        kinds = " ".join(f"*{e}" for e in EXTENSIONS)
        path = filedialog.askopenfilename(title="Import model", filetypes=[("3D models", kinds), ("All files", "*.*")])
        if path:
            self._load(path)

    def _load(self, path):
        if "run" in self._active:
            return
        self._submit("load", self._load_job, path)
        self._progress(0.05, "Loading model")

    def start(self):
        if self.model is None or "run" in self._active or "load" in self._active:
            return
        try:
            length = float(self.length.get())
            if length <= 0:
                raise ValueError
        except ValueError:
            self.status.set("Model length must be a number above 0")
            return
        values = {key: row.get() for key, row in self.rows.items()}
        values["mesh_detail"] = int(values["mesh_detail"])
        values["line_count"] = int(values["line_count"])
        settings = TestSettings(model_length=length, **values)

        self.scene.clear_result()
        self._reset_results()
        self._submit("run", self._run_job, self.model, settings)
        self._progress(0.0, "Starting")

    def restart(self):
        self.pool.cancel_all()
        self._active.clear()
        for row in self.rows.values():
            row.set(row.spec.default)
        self.preset.set("Preset")
        self.result = None
        self._reset_results()
        if self.model is not None:
            self.scene.clear_result()
        self.progress.configure(value=0)
        self.status.set("Ready" if self.model else "Import a model to begin")
        self._refresh()

    def save_data(self):
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save data", defaultextension=".csv", initialfile=self.result.model.name,
            filetypes=[("Per-face table", "*.csv"), ("Summary", "*.json"),
                       ("MATLAB data", "*.mat"), ("View screenshot", "*.png")])
        if not path:
            return
        if path.lower().endswith(".png"):
            self.scene.snapshot(path)
            self.status.set(f"Saved {os.path.basename(path)}")
        else:
            self._submit("save", self._save_job, self.result, path, supersede=False)

    def open_output(self):
        target = self.out_folder or OUTPUT_DIR
        target.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(target)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", target])
        else:
            subprocess.Popen(["xdg-open", target])

    def _renderer_changed(self):
        if self.renderer.get() != "matlab":
            return
        if self.out_folder is not None and self.result is not None:
            self._submit("matlab", self._matlab_job, self.out_folder / "run.mat", supersede=False)
        else:
            self._submit("matlab", self._matlab_job, None, supersede=False)
            self.status.set("Starting MATLAB")

    def _apply_preset(self, _event=None):
        values = PRESETS[self.preset.get()]
        for key, row in self.rows.items():
            row.set(values.get(key, row.spec.default))

    # jobs, these run on the worker threads

    def _load_job(self, ctx, path):
        ctx.progress(0.2, "Reading file")
        model = MeshModel.demo() if path is None else MeshModel.from_file(path)
        ctx.progress(0.8, "Building preview")
        return model, model.scaled_to(1.0).simplified(PREVIEW_FACES)

    def _run_job(self, ctx, model, settings):
        result = run_test(self.solver, model, settings, ctx.progress, self.advisor)
        ctx.progress(0.97, "Saving results")
        return result, write_run(result)

    def _save_job(self, ctx, result, path):
        export_file(result, path)
        return path

    def _matlab_job(self, ctx, mat_file):
        ctx.progress(0.3, "Starting MATLAB")
        if mat_file is None:
            self.matlab.warm_up()
        else:
            self.matlab.show(mat_file)
        return mat_file

    # tk thread

    def _submit(self, kind, fn, *args, supersede=True):
        self._active[kind] = self.pool.submit(kind, fn, *args, supersede=supersede)
        self._refresh()
        if not self._polling:
            self._polling = True
            self.after(40, self._poll)

    def _poll(self):
        for ev in self.pool.drain():
            if self._active.get(ev.kind) != ev.ticket:
                continue
            if ev.type == "progress":
                self._progress(*ev.payload)
                continue
            del self._active[ev.kind]
            if ev.type == "done":
                self._finished(ev.kind, ev.payload)
            elif ev.type == "error":
                exc, trace = ev.payload
                print(trace, file=sys.stderr)
                self.progress.configure(value=0)
                self.status.set(f"{ev.kind} failed")
                messagebox.showerror("Wind Tunnel", str(exc) or exc.__class__.__name__)
            else:
                self.progress.configure(value=0)
        self._refresh()
        if self._active:
            self.after(40, self._poll)
        else:
            self._polling = False

    def _finished(self, kind, value):
        self.progress.configure(value=100 if kind in ("load", "run") else 0)
        self.after(900, lambda: None if self._active else self.progress.configure(value=0))
        if kind == "load":
            self.model, preview = value
            self.result = None
            self._reset_results()
            self.scene.show_model(preview)
            self._update_model_card()
            self.title(f"Wind Tunnel · {self.model.name}")
            self.status.set("Model loaded, press Start test")
        elif kind == "run":
            self.result, self.out_folder = value
            self.scene.show_result(self.result)
            self._show_results(self.result)
            self.status.set(f"Done in {self.result.elapsed:.2f} s · saved to {self.out_folder.name}")
            if self.renderer.get() == "matlab":
                self._submit("matlab", self._matlab_job, self.out_folder / "run.mat", supersede=False)
        elif kind == "save":
            self.status.set(f"Saved {os.path.basename(value)}")
        elif kind == "matlab":
            self.status.set("MATLAB ready" if value is None else "Opened in MATLAB")

    def _progress(self, fraction, message):
        self.progress.configure(value=fraction * 100)
        if message:
            self.status.set(message)

    def _reset_results(self):
        for card in self.cards.values():
            card.value.set("—")
        self._set_advice([])

    def _show_results(self, result):
        for key, _label, _unit, scale, fmt in STATS:
            if key in result.summary:
                self.cards[key].value.set(fmt.format(result.summary[key] * scale))
        self._set_advice(result.advice)

    def _set_advice(self, lines):
        self.advice.configure(state="normal")
        self.advice.delete("1.0", "end")
        if lines:
            self.advice.insert("1.0", "\n\n".join(lines))
        else:
            self.advice.insert("1.0", "No advisor connected yet.")
        self.advice.configure(state="disabled")

    def _update_model_card(self):
        if self.model is None:
            return
        m = self.model
        try:
            length = float(self.length.get())
            scale = length / m.length
            size = " × ".join(f"{d * scale:.2f}" for d in m.extent)
        except (ValueError, ZeroDivisionError):
            size = "—"
        name = m.name if len(m.name) < 26 else m.name[:23] + "…"
        self.model_name.configure(text=name)
        tight = "watertight" if m.is_watertight else "open mesh"
        self.model_info.configure(text=f"{len(m.faces):,} faces · {tight} · {size} m")

    def _refresh(self):
        running = "run" in self._active
        loading = "load" in self._active
        ready = self.model is not None and not running and not loading
        self.start_btn.state(["!disabled"] if ready else ["disabled"])
        self.import_btn.state(["disabled"] if running or loading else ["!disabled"])
        self.save_btn.state(["!disabled"] if self.result is not None else ["disabled"])
        for btn in (self.play_btn, self.replay_btn):
            btn.state(["!disabled"] if self.result is not None else ["disabled"])
        self.start_btn.configure(text="Running…" if running else "Start test")

    def _close(self):
        self.pool.cancel_all()
        self.pool.shutdown()
        self.matlab.close()
        self.destroy()
